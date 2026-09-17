// Compile-only probe: does the udna1 (gfx1250, wave32) register tier's shape
// algebra actually line up with the WMMA operand maps it must feed?
//
// Build:
//   hipcc -std=c++20 -I$ROCM_PATH/include/hip -I<hk>/include \
//         -DKITTENS_UDNA1 --offload-arch=gfx1250 -c shape_algebra_probe.cu
//
// Every check below is a static_assert. No GPU is required, and nothing here
// depends on runtime behaviour -- this probe answers the *structural* half of
// the question. The numeric half (does `transpose` move the right bytes?)
// still needs a device run.

#include "kittens.cuh"
using namespace kittens;

// ---------------------------------------------------------------------------
// 0. Establish the wave width the headers actually believe in.
// ---------------------------------------------------------------------------
static_assert(WARP_THREADS == 32, "udna1 must be wave32");

// ---------------------------------------------------------------------------
// 1. Per-lane element counts, derived purely from rt_shape.cuh.
//    rt_shape.cuh is BYTE-IDENTICAL to cdna4's; only WARP_THREADS differs.
// ---------------------------------------------------------------------------
using S16x16 = ducks::rt_shape::rt_16x16;
using S16x32 = ducks::rt_shape::rt_16x32;
using S32x16 = ducks::rt_shape::rt_32x16;

static_assert(S16x16::elements_per_thread == 8,  "16x16 tile: 256/32 = 8 elems/lane");
static_assert(S16x32::elements_per_thread == 16, "16x32 tile: 512/32 = 16 elems/lane");
static_assert(S32x16::elements_per_thread == 16, "32x16 tile: 512/32 = 16 elems/lane");

// ---------------------------------------------------------------------------
// 2. Do those counts match the WMMA builtin operand widths?
//    wmma161632 (udna1/ops/warp/register/tile/mma.cuh) declares:
//        D/C accumulator : float2[4]  -> 8 floats per lane   (16x16 fp32)
//        A operand       : bf16_2[8]  -> 16 bf16  per lane   (16x32 bf16)
//        B operand       : bf16_2[8]  -> 16 bf16  per lane   (32x16 bf16)
// ---------------------------------------------------------------------------
using Acc  = rt_base<float, ducks::rt_layout::col, S16x16>;  // C / D
using AOpd = rt_base<bf16,  ducks::rt_layout::row, S16x32>;  // A

static_assert(Acc::packed_per_thread  == 4,  "C must be float2[4] to feed wmma_f32_16x16x32");
static_assert(sizeof(Acc::data)  == 32,      "C is v8f32 = 32 bytes per lane");
static_assert(AOpd::packed_per_thread == 8,  "A must be bf16_2[8] to feed wmma_f32_16x16x32");
static_assert(sizeof(AOpd::data) == 32,      "A is v16bf16 = 32 bytes per lane");

// ---------------------------------------------------------------------------
// 3. THE LOAD-BEARING CLAIM.
//
//    conversions.cuh::transpose is a PURE REGISTER RELABEL: for every k it does
//        result.tiles[j][i].data[k] = tile.tiles[i][j].data[k]
//    with no cross-lane movement. For that to compute a real transpose, the
//    source tile's (lane, k) -> (row, col) map and the destination tile's
//    (lane, k) -> (row, col) map must be exact duals.
//
//    A necessary (not sufficient) condition: source and destination must hold
//    the SAME number of elements per lane, or the copy loop cannot even be
//    well-formed. Check that the shape-transpose metafunction preserves it.
// ---------------------------------------------------------------------------
template<class S> using TS = typename ducks::rt_shape::transpose<S>::type;

static_assert(S16x32::elements_per_thread == TS<S16x32>::elements_per_thread,
              "transpose<16x32> must preserve elems/lane");
static_assert(S16x16::elements_per_thread == TS<S16x16>::elements_per_thread,
              "transpose<16x16> must preserve elems/lane");

// ---------------------------------------------------------------------------
// 4. The asymmetry that does NOT exist on CDNA.
//
//    transpose maps a col-layout (accumulator-shaped) 16x16 tile to a
//    row-layout 16x16 tile. But mma_ABt_base consumes A as a 16x32 row tile,
//    NOT a 16x16 row tile. So a transposed accumulator is not directly
//    feedable to the matrix unit -- it must first go through swap_layout.
//
//    Assert the gap explicitly so it is visible at compile time.
// ---------------------------------------------------------------------------
static_assert(!std::is_same_v<TS<S16x16>, S16x32>,
              "transpose of the 16x16 accumulator does NOT land on the 16x32 A-operand shape; "
              "a swap_layout step is mandatory on udna1");

// ---------------------------------------------------------------------------
// 5. swap_layout's hardcoded register indices.
//
//    udna1/ops/warp/register/tile/conversions.cuh, bf16 col->col branch
//    (shape1=rt_16x32, shape2=rt_16x16) reads:
//
//        for (int k = 0; k < 2; k++) {
//            res = permlane16_swap(src...data[k], src...data[k+1]);
//            dst...data[k]     = res.x;
//            dst...data[k + 2] = res.y;
//        }
//
//    Those literals 2 and "+2" are HALF of packed_per_thread for the wave64
//    rt_16x32 bf16 base tile (which is 4). On wave32 the same base tile has
//    packed_per_thread == 8. The loop therefore touches data[0..3] of an
//    8-entry array and leaves data[4..7] untouched.
// ---------------------------------------------------------------------------
using Bf16_16x32 = rt_base<bf16, ducks::rt_layout::col, S16x32>;
static_assert(Bf16_16x32::packed_per_thread == 8,
              "wave32 rt_16x32 bf16 base tile holds 8 bf16_2 per lane");
static_assert(Bf16_16x32::packed_per_thread != 4,
              "PORT BUG: swap_layout's k<2 / data[k+2] literals assume "
              "packed_per_thread==4 (wave64). On wave32 it is 8, so swap_layout "
              "writes only half the destination registers.");

// ---------------------------------------------------------------------------
// 6. num_strides drift. rt_shape hardcodes `stride` per shape but derives
//    num_strides = elements_per_thread / stride. Halving the wave width
//    doubles num_strides for every shape, silently changing the trip count of
//    every stride-group loop in the register tier.
// ---------------------------------------------------------------------------
static_assert(S16x16::stride == 4, "stride literal is wave-width independent");
static_assert(S16x16::num_strides == 2,
              "wave32 doubles num_strides for rt_16x16 (was 1 on wave64)");

int main() { return 0; }
