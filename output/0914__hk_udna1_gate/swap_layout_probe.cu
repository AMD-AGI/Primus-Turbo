// Machine-checked arithmetic for the swap_layout bf16 col->col branch in
// udna1/ops/warp/register/tile/conversions.cuh (shape1=rt_16x32, shape2=rt_16x16).
//
// The loop there is literally:
//     for (int k = 0; k < 2; k++) {
//         res = permlane16_swap(src.tiles[i][2j].data[k], src.tiles[i][2j+1].data[k], ...);
//         dst.tiles[i][j].data[k]     = res.x;
//         dst.tiles[i][j].data[k + 2] = res.y;
//     }
// The literals `2` and `+2` are wave-width dependent and are NOT derived from
// any shape trait. This file asserts what they cover on wave32.
#include "kittens.cuh"
using namespace kittens;

using SrcBase = rt_base<bf16, ducks::rt_layout::col, ducks::rt_shape::rt_16x16>;
using DstBase = rt_base<bf16, ducks::rt_layout::col, ducks::rt_shape::rt_16x32>;

constexpr int K_TRIP      = 2;                          // hardcoded loop bound
constexpr int SRC_READ    = K_TRIP;                     // data[0..K_TRIP-1]
constexpr int DST_WRITTEN = 2 * K_TRIP;                 // data[k] and data[k+2]

constexpr int SRC_HAVE = SrcBase::packed_per_thread;
constexpr int DST_HAVE = DstBase::packed_per_thread;

static_assert(SRC_HAVE == 4, "wave32 rt_16x16 bf16 base tile: 8 bf16 = 4 bf16_2 per lane");
static_assert(DST_HAVE == 8, "wave32 rt_16x32 bf16 base tile: 16 bf16 = 8 bf16_2 per lane");

// On wave64 these would be SRC_HAVE==2, DST_HAVE==4 and both lines below would
// be equalities -- the loop would exactly cover the registers. On wave32:
static_assert(SRC_READ    < SRC_HAVE,
    "CONFIRMED: swap_layout reads only data[0..1] of a 4-entry source base tile. "
    "src.data[2] and src.data[3] are never read on wave32.");
static_assert(DST_WRITTEN < DST_HAVE,
    "CONFIRMED: swap_layout writes only data[0..3] of an 8-entry destination base tile. "
    "dst.data[4..7] are left UNINITIALISED on wave32.");

static_assert(SRC_READ * 2 == SRC_HAVE && DST_WRITTEN * 2 == DST_HAVE,
    "Exactly half the registers are covered in each direction.");

int main() { return 0; }
