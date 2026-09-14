// Host-only exhaustive check of the claim that udna1's ROW and COL register
// layouts are exact duals, transcribed directly from
//   include/udna1/ops/warp/memory/tile/global_to_register.cuh
//
//   ROW (lines 36-37, 46, 52-53, 115-116):
//     row_offset = lane % base_tile_rows
//     col_offset = stride * (lane / base_tile_rows)
//     row = R*i + row_offset
//     col = C*j + col_offset + k*elements_per_stride_group
//     idx = l + k*(stride/packing);  element .x/.y at col + l*packing + {0,1}
//
//   COL (lines 147-148, 155-157, 160-163):
//     row_offset = stride * (lane / base_tile_cols)
//     col_offset = lane % base_tile_cols
//     row = R*i + row_offset + k*elements_per_stride_group
//     col = C*j + col_offset
//     idx = l + k*(stride/packing);  element .x/.y at row + l*2 + {0,1}
//
// If these are duals then for every (lane, idx, half) the ROW map on an RxC
// tile and the COL map on a CxR tile name transposed coordinates -- which is
// exactly what makes conversions.cuh::transpose (a pure data[k]->data[k]
// relabel) compute a real transpose.
#include <cstdio>
#include <vector>
#include <tuple>

struct Shape { int rows, cols, stride; const char* name; };

static int elements_per_thread(const Shape& s, int WARP) { return s.rows*s.cols/WARP; }
static int esg(const Shape& s, int WARP, bool row_layout) {
    int reductions = row_layout ? s.cols : s.rows;
    int ept = elements_per_thread(s, WARP);
    int threads_per_reduction = reductions / ept;
    return threads_per_reduction * s.stride;
}

int main() {
    const int PACK = 2; // bf16_2 / float2: two scalars per packed element
    int fails = 0, checked = 0;

    // (src shape, dst shape) pairs exactly as ducks::rt_shape::transpose maps them.
    std::vector<std::pair<Shape,Shape>> pairs = {
        {{16,16,4,"rt_16x16"}, {16,16,4,"rt_16x16"}},
        {{16,32,8,"rt_16x32"}, {32,16,8,"rt_32x16"}},
        {{32,16,8,"rt_32x16"}, {16,32,8,"rt_16x32"}},
        {{32,32,4,"rt_32x32"}, {32,32,4,"rt_32x32"}},
    };

    for (int WARP : {64, 32}) {
      printf("=== WARP_THREADS = %d %s ===\n", WARP, WARP==64?"(cdna4)":"(udna1/gfx1250)");
      for (auto& pr : pairs) {
        const Shape S = pr.first, D = pr.second;
        int eptS = elements_per_thread(S, WARP), eptD = elements_per_thread(D, WARP);
        if (eptS != eptD) { printf("  %-9s -> %-9s SKIP (elems/lane %d vs %d)\n", S.name, D.name, eptS, eptD); continue; }
        int nsS = eptS / S.stride, nsD = eptD / D.stride;
        int esgS = esg(S, WARP, /*row*/true), esgD = esg(D, WARP, /*row*/false);
        int bad = 0;
        for (int lane = 0; lane < WARP; ++lane)
          for (int k = 0; k < nsS; ++k)
            for (int l = 0; l < S.stride/PACK; ++l)
              for (int h = 0; h < PACK; ++h) {
                // ROW layout on the source tile
                int srow = lane % S.rows;
                int scol = S.stride*(lane / S.rows) + k*esgS + l*PACK + h;
                // COL layout on the destination tile, SAME (lane, idx, half)
                int dcol = lane % D.cols;
                int drow = D.stride*(lane / D.cols) + k*esgD + l*2 + h;
                ++checked;
                if (!(drow == scol && dcol == srow)) { ++bad; ++fails; }
              }
        printf("  %-9s -> %-9s  ept=%2d ns=%d esg(row)=%2d esg(col)=%2d  %s\n",
               S.name, D.name, eptS, nsS, esgS, esgD, bad ? "*** NOT DUAL ***" : "dual OK");
      }
    }
    printf("\nchecked %d (lane,idx,half) triples; %d violations\n", checked, fails);
    printf("VERDICT: row/col register layouts are %s duals on both wave widths\n",
           fails ? "NOT" : "exact");
    return fails ? 1 : 0;
}
