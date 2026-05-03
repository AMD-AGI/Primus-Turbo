#pragma once

#define WARP_SIZE 64

#define UNROLLED_WARP_COPY(UNROLL_FACTOR, LANE_ID, N, DST, SRC, LD_FUNC, ST_FUNC)                  \
    {                                                                                              \
        constexpr int kLoopStride = WARP_SIZE * (UNROLL_FACTOR);                                   \
        typename std::remove_reference<decltype(LD_FUNC((SRC) + 0))>::type                         \
             unrolled_values[(UNROLL_FACTOR)];                                                     \
        auto __src = (SRC);                                                                        \
        auto __dst = (DST);                                                                        \
        for (int __i = (LANE_ID); __i < ((N) / kLoopStride) * kLoopStride; __i += kLoopStride) {   \
            _Pragma("unroll") for (int __j = 0; __j < (UNROLL_FACTOR); ++__j)                      \
                unrolled_values[__j] = LD_FUNC(__src + __i + __j * 32);                            \
            _Pragma("unroll") for (int __j = 0; __j < (UNROLL_FACTOR); ++__j)                      \
                ST_FUNC(__dst + __i + __j * 32, unrolled_values[__j]);                             \
        }                                                                                          \
        {                                                                                          \
            int __i = ((N) / kLoopStride) * kLoopStride + (LANE_ID);                               \
            _Pragma("unroll") for (int __j = 0; __j < (UNROLL_FACTOR); ++__j) {                    \
                if (__i + __j * WARP_SIZE < (N)) {                                                 \
                    unrolled_values[__j] = LD_FUNC(__src + __i + __j * WARP_SIZE);                 \
                }                                                                                  \
            }                                                                                      \
            _Pragma("unroll") for (int __j = 0; __j < (UNROLL_FACTOR); ++__j) {                    \
                if (__i + __j * WARP_SIZE < (N)) {                                                 \
                    ST_FUNC(__dst + __i + __j * WARP_SIZE, unrolled_values[__j]);                  \
                }                                                                                  \
            }                                                                                      \
        }                                                                                          \
    }
