/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// quantize_mxfp6_row_mxfp4_col_impl<float16>: one instantiation of the fused MX packers (see quantization_mxfp6_gfx950_impl.cuh).

#include "quantization_mxfp6_gfx950_impl.cuh"

namespace primus_turbo {

template void quantize_mxfp6_row_mxfp4_col_impl<float16>(const float16 *, uint8_t *, uint8_t *,
                                                         uint8_t *, uint8_t *, const int,
                                                         const int, hipStream_t);

} // namespace primus_turbo
