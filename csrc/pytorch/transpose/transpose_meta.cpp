// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#include "pytorch/extensions.h"
#include "pytorch/utils.h"

namespace primus_turbo::pytorch {

at::Tensor transpose_2d_meta(const at::Tensor input, const int64_t dim0, const int64_t dim1) {
    const int64_t ndim = input.dim();
    PRIMUS_TURBO_CHECK(ndim == 2 || ndim == 3, "transpose_2d only supports 2D or 3D input");
    int64_t d0 = (dim0 >= 0) ? dim0 : dim0 + ndim;
    int64_t d1 = (dim1 >= 0) ? dim1 : dim1 + ndim;
    PRIMUS_TURBO_CHECK(d0 >= 0 && d0 < ndim && d1 >= 0 && d1 < ndim, "transpose dim out of range");
    const int64_t lo = d0 < d1 ? d0 : d1;
    const int64_t hi = d0 < d1 ? d1 : d0;
    PRIMUS_TURBO_CHECK(lo == ndim - 2 && hi == ndim - 1,
                       "transpose_2d only supports transposing the last two dims (-1, -2)");

    std::vector<int64_t> out_shape(input.sizes().begin(), input.sizes().end());
    const int64_t        tmp = out_shape[ndim - 2];
    out_shape[ndim - 2]      = out_shape[ndim - 1];
    out_shape[ndim - 1]      = tmp;
    return at::empty(out_shape, at::dtype(input.scalar_type()).device(at::kMeta));
}

} // namespace primus_turbo::pytorch
