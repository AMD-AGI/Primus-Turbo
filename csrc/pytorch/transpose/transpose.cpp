// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#include "primus_turbo/transpose.h"
#include "pytorch/extensions.h"
#include "pytorch/utils.h"

#include <ATen/cuda/CUDAContext.h>

namespace primus_turbo::pytorch {

// Batched 2D transpose of the last two dims for any dtype. dim0/dim1 must be the
// trailing (-2, -1) dims of a 2D / 3D contiguous tensor.
at::Tensor transpose_2d(const at::Tensor input, const int64_t dim0, const int64_t dim1) {
    PRIMUS_TURBO_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    PRIMUS_TURBO_CHECK(input.is_contiguous(), "input must be contiguous");

    const int64_t ndim = input.dim();
    PRIMUS_TURBO_CHECK(ndim == 2 || ndim == 3, "transpose_2d only supports 2D or 3D input");
    int64_t d0 = (dim0 >= 0) ? dim0 : dim0 + ndim;
    int64_t d1 = (dim1 >= 0) ? dim1 : dim1 + ndim;
    PRIMUS_TURBO_CHECK(d0 >= 0 && d0 < ndim && d1 >= 0 && d1 < ndim, "transpose dim out of range");
    const int64_t lo = d0 < d1 ? d0 : d1;
    const int64_t hi = d0 < d1 ? d1 : d0;
    PRIMUS_TURBO_CHECK(lo == ndim - 2 && hi == ndim - 1,
                       "transpose_2d only supports transposing the last two dims (-1, -2)");

    const auto    sizes = input.sizes();
    const int64_t batch = (ndim == 3) ? sizes[0] : 1;
    const int64_t M     = sizes[ndim - 2];
    const int64_t N     = sizes[ndim - 1];

    std::vector<int64_t> out_shape(sizes.begin(), sizes.end());
    out_shape[ndim - 2] = N;
    out_shape[ndim - 1] = M;

    at::Tensor output = at::empty(out_shape, input.options());
    if (input.numel() == 0) {
        return output;
    }

    auto stream = at::cuda::getCurrentCUDAStream();
    transpose_2d_impl(input.data_ptr(), output.data_ptr(), batch, M, N, input.element_size(),
                      stream);
    return output;
}

} // namespace primus_turbo::pytorch
