// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#include "primus_turbo/normalization.h"
#include "../extensions.h"

namespace primus_turbo::jax {

// TODO: remove
// inline void PrintBufferInfo(const char *name, const ffi::AnyBuffer &buf) {
//     printf("%s:\n", name);
//     printf("Shape: ");
//     const ffi::AnyBuffer::Dimensions dims = buf.dimensions();
//     for (auto d : dims) {
//         printf("%d ", d);
//     }
//     printf("\n");
// }

ffi::Error RMSNormFwdFFI(cudaStream_t stream, ffi::AnyBuffer input, ffi::AnyBuffer gamma,
                         ffi::Result<ffi::AnyBuffer> output, double eps) {
    // printf("JAX/RMSNormFwdFFI\n");
    // PrintBufferInfo("input", input);
    // PrintBufferInfo("gamma", gamma);

    const int64_t inner_len = static_cast<int64_t>(gamma.element_count());
    const int64_t outer_len = static_cast<int64_t>(input.element_count() / inner_len);

    // TODO: refactor
    rmsnorm_fwd_impl<float>(input.typed_data<float>(), gamma.typed_data<float>(),
                            output->typed_data<float>(), inner_len, outer_len,
                            static_cast<float>(eps), stream);

    return ffi::Error::Success();
}

ffi::Error RMSNormBwdFFI(cudaStream_t stream, ffi::AnyBuffer doutput, ffi::AnyBuffer input,
                         ffi::AnyBuffer gamma, ffi::Result<ffi::AnyBuffer> dinput,
                         ffi::Result<ffi::AnyBuffer> dgamma, double eps) {
    // printf("JAX/RMSNormBwdFFI\n");
    // PrintBufferInfo("doutput", doutput);
    // PrintBufferInfo("input", input);
    // PrintBufferInfo("gamma", gamma);
    // PrintBufferInfo("dgamma", *dgamma);

    const int64_t inner_len = static_cast<int64_t>(gamma.element_count());
    const int64_t outer_len = static_cast<int64_t>(input.element_count() / inner_len);

    // TODO: refactor
    rmsnorm_bwd_impl<float>(input.typed_data<float>(), gamma.typed_data<float>(),
                            doutput.typed_data<float>(), dinput->typed_data<float>(),
                            dgamma->typed_data<float>(), inner_len, outer_len,
                            static_cast<float>(eps), stream);
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(RMSNormFwdHandler, RMSNormFwdFFI,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>() // stream
                                  .Arg<ffi::AnyBuffer>()                    // input
                                  .Arg<ffi::AnyBuffer>()                    // gamma
                                  .Ret<ffi::AnyBuffer>()                    // output
                                  .Attr<double>("eps")                      // eps
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(RMSNormBwdHandler, RMSNormBwdFFI,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>() // stream
                                  .Arg<ffi::AnyBuffer>()                    // doutput
                                  .Arg<ffi::AnyBuffer>()                    // input
                                  .Arg<ffi::AnyBuffer>()                    // gamma
                                  .Ret<ffi::AnyBuffer>()                    // dinput
                                  .Ret<ffi::AnyBuffer>()                    // dgamma
                                  .Attr<double>("eps"));

} // namespace primus_turbo::jax
