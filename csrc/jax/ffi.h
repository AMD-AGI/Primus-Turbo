#pragma once
#include "primus_turbo/common.h"
#include <hip/hip_runtime.h>
#include <xla/ffi/api/ffi.h>
namespace ffi = xla::ffi;
namespace primus_turbo::jax {

inline hipDataType FFIDataTypeToHIPDataType(const ffi::DataType &data_type) {
    switch (data_type) {
    case ffi::U8:
        return HIP_R_8U;
    case ffi::S8:
        return HIP_R_8I;
    case ffi::S32:
        return HIP_R_32I;
    case ffi::F16:
        return HIP_R_16F;
    case ffi::F32:
        return HIP_R_32F;
    case ffi::F64:
        return HIP_R_64F;
    case ffi::C64:
        return HIP_C_64F;
    case ffi::S16:
        return HIP_R_16I;
    case ffi::S64:
        return HIP_R_64I;
    case ffi::BF16:
        return HIP_R_16BF;
    case ffi::F8E4M3FNUZ:
        return HIP_R_8F_E4M3_FNUZ;
    case ffi::F8E5M2FNUZ:
        return HIP_R_8F_E5M2_FNUZ;
    default:
        std::stringstream data_type_str;
        data_type_str << data_type;
        PRIMUS_TURBO_CHECK(false, "Cannot convert ffi::DataType ", data_type_str.str(),
                           " to hipDataType.");
    }
}

} // namespace primus_turbo::jax
