#pragma once
#include "primus_turbo/common.h"
#include <hip/hip_runtime.h>
#include <xla/ffi/api/ffi.h>
namespace ffi = xla::ffi;
namespace primus_turbo::jax {

inline cudaDataType FFIDataTypeToHIPDataType(const ffi::DataType &data_type) {
    switch (data_type) {
    case ffi::DataType::U8:
        return HIP_R_8U;
    case ffi::DataType::S8:
        return HIP_R_8I;
    case ffi::DataType::S32:
        return HIP_R_32I;
    case ffi::DataType::F16:
        return HIP_R_16F;
    case ffi::DataType::F32:
        return HIP_R_32F;
    case ffi::DataType::F64:
        return HIP_R_64F;
    case ffi::DataType::F8E4M3FNUZ:
        return HIP_R_8F_E4M3_FNUZ;
    case ffi::DataType::F8E5M2FNUZ:
        return HIP_R_8F_E5M2_FNUZ;
    default:
        PRIMUS_TURBO_CHECK(false, "Cannot convert ffi::DataType to hipDataType.");
    }
}

} // namespace primus_turbo::jax
