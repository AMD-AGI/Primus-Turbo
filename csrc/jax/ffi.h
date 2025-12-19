// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#pragma once
#include "primus_turbo/common.h"
#include <hip/hip_runtime.h>
#include <xla/ffi/api/ffi.h>
namespace ffi = xla::ffi;
namespace primus_turbo::jax {

//==============================================================================
// DType Enum (for pybind11 interface)
//==============================================================================

enum class DType {
    kByte           = 0,
    kInt16          = 1,
    kInt32          = 2,
    kInt64          = 3,
    kFloat32        = 4,
    kFloat16        = 5,
    kBFloat16       = 6,
    kFloat8E4M3FN   = 7,  // OCP FP8 E4M3
    kFloat8E4M3FNUZ = 8,  // AMD FP8 E4M3
    kFloat8E5M2     = 9,  // OCP FP8 E5M2
    kFloat8E5M2FNUZ = 10, // AMD FP8 E5M2
    kFloat8E8M0     = 11, // MX scaling format
};

inline ffi::DataType dtype_to_ffi_dtype(DType dtype) {
    switch (dtype) {
    case DType::kByte:
        return ffi::DataType::U8;
    case DType::kInt16:
        return ffi::DataType::S16;
    case DType::kInt32:
        return ffi::DataType::S32;
    case DType::kInt64:
        return ffi::DataType::S64;
    case DType::kFloat32:
        return ffi::DataType::F32;
    case DType::kFloat16:
        return ffi::DataType::F16;
    case DType::kBFloat16:
        return ffi::DataType::BF16;
    case DType::kFloat8E4M3FN:
        return ffi::DataType::F8E4M3FN;
    case DType::kFloat8E4M3FNUZ:
        return ffi::DataType::F8E4M3FNUZ;
    case DType::kFloat8E5M2:
        return ffi::DataType::F8E5M2;
    case DType::kFloat8E5M2FNUZ:
        return ffi::DataType::F8E5M2FNUZ;
    case DType::kFloat8E8M0:
        return ffi::DataType::F8E8M0FNU;
    default:
        PRIMUS_TURBO_CHECK(false, "Unsupported DType for conversion to ffi::DataType");
    }
}

inline size_t ffi_dtype_to_bytes(ffi::DataType dtype) {
    switch (dtype) {
    case ffi::DataType::U8:
    case ffi::DataType::S8:
    case ffi::DataType::F8E4M3FN:
    case ffi::DataType::F8E4M3FNUZ:
    case ffi::DataType::F8E5M2:
    case ffi::DataType::F8E5M2FNUZ:
    case ffi::DataType::F8E8M0FNU:
        return 1;
    case ffi::DataType::S16:
    case ffi::DataType::F16:
    case ffi::DataType::BF16:
        return 2;
    case ffi::DataType::S32:
    case ffi::DataType::F32:
        return 4;
    case ffi::DataType::S64:
    case ffi::DataType::F64:
        return 8;
    default:
        PRIMUS_TURBO_CHECK(false, "Unsupported ffi::DataType for byte size");
    }
}

inline size_t dtype_to_bytes(DType dtype) {
    return ffi_dtype_to_bytes(dtype_to_ffi_dtype(dtype));
}

//==============================================================================
// ffi::DataType -> hipDataType conversion
//==============================================================================

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
