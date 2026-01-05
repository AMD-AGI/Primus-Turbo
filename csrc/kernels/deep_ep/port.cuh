#pragma once

#define WARP_SIZE 64
#define WARP_MASK 0xffffffffffffffffu
#define MAX_NTHREADS 1024
#define MAX_GROUPS (MAX_NTHREADS / WARP_SIZE) // 16 warps in the block
#define MAX_GROUPS_MASK 0xf

#define nv_bfloat16 hip_bfloat16
#define __nv_fp8x2_storage_t __hip_fp8x2_storage_t
#define __nv_fp8_storage_t __hip_fp8_storage_t
#define __nv_cvt_float2_to_fp8x2 __hip_cvt_float2_to_fp8x2
#define __NV_SATFINITE __HIP_SATFINITE
#define __NV_E4M3 __HIP_E4M3_FNUZ

#define CUDA_R_16BF HIP_R_16BF
