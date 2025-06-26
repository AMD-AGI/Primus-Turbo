#pragma once

namespace primus_turbo {

#ifdef __HIPCC__

#include <hip/hip_fp8.h>

// TODO(ruibzhan): support gfx950 architecture
typedef __hip_fp8_e4m3_fnuz hip_fp8_e4m3;
typedef __hip_fp8_e5m2_fnuz hip_fp8_e5m2;

#else  //__HIPCC__
typedef struct {
    char storage;
} hip_fp8_e4m3;
typedef struct {
    char storage;
} hip_fp8_e5m2;
#endif //__HIPCC__

} // namespace primus_turbo
