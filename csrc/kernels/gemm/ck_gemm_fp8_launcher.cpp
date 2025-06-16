#include "ck_gemm_fp8_launcher.h"

namespace primus_turbo {

template struct CKGemmFP8BlockwiseLauncher<
    CKGemmFP8BlockwiseDescriptor_E4M3_BF16_NT_ScaleBlkM1N128K128_GemmBlkM128N128K128>;

}
