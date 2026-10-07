/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

#include "../extensions.h"
#include "primus_turbo/quantization.h"

namespace primus_turbo::pytorch {

// The base seed of the stochastic-rounding quantizers (both the quantize_mx* packers and the MXFP4
// quantizer); resets their launch counters. The 64 bits of `seed` are taken as unsigned.
void set_sr_seed_next_pack(const int64_t seed) {
    sr_override_next_seed(SRStream::MXPack, static_cast<uint32_t>(seed & 0xffffffff));
}

void set_sr_seed(const int64_t seed) {
    primus_turbo::sr_set_base_seed(static_cast<uint64_t>(seed));
}

} // namespace primus_turbo::pytorch
