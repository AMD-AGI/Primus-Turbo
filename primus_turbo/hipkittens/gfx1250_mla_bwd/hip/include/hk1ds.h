/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// hk1ds.h -- the gfx1250 builtins the dS^T export (k_dkdv64ds / k_dkdv96ds) adds to hk1.h; one
// instruction each. hk1.h itself is unchanged (the kernels without the export keep their binaries).
#pragma once
#include "hk1.h"

namespace hk1 {

typedef uint32_t u32x8 __attribute__((ext_vector_type(8)));

// v_mov_b32_dpp row_ror:8 (DPP16 control 0x128), row_mask 0xF (both 16-lane rows of wave32),
// bank_mask BANK, bound_ctrl off (llvm.amdgcn.update.dpp): lane l with bank (l % 16) / 4 enabled in
// BANK receives `src` of lane 16 * (l / 16) + (l % 16 + 8) % 16, every other lane keeps `old`.
// (emulator model dpp16; the same control and bank masks as the FlyDSL bwd_r7_a / r7_c export,
// whose toys passed on this card.)
template <int BANK> HK1_INLINE uint32_t dpp_ror8(uint32_t old, uint32_t src) {
    return (uint32_t) __builtin_amdgcn_update_dpp((int) old, (int) src, 0x128, 0xF, BANK, false);
}

// buffer resource over [base, base + nbytes): nbytes is an int64 byte count (the backend stores
// nbytes >> 7, so a multiple of 128 is exact); an access past it is dropped by the range check
// (never a fault)
HK1_INLINE __amdgpu_buffer_rsrc_t rsrc64(const void *base, int64_t nbytes) {
    return __builtin_amdgcn_make_buffer_rsrc(const_cast<void *>(base), (short) 0, (long) nbytes,
                                             RSRC_FLAGS);
}
// buffer_store_b128 of four dwords at voffset + soffset (+ the constant part of voffset in the
// offset field)
HK1_INLINE void buf_store_u32x4(u32x4 v, __amdgpu_buffer_rsrc_t r, int voff, int soff) {
    __builtin_amdgcn_raw_buffer_store_b128(v, r, voff, soff, 0);
}

} // namespace hk1
