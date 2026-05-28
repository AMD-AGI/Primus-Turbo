// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#pragma once

#include <cstring>
#include <hip/hip_runtime.h>

namespace primus_turbo {

// =============================================================================
// RMSNorm arch-conditional runtime tunables.
// =============================================================================
//
// The values below were measured ONLY on gfx942 (MI300X / MI325X). Each TODO
// branch returns the gfx942 value as a "safe default" — the kernel runs
// correctly, just may not be optimal. To tune a new arch:
//
//   1. Build the kernel with the gfx942 defaults.
//   2. Run `benchmark/ops/tune_rmsnorm.py` on the target GPU.
//   3. Copy the recommended values into the matching switch arm below.
//   4. Rebuild and re-bench against TransformerEngine to confirm.
//
// See `csrc/kernels/normalization/TUNING.md` for the full methodology.
//
// Tunables defined here (all runtime; the compile-time constants in
// `normalization.h` — `RMSNORM_CTAS_PER_CU`, `RMSNORM_WARPS_PER_BLOCK`,
// `RMSNORM_MAX_WAVES_PER_CU` — would require per-arch builds to vary and are
// kept fixed at safe gfx942 values):
//
//   - finalize_coalesced_threshold: cols above which the new TE-style col-
//     tile finalize beats the warp-per-col layout. Tracks L2 size × num_cus.
//
//   - bwd_half_unroll_lo / hi (bytes-per-row): cols range where halving the
//     uint4 LDG width in bwd wins (more in-flight HBM transactions per
//     thread). Expressed in bytes (rather than elements) so the dispatch is
//     dtype-agnostic at the call site.
//
//   - bwd_half_unroll_outer_len_max: above this outer_len, the half-unroll
//     path's block=1024 occupancy cliff (2 CTAs/CU on CDNA3) starves stage 0
//     more than the wider LDG burst wins; the kernel reverts to the
//     UNROLL_FULL block-per-row path. See struct member for sizing.

enum class RMSNormArch { GFX942, GFX950, GFX90A, GFX908, UNKNOWN };

inline RMSNormArch rmsnorm_detect_arch() {
    static RMSNormArch cached = []() -> RMSNormArch {
        hipDeviceProp_t prop;
        if (hipGetDeviceProperties(&prop, 0) != hipSuccess)
            return RMSNormArch::UNKNOWN;
        // gcnArchName is e.g. "gfx942:sramecc+:xnack-" — match on the gfxNNN
        // prefix to ignore feature suffixes. gfx908 and gfx90a both expose
        // major.minor = 9.0, so the name is the only way to distinguish.
        const char *name = prop.gcnArchName;
        if (std::strncmp(name, "gfx908", 6) == 0)
            return RMSNormArch::GFX908;
        if (std::strncmp(name, "gfx90a", 6) == 0)
            return RMSNormArch::GFX90A;
        if (std::strncmp(name, "gfx942", 6) == 0)
            return RMSNormArch::GFX942;
        if (std::strncmp(name, "gfx950", 6) == 0)
            return RMSNormArch::GFX950;
        return RMSNormArch::UNKNOWN;
    }();
    return cached;
}

struct RMSNormTunables {
    // cols ≥ this → use the coalesced col-tile finalize; below → warp-per-col.
    int finalize_coalesced_threshold;

    // bwd uses UNROLL/2 (half uint4 = uint2) when the row's bytes-per-row
    // falls in this inclusive range. Halving the uint4 LDG width turns
    // LDGS=1 into LDGS=2 → 2× in-flight HBM transactions per thread,
    // better latency hiding while stage 0 is bandwidth-bound. Measured at
    // MI325X: bf16 cols [4096, 8192], fp32 cols [2048, 4096] — same
    // bytes-per-row range [8192, 16384].
    int bwd_half_unroll_lo_bytes;
    int bwd_half_unroll_hi_bytes;

    // Iter-5: above this outer_len, the half-unroll path's block=1024 +
    // 2 CTAs/CU occupancy cap starves stage 0 even though each CTA has
    // wider LDG bursts. Crossover with the UNROLL_FULL path (block=512,
    // 4 CTAs/CU) is at ~25 rows-per-CTA-at-2-ctas/cu (= ~25 × 2 × num_cus
    // ≈ 15K on MI325X); 16384 used as a power-of-2 default that puts the
    // gate exactly between the bench grid's 16K/32K outer_len rows. Set
    // to 0 to disable the cap (use pure row-bytes dispatch).
    int bwd_half_unroll_outer_len_max;
};

inline RMSNormTunables rmsnorm_tunables() {
    // Measured on MI300X / MI325X (gfx942, CDNA3).
    constexpr RMSNormTunables gfx942 = {
        /*.finalize_coalesced_threshold     = */ 2048,
        /*.bwd_half_unroll_lo_bytes         = */ 8192,  // 4096 bf16 / 2048 fp32
        /*.bwd_half_unroll_hi_bytes         = */ 16384, // 8192 bf16 / 4096 fp32
        /*.bwd_half_unroll_outer_len_max    = */ 16384,
    };

    switch (rmsnorm_detect_arch()) {
    case RMSNormArch::GFX942:
        return gfx942;

    case RMSNormArch::GFX950:
        // CDNA4 (MI355). Has more vregs (~80K vs 64K on CDNA3) so the half-
        // unroll range may extend higher; L2 size may shift the finalize
        // threshold. TODO(tune_rmsnorm.py on gfx950): replace with measured.
        return gfx942;

    case RMSNormArch::GFX90A:
        // CDNA2 (MI250). 8 MB L2 per die → finalize threshold likely ~4096
        // (twice the L2 of gfx942's 4 MB). num_cus per die is also lower
        // (104) → smaller dgamma_part for same threshold.
        // TODO(tune_rmsnorm.py on gfx90a): replace with measured.
        return gfx942;

    case RMSNormArch::GFX908:
        // CDNA1 (MI100). 8 MB L2, 120 CUs. Same caveats as gfx90a; older
        // HBM also shifts the bandwidth/compute tradeoff that drives
        // bwd_half_unroll.
        // TODO(tune_rmsnorm.py on gfx908): replace with measured.
        return gfx942;

    default:
        return gfx942;
    }
}

} // namespace primus_turbo
