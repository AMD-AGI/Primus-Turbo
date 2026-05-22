// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.
//
// Device-side view over an IPC-mapped symmetric memory region.  Mirrors
// DeepGEMM's ``deep_gemm::layout::SymBuffer<kNumRanks>`` 1:1 so the
// kernel-side mental model stays familiar — the only platform-specific
// detail is that ``map(...)`` here returns a regular global pointer
// that vector load/store routes over XGMI (HIP IPC) instead of NVLink
// TMA.

#pragma once

#include <cstdint>
#include <hip/hip_runtime.h>

namespace primus_turbo {
namespace mega_moe {
namespace layout {

// Mirrors ``deep_gemm::layout::kNumMaxRanks``.  Kept the same value
// (72) so the on-device layout of ``SymBuffer<kNumRanks>`` matches DG
// byte-for-byte; the host side only fills the first ``world_size``
// slots and leaves the rest at zero.
constexpr static uint32_t kNumMaxRanks = 72;

template <uint32_t kNumRanks = kNumMaxRanks> struct SymBuffer {
    int64_t  base;
    int64_t  offsets[kNumMaxRanks];
    uint32_t rank_idx;

    static_assert(kNumRanks <= kNumMaxRanks, "Too many ranks");

    SymBuffer() = default;

    // Host-side ctor: ``c[r]`` is the IPC-mapped base pointer of rank
    // ``r`` (as visible to this process).  Templated on the container
    // type so callers may pass ``std::vector<int64_t>``, a raw array,
    // or any indexable type with ``.size()``.
    template <typename Container>
    __host__ explicit SymBuffer(const Container &c, const uint32_t &rank_idx) : rank_idx(rank_idx) {
        const auto size = static_cast<uint32_t>(c.size());
        base            = c[rank_idx];
        for (uint32_t i = 0; i < kNumMaxRanks; ++i)
            offsets[i] = i < size ? (c[i] - base) : 0;
    }

    __host__ __device__ __forceinline__ void *get_base_ptr() const {
        return reinterpret_cast<void *>(base);
    }

    // Translate a local pointer (valid on this rank) into the
    // equivalent address inside ``dst_rank_idx``'s allocation.  A
    // subsequent vector load/store on the returned pointer goes over
    // XGMI transparently on a single node.
    template <typename ptr_t>
    __device__ __forceinline__ ptr_t map(const ptr_t &ptr, const uint32_t &dst_rank_idx) const {
        const int64_t mapped = offsets[dst_rank_idx] + reinterpret_cast<int64_t>(ptr);
        return reinterpret_cast<ptr_t>(mapped);
    }
};

} // namespace layout
} // namespace mega_moe
} // namespace primus_turbo
