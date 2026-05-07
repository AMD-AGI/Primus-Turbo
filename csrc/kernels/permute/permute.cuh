#pragma once
#include "../deep_ep/utils.cuh"

#define MAX_NUM_CU 304

struct vsmem_t {
    void  *gmem_ptr;
    size_t bytes_per_block;
};

struct LookbackCache {
    void  *ptr        = nullptr;
    size_t total      = 0;
    size_t buf_bytes  = 0; // bytes per tile_state half
    int    active_idx = 0; // 0 or 1
};

struct TempStorageLayout {
    uint64_t *tile_state;      // active tile_state for THIS launch
    uint64_t *prev_tile_state; // tile_state used by the PREVIOUS launch on this
                               // stream; the last block's Phase 8 zeros it for
                               // the launch that will re-use this slot
    size_t  num_memset_int64;  // length of EACH tile_state buffer, in uint64 units
    vsmem_t vsmem;             // gmem_ptr == nullptr ⇒ kernel uses LDS
};

struct TileState {
    static constexpr uint32_t kInvalid  = 0u;
    static constexpr uint32_t kPartial  = 1u;
    static constexpr uint32_t kComplete = 2u;

    uint32_t flag;
    int32_t  value;
};

inline size_t align_up(size_t x, size_t a) {
    return (x + a - 1) / a * a;
}
__device__ __forceinline__ TileState load_tile_state(uint64_t *p) {
    const uint64_t raw = __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    return {static_cast<uint32_t>(raw), static_cast<int32_t>(static_cast<uint32_t>(raw >> 32))};
}

__device__ __forceinline__ void store_tile_state(uint64_t *p, uint32_t flag, int32_t value) {
    const uint64_t raw =
        static_cast<uint64_t>(flag) | (static_cast<uint64_t>(static_cast<uint32_t>(value)) << 32);
    __hip_atomic_store(p, raw, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

inline constexpr size_t kVsmemCacheLineSize = 128;

template <typename T>
__device__ __forceinline__ T *get_temp_storage(T *static_temp_storage, vsmem_t vsmem) {
    if (vsmem.gmem_ptr == nullptr) {
        return static_temp_storage;
    }
    return reinterpret_cast<T *>(static_cast<char *>(vsmem.gmem_ptr) +
                                 vsmem.bytes_per_block * blockIdx.x);
}
