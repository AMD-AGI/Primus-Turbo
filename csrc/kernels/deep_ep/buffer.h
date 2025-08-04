/*
 * Copyright (c) 2025 DeepSeek. All rights reserved.
 *
 * Modification Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 */

#pragma once

#include "configs.h"

namespace primus_turbo::deep_ep {

template <typename dtype_t> struct Buffer {
  private:
    uint8_t *ptr;

  public:
    int total_bytes;

    __device__ __forceinline__ Buffer() : ptr(nullptr), total_bytes(0) {}

    __device__ __forceinline__ Buffer(void *&gbl_ptr, int num_elems, int offset = 0) {
        total_bytes = num_elems * sizeof(dtype_t);
        ptr         = reinterpret_cast<uint8_t *>(gbl_ptr) + offset * sizeof(dtype_t);
        gbl_ptr     = reinterpret_cast<uint8_t *>(gbl_ptr) + total_bytes;
    }

    __device__ __forceinline__ Buffer advance_also(void *&gbl_ptr) {
        gbl_ptr = reinterpret_cast<uint8_t *>(gbl_ptr) + total_bytes;
        return *this;
    }

    __device__ __forceinline__ dtype_t *buffer() { return reinterpret_cast<dtype_t *>(ptr); }

    __device__ __forceinline__ dtype_t &operator[](int idx) { return buffer()[idx]; }
};

template <typename dtype_t, int kNumRanks = 1> struct AsymBuffer {
  private:
    uint8_t *ptrs[kNumRanks];
    int      num_bytes;

  public:
    int total_bytes;

    __device__ __forceinline__ AsymBuffer(void *&gbl_ptr, int num_elems, int num_ranks,
                                          int sm_id = 0, int num_sms = 1, int offset = 0) {
        PRIMUS_TURBO_STATIC_CHECK(kNumRanks == 1, "");
        num_bytes = num_elems * sizeof(dtype_t);

        int per_channel_bytes = num_bytes * num_ranks;
        total_bytes           = per_channel_bytes * num_sms;
        ptrs[0] =
            reinterpret_cast<uint8_t *>(gbl_ptr) + per_channel_bytes * sm_id + num_bytes * offset;
        gbl_ptr = reinterpret_cast<uint8_t *>(gbl_ptr) + total_bytes;
    }

    __device__ __forceinline__ AsymBuffer(void **gbl_ptrs, int num_elems, int num_ranks,
                                          int sm_id = 0, int num_sms = 1, int offset = 0) {
        PRIMUS_TURBO_STATIC_CHECK(kNumRanks > 1, "");
        num_bytes = num_elems * sizeof(dtype_t);

        int per_channel_bytes = num_bytes * num_ranks;
        total_bytes           = per_channel_bytes * num_sms;
        for (int i = 0; i < kNumRanks; ++i) {
            ptrs[i] = reinterpret_cast<uint8_t *>(gbl_ptrs[i]) + per_channel_bytes * sm_id +
                      num_bytes * offset;
            gbl_ptrs[i] = reinterpret_cast<uint8_t *>(gbl_ptrs[i]) + total_bytes;
        }
    }

    __device__ __forceinline__ void advance(int shift) {
#pragma unroll
        for (int i = 0; i < kNumRanks; ++i)
            ptrs[i] = ptrs[i] + shift * sizeof(dtype_t);
    }

    __device__ __forceinline__ AsymBuffer advance_also(void *&gbl_ptr) {
        gbl_ptr = reinterpret_cast<uint8_t *>(gbl_ptr) + total_bytes;
        return *this;
    }

    template <int kNumAlsoRanks>
    __device__ __forceinline__ AsymBuffer advance_also(void **gbl_ptrs) {
        for (int i = 0; i < kNumAlsoRanks; ++i)
            gbl_ptrs[i] = reinterpret_cast<uint8_t *>(gbl_ptrs[i]) + total_bytes;
        return *this;
    }

    __device__ __forceinline__ dtype_t *buffer(int idx = 0) {
        PRIMUS_TURBO_STATIC_CHECK(kNumRanks == 1,
                                  "`buffer` is only available for single rank case");
        return reinterpret_cast<dtype_t *>(ptrs[0] + num_bytes * idx);
    }

    __device__ __forceinline__ dtype_t *buffer_by(int rank_idx, int idx = 0) {
        PRIMUS_TURBO_STATIC_CHECK(kNumRanks > 1, "`buffer` is only available for single rank case");
        return reinterpret_cast<dtype_t *>(ptrs[rank_idx] + num_bytes * idx);
    }
};

} // namespace primus_turbo::deep_ep
