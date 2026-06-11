/*
 * Copyright (c) 2025 DeepSeek. All rights reserved.
 *
 * Modification Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 */

#pragma once

#include "primus_turbo/deep_ep/configs.h"

namespace primus_turbo::deep_ep {

template <typename dtype_t> struct Buffer {
private:
    uint8_t *ptr;

public:
    int64_t total_bytes;

    __device__ __forceinline__ Buffer() : ptr(nullptr), total_bytes(0) {}

    __device__ __forceinline__ Buffer(void *&gbl_ptr, int64_t num_elems, int64_t offset = 0) {
        total_bytes = num_elems * static_cast<int64_t>(sizeof(dtype_t));
        ptr = reinterpret_cast<uint8_t *>(gbl_ptr) + offset * static_cast<int64_t>(sizeof(dtype_t));
        gbl_ptr = reinterpret_cast<uint8_t *>(gbl_ptr) + total_bytes;
    }

    __device__ __forceinline__ Buffer advance_also(void *&gbl_ptr) {
        gbl_ptr = reinterpret_cast<uint8_t *>(gbl_ptr) + total_bytes;
        return *this;
    }

    __device__ __forceinline__ dtype_t *buffer() { return reinterpret_cast<dtype_t *>(ptr); }

    __device__ __forceinline__ dtype_t &operator[](int64_t idx) { return buffer()[idx]; }
};

template <typename dtype_t, int kNumRanks = 1> struct AsymBuffer {
private:
    uint8_t *ptrs[kNumRanks];
    int64_t  num_bytes;

public:
    int64_t total_bytes;

    __device__ __forceinline__ AsymBuffer(void *&gbl_ptr, int64_t num_elems, int num_ranks,
                                          int sm_id = 0, int num_sms = 1, int64_t offset = 0) {
        PRIMUS_TURBO_STATIC_CHECK(kNumRanks == 1, "");
        num_bytes = num_elems * static_cast<int64_t>(sizeof(dtype_t));

        int64_t per_channel_bytes = num_bytes * num_ranks;
        total_bytes               = per_channel_bytes * num_sms;
        ptrs[0] =
            reinterpret_cast<uint8_t *>(gbl_ptr) + per_channel_bytes * sm_id + num_bytes * offset;
        gbl_ptr = reinterpret_cast<uint8_t *>(gbl_ptr) + total_bytes;
    }

    __device__ __forceinline__ AsymBuffer(void **gbl_ptrs, int64_t num_elems, int num_ranks,
                                          int sm_id = 0, int num_sms = 1, int64_t offset = 0) {
        PRIMUS_TURBO_STATIC_CHECK(kNumRanks > 1, "");
        num_bytes = num_elems * static_cast<int64_t>(sizeof(dtype_t));

        int64_t per_channel_bytes = num_bytes * num_ranks;
        total_bytes               = per_channel_bytes * num_sms;
        for (int i = 0; i < kNumRanks; ++i) {
            ptrs[i] = reinterpret_cast<uint8_t *>(gbl_ptrs[i]) + per_channel_bytes * sm_id +
                      num_bytes * offset;
            gbl_ptrs[i] = reinterpret_cast<uint8_t *>(gbl_ptrs[i]) + total_bytes;
        }
    }

    __device__ __forceinline__ void advance(int64_t shift) {
#pragma unroll
        for (int i = 0; i < kNumRanks; ++i)
            ptrs[i] = ptrs[i] + shift * static_cast<int64_t>(sizeof(dtype_t));
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

    __device__ __forceinline__ dtype_t *buffer(int64_t idx = 0) {
        PRIMUS_TURBO_STATIC_CHECK(kNumRanks == 1,
                                  "`buffer` is only available for single rank case");
        return reinterpret_cast<dtype_t *>(ptrs[0] + num_bytes * idx);
    }

    __device__ __forceinline__ dtype_t *buffer_by(int rank_idx, int64_t idx = 0) {
        PRIMUS_TURBO_STATIC_CHECK(kNumRanks > 1, "`buffer` is only available for single rank case");
        return reinterpret_cast<dtype_t *>(ptrs[rank_idx] + num_bytes * idx);
    }
};

template <typename dtype_t, bool kDecoupled = true> struct SymBuffer {
private:
    // NOTES: for non-decoupled case, `recv_ptr` is not used
    uint8_t *send_ptr;
    uint8_t *recv_ptr;
    int64_t  num_bytes;

public:
    int64_t total_bytes;

    __device__ __forceinline__ SymBuffer(void *&gbl_ptr, int64_t num_elems, int num_ranks,
                                         int sm_id = 0, int num_sms = 1) {
        num_bytes = num_elems * static_cast<int64_t>(sizeof(dtype_t));

        int64_t per_channel_bytes = num_bytes * num_ranks;
        total_bytes = per_channel_bytes * num_sms * (static_cast<int>(kDecoupled) + 1);
        send_ptr    = reinterpret_cast<uint8_t *>(gbl_ptr) + per_channel_bytes * sm_id;
        recv_ptr    = reinterpret_cast<uint8_t *>(gbl_ptr) + per_channel_bytes * (sm_id + num_sms);
        gbl_ptr     = reinterpret_cast<uint8_t *>(gbl_ptr) + total_bytes;
    }

    __device__ __forceinline__ dtype_t *send_buffer(int64_t idx = 0) {
        PRIMUS_TURBO_STATIC_CHECK(kDecoupled,
                                  "`send_buffer` is only available for non-decoupled case");
        return reinterpret_cast<dtype_t *>(send_ptr + num_bytes * idx);
    }

    __device__ __forceinline__ dtype_t *recv_buffer(int64_t idx = 0) {
        PRIMUS_TURBO_STATIC_CHECK(kDecoupled,
                                  "`recv_buffer` is only available for non-decoupled case");
        return reinterpret_cast<dtype_t *>(recv_ptr + num_bytes * idx);
    }

    __device__ __forceinline__ dtype_t *buffer(int64_t idx = 0) {
        PRIMUS_TURBO_STATIC_CHECK(not kDecoupled, "`buffer` is only available for decoupled case");
        return reinterpret_cast<dtype_t *>(send_ptr + num_bytes * idx);
    }
};

} // namespace primus_turbo::deep_ep
