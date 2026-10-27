/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// DeepEP intranode dispatch / combine, V2: direct writes into fixed slots (gfx1250 only).
//
// V1 streams every (channel, rank) pair through a ring buffer, one chunk per head / tail handshake,
// and the receiver copies the ring into `recv_x`. V2 drops the rings: every token has a fixed slot
// on each rank it goes to, so a sender writes it there directly and never waits for a receiver.
//
// Dispatch. A token's slot on rank `d` is our rank's offset in `d`'s receive order (the rank prefix
// matrix of the notify) plus its index in our stream of tokens to `d` -- the order the V1 receiver
// produces, so `recv_x` and every per-token output match V1 exactly. Each block owns one channel (a
// contiguous token range); it scans the range a tile at a time for the per-destination indices
// (also returned as `send_head`), then each warp loads a token's row once and stores it to every
// rank the token goes to: our own rank's straight into the outputs, the others' into their heap.
// When its rows are visible, the block tells each rank how many rows its channel sent there; then
// it copies, from our heap into the outputs, the rows the same channel of every other rank sent us,
// as soon as that rank's count is in.
//
// Combine. The rows `x` of the dispatch's receive order go back to their source rank, each with its
// top-k weights, into a compact region laid out as [destination rank][index in the source's stream
// to it], so the slot of (rank `r`, token `t`) on the source is `base_r + send_head[t, r]`; our own
// rows stay in `x`. Once a row has landed its pusher raises the slot's flag to this combine's
// epoch. Each rank sums the rows of its own tokens (biases first, then ranks in ascending order, as
// the V1 kernel adds them) as soon as their flags are up: warps claim tokens in order from a queue,
// some from the start of the kernel and the pushers once they are done, so the reduce overlaps the
// push.
//
// Rows move through LDS with the Tensor Data Mover: one load, a store per destination. Ordering is
// the V1 cheap fence: the heap is uncached, payload stores carry SCOPE_SYS, a warp waits for its
// own stores before it publishes a count or raises a flag, and readers load at system scope.

#include <algorithm>

#include "hip/hip_runtime.h"
#include "launch.cuh"
#include "primus_turbo/common.h"
#include "primus_turbo/deep_ep/api.h"
#include "primus_turbo/deep_ep/configs.h"
#include "utils.cuh"

namespace primus_turbo::deep_ep::intranode_v2 {

namespace {

constexpr int kNumThreads  = 1024;
constexpr int kV2WarpSize  = 32;
constexpr int kNumWarps    = kNumThreads / kV2WarpSize;
constexpr int kMaxLdsBytes = 320 * 1024;

// what a (source rank, channel) stream publishes to its destination:
// `epoch << 20 | done << 19 | rows sent so far`
constexpr int kTailEpochShift = 20;
constexpr int kTailDoneBit    = 1 << 19;
constexpr int kTailRowsMask   = kTailDoneBit - 1;

// combine: tokens a warp claims from the reduce queue at a time
constexpr int kReduceGrabTokens = 4;

// gfx12 cache policy: TH_NT (1) | SCOPE_SYS (3 << 3)
constexpr int kSysCPol = (3 << 3) | 1;

struct DispatchLayout {
    int64_t x, src_idx, topk_idx, topk_weights, x_scales;
    int64_t capacity;
};

int64_t align_up(int64_t value, int64_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

int64_t half_bytes(int64_t num_nvl_bytes) {
    return (num_nvl_bytes - kDataOffset) / 2 / 4096 * 4096;
}

DispatchLayout dispatch_layout(int64_t num_nvl_bytes, int hidden_bytes, int num_topk,
                               int num_scales) {
    const int64_t half     = half_bytes(num_nvl_bytes);
    const int64_t slots[5] = {hidden_bytes, 4, num_topk * 8, num_topk * 4, num_scales * 4};
    int64_t       sum      = 0;
    for (auto bytes : slots)
        sum += bytes;
    DispatchLayout layout;
    layout.capacity    = std::max<int64_t>(0, (half - 16 * 5) / sum);
    int64_t *fields[5] = {&layout.x, &layout.src_idx, &layout.topk_idx, &layout.topk_weights,
                          &layout.x_scales};
    int64_t  offset    = kDataOffset;
    for (int i = 0; i < 5; ++i) {
        *fields[i] = offset;
        offset     = align_up(offset + layout.capacity * slots[i], 16);
    }
    return layout;
}

// A combine slot: the row, then its top-k weights, so one TDM store moves both; slots start on
// 128-byte boundaries, as TDM stores want.
int combine_slot_bytes(int hidden_bytes, int num_topk) {
    return static_cast<int>(align_up(hidden_bytes + num_topk * 4, 128));
}

// Warps of each combine block that push before they reduce; the others reduce from the start. A
// grid of 96+ blocks saturates the links with about 2/3 of its warps pushing, and the rest overlap
// the reduce with the push; a narrower one is bound by how fast each CU sends, so all push.
int combine_push_warps(int num_channels, int slot_bytes) {
    const int warps = num_channels >= 96 ? kNumWarps * 11 / 16 : kNumWarps;
    // each pusher holds a whole slot in LDS
    return std::min(warps, kMaxLdsBytes / slot_bytes);
}

#if defined(__gfx1250__)

typedef unsigned int tdm_u32x4_t __attribute__((ext_vector_type(4)));
typedef int          tdm_i32x8_t __attribute__((ext_vector_type(8)));
typedef int          tdm_i32x4_t __attribute__((ext_vector_type(4)));

__device__ __forceinline__ int uniform(int value) {
    return __builtin_amdgcn_readfirstlane(value);
}

__device__ __forceinline__ uint64_t uniform_addr(const void *ptr) {
    const auto addr = reinterpret_cast<uint64_t>(ptr);
    const auto lo   = static_cast<uint32_t>(uniform(static_cast<int>(addr)));
    const auto hi   = static_cast<uint32_t>(uniform(static_cast<int>(addr >> 32)));
    return (static_cast<uint64_t>(hi) << 32) | lo;
}

// The LDS byte address of a pointer into shared memory.
__device__ __forceinline__ uint32_t lds_addr(const void *ptr) {
    return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(ptr));
}

// One TDM transfer of `num_bytes` (a multiple of 8, under 512 KiB) between LDS `lds` and global
// `global`, as a 1 x n tile of 8-byte elements; every operand must be wave-uniform.
template <bool kStore, int kCPol>
__device__ __forceinline__ void tdm_row(uint32_t lds, uint64_t global, int num_bytes) {
    const int         elems  = num_bytes / 8;
    const tdm_u32x4_t group0 = {1u, lds, static_cast<uint32_t>(global),
                                static_cast<uint32_t>(global >> 32) | 0x80000000u};
    const tdm_i32x8_t group1 = {3 << 16,                   // 8-byte elements
                                (elems & 0xffff) << 16,    // tensor dim0 [15:0]
                                (elems >> 16) | (1 << 16), // tensor dim0 [31:16], dim1 = 1
                                elems << 16,               // tile dim0
                                1,                         // tile dim1
                                elems,                     // dim0 stride
                                0,
                                0};
    const tdm_i32x4_t zero4  = {0, 0, 0, 0};
    const tdm_i32x8_t zero8  = {0, 0, 0, 0, 0, 0, 0, 0};
    if constexpr (kStore)
        __builtin_amdgcn_tensor_store_from_lds(group0, group1, zero4, zero4, zero8, kCPol);
    else
        __builtin_amdgcn_tensor_load_to_lds(group0, group1, zero4, zero4, zero8, kCPol);
}

__device__ __forceinline__ void wait_tensor() {
    __atomic_signal_fence(__ATOMIC_SEQ_CST);
    __builtin_amdgcn_s_wait_tensorcnt(0);
    __atomic_signal_fence(__ATOMIC_SEQ_CST);
}

// this wave's LDS accesses are done, e.g. before a TDM store reads what it wrote
__device__ __forceinline__ void wait_lds() {
    asm volatile("s_wait_dscnt 0x0" ::: "memory");
}

__device__ __forceinline__ void wait_all_memory() {
    asm volatile(PRIMUS_TURBO_WAIT_ALL_STR ::: "memory");
}

// System-scope accesses as global (not flat) instructions: a flat access also counts on the LDS
// counter, so a wait for this wave's LDS accesses would wait for it to reach the peer as well.
template <typename T> __device__ __forceinline__ T ld_sys(const T *ptr) {
    return __hip_atomic_load((const __attribute__((address_space(1))) T *) ptr, __ATOMIC_RELAXED,
                             __HIP_MEMORY_SCOPE_SYSTEM);
}

template <typename T> __device__ __forceinline__ void st_sys(T *ptr, T value) {
    __hip_atomic_store((__attribute__((address_space(1))) T *) ptr, value, __ATOMIC_RELAXED,
                       __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__ __forceinline__ uint32_t ballot(bool pred) {
    return __builtin_amdgcn_ballot_w32(pred);
}

// Element `k` of a 16-byte vector of `dtype_t`, as float. Works on the vector's words rather than
// through a pointer to it, which would keep the vector out of registers.
template <typename dtype_t> __device__ __forceinline__ float vec_elem(const int4 &v, int k) {
    constexpr int kPerWord = static_cast<int>(sizeof(int) / sizeof(dtype_t));
    const int     i        = k / kPerWord;
    const int     word     = i == 0 ? v.x : (i == 1 ? v.y : (i == 2 ? v.z : v.w));
    if constexpr (kPerWord == 2)
        return __uint_as_float(k % 2 ? static_cast<uint32_t>(word) & 0xffff0000u
                                     : static_cast<uint32_t>(word) << 16);
    else
        return __int_as_float(word);
}

// hip_bfloat16's float conversion (round to nearest even, signaling NaNs kept), as V1 casts
__device__ __forceinline__ uint32_t bf16_bits(float value) {
    uint32_t bits = __float_as_uint(value);
    if (~bits & 0x7f800000u)
        bits += 0x7fffu + ((bits >> 16) & 1u);
    else if (bits & 0xffffu)
        bits |= 0x10000u;
    return bits >> 16;
}

template <typename dtype_t, int kElems>
__device__ __forceinline__ int4 vec_pack(const float (&values)[kElems]) {
    int words[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        if constexpr (sizeof(dtype_t) == 2)
            words[i] =
                static_cast<int>(bf16_bits(values[2 * i]) | (bf16_bits(values[2 * i + 1]) << 16));
        else
            words[i] = __float_as_int(values[i]);
    }
    return make_int4(words[0], words[1], words[2], words[3]);
}

#endif // __gfx1250__

template <typename Kernel> void set_dynamic_lds(Kernel kernel, int bytes) {
    PRIMUS_TURBO_CHECK_HIP(hipFuncSetAttribute(reinterpret_cast<const void *>(kernel),
                                               hipFuncAttributeMaxDynamicSharedMemorySize, bytes));
}

} // namespace

// Dispatch

template <int kNumRanks> constexpr int dispatch_lds_words() {
    // [kNumRanks] running stream indices, [kNumRanks] the channel's first stream indices,
    // [kNumWarps][kNumRanks] warp bases, [kNumThreads][kNumRanks] slots of the tile, and per source
    // rank its received stream of this channel: rows copied, rows arrived, done, first row
    return 2 * kNumRanks + kNumWarps * kNumRanks + kNumThreads * kNumRanks + 4 * kNumRanks;
}

template <int kNumRanks>
__global__ void __launch_bounds__(kNumThreads, 1)
    dispatch_kernel(uint8_t *recv_x, int *recv_x_scales, int *recv_src_idx, int64_t *recv_topk_idx,
                    float *recv_topk_weights, int *recv_channel_offset, int *send_head,
                    const uint8_t *x, const int *x_scales, const int64_t *topk_idx,
                    const float *topk_weights, const bool *is_token_in_rank,
                    const int *channel_prefix_matrix, const int *rank_prefix_matrix,
                    void **buffer_ptrs, DispatchLayout layout, int num_tokens, int num_worst_tokens,
                    int hidden_bytes, int num_topk, int num_experts, int num_scales,
                    int scale_token_stride, int scale_hidden_stride, int rank, int epoch,
                    int lds_rows, int row_buffer_bytes) {
#if defined(__gfx1250__)
    extern __shared__ __align__(128) int smem[];
    constexpr int                        kLdsRunning   = 0;
    constexpr int                        kLdsChanStart = kNumRanks;
    constexpr int                        kLdsWarp      = 2 * kNumRanks;
    constexpr int                        kLdsSlot      = kLdsWarp + kNumWarps * kNumRanks;
    constexpr int                        kLdsRecv      = kLdsSlot + kNumThreads * kNumRanks;
    int                                 *copied        = smem + kLdsRecv;
    int                                 *arrived       = copied + kNumRanks;
    int                                 *done          = arrived + kNumRanks;
    int                                 *first_row     = done + kNumRanks;

    const int thread_id = static_cast<int>(threadIdx.x), lane_id = thread_id % kV2WarpSize;
    const int warp_id = uniform(thread_id / kV2WarpSize);
    const int channel = static_cast<int>(blockIdx.x), num_channels = static_cast<int>(gridDim.x);
    const int num_experts_per_rank = num_topk > 0 ? num_experts / kNumRanks : 0;
    auto     *local                = static_cast<uint8_t *>(buffer_ptrs[rank]);
    int       token_start, token_end;
    get_channel_task_range(num_tokens, num_channels, channel, token_start, token_end);

    // the channel's first index in our stream to each rank, published to that rank before any of
    // its rows; and nothing received yet (our own rows go to the outputs directly)
    if (thread_id < kNumRanks) {
        const int start =
            channel > 0 ? channel_prefix_matrix[thread_id * num_channels + channel - 1] : 0;
        smem[kLdsRunning + thread_id]   = start;
        smem[kLdsChanStart + thread_id] = start;
        const int idx                   = rank * num_channels + channel;
        if (thread_id == rank)
            recv_channel_offset[idx] = start;
        else
            st_sys(reinterpret_cast<int *>(static_cast<uint8_t *>(buffer_ptrs[thread_id]) +
                                           kRecvChannelOffsetOffset) +
                       idx,
                   start);
        copied[thread_id]    = 0;
        arrived[thread_id]   = 0;
        done[thread_id]      = thread_id == rank;
        first_row[thread_id] = -1;
    }
    __syncthreads();

    // per destination: where our rank's tokens start in its receive order, and where each output
    // goes -- its heap, or for our own rank the output tensors themselves
    uint8_t *dst_x[kNumRanks];
    int     *dst_src_idx[kNumRanks];
    int64_t *dst_topk_idx[kNumRanks];
    float   *dst_topk_weights[kNumRanks];
    int     *dst_x_scales[kNumRanks];
    int      rank_offsets[kNumRanks];
#pragma unroll
    for (int d = 0; d < kNumRanks; ++d) {
        auto      *peer   = static_cast<uint8_t *>(buffer_ptrs[d]);
        const bool is_own = d == rank;
        dst_x[d]          = is_own ? recv_x : peer + layout.x;
        dst_src_idx[d]    = is_own ? recv_src_idx : reinterpret_cast<int *>(peer + layout.src_idx);
        dst_topk_idx[d] =
            is_own ? recv_topk_idx : reinterpret_cast<int64_t *>(peer + layout.topk_idx);
        dst_topk_weights[d] =
            is_own ? recv_topk_weights : reinterpret_cast<float *>(peer + layout.topk_weights);
        dst_x_scales[d] = is_own ? recv_x_scales : reinterpret_cast<int *>(peer + layout.x_scales);
        rank_offsets[d] = rank > 0 ? rank_prefix_matrix[(rank - 1) * kNumRanks + d] : 0;
    }
    auto          *row_buffer = reinterpret_cast<uint8_t *>(smem) + lds_rows;
    const uint32_t row_lds    = uniform(lds_addr(row_buffer) + warp_id * row_buffer_bytes);

    for (int tile_start = token_start; tile_start < token_end; tile_start += kNumThreads) {
        // ---- the stream index of each token of the tile, per destination ----
        const int  token      = tile_start + thread_id;
        const bool is_valid   = token < token_end;
        const int  safe_token = is_valid ? token : 0;
        bool       flags[kNumRanks];
        int        belows[kNumRanks];
#pragma unroll
        for (int d = 0; d < kNumRanks; ++d) {
            flags[d] =
                is_valid and is_token_in_rank[static_cast<int64_t>(safe_token) * kNumRanks + d];
            const uint32_t mask = ballot(flags[d]);
            if (lane_id == 0)
                smem[kLdsWarp + warp_id * kNumRanks + d] = __popc(mask);
            belows[d] = __builtin_amdgcn_mbcnt_lo(mask, 0);
        }
        __syncthreads();
        if (thread_id < kNumRanks) {
            int base = smem[kLdsRunning + thread_id];
            for (int w = 0; w < kNumWarps; ++w) {
                const int count = smem[kLdsWarp + w * kNumRanks + thread_id];
                smem[kLdsWarp + w * kNumRanks + thread_id] = base;
                base += count;
            }
            smem[kLdsRunning + thread_id] = base;
        }
        __syncthreads();
#pragma unroll
        for (int d = 0; d < kNumRanks; ++d) {
            const int slot = flags[d] ? smem[kLdsWarp + warp_id * kNumRanks + d] + belows[d] : -1;
            smem[kLdsSlot + thread_id * kNumRanks + d] = slot;
            if (is_valid)
                send_head[static_cast<int64_t>(token) * kNumRanks + d] = slot;
        }
        __syncthreads();

        // ---- one warp per token: its row loaded once and stored to every rank it goes to ----
        const int num_tile_tokens = min(token_end - tile_start, kNumThreads);
        for (int i = warp_id; i < num_tile_tokens; i += kNumWarps) {
            const int t = tile_start + i;
            int       rows[kNumRanks];
            bool      routed[kNumRanks];
#pragma unroll
            for (int d = 0; d < kNumRanks; ++d) {
                const int slot = uniform(smem[kLdsSlot + i * kNumRanks + d]);
                routed[d]      = slot >= 0;
                rows[d]        = rank_offsets[d] + (routed[d] ? slot : 0);
            }
            const uint8_t *src = x + static_cast<int64_t>(t) * hidden_bytes;
            for (int p = 0; p < hidden_bytes; p += row_buffer_bytes) {
                const int piece = min(row_buffer_bytes, hidden_bytes - p);
                // the previous stores are done with the buffer
                wait_tensor();
                tdm_row<false, 0>(row_lds, uniform_addr(src + p), piece);
                wait_tensor();
#pragma unroll
                for (int d = 0; d < kNumRanks; ++d)
                    if (routed[d])
                        tdm_row<true, kSysCPol>(
                            row_lds,
                            uniform_addr(dst_x[d] + static_cast<int64_t>(rows[d]) * hidden_bytes +
                                         p),
                            piece);
            }

#pragma unroll
            for (int d = 0; d < kNumRanks; ++d) {
                if (not routed[d])
                    continue;
                if (lane_id == 0)
                    st_sys(dst_src_idx[d] + rows[d], t);
                if (lane_id < num_topk) {
                    const int64_t elem     = static_cast<int64_t>(t) * num_topk + lane_id;
                    const int64_t dst_elem = static_cast<int64_t>(rows[d]) * num_topk + lane_id;
                    const int64_t expert   = topk_idx[elem];
                    const int64_t begin    = static_cast<int64_t>(d) * num_experts_per_rank;
                    const bool is_local = expert >= begin and expert < begin + num_experts_per_rank;
                    st_sys(dst_topk_idx[d] + dst_elem, is_local ? expert - begin : int64_t(-1));
                    st_sys(dst_topk_weights[d] + dst_elem, is_local ? topk_weights[elem] : 0.0f);
                }
                for (int k = lane_id; k < num_scales; k += kV2WarpSize)
                    st_sys(dst_x_scales[d] + static_cast<int64_t>(rows[d]) * num_scales + k,
                           x_scales[static_cast<int64_t>(t) * scale_token_stride +
                                    k * scale_hidden_stride]);
            }
        }
    }

    // every warp's rows are visible: our streams are done
    wait_tensor();
    wait_all_memory();
    __syncthreads();
    if (thread_id < kNumRanks and thread_id != rank)
        st_sys(
            reinterpret_cast<int *>(static_cast<uint8_t *>(buffer_ptrs[thread_id]) + kTailsOffset) +
                rank * num_channels + channel,
            (epoch << kTailEpochShift) + kTailDoneBit + smem[kLdsRunning + thread_id] -
                smem[kLdsChanStart + thread_id]);

    // ---- the rest of what this channel receives ----
    auto start_time = timeout_clock();
    while (true) {
        // thread `s` reads how far source `s`'s stream of this channel has got; the first time,
        // also where that stream starts in our receive order
        if (thread_id < kNumRanks and thread_id != rank) {
            const int idx   = thread_id * num_channels + channel;
            const int value = ld_sys(reinterpret_cast<const int *>(local + kTailsOffset) + idx);
            if ((value >> kTailEpochShift) == epoch) {
                if (first_row[thread_id] < 0) {
                    const int offset = ld_sys(
                        reinterpret_cast<const int *>(local + kRecvChannelOffsetOffset) + idx);
                    const int prev =
                        thread_id > 0 ? rank_prefix_matrix[(thread_id - 1) * kNumRanks + rank] : 0;
                    first_row[thread_id]     = prev + offset;
                    recv_channel_offset[idx] = offset;
                }
                arrived[thread_id] = value & kTailRowsMask;
                done[thread_id]    = (value & kTailDoneBit) != 0;
            }
        }
        __syncthreads();

        // copy the rows that have arrived and are not copied yet, with their metadata
        int cums[kNumRanks], starts[kNumRanks];
        int total = 0;
#pragma unroll
        for (int s = 0; s < kNumRanks; ++s) {
            cums[s]   = total;
            starts[s] = first_row[s] + copied[s];
            total += arrived[s] - copied[s];
        }
        for (int j = warp_id; j < total; j += kNumWarps) {
            int row = starts[0] + j;
#pragma unroll
            for (int s = 1; s < kNumRanks; ++s)
                row = j >= cums[s] ? starts[s] + j - cums[s] : row;
            const uint8_t *src = local + layout.x + static_cast<int64_t>(row) * hidden_bytes;
            uint8_t       *dst = recv_x + static_cast<int64_t>(row) * hidden_bytes;
            for (int p = 0; p < hidden_bytes; p += row_buffer_bytes) {
                const int piece = min(row_buffer_bytes, hidden_bytes - p);
                wait_tensor();
                tdm_row<false, kSysCPol>(row_lds, uniform_addr(src + p), piece);
                wait_tensor();
                tdm_row<true, 0>(row_lds, uniform_addr(dst + p), piece);
            }
            if (lane_id == 0)
                recv_src_idx[row] =
                    ld_sys(reinterpret_cast<const int *>(local + layout.src_idx) + row);
            if (lane_id < num_topk) {
                const int64_t elem = static_cast<int64_t>(row) * num_topk + lane_id;
                recv_topk_idx[elem] =
                    ld_sys(reinterpret_cast<const int64_t *>(local + layout.topk_idx) + elem);
                recv_topk_weights[elem] =
                    ld_sys(reinterpret_cast<const float *>(local + layout.topk_weights) + elem);
            }
            for (int k = lane_id; k < num_scales; k += kV2WarpSize) {
                const int64_t elem = static_cast<int64_t>(row) * num_scales + k;
                recv_x_scales[elem] =
                    ld_sys(reinterpret_cast<const int *>(local + layout.x_scales) + elem);
            }
        }
        __syncthreads();
        if (thread_id < kNumRanks)
            copied[thread_id] = arrived[thread_id];
        __syncthreads();
        int remaining = 0;
#pragma unroll
        for (int s = 0; s < kNumRanks; ++s)
            remaining += 1 - done[s];
        // everyone has read the counts before the next poll rewrites them
        __syncthreads();
        if (remaining == 0)
            break;
        if (timeout_clock() - start_time > NUM_TIMEOUT_CYCLES) {
            if (thread_id == 0)
                printf("DeepEP V2 dispatch receive timeout: rank %d, channel %d, sources left %d\n",
                       rank, channel, remaining);
            trap();
        }
    }
    wait_tensor();

    // with a worst-case allocation, the rows past the received ones read as unrouted
    if (num_topk > 0 and num_worst_tokens > 0) {
        const int num_recv_tokens = rank_prefix_matrix[(kNumRanks - 1) * kNumRanks + rank];
        for (int64_t i = static_cast<int64_t>(num_recv_tokens) * num_topk + channel * kNumThreads +
                         thread_id;
             i < static_cast<int64_t>(num_worst_tokens) * num_topk;
             i += static_cast<int64_t>(num_channels) * kNumThreads)
            recv_topk_idx[i] = -1;
    }
#endif // __gfx1250__
}

// Combine

template <int kNumRanks, typename dtype_t, bool kHasBias>
__global__ void __launch_bounds__(kNumThreads, 1)
    combine_kernel(dtype_t *recv_x, float *recv_topk_weights, const dtype_t *x,
                   const float *topk_weights, const dtype_t *bias_0, const dtype_t *bias_1,
                   const int *rank_prefix_matrix, const int *send_head, void **buffer_ptrs,
                   int *reduce_counter, int num_rows, int num_recv_tokens, int hidden_bytes,
                   int num_topk, int rank, int epoch, uint32_t grab_base, int push_warps,
                   int64_t region_offset, int slot_bytes) {
#if defined(__gfx1250__)
    extern __shared__ __align__(128) int smem[];
    constexpr int kVecElems = static_cast<int>(sizeof(int4) / sizeof(dtype_t));
    // 16 vectors of rows in flight per lane: more would not fit 1024-thread blocks' 128 VGPRs
    constexpr int kReduceUnroll = 16 / kNumRanks;

    const int   thread_id = static_cast<int>(threadIdx.x), lane_id = thread_id % kV2WarpSize;
    const int   warp_id  = uniform(thread_id / kV2WarpSize);
    const int   block_id = static_cast<int>(blockIdx.x), num_channels = static_cast<int>(gridDim.x);
    auto       *local       = static_cast<uint8_t *>(buffer_ptrs[rank]);
    const auto *x_bytes     = reinterpret_cast<const uint8_t *>(x);
    auto        rank_prefix = [&](int s, int r) {
        return s < 0 ? 0 : rank_prefix_matrix[s * kNumRanks + r];
    };

    // per source rank s: where its rows of x begin, and where our rank's rows start in s's region
    // (the tokens s sends to the ranks before ours)
    int row_begins[kNumRanks + 1], bases[kNumRanks];
#pragma unroll
    for (int s = 0; s < kNumRanks; ++s) {
        row_begins[s] = rank_prefix(s - 1, rank);
        int base      = 0;
#pragma unroll
        for (int r = 0; r < kNumRanks; ++r)
            base += r < rank ? rank_prefix(s, r) - rank_prefix(s - 1, r) : 0;
        bases[s] = base;
    }
    row_begins[kNumRanks] = num_rows;

    // ---- push (the first push_warps warps of each block): each row of x back to its source rank,
    // but for our own rows, which the reduce reads from x; a row's flag goes up once the row is
    // visible. Rows are grouped by source, so a warp sticks to one source: walking them in order
    // would have the whole GPU writing to one peer, over one link, at a time ----
    const int num_peers      = kNumRanks - 1;
    const int warps_per_peer = num_channels * push_warps / num_peers;
    const int global_warp    = block_id * push_warps + warp_id;
    int       warp_source    = global_warp % num_peers;
    warp_source += warp_source >= rank ? 1 : 0;
    uint8_t *peer      = nullptr;
    int      row_begin = 0, row_end = 0, base = 0;
#pragma unroll
    for (int s = 0; s < kNumRanks; ++s) {
        if (warp_source == s) {
            peer      = static_cast<uint8_t *>(buffer_ptrs[s]);
            row_begin = row_begins[s];
            row_end   = row_begins[s + 1];
            base      = bases[s];
        }
    }
    const bool     is_pusher  = warp_id < push_warps and global_warp < warps_per_peer * num_peers;
    int            row        = is_pusher ? row_begin + global_warp / num_peers : row_end;
    const int      push_bytes = static_cast<int>((hidden_bytes + num_topk * 4 + 7) / 8 * 8);
    auto          *row_buffer = reinterpret_cast<uint8_t *>(smem) + warp_id * slot_bytes;
    const uint32_t row_lds    = uniform(lds_addr(row_buffer));
    int           *peer_flags = reinterpret_cast<int *>(peer + kCombineFlagsOffset);
    int            pushed     = -1;
    while (row < row_end) {
        const int slot = row - row_begin + base;
        // the previous row has landed, and is done with the LDS buffer
        wait_tensor();
        float weight = 0.0f;
        if (num_topk > 0)
            weight = topk_weights[static_cast<int64_t>(row) * num_topk +
                                  (lane_id < num_topk ? lane_id : 0)];
        tdm_row<false, 0>(row_lds, uniform_addr(x_bytes + static_cast<int64_t>(row) * hidden_bytes),
                          hidden_bytes);
        if (num_topk > 0) {
            if (lane_id < num_topk)
                reinterpret_cast<float *>(row_buffer + hidden_bytes)[lane_id] = weight;
            wait_lds();
        }
        wait_tensor();
        tdm_row<true, kSysCPol>(
            row_lds, uniform_addr(peer + region_offset + static_cast<int64_t>(slot) * slot_bytes),
            push_bytes);
        // raise the previous row's flag only now: the store waits to be sent before the warp goes
        // on, which this row's transfer hides
        if (pushed >= 0 and lane_id == 0)
            st_sys(peer_flags + pushed, epoch);
        pushed = slot;
        row += warps_per_peer;
    }
    wait_tensor();
    if (pushed >= 0 and lane_id == 0)
        st_sys(peer_flags + pushed, epoch);

    // ---- reduce: each of our tokens over the ranks it went to, as soon as its rows are in; warps
    // claim tokens in order from a queue, the pushers once they are done ----
    // where rank r's rows start in our region: the tokens we sent to the ranks before r
    int own_bases[kNumRanks];
    int running = 0;
#pragma unroll
    for (int r = 0; r < kNumRanks; ++r) {
        own_bases[r] = running;
        running += rank_prefix(rank, r) - rank_prefix(rank - 1, r);
    }
    // lane l looks after rank l % kNumRanks of the claim's token l / kNumRanks
    const int lane_r    = lane_id % kNumRanks;
    int       lane_base = own_bases[0];
#pragma unroll
    for (int r = 1; r < kNumRanks; ++r)
        lane_base = lane_r == r ? own_bases[r] : lane_base;
    const int *local_flags = reinterpret_cast<const int *>(local + kCombineFlagsOffset);
    const int  num_vecs    = hidden_bytes / static_cast<int>(sizeof(int4));

    auto grab = [&]() {
        int value = 0;
        if (lane_id == 0)
            value = atomicAdd(reduce_counter, 1);
        value = uniform(value);
        return static_cast<int>(static_cast<uint32_t>(value) - grab_base) * kReduceGrabTokens;
    };
    for (int first = grab(); first < num_recv_tokens; first = grab()) {
        const int token_end = min(first + kReduceGrabTokens, num_recv_tokens);
        // the claim's send_head entries in one load, and its rows' flags in one wait
        const bool in_claim = lane_id < (token_end - first) * kNumRanks;
        const int  index =
            in_claim ? send_head[static_cast<int64_t>(first) * kNumRanks + lane_id] : -1;
        const bool waits      = lane_r != rank and index >= 0;
        const int *flag       = local_flags + (waits ? lane_base + index : 0);
        auto       spin_start = timeout_clock();
        while (ballot(waits and ld_sys(flag) != epoch) != 0) {
            if (timeout_clock() - spin_start > NUM_TIMEOUT_CYCLES) {
                if (waits)
                    printf("DeepEP V2 combine row timeout: rank %d, token %d, lane %d\n", rank,
                           first, lane_id);
                trap();
            }
        }

        for (int token = first; token < token_end; ++token) {
            const int      claim_lane = (token - first) * kNumRanks;
            const uint8_t *rows[kNumRanks];
            const float   *weights[kNumRanks];
            bool           routed[kNumRanks];
#pragma unroll
            for (int r = 0; r < kNumRanks; ++r) {
                const int index_r = __builtin_amdgcn_readlane(index, claim_lane + r);
                routed[r]         = index_r >= 0;
                // our own rows were not pushed: they are in x
                const bool is_own = r == rank;
                const int  slot =
                    (is_own ? row_begins[r] : own_bases[r]) + (routed[r] ? index_r : 0);
                const uint8_t *heap_slot =
                    local + region_offset + static_cast<int64_t>(slot) * slot_bytes;
                rows[r] = is_own ? x_bytes + static_cast<int64_t>(slot) * hidden_bytes : heap_slot;
                weights[r] = is_own ? topk_weights + static_cast<int64_t>(slot) * num_topk
                                    : reinterpret_cast<const float *>(heap_slot + hidden_bytes);
            }
            device::BufferSRD row_srds[kNumRanks];
#pragma unroll
            for (int r = 0; r < kNumRanks; ++r)
                row_srds[r] = device::BufferSRD(rows[r], routed[r] ? hidden_bytes : 0);
            const auto *bias_0_row = bias_0 == nullptr ? nullptr
                                                       : reinterpret_cast<const int4 *>(bias_0) +
                                                             static_cast<int64_t>(token) * num_vecs;
            const auto *bias_1_row = bias_1 == nullptr ? nullptr
                                                       : reinterpret_cast<const int4 *>(bias_1) +
                                                             static_cast<int64_t>(token) * num_vecs;
            auto       *out_row =
                reinterpret_cast<int4 *>(recv_x) + static_cast<int64_t>(token) * num_vecs;

            for (int i0 = lane_id; i0 < num_vecs; i0 += kV2WarpSize * kReduceUnroll) {
                // every rank's loads in flight at once (an unrouted rank's empty range loads
                // nothing, and past the row the range returns zeros), added in rank order
                int4 recv_values[kNumRanks][kReduceUnroll];
#pragma unroll
                for (int r = 0; r < kNumRanks; ++r) {
#pragma unroll
                    for (int u = 0; u < kReduceUnroll; ++u) {
                        const auto value = llvm_amdgcn_raw_buffer_load_v4i32(
                            row_srds[r].srd,
                            (i0 + u * kV2WarpSize) * static_cast<int>(sizeof(int4)), 0, kSysCPol);
                        recv_values[r][u] = make_int4(value[0], value[1], value[2], value[3]);
                    }
                }
#pragma unroll
                for (int u = 0; u < kReduceUnroll; ++u) {
                    const int i = i0 + u * kV2WarpSize;
                    // V1 starts from the biases' sum, zero for a missing one
                    float values[kVecElems];
#pragma unroll
                    for (int k = 0; k < kVecElems; ++k)
                        values[k] = 0.0f;
                    if constexpr (kHasBias) {
                        int4 bias_a = make_int4(0, 0, 0, 0), bias_b = make_int4(0, 0, 0, 0);
                        if (i < num_vecs and bias_0_row != nullptr)
                            bias_a = bias_0_row[i];
                        if (i < num_vecs and bias_1_row != nullptr)
                            bias_b = bias_1_row[i];
#pragma unroll
                        for (int k = 0; k < kVecElems; ++k)
                            values[k] = vec_elem<dtype_t>(bias_a, k) + vec_elem<dtype_t>(bias_b, k);
                    }
#pragma unroll
                    for (int r = 0; r < kNumRanks; ++r) {
                        if (not routed[r])
                            continue;
#pragma unroll
                        for (int k = 0; k < kVecElems; ++k)
                            values[k] += vec_elem<dtype_t>(recv_values[r][u], k);
                    }
                    if (i < num_vecs)
                        out_row[i] = vec_pack<dtype_t>(values);
                }
            }
            if (lane_id < num_topk) {
                float weight = 0.0f;
#pragma unroll
                for (int r = 0; r < kNumRanks; ++r)
                    if (routed[r])
                        weight += ld_sys(weights[r] + lane_id);
                recv_topk_weights[static_cast<int64_t>(token) * num_topk + lane_id] = weight;
            }
        }
    }
#endif // __gfx1250__
}

void dispatch(void *recv_x, float *recv_x_scales, int *recv_src_idx, int64_t *recv_topk_idx,
              float *recv_topk_weights, int *recv_channel_offset, int *send_head, const void *x,
              const float *x_scales, const int64_t *topk_idx, const float *topk_weights,
              const bool *is_token_in_rank, const int *channel_prefix_matrix,
              const int *rank_prefix_matrix, int num_tokens, int num_recv_tokens,
              int num_worst_tokens, int hidden_bytes, int num_topk, int num_experts, int num_scales,
              int scale_token_stride, int scale_hidden_stride, void **buffer_ptrs,
              int64_t num_nvl_bytes, int rank, int num_ranks, int epoch, hipStream_t stream,
              int num_channels) {
    PRIMUS_TURBO_CHECK(num_channels > 0 and num_channels <= kMaxChannels);
    PRIMUS_TURBO_CHECK(hidden_bytes % static_cast<int>(sizeof(int4)) == 0);
    PRIMUS_TURBO_CHECK(num_topk <= kV2WarpSize);
    PRIMUS_TURBO_CHECK(epoch > 0 and epoch <= kMaxDispatchEpoch);
    PRIMUS_TURBO_CHECK(DIVUP(num_tokens, num_channels) < kTailDoneBit,
                       "DeepEP V2 dispatch takes fewer than 2^19 tokens per channel");
    const auto layout = dispatch_layout(num_nvl_bytes, hidden_bytes, num_topk, num_scales);
    PRIMUS_TURBO_CHECK(num_recv_tokens <= layout.capacity,
                       "DeepEP V2 dispatch receives more tokens than the NVL buffer holds; raise "
                       "num_nvl_bytes");

#define DISPATCH_V2_LAUNCH_CASE(ranks)                                                             \
    {                                                                                              \
        const int lds_rows   = static_cast<int>(align_up(dispatch_lds_words<ranks>() * 4, 128));   \
        const int row_buffer = static_cast<int>(std::min<int64_t>(                                 \
            align_up(hidden_bytes, 128), (kMaxLdsBytes - lds_rows) / kNumWarps / 128 * 128));      \
        const int smem_bytes = lds_rows + kNumWarps * row_buffer;                                  \
        set_dynamic_lds(dispatch_kernel<ranks>, smem_bytes);                                       \
        dispatch_kernel<ranks><<<num_channels, kNumThreads, smem_bytes, stream>>>(                 \
            static_cast<uint8_t *>(recv_x), reinterpret_cast<int *>(recv_x_scales), recv_src_idx,  \
            recv_topk_idx, recv_topk_weights, recv_channel_offset, send_head,                      \
            static_cast<const uint8_t *>(x), reinterpret_cast<const int *>(x_scales), topk_idx,    \
            topk_weights, is_token_in_rank, channel_prefix_matrix, rank_prefix_matrix,             \
            buffer_ptrs, layout, num_tokens, num_worst_tokens, hidden_bytes, num_topk,             \
            num_experts, num_scales, scale_token_stride, scale_hidden_stride, rank, epoch,         \
            lds_rows, row_buffer);                                                                 \
    }                                                                                              \
    break

    SWITCH_RANKS(DISPATCH_V2_LAUNCH_CASE);
#undef DISPATCH_V2_LAUNCH_CASE
}

uint32_t combine_grabs(int num_recv_tokens, int num_channels) {
    // every claim of tokens, plus the one claim per warp that finds the queue empty
    return static_cast<uint32_t>(DIVUP(num_recv_tokens, kReduceGrabTokens) +
                                 num_channels * kNumWarps);
}

void combine(hipDataType type, void *recv_x, float *recv_topk_weights, const void *x,
             const float *topk_weights, const void *bias_0, const void *bias_1,
             const int *rank_prefix_matrix, const int *send_head, int *reduce_counter,
             int num_tokens, int num_recv_tokens, int hidden, int num_topk, void **buffer_ptrs,
             int64_t num_nvl_bytes, int rank, int num_ranks, int epoch, uint32_t grab_base,
             hipStream_t stream, int num_channels) {
    const int elem_bytes   = type == HIP_R_16BF ? 2 : 4;
    const int hidden_bytes = hidden * elem_bytes;
    PRIMUS_TURBO_CHECK(hidden_bytes % static_cast<int>(sizeof(int4)) == 0);
    PRIMUS_TURBO_CHECK(num_topk <= kV2WarpSize and kReduceGrabTokens * num_ranks <= kV2WarpSize);
    PRIMUS_TURBO_CHECK(num_channels > 0 and epoch > 0);
    const int     slot_bytes    = combine_slot_bytes(hidden_bytes, num_topk);
    const int64_t half          = half_bytes(num_nvl_bytes);
    const int64_t region_offset = kDataOffset + half;
    // a token comes back from at most every rank
    PRIMUS_TURBO_CHECK(static_cast<int64_t>(num_recv_tokens) * num_ranks <=
                           (half - 16) / slot_bytes,
                       "DeepEP V2 combine receives more rows than the NVL buffer holds; raise "
                       "num_nvl_bytes");
    PRIMUS_TURBO_CHECK(static_cast<int64_t>(num_recv_tokens) * num_ranks * 4 <= kCombineFlagsBytes,
                       "DeepEP V2 combine has more rows than row flags");
    const int push_warps = combine_push_warps(num_channels, slot_bytes);
    PRIMUS_TURBO_CHECK(push_warps > 0 and num_channels * push_warps >= num_ranks - 1);
    const int  smem_bytes = push_warps * slot_bytes;
    const bool has_bias   = bias_0 != nullptr or bias_1 != nullptr;

#define COMBINE_V2_LAUNCH_CASE(dtype, ranks)                                                       \
    {                                                                                              \
        auto kernel =                                                                              \
            has_bias ? combine_kernel<ranks, dtype, true> : combine_kernel<ranks, dtype, false>;   \
        set_dynamic_lds(kernel, smem_bytes);                                                       \
        kernel<<<num_channels, kNumThreads, smem_bytes, stream>>>(                                 \
            static_cast<dtype *>(recv_x), recv_topk_weights, static_cast<const dtype *>(x),        \
            topk_weights, static_cast<const dtype *>(bias_0), static_cast<const dtype *>(bias_1),  \
            rank_prefix_matrix, send_head, buffer_ptrs, reduce_counter, num_tokens,                \
            num_recv_tokens, hidden_bytes, num_topk, rank, epoch, grab_base, push_warps,           \
            region_offset, slot_bytes);                                                            \
    }                                                                                              \
    break

#define COMBINE_V2_DTYPE_LAUNCH_CASE(dtype)                                                        \
    SWITCH_RANKS_WITH_DTYPE(dtype, COMBINE_V2_LAUNCH_CASE);                                        \
    break

    SWITCH_TYPES(COMBINE_V2_DTYPE_LAUNCH_CASE);
#undef COMBINE_V2_DTYPE_LAUNCH_CASE
#undef COMBINE_V2_LAUNCH_CASE
}

} // namespace primus_turbo::deep_ep::intranode_v2
