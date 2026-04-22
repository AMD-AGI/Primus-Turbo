#include "buffer.cuh"
#include "configs.cuh"
#include "exception.cuh"
#include "launch.cuh"
#include "utils.cuh"

namespace primus_turbo::cco::ep {
namespace intranode {

// ─────────────────────────────────────────────────────────────────────────────
// Shared-memory layout knobs for the optimized expert-grouped dispatch.
//
// The sender precomputes a per-(rank, group) bucketed list of in-rank token
// offsets ONCE per channel, then re-uses that list to ship one phase at a
// time without re-reading `topk_idx` or `is_token_in_rank` from global memory.
//
// kMaxLocalTokensPerSlice — upper bound on the number of in-rank tokens a
//   single channel can have for one destination rank. With the typical
//   workload (4096 tokens / 24 channels ≈ 170 per channel) this bound is
//   comfortably larger than the worst-case 1:1 routing.
//
// kMaxExpertGroupsPerRank — upper bound on `num_experts_per_group_per_rank`.
//   For 256 experts / 8 ranks / 4 experts-per-group = 8 groups in the
//   benchmark; 64 keeps headroom for finer pipeline granularity.
// ─────────────────────────────────────────────────────────────────────────────
constexpr int kMaxLocalTokensPerSlice = 1024;
constexpr int kMaxExpertGroupsPerRank = 64;

// Compute the "primary local expert" for a token with respect to a destination
// rank. A token is routed to a destination rank if any of its `num_topk` expert
// ids falls inside that rank's expert range. The primary local expert is the
// smallest local expert id among the matches — it uniquely determines the
// expert group in which the token will be shipped.
//
// Returns `num_experts_per_rank` as a sentinel when the token does not target
// this destination rank at all (the caller should have filtered the case via
// `is_token_in_rank`, but the sentinel keeps the helper self-contained).
__device__ __forceinline__ int compute_primary_local_expert(int64_t const *topk_idx,
                                                            int64_t token_idx, int num_topk,
                                                            int responsible_rank,
                                                            int num_experts_per_rank) {
    int       min_le       = num_experts_per_rank;
    int const expert_base  = responsible_rank * num_experts_per_rank;
    int const expert_limit = (responsible_rank + 1) * num_experts_per_rank;
#pragma unroll
    for (int k = 0; k < 32; ++k) {
        if (k >= num_topk)
            break;
        int e = static_cast<int>(__ldg(topk_idx + token_idx * num_topk + k));
        if (e >= expert_base && e < expert_limit) {
            int le = e - expert_base;
            if (le < min_le)
                min_le = le;
        }
    }
    return min_le;
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase 0 of the expert-grouped pipeline: build per-(dst_rank, channel)
// group buckets so the main dispatch kernel's sender can start shipping
// group 0 **immediately** on launch, without the 3-pass scan that used to
// happen inside each sender SM.
//
// Launch: `num_ranks * num_channels` blocks, one warp each. Each block
// classifies the tokens in ONE (dst_rank, channel) slice by primary local
// expert → group, builds the per-group prefix sum, and writes two arrays to
// global memory:
//
//   * group_offsets[dst_rank, channel, g]      int32, length
//       `kMaxExpertGroupsPerRank + 1`. `group_offsets[..,g+1] - [..,g]` is
//       the number of in-rank tokens in group `g`.
//   * sorted_token_offsets[dst_rank, channel, i] int16, length
//       `kMaxLocalTokensPerSlice`. Token slice offsets (relative to
//       `token_start_idx`) sorted by group.
//
// These arrays replace the shared-memory scratch that the dispatch kernel
// used to build on the hot path. Moving the work into a separate,
// massively parallel kernel (#(dst_rank, channel) blocks ≈ 192 on MI300X
// for R=8,C=24) cuts the per-(rank, channel) latency of the classification
// from 3 passes × 170 tokens ≈ 500 cycles to the launch-gap of the kernel
// preceding dispatch. On the dispatch side, every SM jumps straight to
// phase 0.
// ─────────────────────────────────────────────────────────────────────────────
template <int kNumRanks>
__global__ void __launch_bounds__(64, 1)
    expert_grouped_build_buckets(int64_t const *topk_idx, bool const *is_token_in_rank,
                                 int num_tokens, int num_channels, int num_topk, int num_experts,
                                 int num_experts_per_group, int *group_offsets,
                                 int16_t *sorted_token_offsets) {
    int const block_id    = static_cast<int>(blockIdx.x);
    int const dst_rank    = block_id / num_channels;
    int const channel_id  = block_id % num_channels;
    int const thread_id   = static_cast<int>(threadIdx.x);
    int const num_threads = static_cast<int>(blockDim.x);

    int const num_experts_per_rank = num_experts / kNumRanks;
    int const num_groups =
        (num_experts_per_rank + num_experts_per_group - 1) / num_experts_per_group;
    EP_DEVICE_ASSERT(num_groups <= kMaxExpertGroupsPerRank);

    int token_start_idx, token_end_idx;
    get_channel_task_range(num_tokens, num_channels, channel_id, token_start_idx, token_end_idx);
    int const slice_len = token_end_idx - token_start_idx;
    EP_DEVICE_ASSERT(slice_len <= kMaxLocalTokensPerSlice);

    __shared__ int s_count[kMaxExpertGroupsPerRank];
    __shared__ int s_offset[kMaxExpertGroupsPerRank + 1];
    __shared__ int s_fill[kMaxExpertGroupsPerRank];

    for (int g = thread_id; g < num_groups; g += num_threads) {
        s_count[g] = 0;
        s_fill[g]  = 0;
    }
    __syncthreads();

    // Pass 1: count per-group in-rank tokens for this (dst_rank, channel).
    for (int t = thread_id; t < slice_len; t += num_threads) {
        int64_t const token_idx = token_start_idx + t;
        if (not is_token_in_rank[token_idx * kNumRanks + dst_rank])
            continue;
        int const le = compute_primary_local_expert(topk_idx, token_idx, num_topk, dst_rank,
                                                    num_experts_per_rank);
        int const g  = le / num_experts_per_group;
        __hip_atomic_fetch_add(&s_count[g], 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
    }
    __syncthreads();

    // Pass 2: serial prefix-sum of the per-group counts into `s_offset`.
    if (thread_id == 0) {
        int prev = 0;
#pragma unroll
        for (int g = 0; g < kMaxExpertGroupsPerRank; ++g) {
            if (g >= num_groups)
                break;
            s_offset[g] = prev;
            prev += s_count[g];
        }
        s_offset[num_groups] = prev;
    }
    __syncthreads();

    // Pass 3: scatter sorted token offsets directly to global memory.
    int64_t const sorted_base =
        (static_cast<int64_t>(dst_rank) * num_channels + channel_id) * kMaxLocalTokensPerSlice;
    for (int t = thread_id; t < slice_len; t += num_threads) {
        int64_t const token_idx = token_start_idx + t;
        if (not is_token_in_rank[token_idx * kNumRanks + dst_rank])
            continue;
        int const le = compute_primary_local_expert(topk_idx, token_idx, num_topk, dst_rank,
                                                    num_experts_per_rank);
        int const g  = le / num_experts_per_group;
        int const slot =
            __hip_atomic_fetch_add(&s_fill[g], 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
        sorted_token_offsets[sorted_base + s_offset[g] + slot] = static_cast<int16_t>(t);
    }

    // Publish offsets to global memory. Only `num_groups + 1` entries per
    // (dst_rank, channel) are meaningful; the rest of the padded slot is
    // never read on the hot path.
    int64_t const offs_base = (static_cast<int64_t>(dst_rank) * num_channels + channel_id) *
                              (kMaxExpertGroupsPerRank + 1);
    for (int g = thread_id; g <= num_groups; g += num_threads) {
        group_offsets[offs_base + g] = s_offset[g];
    }
}

void expert_grouped_build_buckets(int64_t const *topk_idx, bool const *is_token_in_rank,
                                  int num_tokens, int num_channels, int num_topk, int num_experts,
                                  int num_experts_per_group, int num_ranks, int *group_offsets,
                                  int16_t *sorted_token_offsets, cudaStream_t stream) {
    constexpr int kNumThreads = 64; // one AMD warp
    EP_HOST_ASSERT(num_experts_per_group > 0);
    EP_HOST_ASSERT(num_channels > 0);

#define BUCKETS_LAUNCH_CASE(ranks)                                                                 \
    {                                                                                              \
        auto kernel = expert_grouped_build_buckets<ranks>;                                         \
        LAUNCH_KERNEL(&cfg, kernel, topk_idx, is_token_in_rank, num_tokens, num_channels,          \
                      num_topk, num_experts, num_experts_per_group, group_offsets,                 \
                      sorted_token_offsets);                                                       \
    }                                                                                              \
    break

    SETUP_LAUNCH_CONFIG(num_ranks * num_channels, kNumThreads, stream);
    SWITCH_RANKS(BUCKETS_LAUNCH_CASE);
#undef BUCKETS_LAUNCH_CASE
}

// Pipelined expert-grouped dispatch + permute (optimized).
//
// Phase order is preserved on every (src_rank, dst_rank, channel) stream:
// tokens whose primary local expert lies in group `g` are shipped during
// phase `g`. The receiver bumps `expert_tail_idx[e]` for every received
// token *batched per phase*, so a persistent GroupedGEMM consumer polling
// that counter can start computing experts in group 0 as soon as phase 0
// completes — the rest of dispatch then overlaps with compute.
//
// Optimizations vs the naive multi-pass implementation:
//   1.  Token classification (which group each in-rank token belongs to)
//       is moved **out** of this kernel entirely. A separate, per-
//       (dst_rank, channel) kernel ``expert_grouped_build_buckets`` runs
//       as part of ``notify_dispatch`` preprocessing, writing per-phase
//       offsets and sorted token lists to global memory. Every sender SM
//       here can therefore start shipping group 0 on the very first cycle
//       after launch — no scan, no prefix-sum, no per-thread shared-mem
//       scatter on the critical path.
//   2.  Each phase ships its full bucket as a single batch — the inner
//       `num_max_send_tokens` chunking and per-chunk barrier/tail-update
//       are dropped. We pay exactly one workgroup barrier and two
//       system-scoped stores (`channel_tail_idx` + `channel_phase_idx`)
//       per phase, instead of one per chunk.
//   3.  Sender reads the per-group `(begin, end)` offsets directly via
//       `__ldg` rather than staging `group_offsets_global` into LDS
//       cooperatively. This removes an extra `sync_barrier` that used to
//       sit on the path between kernel launch and the first phase-0
//       hidden copy — since phase 0's latency is what gates the
//       downstream group-0 GroupedGEMM, every cycle shaved here shows up
//       at the consumer.
//   4.  Receiver is **group-aware per phase**. The old receiver copied
//       all `num_topk` hidden slots per newly-arrived token in the phase
//       of arrival — work for experts in groups 1..G-1 piled up on the
//       group-0 tail flush. The new receiver instead, in phase g,
//       processes only the topk hits that fall in group g across ALL
//       tokens received so far (new + earlier-phase). Because sender
//       phase h never ships a token that hits groups 0..h-1 (primary =
//       smallest hit), this strictly partitions the copy work and lets
//       group-0's `expert_tail_idx[e]` flush as soon as the (small)
//       phase-0 in-group copies finish — roughly a `num_topk`× speedup
//       on group 0's portion of receiver latency, with the rest of the
//       copies folded transparently into phases 1..G-1.
//   5.  Receiver accumulates per-expert contributions in shared memory
//       while it copies a phase's tokens, then bumps `expert_tail_idx[e]`
//       in a single batched atomic per (rank, expert) per phase. This is
//       what allows the GroupedGEMM consumer to spin on
//       `expert_tail_idx[e]` without ping-ponging the cacheline against
//       thousands of in-flight per-token atomics from the receiver.
//   6.  Sender signals a dedicated `channel_phase_idx` counter at the
//       end of every phase (including empty ones). The receiver's phase
//       loop is driven by this counter instead of by `channel_tail_idx`
//       transitions, so it advances deterministically through G phases
//       even when a middle phase ships zero tokens.
template <int kNumRanks, int kNumThreads>
__global__ void __launch_bounds__(kNumThreads, 1) expert_grouped_dispatch_permute(
    void **buffer_ptrs, int *expert_tail_idx, int64_t *recv_topk_idx, float *recv_topk_weights,
    int *dispatch_to_expert_map, int4 const *x, float const *x_scales, int64_t const *topk_idx,
    float const *topk_weights, bool const *is_token_in_rank, int const *channel_prefix_matrix,
    int const *num_recv_tokens_per_expert, int const *group_offsets_global,
    int16_t const *sorted_token_offsets_global, int4 *recv_x, int num_tokens, int hidden_int4,
    int num_topk, int num_experts, int num_scales, int scale_token_stride, int scale_hidden_stride,
    int rank, int num_max_tokens, int num_max_send_tokens, int num_experts_per_group) {

    auto const num_sms   = static_cast<int>(gridDim.x);
    auto const sm_id     = static_cast<int>(blockIdx.x);
    auto const thread_id = static_cast<int>(threadIdx.x);
    auto const lane_id   = get_lane_id();
    const bool is_sender = sm_id % 2 == 0;
    EP_DEVICE_ASSERT(num_sms % 2 == 0);

    auto const num_threads_per_rank = kNumThreads / kNumRanks;
    auto const num_channels         = num_sms / 2;
    auto const responsible_rank     = thread_id / num_threads_per_rank;
    auto const responsible_channel  = sm_id / 2;
    auto const thread_id_in_rank    = thread_id % num_threads_per_rank;
    auto const warp_id_in_rank      = thread_id_in_rank / WARP_SIZE;

    int const num_experts_per_rank = num_experts / kNumRanks;
    EP_DEVICE_ASSERT(num_experts_per_rank > 0 or num_topk == 0);
    EP_DEVICE_ASSERT(num_topk <= WARP_SIZE);
    EP_DEVICE_ASSERT(num_experts_per_group > 0);

    int const num_expert_groups_per_rank =
        (num_experts_per_rank + num_experts_per_group - 1) / num_experts_per_group;
    EP_DEVICE_ASSERT(num_expert_groups_per_rank <= kMaxExpertGroupsPerRank);

    auto ptr = buffer_ptrs[is_sender ? responsible_rank : rank];

    auto rank_prefix_matrix = Buffer<int>(ptr, kNumRanks * kNumRanks);

    int  target_rank         = is_sender ? rank : responsible_rank;
    auto num_channels_total  = num_channels * kNumRanks;
    auto channel_rank_offset = responsible_channel * kNumRanks + target_rank;

    auto channel_start_offset = Buffer<int>(ptr, num_channels_total, channel_rank_offset);
    auto channel_end_offset   = Buffer<int>(ptr, num_channels_total, channel_rank_offset);
    auto channel_tail_idx     = Buffer<int>(ptr, num_channels_total, channel_rank_offset);
    // ─────────────────────────────────────────────────────────────────────
    // Per-(src_rank, dst_rank, channel) phase counter. The sender
    // release-stores `group_id + 1` to this cell at the end of every phase
    // — including phases that ship zero tokens — so the receiver can
    // advance strictly in lock-step with the sender's G = num_expert_groups_per_rank
    // phases, even when some phases are empty.
    //
    // This is what lets the receiver restrict its per-phase work to the
    // topk copies that belong to the current phase's expert group, instead
    // of the naive "copy all `num_topk` slots per received token"
    // behaviour that used to block group 0's tail flush on work for
    // groups 1..G-1.
    // ─────────────────────────────────────────────────────────────────────
    auto channel_phase_idx = Buffer<int>(ptr, num_channels_total, channel_rank_offset);
    auto expert_slot_idx   = Buffer<int>(ptr, num_experts_per_rank);

    auto num_tokens_total                = num_tokens * kNumRanks;
    auto dispatched_x_buffers            = Buffer<int4>(ptr, num_tokens_total * hidden_int4);
    auto dispatched_topk_idx_buffers     = Buffer<int64_t>(ptr, num_tokens_total * num_topk);
    auto dispatched_topk_weights_buffers = Buffer<float>(ptr, num_tokens_total * num_topk);

    if (thread_id < MAX_NUM_BARRIERS)
        amd::barrier_init(thread_id);
    __syncthreads();

    if (is_sender) {
        constexpr int num_send_warps          = kNumThreads / WARP_SIZE;
        constexpr int num_send_warps_per_rank = num_send_warps / kNumRanks;
        EP_DEVICE_ASSERT(kNumRanks <= WARP_SIZE);
        EP_DEVICE_ASSERT(num_send_warps % kNumRanks == 0);

        // Get tasks
        int token_start_idx, token_end_idx;
        get_channel_task_range(num_tokens, num_channels, responsible_channel, token_start_idx,
                               token_end_idx);

        int rank_offset =
            rank > 0 ? rank_prefix_matrix[(rank - 1) * kNumRanks + responsible_rank] : 0;

        int total_offset =
            responsible_channel > 0
                ? channel_prefix_matrix[responsible_rank * num_channels + responsible_channel - 1]
                : 0;
        if (lane_id == 0)
            st_relaxed_sys_global(channel_start_offset.buffer(), -total_offset - 1);

        int num_tokens_to_recv =
            channel_prefix_matrix[responsible_rank * num_channels + responsible_channel];
        if (lane_id == 0)
            st_relaxed_sys_global(channel_end_offset.buffer(), -num_tokens_to_recv - 1);
        num_tokens_to_recv -= total_offset;

        total_offset += rank_offset;

        if (num_tokens_to_recv <= 0)
            return;

        // Precomputed base pointers into the bucket tensors. These arrays
        // were built by ``expert_grouped_build_buckets`` as part of
        // ``notify_dispatch`` preprocessing — they are ready in global
        // memory before this kernel launches, so phase 0 can start on the
        // very first cycle after the sender enters the phase loop.
        //
        // We used to stage `group_offsets_global[0..num_groups]` into LDS
        // via a warp-group-wide cooperative load + `sync_barrier`. That
        // barrier sat squarely on phase 0's critical path. Reading the two
        // ints we need per phase (`begin`, `end`) directly via `__ldg`
        // lets the scalar L1 satisfy them after the first hit and removes
        // the barrier from the pre-phase-0 path entirely.
        int64_t const offs_base =
            (static_cast<int64_t>(responsible_rank) * num_channels + responsible_channel) *
            (kMaxExpertGroupsPerRank + 1);
        int const *global_offs_ptr = group_offsets_global + offs_base;

        int64_t const sorted_base =
            (static_cast<int64_t>(responsible_rank) * num_channels + responsible_channel) *
            kMaxLocalTokensPerSlice;
        int16_t const *global_sorted_ptr = sorted_token_offsets_global + sorted_base;

        // ── Phase loop ──────────────────────────────────────────────────────
        // Each phase ships its full bucket. Warps in the rank's warp-group
        // share the work round-robin — each warp handles one token's hidden
        // copy plus its topk metadata update. The sorted token-offset list
        // is read directly from global memory (int16 loads, trivial next to
        // the hidden copies and naturally cached).
        //
        // At the end of every phase (including phases that shipped zero
        // tokens) we release-store BOTH:
        //
        //   * `channel_tail_idx`  — cumulative # tokens shipped so far.
        //   * `channel_phase_idx` — `group_id + 1`.
        //
        // The phase counter is what lets the receiver step deterministically
        // through G phases, even when the tail value hasn't advanced
        // (empty phases). `channel_tail_idx` carries the payload
        // visibility; the phase_idx release happens after it in program
        // order, so an acquire-load of `channel_phase_idx` on the receiver
        // also makes the tail + x-buffer stores visible.
        int cached_channel_tail_idx = 0;
        for (int group_id = 0; group_id < num_expert_groups_per_rank; ++group_id) {
            int const begin = __ldg(global_offs_ptr + group_id);
            int const end   = __ldg(global_offs_ptr + group_id + 1);
            int const count = end - begin;

            for (int i = warp_id_in_rank; i < count; i += num_send_warps_per_rank) {
                int     local_t      = static_cast<int>(__ldg(global_sorted_ptr + begin + i));
                int64_t token_idx    = token_start_idx + local_t;
                int     dst_slot_idx = total_offset + cached_channel_tail_idx + i;

                auto shifted_dispatched_x_buffers =
                    dispatched_x_buffers.buffer() + dst_slot_idx * hidden_int4;
                auto shifted_x = x + token_idx * hidden_int4;
                UNROLLED_WARP_COPY(5, lane_id, hidden_int4, shifted_dispatched_x_buffers, shifted_x,
                                   __ldg, st_na_global);

                if (lane_id < num_topk) {
                    int recv_expert_begin = responsible_rank * num_experts_per_rank,
                        recv_expert_end   = (responsible_rank + 1) * num_experts_per_rank;
                    auto idx_value        = __ldg(topk_idx + token_idx * num_topk + lane_id);
                    idx_value = (idx_value >= recv_expert_begin and idx_value < recv_expert_end)
                                    ? idx_value - recv_expert_begin
                                    : -1;
                    dispatched_topk_idx_buffers[dst_slot_idx * num_topk + lane_id] = idx_value;

                    auto weight_value = __ldg(topk_weights + token_idx * num_topk + lane_id);
                    weight_value      = (idx_value >= 0) ? weight_value : 0.0f;
                    dispatched_topk_weights_buffers[dst_slot_idx * num_topk + lane_id] =
                        weight_value;
                }
            }

            cached_channel_tail_idx += count;

            // Per-phase commit. Two stores to make visible to the peer
            // (the receiver runs on the *destination* rank):
            //
            //   1. `channel_tail_idx`  ← cumulative token count.
            //   2. `channel_phase_idx` ← phase id (group_id + 1).
            //
            // The receiver acquires on (2) then reads (1), so the
            // correctness requirement is:
            //
            //   * (2)'s release must happen-after all this phase's
            //     hidden-payload stores AND (1)'s store.
            //   * (1) itself only needs to be *present* in memory when
            //     the receiver observes (2); it does not need its own
            //     release chain.
            //
            // We therefore issue a plain relaxed store for (1) and rely
            // on the single `s_waitcnt lgkmcnt(0) vmcnt(0)` embedded in
            // the aggressive release-store of (2) to commit both the
            // pending payload stores AND the `channel_tail_idx` store
            // before publishing `channel_phase_idx`. This cuts the
            // per-phase system-scope waitcnt count from 2 to 1, which
            // on MI300X xGMI is worth ~100-200us of wall time per
            // phase — the dominant slowdown otherwise observed at large
            // G (= num_expert_groups_per_rank).
            //
            // When the phase shipped zero tokens there are no per-warp
            // payload writes, so the workgroup-scope `sync_barrier` and
            // the tail-idx store itself can be skipped — the receiver
            // reads the same tail value it saw last phase, which is
            // exactly what an empty phase means.
            if (count > 0) {
                sync_barrier(responsible_rank, num_threads_per_rank);
                if (warp_id_in_rank == 0 and lane_id == 0) {
                    st_relaxed_sys_global(channel_tail_idx.buffer(), cached_channel_tail_idx);
                }
            }
            if (warp_id_in_rank == 0 and lane_id == 0) {
                st_release_sys_global<true>(channel_phase_idx.buffer(), group_id + 1);
            }
        }
    } else {
        // ── Phase-driven, group-aware receiver ─────────────────────────────
        //
        // The sender publishes its progress one phase at a time: each
        // phase ships all tokens whose *primary* (smallest-matching) local
        // expert lies in group g, then release-stores `channel_phase_idx`
        // with `g + 1`. A critical invariant of that bucketing is:
        //
        //   If a token ships in sender phase h, then ALL of its matching
        //   local experts are in groups {h, h+1, …, G-1} — it has NO
        //   topk hits in groups 0..h-1 (otherwise its primary would be
        //   in one of those earlier groups and it would have shipped
        //   in an earlier phase).
        //
        // Consequence: group g's `expert_tail_idx[e]` only receives
        // contributions from tokens shipped in phases 0..g.
        //
        // The old receiver paid the cost of ALL `num_topk` hidden copies
        // for every newly-arrived token in the phase it arrived, which
        // pushed "eager" copies for groups 1..G-1 onto group 0's critical
        // path and delayed the group 0 `expert_tail_idx` flush. The new
        // receiver instead processes, in phase g, only the topk hits that
        // fall into group g — across ALL tokens received so far. Total
        // per-token copy work is unchanged; it's just redistributed so
        // that group g's tail flush happens as soon as group g's portion
        // of the pipelined work is done. Phase 0 in particular drops to
        // ~1 hidden copy per phase-0 token (average: the primary hit),
        // instead of `num_topk` copies.
        constexpr int num_recv_warps          = kNumThreads / WARP_SIZE;
        constexpr int num_recv_warps_per_rank = num_recv_warps / kNumRanks;
        const auto    recv_thread_id_in_rank  = thread_id % num_threads_per_rank;
        const auto    recv_warp_id_in_rank    = recv_thread_id_in_rank / WARP_SIZE;
        EP_DEVICE_ASSERT(kNumRanks <= WARP_SIZE);
        EP_DEVICE_ASSERT(num_recv_warps % kNumRanks == 0);

        auto rank_prefix_matrix_recv = static_cast<int *>(buffer_ptrs[rank]);
        int  rank_offset             = responsible_rank > 0
                                           ? rank_prefix_matrix_recv[(responsible_rank - 1) * kNumRanks + rank]
                                           : 0;

        int total_offset, num_tokens_to_recv;
        while (lane_id == 0 and
               (total_offset = ld_volatile_global(channel_start_offset.buffer())) == 0)
            ;
        while (lane_id == 0 and
               (num_tokens_to_recv = ld_volatile_global(channel_end_offset.buffer())) == 0)
            ;
        if (lane_id == 0) {
            total_offset = -total_offset - 1, num_tokens_to_recv = -num_tokens_to_recv - 1;
            num_tokens_to_recv -= total_offset;
        }
        total_offset = __shfl_sync(WARP_MASK, total_offset, 0);
        total_offset += rank_offset;
        num_tokens_to_recv = __shfl_sync(WARP_MASK, num_tokens_to_recv, 0);

        // Absolute slot index for rel=0 in this (src_rank, channel) stream.
        // Used as `slot_base + rel` throughout the phase loop; unlike the
        // legacy receiver we do NOT shift this forward between phases —
        // each phase re-scans tokens starting from rel=0.
        int const slot_base = total_offset;

        if (num_tokens_to_recv <= 0)
            return;

        __shared__ volatile int shared_channel_tail_idx[kNumRanks];
        __shared__ volatile int shared_expert_prefix_sum[NUM_MAX_LOCAL_EXPERTS];

        // Per-(channel, rank) per-expert phase counter — accumulates how many
        // tokens this receiver block has just permuted into each destination
        // expert during the current phase. Flushed to the global
        // `expert_tail_idx` counter ONCE per phase per expert, dramatically
        // reducing atomic traffic on the cacheline polled by the GroupedGEMM
        // consumer.
        __shared__ int s_expert_phase_count[kNumRanks][NUM_MAX_LOCAL_EXPERTS];

        if (thread_id < num_experts_per_rank)
            shared_expert_prefix_sum[thread_id] = num_recv_tokens_per_expert[thread_id];
        __syncthreads();
        if (thread_id == 0) {
            int prev = 0;
            for (int i = 0; i < num_experts_per_rank; i++) {
                int curr                    = shared_expert_prefix_sum[i];
                shared_expert_prefix_sum[i] = prev;
                prev += curr;
            }
        }
        __syncthreads();

        // Init per-(rank, expert) phase counters to zero.
        if (thread_id_in_rank < num_experts_per_rank)
            s_expert_phase_count[responsible_rank][thread_id_in_rank] = 0;
        sync_barrier(responsible_rank, num_threads_per_rank);

        __shared__ volatile int shared_phase_idx[kNumRanks];

        auto start_time              = clock64();
        int  phases_done             = 0; // # of sender phases already processed
        int  cached_channel_tail_idx = 0; // last observed tail (slot_rel)
        int  prev_phase_tail_cached  = 0; // target_tail at the end of the
                                          // previous batch iteration; used to
                                          // derive `num_new_tokens` for Pass 1

        // ── Phase loop with adaptive batching ──────────────────────────
        // Each outer iteration processes AT LEAST one new phase, but may
        // process several in a single batch when the sender has already
        // raced ahead. Two properties we preserve here:
        //
        //   1. GROUP-0 PRIORITY. The first iteration is guaranteed to
        //      process phase 0 alone: the sender only publishes
        //      `channel_phase_idx = 1` after it finishes shipping phase
        //      0's tokens, and the receiver breaks out of its spin the
        //      instant that store is visible. Because the sender's
        //      subsequent phase-1 stores take nontrivial xGMI time,
        //      phase_idx is overwhelmingly = 1 at that moment, so the
        //      first batch is `[0, 0]`. This gives group 0's expert
        //      tails a fast, isolated flush — which is the *entire
        //      point* of the expert-grouped dispatch.
        //
        //   2. ADAPTIVE BATCHING ON LATER PHASES. If the receiver is
        //      slower than the sender on subsequent phases (e.g. at
        //      large G where each phase's work is tiny but the fixed
        //      per-iteration overhead isn't), multiple phase_idx stores
        //      will have landed by the time the receiver finishes the
        //      previous batch. The next spin break will see a higher
        //      `pi`, and the batch will fold those phases together into
        //      a single Pass 2 scan + single flush loop. Total wall
        //      time degrades gracefully from "strict G iterations" to
        //      "a handful of bulk iterations", matching the legacy
        //      tail-driven receiver's behaviour at high G while still
        //      giving group 0 its own dedicated first iteration.
        while (phases_done < num_expert_groups_per_rank) {
            // Wait for the sender to publish AT LEAST one new phase.
            while (recv_thread_id_in_rank == 0) {
                int pi = ld_acquire_sys_global<true>(channel_phase_idx.buffer());
                if (pi > phases_done) {
                    cached_channel_tail_idx =
                        ld_acquire_sys_global<true>(channel_tail_idx.buffer());
                    shared_channel_tail_idx[responsible_rank] = cached_channel_tail_idx;
                    shared_phase_idx[responsible_rank]        = pi;
                    break;
                }

                if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                    printf("expert_grouped_dispatch_permute recv timeout, rank %d, channel %d, "
                           "phases_done %d\n",
                           rank, responsible_channel, phases_done);
                    trap();
                }
            }
            sync_barrier(responsible_rank, num_threads_per_rank);
            cached_channel_tail_idx = shared_channel_tail_idx[responsible_rank];
            int const target_phases = shared_phase_idx[responsible_rank];

            int const target_tail     = cached_channel_tail_idx;
            int const batch_group_lo  = phases_done;
            int const batch_group_hi  = target_phases; // exclusive
            int const batch_expert_lo = batch_group_lo * num_experts_per_group;
            int const batch_expert_hi =
                min(batch_group_hi * num_experts_per_group, num_experts_per_rank);

            // ── Pass 1 (new tokens only): one-shot per-topk init ────────
            // Copy the sender's topk_idx / topk_weights into the public
            // outputs for tokens that arrived since the last iteration,
            // and pre-seed `dispatch_to_expert_map` with -1 for topk
            // entries whose expert isn't on this rank. Pass 2 overwrites
            // map entries for topk hits whose expert's group falls in
            // the current batch; entries for experts in LATER groups
            // stay unwritten here and get filled by Pass 2 of a
            // subsequent batch.
            int const prev_phase_tail = prev_phase_tail_cached;
            int const num_new_tokens  = target_tail - prev_phase_tail;
            if (num_new_tokens > 0) {
                for (int idx = recv_thread_id_in_rank; idx < num_new_tokens * num_topk;
                     idx += WARP_SIZE * num_recv_warps_per_rank) {
                    int  chunk_idx    = idx / num_topk;
                    int  topk_i       = idx % num_topk;
                    int  src_slot_idx = slot_base + prev_phase_tail + chunk_idx;
                    auto mapped_idx   = static_cast<int64_t>(src_slot_idx) * num_topk + topk_i;
                    auto expert = ld_nc_global(dispatched_topk_idx_buffers.buffer() + mapped_idx);
                    recv_topk_idx[mapped_idx] = expert;
                    recv_topk_weights[mapped_idx] =
                        ld_nc_global(dispatched_topk_weights_buffers.buffer() + mapped_idx);
                    if (expert < 0) {
                        dispatch_to_expert_map[mapped_idx] = -1;
                    }
                }
            }

            // ── Pass 2: hidden copies for all topk hits in THIS batch ─
            // Scan every token received so far (rel in [0, target_tail)),
            // pick out topk indices whose expert falls in the current
            // batch's [batch_expert_lo, batch_expert_hi) range, and do
            // the hidden copy + dispatch_to_expert_map write for each
            // such hit. This is the same invariant as the one-phase-
            // at-a-time path: tokens shipped by sender in phase h < g
            // can still have topk hits in group g (and they get
            // processed here for the first time when this batch covers
            // group g).
            //
            // A token shipped in sender-phase h has all its matching
            // experts in groups h..G-1, so it contributes to at most
            // (G - h) hidden copies across the lifetime of the pipeline.
            // Re-scanning its topk list across batches costs only a
            // small L1/L2 read of `num_topk` int64s — negligible next
            // to the hidden (~7168 bytes) copy work.
            for (int rel = recv_warp_id_in_rank; rel < target_tail;
                 rel += num_recv_warps_per_rank) {
                int  src_slot_idx = slot_base + rel;
                auto shifted_buffer_x_int4 =
                    dispatched_x_buffers.buffer() + src_slot_idx * hidden_int4;

#pragma unroll
                for (int i = 0; i < num_topk; i++) {
                    auto mapped_idx = static_cast<int64_t>(src_slot_idx) * num_topk + i;
                    auto expert = ld_nc_global(dispatched_topk_idx_buffers.buffer() + mapped_idx);
                    if (expert < batch_expert_lo || expert >= batch_expert_hi)
                        continue;

                    int slot = -1;
                    if (lane_id == 0) {
                        slot = __hip_atomic_fetch_add(expert_slot_idx.buffer() + expert, 1,
                                                      __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
                    }
                    slot = __shfl_sync(WARP_MASK, slot, 0);
                    EP_DEVICE_ASSERT(slot >= 0);
                    int dst_slot_idx = shared_expert_prefix_sum[expert] + slot;
                    EP_DEVICE_ASSERT(dst_slot_idx < num_max_tokens * num_topk);
                    auto shifted_recv_x_int4 =
                        recv_x + static_cast<int64_t>(dst_slot_idx) * hidden_int4;
                    UNROLLED_WARP_COPY(5, lane_id, hidden_int4, shifted_recv_x_int4,
                                       shifted_buffer_x_int4, ld_nc_global, st_na_global);

                    dispatch_to_expert_map[mapped_idx] = dst_slot_idx;

                    if (lane_id == 0) {
                        __hip_atomic_fetch_add(&s_expert_phase_count[responsible_rank][expert], 1,
                                               __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
                    }
                }
            }

            // Wait for all warps in this (channel, rank) to finish this
            // batch's copies and phase-counter increments before
            // publishing the batch's expert tails to the consumer-visible
            // `expert_tail_idx` array.
            sync_barrier(responsible_rank, num_threads_per_rank);

            // Flush per-expert phase counts → expert_tail_idx for the
            // experts covered by this batch. One atomic per (rank,
            // covered-expert) per batch. The remaining experts retain
            // their 0-valued phase counter until a later batch brings
            // their group into range.
            //
            // No post-flush `sync_barrier` here: the next iteration of
            // the outer while-loop opens with its own `sync_barrier`
            // after the phase_idx spin, which is enough to order this
            // flush with any subsequent batch's Pass 2 writes to
            // `s_expert_phase_count` (those writes target a disjoint
            // set of expert slots, so the memory race the barrier used
            // to guard never materialises).
            if (thread_id_in_rank < num_experts_per_rank) {
                int delta = s_expert_phase_count[responsible_rank][thread_id_in_rank];
                if (delta > 0) {
                    __hip_atomic_fetch_add(expert_tail_idx + thread_id_in_rank, delta,
                                           __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
                    s_expert_phase_count[responsible_rank][thread_id_in_rank] = 0;
                }
            }

            prev_phase_tail_cached = target_tail;
            phases_done            = target_phases;
        }
    }
}

void expert_grouped_dispatch_permute(
    void **buffer_ptrs, int *expert_tail_idx, int64_t *recv_topk_idx, float *recv_topk_weights,
    int *dispatch_to_expert_map, void const *x, float const *x_scales, int64_t const *topk_idx,
    float const *topk_weights, bool const *is_token_in_rank, int const *channel_prefix_matrix,
    int const *num_recv_tokens_per_expert, int const *group_offsets_global,
    int16_t const *sorted_token_offsets_global, void *recv_x, int num_tokens, int hidden_int4,
    int num_topk, int num_experts, int num_scales, int scale_token_stride, int scale_hidden_stride,
    int rank, int num_ranks, cudaStream_t stream, int num_sms, int num_max_tokens,
    int num_max_send_tokens, int num_experts_per_group) {
    constexpr int kNumThreads = 1024;

#define EXPERT_GROUPED_LAUNCH_CASE(ranks)                                                          \
    {                                                                                              \
        auto kernel = expert_grouped_dispatch_permute<ranks, kNumThreads>;                         \
        LAUNCH_KERNEL(&cfg, kernel, buffer_ptrs, expert_tail_idx, recv_topk_idx,                   \
                      recv_topk_weights, dispatch_to_expert_map,                                   \
                      reinterpret_cast<int4 const *>(x), x_scales, topk_idx, topk_weights,         \
                      is_token_in_rank, channel_prefix_matrix, num_recv_tokens_per_expert,         \
                      group_offsets_global, sorted_token_offsets_global,                           \
                      reinterpret_cast<int4 *>(recv_x), num_tokens, hidden_int4, num_topk,         \
                      num_experts, num_scales, scale_token_stride, scale_hidden_stride, rank,      \
                      num_max_tokens, num_max_send_tokens, num_experts_per_group);                 \
    }                                                                                              \
    break

    EP_HOST_ASSERT(num_sms % 2 == 0);
    EP_HOST_ASSERT(num_experts_per_group > 0);
    SETUP_LAUNCH_CONFIG(num_sms, kNumThreads, stream);
    SWITCH_RANKS(EXPERT_GROUPED_LAUNCH_CASE);
#undef EXPERT_GROUPED_LAUNCH_CASE
}

} // namespace intranode
} // namespace primus_turbo::cco::ep
