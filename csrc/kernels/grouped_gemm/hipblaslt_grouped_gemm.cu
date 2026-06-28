// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <hipblaslt/hipblaslt.h>
#include <vector>

#include "primus_turbo/gemm.h"
#include "primus_turbo/grouped_gemm.h"

namespace primus_turbo {

// Number of streams actively used by `run()` (1 caller stream + 3 internal).
static constexpr size_t kMaxNumStreams = 4;
// Number of internal handle / stream / event slots actually pre-allocated at
// construction time. Mirrors upstream `grouped-gemm-ck::HipBlasLt`'s
// `kDefaultInitKmaxNumStream = 8`. With only `kMaxNumStreams = 4` slots
// pre-allocated, BF16 multi-stream grouped GEMM stalls intermittently on
// MI355X after a few dozen fwd+bwd iterations; over-provisioning the slot
// pool empirically eliminates the stall (see PR description for traces).
static constexpr size_t kInitNumStreams = 8;

std::int64_t get_hipblaslt_grouped_gemm_workspace_size() {
    return kMaxNumStreams * get_hipblaslt_workspace_size_in_byte();
}

class HipblasltGroupedGemm {
public:
    HipblasltGroupedGemm() {
        PRIMUS_TURBO_CHECK_HIP(hipEventCreateWithFlags(&sync_event_, hipEventDisableTiming));

        // Slot 0 is the PyTorch current stream (fetched per-call inside
        // `run(...)`); only slots 1..kInitNumStreams-1 own internal compute
        // streams + dedicated hipBLASLt handles. Creating a "ghost" handle
        // for slot 0 was empirically observed to push hipBLASLt past an
        // internal resource limit on MI355X and re-trigger the BF16 stall.
        handles_[0]         = nullptr;
        compute_streams_[0] = nullptr;
        for (size_t i = 0; i < kInitNumStreams; ++i) {
            if (i > 0) {
                PRIMUS_TURBO_CHECK_HIPBLAS(hipblasLtCreate(&handles_[i]));
                PRIMUS_TURBO_CHECK_HIP(
                    hipStreamCreateWithPriority(&compute_streams_[i], hipStreamNonBlocking, -1));
                // Bind each handle to its dedicated compute stream. hipBLASLt's
                // internal state (heuristic cache, workspace accounting) is
                // per-handle and assumes a one-to-one handle<->stream association.
                PRIMUS_TURBO_CHECK_HIPBLAS(hipblasSetStream(
                    reinterpret_cast<hipblasHandle_t>(handles_[i]), compute_streams_[i]));
            }
            PRIMUS_TURBO_CHECK_HIP(
                hipEventCreateWithFlags(&hipblaslt_events_[i], hipEventDisableTiming));
        }
        workspaces_.resize(kMaxNumStreams);
    }

    ~HipblasltGroupedGemm() {
        if (sync_event_ != nullptr) {
            (void) hipEventDestroy(sync_event_);
        }

        for (size_t i = 0; i < kInitNumStreams; ++i) {
            if (compute_streams_[i] != nullptr) {
                (void) hipStreamDestroy(compute_streams_[i]);
            }
            if (handles_[i] != nullptr) {
                (void) hipblasLtDestroy(handles_[i]);
            }
            if (hipblaslt_events_[i] != nullptr) {
                (void) hipEventDestroy(hipblaslt_events_[i]);
            }
        }
    }

    void check(const HipblasltGroupedGemmParams &params) {
        PRIMUS_TURBO_CHECK(params.a_shape.size() == 2);
        if (params.transA) {
            // For a * grad_c = grad_b
            // [m, k]^T * [m, n] = [b, k, n]
            PRIMUS_TURBO_CHECK(params.b_shape.size() == 2);
            PRIMUS_TURBO_CHECK(params.c_shape.size() == 3);
            PRIMUS_TURBO_CHECK(params.a_shape[0] == params.b_shape[0]);
            PRIMUS_TURBO_CHECK(params.c_shape[0] == params.group_num);
            PRIMUS_TURBO_CHECK(params.c_shape[1] == params.a_shape[1]);
            PRIMUS_TURBO_CHECK(params.c_shape[2] == params.b_shape[1]);
        } else {
            // For a * b = c and grad_c * b = grad_a
            PRIMUS_TURBO_CHECK(params.b_shape.size() == 3);
            PRIMUS_TURBO_CHECK(params.c_shape.size() == 2);
            PRIMUS_TURBO_CHECK(params.b_shape[0] == params.group_num);
        }
    }

    void run(const HipblasltGroupedGemmParams &params, const bool pre_sync) {
        if (pre_sync) {
            PRIMUS_TURBO_CHECK_HIP(hipStreamSynchronize(params.stream));
        }

        // Check
        check(params);
        // Compute arguments
        compute_args(params);

        const size_t num_gemms{gemm_ptrs_.size()};
        const size_t num_stream_used{std::min<size_t>(kMaxNumStreams, num_gemms)};

        if (num_gemms > 0) {
            // Slot 0 runs on params.stream directly; only extra slots need
            // fan-out from params.stream via sync_event_.
            if (num_stream_used > 1) {
                PRIMUS_TURBO_CHECK_HIP(hipEventRecord(sync_event_, params.stream));
                for (size_t s = 1; s < num_stream_used; ++s) {
                    PRIMUS_TURBO_CHECK_HIP(hipStreamWaitEvent(compute_streams_[s], sync_event_, 0));
                }
            }

            for (size_t idx = 0; idx < num_gemms; ++idx) {
                const auto stream_idx = idx % kMaxNumStreams;
                // Slot 0 uses the caller's stream and hipBLASLt handle (the
                // PyTorch current stream/handle, which is already bound
                // one-to-one). Extra slots use our internal streams bound via
                // hipblasSetStream above.
                auto stream    = (stream_idx == 0) ? params.stream : compute_streams_[stream_idx];
                auto handle    = (stream_idx == 0) ? params.handle : handles_[stream_idx];
                auto workspace = workspaces_[stream_idx];
                // clang-format off
                hipblaslt_gemm_impl(
                    gemm_ptrs_[idx].b_ptr, params.b_type, rows_b_[idx], cols_b_[idx], ld_b_[idx],
                    gemm_ptrs_[idx].b_scale_ptr,
                    params.transB ? HIPBLAS_OP_T : HIPBLAS_OP_N,
                    gemm_ptrs_[idx].a_ptr, params.a_type, rows_a_[idx], cols_a_[idx], ld_a_[idx],
                    gemm_ptrs_[idx].a_scale_ptr,
                    params.transA ? HIPBLAS_OP_T : HIPBLAS_OP_N,
                    gemm_ptrs_[idx].c_ptr, params.c_type, rows_c_[idx], cols_c_[idx], ld_c_[idx],
                    workspace, get_hipblaslt_workspace_size_in_byte(),
                    params.use_low_precision,
                    params.scale_mode,
                    handle,
                    stream
                );
                // clang-format on
            }

            if (num_stream_used > 1) {
                // Slot 0 already runs on params.stream; fan extra slots back
                // into params.stream.
                for (size_t s = 1; s < num_stream_used; ++s) {
                    PRIMUS_TURBO_CHECK_HIP(
                        hipEventRecord(hipblaslt_events_[s], compute_streams_[s]));
                }
                for (size_t s = 1; s < num_stream_used; ++s) {
                    PRIMUS_TURBO_CHECK_HIP(
                        hipStreamWaitEvent(params.stream, hipblaslt_events_[s], 0));
                }
            }
        }
    }

private:
    struct GemmPtr {
        const void *a_ptr       = nullptr;
        const void *a_scale_ptr = nullptr;
        const void *b_ptr       = nullptr;
        const void *b_scale_ptr = nullptr;
        void       *c_ptr       = nullptr;
    };

    void compute_args(const HipblasltGroupedGemmParams &params) {

        // TODO(xiaobochen): group_lens_ptr is device pointer, but host can access
        // it directly on MI300/MI350. Need to investigate why this works.
        int valid_group_num = 0;
        for (size_t i = 0; i < params.group_num; ++i) {
            valid_group_num += params.group_lens_ptr[i] > 0 ? 1 : 0;
        }

        const char *a_ptr = static_cast<const char *>(params.a_ptr);
        const char *b_ptr = static_cast<const char *>(params.b_ptr);
        char       *c_ptr = static_cast<char *>(params.c_ptr);
        gemm_ptrs_.resize(valid_group_num);
        ld_a_.resize(valid_group_num);
        ld_b_.resize(valid_group_num);
        ld_c_.resize(valid_group_num);
        rows_a_.resize(valid_group_num);
        cols_a_.resize(valid_group_num);
        rows_b_.resize(valid_group_num);
        cols_b_.resize(valid_group_num);
        rows_c_.resize(valid_group_num);
        cols_c_.resize(valid_group_num);

        int write_idx = 0;
        for (size_t i = 0; i < params.group_num; ++i) {
            int64_t len = params.group_lens_ptr[i];

            // For grad_b (transA), if the group len is 0, set the output memory to 0
            // c shape is [group_num, k, n], so each group's c size is k * n
            if (params.transA && len == 0) {
                int64_t c_rows_t = get_dim(params.c_shape, -1);
                int64_t c_cols_t = get_dim(params.c_shape, -2);
                int64_t c_size_t = c_rows_t * c_cols_t * hipblaslt_dtype_bytes(params.c_type);
                PRIMUS_TURBO_CHECK_HIP(hipMemsetAsync(c_ptr, 0, c_size_t, params.stream));
                c_ptr += c_size_t;
            }

            if (len == 0)
                continue;

            // pointers
            gemm_ptrs_[write_idx].a_ptr = a_ptr;
            gemm_ptrs_[write_idx].b_ptr = b_ptr;
            gemm_ptrs_[write_idx].c_ptr = c_ptr;
            if (params.use_low_precision) {
                // TODO(xiaobochen): support variable scale mode
                gemm_ptrs_[write_idx].a_scale_ptr = params.a_scale_ptr;
                gemm_ptrs_[write_idx].b_scale_ptr = params.b_scale_ptr;
            }

            // leading dimension
            ld_a_[write_idx] = get_dim(params.a_shape, -1);
            ld_b_[write_idx] = get_dim(params.b_shape, -1);
            ld_c_[write_idx] = get_dim(params.c_shape, -1);
            // rows and cols of matrices
            rows_a_[write_idx] = get_dim(params.a_shape, -1);
            cols_a_[write_idx] = len;
            rows_b_[write_idx] = get_dim(params.b_shape, -1);
            cols_b_[write_idx] = params.transA ? len : get_dim(params.b_shape, -2);
            rows_c_[write_idx] = get_dim(params.c_shape, -1);
            cols_c_[write_idx] = params.transA ? get_dim(params.c_shape, -2) : len;
            // advance the pointers
            a_ptr += rows_a_[write_idx] * cols_a_[write_idx] * hipblaslt_dtype_bytes(params.a_type);
            b_ptr += rows_b_[write_idx] * cols_b_[write_idx] * hipblaslt_dtype_bytes(params.b_type);
            c_ptr += rows_c_[write_idx] * cols_c_[write_idx] * hipblaslt_dtype_bytes(params.c_type);
            write_idx++;
        }

        char *workspace_ptr = static_cast<char *>(params.workspace);
        for (size_t i = 0; i < kMaxNumStreams; ++i) {
            workspaces_[i] = workspace_ptr + i * get_hipblaslt_workspace_size_in_byte();
        }
    }

    template <typename T> T get_dim(const std::vector<T> &shape, int idx) {
        if (idx < 0) {
            idx += static_cast<int>(shape.size());
        }
        return shape.at(idx);
    }

    // Handles, events, streams, heuristic, epilogue. Sized to
    // `kInitNumStreams` even though only `kMaxNumStreams` are actually
    // used by `run(...)`; see the constant declaration for the rationale.
    hipblasLtHandle_t   handles_[kInitNumStreams]{};
    hipEvent_t          sync_event_{nullptr};
    hipStream_t         compute_streams_[kInitNumStreams]{};
    hipEvent_t          hipblaslt_events_[kInitNumStreams]{};
    hipblasLtEpilogue_t epilogue_{HIPBLASLT_EPILOGUE_DEFAULT};

    // Gemm Pointers
    std::vector<GemmPtr> gemm_ptrs_;

    // workspace
    std::vector<void *> workspaces_;

    // Leading dimensions
    std::vector<int64_t> ld_a_;
    std::vector<int64_t> ld_b_;
    std::vector<int64_t> ld_c_;

    // rows and cols of matrices
    std::vector<int64_t> rows_a_;
    std::vector<int64_t> cols_a_;
    std::vector<int64_t> rows_b_;
    std::vector<int64_t> cols_b_;
    std::vector<int64_t> rows_c_;
    std::vector<int64_t> cols_c_;
};

void hipblaslt_grouped_gemm(const HipblasltGroupedGemmParams &params, const bool pre_sync) {
    // Process-wide singleton (NOT `thread_local`), mirroring upstream
    // `grouped-gemm-ck::HipBlasLt`. With `thread_local` the autograd worker
    // thread allocated its own pool of internal streams, so the process
    // ended up running on >=7 distinct hipBLASLt streams (3 per thread +
    // PyTorch's default), which appears to push hipBLASLt past an implicit
    // resource limit on MI355X and triggers BF16 multi-stream stalls.
    static HipblasltGroupedGemm instance;
    instance.run(params, pre_sync);
}

// ===========================================================================
//  MXFP8 grouped GEMM (gfx1250)
//
//  hipBLASLt VEC32_UE8M0 on gfx1250 is TN-only AND consumes e8m0 block scales
//  only in the GEMM-swizzled "groups-of-4" layout, with the data m-dim padded
//  to a 128 multiple. Per group this is the SAME NT GEMM the dense MX op runs
//  (csrc/kernels/gemm/hipblaslt_gemm.cu): the contraction (K for fwd, M_g for
//  wgrad) is the contiguous trailing dim and the scale is blocked along it.
//
//  C++ port of the validated Python reference (grouped_gemm_fp8_impl.py:
//  _swizzle_mxfp8_scale_groups4 / _grouped_mxfp8_hipblaslt /
//  _grouped_mxfp8_variable_k_hipblaslt). Per group we pad+swizzle then reuse the
//  same multi-stream hipblaslt_gemm_impl fan-out as the BF16 / TENSORWISE path.
//  All device buffers are pre-allocated/freed by the caller (PyTorch caching
//  allocator) -- no per-group host syncs, no hipMalloc here.
// ===========================================================================

namespace {

constexpr int kMXScaleGrp = 4; // e8m0 scales swizzled in groups of 4 along K-blocks.

// Swizzle ALL groups' row-wise e8m0 scales into the hipBLASLt groups-of-4 layout
// in one launch. Group g owns input rows [row_in_off[g], row_in_off[g]+len) of a
// contiguous (total_rows, ks) source and writes a padded (mpad[g], ks_pad) block
// at output element offset out_blk_off[g]:
//   within = (k/4)*(mpad*4) + m*4 + (k%4)
// Padding (m>=len or k>=ks) is 127 (e8m0 bias 0 -> native scale 1.0). One thread
// per output element of the union of padded blocks (blk_pref is the running
// element prefix, blk_pref[group_num] == total_out_tiles).
__global__ void mxfp8_swizzle_scale_grouped_kernel(
    const uint8_t *__restrict__ in_scale, uint8_t *__restrict__ out_scale, const int64_t ks,
    const int64_t ks_pad, const int *__restrict__ row_in_off, const int *__restrict__ row_len,
    const int *__restrict__ mpad, const int64_t *__restrict__ out_blk_off,
    const int64_t *__restrict__ blk_pref, const int group_num, const int64_t total_out_tiles) {
    const int64_t gid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (gid >= total_out_tiles)
        return;
    int g = 0;
    while (g + 1 < group_num && blk_pref[g + 1] <= gid)
        ++g;
    const int64_t local = gid - blk_pref[g];
    const int     m_pad = mpad[g];
    const int64_t m     = local / ks_pad;
    const int64_t k     = local % ks_pad;
    const int     len   = row_len[g];
    uint8_t       val   = 127;
    if (m < len && k < ks) {
        val = in_scale[(static_cast<int64_t>(row_in_off[g]) + m) * ks + k];
    }
    const int64_t within = (k / kMXScaleGrp) * (static_cast<int64_t>(m_pad) * kMXScaleGrp) +
                           m * kMXScaleGrp + (k % kMXScaleGrp);
    out_scale[out_blk_off[g] + within] = val;
}

// Pad each group's row-major fp8 data (len, K) -> (mpad[g], K) zero-filled,
// concatenated at out_row_off[g] rows. One thread per output element (padded);
// elem_pref is the running element prefix (elem_pref[group_num] == total).
__global__ void mxfp8_pad_data_grouped_kernel(
    const uint8_t *__restrict__ in_data, uint8_t *__restrict__ out_data, const int64_t K,
    const int *__restrict__ row_in_off, const int *__restrict__ row_len,
    const int64_t *__restrict__ out_row_off, const int64_t *__restrict__ elem_pref,
    const int group_num, const int64_t total_out_elems) {
    const int64_t gid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (gid >= total_out_elems)
        return;
    int g = 0;
    while (g + 1 < group_num && elem_pref[g + 1] <= gid)
        ++g;
    const int64_t local = gid - elem_pref[g];
    const int64_t m     = local / K;
    const int64_t k     = local % K;
    uint8_t       v     = 0;
    if (m < row_len[g]) {
        v = in_data[(static_cast<int64_t>(row_in_off[g]) + m) * K + k];
    }
    out_data[(out_row_off[g] + m) * K + k] = v;
}

// Pack the per-group real rows (row_len[g]) from the padded-block GEMM output
// (group g starts at src_row_off[g] rows) into the tight output (dst_row_off[g]
// rows). N is the output free dim (bytes per row = N*dtype_bytes); one thread per
// output byte of the tight result. elem_pref is the running byte prefix.
__global__ void mxfp8_pack_output_grouped_kernel(
    const uint8_t *__restrict__ padded, uint8_t *__restrict__ tight, const int64_t row_bytes,
    const int *__restrict__ src_row_off, const int *__restrict__ dst_row_off,
    const int *__restrict__ row_len, const int64_t *__restrict__ elem_pref, const int group_num,
    const int64_t total_out_elems) {
    const int64_t gid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (gid >= total_out_elems)
        return;
    int g = 0;
    while (g + 1 < group_num && elem_pref[g + 1] <= gid)
        ++g;
    const int64_t local = gid - elem_pref[g];
    const int64_t m     = local / row_bytes; // real row within group
    const int64_t b     = local % row_bytes; // byte within row
    tight[(static_cast<int64_t>(dst_row_off[g]) + m) * row_bytes + b] =
        padded[(static_cast<int64_t>(src_row_off[g]) + m) * row_bytes + b];
}

} // namespace

// Multi-stream pool for the MX per-group GEMMs. Mirrors HipblasltGroupedGemm's
// stream/handle pool + fan-out, but consumes ready-made per-group descriptors
// (swizzled scales, 128-padded data, output) instead of the linear-advance
// TENSORWISE layout.
class HipblasltMXGroupedGemm {
public:
    HipblasltMXGroupedGemm() {
        PRIMUS_TURBO_CHECK_HIP(hipEventCreateWithFlags(&sync_event_, hipEventDisableTiming));
        handles_[0]         = nullptr;
        compute_streams_[0] = nullptr;
        for (size_t i = 0; i < kInitNumStreams; ++i) {
            if (i > 0) {
                PRIMUS_TURBO_CHECK_HIPBLAS(hipblasLtCreate(&handles_[i]));
                PRIMUS_TURBO_CHECK_HIP(
                    hipStreamCreateWithPriority(&compute_streams_[i], hipStreamNonBlocking, -1));
                PRIMUS_TURBO_CHECK_HIPBLAS(hipblasSetStream(
                    reinterpret_cast<hipblasHandle_t>(handles_[i]), compute_streams_[i]));
            }
            PRIMUS_TURBO_CHECK_HIP(
                hipEventCreateWithFlags(&hipblaslt_events_[i], hipEventDisableTiming));
        }
        workspaces_.resize(kMaxNumStreams);
    }

    ~HipblasltMXGroupedGemm() {
        if (sync_event_ != nullptr)
            (void) hipEventDestroy(sync_event_);
        for (size_t i = 0; i < kInitNumStreams; ++i) {
            if (compute_streams_[i] != nullptr)
                (void) hipStreamDestroy(compute_streams_[i]);
            if (handles_[i] != nullptr)
                (void) hipblasLtDestroy(handles_[i]);
            if (hipblaslt_events_[i] != nullptr)
                (void) hipEventDestroy(hipblaslt_events_[i]);
        }
    }

    void run(const HipblasltMXGroupedGemmParams &p) {
        // Issue the per-group MX GEMMs on the multi-stream pool. The caller has
        // already (a) filtered to valid groups (p.group_num == #valid, all per-group
        // arrays valid-indexed) and (b) launched the batched pad/swizzle kernels.
        // Per group g the GEMM is the NT product
        //   D[g] (n_pad_g x m_pad_g col-major)  from  A[m_pad_g, K_g] @ B[n_pad_g, K_g]^T
        // with contraction K_g = p.kdim[g] (fwd: fixed K; wgrad: per-group M_g),
        // VEC32_UE8M0 groups-of-4 swizzled scales, and real output row stride p.ldc.
        const size_t num_gemms = static_cast<size_t>(p.group_num);

        char *workspace_ptr = static_cast<char *>(p.workspace);
        for (size_t i = 0; i < kMaxNumStreams; ++i)
            workspaces_[i] = workspace_ptr + i * get_hipblaslt_workspace_size_in_byte();

        if (num_gemms == 0)
            return;
        const size_t num_stream_used = std::min<size_t>(kMaxNumStreams, num_gemms);

        if (num_stream_used > 1) {
            PRIMUS_TURBO_CHECK_HIP(hipEventRecord(sync_event_, p.stream));
            for (size_t s = 1; s < num_stream_used; ++s)
                PRIMUS_TURBO_CHECK_HIP(hipStreamWaitEvent(compute_streams_[s], sync_event_, 0));
        }

        for (size_t idx = 0; idx < num_gemms; ++idx) {
            const int  g          = static_cast<int>(idx);
            const auto stream_idx = idx % kMaxNumStreams;
            auto       stream     = (stream_idx == 0) ? p.stream : compute_streams_[stream_idx];
            auto       handle     = (stream_idx == 0) ? p.handle : handles_[stream_idx];
            auto       workspace  = workspaces_[stream_idx];

            const int64_t K     = p.kdim[g];   // per-group contraction
            const int64_t m_pad = p.a_mpad[g]; // padded A rows (m)
            const int64_t n_pad = p.b_mpad[g]; // padded B rows (n)

            const uint8_t *a_data =
                static_cast<const uint8_t *>(p.a_padded) + p.a_row_off[g] * K;
            const uint8_t *a_scale =
                static_cast<const uint8_t *>(p.a_swz_scale) + p.a_scale_off[g];
            const uint8_t *b_data = static_cast<const uint8_t *>(p.b_data) + p.b_row_off[g] * K;
            const uint8_t *b_scale =
                static_cast<const uint8_t *>(p.b_swz_scale) + p.b_scale_off[g];
            void *c_data = static_cast<char *>(p.c_data) + p.c_off_bytes[g];

            // hipblaslt_gemm_impl takes (B first, A second); the row-major ->
            // col-major operand swap lands it as TN, identical to the dense MX op:
            //   B operand: rows_b=K, cols_b=n_pad, ld_b=K, op_T
            //   A operand: rows_a=K, cols_a=m_pad, ld_a=K, op_N
            //   D:         rows_d=n_pad, cols_d=m_pad, ld_d=p.ldc (real output row stride)
            // clang-format off
            hipblaslt_gemm_impl(
                b_data, p.ab_type, K, n_pad, K,
                b_scale, HIPBLAS_OP_T,
                a_data, p.ab_type, K, m_pad, K,
                a_scale, HIPBLAS_OP_N,
                c_data, p.c_type, n_pad, m_pad, p.ldc,
                workspace, get_hipblaslt_workspace_size_in_byte(),
                /*use_low_precision=*/true,
                HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0,
                handle, stream);
            // clang-format on
        }

        if (num_stream_used > 1) {
            for (size_t s = 1; s < num_stream_used; ++s)
                PRIMUS_TURBO_CHECK_HIP(hipEventRecord(hipblaslt_events_[s], compute_streams_[s]));
            for (size_t s = 1; s < num_stream_used; ++s)
                PRIMUS_TURBO_CHECK_HIP(hipStreamWaitEvent(p.stream, hipblaslt_events_[s], 0));
        }
    }

private:
    hipblasLtHandle_t handles_[kInitNumStreams]{};
    hipEvent_t        sync_event_{nullptr};
    hipStream_t       compute_streams_[kInitNumStreams]{};
    hipEvent_t        hipblaslt_events_[kInitNumStreams]{};
    std::vector<void *> workspaces_;
};

void mxfp8_swizzle_scale_grouped(const uint8_t *in_scale, uint8_t *out_scale, int64_t ks,
                                 int64_t ks_pad, const int *row_in_off, const int *row_len,
                                 const int *mpad, const int64_t *out_blk_off,
                                 const int64_t *blk_pref, int group_num, int64_t total_out_tiles,
                                 hipStream_t stream) {
    if (total_out_tiles == 0)
        return;
    constexpr int kBlock = 256;
    const int64_t grid   = DIVUP<int64_t>(total_out_tiles, kBlock);
    mxfp8_swizzle_scale_grouped_kernel<<<grid, kBlock, 0, stream>>>(
        in_scale, out_scale, ks, ks_pad, row_in_off, row_len, mpad, out_blk_off, blk_pref,
        group_num, total_out_tiles);
    PRIMUS_TURBO_CHECK_HIP(hipGetLastError());
}

void mxfp8_pad_data_grouped(const uint8_t *in_data, uint8_t *out_data, int64_t K,
                            const int *row_in_off, const int *row_len, const int64_t *out_row_off,
                            const int64_t *elem_pref, int group_num, int64_t total_out_elems,
                            hipStream_t stream) {
    if (total_out_elems == 0)
        return;
    constexpr int kBlock = 256;
    const int64_t grid   = DIVUP<int64_t>(total_out_elems, kBlock);
    mxfp8_pad_data_grouped_kernel<<<grid, kBlock, 0, stream>>>(in_data, out_data, K, row_in_off,
                                                               row_len, out_row_off, elem_pref,
                                                               group_num, total_out_elems);
    PRIMUS_TURBO_CHECK_HIP(hipGetLastError());
}

void mxfp8_pack_output_grouped(const void *padded, void *tight, int64_t N, int dtype_bytes,
                               const int *src_row_off, const int *dst_row_off, const int *row_len,
                               const int64_t *elem_pref, int group_num, int64_t total_out_elems,
                               hipStream_t stream) {
    if (total_out_elems == 0)
        return;
    constexpr int kBlock    = 256;
    const int64_t row_bytes = N * dtype_bytes;
    const int64_t grid      = DIVUP<int64_t>(total_out_elems, kBlock);
    mxfp8_pack_output_grouped_kernel<<<grid, kBlock, 0, stream>>>(
        static_cast<const uint8_t *>(padded), static_cast<uint8_t *>(tight), row_bytes, src_row_off,
        dst_row_off, row_len, elem_pref, group_num, total_out_elems);
    PRIMUS_TURBO_CHECK_HIP(hipGetLastError());
}

void hipblaslt_grouped_gemm_mxfp8(const HipblasltMXGroupedGemmParams &params) {
    static HipblasltMXGroupedGemm instance;
    instance.run(params);
}

} // namespace primus_turbo
