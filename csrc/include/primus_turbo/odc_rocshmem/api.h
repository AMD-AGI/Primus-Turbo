// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

// Kernel-level API for the ODC rocSHMEM distributed backends.
//
// Declares the single-node host-API surface and the multi-node GPU-Direct Async
// (GDA) surface (host launchers for the device gather / reduce-scatter kernels).
// The implementations live in csrc/kernels/odc_rocshmem/ and are compiled into
// libprimus_turbo_kernels.so (which links rocSHMEM); the pytorch/jax bindings only
// include this header and link the kernels library, so they need no rocSHMEM
// include/link flags themselves (mirrors DeepEP's csrc/include .. api.h split).
//
// Only plain scalar / pointer types cross this boundary (no rocSHMEM, HIP-device,
// or torch types), so the header is safe to include from a rocSHMEM-agnostic TU.

#pragma once

#include <cstddef>

namespace primus_turbo::odc_rocshmem {

// ---------------------------------------------------------------------------
// Single-node host-API backend (uid bootstrap + symmetric heap + peer-ptr).
// ---------------------------------------------------------------------------
namespace host {

int       rs_uid_bytes();
void      rs_get_uid(char *out);
void      rs_init_uid(int rank, int nranks, const char *bytes);
void      rs_get_ctx_fields(long long *a, long long *b);
int       rs_my_pe();
int       rs_n_pes();
long long rs_malloc(std::size_t n);
long long rs_ptr(long long p, int pe);
void      rs_barrier();
void      rs_finalize();

} // namespace host

// ---------------------------------------------------------------------------
// Multi-node GPU-Direct Async (GDA) backend: uid bootstrap, symmetric heap, and
// host launchers for the device gather / pull-based reduce-scatter-accumulate
// kernels.
// ---------------------------------------------------------------------------
namespace gda {

// Single-node-binding-compatible host surface (so ODC's python backend can reuse it).
int       rs_uid_bytes();
void      rs_get_uid(char *out);
void      rs_init_uid(int rank, int nranks, const char *bytes);
int       rs_my_pe();
int       rs_n_pes();
long long rs_malloc(std::size_t n);
long long rs_ptr(long long p, int pe);
int       rs_is_remote(long long base, int pe);
void      rs_barrier();
void      rs_finalize();

// host<->device copy + memset helpers (used by the standalone kernel numeric test).
void gda_h2d(long long dptr, const void *host, std::size_t nbytes);
void gda_d2h(long long dptr, void *host, std::size_t nbytes);
void gda_memset(long long dptr, int val, std::size_t nbytes);

// Strided page-touch warm-up launcher. Returns hipError_t.
int gda_strided_touch(long long input_sym, std::size_t seg_off_bytes, std::size_t seg_bytes,
                      int n_pes, std::size_t stride_bytes, std::size_t touch_bytes,
                      long long scratch_sym, std::size_t scratch_stride, int nblocks);

// Copy nbytes src->dst (both device) then system-fence; returns hipError_t.
int gda_stage_fence(long long dst, long long src, std::size_t nbytes);

// HDP (Host Data Path) flush register for GPUDirect-RDMA write visibility.
int gda_hdp_init();
int gda_hdp_flush();

// GPU-direct gather: returns hipError_t (0 == success).
int gda_gather(long long target, long long src, std::size_t nbytes, const int *peers_host,
               int n_peers, std::size_t stride_bytes);

// Async gather on a caller-provided stream (no sync; caller orders via stream).
int gda_gather_async(long long target, long long src, std::size_t nbytes, const int *peers_host,
                     int n_peers, std::size_t stride_bytes, long long stream);

// GPU-direct pull-based reduce-scatter accumulate into fp32 acc.
int gda_reduce_scatter_acc(long long acc_fp32, long long input_sym, std::size_t seg_off_bytes,
                           std::size_t shard_elems, int n_pes, long long scratch_sym,
                           std::size_t scratch_stride_bytes, int dtype_code, int nblocks);

// Comm/compute-overlap variant: launch on a side stream and return without sync.
int gda_reduce_scatter_acc_async(long long acc_fp32, long long input_sym, std::size_t seg_off_bytes,
                                 std::size_t shard_elems, int n_pes, long long scratch_sym,
                                 std::size_t scratch_stride_bytes, int dtype_code, int nblocks);

// Wait for all pending overlapped reduce-scatter kernels on the side stream.
int gda_rs_overlap_sync();

// Device-side completion/visibility microbench. Returns mismatch count (>=0) or <0 on error.
int gda_microbench(int n, int reps, int do_quiet);

} // namespace gda

} // namespace primus_turbo::odc_rocshmem
