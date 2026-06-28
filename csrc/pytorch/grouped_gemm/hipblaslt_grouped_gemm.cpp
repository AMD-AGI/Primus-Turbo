// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <mutex>

#include "primus_turbo/gemm.h"
#include "primus_turbo/grouped_gemm.h"
#include "pytorch/extensions.h"
#include "pytorch/type_traits.h"

namespace primus_turbo::pytorch {

inline HipblasltGroupedGemmParams
make_hipblaslt_grouped_gemm_params(const at::Tensor &a, const at::Tensor &b, at::Tensor &c,
                                   const at::Tensor &group_lens, const at::Tensor &group_offs,
                                   bool transA, bool transB, at::Tensor workspace) {
    HipblasltGroupedGemmParams params;

    params.a_ptr   = reinterpret_cast<void *>(a.data_ptr());
    params.a_type  = get_hipblaslt_dtype(a.scalar_type());
    params.a_shape = a.sizes().vec();

    params.b_ptr   = reinterpret_cast<void *>(b.data_ptr());
    params.b_type  = get_hipblaslt_dtype(b.scalar_type());
    params.b_shape = b.sizes().vec();

    params.c_ptr   = reinterpret_cast<void *>(c.data_ptr());
    params.c_type  = get_hipblaslt_dtype(c.scalar_type());
    params.c_shape = c.sizes().vec();

    params.group_lens_ptr = reinterpret_cast<const int64_t *>(group_lens.data_ptr());
    params.group_offs_ptr = reinterpret_cast<const int64_t *>(group_offs.data_ptr());
    params.transA         = transA;
    params.transB         = transB;
    params.group_num      = group_lens.numel();
    params.stream         = at::cuda::getCurrentCUDAStream();
    params.workspace      = workspace.data_ptr();
    params.handle         = at::cuda::getCurrentCUDABlasLtHandle();
    return params;
}

inline HipblasltGroupedGemmParams make_hipblaslt_grouped_gemm_fp8_params(
    const at::Tensor &a, const at::Tensor &b, at::Tensor &c, const at::Tensor &a_scales,
    const at::Tensor &b_scales, const at::Tensor &group_lens, const at::Tensor &group_offs,
    bool transA, bool transB, hipblasLtMatmulMatrixScale_t scale_mode, at::Tensor workspace) {
    HipblasltGroupedGemmParams params;

    params.a_ptr       = reinterpret_cast<const void *>(a.data_ptr());
    params.a_scale_ptr = reinterpret_cast<const void *>(a_scales.data_ptr());
    params.a_type      = get_hipblaslt_dtype(a.scalar_type());
    params.a_shape     = a.sizes().vec();

    params.b_ptr       = reinterpret_cast<const void *>(b.data_ptr());
    params.b_scale_ptr = reinterpret_cast<const void *>(b_scales.data_ptr());
    params.b_type      = get_hipblaslt_dtype(b.scalar_type());
    params.b_shape     = b.sizes().vec();

    params.c_ptr   = reinterpret_cast<void *>(c.data_ptr());
    params.c_type  = get_hipblaslt_dtype(c.scalar_type());
    params.c_shape = c.sizes().vec();

    params.group_lens_ptr = reinterpret_cast<const int64_t *>(group_lens.data_ptr());
    params.group_offs_ptr = reinterpret_cast<const int64_t *>(group_offs.data_ptr());
    params.transA         = transA;
    params.transB         = transB;
    params.group_num      = group_lens.numel();
    params.stream         = at::cuda::getCurrentCUDAStream();
    params.workspace      = workspace.data_ptr();

    params.use_low_precision = true;

    params.handle     = at::cuda::getCurrentCUDABlasLtHandle();
    params.scale_mode = scale_mode;

    return params;
}

at::Tensor hipblaslt_grouped_gemm(at::Tensor &a, at::Tensor &b, at::Tensor &group_lens,
                                  at::Tensor &group_offs, const bool transA, const bool transB,
                                  const bool pre_sync) {
    // Check
    PRIMUS_TURBO_CHECK(is_16bit_floating_point_dtype(a.scalar_type()),
                       "hipblaslt_grouped_gemm only supports float16 and bfloat16");
    PRIMUS_TURBO_CHECK(is_16bit_floating_point_dtype(b.scalar_type()),
                       "hipblaslt_grouped_gemm only supports float16 and bfloat16");
    PRIMUS_TURBO_CHECK(group_lens.scalar_type() == at::kLong);
    PRIMUS_TURBO_CHECK(group_offs.scalar_type() == at::kLong);
    PRIMUS_TURBO_CHECK(a.scalar_type() == b.scalar_type(), "a and b dtype mismatch");

    // Create output tensor
    at::Tensor c;
    if (transA) {
        const int64_t bs = group_lens.numel();
        const int64_t m  = a.size(1);
        const int64_t n  = transB ? b.size(0) : b.size(1);
        c                = at::empty({bs, m, n}, a.options());
    } else {
        const int64_t m = a.size(0);
        const int64_t n = transB ? b.size(1) : b.size(2);
        c               = at::empty({m, n}, a.options());
    }

    const int64_t workspace_size = primus_turbo::get_hipblaslt_grouped_gemm_workspace_size();
    at::Tensor    workspace =
        at::empty({workspace_size}, at::TensorOptions().dtype(at::kByte).device(a.device()));

    auto params = make_hipblaslt_grouped_gemm_params(a, b, c, group_lens, group_offs, transA,
                                                     transB, workspace);
    primus_turbo::hipblaslt_grouped_gemm(params, pre_sync);
    return c;
}

// MXFP8 grouped forward / dgrad (NT, b = [E, N, K]). hipBLASLt VEC32_UE8M0 on
// gfx1250 needs the e8m0 block scales in the groups-of-4 swizzled layout with the
// activation m-dim padded to 128. We pad A + swizzle A/B scales with batched HIP
// kernels into PyTorch-allocated arenas (no host syncs), run the per-group GEMMs
// on the multi-stream pool, then pack the real rows into the tight output.
//   a        : [M_in, K] fp8 (per-group 32-row-padded activation)
//   b        : [E, N, K] fp8 weights  (N a 128-multiple)
//   a_scales : [M_in, K/32] e8m0 ;  b_scales : [E, N, K/32] e8m0
//   group_offs     : [E+1] int64 padded read offsets (into a / a_scales rows)
//   group_offs_out : [E+1] int64 tight write offsets (into the [total_m, N] output)
// Returns the tight [total_m, N] output (out_dtype).
at::Tensor hipblaslt_grouped_gemm_mxfp8(at::Tensor &a, at::Tensor &b, at::Tensor &a_scales,
                                        at::Tensor &b_scales, at::Tensor &group_lens,
                                        at::Tensor &group_offs, at::Tensor &group_offs_out,
                                        at::ScalarType out_dtype) {
    PRIMUS_TURBO_CHECK(is_8bit_floating_point_dtype(a.scalar_type()));
    PRIMUS_TURBO_CHECK(is_8bit_floating_point_dtype(b.scalar_type()));
    PRIMUS_TURBO_CHECK(a.dim() == 2 && b.dim() == 3, "MX fwd: a 2D, b 3D");
    PRIMUS_TURBO_CHECK(group_lens.scalar_type() == at::kLong);
    PRIMUS_TURBO_CHECK(group_offs.scalar_type() == at::kLong);
    PRIMUS_TURBO_CHECK(group_offs_out.scalar_type() == at::kLong);
    PRIMUS_TURBO_CHECK(out_dtype == at::kBFloat16 || out_dtype == at::kHalf);
    PRIMUS_TURBO_CHECK(a_scales.scalar_type() == at::kFloat8_e8m0fnu);
    PRIMUS_TURBO_CHECK(b_scales.scalar_type() == at::kFloat8_e8m0fnu);

    constexpr int64_t kBlk    = 32;
    constexpr int64_t kMPad   = 128;
    constexpr int64_t kSGrp   = 4;
    const int         E       = static_cast<int>(group_lens.numel());
    const int64_t     K       = a.size(1);
    const int64_t     N       = b.size(1);
    const int64_t     ks      = K / kBlk;
    const int64_t     ks_pad  = ((ks + kSGrp - 1) / kSGrp) * kSGrp;
    PRIMUS_TURBO_CHECK(N % kMPad == 0, "MX fwd C++ path requires N % 128 == 0");
    PRIMUS_TURBO_CHECK(K % kBlk == 0, "K must be a multiple of 32");

    auto stream = at::cuda::getCurrentCUDAStream();

    // group_lens / group_offs are device int64 but host-accessible on gfx1250
    // (same assumption the BF16 compute_args relies on).
    const int64_t *gl  = reinterpret_cast<const int64_t *>(group_lens.data_ptr());
    const int64_t *go  = reinterpret_cast<const int64_t *>(group_offs.data_ptr());
    const int64_t *goo = reinterpret_cast<const int64_t *>(group_offs_out.data_ptr());

    // Host per-group metadata (indexed by valid group order).
    std::vector<int>     h_a_read_off, h_len, h_a_mpad_i, h_b_swz_rows;
    std::vector<int>     h_pack_src_row, h_pack_dst_row;
    std::vector<int64_t> h_a_pad_row_off, h_a_scale_off, h_b_row_off, h_b_scale_off, h_c_off_bytes;
    std::vector<int64_t> h_a_mpad, h_b_mpad, h_kdim;
    std::vector<int64_t> h_apad_elem_pref, h_aswz_pref, h_bswz_pref, h_pack_pref;
    // Over-allocate to the padded input rows (a.size(0)), matching the Triton MX
    // backend + the registered fake; pack writes real rows at group_offs_out and
    // the caller slices [:total_m].
    at::Tensor out = at::empty({a.size(0), N}, a.options().dtype(out_dtype));

    int64_t a_pad_rows = 0, a_swz_elems = 0, b_swz_elems = 0, c_pad_rows = 0, pack_bytes_total = 0;
    int     out_bytes = (out_dtype == at::kBFloat16 || out_dtype == at::kHalf) ? 2 : 2;
    for (int g = 0; g < E; ++g) {
        const int64_t len = gl[g];
        if (len <= 0)
            continue;
        const int64_t m_pad = ((len + kMPad - 1) / kMPad) * kMPad;
        h_a_read_off.push_back(static_cast<int>(go[g]));
        h_len.push_back(static_cast<int>(len));
        h_a_mpad_i.push_back(static_cast<int>(m_pad));
        h_b_swz_rows.push_back(static_cast<int>(N));

        h_a_pad_row_off.push_back(a_pad_rows);
        h_apad_elem_pref.push_back(a_pad_rows * K); // running A-pad element prefix
        h_a_scale_off.push_back(a_swz_elems);
        h_aswz_pref.push_back(a_swz_elems);
        h_b_row_off.push_back(static_cast<int64_t>(g) * N); // weight rows (contiguous E*N)
        h_b_scale_off.push_back(b_swz_elems);
        h_bswz_pref.push_back(b_swz_elems);
        h_c_off_bytes.push_back(c_pad_rows * N * out_bytes);

        h_pack_src_row.push_back(static_cast<int>(c_pad_rows));
        h_pack_dst_row.push_back(static_cast<int>(goo[g]));
        h_pack_pref.push_back(pack_bytes_total);

        h_a_mpad.push_back(m_pad);
        h_b_mpad.push_back(N);
        h_kdim.push_back(K);

        a_pad_rows += m_pad;
        a_swz_elems += m_pad * ks_pad;
        b_swz_elems += N * ks_pad;
        c_pad_rows += m_pad;
        pack_bytes_total += len * N * out_bytes;
    }
    const int valid = static_cast<int>(h_len.size());
    if (valid == 0)
        return out;

    // Append the terminal prefix entries (size valid+1) for the group-scan kernels.
    h_apad_elem_pref.push_back(a_pad_rows * K);
    h_aswz_pref.push_back(a_swz_elems);
    h_bswz_pref.push_back(b_swz_elems);
    h_pack_pref.push_back(pack_bytes_total);

    // Arenas (PyTorch caching allocator).
    auto opt_u8  = at::TensorOptions().dtype(at::kByte).device(a.device());
    auto opt_i32 = at::TensorOptions().dtype(at::kInt).device(a.device());
    auto opt_i64 = at::TensorOptions().dtype(at::kLong).device(a.device());
    at::Tensor a_pad      = at::empty({a_pad_rows * K}, opt_u8);
    at::Tensor a_swz      = at::empty({a_swz_elems}, opt_u8);
    at::Tensor b_swz      = at::empty({b_swz_elems}, opt_u8);
    at::Tensor c_pad      = at::empty({c_pad_rows * N * out_bytes}, opt_u8);

    // Upload small host arrays to device for the batched kernels.
    auto to_dev_i32 = [&](const std::vector<int> &v) {
        return at::from_blob((void *) v.data(), {(int64_t) v.size()}, at::kInt).to(opt_i32.device());
    };
    auto to_dev_i64 = [&](const std::vector<int64_t> &v) {
        return at::from_blob((void *) v.data(), {(int64_t) v.size()}, at::kLong)
            .to(opt_i64.device());
    };
    at::Tensor d_a_read_off  = to_dev_i32(h_a_read_off);
    at::Tensor d_len         = to_dev_i32(h_len);
    at::Tensor d_a_mpad_i    = to_dev_i32(h_a_mpad_i);
    at::Tensor d_b_swz_rows  = to_dev_i32(h_b_swz_rows);
    at::Tensor d_a_pad_roff  = to_dev_i64(h_a_pad_row_off);
    at::Tensor d_a_scale_off = to_dev_i64(h_a_scale_off);
    at::Tensor d_b_scale_off = to_dev_i64(h_b_scale_off);
    at::Tensor d_apad_pref   = to_dev_i64(h_apad_elem_pref);
    at::Tensor d_aswz_pref   = to_dev_i64(h_aswz_pref);
    at::Tensor d_bswz_pref   = to_dev_i64(h_bswz_pref);
    at::Tensor d_pack_src    = to_dev_i32(h_pack_src_row);
    at::Tensor d_pack_dst    = to_dev_i32(h_pack_dst_row);
    at::Tensor d_pack_pref   = to_dev_i64(h_pack_pref);
    // B swizzle row offsets: weight g starts at row g*N; per valid group.
    std::vector<int> h_b_read_off;
    for (int g = 0; g < E; ++g) {
        if (gl[g] > 0)
            h_b_read_off.push_back(static_cast<int>(static_cast<int64_t>(g) * N));
    }
    at::Tensor d_b_read_off = to_dev_i32(h_b_read_off);

    // 1) Pad A data (len,K) -> (m_pad,K) per group.
    primus_turbo::mxfp8_pad_data_grouped(
        reinterpret_cast<const uint8_t *>(a.data_ptr()),
        reinterpret_cast<uint8_t *>(a_pad.data_ptr()), K,
        reinterpret_cast<const int *>(d_a_read_off.data_ptr()),
        reinterpret_cast<const int *>(d_len.data_ptr()),
        reinterpret_cast<const int64_t *>(d_a_pad_roff.data_ptr()),
        reinterpret_cast<const int64_t *>(d_apad_pref.data_ptr()), valid, a_pad_rows * K, stream);

    // 2) Swizzle A scales (rows=len -> m_pad) and B scales (rows=N).
    primus_turbo::mxfp8_swizzle_scale_grouped(
        reinterpret_cast<const uint8_t *>(a_scales.data_ptr()),
        reinterpret_cast<uint8_t *>(a_swz.data_ptr()), ks, ks_pad,
        reinterpret_cast<const int *>(d_a_read_off.data_ptr()),
        reinterpret_cast<const int *>(d_len.data_ptr()),
        reinterpret_cast<const int *>(d_a_mpad_i.data_ptr()),
        reinterpret_cast<const int64_t *>(d_a_scale_off.data_ptr()),
        reinterpret_cast<const int64_t *>(d_aswz_pref.data_ptr()), valid, a_swz_elems, stream);
    primus_turbo::mxfp8_swizzle_scale_grouped(
        reinterpret_cast<const uint8_t *>(b_scales.data_ptr()),
        reinterpret_cast<uint8_t *>(b_swz.data_ptr()), ks, ks_pad,
        reinterpret_cast<const int *>(d_b_read_off.data_ptr()),
        reinterpret_cast<const int *>(d_b_swz_rows.data_ptr()),
        reinterpret_cast<const int *>(d_b_swz_rows.data_ptr()),
        reinterpret_cast<const int64_t *>(d_b_scale_off.data_ptr()),
        reinterpret_cast<const int64_t *>(d_bswz_pref.data_ptr()), valid, b_swz_elems, stream);

    // 3) Per-group GEMMs into the padded-block output.
    const int64_t workspace_size = primus_turbo::get_hipblaslt_grouped_gemm_workspace_size();
    at::Tensor    workspace       = at::empty({workspace_size}, opt_u8);
    primus_turbo::HipblasltMXGroupedGemmParams mp;
    mp.a_padded        = a_pad.data_ptr();
    mp.a_swz_scale     = a_swz.data_ptr();
    mp.b_data          = b.data_ptr();
    mp.b_swz_scale     = b_swz.data_ptr();
    mp.c_data      = c_pad.data_ptr();
    mp.ab_type     = get_hipblaslt_dtype(a.scalar_type());
    mp.c_type      = get_hipblaslt_dtype(out_dtype);
    mp.ldc         = N;
    // Per-group descriptor arrays live on host; the runner issues GEMMs host-side.
    // All arrays are already filtered to the `valid` groups (len>0).
    mp.a_row_off   = h_a_pad_row_off.data();
    mp.a_scale_off = h_a_scale_off.data();
    mp.b_row_off   = h_b_row_off.data();
    mp.b_scale_off = h_b_scale_off.data();
    mp.c_off_bytes = h_c_off_bytes.data();
    mp.a_mpad      = h_a_mpad.data();
    mp.b_mpad      = h_b_mpad.data();
    mp.kdim        = h_kdim.data();
    mp.group_num   = valid;
    mp.stream      = stream;
    mp.workspace   = workspace.data_ptr();
    mp.handle      = at::cuda::getCurrentCUDABlasLtHandle();
    primus_turbo::hipblaslt_grouped_gemm_mxfp8(mp);

    // 4) Pack real rows from padded blocks into the tight output.
    primus_turbo::mxfp8_pack_output_grouped(
        c_pad.data_ptr(), out.data_ptr(), N, out_bytes,
        reinterpret_cast<const int *>(d_pack_src.data_ptr()),
        reinterpret_cast<const int *>(d_pack_dst.data_ptr()),
        reinterpret_cast<const int *>(d_len.data_ptr()),
        reinterpret_cast<const int64_t *>(d_pack_pref.data_ptr()), valid, pack_bytes_total, stream);

    return out;
}

at::Tensor hipblaslt_grouped_gemm_fp8(at::Tensor &a, at::Tensor &b, at::Tensor &a_scales,
                                      at::Tensor &b_scales, at::Tensor &group_lens,
                                      at::Tensor &group_offs, const bool transA, const bool transB,
                                      at::ScalarType out_dtype, const std::string &granularity,
                                      const bool pre_sync) {
    // Check
    PRIMUS_TURBO_CHECK(is_8bit_floating_point_dtype(a.scalar_type()));
    PRIMUS_TURBO_CHECK(is_8bit_floating_point_dtype(b.scalar_type()));
    PRIMUS_TURBO_CHECK(group_lens.scalar_type() == at::kLong);
    PRIMUS_TURBO_CHECK(group_offs.scalar_type() == at::kLong);
    PRIMUS_TURBO_CHECK(out_dtype == at::kBFloat16 || out_dtype == at::kHalf,
                       "out_dtype must be kBFloat16 or kHalf");
    PRIMUS_TURBO_CHECK(granularity == "TENSORWISE", "granularity must be 'TENSORWISE'");

    // Scale mode
    hipblasLtMatmulMatrixScale_t scale_mode = HIPBLASLT_MATMUL_MATRIX_SCALE_END;
    if (granularity == "TENSORWISE") {
        scale_mode = HIPBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
    } else {
        PRIMUS_TURBO_ERROR("Invalid granularity.");
    }

    // Create output tensor
    at::Tensor c;
    if (transA) {
        const int64_t bs = group_lens.numel();
        const int64_t m  = a.size(1);
        const int64_t n  = transB ? b.size(0) : b.size(1);
        c                = at::empty({bs, m, n}, a.options().dtype(out_dtype));
    } else {
        const int64_t m = a.size(0);
        const int64_t n = transB ? b.size(1) : b.size(2);
        c               = at::empty({m, n}, a.options().dtype(out_dtype));
    }

    const int64_t workspace_size = primus_turbo::get_hipblaslt_grouped_gemm_workspace_size();
    at::Tensor    workspace =
        at::empty({workspace_size}, at::TensorOptions().dtype(at::kByte).device(a.device()));

    auto params = make_hipblaslt_grouped_gemm_fp8_params(
        a, b, c, a_scales, b_scales, group_lens, group_offs, transA, transB, scale_mode, workspace);
    primus_turbo::hipblaslt_grouped_gemm(params, pre_sync);

    return c;
}

} // namespace primus_turbo::pytorch
