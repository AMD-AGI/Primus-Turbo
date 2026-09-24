/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// Torch entry points for the MXFP4 (E2M1) weight packer used by A6W4.
//
// The kernel is gfx950-only (hardware FP4 conversion), so the whole file sits behind the
// same build gate that drops the kernel from non-gfx950 builds; otherwise these ordinary
// .cpp symbols would still compile and reference a kernel nothing defined.
//
// Structurally this mirrors quantization_mxfp6.cpp. It is deliberately much smaller: the
// MXFP4 packer exists to pack *weights*, which need no fused prologue and no bias-gradient
// column sum, so there is no run_fused, no prologue mode and no optional operands.

#include <ATen/hip/HIPContext.h>
#include <c10/core/DeviceGuard.h>
#include <torch/extension.h>

#include "../extensions.h"
#include "primus_turbo/common.h"
#include "primus_turbo/quantization.h"

#ifdef BUILD_MXFP4_BACKEND

namespace primus_turbo::pytorch {

namespace {

constexpr int64_t kTileRows        = 256;
constexpr int64_t kKTile           = 128;
constexpr int64_t kPackedTileBytes = 16384; // MXFP6's is 24576
constexpr int64_t kScaleTileBytes  = 1024;
constexpr int64_t kBlockSize       = 32;

int64_t cdiv(const int64_t x, const int64_t m) {
    return (x + m - 1) / m;
}

// Byte sizes of the (operand, scale) blobs for a [rows, k] operand. Both include the
// guard tiles: their contents are never read, but the space is mandatory because the
// A6W4 assembly derives its row-tile stride from k/128 + 2. Must agree with
// aiter.ops.gemm_op_a6w4.mxfp4_gemm_pack_size.
std::pair<int64_t, int64_t> pack_sizes(const int64_t rows, const int64_t k) {
    const int64_t row_tiles = cdiv(rows, kTileRows);
    const int64_t k_tiles   = cdiv(k, kKTile) + MXFP4_GUARD_K_TILES;
    return {row_tiles * k_tiles * kPackedTileBytes, row_tiles * k_tiles * kScaleTileBytes};
}

void check_input(const at::Tensor &input) {
    PRIMUS_TURBO_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    PRIMUS_TURBO_CHECK(input.dim() == 2, "Input must be 2D");
    PRIMUS_TURBO_CHECK(input.is_contiguous(), "Input must be contiguous");
    PRIMUS_TURBO_CHECK(input.scalar_type() == at::kBFloat16 || input.scalar_type() == at::kHalf,
                       "Input must be BFloat16 or Half");
    PRIMUS_TURBO_CHECK(input.size(0) % kBlockSize == 0 && input.size(1) % kBlockSize == 0,
                       "MXFP4 scales strictly per 1x", kBlockSize,
                       " along whichever axis is contracted, and this packs both, so both "
                       "dimensions must be multiples of ",
                       kBlockSize, ". Got [", input.size(0), ", ", input.size(1), "]");
}

// Guard tiles are left uninitialised on purpose: they are never read, and zeroing them
// would add a memset over the whole blob to no effect. Anything comparing packed blobs
// has to mask them (see mxfp4_data_region on the Python side).
at::Tensor empty_blob(const int64_t bytes, const at::Tensor &like) {
    return at::empty({bytes}, like.options().dtype(at::kByte));
}

std::vector<at::Tensor> run(const at::Tensor &input, const MXFP4Direction direction) {
    check_input(input);
    // Allocate and launch on the operand's device rather than the ambient one, which the
    // caller is under no obligation to have set.
    const c10::DeviceGuard device_guard(input.device());
    const int64_t          M = input.size(0);
    const int64_t          N = input.size(1);

    const bool want_row = direction != MXFP4Direction::Col;
    const bool want_col = direction != MXFP4Direction::Row;

    const auto [row_p_bytes, row_s_bytes] = pack_sizes(M, N); // contract N
    const auto [col_p_bytes, col_s_bytes] = pack_sizes(N, M); // contract M

    at::Tensor row_p = empty_blob(want_row ? row_p_bytes : 0, input);
    at::Tensor row_s = empty_blob(want_row ? row_s_bytes : 0, input);
    at::Tensor col_p = empty_blob(want_col ? col_p_bytes : 0, input);
    at::Tensor col_s = empty_blob(want_col ? col_s_bytes : 0, input);

    auto stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();

    if (input.scalar_type() == at::kBFloat16) {
        quantize_mxfp4_impl<dtype::bfloat16>(
            reinterpret_cast<const dtype::bfloat16 *>(input.data_ptr()), row_p.data_ptr<uint8_t>(),
            row_s.data_ptr<uint8_t>(), col_p.data_ptr<uint8_t>(), col_s.data_ptr<uint8_t>(),
            static_cast<int32_t>(M), static_cast<int32_t>(N), direction, stream);
    } else {
        quantize_mxfp4_impl<dtype::float16>(
            reinterpret_cast<const dtype::float16 *>(input.data_ptr()), row_p.data_ptr<uint8_t>(),
            row_s.data_ptr<uint8_t>(), col_p.data_ptr<uint8_t>(), col_s.data_ptr<uint8_t>(),
            static_cast<int32_t>(M), static_cast<int32_t>(N), direction, stream);
    }

    if (direction == MXFP4Direction::Row)
        return {row_p, row_s};
    if (direction == MXFP4Direction::Col)
        return {col_p, col_s};
    return {row_p, row_s, col_p, col_s};
}

MXFP4Direction direction_from_axis(const int64_t axis) {
    PRIMUS_TURBO_CHECK(axis == -1 || axis == 0 || axis == 1,
                       "axis must be 0 (contract rows) or 1/-1 (contract columns), got ", axis);
    return axis == 0 ? MXFP4Direction::Col : MXFP4Direction::Row;
}

} // namespace

std::vector<at::Tensor> quantize_mxfp4_gemm(const at::Tensor input, const int64_t axis) {
    return run(input, direction_from_axis(axis));
}

std::vector<at::Tensor> quantize_mxfp4_gemm_dual(const at::Tensor input) {
    return run(input, MXFP4Direction::Dual);
}

// Meta implementations. Shapes are pure arithmetic on M and N, so torch.compile can trace
// through the packer without a graph break.
std::vector<at::Tensor> quantize_mxfp4_gemm_meta(const at::Tensor input, const int64_t axis) {
    const int64_t M            = input.size(0);
    const int64_t N            = input.size(1);
    const bool    row          = direction_from_axis(axis) == MXFP4Direction::Row;
    const auto [packed, scale] = row ? pack_sizes(M, N) : pack_sizes(N, M);
    auto opts                  = input.options().dtype(at::kByte);
    return {at::empty({packed}, opts), at::empty({scale}, opts)};
}

std::vector<at::Tensor> quantize_mxfp4_gemm_dual_meta(const at::Tensor input) {
    const int64_t M                       = input.size(0);
    const int64_t N                       = input.size(1);
    const auto [row_p_bytes, row_s_bytes] = pack_sizes(M, N);
    const auto [col_p_bytes, col_s_bytes] = pack_sizes(N, M);
    auto opts                             = input.options().dtype(at::kByte);
    return {at::empty({row_p_bytes}, opts), at::empty({row_s_bytes}, opts),
            at::empty({col_p_bytes}, opts), at::empty({col_s_bytes}, opts)};
}

} // namespace primus_turbo::pytorch

#endif // BUILD_MXFP4_BACKEND
