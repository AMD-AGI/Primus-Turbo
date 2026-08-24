/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// Torch entry points for the fused MXFP6 packer.
//
// The kernel is gfx950-only (it uses the hardware FP6 conversion), so the whole file sits
// behind the same build gate that drops the kernel from non-gfx950 builds; otherwise these
// ordinary .cpp symbols would still compile and reference a kernel nothing defined.

#include <ATen/hip/HIPContext.h>
#include <torch/extension.h>

#include "../extensions.h"
#include "primus_turbo/common.h"
#include "primus_turbo/quantization.h"

#ifdef BUILD_MXFP6_BACKEND

namespace primus_turbo::pytorch {

namespace {

constexpr int64_t kTileRows        = 256;
constexpr int64_t kKTile           = 128;
constexpr int64_t kPackedTileBytes = 24576;
constexpr int64_t kScaleTileBytes  = 1024;
constexpr int64_t kBlockSize       = 32;

int64_t cdiv(const int64_t x, const int64_t m) { return (x + m - 1) / m; }

// Byte sizes of the (operand, scale) blobs for a [rows, k] operand. Both include the
// guard tiles: their contents are never read, but the space is mandatory because the
// A6W6 assembly derives its row-tile stride from k/128 + 2.
std::pair<int64_t, int64_t> pack_sizes(const int64_t rows, const int64_t k) {
    const int64_t row_tiles = cdiv(rows, kTileRows);
    const int64_t k_tiles   = cdiv(k, kKTile) + MXFP6_GUARD_K_TILES;
    return {row_tiles * k_tiles * kPackedTileBytes, row_tiles * k_tiles * kScaleTileBytes};
}

void check_input(const at::Tensor &input) {
    PRIMUS_TURBO_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    PRIMUS_TURBO_CHECK(input.dim() == 2, "Input must be 2D");
    PRIMUS_TURBO_CHECK(input.is_contiguous(), "Input must be contiguous");
    PRIMUS_TURBO_CHECK(input.scalar_type() == at::kBFloat16 || input.scalar_type() == at::kHalf,
                       "Input must be BFloat16 or Half");
    PRIMUS_TURBO_CHECK(input.size(0) % kBlockSize == 0 && input.size(1) % kBlockSize == 0,
                       "MXFP6 scales strictly per 1x", kBlockSize,
                       " along whichever axis is contracted, and this packs both, so both "
                       "dimensions must be multiples of ",
                       kBlockSize, ". Got [", input.size(0), ", ", input.size(1), "]");
}

// Guard tiles are left uninitialised on purpose: they are never read, and zeroing them
// would add a memset over the whole blob to no effect. Anything comparing packed blobs
// has to mask them (see mxfp6_data_region on the Python side).
at::Tensor empty_blob(const int64_t bytes, const at::Tensor &like) {
    return at::empty({bytes}, like.options().dtype(at::kByte));
}

std::vector<at::Tensor> run(const at::Tensor &input, const MXFP6Direction direction) {
    check_input(input);
    const int64_t M = input.size(0);
    const int64_t N = input.size(1);

    const bool want_row = direction != MXFP6Direction::Col;
    const bool want_col = direction != MXFP6Direction::Row;

    const auto [row_p_bytes, row_s_bytes] = pack_sizes(M, N); // contract N
    const auto [col_p_bytes, col_s_bytes] = pack_sizes(N, M); // contract M

    at::Tensor row_p = empty_blob(want_row ? row_p_bytes : 0, input);
    at::Tensor row_s = empty_blob(want_row ? row_s_bytes : 0, input);
    at::Tensor col_p = empty_blob(want_col ? col_p_bytes : 0, input);
    at::Tensor col_s = empty_blob(want_col ? col_s_bytes : 0, input);

    auto stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();

    if (input.scalar_type() == at::kBFloat16) {
        quantize_mxfp6_impl<dtype::bfloat16>(
            reinterpret_cast<const dtype::bfloat16 *>(input.data_ptr()),
            row_p.data_ptr<uint8_t>(), row_s.data_ptr<uint8_t>(), col_p.data_ptr<uint8_t>(),
            col_s.data_ptr<uint8_t>(), static_cast<int>(M), static_cast<int>(N), direction, stream);
    } else {
        quantize_mxfp6_impl<dtype::float16>(
            reinterpret_cast<const dtype::float16 *>(input.data_ptr()),
            row_p.data_ptr<uint8_t>(), row_s.data_ptr<uint8_t>(), col_p.data_ptr<uint8_t>(),
            col_s.data_ptr<uint8_t>(), static_cast<int>(M), static_cast<int>(N), direction, stream);
    }

    if (direction == MXFP6Direction::Row)
        return {row_p, row_s};
    if (direction == MXFP6Direction::Col)
        return {col_p, col_s};
    return {row_p, row_s, col_p, col_s};
}

MXFP6Direction direction_from_axis(const int64_t axis) {
    PRIMUS_TURBO_CHECK(axis == -1 || axis == 0 || axis == 1,
                       "axis must be 0 (contract rows) or 1/-1 (contract columns), got ", axis);
    return axis == 0 ? MXFP6Direction::Col : MXFP6Direction::Row;
}

} // namespace

std::vector<at::Tensor> quantize_mxfp6(const at::Tensor input, const int64_t axis) {
    return run(input, direction_from_axis(axis));
}

std::vector<at::Tensor> quantize_mxfp6_dual(const at::Tensor input) {
    return run(input, MXFP6Direction::Dual);
}

// Meta implementations. Shapes are pure arithmetic on M and N, so torch.compile can trace
// through the packer without a graph break.
std::vector<at::Tensor> quantize_mxfp6_meta(const at::Tensor input, const int64_t axis) {
    const int64_t M                = input.size(0);
    const int64_t N                = input.size(1);
    const bool    row              = direction_from_axis(axis) == MXFP6Direction::Row;
    const auto [packed, scale]     = row ? pack_sizes(M, N) : pack_sizes(N, M);
    auto opts                      = input.options().dtype(at::kByte);
    return {at::empty({packed}, opts), at::empty({scale}, opts)};
}

std::vector<at::Tensor> quantize_mxfp6_dual_meta(const at::Tensor input) {
    const int64_t M                       = input.size(0);
    const int64_t N                       = input.size(1);
    const auto [row_p_bytes, row_s_bytes] = pack_sizes(M, N);
    const auto [col_p_bytes, col_s_bytes] = pack_sizes(N, M);
    auto        opts                      = input.options().dtype(at::kByte);
    return {at::empty({row_p_bytes}, opts), at::empty({row_s_bytes}, opts),
            at::empty({col_p_bytes}, opts), at::empty({col_s_bytes}, opts)};
}

} // namespace primus_turbo::pytorch

#endif // BUILD_MXFP6_BACKEND
