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
#include <c10/core/DeviceGuard.h>
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

int64_t cdiv(const int64_t x, const int64_t m) {
    return (x + m - 1) / m;
}

// Byte sizes of the (operand, scale) blobs for a [rows, k] operand. Both include the
// guard tiles: their contents are never read, but the space is mandatory because the
// A6W6 assembly derives its row-tile stride from k/128 + 2.
std::pair<int64_t, int64_t> pack_sizes(const int64_t rows, const int64_t k) {
    const int64_t row_tiles = cdiv(rows, kTileRows);
    const int64_t k_tiles   = cdiv(k, kKTile) + MXFP6_GUARD_K_TILES;
    return {row_tiles * k_tiles * kPackedTileBytes, row_tiles * k_tiles * kScaleTileBytes};
}

// The `fmt` argument of the quantize_mx_* ops: which format each direction is emitted in.
//   0  FP6 both ways             -- the A6W6 blobs, as the quantize_mxfp6_* ops
//   1  A4W4 gradient             -- AITER f4gemm A operand both ways (dgrad contracts the
//                                   gradient's columns, wgrad its rows)
//   2  A4W4 activation / weight  -- FP6 rows for the forward, f4gemm B operand columns
//   3  gradient, wgrad-only A4W4 -- FP6 rows (A6W6 dgrad), f4gemm A operand columns
//   4  gradient, dgrad-only A4W4 -- f4gemm A operand rows, FP6 columns (A6W6 wgrad)
//   5, 6, 7  formats 1, 3, 4 with stochastic rounding in their FP4 directions
//   8  gradient, FlyDSL operands both ways (plain codes + plain E8M0 scales); 12 = 8 with SR
//   9  activation / weight for FlyDSL: FP6 rows, plain FP4 columns
// fmt >= 0x1000: FlyDSL packed scales. 0x1000 | sr * 0x800 | col_code << 4 |
// row_code, a direction code being 0 (FP6) or 0x8 | is_b | (nt == 3) << 1 | (ilv == 4) << 2 for the
// consuming GEMM operand (see MXTilePack). Gradients pass two codes, activations / weights FP6 rows.
// Codes without 0x8: 0x1 | is_b << 1 = MXFP6 K128-blocked planes; 0x5 | is_b << 1 = MXFP4 with
// K128-blocked codes (row direction, round to nearest); both with scales at nt 4, no interleave.
constexpr int64_t kTileFmt = 0x1000, kTileSr = 0x800;
//
// fmt bits 16-24: per-direction FP4 options, on top of any of the formats above, for the FP4
// directions only (FP6 directions keep their own rotation and scale rule; setting a row or column
// option on an FP6 direction is an error). All zero is the default emit, bit for bit.
//   bits 16-17 / 18-19  row / column scale rule: 0 RCEIL, 1-3 = scale_rounding_mode 0-2
//   bits 20-21 / 22-23  row / column Hadamard:   0 H32, 1 none, 2 H16
//   bit  24             2-D 32x32 block scaling of every FP4 direction (weights); needs no Hadamard
//   bit  26             the column direction (packed FP4 tile, role B) K256-outer: codes [K/256, rows, 128]
//                       and the scale slab K256-outer, so any 256-aligned K range is contiguous in both
//   bit  25             stochastic rounding of the column direction only (packed-FlyDSL FP4
//   columns): the
//                       backward copy of an activation / weight, whose forward rows stay
//                       round-to-nearest
// The Hadamard is applied along the contraction axis, so a GEMM's two operands must carry the same
// choice; the caller (Primus) derives both operands' flags from one per-GEMM setting.
constexpr int64_t kFmtBaseMask = 0xFFFF;

int64_t fmt_base(const int64_t fmt) {
    PRIMUS_TURBO_CHECK((fmt >> 27) == 0, "unknown fmt bits in ", fmt);
    return fmt & kFmtBaseMask;
}

MXPackFmt ts_dir_fmt(int64_t fmt, const bool col) {
    const bool col_sr  = col && ((fmt >> 25) & 1);
    fmt                = fmt_base(fmt);
    const int64_t code = col ? (fmt >> 4) & 0xF : fmt & 0xF;
    if (code == 0)
        return MXPackFmt::Fp6;
    if (!(code & 0x8)) {
        if (code & 0x4) { // 0x5 | is_b << 1: MXFP4, K128-blocked codes + FlyDSL packed scales (nt 4, ilv 0)
            PRIMUS_TURBO_CHECK((code & 0x1) && !col, "K128-blocked FP4 is a row direction (fmt ", fmt, ")");
            PRIMUS_TURBO_CHECK(!(fmt & kTileSr), "K128-blocked FP4 rows take no stochastic rounding");
            return MXPackFmt::Fp4TileK128;
        }
        // 0x1 | is_b << 1: MXFP6, K128-blocked C0/C1 planes + FlyDSL packed scales (nt 4, ilv 0).
        PRIMUS_TURBO_CHECK((code & 0x1) && !(code & 0x4), "bad FlyDSL direction code ", code, " in fmt ", fmt);
        PRIMUS_TURBO_CHECK(!(fmt & kTileSr), "MXFP6 K128-blocked directions take no stochastic rounding");
        return MXPackFmt::Fp6Tile;
    }
    return (fmt & kTileSr) || col_sr ? MXPackFmt::Fp4TileSr : MXPackFmt::Fp4Tile;
}

// The consuming GEMM's operand parameters for one direction of a [M, N] input (row direction
// contracts N, column direction contracts M).
std::pair<MXPackFmt, MXPackFmt> fmt_pair(int64_t fmt);

// The FP4 options of one direction (fmt bits 16-24), checked against that direction's format.
void set_fp4_options(MXTilePack &p, const int64_t fmt, const bool col) {
    const int64_t round  = (fmt >> (col ? 18 : 16)) & 0x3;
    const int64_t had    = (fmt >> (col ? 22 : 20)) & 0x3;
    const bool    tile2d = (fmt >> 24) & 0x1;
    const auto    dirs   = fmt_pair(fmt);
    const auto    f      = col ? dirs.second : dirs.first;
    if (f == MXPackFmt::Fp6 || f == MXPackFmt::Fp6Tile) {
        PRIMUS_TURBO_CHECK(round == 0 && had == 0,
                           "FP4 scale rule / Hadamard options set on an FP6 ",
                           col ? "column" : "row", " direction (fmt ", fmt, ")");
        return;
    }
    PRIMUS_TURBO_CHECK(had != 3, "fmt Hadamard code 3 is undefined (fmt ", fmt, ")");
    PRIMUS_TURBO_CHECK(!((fmt >> 25) & 1) || (fmt_base(fmt) & kTileFmt),
                       "column-only stochastic rounding is for packed-FlyDSL formats (fmt ", fmt,
                       ")");
    PRIMUS_TURBO_CHECK(!tile2d || had == 1, "2-D block scaling needs no Hadamard (fmt ", fmt, ")");
    p.fp4_round  = static_cast<int32_t>(round);
    p.fp4_had    = static_cast<int32_t>(had);
    p.fp4_tile2d = tile2d ? 1 : 0;
}

MXTilePack ts_dir(const int64_t fmt_in, const int64_t M, const int64_t N, const bool col) {
    MXTilePack p;
    set_fp4_options(p, fmt_in, col);
    const int64_t fmt = fmt_base(fmt_in);
    if (!(fmt & kTileFmt))
        return p;
    const int64_t code = col ? (fmt >> 4) & 0xF : fmt & 0xF;
    if (code == 0)
        return p;
    const int64_t k = col ? M : N;
    if (!(code & 0x8)) { // MXFP6 / MXFP4 K128-blocked: 0x1 (| 0x4) | is_b << 1, scales nt 4 / ilv 0
        p.is_b = (code >> 1) & 1;
        p.nt   = 4;
        p.ilv  = 0;
        p.k128 = static_cast<int32_t>((k + 255) / 256 * 2);
        p.rows = static_cast<int32_t>(col ? N : M);
        return p;
    }
    p.is_b = code & 1;
    p.nt   = (code & 2) ? 3 : 4;
    p.ilv  = (code & 4) ? 4 : 0;
    p.k128 = static_cast<int32_t>((k + 255) / 256 * 2);
    p.rows = static_cast<int32_t>(col ? N : M);
    if (col && ((fmt_in >> 26) & 1)) {
        PRIMUS_TURBO_CHECK(p.is_b && p.nt == 4, "a K256-outer column is a role-B 256-tile direction (fmt ", fmt_in, ")");
        p.kouter = 1;
    }
    return p;
}

std::pair<MXPackFmt, MXPackFmt> fmt_pair(int64_t fmt) {
    if (fmt_base(fmt) & kTileFmt) // ts_dir_fmt reads the column-SR bit, so it takes the full fmt
        return {ts_dir_fmt(fmt, false), ts_dir_fmt(fmt, true)};
    fmt = fmt_base(fmt);
    switch (fmt) {
    case 0:
        return {MXPackFmt::Fp6, MXPackFmt::Fp6};
    case 1:
        return {MXPackFmt::Fp4A, MXPackFmt::Fp4A};
    case 2:
        return {MXPackFmt::Fp6, MXPackFmt::Fp4B};
    case 3:
        return {MXPackFmt::Fp6, MXPackFmt::Fp4A};
    case 4:
        return {MXPackFmt::Fp4A, MXPackFmt::Fp6};
    case 5:
        return {MXPackFmt::Fp4ASr, MXPackFmt::Fp4ASr};
    case 6:
        return {MXPackFmt::Fp6, MXPackFmt::Fp4ASr};
    case 7:
        return {MXPackFmt::Fp4ASr, MXPackFmt::Fp6};
    case 8:
        return {MXPackFmt::Fp4Plain, MXPackFmt::Fp4Plain};
    case 9:
        return {MXPackFmt::Fp6, MXPackFmt::Fp4Plain};
    case 12:
        return {MXPackFmt::Fp4PlainSr, MXPackFmt::Fp4PlainSr};
    case 16:  // A4W4 tile-blob kernels: gradient, the C0 FP4 tile blob both ways
        return {MXPackFmt::Fp4Blob, MXPackFmt::Fp4Blob};
    case 17:  // activation / weight for them: FP6 rows (forward), FP4 blob columns
        return {MXPackFmt::Fp6, MXPackFmt::Fp4Blob};
    case 18:  // 16 with stochastic rounding
        return {MXPackFmt::Fp4BlobSr, MXPackFmt::Fp4BlobSr};
    default:
        PRIMUS_TURBO_CHECK(false, "fmt must be 0 (fp6), 1 (a4w4 gradient), 2 (a4w4 "
                                  "activation/weight), 3 (gradient, wgrad-only a4w4) or 4 "
                                  "(gradient, dgrad-only a4w4), 5-7 (1, 3, 4 with SR), 8 / 9 / 12 (FlyDSL operands), got ", fmt);
        return {MXPackFmt::Fp6, MXPackFmt::Fp6};
    }
}

// Byte sizes of (codes, scales) for one direction. The A4W4 operands carry no guard tiles:
// codes [ceil(rows, 256), ceil(k, 256) / 2], scales [ceil(rows, 256), ceil(k, 256) / 32] in
// shuffle_scale()'s layout (whose padding is exactly these).
std::pair<int64_t, int64_t> sizes_for(const MXPackFmt f, const int64_t rows, const int64_t k,
                                      const MXTilePack &fly = {}) {
    if (f == MXPackFmt::Fp6)
        return pack_sizes(rows, k);
    if (f == MXPackFmt::Fp4Blob || f == MXPackFmt::Fp4BlobSr) {
        // The A6W4 MXFP4 tile blob: 16384 code bytes + 1024 scale bytes per 256x128 tile, K padded by
        // MXFP6_GUARD_K_TILES (2) tiles like the MXFP6 blob.
        const int64_t rt = cdiv(rows, kTileRows), kt = cdiv(k, 128) + 2;
        return {rt * kt * 16384, rt * kt * 1024};
    }
    if (f == MXPackFmt::Fp6Tile) {
        // C0 [rows/16, K/128, 16, 64] then C1 [rows/32, K/128, 32, 32] (rows to 256, K to 256), and FlyDSL's
        // packed scale slab at nt 4.
        const int64_t r = cdiv(rows, kTileRows) * kTileRows, c = cdiv(k, kTileRows) * kTileRows;
        return {r * c * 3 / 4, cdiv(rows, 64 * 4) * 256 * (c / 128) * 4};
    }
    if (f == MXPackFmt::Fp4Tile || f == MXPackFmt::Fp4TileSr || f == MXPackFmt::Fp4TileK128) {
        // FlyDSL's per-tile slab (_get_mxfp4_scale_ws): ceil(rows / tile) * 256 * K/128 dwords.
        const int64_t r = cdiv(rows, kTileRows) * kTileRows, c = cdiv(k, kTileRows) * kTileRows;
        return {r * c / 2, cdiv(rows, 64 * fly.nt) * 256 * (c / 128) * 4};
    }
    PRIMUS_TURBO_CHECK(f == MXPackFmt::Fp4A || f == MXPackFmt::Fp4B || f == MXPackFmt::Fp4ASr ||
                           f == MXPackFmt::Fp4Plain || f == MXPackFmt::Fp4PlainSr,
                       "sizes_for: unsupported format ", int(f));
    const int64_t r = cdiv(rows, kTileRows) * kTileRows;
    const int64_t c = cdiv(k, kTileRows) * kTileRows;
    return {r * c / 2, r * c / kBlockSize};
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

std::vector<at::Tensor> run(const at::Tensor &input, const MXFP6Direction direction,
                            const int64_t fmt = 0) {
    const auto [row_fmt, col_fmt] = fmt_pair(fmt);
    check_input(input);
    // Allocate and launch on the operand's device rather than the ambient one, which the
    // caller is under no obligation to have set.
    const c10::DeviceGuard device_guard(input.device());
    const int64_t          M = input.size(0);
    const int64_t          N = input.size(1);

    const bool want_row = direction != MXFP6Direction::Col;
    const bool want_col = direction != MXFP6Direction::Row;

    const auto [row_p_bytes, row_s_bytes] = sizes_for(row_fmt, M, N, ts_dir(fmt, M, N, false)); // contract N
    const auto [col_p_bytes, col_s_bytes] = sizes_for(col_fmt, N, M, ts_dir(fmt, M, N, true)); // contract M
    MXTilePackScope fly_scope(ts_dir(fmt, M, N, false), ts_dir(fmt, M, N, true));

    at::Tensor row_p = empty_blob(want_row ? row_p_bytes : 0, input);
    at::Tensor row_s = empty_blob(want_row ? row_s_bytes : 0, input);
    at::Tensor col_p = empty_blob(want_col ? col_p_bytes : 0, input);
    at::Tensor col_s = empty_blob(want_col ? col_s_bytes : 0, input);

    auto stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();

    if (input.scalar_type() == at::kBFloat16) {
        quantize_mxfp6_impl<dtype::bfloat16>(
            reinterpret_cast<const dtype::bfloat16 *>(input.data_ptr()), row_p.data_ptr<uint8_t>(),
            row_s.data_ptr<uint8_t>(), col_p.data_ptr<uint8_t>(), col_s.data_ptr<uint8_t>(),
            static_cast<int>(M), static_cast<int>(N), direction, stream, row_fmt, col_fmt);
    } else {
        quantize_mxfp6_impl<dtype::float16>(
            reinterpret_cast<const dtype::float16 *>(input.data_ptr()), row_p.data_ptr<uint8_t>(),
            row_s.data_ptr<uint8_t>(), col_p.data_ptr<uint8_t>(), col_s.data_ptr<uint8_t>(),
            static_cast<int>(M), static_cast<int>(N), direction, stream, row_fmt, col_fmt);
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

MXFP6Prologue prologue_from_mode(const int64_t mode) {
    switch (mode) {
    case 0:
        return MXFP6Prologue::Identity;
    case 1:
        return MXFP6Prologue::BiasGelu;
    case 2:
        return MXFP6Prologue::BiasGeluBackward;
    default:
        // 3 (QK-norm+RoPE backward) and 4 (LnModulate) are deliberately absent: their
        // operands do not fit this entry point's (aux, bias), so each has its own op.
        PRIMUS_TURBO_CHECK(false,
                           "prologue mode must be 0 (identity), 1 (bias+gelu) or 2 "
                           "(bias+gelu backward), got ",
                           mode);
        return MXFP6Prologue::Identity;
    }
}

// The bias is a broadcast vector, so check_input's 2D and %32 rules do not apply to it.
void check_bias(const at::Tensor &bias, const at::Tensor &input, const int64_t N) {
    PRIMUS_TURBO_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
    PRIMUS_TURBO_CHECK(bias.device() == input.device(),
                       "Bias must be on the input's device: input is on ", input.device().str(),
                       " and bias on ", bias.device().str());
    PRIMUS_TURBO_CHECK(bias.dim() == 1, "Bias must be 1D, got ", bias.dim(), "D");
    PRIMUS_TURBO_CHECK(bias.is_contiguous(), "Bias must be contiguous");
    PRIMUS_TURBO_CHECK(bias.scalar_type() == input.scalar_type(),
                       "Bias dtype must match the input's");
    PRIMUS_TURBO_CHECK(bias.size(0) == N, "Bias must have one element per column: expected ", N,
                       ", got ", bias.size(0));
}

std::vector<at::Tensor> run_fused(const at::Tensor &input, const c10::optional<at::Tensor> &aux,
                                  const c10::optional<at::Tensor> &bias,
                                  const MXFP6Prologue prologue, const bool want_col_sum,
                                  const int64_t fmt = 0) {
    const auto [row_fmt, col_fmt] = fmt_pair(fmt);
    check_input(input);
    const c10::DeviceGuard device_guard(input.device());
    const int64_t          M = input.size(0);
    const int64_t          N = input.size(1);

    const bool wants_aux = prologue == MXFP6Prologue::BiasGeluBackward;
    PRIMUS_TURBO_CHECK(wants_aux == aux.has_value(),
                       "aux is required by the backward prologue and unused otherwise");
    if (aux.has_value()) {
        check_input(*aux);
        PRIMUS_TURBO_CHECK(aux->device() == input.device(),
                           "aux must be on the input's device: input is on ", input.device().str(),
                           " and aux on ", aux->device().str());
        PRIMUS_TURBO_CHECK(aux->sizes() == input.sizes(), "aux must have the input's shape");
        PRIMUS_TURBO_CHECK(aux->scalar_type() == input.scalar_type(),
                           "aux dtype must match the input's");
    }
    if (bias.has_value())
        check_bias(*bias, input, N);

    const auto [row_p_bytes, row_s_bytes] = sizes_for(row_fmt, M, N, ts_dir(fmt, M, N, false)); // contract N
    const auto [col_p_bytes, col_s_bytes] = sizes_for(col_fmt, N, M, ts_dir(fmt, M, N, true)); // contract M
    MXTilePackScope fly_scope(ts_dir(fmt, M, N, false), ts_dir(fmt, M, N, true));

    at::Tensor row_p = empty_blob(row_p_bytes, input);
    at::Tensor row_s = empty_blob(row_s_bytes, input);
    at::Tensor col_p = empty_blob(col_p_bytes, input);
    at::Tensor col_s = empty_blob(col_s_bytes, input);

    // Always returned so the op has one shape signature; degenerate when not wanted, which
    // keeps the custom op's schema and its fake free of a conditional output.
    at::Tensor col_sum = at::empty(
        {want_col_sum ? mxfp6_col_sum_rows(static_cast<int>(M)) : 0, want_col_sum ? N : 0},
        input.options().dtype(at::kFloat));

    auto stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();

    if (input.scalar_type() == at::kBFloat16) {
        using T = dtype::bfloat16;
        quantize_mxfp6_fused_impl<T>(
            reinterpret_cast<const T *>(input.data_ptr()),
            aux.has_value() ? reinterpret_cast<const T *>(aux->data_ptr()) : nullptr,
            bias.has_value() ? reinterpret_cast<const T *>(bias->data_ptr()) : nullptr,
            row_p.data_ptr<uint8_t>(), row_s.data_ptr<uint8_t>(), col_p.data_ptr<uint8_t>(),
            col_s.data_ptr<uint8_t>(), want_col_sum ? col_sum.data_ptr<float>() : nullptr,
            static_cast<int>(M), static_cast<int>(N), prologue, stream, row_fmt, col_fmt);
    } else {
        using T = dtype::float16;
        quantize_mxfp6_fused_impl<T>(
            reinterpret_cast<const T *>(input.data_ptr()),
            aux.has_value() ? reinterpret_cast<const T *>(aux->data_ptr()) : nullptr,
            bias.has_value() ? reinterpret_cast<const T *>(bias->data_ptr()) : nullptr,
            row_p.data_ptr<uint8_t>(), row_s.data_ptr<uint8_t>(), col_p.data_ptr<uint8_t>(),
            col_s.data_ptr<uint8_t>(), want_col_sum ? col_sum.data_ptr<float>() : nullptr,
            static_cast<int>(M), static_cast<int>(N), prologue, stream, row_fmt, col_fmt);
    }

    return {row_p, row_s, col_p, col_s, col_sum};
}

// Shared shape/dtype/device checks for the QK-norm+RoPE operands. Every one of them is a
// plain tensor with a shape the caller could plausibly get wrong, and a wrong shape here does
// not fault -- it reads the wrong element and produces a gradient that looks reasonable.
// PRIMUS_TURBO_CHECK's stringifier takes numbers and strings, not IntArrayRef, and a shape
// mismatch is exactly the error whose message needs to carry both shapes.
std::string shape_str(const at::IntArrayRef s) {
    std::string out = "[";
    for (size_t i = 0; i < s.size(); ++i)
        out += (i ? ", " : "") + std::to_string(s[i]);
    return out + "]";
}

void check_operand(const at::Tensor &t, const at::Tensor &input, const char *name,
                   const at::IntArrayRef want, const at::ScalarType dtype) {
    PRIMUS_TURBO_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    PRIMUS_TURBO_CHECK(t.device() == input.device(), name, " must be on the input's device: ",
                       "input is on ", input.device().str(), " and ", name, " on ",
                       t.device().str());
    PRIMUS_TURBO_CHECK(t.is_contiguous(), name, " must be contiguous");
    PRIMUS_TURBO_CHECK(t.scalar_type() == dtype, name, " has the wrong dtype");
    PRIMUS_TURBO_CHECK(t.sizes() == want, name, " has the wrong shape: expected ",
                       shape_str(want), ", got ", shape_str(t.sizes()));
}

std::vector<at::Tensor>
run_qk_norm_rope_bwd(const at::Tensor &input, const at::Tensor &dq, const at::Tensor &dk,
                     const at::Tensor &dv, const at::Tensor &cos, const at::Tensor &sin,
                     const at::Tensor &wq, const at::Tensor &wk, const at::Tensor &rstd_q,
                     const at::Tensor &rstd_k, const bool want_col_sum,
                     const int64_t fmt = 0) {
    const auto [row_fmt, col_fmt] = fmt_pair(fmt);
    check_input(input);
    const c10::DeviceGuard device_guard(input.device());
    const int64_t          M = input.size(0);
    const int64_t          N = input.size(1);

    // head_dim comes from the norm weight rather than an argument: it is the one operand whose
    // length *is* head_dim by definition, so deriving it here means a caller cannot pass a
    // head_dim that disagrees with the weight it also passed.
    const int64_t head_dim = wq.size(0);
    PRIMUS_TURBO_CHECK(head_dim > 0 && N % (3 * head_dim) == 0,
                       "input's N must be num_heads * 3 * head_dim: N is ", N,
                       " and head_dim (from wq) is ", head_dim);
    const int64_t num_heads = N / (3 * head_dim);

    const at::ScalarType dt = input.scalar_type();
    // The per-slice gradients, at [M, num_heads * head_dim]. These are exactly the grad_outputs
    // an autograd Function spanning linear_qkv -> norm -> rope receives.
    for (const auto &[t, name] : {std::pair{std::cref(dq), "dq"}, {std::cref(dk), "dk"},
                                  {std::cref(dv), "dv"}})
        check_operand(t.get(), input, name, {M, num_heads * head_dim}, dt);
    // cos/sin at [M, head_dim] is not a formality -- it is the check that the tables map 1:1
    // onto packer rows. Flux builds them per (position, batch), which is that shape; a
    // batch-shared [S, head_dim] table is also legal upstream and the Triton kernel handles it
    // by dividing the row index. This kernel does not divide, so a shared table has to be
    // rejected here rather than silently read at the wrong row.
    check_operand(cos, input, "cos", {M, head_dim}, dt);
    check_operand(sin, input, "sin", {M, head_dim}, dt);
    check_operand(wq, input, "wq", {head_dim}, dt);
    check_operand(wk, input, "wk", {head_dim}, dt);
    // fp32 and flattened [M * num_heads]: one reciprocal RMS per normed vector, in the
    // (position, batch, head) order the Triton forward writes it.
    check_operand(rstd_q, input, "rstd_q", {M * num_heads}, at::kFloat);
    check_operand(rstd_k, input, "rstd_k", {M * num_heads}, at::kFloat);

    const auto [row_p_bytes, row_s_bytes] = sizes_for(row_fmt, M, N, ts_dir(fmt, M, N, false)); // contract N
    const auto [col_p_bytes, col_s_bytes] = sizes_for(col_fmt, N, M, ts_dir(fmt, M, N, true)); // contract M
    MXTilePackScope fly_scope(ts_dir(fmt, M, N, false), ts_dir(fmt, M, N, true));

    at::Tensor row_p = empty_blob(row_p_bytes, input);
    at::Tensor row_s = empty_blob(row_s_bytes, input);
    at::Tensor col_p = empty_blob(col_p_bytes, input);
    at::Tensor col_s = empty_blob(col_s_bytes, input);

    const int  rows      = mxfp6_col_sum_rows(static_cast<int>(M));
    const auto fp32_opts = input.options().dtype(at::kFloat);
    at::Tensor col_sum =
        at::empty({want_col_sum ? rows : 0, want_col_sum ? N : 0}, fp32_opts);
    // dw partials are not optional the way col_sum is: the norm weight always has a gradient
    // if it requires one, and unlike the bias there is no path that produces it otherwise --
    // the tensor it would be reduced from never reaches HBM. Summed over both leading axes by
    // the caller, which is also where the dtype cast back to the weight's belongs.
    at::Tensor dw_q = at::empty({rows, num_heads, head_dim}, fp32_opts);
    at::Tensor dw_k = at::empty({rows, num_heads, head_dim}, fp32_opts);

    auto stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();

    // One body for both dtypes. The operand struct is templated on the element type, so the
    // alternative is either this or the whole nine-assignment block written twice.
    auto launch = [&]<typename T>() {
        MXFP6QkNormRopeArgs<T> args{};
        args.dq        = reinterpret_cast<const T *>(dq.data_ptr());
        args.dk        = reinterpret_cast<const T *>(dk.data_ptr());
        args.dv        = reinterpret_cast<const T *>(dv.data_ptr());
        args.cos       = reinterpret_cast<const T *>(cos.data_ptr());
        args.sin       = reinterpret_cast<const T *>(sin.data_ptr());
        args.wq        = reinterpret_cast<const T *>(wq.data_ptr());
        args.wk        = reinterpret_cast<const T *>(wk.data_ptr());
        args.rstd_q    = rstd_q.data_ptr<float>();
        args.rstd_k    = rstd_k.data_ptr<float>();
        args.dw_q      = dw_q.data_ptr<float>();
        args.dw_k      = dw_k.data_ptr<float>();
        args.num_heads = static_cast<int32_t>(num_heads);
        args.head_dim  = static_cast<int32_t>(head_dim);
        quantize_mxfp6_qk_norm_rope_bwd_impl<T>(
            reinterpret_cast<const T *>(input.data_ptr()), args, row_p.data_ptr<uint8_t>(),
            row_s.data_ptr<uint8_t>(), col_p.data_ptr<uint8_t>(), col_s.data_ptr<uint8_t>(),
            want_col_sum ? col_sum.data_ptr<float>() : nullptr, static_cast<int>(M),
            static_cast<int>(N), stream, row_fmt, col_fmt);
    };
    if (dt == at::kBFloat16)
        launch.template operator()<dtype::bfloat16>();
    else
        launch.template operator()<dtype::float16>();

    return {row_p, row_s, col_p, col_s, col_sum, dw_q, dw_k};
}

std::vector<at::Tensor> run_ln_modulate(const at::Tensor &input, const at::Tensor &mean,
                                        const at::Tensor &rstd, const at::Tensor &scale,
                                        const at::Tensor &shift, const bool want_col_sum,
                     const int64_t fmt = 0) {
    const auto [row_fmt, col_fmt] = fmt_pair(fmt);
    check_input(input);
    const c10::DeviceGuard device_guard(input.device());
    const int64_t          M = input.size(0);
    const int64_t          N = input.size(1);

    // The batch comes from the modulation's leading axis rather than an argument: scale is
    // [B, N] by construction, so deriving B here means a caller cannot pass a B that
    // disagrees with the tensor it also passed.
    PRIMUS_TURBO_CHECK(scale.dim() == 2, "scale must be 2D [B, N], got ", scale.dim(), "D");
    const int64_t B = scale.size(0);
    PRIMUS_TURBO_CHECK(B > 0 && (B & (B - 1)) == 0,
                       "LnModulate needs a power-of-two batch so the kernel can take the batch "
                       "index as the low bits of the row; got B = ", B);
    PRIMUS_TURBO_CHECK(M % B == 0,
                       "input's rows must be a whole number of batches: M is ", M, " and B is ",
                       B);

    const at::ScalarType dt = input.scalar_type();
    check_operand(mean, input, "mean", {M}, at::kFloat);
    check_operand(rstd, input, "rstd", {M}, at::kFloat);
    check_operand(scale, input, "scale", {B, N}, dt);
    check_operand(shift, input, "shift", {B, N}, dt);

    const auto [row_p_bytes, row_s_bytes] = sizes_for(row_fmt, M, N, ts_dir(fmt, M, N, false)); // contract N
    const auto [col_p_bytes, col_s_bytes] = sizes_for(col_fmt, N, M, ts_dir(fmt, M, N, true)); // contract M
    MXTilePackScope fly_scope(ts_dir(fmt, M, N, false), ts_dir(fmt, M, N, true));

    at::Tensor row_p = empty_blob(row_p_bytes, input);
    at::Tensor row_s = empty_blob(row_s_bytes, input);
    at::Tensor col_p = empty_blob(col_p_bytes, input);
    at::Tensor col_s = empty_blob(col_s_bytes, input);

    const int  rows      = mxfp6_col_sum_rows(static_cast<int>(M));
    const auto fp32_opts = input.options().dtype(at::kFloat);
    at::Tensor col_sum = at::empty({want_col_sum ? rows : 0, want_col_sum ? N : 0}, fp32_opts);

    auto stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();

    auto launch = [&]<typename T>() {
        MXFP6LnModulateArgs<T> args{};
        args.mean       = mean.data_ptr<float>();
        args.rstd       = rstd.data_ptr<float>();
        args.scale      = reinterpret_cast<const T *>(scale.data_ptr());
        args.shift      = reinterpret_cast<const T *>(shift.data_ptr());
        args.batch_mask = static_cast<int32_t>(B - 1);
        quantize_mxfp6_ln_modulate_impl<T>(
            reinterpret_cast<const T *>(input.data_ptr()), args, row_p.data_ptr<uint8_t>(),
            row_s.data_ptr<uint8_t>(), col_p.data_ptr<uint8_t>(), col_s.data_ptr<uint8_t>(),
            want_col_sum ? col_sum.data_ptr<float>() : nullptr, static_cast<int>(M),
            static_cast<int>(N), stream, row_fmt, col_fmt);
    };
    if (dt == at::kBFloat16)
        launch.template operator()<dtype::bfloat16>();
    else
        launch.template operator()<dtype::float16>();

    return {row_p, row_s, col_p, col_s, col_sum};
}

std::vector<at::Tensor> run_gate_mul(const at::Tensor &input, const at::Tensor &gate,
                                     const bool want_col_sum,
                     const int64_t fmt = 0) {
    const auto [row_fmt, col_fmt] = fmt_pair(fmt);
    check_input(input);
    const c10::DeviceGuard device_guard(input.device());
    const int64_t          M = input.size(0);
    const int64_t          N = input.size(1);

    // B from the gate's leading axis, as run_ln_modulate takes it from scale's.
    PRIMUS_TURBO_CHECK(gate.dim() == 2, "gate must be 2D [B, N], got ", gate.dim(), "D");
    const int64_t B = gate.size(0);
    PRIMUS_TURBO_CHECK(B > 0 && (B & (B - 1)) == 0,
                       "GateMul needs a power-of-two batch so the kernel can take the batch "
                       "index as the low bits of the row; got B = ", B);
    PRIMUS_TURBO_CHECK(M % B == 0,
                       "input's rows must be a whole number of batches: M is ", M, " and B is ",
                       B);
    check_operand(gate, input, "gate", {B, N}, input.scalar_type());

    const auto [row_p_bytes, row_s_bytes] = sizes_for(row_fmt, M, N, ts_dir(fmt, M, N, false)); // contract N
    const auto [col_p_bytes, col_s_bytes] = sizes_for(col_fmt, N, M, ts_dir(fmt, M, N, true)); // contract M
    MXTilePackScope fly_scope(ts_dir(fmt, M, N, false), ts_dir(fmt, M, N, true));

    at::Tensor row_p = empty_blob(row_p_bytes, input);
    at::Tensor row_s = empty_blob(row_s_bytes, input);
    at::Tensor col_p = empty_blob(col_p_bytes, input);
    at::Tensor col_s = empty_blob(col_s_bytes, input);

    const int  rows      = mxfp6_col_sum_rows(static_cast<int>(M));
    const auto fp32_opts = input.options().dtype(at::kFloat);
    at::Tensor col_sum = at::empty({want_col_sum ? rows : 0, want_col_sum ? N : 0}, fp32_opts);

    auto stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();

    auto launch = [&]<typename T>() {
        MXFP6GateMulArgs<T> args{};
        args.gate       = reinterpret_cast<const T *>(gate.data_ptr());
        args.batch_mask = static_cast<int32_t>(B - 1);
        quantize_mxfp6_gate_mul_impl<T>(
            reinterpret_cast<const T *>(input.data_ptr()), args, row_p.data_ptr<uint8_t>(),
            row_s.data_ptr<uint8_t>(), col_p.data_ptr<uint8_t>(), col_s.data_ptr<uint8_t>(),
            want_col_sum ? col_sum.data_ptr<float>() : nullptr, static_cast<int>(M),
            static_cast<int>(N), stream, row_fmt, col_fmt);
    };
    if (input.scalar_type() == at::kBFloat16)
        launch.template operator()<dtype::bfloat16>();
    else
        launch.template operator()<dtype::float16>();

    return {row_p, row_s, col_p, col_s, col_sum};
}

} // namespace

std::vector<at::Tensor> quantize_mxfp6(const at::Tensor input, const int64_t axis) {
    return run(input, direction_from_axis(axis));
}

// Out-variant: pack into caller-provided buffers instead of allocating.
//
// A grouped A6W6 GEMM runs G GEMMs in one launch, so its packed operands must form one
// contiguous buffer with each group written into its own slice. The allocating variant
// cannot target a slice, and both workarounds measure NEGATIVE against the grouped
// kernel's gain: aiter's caller-buffer packer is markedly slower than this kernel, and
// pack-then-copy is worse still. Either is enough to turn the grouped GEMM's gain into a
// net loss. The kernel is unchanged; only its destination moves.
void quantize_mxfp6_out(const at::Tensor input, const int64_t axis, at::Tensor packed,
                        at::Tensor scale) {
    check_input(input);
    const c10::DeviceGuard device_guard(input.device());
    const int64_t          M = input.size(0);
    const int64_t          N = input.size(1);

    const MXFP6Direction direction = direction_from_axis(axis);
    TORCH_CHECK(direction != MXFP6Direction::Dual,
                "quantize_mxfp6_out: dual packing yields four blobs; use the allocating variant");
    const bool is_row             = direction == MXFP6Direction::Row;
    const auto [p_bytes, s_bytes] = is_row ? pack_sizes(M, N) : pack_sizes(N, M);
    TORCH_CHECK(packed.numel() == p_bytes && scale.numel() == s_bytes,
                "quantize_mxfp6_out: output buffers do not match the packed layout size");
    TORCH_CHECK(packed.is_contiguous() && scale.is_contiguous(),
                "quantize_mxfp6_out: output buffers must be contiguous");
    TORCH_CHECK(packed.scalar_type() == at::kByte && scale.scalar_type() == at::kByte,
                "quantize_mxfp6_out: output buffers must be uint8");

    at::Tensor unused = empty_blob(0, input);
    uint8_t   *rp     = is_row ? packed.data_ptr<uint8_t>() : unused.data_ptr<uint8_t>();
    uint8_t   *rs     = is_row ? scale.data_ptr<uint8_t>() : unused.data_ptr<uint8_t>();
    uint8_t   *cp     = is_row ? unused.data_ptr<uint8_t>() : packed.data_ptr<uint8_t>();
    uint8_t   *cs     = is_row ? unused.data_ptr<uint8_t>() : scale.data_ptr<uint8_t>();

    auto stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
    if (input.scalar_type() == at::kBFloat16)
        quantize_mxfp6_impl<dtype::bfloat16>(
            reinterpret_cast<const dtype::bfloat16 *>(input.data_ptr()), rp, rs, cp, cs,
            static_cast<int>(M), static_cast<int>(N), direction, stream);
    else
        quantize_mxfp6_impl<dtype::float16>(
            reinterpret_cast<const dtype::float16 *>(input.data_ptr()), rp, rs, cp, cs,
            static_cast<int>(M), static_cast<int>(N), direction, stream);
}

void quantize_mxfp6_out_meta(const at::Tensor, const int64_t, at::Tensor, at::Tensor) {}

std::vector<at::Tensor> quantize_mxfp6_dual(const at::Tensor input) {
    return run(input, MXFP6Direction::Dual);
}

// Dual out-variant: row AND column blobs into caller-provided buffers, one kernel.
//
// A grouped GEMM needs the ROW blob of both groups in one contiguous buffer, while the
// COLUMN blob (which contracts along M) must stay per-stream for wgrad. Splitting the
// fused dual pack into a row call plus a column call to achieve that costs a large fraction
// of what the grouping wins. This keeps it
// a single kernel and hands each direction its own destination, so the split costs
// nothing.
static void dual_out_fmt(const at::Tensor input, at::Tensor row_packed, at::Tensor row_scale,
                         at::Tensor col_packed, at::Tensor col_scale, const int64_t fmt,
                         const c10::optional<at::Tensor> row_c1 = c10::nullopt, const int64_t draws = 1,
                         const int64_t draw_codes = 0, const int64_t draw_scales = 0) {
    check_input(input);
    const c10::DeviceGuard device_guard(input.device());
    const int64_t          M = input.size(0);
    const int64_t          N = input.size(1);

    const auto [row_fmt, col_fmt]   = fmt_pair(fmt);
    const auto [rp_bytes, rs_bytes] = sizes_for(row_fmt, M, N, ts_dir(fmt, M, N, false));
    const auto [cp_bytes, cs_bytes] = sizes_for(col_fmt, N, M, ts_dir(fmt, M, N, true));
    MXTilePack row_ts = ts_dir(fmt, M, N, false);
    // row_c1: MXFP6 K128-blocked rows with the C1 plane in its own buffer (row_packed then holds C0: 2/3 of it)
    const int64_t row_bytes = row_packed.numel() + (row_c1.has_value() ? row_c1->numel() : 0);
    if (row_c1.has_value()) {
        TORCH_CHECK(row_fmt == MXPackFmt::Fp6Tile && row_c1->numel() * 2 == row_packed.numel() &&
                        row_c1->is_contiguous() && row_c1->scalar_type() == at::kByte,
                    "quantize_mx_dual_out: row_c1 is the C1 plane of MXFP6 K128-blocked rows (half C0's bytes)");
        row_ts.c1_split = 1;
        row_ts.c1_delta = reinterpret_cast<intptr_t>(row_c1->data_ptr()) -
                          reinterpret_cast<intptr_t>(row_packed.data_ptr());
    }
    MXTilePack col_ts = ts_dir(fmt, M, N, true);
    if (draws != 1) {
        // Draw d of the column pack lands at d * draw_codes / d * draw_scales bytes past the given buffers, inside
        // the same allocations.
        TORCH_CHECK(draws > 1 && (col_fmt == MXPackFmt::Fp4TileSr || col_fmt == MXPackFmt::Fp4PlainSr ||
                                  col_fmt == MXPackFmt::Fp4ASr || col_fmt == MXPackFmt::Fp4BlobSr),
                    "quantize_mx_dual_out: draws > 1 is for a stochastically rounded FP4 column direction");
        const auto room = [](const at::Tensor &t, int64_t span) {
            return (t.storage_offset() + span) * t.element_size() <= int64_t(t.storage().nbytes());
        };
        TORCH_CHECK(room(col_packed, (draws - 1) * draw_codes + col_packed.numel()) &&
                        room(col_scale, (draws - 1) * draw_scales + col_scale.numel()),
                    "quantize_mx_dual_out: the column draws run past their allocations");
        col_ts.draws       = static_cast<int32_t>(draws);
        col_ts.draw_codes  = draw_codes;
        col_ts.draw_scales = draw_scales;
    }
    MXTilePackScope fly_scope(row_ts, col_ts);
    // A direction whose two buffers are both empty is not emitted (a row-only or column-only pack of the fmt).
    const bool do_row = row_bytes || row_scale.numel(), do_col = col_packed.numel() || col_scale.numel();
    TORCH_CHECK(do_row || do_col, "quantize_mx_dual_out: no direction to emit");
    TORCH_CHECK((!do_row || (row_bytes == rp_bytes && row_scale.numel() == rs_bytes)) &&
                    (!do_col || (col_packed.numel() == cp_bytes && col_scale.numel() == cs_bytes)),
                "quantize_mxfp6_dual_out: output buffers do not match the packed layout size");
    const MXFP6Direction dir = do_row && do_col ? MXFP6Direction::Dual
                               : do_row         ? MXFP6Direction::Row
                                                : MXFP6Direction::Col;
    TORCH_CHECK(row_packed.is_contiguous() && row_scale.is_contiguous() &&
                    col_packed.is_contiguous() && col_scale.is_contiguous(),
                "quantize_mxfp6_dual_out: output buffers must be contiguous");
    TORCH_CHECK(row_packed.scalar_type() == at::kByte && row_scale.scalar_type() == at::kByte &&
                    col_packed.scalar_type() == at::kByte && col_scale.scalar_type() == at::kByte,
                "quantize_mxfp6_dual_out: output buffers must be uint8");

    auto stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
    if (input.scalar_type() == at::kBFloat16)
        quantize_mxfp6_impl<dtype::bfloat16>(
            reinterpret_cast<const dtype::bfloat16 *>(input.data_ptr()),
            row_packed.data_ptr<uint8_t>(), row_scale.data_ptr<uint8_t>(),
            col_packed.data_ptr<uint8_t>(), col_scale.data_ptr<uint8_t>(), static_cast<int>(M),
            static_cast<int>(N), dir, stream, row_fmt, col_fmt);
    else
        quantize_mxfp6_impl<dtype::float16>(
            reinterpret_cast<const dtype::float16 *>(input.data_ptr()),
            row_packed.data_ptr<uint8_t>(), row_scale.data_ptr<uint8_t>(),
            col_packed.data_ptr<uint8_t>(), col_scale.data_ptr<uint8_t>(), static_cast<int>(M),
            static_cast<int>(N), dir, stream, row_fmt, col_fmt);
}

void quantize_mxfp6_dual_out(const at::Tensor input, at::Tensor row_packed, at::Tensor row_scale,
                             at::Tensor col_packed, at::Tensor col_scale) {
    dual_out_fmt(input, row_packed, row_scale, col_packed, col_scale, 0);
}

// The same with an output format (see fmt_pair): e.g. a grouped MLP's packs under A4W4.
void quantize_mx_dual_out(const at::Tensor input, at::Tensor row_packed, at::Tensor row_scale,
                          at::Tensor col_packed, at::Tensor col_scale, const int64_t fmt,
                          const c10::optional<at::Tensor> row_c1, const int64_t draws,
                          const int64_t draw_codes, const int64_t draw_scales) {
    dual_out_fmt(input, row_packed, row_scale, col_packed, col_scale, fmt, row_c1, draws, draw_codes,
                 draw_scales);
}

void quantize_mx_dual_out_meta(const at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                               const int64_t, const c10::optional<at::Tensor>, const int64_t,
                               const int64_t, const int64_t) {}

void quantize_mxfp6_dual_out_meta(const at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                                  at::Tensor) {}

std::vector<at::Tensor> quantize_mxfp6_fused_dual(const at::Tensor                input,
                                                  const c10::optional<at::Tensor> aux,
                                                  const c10::optional<at::Tensor> bias,
                                                  const int64_t mode, const bool want_col_sum) {
    return run_fused(input, aux, bias, prologue_from_mode(mode), want_col_sum);
}

// Out-variant of the fused prologue+pack, for grouped GEMMs.
//
// The MLP's fc2 wants the ROW blob of both streams contiguous (one grouped GEMM), while the
// COLUMN blob contracts along M and must stay per-stream -- the two streams have different
// w2, so a stacked column pack would sum contributions that belong to different weights.
// One kernel, two destinations.
//
// col_sum is an optional third destination. The backward prologue
// (MXFP6_PROLOGUE_BIAS_GELU_BACKWARD) reduces the tensor it is packing, because the bias
// gradient is a sum over a tensor the fusion deliberately never materialises. Without it
// here, fc1's dgrad could not use this out-variant at all and so could not be grouped --
// and copying the row blob out of the allocating form costs about as much as grouping
// saves. Pass None when the prologue has no reduction, which is the forward case.
static void fused_dual_out_fmt(const at::Tensor input, const c10::optional<at::Tensor> aux,
                               const c10::optional<at::Tensor> bias, const int64_t prologue_mode,
                               at::Tensor row_packed, at::Tensor row_scale, at::Tensor col_packed,
                               at::Tensor col_scale, c10::optional<at::Tensor> col_sum,
                               const int64_t fmt) {
    const c10::DeviceGuard device_guard(input.device());
    const int64_t          M = input.size(0);
    const int64_t          N = input.size(1);

    const auto [row_fmt, col_fmt]   = fmt_pair(fmt);
    const auto [rp_bytes, rs_bytes] = sizes_for(row_fmt, M, N, ts_dir(fmt, M, N, false));
    const auto [cp_bytes, cs_bytes] = sizes_for(col_fmt, N, M, ts_dir(fmt, M, N, true));
    MXTilePackScope fly_scope(ts_dir(fmt, M, N, false), ts_dir(fmt, M, N, true));
    TORCH_CHECK(row_packed.numel() == rp_bytes && row_scale.numel() == rs_bytes &&
                    col_packed.numel() == cp_bytes && col_scale.numel() == cs_bytes,
                "quantize_mxfp6_fused_dual_out: buffers do not match the packed layout size");
    TORCH_CHECK(row_packed.is_contiguous() && row_scale.is_contiguous() &&
                    col_packed.is_contiguous() && col_scale.is_contiguous(),
                "quantize_mxfp6_fused_dual_out: buffers must be contiguous");
    if (col_sum.has_value()) {
        TORCH_CHECK(col_sum->is_contiguous() && col_sum->scalar_type() == at::kFloat,
                    "quantize_mxfp6_fused_dual_out: col_sum must be contiguous float32");
        TORCH_CHECK(col_sum->numel() == mxfp6_col_sum_rows(static_cast<int>(M)) * N,
                    "quantize_mxfp6_fused_dual_out: col_sum does not match the reduction shape");
    }
    float *col_sum_ptr = col_sum.has_value() ? col_sum->data_ptr<float>() : nullptr;

    const MXFP6Prologue prologue = prologue_from_mode(prologue_mode);
    auto                stream   = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();

    if (input.scalar_type() == at::kBFloat16) {
        using T = dtype::bfloat16;
        quantize_mxfp6_fused_impl<T>(
            reinterpret_cast<const T *>(input.data_ptr()),
            aux.has_value() ? reinterpret_cast<const T *>(aux->data_ptr()) : nullptr,
            bias.has_value() ? reinterpret_cast<const T *>(bias->data_ptr()) : nullptr,
            row_packed.data_ptr<uint8_t>(), row_scale.data_ptr<uint8_t>(),
            col_packed.data_ptr<uint8_t>(), col_scale.data_ptr<uint8_t>(), col_sum_ptr,
            static_cast<int>(M), static_cast<int>(N), prologue, stream, row_fmt, col_fmt);
    } else {
        using T = dtype::float16;
        quantize_mxfp6_fused_impl<T>(
            reinterpret_cast<const T *>(input.data_ptr()),
            aux.has_value() ? reinterpret_cast<const T *>(aux->data_ptr()) : nullptr,
            bias.has_value() ? reinterpret_cast<const T *>(bias->data_ptr()) : nullptr,
            row_packed.data_ptr<uint8_t>(), row_scale.data_ptr<uint8_t>(),
            col_packed.data_ptr<uint8_t>(), col_scale.data_ptr<uint8_t>(), col_sum_ptr,
            static_cast<int>(M), static_cast<int>(N), prologue, stream, row_fmt, col_fmt);
    }
}

void quantize_mxfp6_fused_dual_out(const at::Tensor input, const c10::optional<at::Tensor> aux,
                                   const c10::optional<at::Tensor> bias,
                                   const int64_t prologue_mode, at::Tensor row_packed,
                                   at::Tensor row_scale, at::Tensor col_packed,
                                   at::Tensor col_scale, c10::optional<at::Tensor> col_sum) {
    fused_dual_out_fmt(input, aux, bias, prologue_mode, row_packed, row_scale, col_packed,
                       col_scale, col_sum, 0);
}

void quantize_mx_fused_dual_out(const at::Tensor input, const c10::optional<at::Tensor> aux,
                                const c10::optional<at::Tensor> bias, const int64_t prologue_mode,
                                at::Tensor row_packed, at::Tensor row_scale, at::Tensor col_packed,
                                at::Tensor col_scale, c10::optional<at::Tensor> col_sum,
                                const int64_t fmt) {
    fused_dual_out_fmt(input, aux, bias, prologue_mode, row_packed, row_scale, col_packed,
                       col_scale, col_sum, fmt);
}

void quantize_mx_fused_dual_out_meta(const at::Tensor, const c10::optional<at::Tensor>,
                                     const c10::optional<at::Tensor>, const int64_t, at::Tensor,
                                     at::Tensor, at::Tensor, at::Tensor, c10::optional<at::Tensor>,
                                     const int64_t) {}

void quantize_mxfp6_fused_dual_out_meta(const at::Tensor, const c10::optional<at::Tensor>,
                                        const c10::optional<at::Tensor>, const int64_t, at::Tensor,
                                        at::Tensor, at::Tensor, at::Tensor,
                                        c10::optional<at::Tensor>) {}

// Kept off quantize_mxfp6_fused_dual's `mode` argument on purpose. This prologue's operands do
// not fit the (aux, bias) shape, and it runs at a different tile width, so routing it through
// the same entry point would mean nine mostly-null optional tensors on every existing call and
// a second tile instantiation behind a runtime branch. `prologue_from_mode` rejects mode 3 for
// the same reason.
std::vector<at::Tensor>
quantize_mxfp6_qk_norm_rope_bwd(const at::Tensor input, const at::Tensor dq, const at::Tensor dk,
                                const at::Tensor dv, const at::Tensor cos, const at::Tensor sin,
                                const at::Tensor wq, const at::Tensor wk, const at::Tensor rstd_q,
                                const at::Tensor rstd_k, const bool want_col_sum) {
    return run_qk_norm_rope_bwd(input, dq, dk, dv, cos, sin, wq, wk, rstd_q, rstd_k,
                                want_col_sum);
}

// Off the `mode` argument for the same reason: four operands that are not (aux, bias). The
// input here is the norm's *input*, not its output -- the point of the fusion is that the
// output never exists.
std::vector<at::Tensor> quantize_mxfp6_ln_modulate(const at::Tensor input, const at::Tensor mean,
                                                   const at::Tensor rstd, const at::Tensor scale,
                                                   const at::Tensor shift,
                                                   const bool       want_col_sum) {
    return run_ln_modulate(input, mean, rstd, scale, shift, want_col_sum);
}

// Dual pack of input * gate[m % B], for a gated residual's incoming gradient. Off `mode`
// because its operand is not (aux, bias).
std::vector<at::Tensor> quantize_mxfp6_gate_mul(const at::Tensor input, const at::Tensor gate,
                                                const bool want_col_sum) {
    return run_gate_mul(input, gate, want_col_sum);
}

// Meta implementations. Shapes are pure arithmetic on M and N, so torch.compile can trace
// through the packer without a graph break.
std::vector<at::Tensor> quantize_mxfp6_meta(const at::Tensor input, const int64_t axis) {
    const int64_t M            = input.size(0);
    const int64_t N            = input.size(1);
    const bool    row          = direction_from_axis(axis) == MXFP6Direction::Row;
    const auto [packed, scale] = row ? pack_sizes(M, N) : pack_sizes(N, M);
    auto opts                  = input.options().dtype(at::kByte);
    return {at::empty({packed}, opts), at::empty({scale}, opts)};
}

std::vector<at::Tensor> quantize_mxfp6_dual_meta(const at::Tensor input) {
    const int64_t M                       = input.size(0);
    const int64_t N                       = input.size(1);
    const auto [row_p_bytes, row_s_bytes] = pack_sizes(M, N);
    const auto [col_p_bytes, col_s_bytes] = pack_sizes(N, M);
    auto opts                             = input.options().dtype(at::kByte);
    return {at::empty({row_p_bytes}, opts), at::empty({row_s_bytes}, opts),
            at::empty({col_p_bytes}, opts), at::empty({col_s_bytes}, opts)};
}

// The prologue does not change the blob geometry, so the four blobs are the dual fake's;
// only the bias-gradient partial is new.
std::vector<at::Tensor> quantize_mxfp6_fused_dual_meta(const at::Tensor                input,
                                                       const c10::optional<at::Tensor> aux,
                                                       const c10::optional<at::Tensor> bias,
                                                       const int64_t                   mode,
                                                       const bool want_col_sum) {
    const int64_t M    = input.size(0);
    const int64_t N    = input.size(1);
    auto          out  = quantize_mxfp6_dual_meta(input);
    const int64_t rows = want_col_sum ? mxfp6_col_sum_rows(static_cast<int>(M)) : 0;
    out.push_back(at::empty({rows, want_col_sum ? N : 0}, input.options().dtype(at::kFloat)));
    return out;
}

std::vector<at::Tensor>
quantize_mxfp6_qk_norm_rope_bwd_meta(const at::Tensor input, const at::Tensor dq,
                                     const at::Tensor dk, const at::Tensor dv,
                                     const at::Tensor cos, const at::Tensor sin,
                                     const at::Tensor wq, const at::Tensor wk,
                                     const at::Tensor rstd_q, const at::Tensor rstd_k,
                                     const bool want_col_sum) {
    const int64_t M         = input.size(0);
    const int64_t N         = input.size(1);
    const int64_t head_dim  = wq.size(0);
    const int64_t num_heads = head_dim > 0 ? N / (3 * head_dim) : 0;
    const int64_t rows      = mxfp6_col_sum_rows(static_cast<int>(M));
    const auto    fp32_opts = input.options().dtype(at::kFloat);

    auto out = quantize_mxfp6_dual_meta(input);
    out.push_back(at::empty({want_col_sum ? rows : 0, want_col_sum ? N : 0}, fp32_opts));
    out.push_back(at::empty({rows, num_heads, head_dim}, fp32_opts));
    out.push_back(at::empty({rows, num_heads, head_dim}, fp32_opts));
    return out;
}

// Same five outputs as the fused packer: the prologue does not touch the blob geometry and
// the modulation's operands are not part of it.
std::vector<at::Tensor>
quantize_mxfp6_ln_modulate_meta(const at::Tensor input, const at::Tensor mean,
                                const at::Tensor rstd, const at::Tensor scale,
                                const at::Tensor shift, const bool want_col_sum) {
    const int64_t M    = input.size(0);
    const int64_t N    = input.size(1);
    auto          out  = quantize_mxfp6_dual_meta(input);
    const int64_t rows = want_col_sum ? mxfp6_col_sum_rows(static_cast<int>(M)) : 0;
    out.push_back(at::empty({rows, want_col_sum ? N : 0}, input.options().dtype(at::kFloat)));
    return out;
}

std::vector<at::Tensor> quantize_mxfp6_gate_mul_meta(const at::Tensor input, const at::Tensor gate,
                                                     const bool want_col_sum) {
    const int64_t M    = input.size(0);
    const int64_t N    = input.size(1);
    auto          out  = quantize_mxfp6_dual_meta(input);
    const int64_t rows = want_col_sum ? mxfp6_col_sum_rows(static_cast<int>(M)) : 0;
    out.push_back(at::empty({rows, want_col_sum ? N : 0}, input.options().dtype(at::kFloat)));
    return out;
}

// ---------------------------------------------------------------------------------------
// quantize_mx_*: the same packers with a trailing `fmt` (see fmt_pair). fmt = 0 is exactly
// the quantize_mxfp6_* op; 1 and 2 emit AITER's A4W4 (f4gemm) operand layouts, for MXFP4 in
// the backward GEMMs. Outputs are the same tensors in the same order.
// ---------------------------------------------------------------------------------------
std::vector<at::Tensor> quantize_mx(const at::Tensor input, const int64_t axis, const int64_t fmt) {
    return run(input, direction_from_axis(axis), fmt);
}

std::vector<at::Tensor> quantize_mx_dual(const at::Tensor input, const int64_t fmt) {
    return run(input, MXFP6Direction::Dual, fmt);
}

std::vector<at::Tensor> quantize_mx_fused_dual(const at::Tensor input,
                                               const c10::optional<at::Tensor> aux,
                                               const c10::optional<at::Tensor> bias,
                                               const int64_t mode, const bool want_col_sum,
                                               const int64_t fmt) {
    return run_fused(input, aux, bias, prologue_from_mode(mode), want_col_sum, fmt);
}

std::vector<at::Tensor>
quantize_mx_qk_norm_rope_bwd(const at::Tensor input, const at::Tensor dq, const at::Tensor dk,
                             const at::Tensor dv, const at::Tensor cos, const at::Tensor sin,
                             const at::Tensor wq, const at::Tensor wk, const at::Tensor rstd_q,
                             const at::Tensor rstd_k, const bool want_col_sum, const int64_t fmt) {
    return run_qk_norm_rope_bwd(input, dq, dk, dv, cos, sin, wq, wk, rstd_q, rstd_k, want_col_sum,
                                fmt);
}

std::vector<at::Tensor> quantize_mx_ln_modulate(const at::Tensor input, const at::Tensor mean,
                                                const at::Tensor rstd, const at::Tensor scale,
                                                const at::Tensor shift, const bool want_col_sum,
                                                const int64_t fmt) {
    return run_ln_modulate(input, mean, rstd, scale, shift, want_col_sum, fmt);
}

std::vector<at::Tensor> quantize_mx_gate_mul(const at::Tensor input, const at::Tensor gate,
                                             const bool want_col_sum, const int64_t fmt) {
    return run_gate_mul(input, gate, want_col_sum, fmt);
}

namespace {
// Replace the four blobs of an FP6 meta result with the sizes `fmt` emits.
std::vector<at::Tensor> with_fmt_blobs(std::vector<at::Tensor> out, const at::Tensor &input,
                                       const int64_t fmt) {
    const int64_t M                = input.size(0);
    const int64_t N                = input.size(1);
    const auto [row_fmt, col_fmt]  = fmt_pair(fmt);
    const auto [rp, rs]            = sizes_for(row_fmt, M, N, ts_dir(fmt, M, N, false));
    const auto [cp, cs]            = sizes_for(col_fmt, N, M, ts_dir(fmt, M, N, true));
    auto opts                      = input.options().dtype(at::kByte);
    out[0] = at::empty({rp}, opts);
    out[1] = at::empty({rs}, opts);
    out[2] = at::empty({cp}, opts);
    out[3] = at::empty({cs}, opts);
    return out;
}
} // namespace

std::vector<at::Tensor> quantize_mx_meta(const at::Tensor input, const int64_t axis,
                                         const int64_t fmt) {
    const int64_t M               = input.size(0);
    const int64_t N               = input.size(1);
    const bool    row             = direction_from_axis(axis) == MXFP6Direction::Row;
    const auto [row_fmt, col_fmt] = fmt_pair(fmt);
    const auto [packed, scale]    = row ? sizes_for(row_fmt, M, N, ts_dir(fmt, M, N, false))
                                        : sizes_for(col_fmt, N, M, ts_dir(fmt, M, N, true));
    auto opts                     = input.options().dtype(at::kByte);
    return {at::empty({packed}, opts), at::empty({scale}, opts)};
}

std::vector<at::Tensor> quantize_mx_dual_meta(const at::Tensor input, const int64_t fmt) {
    return with_fmt_blobs(quantize_mxfp6_dual_meta(input), input, fmt);
}

std::vector<at::Tensor> quantize_mx_fused_dual_meta(const at::Tensor input,
                                                    const c10::optional<at::Tensor> aux,
                                                    const c10::optional<at::Tensor> bias,
                                                    const int64_t mode, const bool want_col_sum,
                                                    const int64_t fmt) {
    return with_fmt_blobs(quantize_mxfp6_fused_dual_meta(input, aux, bias, mode, want_col_sum),
                          input, fmt);
}

std::vector<at::Tensor>
quantize_mx_qk_norm_rope_bwd_meta(const at::Tensor input, const at::Tensor dq, const at::Tensor dk,
                                  const at::Tensor dv, const at::Tensor cos, const at::Tensor sin,
                                  const at::Tensor wq, const at::Tensor wk,
                                  const at::Tensor rstd_q, const at::Tensor rstd_k,
                                  const bool want_col_sum, const int64_t fmt) {
    return with_fmt_blobs(quantize_mxfp6_qk_norm_rope_bwd_meta(input, dq, dk, dv, cos, sin, wq, wk,
                                                               rstd_q, rstd_k, want_col_sum),
                          input, fmt);
}

std::vector<at::Tensor> quantize_mx_ln_modulate_meta(const at::Tensor input, const at::Tensor mean,
                                                     const at::Tensor rstd, const at::Tensor scale,
                                                     const at::Tensor shift,
                                                     const bool want_col_sum, const int64_t fmt) {
    return with_fmt_blobs(
        quantize_mxfp6_ln_modulate_meta(input, mean, rstd, scale, shift, want_col_sum), input,
        fmt);
}

std::vector<at::Tensor> quantize_mx_gate_mul_meta(const at::Tensor input, const at::Tensor gate,
                                                  const bool want_col_sum, const int64_t fmt) {
    return with_fmt_blobs(quantize_mxfp6_gate_mul_meta(input, gate, want_col_sum), input, fmt);
}

} // namespace primus_turbo::pytorch

#endif // BUILD_MXFP6_BACKEND
