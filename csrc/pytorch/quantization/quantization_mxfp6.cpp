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
    // Allocate and launch on the operand's device rather than the ambient one, which the
    // caller is under no obligation to have set.
    const c10::DeviceGuard device_guard(input.device());
    const int64_t          M = input.size(0);
    const int64_t          N = input.size(1);

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
            reinterpret_cast<const dtype::bfloat16 *>(input.data_ptr()), row_p.data_ptr<uint8_t>(),
            row_s.data_ptr<uint8_t>(), col_p.data_ptr<uint8_t>(), col_s.data_ptr<uint8_t>(),
            static_cast<int>(M), static_cast<int>(N), direction, stream);
    } else {
        quantize_mxfp6_impl<dtype::float16>(
            reinterpret_cast<const dtype::float16 *>(input.data_ptr()), row_p.data_ptr<uint8_t>(),
            row_s.data_ptr<uint8_t>(), col_p.data_ptr<uint8_t>(), col_s.data_ptr<uint8_t>(),
            static_cast<int>(M), static_cast<int>(N), direction, stream);
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
                                  const MXFP6Prologue prologue, const bool want_col_sum) {
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

    const auto [row_p_bytes, row_s_bytes] = pack_sizes(M, N); // contract N
    const auto [col_p_bytes, col_s_bytes] = pack_sizes(N, M); // contract M

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
            static_cast<int>(M), static_cast<int>(N), prologue, stream);
    } else {
        using T = dtype::float16;
        quantize_mxfp6_fused_impl<T>(
            reinterpret_cast<const T *>(input.data_ptr()),
            aux.has_value() ? reinterpret_cast<const T *>(aux->data_ptr()) : nullptr,
            bias.has_value() ? reinterpret_cast<const T *>(bias->data_ptr()) : nullptr,
            row_p.data_ptr<uint8_t>(), row_s.data_ptr<uint8_t>(), col_p.data_ptr<uint8_t>(),
            col_s.data_ptr<uint8_t>(), want_col_sum ? col_sum.data_ptr<float>() : nullptr,
            static_cast<int>(M), static_cast<int>(N), prologue, stream);
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
                     const at::Tensor &rstd_k, const bool want_col_sum) {
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

    const auto [row_p_bytes, row_s_bytes] = pack_sizes(M, N); // contract N
    const auto [col_p_bytes, col_s_bytes] = pack_sizes(N, M); // contract M

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
            static_cast<int>(N), stream);
    };
    if (dt == at::kBFloat16)
        launch.template operator()<dtype::bfloat16>();
    else
        launch.template operator()<dtype::float16>();

    return {row_p, row_s, col_p, col_s, col_sum, dw_q, dw_k};
}

} // namespace

std::vector<at::Tensor> quantize_mxfp6(const at::Tensor input, const int64_t axis) {
    return run(input, direction_from_axis(axis));
}

std::vector<at::Tensor> quantize_mxfp6_dual(const at::Tensor input) {
    return run(input, MXFP6Direction::Dual);
}

std::vector<at::Tensor> quantize_mxfp6_fused_dual(const at::Tensor                input,
                                                  const c10::optional<at::Tensor> aux,
                                                  const c10::optional<at::Tensor> bias,
                                                  const int64_t mode, const bool want_col_sum) {
    return run_fused(input, aux, bias, prologue_from_mode(mode), want_col_sum);
}

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

} // namespace primus_turbo::pytorch

#endif // BUILD_MXFP6_BACKEND
