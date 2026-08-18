/***************************************************************************************************
 * Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// Torch entry points for the HipKittens gfx950 attention kernels.
//
// Only the argument marshalling lives here; the schemas and the dispatcher registrations are in
// bindings_pytorch.cpp with every other op, and the declarations in extensions.h.
//
// The kernels behind these are compiled in every arch configuration but do nothing off gfx950 --
// each body sits behind `#if !defined(__gfx950__)` and asserts. Reaching them on another card is
// a caller bug: primus_turbo/hipkittens/attention/gfx950/ refuses a non-gfx950 device, along with
// the rest of the envelope these kernels admit.

#include "../extensions.h"
#include "primus_turbo/hipkittens/attention.h"

namespace primus_turbo::pytorch {

namespace hk = ::primus_turbo::hipkittens;

namespace {

// The kernel layer takes plain buffers, so the tensor checks live here -- this is the only
// place that sees an at::Tensor at all.
hk::HkTensorDesc desc(const at::Tensor &t, const char *name) {
    TORCH_CHECK(t.is_cuda(), "hipkittens attention: ", name, " must be on an AMD GPU");
    TORCH_CHECK(t.is_contiguous(), "hipkittens attention: ", name,
                " must be contiguous; the kernel derives its strides from the shape");
    TORCH_CHECK(t.dim() <= 4, "hipkittens attention: ", name, " must have at most 4 dims, got ",
                t.dim());
    // Right-align the shape into four axes, which is what the kernels index against.
    int d[4] = {1, 1, 1, 1};
    const int64_t nd = t.dim();
    for (int64_t i = 0; i < nd; ++i) {
        d[4 - nd + i] = static_cast<int>(t.size(i));
    }
    return hk::HkTensorDesc{t.data_ptr(), d[0], d[1], d[2], d[3]};
}

}  // namespace

// int64_t across the boundary because that is the only integer type the torch schema carries;
// the kernels take int, and every one of these is a shape or a tile index well inside its range.
void hk_attn_fwd_d64(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v, at::Tensor &o,
                     at::Tensor &lse, int64_t Sq, int64_t Skv, int64_t B, int64_t Hq, int64_t Hkv,
                     int64_t window_left, double softmax_scale) {
    hk::hk_attn_fwd_d64(desc(q, "q"), desc(k, "k"), desc(v, "v"), desc(o, "o"), desc(lse, "lse"), static_cast<int>(Sq), static_cast<int>(Skv),
                        static_cast<int>(B), static_cast<int>(Hq), static_cast<int>(Hkv),
                        static_cast<int>(window_left), static_cast<float>(softmax_scale));
}

void hk_attn_fwd_d128(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v, at::Tensor &o,
                      at::Tensor &lse, int64_t Sq, int64_t Skv, int64_t B, int64_t Hq, int64_t Hkv,
                      int64_t window_left, double softmax_scale) {
    hk::hk_attn_fwd_d128(desc(q, "q"), desc(k, "k"), desc(v, "v"), desc(o, "o"), desc(lse, "lse"), static_cast<int>(Sq), static_cast<int>(Skv),
                         static_cast<int>(B), static_cast<int>(Hq), static_cast<int>(Hkv),
                         static_cast<int>(window_left), static_cast<float>(softmax_scale));
}

void hk_attn_bwd_d64(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
                     const at::Tensor &o, const at::Tensor &dO, at::Tensor &dq, at::Tensor &dk,
                     at::Tensor &dv, const at::Tensor &lse, at::Tensor &delta, at::Tensor &lneg,
                     at::Tensor &wsk, at::Tensor &wsv, int64_t Sq, int64_t Skv, int64_t B,
                     int64_t Hq, int64_t Hkv, int64_t window_left, double softmax_scale,
                     int64_t n_split_req) {
    hk::hk_attn_bwd_d64(desc(q, "q"), desc(k, "k"), desc(v, "v"), desc(o, "o"), desc(dO, "dO"), desc(dq, "dq"), desc(dk, "dk"), desc(dv, "dv"), desc(lse, "lse"), desc(delta, "delta"), desc(lneg, "lneg"), desc(wsk, "wsk"), desc(wsv, "wsv"),
                        static_cast<int>(Sq), static_cast<int>(Skv), static_cast<int>(B),
                        static_cast<int>(Hq), static_cast<int>(Hkv),
                        static_cast<int>(window_left), static_cast<float>(softmax_scale),
                        static_cast<int>(n_split_req));
}

void hk_attn_bwd_d128(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
                      const at::Tensor &o, const at::Tensor &dO, at::Tensor &dq, at::Tensor &dk,
                      at::Tensor &dv, const at::Tensor &lse, at::Tensor &delta, at::Tensor &lneg,
                      at::Tensor &wsk, at::Tensor &wsv, int64_t Sq, int64_t Skv, int64_t B,
                      int64_t Hq, int64_t Hkv, int64_t window_left, double softmax_scale,
                      int64_t n_split_req) {
    hk::hk_attn_bwd_d128(desc(q, "q"), desc(k, "k"), desc(v, "v"), desc(o, "o"), desc(dO, "dO"), desc(dq, "dq"), desc(dk, "dk"), desc(dv, "dv"), desc(lse, "lse"), desc(delta, "delta"), desc(lneg, "lneg"), desc(wsk, "wsk"), desc(wsv, "wsv"),
                         static_cast<int>(Sq), static_cast<int>(Skv), static_cast<int>(B),
                         static_cast<int>(Hq), static_cast<int>(Hkv),
                         static_cast<int>(window_left), static_cast<float>(softmax_scale),
                         static_cast<int>(n_split_req));
}

void hk_attn_bwd_fused_d64(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
                           const at::Tensor &o, const at::Tensor &dO, at::Tensor &dq,
                           at::Tensor &dk, at::Tensor &dv, at::Tensor &ws, const at::Tensor &lse,
                           at::Tensor &delta, int64_t Sq, int64_t Skv, int64_t B, int64_t Hq,
                           int64_t Hkv, int64_t window_left, double softmax_scale) {
    hk::hk_attn_bwd_fused_d64(desc(q, "q"), desc(k, "k"), desc(v, "v"), desc(o, "o"), desc(dO, "dO"), desc(dq, "dq"), desc(dk, "dk"), desc(dv, "dv"), desc(ws, "ws"), desc(lse, "lse"), desc(delta, "delta"), static_cast<int>(Sq),
                              static_cast<int>(Skv), static_cast<int>(B), static_cast<int>(Hq),
                              static_cast<int>(Hkv), static_cast<int>(window_left),
                              static_cast<float>(softmax_scale));
}

void hk_attn_bwd_fused_d128(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
                            const at::Tensor &o, const at::Tensor &dO, at::Tensor &dq,
                            at::Tensor &dk, at::Tensor &dv, at::Tensor &ws, const at::Tensor &lse,
                            at::Tensor &delta, int64_t Sq, int64_t Skv, int64_t B, int64_t Hq,
                            int64_t Hkv, int64_t window_left, double softmax_scale) {
    hk::hk_attn_bwd_fused_d128(desc(q, "q"), desc(k, "k"), desc(v, "v"), desc(o, "o"), desc(dO, "dO"), desc(dq, "dq"), desc(dk, "dk"), desc(dv, "dv"), desc(ws, "ws"), desc(lse, "lse"), desc(delta, "delta"), static_cast<int>(Sq),
                               static_cast<int>(Skv), static_cast<int>(B), static_cast<int>(Hq),
                               static_cast<int>(Hkv), static_cast<int>(window_left),
                               static_cast<float>(softmax_scale));
}

int64_t hk_attn_dkdv_head_split(int64_t head_dim, int64_t Sq, int64_t Skv, int64_t B, int64_t Hq,
                                int64_t Hkv, int64_t window_left) {
    TORCH_CHECK(head_dim == 64 || head_dim == 128,
                "hipkittens attention: head_dim must be 64 or 128, got ", head_dim);
    const auto call = head_dim == 64 ? hk::hk_attn_bwd_d64_dkdv_head_split
                                     : hk::hk_attn_bwd_d128_dkdv_head_split;
    return call(static_cast<int>(Sq), static_cast<int>(Skv), static_cast<int>(B),
                static_cast<int>(Hq), static_cast<int>(Hkv), static_cast<int>(window_left));
}

// The tile sizes the Python layer pads the sequence axes to, reported by the kernels rather than
// restated there: a table that drifted from the build would pad to the wrong multiple and read
// past the end of a tensor instead of failing.
std::vector<int64_t> hk_attn_block_sizes(int64_t head_dim) {
    TORCH_CHECK(head_dim == 64 || head_dim == 128,
                "hipkittens attention: head_dim must be 64 or 128, got ", head_dim);
    int64_t fwd_q = 0, fwd_kv = 0;
    hk::HkBwdBlocks b{};
    if (head_dim == 64) {
        hk::hk_attn_fwd_d64_blocks(&fwd_q, &fwd_kv);
        hk::hk_attn_bwd_d64_blocks(&b);
    } else {
        hk::hk_attn_fwd_d128_blocks(&fwd_q, &fwd_kv);
        hk::hk_attn_bwd_d128_blocks(&b);
    }
    return {fwd_q, fwd_kv, b.dq_q, b.dq_kv, b.dkdv_q, b.dkdv_kv, b.prep_q, b.fused_q, b.fused_kv};
}

}  // namespace primus_turbo::pytorch
