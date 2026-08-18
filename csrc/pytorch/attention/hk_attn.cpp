/***************************************************************************************************
 * Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// Torch op registration for the HipKittens gfx950 attention kernels.
//
// These are registered as real custom ops rather than exposed as pybind functions so a
// torch.compile(fullgraph=True) caller does not trace into them, the same reason the FlyDSL
// entry points are ops. The kernels write through their output arguments, so every op is
// declared mutating on them and returns nothing.
//
// The whole library is only compiled when the build targets gfx950 -- see
// build_hipkittens_extension in setup.py -- so a caller must check the arch before reaching
// here. primus_turbo/hipkittens/attention/gfx950/ does that, along with the rest of the
// envelope these kernels admit.

#include <torch/library.h>

#include "primus_turbo/hipkittens/attention.h"

namespace primus_turbo::hipkittens {
namespace {

// int64_t across the boundary because that is the only integer type the torch schema carries;
// the kernels take int, and every one of these is a shape or a tile index well inside its range.
void fwd_d64_op(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v, at::Tensor &o,
                at::Tensor &lse, int64_t Sq, int64_t Skv, int64_t B, int64_t Hq, int64_t Hkv,
                int64_t window_left, double softmax_scale) {
    hk_attn_fwd_d64(q, k, v, o, lse, static_cast<int>(Sq), static_cast<int>(Skv),
                    static_cast<int>(B), static_cast<int>(Hq), static_cast<int>(Hkv),
                    static_cast<int>(window_left), static_cast<float>(softmax_scale));
}

void fwd_d128_op(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v, at::Tensor &o,
                 at::Tensor &lse, int64_t Sq, int64_t Skv, int64_t B, int64_t Hq, int64_t Hkv,
                 int64_t window_left, double softmax_scale) {
    hk_attn_fwd_d128(q, k, v, o, lse, static_cast<int>(Sq), static_cast<int>(Skv),
                     static_cast<int>(B), static_cast<int>(Hq), static_cast<int>(Hkv),
                     static_cast<int>(window_left), static_cast<float>(softmax_scale));
}

void bwd_d64_op(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v, const at::Tensor &o,
                const at::Tensor &dO, at::Tensor &dq, at::Tensor &dk, at::Tensor &dv,
                const at::Tensor &lse, at::Tensor &delta, at::Tensor &lneg, at::Tensor &wsk,
                at::Tensor &wsv, int64_t Sq, int64_t Skv, int64_t B, int64_t Hq, int64_t Hkv,
                int64_t window_left, double softmax_scale, int64_t n_split_req) {
    hk_attn_bwd_d64(q, k, v, o, dO, dq, dk, dv, lse, delta, lneg, wsk, wsv, static_cast<int>(Sq),
                    static_cast<int>(Skv), static_cast<int>(B), static_cast<int>(Hq),
                    static_cast<int>(Hkv), static_cast<int>(window_left),
                    static_cast<float>(softmax_scale), static_cast<int>(n_split_req));
}

void bwd_d128_op(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v, const at::Tensor &o,
                 const at::Tensor &dO, at::Tensor &dq, at::Tensor &dk, at::Tensor &dv,
                 const at::Tensor &lse, at::Tensor &delta, at::Tensor &lneg, at::Tensor &wsk,
                 at::Tensor &wsv, int64_t Sq, int64_t Skv, int64_t B, int64_t Hq, int64_t Hkv,
                 int64_t window_left, double softmax_scale, int64_t n_split_req) {
    hk_attn_bwd_d128(q, k, v, o, dO, dq, dk, dv, lse, delta, lneg, wsk, wsv, static_cast<int>(Sq),
                     static_cast<int>(Skv), static_cast<int>(B), static_cast<int>(Hq),
                     static_cast<int>(Hkv), static_cast<int>(window_left),
                     static_cast<float>(softmax_scale), static_cast<int>(n_split_req));
}

void bwd_fused_d64_op(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
                      const at::Tensor &o, const at::Tensor &dO, at::Tensor &dq, at::Tensor &dk,
                      at::Tensor &dv, at::Tensor &ws, const at::Tensor &lse, at::Tensor &delta,
                      int64_t Sq, int64_t Skv, int64_t B, int64_t Hq, int64_t Hkv,
                      int64_t window_left, double softmax_scale) {
    hk_attn_bwd_fused_d64(q, k, v, o, dO, dq, dk, dv, ws, lse, delta, static_cast<int>(Sq),
                          static_cast<int>(Skv), static_cast<int>(B), static_cast<int>(Hq),
                          static_cast<int>(Hkv), static_cast<int>(window_left),
                          static_cast<float>(softmax_scale));
}

void bwd_fused_d128_op(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
                       const at::Tensor &o, const at::Tensor &dO, at::Tensor &dq, at::Tensor &dk,
                       at::Tensor &dv, at::Tensor &ws, const at::Tensor &lse, at::Tensor &delta,
                       int64_t Sq, int64_t Skv, int64_t B, int64_t Hq, int64_t Hkv,
                       int64_t window_left, double softmax_scale) {
    hk_attn_bwd_fused_d128(q, k, v, o, dO, dq, dk, dv, ws, lse, delta, static_cast<int>(Sq),
                           static_cast<int>(Skv), static_cast<int>(B), static_cast<int>(Hq),
                           static_cast<int>(Hkv), static_cast<int>(window_left),
                           static_cast<float>(softmax_scale));
}

int64_t dkdv_head_split_op(int64_t head_dim, int64_t Sq, int64_t Skv, int64_t B, int64_t Hq,
                           int64_t Hkv, int64_t window_left) {
    TORCH_CHECK(head_dim == 64 || head_dim == 128,
                "hipkittens attention: head_dim must be 64 or 128, got ", head_dim);
    const auto call =
        head_dim == 64 ? hk_attn_bwd_d64_dkdv_head_split : hk_attn_bwd_d128_dkdv_head_split;
    return call(static_cast<int>(Sq), static_cast<int>(Skv), static_cast<int>(B),
                static_cast<int>(Hq), static_cast<int>(Hkv), static_cast<int>(window_left));
}

// The block sizes the Python layer pads to, read from the kernels rather than restated there.
std::vector<int64_t> block_sizes_op(int64_t head_dim) {
    TORCH_CHECK(head_dim == 64 || head_dim == 128,
                "hipkittens attention: head_dim must be 64 or 128, got ", head_dim);
    int64_t fwd_q = 0, fwd_kv = 0;
    HkBwdBlocks b{};
    if (head_dim == 64) {
        hk_attn_fwd_d64_blocks(&fwd_q, &fwd_kv);
        hk_attn_bwd_d64_blocks(&b);
    } else {
        hk_attn_fwd_d128_blocks(&fwd_q, &fwd_kv);
        hk_attn_bwd_d128_blocks(&b);
    }
    return {fwd_q, fwd_kv, b.dq_q, b.dq_kv, b.dkdv_q, b.dkdv_kv, b.prep_q, b.fused_q, b.fused_kv};
}

}  // namespace

TORCH_LIBRARY(primus_turbo_hipkittens, m) {
    m.def(
        "attn_fwd_d64(Tensor q, Tensor k, Tensor v, Tensor(a!) o, Tensor(b!) lse, int Sq, int Skv, "
        "int B, int Hq, int Hkv, int window_left, float softmax_scale) -> ()");
    m.def(
        "attn_fwd_d128(Tensor q, Tensor k, Tensor v, Tensor(a!) o, Tensor(b!) lse, int Sq, int Skv, "
        "int B, int Hq, int Hkv, int window_left, float softmax_scale) -> ()");
    m.def(
        "attn_bwd_d64(Tensor q, Tensor k, Tensor v, Tensor o, Tensor dO, Tensor(a!) dq, "
        "Tensor(b!) dk, Tensor(c!) dv, Tensor lse, Tensor(d!) delta, Tensor(e!) lneg, "
        "Tensor(f!) wsk, Tensor(g!) wsv, int Sq, int Skv, int B, int Hq, int Hkv, "
        "int window_left, float softmax_scale, int n_split_req) -> ()");
    m.def(
        "attn_bwd_d128(Tensor q, Tensor k, Tensor v, Tensor o, Tensor dO, Tensor(a!) dq, "
        "Tensor(b!) dk, Tensor(c!) dv, Tensor lse, Tensor(d!) delta, Tensor(e!) lneg, "
        "Tensor(f!) wsk, Tensor(g!) wsv, int Sq, int Skv, int B, int Hq, int Hkv, "
        "int window_left, float softmax_scale, int n_split_req) -> ()");
    m.def(
        "attn_bwd_fused_d64(Tensor q, Tensor k, Tensor v, Tensor o, Tensor dO, Tensor(a!) dq, "
        "Tensor(b!) dk, Tensor(c!) dv, Tensor(d!) ws, Tensor lse, Tensor(e!) delta, int Sq, "
        "int Skv, int B, int Hq, int Hkv, int window_left, float softmax_scale) -> ()");
    m.def(
        "attn_bwd_fused_d128(Tensor q, Tensor k, Tensor v, Tensor o, Tensor dO, Tensor(a!) dq, "
        "Tensor(b!) dk, Tensor(c!) dv, Tensor(d!) ws, Tensor lse, Tensor(e!) delta, int Sq, "
        "int Skv, int B, int Hq, int Hkv, int window_left, float softmax_scale) -> ()");
    m.def(
        "dkdv_head_split(int head_dim, int Sq, int Skv, int B, int Hq, int Hkv, "
        "int window_left) -> int");
    m.def("block_sizes(int head_dim) -> int[]");
}

TORCH_LIBRARY_IMPL(primus_turbo_hipkittens, CUDA, m) {
    m.impl("attn_fwd_d64", &fwd_d64_op);
    m.impl("attn_fwd_d128", &fwd_d128_op);
    m.impl("attn_bwd_d64", &bwd_d64_op);
    m.impl("attn_bwd_d128", &bwd_d128_op);
    m.impl("attn_bwd_fused_d64", &bwd_fused_d64_op);
    m.impl("attn_bwd_fused_d128", &bwd_fused_d128_op);
}

// Shape-only entry points: no tensors, so they answer the same on any device.
TORCH_LIBRARY_IMPL(primus_turbo_hipkittens, CompositeExplicitAutograd, m) {
    m.impl("dkdv_head_split", &dkdv_head_split_op);
    m.impl("block_sizes", &block_sizes_op);
}

}  // namespace primus_turbo::hipkittens
