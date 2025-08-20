#pragma once
#include "primus_turbo/ck_attention.h"

namespace primus_turbo::pytorch::attention {

void init_kernel_selection_params(
    FmhaFwdKernelSelectionParams & params,
    const int hdim_q, 
    const int hdim_v, 
    const std::string & data_type, 
    bool is_group_mode,
    bool is_v_rowmajor,
    bool has_logits_soft_cap,
    mask_info mask,
    bias_enum bias_type,
    bool has_lse,
    bool has_dropout,
    bool do_fp8_static_quant,
    bool skip_min_seqlen_q = false
) {
    params.hdim_q        = hdim_q;
    params.hdim_v        = hdim_v;
    params.data_type     = data_type;
    params.is_v_rowmajor = is_v_rowmajor;

    params.is_group_mode       = (mode == mode_enum::group);
    params.has_logits_soft_cap = 0.f < logits_soft_cap;
    params.mask_type           = mask.type;
    params.bias_type           = bias.type;
    params.has_lse             = lse;
    params.do_fp8_static_quant = squant;

    params.has_dropout = (p_drop > 0.0f);
}

FmhaFwdKernelRuntimeParams create_fmha_fwd_runtime_args(
    bool has_lse,
    bool has_dropout_randval,
    const mask_info &mask,
    // sizes
    const int b,
    const int seqlen_q,
    const int seqlen_k,
    const int h,
    const int h_k,
    const int d_qk,
    const int d_v,
    // device pointers
    const at::Tensor q,
    const at::Tensor k,
    const at::Tensor v,
    std::optional<at::Tensor> &alibi_slopes_,
    at::Tensor out,
    at::Tensor softmax_lse,
    at::Tensor dropout_randval,
    float softmax_scale,
    float p_dropout,
    std::pair<uint64_t*, uint64_t*> drop_seed_offset
) {
    // q: (batch_size, seqlen_q, nheads, d_qk)
    // k: (batch_size, seqlen_k, nheads_k, d_qk)
    // v: (batch_size, seqlen_k, nheads_k, d_v)
    // o: (batch_size, seqlen_q, nheads, d_v)

    // alibi_slopes:(batch_size, nheads) or (nhead)
    // lse: (batch_size, nheads, seqlen_q)
    // randval: (batch_size, nheads, seqlen_q, seqlen_k)

    ck_tile::index_t stride_q = q.stride(1);
    ck_tile::index_t stride_k = k.stride(1);
    ck_tile::index_t stride_v = v.stride(1);
    ck_tile::index_t stride_o = out.stride(1);
    ck_tile::index_t stride_randval = has_dropout_randval ? dropout_randval.stride(2) : 0;

    ck_tile::index_t nhead_stride_q = q.stride(2);
    ck_tile::index_t nhead_stride_k = k.stride(2);
    ck_tile::index_t nhead_stride_v = v.stride(2);
    ck_tile::index_t nhead_stride_o = out.stride(2);
    ck_tile::index_t nhead_stride_lse = has_lse ? softmax_lse.stride(1) : 0;
    ck_tile::index_t nhead_stride_randval = has_dropout_randval ? dropout_randval.stride(1) : 0;

    ck_tile::index_t batch_stride_q = q.stride(0);
    ck_tile::index_t batch_stride_k = k.stride(0);
    ck_tile::index_t batch_stride_v = v.stride(0);
    ck_tile::index_t batch_stride_o = out.stride(0);

    ck_tile::index_t batch_stride_lse = has_lse ? softmax_lse.stride(0) : 0;
    ck_tile::index_t batch_stride_randval = has_dropout_randval ? dropout_randval.stride(0) : 0;

    void *alibi_slopes_ptr = nullptr;
    ck_tile::index_t stride_alibi_slopes = 0;

    if (alibi_slopes_.has_value()) {
        auto alibi_slopes = alibi_slopes_.value();
        CHECK_DEVICE(alibi_slopes);
        TORCH_CHECK(alibi_slopes.stride(-1) == 1, "ALiBi slopes tensor must have contiguous last dimension");
        TORCH_CHECK(alibi_slopes.sizes() == torch::IntArrayRef({h}) || alibi_slopes.sizes() == torch::IntArrayRef({b, h}));
        alibi_slopes_ptr = alibi_slopes.data_ptr();
        stride_alibi_slopes = alibi_slopes.dim() == 2 ? alibi_slopes.stride(0) : 0;
    }

    return FmhaFwdKernelRuntimeParams{
        q.data_ptr(),
        k.data_ptr(),
        v.data_ptr(),
        alibi_slopes_ptr, // bias
        has_dropout_randval ? dropout_randval.data_ptr() : nullptr,
        has_lse ? softmax_lse.data_ptr() : nullptr,
        out.data_ptr(),
        nullptr, // seqstart_q
        nullptr, // seqstart_k
        nullptr,
        seqlen_q,
        seqlen_k,
        b,
        seqlen_q,      // max_seqlen_q
        d_qk,             // hdim_q
        d_v,             // hdim_v
        h,             // nhead
        h_k,           // nhead_k
        softmax_scale, // scale_s
        1,             // scale_p
        1,             // scale_o
        0.0f,          // logits_soft_cap
        stride_q,
        stride_k,
        stride_v,
        stride_alibi_slopes,
        stride_randval,
        stride_o,
        nhead_stride_q,
        nhead_stride_k,
        nhead_stride_v,
        0, // nhead_stride_bias, FA without bias
        nhead_stride_randval,
        nhead_stride_lse,
        nhead_stride_o,
        batch_stride_q,
        batch_stride_k,
        batch_stride_v,
        0, // batch_stride_bias, FA without bias
        batch_stride_randval,
        batch_stride_lse,
        batch_stride_o,
        mask.left,
        mask.right,
        static_cast<ck_tile::index_t>(mask.type),
        0, // min_seqlen_q
        p_dropout,
        has_dropout_randval,
        drop_seed_offset
    };
}

}