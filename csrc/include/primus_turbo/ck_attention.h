#pragma once
#include "ck_tile/ops/fmha.hpp"
#include "ck_tile/core.hpp"
#include "ck_tile/host/device_prop.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/epilogue.hpp"
// #include "ck_tile/ops/fmha.hpp"

namespace primus_turbo {

// enum class mask_enum {
//     no_mask = 0,
//     mask_top_left,
//     mask_bottom_right,
//     window_generic,
// };

// enum class bias_enum {
//     no_bias          = 0,
//     elementwise_bias = 1,
//     alibi            = 2,
// };

struct FmhaFwdKernelSelectionParams {
    int hdim_q;
    int hdim_v;
    std::string data_type;
    bool is_group_mode;
    bool is_v_rowmajor;
    bool has_logits_soft_cap;
    ck_tile::GenericAttentionMaskEnum mask_type;
    ck_tile::BlockAttentionBiasEnum bias_type; // 0:no bias, 1:elementwise bias, 2:alibi. sync with BlockAttentionBiasEnum
    bool has_lse;
    bool has_dropout;
    bool do_fp8_static_quant;
    bool skip_min_seqlen_q = false;
};

struct FmhaFwdKernelRuntimeParams {
    const void* q_ptr;
    const void* k_ptr;
    const void* v_ptr;
    const void* bias_ptr; // bias or alibi_slope pointer
    void* rand_val_ptr;
    void* lse_ptr;
    void* o_ptr;

    const void* seqstart_q_ptr;
    const void* seqstart_k_ptr;
    const void*
        seqlen_k_ptr; // only used if both 'seqstart_q_ptr' & 'seqstart_k_ptr' are not nullptr

    ck_tile::index_t seqlen_q;
    ck_tile::index_t seqlen_k;
    ck_tile::index_t batch;
    ck_tile::index_t max_seqlen_q;
    ck_tile::index_t hdim_q;
    ck_tile::index_t hdim_v;
    ck_tile::index_t nhead_q;
    ck_tile::index_t nhead_k;

    float scale_s;
    float scale_p;
    float scale_o;

    float logits_soft_cap;

    ck_tile::index_t stride_q;
    ck_tile::index_t stride_k;
    ck_tile::index_t stride_v;
    ck_tile::index_t stride_bias; // if alibi, b*h need set this to h, 1*h need set this to 0
    ck_tile::index_t stride_randval;
    ck_tile::index_t stride_o;
    ck_tile::index_t nhead_stride_q;
    ck_tile::index_t nhead_stride_k;
    ck_tile::index_t nhead_stride_v;
    ck_tile::index_t nhead_stride_bias;
    ck_tile::index_t nhead_stride_randval;
    ck_tile::index_t nhead_stride_lse;
    ck_tile::index_t nhead_stride_o;
    ck_tile::index_t batch_stride_q;
    ck_tile::index_t batch_stride_k;
    ck_tile::index_t batch_stride_v;
    ck_tile::index_t batch_stride_bias;
    ck_tile::index_t batch_stride_randval;
    ck_tile::index_t batch_stride_lse;
    ck_tile::index_t batch_stride_o;

    ck_tile::index_t window_size_left;
    ck_tile::index_t window_size_right;
    ck_tile::index_t mask_type;
    ck_tile::index_t min_seqlen_q;

    float p_drop;
    bool s_randval;

    std::variant<std::pair<uint64_t, uint64_t>, std::pair<const void*, const void*>> drop_seed_offset;
};

// ck_attention interface
float attention_fwd_impl(const FmhaFwdKernelSelectionParams & t, FmhaFwdKernelRuntimeParams & args, const ck_tile::stream_config &stream_cfg);

}