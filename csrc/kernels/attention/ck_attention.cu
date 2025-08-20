#include <stdexcept>
#include "ck_attention_launcher.h"

namespace primus_turbo {

float attention_fwd_impl(const FmhaFwdKernelSelectionParams & t, FmhaFwdKernelRuntimeParams & args, const ck_tile::stream_config &stream_cfg) {
    // only support bf16 with padded seqlen and hdim
    #define select_attn_fwd_launcher(hdimq, hdimv, group_mode, v_row_major, has_logits_soft_cap, mask_enum, bias_enum, has_lse, has_dropout) \
        // return 0.0;
        if (t.hdim_q == hdimq && t.hdim_v == hdimv && t.is_group_mode == group_mode && t.is_v_rowmajor == v_row_major && t.has_logits_soft_cap == has_logits_soft_cap && \
            t.mask_type == mask_enum::no_mask && t.bias_type == bias_enum && t.has_lse == has_lse && t.has_dropout == has_dropout) { \
                using traits_param = FmhaFwdKernelTraitsParam<hdimq, hdimv, FmhaFwdFp16, group_mode, v_row_major, has_logits_soft_cap, mask_enum, bias_enum, has_lse, has_dropout, true, true, true, true, false>; \
                using kernel_traits = FmhaFwdKernelTraits<traits_param>; \
                auto kernel_launcher = CKAttentionFwdLauncher<kernel_traits>(); \
                return kernel_launcher.run(stream_cfg, args); \
        } \
    // (hdimq, hdimv, group_mode, v_row_major, has_logits_soft_cap, mask_enum, bias_enum, has_lse, has_dropout)
    #define select_attn_fwd_launcher_head_dim(hdimq, hdimv) \
        select_attn_fwd_launcher(hdimq, hdimv, false, true, false, t.mask_type == mask_enum::no_mask, bias_enum::no_bias, false, false); /* no gqa, no mask */
        // select_attn_fwd_launcher(hdimq, hdimv, true, true, false, t.mask_type != mask_enum::no_mask, bias_enum::no_bias, false, false); /* no gqa, casual */ \
        // select_attn_fwd_launcher(hdimq, hdimv, true, true, false, t.mask_type == mask_enum::no_mask, bias_enum::no_bias, false, false); /* gqa no mask */ \
        // select_attn_fwd_launcher(hdimq, hdimv, true, true, false, t.mask_type != mask_enum::no_mask, bias_enum::no_bias, false, false); /* gqa casual */ \
    
    select_attn_fwd_launcher_head_dim(32, 32)
    // select_attn_fwd_launcher_head_dim(64, 64)
    // select_attn_fwd_launcher_head_dim(128, 128)
    // select_attn_fwd_launcher_head_dim(192, 128)
    // select_attn_fwd_launcher_head_dim(256, 256)

    #undef select_attn_fwd_launcher_head_dim
    #undef select_attn_fwd_launcher
    // If no matching kernel found, throw an error
    throw std::runtime_error("Unsupported attention forward kernel configuration.");
    return 0.0;
}

} // namespace primus_turbo