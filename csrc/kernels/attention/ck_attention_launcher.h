#pragma once
#include "ck_attention_traits.h"
#include "primus_turbo/ck_attention.h"

namespace primus_turbo {




template <typename FmhaFwdKernelTraits> 
class CKAttentionFwdLauncher {
public:
    float run(const ck_tile::stream_config &stream_cfg, FmhaFwdKernelRuntimeParams & args) {
        using k_ = FmhaFwdKernelTraits::fmha_kernel_0;
        auto [kargs, grids] = fmha_fwd_create_kargs_and_grids<k_>(args);
        constexpr dim3 blocks             = k_::BlockSize();
        constexpr ck_tile::index_t kBlockPerCu = k_::kBlockPerCu;
        return ck_tile::launch_kernel(stream_cfg, ck_tile::make_kernel<blocks.x, kBlockPerCu>(k_{}, grids, blocks, 0, kargs));
    }
};

}