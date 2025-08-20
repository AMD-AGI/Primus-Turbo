#include <torch/extension.h>

#include "../extensions.h"

namespace primus_turbo::pytorch::attention {

std::vector<at::Tensor> mha_fwd(
    at::Tensor &q,                            // batch_size x seqlen_q x num_heads x round_multiple(head_size_qk, 8)
    const at::Tensor &k,                      // batch_size x seqlen_k x num_heads_k x round_multiple(head_size_qk, 8)
    const at::Tensor &v,                      // batch_size x seqlen_k x num_heads_k x round_multiple(head_size_v, 8)
    std::optional<at::Tensor> &out_,          // batch_size x seqlen_q x num_heads x round_multiple(head_size_v, 8)
    std::optional<at::Tensor> &alibi_slopes_, // num_heads or batch_size x num_heads
    const float p_dropout,
    const float softmax_scale,
    bool is_causal,
    int window_size_left,
    int window_size_right,
    const bool return_dropout_randval,
    std::optional<at::Generator> gen_
) {
    const int64_t b = q.size(0);
    const int64_t s = q.size(1);
    const int64_t h = q.size(2);
    const int64_t d = v.size(3);

    // out, softmax_lse, p, rng_state
    return {at::empty({b, h, s, d}, a.options().device(at::kMeta)), 
            at::empty({b, s, h}, a.options().device(at::kMeta)), 
            at::empty({0}, a.options().device(at::kMeta));
            at::empty({2}, opts.dtype(torch::kInt64), a.options().device(at::kMeta));
           }
}

}