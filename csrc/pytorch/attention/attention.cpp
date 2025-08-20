#include "primus_turbo/ck_attention.h"
#include "ck_attention_helper.hpp"

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
    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kBFloat16, "TurboAttention only support bf16 data type");

    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");

    std::string q_dtype_str = "bf16";

    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");

    const auto sizes = q.sizes();

    const int batch_size = sizes[0];
    int seqlen_q = sizes[1];

    int num_heads = sizes[2];
    const int head_size_qk = sizes[3];

    const int seqlen_k = k.size(1);
    const int num_heads_k = k.size(2);
    const int head_size_v = v.size(3);
    
    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(head_size_qk <= 256 && head_size_v <= 256, "CK only supports head dimension at most 256");
    TORCH_CHECK(head_size_qk % 8 == 0 && head_size_v % 8 == 0, "query, key, value, and out_ must have a head_size that is a multiple of 8");
    
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    if (window_size_left >= seqlen_k) { window_size_left = -1; }
    if (window_size_right >= seqlen_k) { window_size_right = -1; }

    // causal=true is the same as causal=false in this case
    if (seqlen_q == 1 && !alibi_slopes_.has_value()) { is_causal = false; }

    mask_info mask;
    if (is_causal) {
        // Causal is the special case where window_size_right == 0 and window_size_left < 0.
        window_size_right = 0;
        std::string mask_identify = "b:" + std::to_string(window_size_left) + "," + "0";
        mask = mask_info::decode(mask_identify, seqlen_q, seqlen_k); // casual
    } else if (window_size_left == -1 && window_size_right == -1) {
        mask = mask_info::decode("0", seqlen_q, seqlen_k); // no mask
    } else {
        // Local is the more general case where window_size_right >= 0 or window_size_left >= 0.
        std::string mask_identify = "b:" + std::to_string(window_size_left) + "," + std::to_string(window_size_right);
        mask = mask_info::decode(mask_identify, seqlen_q, seqlen_k); // local
    }

    // Faster to transpose q from (b, 1, (nheads_kv ngroups), d) to (b, ngroups, nheads_kv, d) in this case
    // H/t Daniel Haziza
    const int seqlenq_ngroups_swapped = seqlen_q == 1 && num_heads > num_heads_k && window_size_left < 0 && window_size_right < 0 && p_dropout == 0.f && head_size_qk % 8 == 0 && !alibi_slopes_.has_value();
    const int ngroups = num_heads / num_heads_k;
    if (seqlenq_ngroups_swapped) {
        q = q.reshape({batch_size, num_heads_k, ngroups, head_size_qk}).transpose(1, 2);
        seqlen_q = ngroups;
        num_heads = num_heads_k;
    }

    CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size_qk);
    CHECK_SHAPE(k, batch_size, seqlen_k, num_heads_k, head_size_qk);
    CHECK_SHAPE(v, batch_size, seqlen_k, num_heads_k, head_size_v);

    at::Tensor out;
    if (out_.has_value()) {
        out = out_.value();
        TORCH_CHECK(out.dtype() == q_dtype, "Output must have the same dtype as inputs");
        CHECK_DEVICE(out);
        TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
        CHECK_SHAPE(out, batch_size, sizes[1], sizes[2], head_size_v);
        if (seqlenq_ngroups_swapped) {
            out = out.reshape({batch_size, num_heads_k, ngroups, head_size_v}).transpose(1, 2);
        }
    } else {
        out = torch::empty_like(q);
    }

    // Otherwise the kernel will be launched from cuda:0 device
    at::cuda::CUDAGuard device_guard{q.device()};

    auto opts = q.options();
    bool has_lse = true;
    bool has_dropout = p_dropout > 0.0f;

    at::Tensor softmax_lse;
    // TODO - check gradient, only training require lse
    softmax_lse = torch::empty({batch_size, num_heads, seqlen_q}, opts.dtype(torch::kFloat32));

    at::Tensor p;
    if (return_dropout_randval) {
        TORCH_CHECK(has_dropout, "return_dropout_randval require p_dropout > 0");
        p = torch::empty({batch_size, num_heads, seqlen_q, seqlen_k}, opts.dtype(torch::kUInt8));
    }
    else {
        p = torch::empty({ 0 }, opts);
    }

    int64_t counter_offset = batch_size * num_heads * ck_tile::get_warp_size();
    auto rng_state = torch::empty({2}, opts.dtype(torch::kInt64));
    auto rng_state_ptr = reinterpret_cast<uint64_t*>(rng_state.data_ptr());

    if (p_dropout > 0.0)  {
        auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
            gen_, at::cuda::detail::getDefaultCUDAGenerator());
        // See Note [Acquire lock when using random generators]
        std::lock_guard<std::mutex> lock(gen->mutex_);
        auto philox_args = gen->philox_cuda_state(counter_offset);
        hipLaunchKernelGGL(flash::ParsePhiloxCudaState, dim3(1), dim3(64), 0, 0, philox_args, rng_state_ptr);
    }

    if (seqlen_k > 0) {
        auto drop_seed_offset = std::make_pair(rng_state_ptr, rng_state_ptr + 1);
        auto stream = at::cuda::getCurrentHIPStream().stream();
        ck_tile::stream_config stream_config{stream};

        auto args = create_fmha_fwd_runtime_args(
            has_lse,
            return_dropout_randval,
            mask,
            batch_size,
            seqlen_q,
            seqlen_k,
            num_heads,
            num_heads_k,
            head_size_qk,
            head_size_v,
            q,
            k,
            v,
            alibi_slopes_,
            out,
            softmax_lse,
            p,
            softmax_scale,
            p_dropout,
            drop_seed_offset
        );

        FmhaFwdKernelSelectionParams params;
        init_kernel_selection_params(
            params,
            hdim_q,
            hdim_v,
            data_type,
            is_group_mode,
            is_v_rowmajor,
            has_logits_soft_cap,
            mask_type,
            bias_type,
            has_lse,
            has_dropout,
            do_fp8_static_quant,
            skip_min_seqlen_q
        );

        float t = attention_fwd_impl(params, args, stream_config);
        // float t = fmha_fwd(traits, args, stream_config);
        TORCH_CHECK(t >= 0, "invalid argument for fmha_fwd");
    } else {
        // If seqlen_k == 0, then we have an empty tensor. We need to set the output to 0.
        out.zero_();
        softmax_lse.fill_(std::numeric_limits<float>::infinity());
    }

    if (seqlenq_ngroups_swapped) {
        out = out.transpose(1, 2).reshape({batch_size, 1, num_heads_k * seqlen_q, head_size_v});
        q = q.transpose(1, 2).reshape({batch_size, 1, num_heads_k * seqlen_q, head_size_qk});
        softmax_lse = softmax_lse.reshape({batch_size, num_heads_k * seqlen_q, 1});
    }
    return {out, softmax_lse, p, rng_state};
}

}