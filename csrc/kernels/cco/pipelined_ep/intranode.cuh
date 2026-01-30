#pragma once

namespace primus_turbo::cco::pipelined_ep {

namespace intranode {

struct expert_stage_config_t {
    int num_stages;
    int experts_per_stage;
    int stage_expert_start[8];
};

void dispatch_preprocess(const int64_t *topk_idx, int num_tokens, int num_topk, hipStream_t stream);

} // namespace intranode

} // namespace primus_turbo::cco::pipelined_ep
