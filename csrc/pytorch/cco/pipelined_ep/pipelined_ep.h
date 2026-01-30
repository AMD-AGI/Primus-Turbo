#pragma once

#include <string>
#include <torch/types.h>
#include <tuple>
#include <vector>

namespace primus_turbo::pytorch::cco::pipelined_ep {

class PipelinedBuffer {

public:
    PipelinedBuffer(std::string group_name);
    ~PipelinedBuffer() noexcept(false);

    std::tuple<torch::Tensor, torch::Tensor> dispatch();
};

} // namespace primus_turbo::pytorch::cco::pipelined_ep
