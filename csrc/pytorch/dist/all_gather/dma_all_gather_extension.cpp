// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#include "primus_turbo/dist/all_gather/dma_all_gather.h"
#include "pytorch/extensions.h"

#include <unordered_map>

namespace primus_turbo::pytorch {

namespace dist {

class DMAAllGahterWork : public c10d::Work {
public:
    DMAAllGahterWork(DMAHandle *dma_handle, hipStream_t stream)
        : c10d::Work(-1), dma_handle_(dma_handle), stream_(stream) {}
    bool wait(std::chrono::milliseconds timeout = kNoTimeout) override {
        if (waited_) {
            return true;
        }
        run_dma_stream_wait(dma_handle_, stream_);
        return waited_ = true;
    }

private:
    DMAHandle  *dma_handle_;
    hipStream_t stream_;
    bool        waited_{false};
};

} // namespace dist

using dist::DMAHandle;
using RankInfo = dist::DMAHandle::RankInfo;

std::unordered_map<std::string, std::unique_ptr<DMAHandle>> DMAHandle::dma_handles_;

c10::intrusive_ptr<c10d::Work> dma_all_gather_into_tensor(at::Tensor       output_tensor,
                                                          const at::Tensor input_tensor,
                                                          c10::intrusive_ptr<c10d::ProcessGroup> pg,
                                                          const std::string &group_tag) {
    size_t group_rank = pg->getRank();
    size_t group_size = pg->getSize();

    DMAHandle *dma_handle = DMAHandle::get_handle(group_tag, group_rank, group_size);
    RankInfo  *rankinfo   = dma_handle->get_rankinfo();
    PRIMUS_TURBO_CHECK(group_rank == rankinfo->group_rank, "group_rank check failed");
    PRIMUS_TURBO_CHECK(group_size == rankinfo->group_size, "group_size check failed");
    size_t size_bytes = input_tensor.numel() * input_tensor.element_size();

    hipStream_t stream = c10::hip::getCurrentHIPStream().stream();
    dist::run_dma_all_gather_into_tensor_nobuffer(dma_handle, output_tensor.data_ptr(),
                                                  input_tensor.data_ptr(), size_bytes, stream);
    return c10::make_intrusive<dist::DMAAllGahterWork>(dma_handle, stream);
}

} // namespace primus_turbo::pytorch
