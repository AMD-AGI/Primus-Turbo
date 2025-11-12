// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#include "primus_turbo/dist/all_gather/dma_all_gather.h"
#include "pytorch/extensions.h"

#include <unordered_map>

namespace primus_turbo::pytorch {

using dist::DMAHandle;

std::unordered_map<std::string, std::unique_ptr<DMAHandle>> DMAHandle::dma_handles_;

c10::intrusive_ptr<c10d::Work> dma_all_gather_into_tensor(at::Tensor       output_tensor,
                                                          const at::Tensor input_tensor,
                                                          c10::intrusive_ptr<c10d::ProcessGroup> pg,
                                                          const std::string &group_tag) {
    size_t group_rank = pg->getRank();
    size_t group_size = pg->getSize();

    DMAHandle *dma_handle = DMAHandle::get_handle(group_tag, group_rank, group_size);
    PRIMUS_TURBO_CHECK(group_rank == dma_handle->get_group_rank(), "group_rank check failed");
    PRIMUS_TURBO_CHECK(group_size == dma_handle->get_group_size(), "group_size check failed");
    size_t size_bytes = input_tensor.numel() * input_tensor.element_size();

    hipStream_t stream = c10::hip::getCurrentHIPStream().stream();
    dist::run_dma_all_gather_into_tensor_nobuffer(dma_handle, output_tensor.data_ptr(),
                                                  input_tensor.data_ptr(), size_bytes, stream);
    return c10::intrusive_ptr<c10d::Work>(nullptr);
}

} // namespace primus_turbo::pytorch
