#include <ATen/ATen.h> // for TORCH_CHECK
#include <torch/extension.h>

#include "primus_turbo/dist/all_gather/dma_all_gather.h"

namespace primus_turbo::pytorch {

void dma_all_gather_into_tensor_nobuffer(uintptr_t handle_ptr, torch::Tensor output_tensor,
                                         torch::Tensor input_tensor, size_t group_rank,
                                         size_t group_world_size, uintptr_t stream_ptr) {
    printf("C++ dma_all_gather_into_tensor_nobuffer, group_rank=%d\n", (int) group_rank);
    return;

    // size_t num_ranks, uintptr_t stream_ptr) {
    size_t input_numel  = input_tensor.numel();
    size_t element_size = input_tensor.element_size(); // Size of each element in bytes
    size_t size_bytes   = input_numel * element_size;

    c10::Device device    = input_tensor.device();
    int         device_id = device.index();
    TORCH_CHECK(device_id == output_tensor.device().index());
    TORCH_CHECK(device_id >= 0 && device_id < 8);
    // std::cout << "Device type: " << device.type() << "Device ID: " << device_id
    // << std::endl;

    // TODO(wenx): check output tensor size

    run_dma_all_gather_into_tensor_nobuffer(reinterpret_cast<dmaHandle_t>(handle_ptr),
                                            output_tensor.data_ptr(), input_tensor.data_ptr(),
                                            size_bytes, group_rank, group_world_size,
                                            reinterpret_cast<hipStream_t>(stream_ptr));
}

} // namespace primus_turbo::pytorch