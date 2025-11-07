// #include <ATen/ATen.h>
#include "pytorch/extensions.h"

#include "primus_turbo/dist/all_gather/dma_all_gather.h"

namespace {

uintptr_t getGlobalHandle(int64_t pg_id, size_t group_rank, size_t group_size) {
    // TODO (limou)
    // different handles for each process group id
    // ...
    static uintptr_t global_handle = primus_turbo::pytorch::create_all_gather_handle(
        "allgather_shm_tag", group_rank, group_size);
    return global_handle;
}

} // namespace

namespace primus_turbo::pytorch {

c10::intrusive_ptr<c10d::Work>
dma_all_gather_into_tensor(at::Tensor output_tensor, const at::Tensor input_tensor,
                           c10::intrusive_ptr<c10d::ProcessGroup> pg) {
    int       group_rank = pg->getRank();
    int       group_size = pg->getSize();
    uintptr_t handle     = getGlobalHandle(pg->getID(), group_rank, group_size);
    // printf("C++ dma_all_gather_into_tensor, rank=%d, size=%d, handle=%lu\n", rank, size,
    //        (uintptr_t) handle);

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

    hipStream_t stream = c10::hip::getCurrentHIPStream().stream();
    run_dma_all_gather_into_tensor_nobuffer(reinterpret_cast<dmaHandle_t>(handle),
                                            output_tensor.data_ptr(), input_tensor.data_ptr(),
                                            size_bytes, group_rank, group_size, stream);
    return c10::intrusive_ptr<c10d::Work>(nullptr);
}

} // namespace primus_turbo::pytorch