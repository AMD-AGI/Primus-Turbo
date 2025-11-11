// #include <ATen/ATen.h>
#include "pytorch/extensions.h"
// #include "primus_turbo/macros.h"
#include "primus_turbo/dist/all_gather/dma_all_gather.h"

#include <sstream>
#include <unordered_map>

namespace {

class DMAHandle final {
    static inline std::unordered_map<std::string, std::unique_ptr<DMAHandle>> dma_handles_;

public:
    static DMAHandle *GetHandle(const std::string &group_tag, size_t group_rank,
                                size_t group_size) {
        // TODO (limou)
        // multiple-threads cases
        auto it = dma_handles_.find(group_tag);
        if (it == dma_handles_.end()) {
            uintptr_t new_handle_ptr =
                primus_turbo::pytorch::create_all_gather_handle(group_tag, group_rank, group_size);
            auto uptr = std::unique_ptr<DMAHandle>(
                new DMAHandle(new_handle_ptr, group_tag, group_rank, group_size));

            auto result = dma_handles_.emplace(group_tag, std::move(uptr));
            PRIMUS_TURBO_CHECK(result.second, "emplace DMAHandle failed");
            it = result.first;
        }
        return it->second.get();
    }
    ~DMAHandle() {
        if (handle_ptr_ != 0) {
            primus_turbo::pytorch::destroy_all_gather_handle(handle_ptr_, shm_tag_, group_rank_,
                                                             group_size_);
            handle_ptr_ = 0;
        }
    }
    uintptr_t GetPtr() const { return handle_ptr_; }
    DMAHandle(const DMAHandle &)            = delete;
    DMAHandle(DMAHandle &&)                 = delete;
    DMAHandle &operator=(const DMAHandle &) = delete;
    DMAHandle &operator=(DMAHandle &&)      = delete;

private:
    DMAHandle(uintptr_t handle_ptr, const std::string &shm_tag, size_t group_rank,
              size_t group_size)
        : handle_ptr_(handle_ptr), shm_tag_(shm_tag), group_rank_(group_rank),
          group_size_(group_size) {}

private:
    uintptr_t         handle_ptr_{0};
    const std::string shm_tag_;
    size_t            group_rank_{0};
    size_t            group_size_{0};
};

} // namespace

namespace primus_turbo::pytorch {

c10::intrusive_ptr<c10d::Work> dma_all_gather_into_tensor(at::Tensor       output_tensor,
                                                          const at::Tensor input_tensor,
                                                          c10::intrusive_ptr<c10d::ProcessGroup> pg,
                                                          const std::string &group_tag) {
    int group_rank = pg->getRank();
    int group_size = pg->getSize();

    DMAHandle *dma_handle = DMAHandle::GetHandle(group_tag, group_rank, group_size);
    size_t     size_bytes = input_tensor.numel() * input_tensor.element_size();

    hipStream_t stream = c10::hip::getCurrentHIPStream().stream();
    run_dma_all_gather_into_tensor_nobuffer(reinterpret_cast<dmaHandle_t>(dma_handle->GetPtr()),
                                            output_tensor.data_ptr(), input_tensor.data_ptr(),
                                            size_bytes, group_rank, group_size, stream);
    return c10::intrusive_ptr<c10d::Work>(nullptr);
}

} // namespace primus_turbo::pytorch