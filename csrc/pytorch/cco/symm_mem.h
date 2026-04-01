#pragma once

namespace primus_turbo::pytorch::cco::symm_mem {
class SymmetricMemoryAllocator {
public:
    ~SymmetricMemoryAllocator() override = default;

    void alloc(void **ptr, size_t size);

    void free(void *ptr);
};

class TORCH_API SymmetricMemory : public torch::CustomClassHolder {
public:
    ~SymmetricMemory() override = default;

    virtual std::vector<void *> get_buffer_ptrs()     = 0;
    virtual std::vector<void *> get_signal_pad_ptrs() = 0;

    // get_buffer_ptrs_dev() and get_signal_pad_ptrs_dev() each return a pointer
    // to a device array of size world_size, containing buffer pointers and
    // signal pad pointers, respectively.
    virtual void **get_buffer_ptrs_dev()     = 0;
    virtual void **get_signal_pad_ptrs_dev() = 0;
    virtual size_t get_buffer_size()         = 0;
    size_t         get_signal_pad_size();

    virtual size_t get_offset() = 0;

    virtual bool  has_multicast_support() = 0;
    virtual void *get_multicast_ptr()     = 0;

    at::Tensor get_buffer(int rank, c10::IntArrayRef sizes, c10::ScalarType dtype,
                          int64_t storage_offset);

    at::Tensor get_signal_pad(int rank, c10::IntArrayRef sizes,
                              std::optional<c10::ScalarType> dtype          = std::nullopt,
                              int64_t                        storage_offset = 0);

    at::Tensor get_remote_tensor(int peer, c10::IntArrayRef sizes, c10::ScalarType dtype);

    virtual void barrier(int channel, size_t timeout_ms)                   = 0;
    virtual void put_signal(int dst_rank, int channel, size_t timeout_ms)  = 0;
    virtual void wait_signal(int src_rank, int channel, size_t timeout_ms) = 0;

    virtual int         get_rank()       = 0;
    virtual int         get_world_size() = 0;
    virtual c10::Device get_device()     = 0;

    virtual const std::vector<int> &get_rank_to_global_rank() { TORCH_CHECK(false, "NYI"); }

    virtual int *get_rank_to_global_rank_dev() { TORCH_CHECK(false, "NYI"); }

    // Returns true if *all* peers within the group are accessible via direct
    // memory load and store.
    virtual bool world_within_direct_access() { TORCH_CHECK(false, "NYI"); }
};

} // namespace primus_turbo::pytorch::cco::symm_mem