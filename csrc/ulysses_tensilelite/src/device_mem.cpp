/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// Adapted from attn-a2a-demo (https://github.com/DDEle/attn-a2a-demo) at 7a7343ef47,
// csrc/device_mem.cpp.
#include "device_mem.hpp"

#include <hip/hip_runtime.h>

#include <cstring>
#include <stdexcept>

namespace aa {
namespace {
void check(hipError_t e, char const *what) {
    if (e != hipSuccess) {
        (void) hipGetLastError();
        throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
    }
}
} // namespace

torch::Tensor alloc(int64_t nbytes, bool fineGrained) {
    void *p = nullptr;
    if (fineGrained)
        check(hipExtMallocWithFlags(&p, nbytes, hipDeviceMallocFinegrained),
              "hipExtMallocWithFlags");
    else
        check(hipMalloc(&p, nbytes), "hipMalloc");
    int dev = 0;
    check(hipGetDevice(&dev), "hipGetDevice");
    return torch::from_blob(
        p, {nbytes}, [](void *q) { (void) hipFree(q); },
        torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA, dev));
}

std::string ipcHandle(torch::Tensor const &t) {
    hipIpcMemHandle_t h{};
    check(hipIpcGetMemHandle(&h, t.data_ptr()), "hipIpcGetMemHandle");
    return std::string(reinterpret_cast<char const *>(&h), sizeof(h));
}

uintptr_t ipcOpen(std::string const &handle) {
    if (handle.size() != sizeof(hipIpcMemHandle_t))
        throw std::runtime_error("ipcOpen: bad handle size");
    hipIpcMemHandle_t h{};
    std::memcpy(&h, handle.data(), sizeof(h));
    void *p = nullptr;
    check(hipIpcOpenMemHandle(&p, h, hipIpcMemLazyEnablePeerAccess), "hipIpcOpenMemHandle");
    return reinterpret_cast<uintptr_t>(p);
}

void ipcClose(uintptr_t ptr) {
    (void) hipIpcCloseMemHandle(reinterpret_cast<void *>(ptr));
}

void memcpyD2D(uintptr_t dst, uintptr_t src, int64_t nbytes) {
    check(hipMemcpy(reinterpret_cast<void *>(dst), reinterpret_cast<void const *>(src), nbytes,
                    hipMemcpyDeviceToDevice),
          "hipMemcpy");
}
} // namespace aa
