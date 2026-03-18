/*
 * Copyright (c) 2025 DeepSeek. All rights reserved.
 *
 * Modification Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 */

#pragma once

#include "primus_turbo/macros.h"
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>
#include <c10/cuda/CUDAStream.h>

// NOTE: pytorch hipify v2 support
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// We needed to create overloads of the inline functions here for CUDAStream.
//
// HIPStreamMasqueradingAsCUDA is no longer needed but exists for backward
// compatibility. In hipify v1 behavior, it used to be necessary to use
// HIPStreamMasqueradingAsCUDA instead of HIPStream for proper functioning of
// ROCm builds. When pytorch switched to v2 of the hipify strategy, HIPStream
// no longer exists and HIPStreamMasqueradingAsCUDA inherits from CUDAStream.
// Functions like getCurrentHIPStreamMasqueradingAsCUDA now return CUDAStream.

namespace primus_turbo::pytorch::deep_ep {

struct EventHandle {
    std::shared_ptr<torch::Event> event;

    EventHandle() {
        event = std::make_shared<torch::Event>(torch::kCUDA);
        event->record(at::hip::getCurrentHIPStreamMasqueradingAsCUDA());
    }

    explicit EventHandle(const at::hip::HIPStreamMasqueradingAsCUDA &stream) {
        event = std::make_shared<torch::Event>(torch::kCUDA);
        event->record(stream);
    }

    EventHandle(const EventHandle &other) = default;

    void current_stream_wait() const {
        at::hip::getCurrentHIPStreamMasqueradingAsCUDA().unwrap().wait(*event);
    }
};

inline torch::Event create_event(const at::hip::HIPStreamMasqueradingAsCUDA &s) {
    auto event = torch::Event(torch::kCUDA);
    event.record(s);
    return event;
}

inline torch::Event create_event(const c10::cuda::CUDAStream &s) {
    auto event = torch::Event(torch::kCUDA);
    event.record(s);
    return event;
}

inline void stream_wait(const at::hip::HIPStreamMasqueradingAsCUDA &s_0,
                        const at::hip::HIPStreamMasqueradingAsCUDA &s_1) {
    PRIMUS_TURBO_CHECK(s_0.id() != s_1.id());
    s_0.unwrap().wait(create_event(s_1));
}

inline void stream_wait(const c10::cuda::CUDAStream &s_0,
                        const c10::cuda::CUDAStream &s_1) {
    PRIMUS_TURBO_CHECK(s_0.id() != s_1.id());
    s_0.unwrap().wait(create_event(s_1));
}

inline void stream_wait(const at::hip::HIPStreamMasqueradingAsCUDA &s, const EventHandle &event) {
    s.unwrap().wait(*event.event);
}

inline void stream_wait(const c10::cuda::CUDAStream &s, const EventHandle &event) {
    s.unwrap().wait(*event.event);
}

} // namespace primus_turbo::pytorch::deep_ep
