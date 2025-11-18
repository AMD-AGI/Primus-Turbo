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
#include <torch/custom_class.h>
namespace primus_turbo::pytorch::deep_ep {

struct EventHandle : torch::CustomClassHolder {
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

    std::tuple<> obj_flatten() const { return std::make_tuple(); }
};

inline torch::Event create_event(const at::hip::HIPStreamMasqueradingAsCUDA &s) {
    auto event = torch::Event(torch::kCUDA);
    event.record(s);
    return event;
}

inline void stream_wait(const at::hip::HIPStreamMasqueradingAsCUDA &s_0,
                        const at::hip::HIPStreamMasqueradingAsCUDA &s_1) {
    PRIMUS_TURBO_CHECK(s_0.id() != s_1.id());
    s_0.unwrap().wait(create_event(s_1));
}

inline void stream_wait(const at::hip::HIPStreamMasqueradingAsCUDA &s, const EventHandle &event) {
    s.unwrap().wait(*event.event);
}

} // namespace primus_turbo::pytorch::deep_ep
