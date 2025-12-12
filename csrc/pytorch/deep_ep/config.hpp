/*
 * Copyright (c) 2025 DeepSeek. All rights reserved.
 *
 * Modification Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 */

#pragma once

#include <climits>

#include "primus_turbo/common.h"

#include "primus_turbo/deep_ep/api.h"
#include "primus_turbo/deep_ep/config.hpp"
#include "primus_turbo/deep_ep/configs.h"
#include <torch/custom_class.h>

namespace primus_turbo::pytorch::deep_ep {

using BaseConfig = primus_turbo::deep_ep::Config;

struct Config : public BaseConfig, torch::CustomClassHolder {

    Config(int64_t num_sms, int64_t num_max_nvl_chunked_send_tokens,
           int64_t num_max_nvl_chunked_recv_tokens, int64_t num_max_rdma_chunked_send_tokens,
           int64_t num_max_rdma_chunked_recv_tokens)
        : BaseConfig(static_cast<int>(num_sms), static_cast<int>(num_max_nvl_chunked_send_tokens),
                     static_cast<int>(num_max_nvl_chunked_recv_tokens),
                     static_cast<int>(num_max_rdma_chunked_send_tokens),
                     static_cast<int>(num_max_rdma_chunked_recv_tokens)) {
        // Validate that int64_t parameters fit within int range
        PRIMUS_TURBO_CHECK(num_sms <= INT_MAX && num_sms >= INT_MIN, "num_sms out of int range");
        PRIMUS_TURBO_CHECK(num_max_nvl_chunked_send_tokens <= INT_MAX &&
                               num_max_nvl_chunked_send_tokens >= INT_MIN,
                           "num_max_nvl_chunked_send_tokens out of int range");
        PRIMUS_TURBO_CHECK(num_max_nvl_chunked_recv_tokens <= INT_MAX &&
                               num_max_nvl_chunked_recv_tokens >= INT_MIN,
                           "num_max_nvl_chunked_recv_tokens out of int range");
        PRIMUS_TURBO_CHECK(num_max_rdma_chunked_send_tokens <= INT_MAX &&
                               num_max_rdma_chunked_send_tokens >= INT_MIN,
                           "num_max_rdma_chunked_send_tokens out of int range");
        PRIMUS_TURBO_CHECK(num_max_rdma_chunked_recv_tokens <= INT_MAX &&
                               num_max_rdma_chunked_recv_tokens >= INT_MIN,
                           "num_max_rdma_chunked_recv_tokens out of int range");
    }

    int64_t get_rdma_buffer_size_hint(int64_t hidden_bytes, int64_t num_ranks) const {
        return static_cast<int64_t>(
            BaseConfig::get_rdma_buffer_size_hint(hidden_bytes, static_cast<int>(num_ranks)));
    }

    int64_t get_nvl_buffer_size_hint(int64_t hidden_bytes, int64_t num_ranks) const {
        return static_cast<int64_t>(BaseConfig::get_nvl_buffer_size_hint(
            static_cast<size_t>(hidden_bytes), static_cast<int>(num_ranks)));
    }
};

} // namespace primus_turbo::pytorch::deep_ep
