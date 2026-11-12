/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// Adapted from attn-a2a-demo (https://github.com/DDEle/attn-a2a-demo) at 7a7343ef47,
// csrc/device_mem.hpp.
#pragma once

#include <torch/extension.h>

#include <cstdint>
#include <string>

namespace aa {
torch::Tensor alloc(int64_t nbytes, bool fineGrained);
std::string   ipcHandle(torch::Tensor const &t);
uintptr_t     ipcOpen(std::string const &handle);
void          ipcClose(uintptr_t ptr);
void          memcpyD2D(uintptr_t dst, uintptr_t src, int64_t nbytes);
} // namespace aa
