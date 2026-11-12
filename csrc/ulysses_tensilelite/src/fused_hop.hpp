/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// Adapted from attn-a2a-demo (https://github.com/DDEle/attn-a2a-demo) at 7a7343ef47,
// csrc/fused_hop.hpp.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace aa {
extern const size_t kFlagBlockBytes;

struct HopShape {
    std::string mode;
    size_t      m, n, k, extent;
    int         rank, world;
    double      beta    = 0.0;
    bool        trans_a = true; // false: A read as stored (NN)
};

class FusedHop {
public:
    static constexpr int kChannels = 2;

    FusedHop(HopShape const &shape, std::string const &dat, std::string const &coDir,
             std::string const &co);
    ~FusedHop();

    std::string           kernelName() const;
    size_t                counterBytes() const;
    std::array<size_t, 2> macroTile() const;
    int                   world() const;
    std::string           modeName() const;

    // One group per rank, in rank order: flagPtr, bufPtr, queueBuf, rptr, wptr, doorbell.
    using PeerFields = std::array<void *, 6>;

    void bind(int channel, std::vector<PeerFields> const &peers, void const *a, void const *b,
              void *d, void *counter, void const *c = nullptr);
    void launch(int channel, uintptr_t stream);

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};
} // namespace aa
