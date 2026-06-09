// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#pragma once

#include "primus_turbo/arch.h"

#include <cstdint>
#include <hipblaslt/hipblaslt.h>
#include <mutex>
#include <unordered_map>

namespace primus_turbo {

class HipblasltAlgoCache {
public:
    struct Key {
        int                          device_cap;
        hipDataType                  a_type, b_type, d_type;
        int64_t                      m, n, k;
        int64_t                      lda, ldb, ldd;
        hipblasOperation_t           trans_a, trans_b;
        hipblasLtMatmulMatrixScale_t scale_mode;

        bool operator==(const Key &o) const {
            return device_cap == o.device_cap && a_type == o.a_type && b_type == o.b_type &&
                   d_type == o.d_type && m == o.m && n == o.n && k == o.k && lda == o.lda &&
                   ldb == o.ldb && ldd == o.ldd && trans_a == o.trans_a && trans_b == o.trans_b &&
                   scale_mode == o.scale_mode;
        }

        struct Hash {
            size_t operator()(const Key &k) const {
                // FNV-1a over all fields
                size_t h   = 14695981039346656037ULL;
                auto   mix = [&](auto val) {
                    const auto *p = reinterpret_cast<const unsigned char *>(&val);
                    for (size_t i = 0; i < sizeof(val); ++i) {
                        h ^= static_cast<size_t>(p[i]);
                        h *= 1099511628211ULL;
                    }
                };
                mix(k.device_cap);
                mix(k.a_type);
                mix(k.b_type);
                mix(k.d_type);
                mix(k.m);
                mix(k.n);
                mix(k.k);
                mix(k.lda);
                mix(k.ldb);
                mix(k.ldd);
                mix(k.trans_a);
                mix(k.trans_b);
                mix(k.scale_mode);
                return h;
            }
        };
    };

    struct CachedAlgo {
        hipblasLtMatmulAlgo_t algo;
    };

    static HipblasltAlgoCache &instance() {
        static HipblasltAlgoCache inst;
        return inst;
    }

    bool find(const Key &key, CachedAlgo &out) {
        std::lock_guard<std::mutex> lock(mtx_);
        auto                        it = cache_.find(key);
        if (it != cache_.end()) {
            out = it->second;
            return true;
        }
        return false;
    }

    void store(const Key &key, const CachedAlgo &algo) {
        std::lock_guard<std::mutex> lock(mtx_);
        // Prevent unbounded growth under dynamic-shape workloads
        // (e.g. MoE without capacity factor produces near-unique M per expert per step)
        if (cache_.size() >= kMaxEntries) {
            cache_.clear();
        }
        cache_[key] = algo;
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mtx_);
        cache_.clear();
    }

    static int device_cap() {
        static const int cap = [] {
            switch (get_current_arch()) {
            case GPUArch::GFX942:
                return 942;
            case GPUArch::GFX950:
                return 950;
            case GPUArch::UNKNOWN:
            default:
                return 0;
            }
        }();
        return cap;
    }

    static constexpr size_t kMaxEntries = 256;

private:
    HipblasltAlgoCache() = default;

    std::mutex                                     mtx_;
    std::unordered_map<Key, CachedAlgo, Key::Hash> cache_;
};

} // namespace primus_turbo
