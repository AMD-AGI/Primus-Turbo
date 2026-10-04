/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

#pragma once

#include "primus_turbo/common.h"

namespace primus_turbo {

// Direct BF16-rounded MXFP4 QDQ of a flattened local shard, followed by
// snapshot/tracker update and optional period closure. Missing shard-boundary
// values are zero, exactly as in Primus's tile-staging reference path.
void weight_deosc_qdq(float *master, dtype::bfloat16 *previous, dtype::bfloat16 *previous_qdq,
                      float *dist, float *dist_qdq, int64_t *reset_count, int64_t numel,
                      int64_t rows, int64_t cols, int64_t start, int scale_rounding_mode, bool seed,
                      bool close, float ratio_threshold, float eps, bool grouped,
                      hipStream_t stream);

void weight_deosc_update(const dtype::bfloat16 *current, const dtype::bfloat16 *current_qdq,
                         const dtype::bfloat16 *previous, const dtype::bfloat16 *previous_qdq,
                         float *dist, float *dist_qdq, int64_t numel, hipStream_t stream);

void weight_deosc_close(float *master, dtype::bfloat16 *previous,
                        const dtype::bfloat16 *current_qdq, float *dist, float *dist_qdq,
                        int64_t *reset_count, int64_t numel, float ratio_threshold, float eps,
                        bool collect_count, hipStream_t stream);

} // namespace primus_turbo
