/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

#pragma once

#include "primus_turbo/common.h"

namespace primus_turbo {

void weight_deosc_update(const dtype::bfloat16 *current,
                         const dtype::bfloat16 *current_qdq,
                         const dtype::bfloat16 *previous,
                         const dtype::bfloat16 *previous_qdq, float *dist,
                         float *dist_qdq, int64_t numel, hipStream_t stream);

void weight_deosc_close(float *master, dtype::bfloat16 *previous,
                        const dtype::bfloat16 *current_qdq, float *dist,
                        float *dist_qdq, int64_t *reset_count, int64_t numel,
                        float ratio_threshold, float eps, bool collect_count,
                        hipStream_t stream);

} // namespace primus_turbo
