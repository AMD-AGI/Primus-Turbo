/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

#include "pytorch/extensions.h"

namespace primus_turbo::pytorch {

void weight_deosc_update_meta(const at::Tensor current, const at::Tensor current_qdq,
                              const at::Tensor previous, const at::Tensor previous_qdq,
                              at::Tensor dist, at::Tensor dist_qdq) {}

void weight_deosc_close_meta(at::Tensor master, at::Tensor previous,
                             const at::Tensor current_qdq, at::Tensor dist,
                             at::Tensor dist_qdq, double ratio_threshold, double eps,
                             c10::optional<at::Tensor> reset_count) {}

} // namespace primus_turbo::pytorch
