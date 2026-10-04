###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 FlyDSL Project Contributors
#
# Adapted from FlyDSL (https://github.com/ROCm/FlyDSL)
# Modified by the Primus-Turbo team.
#
# This file is distributed under the Apache License 2.0 (see LICENSE-APACHE),
# not the MIT license that covers the rest of Primus-Turbo (see LICENSE).
###############################################################################

"""FlyDSL flash-attention forward for gfx1250 (MI455X), DeepSeek-V3 MLA head dims.

q/k head dim 192, v/o head dim 128, bf16, bottom-right causal or full attention. The
kernel is aiter's gfx1250 ``m32x8`` forward prefill (vendored, see PROVENANCE.md) with the
softmax scale applied in fp32 inside the exp2 argument instead of being folded into Q in
bf16. Import only on gfx1250 with flydsl 0.3.4.x; the entry point lives in ``interface``.
"""
