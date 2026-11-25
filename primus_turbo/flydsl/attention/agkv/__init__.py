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

"""FlyDSL kernels of the fused all-gather-KV context-parallel attention (gfx950, single node).

``fwd.forward`` and ``bwd.backward`` are one persistent kernel each with the K / V exchange inside it;
``comm`` holds the XGMI push loop and flag waits they share. The host side (exchange buffers, copy unit
tables, autograd) is ``primus_turbo.pytorch.ops.attention.agkv``.
"""
