###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""BF16 Mega MoE building blocks for gfx1250 (MI455X): wave32, WMMA, TDM.

Import only after ``is_gfx1250()``: the tiles build WMMA / TDM atoms that do not exist on gfx9.
"""
