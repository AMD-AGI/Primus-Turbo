###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from .token_dispatcher import *

# NOTE: MegaMoEFP8 (mega_moe_fp8) is intentionally NOT re-exported here. Importing it eagerly pulls
# the fp8 op -> flydsl.mega.fp8 -> pytorch.core -> back into pytorch package init (circular). Like
# the bf16 mega op, import it by path: `from primus_turbo.pytorch.modules.moe.mega_moe_fp8 import
# MegaMoEFP8`.
