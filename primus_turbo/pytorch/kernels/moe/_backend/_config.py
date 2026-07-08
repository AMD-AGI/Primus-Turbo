###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
from dataclasses import dataclass
from typing import Any


@dataclass
class EPBufferConfig:
    """Configuration for EP communication buffer initialization.

    Attributes:
        num_sms: Number of SMs used by high-throughput kernels.
        dispatch_config: Dispatch config; ``None`` means use backend default.
        combine_config: Combine config; ``None`` means use backend default.
    """

    num_sms: int = 64
    dispatch_config: Any = None
    combine_config: Any = None
