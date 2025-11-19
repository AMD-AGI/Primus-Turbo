###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from .quantization import dequantize_fp8, quantize_fp8

__all__ = ["quantize_fp8", "dequantize_fp8"]
