###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

# ---------------------------------------------------------------------------
# Centralized environment-variable keys used throughout Primus-Turbo.
#
# Keeping every key in one place avoids typo-induced mismatches and makes it
# easy to discover which knobs the library exposes.
# ---------------------------------------------------------------------------

# Log level for the primus_turbo logger (DEBUG / INFO / WARNING / ERROR / CRITICAL).
# Default: WARNING
ENV_LOG_LEVEL = "PRIMUS_TURBO_LOG_LEVEL"

# GEMM backend selection (e.g. HIPBLASLT, AITER).
# Supports per-precision format: "FP4:HIPBLASLT,FP8:AITER" or a single value.
# Default: None (auto-select)
ENV_GEMM_BACKEND = "PRIMUS_TURBO_GEMM_BACKEND"

# Grouped GEMM backend selection. Same format as ENV_GEMM_BACKEND.
# Default: None (auto-select)
ENV_GROUPED_GEMM_BACKEND = "PRIMUS_TURBO_GROUPED_GEMM_BACKEND"

# MoE dispatch/combine EP backend (TURBO, DEEP_EP, or custom names like UCCL_EP).
# Default: TURBO
ENV_EP_BACKEND = "PRIMUS_TURBO_EP_BACKEND"

# Enable auto-tuning across registered kernel backends ("1" to enable).
# Default: "0" (disabled)
ENV_AUTO_TUNE = "PRIMUS_TURBO_AUTO_TUNE"

# Whether Attention V3 uses FP32 atomic accumulation ("1" to enable, "0" to disable).
# Default: "1" (enabled)
ENV_ATTN_V3_ATOMIC_FP32 = "PRIMUS_TURBO_ATTN_V3_ATOMIC_FP32"
