###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Regression tests pinning the public names and string values of the centralized
environment-variable keys declared in ``primus_turbo.common.constants``.

These values are part of Primus-Turbo's public configuration contract:
downstream users set environment variables using these literal strings. A
renamed or typo-introducing edit would silently break backend selection,
auto-tune, logging, and FP32 atomic attention.
"""

import pytest

from primus_turbo.common import constants


# Pin the expected literal string for every documented key. If one of these
# must change, update the constant and this table together so the contract
# change is explicit in code review.
_EXPECTED_ENV_KEYS = {
    "ENV_LOG_LEVEL": "PRIMUS_TURBO_LOG_LEVEL",
    "ENV_GEMM_BACKEND": "PRIMUS_TURBO_GEMM_BACKEND",
    "ENV_GROUPED_GEMM_BACKEND": "PRIMUS_TURBO_GROUPED_GEMM_BACKEND",
    "ENV_MOE_DISPATCH_COMBINE_BACKEND": "PRIMUS_TURBO_MOE_DISPATCH_COMBINE_BACKEND",
    "ENV_AUTO_TUNE": "PRIMUS_TURBO_AUTO_TUNE",
    "ENV_ATTN_V3_ATOMIC_FP32": "PRIMUS_TURBO_ATTN_V3_ATOMIC_FP32",
}


@pytest.mark.parametrize("attr,expected", sorted(_EXPECTED_ENV_KEYS.items()))
def test_env_var_string_is_stable(attr, expected):
    """Each ENV_* constant must resolve to the exact documented env var name."""
    assert hasattr(constants, attr), f"constants.{attr} was removed"
    assert getattr(constants, attr) == expected, (
        f"constants.{attr} changed from {expected!r} to "
        f"{getattr(constants, attr)!r}; this is a public env-var contract change."
    )


def test_env_keys_all_have_primus_turbo_prefix():
    """All Primus-Turbo env vars share a common prefix to avoid collisions."""
    for attr in _EXPECTED_ENV_KEYS:
        value = getattr(constants, attr)
        assert value.startswith("PRIMUS_TURBO_"), f"{attr}={value!r} missing PRIMUS_TURBO_ prefix"


def test_env_keys_are_unique():
    """Different knobs must not alias to the same environment variable name."""
    values = [getattr(constants, attr) for attr in _EXPECTED_ENV_KEYS]
    assert len(values) == len(set(values)), f"duplicate env var names detected: {values}"
