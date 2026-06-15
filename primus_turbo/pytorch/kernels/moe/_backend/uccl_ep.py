###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""UCCL-EP backend (optional)."""

from typing import Optional

from primus_turbo.common.uccl_utils import get_uccl

from .base import _DeepEPLikeBackend, apply_uccl_network_env, call_once


class UCCLEPBackend(_DeepEPLikeBackend):
    """UCCL-EP backend (optional)."""

    @staticmethod
    def is_available() -> bool:
        # Check the API surface, not just the import (some UCCL builds lack ``Buffer``).
        try:
            get_uccl()
            import uccl.ep  # noqa: F401
        except ImportError:
            return False
        return hasattr(uccl.ep, "Buffer")

    @classmethod
    def can_release(cls, *, will_reinit: bool) -> bool:  # noqa: ARG003
        # destroy() corrupts the HIP context: never release.
        return False

    @staticmethod
    def _get_module():
        get_uccl()
        import uccl.ep as uccl_ep

        return uccl_ep

    @call_once
    def setup_env(self, **overrides: Optional[str]) -> None:
        apply_uccl_network_env(**overrides)
