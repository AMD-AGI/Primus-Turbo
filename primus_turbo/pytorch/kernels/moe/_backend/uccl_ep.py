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
    """UCCL-EP backend via uccl's deep_ep-compatible wrapper ``Buffer``."""

    @staticmethod
    def is_available() -> bool:
        # Needs uccl and its deep_ep wrapper (raw uccl.ep lacks the high-level API).
        try:
            import deep_ep  # noqa: F401
            import uccl.ep  # noqa: F401
        except ImportError:
            return False
        # Only the uccl-backed wrapper (Config re-exported from uccl.ep) is ours.
        return getattr(getattr(deep_ep, "Config", None), "__module__", "").startswith("uccl")

    @classmethod
    def can_release(cls, *, will_reinit: bool) -> bool:  # noqa: ARG003
        # destroy() corrupts the HIP context: never release.
        return False

    @staticmethod
    def _get_module():
        get_uccl()
        import deep_ep

        return deep_ep

    @call_once
    def setup_env(self, **overrides: Optional[str]) -> None:
        apply_uccl_network_env(**overrides)
