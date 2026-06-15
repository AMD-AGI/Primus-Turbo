###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""External ``deep_ep`` package backend (optional)."""

from typing import Optional

from .base import _DeepEPLikeBackend, apply_uccl_network_env, call_once


class DeepEPBackend(_DeepEPLikeBackend):
    """External ``deep_ep`` package backend (optional)."""

    @staticmethod
    def is_available() -> bool:
        try:
            import deep_ep  # noqa: F401

            return True
        except ImportError:
            return False

    @staticmethod
    def _get_module():
        import deep_ep

        return deep_ep

    @classmethod
    def _is_uccl_backed(cls) -> bool:
        # uccl wrapper re-exports Config from uccl.ep; on import error assume uccl (never free).
        try:
            config = getattr(cls._get_module(), "Config", None)
        except Exception:  # noqa: BLE001 - import quirks must not break capability checks
            return True
        return getattr(config, "__module__", "").startswith("uccl")

    @classmethod
    def can_release(cls, *, will_reinit: bool) -> bool:  # noqa: ARG003
        # uccl destroy() corrupts the HIP context: never release.
        return not cls._is_uccl_backed()

    @call_once
    def setup_env(self, **overrides: Optional[str]) -> None:
        # The uccl-backed wrapper needs the UCCL_* RDMA env like the UCCL backend.
        if self._is_uccl_backed():
            apply_uccl_network_env(**overrides)
