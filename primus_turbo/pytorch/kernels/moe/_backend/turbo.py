###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""In-tree Primus-Turbo DeepEP backend."""

from .base import _DeepEPLikeBackend


class TurboEPBackend(_DeepEPLikeBackend):
    """In-tree Primus-Turbo DeepEP backend (always available)."""

    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def _get_module():
        import primus_turbo.pytorch.deep_ep as turbo_ep

        return turbo_ep
