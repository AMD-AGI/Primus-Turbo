###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""In-tree Primus-Turbo DeepEP backend."""

import torch.distributed as dist

from .base import _DeepEPLikeBackend, detect_group_topology


class TurboEPBackend(_DeepEPLikeBackend):
    """In-tree Primus-Turbo DeepEP backend (always available)."""

    @staticmethod
    def is_available() -> bool:
        return True

    @classmethod
    def is_feasible(cls, group: dist.ProcessGroup) -> bool:
        # rocSHMEM internode is unsupported on AINIC (SIGABRTs); intra-node only.
        _, num_nodes = detect_group_topology(group)
        return num_nodes <= 1

    @staticmethod
    def _get_module():
        import primus_turbo.pytorch.deep_ep as turbo_ep

        return turbo_ep
