###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""External ``deep_ep`` package backend (optional)."""

import inspect

import torch.distributed as dist

from .base import _DeepEPLikeBackend


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

    def _make_buffer_kwargs(self, group: dist.ProcessGroup) -> dict:
        BufferClass = self._get_module().Buffer
        try:
            param = inspect.signature(BufferClass).parameters.get("is_intranode")
        except (TypeError, ValueError):
            param = None
        if param is not None and param.default is False:
            return {"is_intranode": group.size() <= 8}
        return {}
