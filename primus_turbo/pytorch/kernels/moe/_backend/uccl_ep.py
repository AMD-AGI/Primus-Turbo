###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""UCCL-EP backend (optional)."""

import inspect
from typing import Optional

import torch.distributed as dist

from .base import _apply_env_with_nccl_fallback, _DeepEPLikeBackend


class UCCLEPBackend(_DeepEPLikeBackend):
    """UCCL-EP backend (optional)."""

    @staticmethod
    def is_available() -> bool:
        try:
            import uccl.ep  # noqa: F401

            return True
        except ImportError:
            return False

    @staticmethod
    def _get_module():
        import uccl.ep as uccl_ep

        return uccl_ep

    def _make_buffer_kwargs(self, group: dist.ProcessGroup) -> dict:
        BufferClass = self._get_module().Buffer
        try:
            param = inspect.signature(BufferClass).parameters.get("is_intranode")
        except (TypeError, ValueError):
            param = None
        if param is not None and param.default is False:
            return {"is_intranode": group.size() <= 8}
        return {}

    def setup_env(
        self,
        *,
        ib_gid_index: Optional[str] = None,
        ib_hca: Optional[str] = None,
        socket_ifname: Optional[str] = None,
        ib_tc: Optional[str] = None,
        ib_sl: Optional[str] = None,
    ) -> None:
        """Populate UCCL_* RDMA env vars, falling back to NCCL_* when unset.

        Args:
            ib_gid_index: Override for ``UCCL_IB_GID_INDEX``.
            ib_hca: Override for ``UCCL_IB_HCA``.
            socket_ifname: Override for ``UCCL_SOCKET_IFNAME``.
            ib_tc: Override for ``UCCL_IB_TC``.
            ib_sl: Override for ``UCCL_IB_SL``.

        Any argument left ``None`` inherits from the corresponding ``NCCL_*``
        env var when it is set; otherwise the variable stays unset and UCCL
        applies its own default.
        """
        _apply_env_with_nccl_fallback(
            [
                ("UCCL_IB_GID_INDEX", "NCCL_IB_GID_INDEX", ib_gid_index),
                ("UCCL_IB_HCA", "NCCL_IB_HCA", ib_hca),
                ("UCCL_SOCKET_IFNAME", "NCCL_SOCKET_IFNAME", socket_ifname),
                ("UCCL_IB_TC", "NCCL_IB_TC", ib_tc),
                ("UCCL_IB_SL", "NCCL_IB_SL", ib_sl),
            ]
        )
