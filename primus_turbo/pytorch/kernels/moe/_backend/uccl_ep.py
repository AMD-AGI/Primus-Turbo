###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""UCCL-EP backend (optional)."""

import inspect
import os
from typing import Dict, Optional

import torch.distributed as dist

from primus_turbo.common.logger import logger

from .base import _apply_env_with_nccl_fallback, _DeepEPLikeBackend, call_once
from .nic_detect import detect_nic_type

# Per-NIC inflight-env defaults; avoid hangs on AINIC (ionic_*) and Thor-2 (bnxt_re_*).
_UCCL_NIC_INFLIGHT_DEFAULTS: Dict[str, Dict[str, str]] = {
    "ionic": {
        "UCCL_IB_MAX_INFLIGHT_NORMAL": "1",
        "UCCL_IB_MAX_INFLIGHT_LOW_LATENCY": "1",
    },
    "bnxt": {
        "UCCL_IB_MAX_INFLIGHT_NORMAL": "1",
        "UCCL_IB_MAX_INFLIGHT_LOW_LATENCY": "1",
    },
}


class UCCLEPBackend(_DeepEPLikeBackend):
    """UCCL-EP backend (optional)."""

    @staticmethod
    def is_available() -> bool:
        # Check the API surface, not just the import (some UCCL builds lack ``Buffer``).
        try:
            import uccl.ep  # noqa: F401
        except ImportError:
            return False
        return hasattr(uccl.ep, "Buffer")

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

    @call_once
    def setup_env(
        self,
        *,
        ib_gid_index: Optional[str] = None,
        ib_hca: Optional[str] = None,
        socket_ifname: Optional[str] = None,
        ib_tc: Optional[str] = None,
        ib_sl: Optional[str] = None,
        ib_max_inflight_normal: Optional[str] = None,
        ib_max_inflight_low_latency: Optional[str] = None,
        ib_max_inflight_bytes: Optional[str] = None,
    ) -> None:
        """Set UCCL_* RDMA env vars, falling back to ``NCCL_*`` equivalents.
        ``None`` kwargs inherit from ``NCCL_*`` if set, else UCCL defaults.
        """
        _apply_env_with_nccl_fallback(
            [
                ("UCCL_IB_GID_INDEX", "NCCL_IB_GID_INDEX", ib_gid_index),
                ("UCCL_IB_HCA", "NCCL_IB_HCA", ib_hca),
                ("UCCL_SOCKET_IFNAME", "NCCL_SOCKET_IFNAME", socket_ifname),
                ("UCCL_IB_TC", "NCCL_IB_TC", ib_tc),
                ("UCCL_IB_SL", "NCCL_IB_SL", ib_sl),
            ],
            backend_name="UCCL",
        )

        # NIC-specific defaults for inflight envs (run after HCA is resolved).
        nic_defaults = _UCCL_NIC_INFLIGHT_DEFAULTS.get(detect_nic_type() or "", {})
        inflight_envs = [
            ("UCCL_IB_MAX_INFLIGHT_NORMAL", ib_max_inflight_normal),
            ("UCCL_IB_MAX_INFLIGHT_LOW_LATENCY", ib_max_inflight_low_latency),
            ("UCCL_IB_MAX_INFLIGHT_BYTES", ib_max_inflight_bytes),
        ]
        for env_name, explicit in inflight_envs:
            if explicit is not None:
                os.environ[env_name] = str(explicit)
            elif env_name not in os.environ and env_name in nic_defaults:
                os.environ[env_name] = nic_defaults[env_name]
            if env_name in os.environ:
                logger.info(
                    f"[UCCL Network Settings] {env_name}: {os.environ[env_name]}",
                    rank=0,
                )
