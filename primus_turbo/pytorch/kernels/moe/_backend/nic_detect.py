###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
from __future__ import annotations

import os
from typing import List, Optional, Tuple

__all__ = ["detect_nic_type"]

# Ordered ``(device_name_prefix, canonical_family)`` map. First match wins,
# so put more specific prefixes first if a generic one could shadow them.
_NIC_NAME_PREFIXES: Tuple[Tuple[str, str], ...] = (
    ("ionic", "ionic"),  # AMD Pollara AINIC
    ("bnxt_re", "bnxt"),  # Broadcom Thor / Thor-2
    ("mlx5", "mlx5"),  # Mellanox ConnectX
)

_SYSFS_IB_PATH = "/sys/class/infiniband"


def _classify(name: str) -> Optional[str]:
    """Map an IB device name to its canonical NIC family, or ``None``."""
    for prefix, family in _NIC_NAME_PREFIXES:
        if name.startswith(prefix):
            return family
    return None


def _read_hca_env() -> List[str]:
    """Parse ``UCCL_IB_HCA`` / ``NCCL_IB_HCA`` into bare device-name tokens."""
    value = (
        os.environ.get("NCCL_IB_HCA")
        or os.environ.get("UCCL_IB_HCA")
        or os.environ.get("MORI_RDMA_DEVICES")
        or ""
    )
    seen: set = set()
    tokens: List[str] = []
    for raw in value.split(","):
        tok = raw.strip().lstrip("^").split(":", 1)[0]
        if tok and tok not in seen:
            seen.add(tok)
            tokens.append(tok)
    return tokens


def _list_local_ib_devices() -> List[str]:
    """Return sorted IB device names from sysfs, or ``[]`` if unreadable."""
    try:
        return sorted(os.listdir(_SYSFS_IB_PATH))
    except OSError:
        return []


def detect_nic_type() -> Optional[str]:
    """Return canonical NIC family (``"ionic"`` / ``"bnxt"`` / ``"mlx5"``) or ``None``.

    Resolution order:
        1. Enumerate ``/sys/class/infiniband``; if the HCA env is set, prefer
           devices it matches (NCCL substring rule). Fall through to the
           unfiltered list when nothing matches so we still classify
           *something* on mixed-NIC hosts.
        2. If sysfs is empty/unreadable, classify the env tokens directly.
    """
    env_tokens = _read_hca_env()
    local = _list_local_ib_devices()

    if local and env_tokens:
        preferred = [n for n in local if any(t in n for t in env_tokens)]
        candidates = preferred or local
    else:
        candidates = local or env_tokens

    for name in candidates:
        family = _classify(name)
        if family is not None:
            return family
    return None
