###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""ROCm Mori EP backend (optional)."""

import dataclasses
import os
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.distributed as dist

from primus_turbo.common.constants import ENV_MORI_NUM_QP_PER_PE
from primus_turbo.common.logger import logger
from primus_turbo.pytorch.kernels.moe.moe_utils import detect_group_topology

from .base import _apply_env_with_nccl_fallback

# ==========================================================================
# Mori EP backend helpers
# ==========================================================================


class _MoriEpMode(Enum):
    """Mori EP deployment mode (normal dispatch only)."""

    INTRA_NODE = "intra_node"
    INTER_NODE = "inter_node"


@dataclass(frozen=True)
class _MoriDispatchCfg:
    """Mori kernel launch config for a given mode / token budget."""

    kernel_type: Any  # mori.ops.EpDispatchCombineKernelType
    warp_num_per_block: int
    block_num: int
    rdma_block_num: int


def _get_mori_dispatch_configs(
    num_max_dispatch_tokens_per_rank: int,
) -> Dict[_MoriEpMode, _MoriDispatchCfg]:
    """Return per-mode Mori kernel launch configs."""
    import mori.ops

    return {
        _MoriEpMode.INTRA_NODE: _MoriDispatchCfg(
            kernel_type=mori.ops.EpDispatchCombineKernelType.IntraNode,
            warp_num_per_block=16,
            block_num=80,
            rdma_block_num=0,
        ),
        _MoriEpMode.INTER_NODE: _MoriDispatchCfg(
            kernel_type=mori.ops.EpDispatchCombineKernelType.InterNodeV1,
            warp_num_per_block=8,
            block_num=64,
            rdma_block_num=32,
        ),
    }


_MORI_SHMEM_PG_NAME = "mori"


def _register_and_init_mori_shmem(group: dist.ProcessGroup) -> None:
    """Register ``group`` with the Mori SHMEM runtime."""
    import mori.shmem

    assert dist.is_initialized(), "torch.distributed must be initialized before Mori SHMEM init."

    try:
        torch._C._distributed_c10d._register_process_group(_MORI_SHMEM_PG_NAME, group)
    except Exception as e:  # noqa: BLE001 - mori binds raise a mix of exception types.
        if "already registered" in str(e):
            logger.info(
                f"[MORI init] Process group already registered under "
                f"'{_MORI_SHMEM_PG_NAME}'; reusing existing SHMEM binding "
                f"({e}).",
                rank=0,
            )
        else:
            raise
    else:
        mori.shmem.shmem_torch_process_group_init(_MORI_SHMEM_PG_NAME)


def _resolve_mori_dispatch_cfg(
    group: dist.ProcessGroup,
    num_max_dispatch_tokens_per_rank: int,
    config: Optional["EPBufferConfig"] = None,
) -> _MoriDispatchCfg:
    """Resolve Mori kernel launch cfg: topology default, overridden by
    ``config.num_sms`` (``block_num``) and ``config.dispatch_config`` (full).
    """
    _, num_nodes = detect_group_topology(group)
    mode = _MoriEpMode.INTRA_NODE if num_nodes <= 1 else _MoriEpMode.INTER_NODE
    base = _get_mori_dispatch_configs(num_max_dispatch_tokens_per_rank)[mode]

    kernel_type = base.kernel_type
    warp_num_per_block = base.warp_num_per_block
    block_num = base.block_num
    rdma_block_num = base.rdma_block_num

    num_sms_override = config.num_sms if config is not None else 0
    full_override = (
        config.dispatch_config
        if config is not None and isinstance(config.dispatch_config, _MoriDispatchCfg)
        else None
    )

    if num_sms_override > 0:
        block_num = int(num_sms_override)

    if full_override is not None:
        kernel_type = full_override.kernel_type
        warp_num_per_block = full_override.warp_num_per_block
        block_num = full_override.block_num
        rdma_block_num = full_override.rdma_block_num

    return _MoriDispatchCfg(
        kernel_type=kernel_type,
        warp_num_per_block=warp_num_per_block,
        block_num=block_num,
        rdma_block_num=rdma_block_num,
    )


@lru_cache(maxsize=8)
def _build_mori_op(
    rank: int,
    world_size: int,
    hidden: int,
    params_dtype: torch.dtype,
    num_local_experts: int,
    num_topk: int,
    num_max_dispatch_tokens_per_rank: int,
    kernel_type: Any,
    warp_num_per_block: int,
    block_num: int,
    rdma_block_num: int,
):
    """Build and cache a :class:`mori.ops.EpDispatchCombineOp`."""
    import mori.ops

    num_qp_per_pe = int(os.environ.get(ENV_MORI_NUM_QP_PER_PE, "2"))

    common_kwargs = dict(
        data_type=params_dtype,
        rank=rank,
        world_size=world_size,
        hidden_dim=hidden,
        scale_dim=0,
        scale_type_size=0,
        max_token_type_size=max(2, params_dtype.itemsize),
        max_num_inp_token_per_rank=num_max_dispatch_tokens_per_rank,
        num_experts_per_rank=num_local_experts,
        num_experts_per_token=num_topk,
        warp_num_per_block=warp_num_per_block,
        block_num=block_num,
        max_total_recv_tokens=0,
        use_external_inp_buf=True,
        kernel_type=kernel_type,
        gpu_per_node=torch.cuda.device_count(),
        rdma_block_num=rdma_block_num,
        num_qp_per_pe=num_qp_per_pe,
        quant_type="none",
    )

    def _check_mori_compatibility(kwargs: dict) -> dict:
        """Drop kwargs not accepted by the installed Mori config."""
        valid = {f.name for f in dataclasses.fields(mori.ops.EpDispatchCombineConfig)}
        cleaned = dict(kwargs)
        for key in list(cleaned.keys()):
            if key not in valid:
                logger.warning(
                    f"[MORI compat] Dropping incompatible EpDispatchCombineConfig " f"argument '{key}'.",
                    once=True,
                )
                del cleaned[key]
        return cleaned

    common_kwargs = _check_mori_compatibility(common_kwargs)

    logger.info(
        f"[MORI init] world_size={world_size} rank={rank} hidden={hidden} "
        f"dtype={params_dtype} max_inp_tokens={num_max_dispatch_tokens_per_rank} "
        f"num_local_experts={num_local_experts} num_topk={num_topk} "
        f"kernel={kernel_type} block_num={block_num} "
        f"warp_num_per_block={warp_num_per_block} rdma_block_num={rdma_block_num}",
        rank=0,
    )

    config = mori.ops.EpDispatchCombineConfig(**common_kwargs)
    return mori.ops.EpDispatchCombineOp(config)


class MoriEPBackend:
    """ROCm Mori EP backend for MoE dispatch/combine (normal mode)."""

    def __init__(self) -> None:
        self._group: Optional[dist.ProcessGroup] = None
        self._op = None
        self._hidden_size: int = 0
        self._num_local_experts: int = 0
        self._num_topk: int = 0
        self._seqlen: int = 0
        self._fp8_dispatch: bool = False
        self._params_dtype: Optional[torch.dtype] = None
        # Last resolved kernel cfg; used to detect cfg-only rebuilds.
        self._kernel_cfg: Optional[_MoriDispatchCfg] = None

    # ------------------------------------------------------------------
    # EPBackend protocol
    # ------------------------------------------------------------------

    def is_initialized(self) -> bool:
        """Return True if the backend is initialized."""
        return self._op is not None

    @staticmethod
    def is_available() -> bool:
        try:
            import mori  # noqa: F401
            import mori.ops  # noqa: F401
            import mori.shmem  # noqa: F401

            return True
        except ImportError:
            return False

    @staticmethod
    def _derive_params_dtype(fp8_dispatch: bool) -> torch.dtype:
        """Pick the Mori ``data_type`` argument from ``fp8_dispatch``."""
        assert not fp8_dispatch, "Not implemented"
        return torch.bfloat16

    def setup_env(
        self,
        *,
        ib_gid_index: Optional[str] = None,
        ib_hca: Optional[str] = None,
        socket_ifname: Optional[str] = None,
        ib_tc: Optional[str] = None,
        ib_sl: Optional[str] = None,
    ) -> None:
        """Populate MORI_* RDMA env vars, falling back to NCCL_* when unset.

        Mapping (MORI doc → NCCL equivalent):

        - ``MORI_IB_GID_INDEX``  ↔ ``NCCL_IB_GID_INDEX``
        - ``MORI_RDMA_DEVICES``  ↔ ``NCCL_IB_HCA`` (both accept ``mlx5_0,mlx5_1``
          include lists and ``^mlx5_2`` exclude lists)
        - ``MORI_SOCKET_IFNAME`` ↔ ``NCCL_SOCKET_IFNAME``
        - ``MORI_RDMA_TC``       ↔ ``NCCL_IB_TC``
        - ``MORI_RDMA_SL``       ↔ ``NCCL_IB_SL``

        Args:
            ib_gid_index: Override for ``MORI_IB_GID_INDEX``.
            ib_hca: Override for ``MORI_RDMA_DEVICES``.
            socket_ifname: Override for ``MORI_SOCKET_IFNAME``.
            ib_tc: Override for ``MORI_RDMA_TC``.
            ib_sl: Override for ``MORI_RDMA_SL``.

        Any argument left ``None`` inherits from the corresponding ``NCCL_*``
        env var when it is set; otherwise the variable stays unset and Mori
        applies its own default (typically auto-detect).
        """
        _apply_env_with_nccl_fallback(
            [
                ("MORI_IB_GID_INDEX", "NCCL_IB_GID_INDEX", ib_gid_index),
                ("MORI_RDMA_DEVICES", "NCCL_IB_HCA", ib_hca),
                ("MORI_SOCKET_IFNAME", "NCCL_SOCKET_IFNAME", socket_ifname),
                ("MORI_RDMA_TC", "NCCL_IB_TC", ib_tc),
                ("MORI_RDMA_SL", "NCCL_IB_SL", ib_sl),
            ]
        )

    def init_buffer(
        self,
        group: dist.ProcessGroup,
        hidden_size: int,
        num_experts: int,
        num_topk: int,
        seqlen: int,
        fp8_dispatch: bool,
        config: "EPBufferConfig",
    ) -> None:
        """Register Mori SHMEM and build/rebuild the Mori op as needed.

        Rebuilds when any shape signature field or kernel cfg changes. Reads
        ``config.num_sms`` as a ``block_num`` override and
        ``config.dispatch_config`` (if a :class:`_MoriDispatchCfg`) as a full
        kernel cfg override; see :func:`_resolve_mori_dispatch_cfg`.
        """
        assert not fp8_dispatch, "not implemented"

        # Best-effort NCCL-fallback for Mori RDMA env vars. Must run before
        # SHMEM init so Mori picks up the right network configuration.
        self.setup_env()

        if self._group is not group:
            self._group = group
            _register_and_init_mori_shmem(group)
            logger.info("[MoriEPBackend init] Mori SHMEM initialized.", rank=0)

        num_local_experts = num_experts // group.size()
        params_dtype = self._derive_params_dtype(fp8_dispatch)
        kernel_cfg = _resolve_mori_dispatch_cfg(group, seqlen, config)

        # Rebuild on shape-signature or kernel-cfg change (_build_mori_op is cached).
        needs_rebuild = (
            self._op is None
            or self._hidden_size != hidden_size
            or self._num_local_experts != num_local_experts
            or self._num_topk != num_topk
            or self._seqlen != seqlen
            or self._fp8_dispatch != fp8_dispatch
            or self._params_dtype != params_dtype
            or self._kernel_cfg != kernel_cfg
        )

        self._hidden_size = hidden_size
        self._num_local_experts = num_local_experts
        self._num_topk = num_topk
        self._seqlen = seqlen
        self._fp8_dispatch = fp8_dispatch
        self._params_dtype = params_dtype
        self._kernel_cfg = kernel_cfg

        if needs_rebuild:
            self._op = _build_mori_op(
                rank=group.rank(),
                world_size=group.size(),
                hidden=hidden_size,
                params_dtype=params_dtype,
                num_local_experts=num_local_experts,
                num_topk=num_topk,
                num_max_dispatch_tokens_per_rank=seqlen,
                kernel_type=kernel_cfg.kernel_type,
                warp_num_per_block=kernel_cfg.warp_num_per_block,
                block_num=kernel_cfg.block_num,
                rdma_block_num=kernel_cfg.rdma_block_num,
            )

    def dispatch(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        handle: Optional[tuple] = None,
        topk_idx: Optional[torch.Tensor] = None,
        token_weights: Optional[torch.Tensor] = None,
        num_experts: Optional[int] = None,
        async_finish: bool = False,  # noqa: ARG002 - API compat only.
        allocate_on_comm_stream: bool = False,  # noqa: ARG002 - API compat only.
        num_worst_tokens: int = 0,  # noqa: ARG002 - API compat only.
        config: Optional[Any] = None,  # noqa: ARG002 - API compat only.
    ):
        from primus_turbo.pytorch.kernels.moe.moe_dispatch_combine_impl import (
            compute_expert_token_info,
        )

        assert self.is_initialized(), "Backend is not initialized"
        scale = None
        non_blocking = num_worst_tokens > 0

        if handle is None:
            assert topk_idx is not None
            topk_idx_i32 = topk_idx.to(torch.int32)
        else:
            assert topk_idx is None, token_weights is None
            (topk_idx_i32, token_weights, _) = handle

        recv_x, recv_topk_weights, recv_x_scales, recv_topk_idx, total_recv = self._op.dispatch(
            x,
            token_weights,
            scale,
            topk_idx_i32,
        )
        expert_base = self._group.rank() * self._num_local_experts
        num_recv_tokens_per_expert, deepep_like_recv_topk_idx = compute_expert_token_info(
            recv_topk_idx,
            self._num_local_experts,
            expert_base=expert_base,
            total_recv=total_recv,
        )
        if not non_blocking:
            num_recv_tokens_per_expert = num_recv_tokens_per_expert.tolist()

        # hold token_weights for dispatch weights in backward
        # it's a workaround to aviod illegal access when token_weights is None
        handle = (topk_idx_i32, token_weights, recv_topk_idx)
        return (recv_x, deepep_like_recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert, handle)

    def combine(
        self,
        x: torch.Tensor,
        handle: tuple,
        topk_weights: Optional[torch.Tensor] = None,
        async_finish: bool = False,  # noqa: ARG002 - API compat only.
        allocate_on_comm_stream: bool = False,  # noqa: ARG002 - API compat only.
        config: Optional[Any] = None,  # noqa: ARG002 - API compat only.
    ):
        assert self.is_initialized(), "Backend is not initialized"

        (_, _, recv_topk_idx) = handle
        combined_x, combined_topk_weights = self._op.combine(
            x.contiguous(),
            topk_weights,
            recv_topk_idx,
        )
        return combined_x, combined_topk_weights

    @torch.no_grad()
    def tune_configs(
        self,
        group: dist.ProcessGroup,  # noqa: ARG002 - stub.
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],  # noqa: ARG002
        num_experts: int,  # noqa: ARG002
        *,
        topk_idx: Optional[torch.Tensor] = None,  # noqa: ARG002
        topk_weights: Optional[torch.Tensor] = None,  # noqa: ARG002
        num_sms: int = 32,  # noqa: ARG002
        num_tests: int = 20,  # noqa: ARG002
        num_topk: Optional[int] = None,  # noqa: ARG002
        uniform_dispatch: bool = True,  # noqa: ARG002
    ) -> Tuple[Any, Any, float, float]:
        raise NotImplementedError("tune_configs is not implemented for MoriEPBackend")

    def release_buffer(self) -> None:
        """Drop the fast-path op reference and SHMEM binding state."""
        self._op = None
        self._hidden_size = 0
        self._num_local_experts = 0
        self._num_topk = 0
        self._seqlen = 0
        self._fp8_dispatch = False
        self._params_dtype = None
        self._kernel_cfg = None
