###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""ROCm Mori EP backend (optional)."""

import dataclasses
import gc
import os
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import triton
import triton.language as tl

from primus_turbo.common.constants import ENV_MORI_NUM_QP_PER_PE
from primus_turbo.common.logger import logger

from ._config import EPBufferConfig
from .base import (
    _apply_env_with_nccl_fallback,
    _broadcast_from_rank0_float,
    _broadcast_from_rank0_int,
    call_once,
    detect_group_topology,
    sweep_configs,
)

# ==========================================================================
# Per-expert token counting (Mori dispatch post-processing)
# ==========================================================================


def _compute_expert_token_info_configs() -> List[triton.Config]:
    """Autotune space for :func:`compute_expert_token_info_kernel`."""
    configs: List[triton.Config] = []
    for block_m in (64, 128, 256, 512, 1024):
        for num_warps in (1, 2, 4, 8):
            # Wave=64, cap threads/CTA to 1024 (HW limit on CDNA).
            if num_warps * 64 > 1024:
                continue
            # Keep at least one element per thread in the M dimension.
            if block_m < num_warps * 64:
                continue
            for num_stages in (1, 2):
                configs.append(
                    triton.Config(
                        {"BLOCK_SIZE_M": block_m},
                        num_warps=num_warps,
                        num_stages=num_stages,
                    )
                )
    return configs


@triton.autotune(
    configs=_compute_expert_token_info_configs(),
    key=["num_tokens", "num_topk"],
    reset_to_zero=["num_recv_tokens_per_expert_ptr"],
)
@triton.jit
def compute_expert_token_info_kernel(
    recv_topk_idx_ptr,
    deepep_topk_idx_ptr,
    num_recv_tokens_per_expert_ptr,
    total_recv_ptr,
    num_tokens: tl.int32,
    num_topk: tl.int32,
    num_local_experts: tl.int32,
    expert_base: tl.int32,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    HAS_TOTAL_RECV: tl.constexpr,
):
    """Count per-expert received tokens and emit a DeepEP-format topk_idx."""
    pid = tl.program_id(0)
    row_offs = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    col_offs = tl.arange(0, BLOCK_SIZE_K)
    bin_offs = tl.arange(0, BLOCK_SIZE_N)

    if HAS_TOTAL_RECV:
        total_recv = tl.load(total_recv_ptr)
        row_limit = tl.minimum(total_recv, num_tokens)
    else:
        row_limit = num_tokens

    in_footprint = row_offs < num_tokens
    in_valid_rows = row_offs < row_limit
    col_mask = col_offs < num_topk

    footprint_mask = in_footprint[:, None] & col_mask[None, :]
    load_mask = in_valid_rows[:, None] & col_mask[None, :]

    safe_row = tl.where(in_footprint, row_offs, 0)
    expert_offs = safe_row[:, None] * num_topk + col_offs[None, :]

    topk_experts = tl.load(
        recv_topk_idx_ptr + expert_offs,
        mask=load_mask,
        other=0,
    )

    local_experts = topk_experts - expert_base
    valid = load_mask & (local_experts >= 0) & (local_experts < num_local_experts)
    # Clamp invalid lanes to a legal bin; the histogram mask excludes them.
    safe_experts = tl.where(valid, local_experts, 0)

    # ``tl.histogram`` requires a flat 1D input; reshape the 2D tile.
    flat_experts = tl.reshape(safe_experts, [BLOCK_SIZE_M * BLOCK_SIZE_K])
    flat_mask = tl.reshape(valid, [BLOCK_SIZE_M * BLOCK_SIZE_K])
    local_counts = tl.histogram(flat_experts, BLOCK_SIZE_N, mask=flat_mask)

    # Flush CTA-local accumulator to global memory with one atomic per bin.
    bin_mask = (bin_offs < num_local_experts) & (local_counts > 0)
    tl.atomic_add(
        num_recv_tokens_per_expert_ptr + bin_offs,
        local_counts,
        sem="relaxed",
        scope="gpu",
        mask=bin_mask,
    )

    deepep_experts = tl.where(valid, local_experts, -1).to(topk_experts.dtype)
    tl.store(deepep_topk_idx_ptr + expert_offs, deepep_experts, mask=footprint_mask)


def compute_expert_token_info(
    recv_topk_idx: torch.Tensor,
    num_local_experts: int,
    *,
    expert_base: int = 0,
    total_recv: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Count per-expert received tokens and emit ``recv_topk_idx`` in DeepEP
    layout (as a new tensor).
    """
    assert recv_topk_idx.is_cuda, "recv_topk_idx must be a CUDA tensor"
    assert recv_topk_idx.dim() == 2, "recv_topk_idx must be 2D [num_tokens, num_topk]"

    device = recv_topk_idx.device
    num_tokens, num_topk = recv_topk_idx.shape

    num_recv_tokens_per_expert = torch.zeros(num_local_experts, dtype=recv_topk_idx.dtype, device=device)
    deepep_like_topk_idx = torch.empty_like(recv_topk_idx)

    if num_tokens == 0 or num_topk == 0 or num_local_experts == 0:
        # Nothing to count; fill with the DeepEP padding sentinel.
        if deepep_like_topk_idx.numel() > 0:
            deepep_like_topk_idx.fill_(-1)
        return num_recv_tokens_per_expert, deepep_like_topk_idx

    has_total_recv = total_recv is not None
    if has_total_recv:
        assert total_recv.is_cuda, "total_recv must be a CUDA tensor"
        assert total_recv.numel() >= 1, "total_recv must have at least 1 element"
        # Triton expects an int32 pointer; coerce if needed.
        if total_recv.dtype != torch.int32:
            total_recv = total_recv.to(torch.int32)
        total_recv_ptr = total_recv
    else:
        # Dummy pointer; unused when HAS_TOTAL_RECV=False.
        total_recv_ptr = recv_topk_idx

    # ``tl.histogram`` requires ``num_bins`` to be a power of 2 on AMD Triton.
    block_size_n = triton.next_power_of_2(num_local_experts)

    # ``BLOCK_SIZE_M`` is picked by the autotuner; the grid size depends on it.
    def grid(META):
        return (triton.cdiv(num_tokens, META["BLOCK_SIZE_M"]),)  # noqa: E731

    compute_expert_token_info_kernel[grid](
        recv_topk_idx,
        deepep_like_topk_idx,
        num_recv_tokens_per_expert,
        total_recv_ptr,
        num_tokens,
        num_topk,
        num_local_experts,
        expert_base,
        BLOCK_SIZE_K=triton.next_power_of_2(num_topk),
        BLOCK_SIZE_N=block_size_n,
        HAS_TOTAL_RECV=has_total_recv,
    )
    return num_recv_tokens_per_expert, deepep_like_topk_idx


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


def _get_mori_dispatch_configs() -> Dict[_MoriEpMode, _MoriDispatchCfg]:
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


def _align_mori_hip_with_torch() -> None:
    """Make Mori's JIT reuse torch's ``libamdhip64.so``.

    On ROCm 7.2+ Mori's separate ``dlopen`` of system HIP gets a different
    module table than torch's, causing HIP error 500 in ``hipModuleGetFunction``.
    Preloading torch's copy into Mori's lazy handle fixes it.
    """
    try:
        import ctypes

        import mori.jit.hip_driver as _hd
    except ImportError:
        return

    torch_hip = os.path.join(os.path.dirname(torch.__file__), "lib", "libamdhip64.so")
    if not os.path.isfile(torch_hip):
        return

    try:
        _hd._hip = ctypes.CDLL(torch_hip)
        logger.info(f"[MORI init] Aligned Mori HIP with torch ({torch_hip}).", rank=0, once=True)
    except OSError as e:  # pragma: no cover - best-effort.
        logger.warning(f"[MORI init] Failed to preload {torch_hip}: {e}.", once=True)


def _register_and_init_mori_shmem(group: dist.ProcessGroup) -> None:
    """Register ``group`` with the Mori SHMEM runtime (idempotent)."""
    import mori.shmem

    assert dist.is_initialized(), "torch.distributed must be initialized before Mori SHMEM init."

    try:
        torch._C._distributed_c10d._register_process_group(_MORI_SHMEM_PG_NAME, group)
    except Exception as e:  # noqa: BLE001 - mori binds raise a mix of exception types.
        if "already registered" not in str(e):
            raise
        logger.info(
            f"[MORI init] Process group already registered under "
            f"'{_MORI_SHMEM_PG_NAME}'; reusing existing SHMEM binding "
            f"({e}).",
            rank=0,
        )
        return
    mori.shmem.shmem_torch_process_group_init(_MORI_SHMEM_PG_NAME)


def _extract_mori_launch_override(
    config: Optional[Any],
) -> Tuple[int, int, int]:
    """Return ``(block_num, warp_per_block, rdma_block_num)`` from ``config``.

    Mori's ``op.dispatch``/``op.combine`` accept ``-1`` as a "use config
    default" sentinel, so unknown / missing overrides degrade to that.
    """
    if isinstance(config, _MoriDispatchCfg):
        return config.block_num, config.warp_num_per_block, config.rdma_block_num
    return -1, -1, -1


def _resolve_mori_dispatch_cfg(
    group: dist.ProcessGroup,
    config: Optional[EPBufferConfig] = None,
) -> _MoriDispatchCfg:
    """Resolve Mori kernel launch cfg: topology default, overridden by
    ``config.num_sms`` (``block_num``) and ``config.dispatch_config`` (full).
    """
    _, num_nodes = detect_group_topology(group)
    mode = _MoriEpMode.INTRA_NODE if num_nodes <= 1 else _MoriEpMode.INTER_NODE
    base = _get_mori_dispatch_configs()[mode]

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
    def supports_cuda_graph() -> bool:
        """not supported"""
        return False

    @staticmethod
    def _derive_params_dtype(fp8_dispatch: bool) -> torch.dtype:
        """Pick the Mori ``data_type`` argument from ``fp8_dispatch``."""
        assert not fp8_dispatch, "Not implemented"
        return torch.bfloat16

    @call_once
    def setup_env(
        self,
        *,
        ib_gid_index: Optional[str] = None,
        ib_hca: Optional[str] = None,
        socket_ifname: Optional[str] = None,
        ib_tc: Optional[str] = None,
        ib_sl: Optional[str] = None,
    ) -> None:
        """Set MORI_* RDMA env vars, falling back to NCCL_* equivalents.

        Mapping: ``MORI_IB_GID_INDEX``/``NCCL_IB_GID_INDEX``,
        ``MORI_RDMA_DEVICES``/``NCCL_IB_HCA``, ``MORI_SOCKET_IFNAME``/
        ``NCCL_SOCKET_IFNAME``, ``MORI_RDMA_TC``/``NCCL_IB_TC``,
        ``MORI_RDMA_SL``/``NCCL_IB_SL``. Each kwarg overrides the
        corresponding MORI var; ``None`` keeps it unset (Mori auto-detects).
        """
        _apply_env_with_nccl_fallback(
            [
                ("MORI_IB_GID_INDEX", "NCCL_IB_GID_INDEX", ib_gid_index),
                ("MORI_RDMA_DEVICES", "NCCL_IB_HCA", ib_hca),
                ("MORI_SOCKET_IFNAME", "NCCL_SOCKET_IFNAME", socket_ifname),
                ("MORI_RDMA_TC", "NCCL_IB_TC", ib_tc),
                ("MORI_RDMA_SL", "NCCL_IB_SL", ib_sl),
            ],
            backend_name="MORI",
        )

    def init_buffer(
        self,
        group: dist.ProcessGroup,
        hidden_size: int,
        num_experts: int,
        num_topk: int,
        seqlen: int,
        fp8_dispatch: bool,
        config: EPBufferConfig,
    ) -> None:
        """Register Mori SHMEM and build/rebuild the Mori op as needed.

        Rebuilds when any shape signature field or kernel cfg changes. Reads
        ``config.num_sms`` as a ``block_num`` override and
        ``config.dispatch_config`` (if a :class:`_MoriDispatchCfg`) as a full
        kernel cfg override; see :func:`_resolve_mori_dispatch_cfg`.
        """
        assert not fp8_dispatch, "not implemented"

        # Must run before any Mori JIT/SHMEM activity (fixes HIP error 500 on ROCm 7.2+).
        _align_mori_hip_with_torch()

        # Apply MORI_* fallbacks from NCCL_* before SHMEM init.
        self.setup_env()

        if self._group is not group:
            self._group = group
            _register_and_init_mori_shmem(group)
            logger.info("[MoriEPBackend init] Mori SHMEM initialized.", rank=0)

        num_local_experts = num_experts // group.size()
        params_dtype = self._derive_params_dtype(fp8_dispatch)
        kernel_cfg = _resolve_mori_dispatch_cfg(group, config)

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
        num_experts: Optional[int] = None,  # noqa: ARG002 - sized via init_buffer.
        async_finish: bool = False,  # noqa: ARG002 - Mori runs sync on caller stream.
        allocate_on_comm_stream: bool = False,  # noqa: ARG002 - Mori runs sync on caller stream.
        num_worst_tokens: int = 0,
        config: Optional[Any] = None,
    ):
        assert self.is_initialized(), "Backend is not initialized"

        # num_worst_tokens > 0: graph-friendly caller; keep counts on device (no D2H sync).
        keep_tokens_per_expert_on_device = num_worst_tokens > 0

        if handle is None:
            assert topk_idx is not None
            topk_idx_i32 = topk_idx.to(torch.int32)
        else:
            assert topk_idx is None and token_weights is None
            (topk_idx_i32, token_weights, _) = handle

        # Autotuner launch override; -1 keeps the op-level default.
        block_num, warp_per_block, rdma_block_num = _extract_mori_launch_override(config)

        # ``scale`` is the FP8 scales arg; Mori path is BF16-only for now.
        scale = None
        recv_x, recv_topk_weights, _recv_x_scales, recv_topk_idx, total_recv = self._op.dispatch(
            x,
            token_weights,
            scale,
            topk_idx_i32,
            block_num=block_num,
            rdma_block_num=rdma_block_num,
            warp_per_block=warp_per_block,
        )
        expert_base = self._group.rank() * self._num_local_experts
        num_recv_tokens_per_expert, deepep_like_recv_topk_idx = compute_expert_token_info(
            recv_topk_idx,
            self._num_local_experts,
            expert_base=expert_base,
            total_recv=total_recv,
        )
        if not keep_tokens_per_expert_on_device:
            num_recv_tokens_per_expert = num_recv_tokens_per_expert.tolist()

        # Keep token_weights in the handle so backward can reach it.
        handle = (topk_idx_i32, token_weights, recv_topk_idx)
        return (recv_x, deepep_like_recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert, handle)

    def combine(
        self,
        x: torch.Tensor,
        handle: tuple,
        topk_weights: Optional[torch.Tensor] = None,
        async_finish: bool = False,  # noqa: ARG002 - API compat only.
        allocate_on_comm_stream: bool = False,  # noqa: ARG002 - API compat only.
        config: Optional[Any] = None,
    ):
        assert self.is_initialized(), "Backend is not initialized"

        (_, _, recv_topk_idx) = handle
        block_num, warp_per_block, rdma_block_num = _extract_mori_launch_override(config)
        combined_x, combined_topk_weights = self._op.combine(
            x.contiguous(),
            topk_weights,
            recv_topk_idx,
            block_num=block_num,
            rdma_block_num=rdma_block_num,
            warp_per_block=warp_per_block,
        )
        return combined_x, combined_topk_weights

    @torch.no_grad()
    def tune_configs(
        self,
        group: dist.ProcessGroup,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        num_experts: int,
        *,
        topk_idx: Optional[torch.Tensor] = None,
        topk_weights: Optional[torch.Tensor] = None,
        num_sms: int = 32,
        num_tests: int = 20,
        num_topk: Optional[int] = None,
        uniform_dispatch: bool = True,
    ) -> Tuple[_MoriDispatchCfg, _MoriDispatchCfg, float, float]:
        """Sweep Mori ``warp_per_block`` and return best per-call cfgs.

        Dispatch and combine are tuned independently over the ``warp_per_block``
        candidates; ``block_num`` is pinned to the configured ``num_sms`` (not
        swept). RDMA block count is *not* swept; the kernel type's base value is
        preserved (0 intra-node, 32 inter-node).

        Args:
            group: EP process group.
            x: Dispatch input. ``(fp8_tensor, scales)`` tuples are rejected
                because Mori's BF16-only path is the only one wired up.
            num_experts: Total expert count.
            topk_idx: ``[num_tokens, num_topk]`` indices. Required when
                ``uniform_dispatch=False``; only consulted for ``num_topk``
                in uniform mode.
            topk_weights: ``[num_tokens, num_topk]`` weights. Required when
                ``uniform_dispatch=False``.
            num_sms: Reserved for API compatibility with other backends;
                Mori sweeps ``block_num`` directly off the device's SM count.
            num_tests: Timed iterations per candidate.
            num_topk: Falls back to ``topk_idx.size(1)`` when unset.
            uniform_dispatch: If True, resample a uniform topk distribution
                so tuning is robust to caller-side routing skew.

        Returns:
            ``(dispatch_cfg, combine_cfg, dispatch_time_s, combine_time_s)``
            where the cfgs are :class:`_MoriDispatchCfg` instances suitable
            for passing as the ``config=`` argument to
            :meth:`MoriEPBackend.dispatch` / :meth:`MoriEPBackend.combine`.
        """
        if isinstance(x, tuple):
            raise NotImplementedError(
                "MoriEPBackend.tune_configs: FP8 dispatch (tuple ``x``) is not supported."
            )

        # Topology-derived kernel pinning: matches the live ``init_buffer`` path.
        _, num_nodes = detect_group_topology(group)
        mode = _MoriEpMode.INTRA_NODE if num_nodes <= 1 else _MoriEpMode.INTER_NODE
        base_cfg = _get_mori_dispatch_configs()[mode]
        kernel_type = base_cfg.kernel_type
        rdma_block_num = base_cfg.rdma_block_num

        # Inter-node mori isn't benchmark-safe (reuse NaNs, op free/rebuild hangs):
        # skip tuning and return the base config; the live path builds one fresh op.
        if num_nodes > 1:
            fixed_block_num = int(num_sms) if num_sms and num_sms > 0 else base_cfg.block_num
            base_live = _MoriDispatchCfg(
                kernel_type=kernel_type,
                warp_num_per_block=base_cfg.warp_num_per_block,
                block_num=fixed_block_num,
                rdma_block_num=rdma_block_num,
            )
            if group.rank() == 0:
                logger.info(
                    f"[Mori tuning] inter-node ({num_nodes} nodes): autotune disabled "
                    f"(mori inter-node is not benchmark-safe); using base config "
                    f"block_num={fixed_block_num} warp_per_block={base_cfg.warp_num_per_block}.",
                    rank=0,
                )
            return base_live, base_live, 0.0, 0.0

        x_tensor = x
        hidden_size = int(x_tensor.size(1))
        seqlen = int(x_tensor.size(0))
        device = x_tensor.device

        if uniform_dispatch:
            if num_topk is None:
                if topk_idx is None:
                    raise ValueError(
                        "tune_configs(uniform_dispatch=True): need either "
                        "``num_topk`` or a shape-carrying ``topk_idx`` to "
                        "sample a uniform distribution."
                    )
                num_topk = int(topk_idx.size(1))
            if num_topk > num_experts:
                raise ValueError(
                    "tune_configs(uniform_dispatch=True): ``num_topk`` "
                    f"({num_topk}) cannot exceed ``num_experts`` ({num_experts})."
                )
            num_tokens = int(x_tensor.size(0))
            scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device=device).abs() + 1
            topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)[1]
            topk_weights = torch.randn((num_tokens, num_topk), dtype=torch.float32, device=device)
        else:
            if topk_idx is None or topk_weights is None:
                raise ValueError(
                    "tune_configs(uniform_dispatch=False): ``topk_idx`` and "
                    "``topk_weights`` are both required."
                )
            if num_topk is None:
                num_topk = int(topk_idx.size(1))

        topk_idx_i32 = topk_idx.to(torch.int32)
        topk_weights_f = topk_weights.float() if topk_weights is not None else None

        sm_count = torch.cuda.get_device_properties(device).multi_processor_count
        # Intra-node warp sweep (inter-node returned the base config above).
        warp_list = [4, 5, 6, 8, 10, 12, 14, 15, 16]
        fixed_block_num = int(num_sms) if num_sms and num_sms > 0 else base_cfg.block_num
        block_list = [fixed_block_num]

        # Size the op for the largest candidate so per-call overrides fit.
        worst_cfg = _MoriDispatchCfg(
            kernel_type=kernel_type,
            warp_num_per_block=max(warp_list),
            block_num=max(block_list),
            rdma_block_num=rdma_block_num,
        )
        self.init_buffer(
            group,
            hidden_size=hidden_size,
            num_experts=num_experts,
            num_topk=num_topk,
            seqlen=seqlen,
            fp8_dispatch=False,
            config=EPBufferConfig(
                num_sms=num_sms,
                dispatch_config=worst_cfg,
                combine_config=worst_cfg,
            ),
        )

        num_warmup = max(1, num_tests // 5)
        is_rank0 = group.rank() == 0
        total_cfgs = len(block_list) * len(warp_list)
        if is_rank0:
            logger.info(
                f"[Mori tuning] sm_count={sm_count} "
                f"kernel={kernel_type} block_num={fixed_block_num} "
                f"warp_per_block candidates={len(warp_list)} "
                f"total={total_cfgs}",
                rank=0,
            )

        # Paired dispatch->combine sweep; _combine matches the live contiguous() prep.
        candidates = [(block_num, warp_per_block) for block_num in block_list for warp_per_block in warp_list]

        def _dispatch(cand):
            block_num, warp_per_block = cand
            recv_x, recv_weights, _, recv_idx, _ = self._op.dispatch(
                x_tensor,
                topk_weights_f,
                None,
                topk_idx_i32,
                block_num=block_num,
                rdma_block_num=rdma_block_num,
                warp_per_block=warp_per_block,
            )
            return (recv_x, recv_weights, recv_idx, block_num, warp_per_block)

        def _combine(cand, recv):
            recv_x, recv_weights, recv_idx, block_num, warp_per_block = recv
            self._op.combine(
                recv_x.contiguous(),
                recv_weights,
                recv_idx,
                block_num=block_num,
                rdma_block_num=rdma_block_num,
                warp_per_block=warp_per_block,
            )

        def _on_skip(cand, exc):
            if is_rank0:
                logger.debug(f"[Mori tuning] skip invalid cfg {cand}: {exc!r}")

        best_d, best_c, best_d_time, best_c_time = sweep_configs(
            candidates,
            _dispatch,
            _combine,
            group=group,
            num_tests=num_tests,
            num_warmup=num_warmup,
            on_skip=_on_skip,
        )
        if best_d is None or best_d_time == float("inf"):
            raise RuntimeError("MoriEPBackend.tune_configs: no valid dispatch config in sweep.")
        if best_c is None or best_c_time == float("inf"):
            raise RuntimeError("MoriEPBackend.tune_configs: no valid combine config in sweep.")

        # Broadcast rank 0's pick so every rank agrees on the winners.
        disp_winner = _broadcast_from_rank0_int(list(best_d), group)
        comb_winner = _broadcast_from_rank0_int(list(best_c), group)
        best_d_time = _broadcast_from_rank0_float(best_d_time, group)
        best_c_time = _broadcast_from_rank0_float(best_c_time, group)

        final_d = _MoriDispatchCfg(
            kernel_type=kernel_type,
            warp_num_per_block=int(disp_winner[1]),
            block_num=int(disp_winner[0]),
            rdma_block_num=rdma_block_num,
        )
        final_c = _MoriDispatchCfg(
            kernel_type=kernel_type,
            warp_num_per_block=int(comb_winner[1]),
            block_num=int(comb_winner[0]),
            rdma_block_num=rdma_block_num,
        )

        if is_rank0:
            logger.info(
                f"[Mori tuning] best dispatch block_num={final_d.block_num} "
                f"warp_per_block={final_d.warp_num_per_block} t={best_d_time * 1e6:.2f} us; "
                f"best combine block_num={final_c.block_num} "
                f"warp_per_block={final_c.warp_num_per_block} t={best_c_time * 1e6:.2f} us",
                rank=0,
            )

        return final_d, final_c, best_d_time, best_c_time

    def release_buffer(self) -> None:
        """Drop the op and free its Mori SHMEM (kept ``self._group``; SHMEM binding is global).

        Nulling ``self._op`` alone leaves the op pinned by ``_build_mori_op``'s
        lru_cache, so its dtor (``ShmemFree``) never runs and autotune leaks the
        tuning op; ``cache_clear()`` lets the dtor reclaim the static heap.
        """
        # Drain in-flight kernels touching the buffers before they are freed.
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.synchronize()
        self._op = None
        self._hidden_size = 0
        self._num_local_experts = 0
        self._num_topk = 0
        self._seqlen = 0
        self._fp8_dispatch = False
        self._params_dtype = None
        self._kernel_cfg = None
        # Drop the cache's ref so the handle dtor runs (ShmemFree).
        _build_mori_op.cache_clear()
        gc.collect()
