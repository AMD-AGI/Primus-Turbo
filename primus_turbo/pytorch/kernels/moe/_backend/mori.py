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

from primus_turbo.common.constants import ENV_MORI_NUM_QP_PER_PE
from primus_turbo.common.logger import logger
from primus_turbo.common.mori_utils import get_mori

from ._config import EPBufferConfig
from .base import (
    _apply_env_with_nccl_fallback,
    _broadcast_from_rank0_float,
    _broadcast_from_rank0_int,
    _EPCapabilities,
    call_once,
    detect_group_topology,
    sweep_configs,
)
from .expert_token_count import compute_expert_token_info

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

    # Defaults aligned with mori's official gfx950 tuning (used when autotune off).
    return {
        _MoriEpMode.INTRA_NODE: _MoriDispatchCfg(
            kernel_type=mori.ops.EpDispatchCombineKernelType.IntraNode,
            warp_num_per_block=16,
            block_num=256,
            rdma_block_num=0,
        ),
        _MoriEpMode.INTER_NODE: _MoriDispatchCfg(
            kernel_type=mori.ops.EpDispatchCombineKernelType.InterNodeV1,
            warp_num_per_block=8,
            block_num=128,
            rdma_block_num=64,
        ),
    }


def _mori_block_num_candidates(num_sms: int, sm_count: int) -> List[int]:
    """block_num sweep set: powers of two up to sm_count, plus sm_count and num_sms.

    Mirrors mori's bench; excludes over-subscription (>sm_count, hang risk).
    block_num does not change op buffer size, so the sweep is memory-free.
    """
    cap = int(sm_count) if sm_count and sm_count > 0 else int(num_sms or 64)
    candidates = {cap}
    p = 32
    while p <= cap:
        candidates.add(p)
        p *= 2
    if num_sms and num_sms > 0:
        candidates.add(min(int(num_sms), cap))
    return sorted(v for v in candidates if 0 < v <= cap)


# mori's native env: AUTO makes the op auto-apply its shipped tuning JSON per-call.
_MORI_LAUNCH_CONFIG_MODE_ENV = "MORI_EP_LAUNCH_CONFIG_MODE"


def _mori_auto_mode() -> bool:
    """True when mori auto-tunes from its shipped JSON DB (opt-in via env=AUTO).

    Default MANUAL: our benchmark sweep beats the JSON for bf16 (mori's dispatch
    JSON is fp8-only, so bf16 dispatch falls back to a slower hard-coded cfg).
    """
    return os.environ.get(_MORI_LAUNCH_CONFIG_MODE_ENV, "MANUAL").upper() == "AUTO"


_MORI_SHMEM_PG_NAME = "mori"


def _align_mori_hip_with_torch() -> None:
    """Make Mori's JIT reuse torch's libamdhip64.so (fixes HIP error 500 on ROCm 7.2+)."""
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
    """Return (block_num, warp_per_block, rdma_block_num); -1 means op default."""
    if isinstance(config, _MoriDispatchCfg):
        return config.block_num, config.warp_num_per_block, config.rdma_block_num
    return -1, -1, -1


def _resolve_mori_dispatch_cfg(
    group: dist.ProcessGroup,
    config: Optional[EPBufferConfig] = None,
) -> _MoriDispatchCfg:
    """Resolve launch cfg: topology default, overridden by num_sms and dispatch_config."""
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

    # mori reads MORI_EP_LAUNCH_CONFIG_MODE itself (native default MANUAL); set it
    # to AUTO to make the op auto-apply its shipped tuning JSON per-call.
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


class MoriEPBackend(_EPCapabilities):
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
            get_mori()
            return True
        except ImportError:
            return False

    @staticmethod
    def supports_cuda_graph() -> bool:
        """not supported"""
        return False

    @classmethod
    def can_release(cls, *, will_reinit: bool) -> bool:
        # inter-node free+rebuild hangs: keep if reinit'd, free a dead loser.
        return not will_reinit

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
        """Set MORI_* RDMA env from NCCL_* fallbacks; None kwargs stay unset (auto-detect)."""
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
        """Register Mori SHMEM and (re)build the op on shape/kernel-type change.

        block/warp/rdma are per-call, so the op is reused (not rebuilt) on cfg
        change — the autotune winner reuses the tuned op (inter-node rebuild hangs).
        """
        assert not fp8_dispatch, "not implemented"

        # Lazy mori import (install hint + version check).
        get_mori()

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

        # Rebuild on shape/kernel-type change only; block/warp/rdma are per-call.
        needs_rebuild = (
            self._op is None
            or self._hidden_size != hidden_size
            or self._num_local_experts != num_local_experts
            or self._num_topk != num_topk
            or self._seqlen != seqlen
            or self._fp8_dispatch != fp8_dispatch
            or self._params_dtype != params_dtype
            or self._kernel_cfg is None
            or self._kernel_cfg.kernel_type != kernel_cfg.kernel_type
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

        # Graph-friendly caller: keep counts on device (no D2H sync).
        keep_tokens_per_expert_on_device = num_worst_tokens > 0

        if handle is None:
            assert topk_idx is not None
            topk_idx_i32 = topk_idx.to(torch.int32)
        else:
            assert topk_idx is None and token_weights is None
            (topk_idx_i32, token_weights, _) = handle

        # Autotuner launch override; -1 keeps the op-level default.
        block_num, warp_per_block, rdma_block_num = _extract_mori_launch_override(config)

        # FP8 scales arg; BF16-only path for now.
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
        """Sweep (block_num, warp, rdma) on one op via per-call overrides (mirrors mori's bench).

        Intra pins rdma=0; inter sweeps it. Best dispatch/combine tracked
        independently. Returns (dispatch_cfg, combine_cfg, dispatch_s, combine_s).
        """
        # Lazy mori import (install hint + version check).
        get_mori()

        if isinstance(x, tuple):
            raise NotImplementedError(
                "MoriEPBackend.tune_configs: FP8 dispatch (tuple ``x``) is not supported."
            )

        # Topology-derived kernel pinning: matches the live ``init_buffer`` path.
        _, num_nodes = detect_group_topology(group)
        mode = _MoriEpMode.INTRA_NODE if num_nodes <= 1 else _MoriEpMode.INTER_NODE
        base_cfg = _get_mori_dispatch_configs()[mode]
        kernel_type = base_cfg.kernel_type

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

        if _mori_auto_mode():
            # mori self-tunes from its JSON per-call; just measure one representative config.
            warp_list = [base_cfg.warp_num_per_block]
            block_list = [base_cfg.block_num]
            candidates = [(base_cfg.block_num, base_cfg.warp_num_per_block, base_cfg.rdma_block_num)]
        else:
            block_list = _mori_block_num_candidates(num_sms, sm_count)
            # Intra pins rdma=0; inter sweeps rdma ~ {block/2, block*2/3} (fewer warps to bound cost).
            if num_nodes <= 1:
                warp_list = [4, 5, 6, 8, 10, 12, 14, 15, 16]

                def _rdma_candidates(block_num: int) -> List[int]:
                    return [0]

            else:
                warp_list = [4, 8, 16]

                def _rdma_candidates(block_num: int) -> List[int]:
                    cands = sorted(
                        {v for v in (max(block_num // 2, 1), block_num * 2 // 3) if 1 <= v < block_num}
                    )
                    return cands or [max(block_num // 2, 1)]

            candidates = [
                (block_num, warp_per_block, rdma_block_num)
                for block_num in block_list
                for warp_per_block in warp_list
                for rdma_block_num in _rdma_candidates(block_num)
            ]

        # Size the op for the largest candidate so per-call overrides fit.
        worst_cfg = _MoriDispatchCfg(
            kernel_type=kernel_type,
            warp_num_per_block=max(warp_list),
            block_num=max(block_list),
            rdma_block_num=base_cfg.rdma_block_num,
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
        if is_rank0:
            mode = "AUTO (mori tuning DB)" if _mori_auto_mode() else "MANUAL sweep"
            logger.info(
                f"[Mori tuning] mode={mode} sm_count={sm_count} kernel={kernel_type} "
                f"block candidates={block_list} warp candidates={warp_list} "
                f"rdma swept={num_nodes > 1 and not _mori_auto_mode()} total={len(candidates)}",
                rank=0,
            )

        # Paired dispatch->combine sweep; _combine matches the live contiguous() prep.
        def _dispatch(cand):
            block_num, warp_per_block, rdma_block_num = cand
            recv_x, recv_weights, _, recv_idx, _ = self._op.dispatch(
                x_tensor,
                topk_weights_f,
                None,
                topk_idx_i32,
                block_num=block_num,
                rdma_block_num=rdma_block_num,
                warp_per_block=warp_per_block,
            )
            return (recv_x, recv_weights, recv_idx, block_num, warp_per_block, rdma_block_num)

        def _combine(cand, recv):
            recv_x, recv_weights, recv_idx, block_num, warp_per_block, rdma_block_num = recv
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
        disp_winner = _broadcast_from_rank0_int(list(best_d), group)  # [block, warp, rdma]
        comb_winner = _broadcast_from_rank0_int(list(best_c), group)
        best_d_time = _broadcast_from_rank0_float(best_d_time, group)
        best_c_time = _broadcast_from_rank0_float(best_c_time, group)

        final_d = _MoriDispatchCfg(
            kernel_type=kernel_type,
            warp_num_per_block=int(disp_winner[1]),
            block_num=int(disp_winner[0]),
            rdma_block_num=int(disp_winner[2]),
        )
        final_c = _MoriDispatchCfg(
            kernel_type=kernel_type,
            warp_num_per_block=int(comb_winner[1]),
            block_num=int(comb_winner[0]),
            rdma_block_num=int(comb_winner[2]),
        )

        if is_rank0:
            logger.info(
                f"[Mori tuning] best dispatch block_num={final_d.block_num} "
                f"warp={final_d.warp_num_per_block} rdma={final_d.rdma_block_num} "
                f"t={best_d_time * 1e6:.2f} us; best combine block_num={final_c.block_num} "
                f"warp={final_c.warp_num_per_block} rdma={final_c.rdma_block_num} "
                f"t={best_c_time * 1e6:.2f} us",
                rank=0,
            )

        return final_d, final_c, best_d_time, best_c_time

    def release_buffer(self) -> None:
        """Drop the op and free its Mori SHMEM (keeps the global SHMEM binding).

        cache_clear() is required: the lru_cache ref otherwise pins the op so its
        dtor (ShmemFree) never runs and autotune leaks the tuning op.
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
