###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Precision-neutral MegaMoE runtime contracts.

The BF16 and MXFP8 data planes keep different physical heap layouts and
different FlyDSL kernels.  This module owns the logical shape, handle and
workspace contracts shared by their Python orchestration layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence

import torch


class MegaMoEPrecision(str, Enum):
    BF16 = "bf16"
    MXFP8 = "mxfp8"


@dataclass(frozen=True)
class MegaShape:
    """Shape and routing geometry shared by one MegaMoE workspace."""

    world_size: int
    num_experts: int
    num_max_tokens_per_rank: int
    num_topk: int
    hidden: int
    intermediate_hidden: int
    num_max_pool_tokens: int
    block_m: int = 256
    block_n: int = 256

    @property
    def experts_per_rank(self) -> int:
        return self.num_experts // self.world_size

    @property
    def num_pool_blocks(self) -> int:
        return self.num_max_pool_tokens // self.block_m

    @property
    def combine_slots(self) -> int:
        return self.num_max_tokens_per_rank * self.num_topk

    @property
    def workspace_key(self) -> tuple:
        return (
            self.world_size,
            self.num_experts,
            self.num_max_tokens_per_rank,
            self.num_topk,
            self.hidden,
            self.intermediate_hidden,
            self.num_max_pool_tokens,
            self.block_m,
            self.block_n,
        )


@dataclass(frozen=True)
class WorkspaceRequest:
    """Input to the backend-specific workspace adapter."""

    process_group: Any
    precision: MegaMoEPrecision
    shape: MegaShape
    tile_config: tuple = ()

    @property
    def key(self) -> tuple:
        # ProcessGroup objects are identity-scoped and are not guaranteed to be
        # hashable.  Their identity is the correct lifetime boundary here:
        # symmetric allocations cannot be shared across process groups.
        return id(self.process_group), self.precision, self.shape.workspace_key, self.tile_config


class WorkspaceAdapter(Protocol):
    """Minimal host-side workspace interface exposed to orchestration."""

    precision: MegaMoEPrecision
    shape: MegaShape

    def destroy(self) -> None: ...


@dataclass(frozen=True)
class HandleSchema:
    """Named view over a backend-specific flat handle tuple."""

    precision: MegaMoEPrecision
    length: int
    fields: Mapping[str, int]

    def validate(self, handle: Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
        handle = tuple(handle)
        if len(handle) != self.length:
            raise ValueError(
                f"{self.precision.value} MegaMoE handle length {len(handle)} != "
                f"{self.length}; handle ABI changed"
            )
        missing = [name for name, index in self.fields.items() if index >= len(handle)]
        if missing:
            raise ValueError(
                f"{self.precision.value} MegaMoE handle is missing fields {missing}; ABI changed"
            )
        return handle

    def get(self, handle: Sequence[torch.Tensor], field: str) -> torch.Tensor:
        if field not in self.fields:
            raise KeyError(f"{self.precision.value} MegaMoE handle has no field {field!r}")
        handle = self.validate(handle)
        return handle[self.fields[field]]


BF16_HANDLE_SCHEMA = HandleSchema(
    precision=MegaMoEPrecision.BF16,
    length=13,
    fields={
        "expert_send_dst_rank": 0,
        "expert_send_dst_row": 1,
        "expert_send_count": 2,
        "expert_send_offset": 3,
        "dispatched_token_idx": 4,
        "tile_to_expert": 5,
        "real_count_per_expert": 6,
        "group_offsets": 7,
        "tile_count": 8,
        "recv_dst_rank": 9,
        "recv_start_row": 10,
        "recv_count": 11,
        "pool_src_slot": 12,
    },
)

MXFP8_HANDLE_SCHEMA = HandleSchema(
    precision=MegaMoEPrecision.MXFP8,
    length=14,
    fields={
        "expert_send_dst_rank": 0,
        "expert_send_dst_row": 1,
        "expert_send_count": 2,
        "expert_send_offset": 3,
        "dispatched_token_idx": 4,
        "dispatched_topk_slot": 5,
        "src_token_weight": 6,
        "tile_to_expert": 7,
        "tile_expected": 8,
        "group_lens": 9,
        "group_offsets": 10,
        "tile_count": 11,
        "origin_rank": 12,
        "origin_slot": 13,
    },
)


@dataclass
class RouteState:
    """Logical routing state shared by both staged implementations."""

    precision: MegaMoEPrecision
    handle: tuple[torch.Tensor, ...]
    dispatch_weights: torch.Tensor
    workspace: WorkspaceAdapter | None = None

    def __post_init__(self) -> None:
        self.handle = self.schema.validate(self.handle)

    @property
    def schema(self) -> HandleSchema:
        return handle_schema(self.precision)

    def field(self, name: str) -> torch.Tensor:
        return self.schema.get(self.handle, name)

    @property
    def tile_count(self) -> torch.Tensor:
        return self.field("tile_count")

    def assert_same_handle(self, handle: Sequence[torch.Tensor]) -> None:
        candidate = self.schema.validate(handle)
        if len(candidate) != len(self.handle) or any(
            lhs.data_ptr() != rhs.data_ptr() for lhs, rhs in zip(self.handle, candidate)
        ):
            raise ValueError("staged MegaMoE received a handle different from stage1's route state")


@dataclass
class DispatchState:
    """Backend-neutral dispatch result returned by a prologue facade."""

    precision: MegaMoEPrecision
    shape: MegaShape
    handle: tuple[torch.Tensor, ...]
    workspace: WorkspaceAdapter
    dispatch_weights: torch.Tensor

    def __post_init__(self) -> None:
        self.handle = handle_schema(self.precision).validate(self.handle)

    @property
    def route(self) -> RouteState:
        return RouteState(
            precision=self.precision,
            handle=self.handle,
            dispatch_weights=self.dispatch_weights,
            workspace=self.workspace,
        )


class StageState:
    """Non-differentiable state carrier shared by stage1 and stage2 edges."""

    __slots__ = ("precision", "route", "dispatch_state", "payload")

    def __init__(
        self,
        precision: MegaMoEPrecision,
        *,
        route: RouteState | None = None,
        dispatch_state: DispatchState | None = None,
        payload: Any = None,
    ) -> None:
        self.precision = precision
        self.route = route
        self.dispatch_state = None
        self.payload = payload
        if dispatch_state is not None:
            self.set_dispatch(dispatch_state)

    def set_route(self, route: RouteState) -> None:
        if route.precision is not self.precision:
            raise ValueError(
                f"stage state precision {self.precision.value} cannot carry "
                f"{route.precision.value} route state"
            )
        self.route = route

    def require_route(self) -> RouteState:
        if self.route is None:
            raise RuntimeError("staged MegaMoE route state was not initialized")
        return self.route

    def set_dispatch(self, dispatch_state: DispatchState) -> None:
        if dispatch_state.precision is not self.precision:
            raise ValueError(
                f"stage state precision {self.precision.value} cannot carry "
                f"{dispatch_state.precision.value} dispatch state"
            )
        self.dispatch_state = dispatch_state
        self.route = dispatch_state.route

    def require_dispatch(self) -> DispatchState:
        if self.dispatch_state is None:
            raise RuntimeError("staged MegaMoE dispatch state was not initialized")
        return self.dispatch_state


class WorkspaceRegistry:
    """Small lifecycle registry; backend adapters remain responsible for allocation."""

    def __init__(self) -> None:
        self._workspaces: dict[tuple, WorkspaceAdapter] = {}
        self._active: dict[MegaMoEPrecision, WorkspaceAdapter] = {}

    def get(self, request: WorkspaceRequest) -> WorkspaceAdapter | None:
        return self._workspaces.get(request.key)

    def put(self, request: WorkspaceRequest, workspace: WorkspaceAdapter) -> WorkspaceAdapter:
        old = self._workspaces.get(request.key)
        if old is not None and old is not workspace:
            old.destroy()
        self._workspaces[request.key] = workspace
        self._active[request.precision] = workspace
        return workspace

    def pop(self, request: WorkspaceRequest) -> WorkspaceAdapter | None:
        workspace = self._workspaces.pop(request.key, None)
        if workspace is not None and self._active.get(request.precision) is workspace:
            self._active.pop(request.precision, None)
        return workspace

    def destroy(self, request: WorkspaceRequest) -> None:
        workspace = self._workspaces.pop(request.key, None)
        if workspace is not None:
            workspace.destroy()
            if self._active.get(request.precision) is workspace:
                self._active.pop(request.precision, None)

    def destroy_all(self) -> None:
        for workspace in tuple(self._workspaces.values()):
            workspace.destroy()
        self._workspaces.clear()
        self._active.clear()

    def get_active(self, precision: MegaMoEPrecision | str) -> WorkspaceAdapter | None:
        return self._active.get(MegaMoEPrecision(precision))


class MegaRuntime:
    """One logical MegaMoE runtime bound to a process group and shape."""

    def __init__(
        self,
        request: WorkspaceRequest,
        workspace: WorkspaceAdapter,
        *,
        registry: "MegaRuntimeRegistry | None" = None,
    ) -> None:
        self.request = request
        self.workspace = workspace
        self._registry = registry

    @property
    def precision(self) -> MegaMoEPrecision:
        return self.request.precision

    @property
    def shape(self) -> MegaShape:
        return self.request.shape

    def dispatch_state(
        self,
        handle: Sequence[torch.Tensor],
        dispatch_weights: torch.Tensor,
    ) -> DispatchState:
        return DispatchState(
            precision=self.precision,
            shape=self.shape,
            handle=tuple(handle),
            workspace=self.workspace,
            dispatch_weights=dispatch_weights,
        )

    def destroy(self) -> None:
        if self._registry is not None:
            self._registry.destroy(self.request)
        else:
            self.workspace.destroy()


class MegaPrologueFacade:
    """Dispatch/prologue facade over the two physical FlyDSL launchers."""

    @staticmethod
    def launch(
        runtime: MegaRuntime,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor | None,
        *,
        num_cu: int | None = None,
    ) -> DispatchState:
        shape = runtime.shape
        workspace = runtime.workspace
        num_tokens = int(topk_idx.shape[0])
        num_topk = int(topk_idx.shape[-1])
        if runtime.precision is MegaMoEPrecision.BF16:
            from primus_turbo.flydsl.mega.dispatch_prologue_kernel import (
                dispatch_prologue_flydsl_kernel,
            )

            handle = dispatch_prologue_flydsl_kernel(
                topk_idx,
                topk_weights,
                sym_buffer=workspace.get_sym_buffer(),
                num_tokens=num_tokens,
                num_topk=num_topk,
                num_experts=shape.num_experts,
                num_ranks=shape.world_size,
                rank=workspace.rank,
                experts_per_rank=shape.experts_per_rank,
                block_m=shape.block_m,
                num_max_pool_tokens=shape.num_max_pool_tokens,
                hidden=shape.hidden,
                num_max_tokens_per_rank=shape.num_max_tokens_per_rank,
            )
            # BF16's pool source slot is a host snapshot appended by the
            # launcher, whereas MXFP8 stores origin metadata in its heap.
            handle = tuple(handle) + (workspace.pool_src_slot.clone(),)
        else:
            from primus_turbo.flydsl.mega.fp8.dispatch_prologue import dispatch_prologue

            kwargs = {}
            if num_cu is not None:
                kwargs["num_cu"] = int(num_cu)
            handle = dispatch_prologue(
                topk_idx,
                topk_weights,
                sym_layout=workspace.make_sym_layout(),
                num_tokens=num_tokens,
                num_topk=num_topk,
                num_experts=shape.num_experts,
                world_size=shape.world_size,
                rank=workspace.rank,
                experts_per_rank=shape.experts_per_rank,
                block_m=shape.block_m,
                num_max_pool_tokens=shape.num_max_pool_tokens,
                **kwargs,
            )
            # The FP8 launcher owns the logical route tables; these three
            # snapshots are the per-call origin/tile extension consumed by
            # staged backward.  Keep the extension inside the facade so its
            # returned DispatchState always satisfies the 14-slot ABI.
            handle = tuple(handle) + (
                workspace.meta_scalars[1:2].clone(),
                workspace.origin_rank.clone(),
                workspace.origin_slot.clone(),
            )
        return runtime.dispatch_state(tuple(handle), topk_weights)


class MegaRuntimeRegistry:
    """Shared runtime registry for BF16 and MXFP8 physical workspace adapters."""

    def __init__(self) -> None:
        self._workspaces = WorkspaceRegistry()
        self._runtimes: dict[tuple, MegaRuntime] = {}

    def get(self, request: WorkspaceRequest) -> MegaRuntime | None:
        return self._runtimes.get(request.key)

    def acquire(
        self,
        request: WorkspaceRequest,
        factory: Any,
    ) -> MegaRuntime:
        runtime = self._runtimes.get(request.key)
        if runtime is not None:
            return runtime
        workspace = self._workspaces.get(request)
        if workspace is None:
            # A symmetric heap is a multi-GB allocation and both backends assume a
            # single live workspace per precision, so retire the other shapes
            # instead of letting them accumulate.
            self._evict_precision(request.precision, keep=request.key)
            workspace = factory()
            self._workspaces.put(request, workspace)
            workspace._mega_runtime_registry = self
            workspace._mega_runtime_request = request
        runtime = MegaRuntime(request, workspace, registry=self)
        self._runtimes[request.key] = runtime
        return runtime

    def _evict_precision(self, precision: MegaMoEPrecision, *, keep: tuple | None = None) -> None:
        stale = [
            runtime.request
            for key, runtime in self._runtimes.items()
            if runtime.precision is precision and key != keep
        ]
        for request in stale:
            self.destroy(request)

    def active(self, precision: MegaMoEPrecision | str) -> MegaRuntime | None:
        workspace = self._workspaces.get_active(precision)
        if workspace is None:
            return None
        for runtime in self._runtimes.values():
            if runtime.workspace is workspace:
                return runtime
        return None

    def destroy(self, request: WorkspaceRequest) -> None:
        self._runtimes.pop(request.key, None)
        workspace = self._workspaces.pop(request)
        if workspace is not None:
            workspace._mega_runtime_destroying = True
            try:
                workspace.destroy()
            finally:
                workspace._mega_runtime_destroying = False

    def destroy_all(self) -> None:
        self._runtimes.clear()
        self._workspaces.destroy_all()


_MEGA_RUNTIME_REGISTRY = MegaRuntimeRegistry()


def get_mega_runtime_registry() -> MegaRuntimeRegistry:
    return _MEGA_RUNTIME_REGISTRY


def handle_schema(precision: MegaMoEPrecision | str) -> HandleSchema:
    precision = MegaMoEPrecision(precision)
    if precision is MegaMoEPrecision.BF16:
        return BF16_HANDLE_SCHEMA
    return MXFP8_HANDLE_SCHEMA


def make_route_state(
    precision: MegaMoEPrecision | str,
    dispatch_weights: torch.Tensor,
    handle: Sequence[torch.Tensor],
    *,
    workspace: WorkspaceAdapter | None = None,
) -> RouteState:
    return RouteState(
        precision=MegaMoEPrecision(precision),
        handle=tuple(handle),
        dispatch_weights=dispatch_weights,
        workspace=workspace,
    )


__all__ = [
    "BF16_HANDLE_SCHEMA",
    "MXFP8_HANDLE_SCHEMA",
    "DispatchState",
    "HandleSchema",
    "MegaMoEPrecision",
    "MegaPrologueFacade",
    "MegaRuntime",
    "MegaRuntimeRegistry",
    "MegaShape",
    "RouteState",
    "StageState",
    "WorkspaceAdapter",
    "WorkspaceRegistry",
    "WorkspaceRequest",
    "get_mega_runtime_registry",
    "handle_schema",
    "make_route_state",
]
