###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Shared staged MegaMoE state and handle contracts.

BF16 and MXFP8 intentionally keep different device-side workspace layouts and
kernel ABIs.  This module unifies the logical contract at the Python
orchestration boundary without making either hot path carry dtype branches.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

import torch


class MegaMoEPrecision(str, Enum):
    BF16 = "bf16"
    MXFP8 = "mxfp8"


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


class StageState:
    """Non-differentiable state carrier shared by stage1 and stage2 autograd edges.

    ``payload`` is deliberately backend-owned: MXFP8 stores quantized backward
    operands here, while BF16 leaves it unused.  Keeping this carrier common
    prevents the autograd wrapper from depending on a particular gradient
    representation.
    """

    __slots__ = ("precision", "route", "payload")

    def __init__(
        self,
        precision: MegaMoEPrecision,
        *,
        route: RouteState | None = None,
        payload: Any = None,
    ) -> None:
        self.precision = precision
        self.route = route
        self.payload = payload

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


def handle_schema(precision: MegaMoEPrecision | str) -> HandleSchema:
    precision = MegaMoEPrecision(precision)
    if precision is MegaMoEPrecision.BF16:
        return BF16_HANDLE_SCHEMA
    return MXFP8_HANDLE_SCHEMA


def make_route_state(
    precision: MegaMoEPrecision | str,
    dispatch_weights: torch.Tensor,
    handle: Sequence[torch.Tensor],
) -> RouteState:
    return RouteState(
        precision=MegaMoEPrecision(precision),
        handle=tuple(handle),
        dispatch_weights=dispatch_weights,
    )
