###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

"""CPU-only tests for the shared staged MegaMoE Python contract."""

import pytest
import torch

from primus_turbo.flydsl.mega.runtime import MegaRuntimeRegistry
from primus_turbo.pytorch.kernels.fused_mega_moe.staged_contract import (
    BF16_HANDLE_SCHEMA,
    MXFP8_HANDLE_SCHEMA,
    MegaMoEPrecision,
    MegaShape,
    StageState,
    WorkspaceRequest,
    make_route_state,
)


def _handle(length):
    return tuple(torch.empty(1, dtype=torch.int32) for _ in range(length))


@pytest.mark.parametrize(
    ("schema", "precision"),
    [
        (BF16_HANDLE_SCHEMA, MegaMoEPrecision.BF16),
        (MXFP8_HANDLE_SCHEMA, MegaMoEPrecision.MXFP8),
    ],
)
def test_handle_schema_exposes_named_fields(schema, precision):
    handle = _handle(schema.length)

    assert schema.validate(handle) == handle
    assert schema.get(handle, "tile_count") is handle[schema.fields["tile_count"]]
    route = make_route_state(precision, torch.empty(1), handle)
    assert route.tile_count is handle[schema.fields["tile_count"]]


def test_handle_schema_rejects_wrong_abi():
    with pytest.raises(ValueError, match="handle length"):
        BF16_HANDLE_SCHEMA.validate(_handle(BF16_HANDLE_SCHEMA.length - 1))


def test_stage_state_rejects_mismatched_precision():
    state = StageState(MegaMoEPrecision.BF16)
    route = make_route_state(
        MegaMoEPrecision.MXFP8,
        torch.empty(1),
        _handle(MXFP8_HANDLE_SCHEMA.length),
    )

    with pytest.raises(ValueError, match="cannot carry"):
        state.set_route(route)


class _FakeWorkspace:
    def __init__(self, tag):
        self.tag = tag
        self.destroyed = False

    def destroy(self):
        self.destroyed = True


def _request(group, precision, hidden):
    return WorkspaceRequest(
        process_group=group,
        precision=precision,
        shape=MegaShape(
            world_size=8,
            num_experts=256,
            num_max_tokens_per_rank=8192,
            num_topk=8,
            hidden=hidden,
            intermediate_hidden=2048,
            num_max_pool_tokens=16384,
        ),
    )


def test_registry_keeps_one_live_workspace_per_precision():
    """A symmetric heap is multi-GB; a new shape must retire the previous one."""

    group = object()
    registry = MegaRuntimeRegistry()
    first = _FakeWorkspace("first")
    second = _FakeWorkspace("second")

    registry.acquire(_request(group, MegaMoEPrecision.BF16, 7168), lambda: first)
    runtime = registry.acquire(_request(group, MegaMoEPrecision.BF16, 4096), lambda: second)

    assert first.destroyed
    assert not second.destroyed
    assert runtime.workspace is second
    assert registry.active(MegaMoEPrecision.BF16) is runtime


def test_registry_keeps_precisions_independent():
    group = object()
    registry = MegaRuntimeRegistry()
    bf16 = _FakeWorkspace("bf16")
    mxfp8 = _FakeWorkspace("mxfp8")

    registry.acquire(_request(group, MegaMoEPrecision.BF16, 7168), lambda: bf16)
    registry.acquire(_request(group, MegaMoEPrecision.MXFP8, 7168), lambda: mxfp8)

    assert not bf16.destroyed
    assert not mxfp8.destroyed
    assert registry.active(MegaMoEPrecision.BF16).workspace is bf16
    assert registry.active(MegaMoEPrecision.MXFP8).workspace is mxfp8


def test_registry_reuses_workspace_for_same_request():
    group = object()
    registry = MegaRuntimeRegistry()
    workspace = _FakeWorkspace("only")

    first = registry.acquire(_request(group, MegaMoEPrecision.BF16, 7168), lambda: workspace)
    second = registry.acquire(
        _request(group, MegaMoEPrecision.BF16, 7168),
        lambda: pytest.fail("factory must not run for a cached request"),
    )

    assert first is second
    assert not workspace.destroyed
