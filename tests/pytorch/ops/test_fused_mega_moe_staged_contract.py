###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

"""CPU-only tests for the shared staged MegaMoE Python contract."""

import pytest
import torch

from primus_turbo.pytorch.kernels.fused_mega_moe.staged_contract import (
    BF16_HANDLE_SCHEMA,
    MXFP8_HANDLE_SCHEMA,
    MegaMoEPrecision,
    StageState,
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
