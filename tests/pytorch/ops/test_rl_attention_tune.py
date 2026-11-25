###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import pytest
import torch

from primus_turbo.pytorch.ops.attention.rl_attention_tune import (
    format_options,
    search_space,
    tune,
)

# Fake call: the time is a function of the options; BLOCK_M=32 changes the bits.
_COST = {("BLOCK_M", 16): 8.0, ("BLOCK_M", 32): 1.0, ("num_warps", 2): 6.0, ("num_warps", 4): 9.0}


def _fake():
    last = {}

    def run(options):
        last.clear()
        last.update(options)
        out = torch.zeros(4)
        if options.get("BLOCK_M") == 32:
            out[0] = 1.0
        return out

    def bench(fn):
        fn()
        return 10.0 - sum(10.0 - _COST.get(item, 10.0) for item in last.items())

    return run, bench


@pytest.mark.parametrize("exhaustive", [False, True])
def test_tune_picks_fastest_bit_identical(exhaustive):
    run, bench = _fake()
    space = {"BLOCK_M": [0, 16, 32], "num_warps": [0, 2, 4]}
    res = tune(run, space, bench=bench, exhaustive=exhaustive)
    assert res.options == {"BLOCK_M": 16, "num_warps": 2}
    assert res.default_us == 10.0 and res.best_us == 4.0
    assert any(opts.get("BLOCK_M") == 32 and why.startswith("bits") for opts, why in res.rejected)


def test_tune_keeps_default_below_min_gain():
    run, bench = _fake()
    res = tune(run, {"num_warps": [0, 2]}, bench=bench, min_gain=0.5)
    assert res.options == {} and res.speedup == 1.0


def test_tune_skips_failing_candidates():
    def run(options):
        if options.get("MODE") == "split":
            raise ValueError("unsupported")
        return torch.zeros(2)

    res = tune(run, {"MODE": ["auto", "split"]}, bench=lambda fn: 1.0)
    assert res.options == {} and res.rejected[0][1].startswith("ValueError")


def test_only_schedule_options():
    with pytest.raises(ValueError):
        tune(lambda o: torch.zeros(1), {"TILE_SIZE": [32, 64]}, bench=lambda fn: 1.0)
    for backend in ("TRITON", "FLYDSL"):
        for q in (1, 512):
            assert all(values[0] in (0, "auto") for values in search_space(backend, q).values())
    assert format_options({"MODE": "split", "BLOCK_M": 16}) == "MODE=split,BLOCK_M=16"
