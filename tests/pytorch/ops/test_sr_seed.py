###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""set_sr_seed: the stochastic-rounding seeds of both SR quantizers (the quantize_mx* packers and the MXFP4 quantizer)
are a function of the base seed and the launch index since it was set.

  * reproducible: setting the same base again replays the same SR codes, launch for launch (a run resumed at a step);
  * independent: a different base -- another rank, run seed or iteration (sr_step_seed) -- draws different codes;
  * per quantizer: one quantizer's launches do not shift the other's stream;
  * still stochastic: consecutive launches after one set draw differently.
"""

import pytest
import torch

import primus_turbo.pytorch as turbo
from primus_turbo.pytorch.kernels.quantization.mx_a4w4_pack import (
    FLY_A,
    fly_fmt,
    quantize_mx_dual,
)
from primus_turbo.pytorch.ops.quantization import set_sr_seed, sr_step_seed


def _skip():
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    if not hasattr(torch.ops.primus_turbo_cpp_extension, "set_sr_seed"):
        pytest.skip("Primus-Turbo built without set_sr_seed")


def _x(seed=0, rows=512, cols=3072):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn((rows, cols), device="cuda", dtype=torch.bfloat16, generator=g)


def _mx_sr(x):
    """quantize_mx* packer, gradient format with SR on both FP4 directions: the codes of both directions."""
    p = quantize_mx_dual(x, fly_fmt(FLY_A, FLY_A, sr=True))
    return p[0].clone(), p[2].clone()


def _mxfp4_sr(x):
    """The MXFP4 quantizer (HIP), SR in both directions: the codes of both directions."""
    out = torch.ops.primus_turbo_cpp_extension.quantize_mxfp4_dual(
        x,
        turbo.float4_e2m1fn_x2,
        128,
        False,
        True,
        False,
        False,
        True,
        False,
        False,
        False,
        False,
        False,
        0,
    )
    return out[0].view(torch.uint8).clone(), out[2].view(torch.uint8).clone()


def _same(a, b):
    return all(torch.equal(u, v) for u, v in zip(a, b))


@pytest.mark.parametrize("quant", [_mx_sr, _mxfp4_sr], ids=["quantize_mx", "mxfp4"])
def test_same_base_replays_the_same_codes(quant):
    _skip()
    x = _x()
    set_sr_seed(sr_step_seed(1234, 3, 100))
    first = [quant(x) for _ in range(3)]
    set_sr_seed(sr_step_seed(1234, 3, 100))
    again = [quant(x) for _ in range(3)]
    for a, b in zip(first, again):
        assert _same(
            a, b
        ), "the same base must replay the same SR codes, launch for launch"
    assert not _same(first[0], first[1]), "consecutive launches must draw independently"


@pytest.mark.parametrize("quant", [_mx_sr, _mxfp4_sr], ids=["quantize_mx", "mxfp4"])
@pytest.mark.parametrize(
    "other",
    [(1234, 4, 100), (1235, 3, 100), (1234, 3, 101)],
    ids=["other_rank", "other_run_seed", "next_iteration"],
)
def test_other_rank_seed_or_iteration_draws_differently(quant, other):
    _skip()
    x = _x(1)
    set_sr_seed(sr_step_seed(1234, 3, 100))
    a = quant(x)
    set_sr_seed(sr_step_seed(*other))
    b = quant(x)
    assert not _same(a, b)


def test_quantizers_keep_separate_streams():
    """MXFP4-quantizer launches between a set and a quantize_mx pack do not change the pack's SR codes."""
    _skip()
    x = _x(2)
    set_sr_seed(77)
    ref = _mx_sr(x)
    set_sr_seed(77)
    _mxfp4_sr(x)
    _mxfp4_sr(x)
    assert _same(_mx_sr(x), ref)


def test_sr_step_seed():
    seeds = {
        sr_step_seed(s, r, i) for s in (1, 2) for r in range(32) for i in range(50)
    }
    assert len(seeds) == 2 * 32 * 50, "distinct per (run seed, rank, iteration)"
    assert all(0 <= s < 1 << 64 for s in seeds)
    assert sr_step_seed(5, 6, 7) == sr_step_seed(5, 6, 7)
    _skip()
    set_sr_seed(max(seeds))  # the top bit set: passed through as a signed int64
