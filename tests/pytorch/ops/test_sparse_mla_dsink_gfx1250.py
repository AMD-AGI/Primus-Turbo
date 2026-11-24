###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Sparse-MLA attention-sink gradient kernel: dsink[h] = -sum_t exp(sink[h] - lse[t, h]) * delta[t, h].

The partial-sum kernel must match a float64 reference, be bit-repeatable, and the variants that write straight
into a bf16 ``main_grad`` (overwrite and accumulate) must equal bf16 rounding of its result.
"""

import pytest
import torch

from primus_turbo.triton.attention import sparse_mla_dsink as sd
from tests.pytorch.ops.gfx1250_helper import bits_equal, rel_l2, requires_gfx1250

pytestmark = requires_gfx1250


def _inputs(t, h, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    sink = torch.randn(h, device="cuda", generator=g) * 0.1
    lse = torch.randn(t, h, device="cuda", generator=g) + 3.0
    delta = torch.randn(t, h, device="cuda", generator=g)
    return sink, lse, delta


def _ref(sink, lse, delta):
    return -(torch.exp(sink.double()[None] - lse.double()) * delta.double()).sum(0)


@pytest.mark.parametrize("t, h", [(4096, 64), (1500, 32), (3000, 128)])
def test_dsink_matches_reference_and_is_repeatable(t, h):
    sink, lse, delta = _inputs(t, h)
    out = sd.sparse_mla_dsink(sink, lse, delta)
    assert rel_l2(out, _ref(sink, lse, delta)) < 1e-4
    assert bits_equal(out, sd.sparse_mla_dsink(sink, lse, delta))


@pytest.mark.parametrize("accumulate", [False, True])
@pytest.mark.parametrize("fn", ["sparse_mla_dsink_mg2"])
def test_dsink_main_grad_variants(fn, accumulate):
    t, h = 3000, 64
    sink, lse, delta = _inputs(t, h, seed=1)
    ref = _ref(sink, lse, delta)
    mg0 = torch.randn(h, device="cuda").bfloat16()
    mg = mg0.clone()
    getattr(sd, fn)(sink, lse, delta, mg, accumulate)
    torch.cuda.synchronize()
    want = (mg0.float() + ref.float()).bfloat16() if accumulate else ref.float().bfloat16()
    assert rel_l2(mg, want) < 2e-2
