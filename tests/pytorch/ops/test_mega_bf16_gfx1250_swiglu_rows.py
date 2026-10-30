###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""``rows`` on the gfx1250 Mega SwiGLU backward: several rows a grid-stride pass, bit for bit.

``rows > 1`` only reorders the kernel's loads and shares the gate sums' barriers between rows, so
every output -- ``dx``, the routing-weight gradient and the weighted activation -- must equal the
one-row kernel's at the same grid and block size. A small odd grid leaves each block a tail of fewer
than ``rows`` rows, which it takes one row at a time; ``I = 2880`` puts a partial column tile at the
end of each row.
"""

import pytest
import torch

import primus_turbo.pytorch  # noqa: F401
from primus_turbo.pytorch.core.utils import is_gfx1250

pytestmark = pytest.mark.skipif(not is_gfx1250(), reason="gfx1250 kernel")

_BM = 256
_BLOCKS = 3  # pool blocks, one expert each
_GRID_X = 7
_THREADS = 256


def _run(I, act_name, with_scale, with_gate, with_act_w, with_bias, rows):
    from primus_turbo.flydsl.mega.gfx1250 import swiglu_kernel as sk
    from primus_turbo.flydsl.utils.glu_activation import (
        GLUActivation,
        activation_constexpr,
    )

    m = _BLOCKS * _BM
    g = torch.Generator(device="cuda").manual_seed(0)
    x = (torch.randn(m, 2 * I, device="cuda", generator=g) * 4).to(torch.bfloat16)
    dact = torch.randn(m, I, device="cuda", generator=g).to(torch.bfloat16)
    scale = torch.rand(m, device="cuda", generator=g, dtype=torch.float32)
    bias = torch.randn(_BLOCKS, 2 * I, device="cuda", generator=g).to(torch.bfloat16)
    ntb = torch.tensor([_BLOCKS], dtype=torch.int32, device="cuda")
    t2e = torch.arange(_BLOCKS, dtype=torch.int32, device="cuda")
    dx = torch.full((m, 2 * I), float("nan"), dtype=torch.bfloat16, device="cuda")
    gate = torch.full((m,), float("nan"), dtype=torch.float32, device="cuda")
    act_w = torch.full((m, I), float("nan"), dtype=torch.bfloat16, device="cuda")
    activation = GLUActivation.swigluoai() if act_name == "swigluoai" else None
    sk._compiled_swiglu_bwd.fn(
        DACT=dact,
        ACC1=x,
        DACC1=dx,
        SCALE=scale if with_scale else x,
        NUM_TILE_BLOCKS=ntb,
        GRAD_GATE=gate if with_gate else x,
        ACT_W=act_w if with_act_w else x,
        BIAS=bias if with_bias else x,
        TILE_TO_EXPERT=t2e if with_bias else ntb,
        I=I,
        with_scale=with_scale,
        BM=_BM,
        with_gate=with_gate,
        with_act_w=with_act_w,
        activation=activation_constexpr(activation),
        with_bias=with_bias,
        grid_x=_GRID_X,
        block_threads=_THREADS,
        stream=torch.cuda.current_stream(),
        rows=rows,
    )
    torch.cuda.synchronize()
    outs = {"dx": dx}
    if with_gate:
        outs["grad_gate"] = gate
    if with_act_w:
        outs["act_w"] = act_w
    return outs


@pytest.mark.parametrize("I", [2048, 2880])
@pytest.mark.parametrize("act_name", ["silu", "swigluoai"])
@pytest.mark.parametrize(
    "with_scale,with_gate,with_act_w,with_bias",
    [(1, 1, 1, 0), (1, 1, 0, 1), (0, 1, 0, 0), (1, 0, 0, 0), (0, 0, 0, 1)],
)
@pytest.mark.parametrize("rows", [2, 4])
def test_swiglu_bwd_rows_bit_identical(I, act_name, with_scale, with_gate, with_act_w, with_bias, rows):
    ref = _run(I, act_name, with_scale, with_gate, with_act_w, with_bias, 1)
    got = _run(I, act_name, with_scale, with_gate, with_act_w, with_bias, rows)
    for name, want in ref.items():
        assert not want.isnan().any(), f"{name}: the one-row kernel left rows unwritten"
        assert torch.equal(
            got[name].view(torch.int16 if want.dtype == torch.bfloat16 else torch.int32),
            want.view(torch.int16 if want.dtype == torch.bfloat16 else torch.int32),
        ), name
