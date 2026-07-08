###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Phase-1 groundwork tests for mega mxfp8: symmetric-layout offset stability (the
bf16 byte layout must be unchanged when use_mxfp8 is added) and the mxfp8 region
sizes, plus the mxfp8 quant helpers' shapes/dtypes."""

import pytest
import torch

from primus_turbo.flydsl.mega.symm_buffer import get_symm_buffer_size_for_mega_moe

_SHAPE = dict(
    world_size=8,
    num_experts=32,
    num_max_tokens_per_rank=128,
    num_topk=8,
    hidden=7168,
    intermediate_hidden=2048,
)


def test_layout_offset_stability():
    # bf16 layout (use_mxfp8=False) must be byte-identical to enabling it for every
    # pre-existing region (the mxfp8 regions are appended after all of them).
    nb0, spec0, sb0, meta0 = get_symm_buffer_size_for_mega_moe(**_SHAPE, use_mxfp8=False)
    nb1, spec1, sb1, meta1 = get_symm_buffer_size_for_mega_moe(**_SHAPE, use_mxfp8=True)

    for name, (off, dt, numel) in spec0.items():
        assert name in spec1, f"{name} vanished under use_mxfp8"
        assert spec1[name][0] == off, f"{name} offset drifted: {off} -> {spec1[name][0]}"
        assert spec1[name][2] == numel, f"{name} numel drifted"
    assert sb0 == sb1, "signal heap must be unchanged (mxfp8 regions are all MAIN heap)"
    assert nb1 >= nb0, "mxfp8 layout should be >= bf16 (adds pool_fp8/scale, act_fp8/scale)"


def test_mxfp8_regions_present_and_sized():
    _, spec, _, meta = get_symm_buffer_size_for_mega_moe(**_SHAPE, use_mxfp8=True)
    P = meta["num_max_pool_tokens"]
    H = _SHAPE["hidden"]
    I = _SHAPE["intermediate_hidden"]
    for name in ("pool_fp8", "pool_scale", "act_fp8", "act_scale"):
        assert name in spec, f"missing mxfp8 region {name}"
    assert spec["pool_fp8"][1] == torch.float8_e4m3fn and spec["pool_fp8"][2] == P * H
    assert spec["pool_scale"][1] == torch.uint8 and spec["pool_scale"][2] == P * (H // 32)
    assert spec["act_fp8"][1] == torch.float8_e4m3fn and spec["act_fp8"][2] == P * I
    assert spec["act_scale"][1] == torch.uint8 and spec["act_scale"][2] == P * (I // 32)

    _, spec_bf16, _, _ = get_symm_buffer_size_for_mega_moe(**_SHAPE, use_mxfp8=False)
    for name in ("pool_fp8", "pool_scale", "act_fp8", "act_scale"):
        assert name not in spec_bf16, f"{name} must be absent in bf16 layout"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
def test_quant_helpers_shapes():
    from primus_turbo.flydsl.mega.fp8.quant import (
        quantize_grouped_weight_mxfp8,
        quantize_rowwise_mxfp8,
    )

    dev = "cuda:0"
    M, H, I, G = 512, 7168, 2048, 4
    x = torch.randn(M, H, device=dev, dtype=torch.bfloat16)
    xq, xs = quantize_rowwise_mxfp8(x)
    assert xq.shape == (M, H) and xq.dtype == torch.float8_e4m3fn
    assert xs.shape == (M, H // 32)

    w = torch.randn(G, 2 * I, H, device=dev, dtype=torch.bfloat16)
    wq, ws = quantize_grouped_weight_mxfp8(w)
    assert wq.shape == (G, 2 * I, H) and wq.dtype == torch.float8_e4m3fn
    assert ws.shape == (G, 2 * I, H // 32)
