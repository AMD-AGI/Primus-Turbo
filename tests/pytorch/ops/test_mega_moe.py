###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Skeleton tests for the fused Mega MoE FFN frontend.

These tests exercise the host-side layout helpers and weight
pre-processing without depending on a working device kernel.  The
distributed end-to-end test (``test_mega_moe_e2e``) is parked as a
skip until the GPU kernel lands.
"""

from __future__ import annotations

import unittest

import torch

from primus_turbo.pytorch.kernels.mega_moe import (
    MegaMoEConfig,
    get_mega_moe_config,
    get_symm_buffer_layout,
    get_token_alignment,
    transform_l1_weights_for_mega_moe,
    transform_l2_weights_for_mega_moe,
)


def _cpp_extension_has(symbol: str) -> bool:
    try:
        ns = torch.ops.primus_turbo_cpp_extension
    except (AttributeError, RuntimeError):
        return False
    return hasattr(ns, symbol)


def _make_l1_weights(num_groups: int, n: int, k: int):
    """Generate dummy FP4 packed weight + UE8M0 SF tensors for L1."""
    # Pack two FP4 lanes per byte → K is halved for the weight tensor.
    w = torch.randint(0, 255, (num_groups, n, k // 2), dtype=torch.int8)
    sf = torch.randint(0, 255, (num_groups, n, k // 32), dtype=torch.int32)
    return w, sf


def _make_l2_weights(num_groups: int, n: int, k: int):
    w = torch.randint(0, 255, (num_groups, n, k // 2), dtype=torch.int8)
    sf = torch.randint(0, 255, (num_groups, n, k // 32), dtype=torch.int32)
    return w, sf


class MegaMoELayoutTest(unittest.TestCase):
    def setUp(self) -> None:
        self.num_ranks = 8
        self.num_experts = 64
        self.num_max_tokens_per_rank = 128
        self.num_topk = 6
        self.hidden = 512
        self.intermediate_hidden = 256

    @unittest.skipUnless(
        _cpp_extension_has("mega_moe_get_token_alignment"),
        "primus_turbo C extension is not built with mega_moe symbols.",
    )
    def test_token_alignment(self) -> None:
        align = get_token_alignment()
        self.assertGreater(align, 0)
        self.assertEqual(align & (align - 1), 0, "token alignment should be a power of two")

    @unittest.skipUnless(
        _cpp_extension_has("mega_moe_get_symm_buffer_layout"),
        "primus_turbo C extension is not built with mega_moe symbols.",
    )
    def test_symm_buffer_layout_offsets_monotonic(self) -> None:
        layout = get_symm_buffer_layout(
            num_ranks=self.num_ranks,
            num_experts=self.num_experts,
            num_max_tokens_per_rank=self.num_max_tokens_per_rank,
            num_topk=self.num_topk,
            hidden=self.hidden,
            intermediate_hidden=self.intermediate_hidden,
        )
        offsets = [
            layout.workspace_offset,
            layout.input_x_offset,
            layout.input_x_sf_offset,
            layout.input_topk_idx_offset,
            layout.input_topk_weights_offset,
            layout.l1_pool_x_offset,
            layout.l1_pool_x_sf_offset,
            layout.l1_pool_weights_offset,
            layout.l2_pool_x_offset,
            layout.l2_pool_x_sf_offset,
            layout.combine_buffer_offset,
        ]
        for prev, nxt in zip(offsets, offsets[1:]):
            self.assertLessEqual(prev, nxt, "buffer offsets must be monotonically increasing")
        self.assertGreater(layout.total_bytes, layout.combine_buffer_offset)
        self.assertGreater(layout.num_max_pool_tokens, 0)
        self.assertGreater(layout.num_padded_sf_pool_tokens, 0)


class MegaMoEConfigTest(unittest.TestCase):
    def test_config_pick_returns_valid_struct(self) -> None:
        cfg = get_mega_moe_config(
            num_ranks=8,
            num_experts=64,
            num_experts_per_rank=8,
            num_max_tokens_per_rank=4096,
            num_tokens=4096,
            num_topk=8,
            hidden=7168,
            intermediate_hidden=2048,
            num_padded_sf_pool_tokens=4096,
            num_max_pool_tokens=4096,
        )
        self.assertIsInstance(cfg, MegaMoEConfig)
        self.assertGreater(cfg.block_m, 0)
        self.assertEqual(cfg.block_n, 128)
        self.assertEqual(cfg.block_k, 128)
        self.assertGreaterEqual(cfg.num_experts_per_wave, 1)
        self.assertGreaterEqual(cfg.num_stages, 2)


class MegaMoEWeightTransformTest(unittest.TestCase):
    def test_l1_transform_preserves_shape(self) -> None:
        w, sf = _make_l1_weights(num_groups=4, n=256, k=512)
        out_w, out_sf = transform_l1_weights_for_mega_moe((w, sf))
        self.assertEqual(out_w.shape, w.shape)
        self.assertEqual(out_sf.shape, sf.shape)
        # Weight tensor should still be packed FP4.
        self.assertEqual(out_w.dtype, torch.int8)
        self.assertEqual(out_sf.dtype, torch.int32)

    def test_l2_transform_preserves_shape(self) -> None:
        w, sf = _make_l2_weights(num_groups=4, n=256, k=512)
        out_w, out_sf = transform_l2_weights_for_mega_moe((w, sf))
        self.assertEqual(out_w.shape, w.shape)
        self.assertEqual(out_sf.shape, sf.shape)
        # L2 transform only permutes the SF tensor, weight is unchanged.
        torch.testing.assert_close(out_w, w)


@unittest.skip("Mega MoE kernel implementation pending; skeleton smoke test only.")
class MegaMoEEndToEndTest(unittest.TestCase):
    """End-to-end test: dispatch + L1 GEMM + SwiGLU + L2 GEMM + combine.

    Requires a multi-process launch and a working device kernel.  Kept
    here so the skeleton documents the intended call sequence:

        buffer = get_symm_buffer_for_mega_moe(group, ...)
        l1, l2 = transform_weights_for_mega_moe(l1, l2)
        buffer.x[:n].copy_(x_fp8)
        buffer.x_sf[:n].copy_(x_sf)
        buffer.topk_idx[:n].copy_(topk_idx)
        buffer.topk_weights[:n].copy_(topk_weights)
        y = torch.empty((n, hidden), dtype=torch.bfloat16, device='cuda')
        fp8_mega_moe(y, l1, l2, buffer)
    """

    def test_e2e(self) -> None:
        self.fail("Device kernel not implemented yet.")


if __name__ == "__main__":
    unittest.main()
