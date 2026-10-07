###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""CPU wire-format tests: run this file directly to avoid GPU suite setup."""

import importlib.util
import sys
import unittest
from pathlib import Path

import torch

_PATH = Path(__file__).resolve().parents[3] / "primus_turbo/pytorch/core/mxfp4_comm.py"
_SPEC = importlib.util.spec_from_file_location("mxfp4_comm_under_test", _PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
MXFP4WireLayout = _MODULE.MXFP4WireLayout


class TestMXFP4WireLayout(unittest.TestCase):
    def test_rank_and_expert_order_with_padding(self):
        # N=96, K=160 exercise local padding and a different full-weight pad.
        layout = MXFP4WireLayout((3, 96, 160), scale_rounding_mode=2)
        shards, payloads = [], []
        for rank in range(3):
            components = []
            for index, shape in enumerate(layout.component_shapes):
                component = torch.full((*shape[:-1], shape[-1] + 8), 255, dtype=torch.uint8)
                values = torch.arange(torch.tensor(shape).prod().item()).reshape(shape)
                component[..., : shape[-1]] = (values + rank * 43 + index * 19).to(torch.uint8)
                components.append(component)
            shards.append(components)
            payload = layout.pack(components)
            self.assertEqual(payload.numel(), 17 * 3 * 96 * 160 // 16)
            payloads.append(payload)
        actual = layout.assemble(torch.stack(payloads))
        for index, shape in enumerate(layout.component_shapes):
            expected = torch.cat([s[index][..., : shape[-1]] for s in shards], dim=1 if index < 2 else 2)
            torch.testing.assert_close(actual[index][..., : expected.shape[-1]], expected, rtol=0, atol=0)
            self.assertEqual(actual[index][..., expected.shape[-1] :].count_nonzero().item(), 0)
            self.assertTrue(actual[index].is_contiguous())

    def test_bitcasts_do_not_convert_scale_values(self):
        layout = MXFP4WireLayout((1, 32, 32))
        components = [
            torch.arange(torch.tensor(s).prod().item()).to(torch.uint8).reshape(s)
            for s in layout.component_shapes
        ]
        expected = layout.pack(components)
        components[1] = components[1].view(torch.float8_e8m0fnu)
        components[3] = components[3].view(torch.float8_e8m0fnu)
        torch.testing.assert_close(layout.pack(components), expected, rtol=0, atol=0)

    def test_invalid_inputs_fail_before_transport(self):
        for shape in ((1, 31, 32), (1, 32, 33), (0, 32, 32), (32, 32)):
            with self.assertRaises(ValueError):
                MXFP4WireLayout(shape)
        layout = MXFP4WireLayout((1, 32, 32))
        for payload in (torch.zeros(layout.nbytes), torch.zeros(layout.nbytes - 1, dtype=torch.uint8)):
            with self.assertRaises(ValueError):
                layout.views(payload)
        with self.assertRaises(ValueError):
            layout.quantize(torch.zeros(layout.shape, dtype=torch.float32))
        with self.assertRaises(ValueError):
            layout.assemble(torch.zeros(0, layout.nbytes, dtype=torch.uint8))


if __name__ == "__main__":
    unittest.main()
