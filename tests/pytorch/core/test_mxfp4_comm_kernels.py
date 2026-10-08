###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""CPU address tests: TRITON_INTERPRET=1 python test_mxfp4_comm_kernels.py.

The GPU training preflight separately checks this kernel on real gathered data.
"""

import importlib.util
import os
import random
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(os.getenv("TRITON_INTERPRET") == "1", "CPU Triton interpreter must be enabled")
class TestMXFP4GatherKernel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load(
            "primus_turbo.pytorch.kernels.quantization.mxfp4_comm",
            "primus_turbo/pytorch/kernels/quantization/mxfp4_comm.py",
        )
        cls.wire = load("mxfp4_wire_kernel_test", "primus_turbo/pytorch/core/mxfp4_comm.py")

    def test_uneven_ranks_padding_and_byte_values(self):
        rng = random.Random(3612)
        for shape in ((1, 32, 32), (3, 96, 160), (2, 128, 64), (1, 320, 96)):
            with self.subTest(shape=shape):
                layout = self.wire.MXFP4WireLayout(shape, 2)
                g, n, k = shape
                total = g * n // 32
                cuts = [0] + sorted(rng.randint(0, total) for _ in range(4)) + [total]
                ownership = [(a, b - a) for a, b in zip(cuts, cuts[1:])]
                offsets = [5, 14, 33, 2, 127]
                width = max(offset + count * 34 * k for (_, count), offset in zip(ownership, offsets))
                width = (width + 255) // 256 * 256
                storage = torch.full((len(ownership) * width,), 205, dtype=torch.uint8)
                pieces = []
                for rank, ((first, count), offset) in enumerate(zip(ownership, offsets)):
                    if not count:
                        continue
                    shard_layout = self.wire.MXFP4WireLayout((count, 32, k), 2)
                    payload = (torch.arange(shard_layout.nbytes) * 13 + rank * 31).to(torch.uint8)
                    storage[rank * width + offset : rank * width + offset + payload.numel()].copy_(payload)
                    pieces.append((first, count, payload))
                original = storage.clone()
                plan = self.wire.MXFP4StripGatherPlan(layout, ownership, offsets, width, "cpu")
                actual = plan.assemble(storage)
                expected = layout.assemble_strip_shards(pieces)
                for a, b in zip(actual, expected):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                    self.assertTrue(a.is_contiguous())
                torch.testing.assert_close(storage, original, rtol=0, atol=0)

    def test_shared_2d_reconstructs_both_orientations_and_scales(self):
        for shape in ((1, 32, 32), (3, 96, 160), (2, 128, 64), (1, 320, 96)):
            with self.subTest(shape=shape):
                g, n, k = shape
                layout = self.wire.MXFP4WireLayout(shape, 2, shared_2d=True)
                values = torch.randint(
                    0, 16, shape, dtype=torch.uint8, generator=torch.Generator().manual_seed(481)
                )
                tile_scales = torch.arange(g * n * k // 1024).to(torch.uint8).reshape(g, n // 32, k // 32)
                transposed = values.transpose(1, 2)
                expected = (
                    values[..., ::2] | (values[..., 1::2] << 4),
                    tile_scales.repeat_interleave(32, dim=1),
                    transposed[..., ::2] | (transposed[..., 1::2] << 4),
                    tile_scales.transpose(1, 2).repeat_interleave(32, dim=1),
                )
                packed = layout.pack(expected)
                self.assertEqual(packed.numel(), g * n * k * 513 // 1024)
                for decoded, reference in zip(layout.views(packed), expected):
                    torch.testing.assert_close(decoded, reference, rtol=0, atol=0)
                total = g * n // 32
                counts = [1, 0, total // 2, total - 1 - total // 2]
                ownership, first = [], 0
                for count in counts:
                    ownership.append((first, count))
                    first += count
                offsets = [3, 17, 91, 5]
                width = max(offset + count * (16 * k + k // 32) for offset, count in zip(offsets, counts))
                storage = torch.full((4 * width,), 205, dtype=torch.uint8)
                for rank, ((first, count), offset) in enumerate(zip(ownership, offsets)):
                    if not count:
                        continue
                    row = expected[0].reshape(total, 32, k // 2)[first : first + count]
                    scale = tile_scales.reshape(total, k // 32)[first : first + count]
                    payload = torch.cat((row.flatten(), scale.flatten()))
                    storage[rank * width + offset : rank * width + offset + payload.numel()] = payload
                original = storage.clone()
                plan = self.wire.MXFP4StripGatherPlan(layout, ownership, offsets, width, "cpu")
                for actual, reference in zip(plan.assemble(storage), expected):
                    torch.testing.assert_close(actual[..., : reference.shape[-1]], reference, rtol=0, atol=0)
                    self.assertEqual(actual[..., reference.shape[-1] :].count_nonzero().item(), 0)
                    self.assertTrue(actual.is_contiguous())
                torch.testing.assert_close(storage, original, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
