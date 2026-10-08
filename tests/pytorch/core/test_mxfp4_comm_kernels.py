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


if __name__ == "__main__":
    unittest.main()
