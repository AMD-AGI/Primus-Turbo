###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""CPU import checks; each recipe gets a fresh custom-op registration namespace.

Run with: python tests/test_flux_selection.py
Requires the FlyDSL/Triton runtime, but no GPU or Turbo native extension.
"""

import importlib.util
import os
import subprocess
import sys
import unittest


@unittest.skipUnless(importlib.util.find_spec("flydsl"), "requires FlyDSL")
class FluxSelectionTest(unittest.TestCase):
    def test_recipes(self):
        cases = [
            ({}, ("stock", "stock", "stock", "stock", "tuned")),
            (
                {
                    "FLUX_FP4_PASSES": "dgrad,wgrad",
                    "FLUX_FP8_FUSE_BIAS_EPILOGUE": "1",
                    "FLUX_ATTN_FLYDSL": "1",
                },
                ("p3", "flux", "flux", "flux", "tuned"),
            ),
            (
                {"FLUX_FP4_PASSES": "dgrad,wgrad", "FLUX_FP4_H16_STOCK": "1"},
                ("stock", "pre_dual", "stock", "stock", "tuned"),
            ),
            (
                {"FLUX_FP4_PASSES": "dgrad,wgrad", "FLUX_MXFP4_P3_GEMM": "0", "FLUX_A6W4_GEMM_V0": "1"},
                ("stock", "flux", "stock", "stock", "v0"),
            ),
            (
                {"FLUX_FP4_PASSES": "dgrad,wgrad", "FLUX_FP4_DISPATCH": "host_dispatch"},
                ("p3", "flux", "stock", "stock", "tuned"),
            ),
        ]
        for flags, expected in cases:
            with self.subTest(flags=flags):
                env = {k: v for k, v in os.environ.items() if not k.startswith("FLUX_")}
                env.update(flags)
                code = """
import importlib
import sys
modules = [
    'gemm.gemm_mxfp4_kernel', 'quantization.mxfp4_quant_kernel',
    'gemm.gemm_fp8_kernel', 'attention.flash_attn_fwd', 'gemm.gemm_a6w4_kernel',
]
for name, suffix in zip(modules, sys.argv[1:]):
    full = 'primus_turbo.flydsl.' + name
    m = importlib.import_module(full)
    assert m._implementation.__name__ == full + '_' + suffix, (name, suffix)
    assert not any(full + '_' + other in sys.modules
                   for other in ('stock', 'flux', 'p3', 'pre_dual', 'v0', 'tuned')
                   if other != suffix), full
importlib.import_module('primus_turbo.flydsl.attention.flash_attn_bwd')
importlib.import_module('primus_turbo.flydsl.quantization.a6w4_quant')
importlib.import_module('primus_turbo.flydsl.quantization.a6w4_quant_triton')
"""
                result = subprocess.run(
                    [sys.executable, "-c", code, *expected], env=env, capture_output=True, text=True
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
