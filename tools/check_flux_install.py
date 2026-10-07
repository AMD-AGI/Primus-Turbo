###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Verify the installed FLUX Python sources and FlyDSL APIs without requiring a GPU.

Run after installation: python tools/check_flux_install.py <expected-git-revision>
"""

import argparse
import ast
import subprocess
from importlib.metadata import distribution
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("expected_revision", help="Full commit SHA used to build this installation")
    args = parser.parse_args()
    checkout = Path(__file__).resolve().parents[1]
    sha = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    assert sha == args.expected_revision, (sha, args.expected_revision)
    dist = distribution("primus-turbo")
    for rel, symbol in (
        ("flydsl/gemm/gemm_fp8_kernel_stock.py", "gemm_fp8_tensorwise_flydsl_kernel"),
        ("flydsl/gemm/gemm_fp8_kernel_flux.py", "gemm_fp8_tensorwise_flydsl_kernel"),
        ("flydsl/gemm/gemm_mxfp4_kernel_p3.py", "gemm_mxfp4_flydsl_kernel"),
        ("pytorch/kernels/gemm/gemm_fp4_impl_stock.py", "gemm_fp4_impl"),
        ("pytorch/kernels/gemm/gemm_fp4_impl_p3.py", "gemm_fp4_impl"),
        ("flydsl/quantization/mxfp4_quant_kernel_flux.py", "flydsl_quant_mxfp4_h16_dual"),
        ("flydsl/quantization/mxfp4_quant_kernel_pre_dual.py", "flydsl_quant_mxfp4_h16"),
        ("flydsl/gemm/gemm_a6w4_kernel_tuned.py", "gemm_a6w4"),
        ("flydsl/gemm/gemm_a6w4_kernel_v0.py", "gemm_a6w4"),
        ("flydsl/quantization/a6w4_quant.py", "quant_act_a6w4"),
        ("flydsl/quantization/a6w4_quant_triton.py", "quant_act_a6w4"),
        ("flydsl/attention/flash_attn_fwd_flux.py", "flux_attn_fwd"),
        ("flydsl/attention/flash_attn_bwd_flux.py", "flux_attn_bwd"),
    ):
        path = dist.locate_file("primus_turbo/" + rel)
        assert path.is_file(), f"{rel} not installed"
        definitions = {
            node.name
            for node in ast.parse(path.read_text()).body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        }
        assert symbol in definitions, f"{symbol} missing from {rel}"

    root = dist.locate_file("primus_turbo/flydsl")
    for rel in (
        "gemm/gemm_mxfp4_kernel.py",
        "gemm/gemm_fp8_kernel.py",
        "utils/gemm_helper.py",
        "utils/prims.py",
        "quantization/mxfp4_quant_kernel.py",
    ):
        assert (root / rel).is_file(), f"flydsl/{rel} missing from the image"
    from primus_turbo.flydsl.gemm import gemm_mxfp4_kernel as g

    assert g.gemm_mxfp4_flydsl_kernel

    _glu = [
        n for n in ("dense_glu_epi_quant_supported", "gemm_mxfp4_glu_quant_flydsl_kernel") if hasattr(g, n)
    ]
    if len(_glu) == 2:
        print("flydsl: fused GLU epilogue present")
    elif not _glu:
        print("flydsl: no fused GLU epilogue at this pin (expected on the 0.4.x line)")
    else:
        raise AssertionError(f"flydsl: partial GLU epilogue, only {_glu} present")

    from primus_turbo.flydsl.gemm.gemm_fp8_kernel import gemm_fp8_tensorwise_flydsl_kernel

    assert gemm_fp8_tensorwise_flydsl_kernel

    print("Primus-Turbo FLUX installation verified at", sha)


if __name__ == "__main__":
    main()
