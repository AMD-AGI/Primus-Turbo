###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import importlib.util
from pathlib import Path

import pytest

AUDIT_PATH = Path(__file__).resolve().parents[2] / "tools" / "ci" / "check_gluon_vendor.py"
SPEC = importlib.util.spec_from_file_location("check_gluon_vendor", AUDIT_PATH)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)

PROVENANCE = "\n".join(AUDIT.PROVENANCE_MARKERS)
VALID_KERNEL = f"""{PROVENANCE}

VGPR_ONLY_FN_ATTRS = (("amdgpu-agpr-alloc", "0,0"),)
NO_DISPATCH_ID_FN_ATTRS = (("amdgpu-no-dispatch-id", ""),)
VGPR_ONLY_PRELOAD_FN_ATTRS = VGPR_ONLY_FN_ATTRS + NO_DISPATCH_ID_FN_ATTRS
VGPR_ONLY_PRELOAD_MAX_ILP_FN_ATTRS = (
    VGPR_ONLY_PRELOAD_FN_ATTRS + (("amdgpu-sched-strategy", "max-ilp"),)
)

def gluon_attn_fwd():
    pass

def gluon_attn_fwd_short_causal_classes():
    pass

def configs():
    return [
        triton.Config({{"BLOCK_M": 256, "BLOCK_N": 32, "PRE_LOAD_V": False,
                       "NUM_STAGES": 2, "waves_per_eu": 2,
                       "llvm_fn_attrs": VGPR_ONLY_FN_ATTRS}}, num_warps=8),
        triton.Config({{"BLOCK_M": 256, "BLOCK_N": 64, "PRE_LOAD_V": False,
                       "NUM_STAGES": 3, "waves_per_eu": 2,
                       "llvm_fn_attrs": VGPR_ONLY_FN_ATTRS}}, num_warps=8),
        triton.Config({{"BLOCK_M": 256, "BLOCK_N": 64, "PRE_LOAD_V": False,
                       "NUM_STAGES": 4, "waves_per_eu": 2,
                       "llvm_fn_attrs": VGPR_ONLY_PRELOAD_FN_ATTRS}}, num_warps=8),
        triton.Config({{"BLOCK_M": 256, "BLOCK_N": 64, "PRE_LOAD_V": False,
                       "NUM_STAGES": 4, "waves_per_eu": 2,
                       "llvm_fn_attrs": VGPR_ONLY_PRELOAD_MAX_ILP_FN_ATTRS}}, num_warps=8),
    ]
"""


def _write_vendor_fixture(tmp_path: Path, kernel_source: str) -> None:
    vendor_dir = tmp_path / "primus_turbo" / "gluon" / "attention"
    vendor_dir.mkdir(parents=True)
    (vendor_dir / "f16_fa_gfx950_common.py").write_text(f"{PROVENANCE}\n", encoding="utf-8")
    (vendor_dir / "f16_fa_gfx950_rotated_4cluster.py").write_text(kernel_source, encoding="utf-8")


def test_audit_accepts_exact_pinned_optimized_candidates(tmp_path, monkeypatch, capsys):
    _write_vendor_fixture(tmp_path, VALID_KERNEL)
    monkeypatch.setattr(AUDIT, "REPO_ROOT", tmp_path)

    assert AUDIT.main() == 0
    assert "passed" in capsys.readouterr().out


def test_audit_rejects_wrong_optimized_candidate_attributes(tmp_path, monkeypatch, capsys):
    broken = VALID_KERNEL.replace(
        '"llvm_fn_attrs": VGPR_ONLY_PRELOAD_MAX_ILP_FN_ATTRS',
        '"llvm_fn_attrs": VGPR_ONLY_PRELOAD_FN_ATTRS',
    )
    _write_vendor_fixture(tmp_path, broken)
    monkeypatch.setattr(AUDIT, "REPO_ROOT", tmp_path)

    assert AUDIT.main() == 1
    assert "optimized AMDGPU autotune candidates do not match" in capsys.readouterr().err


@pytest.mark.parametrize(
    "loader_source",
    (
        'LLVM_PASS_PLUGIN_FILE = "/tmp/pass.so"',
        'ctypes.PyDLL("/tmp/pass.so")',
        "sys.setdlopenflags(sys.getdlopenflags())",
        'import ctypes as c\nc.CDLL("/tmp/pass.so")',
        'from ctypes import PyDLL as load\nload("/tmp/pass.so")',
        "from sys import setdlopenflags as set_flags\nset_flags(0)",
        "import sys as system\nsystem.setdlopenflags(0)",
    ),
)
def test_audit_rejects_plugin_and_dynamic_loader_variants(tmp_path, monkeypatch, capsys, loader_source):
    _write_vendor_fixture(tmp_path, f"{VALID_KERNEL}\n{loader_source}\n")
    monkeypatch.setattr(AUDIT, "REPO_ROOT", tmp_path)

    assert AUDIT.main() == 1
    assert "forbidden" in capsys.readouterr().err
