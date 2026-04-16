###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import ast
import importlib.util
import os
import sys
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SETUP_PATH = REPO_ROOT / "setup.py"


def _load_setup_module(monkeypatch):
    """Load setup.py with lightweight stubs for build-only imports."""

    tools_module = types.ModuleType("tools")
    tools_module.__path__ = []  # Mark as package to satisfy nested imports.

    build_ext_module = types.ModuleType("tools.build_ext")

    class _FakeTurboBuildExt:
        @classmethod
        def with_options(cls, **_kwargs):
            return cls

    build_ext_module.TurboBuildExt = _FakeTurboBuildExt
    build_ext_module._join_rocm_home = lambda *paths: os.path.join("/opt/rocm", *paths)

    build_utils_module = types.ModuleType("tools.build_utils")
    build_utils_module.HIPExtension = lambda *args, **kwargs: (args, kwargs)
    build_utils_module.find_rocshmem_library = lambda: None
    build_utils_module.get_gpu_arch = lambda: "gfx942"

    monkeypatch.setitem(sys.modules, "tools", tools_module)
    monkeypatch.setitem(sys.modules, "tools.build_ext", build_ext_module)
    monkeypatch.setitem(sys.modules, "tools.build_utils", build_utils_module)

    spec = importlib.util.spec_from_file_location("setup_under_test", SETUP_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_is_package_installed_returns_true_with_distribution(monkeypatch):
    setup_module = _load_setup_module(monkeypatch)

    called = []

    def fake_distribution(package_name):
        called.append(package_name)
        return object()

    monkeypatch.setattr(setup_module.importlib.metadata, "distribution", fake_distribution)

    assert setup_module.is_package_installed("amd-aiter") is True
    assert called == ["amd-aiter"]


def test_is_package_installed_returns_false_when_distribution_missing(monkeypatch):
    setup_module = _load_setup_module(monkeypatch)

    def fake_distribution(_package_name):
        raise setup_module.importlib.metadata.PackageNotFoundError

    monkeypatch.setattr(setup_module.importlib.metadata, "distribution", fake_distribution)

    assert setup_module.is_package_installed("amd-aiter") is False


def test_setup_script_checks_amd_aiter_distribution_name():
    setup_tree = ast.parse(SETUP_PATH.read_text(encoding="utf-8"))

    queried_packages = []
    for node in ast.walk(setup_tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "is_package_installed"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            queried_packages.append(node.args[0].value)

    assert "amd-aiter" in queried_packages
    assert "aiter" not in queried_packages
