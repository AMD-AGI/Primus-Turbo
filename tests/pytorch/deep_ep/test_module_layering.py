###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The FlyDSL DeepEP V1 and V2 share only neutral modules, never each other's code.

Read from the source with ``ast``, so it needs neither a GPU nor an import of primus_turbo.
"""

import ast
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
KERNELS = REPO / "primus_turbo" / "flydsl" / "deep_ep"
KERNEL_PACKAGE = "primus_turbo.flydsl.deep_ep"
HOST_RUNTIME = REPO / "primus_turbo" / "pytorch" / "deep_ep" / "flydsl_intranode.py"


def _imported_modules(path: Path) -> set:
    modules = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
        elif isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
    return modules


def _kernel_deps(name: str) -> set:
    prefix = KERNEL_PACKAGE + "."
    return {m[len(prefix) :] for m in _imported_modules(KERNELS / f"{name}.py") if m.startswith(prefix)}


class KernelModuleLayeringTestCase(unittest.TestCase):
    """V1 and V2 build on ``arch`` / ``common`` / ``notify_kernel``, which build on neither."""

    ALLOWED_DEPS = {
        "arch": set(),
        "common": {"arch"},
        "notify_kernel": {"common"},
        "intranode_v1_kernel": {"arch", "common"},
        "intranode_v2_kernel": {"arch", "common", "notify_kernel"},
    }

    def test_kernel_modules_import_only_the_shared_ones(self):
        for name, allowed in self.ALLOWED_DEPS.items():
            deps = _kernel_deps(name)
            assert deps <= allowed, f"{name} imports {sorted(deps - allowed)}"


class HostRuntimeLayeringTestCase(unittest.TestCase):
    """In ``flydsl_intranode.py`` only the V1 runtime uses the V1 kernels, and only the V2 runtime and
    its heap size the V2 kernels; the shared base class uses neither."""

    VERSION_OF = {
        "FlyDSLIntranodeV1Runtime": "intranode_v1_kernel",
        "FlyDSLIntranodeV2Runtime": "intranode_v2_kernel",
        "v2_heap_bytes": "intranode_v2_kernel",
    }

    def test_each_runtime_uses_only_its_own_kernels(self):
        tree = ast.parse(HOST_RUNTIME.read_text())
        imported = {
            version: {
                alias.asname or alias.name
                for node in tree.body
                if isinstance(node, ast.ImportFrom) and node.module == f"{KERNEL_PACKAGE}.{version}"
                for alias in node.names
            }
            for version in ("intranode_v1_kernel", "intranode_v2_kernel")
        }
        assert all(imported.values()), imported
        defs = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))]
        assert {node.name for node in defs} >= set(self.VERSION_OF) | {"FlyDSLIntranodeRuntime"}
        for node in defs:
            used = {name.id for name in ast.walk(node) if isinstance(name, ast.Name)}
            for version, names in imported.items():
                if self.VERSION_OF.get(node.name) != version:
                    assert not used & names, f"{node.name} uses {sorted(used & names)} of {version}"


if __name__ == "__main__":
    unittest.main()
