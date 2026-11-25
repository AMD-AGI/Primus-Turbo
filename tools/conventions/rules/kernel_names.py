###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""NAME: kernel file names (CONTRIBUTING.md, "Kernel File Naming Convention")."""

from __future__ import annotations

import sys

from tools.conventions.core import PKG, REPO_ROOT, Finding, Tree, rule

sys.path.insert(0, str(REPO_ROOT))
try:
    from primus_turbo.arch import KNOWN_ARCHS
finally:
    sys.path.pop(0)

# The layers the naming convention is written for: <layer>/<op>/<file>.py
LAYERS = (f"{PKG}/pytorch/kernels/", f"{PKG}/flydsl/", f"{PKG}/triton/")
ROLES = ("impl", "kernel", "utils", "heuristic", "registry")
# Directories of a layer that are not an operation: code several operations share.
SHARED_DIRS = ("utils", "autotune")


@rule("NAME001", "a kernel file ends with its role", baseline=True)
def kernel_file_ends_with_role(tree: Tree):
    """A Python file directly in an operation directory of primus_turbo/pytorch/kernels/,
    primus_turbo/flydsl/ or primus_turbo/triton/ is named <op>_<dtype>_<role>.py: its last name
    segment is its role (impl, kernel, utils, heuristic, registry), after which only an
    architecture tag may follow (<...>_kernel_gfx1250.py). No file sits at the root of a layer.

    Why: the role says what a file is without opening it, and the fixed position lets tools and
    agents find the frontend (impl), the kernels and the tuning tables of an operation by name.

    Not checked here: files inside a kernel package (a subdirectory of the operation), shared
    directories (utils/, autotune/), __init__.py, and the <op> / <dtype> parts of the name.

    Fix: rename a new file so that it ends with its role. Files that predate the rule are listed
    in tools/conventions/baseline.txt; renaming one of those is a change of its own (imports, and
    for a FlyDSL launcher its disk-cache key), not something to fold into another change.
    """
    for path in tree.under(*LAYERS):
        layer = next(prefix for prefix in LAYERS if path.startswith(prefix))
        parts = path[len(layer) :].split("/")
        name = parts[-1]
        if not name.endswith(".py") or name == "__init__.py":
            continue
        if len(parts) == 1:
            yield Finding(
                "NAME001",
                path,
                0,
                f"is at the root of {layer}; a kernel file lives in an operation directory",
            )
            continue
        if parts[0] in SHARED_DIRS or len(parts) > 2:
            continue
        segments = name[: -len(".py")].split("_")
        if segments[-1] in KNOWN_ARCHS:
            segments = segments[:-1]
        if not segments or segments[-1] not in ROLES:
            yield Finding(
                "NAME001",
                path,
                0,
                f"does not end with a role ({', '.join(ROLES)}), optionally followed by an architecture "
                "tag: <op>_<dtype>_<role>[_<arch>].py",
            )
