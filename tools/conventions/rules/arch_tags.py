###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""ARCH: the architecture tag in paths.

Architecture-specific code stays in the directory of its operation and says which architecture
it is for with the gfx ID as a whole segment of a file or directory name, segments being
separated by ``_`` or ``.`` (``grouped_gemm_bf16_kernel_gfx1250.py``, ``attention/gfx1250_mla/``).
"""

from __future__ import annotations

import re
import sys
from typing import Dict, Iterator, List, Tuple

from tools.conventions.core import PKG, REPO_ROOT, Finding, Tree, rule

sys.path.insert(0, str(REPO_ROOT))
try:
    from primus_turbo.arch import PRODUCTS
finally:
    sys.path.pop(0)

# The architectures whose tagged code must be registered. Add one here once its tagged files are
# owned by registry entries (the older gfx942 / gfx950 kernels are not registered yet).
ENFORCED_ARCHS = ("gfx1250",)

CODE_TREES = (f"{PKG}/", "csrc/")
VERIFY_TREES = ("tests/", "benchmark/")
BUILD_PRODUCTS = (f"{PKG}/lib/",)


def tag_prefix(path: str, arch: str) -> str:
    """The path up to and including its first component that carries ``arch`` as a whole name
    segment (``a/gfx1250_mla/b.py`` -> ``a/gfx1250_mla``), ``""`` when none does."""
    parts = path.split("/")
    for i, part in enumerate(parts):
        if arch in re.split(r"[_.]", part):
            return "/".join(parts[: i + 1])
    return ""


def grouped(found: Dict[str, List[str]]) -> Iterator[Tuple[str, str, str]]:
    """(first file, prefix, " (and N more files below it)" or "") for {prefix: files}."""
    for prefix, files in sorted(found.items()):
        more = f" (and {len(files) - 1} more file(s) below it)" if len(files) > 1 else ""
        yield sorted(files)[0], prefix, more


@rule("ARCH001", "architecture-tagged code is owned by a kernel registry entry")
def tagged_code_is_registered(tree: Tree):
    """Every file under primus_turbo/ and csrc/ whose path carries an architecture tag must be
    owned by a kernel registry entry for that architecture.

    Why: the registry is the complete, machine-readable answer to "where is this architecture's
    code, when does it apply and what verifies it". Code that is tagged but not registered is
    invisible to primus_turbo.capabilities, to reviewers routed by the registry and to an agent
    looking for the tests of a kernel.

    Fix: add the path to an entry of the operation's registry file
    (primus_turbo/pytorch/kernels/<op>/<op>_registry.py) as its 'kernel' (module or package),
    'impl' (module) or one of its 'sources' (any repository path); or add a new entry there,
    with its tests. `python -m primus_turbo.registry --arch <arch>` shows the entries.
    """
    entries, _ = tree.registry
    for arch in ENFORCED_ARCHS:
        owners = [k for k in entries if arch in k.arch]
        found: Dict[str, List[str]] = {}
        for path in tree.under(*CODE_TREES):
            prefix = tag_prefix(path, arch)
            if prefix and not path.startswith(BUILD_PRODUCTS) and not any(k.owns(path) for k in owners):
                found.setdefault(prefix, []).append(path)
        for first, prefix, more in grouped(found):
            yield Finding(
                "ARCH001",
                first,
                0,
                f"{prefix}{more} carries the {arch} tag but no kernel registry entry for {arch} owns it; "
                f"add it to an entry of the operation's <op>_registry.py ('kernel', 'impl' or 'sources')",
            )


@rule("ARCH002", "the architecture tag is a whole name segment")
def tag_is_a_whole_segment(tree: Tree):
    """A path that names an architecture names it as a whole segment of a file or directory name,
    segments being separated by '_' or '.': attention/gfx1250_mla/, test_mega_bf16_gfx1250_gemm.py.

    Why: the tag is what tools match on. tests/conftest.py runs a test file on gfx1250 when its
    name carries the tag, setup.py filters C++ sources by it, and ARCH001 / REG004 find
    unregistered code and tests by it. 'gfx1250mla' or 'Gfx1250' is seen by none of them.

    Fix: rename so that the gfx ID stands between '_', '.' or '/' in lower case.
    """
    for arch in ENFORCED_ARCHS:
        found: Dict[str, List[str]] = {}
        for path in tree.under(*CODE_TREES, *VERIFY_TREES):
            parts = path.split("/")
            for i, part in enumerate(parts):
                if arch in part.lower() and arch not in re.split(r"[_.]", part):
                    found.setdefault("/".join(parts[: i + 1]), []).append(path)
                    break
        for first, prefix, more in grouped(found):
            yield Finding(
                "ARCH002",
                first,
                0,
                f"{prefix}{more}: '{arch}' must be a whole name segment (between '_', '.' or '/'), "
                f"as in attention/{arch}_mla/ or grouped_gemm_bf16_kernel_{arch}.py",
            )


@rule("ARCH003", "paths use the gfx ID, not the product name")
def no_product_name_in_paths(tree: Tree):
    """Paths under primus_turbo/, csrc/, tests/ and benchmark/ name an architecture by its gfx ID
    (gfx1250), never by a product name (MI455X).

    Why: one spelling per architecture. Several products share an architecture, the tools match
    on the gfx ID only (see ARCH002), and a file named after the product would be neither
    registered nor run on that architecture.

    Fix: rename the path to use the gfx ID as its tag.
    """
    for arch in ENFORCED_ARCHS:
        aliases = sorted({product.lower().rstrip("x") for product in PRODUCTS.get(arch, ())})
        found: Dict[str, List[str]] = {}
        for path in tree.under(*CODE_TREES, *VERIFY_TREES):
            parts = path.split("/")
            for i, part in enumerate(parts):
                if any(alias in part.lower() for alias in aliases):
                    found.setdefault("/".join(parts[: i + 1]), []).append(path)
                    break
        for first, prefix, more in grouped(found):
            yield Finding(
                "ARCH003",
                first,
                0,
                f"{prefix}{more}: use the gfx ID '{arch}' in paths, not the product name",
            )
