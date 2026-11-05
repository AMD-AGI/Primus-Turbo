###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Build parameterized ``@triton.jit`` mods from generated source.

Triton keys compiled kernels partly on a JIT function's *source*, so two closures made
from one ``def`` with different captured constants hash alike and the second silently
reuses the first's binary. Baking each constant into the source as a literal keeps the
variants distinct. The source must live in a real file: Triton reads it with
``inspect.getsource``, which rejects code created by ``exec`` of a string.
"""

import importlib.util
import os
import sys
import tempfile

__all__ = ["build_jit", "literal_tag"]

_GENERATED_DIR = None


def literal_tag(value) -> str:
    """Identifier-safe tag for a number, used to keep generated function names unique."""
    return repr(value).replace("-", "neg").replace(".", "p").replace("+", "")


def build_jit(src: str, names, tag: str):
    """Write ``src`` (``@triton.jit`` functions) to a module file and return ``names`` from it."""
    global _GENERATED_DIR
    if _GENERATED_DIR is None:
        _GENERATED_DIR = tempfile.mkdtemp(prefix="primus_turbo_flex_mods_")
    module_name = f"_primus_turbo_flex_mod_{tag}"
    path = os.path.join(_GENERATED_DIR, f"{module_name}.py")
    header = "import triton\nimport triton.language as tl\nfrom triton.language.extra import libdevice\n"
    with open(path, "w") as fh:
        fh.write(header + src)
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module  # triton resolves a JIT function's module by name
    spec.loader.exec_module(module)
    if isinstance(names, str):
        return getattr(module, names)
    return tuple(getattr(module, n) for n in names)
