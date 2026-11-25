###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The kernel registry: which kernels Primus-Turbo has, for which architecture, where each one
is, when it applies and how it is verified.

**Where it is written.** Not in one table: every operation keeps its own registry file next
to its dispatch code, ``primus_turbo/pytorch/kernels/<op>/<op>_registry.py`` (any
``*_registry.py`` in the package is one). A pull request that adds a kernel to an operation
edits that operation's file only, so work on different operations does not meet in a shared
file. A registry file is data: a docstring and one assignment, ``KERNELS = (...)``, of dict
literals. It is parsed, never imported, so reading it needs neither torch nor a GPU and cannot
fail on a kernel's import.

**One entry** describes one implementation of the operation::

    {
        "name": "attention.flydsl.mla",       # what primus_turbo.capabilities answers to
        "summary": "multi-head latent attention at the DeepSeek-V3 head dims",
        "arch": ("gfx1250",),                 # gfx IDs it is for
        "backend": "flydsl",                  # a BackendType, lower case
        "dtype": ("bf16",),                   # the formats it computes in; left out when it has none
        "kernel": "primus_turbo.flydsl.attention.gfx1250_mla",   # the kernels (module or package)
        "impl": "primus_turbo.pytorch.kernels.attention.attention_flydsl_gfx1250_mla_impl",
        "gates": ("<module>:<symbol>",),      # the code that decides when it applies
        "tests": ("tests/pytorch/ops/test_attention_flydsl_gfx1250_mla.py",),
        # optional: "sources" (further repository paths), "products" (build products it needs at
        # run time, globs relative to the package), "provides" ({sub-feature: symbol the kernel
        # module defines}), "benchmarks"
    }

``name`` may repeat across entries whose ``arch`` do not overlap: the same feature, implemented
per architecture. The operation of an entry is the registry file's name (``<op>_registry.py``).

**Who reads it.**

  * ``primus_turbo.capabilities``: code outside Primus-Turbo asks whether a feature is there
    instead of knowing a path.
  * ``tools/conventions`` (pre-commit, CI): everything an entry names exists, every path that
    carries the tag of an architecture whose code must be registered (gfx1250 today) is owned by
    an entry, every entry has tests and they run on the entry's architecture. See
    ``python tools/conventions --list-rules``.
  * People and agents: ``python -m primus_turbo.registry`` lists the entries; ``--arch``,
    ``--op``, ``--name`` filter; ``--owning <path>...`` gives the entries that own the given
    files; ``--tests`` / ``--benchmarks`` print what verifies the selection (feed it to pytest);
    ``--json`` prints the entries in full.

Importing this module imports neither torch nor a kernel module, and neither does a query that
names the architecture. A query that leaves it out asks the current device, which imports torch
and initializes its device runtime: a launch script names the architecture.
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from primus_turbo.arch import KNOWN_ARCHS, device_arch, normalize_arch

__all__ = [
    "BACKENDS",
    "DTYPES",
    "FILE_SUFFIX",
    "Kernel",
    "Problem",
    "features",
    "find",
    "has",
    "kernels",
    "load",
    "module_file",
    "owners",
    "present",
]

FILE_SUFFIX = "_registry.py"

# primus_turbo.pytorch.core.backend.BackendType, lower case (a test holds the two together).
BACKENDS = ("aiter", "ck", "deep_ep", "flydsl", "gluon", "hipblaslt", "hipkittens", "triton", "turbo")
DTYPES = ("bf16", "fp16", "fp32", "fp8", "fp4", "mxfp8", "mxfp4")

_PKG_DIR = Path(__file__).resolve().parent
_PKG = _PKG_DIR.name

_NAME_RE = re.compile(r"[a-z0-9_]+(\.[a-z0-9_]+)*")
_MODULE_RE = re.compile(rf"{_PKG}(\.[A-Za-z_][A-Za-z0-9_]*)+")
_SYMBOL_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

_REQUIRED_STR = ("name", "summary", "backend")
_REQUIRED_SEQ = ("arch", "tests")
_OPTIONAL_STR = ("kernel", "impl")
_OPTIONAL_SEQ = ("dtype", "gates", "sources", "products", "benchmarks")
_FIELDS = _REQUIRED_STR + _REQUIRED_SEQ + _OPTIONAL_STR + _OPTIONAL_SEQ + ("provides",)


@dataclass(frozen=True)
class Kernel:
    """One registry entry. ``op``, ``file`` and ``line`` say where it was read from."""

    name: str
    summary: str
    arch: Tuple[str, ...]
    backend: str
    tests: Tuple[str, ...]
    dtype: Tuple[str, ...] = ()
    kernel: str = ""
    impl: str = ""
    gates: Tuple[str, ...] = ()
    sources: Tuple[str, ...] = ()
    products: Tuple[str, ...] = ()
    benchmarks: Tuple[str, ...] = ()
    provides: Dict[str, str] = field(default_factory=dict)
    op: str = ""
    file: str = ""  # the registry file, relative to the directory that holds the package
    line: int = 0
    valid: bool = True  # False: load() reported a problem in it; lint reads it, queries do not use it

    def __hash__(self):
        return hash((self.name, self.arch, self.file, self.line))

    def modules(self) -> Tuple[str, ...]:
        """The modules the entry consists of: ``kernel`` and ``impl``, those that are given."""
        return tuple(m for m in (self.kernel, self.impl) if m)

    def feature_names(self) -> Tuple[str, ...]:
        """The names ``primus_turbo.capabilities`` answers to: the entry's and its sub-features'."""
        return (self.name, *(f"{self.name}.{extra}" for extra in self.provides))

    def owns(self, path: str) -> bool:
        """Whether a file of the repository (``primus_turbo/...``, ``csrc/...``, ``tests/...``)
        belongs to this entry: its code (also the modules of its gates), its build products, its
        tests or its benchmarks. A module that is a package owns what is below it."""
        for module in (*self.modules(), *(gate.partition(":")[0] for gate in self.gates)):
            base = module.replace(".", "/")
            if path == base + ".py" or path.startswith(base + "/"):
                return True
        for source in self.sources:
            source = source.rstrip("/")
            if path == source or path.startswith(source + "/"):
                return True
        if path.startswith(_PKG + "/") and any(
            fnmatch.fnmatchcase(path[len(_PKG) + 1 :], pattern) for pattern in self.products
        ):
            return True
        return any(path == t.split("::", 1)[0] for t in (*self.tests, *self.benchmarks))


@dataclass(frozen=True)
class Problem:
    """Something wrong in a registry file."""

    file: str
    line: int
    message: str


def _strings(value, allow_empty: bool) -> Optional[Tuple[str, ...]]:
    """``value`` as a tuple of non-empty strings, None when it is not one."""
    if not isinstance(value, (tuple, list)) or not all(isinstance(v, str) and v for v in value):
        return None
    return tuple(value) if (value or allow_empty) else None


def _entry(raw, op: str, file: str, line: int) -> Tuple[Optional[Kernel], List[str]]:
    """(the entry, what is wrong with it); the entry is None when it cannot be used at all."""
    if not isinstance(raw, dict):
        return None, ["an entry is a dict literal"]
    errors = [f"unknown field {key!r} (fields: {', '.join(_FIELDS)})" for key in raw if key not in _FIELDS]
    values = {}
    for key in _REQUIRED_STR + _OPTIONAL_STR:
        value = raw.get(key, "" if key in _OPTIONAL_STR else None)
        if not isinstance(value, str) or (key in _REQUIRED_STR and not value.strip()):
            errors.append(f"{key!r} must be a non-empty string")
            value = ""
        values[key] = value
    for key in _REQUIRED_SEQ + _OPTIONAL_SEQ:
        value = _strings(raw.get(key, () if key in _OPTIONAL_SEQ else None), key in _OPTIONAL_SEQ)
        if value is None:
            errors.append(
                f"{key!r} must be a tuple of strings" + ("" if key in _OPTIONAL_SEQ else ", not empty")
            )
            value = ()
        values[key] = value
    provides = raw.get("provides", {})
    if not isinstance(provides, dict) or not all(
        isinstance(k, str) and isinstance(v, str) and _NAME_RE.fullmatch(k) and _SYMBOL_RE.fullmatch(v)
        for k, v in provides.items()
    ):
        errors.append("'provides' must be a dict {sub-feature name: symbol of the kernel module}")
        provides = {}
    if errors and not values["name"]:
        return None, errors

    name = values["name"]
    if not _NAME_RE.fullmatch(name):
        errors.append(f"name {name!r}: lower-case words joined by '.', e.g. attention.flydsl.mla")
    for arch in values["arch"]:
        if arch not in KNOWN_ARCHS:
            errors.append(f"arch {arch!r} is not one of {', '.join(KNOWN_ARCHS)} (primus_turbo/arch.py)")
    if values["backend"] and values["backend"] not in BACKENDS:
        errors.append(f"backend {values['backend']!r} is not one of {', '.join(BACKENDS)}")
    for dtype in values["dtype"]:
        if dtype not in DTYPES:
            errors.append(f"dtype {dtype!r} is not one of {', '.join(DTYPES)}")
    for key in _OPTIONAL_STR:
        if values[key] and not _MODULE_RE.fullmatch(values[key]):
            errors.append(f"{key!r}: {values[key]!r} is not a module of {_PKG}")
    for gate in values["gates"]:
        module, _, symbol = gate.partition(":")
        if not (_MODULE_RE.fullmatch(module) and _SYMBOL_RE.fullmatch(symbol)):
            errors.append(f"gate {gate!r}: write '<module of {_PKG}>:<symbol>'")
    for pattern in values["products"]:
        if pattern.startswith("/") or ".." in pattern.split("/"):
            errors.append(f"product {pattern!r}: a pattern relative to the package, e.g. lib/<dir>/*.hsaco")
    if not (values["kernel"] or values["impl"] or values["sources"]):
        errors.append("an entry names its code: 'kernel', 'impl' or 'sources'")
    if provides and not values["kernel"]:
        errors.append("'provides' names symbols of the 'kernel' module, which is not given")
    entry = Kernel(provides=dict(provides), op=op, file=file, line=line, valid=not errors, **values)
    return entry, errors


def _parse(path: Path, file: str) -> Tuple[List[Kernel], List[Problem]]:
    op = path.name[: -len(FILE_SUFFIX)]
    try:
        body = list(ast.parse(path.read_text(encoding="utf-8")).body)
    except (OSError, SyntaxError, ValueError) as error:
        return [], [Problem(file, getattr(error, "lineno", 0) or 0, f"cannot be read: {error}")]
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    node = body[0] if len(body) == 1 else None
    if not (
        isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and getattr(node.targets[0], "id", None) == "KERNELS"
        and isinstance(node.value, (ast.Tuple, ast.List))
    ):
        line = body[1].lineno if len(body) > 1 else (body[0].lineno if body else 1)
        return [], [
            Problem(
                file,
                line,
                "a registry file is data only: a docstring and one assignment, KERNELS = (...), "
                "of dict literals (no imports, no code)",
            )
        ]
    entries, problems = [], []
    for element in node.value.elts:
        try:
            raw = ast.literal_eval(element)
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            problems.append(
                Problem(file, element.lineno, "an entry is a literal: no names, calls or operators")
            )
            continue
        entry, errors = _entry(raw, op, file, element.lineno)
        if isinstance(element, ast.Dict):  # a dict literal keeps the last of two equal keys without a word
            keys = [k.value for k in element.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)]
            repeated = sorted({k for k in keys if keys.count(k) > 1})
            if repeated and entry is not None:
                errors = errors + [f"field {k!r} is given twice" for k in repeated]
                entry = Kernel(**{**entry.__dict__, "valid": False})
        label = f"{raw.get('name')!r}: " if isinstance(raw, dict) and isinstance(raw.get("name"), str) else ""
        problems += [Problem(file, element.lineno, label + error) for error in errors]
        if entry is not None:
            entries.append(entry)
    return entries, problems


def load(pkg_dir: Optional[Path] = None) -> Tuple[List[Kernel], List[Problem]]:
    """Read every registry file of a package directory (default: this install).

    Returns (entries, problems). An entry that is wrong in some field is still returned when it
    has a name; the problems say what is wrong, and ``tools/conventions`` fails on any of them.
    """
    pkg_dir = Path(pkg_dir) if pkg_dir is not None else _PKG_DIR
    entries: List[Kernel] = []
    problems: List[Problem] = []
    for path in sorted(pkg_dir.rglob("*" + FILE_SUFFIX)):
        found, wrong = _parse(path, path.relative_to(pkg_dir.parent).as_posix())
        entries += found
        problems += wrong
    seen: Dict[Tuple[str, str], Kernel] = {}
    for entry in entries:
        for feature in entry.feature_names():
            for arch in entry.arch:
                first = seen.setdefault((feature, arch), entry)
                if first is not entry:
                    problems.append(
                        Problem(
                            entry.file,
                            entry.line,
                            f"{feature!r} is registered twice for {arch} (also {first.file}:{first.line})",
                        )
                    )
    return entries, problems


_LOADED: Dict[Path, List[Kernel]] = {}


def kernels(
    arch: Optional[str] = None,
    op: Optional[str] = None,
    name: Optional[str] = None,
    pkg_dir: Optional[Path] = None,
) -> List[Kernel]:
    """The entries, narrowed to an architecture, an operation and / or a name (each optional).
    An entry that ``load`` reported a problem in is left out: it answers no query until it is fixed."""
    root = Path(pkg_dir) if pkg_dir is not None else _PKG_DIR
    if root not in _LOADED:
        _LOADED[root] = [k for k in load(root)[0] if k.valid]
    arch = normalize_arch(arch) if arch else None
    return [
        k
        for k in _LOADED[root]
        if (arch is None or arch in k.arch)
        and (op is None or k.op == op)
        and (name is None or k.name == name)
    ]


def owners(paths: Sequence[str], pkg_dir: Optional[Path] = None) -> List[Kernel]:
    """The entries that own any of the given files (paths relative to the repository root)."""
    return [k for k in kernels(pkg_dir=pkg_dir) if any(k.owns(p) for p in paths)]


def module_file(module: str, pkg_dir: Optional[Path] = None) -> Optional[Path]:
    """The file of ``primus_turbo.<...>`` (a module, or a package's ``__init__.py``) in a package
    directory (default: this install), None when it is not there. Not an import."""
    parts = module.split(".")
    if parts[0] != _PKG or len(parts) < 2:
        return None
    base = (Path(pkg_dir) if pkg_dir is not None else _PKG_DIR).joinpath(*parts[1:])
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def find(feature: str, arch: Optional[str], pkg_dir: Optional[Path] = None) -> Optional[Tuple[Kernel, str]]:
    """The entry that provides ``feature`` on ``arch``, with the symbol a sub-feature needs (``""``
    for the entry's own name); None when no entry does."""
    arch = normalize_arch(arch)
    if not arch:
        return None
    for entry in kernels(arch=arch, pkg_dir=pkg_dir):
        if feature == entry.name:
            return entry, ""
        if feature.startswith(entry.name + ".") and feature[len(entry.name) + 1 :] in entry.provides:
            return entry, entry.provides[feature[len(entry.name) + 1 :]]
    return None


def present(entry: Kernel, pkg_dir: Optional[Path] = None) -> bool:
    """Whether what an entry consists of is in a package directory (default: this install): its
    modules, and for each ``products`` pattern at least one file. Not an import."""
    root = Path(pkg_dir) if pkg_dir is not None else _PKG_DIR
    return all(module_file(m, root) is not None for m in entry.modules()) and all(
        any(root.glob(pattern)) for pattern in entry.products
    )


def has(feature: str, arch: Optional[str] = None) -> bool:
    """Whether ``feature`` is available for ``arch`` (default: the current device's architecture).

    False for a feature or an architecture the registry does not list (to a caller written for a
    newer Primus-Turbo, a name this version does not know is a feature it does not have;
    ``features`` gives the names it knows), and, with ``arch`` left out, when there is no ROCm
    device to ask. With ``arch`` given nothing is imported to answer: a feature is available when
    what its entry names (module files, and for build products at least one matching file) is in
    the installed package. That the kernel module defines the symbol of a sub-feature is held by
    ``tools/conventions`` (REG002) when the entry is written, not looked up again here.
    """
    found = find(feature, arch if arch else device_arch())
    return found is not None and present(found[0])


def features(arch: Optional[str] = None) -> Dict[str, bool]:
    """``{feature: has(feature, arch)}`` for every feature the registry lists for the architecture."""
    arch = normalize_arch(arch) if arch else device_arch()
    if not arch:
        return {}
    names = sorted({f for entry in kernels(arch=arch) for f in entry.feature_names()})
    return {name: has(name, arch) for name in names}


def _main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m primus_turbo.registry",
        description="List the kernel registry: one line per entry, or with --json the entries in full. "
        "With --tests / --benchmarks, the test / benchmark files of the selected entries, one per line.",
    )
    parser.add_argument("--arch", help="only entries for this gfx ID, e.g. gfx1250")
    parser.add_argument("--op", help="only entries of this operation (the registry file's <op>)")
    parser.add_argument("--name", help="only entries with this name")
    parser.add_argument(
        "--owning",
        nargs="+",
        metavar="PATH",
        help="only entries that own one of these files: their code, the modules of their gates, their tests "
        "and benchmarks (paths relative to the repository root, e.g. the output of `git diff --name-only`); "
        "exit code 1 when no entry owns any of them",
    )
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--json", action="store_true", help="print the selected entries in full, as JSON")
    output.add_argument("--tests", action="store_true", help="print the selected entries' tests")
    output.add_argument("--benchmarks", action="store_true", help="print the selected entries' benchmarks")
    args = parser.parse_args(argv)

    selected = kernels(arch=args.arch, op=args.op, name=args.name)
    if args.owning:
        owned = owners([p[2:] if p.startswith("./") else p for p in args.owning])
        selected = [k for k in selected if k in owned]
        if not selected:
            # not an empty answer on stdout alone: `pytest $(... --tests)` would then run everything
            print("no registry entry owns any of the given paths", file=sys.stderr)
            return 1
    if args.json:
        print(json.dumps([asdict(k) for k in selected], indent=2))
    elif args.tests or args.benchmarks:
        seen = []
        for k in selected:
            for item in k.tests if args.tests else k.benchmarks:
                if item not in seen:
                    seen.append(item)
        print("\n".join(seen))
    else:
        for k in selected:
            print(
                f"{k.name:40s} {','.join(k.arch):10s} {k.backend:10s} {','.join(k.dtype) or '-':9s} {k.summary}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(_main())
