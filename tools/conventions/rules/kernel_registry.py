###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""REG: the kernel registry (``primus_turbo.registry``, the ``<op>_registry.py`` files) and the tree
agree."""

from __future__ import annotations

import ast
from typing import Dict, List

from tools.conventions.core import Finding, Tree, rule
from tools.conventions.rules.arch_tags import ENFORCED_ARCHS, VERIFY_TREES, grouped, tag_prefix

# Architectures on which tests/conftest.py skips every test that does not opt in.
OPT_IN_ARCHS = ("gfx1250",)

REGISTRY_SUFFIX = "_registry.py"


@rule("REG001", "kernel registry files are well-formed data")
def registry_is_well_formed(tree: Tree):
    """A registry file is <op>/<op>_registry.py: a docstring and one assignment KERNELS = (...) of
    dict literals with the fields primus_turbo/registry.py documents; a feature name is registered
    once per architecture.

    Why: the registry is read by primus_turbo.capabilities at run time, by this tool and by
    agents, without importing anything. Code in a registry file, a misspelled field or a name
    registered twice makes one of them read something else than the author meant.

    Fix: follow the message; `python -m primus_turbo.registry --json` shows what was read.
    """
    _, problems = tree.registry
    for problem in problems:
        yield Finding("REG001", problem.file, problem.line, problem.message)
    for path in tree.files:
        if path.startswith("primus_turbo/") and path.endswith(REGISTRY_SUFFIX):
            directory, _, name = path.rpartition("/")
            op = name[: -len(REGISTRY_SUFFIX)]
            if directory.rsplit("/", 1)[-1] != op:
                yield Finding(
                    "REG001",
                    path,
                    0,
                    f"a registry file is named after its directory, the operation: "
                    f"{directory}/{directory.rsplit('/', 1)[-1]}{REGISTRY_SUFFIX}",
                )


@rule("REG002", "what a kernel registry entry names exists")
def registry_entries_exist(tree: Tree):
    """The modules of an entry ('kernel', 'impl'), its 'sources', its 'tests' and 'benchmarks' are
    in the tree; each of its 'gates' (<module>:<symbol>) and each symbol of 'provides' is defined
    at module level; a test given as <file>::<name> is defined in that file.

    Why: an entry that points at something moved, renamed or deleted answers
    primus_turbo.capabilities wrongly (the feature silently reads False) and sends whoever follows
    it to nothing.

    Fix: update the entry in the same change that moves or removes the code.
    """
    entries, _ = tree.registry
    for entry in entries:

        def finding(message: str, entry=entry):
            return Finding("REG002", entry.file, entry.line, f"{entry.name!r}: {message}")

        missing = set()  # one finding per missing module, however many fields name it
        for field in ("kernel", "impl"):
            module = getattr(entry, field)
            if module and module not in missing and tree.module_path(module) is None:
                missing.add(module)
                yield finding(f"'{field}': no module {module}")
        for gate in entry.gates:
            module, _, symbol = gate.partition(":")
            path = tree.module_path(module)
            if path is None:
                if module not in missing:
                    missing.add(module)
                    yield finding(f"gate {gate}: no module {module}")
            elif not tree.defines(path, symbol):
                yield finding(f"gate {gate}: {path} does not define {symbol}")
        kernel = tree.module_path(entry.kernel) if entry.kernel else None
        for extra, symbol in entry.provides.items():
            if kernel is not None and not tree.defines(kernel, symbol):
                yield finding(f"'provides' {extra!r}: {kernel} does not define {symbol}")
        for source in entry.sources:
            if not tree.exists(source):
                yield finding(f"'sources': no path {source}")
        for field in ("tests", "benchmarks"):
            for item in getattr(entry, field):
                path, _, name = item.partition("::")
                if not tree.is_file(path):
                    yield finding(f"'{field}': no file {path}")
                elif name and not tree.defines(path, _top_name(name)):
                    yield finding(f"'{field}': {path} does not define {_top_name(name)}")


def _top_name(test: str) -> str:
    """The module-level name of a test given as ``name``, ``Class::name`` or ``name[parameters]``."""
    return test.split("::", 1)[0].split("[", 1)[0]


def _marks(node, arch: str) -> bool:
    """Whether an expression is, or holds, ``pytest.mark.<arch>`` (also called, also in a list)."""
    for sub in ast.walk(node):
        if isinstance(sub, ast.Attribute) and sub.attr == arch:
            value = sub.value
            if isinstance(value, ast.Attribute) and value.attr == "mark":
                return True
    return False


def _opts_in(source: str, name: str, arch: str) -> bool:
    """Whether a test file marks itself (``pytestmark = ...``) or, for a test given by ``name``, that
    test or class with ``pytest.mark.<arch>``. A mark on some other test of the file does not count."""
    try:
        body = ast.parse(source).body
    except (SyntaxError, ValueError):
        return False
    for node in body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "pytestmark" for t in node.targets):
            if _marks(node.value, arch):
                return True
        if (
            name
            and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node.name == name
            and any(_marks(decorator, arch) for decorator in node.decorator_list)
        ):
            return True
    return False


@rule("REG003", "a registered kernel's tests run on its architecture")
def registered_tests_opt_in(tree: Tree):
    """Every test a registry entry lists runs on the entry's architecture. On gfx1250,
    tests/conftest.py skips everything that does not opt in: a test file opts in by carrying the
    architecture tag in its name, a test or a file without it by pytest.mark.gfx1250. An entry
    that lists a whole file needs the file to opt in (name tag or module-level pytestmark); one
    that lists <file>::<test> needs that test (or the file) to.

    Why: a test that is skipped on the architecture it was written for verifies nothing there,
    and a skip does not fail CI. The registry is where an agent reads how to verify a kernel; what
    it lists must actually run.

    Fix: put the tag in the test file's name (test_<...>_gfx1250_<...>.py), or add
    `pytestmark = pytest.mark.gfx1250` to the file, or `@pytest.mark.gfx1250` to the test and
    list it as <file>::<test>.
    """
    entries, _ = tree.registry
    for entry in entries:
        for arch in entry.arch:
            if arch not in OPT_IN_ARCHS:
                continue
            for item in entry.tests:
                path, _, name = item.partition("::")
                if not tree.is_file(path) or tag_prefix(path, arch):
                    continue
                if not _opts_in(tree.text(path), _top_name(name) if name else "", arch):
                    what = f"{path}::{name}" if name else path
                    yield Finding(
                        "REG003",
                        path,
                        0,
                        f"{what} is listed as a test of {entry.name!r} ({arch}) but skipped on {arch}: the "
                        f"file name has no {arch} tag and "
                        + ("neither the test nor the file" if name else "the file as a whole")
                        + f" is marked pytest.mark.{arch}",
                    )


@rule("REG004", "architecture-tagged tests and benchmarks belong to a kernel registry entry")
def tagged_tests_are_registered(tree: Tree):
    """Every test file (tests/**/test_*.py) and benchmark (benchmark/**/bench_*.py) whose path
    carries an architecture tag is listed by a kernel registry entry for that architecture, under
    'tests' or 'benchmarks'. Helper modules next to them are not.

    Why: `python -m primus_turbo.registry --owning <changed files> --tests` is how a change to a
    kernel finds what verifies it. A test that no entry lists is not found that way, so it is not
    run before the change is pushed.

    Fix: add the file to 'tests' (or 'benchmarks') of the entry it verifies, in the operation's
    <op>_registry.py.
    """
    entries, _ = tree.registry
    for arch in ENFORCED_ARCHS:
        owners = [k for k in entries if arch in k.arch]
        found: Dict[str, List[str]] = {}
        for path in tree.under(*VERIFY_TREES):
            name = path.rsplit("/", 1)[-1]
            if not name.endswith(".py") or not name.startswith(
                "bench_" if path.startswith("benchmark/") else "test_"
            ):
                continue
            if tag_prefix(path, arch) and not any(k.owns(path) for k in owners):
                found.setdefault(path, []).append(path)
        for first, _, _ in grouped(found):
            field = "benchmarks" if first.startswith("benchmark/") else "tests"
            yield Finding(
                "REG004",
                first,
                0,
                f"carries the {arch} tag but no kernel registry entry for {arch} lists it under '{field}'",
            )
