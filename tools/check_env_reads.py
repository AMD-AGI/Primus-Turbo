#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Check that Primus-Turbo code reads environment variables only through ``primus_turbo.common.envs``.

Usage: ``python tools/check_env_reads.py PATH...`` (directories are searched for ``*.py``).

A read is ``os.environ[...]``, ``os.environ.get(...)``, ``os.getenv(...)``, ``"X" in os.environ``,
or any other use of ``os.environ`` as a value (passing it on, ``.copy()``, iterating it), through
whatever names ``os``, ``environ`` and ``getenv`` are imported as, except where a parameter or local
assignment of an enclosing function rebinds that name. Writes are allowed:
``os.environ[k] = v``, ``del os.environ[k]``, ``update()`` / ``clear()``, ``setdefault()`` /
``pop()`` whose result is discarded, and ``mock.patch.dict(os.environ, ...)``. The exception is a
write of a variable with Primus-Turbo's own prefix (``OWN_PREFIXES``) by ``os.environ[k] = v``,
``del os.environ[k]`` or a discarded ``setdefault()`` / ``pop()``: ``envs`` caches each variable at
its first read, so such a write goes through ``envs.set_env()``, which also drops the cached value.

``envs.py`` holds only Primus-Turbo's own variables, so a read of another project's variable
(``THIRD_PARTY_NAMES`` / ``THIRD_PARTY_PREFIXES``) passes. The name is taken from a string
literal, a loop over a tuple of string literals or a module-level tuple of them, or a module-level
string constant, in that order; a parameter or local assignment of an enclosing function makes the
name dynamic. A loop passes only when every name in it is third-party. Names with Primus-Turbo's
own prefixes (``OWN_PREFIXES``) are never third-party.

``primus_turbo/common/envs.py`` itself is skipped. Any other read that must stay is exempted by
``# envs: allow`` on any line it spans. Exits 1 if anything is reported.
"""

from __future__ import annotations

import ast
import io
import re
import sys
import tokenize
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ENVS_FILE = REPO_ROOT / "primus_turbo" / "common" / "envs.py"
ENVS_DISPLAY = "primus_turbo/common/envs.py"

ALLOW_RE = re.compile(r"#\s*envs:\s*allow\b")

# Variables owned by other projects. Lists only what code under primus_turbo/ touches; add an
# entry when new code reads another project's variable.
THIRD_PARTY_PREFIXES = (
    "AMDGCN_",  # Triton's AMD backend
    "FLUX_",  # MLPerf Flux recipe flags, owned by Primus
    "FLYDSL_",
    "NVSHMEM_",  # NVSHMEM / rocSHMEM, configured by the DeepEP buffer
    "PRIMUS_MXFP4_",  # MXFP4 packing flag, owned by Primus
    "TRITON_",
)
THIRD_PARTY_NAMES = frozenset({"RANK", "WORLD_SIZE"})  # set by torchrun and other launchers
OWN_PREFIXES = ("PRIMUS_TURBO_", "PT_")

_LOOP_SCOPES = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp, ast.For, ast.AsyncFor)
_SCOPE_BOUNDARIES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef, ast.Module)
_FUNCTION_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)

ALWAYS_WRITE_METHODS = {"update", "clear"}
DISCARDED_WRITE_METHODS = {"setdefault", "pop"}
NAMED_READ_METHODS = {"get"} | DISCARDED_WRITE_METHODS


def is_third_party(name: str) -> bool:
    if name.startswith(OWN_PREFIXES):
        return False
    return name in THIRD_PARTY_NAMES or name.startswith(THIRD_PARTY_PREFIXES)


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    via: str
    # The variable name when it resolves to a single string, else None.
    name: str | None
    # False when the code reads the whole environment rather than one variable.
    single: bool = True
    # True for a write of one of Primus-Turbo's own variables that bypasses envs.set_env().
    write: bool = False

    def render(self, declared: set[str]) -> str:
        where = f"{self.path}:{self.line}"
        if self.write:
            return (
                f"{where}: writes {self.name} via {self.via}\n"
                f"    write it with envs.set_env({self.name!r}, value) (None removes it): envs caches "
                f"each variable at its first read and would not see this write"
            )
        if not self.single:
            return (
                f"{where}: uses the whole environment via {self.via}\n"
                f"    read each variable it needs through envs instead, or end the line with "
                f'"# envs: allow" if it must see os.environ itself'
            )
        if self.name is None:
            return (
                f"{where}: reads an environment variable via {self.via}, "
                f"but the name is not a single string constant\n"
                f"    declare each variable it can read in {ENVS_DISPLAY} and read it as envs.NAME, "
                f'or end the line with "# envs: allow"'
            )
        if self.name in declared:
            return (
                f"{where}: reads {self.name} via {self.via}\n"
                f"    {self.name} is declared in {ENVS_DISPLAY}: read it as envs.{self.name}"
            )
        declare = (
            f"add it to {ENVS_DISPLAY} (an environment_variables entry with a comment, and a "
            f"TYPE_CHECKING line), then read it as envs.{self.name}"
        )
        head = f"{where}: reads {self.name} via {self.via}\n    {self.name} is not declared"
        if self.name.startswith(OWN_PREFIXES):
            return f"{head}: {declare}"
        return (
            f"{head}. If it is a Primus-Turbo variable, {declare}; "
            f"if another project owns it, add its name or prefix to THIRD_PARTY_NAMES / "
            f"THIRD_PARTY_PREFIXES in tools/check_env_reads.py"
        )


def declared_names(envs_file: Path = ENVS_FILE) -> set[str]:
    """The string keys of the ``environment_variables`` dict literal in ``envs_file``."""
    tree = ast.parse(envs_file.read_text(encoding="utf-8"), filename=str(envs_file))
    for node in tree.body:
        if isinstance(node, ast.AnnAssign):
            targets, value = [node.target], node.value
        elif isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        else:
            continue
        if any(isinstance(t, ast.Name) and t.id == "environment_variables" for t in targets):
            if isinstance(value, ast.Dict):
                return {
                    k.value for k in value.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)
                }
    raise RuntimeError(f"{envs_file}: no environment_variables dict literal found")


def _module_str_constants(tree: ast.Module) -> dict[str, str]:
    """The module-level ``NAME = "LITERAL"`` assignments in ``tree``."""
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            if isinstance(node.value.value, str):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        constants[target.id] = node.value.value
    return constants


def _literal_strings(node: ast.AST) -> tuple[str, ...] | None:
    if isinstance(node, (ast.Tuple, ast.List)) and node.elts:
        if all(isinstance(e, ast.Constant) and isinstance(e.value, str) for e in node.elts):
            return tuple(e.value for e in node.elts)
    return None


def _local_names(func: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda) -> set[str]:
    """The names ``func`` binds as parameters or by assignment, not counting nested scopes."""
    args = func.args
    names = {a.arg for a in args.posonlyargs + args.args + args.kwonlyargs}
    names.update(a.arg for a in (args.vararg, args.kwarg) if a is not None)
    stack: list[ast.AST] = [func.body] if isinstance(func, ast.Lambda) else list(func.body)
    while stack:
        node = stack.pop()
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif not isinstance(node, (*_FUNCTION_SCOPES, ast.ClassDef)):
            stack.extend(ast.iter_child_nodes(node))
    return names


def _allowed_lines(source: str) -> set[int]:
    lines = set()
    try:
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type == tokenize.COMMENT and ALLOW_RE.search(tok.string):
                lines.add(tok.start[0])
    except (tokenize.TokenError, SyntaxError):
        pass
    return lines


class _Checker:
    def __init__(self, path: str, tree: ast.Module):
        self.path = path
        # Each finding with the first and last line of the code it reports.
        self.results: list[tuple[Finding, int, int]] = []
        self.parent: dict[ast.AST, ast.AST] = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                self.parent[child] = node
        self.local_names: dict[ast.AST, set[str]] = {}

        self.os_names: set[str] = set()
        self.environ_names: set[str] = set()
        self.getenv_names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "os":
                        self.os_names.add(alias.asname or "os")
                    elif alias.name.startswith("os.") and alias.asname is None:
                        self.os_names.add("os")
            elif isinstance(node, ast.ImportFrom) and node.module == "os" and node.level == 0:
                for alias in node.names:
                    if alias.name == "environ":
                        self.environ_names.add(alias.asname or "environ")
                    elif alias.name == "getenv":
                        self.getenv_names.add(alias.asname or "getenv")

        # Module-level string constants, so ``os.environ.get(_ENV)`` names its variable.
        self.str_constants: dict[str, str] = _module_str_constants(tree)
        # Module-level tuples / lists of string literals, for ``for name in _ENVS`` loops.
        self.str_sequences: dict[str, tuple[str, ...]] = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and (names := _literal_strings(node.value)) is not None:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.str_sequences[target.id] = names

        for node in ast.walk(tree):
            if self._is_environ(node):
                self._check_environ_use(node)
            elif self._is_getenv(node):
                self._check_getenv_use(node)

    def _shadowed(self, name: ast.Name) -> bool:
        """Whether a parameter or local assignment of an enclosing function binds ``name``."""
        node = self.parent.get(name)
        while node is not None:
            if isinstance(node, _FUNCTION_SCOPES):
                if node not in self.local_names:
                    self.local_names[node] = _local_names(node)
                if name.id in self.local_names[node]:
                    return True
            node = self.parent.get(node)
        return False

    def _is_os(self, node: ast.AST) -> bool:
        return isinstance(node, ast.Name) and node.id in self.os_names and not self._shadowed(node)

    def _is_environ(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Attribute):
            return node.attr == "environ" and self._is_os(node.value)
        return (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id in self.environ_names
            and not self._shadowed(node)
        )

    def _is_getenv(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Attribute):
            return node.attr == "getenv" and self._is_os(node.value)
        return (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id in self.getenv_names
            and not self._shadowed(node)
        )

    def _loop_names(self, name: ast.Name) -> tuple[str, ...] | None:
        """The values ``name`` takes when an enclosing loop binds it over string constants.

        ``()`` when an enclosing loop binds it over anything else, None when no loop binds it.
        """
        node: ast.AST | None = name
        while node is not None and not isinstance(node, _SCOPE_BOUNDARIES):
            if isinstance(node, _LOOP_SCOPES):
                loops = node.generators if hasattr(node, "generators") else [node]
                for loop in loops:
                    if isinstance(loop.target, ast.Name) and loop.target.id == name.id:
                        if isinstance(loop.iter, ast.Name):
                            if self._shadowed(loop.iter):
                                return ()
                            return self.str_sequences.get(loop.iter.id, ())
                        return _literal_strings(loop.iter) or ()
            node = self.parent.get(node)
        return None

    def _names_of(self, node: ast.AST | None) -> tuple[str, ...] | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return (node.value,)
        if isinstance(node, ast.Name):
            # Nearest binding first: a loop variable, a parameter or local assignment, a module constant.
            if (names := self._loop_names(node)) is not None:
                return names or None
            if not self._shadowed(node) and node.id in self.str_constants:
                return (self.str_constants[node.id],)
        return None

    def _report_write(self, node: ast.AST, via: str, key: ast.AST | None) -> None:
        for name in self._names_of(key) or ():
            if name.startswith(OWN_PREFIXES):
                finding = Finding(path=self.path, line=node.lineno, via=via, name=name, write=True)
                self.results.append((finding, node.lineno, node.end_lineno or node.lineno))

    def _report(self, node: ast.AST, via: str, key: ast.AST | None, single: bool = True) -> None:
        names = self._names_of(key) if single else None
        if names and all(is_third_party(n) for n in names):
            return
        finding = Finding(
            path=self.path,
            line=node.lineno,
            via=via,
            name=names[0] if names and len(names) == 1 else None,
            single=single,
        )
        self.results.append((finding, node.lineno, node.end_lineno or node.lineno))

    @staticmethod
    def _first_arg(call: ast.Call, keyword: str) -> ast.AST | None:
        if call.args:
            return call.args[0]
        return next((kw.value for kw in call.keywords if kw.arg == keyword), None)

    def _is_patch_dict(self, call: ast.Call) -> bool:
        func = call.func
        if not (isinstance(func, ast.Attribute) and func.attr == "dict"):
            return False
        owner = func.value
        return (isinstance(owner, ast.Name) and owner.id == "patch") or (
            isinstance(owner, ast.Attribute) and owner.attr == "patch"
        )

    def _check_environ_use(self, env: ast.AST) -> None:
        env_src = ast.unparse(env)
        parent = self.parent.get(env)

        if isinstance(parent, ast.Subscript) and parent.value is env:
            augmented = isinstance(self.parent.get(parent), ast.AugAssign)
            if isinstance(parent.ctx, ast.Load) or augmented:
                self._report(parent, f"{env_src}[...]", parent.slice)
            else:
                self._report_write(parent, f"{env_src}[...]", parent.slice)
            return

        if isinstance(parent, ast.Attribute) and parent.value is env:
            call = self.parent.get(parent)
            method = parent.attr
            if not (isinstance(call, ast.Call) and call.func is parent):
                self._report(parent, f"{env_src}.{method}", None, single=False)
                return
            if method in ALWAYS_WRITE_METHODS:
                return
            if method in DISCARDED_WRITE_METHODS and isinstance(self.parent.get(call), ast.Expr):
                self._report_write(call, f"{env_src}.{method}()", self._first_arg(call, "key"))
                return
            if method in NAMED_READ_METHODS:
                self._report(call, f"{env_src}.{method}()", self._first_arg(call, "key"))
            else:
                self._report(call, f"{env_src}.{method}()", None, single=False)
            return

        if (
            isinstance(parent, ast.Compare)
            and len(parent.ops) == 1
            and isinstance(parent.ops[0], (ast.In, ast.NotIn))
            and parent.comparators[0] is env
        ):
            op = "in" if isinstance(parent.ops[0], ast.In) else "not in"
            self._report(parent, f"... {op} {env_src}", parent.left)
            return

        if (
            isinstance(parent, ast.Call)
            and parent.args
            and parent.args[0] is env
            and self._is_patch_dict(parent)
        ):
            return

        self._report(env, env_src, None, single=False)

    def _check_getenv_use(self, getenv: ast.AST) -> None:
        src = ast.unparse(getenv)
        call = self.parent.get(getenv)
        if isinstance(call, ast.Call) and call.func is getenv:
            self._report(call, f"{src}()", self._first_arg(call, "key"))
        else:
            self._report(getenv, src, None)


def check_source(source: str, path: str = "<string>") -> list[Finding]:
    """The environment reads in ``source`` that are not exempted by ``# envs: allow``."""
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError:
        # check-ast reports the syntax error; there is nothing to check here.
        return []
    allowed = _allowed_lines(source)
    found = {
        finding
        for finding, start, end in _Checker(path, tree).results
        if not any(line in allowed for line in range(start, end + 1))
    }
    return sorted(found, key=lambda f: (f.line, f.via))


def _python_files(paths: list[str]) -> list[Path]:
    files = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            files.extend(sorted(p for p in path.rglob("*.py") if "__pycache__" not in p.parts))
        elif path.suffix == ".py":
            files.append(path)
    return files


def _display(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def main(argv: list[str] | None = None, envs_file: Path = ENVS_FILE) -> int:
    paths = sys.argv[1:] if argv is None else argv
    declared = declared_names(envs_file)
    findings: list[Finding] = []
    for path in _python_files(paths):
        if path.resolve() == envs_file.resolve():
            continue
        findings.extend(check_source(path.read_text(encoding="utf-8"), _display(path)))
    if not findings:
        return 0
    for finding in findings:
        print(finding.render(declared))
    writes = sum(f.write for f in findings)
    if writes:
        print(f"\n{writes} write(s) of Primus-Turbo variables that bypass envs.set_env().")
    if writes == len(findings):
        return 1
    print(
        f"\n{len(findings) - writes} environment variable read(s) outside {ENVS_DISPLAY}. Declare each variable "
        f"in environment_variables there and read it as envs.NAME (import primus_turbo.common.envs as envs). "
        f"Another project's variable belongs in THIRD_PARTY_NAMES / THIRD_PARTY_PREFIXES in "
        f'tools/check_env_reads.py instead; any other read that must stay can end with "# envs: allow".'
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
