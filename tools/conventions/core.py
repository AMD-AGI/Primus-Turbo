###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The tree, findings, the baseline, the runner and the command line of ``tools/conventions``.

Imports nothing heavier than ``primus_turbo.registry`` (no torch, no GPU), so it runs in the
lint job and in a pre-commit hook.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import json
import os
import pkgutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from functools import cached_property
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE = Path(__file__).resolve().parent / "baseline.txt"
PKG = "primus_turbo"

# Not part of the source when the tree is walked instead of read from git: caches at any depth, build
# output and vendored code at the top only (primus_turbo/pytorch/dist/ and tests/pytorch/dist/ are source).
_SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".ruff_cache"}
_SKIP_TOP_DIRS = {"build", "dist", "3rdparty"}


@dataclass(frozen=True)
class Finding:
    rule: str
    path: str  # relative to the root of the tree
    line: int  # 0: the file as a whole
    message: str

    def key(self) -> Tuple[str, str]:
        return self.rule, self.path


@dataclass(frozen=True)
class Rule:
    id: str
    title: str
    check: Callable[["Tree"], Iterable[Finding]]
    baseline: bool  # whether findings that predate the rule may be listed in baseline.txt
    doc: str


RULES: Dict[str, Rule] = {}


def rule(rule_id: str, title: str, baseline: bool = False):
    """Register ``check(tree) -> findings`` as a rule. Its docstring is what ``--explain`` prints:
    why the rule exists and how to fix a finding."""

    def register(check):
        if rule_id in RULES:
            raise ValueError(f"rule {rule_id} is defined twice")
        if not (check.__doc__ or "").strip():
            raise ValueError(f"rule {rule_id} has no docstring (why it exists, how to fix a finding)")
        RULES[rule_id] = Rule(rule_id, title, check, baseline, "\n".join(_dedent(check.__doc__)))
        return check

    return register


def _dedent(doc: str) -> List[str]:
    lines = doc.strip("\n").splitlines()
    rest = [line for line in lines[1:] if line.strip()]
    indent = min((len(line) - len(line.lstrip()) for line in rest), default=0)
    return [lines[0].strip()] + [line[indent:].rstrip() for line in lines[1:]]


def load_rules() -> None:
    """Import every module of ``rules/`` (each registers its rules)."""
    from tools.conventions import rules

    for module in sorted(m.name for m in pkgutil.iter_modules(rules.__path__)):
        importlib.import_module(f"{rules.__name__}.{module}")


class Tree:
    """The source tree being checked: its files and what the rules read from them."""

    def __init__(self, root: Path, files: Optional[Sequence[str]] = None, tracked: bool = False):
        self.root = Path(root).resolve()
        if files is None:
            files = _source_files(self.root, tracked)
        self.files: Tuple[str, ...] = tuple(sorted(files))
        self._files = set(self.files)

    def is_file(self, path: str) -> bool:
        return path in self._files

    def exists(self, path: str) -> bool:
        """A file of the tree, or a directory that holds one."""
        path = path.rstrip("/")
        return path in self._files or any(f.startswith(path + "/") for f in self.files)

    def under(self, *prefixes: str) -> Iterator[str]:
        return (f for f in self.files if f.startswith(prefixes))

    def text(self, path: str) -> str:
        return (self.root / path).read_text(encoding="utf-8", errors="replace")

    def module_path(self, module: str) -> Optional[str]:
        """The file of a dotted module (``a.b`` -> ``a/b.py`` or ``a/b/__init__.py``), None when
        the tree does not have it."""
        base = module.replace(".", "/")
        for candidate in (base + ".py", base + "/__init__.py"):
            if candidate in self._files:
                return candidate
        return None

    def defines(self, path: str, name: str) -> bool:
        """Whether a Python file binds ``name`` at module level: def / class / assignment / import,
        also inside a module-level ``if`` or ``try``."""
        try:
            body = ast.parse(self.text(path)).body
        except (OSError, SyntaxError, ValueError):
            return False
        return _binds(body, name)

    @cached_property
    def registry(self):
        """(entries, problems) of the tree's kernel registry (``primus_turbo.registry.load``)."""
        sys.path.insert(0, str(REPO_ROOT))
        try:
            from primus_turbo import registry
        finally:
            sys.path.pop(0)
        return registry.load(self.root / PKG)


def _binds(body, name: str) -> bool:
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name:
            return True
        if isinstance(node, ast.Assign) and any(name in _names(t) for t in node.targets):
            return True
        if isinstance(node, (ast.For, ast.AsyncFor)) and name in _names(node.target):
            return True
        if isinstance(node, (ast.With, ast.AsyncWith)) and (
            any(i.optional_vars is not None and name in _names(i.optional_vars) for i in node.items)
            or _binds(node.body, name)
        ):
            return True
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == name:
            return True
        if isinstance(node, (ast.Import, ast.ImportFrom)) and any(
            (a.asname or a.name) == name for a in node.names
        ):
            return True
        if isinstance(node, ast.If) and (_binds(node.body, name) or _binds(node.orelse, name)):
            return True
        if isinstance(node, ast.Try) and (
            _binds(node.body, name)
            or _binds(node.orelse, name)
            or _binds(node.finalbody, name)
            or any(_binds(h.body, name) for h in node.handlers)
        ):
            return True
    return False


def _names(target) -> List[str]:
    """The names an assignment target binds (``a``, ``a, b``, ``[a, *b]``)."""
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, ast.Starred):
        return _names(target.value)
    if isinstance(target, (ast.Tuple, ast.List)):
        return [name for element in target.elts for name in _names(element)]
    return []


def _source_files(root: Path, tracked: bool = False) -> List[str]:
    """The files of the source tree. When ``root`` is the top of a git checkout: what git tracks,
    and unless ``tracked`` also what is new and not ignored (so a file is checked before it is
    staged; a commit hook and CI ask for ``tracked``, which is what the commit will hold).
    Otherwise (an unpacked sdist, a tree built by a test, also one that sits inside some other
    repository) every file below it."""
    try:
        top = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
        if Path(top).resolve() == root:
            command = ["git", "ls-files", "-z", "--cached"]
            if not tracked:
                command += ["--others", "--exclude-standard"]
            out = subprocess.run(command, cwd=root, capture_output=True, text=True, check=True).stdout
            files = sorted({p for p in out.split("\0") if p and (root / p).is_file()})
            if files:
                return files
    except (OSError, subprocess.CalledProcessError):
        pass
    found = []
    for directory, subdirs, names in os.walk(root):
        relative = Path(directory).relative_to(root)
        skip = _SKIP_DIRS | (_SKIP_TOP_DIRS if relative == Path(".") else set())
        subdirs[:] = [d for d in subdirs if d not in skip and not d.endswith(".egg-info")]
        found += [(relative / name).as_posix() for name in names]
    return found


def read_baseline(path: Path) -> List[Tuple[str, str]]:
    """``[(rule id, path)]`` of a baseline file: one ``<rule id> <path>`` per line, ``#`` comments."""
    if not path.is_file():
        return []
    keys = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            rule_id, _, file = line.partition(" ")
            keys.append((rule_id, file.strip()))
    return keys


_BASELINE_HEADER = """\
# Findings that predate a rule of tools/conventions: `<rule id> <path>`, sorted. Written by the tool:
# comments on the lines below are not kept.
# A file listed here passes that rule; any other file must follow it. A listed file that no
# longer breaks the rule must be removed (`python tools/conventions --update-baseline`), so this
# list only shrinks. Adding a line is accepting a new exception: say why in the pull request.
"""


def write_baseline(path: Path, keys: Iterable[Tuple[str, str]]) -> None:
    lines = [f"{rule_id} {file}" for rule_id, file in sorted(set(keys))]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_BASELINE_HEADER + "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


@dataclass
class Result:
    new: List[Finding]  # findings that fail the run
    known: List[Finding]  # findings covered by the baseline
    stale: List[Tuple[str, str]]  # baseline lines that no longer match a finding

    @property
    def ok(self) -> bool:
        return not self.new and not self.stale


def run(tree: Tree, rule_ids: Sequence[str], baseline: Sequence[Tuple[str, str]]) -> Result:
    findings: List[Finding] = []
    for rule_id in rule_ids:
        findings += sorted(set(RULES[rule_id].check(tree)), key=lambda f: (f.path, f.line, f.message))
    allowed = {key for key in baseline if key[0] in RULES and RULES[key[0]].baseline}
    new = [f for f in findings if f.key() not in allowed]
    known = [f for f in findings if f.key() in allowed]
    found = {f.key() for f in findings}
    # a line whose file no longer breaks the rule, and a line for a rule that does not exist (any
    # more) or takes no baseline
    stale = [key for key in baseline if key not in allowed or (key[0] in rule_ids and key not in found)]
    return Result(new, known, sorted(set(stale)))


def _report(result: Result, rule_ids: Sequence[str], fmt: str, out) -> None:
    if fmt == "json":
        json.dump(
            {
                "ok": result.ok,
                "rules": list(rule_ids),
                "new": [asdict(f) for f in result.new],
                "known": [asdict(f) for f in result.known],
                "stale_baseline": [{"rule": r, "path": p} for r, p in result.stale],
            },
            out,
            indent=2,
        )
        out.write("\n")
        return
    for f in result.new:
        where = f"{f.path}:{f.line}" if f.line else f.path
        print(f"{where}: {f.rule} {f.message}", file=out)
        if fmt == "github":
            line = f",line={f.line}" if f.line else ""
            print(f"::error file={f.path}{line},title={f.rule} {RULES[f.rule].title}::{f.message}", file=out)
    for rule_id, path in result.stale:
        why = (
            f"{path} no longer breaks {rule_id}"
            if rule_id in RULES and RULES[rule_id].baseline
            else f"{rule_id} is not a rule that takes a baseline ({path})"
        )
        message = f"{why}: remove its line from tools/conventions/baseline.txt (python tools/conventions --update-baseline)"
        print(f"tools/conventions/baseline.txt: {message}", file=out)
        if fmt == "github":
            print(
                f"::error file=tools/conventions/baseline.txt,title=stale baseline line::{message}", file=out
            )
    broken = sorted({f.rule for f in result.new})
    print(
        f"conventions: {len(rule_ids)} rules, {len(result.new)} new finding(s), "
        f"{len(result.known)} known (baseline), {len(result.stale)} stale baseline line(s)"
        + (f". Explain a rule: python tools/conventions --explain {broken[0]}" if broken else ""),
        file=out,
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python tools/conventions",
        description="Check the repository against its conventions (rules with an ID). Exit code 1 when a "
        "rule is broken by something that is not in the baseline, or a baseline line is stale.",
    )
    parser.add_argument("--root", default=str(REPO_ROOT), help="the tree to check (default: this repository)")
    parser.add_argument("--rule", action="append", metavar="ID", help="only this rule (repeatable)")
    parser.add_argument(
        "--baseline", default=None, help="baseline file (default: tools/conventions/baseline.txt)"
    )
    parser.add_argument(
        "--format",
        choices=("text", "github", "json"),
        default=None,
        help="github adds workflow annotations (the default under GitHub Actions); json is for tools",
    )
    parser.add_argument("--list-rules", action="store_true", help="print the rules and exit")
    parser.add_argument("--explain", metavar="ID", help="print why a rule exists and how to fix a finding")
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="rewrite the baseline without its stale lines; with --accept-new also add today's findings "
        "of the rules that take a baseline (adopting a rule, or accepting an exception on purpose)",
    )
    parser.add_argument("--accept-new", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--tracked",
        action="store_true",
        help="in a git checkout, check only what git tracks (the commit hook and CI); by default files "
        "that are new and not ignored are checked too, so a finding shows before the file is staged",
    )
    args = parser.parse_args(argv)
    if args.accept_new and not args.update_baseline:
        parser.error("--accept-new goes with --update-baseline")

    load_rules()
    if args.explain:
        if args.explain not in RULES:
            parser.error(f"no rule {args.explain}; rules: {', '.join(sorted(RULES))}")
        chosen = RULES[args.explain]
        print(f"{chosen.id}  {chosen.title}\n\n{chosen.doc}")
        return 0
    if args.list_rules:
        for chosen in sorted(RULES.values(), key=lambda r: r.id):
            print(f"{chosen.id:9s} {chosen.title}" + ("  [baseline]" if chosen.baseline else ""))
        return 0
    rule_ids = sorted(RULES) if not args.rule else sorted(set(args.rule))
    unknown = [r for r in rule_ids if r not in RULES]
    if unknown:
        parser.error(f"no rule {', '.join(unknown)}; rules: {', '.join(sorted(RULES))}")

    root = Path(args.root).resolve()
    if not (root / PKG).is_dir():
        # e.g. the release job, which tests the installed package with the source directory removed:
        # every rule would report the whole tree as unregistered
        print(
            f"tools/conventions: no {PKG}/ under {root}: not a source tree, nothing checked", file=sys.stderr
        )
        return 2
    baseline_file = Path(args.baseline) if args.baseline else root / BASELINE.relative_to(REPO_ROOT)
    baseline = read_baseline(baseline_file)
    result = run(Tree(root, tracked=args.tracked), rule_ids, baseline)
    if args.update_baseline:
        keep = [key for key in baseline if key not in set(result.stale)]
        if args.accept_new:
            keep += [f.key() for f in result.new if RULES[f.rule].baseline]
        write_baseline(baseline_file, keep)
        result = run(Tree(root, tracked=args.tracked), rule_ids, read_baseline(baseline_file))
    fmt = args.format or ("github" if os.environ.get("GITHUB_ACTIONS") == "true" else "text")
    _report(result, rule_ids, fmt, sys.stdout)
    return 0 if result.ok else 1
