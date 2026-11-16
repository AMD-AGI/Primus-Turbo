#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Generate ``docs/env.md`` from ``primus_turbo/common/envs.py``.

Usage::

    python tools/gen_env_docs.py           # rewrite docs/env.md
    python tools/gen_env_docs.py --check   # exit 1 if docs/env.md is out of date

``envs.py`` is parsed, not imported, so this runs without torch or a GPU. Each variable's type and
default come from its declaration in the ``TYPE_CHECKING`` block, and its description from the
comment block above its entry in ``environment_variables``:

* ``# ===== Title =====`` starts a section.
* A comment block followed by a bare ``#`` line is a note for the variables that follow it.
* The comment lines directly above an entry describe it. An entry with no comment lines directly
  above it shares the previous entry's description, which needs the same type and default.

A declared variable without a description, or a ``TYPE_CHECKING`` declaration and an entry that
disagree, is an error, so ``--check`` also keeps every new variable documented.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ENVS_FILE = REPO_ROOT / "primus_turbo" / "common" / "envs.py"
DOC_FILE = REPO_ROOT / "docs" / "env.md"

SECTION_RE = re.compile(r"^#\s*=+\s*(?P<title>.+?)\s*=+\s*$")

# Build-time variables are read by setup.py and tools/, which envs.py does not cover.
BUILD_VARIABLES = (
    (
        "PRIMUS_TURBO_FRAMEWORK",
        "`PYTORCH`",
        "Frameworks to build the extension for, `;`-separated: `PYTORCH`, `JAX`, or `PYTORCH;JAX`.",
    ),
    (
        "GPU_ARCHS",
        "the GPU of the build machine",
        "Target architectures, `;`-separated, e.g. `gfx942;gfx950`. `native` means the build "
        "machine's GPU. Supported: `gfx942`, `gfx950`, `gfx1250`.",
    ),
    (
        "PRIMUS_TURBO_BUILD_CK",
        "on",
        "Build the CK (ck_tile) GEMM backend. `0`, `false`, `off` or empty disable it. Always off "
        "on gfx1250.",
    ),
    (
        "PRIMUS_TURBO_BUILD_HK_MLA",
        "on",
        "On a gfx1250 build, build the HipKittens MLA backward code objects. `0`, `false`, `off` "
        "or empty skip them, and the MLA backward stays on FlyDSL.",
    ),
    (
        "PRIMUS_TURBO_HK_CXX",
        "`amdclang++` from `PATH`, `ROCM_PATH`, `ROCM_HOME` or `/opt/rocm`",
        "Compiler for the HipKittens gfx1250 MLA code objects.",
    ),
    (
        "MAX_JOBS",
        "ninja's worker count; 64 jobs per HIP compile",
        "Number of ninja workers, and of parallel jobs within each HIP compile (`-parallel-jobs`).",
    ),
    (
        "CXX",
        "`hipcc` under the ROCm home",
        "Compiler for the extension; also the default for `CMAKE_CXX_COMPILER` and `CMAKE_HIP_COMPILER`.",
    ),
)


@dataclass
class Variable:
    names: list[str]
    type: str
    default: str
    description: str


@dataclass
class Section:
    title: str
    # Notes and variables in source order: a str is a note, a list of Variable is a table.
    blocks: list[str | list[Variable]] = field(default_factory=list)


def _table_cell(text: str) -> str:
    """Make ``text`` safe inside a Markdown table cell: escape ``|`` everywhere (GitHub honours
    ``\\|`` inside code spans too), and ``<`` / ``>`` outside code spans."""
    parts = text.replace("|", "\\|").split("`")
    for i in range(0, len(parts), 2):
        parts[i] = parts[i].replace("<", "&lt;").replace(">", "&gt;")
    return "`".join(parts)


def _comment_text(lines: list[str]) -> str:
    return " ".join(line.lstrip("#").strip() for line in lines).strip().replace("``", "`")


def _find_declarations(tree: ast.Module) -> dict[str, tuple[str, str]]:
    for node in tree.body:
        if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING":
            return {
                stmt.target.id: (ast.unparse(stmt.annotation), ast.unparse(stmt.value))
                for stmt in node.body
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.value
            }
    raise ValueError("no `if TYPE_CHECKING:` block of declarations")


def _find_registry(tree: ast.Module) -> ast.Dict:
    for node in tree.body:
        target = node.target if isinstance(node, ast.AnnAssign) else None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
        if isinstance(target, ast.Name) and target.id == "environment_variables":
            if isinstance(node.value, ast.Dict):
                return node.value
    raise ValueError("no `environment_variables = {...}` dict")


def _format_default(value: str) -> str:
    if value == "None":
        return "unset"
    try:
        literal = ast.literal_eval(value)
    except ValueError:
        return f"`{value}`"
    if isinstance(literal, str):
        return f"`{literal}`"
    if isinstance(literal, tuple):
        return "`" + ", ".join(str(v) for v in literal) + "`"
    return f"`{literal}`"


def parse_envs(source: str) -> list[Section]:
    tree = ast.parse(source)
    declarations = _find_declarations(tree)
    registry = _find_registry(tree)

    key_lines: dict[int, str] = {}
    for key in registry.keys:
        if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
            raise ValueError(f"line {key.lineno}: environment_variables keys must be string literals")
        key_lines[key.lineno] = key.value

    names = set(key_lines.values())
    missing = sorted(set(declarations) - names)
    undeclared = sorted(names - set(declarations))
    if missing:
        raise ValueError(f"declared under TYPE_CHECKING but not in environment_variables: {missing}")
    if undeclared:
        raise ValueError(f"in environment_variables but not declared under TYPE_CHECKING: {undeclared}")

    sections: list[Section] = []
    pending: list[str] = []
    table: list[Variable] | None = None
    previous: Variable | None = None

    lines = source.splitlines()
    for lineno in range(registry.lineno, registry.end_lineno + 1):
        stripped = lines[lineno - 1].strip()
        if stripped.startswith("#"):
            match = SECTION_RE.match(stripped)
            if match:
                sections.append(Section(match.group("title")))
                pending, table, previous = [], None, None
            elif stripped == "#":
                if not sections:
                    raise ValueError(f"line {lineno}: a note before the first section")
                sections[-1].blocks.append(_comment_text(pending))
                pending, table, previous = [], None, None
            else:
                pending.append(stripped)
            continue

        name = key_lines.get(lineno)
        if name is None:
            continue
        if not sections:
            raise ValueError(f"line {lineno}: {name} comes before the first section")
        var_type, default = declarations[name]
        if pending:
            previous = Variable([name], var_type, _format_default(default), _comment_text(pending))
            if table is None:
                table = []
                sections[-1].blocks.append(table)
            table.append(previous)
        elif previous is not None and (var_type, _format_default(default)) == (
            previous.type,
            previous.default,
        ):
            previous.names.append(name)
        else:
            raise ValueError(f"line {lineno}: {name} has no description (a comment block above its entry)")
        pending = []

    return sections


def render(sections: list[Section]) -> str:
    out = [
        "<!-- Generated by tools/gen_env_docs.py from primus_turbo/common/envs.py. Do not edit by hand:",
        "     change the comments in envs.py, then run `python tools/gen_env_docs.py`. -->",
        "",
        "# Environment variables",
        "",
        "Primus-Turbo reads the runtime variables below through `primus_turbo.common.envs`.",
        "",
        "- Each value is parsed the first time it is read, and that value is kept for the rest of the "
        "process. Set variables before the first use of Primus-Turbo; setting one later has no effect.",
        "- Code that must change a variable at run time calls `envs.set_env(name, value)`, which also "
        "drops the cached value.",
        "- Variables read by the C++ extension are cached by C++ at its own first read, so "
        "`set_env()` reaches C++ only before that.",
        "- To see the value Primus-Turbo uses, read `primus_turbo.common.envs.<NAME>`.",
        "",
        "Variables owned by other projects (FlyDSL, Triton, NVSHMEM, the launcher's `RANK` / "
        "`WORLD_SIZE`, and the Primus `FLUX_*` / `PRIMUS_MXFP4_*` recipe flags) are not listed here.",
        "",
        "## Runtime variables",
        "",
    ]
    for section in sections:
        out += [f"### {section.title}", ""]
        for block in section.blocks:
            if isinstance(block, str):
                out += [_table_cell(block), ""]
                continue
            out += ["| Variable | Type | Default | Description |", "|---|---|---|---|"]
            for var in block:
                names = "<br>".join(f"`{n}`" for n in var.names)
                cells = (f"`{var.type}`", var.default, var.description)
                out.append(f"| {names} | " + " | ".join(_table_cell(c) for c in cells) + " |")
            out.append("")

    out += [
        "## Build-time variables",
        "",
        "Read by `setup.py` and `tools/` when the extension is built, not at run time.",
        "",
        "| Variable | Default | Description |",
        "|---|---|---|",
    ]
    out += [f"| `{name}` | {default} | {description} |" for name, default, description in BUILD_VARIABLES]
    out.append("")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="fail if docs/env.md is out of date")
    args = parser.parse_args(argv)

    try:
        content = render(parse_envs(ENVS_FILE.read_text(encoding="utf-8")))
    except ValueError as e:
        print(f"{ENVS_FILE.relative_to(REPO_ROOT)}: {e}", file=sys.stderr)
        return 1

    if args.check:
        current = DOC_FILE.read_text(encoding="utf-8") if DOC_FILE.exists() else None
        if current != content:
            print(
                f"{DOC_FILE.relative_to(REPO_ROOT)} is out of date; run `python tools/gen_env_docs.py`.",
                file=sys.stderr,
            )
            return 1
        return 0

    DOC_FILE.write_text(content, encoding="utf-8")
    print(f"wrote {DOC_FILE.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
