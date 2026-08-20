#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from __future__ import annotations

import argparse
import datetime
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FLYDSL_DIR = REPO_ROOT / "primus_turbo" / "flydsl"

CPP_EXTENSIONS = {".cpp", ".cc", ".cxx", ".c", ".h", ".hpp", ".cu", ".cuh", ".hip"}
SOURCE_SUFFIXES = {".py", *CPP_EXTENSIONS}

SKIP_RELATIVE_PATHS = {
    Path("primus_turbo/_version.py"),
    Path("primus_turbo/_build_info.py"),
}

SKIP_DIR_NAMES = {
    "3rdparty",
    ".git",
    "__pycache__",
    "build",
    "dist",
    ".eggs",
}

COPYRIGHT_YEAR_RE = re.compile(r"Copyright \(c\) (\d{4})")
AMD_COPYRIGHT_RE = re.compile(r"Copyright \(c\) \d{4}, Advanced Micro Devices, Inc\. All rights reserved\.")
AMD_COPYRIGHT_RELAXED_RE = re.compile(r"Advanced Micro Devices, Inc\. All rights reserved\.")
PYTHON_HEADER_START = "###############################################################################"
PYTHON_INLINE_AMD_COPYRIGHT_RE = re.compile(
    r"^# Copyright \(c\) \d{4}, Advanced Micro Devices, Inc\. All rights reserved\.$"
)
PYTHON_INLINE_LICENSE_RE = re.compile(r"^# See LICENSE for license information\.$")
PYTHON_CODING_RE = re.compile(r"^#.*coding[:=]", re.IGNORECASE)
HEADER_SCAN_LINES = 30
CPP_BANNER_START = f"/{'*' * 99}"
CPP_BANNER_END = f" {'*' * 98}/"
CPP_BANNER_START_RE = re.compile(r"^/\*+/?$")
CPP_BANNER_END_RE = re.compile(r"^\s*\*+/$")


def current_year() -> str:
    return str(datetime.date.today().year)


def flydsl_header(year: str) -> list[str]:
    return [
        "###############################################################################",
        "# SPDX-License-Identifier: Apache-2.0",
        "#",
        f"# Copyright (c) {year}, Advanced Micro Devices, Inc. All rights reserved.",
        f"# Copyright (c) {year} FlyDSL Project Contributors",
        "#",
        "# Adapted from FlyDSL (https://github.com/ROCm/FlyDSL)",
        "# Modified by the Primus-Turbo team.",
        "#",
        "# This file is distributed under the Apache License 2.0 (see LICENSE-APACHE),",
        "# not the MIT license that covers the rest of Primus-Turbo (see LICENSE).",
        "###############################################################################",
    ]


def normalize_copyright_years(lines: list[str]) -> list[str]:
    return [COPYRIGHT_YEAR_RE.sub("Copyright (c) YEAR", line) for line in lines]


def is_cpp_file(path: Path) -> bool:
    return path.suffix.lower() in CPP_EXTENSIONS


def is_flydsl_file(path: Path) -> bool:
    try:
        path.resolve().relative_to(FLYDSL_DIR.resolve())
    except ValueError:
        return False
    return True


def requires_strict_header(path: Path) -> bool:
    return path.suffix == ".py" and is_flydsl_file(path)


def has_amd_copyright(lines: list[str], scan_lines: int = HEADER_SCAN_LINES) -> bool:
    return any(AMD_COPYRIGHT_RE.search(line) for line in lines[:scan_lines])


def has_amd_copyright_relaxed(lines: list[str], scan_lines: int = HEADER_SCAN_LINES) -> bool:
    return any(AMD_COPYRIGHT_RELAXED_RE.search(line) for line in lines[:scan_lines])


def has_python_block_header(lines: list[str]) -> bool:
    index = 0
    while index < len(lines) and index < HEADER_SCAN_LINES:
        line = lines[index]
        if line.startswith("#!") or PYTHON_CODING_RE.match(line):
            index += 1
            continue
        return line == PYTHON_HEADER_START
    return False


def default_python_block_header(year: str) -> list[str]:
    return [
        "###############################################################################",
        f"# Copyright (c) {year}, Advanced Micro Devices, Inc. All rights reserved.",
        "#",
        "# See LICENSE for license information.",
        "###############################################################################",
    ]


def strip_duplicate_inline_python_header(body: list[str]) -> list[str]:
    body = strip_leading_blank_lines(body)
    if not body or not PYTHON_INLINE_AMD_COPYRIGHT_RE.match(body[0].strip()):
        return body

    index = 1
    while index < len(body) and body[index].strip() in {"", "#"}:
        index += 1
    if index < len(body) and PYTHON_INLINE_LICENSE_RE.match(body[index].strip()):
        index += 1
    return strip_leading_blank_lines(body[index:])


def has_duplicate_inline_python_header(lines: list[str]) -> bool:
    prefix = extract_python_prefix(lines)
    body = lines[len(prefix) :]
    if not has_python_block_header(body):
        return False

    header_start = find_header_start(body, "python")
    if header_start is None:
        return False

    header_end = find_python_header_end(body, header_start)
    rest = strip_leading_blank_lines(body[header_end + 1 :])
    return bool(rest) and bool(PYTHON_INLINE_AMD_COPYRIGHT_RE.match(rest[0].strip()))


def extract_leading_inline_python_header(lines: list[str]) -> tuple[list[str], list[str]]:
    header: list[str] = []
    index = 0
    while index < len(lines) and index < HEADER_SCAN_LINES:
        stripped = lines[index].strip()
        if stripped.startswith("# Copyright") or stripped == "#" or stripped.startswith("# See LICENSE"):
            header.append(lines[index])
            index += 1
            continue
        if stripped == "" and header:
            header.append(lines[index])
            index += 1
            continue
        break

    if header and any(AMD_COPYRIGHT_RELAXED_RE.search(line) for line in header):
        return header, lines[index:]
    return [], lines


def uses_cpp_banner_format(lines: list[str]) -> bool:
    for line in lines[:HEADER_SCAN_LINES]:
        if line.strip():
            return bool(CPP_BANNER_START_RE.match(line.strip()))
    return False


def amd_copyright_line(year: str) -> str:
    return f"# Copyright (c) {year}, Advanced Micro Devices, Inc. All rights reserved."


def default_cpp_banner_inner(year: str) -> list[str]:
    return [
        f" * Copyright (c) {year}, Advanced Micro Devices, Inc. All rights reserved.",
        " *",
        " * See LICENSE for license information.",
    ]


def build_cpp_banner_header(inner_lines: list[str]) -> list[str]:
    return [CPP_BANNER_START, *inner_lines, CPP_BANNER_END]


def extract_amd_copyright_year(lines: list[str], scan_lines: int = HEADER_SCAN_LINES) -> str:
    for line in lines[:scan_lines]:
        if AMD_COPYRIGHT_RE.search(line):
            year_match = COPYRIGHT_YEAR_RE.search(line)
            if year_match:
                return year_match.group(1)
    return current_year()


def should_skip_file(path: Path) -> bool:
    resolved = path.resolve()
    try:
        rel = resolved.relative_to(REPO_ROOT.resolve())
    except ValueError:
        return False

    if rel in SKIP_RELATIVE_PATHS:
        return True

    return any(part in SKIP_DIR_NAMES for part in rel.parts)


def read_file(path: Path) -> tuple[list[str], bool]:
    text = path.read_text(encoding="utf-8")
    return text.splitlines(), text.endswith("\n")


def write_file(path: Path, lines: list[str], trailing_newline: bool) -> None:
    content = "\n".join(lines)
    if trailing_newline:
        content += "\n"
    path.write_text(content, encoding="utf-8")


def extract_python_prefix(lines: list[str]) -> list[str]:
    prefix: list[str] = []
    for line in lines:
        if line.startswith("#!") or PYTHON_CODING_RE.match(line):
            prefix.append(line)
            continue
        break
    return prefix


def find_header_start(lines: list[str], file_type: str) -> int | None:
    for index, line in enumerate(lines):
        if file_type == "python":
            if line.startswith("#!") or PYTHON_CODING_RE.match(line):
                continue
            if line == PYTHON_HEADER_START:
                return index
            if line and not line.startswith("#"):
                return None
        elif line.startswith("// Copyright (c)"):
            return index
        elif line and not line.startswith("//"):
            return None
    return None


def find_python_header_end(lines: list[str], start: int) -> int:
    for index in range(start + 1, len(lines)):
        if lines[index] == PYTHON_HEADER_START:
            return index
    return start


def find_cpp_header_end(lines: list[str], start: int) -> int:
    index = start
    while index < len(lines) and lines[index].startswith("//"):
        index += 1
    return index - 1


def strip_leading_blank_lines(lines: list[str]) -> list[str]:
    body = list(lines)
    while body and body[0] == "":
        body.pop(0)
    return body


def join_header_and_body(header: list[str], body: list[str]) -> list[str]:
    if not body:
        return header
    return header + [""] + body


def file_type_for(path: Path) -> str | None:
    if path.suffix == ".py":
        return "python"
    if is_cpp_file(path):
        return "cpp"
    return None


def expected_header(path: Path, year: str) -> list[str]:
    if not requires_strict_header(path):
        raise ValueError(f"Strict header template is not used for: {path}")
    return flydsl_header(year)


def extract_header_year(lines: list[str], header_start: int, file_type: str) -> str:
    if file_type == "python":
        header_end = find_python_header_end(lines, header_start)
        header_region = lines[header_start : header_end + 1]
    else:
        header_end = find_cpp_header_end(lines, header_start)
        header_region = lines[header_start : header_end + 1]

    for line in header_region:
        match = COPYRIGHT_YEAR_RE.search(line)
        if match:
            return match.group(1)
    return current_year()


def extract_cpp_banner_block(lines: list[str]) -> tuple[list[str], list[str]] | None:
    if not lines or not CPP_BANNER_START_RE.match(lines[0].strip()):
        return None

    for index in range(1, min(len(lines), HEADER_SCAN_LINES)):
        if CPP_BANNER_END_RE.match(lines[index].strip()):
            return lines[: index + 1], lines[index + 1 :]
    return None


def extract_inner_from_banner(header: list[str]) -> list[str]:
    inner: list[str] = []
    for line in header[1:-1]:
        stripped = line.strip()
        if not stripped.startswith("*"):
            continue
        content = stripped[1:].strip()
        inner.append(f" * {content}" if content else " *")
    return inner


def collect_leading_banner_blocks(lines: list[str]) -> tuple[list[list[str]], list[str]]:
    blocks: list[list[str]] = []
    remainder = list(lines)
    while True:
        remainder = strip_leading_blank_lines(remainder)
        block = extract_cpp_banner_block(remainder)
        if block is None:
            break
        header, remainder = block
        blocks.append(header)
    return blocks, remainder


def choose_best_banner_block(blocks: list[list[str]]) -> list[str] | None:
    if not blocks:
        return None

    def score(block: list[str]) -> tuple[int, int, int, int]:
        text = "\n".join(block)
        return (
            len(block),
            1 if "Adapted from" in text else 0,
            1 if "HipKittens" in text else 0,
            1 if AMD_COPYRIGHT_RE.search(text) else 0,
        )

    return max(blocks, key=score)


def extract_leading_line_comment_header(lines: list[str]) -> tuple[list[str], list[str]]:
    header: list[str] = []
    index = 0
    while index < len(lines) and index < HEADER_SCAN_LINES:
        stripped = lines[index].strip()
        if stripped.startswith("//") or (stripped == "" and header):
            header.append(lines[index])
            index += 1
            continue
        break

    if header and any(AMD_COPYRIGHT_RE.search(line) for line in header):
        return header, lines[index:]
    return [], lines


def convert_comment_lines_to_banner_inner(header_lines: list[str]) -> list[str]:
    inner: list[str] = []
    for line in header_lines:
        stripped = line.strip()
        if not stripped.startswith("//"):
            continue
        content = stripped[2:].strip()
        inner.append(f" * {content}" if content else " *")
    return inner


def build_cpp_fixed_lines(lines: list[str]) -> list[str]:
    blocks, body = collect_leading_banner_blocks(lines)

    if blocks:
        best = choose_best_banner_block(blocks)
        inner = extract_inner_from_banner(best) if best else default_cpp_banner_inner(current_year())
    else:
        header_lines, body = extract_leading_line_comment_header(lines)
        if header_lines:
            inner = convert_comment_lines_to_banner_inner(header_lines)
            body = strip_leading_blank_lines(body)
        else:
            inner = default_cpp_banner_inner(extract_amd_copyright_year(lines))
            body = strip_leading_blank_lines(lines)

    if not any(AMD_COPYRIGHT_RE.search(line) for line in inner):
        year = extract_amd_copyright_year(lines)
        inner.insert(0, f" * Copyright (c) {year}, Advanced Micro Devices, Inc. All rights reserved.")

    new_header = build_cpp_banner_header(inner)
    new_lines = join_header_and_body(new_header, body)
    if len(blocks) > 1 or new_lines != lines:
        return new_lines
    return lines


def build_python_fixed_lines(path: Path, lines: list[str]) -> list[str]:
    prefix = extract_python_prefix(lines)
    body = lines[len(prefix) :]

    if has_python_block_header(body):
        header_start = find_header_start(body, "python")
        if header_start is None:
            return lines

        header_end = find_python_header_end(body, header_start)
        block = body[header_start : header_end + 1]
        rest = strip_duplicate_inline_python_header(body[header_end + 1 :])
        new_lines = prefix + join_header_and_body(block, rest)
        return new_lines if new_lines != lines else lines

    inline_header, rest = extract_leading_inline_python_header(body)
    year = extract_amd_copyright_year(body if inline_header else lines)
    header = default_python_block_header(year)
    rest = strip_leading_blank_lines(rest if inline_header else body)
    return prefix + join_header_and_body(header, rest)


def build_fixed_lines(path: Path, lines: list[str]) -> list[str]:
    if requires_strict_header(path):
        return build_strict_fixed_lines(path, lines)

    if is_cpp_file(path):
        return build_cpp_fixed_lines(lines)

    return build_python_fixed_lines(path, lines)


def build_strict_fixed_lines(path: Path, lines: list[str]) -> list[str]:
    file_type = file_type_for(path)
    if file_type is None:
        return lines

    header_start = find_header_start(lines, file_type)
    year = extract_header_year(lines, header_start, file_type) if header_start is not None else current_year()
    expected_lines = expected_header(path, year)

    prefix = extract_python_prefix(lines)
    if header_start is not None:
        header_end = find_python_header_end(lines, header_start)
        body = strip_leading_blank_lines(lines[header_end + 1 :])
    else:
        body = strip_leading_blank_lines(lines[len(prefix) :])
    return prefix + join_header_and_body(expected_lines, body)


def compare_header(path: Path, actual_lines: list[str], expected_lines: list[str]) -> list[str]:
    if normalize_copyright_years(actual_lines) == normalize_copyright_years(expected_lines):
        return []

    message = [f"ERROR: Invalid copyright header in {path}", "  Expected:"]
    message.extend(f"    {line}" for line in expected_lines)
    message.append("  Actual:")
    message.extend(f"    {line}" for line in actual_lines)
    return message


def check_file(path: Path) -> list[str]:
    if should_skip_file(path):
        return []

    if path.suffix not in SOURCE_SUFFIXES:
        return []

    try:
        lines, _ = read_file(path)
    except OSError as exc:
        return [f"ERROR: Failed to read {path}: {exc}"]

    if not requires_strict_header(path):
        if is_cpp_file(path):
            errors: list[str] = []
            if not uses_cpp_banner_format(lines):
                errors.append(f"ERROR: C++ license header must use banner block format in {path}")
            if not has_amd_copyright(lines):
                errors.append(
                    f"ERROR: Missing AMD copyright in {path}: expected a line containing "
                    "'Copyright (c) <year>, Advanced Micro Devices, Inc. All rights reserved.'"
                )
            return errors

        errors: list[str] = []
        if not has_python_block_header(lines):
            errors.append(f"ERROR: Python license header must use block comment format in {path}")
        elif has_duplicate_inline_python_header(lines):
            errors.append(
                f"ERROR: Duplicate inline copyright header after block header in {path}: "
                "keep only the ############################################################################### block"
            )
        if not has_amd_copyright_relaxed(lines):
            errors.append(
                f"ERROR: Missing AMD copyright in {path}: expected a line containing "
                "'Advanced Micro Devices, Inc. All rights reserved.'"
            )
        return errors

    file_type = file_type_for(path)
    if file_type is None:
        return []

    header_start = find_header_start(lines, file_type)
    if header_start is None:
        return [f"ERROR: Missing copyright header in {path}"]

    year = extract_header_year(lines, header_start, file_type)
    expected_lines = expected_header(path, year)
    actual_lines = lines[header_start : header_start + len(expected_lines)]

    return compare_header(path, actual_lines, expected_lines)


def fix_file(path: Path) -> bool:
    if should_skip_file(path) or path.suffix not in SOURCE_SUFFIXES:
        return False

    try:
        lines, trailing_newline = read_file(path)
    except OSError as exc:
        print(f"ERROR: Failed to read {path}: {exc}", file=sys.stderr)
        return False

    if file_type_for(path) is None:
        return False

    new_lines = build_fixed_lines(path, lines)
    if new_lines == lines:
        return False

    write_file(path, new_lines, trailing_newline)
    return True


def discover_files() -> list[Path]:
    files: list[Path] = []
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix not in SOURCE_SUFFIXES:
            continue
        if should_skip_file(path):
            continue
        files.append(path)
    return sorted(files)


def collect_input_files(inputs: list[str]) -> list[Path]:
    if not inputs:
        return discover_files()

    files: list[Path] = []
    for raw_input in inputs:
        path = Path(raw_input)
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()

        if path.is_dir():
            for candidate in sorted(path.rglob("*")):
                if candidate.is_file() and candidate.suffix in SOURCE_SUFFIXES:
                    if not should_skip_file(candidate.resolve()):
                        files.append(candidate.resolve())
        elif path.is_file():
            files.append(path.resolve())
        else:
            print(f"WARNING: Skipping missing path: {raw_input}", file=sys.stderr)

    unique_files: list[Path] = []
    seen: set[Path] = set()
    for file in files:
        if file not in seen:
            seen.add(file)
            unique_files.append(file)
    return unique_files


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check and optionally fix copyright header format in source files.",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="Files or directories to check. Defaults to scanning the repository.",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Automatically fix copyright header format.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    files = collect_input_files(args.paths)

    if not files:
        print("No files to check.")
        return 0

    if args.fix:
        fixed_files = [file for file in files if fix_file(file)]
        for file in fixed_files:
            print(f"FIXED: {file}")
        if fixed_files:
            print(f"Updated {len(fixed_files)} file(s).")

    errors: list[str] = []
    for file in files:
        errors.extend(check_file(file))

    for error in errors:
        print(error)

    if errors:
        print("License check failed.")
        return 1

    print(f"License check passed for {len(files)} file(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
