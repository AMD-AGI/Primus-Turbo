#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Verify that the flex-attention compat layer is not a copy of PyTorch's.

``primus_turbo/pytorch/ops/attention/flex_attention.py`` deliberately mirrors the
*interface* of ``torch/nn/attention/flex_attention.py`` -- same module name, same
``flex_attention`` / ``create_block_mask`` entry points, same ``score_mod`` /
``mask_mod`` calling convention -- because it is a drop-in replacement. That
resemblance invites a fair question from reviewers: is the implementation copied?
If it were, the repo's own convention (see ``tools/check_license.py``, which emits
a dual-copyright ``Adapted from ...`` banner for the FlyDSL tree) would require a
PyTorch BSD-3-Clause notice on the file.

This script answers that question mechanically instead of by assertion, and keeps
answering it: if someone later pastes torch code into the compat layer, it fails.

Three independent measures, all computed against the *installed* torch:

1. Shared top-level names, minus the interface names we intentionally match.
2. Per-function best textual similarity, comparing every local function against
   every torch function -- not just same-named ones, so a renamed copy is caught.
   Comments and blank lines are stripped so only code shape is compared.
3. K-gram fingerprint overlap over the raw text, which catches copied prose and
   copied fragments that are too small to dominate any single function.

Exit code 0 = clean, 1 = a threshold was exceeded, 0 with a notice = torch (or its
flex-attention module) is unavailable, so the check could not run.

Usage::

    python tools/check_flex_provenance.py
    python tools/check_flex_provenance.py --verbose
"""

from __future__ import annotations

import argparse
import ast
import difflib
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LOCAL_FILE = REPO_ROOT / "primus_turbo" / "pytorch" / "ops" / "attention" / "flex_attention.py"

# Names we match on purpose: this module is a drop-in replacement, so the public
# entry points share torch's names. Sharing a *name* is the point; sharing an
# *implementation* is what this script looks for.
ALLOWED_SHARED_NAMES = frozenset({"flex_attention", "create_block_mask"})

# A function that is >= 60% textually identical to a torch function (after comment
# stripping) is not plausibly independent work. Observed max on a clean tree is
# ~0.42, and that is a 6-line helper coinciding with another 6-line helper.
FUNCTION_SIMILARITY_LIMIT = 0.60

# Shared k-grams are dominated by unavoidable idiom -- `query: torch.Tensor, key:
# torch.Tensor`, `query.transpose(1, 2)`, the device-mismatch guard.
#
# Calibrated against measurement, not guessed. On a clean tree the overlap is
# 0.22% (52/23289). Pasting three whole torch functions (`or_masks`, `and_masks`,
# `_convert_mask_to_block_mask`) into the file raises it to only 2.58% -- so a
# threshold of 5% would have missed that injection entirely. 2% sits ~9x above the
# clean baseline and below a three-function paste, which makes this measure an
# actual tripwire rather than decoration. Note it is the weakest of the three
# checks by design: a copy large enough to matter is caught by the shared-name and
# per-function-similarity checks first, and this one exists to catch copied prose
# and fragments too small to dominate any single function.
SHINGLE_K = 12
SHINGLE_OVERLAP_LIMIT = 0.02

_TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def torch_attention_sources() -> list[Path]:
    """Locate the installed torch's attention sources, or return [] if absent."""
    try:
        import torch  # noqa: PLC0415  (optional dependency for this check)
    except Exception:
        return []

    attention_dir = Path(torch.__file__).resolve().parent / "nn" / "attention"
    if not attention_dir.is_dir():
        return []
    sources = sorted(p for p in attention_dir.glob("*.py") if p.is_file())

    # The HOP lowering lives outside nn/attention but is part of flex-attention.
    hop = Path(torch.__file__).resolve().parent / "_higher_order_ops" / "flex_attention.py"
    if hop.is_file():
        sources.append(hop)
    return sources


def top_level_units(path: Path) -> dict[str, str]:
    """Map top-level function/class name -> its source text."""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    units: dict[str, str] = {}
    for node in ast.parse(text).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            units[node.name] = "".join(lines[node.lineno - 1 : node.end_lineno])
    return units


def code_shape(source: str) -> str:
    """Drop comments and blank lines so only executable structure is compared."""
    kept = []
    for line in source.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            kept.append(stripped)
    return "\n".join(kept)


def shingles(text: str, k: int = SHINGLE_K) -> set[tuple[str, ...]]:
    tokens = _TOKEN_RE.findall(text)
    return {tuple(tokens[i : i + k]) for i in range(len(tokens) - k + 1)}


def best_similarity(local_src: str, torch_units: dict[str, str]) -> tuple[float, str | None]:
    """Best ratio of one local function against every torch function."""
    local_shape = code_shape(local_src)
    best, best_key = 0.0, None
    for key, torch_src in torch_units.items():
        torch_shape = code_shape(torch_src)
        if not torch_shape:
            continue
        matcher = difflib.SequenceMatcher(None, local_shape, torch_shape)
        # Cheap upper bounds first; both are >= the true ratio.
        if matcher.real_quick_ratio() <= best or matcher.quick_ratio() <= best:
            continue
        ratio = matcher.ratio()
        if ratio > best:
            best, best_key = ratio, key
    return best, best_key


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--verbose", action="store_true", help="Print every per-function score.")
    args = parser.parse_args(argv)

    if not LOCAL_FILE.is_file():
        print(f"ERROR: compat layer not found at {LOCAL_FILE}")
        return 1

    sources = torch_attention_sources()
    if not sources:
        print("NOTICE: torch (or torch.nn.attention) is unavailable; provenance check skipped.")
        return 0

    local_units = top_level_units(LOCAL_FILE)
    torch_units: dict[str, str] = {}
    for source in sources:
        try:
            for name, src in top_level_units(source).items():
                torch_units[f"{source.name}::{name}"] = src
        except SyntaxError:
            # A torch build newer than this interpreter can contain syntax we
            # cannot parse. Skipping one file only weakens the check, never
            # produces a false accusation.
            print(f"NOTICE: could not parse {source}; skipped.")

    print(f"local units: {len(local_units)}   torch units: {len(torch_units)}   torch files: {len(sources)}")

    errors: list[str] = []

    # --- 1. shared names -----------------------------------------------------
    torch_names = {key.split("::", 1)[1] for key in torch_units}
    shared = (set(local_units) & torch_names) - ALLOWED_SHARED_NAMES
    print(f"shared top-level names (excluding interface names): {sorted(shared) or 'none'}")
    if shared:
        errors.append(
            f"Unexpected shared top-level name(s) with torch: {sorted(shared)}. "
            "Either delegate to torch by import, or add to ALLOWED_SHARED_NAMES "
            "with a justification."
        )

    # --- 2. per-function similarity ------------------------------------------
    scored = []
    for name, src in local_units.items():
        ratio, match = best_similarity(src, torch_units)
        scored.append((ratio, name, match))
    scored.sort(reverse=True)

    top = scored[:10] if not args.verbose else scored
    print(f"\nper-function best similarity vs torch (top {len(top)} of {len(scored)}):")
    for ratio, name, match in top:
        print(f"  {ratio:5.3f}  {name:44s} -> {match}")

    for ratio, name, match in scored:
        if ratio >= FUNCTION_SIMILARITY_LIMIT:
            errors.append(
                f"{name} is {ratio:.1%} similar to torch's {match} "
                f"(limit {FUNCTION_SIMILARITY_LIMIT:.0%}). If it is derived from torch, this "
                "file needs a PyTorch BSD-3-Clause attribution banner; if it is not, "
                "restructure it or import torch's version."
            )

    # --- 3. fingerprint overlap ----------------------------------------------
    local_shingles = shingles(LOCAL_FILE.read_text(encoding="utf-8"))
    torch_shingles: set[tuple[str, ...]] = set()
    for source in sources:
        torch_shingles |= shingles(source.read_text(encoding="utf-8"))
    common = local_shingles & torch_shingles
    overlap = len(common) / max(1, len(local_shingles))
    print(
        f"\n{SHINGLE_K}-token fingerprint overlap: {len(common)}/{len(local_shingles)} "
        f"= {overlap:.4%}  (limit {SHINGLE_OVERLAP_LIMIT:.0%})"
    )
    if args.verbose and common:
        print("  shared fragments:")
        for shingle in sorted(common)[:20]:
            print("    " + " ".join(shingle)[:140])
    if overlap >= SHINGLE_OVERLAP_LIMIT:
        errors.append(
            f"Raw-text fingerprint overlap with torch is {overlap:.2%}, at or above the "
            f"{SHINGLE_OVERLAP_LIMIT:.0%} limit."
        )

    print()
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        print("Flex-attention provenance check FAILED.")
        return 1

    print(
        "Flex-attention provenance check passed: the compat layer matches torch's "
        "interface but shares no implementation, so the AMD-only header is correct."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
