###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Enforce provenance and integration policy for vendored Gluon kernels."""

import ast
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
VENDORED_FILES = (
    Path("primus_turbo/gluon/attention/f16_fa_gfx950_common.py"),
    Path("primus_turbo/gluon/attention/f16_fa_gfx950_rotated_4cluster.py"),
)
KERNEL_FILE = VENDORED_FILES[1]

PROVENANCE_MARKERS = (
    "# Vendored from https://github.com/AMD-Triton/gluon-kernels",
    "# Source branch: bangtian/fa-fwd-gfx950-gluon-optimized",
    "# Source commit: 05b349b545ef713cd0ba41a3d89ddf3e3eb6b2c3",
)
REQUIRED_KERNEL_ENTRY_POINTS = (
    "gluon_attn_fwd",
    "gluon_attn_fwd_short_causal_classes",
)
FORBIDDEN_TOKENS = (
    "argparse",
    "gluon_attn_fwd_persistent",
    "flash_attn_rotated_4cluster_persistent",
)
FORBIDDEN_TOKEN_PREFIXES = ("LLVM_PASS_PLUGIN",)
FORBIDDEN_IMPORT_ROOTS = {"ctypes"}
FORBIDDEN_FROM_IMPORTS = {("sys", "*"), ("sys", "setdlopenflags")}
FORBIDDEN_DYNAMIC_LOADER_CALLS = {
    "CDLL",
    "OleDLL",
    "PyDLL",
    "WinDLL",
    "ctypes.CDLL",
    "ctypes.OleDLL",
    "ctypes.PyDLL",
    "ctypes.WinDLL",
    "ctypes.cdll.LoadLibrary",
    "ctypes.pydll.LoadLibrary",
    "ctypes.windll.LoadLibrary",
    "sys.setdlopenflags",
}

EXPECTED_ATTRIBUTE_CONSTANTS = {
    "VGPR_ONLY_FN_ATTRS": (("amdgpu-agpr-alloc", "0,0"),),
    "NO_DISPATCH_ID_FN_ATTRS": (("amdgpu-no-dispatch-id", ""),),
    "VGPR_ONLY_PRELOAD_FN_ATTRS": (
        ("amdgpu-agpr-alloc", "0,0"),
        ("amdgpu-no-dispatch-id", ""),
    ),
    "VGPR_ONLY_PRELOAD_MAX_ILP_FN_ATTRS": (
        ("amdgpu-agpr-alloc", "0,0"),
        ("amdgpu-no-dispatch-id", ""),
        ("amdgpu-sched-strategy", "max-ilp"),
    ),
}
EXPECTED_OPTIMIZED_CONFIGS = (
    (256, 32, False, 2, 2, 8, EXPECTED_ATTRIBUTE_CONSTANTS["VGPR_ONLY_FN_ATTRS"]),
    (256, 64, False, 3, 2, 8, EXPECTED_ATTRIBUTE_CONSTANTS["VGPR_ONLY_FN_ATTRS"]),
    (
        256,
        64,
        False,
        4,
        2,
        8,
        EXPECTED_ATTRIBUTE_CONSTANTS["VGPR_ONLY_PRELOAD_FN_ATTRS"],
    ),
    (
        256,
        64,
        False,
        4,
        2,
        8,
        EXPECTED_ATTRIBUTE_CONSTANTS["VGPR_ONLY_PRELOAD_MAX_ILP_FN_ATTRS"],
    ),
)


def _dotted_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        owner = _dotted_name(node.value)
        return f"{owner}.{node.attr}" if owner else node.attr
    return None


def _import_aliases(tree):
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for imported in node.names:
                bound_name = imported.asname or imported.name.split(".", 1)[0]
                aliases[bound_name] = imported.name if imported.asname else imported.name.split(".", 1)[0]
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            for imported in node.names:
                if imported.name != "*":
                    aliases[imported.asname or imported.name] = f"{node.module}.{imported.name}"
    return aliases


def _canonical_dotted_name(node, aliases):
    name = _dotted_name(node)
    if name is None:
        return None
    root, separator, remainder = name.partition(".")
    canonical_root = aliases.get(root, root)
    return f"{canonical_root}.{remainder}" if separator else canonical_root


def _string_pair(node):
    if not isinstance(node, ast.Tuple) or len(node.elts) != 2:
        return None
    if not all(isinstance(element, ast.Constant) and isinstance(element.value, str) for element in node.elts):
        return None
    return tuple(element.value for element in node.elts)


def _attribute_assignments(tree):
    return {
        node.targets[0].id: node.value
        for node in tree.body
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
    }


def _resolve_attribute_sequence(node, assignments, resolving=()):
    if isinstance(node, ast.Name):
        if node.id in resolving or node.id not in assignments:
            return None
        return _resolve_attribute_sequence(assignments[node.id], assignments, (*resolving, node.id))
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _resolve_attribute_sequence(node.left, assignments, resolving)
        right = _resolve_attribute_sequence(node.right, assignments, resolving)
        return None if left is None or right is None else left + right
    if isinstance(node, ast.Tuple):
        attributes = []
        for element in node.elts:
            attribute = _string_pair(element)
            if attribute is None:
                return None
            attributes.append(attribute)
        return tuple(attributes)
    return None


def _literal_dict(node):
    if not isinstance(node, ast.Dict):
        return None
    result = {}
    for key, value in zip(node.keys, node.values, strict=True):
        if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
            return None
        result[key.value] = value
    return result


def _literal_constant(node):
    return node.value if isinstance(node, ast.Constant) else None


def _optimized_config_signature(node, assignments):
    if not isinstance(node, ast.Call) or _dotted_name(node.func) != "triton.Config":
        return None
    if not node.args or (config := _literal_dict(node.args[0])) is None:
        return None
    if "llvm_fn_attrs" not in config:
        return None

    num_warps = next(
        (_literal_constant(keyword.value) for keyword in node.keywords if keyword.arg == "num_warps"),
        None,
    )
    return (
        _literal_constant(config.get("BLOCK_M")),
        _literal_constant(config.get("BLOCK_N")),
        _literal_constant(config.get("PRE_LOAD_V")),
        _literal_constant(config.get("NUM_STAGES")),
        _literal_constant(config.get("waves_per_eu")),
        num_warps,
        _resolve_attribute_sequence(config["llvm_fn_attrs"], assignments),
    )


def main() -> int:
    errors = []
    sources = {}
    trees = {}

    for relative_path in VENDORED_FILES:
        path = REPO_ROOT / relative_path
        try:
            source = path.read_text(encoding="utf-8")
        except OSError as error:
            errors.append(f"{relative_path}: cannot read file: {error}")
            continue

        sources[relative_path] = source
        for marker in PROVENANCE_MARKERS:
            if marker not in source:
                errors.append(f"{relative_path}: missing provenance marker: {marker}")
        for token in FORBIDDEN_TOKENS:
            if token in source:
                errors.append(f"{relative_path}: forbidden token: {token}")
        for prefix in FORBIDDEN_TOKEN_PREFIXES:
            if prefix in source:
                errors.append(f"{relative_path}: forbidden token prefix: {prefix}*")

        try:
            tree = ast.parse(source, filename=str(relative_path))
        except SyntaxError as error:
            errors.append(f"{relative_path}: cannot parse file: {error}")
            continue
        trees[relative_path] = tree
        aliases = _import_aliases(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for imported in node.names:
                    if imported.name.split(".", 1)[0] in FORBIDDEN_IMPORT_ROOTS:
                        errors.append(f"{relative_path}: forbidden import: {imported.name}")
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                module_root = node.module.split(".", 1)[0]
                for imported in node.names:
                    if (
                        module_root in FORBIDDEN_IMPORT_ROOTS
                        or (node.module, imported.name) in FORBIDDEN_FROM_IMPORTS
                    ):
                        errors.append(
                            f"{relative_path}: forbidden import: from {node.module} import {imported.name}"
                        )
            elif isinstance(node, ast.Call):
                call_name = _canonical_dotted_name(node.func, aliases)
                if call_name in FORBIDDEN_DYNAMIC_LOADER_CALLS:
                    errors.append(f"{relative_path}: forbidden dynamic-loader call: {call_name}")

    kernel_source = sources.get(KERNEL_FILE)
    tree = trees.get(KERNEL_FILE)
    if kernel_source is not None and tree is not None:
        top_level_functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
        for entry_point in REQUIRED_KERNEL_ENTRY_POINTS:
            if entry_point not in top_level_functions:
                errors.append(f"{KERNEL_FILE}: missing entry point: {entry_point}")

        assignments = _attribute_assignments(tree)
        for name, expected in EXPECTED_ATTRIBUTE_CONSTANTS.items():
            actual = _resolve_attribute_sequence(assignments.get(name), assignments)
            if actual != expected:
                errors.append(f"{KERNEL_FILE}: {name} must resolve to {expected}, found {actual}")

        actual_configs = tuple(
            signature
            for node in ast.walk(tree)
            if (signature := _optimized_config_signature(node, assignments)) is not None
        )
        if Counter(actual_configs) != Counter(EXPECTED_OPTIMIZED_CONFIGS):
            errors.append(
                f"{KERNEL_FILE}: optimized AMDGPU autotune candidates do not match "
                f"the pinned source; expected {EXPECTED_OPTIMIZED_CONFIGS}, "
                f"found {actual_configs}"
            )

    if errors:
        print("Gluon vendor policy audit failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print("Gluon vendor policy audit passed for both vendored source files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
