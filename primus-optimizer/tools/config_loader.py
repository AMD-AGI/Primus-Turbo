"""Load optimizer configuration from YAML and parse CLI task specifications."""

from __future__ import annotations

import os
from itertools import product
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "optimizer.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML config, falling back to the default bundled config."""
    path = Path(path) if path else _DEFAULT_CONFIG_PATH
    with open(path) as f:
        return yaml.safe_load(f)


def parse_tasks(
    tasks_str: str | None = None,
    operators: str | None = None,
    backends: str | None = None,
) -> list[dict[str, str]]:
    """Parse task specifications into a list of {operator, backend} dicts.

    Supports explicit ('gemm:ck,attn:triton'), cartesian (operators x backends),
    or mixed (both).
    """
    result: list[dict[str, str]] = []

    # Explicit tasks
    if tasks_str:
        for item in tasks_str.split(","):
            op, be = item.strip().split(":")
            result.append({"operator": op, "backend": be})

    # Cartesian product
    if operators and backends:
        ops = [o.strip() for o in operators.split(",")]
        bes = [b.strip() for b in backends.split(",")]
        for op, be in product(ops, bes):
            entry = {"operator": op, "backend": be}
            if entry not in result:
                result.append(entry)

    return result


def resolve_profile(
    cfg: dict[str, Any], profile_name: str
) -> tuple[str, list[dict[str, str]]]:
    """Resolve a named profile into (hw, tasks) pair."""
    profile = cfg["profiles"][profile_name]
    hw = profile["hw"]

    if "tasks" in profile:
        return hw, profile["tasks"]

    # Cartesian product from operators x backends
    ops = profile["operators"]
    bes = profile["backends"]
    tasks = [{"operator": op, "backend": be} for op, be in product(ops, bes)]
    return hw, tasks
