import pytest
import os
import tempfile
import yaml
from tools.config_loader import load_config, parse_tasks, resolve_profile


def test_load_config_from_file():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump({"defaults": {"max_rounds": 5}, "gpu_pool": {"device_ids": [0, 1]}}, f)
        f.flush()
        cfg = load_config(f.name)
    os.unlink(f.name)
    assert cfg["defaults"]["max_rounds"] == 5
    assert cfg["gpu_pool"]["device_ids"] == [0, 1]


def test_parse_tasks_explicit():
    tasks = parse_tasks(tasks_str="gemm:ck,attention:triton", operators=None, backends=None)
    assert tasks == [
        {"operator": "gemm", "backend": "ck"},
        {"operator": "attention", "backend": "triton"},
    ]


def test_parse_tasks_cartesian():
    tasks = parse_tasks(tasks_str=None, operators="gemm,attention", backends="triton,ck")
    assert len(tasks) == 4
    assert {"operator": "gemm", "backend": "triton"} in tasks
    assert {"operator": "attention", "backend": "ck"} in tasks


def test_parse_tasks_mixed():
    tasks = parse_tasks(tasks_str="gemm:ck", operators="attention", backends="triton")
    assert len(tasks) == 2
    assert {"operator": "gemm", "backend": "ck"} in tasks
    assert {"operator": "attention", "backend": "triton"} in tasks


def test_resolve_profile():
    cfg = {
        "profiles": {
            "test-profile": {
                "hw": "mi355x",
                "tasks": [{"operator": "gemm", "backend": "ck", "max_rounds": 10}],
            }
        },
        "defaults": {"max_rounds": 10},
    }
    hw, tasks = resolve_profile(cfg, "test-profile")
    assert hw == "mi355x"
    assert tasks[0]["operator"] == "gemm"


def test_resolve_profile_cartesian():
    cfg = {
        "profiles": {
            "sweep": {
                "hw": "mi300x",
                "operators": ["gemm", "attention"],
                "backends": ["triton", "ck"],
            }
        },
        "defaults": {"max_rounds": 5},
    }
    hw, tasks = resolve_profile(cfg, "sweep")
    assert hw == "mi300x"
    assert len(tasks) == 4
