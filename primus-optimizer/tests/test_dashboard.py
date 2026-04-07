import json
import os
import tempfile
import pytest
from tools.dashboard import (
    format_status_icon,
    format_worker_table_data,
    format_gain,
)


def test_format_status_icon():
    assert format_status_icon("running") == "▶ RUN"
    assert format_status_icon("bottleneck") == "⚠ BTNK"
    assert format_status_icon("completed") == "✓ DONE"
    assert format_status_icon("failed") == "✗ FAIL"
    assert format_status_icon("pending") == "◻ WAIT"


def test_format_gain():
    assert format_gain(62.5) == "+62.5%"
    assert format_gain(-3.2) == "-3.2%"
    assert format_gain(0.0) == "+0.0%"
    assert format_gain(None) == "-"


def test_format_worker_table_data():
    workers = {
        "gemm:triton": {
            "status": "running",
            "gpu_id": 0,
            "current_round": 5,
            "rounds": [
                {"round": 1, "baseline_tflops": 488.1, "result_tflops": 793.2, "improvement_pct": 62.5}
            ],
            "bottleneck": None,
        }
    }
    rows = format_worker_table_data(workers, max_rounds=10)
    assert len(rows) == 1
    row = rows[0]
    assert row["task"] == "gemm:triton"
    assert row["status"] == "▶ RUN"
    assert row["gpu"] == "0"
    assert row["round"] == "5/10"
    assert "488" in row["tflops"]
    assert "793" in row["tflops"]


def test_format_worker_table_pending():
    workers = {
        "gemm:ck": {
            "status": "pending",
            "gpu_id": None,
            "current_round": 0,
            "rounds": [],
            "bottleneck": None,
        }
    }
    rows = format_worker_table_data(workers, max_rounds=10)
    assert rows[0]["gpu"] == "-"
    assert rows[0]["round"] == "-"
