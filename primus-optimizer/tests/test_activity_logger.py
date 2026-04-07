import json
import os
import tempfile
import pytest
from tools.activity_logger import ActivityLogger


@pytest.fixture
def tmp_log():
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.unlink(path)


def test_log_appends_jsonl(tmp_log):
    logger = ActivityLogger(tmp_log, worker_id="gemm:triton")
    logger.log("VERIFY", 3, "accuracy check passed")
    logger.log("VERIFY", 3, "benchmark complete: 793.2 TFLOPS")

    with open(tmp_log) as f:
        lines = [json.loads(line) for line in f if line.strip()]
    assert len(lines) == 2
    assert lines[0]["worker"] == "gemm:triton"
    assert lines[0]["phase"] == "VERIFY"
    assert lines[0]["round"] == 3
    assert "ts" in lines[0]


def test_read_recent(tmp_log):
    logger = ActivityLogger(tmp_log, worker_id="gemm:ck")
    for i in range(20):
        logger.log("PROFILE", i, f"msg {i}")
    entries = ActivityLogger.read_recent(tmp_log, n=5)
    assert len(entries) == 5
    assert entries[-1]["msg"] == "msg 19"
