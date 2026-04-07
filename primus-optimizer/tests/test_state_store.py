import json
import os
import tempfile
import pytest
from tools.state_store import StateStore, WorkerStatus


@pytest.fixture
def tmp_state_path():
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.unlink(path)


def test_init_creates_state(tmp_state_path):
    store = StateStore(tmp_state_path)
    store.init_session(
        session_id="test-001",
        hw="mi355x",
        tasks=[{"operator": "gemm", "backend": "triton"}],
        gpu_ids=[0, 1],
    )
    state = store.load()
    assert state["session_id"] == "test-001"
    assert state["config"]["hw"] == "mi355x"
    assert len(state["workers"]) == 0
    assert state["task_queue"] == ["gemm:triton"]


def test_add_worker(tmp_state_path):
    store = StateStore(tmp_state_path)
    store.init_session("s1", "mi355x", [{"operator": "gemm", "backend": "ck"}], [0])
    store.add_worker("gemm:ck", gpu_id=0, worktree_branch="opt/gemm-ck-001")
    state = store.load()
    assert "gemm:ck" in state["workers"]
    assert state["workers"]["gemm:ck"]["status"] == WorkerStatus.RUNNING
    assert state["workers"]["gemm:ck"]["gpu_id"] == 0


def test_record_round(tmp_state_path):
    store = StateStore(tmp_state_path)
    store.init_session("s1", "mi355x", [{"operator": "gemm", "backend": "triton"}], [0])
    store.add_worker("gemm:triton", 0, "opt/gemm-triton-001")
    store.record_round(
        worker_id="gemm:triton",
        round_num=1,
        strategy="persistent_kernel",
        baseline_tflops=488.1,
        result_tflops=614.2,
        status="success",
    )
    state = store.load()
    w = state["workers"]["gemm:triton"]
    assert w["current_round"] == 1
    assert len(w["rounds"]) == 1
    assert w["rounds"][0]["improvement_pct"] == pytest.approx(25.8, abs=0.1)


def test_update_worker_status(tmp_state_path):
    store = StateStore(tmp_state_path)
    store.init_session("s1", "mi355x", [{"operator": "gemm", "backend": "triton"}], [0])
    store.add_worker("gemm:triton", 0, "opt/gemm-triton-001")
    store.update_worker_status("gemm:triton", WorkerStatus.BOTTLENECK, bottleneck_level="L1")
    state = store.load()
    assert state["workers"]["gemm:triton"]["status"] == WorkerStatus.BOTTLENECK
    assert state["workers"]["gemm:triton"]["bottleneck"] == "L1"


def test_add_review_log(tmp_state_path):
    store = StateStore(tmp_state_path)
    store.init_session("s1", "mi355x", [], [0])
    store.add_review_log(trigger="milestone", worker="gemm:triton", round_num=5, decision="continue")
    state = store.load()
    assert len(state["review_log"]) == 1
    assert state["review_log"][0]["trigger"] == "milestone"


def test_pop_task_queue(tmp_state_path):
    store = StateStore(tmp_state_path)
    store.init_session("s1", "mi355x", [
        {"operator": "gemm", "backend": "triton"},
        {"operator": "gemm", "backend": "ck"},
    ], [0])
    task = store.pop_task_queue()
    assert task == "gemm:triton"
    task2 = store.pop_task_queue()
    assert task2 == "gemm:ck"
    task3 = store.pop_task_queue()
    assert task3 is None
