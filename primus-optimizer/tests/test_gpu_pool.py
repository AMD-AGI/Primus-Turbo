import pytest
from tools.gpu_pool import GPUPool


def test_acquire_returns_gpu():
    pool = GPUPool([0, 1, 2])
    gpu = pool.acquire("gemm:triton")
    assert gpu in [0, 1, 2]
    assert pool.available_count() == 2


def test_release_returns_gpu_to_pool():
    pool = GPUPool([0, 1])
    pool.acquire("task1")
    pool.release("task1")
    assert pool.available_count() == 2


def test_acquire_exhausted_raises():
    pool = GPUPool([0])
    pool.acquire("task1")
    with pytest.raises(RuntimeError, match="No GPUs available"):
        pool.acquire("task2")


def test_release_unknown_raises():
    pool = GPUPool([0])
    with pytest.raises(KeyError):
        pool.release("unknown")


def test_allocated_map():
    pool = GPUPool([0, 1])
    pool.acquire("task1")
    pool.acquire("task2")
    alloc = pool.allocated_map()
    assert len(alloc) == 2
    assert "task1" in alloc
    assert "task2" in alloc


def test_has_available():
    pool = GPUPool([0])
    assert pool.has_available() is True
    pool.acquire("t")
    assert pool.has_available() is False


def test_env_var():
    pool = GPUPool([3])
    gpu = pool.acquire("task1")
    assert pool.env_for("task1") == {"HIP_VISIBLE_DEVICES": "3"}
