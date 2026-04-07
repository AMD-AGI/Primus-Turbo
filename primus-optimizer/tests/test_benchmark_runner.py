import pytest
from tools.benchmark_runner import BenchmarkRunner


def test_build_benchmark_cmd_gemm():
    runner = BenchmarkRunner(repo_root="/repo", gpu_id=2)
    cmd = runner.build_benchmark_cmd(
        operator="gemm", dtype="fp8", granularity="blockwise", output_csv="/out/result.csv"
    )
    assert "HIP_VISIBLE_DEVICES=2" in cmd
    assert "benchmark/ops/bench_gemm_turbo.py" in cmd
    assert "--dtype fp8" in cmd
    assert "--granularity blockwise" in cmd
    assert "--output /out/result.csv" in cmd


def test_build_benchmark_cmd_grouped_gemm():
    runner = BenchmarkRunner(repo_root="/repo", gpu_id=0)
    cmd = runner.build_benchmark_cmd(operator="grouped_gemm", dtype="bf16")
    assert "bench_grouped_gemm_turbo.py" in cmd
    assert "--dtype bf16" in cmd


def test_build_benchmark_cmd_attention():
    runner = BenchmarkRunner(repo_root="/repo", gpu_id=1)
    cmd = runner.build_benchmark_cmd(operator="attention")
    assert "bench_attention_turbo.py" in cmd


def test_build_accuracy_cmd():
    runner = BenchmarkRunner(repo_root="/repo", gpu_id=3)
    cmd = runner.build_accuracy_cmd(operator="gemm", report_dir="/out/accuracy")
    assert "HIP_VISIBLE_DEVICES=3" in cmd
    assert "eval_gemm_accuracy.py" in cmd
    assert "--report-dir-path /out/accuracy" in cmd


def test_unknown_operator_raises():
    runner = BenchmarkRunner(repo_root="/repo", gpu_id=0)
    with pytest.raises(ValueError, match="Unknown operator"):
        runner.build_benchmark_cmd(operator="unknown_op")
