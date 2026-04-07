"""Standardized benchmark command builder with GPU isolation."""

from __future__ import annotations

from pathlib import Path

_BENCH_SCRIPTS = {
    "gemm": "benchmark/ops/bench_gemm_turbo.py",
    "grouped_gemm": "benchmark/ops/bench_grouped_gemm_turbo.py",
    "attention": "benchmark/ops/bench_attention_turbo.py",
}

_ACCURACY_SCRIPTS = {
    "gemm": "benchmark/accuracy/eval_gemm_accuracy.py",
}


class BenchmarkRunner:
    def __init__(self, repo_root: str | Path, gpu_id: int):
        self.repo_root = Path(repo_root)
        self.gpu_id = gpu_id

    def _env_prefix(self) -> str:
        return f"HIP_VISIBLE_DEVICES={self.gpu_id}"

    def build_benchmark_cmd(
        self,
        operator: str,
        dtype: str = "bf16",
        granularity: str | None = None,
        output_csv: str | None = None,
    ) -> str:
        if operator not in _BENCH_SCRIPTS:
            raise ValueError(f"Unknown operator: {operator}. Valid: {list(_BENCH_SCRIPTS)}")

        script = self.repo_root / _BENCH_SCRIPTS[operator]
        parts = [self._env_prefix(), "python", str(script), f"--dtype {dtype}"]

        if granularity and dtype in ("fp8", "fp4"):
            parts.append(f"--granularity {granularity}")
        if output_csv:
            parts.append(f"--output {output_csv}")

        return " ".join(parts)

    def build_accuracy_cmd(
        self, operator: str, report_dir: str, seed: int = 42
    ) -> str:
        if operator not in _ACCURACY_SCRIPTS:
            raise ValueError(f"No accuracy script for operator: {operator}")

        script = self.repo_root / _ACCURACY_SCRIPTS[operator]
        return f"{self._env_prefix()} python {script} --report-dir-path {report_dir} --seed {seed}"
