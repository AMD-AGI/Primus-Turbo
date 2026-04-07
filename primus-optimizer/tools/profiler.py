"""Profiling command builders for rocprof and omniperf."""

from __future__ import annotations


class Profiler:
    def __init__(self, gpu_id: int):
        self.gpu_id = gpu_id

    def _env_prefix(self) -> str:
        return f"HIP_VISIBLE_DEVICES={self.gpu_id}"

    def build_rocprof_cmd(self, bench_cmd: str, output_dir: str) -> str:
        return (
            f"{self._env_prefix()} rocprof --stats "
            f"-o {output_dir}/rocprof_stats.csv "
            f"{bench_cmd}"
        )

    def build_omniperf_cmd(self, bench_cmd: str, output_dir: str) -> str:
        return (
            f"{self._env_prefix()} omniperf profile "
            f"--path {output_dir}/omniperf "
            f"-- {bench_cmd}"
        )

    def build_roofline_cmd(self, omniperf_dir: str) -> str:
        return f"omniperf analyze --path {omniperf_dir} --roof-only"
