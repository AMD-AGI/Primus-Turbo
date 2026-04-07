import pytest
from tools.profiler import Profiler


def test_rocprof_cmd():
    p = Profiler(gpu_id=2)
    cmd = p.build_rocprof_cmd("python bench.py", output_dir="/out")
    assert "HIP_VISIBLE_DEVICES=2" in cmd
    assert "rocprof --stats" in cmd
    assert "-o /out/rocprof_stats.csv" in cmd
    assert "python bench.py" in cmd


def test_omniperf_cmd():
    p = Profiler(gpu_id=1)
    cmd = p.build_omniperf_cmd("python bench.py", output_dir="/out")
    assert "HIP_VISIBLE_DEVICES=1" in cmd
    assert "omniperf profile" in cmd
    assert "--path /out/omniperf" in cmd


def test_roofline_analysis_cmd():
    p = Profiler(gpu_id=0)
    cmd = p.build_roofline_cmd(omniperf_dir="/out/omniperf")
    assert "omniperf analyze" in cmd
    assert "--path /out/omniperf" in cmd
