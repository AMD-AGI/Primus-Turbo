"""sys.path / env preamble shared by every module in this job.

Import this BEFORE `import flydsl` or `import aiter`, from any implementation directory.
It is duplicated per implementation on purpose: `op/baseline/` must stay importable on its
own after the framework copies it into `rounds/<n>/op/`.

Order is fixed by runtime.python_path in the spec:
  /home/lihuzhan/.local/flydsl032   flydsl 0.3.2, ahead of the image's 0.2.4
  /home/lihuzhan/code/aiter-src     aiter, not installed in the container
NEVER import primus_turbo in this process: its FlyDSL tree imports
`flydsl.expr.buffer_ops`, which 0.3.2 removed.
"""
import os
import sys

os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "0")

for _p in ("/home/lihuzhan/.local/flydsl0341",):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
        break
if "/home/lihuzhan/code/aiter-src" not in sys.path:
    sys.path.insert(0, "/home/lihuzhan/code/aiter-src")


def assert_environment():
    """Assert the two things that silently produce wrong answers if they are wrong."""
    import flydsl
    import torch
    assert flydsl.__version__.startswith("0.3.4"), (
        f"flydsl 0.3.4.x required, got {flydsl.__version__} at {flydsl.__file__}")
    assert "flydsl0341" in flydsl.__file__, (
        f"flydsl resolved to the image copy, not the 0.3.4.1 install: {flydsl.__file__}")
    arch = torch.cuda.get_device_properties(0).gcnArchName
    assert "gfx1250" in arch, f"these kernels target gfx1250, got {arch}"
    return flydsl.__version__, flydsl.__file__, arch
