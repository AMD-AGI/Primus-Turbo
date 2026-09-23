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

# [stage2 S0-b, 2026-09-23] was "0". hipBLASLt is NOT broken on this image; its
# default search path just misses the gfx1250 payload. Point it at the complete
# 328-file library copied under $HOME and prefer it. See the job spec for the
# full three-directory autopsy.
os.environ.setdefault(
    "HIPBLASLT_TENSILE_LIBPATH", "/home/lihuzhan/.local/hipblaslt-gfx1250/gfx1250"
)
# setdefault would be a NO-OP here: the container ships
# /usr/lib/python3.12/sitecustomize.py, whose line 18 pins this to "0" at every
# interpreter start, before any of our code runs. That -- not anything about the
# hardware -- is why 12 rounds of this campaign silently ran on rocBLAS. Assign.
if os.environ.get("OPEVOLVE_KEEP_BLAS_ENV") != "1":
    os.environ["TORCH_BLAS_PREFER_HIPBLASLT"] = "1"

for _p in ("/home/lihuzhan/.local/flydsl032", "/tmp/flydsl032"):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
        break
if "/home/lihuzhan/code/aiter-src" not in sys.path:
    sys.path.insert(0, "/home/lihuzhan/code/aiter-src")


def assert_environment():
    """Assert the two things that silently produce wrong answers if they are wrong."""
    import flydsl
    import torch
    assert flydsl.__version__.startswith("0.3.2"), (
        f"flydsl 0.3.2 required, got {flydsl.__version__} at {flydsl.__file__}")
    assert "flydsl032" in flydsl.__file__, (
        f"flydsl resolved to the image copy, not the 0.3.2 install: {flydsl.__file__}")
    arch = torch.cuda.get_device_properties(0).gcnArchName
    assert "gfx1250" in arch, f"these kernels target gfx1250, got {arch}"
    return flydsl.__version__, flydsl.__file__, arch
