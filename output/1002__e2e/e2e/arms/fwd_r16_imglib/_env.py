"""sys.path / env preamble shared by every module in this job.

Import this BEFORE `import flydsl`, from any implementation directory. It is
duplicated per implementation on purpose: `op/baseline/` must stay importable on its
own after the framework copies it into `rounds/<n>/op/`.

Adapted from the backward job's
/home/lihuzhan/code/2026_0910__op-evolve/op-evolve/artifacts/gfx1250-flydsl-attn-bwd-20260917-115934/job_context/op/baseline/_env.py
with one change: the forward kernel is VENDORED under flydsl_fwd/, so
/home/lihuzhan/code/aiter-src no longer has to be on sys.path and is deliberately
NOT added -- an `import aiter` from this tree should fail loudly rather than
silently pick up the installed package's kernels.

Order that still matters:
  /home/lihuzhan/.local/flydsl0341  flydsl 0.3.4.1, ahead of the image's 0.2.4
NEVER import primus_turbo in this process: its FlyDSL tree imports
`flydsl.expr.buffer_ops`, which 0.3.2 removed.
"""
import os
import sys

# E2E COPY (output/1002__e2e/e2e/arms/fwd_r16_imglib, 2026-10-02): the two BLAS assignments of the
# champion's _env.py are REMOVED here and nowhere else. The champion assigned
#     TORCH_BLAS_PREFER_HIPBLASLT = "1"
#     HIPBLASLT_TENSILE_LIBPATH   = "/home/lihuzhan/.local/hipblaslt-gfx1250/gfx1250"   (host library)
# at import, i.e. in the middle of a training process (first FlyDSL call). On A0 09-28 both runs that
# loaded this tree in step 1 went bad (a0_p3b NaN in the backward GEMMs, a0_p4b wedge;
# output/0928__a0_repro/REPORT.md section 5.3). In the e2e the launcher assigns the IMAGE library
# before python starts and the e2e adapter restores it if anything re-points it. The forward kernel
# never calls hipBLASLt, so this file is the only difference from the champion and changes no kernel.

for _p in ("/home/lihuzhan/.local/flydsl0341",):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
        break


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


def env_line():
    """The BLAS env as seen by THIS process -- print it from the measuring process."""
    return ("ENV TORCH_BLAS_PREFER_HIPBLASLT=%s HIPBLASLT_TENSILE_LIBPATH=%s"
            % (os.environ.get("TORCH_BLAS_PREFER_HIPBLASLT"),
               os.environ.get("HIPBLASLT_TENSILE_LIBPATH")))
