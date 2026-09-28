"""sys.path / env preamble for the beat arm. Import BEFORE torch/aiter.

The BLAS variables are ASSIGNED, not setdefault: /usr/lib/python3.12/sitecustomize.py:18
already did setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "0") before this runs.

sys.path order: flydsl 0.3.4.1 ahead of the image's 0.2.4, and the aiter checkout
(/home/lihuzhan/code/aiter-src -- READ ONLY, never edit it; aiter is not pip-installed in
this image). NEVER import primus_turbo in this process.
"""
import os
import sys

os.environ["TORCH_BLAS_PREFER_HIPBLASLT"] = "1"
os.environ["HIPBLASLT_TENSILE_LIBPATH"] = "/home/lihuzhan/.local/hipblaslt-gfx1250/gfx1250"

AITER_SRC = "/home/lihuzhan/code/aiter-src"
for _p in (AITER_SRC, "/home/lihuzhan/.local/flydsl0341"):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def env_line():
    return ("ENV TORCH_BLAS_PREFER_HIPBLASLT=%s HIPBLASLT_TENSILE_LIBPATH=%s"
            % (os.environ.get("TORCH_BLAS_PREFER_HIPBLASLT"),
               os.environ.get("HIPBLASLT_TENSILE_LIBPATH")))
