###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

try:
    from ._version import version as __version__
except Exception:
    __version__ = "0.0.0.dev0"

try:
    from ._build_info import __build_time__, __git_commit__
except Exception:
    __git_commit__ = "unknown"
    __build_time__ = "unknown"

try:
    from ._build_info import DEV_NO_ROCSHMEM as _DEV_NO_ROCSHMEM
except Exception:
    _DEV_NO_ROCSHMEM = False
if _DEV_NO_ROCSHMEM:
    # A development build (PRIMUS_TURBO_DEV_NO_ROCSHMEM=1 at build time: rocSHMEM, DeepEP internode and
    # ODC compiled out). Fail closed: only an environment that opts in may load it.
    import os as _os

    if _os.environ.get("PRIMUS_TURBO_ALLOW_DEV_BUILD", "0") != "1":
        raise ImportError(
            "This Primus-Turbo is a development build without rocSHMEM (PRIMUS_TURBO_DEV_NO_ROCSHMEM=1) and "
            "must not be used in production. Rebuild without that variable, or set "
            "PRIMUS_TURBO_ALLOW_DEV_BUILD=1 in a development environment."
        )
    import warnings as _warnings

    _warnings.warn("Primus-Turbo development build: rocSHMEM, DeepEP internode and ODC are compiled out.")
    try:
        from ._build_info import DEV_NO_RDC as _DEV_NO_RDC
    except Exception:
        _DEV_NO_RDC = False
    if _DEV_NO_RDC:
        _warnings.warn(
            "Primus-Turbo PRIMUS_TURBO_DEV_NO_RDC build: kernels compiled without device LTO; not for timing."
        )
