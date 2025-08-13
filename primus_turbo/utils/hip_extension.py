###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import os
import shutil
from pathlib import Path

import setuptools
from hipify_torch.hipify_python import hipify


def _find_hip_home() -> str:
    hip_home = os.environ.get("HIP_HOME") or os.environ.get("HIP_PATH")
    if hip_home is None:
        hipcc_path = shutil.which("hipcc")
        if hipcc_path is not None:
            hip_home = os.path.dirname(os.path.dirname(os.path.realpath(hipcc_path)))
            if os.path.basename(hip_home) == "hip":
                hip_home = os.path.dirname(hip_home)
        else:
            fallback_path = "/opt/rocm"
            if os.path.exists(fallback_path):
                hip_home = fallback_path

    if not os.path.exists(hip_home):
        raise ValueError(f"No HIP runtime is found, using HIP_HOME='{hip_home}'")
    return hip_home


HIP_HOME: Path = _find_hip_home()
HIP_LIBRARY_PATH = os.path.join(HIP_HOME, "lib")
HIP_INCLUDE_PATH = os.path.join(HIP_HOME, "include")


def HIPExtension(name, sources, *args, **kwargs):
    library_dirs = kwargs.get("library_dirs", [])
    library_dirs += [HIP_LIBRARY_PATH]
    kwargs["library_dirs"] = library_dirs

    libraries = kwargs.get("libraries", [])
    libraries.append("amdhip64")
    kwargs["libraries"] = libraries

    include_dirs = kwargs.get("include_dirs", [])

    build_dir = os.getcwd()
    hipify_result = hipify(
        project_directory=build_dir,
        output_directory=build_dir,
        header_include_dirs=include_dirs,
        includes=[os.path.join(build_dir, "*")],
        extra_files=[os.path.abspath(s) for s in sources],
        show_detailed=True,
        is_pytorch_extension=True,
        hipify_extra_files_only=True,
    )

    hipified_sources = set()
    for source in sources:
        s_abs = os.path.abspath(source)
        hipified_s_abs = (
            hipify_result[s_abs].hipified_path
            if (s_abs in hipify_result and hipify_result[s_abs].hipified_path is not None)
            else s_abs
        )
        hipified_sources.add(os.path.relpath(hipified_s_abs, build_dir))

    sources = list(hipified_sources)

    include_dirs += [HIP_INCLUDE_PATH]
    kwargs["include_dirs"] = include_dirs

    kwargs["language"] = "c++"

    dlink_libraries = kwargs.get("dlink_libraries", [])
    dlink = kwargs.get("dlink", False) or dlink_libraries
    if dlink:
        extra_compile_args = kwargs.get("extra_compile_args", {})

        extra_compile_args_dlink = extra_compile_args.get("nvcc_dlink", [])
        extra_compile_args_dlink += ["-dlink"]
        extra_compile_args_dlink += [f"-L{x}" for x in library_dirs]
        extra_compile_args_dlink += [f"-l{x}" for x in dlink_libraries]
        extra_compile_args_dlink += ["-dlto"]

        extra_compile_args["nvcc_dlink"] = extra_compile_args_dlink

        kwargs["extra_compile_args"] = extra_compile_args

    return setuptools.Extension(name, sources, *args, **kwargs)
