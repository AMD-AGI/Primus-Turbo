###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import os
import re
import shutil
import subprocess
import sys
import sysconfig
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import setuptools

# Use the vendored hipify_torch submodule; no separate install needed.
_HIPIFY_VENDOR_DIR = Path(__file__).resolve().parent.parent / "3rdparty" / "hipify_torch"
if _HIPIFY_VENDOR_DIR.is_dir() and str(_HIPIFY_VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(_HIPIFY_VENDOR_DIR))

try:
    from hipify_torch.hipify_python import hipify
except ImportError as exc:
    raise RuntimeError(
        "hipify_torch is required to build primus_turbo but was not found.\n"
        "It is vendored as a git submodule; initialize it with:\n"
        "  git submodule sync && git submodule update --init --recursive"
    ) from exc

try:
    import torch  # noqa: F401

    from .patch import patch_torch_extension
except ImportError:
    warnings.warn("PyTorch not found, skip patch_torch_extension.", ImportWarning)
else:
    # TODO: pytorch version check
    patch_torch_extension()


TURBO_FALLBACK_LIBRARY_HOME = "/opt/rocm"


@dataclass
class Library:
    name: str
    include_dirs: List[str]
    libraries: List[str]
    library_dirs: List[str]
    extra_link_args: List[str]


def _guess_library_home(exec_names: Optional[List[str]] = None) -> Optional[str]:
    if exec_names is None:
        exec_names = []

    all_exec_paths = []
    for execute_name in exec_names:
        exec_home = shutil.which(execute_name)
        if exec_home is not None:
            exec_home = os.path.dirname(os.path.dirname(os.path.realpath(exec_home)))
            all_exec_paths.append(exec_home)

    if len(all_exec_paths) > 1:
        return all_exec_paths[0]


def _find_library_home(
    env_names: List[str],
    exec_names: Optional[List[str]] = None,
    fallback_path: Optional[str] = None,
) -> Optional[str]:
    lib_home = None
    for env in env_names:
        lib_home = lib_home or os.getenv(env)
        if lib_home:
            break

    # 1. specified by user, warning if not exist
    if lib_home:
        if not os.path.exists(lib_home):
            lib_home = None
            warnings.warn(f"Not found library dir at {lib_home}")
        return lib_home

    # 2. lib_home must be None, so guess from executable
    lib_home = _guess_library_home(exec_names)

    # 3. use fallback path when guess failed
    if lib_home is None:
        if fallback_path and os.path.exists(fallback_path):
            lib_home = fallback_path

    return lib_home


def find_rocm_home() -> Optional[str]:
    """Find the ROCm install path."""
    hipcc_path = shutil.which("hipcc")
    if hipcc_path is not None:
        rocm_home = os.path.dirname(os.path.dirname(os.path.realpath(hipcc_path)))
        if os.path.basename(rocm_home) == "hip":
            rocm_home = os.path.dirname(rocm_home)
    else:
        fallback_path = TURBO_FALLBACK_LIBRARY_HOME
        if os.path.exists(fallback_path):
            rocm_home = fallback_path
    return rocm_home


def find_hip_home() -> str:
    return _find_library_home(["HIP_HOME", "HIP_PATH", "HIP_DIR"], ["hipcc"], fallback_path="/opt/rocm")


HIP_HOME: Path = find_hip_home()
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


def find_mpi_home():
    return _find_library_home(
        ["MPI_HOME", "MPI_PATH", "MPI_DIR"],
        ["mpirun", "mpicc", "ompi_info"],
        fallback_path="/opt/rocm/ompi",
    )


def find_rocshmem_home():
    return _find_library_home(
        ["ROCSHMEM_HOME", "ROCSHMEM_PATH", "ROCSHMEM_DIR"], fallback_path="/opt/rocm/rocshmem"
    )


def _default_library_search_dirs() -> List[str]:
    library_dirs = []
    for env_name in ("LD_LIBRARY_PATH", "LIBRARY_PATH"):
        library_dirs.extend(path for path in os.getenv(env_name, "").split(os.pathsep) if path)

    multiarch = sysconfig.get_config_var("MULTIARCH")
    if multiarch:
        library_dirs.extend([f"/usr/lib/{multiarch}", f"/lib/{multiarch}"])

    library_dirs.extend(["/usr/local/lib", "/usr/lib", "/lib"])
    return library_dirs


def _find_linkable_library(name: str, library_dirs: List[str]) -> Optional[Path]:
    candidates = []
    for library_dir in [*library_dirs, *_default_library_search_dirs()]:
        if not library_dir:
            continue
        lib_dir = Path(library_dir)
        candidates.extend(
            [
                lib_dir / f"lib{name}.so",
                lib_dir / f"lib{name}.a",
            ]
        )
        candidates.extend(sorted(lib_dir.glob(f"lib{name}.so.*")))

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def find_rocshmem_library() -> Library:
    rocshmem_home = find_rocshmem_home()
    mpi_home = find_mpi_home()
    if rocshmem_home is None or mpi_home is None:
        warnings.warn(
            "rocSHMEM or MPI library is not found, internode of DeepEP will be disabled. Please set 'MPI_HOME' and 'ROCSHMEM_HOME' env variables and reinstall."
        )
        return None

    rocshmem_home = Path(rocshmem_home).resolve()
    mpi_home = Path(mpi_home).resolve()

    rocshmem_library_dir = str(rocshmem_home / "lib")
    rocshmem_include_dir = str(rocshmem_home / "include")
    mpi_library_dir = str(mpi_home / "lib")
    mpi_include_dir = str(mpi_home / "include")
    library_dirs = [rocshmem_library_dir, mpi_library_dir]
    extra_link_args = [
        f"-Wl,-rpath,{rocshmem_library_dir}",
        "-l:librocshmem.a",
        "-fgpu-rdc",
        "--hip-link",
        "-lamdhip64",
        "-lhsa-runtime64",
        "-l:libmpi.so",
        f"-Wl,-rpath,{mpi_library_dir}",
        "-libverbs",
        "-lmlx5",
    ]
    ionic_library = _find_linkable_library("ionic", library_dirs)
    if ionic_library is not None:
        extra_link_args.extend([str(ionic_library), f"-Wl,-rpath,{ionic_library.parent}"])

    rocshmem_library = Library(
        name="rocshmem",
        libraries=[],
        library_dirs=library_dirs,
        include_dirs=[rocshmem_include_dir, mpi_include_dir],
        extra_link_args=extra_link_args,
    )
    return rocshmem_library


def get_gpu_arch(gpu_id: int = 0) -> str:
    GPU_ARCH_PATTERN = re.compile(r"Name:\s*(gfx\d+\w*)")
    try:
        result = subprocess.run(
            ["rocminfo"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
    except FileNotFoundError:
        raise RuntimeError("rocminfo not found. Please ensure ROCm is installed.")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"rocminfo failed: {e.stderr}")

    matches = GPU_ARCH_PATTERN.findall(result.stdout)

    if not matches:
        raise RuntimeError("No gfx architecture found in rocminfo output.")
    if gpu_id >= len(matches):
        raise IndexError(f"gpu_id {gpu_id} out of range (found {len(matches)} GPUs).")

    return matches[gpu_id].lower()
