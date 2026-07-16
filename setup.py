import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from setuptools import find_packages, setup

PROJECT_ROOT = Path(os.path.dirname(__file__)).resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.build_ext import TurboBuildExt, _join_rocm_home
from tools.build_utils import (
    HIPExtension,
    build_ck_backend,
    build_turbo_backend,
    enabled_backends,
    find_rocshmem_library,
    get_gpu_arch,
    is_package_installed,
)

# -------- Framework Switches ---------
# PRIMUS_TURBO_FRAMEWORK="PYTORCH;JAX"
PRIMUS_TURBO_FRAMEWORK = [
    fw.strip().upper() for fw in os.environ.get("PRIMUS_TURBO_FRAMEWORK", "PYTORCH").split(";")
]
BUILD_TORCH = "PYTORCH" in PRIMUS_TURBO_FRAMEWORK
BUILD_JAX = "JAX" in PRIMUS_TURBO_FRAMEWORK

# -------- Supported GPU ARCHS --------
SUPPORTED_GPU_ARCHS = ["gfx942", "gfx950", "gfx1250"]

# -------- Optional GEMM backends --------
# The CK (ck_tile) and turbo (MFMA) GEMM kernels are optional backends selected via the
# PRIMUS_TURBO_BUILD_BACKEND env var (comma-separated, e.g. "ck,turbo"). When unset, all
# known backends are built. Each enabled backend defines a matching guard macro for the
# host bindings (-DBUILD_CK_BACKEND / -DBUILD_TURBO_BACKEND) and
# includes its kernel sources; disabled backends are excluded and fall back to stubs.

# csrc/kernels path fragments that depend on CK, skipped when the CK backend is disabled.
CK_DISABLED_SOURCE_MARKERS = (
    "/gemm/ck/",
    "/gemm/ck_gemm.cu",
    "/grouped_gemm/ck/",
    "/grouped_gemm/ck_grouped_gemm.cu",
)

# csrc/kernels path fragments for the turbo (MFMA) GEMM backend, skipped when it is disabled.
TURBO_DISABLED_SOURCE_MARKERS = (
    "/gemm/turbo/",
    "/gemm/turbo_gemm.cu",
)

# -------- ROCSHMEM LIB ---------------
# try to found rocshmem in default path or enviorment
ROCSHMEM_LIBRARY = find_rocshmem_library()


def get_submodule_folders():
    """Parse .gitmodules and return list of submodule paths."""
    gitmodules_path = os.path.join(PROJECT_ROOT, ".gitmodules")
    if not os.path.exists(gitmodules_path):
        return []
    with open(gitmodules_path) as f:
        return [
            os.path.join(PROJECT_ROOT, line.split("=")[1].strip())
            for line in f
            if line.strip().startswith("path")
        ]


def check_submodules():
    """Check and initialize git submodules if needed."""
    submodule_folders = get_submodule_folders()
    if not submodule_folders:
        return

    for folder in submodule_folders:
        if not os.path.isdir(folder) or not os.listdir(folder):
            try:
                print("[Primus-Turbo] Initializing git submodules...")
                subprocess.run(
                    ["git", "submodule", "sync", "--recursive"],
                    cwd=PROJECT_ROOT,
                    check=True,
                )
                subprocess.run(
                    ["git", "submodule", "update", "--init", "--recursive"],
                    cwd=PROJECT_ROOT,
                    check=True,
                )
            except Exception as e:
                print(f"[Primus-Turbo] Failed to initialize submodules: {e}")
                print(
                    "[Primus-Turbo] Please run: git submodule sync && git submodule update --init --recursive"
                )
                sys.exit(1)
            return
    print("[Primus-Turbo] Submodules already initialized.")


def get_extras_require():
    extras = {
        "pytorch": [
            "torch",
        ],
        "jax": [
            "jax",
            "jaxlib",
            "pybind11>=3.0.1",
        ],
    }
    return extras


def all_files_in_dir(path, name_extensions=None):
    all_files = []
    for dirname, _, names in os.walk(path):
        for name in names:
            suffix = Path(name).suffix.lstrip(".")
            if name_extensions and suffix not in name_extensions:
                continue
            all_files.append(Path(dirname, name))
    return filter_files_by_arch(all_files)


def filter_files_by_arch(files):
    offload_arch_list, _ = get_offload_archs()
    enabled_gfx942 = "--offload-arch=gfx942" in offload_arch_list
    enabled_gfx950 = "--offload-arch=gfx950" in offload_arch_list
    enable_gfx1250 = "--offload-arch=gfx1250" in offload_arch_list
    ck_disabled = not build_ck_backend()
    turbo_disabled = not build_turbo_backend()

    filtered_files = []
    for file in files:
        # Normalize to forward slashes so markers match regardless of platform.
        file_str = str(file).replace(os.sep, "/")
        if ck_disabled and any(marker in file_str for marker in CK_DISABLED_SOURCE_MARKERS):
            continue
        if turbo_disabled and any(marker in file_str for marker in TURBO_DISABLED_SOURCE_MARKERS):
            continue
        if file_str.endswith("_gfx942.cu") or file_str.endswith("_gfx942.hip"):
            if enabled_gfx942:
                filtered_files.append(file)
        elif file_str.endswith("_gfx950.cu") or file_str.endswith("_gfx950.hip"):
            if enabled_gfx950:
                filtered_files.append(file)
        elif file_str.endswith("_gfx1250.cu") or file_str.endswith("_gfx1250.hip"):
            if enable_gfx1250:
                filtered_files.append(file)
        else:
            filtered_files.append(file)
    return filtered_files


def setup_cxx_env():
    DEFAULT_HIPCC = _join_rocm_home("bin", "hipcc")
    user_cxx = os.environ.get("CXX")
    if user_cxx:
        print(f"[Primus-Turbo Setup] Using user-provided CXX: {user_cxx}")
    else:
        os.environ["CXX"] = DEFAULT_HIPCC
        print(f"[Primus-Turbo Setup] No CXX provided. Defaulting to: {DEFAULT_HIPCC}")

    os.environ.setdefault("CMAKE_CXX_COMPILER", os.environ["CXX"])
    os.environ.setdefault("CMAKE_HIP_COMPILER", os.environ["CXX"])
    print(f"[Primus-Turbo Setup] CMAKE_CXX_COMPILER set to: {os.environ['CMAKE_CXX_COMPILER']}")
    print(f"[Primus-Turbo Setup] CMAKE_HIP_COMPILER set to: {os.environ['CMAKE_HIP_COMPILER']}")


def get_commit():
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=PROJECT_ROOT,
            )
            .decode("ascii")
            .strip()
        )
    except Exception:
        return "unknown"


def write_build_info():
    """Codegen primus_turbo/_build_info.py: git/build metadata + enabled backends."""
    build_info_path = PROJECT_ROOT / "primus_turbo" / "_build_info.py"
    old_content = build_info_path.read_text(encoding="utf-8") if build_info_path.exists() else None

    commit = get_commit()
    # Preserve a previously recorded commit when git is unavailable this run.
    if commit == "unknown" and old_content is not None:
        match = re.search(r'__git_commit__ = "([^"]*)"', old_content)
        if match:
            commit = match.group(1)
    build_time = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    backends = enabled_backends()
    content = (
        "# Auto-generated by setup.py during build/install.\n"
        f'__git_commit__ = "{commit}"\n'
        f'__build_time__ = "{build_time}"\n'
        f"BUILD_CK_BACKEND = {'ck' in backends}\n"
        f"BUILD_TURBO_BACKEND = {'turbo' in backends}\n"
    )
    if old_content != content:
        build_info_path.write_text(content, encoding="utf-8")


def get_offload_archs():
    gpu_archs = os.environ.get("GPU_ARCHS", None)

    arch_list = []
    if gpu_archs is None or gpu_archs.strip() == "":
        arch_list = [get_gpu_arch()]
    else:
        for arch in gpu_archs.split(";"):
            arch = arch.strip().lower()
            if arch == "native":
                arch = get_gpu_arch()
            if arch not in arch_list:
                arch_list.append(arch)

    # TODO(ruibin): add JAX support for gfx1250
    if BUILD_JAX and "gfx1250" in arch_list:
        raise ValueError("The JAX backend is not supported on gfx1250.")

    # TODO(ruibin): add CK and turbo support for gfx1250
    if "gfx1250" in arch_list:
        if build_ck_backend() or build_turbo_backend():
            raise ValueError(
                "The CK or turbo backend is not supported on gfx1250. Please set PRIMUS_TURBO_BUILD_BACKEND='' to disable build the CK and turbo backends."
            )

    macro_arch_list = []
    offload_arch_list = []
    for arch in arch_list:
        if arch in SUPPORTED_GPU_ARCHS:
            offload_arch_list.append(f"--offload-arch={arch.lower()}")
            macro_arch_list.append(f"-DPRIMUS_TURBO_{arch.upper()}")
        else:
            print(f"[WARNING] Ignoring unsupported GPU_ARCHS entry: {arch}")
    assert len(offload_arch_list) >= 1, "Primus Turbo: expected at least one --offload-arch."
    return offload_arch_list, macro_arch_list


def check_hip_compiler_flag(flag):
    """Check if the HIP compiler supports a given flag."""
    import tempfile

    hipcc = _join_rocm_home("bin", "hipcc")
    with tempfile.NamedTemporaryFile(suffix=".cpp", delete=True) as f:
        f.write(b"int main() { return 0; }\n")
        f.flush()
        try:
            result = subprocess.run(
                [hipcc, flag, "-c", f.name, "-o", "/dev/null"],
                capture_output=True,
                text=True,
            )
            return result.returncode == 0
        except Exception:
            return False


def get_common_flags():
    from tools.build_utils import HIP_LIBRARY_PATH

    arch = platform.machine().lower()
    extra_link_args = [
        f"-Wl,-rpath,{HIP_LIBRARY_PATH}",
        f"-L/usr/lib/{arch}-linux-gnu",
        "-fgpu-rdc",
        "--hip-link",
        "-Xoffload-linker",
        "--allow-multiple-definition",
    ]

    cxx_flags = [
        "-O3",
        "-fvisibility=hidden",
        "-std=c++20",
        "-fgpu-rdc",
        "-Wno-unknown-warning-option",
    ]

    nvcc_flags = [
        "-O3",
        "-DHIP_ENABLE_WARP_SYNC_BUILTINS=1",
        "-U__HIP_NO_HALF_OPERATORS__",
        "-U__HIP_NO_HALF_CONVERSIONS__",
        "-U__HIP_NO_BFLOAT16_OPERATORS__",
        "-U__HIP_NO_BFLOAT16_CONVERSIONS__",
        "-U__HIP_NO_BFLOAT162_OPERATORS__",
        "-U__HIP_NO_BFLOAT162_CONVERSIONS__",
        "-fno-offload-uniform-block",
        "-mllvm",
        "--lsr-drop-solution=1",
        "-mllvm",
        "-enable-post-misched=0",
        "-mllvm",
        "-amdgpu-early-inline-all=true",
        # "-mllvm",
        # "-amdgpu-function-calls=false",
        "-std=c++20",
        "-fgpu-rdc",
        "-Wno-unknown-warning-option",
    ]

    # Check and add optional compiler flags (for ROCm version compatibility)
    if check_hip_compiler_flag("-mllvm -amdgpu-coerce-illegal-types=1"):
        nvcc_flags.extend(["-mllvm", "-amdgpu-coerce-illegal-types=1"])

    # Device Archs
    offload_arch_list, macro_arch_list = get_offload_archs()
    cxx_flags += macro_arch_list
    cxx_flags += offload_arch_list  # hipcc needs --offload-arch for CK headers in .cpp files
    nvcc_flags += macro_arch_list
    nvcc_flags += offload_arch_list

    # Composable-Kernel / MFMA backend: define the guard macro only when the backend is built.
    if build_ck_backend():
        cxx_flags.append("-DBUILD_CK_BACKEND")
        nvcc_flags.append("-DBUILD_CK_BACKEND")

    # Turbo (MFMA) GEMM backend: define the guard macro only when the backend is built.
    if build_turbo_backend():
        cxx_flags.append("-DBUILD_TURBO_BACKEND")
        nvcc_flags.append("-DBUILD_TURBO_BACKEND")

    # Max Jobs
    max_jobs = int(os.getenv("MAX_JOBS", "64"))
    nvcc_flags.append(f"-parallel-jobs={max_jobs}")

    if "--offload-arch=gfx950" in nvcc_flags:
        cxx_flags.append("-DCK_TILE_USE_OCP_FP8")
        nvcc_flags.append("-DCK_TILE_USE_OCP_FP8")

    return {
        "extra_link_args": extra_link_args,
        "extra_compile_args": {
            "cxx": cxx_flags,
            "nvcc": nvcc_flags,
        },
    }


def build_kernels_extension():
    extra_flags = get_common_flags()
    extra_flags["extra_link_args"] += [
        "-shared",
        "-Wl,-soname,libprimus_turbo_kernels.so",
        # [experiment A] Force a single device-LTO partition so the rocSHMEM
        # device runtime (default context) and the ODC GDA device kernels stay
        # in ONE device-LTO unit. The default multi-partition device link
        # (--lto-partitions=8 + -amdgpu-internalize-symbols) splits the rocSHMEM
        # device default context away from the ODC kernels -> device getmem reads
        # zero -> dual-node grad_norm=0. Keeping them in one partition mirrors the
        # single-TU pinfix .so that works.
        "-Xoffload-linker",
        "--lto-partitions=1",
    ]

    kernels_source_files = Path(PROJECT_ROOT / "csrc" / "kernels")
    kernels_sources = all_files_in_dir(kernels_source_files, name_extensions=["cpp", "cc", "cu"])

    include_dirs = [
        Path(PROJECT_ROOT / "csrc"),
        Path(PROJECT_ROOT / "csrc" / "include"),
        Path(PROJECT_ROOT / "3rdparty" / "composable_kernel" / "include"),
    ]
    library_dirs = []

    if ROCSHMEM_LIBRARY is None:
        extra_flags["extra_compile_args"]["nvcc"].append("-DDISABLE_ROCSHMEM")
        extra_flags["extra_compile_args"]["cxx"].append("-DDISABLE_ROCSHMEM")
    else:
        include_dirs.extend(ROCSHMEM_LIBRARY.include_dirs)
        library_dirs.extend(ROCSHMEM_LIBRARY.library_dirs)
        extra_flags["extra_link_args"].extend(ROCSHMEM_LIBRARY.extra_link_args)

        if (
            "-fgpu-rdc" in ROCSHMEM_LIBRARY.extra_link_args
            or "--hip-link" in ROCSHMEM_LIBRARY.extra_link_args
        ):
            extra_flags["extra_compile_args"]["nvcc"] += ["-fgpu-rdc"]

    return HIPExtension(
        name="libprimus_turbo_kernels",
        include_dirs=include_dirs,
        sources=kernels_sources,
        library_dirs=library_dirs,
        libraries=["hipblaslt", "hipblas"],
        **extra_flags,
    )


def build_torch_extension():
    if not BUILD_TORCH:
        return None

    from torch.utils.cpp_extension import CUDAExtension

    # Link and Compile flags
    extra_flags = get_common_flags()
    extra_flags["extra_link_args"] = [
        "-Wl,-rpath,$ORIGIN/../lib",
        f"-L{PROJECT_ROOT / 'primus_turbo' / 'lib'}",
        "-lprimus_turbo_kernels",
        *extra_flags.get("extra_link_args", []),
    ]

    if ROCSHMEM_LIBRARY is None:
        extra_flags["extra_compile_args"]["nvcc"].append("-DDISABLE_ROCSHMEM")
        extra_flags["extra_compile_args"]["cxx"].append("-DDISABLE_ROCSHMEM")

    # CPP
    pytorch_csrc_source_files = Path(PROJECT_ROOT / "csrc" / "pytorch")
    sources = all_files_in_dir(pytorch_csrc_source_files, name_extensions=["cpp", "cc", "cu"])

    return CUDAExtension(
        name="primus_turbo.pytorch._C",
        sources=sources,
        include_dirs=[
            Path(PROJECT_ROOT / "csrc"),
            Path(PROJECT_ROOT / "csrc" / "include"),
            Path(PROJECT_ROOT / "3rdparty" / "composable_kernel" / "include"),
        ],
        **extra_flags,
    )


def build_jax_extension():
    if not BUILD_JAX:
        return None

    import pybind11
    from jax import ffi

    # Link and Compile flags
    extra_flags = get_common_flags()
    extra_flags["extra_link_args"] = [
        "-Wl,-rpath,$ORIGIN/../lib",
        f"-L{PROJECT_ROOT / 'primus_turbo' / 'lib'}",
        "-lprimus_turbo_kernels",
        *extra_flags.get("extra_link_args", []),
    ]

    if ROCSHMEM_LIBRARY is None:
        extra_flags["extra_compile_args"]["nvcc"].append("-DDISABLE_ROCSHMEM")
        extra_flags["extra_compile_args"]["cxx"].append("-DDISABLE_ROCSHMEM")

    # CPP
    jax_csrc_source_files = Path(PROJECT_ROOT / "csrc" / "jax")
    sources = all_files_in_dir(jax_csrc_source_files, name_extensions=["cpp", "cc", "cu"])

    return HIPExtension(
        name="primus_turbo.jax._C",
        sources=sources,
        include_dirs=[
            Path(PROJECT_ROOT / "csrc"),
            Path(PROJECT_ROOT / "csrc" / "include"),
            Path(PROJECT_ROOT / "3rdparty" / "composable_kernel" / "include"),
            ffi.include_dir(),
            pybind11.get_include(),
        ],
        **extra_flags,
    )


if __name__ == "__main__":
    # Initialize submodules
    check_submodules()

    # set cxx
    setup_cxx_env()

    # Build metadata
    write_build_info()

    # Extensions
    kernels_ext = build_kernels_extension()

    torch_ext = build_torch_extension()
    jax_ext = build_jax_extension()
    ext_modules = [kernels_ext] + [e for e in (torch_ext, jax_ext) if e is not None]

    # Entry points and Install Requires
    entry_points = {}
    install_requires = [
        "scipy",
    ]

    # TODO(ruibin): Triton 3.7.0 and flydsl 0.2.0 does not support gfx1250, so we skip their installation when building for gfx1250.
    offload_arch_list, _ = get_offload_archs()
    if "--offload-arch=gfx1250" not in offload_arch_list:
        install_requires.append("triton==3.7.0")
        install_requires.append("flydsl>=0.2.0")
    else:
        print("[Primus-Turbo Setup] Building for gfx1250; skipping triton dependency.")

    if torch_ext is not None:
        if is_package_installed("origami"):
            print("[Primus-Turbo Setup] Found optional origami installation.")
        else:
            print("[Primus-Turbo Setup] origami not found; Triton selectors will use heuristic fallback.")

    if BUILD_JAX:
        entry_points["jax_plugins"] = ["primus_turbo = primus_turbo.jax"]

    setup(
        name="primus-turbo",
        use_scm_version=True,
        description="High-performance acceleration library for large-scale model training on AMD GPUs",
        long_description=(PROJECT_ROOT / "README.md").read_text(encoding="utf-8"),
        long_description_content_type="text/markdown",
        packages=find_packages(exclude=["tests", "tests.*"]),
        package_data={"primus_turbo": ["lib/*.so"]},
        ext_modules=ext_modules,
        cmdclass={"build_ext": TurboBuildExt.with_options(use_ninja=True)},
        entry_points=entry_points,
        install_requires=install_requires,
        extras_require=get_extras_require(),
    )
