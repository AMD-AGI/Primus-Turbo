import os
import platform
import re
import shutil
import subprocess
from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

from primus_turbo.utils.hip_extension import HIPExtension

PROJECT_ROOT = Path(os.path.dirname(__file__)).resolve()
DEFAULT_HIPCC = "/opt/rocm/bin/hipcc"

# -------- env switches --------
BUILD_TORCH = os.environ.get("PRIMUS_TURBO_BUILD_TORCH", "1") == "1"
BUILD_JAX = os.environ.get("PRIMUS_TURBO_BUILD_JAX", "0") == "1"


class TurboBuildExt(BuildExtension):
    KERNEL_EXT_NAME = "libprimus_turbo_kernels"

    def get_ext_filename(self, ext_name: str) -> str:
        filename = super().get_ext_filename(ext_name)
        if ext_name == self.KERNEL_EXT_NAME:
            filename = os.path.join(*filename.split(os.sep)[:-1], "libprimus_turbo_kernels.so")
        return filename

    def build_extension(self, ext):
        super().build_extension(ext)

        if ext.name == self.KERNEL_EXT_NAME:
            built_path = Path(self.get_ext_fullpath(ext.name))
            filename = built_path.name
            #
            src_dst_dir = PROJECT_ROOT / "primus_turbo" / "lib"
            src_dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(built_path, src_dst_dir / filename)
            #
            build_dst_dir = Path(self.build_lib) / "primus_turbo" / "lib"
            build_dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(built_path, build_dst_dir / filename)
            print(f"[TurboBuildExt] Copied {filename} to:")
            print(f"  - {src_dst_dir}")
            print(f"  - {build_dst_dir}")


def all_files_in_dir(path, name_extensions=None):
    all_files = []
    for dirname, _, names in os.walk(path):
        for name in names:
            suffix = Path(name).suffix.lstrip(".")
            if name_extensions and suffix not in name_extensions:
                continue
            all_files.append(Path(dirname, name))
    return all_files


def setup_cxx_env():
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


def get_version():
    base_version = None
    with open(os.path.join("primus_turbo", "__init__.py")) as f:
        for line in f:
            match = re.match(r"^__version__\s*=\s*[\"'](.+?)[\"']", line)
            if match:
                base_version = match.group(1)
                break
    if base_version is None:
        raise RuntimeError("Cannot find version.")

    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("ascii").strip()
        return f"{base_version}+{commit}"  # PEP440
    except Exception:
        return base_version


def get_common_flags():
    arch = platform.machine().lower()
    extra_link_args = [
        "-Wl,-rpath,/opt/rocm/lib",
        f"-L/usr/lib/{arch}-linux-gnu",
    ]

    cxx_flags = [
        "-O3",
        "-fvisibility=hidden",
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
        "-amdgpu-coerce-illegal-types=1",
        "-mllvm",
        "-amdgpu-early-inline-all=true",
        "-mllvm",
        "-amdgpu-function-calls=false",
    ]

    # Device Arch
    # TODO: Add ENV Setting
    # TODO: ROCM Version support
    # nvcc_flags += [
    #     "--offload-arch=gfx942",
    #     "--offload-arch=gfx950",
    # ]

    max_jobs = int(os.getenv("MAX_JOBS", "4"))
    nvcc_flags.append(f"-parallel-jobs={max_jobs}")

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
    ]

    kernels_source_files = Path(PROJECT_ROOT / "csrc" / "kernels")
    kernels_sources = all_files_in_dir(kernels_source_files, name_extensions=["cpp", "cc", "cu"])

    return HIPExtension(
        name="libprimus_turbo_kernels",
        include_dirs=[
            Path(PROJECT_ROOT / "csrc" / "include"),
            Path(PROJECT_ROOT / "3rdparty" / "composable_kernel" / "include"),
            Path(PROJECT_ROOT / "csrc"),
        ],
        sources=kernels_sources,
        libraries=["hipblas"],
        **extra_flags,
    )


def build_torch_extension():
    if not BUILD_TORCH:
        return None

    # Link and Compile flags
    extra_flags = get_common_flags()
    extra_flags["extra_link_args"] = [
        "-Wl,-rpath,$ORIGIN/../lib",
        f"-L{PROJECT_ROOT / 'primus_turbo' / 'lib'}",
        "-lprimus_turbo_kernels",
        *extra_flags.get("extra_link_args", []),
    ]

    # CPP
    pytorch_csrc_source_files = Path(PROJECT_ROOT / "csrc" / "pytorch")
    sources = all_files_in_dir(pytorch_csrc_source_files, name_extensions=["cpp", "cc", "cu"])

    return CUDAExtension(
        name="primus_turbo.pytorch._C",
        sources=sources,
        include_dirs=[
            Path(PROJECT_ROOT / "csrc" / "include"),
            Path(PROJECT_ROOT / "3rdparty" / "composable_kernel" / "include"),
            Path(PROJECT_ROOT / "csrc"),
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

    # CPP
    jax_csrc_source_files = Path(PROJECT_ROOT / "csrc" / "jax")
    sources = all_files_in_dir(jax_csrc_source_files, name_extensions=["cpp", "cc", "cu"])

    return HIPExtension(
        name="primus_turbo.jax._C",
        sources=sources,
        include_dirs=[
            Path(PROJECT_ROOT / "csrc" / "include"),
            Path(PROJECT_ROOT / "3rdparty" / "composable_kernel" / "include"),
            Path(PROJECT_ROOT / "csrc"),
            ffi.include_dir(),
            pybind11.get_include(),
        ],
        **extra_flags,
    )


if __name__ == "__main__":
    # set cxx
    setup_cxx_env()

    # Extensions
    kernels_ext = build_kernels_extension()
    torch_ext = build_torch_extension()
    jax_ext = build_jax_extension()
    ext_modules = [kernels_ext] + [e for e in (torch_ext, jax_ext) if e is not None]

    # Entry points and Install Requires
    entry_points = {}
    install_requires = [
        "aiter @ git+https://github.com/ROCm/aiter.git@4822e6755ae66ba727f0d1d33d348673972cbe9c",
        "hip-python",
    ]
    if BUILD_JAX:
        entry_points["jax_plugins"] = ["primus_turbo = primus_turbo.jax"]
        install_requires.append("jax[rocm]")

    setup(
        name="primus_turbo",
        version=get_version(),
        packages=find_packages(exclude=["tests", "tests.*"]),
        package_data={"primus_turbo": ["lib/*.so"]},
        ext_modules=ext_modules,
        cmdclass={"build_ext": TurboBuildExt.with_options(use_ninja=True)},
        entry_points=entry_points,
        install_requires=install_requires,
    )
