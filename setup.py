import os
import platform
import re
import shutil
import subprocess
from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

from tools.build_utils import HIPExtension, find_rocshmem_library

PROJECT_ROOT = Path(os.path.dirname(__file__)).resolve()
DEFAULT_HIPCC = "/opt/rocm/bin/hipcc"

# try to found rocshmem in default path or enviorment
ROCSHMEM_LIBRARY = find_rocshmem_library()

# -------- env switches --------
BUILD_TORCH = os.environ.get("PRIMUS_TURBO_BUILD_TORCH", "1") == "1"
BUILD_JAX = os.environ.get("PRIMUS_TURBO_BUILD_JAX", "0") == "1"

# -------- Supported GPU ARCHS --------
SUPPORTED_GPU_ARCHS = ["gfx942", "gfx950"]


class TurboBuildExt(BuildExtension):
    KERNEL_EXT_NAME = "libprimus_turbo_kernels"

    def _is_hip_src(self, p: str) -> bool:
        p = p.lower()
        return p.endswith(".cu") or p.endswith(".hip")

    def _filter_nvcc_compile_args(self, nvcc_compile_args: list[str], arch: str) -> list[str]:
        offload_arch = f"--offload-arch={arch.lower()}"
        macro_arch = f"-DPRIMUS_TURBO_{arch.upper()}"
        exists = any(a == offload_arch or a == macro_arch for a in nvcc_compile_args)

        new_nvcc_compile_args = []
        for arg in nvcc_compile_args:
            if arg.startswith("--offload-arch=") or arg.startswith("-DPRIMUS_TURBO_"):
                continue
            new_nvcc_compile_args.append(arg)
        new_nvcc_compile_args.append(offload_arch)
        new_nvcc_compile_args.append(macro_arch)
        return new_nvcc_compile_args, exists

    def get_ext_filename(self, ext_name: str) -> str:
        filename = super().get_ext_filename(ext_name)
        if ext_name == self.KERNEL_EXT_NAME:
            filename = os.path.join(*filename.split(os.sep)[:-1], "libprimus_turbo_kernels.so")
        return filename

    def build_extension(self, ext):
        if ext.name != self.KERNEL_EXT_NAME:
            return super().build_extension(ext)

        build_temp = Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)

        # Args
        cxx_compile_args = list(ext.extra_compile_args.get("cxx", []))
        nvcc_compile_args = list(ext.extra_compile_args.get("nvcc", []))
        include_dirs = list(ext.include_dirs or [])
        macros = list(ext.define_macros or [])
        library_dirs = list(ext.library_dirs or [])
        libraries = list(ext.libraries or [])
        extra_link_args = list(ext.extra_link_args or [])

        # print("*** cxx_compile_args", cxx_compile_args)
        # print("*** nvcc_compile_args", nvcc_compile_args)
        # print("*** include_dirs", include_dirs)
        # print("*** macros", macros)
        # print("*** library_dirs", library_dirs)
        # print("*** libraries", libraries)
        # print("*** extra_link_args", extra_link_args)

        cxx_srcs = []
        hip_srcs = []
        hip_srcs_gfx942 = []
        hip_srcs_gfx950 = []
        for source_file in ext.sources:
            if self._is_hip_src(source_file):
                if source_file.endswith("_gfx942.cu") or source_file.endswith("_gfx942.hip"):
                    hip_srcs_gfx942.append(source_file)
                elif source_file.endswith("_gfx950.cu") or source_file.endswith("_gfx950.hip"):
                    hip_srcs_gfx950.append(source_file)
                else:
                    hip_srcs.append(source_file)
            else:
                cxx_srcs.append(source_file)

        objects = []
        # Compile cxx files
        if cxx_srcs:
            cxx_objs = self.compiler.compile(
                sources=cxx_srcs,
                output_dir=str(build_temp),
                include_dirs=include_dirs,
                extra_postargs=cxx_compile_args,
                macros=macros,
                debug=self.debug,
            )
            objects.extend(cxx_objs)

        # Compile hip general files
        if hip_srcs:
            hip_objs = self.compiler.compile(
                sources=hip_srcs,
                output_dir=str(build_temp),
                include_dirs=include_dirs,
                extra_postargs=nvcc_compile_args,
                macros=macros,
                debug=self.debug,
            )
            objects.extend(hip_objs)

        # Compile hip gfx942 files
        nvcc_compile_args_only_gfx942, has_gfx942_arch = self._filter_nvcc_compile_args(
            nvcc_compile_args, "gfx942"
        )
        if hip_srcs_gfx942 and has_gfx942_arch:
            hip_objs_gfx942 = self.compiler.compile(
                sources=hip_srcs_gfx942,
                output_dir=str(build_temp),
                include_dirs=include_dirs,
                extra_postargs=nvcc_compile_args_only_gfx942,
                macros=macros,
                debug=self.debug,
            )
            objects.extend(hip_objs_gfx942)

        # Compile hip gfx950 files
        nvcc_compile_args_only_gfx950, has_gfx950_arch = self._filter_nvcc_compile_args(
            nvcc_compile_args, "gfx950"
        )
        if hip_srcs_gfx950 and has_gfx950_arch:
            hip_objs_gfx950 = self.compiler.compile(
                sources=hip_srcs_gfx950,
                output_dir=str(build_temp),
                include_dirs=include_dirs,
                extra_postargs=nvcc_compile_args_only_gfx950,
                macros=macros,
                debug=self.debug,
            )
            objects.extend(hip_objs_gfx950)

        # Link
        self.compiler.link_shared_object(
            objects=objects,
            output_filename=self.get_ext_fullpath(ext.name),
            library_dirs=library_dirs,
            libraries=libraries,
            extra_postargs=extra_link_args,
            debug=self.debug,
            target_lang="c++",
        )

        # Copy to primus_turbo/lib
        built_path = Path(self.get_ext_fullpath(ext.name))
        filename = built_path.name
        src_dst_dir = PROJECT_ROOT / "primus_turbo" / "lib"
        src_dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(built_path, src_dst_dir / filename)
        build_dst_dir = Path(self.build_lib) / "primus_turbo" / "lib"
        build_dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(built_path, build_dst_dir / filename)
        print(f"[TurboBuildExt] Copied {filename} to:")
        print(f"  -  {src_dst_dir}")
        print(f"  -  {build_dst_dir}")


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


def get_offload_archs():
    def _get_device_arch():
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("No GPU found!")
        return torch.cuda.get_device_properties(0).gcnArchName.split(":")[0].lower()

    gpu_archs = os.environ.get("GPU_ARCHS", None)

    arch_list = []
    if gpu_archs is None or gpu_archs.strip() == "":
        arch_list = [_get_device_arch()]
    else:
        for arch in gpu_archs.split(";"):
            arch = arch.strip().lower()
            if arch == "native":
                arch = _get_device_arch()
            if arch not in arch_list:
                arch_list.append(arch)

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


def get_common_flags():
    arch = platform.machine().lower()
    extra_link_args = [
        "-Wl,-rpath,/opt/rocm/lib",
        f"-L/usr/lib/{arch}-linux-gnu",
    ]

    cxx_flags = [
        "-O3",
        "-fvisibility=hidden",
        "-std=c++20",
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
        "-std=c++20",
    ]

    # Device Archs
    offload_arch_list, macro_arch_list = get_offload_archs()
    cxx_flags += macro_arch_list
    nvcc_flags += macro_arch_list
    nvcc_flags += offload_arch_list

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
        libraries=["hipblaslt"],
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
        "aiter @ git+https://github.com/ROCm/aiter.git@97007320d4b1d7b882d99af02cad02fbb9957559",
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
