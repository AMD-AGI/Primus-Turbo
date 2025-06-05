import os
import platform
import re

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def read_version():
    with open(os.path.join("primus_turbo", "__init__.py")) as f:
        for line in f:
            match = re.match(r"^__version__\s*=\s*[\"'](.+?)[\"']", line)
            if match:
                return match.group(1)
    raise RuntimeError("Cannot find version.")


def build_torch_extension():
    arch = platform.machine().lower()
    libraries = ["hipblas"]
    extra_link_args = [
        "-Wl,-rpath,$ORIGIN/../../torch/lib",
        "-Wl,-rpath,/opt/rocm/lib",
        f"-L/usr/lib/{arch}-linux-gnu",
    ]
    return CUDAExtension(
        name="primus_turbo.pytorch._C",
        sources=[
            "csrc/pytorch/bindings_pytorch.cpp",
            "csrc/pytorch/gemm/gemm.cpp",
            "csrc/pytorch/gemm/gemm_meta.cpp",
        ],
        include_dirs=["csrc/include"],
        libraries=libraries,
        extra_link_args=extra_link_args,
        extra_compile_args={
            "cxx": ["-O3"],
            "nvcc": ["-O3"],
        },
    )


if __name__ == "__main__":
    setup(
        name="primus_turbo",
        version=read_version(),
        packages=find_packages(),
        ext_modules=[build_torch_extension()],
        cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
    )
