import os
import shlex
import shutil
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Optional

from setuptools.command.build_ext import build_ext

from .build_utils import find_rocm_home


def _join_rocm_home(*paths) -> str:
    return os.path.join(find_rocm_home(), *paths)


SUBPROCESS_DECODE_ARGS = ()

COMMON_HIP_FLAGS = [
    "-D__HIP_PLATFORM_AMD__=1",
    "-DUSE_ROCM=1",
    "-DHIPBLAS_V2",
    "-fPIC",
]

COMMON_HIPCC_FLAGS = [
    "-DCUDA_HAS_FP16=1",
    "-D__HIP_NO_HALF_OPERATORS__=1",
    "-D__HIP_NO_HALF_CONVERSIONS__=1",
    "-DHIP_ENABLE_WARP_SYNC_BUILTINS=1",
]


def is_ninja_available():
    """Return ``True`` if the `ninja <https://ninja-build.org/>`_ build system is available on the system, ``False`` otherwise."""
    try:
        subprocess.check_output("ninja --version".split())
    except Exception:
        return False
    else:
        return True


def verify_ninja_availability():
    """Raise ``RuntimeError`` if `ninja <https://ninja-build.org/>`_ build system is not available on the system, does nothing otherwise."""
    if not is_ninja_available():
        raise RuntimeError("Ninja is required to load C++ extensions (pip install ninja to get it)")


def get_cxx_compiler():
    compiler = os.environ.get("CXX", "c++")
    return compiler


class BuildExtension(build_ext):
    @classmethod
    def with_options(cls, **options):
        class cls_with_options(cls):  # type: ignore[misc, valid-type]
            def __init__(self, *args, **kwargs):
                kwargs.update(options)
                super().__init__(*args, **kwargs)

        return cls_with_options

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.no_python_abi_suffix = kwargs.get("no_python_abi_suffix", False)

        self.use_ninja = kwargs.get("use_ninja", True)
        if self.use_ninja:
            # Test if we can use ninja. Fallback otherwise.
            msg = (
                "Attempted to use ninja as the BuildExtension backend but "
                "%s. Falling back to using the slow distutils backend."
            )
            if not is_ninja_available():
                warnings.warn(msg % "we could not find ninja.")
                self.use_ninja = False

    def finalize_options(self) -> None:
        super().finalize_options()
        if self.use_ninja:
            self.force = True

    def build_extensions(self) -> None:

        cuda_ext = False
        extension_iter = iter(self.extensions)
        extension = next(extension_iter, None)
        while not cuda_ext and extension:
            for source in extension.sources:
                _, ext = os.path.splitext(source)
                if ext == ".cu":
                    cuda_ext = True
                    break
            extension = next(extension_iter, None)

        for extension in self.extensions:
            if isinstance(extension.extra_compile_args, dict):
                for ext in ["cxx", "nvcc"]:
                    if ext not in extension.extra_compile_args:
                        extension.extra_compile_args[ext] = []

            # self._add_compile_flag(extension, '-DTORCH_API_INCLUDE_EXTENSION_H')
            self._hipify_compile_flags(extension)
            # self._define_torch_extension_name(extension)

        self.compiler.src_extensions += [".cu", ".cuh", ".hip"]

        def append_std17_if_no_std_present(cflags) -> None:
            # NVCC does not allow multiple -std to be passed, so we avoid
            # overriding the option if the user explicitly passed it.
            cpp_format_prefix = "/{}:" if self.compiler.compiler_type == "msvc" else "-{}="
            cpp_flag_prefix = cpp_format_prefix.format("std")
            cpp_flag = cpp_flag_prefix + "c++17"
            if not any(flag.startswith(cpp_flag_prefix) for flag in cflags):
                cflags.append(cpp_flag)

        def convert_to_absolute_paths_inplace(paths):
            # Helper function. See Note [Absolute include_dirs]
            if paths is not None:
                for i in range(len(paths)):
                    if not os.path.isabs(paths[i]):
                        paths[i] = os.path.abspath(paths[i])

        def unix_wrap_ninja_compile(
            sources,
            output_dir=None,
            macros=None,
            include_dirs=None,
            debug=0,
            extra_preargs=None,
            extra_postargs=None,
            depends=None,
        ):
            output_dir = os.path.abspath(output_dir)
            # See Note [Absolute include_dirs]
            convert_to_absolute_paths_inplace(self.compiler.include_dirs)

            _, objects, extra_postargs, pp_opts, _ = self.compiler._setup_compile(
                output_dir, macros, include_dirs, sources, depends, extra_postargs
            )
            common_cflags = self.compiler._get_cc_args(pp_opts, debug, extra_preargs)
            extra_cc_cflags = self.compiler.compiler_so[1:]
            with_cuda = any(map(_is_cuda_file, sources))

            if isinstance(extra_postargs, dict):
                post_cflags = extra_postargs["cxx"]
            else:
                post_cflags = list(extra_postargs)
            post_cflags = COMMON_HIP_FLAGS + post_cflags
            append_std17_if_no_std_present(post_cflags)

            cuda_post_cflags = None
            cuda_cflags = None
            if with_cuda:
                cuda_cflags = common_cflags
                if isinstance(extra_postargs, dict):
                    cuda_post_cflags = extra_postargs["nvcc"]
                else:
                    cuda_post_cflags = list(extra_postargs)
                cuda_post_cflags = cuda_post_cflags + _get_rocm_gpu_rdc_flags(cuda_post_cflags)
                cuda_post_cflags = COMMON_HIP_FLAGS + COMMON_HIPCC_FLAGS + cuda_post_cflags
                append_std17_if_no_std_present(cuda_post_cflags)
                cuda_cflags = [shlex.quote(f) for f in cuda_cflags]
                cuda_post_cflags = [shlex.quote(f) for f in cuda_post_cflags]

            _write_ninja_file_and_compile_objects(
                sources=sources,
                objects=objects,
                cflags=[shlex.quote(f) for f in extra_cc_cflags + common_cflags],
                post_cflags=[shlex.quote(f) for f in post_cflags],
                cuda_cflags=cuda_cflags,
                cuda_post_cflags=cuda_post_cflags,
                cuda_dlink_post_cflags=None,
                build_directory=output_dir,
                verbose=True,
                with_cuda=with_cuda,
            )
            return objects

        if self.compiler.compiler_type == "unix":
            if self.use_ninja:
                self.compiler.compile = unix_wrap_ninja_compile
            else:
                raise RuntimeError(
                    "[Primus-Turbo BuildExtension] Non-ninja backend is not supported yet. "
                    "Please enable ninja to build."
                )
        else:
            raise RuntimeError(
                f"[Primus-Turbo BuildExtension] Unsupported compiler type: {self.compiler.compiler_type}"
            )

        build_ext.build_extensions(self)

    # Simple hipify, replace the first occurrence of CUDA with HIP
    # in flags starting with "-" and containing "CUDA", but exclude -I flags
    def _hipify_compile_flags(self, extension):
        if isinstance(extension.extra_compile_args, dict) and "nvcc" in extension.extra_compile_args:
            modified_flags = []
            for flag in extension.extra_compile_args["nvcc"]:
                if flag.startswith("-") and "CUDA" in flag and not flag.startswith("-I"):
                    # check/split flag into flag and value
                    parts = flag.split("=", 1)
                    if len(parts) == 2:
                        flag_part, value_part = parts
                        # replace fist instance of "CUDA" with "HIP" only in the flag and not flag value
                        modified_flag_part = flag_part.replace("CUDA", "HIP", 1)
                        modified_flag = f"{modified_flag_part}={value_part}"
                    else:
                        # replace fist instance of "CUDA" with "HIP" in flag
                        modified_flag = flag.replace("CUDA", "HIP", 1)
                    modified_flags.append(modified_flag)
                    print(f"Modified flag: {flag} -> {modified_flag}")
                else:
                    modified_flags.append(flag)
            extension.extra_compile_args["nvcc"] = modified_flags


def _is_cuda_file(path: str) -> bool:
    valid_ext = [".cu", ".cuh", ".hip"]
    return os.path.splitext(path)[1] in valid_ext


def _get_rocm_gpu_rdc_flags(cflags: Optional[list[str]] = None) -> list[str]:
    has_gpu_rdc_flag = False
    if cflags is not None:
        has_custom_flags = False
        for flag in cflags:
            if "amdgpu-target" in flag or "offload-arch" in flag:
                has_custom_flags = True
            elif "gpu-rdc" in flag:
                has_gpu_rdc_flag = True
        if has_custom_flags:
            return [] if has_gpu_rdc_flag else ["-fno-gpu-rdc"]
    flags += [] if has_gpu_rdc_flag else ["-fno-gpu-rdc"]
    return flags


def _get_num_workers(verbose: bool) -> Optional[int]:
    max_jobs = os.environ.get("MAX_JOBS")
    if max_jobs is not None and max_jobs.isdigit():
        if verbose:
            print(f"Using envvar MAX_JOBS ({max_jobs}) as the number of workers...")
        return int(max_jobs)
    if verbose:
        print(
            "Allowing ninja to set a default number of workers... "
            "(overridable by setting the environment variable MAX_JOBS=N)"
        )
    return None


def _maybe_write(filename, new_content):
    r"""
    Equivalent to writing the content into the file but will not touch the file
    if it already had the right content (to avoid triggering recompile).
    """
    if os.path.exists(filename):
        with open(filename) as f:
            content = f.read()

        if content == new_content:
            # The file already contains the right thing!
            return

    with open(filename, "w") as source_file:
        source_file.write(new_content)


def _write_ninja_file_and_compile_objects(
    sources: list[str],
    objects,
    cflags,
    post_cflags,
    cuda_cflags,
    cuda_post_cflags,
    cuda_dlink_post_cflags,
    build_directory: str,
    verbose: bool,
    with_cuda: Optional[bool],
) -> None:
    verify_ninja_availability()

    # TODO:
    # compiler = get_cxx_compiler()
    # get_compiler_abi_compatibility_and_version(compiler)

    if with_cuda is None:
        with_cuda = any(map(_is_cuda_file, sources))
    build_file_path = os.path.join(build_directory, "build.ninja")
    if verbose:
        print(f"Emitting ninja build file {build_file_path}...")

    # Create build_directory if it does not exist
    if not os.path.exists(build_directory):
        if verbose:
            print(f"Creating directory {build_directory}...")
        # This is like mkdir -p, i.e. will also create parent directories.
        os.makedirs(build_directory, exist_ok=True)

    _write_ninja_file(
        path=build_file_path,
        cflags=cflags,
        post_cflags=post_cflags,
        cuda_cflags=cuda_cflags,
        cuda_post_cflags=cuda_post_cflags,
        cuda_dlink_post_cflags=cuda_dlink_post_cflags,
        sources=sources,
        objects=objects,
        ldflags=None,
        library_target=None,
        with_cuda=with_cuda,
    )
    if verbose:
        print("Compiling objects...")
    _run_ninja_build(
        build_directory,
        verbose,
        # It would be better if we could tell users the name of the extension
        # that failed to build but there isn't a good way to get it here.
        error_prefix="Error compiling objects for extension",
    )


def _run_ninja_build(build_directory: str, verbose: bool, error_prefix: str) -> None:
    command = ["ninja", "-v"]
    num_workers = _get_num_workers(verbose)
    if num_workers is not None:
        command.extend(["-j", str(num_workers)])
    env = os.environ.copy()
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        # Warning: don't pass stdout=None to subprocess.run to get output.
        # subprocess.run assumes that sys.__stdout__ has not been modified and
        # attempts to write to it by default.  However, when we call _run_ninja_build
        # from ahead-of-time cpp extensions, the following happens:
        # 1) If the stdout encoding is not utf-8, setuptools detaches __stdout__.
        #    https://github.com/pypa/setuptools/blob/7e97def47723303fafabe48b22168bbc11bb4821/setuptools/dist.py#L1110
        #    (it probably shouldn't do this)
        # 2) subprocess.run (on POSIX, with no stdout override) relies on
        #    __stdout__ not being detached:
        #    https://github.com/python/cpython/blob/c352e6c7446c894b13643f538db312092b351789/Lib/subprocess.py#L1214
        # To work around this, we pass in the fileno directly and hope that
        stdout_fileno = 1
        subprocess.run(
            command,
            shell=True,
            stdout=stdout_fileno if verbose else subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=build_directory,
            check=True,
            env=env,
        )
    except subprocess.CalledProcessError as e:
        # Python 2 and 3 compatible way of getting the error object.
        _, error, _ = sys.exc_info()
        # error.output contains the stdout and stderr of the build attempt.
        message = error_prefix
        # `error` is a CalledProcessError (which has an `output`) attribute, but
        # mypy thinks it's Optional[BaseException] and doesn't narrow
        if hasattr(error, "output") and error.output:  # type: ignore[union-attr]
            message += f": {error.output.decode(*SUBPROCESS_DECODE_ARGS)}"  # type: ignore[union-attr]
        raise RuntimeError(message) from e


def _write_ninja_file(
    path,
    cflags,
    post_cflags,
    cuda_cflags,
    cuda_post_cflags,
    cuda_dlink_post_cflags,
    sources,
    objects,
    ldflags,
    library_target,
    with_cuda,
    **kwargs,  # kwargs (ignored) to absorb new flags in torch.utils.cpp_extension
) -> None:
    r"""Write a ninja file that does the desired compiling and linking.

    `path`: Where to write this file
    `cflags`: list of flags to pass to $cxx. Can be None.
    `post_cflags`: list of flags to append to the $cxx invocation. Can be None.
    `cuda_cflags`: list of flags to pass to $nvcc. Can be None.
    `cuda_post_cflags`: list of flags to append to the $nvcc invocation. Can be None.
    `cuda_dlink_post_cflags`: list of flags to append to the $nvcc device code link invocation. Can be None.
    `sources`: list of paths to source files
    `objects`: list of desired paths to objects, one per source.
    `ldflags`: list of flags to pass to linker. Can be None.
    `library_target`: Name of the output library. Can be None; in that case, we do no linking.
    `with_cuda`: If we should be compiling with CUDA.
    """

    def sanitize_flags(flags):
        if flags is None:
            return []
        else:
            return [flag.strip() for flag in flags]

    cflags = sanitize_flags(cflags)
    post_cflags = sanitize_flags(post_cflags)
    cuda_cflags = sanitize_flags(cuda_cflags)
    cuda_post_cflags = sanitize_flags(cuda_post_cflags)
    cuda_dlink_post_cflags = sanitize_flags(cuda_dlink_post_cflags)
    ldflags = sanitize_flags(ldflags)

    # print("***************************************************")
    # print("cflags\n", cflags)
    # print("post_cflags\n", post_cflags)
    # print("cuda_cflags\n", cuda_cflags)
    # print("cuda_post_cflags\n", cuda_post_cflags)
    # print("cuda_dlink_post_cflags\n", cuda_dlink_post_cflags)
    # print("ldflags\n", ldflags)
    # print("sources\n", sources)
    # print("objects\n", objects)
    # print("library_target\n", library_target)
    # print("***************************************************")

    # Sanity checks...
    if len(sources) != len(objects):
        raise AssertionError("sources and objects lists must be the same length")
    if len(sources) == 0:
        raise AssertionError("At least one source is required to build a library")

    compiler = get_cxx_compiler()

    # Version 1.3 is required for the `deps` directive.
    config = ["ninja_required_version = 1.3"]
    config.append(f"cxx = {compiler}")
    if with_cuda or cuda_dlink_post_cflags:
        nvcc = _join_rocm_home("bin", "hipcc")
        config.append(f"nvcc = {nvcc}")

    post_cflags = COMMON_HIP_FLAGS + post_cflags
    flags = [f'cflags = {" ".join(cflags)}']
    flags.append(f'post_cflags = {" ".join(post_cflags)}')
    if with_cuda:
        flags.append(f'cuda_cflags = {" ".join(cuda_cflags)}')
        flags.append(f'cuda_post_cflags = {" ".join(cuda_post_cflags)}')
        cuda_post_cflags_gfx942, _ = _filter_compile_arch_args(cuda_post_cflags, "gfx942")
        flags.append(f'cuda_post_cflags_gfx942 = {" ".join(cuda_post_cflags_gfx942)}')
        cuda_post_cflags_gfx950, _ = _filter_compile_arch_args(cuda_post_cflags, "gfx950")
        flags.append(f'cuda_post_cflags_gfx950 = {" ".join(cuda_post_cflags_gfx950)}')
    flags.append(f'cuda_dlink_post_cflags = {" ".join(cuda_dlink_post_cflags)}')

    # Turn into absolute paths so we can emit them into the ninja build
    # file wherever it is.
    sources = [os.path.abspath(file) for file in sources]

    # See https://ninja-build.org/build.ninja.html for reference.
    compile_rule = ["rule compile"]
    compile_rule.append("  command = $cxx -MMD -MF $out.d $cflags -c $in -o $out $post_cflags")
    compile_rule.append("  depfile = $out.d")
    compile_rule.append("  deps = gcc")

    if with_cuda:
        nvcc_gendeps = ""
        cuda_compile_rule = [
            "rule cuda_compile",
            f"  command = $nvcc {nvcc_gendeps} $cuda_cflags -c $in -o $out $cuda_post_cflags",
        ]
        cuda_compile_rule_gfx942 = [
            "rule cuda_compile_gfx942",
            f"  command = $nvcc {nvcc_gendeps} $cuda_cflags -c $in -o $out $cuda_post_cflags_gfx942",
        ]
        cuda_compile_rule_gfx950 = [
            "rule cuda_compile_gfx950",
            f"  command = $nvcc {nvcc_gendeps} $cuda_cflags -c $in -o $out $cuda_post_cflags_gfx950",
        ]

    # Emit one build rule per source to enable incremental build.
    build = []
    for source_file, object_file in zip(sources, objects):
        is_cuda_source = _is_cuda_file(source_file) and with_cuda
        if is_cuda_source:
            if source_file.endswith("_gfx942.cu") or source_file.endswith("_gfx942.hip"):
                rule = "cuda_compile_gfx942"
            elif source_file.endswith("_gfx950.cu") or source_file.endswith("_gfx950.hip"):
                rule = "cuda_compile_gfx950"
            else:
                rule = "cuda_compile"
        else:
            rule = "compile"
        source_file = source_file.replace(" ", "$ ")
        object_file = object_file.replace(" ", "$ ")
        build.append(f"build {object_file}: {rule} {source_file}")

    if cuda_dlink_post_cflags:
        cuda_devlink_out = os.path.join(os.path.dirname(objects[0]), "dlink.o")
        cuda_devlink_rule = ["rule cuda_devlink"]
        cuda_devlink_rule.append("  command = $nvcc $in -o $out $cuda_dlink_post_cflags")
        cuda_devlink = [f'build {cuda_devlink_out}: cuda_devlink {" ".join(objects)}']
        objects += [cuda_devlink_out]
    else:
        cuda_devlink_rule, cuda_devlink = [], []

    if library_target is not None:
        link_rule = ["rule link"]
        link_rule.append("  command = $cxx $in $ldflags -o $out")
        link = [f'build {library_target}: link {" ".join(objects)}']
        default = [f"default {library_target}"]
    else:
        link_rule, link, default = [], [], []

    # 'Blocks' should be separated by newlines, for visual benefit.
    blocks = [config, flags, compile_rule]
    if with_cuda:
        blocks.append(cuda_compile_rule)  # type: ignore[possibly-undefined]
        blocks.append(cuda_compile_rule_gfx942)  # type: ignore[possibly-undefined]
        blocks.append(cuda_compile_rule_gfx950)  # type: ignore[possibly-undefined]

    blocks += [cuda_devlink_rule, link_rule, build, cuda_devlink, link, default]
    content = "\n\n".join("\n".join(b) for b in blocks)
    # Ninja requires a new lines at the end of the .ninja file
    content += "\n"
    _maybe_write(path, content)


def _filter_compile_arch_args(compile_args: list[str], arch: str) -> list[str]:
    offload_arch = f"--offload-arch={arch.lower()}"
    macro_arch = f"-DPRIMUS_TURBO_{arch.upper()}"
    exists = any(a == offload_arch or a == macro_arch for a in compile_args)

    new_compile_args = []
    for arg in compile_args:
        if arg.startswith("--offload-arch=") or arg.startswith("-DPRIMUS_TURBO_"):
            continue
        new_compile_args.append(arg)
    new_compile_args.append(offload_arch)
    new_compile_args.append(macro_arch)
    return new_compile_args, exists


# *****************************************
# *****************************************


def _select_base_build_ext():
    try:
        import torch  # noqa: F401
        from torch.utils.cpp_extension import BuildExtension as TorchBuildExtension

        # Monkey patching
        torch.utils.cpp_extension._write_ninja_file = _write_ninja_file
        return TorchBuildExtension
    except Exception:
        return BuildExtension


BaseBuildExtension = _select_base_build_ext()


class TurboBuildExt(BaseBuildExtension):
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

            print(f"[TurboBuildExt] Copied {filename} to:")

            src_dst_dir = Path("primus_turbo/lib")
            src_dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(built_path, src_dst_dir / filename)
            print(f"  -  {src_dst_dir}")

            build_dst_dir = Path(self.build_lib) / "primus_turbo" / "lib"
            build_dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(built_path, build_dst_dir / filename)
            print(f"  -  {build_dst_dir}")
