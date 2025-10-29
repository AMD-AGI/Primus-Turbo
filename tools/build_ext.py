import os
import shutil
from pathlib import Path

from torch.utils.cpp_extension import BuildExtension


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

        src_dst_dir = Path("primus_turbo/lib")
        src_dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(built_path, src_dst_dir / filename)
        build_dst_dir = Path(self.build_lib) / "primus_turbo" / "lib"
        build_dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(built_path, build_dst_dir / filename)
        print(f"[TurboBuildExt] Copied {filename} to:")
        print(f"  -  {src_dst_dir}")
        print(f"  -  {build_dst_dir}")
