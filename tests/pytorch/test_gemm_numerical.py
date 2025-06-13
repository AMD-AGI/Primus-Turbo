import pytest
import torch

from tests.utils.numerical_utils import (
    dump_tensor,
    get_device_name,
    get_device_type,
    get_file_path,
    get_format_name,
    get_subdir,
    load_tensor,
    merge_excels,
    post_process,
    save_result_to_excel,
)

results, load_results = [], []


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shapes", [(512, 128, 256), (8192, 8192, 8192), (1, 2048, 128)])
def test_gemm_numerical(dtype, shapes):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    torch.manual_seed(42)
    device_name = get_device_name()
    device_type = get_device_type()

    m, n, k = shapes
    device = "cuda"

    a_cpu = torch.randn(m, k, dtype=dtype, requires_grad=True)
    a = a_cpu.detach().to(device).requires_grad_()

    b_cpu = torch.randn(n, k, dtype=dtype, requires_grad=True)
    b = b_cpu.detach().to(device).requires_grad_()

    out = torch.matmul(a, b.T)
    out = out.cpu()
    ref = torch.matmul(a_cpu, b_cpu.T)

    save_dir = get_subdir()
    device_type_load = get_device_type(is_load=True)
    out_load = load_tensor(save_dir, device_type_load, "gemm", dtype, shapes)

    dump_tensor(out, save_dir, device_type, "gemm", dtype, shapes)
    if out_load is not None:
        post_process(
            get_device_name(is_load=True),
            device_name,
            "gemm",
            dtype,
            shapes,
            out,
            out_load,
            load_results,
        )

    post_process("CPU", device_name, "gemm", dtype, shapes, out, ref, results)


@pytest.fixture(scope="session", autouse=True)
def finalize_results_on_exit(request):
    def finalizer():
        save_dir = get_subdir()
        print(f"{load_results=}")
        if load_results:
            load_file_path = get_file_path(save_dir, get_format_name("gemm"))
            save_result_to_excel(load_results, load_file_path)

        if results:
            device_type = get_device_type()
            file_path = get_file_path(save_dir, get_format_name(device_type, "gemm"))
            save_result_to_excel(results, file_path)

        amd_file = save_dir / "AMD_gemm.xlsx"
        nv_file = save_dir / "NVIDIA_gemm.xlsx"
        comp_file = save_dir / "GPU_gemm.xlsx"
        numerical_file = save_dir / "numerical_gemm.xlsx"

        merge_files = []
        for file in [amd_file, nv_file, comp_file]:
            if file.exists():
                merge_files.append(file)
        if len(merge_files) > 1:
            merge_excels(merge_files, numerical_file)

    request.addfinalizer(finalizer)
