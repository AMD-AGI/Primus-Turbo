import pytest
import torch

from tests.utils.numerical_utils import (
    get_device_name,
    get_device_type,
    get_file_path,
    get_format_name,
    get_subdir,
    post_process,
    merge_excels,
    save_result_to_excel,
    load_tensor,
    dump_tensor,
)

results, load_results = [], []


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "shapes", [(512, 128, 256), (8192, 8192, 8192), (1, 2048, 128)]
)
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


# @pytest.mark.parametrize("dtype", [torch.float8_e4m3fnuz])
# @pytest.mark.parametrize("shapes", [(512, 128, 256),
#                                     (8192, 8192, 8192),
#                                     (1, 2048, 128)])
# def test_fp8_gemm_numerical(dtype, shapes):
#     if not torch.cuda.is_available():
#         pytest.skip("CUDA not available")
#     torch.manual_seed(42)

#     m, n, k = shapes
#     device = "cuda"

#     a = torch.randn(m, k, device=device, requires_grad=True)
#     a_cpu = a.float().detach().clone().cpu().requires_grad_()
#     scale_a = torch.tensor(1.0, dtype=torch.float32, device=device)
#     a = a.to(dtype)

#     b = torch.randn(n, k, device=device, requires_grad=True)
#     b_cpu = b.float().detach().clone().cpu().requires_grad_()
#     scale_b = torch.tensor(1.0, dtype=torch.float32, device=device)
#     b = b.to(dtype)

#     out = torch._scaled_mm(a, b.T, scale_a=scale_a, scale_b=scale_b, out_dtype=torch.float32)
#     out = out.float().cpu()
#     ref = torch.matmul(a_cpu, b_cpu.T)

#     ulp = ulp_error(out, ref)

#     print(f"\n[GEMM] dtype={dtype}, shape={shapes}, result:")
#     print(f"RelError:   {relative_error(ref, out):.3e}")
#     print(f"MAE:        {max_abs_error(ref, out):.3e}")
#     print(f"MSE:        {mean_squared_error(ref, out):.3e}")
#     print(f"CosSim:     {cosine_similarity(ref, out):.6f}")
#     print(f"ULP(max):   {ulp.max().item()}, ULP(mean): {ulp.float().mean().item():.2f}")
