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


FUNC_TABLE = {
    "exp": lambda x: torch.exp(x),
    "log": lambda x: torch.log(x.abs() + 1e-3),
    "sqrt": lambda x: torch.sqrt(x.abs()),
    "sigmoid": lambda x: torch.sigmoid(x),
    "tanh": lambda x: torch.tanh(x),
    "pow": lambda x, y: torch.pow(x.abs() + 1e-3, y.abs()),
}

results, load_results = [], []


@pytest.mark.parametrize("func_name", FUNC_TABLE.keys())
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(1024,), (64, 64)])
def test_special_function_accuracy(func_name, dtype, shape):
    torch.manual_seed(0)
    device = torch.device("cuda")
    device_name = get_device_name()
    device_type = get_device_type()

    x = torch.randn(*shape, device=device).to(dtype)

    if func_name == "pow":
        y = torch.randn(*shape, device=device).to(dtype)
        ref = FUNC_TABLE[func_name](x.cpu(), y.cpu())
        out = FUNC_TABLE[func_name](x, y)
    else:
        ref = FUNC_TABLE[func_name](x.cpu())
        out = FUNC_TABLE[func_name](x)

    out = out.cpu()
    ref = ref

    save_dir = get_subdir()
    device_type_load = get_device_type(is_load=True)
    out_load = load_tensor(save_dir, device_type_load, func_name, dtype, shape)

    dump_tensor(out, save_dir, device_type, func_name, dtype, shape)
    print(f"{out_load=}")
    if out_load is not None:
        post_process(
            "CPU",
            get_device_name(is_load=True),
            func_name,
            dtype,
            shape,
            out_load,
            ref,
            load_results,
        )
        # post_process(
        #     get_device_name(is_load=True),
        #     device_name,
        #     func_name,
        #     dtype,
        #     shape,
        #     out,
        #     out_load,
        #     load_results,
        # )

    post_process("CPU", device_name, func_name, dtype, shape, out, ref, results)


@pytest.fixture(scope="session", autouse=True)
def finalize_results_on_exit(request):
    def finalizer():
        save_dir = get_subdir()
        print(f"{load_results=}")
        if load_results:
            load_file_path = get_file_path(save_dir, get_format_name())
            save_result_to_excel(load_results, load_file_path)

        if results:
            device_type = get_device_type()
            file_path = get_file_path(save_dir, get_format_name(device_type))
            save_result_to_excel(results, file_path)

        amd_file = save_dir / "AMD_special_func.xlsx"
        nv_file = save_dir / "NVIDIA_special_func.xlsx"
        comp_file = save_dir / "GPU_special_func.xlsx"
        numerical_file = save_dir / "numerical_special_func.xlsx"

        merge_files = []
        for file in [amd_file, nv_file, comp_file]:
            if file.exists():
                merge_files.append(file)
        if len(merge_files) > 1:
            merge_excels(merge_files, numerical_file)

    request.addfinalizer(finalizer)
