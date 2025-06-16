import argparse
import json
from pathlib import Path

import torch
from metrics import (
    cosine_similarity,
    max_abs_error,
    mean_squared_error,
    relative_error,
    ulp_error,
)
from utils import (
    DEVICE,
    dump_tensor,
    get_device_name,
    get_device_type,
    get_tensor_name,
    load_tensor,
    merge_excels,
    save_to_excel,
)

FUNC_TABLE = {
    "exp": lambda x: torch.exp(x),
    "log": lambda x: torch.log(x.abs() + 1e-3),
    "sqrt": lambda x: torch.sqrt(x.abs()),
    "sigmoid": lambda x: torch.sigmoid(x),
    "tanh": lambda x: torch.tanh(x),
    "pow": lambda x, y: torch.pow(x.abs() + 1e-3, y.abs()),
}


def special_func(func_name, shape, dtype, seed):
    torch.manual_seed(seed)
    x_cpu = torch.randn(*shape).to(dtype)
    x = x_cpu.to(DEVICE)

    if func_name == "pow":
        y_cpu = torch.randn(*shape).to(dtype)
        ref = FUNC_TABLE[func_name](x_cpu, y_cpu)
        out = FUNC_TABLE[func_name](x, y_cpu.to(DEVICE))
    else:
        ref = FUNC_TABLE[func_name](x_cpu)
        out = FUNC_TABLE[func_name](x)

    out = out.cpu()
    ref = ref
    return out, ref


def benchmark(seed, report_dir_path, load_config_path=None, dump_dir_path=None):
    device_type = get_device_type()
    device_name = get_device_name()
    ref_device = "CPU"

    report_dir = Path(report_dir_path)
    report_dir.mkdir(parents=True, exist_ok=True)

    if load_config_path is not None:
        with open(load_config_path, "r", encoding="utf-8") as f:
            load_config: dict = json.load(f)
            load_dir = Path(load_config.get("load_dir"))

    if dump_dir_path is not None:
        dump_dir = Path(dump_dir_path)
        dump_dir.mkdir(parents=True, exist_ok=True)

    results_with_cpu = []
    results_with_gpu = []

    def item(metric):
        return f"{metric:.3e}" if isinstance(metric, float) else str(metric)

    for shape in [(1024,), (64, 64)]:
        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            for func_name in FUNC_TABLE.keys():
                out, ref = special_func(func_name, shape, dtype, seed)
                ulp = ulp_error(out, ref)

                result = {
                    "func": f"{func_name.upper()} ({device_name} vs {ref_device})",
                    "dtype": str(dtype).split(".")[-1],
                    "shape": str(shape),
                    "RelError": item(relative_error(ref, out)),
                    "MaxAbsErr": item(max_abs_error(ref, out)),
                    "MSE": item(mean_squared_error(ref, out)),
                    "CosSim": f"{cosine_similarity(ref, out):.6f}",
                    "ULP_max": str(ulp.max().item()),
                    "ULP_mean": f"{ulp.float().mean().item():.2f}",
                }
                results_with_cpu.append(result)

                if dump_dir_path is not None:
                    dump_file = get_tensor_name(device_type, func_name, dtype, shape)
                    dump_tensor(out, dump_dir, dump_file)

                if load_config_path is not None:
                    load_file = get_tensor_name(load_config.get("device_type"), func_name, dtype, shape)
                    out_load = load_tensor(load_dir, load_file)
                    ulp_gpu = ulp_error(out, out_load)

                    gpu_result = {
                        "func": f"{func_name.upper()} ({device_name} vs {load_config.get('device_name')})",
                        "dtype": str(dtype).split(".")[-1],
                        "shape": str(shape),
                        "RelError": item(relative_error(out_load, out)),
                        "MaxAbsErr": item(max_abs_error(out_load, out)),
                        "MSE": item(mean_squared_error(out_load, out)),
                        "CosSim": f"{cosine_similarity(out_load, out):.6f}",
                        "ULP_max": str(ulp_gpu.max().item()),
                        "ULP_mean": f"{ulp_gpu.float().mean().item():.2f}",
                    }
                    results_with_gpu.append(gpu_result)

            results_with_cpu.append({k: "" for k in results_with_cpu[-1].keys()})
            if len(results_with_gpu) > 0:
                results_with_gpu.append({k: "" for k in results_with_gpu[-1].keys()})

    report_with_cpu = report_dir / f"benchmark_{device_type}_special_func.xlsx"
    report_with_gpu = report_dir / f"benchmark_GPU_special_func.xlsx"

    save_to_excel(results_with_cpu, report_with_cpu)
    save_to_excel(results_with_gpu, report_with_gpu)

    benchmark_reports = [report_with_cpu]
    if load_config_path and load_config.get("report_path") and Path(load_config.get("report_path")).exists():
        benchmark_reports.append(Path(load_config.get("report_path")))
    if Path(report_with_gpu).exists():
        benchmark_reports.append(report_with_gpu)
    if len(benchmark_reports) >= 2:
        report_path = report_dir / "benchmark_special_func.xlsx"
        merge_excels(benchmark_reports, report_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-config-path", default=None, type=str)
    parser.add_argument("--dump-dir-path", default=None, type=str)
    parser.add_argument("--report-dir-path", type=str)
    parser.add_argument("--seed", default=0, type=int)

    args = parser.parse_args()
    benchmark(args.seed, args.report_dir_path, args.load_config_path, args.dump_dir_path)
