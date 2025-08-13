###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

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

FUNC_NAME = "gemm"


def gemm(m, n, k, dtype, seed):
    torch.manual_seed(seed)
    a_cpu = torch.randn(m, k, dtype=dtype, requires_grad=True)
    a = a_cpu.detach().to(DEVICE).requires_grad_()

    b_cpu = torch.randn(n, k, dtype=dtype, requires_grad=True)
    b = b_cpu.detach().to(DEVICE).requires_grad_()

    out = torch.matmul(a, b.T)
    out = out.cpu()
    ref = torch.matmul(a_cpu, b_cpu.T)
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

    for shape in [(512, 128, 256), (8192, 8192, 8192), (1, 2048, 128)]:
        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            m, n, k = shape
            out, ref = gemm(m, n, k, dtype, seed)
            ulp = ulp_error(out, ref)

            result = {
                "func": f"{FUNC_NAME.upper()} ({device_name} vs {ref_device})",
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
                dump_file = get_tensor_name(device_type, FUNC_NAME, dtype, shape)
                dump_tensor(out, dump_dir, dump_file)

            if load_config_path is not None:
                load_file = get_tensor_name(load_config.get("device_type"), FUNC_NAME, dtype, shape)
                out_load = load_tensor(load_dir, load_file)
                ulp_gpu = ulp_error(out, out_load)

                gpu_result = {
                    "func": f"{FUNC_NAME.upper()} ({device_name} vs {load_config.get('device_name')})",
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

    report_with_cpu = report_dir / f"benchmark_{device_type}_{FUNC_NAME}.xlsx"
    report_with_gpu = report_dir / f"benchmark_GPU_{FUNC_NAME}.xlsx"

    save_to_excel(results_with_cpu, report_with_cpu)
    save_to_excel(results_with_gpu, report_with_gpu)

    benchmark_reports = [report_with_cpu]
    if load_config_path and load_config.get("report_path") and Path(load_config.get("report_path")).exists():
        benchmark_reports.append(Path(load_config.get("report_path")))
    if Path(report_with_gpu).exists():
        benchmark_reports.append(report_with_gpu)
    if len(benchmark_reports) >= 2:
        report_path = report_dir / f"benchmark_{FUNC_NAME}.xlsx"
        merge_excels(benchmark_reports, report_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-config-path", default=None, type=str)
    parser.add_argument("--dump-dir-path", default=None, type=str)
    parser.add_argument("--report-dir-path", type=str)
    parser.add_argument("--seed", default=42, type=int)

    args = parser.parse_args()
    benchmark(args.seed, args.report_dir_path, args.load_config_path, args.dump_dir_path)
