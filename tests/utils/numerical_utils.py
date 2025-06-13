import os
from pathlib import Path

import pandas as pd
import torch

from .metric_utils import (
    cosine_similarity,
    max_abs_error,
    mean_squared_error,
    relative_error,
    ulp_error,
)


def get_device_name(is_load=False):
    if is_load:
        if os.getenv("NUMERICAL_LOAD_DEVICE"):
            return os.getenv("NUMERICAL_LOAD_DEVICE").split("_")[1]
        return None

    device_name = (
        torch.cuda.get_device_name(0).split()[2]
        if torch.version.hip is not None
        else torch.cuda.get_device_name(0).split()[1]
    )
    return device_name


def get_device_type(is_load=False):
    if is_load:
        if os.getenv("NUMERICAL_LOAD_DEVICE"):
            return os.getenv("NUMERICAL_LOAD_DEVICE").split("_")[0]
        return None
    return torch.cuda.get_device_name(0).split()[0]


def get_format_name(device_type="GPU", func_type="special_func"):
    return f"{device_type}_{func_type}.xlsx"


def get_tensor_name(device_name, func_name, dtype, shape):
    return f"{device_name}_{func_name}_{str(dtype).split('.')[-1]}_{'_'.join(map(str, shape))}.pt"


def get_subdir(save_name="numerical_results"):
    try:
        current_dir = Path(__file__).resolve().parent.parent
    except NameError:
        current_dir = Path.cwd()
    path = current_dir / save_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_file_path(file_dir, file_name):
    return Path(file_dir / file_name)


def load_tensor(subdir, device_type, func_name, dtype, shape):
    if os.getenv("NUMERICAL_LOAD_DEVICE"):
        file_name = get_tensor_name(device_type, func_name, dtype, shape)
        file_path = get_file_path(subdir, file_name)
        if file_path.exists():
            return torch.load(file_path)
    os.environ.pop("NUMERICAL_LOAD_DEVICE", None)
    return None


def dump_tensor(tensor, subdir, device_type, func_name, dtype, shape):
    if os.getenv("NUMERICAL_DUMP", "0") != "0":
        file_name = get_tensor_name(device_type, func_name, dtype, shape)
        torch.save(tensor, get_file_path(subdir, file_name))


def save_result_to_excel(data, file_path):
    if os.getenv("NUMERICAL_SAVE") and data:
        df = pd.DataFrame(data)
        df.to_excel(file_path, index=False)
        print(f"✅ Saved: {file_path}")


def merge_excels(files, output_path):
    if not (2 <= len(files) <= 3):
        raise ValueError("Only 2 or 3 Excel files are supported.")

    for file in files:
        if not file.exists():
            raise FileNotFoundError(f"Missing file: {file}")

    dfs = [pd.read_excel(f) for f in files]
    if not all(df.shape[0] == dfs[0].shape[0] for df in dfs):
        raise ValueError("All files must have the same number of rows.")

    if not all(list(df.columns) == list(dfs[0].columns) for df in dfs[1:]):
        raise ValueError("All files must have the same columns.")

    interleaved = []
    for i in range(dfs[0].shape[0]):
        for df in dfs:
            interleaved.append(df.iloc[i])

    merged_df = pd.DataFrame(interleaved, columns=dfs[0].columns)
    merged_df.to_excel(output_path, index=False)
    print(f"✅ Merged Excel saved to {output_path}")


def post_process(ref_device, device_name, func_name, dtype, shape, out, ref, data):
    ulp = ulp_error(out, ref)

    def item(metric):
        return f"{metric:.3e}" if isinstance(metric, float) else str(metric)

    print(f"\n[{func_name.upper()}][{device_name} vs {ref_device}] dtype={dtype}, shape={shape}")
    print(f"RelError:   {relative_error(ref, out):.3e}")
    print(f"MaxAbsErr:  {max_abs_error(ref, out):.3e}")
    print(f"MSE:        {mean_squared_error(ref, out):.3e}")
    print(f"CosSim:     {cosine_similarity(ref, out):.6f}")
    print(f"ULP(max):   {ulp.max().item()}, ULP(mean): {ulp.float().mean().item():.2f}")

    data.append(
        {
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
    )

    if func_name == "pow" or ((func_name == "gemm") and (dtype == torch.bfloat16)):
        data.append({k: "" for k in data[-1].keys()})
