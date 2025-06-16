import pandas as pd
import torch

DEVICE = "cuda:0"


def is_ROCM():
    return torch.cuda.is_available() and torch.version.hip


def load_tensor(dir_path, file_name):
    file_path = dir_path / file_name
    return torch.load(file_path)


def dump_tensor(tensor, dir_path, file_name):
    file_path = dir_path / file_name
    torch.save(tensor, file_path)


def get_device_name():
    return torch.cuda.get_device_name(0).split()[2] if is_ROCM() else torch.cuda.get_device_name(0).split()[1]


def get_device_type():
    return torch.cuda.get_device_name(0).split()[0]


def get_tensor_name(device_type, func_name, dtype, shape):
    return f"{device_type}_{func_name}_{str(dtype).split('.')[-1]}_{'_'.join(map(str, shape))}.pt"


def save_to_excel(data, file_path):
    if data:
        df = pd.DataFrame(data)
        df.to_excel(file_path, index=False)
        print(f"✅ Saved: {file_path}")


def merge_excels(files, output_path):
    if len(files) < 2:
        raise ValueError("Only 2 or more Excel files are supported.")

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
