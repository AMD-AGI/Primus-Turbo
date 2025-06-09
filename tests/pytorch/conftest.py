from pathlib import Path
import torch
import pandas as pd
from .test_special_functions import results


def merge_excels(amd_file, nv_file, subdir_path):
    if amd_file.exists() and nv_file.exists():
        df_amd = pd.read_excel(amd_file)
        df_nv = pd.read_excel(nv_file)

        # check columns
        if list(df_amd.columns) != list(df_nv.columns):
            print("❌ The column number of two excel is not equal")
            return

        if len(df_amd) != len(df_nv):
            print("❌ The row number of two excel is not equal")
            return

        min_len = min(len(df_amd), len(df_nv))

        # interleaved concat
        merged_rows = []
        for i in range(min_len):
            merged_rows.append(df_amd.iloc[i])
            merged_rows.append(df_nv.iloc[i])

        # merged DataFrame
        merged_df = pd.DataFrame(merged_rows, columns=df_amd.columns)

        # save results
        output_path = subdir_path / "merged.xlsx"
        merged_df.to_excel(output_path, index=False)
        print(f"✅ merged excel saved as {output_path}")


def pytest_sessionfinish(session, exitstatus):
    # save to excel
    file_name = (
        f"test_function_accuracy_{torch.cuda.get_device_name(0).split()[0]}.xlsx"
    )
    current_dir = Path(__file__).resolve().parent
    subdir_path = current_dir / "test_accu_results"
    saved_file = subdir_path / file_name

    amd_file = subdir_path / "test_function_accuracy_AMD.xlsx"
    nv_file = subdir_path / "test_function_accuracy_NVIDIA.xlsx"

    if results:
        subdir_path.mkdir(exist_ok=True)

        df = pd.DataFrame(results)
        df.to_excel(saved_file, index=False)
        print("✅ Test sccuracy results has be saved as test_function_accuracy.xlsx")

    merge_excels(amd_file, nv_file, subdir_path)
