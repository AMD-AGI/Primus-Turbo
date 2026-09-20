###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""CPU-only regression tests for the MXFP4 variable-K dispatch contract.

The production package eagerly loads its CUDA extension and queries the current
GPU during import. Run the dispatch checks in a subprocess with lightweight
parent packages and a mocked gfx950 capability so this file remains runnable on
a host with no visible GPU. No kernel is launched and all tensors are on ``meta``.
"""

import ast
import os
import subprocess
import sys
import textwrap
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _keyword(call: ast.Call, name: str) -> ast.expr:
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    raise AssertionError(f"{name} keyword not found")


def test_variable_k_callers_use_nt_tag():
    grouped_gemm_tree = ast.parse((_REPO_ROOT / "primus_turbo/pytorch/ops/grouped_gemm_fp4.py").read_text())
    bgrad_calls = [
        node
        for node in ast.walk(grouped_gemm_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_bgrad_grouped_gemm_fp4_impl_wrapper"
    ]
    assert len(bgrad_calls) == 1
    assert ast.literal_eval(_keyword(bgrad_calls[0], "trans_b")) is True

    grouped_mlp_tree = ast.parse((_REPO_ROOT / "primus_turbo/pytorch/ops/grouped_mlp_fp4.py").read_text())
    wgrad_wrapper = next(
        node
        for node in grouped_mlp_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_wgrad_grouped_gemm_fp4_impl_wrapper"
    )
    option_builders = [
        node
        for node in ast.walk(wgrad_wrapper)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "dict"
        and any(keyword.arg == "trans_b" for keyword in node.keywords)
    ]
    assert len(option_builders) == 1
    assert ast.literal_eval(_keyword(option_builders[0], "trans_b")) is True


_CPU_DISPATCH_CHECK = textwrap.dedent(
    r"""
    import importlib
    import sys
    import types
    from pathlib import Path

    import torch

    repo_root = Path.cwd()
    pytorch_root = repo_root / "primus_turbo/pytorch"
    for name, path in (
        ("primus_turbo.pytorch", pytorch_root),
        ("primus_turbo.pytorch.core", pytorch_root / "core"),
        ("primus_turbo.pytorch.kernels", pytorch_root / "kernels"),
        (
            "primus_turbo.pytorch.kernels.grouped_gemm",
            pytorch_root / "kernels/grouped_gemm",
        ),
    ):
        package = types.ModuleType(name)
        package.__path__ = [str(path)]
        package.__package__ = name
        sys.modules[name] = package

    torch.cuda.current_device = lambda: 0
    torch.cuda.get_device_properties = lambda _: types.SimpleNamespace(
        major=9,
        minor=5,
        multi_processor_count=256,
    )

    fp4_impl = importlib.import_module(
        "primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_fp4_impl"
    )

    class TensorSpec:
        def __init__(self, shape, dtype):
            self.shape = shape
            self.dtype = dtype

    dtype_token = object()
    out_dtype_token = object()
    granularity_token = object()
    key_common = dict(
        a=TensorSpec((512, 2048), dtype_token),
        a_scales=None,
        b_scales=None,
        group_lens=TensorSpec((8,), None),
        group_offs=None,
        trans_a=False,
        trans_b=True,
        trans_c=False,
        out_dtype=out_dtype_token,
        granularity=granularity_token,
        num_cu=None,
    )
    key_n3072 = fp4_impl.GroupedGEMMFP4VariableKKernelDispatcher.make_key(
        b=TensorSpec((3072, 2048), dtype_token),
        **key_common,
    )
    key_n4096 = fp4_impl.GroupedGEMMFP4VariableKKernelDispatcher.make_key(
        b=TensorSpec((4096, 2048), dtype_token),
        **key_common,
    )
    assert key_n3072 != key_n4096
    assert key_n3072[1:4] == (512, 3072, 2048)
    assert key_n4096[1:4] == (512, 4096, 2048)

    accumulate_key = fp4_impl.GroupedGEMMFP4VariableKKernelDispatcher.make_key(
        b=TensorSpec((3072, 2048), dtype_token),
        inplace_add_to_out=True,
        **key_common,
    )
    assert key_n3072 != accumulate_key

    fp4 = fp4_impl.float4_e2m1fn_x2
    a = torch.empty((128, 256), device="meta", dtype=fp4)
    b = torch.empty((64, 256), device="meta", dtype=fp4)
    a_scales = torch.empty((128, 16), device="meta", dtype=torch.uint8)
    b_scales = torch.empty((64, 16), device="meta", dtype=torch.uint8)
    group_lens = torch.empty((4,), device="meta", dtype=torch.int64)
    group_offs = torch.empty((5,), device="meta", dtype=torch.int64)
    backend_kwargs = dict(
        a=a,
        b=b,
        a_scales=a_scales,
        b_scales=b_scales,
        group_lens=group_lens,
        group_offs=group_offs,
        trans_a=False,
        trans_b=True,
        trans_c=False,
        out_dtype=torch.bfloat16,
        granularity=fp4_impl.ScalingGranularity.MX_BLOCKWISE,
        num_cu=None,
    )

    backends = (
        fp4_impl.GroupedGEMMFP4VariableKTritonBackend,
        fp4_impl.GroupedGEMMFP4VariableKFlyDSLBackend,
    )
    for backend in backends:
        assert backend.can_handle(**backend_kwargs) is True
        assert backend.can_handle(**{**backend_kwargs, "trans_b": False}) is False
        assert backend.can_handle(**{**backend_kwargs, "trans_a": True}) is False

    out = torch.empty((4, 128, 64), device="meta", dtype=torch.bfloat16)
    assert (
        fp4_impl.GroupedGEMMFP4VariableKTritonBackend.can_handle(
            **backend_kwargs,
            inplace_add_to_out=True,
            out=out,
        )
        is False
    )
    assert (
        fp4_impl.GroupedGEMMFP4VariableKFlyDSLBackend.can_handle(
            **backend_kwargs,
            inplace_add_to_out=True,
            out=out,
        )
        is True
    )

    meta_kwargs = {
        **backend_kwargs,
        "granularity": fp4_impl.ScalingGranularity.MX_BLOCKWISE.value,
        "default_backend": 0,
    }
    result = fp4_impl.grouped_gemm_fp4_variable_k_impl_meta(**meta_kwargs)
    assert result.shape == (4, 128, 64)
    assert result.device.type == "meta"
    assert result.dtype == torch.bfloat16

    swapped = fp4_impl.grouped_gemm_fp4_variable_k_impl_meta(
        **{**meta_kwargs, "trans_c": True}
    )
    assert swapped.shape == (4, 64, 128)

    def must_reject(function, kwargs):
        try:
            function(**kwargs)
        except AssertionError:
            return
        raise AssertionError("invalid variable-K contract was accepted")

    must_reject(
        fp4_impl.grouped_gemm_fp4_variable_k_impl_meta,
        {**meta_kwargs, "trans_b": False},
    )
    mismatched_b = torch.empty((64, 128), device="meta", dtype=fp4)
    must_reject(
        fp4_impl.grouped_gemm_fp4_variable_k_impl_meta,
        {**meta_kwargs, "b": mismatched_b},
    )

    accum_meta_kwargs = {**meta_kwargs, "out": out}
    assert fp4_impl.grouped_gemm_fp4_variable_k_accum_impl_meta(**accum_meta_kwargs) is None
    must_reject(
        fp4_impl.grouped_gemm_fp4_variable_k_accum_impl_meta,
        {**accum_meta_kwargs, "trans_b": False},
    )
    must_reject(
        fp4_impl.grouped_gemm_fp4_variable_k_accum_impl_meta,
        {**accum_meta_kwargs, "b": mismatched_b},
    )
    """
)


def test_variable_k_dispatch_contract_without_gpu():
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(_REPO_ROOT) if not pythonpath else f"{_REPO_ROOT}{os.pathsep}{pythonpath}"
    result = subprocess.run(
        [sys.executable, "-c", _CPU_DISPATCH_CHECK],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"CPU-only MXFP4 dispatch checks failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
