###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Exercise real accumulation wrappers and tuning loops with CPU arithmetic backends."""

import importlib
import inspect

import pytest
import torch

from primus_turbo.pytorch.core.backend import (
    BackendChoice,
    BackendEntry,
    BackendType,
    GlobalBackendManager,
    KernelBackend,
    TuneCache,
)
from primus_turbo.pytorch.core.low_precision import ScalingGranularity

CASES = [
    ("gemm.gemm_impl", "gemm_accum_impl", "GEMMKernelDispatcher"),
    ("gemm.gemm_fp8_impl", "gemm_fp8_accum_impl", "GEMMFP8KernelDispatcher"),
    ("gemm.gemm_fp4_impl", "gemm_fp4_accum_impl", "GEMMFP4KernelDispatcher"),
    (
        "grouped_gemm.grouped_gemm_impl",
        "grouped_gemm_variable_k_accum_impl",
        "GroupedGEMMVariableKKernelDispatcher",
    ),
    (
        "grouped_gemm.grouped_gemm_fp8_impl",
        "grouped_gemm_fp8_variable_k_accum_impl",
        "GroupedGEMMFP8VariableKKernelDispatcher",
    ),
    (
        "grouped_gemm.grouped_gemm_fp4_impl",
        "grouped_gemm_fp4_variable_k_accum_impl",
        "GroupedGEMMFP4VariableKKernelDispatcher",
    ),
]


@pytest.mark.parametrize("module_name,op_name,dispatcher_name", CASES)
@pytest.mark.parametrize("mode", ["global", "per_op", "pinned", "capture"])
@pytest.mark.parametrize("fail_profile", [False, True])
def test_accumulation_profiles_scratch_once_per_cache_miss(
    monkeypatch, module_name, op_name, dispatcher_name, mode, fail_profile
):
    module = importlib.import_module("primus_turbo.pytorch.kernels." + module_name)
    dispatcher = getattr(module, dispatcher_name)
    # Bypass custom-op device routing only; run the registered Python implementation.
    op = getattr(module, op_name)._init_fn
    grouped = module_name.startswith("grouped")
    choice = (
        BackendChoice(backend=BackendType.HIPBLASLT)
        if mode == "pinned"
        else BackendChoice(auto_tune=mode == "per_op")
    )
    monkeypatch.setattr(GlobalBackendManager, "auto_tune_enabled", lambda: mode != "per_op")
    monkeypatch.setattr(
        GlobalBackendManager, "get_grouped_gemm_backend" if grouped else "get_gemm_backend", lambda _: choice
    )
    monkeypatch.setattr(dispatcher, "_is_graph_capturing", lambda: mode == "capture")
    monkeypatch.setattr(dispatcher, "_cache", TuneCache())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    if grouped:
        utils = importlib.import_module("primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_utils")
        monkeypatch.setattr(
            utils, "group_offs_from_lens", lambda lens: torch.cat((lens.new_zeros(1), lens.cumsum(0)))
        )
    profile_outputs = []

    class Add(KernelBackend):
        @staticmethod
        def can_handle(**kwargs):
            return True

        @staticmethod
        def execute(a, b, out=None, inplace_add_to_out=False, **kwargs):
            product = a.T @ b if grouped else a @ b.T
            if grouped:
                product = product.unsqueeze(0)
            return out.add_(product) if inplace_add_to_out else product

    class OrdinaryOnly(Add):
        @staticmethod
        def can_handle(inplace_add_to_out=False, **kwargs):
            return not inplace_add_to_out

        @staticmethod
        def execute(**kwargs):
            assert not kwargs.get("inplace_add_to_out", False)
            return Add.execute(**kwargs)

    def profile(cls, backend, **kwargs):
        if kwargs.get("inplace_add_to_out", False):
            profile_outputs.append(kwargs["out"])
        for _ in range(3):
            backend.execute(**kwargs)
        if fail_profile and kwargs.get("inplace_add_to_out", False):
            raise RuntimeError("profile failed after writing scratch")
        return 0.5 if backend is OrdinaryOnly else 1.0

    monkeypatch.setattr(dispatcher, "profile", classmethod(profile))
    monkeypatch.setattr(
        dispatcher,
        "_backends",
        {
            BackendType.CK: BackendEntry(OrdinaryOnly),
            BackendType.HIPBLASLT: BackendEntry(Add),
        },
    )
    out = torch.full((1, 2, 2) if grouped else (2, 2), 7.0)
    values = dict(
        a=torch.ones(2, 2),
        b=torch.ones(2, 2),
        a_scale_inv=torch.ones(1),
        b_scale_inv=torch.ones(1),
        a_scales=torch.ones(1),
        b_scales=torch.ones(1),
        trans_a=grouped,
        trans_b=not grouped,
        trans_c=False,
        group_lens=torch.tensor([2]),
        group_offs=torch.tensor([0, 2]),
        out_dtype=torch.float32,
        granularity=ScalingGranularity.MX_BLOCKWISE.value,
        num_cu=None,
        maybe_pre_sync=False,
        schedule="static",
        preshuffled=False,
        out=out,
        default_backend=BackendType.HIPBLASLT.value,
    )
    args = {name: values.get(name, p.default) for name, p in inspect.signature(op).parameters.items()}
    ordinary = {k: v for k, v in args.items() if k not in ("default_backend", "out")}
    ordinary["granularity"] = ScalingGranularity.MX_BLOCKWISE
    # A cached winner without beta=1 must not be reused for accumulation.
    dispatcher._cache.put(dispatcher.make_key(**ordinary), OrdinaryOnly)
    op(**args)
    torch.testing.assert_close(out, torch.full_like(out, 9.0))
    expected_profiling = mode in ("global", "per_op")
    assert bool(profile_outputs) == expected_profiling
    assert all(scratch.data_ptr() != out.data_ptr() for scratch in profile_outputs)
    count = len(profile_outputs)
    op(**args)
    torch.testing.assert_close(out, torch.full_like(out, 11.0))
    if not fail_profile or not expected_profiling:
        assert len(profile_outputs) == count
    else:
        assert len(profile_outputs) > count


@pytest.mark.parametrize(
    "first,second", [(BackendType.AITER, BackendType.FLYDSL), (BackendType.FLYDSL, BackendType.AITER)]
)
def test_fp4_shape_resolution_observes_changed_backend(monkeypatch, first, second):
    module = importlib.import_module("primus_turbo.pytorch.kernels.gemm.gemm_fp4_impl")
    monkeypatch.setattr(module, "_FP4_BACKEND_FOR_SHAPE", {})
    monkeypatch.setattr(module, "float4_e2m1fn_x2", torch.uint8)
    monkeypatch.setattr(GlobalBackendManager, "auto_tune_enabled", lambda: False)
    monkeypatch.setattr(
        module.GEMMFP4KernelDispatcher, "resolve", lambda default, user, **kwargs: user.backend
    )
    for backend in (first, second):
        monkeypatch.setattr(
            GlobalBackendManager,
            "get_gemm_backend",
            lambda _, backend=backend: BackendChoice(backend=backend),
        )
        selected = module.resolve_fp4_backend(
            128, 128, 256, out_dtype=torch.float32, device=torch.device("cpu")
        )
        assert selected is backend
