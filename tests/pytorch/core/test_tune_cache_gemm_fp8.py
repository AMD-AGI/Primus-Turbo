###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""End-to-end: offline tune -> dump -> reload -> dispatch hit, on the real fp8 GEMM."""

import pytest
import torch

from primus_turbo.pytorch.core.backend import GlobalBackendManager
from primus_turbo.pytorch.core.low_precision import (
    Float8QuantConfig,
    Format,
    ScalingGranularity,
)
from primus_turbo.pytorch.kernels.gemm.gemm_fp8_impl import GEMMFP8KernelDispatcher
from primus_turbo.pytorch.ops import gemm_fp8


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_offline_cache_end_to_end_gemm_fp8(tmp_path, monkeypatch):
    device = "cuda:0"
    torch.manual_seed(0)
    m, n, k = 512, 1024, 2048
    a = torch.randn(m, k, dtype=torch.bfloat16, device=device)   # NT: a[m,k], b[n,k]
    b = torch.randn(n, k, dtype=torch.bfloat16, device=device)
    cfg = Float8QuantConfig(granularity=ScalingGranularity.TENSORWISE, format=Format.E4M3)

    disp = GEMMFP8KernelDispatcher
    monkeypatch.delenv("PRIMUS_TURBO_GEMM_BACKEND", raising=False)
    monkeypatch.setattr(disp, "_tune_config_name", None)  # isolate: no auto-load of the packaged asset
    GlobalBackendManager.reset()
    disp._cache.clear()
    try:
        # 1. autotune on -> dispatcher profiles and caches the winning backend.
        GlobalBackendManager.set_auto_tune(True)
        out_tuned = gemm_fp8(a, b, False, True, torch.bfloat16, cfg)
        torch.cuda.synchronize()
        assert len(disp._cache) >= 1
        snapshot = disp._cache.items()

        # 2. dump -> clear -> load: the real key (m/n/k + fp8 dtypes + granularity + flags) round-trips.
        path = tmp_path / "gemm_fp8.json"
        disp.dump_cache(str(path))
        disp._cache.clear()
        assert len(disp._cache) == 0
        disp.load_cache(str(path))
        assert disp._cache.items() == snapshot

        # 3. autotune off + loaded cache -> dispatch hits the cached backend (spy on its execute).
        GlobalBackendManager.set_auto_tune(False)
        cached_impl = snapshot[0][1]
        hits = {"n": 0}
        orig_execute = cached_impl.execute

        def _spy(**kwargs):
            hits["n"] += 1
            return orig_execute(**kwargs)

        monkeypatch.setattr(cached_impl, "execute", staticmethod(_spy))
        out_cached = gemm_fp8(a, b, False, True, torch.bfloat16, cfg)
        torch.cuda.synchronize()

        assert hits["n"] >= 1  # cached backend was used (cache lookup, not profiling/fallback)
        torch.testing.assert_close(out_cached, out_tuned, rtol=1e-2, atol=1e-2)
    finally:
        GlobalBackendManager.reset()
        disp._cache.clear()
