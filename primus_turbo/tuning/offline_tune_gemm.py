###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Offline autotune driver for the dense GEMM family.

The GEMM family shares one shape axis ``(m, n, k)``. This driver takes a shape
list and tunes each GEMM precision in turn (each enumerates its own
dtype / format / granularity), dumping one JSON per dispatcher under
``tuning/configs/<framework>/<arch>/``.

    python -m primus_turbo.tuning.offline_tune_gemm [--shapes s.json]

``shapes.json``: ``{"mnk": [[m, n, k], ...]}``. Omit for a small default grid.

Add a precision by filling in its ``tune_gemm_*`` and calling it from ``run``.
"""

import argparse
import itertools
import json
import logging
import os
import time

import torch

from primus_turbo.common.logger import logger
from primus_turbo.pytorch.core.backend import GlobalBackendManager, TuneCache
from primus_turbo.pytorch.core.low_precision import (
    Float8QuantConfig,
    Format,
    ScaleDtype,
    ScalingGranularity,
)
from primus_turbo.pytorch.kernels.gemm.gemm_fp8_impl import GEMMFP8KernelDispatcher
from primus_turbo.pytorch.ops import gemm_fp8

# (m, n, k); n/k are per-weight (fixed), m is the varying token dim.
_DEFAULT_MNK = [(m, 4096, 4096) for m in (16, 64, 256, 1024)]


def _arch_tag() -> str:
    """e.g. 'gfx950' / 'gfx942'."""
    return torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]


def _dump(dispatcher, out_dir, fname) -> int:
    os.makedirs(out_dir, exist_ok=True)
    n = dispatcher.dump_cache(os.path.join(out_dir, fname))
    logger.info(f"[offline_tune_gemm] {fname}: {n} entries")
    return n


def tune_gemm_fp8(mnk_list, out_dir, device="cuda:0"):
    """fp8 GEMM (NT layout): tune dtype x format x granularity over the shapes,
    fwd + bwd, dump gemm_fp8.json."""
    dtypes = (torch.bfloat16, torch.float16)
    fmts = (Format.E4M3, Format.E5M2, Format.HYBRID)
    grans = (
        ScalingGranularity.TENSORWISE,
        ScalingGranularity.ROWWISE,
        # ScalingGranularity.BLOCKWISE,  # TODO: slow (per-M triton 144-config autotune); prune candidates first
        ScalingGranularity.MX_BLOCKWISE,
    )
    combos = list(itertools.product(dtypes, fmts, grans))
    # Offline tuning must retain every tuned key until dump. Each fwd+bwd emits up to
    # 3 dispatch keys (1 forward gemm + 2 backward grad gemms), so size the LRU to the
    # full grid x 3 (upper bound on distinct keys) and never evict.
    capacity = 3 * len(combos) * len(mnk_list)
    GEMMFP8KernelDispatcher._cache = TuneCache(capacity=max(capacity, 1024))
    for i, (dtype, fmt, gran) in enumerate(combos, 1):
        if gran == ScalingGranularity.BLOCKWISE:
            cfg = Float8QuantConfig(granularity=gran, format=fmt, block_size=128)
        elif gran == ScalingGranularity.MX_BLOCKWISE:
            cfg = Float8QuantConfig(granularity=gran, format=fmt, block_size=32, scale_dtype=ScaleDtype.E8M0)
        else:
            cfg = Float8QuantConfig(granularity=gran, format=fmt)
        logger.info(f"[gemm_fp8] ({i}/{len(combos)}) {dtype} {fmt.name}/{gran.name} x {len(mnk_list)} shapes")
        for m, n, k in mnk_list:
            # NT: a is [m, k], b is [n, k]; both need grad so bwd gemms are tuned too.
            a = torch.randn(m, k, dtype=dtype, device=device, requires_grad=True)
            b = torch.randn(n, k, dtype=dtype, device=device, requires_grad=True)
            try:
                out = gemm_fp8(a, b, False, True, dtype, cfg)
                out.backward(torch.randn_like(out))
                torch.cuda.synchronize()
            except Exception as e:  # unsupported combo on this arch -> skip, keep sweeping
                logger.warning(f"[offline_tune_gemm] skip fp8 {fmt.name}/{gran.name}: {e}", once=True)
    torch.cuda.synchronize()
    return _dump(GEMMFP8KernelDispatcher, out_dir, "gemm_fp8.json")


def run(mnk_list, out_dir):
    GlobalBackendManager.reset()
    GlobalBackendManager.set_auto_tune(True)
    try:
        tune_gemm_fp8(mnk_list, out_dir)
        # TODO: tune_gemm (bf16/fp16 dense), tune_gemm_fp4
    finally:
        GlobalBackendManager.set_auto_tune(None)
        GlobalBackendManager.reset()


def main():
    p = argparse.ArgumentParser(description="Offline autotune for the GEMM family.")
    p.add_argument("--shapes", default=None, help="JSON: {'mnk': [[m,n,k], ...]}")
    args = p.parse_args()

    logger.set_level(logging.INFO)  # progress logs are INFO; default level would hide them

    mnk = [tuple(x) for x in json.load(open(args.shapes))["mnk"]] if args.shapes else _DEFAULT_MNK
    # Always the canonical packaged path the runtime auto-loads from (no override on purpose).
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "pytorch", _arch_tag())
    t0 = time.perf_counter()
    run(mnk, out_dir)
    logger.info(f"[offline_tune_gemm] done -> {out_dir} ({time.perf_counter() - t0:.1f}s)")


if __name__ == "__main__":
    main()
