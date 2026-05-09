#!/usr/bin/env python3
"""Round-21 — Direction G probe: cross-shape num_slots co-optimization.

Sister probe to ``_probe_round_20_cross_shape_chunk_size.py``. R20 closed
the chunk_size axis as FALSIFIED (3-sample sweep on GPU 3 cs ∈ {16,24,
32,48} → median 684/686/692/685 vs baseline 697; -5 to -13 score).

This probe sweeps ``num_slots`` global override on cells where
``cfg.num_slots == 0`` (kernel default), only for FP8 RCR/RRR layouts.
``num_slots`` controls the chunked-partition slot count per CTA in the
persistent grid; reducing it can lower scheduling overhead on
under-saturated cells, but at the risk of smaller chunked-partitions
(less L2 reuse). Together with R20, these two axes exhaust the
"single-knob global override" sub-direction of direction G.

Env:
  PROBE_GPT_OSS_SLOTS_OVERRIDE=N — int num_slots to inject (skip cells
                                  where cfg.num_slots != 0).

Run via dbg_remote.sh.
"""
from __future__ import annotations

import dataclasses
import os
import sys

_WS = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _WS not in sys.path:
    sys.path.insert(0, _WS)

OVERRIDE = int(os.environ.get("PROBE_GPT_OSS_SLOTS_OVERRIDE", "-1"))

if OVERRIDE >= 0:
    from primus_turbo.pytorch.kernels import hipkitten as hk_mod
    from primus_turbo.pytorch.kernels.hipkitten import config as hk_cfg

    _orig_select = hk_cfg.select_default_config
    try:
        _orig_select.cache_clear()
    except AttributeError:
        pass

    def _patched_select(*args, **kwargs):
        cfg = _orig_select(*args, **kwargs)
        if cfg.num_slots == 0 and cfg.layout in ("rcr", "rrr"):
            return dataclasses.replace(cfg, num_slots=OVERRIDE)
        return cfg

    hk_cfg.select_default_config = _patched_select
    hk_mod.select_default_config = _patched_select
    import primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_fp8_impl as fp8_disp
    if hasattr(fp8_disp, "hipkitten"):
        fp8_disp.hipkitten.select_default_config = _patched_select
    print(
        f"[probe_round_21] override active: num_slots={OVERRIDE} on cells with cfg.num_slots==0 (RCR/RRR only)",
        file=sys.stderr,
    )

_HERE = os.path.abspath(os.path.dirname(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import _metric_gpt_oss_fp8_kernel as metric  # noqa: E402

raise SystemExit(metric._run())
