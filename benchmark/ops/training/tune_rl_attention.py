###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Offline schedule tuning for the batch-invariant RL attention (TRITON and FLYDSL backends).

For each setting (phase, model heads, batch, context, query length) and backend, searches the
schedule kernel options (MODE, BLOCK_M, num_warps, num_stages, waves_per_eu, SPLIT_SEGS, PACK) and
prints the fastest ``kernel_options`` whose output is bitwise equal to the default schedule's.
Options that change bits (TILE_SIZE, SEGMENT_SIZE) stay at --tile-size / --segment-size, which
must match the training process. Training settings tune the forward (the backward uses a fixed
configuration).

    python benchmark/ops/training/tune_rl_attention.py [--phase decode prefill train]
        [--model llama3-8b] [--backend TRITON FLYDSL] [--case BATCH,CTX[,Q_LEN] ...]
        [--tp-size N] [--exhaustive] [-v] [-o tuned.json]

The JSON output lists each setting with its tuned options; pass them per call as
``kernel_options`` (or process-wide as PRIMUS_TURBO_RL_ATTN_KERNEL_OPTIONS when one setting
dominates).
"""

import argparse
import json

import torch
from bench_rl_attention import (
    _DTYPES,
    DECODE_CASES,
    MODELS,
    PREFILL_CASES,
    TRAIN_CASES,
    heads,
    paged_inputs,
)
from tabulate import tabulate

from primus_turbo.pytorch.ops.attention import (
    rl_flash_attn_varlen_func,
    rl_paged_attention_func,
)
from primus_turbo.pytorch.ops.attention.rl_attention_tune import (
    format_options,
    search_space,
    tune,
)


def _rollout_run(batch, ctx, q_len, shape, dtype, fixed):
    hq, hkv, d, window, has_sinks = shape
    sinks = torch.randn(hq, device="cuda") if has_sinks else None
    q, k_cache, v_cache, block_table, seq_lens, cu_q = paged_inputs(batch, ctx, q_len, hq, hkv, d, dtype)

    def run(options):
        return rl_paged_attention_func(
            q, k_cache, v_cache, cu_q, seq_lens, block_table, q_len, ctx, d**-0.5, window, sinks=sinks,
            kernel_options={**fixed, **options},
        )  # fmt: skip

    return run


def _train_run(num_seqs, seqlen, shape, dtype, fixed):
    hq, hkv, d, window, has_sinks = shape
    total = num_seqs * seqlen
    q = torch.randn(total, hq, d, dtype=dtype, device="cuda")
    k = torch.randn(total, hkv, d, dtype=dtype, device="cuda")
    v = torch.randn(total, hkv, d, dtype=dtype, device="cuda")
    sinks = torch.randn(hq, device="cuda") if has_sinks else None
    cu = torch.arange(0, (num_seqs + 1) * seqlen, seqlen, dtype=torch.int32, device="cuda")

    @torch.no_grad()
    def run(options):
        return rl_flash_attn_varlen_func(
            q, k, v, cu, cu, seqlen, seqlen, d**-0.5, window, sinks=sinks, kernel_options={**fixed, **options}
        )

    return run


def _settings(phase, cases):
    if cases:
        return [(b, c, q if phase != "decode" else 1) for b, c, q in cases]
    if phase == "decode":
        return [(b, c, 1) for b, c in DECODE_CASES]
    if phase == "prefill":
        return list(PREFILL_CASES)
    return [(n, s, s) for n, s in TRAIN_CASES]


def _parse_case(text):
    parts = [int(x) for x in text.split(",")]
    if len(parts) not in (2, 3):
        raise argparse.ArgumentTypeError(f"expected BATCH,CTX[,Q_LEN], got {text!r}")
    return parts[0], parts[1], parts[2] if len(parts) == 3 else parts[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Tune the RL attention schedule kernel options")
    parser.add_argument("--phase", nargs="+", default=["decode"], choices=["decode", "prefill", "train"])
    parser.add_argument("--model", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--backend", nargs="+", default=["TRITON", "FLYDSL"], choices=["TRITON", "FLYDSL"])
    parser.add_argument(
        "--case", nargs="+", type=_parse_case, help="BATCH,CTX[,Q_LEN] (train: NUM_SEQS,SEQLEN) instead of the defaults"
    )
    parser.add_argument("--tp-size", type=int, default=1, help="shard attention heads over this many ranks")
    parser.add_argument("--dtype", default="bf16", choices=list(_DTYPES))
    parser.add_argument("--tile-size", type=int, default=0, help="TILE_SIZE (changes bits; 0 = default)")
    parser.add_argument("--segment-size", type=int, default=0, help="SEGMENT_SIZE (changes bits; 0 = default)")
    parser.add_argument("--exhaustive", action="store_true", help="time the full grid, not coordinate descent")
    parser.add_argument("--min-gain", type=float, default=0.03, help="relative gain needed to leave a default")
    parser.add_argument("-v", "--verbose", action="store_true", help="print every candidate's time")
    parser.add_argument("-o", "--output", default="rl_attention_tuned.json")
    args = parser.parse_args(argv)

    dtype = _DTYPES[args.dtype]
    print(f"{torch.cuda.get_device_name(0)}, {args.dtype}, tp {args.tp_size}")
    rows, entries = [], []
    for phase in args.phase:
        for model in args.model:
            shape = heads(model, args.tp_size)
            for batch, ctx, q_len in _settings(phase, args.case):
                for backend in args.backend:
                    fixed = {"BACKEND": backend}
                    if args.tile_size:
                        fixed["TILE_SIZE"] = args.tile_size
                    if args.segment_size:
                        fixed["SEGMENT_SIZE"] = args.segment_size
                    if phase == "train":
                        run = _train_run(batch, ctx, shape, dtype, fixed)
                    else:
                        run = _rollout_run(batch, ctx, q_len, shape, dtype, fixed)
                    label = f"{phase} {model} {batch}x{ctx}" + (f" q{q_len}" if phase == "prefill" else "")
                    print(f"{label} {backend}", flush=True)
                    res = tune(
                        run, search_space(backend, q_len), min_gain=args.min_gain, exhaustive=args.exhaustive,
                        log=print if args.verbose else None,
                    )  # fmt: skip
                    for options, why in res.rejected:
                        if why.startswith("bits"):
                            print(f"  WARNING {format_options(options)}: {why}")
                    rows.append(
                        {
                            "phase": phase, "model": model, "batch": batch, "ctx": ctx, "q_len": q_len,
                            "backend": backend, "default_us": round(res.default_us, 1),
                            "tuned_us": round(res.best_us, 1), "speedup": round(res.speedup, 2),
                            "tried": res.tried, "kernel_options": format_options(res.options) or "default",
                        }
                    )  # fmt: skip
                    head_q, head_kv, head_dim, window, has_sinks = shape
                    entries.append(
                        {
                            "phase": phase, "model": model, "num_q_heads": head_q, "num_kv_heads": head_kv,
                            "head_dim": head_dim, "window_left": window, "sinks": has_sinks,
                            "dtype": args.dtype, "batch": batch, "ctx": ctx, "q_len": q_len,
                            "default_us": res.default_us, "tuned_us": res.best_us,
                            "kernel_options": {**fixed, **res.options},
                        }
                    )  # fmt: skip
    print()
    print(tabulate(rows, headers="keys", tablefmt="github"))
    with open(args.output, "w") as f:
        json.dump({"device": torch.cuda.get_device_name(0), "settings": entries}, f, indent=1)
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
