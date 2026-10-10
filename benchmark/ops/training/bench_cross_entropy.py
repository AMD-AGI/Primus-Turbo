# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.

"""Time full-vocabulary CE forward/backward, optionally against installed TE.

GPT-OSS MLPerf shape: --sequence 8192 --batch 4 --vocab 128256 --compare-te
Input restoration is outside the timed region because either backend may
reuse input storage. These operator timings do not measure training throughput.
"""

import argparse
import importlib.metadata
import json
import statistics

import torch

from primus_turbo.pytorch.ops.cross_entropy import cross_entropy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=int, default=4096)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--vocab", type=int, default=128256)
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--compare-te", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    if min(args.sequence, args.batch, args.vocab, args.repetitions) <= 0 or args.warmup < 0:
        parser.error("shapes and repetitions must be positive; warmup must be nonnegative")
    torch.manual_seed(30279)
    shape = (args.sequence, args.batch, args.vocab)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    source = torch.randn(shape, device="cuda", dtype=dtype)
    target = torch.randint(args.vocab, shape[:-1], device="cuda")
    grad = torch.full(shape[:-1], 1.0 / target.numel(), device="cuda")
    work = torch.empty_like(source)
    variants = {
        "turbo_copy": lambda x: cross_entropy(x, target),
        "turbo_overwrite": lambda x: cross_entropy(x, target, overwrite_input=True),
    }
    versions = {"torch": torch.__version__, "triton": importlib.metadata.version("triton")}
    if args.compare_te:
        import transformer_engine
        from transformer_engine.pytorch import parallel_cross_entropy

        variants = {"installed_te": lambda x: parallel_cross_entropy(x, target), **variants}
        versions["transformer_engine"] = transformer_engine.__version__
    results = {}
    for name, op in variants.items():
        forward_ms, backward_ms = [], []
        for i in range(args.warmup + args.repetitions):
            work.copy_(source)
            logits = work.detach().requires_grad_()
            start, middle, end = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
            start.record()
            loss = op(logits)
            middle.record()
            torch.autograd.grad(loss, logits, grad)
            end.record()
            end.synchronize()
            if i >= args.warmup:
                forward_ms.append(start.elapsed_time(middle))
                backward_ms.append(middle.elapsed_time(end))
        totals = [f + b for f, b in zip(forward_ms, backward_ms)]
        results[name] = {
            "forward_median_ms": statistics.median(forward_ms),
            "backward_median_ms": statistics.median(backward_ms),
            "total_median_ms": statistics.median(totals),
            "total_min_ms": min(totals),
            "total_max_ms": max(totals),
            "total_samples_ms": totals,
        }
    report = {
        "gpu": torch.cuda.get_device_name(),
        "shape": shape,
        "dtype": args.dtype,
        "warmup": args.warmup,
        "repetitions": args.repetitions,
        "versions": versions,
        "results": results,
        "note": "Operator-only timing; input restoration excluded; no E2E speedup implied.",
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        with open(args.output, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
