"""GQA Ulysses ops (fwd QKV GemmA2A, fwd o_proj A2AGemm, bwd o_proj GemmA2A + wgrad, bwd QKV A2AGemm +
wgrad): correctness of both fused autograd ops against the fp32 reference at the model's shape, then per op
the fused time and its floor max(hipBLASLt GEMM + wgrad, RCCL all-to-all), all median of 30, max over
ranks; one JSON line per shape on rank 0.

    torchrun --nproc_per_node 4 gqa_quick_bench.py --shapes glm-4.7:32768 qwen3-30b-a3b:32768 [--unfused]
"""

import argparse
import json
import os
import sys
from datetime import timedelta

import torch
import torch.distributed as dist

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("TURBO_ROOT", os.path.abspath(os.path.join(HERE, "..", "..", "..", "..")))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "benchmark", "ops", "training"))
from bench_ulysses_overlap import MODELS, a2a_fn, build_ops, timeit
from tests.pytorch.ops.test_ulysses_fused_a2a import check_o_gemm, check_qkv

OPS = ("fwd QKV (GemmA2A)", "fwd o_proj (A2AGemm)", "bwd o_proj (GemmA2A+wgrad)", "bwd QKV (A2AGemm+wgrad)")
KEYS = ("fwd_qkv", "fwd_o", "bwd_o", "bwd_qkv")


def run_shape(model, seq, group, unfused, check):
    n = group.size()
    cfg = MODELS[model]
    hidden, hq, hkv, d = cfg["hidden"], cfg["hq"], cfg["hkv"], cfg["d"]
    assert hkv % n == 0 and seq % n == 0, f"{model}: {hkv} KV heads or seq {seq} not divisible by {n}"
    failures = []
    if check:
        check_qkv(group, hq, hkv, d, 1, seq // n, hidden, "bshd", "interleaved", 1, failures)
        check_o_gemm(group, hq, d, 1, seq // n, hidden, "bshd", True, 1, failures)
        torch.cuda.empty_cache()
    ops = build_ops(cfg, seq, 1, group)
    out = {"shape": f"{model}:{seq}", "n": n}
    for key, name in zip(KEYS, OPS):
        gemm, wgrad, numel, _pack, unfused_fn, fused, _flops = ops[name]
        g = timeit(gemm, group)
        w = timeit(wgrad, group) if wgrad else 0.0
        a = timeit(a2a_fn(numel, group), group)
        out[key] = timeit(fused, group)
        out[key + "_floor"] = max(g + w, a)
        out[key + "_gemm"], out[key + "_wgrad"], out[key + "_a2a"] = g, w, a
        if unfused:
            out[key + "_unfused"] = timeit(unfused_fn, group)
    out["step_us"] = sum(out[k] for k in KEYS)
    out["floor_us"] = sum(out[k + "_floor"] for k in KEYS)
    gathered = [None] * n
    dist.all_gather_object(gathered, failures, group=group)
    if not check:
        out["check"] = "SKIPPED"
    else:
        out["check"] = (
            "PASS" if not any(gathered) else "FAIL: " + "; ".join(f for fs in gathered for f in fs)[:400]
        )
    del ops
    torch.cuda.empty_cache()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--shapes", nargs="+", required=True, help="model:seq, model from bench_ulysses_overlap.MODELS"
    )
    p.add_argument(
        "--unfused", action="store_true", help="also time the unfused chain (pack + RCCL + hipBLASLt)"
    )
    p.add_argument("--no-check", action="store_true", help="skip the fp32 reference check (timing only)")
    args = p.parse_args()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    dist.init_process_group("nccl", timeout=timedelta(minutes=30))
    group = dist.group.WORLD
    for s in args.shapes:
        model, seq = s.split(":")
        r = run_shape(model, int(seq), group, args.unfused, not args.no_check)
        if group.rank() == 0:
            print("RESULT " + json.dumps(r), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
