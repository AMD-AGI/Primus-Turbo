"""Exit ops (A2AGemm o_proj fwd, GemmA2A dO + dW_o bwd): correctness against the fp32 reference, then
fused op time (median of 30, max over ranks), per shape; one JSON line per shape on rank 0.

    torchrun --nproc_per_node 8 quick_test_bench.py --shapes deepseek-v3:16384 deepseek-v3:32768 ...
"""

import argparse
import json
import os
import sys
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn.functional as F

REPO = os.environ.get(
    "TURBO_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..")),
)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "benchmark", "ops", "training"))
from bench_ulysses_overlap import MODELS, timeit
from primus_turbo.pytorch.ops.attention.ulysses_fused_a2a import (
    _o_gemm_backward,
    _o_gemm_forward,
)
from tests.pytorch.ops.test_ulysses_fused_a2a import _close, reference_a2a_o_gemm


def run_shape(model, seq, group, unfused):
    n, rank = group.size(), group.rank()
    cfg = MODELS[model]
    hidden, hq, d = cfg["hidden"], cfg["hq"], cfg["d"]
    s_local, K = seq // n, hq * d
    gen = torch.Generator(device="cuda").manual_seed(1000 + rank)
    o = torch.randn(seq, 1, hq // n, d, device="cuda", generator=gen).bfloat16()
    w = (
        torch.randn(hidden, K, device="cuda", generator=torch.Generator(device="cuda").manual_seed(7)) * 0.02
    ).bfloat16()
    res = torch.randn(s_local, 1, hidden, device="cuda", generator=gen).bfloat16()
    dy = torch.randn(s_local, 1, hidden, device="cuda", generator=gen).bfloat16()
    dy2d = dy.view(s_local, hidden)

    def fwd():
        return _o_gemm_forward(o, w, res, group, "sbhd", True)

    y, og = fwd()

    def bwd():
        return _o_gemm_backward(dy2d, w, og, group, hq // n, d, s_local, 1, "sbhd", True, True)

    do, dw = bwd()
    failures = []
    o_ref, w_ref, res_ref = (t.detach().float().requires_grad_() for t in (o, w, res))
    y_ref = reference_a2a_o_gemm(o_ref, w_ref, res_ref, group, "sbhd")
    y_ref.backward(dy.float())
    tag = f"[{model}:{seq} n={n}]"
    _close("y", y, y_ref, failures, tag)
    _close("do", do, o_ref.grad, failures, tag)
    _close("dw_o", dw, w_ref.grad, failures, tag)
    del o_ref, w_ref, res_ref, y_ref
    torch.cuda.empty_cache()
    out = {"shape": f"{model}:{seq}", "n": n, "fwd_us": timeit(fwd, group), "bwd_us": timeit(bwd, group)}
    if unfused:
        from primus_turbo.pytorch.ops.attention.usp.attention_a2a_helper import get_attention_cp_a2a_helper
        from torch.distributed import all_to_all_single

        helper = get_attention_cp_a2a_helper(1, seq, hq, hq, d, d, 1, n)

        def a2a(t):
            r = torch.empty_like(t)
            all_to_all_single(r, t, group=group)
            return r

        def fwd_u():
            g = helper.reshape_o_after_a2a(a2a(helper.reshape_o_before_a2a(o.view(1, seq, hq // n, d))))
            return F.linear(g.reshape(s_local, K), w) + res.view(s_local, hidden)

        ogv = og.view(s_local, K)

        def bwd_u():
            g = F.linear(dy2d, w.t()).view(1, s_local, hq, d)
            return helper.reshape_do_after_a2a(a2a(helper.reshape_do_before_a2a(g))), dy2d.t() @ ogv

        out["fwd_unfused_us"], out["bwd_unfused_us"] = timeit(fwd_u, group), timeit(bwd_u, group)
    gathered = [None] * n
    dist.all_gather_object(gathered, failures, group=group)
    out["check"] = (
        "PASS" if not any(gathered) else "FAIL: " + "; ".join(f for fs in gathered for f in fs)[:400]
    )
    del o, w, res, dy, y, og, do, dw
    torch.cuda.empty_cache()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shapes", nargs="+", required=True)
    p.add_argument("--unfused", action="store_true")
    args = p.parse_args()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    dist.init_process_group("nccl", timeout=timedelta(minutes=10))
    group = dist.group.WORLD
    for s in args.shapes:
        model, seq = s.split(":")
        r = run_shape(model, int(seq), group, args.unfused)
        if group.rank() == 0:
            print("RESULT " + json.dumps(r), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
