###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""All-gather-KV context-parallel attention against single-GPU attention over the whole sequence.

Every rank builds the same full-sequence q / k / v, runs the reference (one causal FlyDSL call), takes
its zigzag shard (chunks r and 2n-1-r of 2n) through ``flash_attn_agkv_func``, and compares the output
and gradients with the matching rows of the reference.

    pytest tests/pytorch/ops/test_attention_agkv.py --dist-only             # quick, all world sizes
    torchrun --nproc_per_node 8 tests/pytorch/ops/test_attention_agkv.py    # full
"""

import argparse
import itertools
import os
import subprocess
import sys
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist

from primus_turbo.pytorch.core.utils import is_gfx950

# name: (hq, hkv, d)
HEAD_CONFIGS = {
    "qwen3_30b_a3b": (32, 4, 128),
    "llama3_8b": (32, 8, 128),
    "gpt_oss_120b": (64, 8, 64),
}
# relative L2 error against one FlyDSL call over the whole sequence; splitting the keys over several
# parts changes the bf16 accumulation order, which at 32K puts dq of some ranks near 1e-2
TOL = {"out": 1e-2, "dq": 2e-2, "dk": 2e-2, "dv": 2e-2}


def _rel(x, y):
    return ((x.float() - y.float()).norm() / y.float().norm().clamp_min(1e-12)).item()


def _zigzag(t, n, r):
    c = t.shape[0] // (2 * n)
    return torch.cat((t[r * c : (r + 1) * c], t[(2 * n - 1 - r) * c : (2 * n - r) * c]))


def _reference(q, k, v, dout):
    from primus_turbo.pytorch.kernels.attention.attention_flydsl_impl import (
        flash_attn_sbhd_flydsl_backward_impl,
        flash_attn_sbhd_flydsl_forward_impl,
    )

    out, lse = flash_attn_sbhd_flydsl_forward_impl(q, k, v, causal=True, return_lse=True)
    S, B, Hq = q.shape[:3]
    dq, dk, dv = flash_attn_sbhd_flydsl_backward_impl(
        dout, q, k, v, out, lse.view(B, S, Hq).permute(0, 2, 1), causal=True
    )[:3]
    return out, dq, dk, dv


def run_case(seq, b, heads, group, layers, verbose):
    """``layers`` calls with different inputs, all forwards and then the backwards in reverse, as the
    layers of a model run: consecutive calls reuse the exchange buffers while earlier ones are saved."""
    from primus_turbo.pytorch.ops import flash_attn_agkv_func

    n, r = group.size(), group.rank()
    hq, hkv, d = HEAD_CONFIGS[heads]
    gen = torch.Generator(device="cuda").manual_seed(seq * 31 + b * 7 + hq + d)
    full = lambda h: torch.randn(seq, b, h, d, device="cuda", dtype=torch.bfloat16, generator=gen)
    cases = []
    for _ in range(layers):
        q, k, v, dout = full(hq), full(hkv), full(hkv), full(hq)
        ref = [_zigzag(t, n, r) for t in _reference(q, k, v, dout)]
        ql, kl, vl = (_zigzag(t, n, r).detach().requires_grad_() for t in (q, k, v))
        cases.append((ql, kl, vl, _zigzag(dout, n, r), ref))
    outs = [flash_attn_agkv_func(ql, kl, vl, group) for ql, kl, vl, _, _ in cases]
    for out, (_, _, _, dout, _) in reversed(list(zip(outs, cases))):
        out.backward(dout)

    failures = []
    for i, (out, (ql, kl, vl, _, ref)) in enumerate(zip(outs, cases)):
        errs = {name: _rel(x, y) for name, x, y in zip(TOL, (out, ql.grad, kl.grad, vl.grad), ref)}
        tag = f"seq={seq} b={b} {heads} layer {i}"
        if verbose:
            print(f"rank {r} {tag}: " + ", ".join(f"{k} {e:.1e}" for k, e in errs.items()), flush=True)
        bad = [f"{name} {e:.2e}" for name, e in errs.items() if not e < TOL[name]]
        if bad:
            failures.append(f"{tag}: " + ", ".join(bad))
    return failures


def run_suite(quick, verbose=False, seqs=None):
    group = dist.group.WORLD
    n, rank = group.size(), group.rank()
    seqs = seqs or ([4096] if quick else [4096, 16384, 32768])
    batches = [1] if quick else [1, 2]
    failures = []
    for seq, b, heads in itertools.product(seqs, batches, HEAD_CONFIGS):
        if seq % (2 * n * 128) or seq // (2 * n) < 256:
            continue
        failures += run_case(seq, b, heads, group, layers=3, verbose=verbose)
    gathered = [None] * n
    dist.all_gather_object(gathered, failures)
    if rank == 0:
        all_failures = [f"rank {r}: {f}" for r, fs in enumerate(gathered) for f in fs]
        print("\n".join(all_failures) if all_failures else f"agkv attention: all cases passed (n={n})")
    return not any(gathered)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--seqs", nargs="+", type=int)
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    dist.init_process_group("nccl", timeout=timedelta(minutes=5))
    ok = run_suite(args.quick, args.verbose, args.seqs)
    dist.barrier()
    dist.destroy_process_group()
    sys.exit(0 if ok else 1)


@pytest.mark.multigpu
@pytest.mark.skipif(
    not (torch.cuda.is_available() and is_gfx950()), reason="the FlyDSL attention kernels need gfx950"
)
@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_attention_agkv(world_size):
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"needs {world_size} GPUs")
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={world_size}",
        __file__,
        "--quick",
    ]
    env = {**os.environ, "PYTHONPATH": repo_root}
    result = subprocess.run(cmd, cwd=repo_root, env=env, timeout=1800)
    assert result.returncode == 0


if __name__ == "__main__":
    main()
