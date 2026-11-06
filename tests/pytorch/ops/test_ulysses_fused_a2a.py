###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Ulysses GemmA2A / A2AGemm fused projections against the unfused reference.

Reference: fp32 ``F.linear`` -> Turbo's ``AttentionCPA2AHelper`` packing -> ``all_to_all_single``
-> unpacking, i.e. the ``flash_attn_usp_func`` sequence order.

The pytest entry launches this file under torchrun for every world size the node has GPUs for:

    pytest tests/pytorch/ops/test_ulysses_fused_a2a.py --dist-only
    torchrun --nproc_per_node 8 tests/pytorch/ops/test_ulysses_fused_a2a.py
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
import torch.nn.functional as F

# (hq, hkv) as multiples of n: GQA 32/8, MHA 40/40, hkv == n (Qwen3 64/4 at n = 4)
HEAD_CONFIGS = {
    "gqa": lambda n: (32, 8),
    "mha": lambda n: (40, 40),
    "hkv_eq_n": lambda n: (16 * n, n),
}


def _split_qkv_reference(qkv, hq, hkv, d, weight_layout):
    lead = qkv.shape[:2]
    if weight_layout == "interleaved":
        qkv = qkv.view(*lead, hkv, hq // hkv + 2, d)
        q = qkv[..., : hq // hkv, :].reshape(*lead, hq, d)
        return q, qkv[..., hq // hkv, :], qkv[..., hq // hkv + 1, :]
    q, k, v = torch.split(qkv, [hq * d, hkv * d, hkv * d], dim=-1)
    return q.reshape(*lead, hq, d), k.reshape(*lead, hkv, d), v.reshape(*lead, hkv, d)


def _helper(x, hq, hkv, d, n, qkv_format):
    from primus_turbo.pytorch.ops.attention.usp.attention_a2a_helper import (
        get_attention_cp_a2a_helper,
    )

    if qkv_format == "sbhd":
        s_local, b = x.shape[:2]
    else:
        b, s_local = x.shape[:2]
    return get_attention_cp_a2a_helper(b, s_local * n, hq, hkv, d, d, 0 if qkv_format == "sbhd" else 1, n)


def reference_qkv_gemm_a2a(x, w_qkv, hq, hkv, d, group, qkv_format, weight_layout):
    n = group.size()
    helper = _helper(x, hq, hkv, d, n, qkv_format)
    qkv = F.linear(x.float(), w_qkv.float())
    q, k, v = _split_qkv_reference(qkv, hq, hkv, d, weight_layout)
    packed = helper.combine_qkv_before_a2a(q, k, v)
    out = torch.empty_like(packed)
    dist.all_to_all_single(out, packed, group=group)
    return helper.splits_qkv_after_a2a(out)


def _close(name, out, ref, failures, tag):
    from tests.pytorch.test_utils import get_tolerances

    tol = get_tolerances(torch.bfloat16)
    if out.shape != ref.shape:
        failures.append(f"{tag} {name}: shape {tuple(out.shape)} != {tuple(ref.shape)}")
        return
    if not torch.allclose(out.float(), ref.float(), **tol):
        err = (out.float() - ref.float()).abs().max().item()
        failures.append(f"{tag} {name}: max abs err {err:.4g}")


def check_qkv_forward(group, hq, hkv, d, b, s_local, hidden, qkv_format, weight_layout, repeats, failures):
    from primus_turbo.pytorch.ops.attention.ulysses_fused_a2a import ulysses_qkv_gemm_a2a

    rank = group.rank()
    gen = torch.Generator(device="cuda").manual_seed(1234)
    w_qkv = torch.randn((hq + 2 * hkv) * d, hidden, device="cuda", generator=gen).mul_(0.02).bfloat16()
    tag = f"[qkv fwd hq={hq} hkv={hkv} d={d} b={b} s/n={s_local} {qkv_format} {weight_layout}]"
    for it in range(repeats):
        gen_x = torch.Generator(device="cuda").manual_seed(100 * it + rank)
        shape = (s_local, b, hidden) if qkv_format == "sbhd" else (b, s_local, hidden)
        x = torch.randn(*shape, device="cuda", generator=gen_x).bfloat16()
        outs = ulysses_qkv_gemm_a2a(x, w_qkv, hq, hkv, d, group, qkv_format, weight_layout)
        refs = reference_qkv_gemm_a2a(x, w_qkv, hq, hkv, d, group, qkv_format, weight_layout)
        for name, out, ref in zip("qkv", outs, refs):
            _close(name, out, ref, failures, f"{tag} call {it}")


def run_suite(quick):
    group = dist.group.WORLD
    n = group.size()
    rank = group.rank()
    failures = []
    head_names = ["gqa"] if quick else list(HEAD_CONFIGS)
    dims = [128] if quick else [64, 128]
    batches = [1] if quick else [1, 2]
    formats = ["sbhd"] if quick else ["sbhd", "bshd"]
    layouts = ["interleaved"] if quick else ["interleaved", "concat"]
    # s/n = 300 is not a multiple of any tile
    for head_name, d, b, qkv_format, weight_layout in itertools.product(
        head_names, dims, batches, formats, layouts
    ):
        hq, hkv = HEAD_CONFIGS[head_name](n)
        if hq % n or hkv % n:
            continue
        check_qkv_forward(group, hq, hkv, d, b, 300, 1024, qkv_format, weight_layout, 3, failures)

    gathered = [None] * n
    dist.all_gather_object(gathered, failures, group=group)
    if rank == 0:
        all_failures = [f"rank {r}: {f}" for r, fs in enumerate(gathered) for f in fs]
        print("\n".join(all_failures) if all_failures else f"ulysses fused a2a: all cases passed (n={n})")
    return not any(gathered)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    dist.init_process_group("nccl", timeout=timedelta(minutes=5))
    ok = run_suite(args.quick)
    dist.barrier()
    dist.destroy_process_group()
    sys.exit(0 if ok else 1)


@pytest.mark.multigpu
@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_ulysses_fused_a2a(world_size):
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"needs {world_size} GPUs")
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    cmd = [sys.executable, "-m", "torch.distributed.run", f"--nproc_per_node={world_size}", __file__]
    result = subprocess.run(cmd, cwd=repo_root, env={**os.environ, "PYTHONPATH": repo_root}, timeout=1800)
    assert result.returncode == 0


if __name__ == "__main__":
    main()
