###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fused Ulysses all-to-all + causal attention + all-to-all forward against the unfused reference.

Every rank draws the same global q / k / v, takes its tokens under the striped split and runs the
fused kernel; the reference is the production SBHD FlyDSL forward on the global tensors. The pulled
q / k / v must match bitwise; O / LSE (this rank's heads) and the gathered output (this rank's
tokens, every head) are compared bitwise and with a tolerance.

    torchrun --nproc_per_node 4 tests/pytorch/ops/test_mega_attn_a2a.py
    pytest tests/pytorch/ops/test_mega_attn_a2a.py
"""

import argparse
import os
import subprocess
import sys
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist

from primus_turbo.pytorch.core.utils import is_gfx950


def local_rows(M, n, rank, cs, device):
    """Global sequence positions of ``rank``'s tokens under the striped split."""
    t = torch.arange(M, device=device)
    return (t // cs * n + rank) * cs + t % cs


def _report(name, out, ref, failures, tag, bitwise):
    diff = (out.float() - ref.float()).abs().max().item()
    same = torch.equal(out, ref)
    ok = same or (not bitwise and diff <= 2e-2 * max(ref.float().abs().max().item(), 1.0))
    if not ok:
        failures.append(f"{tag} {name}: max|diff| {diff:.3e} bitwise={same}")
    return same, diff


def check_forward(group, hq, hkv, d, s_local, seq_chunk, repeats, failures):
    from primus_turbo.flydsl.mega_attn.a2a_attn_kernel import (
        a2a_attn_fwd_bank_bytes,
        ulysses_a2a_attn_fwd,
    )
    from primus_turbo.flydsl.ulysses.workspace import get_ulysses_workspace
    from primus_turbo.pytorch.kernels.attention.attention_flydsl_impl import (
        flash_attn_sbhd_flydsl_forward_impl,
    )

    n, rank = group.size(), group.rank()
    dev = torch.device("cuda")
    S, M = s_local * n, s_local
    cs = seq_chunk if seq_chunk else M
    hq_r, hkv_r = hq // n, hkv // n
    tag = f"[n={n} hq={hq} hkv={hkv} d={d} M={M} cs={cs}]"
    gen = torch.Generator(device=dev).manual_seed(1234)
    qg = torch.randn(S, hq, d, device=dev, dtype=torch.bfloat16, generator=gen)
    kg = torch.randn(S, hkv, d, device=dev, dtype=torch.bfloat16, generator=gen)
    vg = torch.randn(S, hkv, d, device=dev, dtype=torch.bfloat16, generator=gen)
    rows = local_rows(M, n, rank, cs, dev)
    q_in, k_in, v_in = (t[rows].contiguous() for t in (qg, kg, vg))
    heads_q = slice(rank * hq_r, (rank + 1) * hq_r)
    heads_kv = slice(rank * hkv_r, (rank + 1) * hkv_r)

    o_all, _ = flash_attn_sbhd_flydsl_forward_impl(
        qg[:, None], kg[:, None], vg[:, None], causal=True, return_lse=True
    )
    q_ref = qg[:, heads_q].contiguous()
    k_ref = kg[:, heads_kv].contiguous()
    v_ref = vg[:, heads_kv].contiguous()
    o_ref, lse_ref = flash_attn_sbhd_flydsl_forward_impl(
        q_ref[:, None], k_ref[:, None], v_ref[:, None], causal=True, return_lse=True
    )
    out_ref = o_all[:, 0][rows]

    ws = get_ulysses_workspace(group, a2a_attn_fwd_bank_bytes(M, n, hq_r, hkv_r, d), name="mega_attn")
    for it in range(repeats):
        q = torch.full((S, hq_r, d), float("nan"), device=dev, dtype=torch.bfloat16)
        k = torch.full((S, hkv_r, d), float("nan"), device=dev, dtype=torch.bfloat16)
        v = torch.full_like(k, float("nan"))
        o = torch.full_like(q, float("nan"))
        lse = torch.full((S, hq_r), float("nan"), device=dev, dtype=torch.float32)
        out = torch.full_like(q_in, float("nan"))
        ulysses_a2a_attn_fwd(q_in, k_in, v_in, q, k, v, o, lse, out, ws, seq_chunk=seq_chunk)
        torch.cuda.synchronize()
        t = f"{tag} it={it}"
        _report("q", q, q_ref, failures, t, True)
        _report("k", k, k_ref, failures, t, True)
        _report("v", v, v_ref, failures, t, True)
        so, do = _report("o", o, o_ref[:, 0], failures, t, False)
        sl, dl = _report("lse", lse, lse_ref, failures, t, False)
        sg, dg = _report("out", out, out_ref, failures, t, False)
        if rank == 0:
            print(
                f"{t} o bitwise={so} {do:.2e} | lse bitwise={sl} {dl:.2e} | out bitwise={sg} {dg:.2e}",
                flush=True,
            )


def run_suite(quick):
    group = dist.group.WORLD
    n = group.size()
    failures = []
    configs = [(32 if n <= 4 else 64, 4 if n <= 4 else 8, 128)]
    if not quick:
        configs.append((8 * n, n, 128))
    for hq, hkv, d in configs:
        for s_local, cs in ((1024, 128), (1024, 256), (1024, 0), (2048, 512)):
            check_forward(group, hq, hkv, d, s_local, cs, 3, failures)
    return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    dist.init_process_group("nccl", timeout=timedelta(minutes=5))
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    failures = run_suite(args.quick)
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, failures)
    if dist.get_rank() == 0:
        flat = [f for fs in gathered for f in fs]
        print("\n".join(flat) if flat else "ALL PASSED", flush=True)
    dist.destroy_process_group()
    sys.exit(1 if any(gathered) else 0)


@pytest.mark.skipif(not is_gfx950(), reason="gfx950 only")
@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_mega_attn_a2a_forward(world_size):
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"needs {world_size} GPUs")
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={world_size}",
        __file__,
        "--quick",
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
