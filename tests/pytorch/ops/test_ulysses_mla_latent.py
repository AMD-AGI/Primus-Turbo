###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Ulysses MLA latent up-projection against a single-device fp32 reference over the full sequence.

Every rank rebuilds all ranks' latents and output grads from fixed seeds, runs the reference over the
whole sequence and all heads, and checks its own head slice (forward), its own sequence chunk
(latent grads) and the rank-summed weight grads (what the gradient all-reduce produces).

    pytest tests/pytorch/ops/test_ulysses_mla_latent.py --dist-only
    torchrun --nproc_per_node 4 tests/pytorch/ops/test_ulysses_mla_latent.py
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

# (q_lora_rank, kv_lora_rank, heads as a function of n)
DIM_CONFIGS = {
    "small": (256, 128, lambda n: 2 * n),
    "dsv3": (1536, 512, lambda n: 128),
}
QK_NOPE, QK_ROPE, V_DIM = 128, 64, 128


def _randn(*shape, seed, scale=1.0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(*shape, device="cuda", generator=gen).mul_(scale).bfloat16()


def _close(name, out, ref, failures, tag, scale=None):
    from tests.pytorch.test_utils import get_tolerances

    tol = get_tolerances(torch.bfloat16)
    if out.shape != ref.shape:
        failures.append(f"{tag} {name}: shape {tuple(out.shape)} != {tuple(ref.shape)}")
        return
    scale = ref.float().abs() if scale is None else scale
    diff = (out.float() - ref.float()).abs()
    excess = diff - tol["atol"] - tol["rtol"] * scale
    if (excess > 0).any():
        i = excess.argmax().item()
        failures.append(
            f"{tag} {name}: max abs err {diff.max().item():.4g}, {(excess > 0).sum().item()} elements out of "
            f"tolerance, worst got {out.flatten()[i].item():.5g} expected {ref.flatten()[i].item():.5g}"
        )


def reference(lq, lkv, wq, wkv, heads):
    """fp32 MLA up-projection of every head; ``lq`` / ``lkv`` cover the whole sequence."""
    ckv = wkv.shape[1]
    lead = lq.shape[:-1]
    q = (lq @ wq.t()).view(*lead, heads, QK_NOPE + QK_ROPE)
    kv = (lkv[..., :ckv] @ wkv.t()).view(*lead, heads, QK_NOPE + V_DIM)
    k_pe = lkv[..., None, ckv:].expand(*lead, heads, QK_ROPE)
    k = torch.cat((kv[..., :QK_NOPE], k_pe), dim=-1)
    return q, k, kv[..., QK_NOPE:]


def _poison_banks(group):
    """Fill both banks of every workspace with bf16 NaN: a GEMM tile that reads a gathered row before
    its producer wrote it this epoch (or a padding row into a live output) shows up as NaN."""
    from primus_turbo.flydsl.ulysses.workspace import _WORKSPACES, FLAG_BYTES

    torch.cuda.synchronize()
    dist.barrier(group=group)
    for ws in _WORKSPACES.values():
        banks = ws.symm_mem.get_buffer(ws.rank, (ws.bank_bytes,), torch.int16, storage_offset=FLAG_BYTES // 2)
        banks.fill_(0x7FC0)
    torch.cuda.synchronize()
    dist.barrier(group=group)


def check(group, dims, b, s_local, qkv_format, fused, repeats, failures, poison=False):
    from primus_turbo.pytorch.ops.attention.ulysses_mla_latent import (
        ulysses_mla_latent_up_proj,
    )

    n, rank = group.size(), group.rank()
    cq, ckv, heads_of = DIM_CONFIGS[dims]
    heads = heads_of(n)
    if heads % n:
        return
    hl = heads // n
    seq_dim, head_dim = (0, 2) if qkv_format == "sbhd" else (1, 2)
    local = (s_local, b) if qkv_format == "sbhd" else (b, s_local)
    full = (s_local * n, b) if qkv_format == "sbhd" else (b, s_local * n)
    tag = f"[{dims} h={heads} b={b} s/n={s_local} {qkv_format} {'fused' if fused else 'rccl'}]"
    wq = _randn(heads * (QK_NOPE + QK_ROPE), cq, seed=11, scale=0.02).requires_grad_()
    wkv = _randn(heads * (QK_NOPE + V_DIM), ckv, seed=12, scale=0.02).requires_grad_()
    for it in range(repeats):
        lqs = [_randn(*local, cq, seed=1000 * it + j) for j in range(n)]
        lkvs = [_randn(*local, ckv + QK_ROPE, seed=1000 * it + 100 + j) for j in range(n)]
        # grads[j][t]: rank j's grad of q / k / v
        out_dims = (QK_NOPE + QK_ROPE, QK_NOPE + QK_ROPE, V_DIM)
        grads = [
            [_randn(*full, hl, d, seed=7 * it + 31 * t + 1000 * j) for t, d in enumerate(out_dims)]
            for j in range(n)
        ]

        lq, lkv = lqs[rank].clone().requires_grad_(), lkvs[rank].clone().requires_grad_()
        wq.grad = wkv.grad = None
        if poison:
            _poison_banks(group)
            # stagger the ranks so peers' rows land well after this rank's tiles start
            torch.cuda._sleep(((rank + it) % n) * 200_000)
        outs = ulysses_mla_latent_up_proj(
            lq, lkv, wq, wkv, heads, group, QK_NOPE, QK_ROPE, V_DIM, qkv_format, fused=fused
        )
        torch.autograd.backward(outs, grads[rank])

        lq_ref = torch.cat(lqs, dim=seq_dim).float().requires_grad_()
        lkv_ref = torch.cat(lkvs, dim=seq_dim).float().requires_grad_()
        wq_ref, wkv_ref = wq.detach().float().requires_grad_(), wkv.detach().float().requires_grad_()
        refs = reference(lq_ref, lkv_ref, wq_ref, wkv_ref, heads)
        for name, out, ref in zip("qkv", outs, refs):
            _close(name, out, ref.narrow(head_dim, rank * hl, hl), failures, f"{tag} call {it}")
        torch.autograd.backward(
            refs, [torch.cat([g[t] for g in grads], dim=head_dim).float() for t in range(3)]
        )

        # latent grads sum every head of every rank: tolerance relative to the summed magnitudes
        mag_grads = [torch.cat([g[t] for g in grads], dim=head_dim).float().abs() for t in range(3)]
        lq_mag = lq_ref.detach().clone().requires_grad_()
        lkv_mag = lkv_ref.detach().clone().requires_grad_()
        refs_mag = reference(lq_mag, lkv_mag, wq_ref.detach().abs(), wkv_ref.detach().abs(), heads)
        torch.autograd.backward(refs_mag, mag_grads)
        chunk = slice(rank * s_local, (rank + 1) * s_local)
        sel = (chunk,) if qkv_format == "sbhd" else (slice(None), chunk)
        _close("d_q_latent", lq.grad, lq_ref.grad[sel], failures, f"{tag} call {it}", lq_mag.grad[sel])
        _close("d_kv_latent", lkv.grad, lkv_ref.grad[sel], failures, f"{tag} call {it}", lkv_mag.grad[sel])

        dwq, dwkv = wq.grad.clone(), wkv.grad.clone()
        for g in (dwq, dwkv):
            dist.all_reduce(g, group=group)
        _close("d_w_q_b", dwq, wq_ref.grad, failures, f"{tag} call {it}")
        _close("d_w_kv_b", dwkv, wkv_ref.grad, failures, f"{tag} call {it}")


def run_suite(quick):
    group = dist.group.WORLD
    n, rank = group.size(), group.rank()
    failures = []
    dims_list = ["small"] if quick else list(DIM_CONFIGS)
    batches = [1] if quick else [1, 2]
    formats = ["sbhd"] if quick else ["sbhd", "bshd"]
    for dims, b, qkv_format, s_local, fused in itertools.product(
        dims_list, batches, formats, [300, 512], [False, True]
    ):
        if fused and qkv_format != "sbhd":
            continue
        check(group, dims, b, s_local, qkv_format, fused, 2, failures)
    # the fused cases above already grew the workspace to these shapes, so every call reads poisoned banks
    for dims, s_local in itertools.product(dims_list, [300, 512]):
        check(group, dims, 2, s_local, "sbhd", True, 3, failures, poison=True)

    gathered = [None] * n
    dist.all_gather_object(gathered, failures, group=group)
    if rank == 0:
        all_failures = [f"rank {r}: {f}" for r, fs in enumerate(gathered) for f in fs]
        print("\n".join(all_failures) if all_failures else f"ulysses mla latent: all cases passed (n={n})")
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


@pytest.mark.parametrize("BM,SUB", [(256, 64), (256, 32), (128, 64), (256, 256)])
@pytest.mark.parametrize("n", [2, 4, 8])
def test_ag_gemm_row_slots_cover(n, BM, SUB):
    """Every (source, sub-tile) flag a GEMM m-tile reads is among the ones it waits on: a missed short
    last sub-tile is a race the GPU tests rarely lose."""
    from primus_turbo.flydsl.ulysses.ag_gemm_kernel import _row_slots

    for T in list(range(1, 2 * BM + 2)) + [600, 1000, 4100]:
        R, n_sub = n * T, -(-T // SUB)
        for m in range(-(-R // BM)):
            need = {(r // T) * n_sub + (r % T) // SUB for r in range(m * BM, min((m + 1) * BM, R))}
            got = _row_slots(m, BM, SUB, T, R, n_sub, i32=int, select=lambda c, a, b: a if c else b)
            assert need <= set(got), f"T={T} tile {m}: missed slots {sorted(need - set(got))}"


@pytest.mark.multigpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPUs")
@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_ulysses_mla_latent(world_size):
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"needs {world_size} GPUs")
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    cmd = [sys.executable, "-m", "torch.distributed.run", f"--nproc_per_node={world_size}", __file__]
    result = subprocess.run(cmd, cwd=repo_root, env={**os.environ, "PYTHONPATH": repo_root}, timeout=1800)
    assert result.returncode == 0


if __name__ == "__main__":
    main()
