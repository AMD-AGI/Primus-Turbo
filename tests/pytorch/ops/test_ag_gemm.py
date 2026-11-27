###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fused all-gather + GEMM kernel (``fused_collective_gemm.ag_gemm``, the MLA latent up-projection)
against an fp32 reference.

Every rank rebuilds all ranks' latent rows from fixed seeds. Reference: those rows stacked in rank
order (the gathered sequence) times this rank's up-projection rows; ``k``'s rope columns against the
gathered ``k_pe``; ``lq`` / ``lc`` against the gathered latents exactly. Outputs start as NaN, so an
element the kernel does not write fails.

The pytest entry launches the ``--quick`` suite under torchrun for every world size the node has
GPUs for. The full suite takes longer than the CI timeout on a cold FlyDSL cache, so it is run by hand:

    pytest tests/pytorch/ops/test_ag_gemm.py --dist-only            # quick, all world sizes
    torchrun --nproc_per_node 8 tests/pytorch/ops/test_ag_gemm.py   # full
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

# (q_lora_rank, kv_lora_rank, local heads as a function of n)
DIMS = {
    "small": (256, 128, lambda n: 2),
    "deepseek-v3": (1536, 512, lambda n: 128 // n),
}


def _randn(*shape, seed, scale=1.0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(*shape, device="cuda", generator=gen).mul_(scale).bfloat16()


def _close(name, out, ref, failures, tag):
    from tests.pytorch.test_utils import get_tolerances

    tol = get_tolerances(torch.bfloat16)
    if out.shape != ref.shape:
        failures.append(f"{tag} {name}: shape {tuple(out.shape)} != {tuple(ref.shape)}")
        return
    diff = (out.float() - ref.float()).abs()
    excess = diff - tol["atol"] - tol["rtol"] * ref.float().abs()
    if not torch.isfinite(out).all() or (excess > 0).any():
        i = excess.nan_to_num(float("inf")).argmax().item()
        failures.append(
            f"{tag} {name}: max abs err {diff.max().item():.4g}, {(~(excess <= 0)).sum().item()} elements "
            f"out of tolerance, worst got {out.flatten()[i].item():.5g} expected {ref.flatten()[i].item():.5g}"
        )


def _poison_banks(group):
    """Fill both banks of the workspace with bf16 NaN: a GEMM tile reading a gathered row before its
    producer wrote it this epoch, or a padding row into a live output, shows up as NaN."""
    from primus_turbo.flydsl.fused_collective_gemm.workspace import (
        _WORKSPACES,
        FLAG_BYTES,
    )

    torch.cuda.synchronize()
    dist.barrier(group=group)
    for ws in _WORKSPACES.values():
        banks = ws.symm_mem.get_buffer(ws.rank, (ws.bank_bytes,), torch.int16, storage_offset=FLAG_BYTES // 2)
        banks.fill_(0x7FC0)
    torch.cuda.synchronize()
    dist.barrier(group=group)


def check_ag_gemm(
    group,
    failures,
    *,
    dims,
    T,
    dn=128,
    dr=64,
    dv=128,
    hl=None,
    save=True,
    flag_region=4,
    repeats=2,
    poison=False,
    **tuning,
):
    """``T`` local rows (sbhd ``s_local * b`` flattened) of ``[q_latent | c_kv | k_pe]`` per rank;
    ``hl`` local heads (default from ``DIMS``)."""
    from primus_turbo.flydsl.fused_collective_gemm.ag_gemm_kernel import (
        ag_gemm,
        ag_gemm_bank_bytes,
    )
    from primus_turbo.flydsl.fused_collective_gemm.workspace import (
        get_collective_gemm_workspace,
    )

    n, rank = group.size(), group.rank()
    cq, ckv, hl_of = DIMS[dims]
    hl = hl_of(n) if hl is None else hl
    R = n * T
    tag = f"[{dims} hl={hl} dn={dn} dr={dr} dv={dv} T={T} save={save} region={flag_region} {tuning}]"
    wq = _randn(hl * (dn + dr), cq, seed=11 + rank, scale=0.02)
    # rows: k_nope of every local head, then v of every local head
    wkv = _randn(hl * (dn + dv), ckv, seed=31 + rank, scale=0.02)
    workspace = get_collective_gemm_workspace(
        group, ag_gemm_bank_bytes(R, cq, ckv, dr, tuning.get("BM", 256))
    )
    nan = float("nan")
    for it in range(repeats):
        xqs = [_randn(T, cq, seed=1000 * it + j) for j in range(n)]
        xkvs = [_randn(T, ckv + dr, seed=1000 * it + 500 + j) for j in range(n)]
        q, k = (torch.full((R, hl, dn + dr), nan, dtype=torch.bfloat16, device="cuda") for _ in range(2))
        v = torch.full((R, hl, dv), nan, dtype=torch.bfloat16, device="cuda")
        lq = torch.full((R, cq), nan, dtype=torch.bfloat16, device="cuda") if save else None
        lc = torch.full((R, ckv), nan, dtype=torch.bfloat16, device="cuda") if save else None
        if poison:
            _poison_banks(group)
            # stagger the ranks so peers' rows land well after this rank's tiles start
            torch.cuda._sleep(((rank + it) % n) * 200_000)
        ag_gemm(
            xqs[rank],
            xkvs[rank],
            wq,
            wkv,
            q,
            k,
            v,
            lq,
            lc,
            workspace,
            dn=dn,
            dr=dr,
            dv=dv,
            flag_region=flag_region,
            **tuning,
        )
        gq, gkv = torch.cat(xqs), torch.cat(xkvs)
        gc, gpe = gkv[:, :ckv], gkv[:, ckv:]
        wk, wv = wkv[: hl * dn].float(), wkv[hl * dn :].float()
        k_ref = torch.cat(
            ((gc.float() @ wk.t()).view(R, hl, dn), gpe.float()[:, None].expand(R, hl, dr)), dim=-1
        )
        _close("q", q, (gq.float() @ wq.float().t()).view(R, hl, dn + dr), failures, f"{tag} call {it}")
        _close("k", k, k_ref, failures, f"{tag} call {it}")
        _close("v", v, (gc.float() @ wv.t()).view(R, hl, dv), failures, f"{tag} call {it}")
        if save:
            for name, out, ref in (("lq", lq, gq), ("lc", lc, gc)):
                if not torch.equal(out, ref):
                    failures.append(f"{tag} call {it} {name}: {(out != ref).sum().item()} elements differ")


def run_suite(quick):
    group = dist.group.WORLD
    n = group.size()
    failures = []
    dims_list = ["small"] if quick else list(DIMS)

    def check(**kw):
        check_ag_gemm(group, failures, **kw)

    # T = 300 is not a multiple of a sub-tile or an m-tile: m-tiles straddle sources and short last
    # sub-tiles; with and without the latent copies
    saves = [True] if quick else [True, False]
    for dims, T, save in itertools.product(dims_list, [300, 512], saves):
        check(dims=dims, T=T, save=save)
    # narrower heads (dn = 64: four heads per 256-wide k tile)
    check(dims="small", T=300, dn=64, dr=32, dv=64, hl=4)
    # launch knobs: tile rows, push sub-tile rows (one sub-tile per m-tile at 256), pushers (1 per
    # rank, many), copy units per m-tile
    knobs = [
        {"BM": 128},
        {"SUB": 32},
        {"SUB": 256},
        {"num_push": n},
        {"num_push": 64},
        {"copy_parts": 1},
        {"copy_parts": 8},
    ]
    for kn in knobs[:2] if quick else knobs:
        for T in (300, 512):
            check(dims="small", T=T, **kn)
    # DeepSeek-V3 at a longer local sequence: several waves of tiles per XCD list
    if not quick:
        check(dims="deepseek-v3", T=2048, repeats=1)
    # banks poisoned before every call, flag regions interleaved on one workspace
    for region in (4, 0, 4):
        for T in (300, 512):
            check(dims="small", T=T, flag_region=region, poison=True, repeats=3)
    check(dims=dims_list[-1], T=512, poison=True, repeats=2)

    gathered = [None] * n
    dist.all_gather_object(gathered, failures, group=group)
    if group.rank() == 0:
        all_failures = [f"rank {r}: {f}" for r, fs in enumerate(gathered) for f in fs]
        print("\n".join(all_failures) if all_failures else f"ag_gemm: all cases passed (n={n})")
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
    from primus_turbo.flydsl.fused_collective_gemm.ag_gemm_kernel import _row_slots

    for T in list(range(1, 2 * BM + 2)) + [600, 1000, 4100]:
        R, n_sub = n * T, -(-T // SUB)
        for m in range(-(-R // BM)):
            need = {(r // T) * n_sub + (r % T) // SUB for r in range(m * BM, min((m + 1) * BM, R))}
            got = _row_slots(m, BM, SUB, T, R, n_sub, i32=int, select=lambda c, a, b: a if c else b)
            assert need <= set(got), f"T={T} tile {m}: missed slots {sorted(need - set(got))}"


@pytest.mark.multigpu
@pytest.mark.skipif(
    not (torch.cuda.is_available() and is_gfx950()), reason="the fused collective GEMM kernels need gfx950"
)
# A cold FlyDSL cache takes minutes per world size to compile its kernels; allow the subprocess timeout.
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_ag_gemm(world_size):
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
    result = subprocess.run(cmd, cwd=repo_root, env={**os.environ, "PYTHONPATH": repo_root}, timeout=1800)
    assert result.returncode == 0


if __name__ == "__main__":
    main()
