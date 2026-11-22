###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fused GEMM + all-to-all kernel (``fused_collective_gemm.gemm_a2a``) against an fp32 reference.

Reference: all-gather every rank's ``a``, multiply by this rank's column block of ``b`` and lay the rows
out at their source rank's sequence offset, split into the q / k / v head pieces; the in-kernel wgrad
against ``a^T @ wb``.

The pytest entry launches the ``--quick`` suite under torchrun for every world size the node has
GPUs for. The full suite takes longer than the CI timeout on a cold FlyDSL cache, so it is run by hand:

    pytest tests/pytorch/ops/test_gemm_a2a.py --dist-only            # quick, all world sizes
    torchrun --nproc_per_node 8 tests/pytorch/ops/test_gemm_a2a.py   # full
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


def reference_gemm_a2a(a, b, layout, d, G, nq, nk, nv, batch, seq_local, qkv_format, group):
    """(q, k, v) ``[n * M, heads, d]`` in fp32: rank ``r``'s column block of ``a_src @ b`` for every
    source rank, rows at the source's sequence offset (``sbhd`` [s, b], ``bshd`` [b, s])."""
    n, rank = group.size(), group.rank()
    a_all = torch.empty(n, *a.shape, dtype=a.dtype, device=a.device)
    dist.all_gather_into_tensor(a_all, a, group=group)
    w = b if layout == "nt" else b.t()
    N_r = w.shape[0] // n
    c = a_all.float() @ w[rank * N_r : (rank + 1) * N_r].float().t()
    if qkv_format == "bshd":
        c = c.view(n, batch, seq_local, N_r).transpose(0, 1)
    c = c.reshape(n * a.shape[0], G, nq + nk + nv, d)
    q, k, v = torch.split(c, [nq, nk, nv], dim=2)
    return tuple(t.reshape(n * a.shape[0], G * t.shape[2], d) for t in (q, k, v))


def _poison_banks(group):
    """Fill both banks of the workspace with bf16 NaN: a pull of a tile before its producer wrote it
    this epoch, or of a padding row into a live output, shows up as NaN."""
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


def check_gemm_a2a(
    group,
    failures,
    *,
    K,
    heads,
    d,
    batch,
    seq_local,
    qkv_format,
    layout="nt",
    wgrad_n=0,
    flag_region=0,
    repeats=2,
    poison=False,
    **tuning,
):
    """``heads = (G, nq, nk, nv)``: one rank's column block holds ``G`` groups of ``nq`` q, ``nk`` k and
    ``nv`` v heads (``nk = nv = 0``: a single output)."""
    from primus_turbo.flydsl.fused_collective_gemm.gemm_a2a_kernel import (
        gemm_a2a,
        gemm_a2a_bank_bytes,
    )
    from primus_turbo.flydsl.fused_collective_gemm.workspace import (
        get_collective_gemm_workspace,
    )

    n, rank = group.size(), group.rank()
    G, nq, nk, nv = heads
    M = seq_local * batch
    N = n * G * (nq + nk + nv) * d
    tag = (
        f"[K={K} heads={heads} d={d} b={batch} s/n={seq_local} {qkv_format} {layout} wgrad_n={wgrad_n} "
        f"region={flag_region} {tuning}]"
    )
    b = _randn(*((N, K) if layout == "nt" else (K, N)), seed=1234, scale=0.02)
    workspace = get_collective_gemm_workspace(group, gemm_a2a_bank_bytes(M, N, 256))
    for it in range(repeats):
        a = _randn(M, K, seed=100 * it + rank)
        outs = tuple(torch.empty(n * M, G * h, d, dtype=torch.bfloat16, device="cuda") for h in (nq, nk, nv))
        if nk == 0:
            outs = (outs[0],) * 3
        wgrad = None
        if wgrad_n:
            wb = _randn(M, wgrad_n, seed=200 * it + rank)
            wgrad = (a, wb, torch.empty(K, wgrad_n, dtype=torch.bfloat16, device="cuda"))
        if poison:
            _poison_banks(group)
        gemm_a2a(
            a,
            b,
            outs,
            workspace,
            layout=layout,
            head_dim=d,
            groups=G,
            nq=nq,
            nk=nk,
            nv=nv,
            seq_local=seq_local,
            qkv_format=qkv_format,
            flag_region=flag_region,
            wgrad=wgrad,
            **tuning,
        )
        refs = reference_gemm_a2a(a, b, layout, d, G, nq, nk, nv, batch, seq_local, qkv_format, group)
        for name, out, ref, h in zip("qkv", outs, refs, (nq, nk, nv)):
            if h:
                _close(name, out, ref, failures, f"{tag} call {it}")
        if wgrad_n:
            _close("dw", wgrad[2], a.float().t() @ wgrad[1].float(), failures, f"{tag} call {it}")


def run_suite(quick):
    group = dist.group.WORLD
    n = group.size()
    failures = []
    dims = [128] if quick else [64, 128]
    batches = [1] if quick else [1, 2]
    formats = ["sbhd"] if quick else ["sbhd", "bshd"]

    def qkv(hq, hkv, block="interleaved", **kw):
        """QKV projection (layout nt). Megatron's interleaved rows: G = hkv / n groups of hq / hkv q, 1 k
        and 1 v heads; the rank-major concat: one group of hq / n q, hkv / n k and hkv / n v heads."""
        heads = (hkv // n, hq // hkv, 1, 1) if block == "interleaved" else (1, hq // n, hkv // n, hkv // n)
        check_gemm_a2a(group, failures, heads=heads, **kw)

    def dout(heads, **kw):
        """dO of the output projection (layout nn): ``heads * d`` columns into one output."""
        check_gemm_a2a(group, failures, heads=(1, heads // n, 0, 0), layout="nn", flag_region=1, **kw)

    # GQA 32 / 8; s/n = 300 is not a multiple of any tile
    blocks = ["interleaved"] if quick else ["interleaved", "concat"]
    for d, batch, fmt, block, s in itertools.product(dims, batches, formats, blocks, [300, 512]):
        qkv(32, 8, block, K=1024, d=d, batch=batch, seq_local=s, qkv_format=fmt)
    # MHA (one k and one v head per q head), 256-row tiles
    for fmt in formats:
        qkv(8 * n, 8 * n, "concat", K=1024, d=128, batch=1, seq_local=300, qkv_format=fmt, BM=256)
    # dispatch-order knobs: no lead wave (the pulls wait from the start), no tail pulls, many pull blocks
    for knobs in ({"num_lead": 0}, {"num_tail_pull": 0}, {"num_pull_cu": 64, "num_tail_pull": 0}):
        qkv(32, 8, K=1024, d=128, batch=1, seq_local=512, qkv_format="sbhd", **knobs)
    # dO with the dW = dY^T @ O tiles in the same launch, dispatched first, after 1/8 of the GEMM tiles
    # (default) and last; then without them
    for fmt, wgrad_at in itertools.product(formats, [0, None, 1 << 20]):
        dout(
            32,
            K=1024,
            d=128,
            batch=1,
            seq_local=512,
            qkv_format=fmt,
            wgrad_n=32 * 128,
            BM=256,
            wgrad_at=wgrad_at,
        )
    dout(32, K=1024, d=128, batch=2, seq_local=300, qkv_format="bshd", BM=256)
    # GLM-4.7 widths (hidden 5120, 96 / 8 heads): several waves of GEMM tiles; the backward has more
    # dW tiles than CUs, which go last
    qkv(
        96,
        8,
        K=5120,
        d=128,
        batch=1,
        seq_local=4096,
        qkv_format="sbhd",
        BM=256,
        num_pull_cu=24,
        num_tail_pull=8,
        repeats=1,
    )
    dout(96, K=5120, d=128, batch=1, seq_local=4096, qkv_format="sbhd", wgrad_n=96 * 128, BM=256, repeats=1)
    # banks poisoned before every call, flag regions interleaved on one workspace
    for region in (0, 3, 0):
        qkv(
            32,
            8,
            "concat",
            K=1024,
            d=128,
            batch=2,
            seq_local=300,
            qkv_format="bshd",
            flag_region=region,
            repeats=1,
            poison=True,
        )
        dout(
            32,
            K=1024,
            d=128,
            batch=1,
            seq_local=512,
            qkv_format="sbhd",
            wgrad_n=1024,
            BM=256,
            repeats=1,
            poison=True,
        )

    gathered = [None] * n
    dist.all_gather_object(gathered, failures, group=group)
    if group.rank() == 0:
        all_failures = [f"rank {r}: {f}" for r, fs in enumerate(gathered) for f in fs]
        print("\n".join(all_failures) if all_failures else f"gemm_a2a: all cases passed (n={n})")
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
@pytest.mark.skipif(
    not (torch.cuda.is_available() and is_gfx950()), reason="the fused collective GEMM kernels need gfx950"
)
# A cold FlyDSL cache takes minutes per world size to compile its kernels; allow the subprocess timeout.
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_gemm_a2a(world_size):
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
