###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fused all-to-all + GEMM kernel (``fused_collective_gemm.a2a_gemm``) against an fp32 reference.

Reference: every rank's head columns over the full sequence, all-gathered; this rank's sequence slice
of every source's columns side by side is the gathered ``A``, multiplied by the weight, plus the
residual; ``o_gathered`` against ``A`` exactly; the in-kernel wgrad against ``A^T @ x``.

The pytest entry launches the ``--quick`` suite under torchrun for every world size the node has
GPUs for. The full suite takes longer than the CI timeout on a cold FlyDSL cache, so it is run by hand:

    pytest tests/pytorch/ops/test_a2a_gemm.py --dist-only            # quick, all world sizes
    torchrun --nproc_per_node 8 tests/pytorch/ops/test_a2a_gemm.py   # full
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


def reference_gathered(parts, G, d, batch, seq_local, qkv_format, group):
    """The gathered ``A`` ``[M, n * K_r]``: this rank's sequence slice (``sbhd`` [s, b], ``bshd``
    [b, s]) of every source's head columns, source-major, each source's columns in the weight's row
    order (``G`` groups of its q, k, v heads)."""
    n, rank = group.size(), group.rank()
    rows = parts[0].shape[0]
    M = rows // n
    cols = torch.cat([p.reshape(rows, G, -1, d) for p in parts], dim=2).reshape(rows, -1)
    cols_all = torch.empty(n, *cols.shape, dtype=cols.dtype, device=cols.device)
    dist.all_gather_into_tensor(cols_all, cols.contiguous(), group=group)
    if qkv_format == "sbhd":
        mine = cols_all[:, rank * M : (rank + 1) * M]
    else:
        mine = cols_all.view(n, batch, n, seq_local, -1)[:, :, rank].reshape(n, M, -1)
    return mine.transpose(0, 1).reshape(M, -1)


def _poison_banks(group):
    """Fill both banks of the workspace with bf16 NaN: a GEMM or wgrad tile reading a recv row before
    its producer wrote it this epoch, or a padding row into a live output, shows up as NaN."""
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


def check_a2a_gemm(
    group,
    failures,
    *,
    N,
    heads,
    d,
    batch,
    seq_local,
    qkv_format,
    layout="nt",
    residual=False,
    save_o=False,
    wgrad=False,
    flag_region=2,
    repeats=2,
    poison=False,
    **tuning,
):
    """``heads = (G, nq, nk, nv)``: one rank's columns are ``G`` groups of ``nq`` q, ``nk`` k and ``nv``
    v heads, passed as ``(dq, dk, dv)``; ``nk = nv = 0``: one ``[n * M, nq * d]`` tensor."""
    from primus_turbo.flydsl.fused_collective_gemm.a2a_gemm_kernel import (
        a2a_gemm,
        a2a_gemm_bank_bytes,
    )
    from primus_turbo.flydsl.fused_collective_gemm.workspace import (
        get_collective_gemm_workspace,
    )

    n, rank = group.size(), group.rank()
    G, nq, nk, nv = heads
    M = seq_local * batch
    K = n * G * (nq + nk + nv) * d
    tag = (
        f"[N={N} heads={heads} d={d} b={batch} s/n={seq_local} {qkv_format} {layout} residual={residual} "
        f"save_o={save_o} wgrad={wgrad} region={flag_region} {tuning}]"
    )
    w = _randn(*((N, K) if layout == "nt" else (K, N)), seed=1234, scale=0.02)
    workspace = get_collective_gemm_workspace(group, a2a_gemm_bank_bytes(M, K))
    for it in range(repeats):
        if nk == 0:
            parts = (_randn(n * M, G * nq * d, seed=100 * it + rank),)
            o = parts[0]
        else:
            parts = tuple(
                _randn(n * M, G * h, d, seed=100 * it + 10 * j + rank) for j, h in enumerate((nq, nk, nv))
            )
            o = parts
        res = _randn(M, N, seed=200 * it + rank) if residual else None
        y = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
        og = torch.empty(M, K, dtype=torch.bfloat16, device="cuda") if save_o else None
        wg = None
        if wgrad:
            wg = (_randn(M, N, seed=300 * it + rank), torch.empty(K, N, dtype=torch.bfloat16, device="cuda"))
        if poison:
            _poison_banks(group)
        a2a_gemm(
            o,
            w,
            res,
            y,
            og,
            workspace,
            seq_local=seq_local,
            qkv_format=qkv_format,
            flag_region=flag_region,
            layout=layout,
            head_dim=d,
            groups=G,
            wgrad=wg,
            **tuning,
        )
        a = reference_gathered(parts, G, d, batch, seq_local, qkv_format, group)
        ref = a.float() @ (w.float().t() if layout == "nt" else w.float())
        if residual:
            ref += res.float()
        _close("y", y, ref, failures, f"{tag} call {it}")
        if save_o and not torch.equal(og, a):
            failures.append(f"{tag} call {it} o_gathered: {(og != a).sum().item()} elements differ")
        if wgrad:
            _close("dw", wg[1], a.float().t() @ wg[0].float(), failures, f"{tag} call {it}")


def run_suite(quick):
    group = dist.group.WORLD
    n = group.size()
    failures = []
    dims = [128] if quick else [64, 128]
    batches = [1] if quick else [1, 2]
    formats = ["sbhd"] if quick else ["sbhd", "bshd"]

    def qkv_bwd(hq, hkv, block="interleaved", hidden=1024, **kw):
        """Backward of the QKV projection (layout nn): ``dx = [dq dk dv] @ W_qkv`` in Megatron's
        interleaved row order (G = hkv / n groups of hq / hkv q, 1 k and 1 v heads) or the rank-major
        concat (one group of hq / n q, hkv / n k and hkv / n v heads), wgrad in the same launch where
        the shape tiles (as the op decides: whole 256-row m-tiles and dw tiles), else the gathered rows
        copied out for a separate wgrad."""
        heads = (hkv // n, hq // hkv, 1, 1) if block == "interleaved" else (1, hq // n, hkv // n, hkv // n)
        M, K_r = kw["seq_local"] * kw["batch"], sum(heads[1:]) * heads[0] * kw["d"]
        if "wgrad" not in kw:
            kw["wgrad"] = M % 256 == 0 and K_r % 256 == 0 and hidden % 256 == 0
        kw.setdefault("save_o", not kw["wgrad"])
        kw.setdefault("flag_region", 3)
        check_a2a_gemm(group, failures, N=hidden, heads=heads, layout="nn", **kw)

    def o_fwd(heads, hidden=1024, **kw):
        """Forward of the output projection (layout nt): ``heads * d`` columns from one tensor."""
        kw.setdefault("residual", True)
        kw.setdefault("save_o", True)
        check_a2a_gemm(group, failures, N=hidden, heads=(1, heads // n, 0, 0), **kw)

    # GQA 32 / 8; s/n = 300 is not a multiple of any tile (no in-kernel wgrad, rows copied out); 512
    # runs the wgrad tiles in the launch where the rank's columns divide by 256
    blocks = ["interleaved"] if quick else ["interleaved", "concat"]
    for d, batch, fmt, block, s in itertools.product(dims, batches, formats, blocks, [300, 512]):
        qkv_bwd(32, 8, block, d=d, batch=batch, seq_local=s, qkv_format=fmt)
    # MHA (one k and one v head per q head)
    for fmt in formats:
        qkv_bwd(8 * n, 8 * n, "concat", d=128, batch=1, seq_local=512, qkv_format=fmt)
    # o_proj with and without the residual and the gathered copy
    residuals = [True] if quick else [True, False]
    for d, batch, fmt, res in itertools.product(dims, batches, formats, residuals):
        o_fwd(32, d=d, batch=batch, seq_local=300, qkv_format=fmt, residual=res, save_o=res)
    # launch knobs: tile rows, pushers, push sub-tiles, wgrad claim position (first, middle, last of
    # the 128-row GEMM tiles)
    tiles = (512 // 128) * (1024 // 256)
    knobs = [
        {"BM": 128},
        {"BM": 256},
        {"num_push": 4},
        {"num_push": 64},
        {"push_split": 1},
        {"push_split": 2},
        {"BM": 128, "wgrad_at": 0},
        {"BM": 128, "wgrad_at": tiles // 2},
        {"BM": 128, "wgrad_at": tiles},
    ]
    for kn in knobs[:2] if quick else knobs:
        qkv_bwd(32, 8, d=128, batch=1, seq_local=512, qkv_format="sbhd", **kn)
    for kn in knobs[:6:2]:
        o_fwd(32, d=128, batch=1, seq_local=512, qkv_format="sbhd", **kn)
    # GLM-4.7 widths (hidden 5120, 96 / 8 heads): over two waves of wgrad tiles, which go last, after
    # 256-row tiles, with the own rows copied first by every block; the o_proj makes several waves
    qkv_bwd(96, 8, hidden=5120, d=128, batch=1, seq_local=512, qkv_format="sbhd", repeats=1)
    o_fwd(96, hidden=5120, d=128, batch=1, seq_local=512, qkv_format="sbhd", repeats=1)
    # MiniMax-M2.5 widths (hidden 3072, 48 / 8 heads): few wgrad tiles, after 128-row tiles
    qkv_bwd(48, 8, hidden=3072, d=128, batch=1, seq_local=512, qkv_format="sbhd", repeats=1)
    # Llama-70B widths (hidden 8192, 64 / 8 heads): more GEMM tiles than CUs, GEMM-bound launch (K walk
    # in step)
    qkv_bwd(64, 8, hidden=8192, d=128, batch=1, seq_local=1024, qkv_format="sbhd", repeats=1)
    o_fwd(64, hidden=8192, d=128, batch=1, seq_local=2048, qkv_format="bshd", repeats=1)
    if not quick:
        # long sequence, narrow QKV (M > K): a wave of GEMM tiles after the longer wgrad tiles
        qkv_bwd(16, 8, hidden=4096, d=64, batch=1, seq_local=4096, qkv_format="sbhd", repeats=1)
    # banks poisoned before every call, flag regions interleaved on one workspace
    for region in (3, 0, 3):
        for s in (300, 512):
            qkv_bwd(32, 8, d=128, batch=2, seq_local=s, qkv_format="bshd", flag_region=region, poison=True)
        o_fwd(32, d=128, batch=2, seq_local=300, qkv_format="sbhd", flag_region=region, poison=True)
    for fmt in formats:
        qkv_bwd(96, 8, hidden=5120, d=128, batch=1, seq_local=512, qkv_format=fmt, repeats=1, poison=True)
        qkv_bwd(48, 8, hidden=3072, d=128, batch=1, seq_local=512, qkv_format=fmt, repeats=1, poison=True)

    gathered = [None] * n
    dist.all_gather_object(gathered, failures, group=group)
    if group.rank() == 0:
        all_failures = [f"rank {r}: {f}" for r, fs in enumerate(gathered) for f in fs]
        print("\n".join(all_failures) if all_failures else f"a2a_gemm: all cases passed (n={n})")
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
def test_a2a_gemm(world_size):
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
