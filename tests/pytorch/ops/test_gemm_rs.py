###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fused GEMM + reduce-scatter kernel (``fused_collective_gemm.gemm_rs``, the MLA latent backward),
with the split-K weight gradients it can run in the same launch, against fp32 references.

Every rank draws its own head gradients and weight rows. Reference for the latent gradients: this
rank's partial ``[gq @ Wq | gk_nope @ Wk + gv @ Wv | sum over heads of gk_pe]`` in fp32, rounded to
bf16 as the kernel stages it, summed over the ranks in fp32; this rank's rows of the sum. Weight
gradients, as the MLA op runs them: ``gq^T @ lq`` (one operand) and ``[gk_nope | gv]^T @ lc`` (a pair
of views) in fp32. Outputs start as NaN, so an element the kernel does not write fails.

The pytest entry launches this file under torchrun for every world size the node has GPUs for:

    pytest tests/pytorch/ops/test_gemm_rs.py
    torchrun --nproc_per_node 8 tests/pytorch/ops/test_gemm_rs.py [--quick]
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
    "small": (256, 256, lambda n: 4),
    "deepseek-v3": (1536, 512, lambda n: 128 // n),
}


def _randn(*shape, seed, scale=1.0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(*shape, device="cuda", generator=gen).mul_(scale).bfloat16()


def _nan(*shape):
    return torch.full(shape, float("nan"), dtype=torch.bfloat16, device="cuda")


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
    """Fill both banks of the workspace with bf16 NaN: a reduce unit summing a partial before its GEMM
    unit wrote it this epoch shows up as NaN."""
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


def _weights(hl, cq, ckv, dn, dr, dv, seed):
    """This rank's weight rows (``Wq`` [hl * (dn + dr), Cq], ``Wkv`` [hl, dn + dv, Ckv]) and the GemmRS
    operands built from them: ``Wq^T``, ``B'^T`` (every head's ``Wk_nope`` over an identity block
    summing the heads' ``k_pe`` gradients) and ``Wv^T``."""
    nk = hl * (dn + dr)
    wq = _randn(nk, cq, seed=seed, scale=nk**-0.5)
    wkv = _randn(hl, dn + dv, ckv, seed=seed + 1, scale=(hl * (dn + dv)) ** -0.5)
    wk_t = torch.zeros(ckv + dr, hl, dn + dr, dtype=torch.bfloat16, device="cuda")
    wk_t[:ckv, :, :dn] = wkv[:, :dn].permute(2, 0, 1)
    wk_t[ckv:, :, dn:] = torch.eye(dr, dtype=torch.bfloat16, device="cuda")[:, None, :]
    wv_t = wkv[:, dn:].permute(2, 0, 1).reshape(ckv, hl * dv).contiguous()
    return wq, wkv, (wq.t().contiguous(), wk_t.view(ckv + dr, nk), wv_t)


def _kv_wgrad_ref(gk, gv, lc, dn):
    return torch.einsum("rhd,rc->hdc", torch.cat((gk[..., :dn], gv), dim=-1).float(), lc.float())


def check_gemm_rs(
    group,
    failures,
    *,
    dims,
    T,
    dn=128,
    dr=64,
    dv=128,
    hl=None,
    kv=False,
    q=False,
    swap=False,
    flag_region=5,
    repeats=2,
    poison=False,
    **tuning,
):
    """``T`` local rows per rank, ``hl`` local heads (default from ``DIMS``); ``kv`` / ``q`` add the kv
    / q weight gradients to the launch, as ``wgrad`` / ``lead_wgrad`` (the other way round with
    ``swap``)."""
    from primus_turbo.flydsl.fused_collective_gemm.gemm_rs_kernel import (
        gemm_rs,
        gemm_rs_bank_bytes,
    )
    from primus_turbo.flydsl.fused_collective_gemm.workspace import (
        get_collective_gemm_workspace,
    )

    n, rank = group.size(), group.rank()
    cq, ckv, hl_of = DIMS[dims]
    hl = hl_of(n) if hl is None else hl
    R, nk = n * T, hl * (dn + dr)
    tag = f"[{dims} hl={hl} dn={dn} dr={dr} dv={dv} T={T} kv={kv} q={q} swap={swap} region={flag_region} {tuning}]"
    kv_slot, q_slot = ("lead_wgrad", "wgrad") if swap else ("wgrad", "lead_wgrad")
    wq, wkv, operands = _weights(hl, cq, ckv, dn, dr, dv, seed=11 + 7 * rank)
    workspace = get_collective_gemm_workspace(
        group, gemm_rs_bank_bytes(R, cq, ckv, dr, tuning.get("BM", 256))
    )
    for it in range(repeats):
        seed = 1000 * it + 100 * rank
        gq = _randn(R, nk, seed=seed, scale=0.1)
        gk = _randn(R, hl, dn + dr, seed=seed + 1, scale=0.1)
        gv = _randn(R, hl, dv, seed=seed + 2, scale=0.1)
        d_q_latent, d_kv_latent = _nan(T, cq), _nan(T, ckv + dr)
        extra = {}
        if kv:
            lc = _randn(R, ckv, seed=seed + 3)
            d_w_kv = _nan(hl, dn + dv, ckv)
            extra[kv_slot] = ((gk[..., :dn], gv), lc, d_w_kv)
        if q:
            lq = _randn(R, cq, seed=seed + 4)
            d_w_q = _nan(nk, cq)
            extra[q_slot] = (gq, lq, d_w_q)
        if poison:
            _poison_banks(group)
            # stagger the ranks so peers' partials land well after this rank's reduce units start
            torch.cuda._sleep(((rank + it) % n) * 200_000)
        gemm_rs(
            gq,
            gk.view(R, nk),
            gv.view(R, hl * dv),
            *operands,
            d_q_latent,
            d_kv_latent,
            workspace,
            dn=dn,
            dr=dr,
            dv=dv,
            flag_region=flag_region,
            **extra,
            **tuning,
        )
        partial = torch.cat(
            (
                gq.float() @ wq.float(),
                gk[..., :dn].reshape(R, hl * dn).float() @ wkv[:, :dn].reshape(hl * dn, ckv).float()
                + gv.view(R, hl * dv).float() @ wkv[:, dn:].reshape(hl * dv, ckv).float(),
                gk[..., dn:].float().sum(dim=1),
            ),
            dim=1,
        )
        partial = partial.bfloat16().float()
        dist.all_reduce(partial, group=group)
        mine = partial[rank * T : (rank + 1) * T]
        _close("d_q_latent", d_q_latent, mine[:, :cq], failures, f"{tag} call {it}")
        _close("d_kv_latent", d_kv_latent, mine[:, cq:], failures, f"{tag} call {it}")
        if kv:
            _close("d_w_kv", d_w_kv, _kv_wgrad_ref(gk, gv, lc, dn), failures, f"{tag} call {it}")
        if q:
            _close("d_w_q", d_w_q, gq.float().t() @ lq.float(), failures, f"{tag} call {it}")


def run_suite(quick):
    group = dist.group.WORLD
    n = group.size()
    failures = []
    dims_list = ["small"] if quick else list(DIMS)

    def check(**kw):
        check_gemm_rs(group, failures, **kw)

    # T = 300: reduce units straddle m-tiles and a short last one; with the weight gradients in the
    # launch (the op's configuration) and without
    wgrads = [dict(kv=True, q=True)] if quick else [{}, dict(kv=True, q=True)]
    for dims, T, wg in itertools.product(dims_list, [300, 512], wgrads):
        check(dims=dims, T=T, **wg)
    # narrow heads: K chunks straddle heads, so the kv units walk B' whole (zero rows included)
    check(dims="small", T=300, dn=32, dr=32, dv=32, hl=8, q=True)
    # launch knobs: tile rows, reduce-unit rows, GEMM blocks ahead of the reduce, no early / no tail
    # reduce blocks
    knobs = [{"BM": 128}, {"RB": 16}, {"RB": 256}, {"num_lead": 16}, {"num_red_cu": 0}, {"num_tail_red": 0}]
    # weight-gradient placement: kv after GEMM unit 8 (one slice, 4 GEMM blocks ahead so unit 8 is
    # past them), kv on 2 / 3 slices behind the tail, q on short slices, the pair ahead and the single
    # operand behind the tail
    knobs += [
        dict(kv=True, wgrad_s=1, wgrad_at=8, num_lead=4),
        dict(kv=True, wgrad_s=2),
        dict(kv=True, wgrad_s=3),
        dict(q=True, lead_wgrad_rows=128),
        dict(kv=True, q=True, swap=True),
    ]
    for kn in knobs[:2] if quick else knobs:
        for T in (300, 512):
            check(dims="small", T=T, **kn)
    # banks poisoned before every call, flag regions interleaved on one workspace
    for region in (5, 0, 5):
        for T in (300, 512):
            check(dims="small", T=T, kv=True, q=True, flag_region=region, poison=True, repeats=3)
    check(dims=dims_list[-1], T=512, kv=True, q=True, poison=True, repeats=2)
    if not quick:
        # DeepSeek-V3 widths at 2048 local rows: longer K slices
        check(dims="deepseek-v3", T=2048, kv=True, q=True)

    gathered = [None] * n
    dist.all_gather_object(gathered, failures, group=group)
    if group.rank() == 0:
        all_failures = [f"rank {r}: {f}" for r, fs in enumerate(gathered) for f in fs]
        print("\n".join(all_failures) if all_failures else f"gemm_rs: all cases passed (n={n})")
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


@pytest.mark.parametrize("num_cus", [80, 256, 304])
def test_wgrad_splits(num_cus):
    """Split-K slices: every slice non-empty and ``BLOCK_K``-aligned, the slices of a launch at most
    one per CU (they wait for each other)."""
    from primus_turbo.flydsl.fused_collective_gemm.split_k_wgrad import (
        wgrad_lead_splits,
        wgrad_splits,
        wgrad_tail_splits,
        wgrad_tiles,
    )
    from primus_turbo.flydsl.utils.gemm_helper import BLOCK_K

    def covers(R, S, k_split):
        return k_split % BLOCK_K == 0 and (S - 1) * k_split < R <= S * k_split

    for R in list(range(64, 2048, 61)) + [4096, 8192, 65536]:
        for M, N in ((1024, 256), (4096, 512), (16384, 512), (768, 256), (3072, 1536)):
            tiles = wgrad_tiles(M, N, 1)
            S, k_split = wgrad_splits(R, tiles, num_cus)
            assert covers(R, S, k_split) and (S == 1 or S * tiles <= num_cus)
            (S, k_split), (S2, k_split2, t2) = wgrad_tail_splits(R, tiles, num_cus)
            assert covers(R, S, k_split) and (not t2 or covers(R, S2, k_split2))
            assert (tiles - t2) * S + t2 * S2 <= max(num_cus, tiles)
            S, k_split = wgrad_lead_splits(R, tiles, num_cus)
            assert covers(R, S, k_split) and (S == 1 or S * tiles <= num_cus)


@pytest.mark.multigpu
@pytest.mark.skipif(
    not (torch.cuda.is_available() and is_gfx950()), reason="the fused collective GEMM kernels need gfx950"
)
@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_gemm_rs(world_size):
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"needs {world_size} GPUs")
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    cmd = [sys.executable, "-m", "torch.distributed.run", f"--nproc_per_node={world_size}", __file__]
    result = subprocess.run(cmd, cwd=repo_root, env={**os.environ, "PYTHONPATH": repo_root}, timeout=1800)
    assert result.returncode == 0


if __name__ == "__main__":
    main()
