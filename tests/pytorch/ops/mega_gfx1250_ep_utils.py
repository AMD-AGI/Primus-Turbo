###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Helpers shared by the gfx1250 Mega MoE multi-rank suites (bf16 and MXFP8)."""

import torch
import torch.distributed as dist


def gather_rows(t: torch.Tensor, group=None):
    """Every rank's rows of ``t`` rank-major, and where this rank's start; counts may differ."""
    group = group or dist.group.WORLD
    n = torch.tensor([t.shape[0]], device=t.device)
    counts = [torch.empty_like(n) for _ in range(group.size())]
    dist.all_gather(counts, n, group=group)
    counts = [int(c) for c in counts]
    buf = t.new_zeros((max(counts),) + t.shape[1:])
    buf[: t.shape[0]] = t
    out = [torch.empty_like(buf) for _ in counts]
    dist.all_gather(out, buf, group=group)
    return torch.cat([o[:c] for o, c in zip(out, counts)]), sum(counts[: group.rank()])


def poison_mega_heap(group=None) -> None:
    """Fill this rank's dispatch and combine token buffers with NaN bytes (0xFF: NaN as bf16, E4M3 and E8M0), once
    every rank is past its previous call and before any rank starts the next: a read the protocol does not order
    after the write it needs then sees NaN instead of the right bytes left by an identical earlier call."""
    from primus_turbo.flydsl.mega.gfx1250.symm_buffer import _CURRENT_SYMM_BUFFER as symm

    group = group or dist.group.WORLD
    torch.cuda.synchronize()
    dist.barrier(group=group)
    if symm is not None:
        symm.dispatch_token_buffer.view(torch.uint8).fill_(0xFF)
        symm.combine_token_buffer.view(torch.uint8).fill_(0xFF)
    torch.cuda.synchronize()
    dist.barrier(group=group)


def late_rank(delay_rank, group=None, cycles: int = 200_000_000) -> None:
    """Stall the device queue of ``delay_rank`` (None: no rank) for ``cycles`` clocks before the next launch, so
    the other ranks run ahead of it into whatever waits the protocol has."""
    group = group or dist.group.WORLD
    if delay_rank is not None and group.rank() == delay_rank % group.size():
        torch.cuda._sleep(cycles)


def _verdict(got: torch.Tensor, want: torch.Tensor, snr) -> float:
    """``snr(got, want)`` in dB, with -inf for a non-finite ``got`` and an all-zero ``want`` judged exactly: +inf if
    ``got`` is all zero too, -inf if not (an SNR against zero is NaN, which a MIN reduce may drop or keep)."""
    got = got.float()
    if not bool(torch.isfinite(got).all()):
        return float("-inf")
    if not bool(want.any()):
        return float("inf") if not bool(got.any()) else float("-inf")
    return float(snr(got, want))


def assert_snr_all_ranks(case, results, threshold: float, snr, group=None) -> None:
    """Every ``(name, got, want)`` above ``threshold`` dB on every rank (NaN-safe: see ``_verdict``); the weakest
    rank decides, so all ranks agree."""
    group = group or dist.group.WORLD
    values = torch.tensor([_verdict(got, want, snr) for _, got, want in results], device=results[0][1].device)
    dist.all_reduce(values.nan_to_num_(nan=float("-inf")), op=dist.ReduceOp.MIN, group=group)
    for (name, _, _), value in zip(results, values.tolist()):
        case.assertGreater(value, threshold, f"{name} SNR {value:.1f} dB on the weakest rank")


def assert_rows_close_all_ranks(case, results, bound: float, group=None) -> None:
    """Every row of every ``(name, got, want)`` within ``bound`` relative (row norm) of the reference on every rank:
    an aggregate SNR does not move when a few of a rank's rows are stale or missing."""
    group = group or dist.group.WORLD
    worst = []
    for _, got, want in results:
        got, want = got.float().reshape(got.shape[0], -1), want.float().reshape(want.shape[0], -1)
        err = (got - want).norm(dim=1) / want.norm(dim=1).clamp_min(1e-6)
        worst.append(float(err.nan_to_num(nan=float("inf")).max()) if err.numel() else 0.0)
    values = torch.tensor(worst, device=results[0][1].device)
    dist.all_reduce(values, op=dist.ReduceOp.MAX, group=group)
    for (name, _, _), value in zip(results, values.tolist()):
        case.assertLess(value, bound, f"{name}: a row off by {value:.3f} relative on some rank")
