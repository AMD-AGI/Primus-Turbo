"""EP dispatch × grouped-gemm overlap benchmark.

Compares five approaches for the MoE dispatch → grouped-gemm pipeline:

    Baseline (no overlap)
    ─────────────────────
    * ``baseline``            — primus_turbo.pytorch.cco._fused_dispatch_permute
                                followed by grouped_gemm_triton_kernel on the
                                *same* stream. This is the DeepEP-style fused
                                dispatch + gemm serial reference.

    Overlap A: 2-stream tile-polling overlap
    ────────────────────────────────────────
    * ``fused_dispatch_overlap`` — ``_fused_dispatch_permute`` on a dedicated
                                comm stream, grouped_gemm on a dedicated
                                compute stream. Each gemm tile polls
                                ``group_tail_idx`` so it doesn't start on an
                                expert whose tokens haven't arrived yet. HIP
                                schedules the two kernels concurrently.
    * ``expert_group_overlap``  — same 2-stream structure, but uses
                                ``_expert_grouped_dispatch_permute`` instead.
                                Because the grouped dispatch fills
                                ``tail[e]`` in group order, gemm tiles for
                                early experts unlock earlier than with
                                ``_fused_dispatch_permute``. Responds to
                                ``num_experts_per_group``: smaller groups →
                                earlier first-tile unlock.

    Overlap B: per-group tail-signaled pipelining
    ─────────────────────────────────────────────
    * ``expert_group_1s``       uses ``_expert_grouped_dispatch_permute`` to
                                dispatch tokens in expert-group order and a
                                per-group wait → gemm loop on **one** compute
                                stream.
    * ``expert_group_ms``       same dispatch, but the per-group waits/gemms
                                round-robin across **K + 1** streams (the comm
                                stream itself is the last slot of the pool, so
                                its HW queue is reused for compute after
                                dispatch finishes).

The three ``*_group_*`` variants sweep
``num_experts_per_group ∈ {1, 2, 4, 8, 16, 32}``.

Timing mode: single-iter (``_bench_joined``) — stream join happens after
every iteration, i.e. the measured wall time includes the final stream
synchronization.

Usage
─────
    OMP_NUM_THREADS=8 torchrun --standalone --nproc_per_node=8 \\
        tests/pytorch/cco/test_ep.py \\
        --num-tokens 4096 --hidden 7168 --num-topk 8 --num-experts 256
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Callable, Dict, List, Tuple

import torch
import torch.distributed as dist
import triton
import triton.language as tl

from primus_turbo.pytorch.cco import (
    _expert_grouped_dispatch_permute,
    _fused_dispatch_permute,
)
from primus_turbo.triton.grouped_gemm.grouped_gemm_kernel import (
    _get_num_cus,
    grouped_gemm_triton_kernel,
)

# MI300X exposes 4 HW queues by default; expose 8 so our (comm + compute +
# default) streams can each land on a separate hardware queue.
os.environ.setdefault("GPU_MAX_HW_QUEUES", "8")


# ═══════════════════════════════════════════════════════════════════════════════
# Triton wait kernel — single-block busy-wait on a contiguous expert range.
# Used by the expert-group overlap schemes to gate each per-group gemm on the
# tail counter written by ``_expert_grouped_dispatch_permute``.
# ═══════════════════════════════════════════════════════════════════════════════


@triton.jit
def _wait_group_kernel(
    tail_ptr,
    expected_ptr,
    group_start,
    GROUP_LEN: tl.constexpr,
):
    for i in tl.static_range(GROUP_LEN):
        e = group_start + i
        expected = tl.load(expected_ptr + e).to(tl.int32)
        req = expected
        while req >= 0:
            current = tl.atomic_add(tail_ptr + e, 0, sem="acquire", scope="gpu")
            if current >= req:
                req = -1
            tl.inline_asm_elementwise(
                "s_sleep 128\n// dummy $0",
                "=r",
                [],
                dtype=tl.int32,
                is_pure=False,
                pack=1,
            )


def wait_group_ready(
    tail: torch.Tensor,
    expected: torch.Tensor,
    group_start: int,
    group_len: int,
) -> None:
    """Launch a single-block wait kernel on the current stream."""
    assert tail.dtype == torch.int32 and expected.dtype == torch.int32
    _wait_group_kernel[(1,)](
        tail,
        expected,
        group_start,
        GROUP_LEN=group_len,
        num_warps=1,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Group-offs builders — used to slice the permuted-x / weight tensors so the
# per-group gemm only computes the tokens of the current group.
# ═══════════════════════════════════════════════════════════════════════════════


def _build_full_offs(expert_counts: torch.Tensor) -> torch.Tensor:
    offs = torch.empty((expert_counts.numel() + 1,), dtype=torch.int64, device=expert_counts.device)
    offs[0] = 0
    offs[1:] = torch.cumsum(expert_counts.to(torch.int64), dim=0)
    return offs


def _build_per_group_offs(expert_counts: torch.Tensor, num_experts_per_group: int) -> List[torch.Tensor]:
    """Build a list of ``[E+1]`` offsets where, at index ``g``, only experts
    in group ``g`` have non-zero counts. Enables running a gemm that touches
    only one group's tokens while using the standard kernel signature.
    """
    E = expert_counts.numel()
    device = expert_counts.device
    num_groups = (E + num_experts_per_group - 1) // num_experts_per_group
    offs_list: List[torch.Tensor] = []
    for g in range(num_groups):
        s = g * num_experts_per_group
        e = min((g + 1) * num_experts_per_group, E)
        masked = torch.zeros(E, dtype=torch.int64, device=device)
        masked[s:e] = expert_counts[s:e].to(torch.int64)
        offs = torch.empty(E + 1, dtype=torch.int64, device=device)
        offs[0] = 0
        offs[1:] = masked.cumsum(0)
        offs_list.append(offs)
    return offs_list


# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark harness
# ═══════════════════════════════════════════════════════════════════════════════


def _bench_joined(
    fn: Callable[[], None],
    join_fn: Callable[[torch.cuda.Stream], None],
    num_warmups: int,
    num_tests: int,
) -> float:
    """Single-iter bench that joins streams to ``current`` after each ``fn``."""
    torch.cuda.synchronize()
    current = torch.cuda.current_stream()
    for _ in range(num_warmups):
        fn()
        join_fn(current)
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(num_tests)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(num_tests)]
    for s, e in zip(starts, ends):
        s.record()
        fn()
        join_fn(current)
        e.record()
    torch.cuda.synchronize()
    return sum(s.elapsed_time(e) for s, e in zip(starts, ends)) / num_tests / 1e3


# ═══════════════════════════════════════════════════════════════════════════════
# Runner builders — each returns ``(runners_dict, join_fn)`` where ``join_fn``
# is how the harness makes the current stream wait on all launched work.
# ═══════════════════════════════════════════════════════════════════════════════


def _build_runners(
    *,
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    weight: torch.Tensor,
    group: dist.ProcessGroup,
    num_experts: int,
    num_experts_per_rank: int,
    num_experts_per_group: int,
    num_sms: int,
    total_sms: int,
    expected_counts: torch.Tensor,
    full_offs: torch.Tensor,
    per_group_offs: List[torch.Tensor],
    num_groups: int,
    group_starts: List[int],
    group_lens: List[int],
    compute_stream_single: torch.cuda.Stream,
    compute_streams_multi: List[torch.cuda.Stream],
    persistent_tail: torch.Tensor,
    fused_comm_stream: torch.cuda.Stream,
    fused_comp_stream: torch.cuda.Stream,
) -> Tuple[Dict[str, Callable[[], None]], Callable[[torch.cuda.Stream], None]]:
    len(compute_streams_multi)
    # Reserve CUs for dispatch with EXTRA headroom. The fused_dispatch_overlap
    # path runs two concurrent persistent kernels on plain (non-CU-masked)
    # streams; if the gemm's persistent blocks saturate all but
    # ``num_sms`` CUs, the dispatch kernel chain can deadlock — notify_dispatch
    # needs a few CUs for its cross-rank symm-mem barrier BEFORE
    # fused_dispatch_permute's 48 blocks can be admitted, and any rank that
    # can't admit notify_dispatch stalls the barrier for everyone else.
    #
    # Leaving ``2 * num_sms`` CUs off the gemm gives dispatch plenty of
    # room. On MI300X (304 CUs, num_sms=48) this is 208 CUs for gemm, 96
    # for dispatch.
    safe_gemm_sms = max(16, total_sms - 2 * num_sms)

    def _fresh_tail() -> torch.Tensor:
        persistent_tail.zero_()
        return persistent_tail

    # ─────────────────── Baseline: serial fused_dispatch + gemm ───────────────────
    def baseline():
        """primus_turbo.pytorch.cco._fused_dispatch_permute + grouped_gemm,
        both on the current stream (no overlap). This is the DeepEP-style
        reference implementation.
        """
        tail = _fresh_tail()
        (out, _) = _fused_dispatch_permute(
            x,
            group,
            tail,
            topk_idx,
            topk_weights,
            num_experts,
            num_sms=num_sms,
        )
        grouped_gemm_triton_kernel(
            out[0],
            weight,
            tail,
            full_offs,
            num_sms=total_sms,
        )

    # ─────────────────── Overlap A: fused_dispatch + gemm on separate streams ───────────────────
    def fused_dispatch_overlap():
        """Dispatch on a dedicated comm stream, gemm on a dedicated compute
        stream — plain ``torch.cuda.Stream`` objects (no CU mask).

        Uses ``_fused_dispatch_permute`` (unmodified) and overlaps the gemm
        via ``group_tail_idx`` tile-polling: every gemm tile checks that its
        expert's tail counter has advanced past its token range before
        starting, so correctness is independent of whether dispatch has
        finished globally. The HIP scheduler multiplexes CUs between the
        two concurrent kernels.
        """
        current = torch.cuda.current_stream()
        tail = _fresh_tail()
        fused_comm_stream.wait_stream(current)
        with torch.cuda.stream(fused_comm_stream):
            (out, _) = _fused_dispatch_permute(
                x,
                group,
                tail,
                topk_idx,
                topk_weights,
                num_experts,
                num_sms=num_sms,
            )
        permuted_x = out[0]
        # comp stream must wait for ``current`` so gemm reads the freshly
        # zeroed ``persistent_tail`` — otherwise it can observe leftover
        # M_g values from the previous iteration, think dispatch is done,
        # and read stale ``permuted_x``.
        fused_comp_stream.wait_stream(current)
        permuted_x.record_stream(fused_comp_stream)
        tail.record_stream(fused_comp_stream)
        with torch.cuda.stream(fused_comp_stream):
            grouped_gemm_triton_kernel(
                permuted_x,
                weight,
                tail,
                full_offs,
                num_sms=safe_gemm_sms,
            )

    # ─────────────────── Overlap A′: fused_dispatch_overlap variant with expert-grouped dispatch ───────────────────
    def expert_group_overlap():
        """Same 2-stream structure as ``fused_dispatch_overlap`` but uses
        ``_expert_grouped_dispatch_permute`` (phase-ordered dispatch).

        The grouped gemm is launched ONCE over the full permuted range and
        polls ``group_tail_idx`` per tile. Because the expert-grouped
        dispatch fills ``tail[e]`` in group order (group 0 experts'
        counters reach their final value first, then group 1, ...), gemm
        tiles for early experts unlock earlier than they would under the
        monolithic ``_fused_dispatch_permute``. This variant therefore
        responds to ``num_experts_per_group``: smaller groups → earlier
        unlock of the first tiles.

        Uses the SAME two streams (``fused_comm_stream`` / ``fused_comp_stream``)
        and the same CU-reservation ``safe_gemm_sms`` as ``fused_dispatch_overlap``,
        so results are directly comparable — the only knob that changes is
        which dispatch kernel fills ``tail``.
        """
        current = torch.cuda.current_stream()
        tail = _fresh_tail()
        fused_comm_stream.wait_stream(current)
        with torch.cuda.stream(fused_comm_stream):
            (out, _) = _expert_grouped_dispatch_permute(
                x,
                group,
                tail,
                topk_idx,
                topk_weights,
                num_experts,
                num_experts_per_group=num_experts_per_group,
                num_sms=num_sms,
            )
        permuted_x = out[0]
        fused_comp_stream.wait_stream(current)
        permuted_x.record_stream(fused_comp_stream)
        tail.record_stream(fused_comp_stream)
        with torch.cuda.stream(fused_comp_stream):
            grouped_gemm_triton_kernel(
                permuted_x,
                weight,
                tail,
                full_offs,
                num_sms=safe_gemm_sms,
            )

    # ─────────────────── Overlap B: per-group tail-signaled pipelining ───────────────────
    # Dedicated comm stream for the expert-group variant so dispatch and
    # compute land on separate HW queues.
    expert_group_comm_stream = torch.cuda.Stream()

    def expert_group_1s():
        """expert_grouped_dispatch_permute on a comm stream, per-group
        wait→gemm sequentially on **one** compute stream."""
        current = torch.cuda.current_stream()
        tail = _fresh_tail()
        expert_group_comm_stream.wait_stream(current)
        with torch.cuda.stream(expert_group_comm_stream):
            (out, _) = _expert_grouped_dispatch_permute(
                x,
                group,
                tail,
                topk_idx,
                topk_weights,
                num_experts,
                num_experts_per_group=num_experts_per_group,
                num_sms=num_sms,
            )
        permuted_x = out[0]
        compute_stream_single.wait_stream(current)
        permuted_x.record_stream(compute_stream_single)
        tail.record_stream(compute_stream_single)
        with torch.cuda.stream(compute_stream_single):
            for g in range(num_groups):
                wait_group_ready(tail, expected_counts, group_starts[g], group_lens[g])
                grouped_gemm_triton_kernel(
                    permuted_x,
                    weight,
                    None,
                    per_group_offs[g],
                    num_sms=total_sms,
                )

    # Unified stream pool for expert_group_ms: the dispatch comm stream is
    # placed LAST in the round-robin pool of (K + 1) streams. Dispatch is
    # enqueued on that last stream first; subsequent per-group gemms are
    # then distributed round-robin across **all K + 1 streams**, including
    # the comm stream. The gemm(s) landing on the comm stream naturally
    # wait for dispatch to finish (via HIP in-stream ordering) and reuse
    # the comm HW queue for compute instead of letting it sit idle for
    # the rest of the iteration.
    expert_group_ms_streams: List[torch.cuda.Stream] = list(compute_streams_multi) + [
        expert_group_comm_stream
    ]
    Kp1 = len(expert_group_ms_streams)

    def expert_group_ms():
        """Unified K+1 stream pool:

            stream[0..K-1] : compute-only, round-robin gemms
            stream[K]      : dispatch kernel first, then any gemms whose
                             round-robin slot lands here (at least the
                             last group).

        Every stream carries at least one gemm, so no HW queue is left
        idle after dispatch completes. Per-group ``wait_group_ready``
        still gates the gemm kernels on the dispatch's expert-tail
        counter, so correctness is independent of scheduling.
        """
        current = torch.cuda.current_stream()
        tail = _fresh_tail()
        # Dispatch goes on the LAST stream of the pool. Its completion is
        # tracked by ``expert_group_comm_stream`` (==
        # ``expert_group_ms_streams[-1]``), so any gemm round-robin'd to
        # that same stream is automatically ordered after dispatch.
        expert_group_comm_stream.wait_stream(current)
        with torch.cuda.stream(expert_group_comm_stream):
            (out, _) = _expert_grouped_dispatch_permute(
                x,
                group,
                tail,
                topk_idx,
                topk_weights,
                num_experts,
                num_experts_per_group=num_experts_per_group,
                num_sms=num_sms,
            )
        permuted_x = out[0]
        # Every compute-pool stream needs to see the dependency on
        # ``current`` (the caller's data-ready event). The comm stream
        # already waited above; the rest wait here.
        for s in expert_group_ms_streams:
            if s is expert_group_comm_stream:
                continue
            s.wait_stream(current)
        for s in expert_group_ms_streams:
            permuted_x.record_stream(s)
            tail.record_stream(s)
        for g in range(num_groups):
            s = expert_group_ms_streams[g % Kp1]
            with torch.cuda.stream(s):
                wait_group_ready(tail, expected_counts, group_starts[g], group_lens[g])
                grouped_gemm_triton_kernel(
                    permuted_x,
                    weight,
                    None,
                    per_group_offs[g],
                    num_sms=total_sms,
                )

    def _wait_all(current: torch.cuda.Stream) -> None:
        current.wait_stream(fused_comm_stream)
        current.wait_stream(fused_comp_stream)
        current.wait_stream(expert_group_comm_stream)
        current.wait_stream(compute_stream_single)
        for s in compute_streams_multi:
            current.wait_stream(s)

    runners: Dict[str, Callable[[], None]] = {
        "baseline": baseline,
        "fused_dispatch_overlap": fused_dispatch_overlap,
        "expert_group_overlap": expert_group_overlap,
        "expert_group_1s": expert_group_1s,
        "expert_group_ms": expert_group_ms,
    }
    return runners, _wait_all


# ═══════════════════════════════════════════════════════════════════════════════
# Correctness: verify ``_expert_grouped_dispatch_permute`` matches
# ``_fused_dispatch_permute`` on per-expert counters, tail and permuted tokens.
# ═══════════════════════════════════════════════════════════════════════════════


def _gather_permuted_by_expert(permuted_x: torch.Tensor, counts: torch.Tensor) -> List[torch.Tensor]:
    slices: List[torch.Tensor] = []
    start = 0
    for c in counts.tolist():
        slices.append(permuted_x[start : start + c])
        start += c
    return slices


def _token_multiset_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.shape != b.shape:
        return False
    if a.numel() == 0:
        return True
    a_h = a.reshape(a.shape[0], -1).to(torch.int32).sum(dim=-1)
    b_h = b.reshape(b.shape[0], -1).to(torch.int32).sum(dim=-1)
    a_sorted, _ = torch.sort(a_h)
    b_sorted, _ = torch.sort(b_h)
    return bool(torch.equal(a_sorted, b_sorted))


def _run_correctness(
    *,
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    group: dist.ProcessGroup,
    num_experts: int,
    num_experts_per_rank: int,
    num_experts_per_group: int,
    num_sms: int,
    rank: int,
) -> None:
    tail_ref = torch.zeros((num_experts_per_rank,), dtype=torch.int32, device="cuda")
    (ref_tuple, _) = _fused_dispatch_permute(
        x,
        group,
        tail_ref,
        topk_idx,
        topk_weights,
        num_experts,
        num_sms=num_sms,
    )
    group.barrier()

    tail_pipe = torch.zeros((num_experts_per_rank,), dtype=torch.int32, device="cuda")
    (pipe_tuple, _) = _expert_grouped_dispatch_permute(
        x,
        group,
        tail_pipe,
        topk_idx,
        topk_weights,
        num_experts,
        num_experts_per_group=num_experts_per_group,
        num_sms=num_sms,
    )
    group.barrier()

    ref_counts = ref_tuple[5]
    pipe_counts = pipe_tuple[5]
    assert torch.equal(ref_counts, pipe_counts), f"[rank {rank}] per-expert counter mismatch"
    assert torch.equal(tail_ref, tail_pipe), f"[rank {rank}] tail mismatch"

    ref_slices = _gather_permuted_by_expert(ref_tuple[0], ref_counts)
    pipe_slices = _gather_permuted_by_expert(pipe_tuple[0], pipe_counts)
    for le in range(num_experts_per_rank):
        assert _token_multiset_equal(
            ref_slices[le], pipe_slices[le]
        ), f"[rank {rank}] expert {le} multiset mismatch"

    if rank == 0:
        print(
            f"[correctness] OK num_experts_per_group={num_experts_per_group} "
            f"(num_experts_per_rank={num_experts_per_rank})",
            flush=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Sweep driver
# ═══════════════════════════════════════════════════════════════════════════════


def _run_single_config(
    *,
    args: argparse.Namespace,
    num_experts_per_group: int,
    rank: int,
    num_ranks: int,
    group: dist.ProcessGroup,
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    weight: torch.Tensor,
    persistent_tail: torch.Tensor,
    compute_stream_single: torch.cuda.Stream,
    compute_streams_multi: List[torch.cuda.Stream],
    fused_comm_stream: torch.cuda.Stream,
    fused_comp_stream: torch.cuda.Stream,
    run_baseline: bool,
    run_fused_overlap: bool,
) -> Dict[str, float]:
    num_experts_per_rank = args.num_experts // num_ranks
    num_groups = (num_experts_per_rank + num_experts_per_group - 1) // num_experts_per_group
    total_sms = _get_num_cus()

    # Warmup dispatch to capture expected_counts. These are deterministic
    # for the fixed topk_idx so we only need to compute them once.
    warm_tail = torch.zeros((num_experts_per_rank,), dtype=torch.int32, device="cuda")
    (warm_tuple, _) = _expert_grouped_dispatch_permute(
        x,
        group,
        warm_tail,
        topk_idx,
        topk_weights,
        args.num_experts,
        num_experts_per_group=num_experts_per_group,
        num_sms=args.num_sms,
    )
    group.barrier()
    expected_counts = warm_tuple[5].clone().contiguous()

    full_offs = _build_full_offs(expected_counts)
    per_group_offs = _build_per_group_offs(expected_counts, num_experts_per_group)
    group_starts = [g * num_experts_per_group for g in range(num_groups)]
    group_lens = [
        min((g + 1) * num_experts_per_group, num_experts_per_rank) - group_starts[g]
        for g in range(num_groups)
    ]

    if rank == 0:
        print(
            f"\n────── num_experts_per_group={num_experts_per_group} (num_groups={num_groups}) ──────",
            flush=True,
        )

    runners, join_fn = _build_runners(
        x=x,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        weight=weight,
        group=group,
        num_experts=args.num_experts,
        num_experts_per_rank=num_experts_per_rank,
        num_experts_per_group=num_experts_per_group,
        num_sms=args.num_sms,
        total_sms=total_sms,
        expected_counts=expected_counts,
        full_offs=full_offs,
        per_group_offs=per_group_offs,
        num_groups=num_groups,
        group_starts=group_starts,
        group_lens=group_lens,
        compute_stream_single=compute_stream_single,
        compute_streams_multi=compute_streams_multi,
        persistent_tail=persistent_tail,
        fused_comm_stream=fused_comm_stream,
        fused_comp_stream=fused_comp_stream,
    )

    # Filter runners: the baseline and fused_dispatch_overlap do not depend
    # on num_experts_per_group, so we only run them on the first pass of
    # the sweep (caller flags).
    active_runners: Dict[str, Callable[[], None]] = {}
    for name, fn in runners.items():
        if name == "baseline" and not run_baseline:
            continue
        if name == "fused_dispatch_overlap" and not run_fused_overlap:
            continue
        active_runners[name] = fn

    results: Dict[str, float] = {}
    for name, fn in active_runners.items():
        dist.barrier(group=group)
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        ) as prof:
            try:
                ms_single = _bench_joined(fn, join_fn, args.warmups, args.iters)
            except Exception as exc:
                if rank == 0:
                    print(f"  {name:24s}: FAILED ({exc})", flush=True)
                results[name] = float("nan")
                continue
            dist.barrier(group=group)
            results[name] = ms_single
        if rank == 0:
            print(f"  {name:24s}: {ms_single*1e6:8.1f}us", flush=True)
            os.makedirs("prof", exist_ok=True)
            prof.export_chrome_trace(f"prof/num_exp_{num_experts_per_group}_{name}.gz")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Main entry
# ═══════════════════════════════════════════════════════════════════════════════


def _test_entry(local_rank: int, world_size: int, args: argparse.Namespace) -> None:
    try:
        from utils import init_dist  # type: ignore
    except ImportError:
        from tests.pytorch.cco.utils import init_dist  # type: ignore

    rank, num_ranks, group = init_dist(local_rank, world_size)
    torch.manual_seed(rank + 17)

    num_experts_per_rank = args.num_experts // num_ranks
    if rank == 0:
        print(
            f"[config] num_tokens={args.num_tokens} hidden={args.hidden} "
            f"num_topk={args.num_topk} num_experts={args.num_experts} "
            f"num_experts_per_rank={num_experts_per_rank} num_sms={args.num_sms}",
            flush=True,
        )

    # Shared tensors — reused across all sweeps.
    x = (
        torch.arange(args.num_tokens, device="cuda", dtype=torch.int32).view(-1, 1)
        + rank * args.num_tokens
        + 1
    )
    x = x.to(torch.bfloat16).expand(-1, args.hidden).contiguous()

    scores = torch.randn((args.num_tokens, args.num_experts), dtype=torch.float32, device="cuda").abs() + 1.0
    topk_idx = torch.topk(scores, args.num_topk, dim=-1, largest=True, sorted=False)[1]
    topk_weights = torch.ones((args.num_tokens, args.num_topk), dtype=torch.float32, device="cuda") * (
        rank + 1.0
    )
    weight = torch.randn(
        (num_experts_per_rank, args.hidden, args.hidden_out),
        dtype=torch.bfloat16,
        device="cuda",
    )

    persistent_tail = torch.zeros((num_experts_per_rank,), dtype=torch.int32, device="cuda")

    # Streams for expert-group overlap (created once, reused across configs).
    compute_stream_single = torch.cuda.Stream()
    compute_streams_multi = [torch.cuda.Stream() for _ in range(args.num_multi_streams)]

    # Plain (non-masked) dedicated streams for the fused_dispatch_overlap
    # variant. The HIP scheduler multiplexes CUs between them based on
    # their HW-queue priority and CU availability.
    #
    # ``fused_comm_stream`` is HIGH-priority so its kernel chain (notify_dispatch
    # cross-rank barrier + fused_dispatch_permute) gets CU admission before
    # the low-priority persistent gemm on ``fused_comp_stream`` can hold
    # the scheduler hostage. Without this, the tile-polling overlap can
    # distributed-deadlock: one rank's gemm admits first, fills its CUs,
    # starves notify_dispatch, and every other rank then waits forever on
    # that rank's symm-mem barrier signal.
    fused_comm_stream = torch.cuda.Stream(priority=-1)
    fused_comp_stream = torch.cuda.Stream(priority=0)

    # Correctness once per num_experts_per_group.
    for g in args.sweep:
        if num_experts_per_rank % g != 0:
            if rank == 0:
                print(
                    f"[skip correctness] num_experts_per_rank={num_experts_per_rank} not divisible by "
                    f"num_experts_per_group={g}",
                    flush=True,
                )
            continue
        _run_correctness(
            x=x,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            group=group,
            num_experts=args.num_experts,
            num_experts_per_rank=num_experts_per_rank,
            num_experts_per_group=g,
            num_sms=args.num_sms,
            rank=rank,
        )

    # Performance sweep. baseline and fused_dispatch_overlap are
    # independent of num_experts_per_group so we only time them on the
    # first valid config.
    sweep_results: List[Tuple[int, Dict[str, float]]] = []
    baseline_done = False
    fused_overlap_done = False
    for g in args.sweep:
        if num_experts_per_rank % g != 0:
            continue
        res = _run_single_config(
            args=args,
            num_experts_per_group=g,
            rank=rank,
            num_ranks=num_ranks,
            group=group,
            x=x,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            weight=weight,
            persistent_tail=persistent_tail,
            compute_stream_single=compute_stream_single,
            compute_streams_multi=compute_streams_multi,
            fused_comm_stream=fused_comm_stream,
            fused_comp_stream=fused_comp_stream,
            run_baseline=not baseline_done,
            run_fused_overlap=not fused_overlap_done,
        )
        if "baseline" in res:
            baseline_done = True
        if "fused_dispatch_overlap" in res:
            fused_overlap_done = True
        sweep_results.append((g, res))

    # ── Final summary ────────────────────────────────────────────────────
    if rank == 0:
        _print_summary(sweep_results)

    # Cleanup: sync, barrier, then tear down the process group. All
    # streams are plain ``torch.cuda.Stream`` objects that PyTorch owns and
    # will free as part of the caching allocator shutdown.
    torch.cuda.synchronize()
    dist.barrier()
    dist.destroy_process_group()
    torch.cuda.synchronize()

    # Hard-exit with success status.
    #
    # The default Python interpreter shutdown still occasionally SIGSEGVs
    # one worker on AMD because several finalizers race against each
    # other (module-level tensor caches, PyTorch caching allocator, NCCL /
    # symmetric-memory). The benchmark has already printed all results by
    # this point, so we bypass finalizers entirely. The parent ``spawn``
    # context observes a clean exit code.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def _print_summary(sweep_results: List[Tuple[int, Dict[str, float]]]) -> None:
    if not sweep_results:
        return

    # Collect baseline + fused_dispatch_overlap once (they are only timed
    # on the first valid sweep config).
    baseline_us = float("nan")
    fused_us = float("nan")
    for _, res in sweep_results:
        if "baseline" in res:
            baseline_us = res["baseline"]
        if "fused_dispatch_overlap" in res:
            fused_us = res["fused_dispatch_overlap"]

    print("\n══════════════════════ fixed variants (us) ══════════════════════")
    print(f"{'variant':30s}  {'time':>12s}")
    print(f"{'baseline':30s}  {baseline_us*1e6:>12.1f}")
    print(f"{'fused_dispatch_overlap':30s}  {fused_us*1e6:>12.1f}")

    print("\n══════════════════════ expert-group sweep (us) ══════════════════════")
    bp = baseline_us
    print(
        f"{'num_ep/grp':>10s}  {'expert_group_overlap':>22s}  {'expert_group_1s':>18s}  "
        f"{'expert_group_ms':>18s}  {'speedup_ov':>12s}  {'speedup_1s':>12s}  {'speedup_ms':>12s}"
    )
    for g, res in sweep_results:
        vo = res.get("expert_group_overlap", float("nan"))
        v1 = res.get("expert_group_1s", float("nan"))
        vm = res.get("expert_group_ms", float("nan"))
        spo = bp / vo if vo and vo == vo and bp == bp else float("nan")
        sp1 = bp / v1 if v1 and v1 == v1 and bp == bp else float("nan")
        spm = bp / vm if vm and vm == vm and bp == bp else float("nan")
        print(
            f"{g:>10d}  {vo*1e6:>22.1f}  {v1*1e6:>18.1f}  {vm*1e6:>18.1f}  "
            f"{spo:>11.2f}x  {sp1:>11.2f}x  {spm:>11.2f}x"
        )

    # Best-config report across all variants.
    all_candidates: List[Tuple[str, int, float]] = []
    for g, res in sweep_results:
        for k in ("expert_group_overlap", "expert_group_1s", "expert_group_ms"):
            all_candidates.append((k, g, res.get(k, float("nan"))))
    if baseline_us == baseline_us:  # not NaN
        all_candidates.append(("baseline", 0, baseline_us))
    if fused_us == fused_us:
        all_candidates.append(("fused_dispatch_overlap", 0, fused_us))

    best = min(
        (c for c in all_candidates if c[2] == c[2]),
        key=lambda c: c[2],
        default=None,
    )
    if best is not None:
        print(f"\n[winner] {best[0]:>24s} @ num_ep/grp={best[1]:>2d}: {best[2]*1e6:.1f} us")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-processes", type=int, default=8)
    p.add_argument("--num-tokens", type=int, default=4096)
    p.add_argument("--hidden", type=int, default=7168)
    p.add_argument("--hidden-out", type=int, default=4096)
    p.add_argument("--num-topk", type=int, default=8)
    p.add_argument("--num-experts", type=int, default=256)
    p.add_argument(
        "--num-sms",
        type=int,
        default=48,
        help="CUs used by dispatch (sender + receiver). Must be even.",
    )
    p.add_argument(
        "--num-multi-streams",
        type=int,
        default=3,
        help="Number of compute streams for expert_group_ms variant.",
    )
    p.add_argument("--warmups", type=int, default=10)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument(
        "--sweep",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32],
        help="num_experts_per_group values to sweep (for expert_group_* only).",
    )
    return p.parse_args()


if __name__ == "__main__":
    _args = _parse_args()
    torch.multiprocessing.spawn(
        _test_entry,
        args=(_args.num_processes, _args),
        nprocs=_args.num_processes,
    )
