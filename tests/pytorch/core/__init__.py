"""Pipelined dispatch × grouped-gemm overlap benchmark.

Rewritten for clarity — compares the following schedules under one roof:

    Sequential baselines
    ────────────────────
    * `baseline_fused`    — original fused_dispatch_permute + gemm_full (serial)
    * `baseline_pipe`     — optimized expert_grouped_dispatch_permute + gemm_full (serial)

    Overlap schemes (async dispatch + async compute on separate streams)
    ─────────────────────────────────────────────────────────────────────
    * `single_wait`       — dispatch || (wait_all → gemm_full)
    * `per_group_1s`      — dispatch || (N × per-group wait→gemm, 1 compute stream)
    * `per_group_ms`      — dispatch || (N × per-group wait→gemm, K compute streams)
    * `tile_wait`         — dispatch || gemm_full with `group_tail_idx=tail`
                             (each gemm block polls its expert tail per tile)
    * `xcd_masked`        — dispatch and gemm_full isolated on different XCDs
                             via `hipExtStreamCreateWithCUMask`, so the two
                             kernels never compete for HBM bandwidth.

Runs a sweep over ``num_experts_per_group ∈ {1, 2, 4, 8, 16, 32}`` and reports
which configuration delivers the best speedup against
``baseline_pipelined_sequential``.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

_HERE = os.path.dirname(os.path.abspath(__file__))
_SIBLING_CCO = os.path.abspath(os.path.join(_HERE, "..", "cco"))
if _SIBLING_CCO not in sys.path:
    sys.path.insert(0, _SIBLING_CCO)

# MI300X exposes 4 HW queues by default; expose 8 so our (comm + 3 compute +
# default) streams can each land on a separate hardware queue.
os.environ.setdefault("GPU_MAX_HW_QUEUES", "8")


# ═══════════════════════════════════════════════════════════════════════════════
# Triton wait kernel — single-block busy-wait on a contiguous expert range.
# ═══════════════════════════════════════════════════════════════════════════════

import triton
import triton.language as tl


@triton.jit
def _wait_group_kernel(
    tail_ptr,  # [E] int32 — atomic counter bumped by dispatch
    expected_ptr,  # [E] int32 — per-expert expected count
    group_start,  # runtime: first expert id in the group
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
    """Standard prefix-sum over expert counts, used by the full-range gemm."""
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
    """Single-iter bench that joins streams to ``current`` after each ``fn``.

    Captures wall time of one iteration *including* the blocking join.
    ``single`` in the output table.
    """
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


def _bench_pipelined(
    fn: Callable[[], None],
    join_fn: Callable[[torch.cuda.Stream], None],
    num_warmups: int,
    num_tests: int,
    inner_loop: int = 4,
) -> float:
    """Steady-state bench — queues ``inner_loop`` iterations back-to-back
    WITHOUT joining between them, and only joins at the very end.

    This is the right bench mode for overlap schemes: it allows iteration
    ``N+1``'s dispatch to start while iteration ``N``'s compute is still
    running on the compute streams, which is how real MoE layers run.
    """
    torch.cuda.synchronize()
    current = torch.cuda.current_stream()
    for _ in range(num_warmups):
        for _ in range(inner_loop):
            fn()
        join_fn(current)
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(num_tests)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(num_tests)]
    for s, e in zip(starts, ends):
        s.record()
        for _ in range(inner_loop):
            fn()
        join_fn(current)
        e.record()
    torch.cuda.synchronize()
    total = sum(s.elapsed_time(e) for s, e in zip(starts, ends)) / num_tests / 1e3
    return total / inner_loop


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
    comm_stream: torch.cuda.Stream,
    compute_stream_single: torch.cuda.Stream,
    compute_streams_multi: List[torch.cuda.Stream],
    persistent_tail: torch.Tensor,
    masked_streams: Optional[Tuple[int, int]] = None,  # (comm_sms, comp_sms)
    masked_comm_stream: Optional[torch.cuda.Stream] = None,
    masked_comp_stream: Optional[torch.cuda.Stream] = None,
) -> Dict[str, Callable[[], None]]:
    from primus_turbo.pytorch.cco import (
        _expert_grouped_dispatch_permute,
        _fused_dispatch_permute,
    )
    from primus_turbo.triton.grouped_gemm.grouped_gemm_kernel import (
        grouped_gemm_triton_kernel,
    )

    K = len(compute_streams_multi)
    # Reserve CUs for dispatch during overlap to avoid CU oversubscription.
    safe_gemm_sms = max(16, total_sms - num_sms)

    def _fresh_tail() -> torch.Tensor:
        persistent_tail.zero_()
        return persistent_tail

    # ─────────────────── baselines ───────────────────
    def baseline_fused():
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
            BLOCK_SIZE_M=128,
            num_sms=total_sms,
        )

    def baseline_pipe():
        tail = _fresh_tail()
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
        grouped_gemm_triton_kernel(
            out[0],
            weight,
            tail,
            full_offs,
            BLOCK_SIZE_M=128,
            num_sms=total_sms,
        )

    # ─────────────────── overlap schemes ───────────────────
    #
    # Each overlap scheme returns via sync points: the *caller* joins the
    # streams back to the current stream. This lets the bench harness
    # distinguish between "single-iter" measurement (join every iteration)
    # and "steady-state" measurement (queue N iterations back-to-back, join
    # only at the end — this is what realistic model-level overlap sees).
    #
    # Each runner obeys the contract:
    #   * at entry: current stream holds the data dependency on the tail
    #     reset.
    #   * during: launches work on comm_stream and compute_stream(s), which
    #     wait on current at the top.
    #   * at exit: leaves comm_stream and compute_stream(s) busy; does NOT
    #     join. Callers use ``_wait_all`` to join.

    def _wait_all(current):
        current.wait_stream(comm_stream)
        current.wait_stream(compute_stream_single)
        for s in compute_streams_multi:
            current.wait_stream(s)

    def single_wait():
        """wait_all (overlaps with dispatch) → single gemm_full."""
        current = torch.cuda.current_stream()
        tail = _fresh_tail()
        comm_stream.wait_stream(current)
        with torch.cuda.stream(comm_stream):
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
            wait_group_ready(tail, expected_counts, 0, num_experts_per_rank)
            grouped_gemm_triton_kernel(
                permuted_x,
                weight,
                None,
                full_offs,
                BLOCK_SIZE_M=128,
                num_sms=total_sms,
            )

    def per_group_1s():
        """Sequential per-group wait→gemm on 1 compute stream (pipelined)."""
        current = torch.cuda.current_stream()
        tail = _fresh_tail()
        comm_stream.wait_stream(current)
        with torch.cuda.stream(comm_stream):
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
                    BLOCK_SIZE_M=128,
                    num_sms=total_sms,
                )

    def per_group_ms():
        """Per-group wait→gemm spread across K compute streams (round-robin),
        each gemm using a safe CU budget (``(total_sms − num_sms) / K``)."""
        current = torch.cuda.current_stream()
        tail = _fresh_tail()
        comm_stream.wait_stream(current)
        with torch.cuda.stream(comm_stream):
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
        for s in compute_streams_multi:
            s.wait_stream(current)
            permuted_x.record_stream(s)
            tail.record_stream(s)
        per_stream_sms = max(16, safe_gemm_sms // K)
        for g in range(num_groups):
            s = compute_streams_multi[g % K]
            with torch.cuda.stream(s):
                wait_group_ready(tail, expected_counts, group_starts[g], group_lens[g])
                grouped_gemm_triton_kernel(
                    permuted_x,
                    weight,
                    None,
                    per_group_offs[g],
                    BLOCK_SIZE_M=128,
                    num_sms=per_stream_sms,
                )

    def per_group_ms_full():
        """Multi-stream per-group wait→gemm, each gemm launched with the full
        SM count. Blocks in different streams share the GPU's CU pool — the
        HIP scheduler multiplexes, but each gemm sees enough CUs to run at
        near-peak throughput once its predecessors on the same stream finish.

        This is the winning scheme for cross-iteration pipelining: iteration
        N+1's dispatch starts on comm_stream as soon as iteration N's
        compute_streams flush their last gemm, because neither comm_stream
        nor the compute streams are blocked by the current stream between
        iterations.
        """
        current = torch.cuda.current_stream()
        tail = _fresh_tail()
        comm_stream.wait_stream(current)
        with torch.cuda.stream(comm_stream):
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
        for s in compute_streams_multi:
            s.wait_stream(current)
            permuted_x.record_stream(s)
            tail.record_stream(s)
        for g in range(num_groups):
            s = compute_streams_multi[g % K]
            with torch.cuda.stream(s):
                wait_group_ready(tail, expected_counts, group_starts[g], group_lens[g])
                grouped_gemm_triton_kernel(
                    permuted_x,
                    weight,
                    None,
                    per_group_offs[g],
                    BLOCK_SIZE_M=128,
                    num_sms=total_sms,
                )

    def tile_wait():
        """Single gemm_full with per-tile tail poll (``group_tail_idx=tail``)."""
        current = torch.cuda.current_stream()
        tail = _fresh_tail()
        comm_stream.wait_stream(current)
        with torch.cuda.stream(comm_stream):
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
            grouped_gemm_triton_kernel(
                permuted_x,
                weight,
                tail,
                full_offs,
                BLOCK_SIZE_M=128,
                num_sms=safe_gemm_sms,
            )

    runners: Dict[str, Callable[[], None]] = {
        "baseline_fused": baseline_fused,  # no streams, no join needed
        "baseline_pipe": baseline_pipe,
        "single_wait": single_wait,
        "per_group_1s": per_group_1s,
        "per_group_ms": per_group_ms,
        "per_group_ms_full": per_group_ms_full,
        "tile_wait": tile_wait,
    }

    # ─────────────────── XCD-masked overlap (only if available) ───────────────────
    if masked_streams is not None and masked_comm_stream is not None and masked_comp_stream is not None:
        comm_cu, comp_cu = masked_streams

        def xcd_masked():
            """Dispatch on a dedicated XCD mask, gemm on the remaining XCDs.
            No HBM BW contention — this is the scheme test_ep.py uses.
            """
            current = torch.cuda.current_stream()
            tail = _fresh_tail()
            masked_comm_stream.wait_stream(current)
            with torch.cuda.stream(masked_comm_stream):
                (out, _) = _expert_grouped_dispatch_permute(
                    x,
                    group,
                    tail,
                    topk_idx,
                    topk_weights,
                    num_experts,
                    num_experts_per_group=num_experts_per_group,
                    num_sms=comm_cu,
                )
            permuted_x = out[0]
            masked_comp_stream.wait_stream(current)
            permuted_x.record_stream(masked_comp_stream)
            tail.record_stream(masked_comp_stream)
            with torch.cuda.stream(masked_comp_stream):
                grouped_gemm_triton_kernel(
                    permuted_x,
                    weight,
                    tail,
                    full_offs,
                    BLOCK_SIZE_M=128,
                    num_sms=comp_cu,
                )

        runners["xcd_masked"] = xcd_masked

    # Return runners and the helper so the bench harness can do the final join.
    return runners, _wait_all


def _gather_permuted_by_expert(
    permuted_x: torch.Tensor,
    counts: torch.Tensor,
) -> List[torch.Tensor]:
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
    """Validate the optimized ``_expert_grouped_dispatch_permute`` against the
    stock ``_fused_dispatch_permute`` reference:

    * per-expert received count matches exactly,
    * per-expert tail counter matches exactly,
    * for every local expert ``e``, the multiset of tokens written to
      ``permuted_x[offsets[e]:offsets[e+1]]`` matches.
    """
    from primus_turbo.pytorch.cco import (
        _expert_grouped_dispatch_permute,
        _fused_dispatch_permute,
    )

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

    # Verify permuted_x content per-expert multiset matches.
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
# Main entry
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
    comm_stream: torch.cuda.Stream,
    compute_stream_single: torch.cuda.Stream,
    compute_streams_multi: List[torch.cuda.Stream],
    masked_streams_info: Optional[Tuple[torch.cuda.Stream, torch.cuda.Stream, int, int]],
) -> Dict[str, float]:
    from primus_turbo.pytorch.cco import _expert_grouped_dispatch_permute
    from primus_turbo.triton.grouped_gemm.grouped_gemm_kernel import _get_num_cus

    num_experts_per_rank = args.num_experts // num_ranks
    num_groups = (num_experts_per_rank + num_experts_per_group - 1) // num_experts_per_group
    total_sms = _get_num_cus()

    # Warmup dispatch to capture expected_counts — the same values will be
    # used every iteration because the topk_idx is deterministic.
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
            f"\n────── num_experts_per_group={num_experts_per_group} " f"(num_groups={num_groups}) ──────",
            flush=True,
        )

    masked_comm_stream = None
    masked_comp_stream = None
    masked_sms = None
    if masked_streams_info is not None:
        masked_comm_stream, masked_comp_stream, comm_cu, comp_cu = masked_streams_info
        masked_sms = (comm_cu, comp_cu)

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
        comm_stream=comm_stream,
        compute_stream_single=compute_stream_single,
        compute_streams_multi=compute_streams_multi,
        persistent_tail=persistent_tail,
        masked_streams=masked_sms,
        masked_comm_stream=masked_comm_stream,
        masked_comp_stream=masked_comp_stream,
    )

    # Per-iter (join every iter) and steady-state (queue N iters, join once)
    # timings side-by-side.
    results: Dict[str, float] = {}
    for name, fn in runners.items():
        dist.barrier(group=group)
        try:
            ms_single = _bench_joined(fn, join_fn, args.warmups, args.iters)
        except Exception as exc:
            if rank == 0:
                print(f"  {name:17s}: FAILED ({exc})", flush=True)
            results[name] = float("nan")
            results[f"{name}_steady"] = float("nan")
            continue
        dist.barrier(group=group)
        try:
            ms_steady = _bench_pipelined(
                fn,
                join_fn,
                args.warmups,
                args.iters,
                inner_loop=args.steady_loop,
            )
        except Exception as exc:
            ms_steady = float("nan")
            if rank == 0:
                print(f"  [steady] {name}: FAILED ({exc})", flush=True)
        dist.barrier(group=group)
        results[name] = ms_single
        results[f"{name}_steady"] = ms_steady
        if rank == 0:
            print(
                f"  {name:17s}: single={ms_single*1e6:8.1f}us  " f"steady={ms_steady*1e6:8.1f}us",
                flush=True,
            )
    return results


def _test_entry(local_rank: int, world_size: int, args: argparse.Namespace) -> None:
    try:
        from utils import init_dist  # type: ignore
    except ImportError:
        from tests.pytorch.cco.utils import init_dist  # type: ignore

    rank, num_ranks, group = init_dist(local_rank, world_size)
    torch.manual_seed(rank + 17)

    num_experts_per_rank = args.num_experts // num_ranks

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

    # Streams (created once, reused across all configs)
    comm_stream = torch.cuda.Stream()
    compute_stream_single = torch.cuda.Stream()
    compute_streams_multi = [torch.cuda.Stream() for _ in range(args.num_multi_streams)]

    # XCD-masked streams (for the xcd_masked variant).
    masked_streams_info = None
    if args.enable_xcd_masked:
        try:
            from test_ep import _create_partitioned_masked_streams  # type: ignore
        except ImportError:
            from tests.pytorch.cco.test_ep import _create_partitioned_masked_streams  # type: ignore
        from primus_turbo.triton.grouped_gemm.grouped_gemm_kernel import _get_num_cus

        total_sms = _get_num_cus()
        device = torch.cuda.current_device()
        try:
            comm_owner, comp_owner, actual_comm, actual_comp = _create_partitioned_masked_streams(
                args.num_sms, total_sms, device
            )
            masked_streams_info = (comm_owner.stream, comp_owner.stream, actual_comm, actual_comp)
            if rank == 0:
                print(
                    f"[xcd_masked] actual_comm_sms={actual_comm} " f"actual_comp_sms={actual_comp}",
                    flush=True,
                )
        except Exception as exc:
            if rank == 0:
                print(f"[xcd_masked] failed: {exc}", flush=True)
            masked_streams_info = None

    # Correctness once per ``num_experts_per_group``.
    for g in args.sweep:
        if num_experts_per_rank % g != 0:
            if rank == 0:
                print(
                    f"[skip] num_experts_per_rank={num_experts_per_rank} "
                    f"not divisible by num_experts_per_group={g}",
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

    # Performance sweep.
    sweep_results: List[Tuple[int, Dict[str, float]]] = []
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
            comm_stream=comm_stream,
            compute_stream_single=compute_stream_single,
            compute_streams_multi=compute_streams_multi,
            masked_streams_info=masked_streams_info,
        )
        sweep_results.append((g, res))

    # Optional profiler dump — captures one iteration per variant so the
    # kernel timeline can be inspected offline.
    if args.dump_trace and sweep_results:
        best_g, best_res = min(sweep_results, key=lambda kv: kv[1].get("per_group_ms_full", float("inf")))
        if rank == 0:
            print(f"\n[profiler] dumping traces for num_experts_per_group={best_g}", flush=True)
            os.makedirs(args.trace_dir, exist_ok=True)

        # Rebuild runners for the best config + run with profiler.
        from primus_turbo.pytorch.cco import _expert_grouped_dispatch_permute
        from primus_turbo.triton.grouped_gemm.grouped_gemm_kernel import _get_num_cus

        nr = num_experts_per_rank
        epg = best_g
        ng = (nr + epg - 1) // epg

        # Re-warm to get expected counts fresh.
        warm = torch.zeros((nr,), dtype=torch.int32, device="cuda")
        (wt, _) = _expert_grouped_dispatch_permute(
            x,
            group,
            warm,
            topk_idx,
            topk_weights,
            args.num_experts,
            num_experts_per_group=epg,
            num_sms=args.num_sms,
        )
        group.barrier()
        exp = wt[5].clone().contiguous()
        fo = _build_full_offs(exp)
        pgo = _build_per_group_offs(exp, epg)
        gs = [g * epg for g in range(ng)]
        gl = [min((g + 1) * epg, nr) - gs[g] for g in range(ng)]

        masked_comm = masked_comp = None
        masked_sms = None
        if masked_streams_info is not None:
            masked_comm, masked_comp, cc, cp = masked_streams_info
            masked_sms = (cc, cp)

        runners, join_fn = _build_runners(
            x=x,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            weight=weight,
            group=group,
            num_experts=args.num_experts,
            num_experts_per_rank=nr,
            num_experts_per_group=epg,
            num_sms=args.num_sms,
            total_sms=_get_num_cus(),
            expected_counts=exp,
            full_offs=fo,
            per_group_offs=pgo,
            num_groups=ng,
            group_starts=gs,
            group_lens=gl,
            comm_stream=comm_stream,
            compute_stream_single=compute_stream_single,
            compute_streams_multi=compute_streams_multi,
            persistent_tail=persistent_tail,
            masked_streams=masked_sms,
            masked_comm_stream=masked_comm,
            masked_comp_stream=masked_comp,
        )

        current = torch.cuda.current_stream()
        for name in args.trace_variants:
            if name not in runners:
                continue
            fn = runners[name]
            for _ in range(5):
                fn()
                join_fn(current)
            torch.cuda.synchronize()
            group.barrier()
            _ctx = (
                torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                )
                if rank == 0
                else None
            )
            if _ctx is not None:
                _ctx.__enter__()
            # Dump 3 back-to-back iterations so the pipelining pattern
            # (iter N+1 dispatch overlapping iter N compute) is visible.
            for _ in range(3):
                fn()
            join_fn(current)
            torch.cuda.synchronize()
            if _ctx is not None:
                _ctx.__exit__(None, None, None)
                path = os.path.join(args.trace_dir, f"bench_{name}_g{epg}.json")
                _ctx.export_chrome_trace(path)
                print(f"[profiler] {name} → {path}", flush=True)
            group.barrier()

    # ── Final summary ────────────────────────────────────────────────────
    if rank == 0:
        # Separate single-iter and steady-state variants for readability.
        def _print_table(label: str, key_filter):
            vs = sorted({k for _, r in sweep_results for k in r if key_filter(k)})
            print(f"\n══════════════════════ {label} (us) ══════════════════════")
            print(f"{'num_ep/grp':>10s}" + "".join(f"  {v:>20s}" for v in vs))
            for g, res in sweep_results:
                row = f"{g:>10d}"
                for v in vs:
                    val = res.get(v, float("nan"))
                    row += f"  {val*1e6:>18.1f}"
                print(row)
            # Speedup vs baseline_pipe (same key_filter).
            bp_key = "baseline_pipe" if key_filter("baseline_pipe") else "baseline_pipe_steady"
            print(f"─── speedup vs {bp_key} ───")
            print(f"{'num_ep/grp':>10s}" + "".join(f"  {v:>20s}" for v in vs))
            for g, res in sweep_results:
                bp = res.get(bp_key, float("nan"))
                row = f"{g:>10d}"
                for v in vs:
                    val = res.get(v, float("nan"))
                    if bp and val:
                        row += f"  {bp/val:>17.2f}x"
                    else:
                        row += f"  {'n/a':>20s}"
                print(row)

        _print_table("SINGLE-ITER (join-inclusive)", lambda k: not k.endswith("_steady"))
        _print_table("STEADY-STATE (pipelined)", lambda k: k.endswith("_steady"))

        # Best-config report.
        best_single = min(
            ((g, k, v) for g, r in sweep_results for k, v in r.items() if not k.endswith("_steady")),
            key=lambda x: x[2] if x[2] == x[2] else float("inf"),
        )
        best_steady = min(
            ((g, k, v) for g, r in sweep_results for k, v in r.items() if k.endswith("_steady")),
            key=lambda x: x[2] if x[2] == x[2] else float("inf"),
        )
        print(
            f"\n[winner single-iter]  "
            f"{best_single[1]:>18s} @ num_ep/grp={best_single[0]:>2d}:"
            f" {best_single[2]*1e6:.1f} us"
        )
        print(
            f"[winner steady-state] "
            f"{best_steady[1]:>18s} @ num_ep/grp={best_steady[0]:>2d}:"
            f" {best_steady[2]*1e6:.1f} us"
        )

    # Cleanup
    if masked_streams_info is not None:
        torch.cuda.synchronize()
        # The owners go out of scope here — destructors clean up HIP streams.

    dist.barrier()
    dist.destroy_process_group()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-processes", type=int, default=8)
    p.add_argument("--num-tokens", type=int, default=4096)
    p.add_argument("--hidden", type=int, default=7168)
    p.add_argument("--hidden-out", type=int, default=4096)
    p.add_argument("--num-topk", type=int, default=8)
    p.add_argument("--num-experts", type=int, default=256)
    p.add_argument(
        "--num-sms", type=int, default=48, help="CUs used by dispatch (sender + receiver). Must be even."
    )
    p.add_argument(
        "--num-multi-streams", type=int, default=3, help="Number of compute streams for per_group_ms variant."
    )
    p.add_argument("--warmups", type=int, default=10)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument(
        "--steady-loop", type=int, default=4, help="Inner loop length for the steady-state pipelined bench"
    )
    p.add_argument(
        "--sweep",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32],
        help="num_experts_per_group values to sweep.",
    )
    p.add_argument("--enable-xcd-masked", action="store_true", default=True)
    p.add_argument("--disable-xcd-masked", dest="enable_xcd_masked", action="store_false")
    p.add_argument(
        "--dump-trace",
        action="store_true",
        default=False,
        help="Export Chrome traces for selected variants to --trace-dir",
    )
    p.add_argument(
        "--trace-dir", type=str, default="prof", help="Directory to write Chrome traces into (default: prof/)"
    )
    p.add_argument(
        "--trace-variants",
        nargs="+",
        default=["baseline_pipe", "per_group_1s", "per_group_ms_full"],
        help="Variants to dump traces for (best num_experts_per_group)",
    )
    return p.parse_args()


if __name__ == "__main__":
    _args = _parse_args()
    torch.multiprocessing.spawn(
        _test_entry,
        args=(_args.num_processes, _args),
        nprocs=_args.num_processes,
    )
