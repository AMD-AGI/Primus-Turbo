###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Benchmark for the MoE permute / unpermute paths in Primus-Turbo.
Usage::

    python benchmark/ops/bench_permute.py
    python benchmark/ops/bench_permute.py -o out.csv
    python benchmark/ops/bench_permute.py --backends hip
    python benchmark/ops/bench_permute.py --backends triton
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from config import (
    BATCH_SIZE_LIST,
    GROUPED_GEMM_EP_SIZE_LIST,
    MoEModelConfigs,
    get_platform_info,
)
from tabulate import tabulate

from primus_turbo.pytorch.ops.moe import (
    indices_to_multihot,
    moe_permute,
    moe_unpermute,
    token_permute,
    token_unpermute,
)
from primus_turbo.triton.moe import permutation as triton_permute

SUPPORTED_BACKENDS = ("hip", "triton")

# -----------------------------------------------------------------------------
# DeepEP-style dispatch simulator.
# -----------------------------------------------------------------------------


def expected_recv_num_tokens(
    num_tokens_per_rank: int, num_experts: int, num_topk: int, num_ranks: int
) -> float:
    """Closed-form E[recv_num_tokens] for the simulator below."""
    if num_experts <= 0 or num_topk <= 0 or num_ranks <= 0:
        return 0.0
    num_local_experts = num_experts // num_ranks
    p_miss = 1.0
    for i in range(num_topk):
        num = (num_experts - num_local_experts) - i
        den = num_experts - i
        if num <= 0:
            p_miss = 0.0
            break
        p_miss *= num / den
    return num_ranks * num_tokens_per_rank * (1.0 - p_miss)


def simulate_dispatch_recv(
    num_tokens_per_rank: int,
    num_experts: int,
    num_topk: int,
    num_ranks: int,
    *,
    device: torch.device,
    seed: int = 0,
):
    """Simulate the receive side of a DeepEP intranode dispatch.

    Returns
    -------
    recv_topk_idx : torch.Tensor
        ``int64 [recv_num_tokens, num_topk]``, entries are local expert ids in
        ``[0, num_local_experts)`` with ``-1`` padding.
    recv_num_tokens : int
    """
    assert (
        num_experts % num_ranks == 0
    ), f"num_experts ({num_experts}) must be divisible by num_ranks ({num_ranks})"
    num_local_experts = num_experts // num_ranks
    total = num_ranks * num_tokens_per_rank

    g = torch.Generator(device="cpu").manual_seed(seed)
    scores = torch.rand((total, num_experts), generator=g)
    _, topk_idx = scores.topk(num_topk, dim=1)

    local_mask = topk_idx < num_local_experts
    has_local = local_mask.any(dim=1)
    recv_num_tokens = int(has_local.sum().item())

    sentinel = num_experts
    masked = torch.where(local_mask, topk_idx, torch.full_like(topk_idx, sentinel))
    sorted_idx, _ = torch.sort(masked, dim=1)
    sorted_idx = torch.where(sorted_idx == sentinel, torch.full_like(sorted_idx, -1), sorted_idx)

    recv_topk_idx = sorted_idx[has_local].to(dtype=torch.int64, device=device)
    return recv_topk_idx, recv_num_tokens


def routing_map_from_recv_topk_idx(recv_topk_idx: torch.Tensor, num_local_experts: int) -> torch.Tensor:
    """Build the boolean ``routing_map [recv_num_tokens, num_local_experts]``
    via the production fused triton kernel.

    Mirrors ``token_dispatcher.py::_post_dispatch``: feed ``topk_idx`` (with
    ``-1`` padding) into ``indices_to_multihot(fused=True)`` to materialise
    the bool routing_map. The probs slot is unused on this code path so we
    pass a zeroed dummy tensor and discard the multihot probs output.
    """
    T = recv_topk_idx.shape[0]
    device = recv_topk_idx.device
    if T == 0:
        return torch.zeros((0, num_local_experts), dtype=torch.bool, device=device)
    fake_probs = torch.zeros_like(recv_topk_idx, dtype=torch.float32)
    routing_map, _ = indices_to_multihot(recv_topk_idx, fake_probs, num_local_experts, fused=True)
    return routing_map


# -----------------------------------------------------------------------------
# Reference implementations (pure torch).
# -----------------------------------------------------------------------------


def reference_permute_from_routing(tokens: torch.Tensor, routing_map: torch.Tensor) -> torch.Tensor:
    """Permute ``tokens`` into expert-grouped order driven by ``routing_map``.

    For each expert ``e`` (in order), append every token routed to ``e``
    (in original token order). This matches the layout produced by both the
    CUDA and Triton kernels.
    """
    H = tokens.shape[-1]
    expert_token = routing_map.T.nonzero(as_tuple=False)  # [num_pairs, 2]
    src_token = expert_token[:, 1]
    if src_token.numel() == 0:
        return torch.zeros((0, H), dtype=tokens.dtype, device=tokens.device)
    return tokens[src_token]


def reference_unpermute_from_routing(permuted: torch.Tensor, routing_map: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`reference_permute_from_routing`: scatter-add back."""
    T = routing_map.shape[0]
    H = permuted.shape[-1]
    expert_token = routing_map.T.nonzero(as_tuple=False)  # [num_pairs, 2]
    src_token = expert_token[:, 1]
    acc = torch.zeros((T, H), dtype=torch.float32, device=permuted.device)
    if src_token.numel() > 0:
        acc.index_add_(0, src_token, permuted[: src_token.numel()].float())
    return acc.to(permuted.dtype)


# -----------------------------------------------------------------------------
# Test-case generator (driven by config.MoEModelConfigs).
# -----------------------------------------------------------------------------


@dataclass
class PermuteCase:
    label: str
    model: str
    mbs: int
    ep: int
    num_tokens_per_rank: int
    hidden_size: int
    num_experts: int
    num_topk: int
    num_local_experts: int


def gen_permute_test_cases() -> List[PermuteCase]:
    cases: List[PermuteCase] = []
    for model_name, cfg in MoEModelConfigs.items():
        seqlen = cfg["seqlen"]
        hidden = cfg["hidden_size"]
        num_experts = cfg["num_experts"]
        num_topk = cfg["num_topk"]
        for mbs in BATCH_SIZE_LIST:
            num_tokens_per_rank = seqlen * mbs
            for ep in GROUPED_GEMM_EP_SIZE_LIST:
                if num_experts % ep != 0:
                    continue
                num_local_experts = num_experts // ep
                if num_local_experts == 0:
                    continue
                cases.append(
                    PermuteCase(
                        label=f"{model_name}/MBS={mbs}/EP={ep}",
                        model=model_name,
                        mbs=mbs,
                        ep=ep,
                        num_tokens_per_rank=num_tokens_per_rank,
                        hidden_size=hidden,
                        num_experts=num_experts,
                        num_topk=num_topk,
                        num_local_experts=num_local_experts,
                    )
                )
    return cases


@dataclass
class BackendResult:
    """Result of one backend on one case.

    ``fwd_ms`` is the **derived** total ``preproc_ms + permute_ms`` (so the
    timed wall-clock matches what users would see if preproc and permute ran
    back-to-back on the same stream — same as the high-level fused API).
    """

    correct: bool
    preproc_ms: float
    permute_ms: float
    fwd_ms: float
    fwd_gbps: float
    bwd_ms: float
    bwd_gbps: float


def _bench_callable(fn: Callable[[], None], *, warmup: int, iters: int) -> float:
    """Return the average wall-clock time for ``fn`` in milliseconds.

    Modeled on ``benchmark/ops/deep_ep/utils.py::bench``: record each
    ``fn()`` call inside its own ``(start, end)`` ``cuda.Event`` pair,
    drop the first measurement as extra warmup margin, and **flush L2
    before every timed iteration** by zeroing a 256 MB scratch tensor.

    The L2 flush is the load-bearing piece for memory-bound permute /
    unpermute kernels: without it, the previous iteration leaves the
    ~33 K * H * 2 byte input/output footprint warm in cache, which biases
    the next ``fn()`` toward an artificially high bandwidth that is not
    representative of the steady-state production stream where each
    permute step sees freshly-arrived dispatch tokens.
    """
    torch.cuda.synchronize()

    # Flush cache
    cache = torch.empty(int(1e7 // 4), dtype=torch.int, device="cuda")

    for _ in range(warmup):
        fn()

    n = max(iters, 1) + 1  # +1 so we can drop the first sample
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n)]

    for i in range(n):
        cache.zero_()  # flush L2 before each timed iteration
        start_events[i].record()
        fn()
        end_events[i].record()
    torch.cuda.synchronize()

    times_ms = np.array([s.elapsed_time(e) for s, e in zip(start_events, end_events)])
    # Drop the first measurement to avoid any residual warmup spike.
    return float(times_ms[1:].mean())


def _check_close(name: str, ref: torch.Tensor, got: torch.Tensor) -> bool:
    if ref.shape != got.shape:
        print(f"  [{name}] FAIL: shape mismatch ref={tuple(ref.shape)} got={tuple(got.shape)}")
        return False
    if not torch.allclose(ref.float(), got.float(), atol=1e-2, rtol=1e-2):
        diff = (ref.float() - got.float()).abs()
        print(f"  [{name}] FAIL: max-abs-diff={diff.max().item():.4f}")
        return False
    return True


def run_hip_backend(
    case: PermuteCase,
    *,
    topk_idx: torch.Tensor,
    tokens: torch.Tensor,
    num_dispatched: int,
    num_dispatched_t: torch.Tensor,
    ref_perm: torch.Tensor,
    ref_unperm: torch.Tensor,
    warmup: int,
    iters: int,
) -> BackendResult:
    """HIP backend: feeds ``topk_idx`` straight into ``moe_permute``."""
    H = case.hidden_size
    E = case.num_local_experts
    K = case.num_topk
    bytes_per_elem = tokens.element_size()

    # ---- correctness pass via the high-level autograd-aware API ------------
    permuted_tokens, row_id_map, _, overflow_flag, _, _ = moe_permute(
        tokens,
        topk_idx,
        num_dispatched_t,
        num_local_experts=E,
        num_topk=K,  # topk_idx path
        pad_multiple=0,
    )
    torch.cuda.synchronize()
    if int(overflow_flag.item()) != 0:
        print("  [hip] preproc overflow flag set unexpectedly")
        return BackendResult(False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    ok_p = _check_close("hip/permute", ref_perm, permuted_tokens)

    recovered, _ = moe_unpermute(
        permuted_tokens,
        row_id_map,
        num_dispatched_t,
        num_local_experts=E,
    )
    torch.cuda.synchronize()
    ok_u = _check_close("hip/unpermute", ref_unperm, recovered)
    correct = ok_p and ok_u
    num_permuted = int(permuted_tokens.shape[0])

    # ---- timing: split preproc / permute / unpermute via the low-level
    # ``primus_turbo_cpp_extension`` ops so each cost is attributable. The
    # high-level ``moe_permute`` would fuse preproc+permute into a single
    # autograd op, hiding the breakdown.
    cpp_ext = torch.ops.primus_turbo_cpp_extension

    def preproc_call():
        cpp_ext.permute_preprocessing(
            topk_idx,
            num_dispatched_t,
            E,
            K,
            0,  # pad_multiple
            num_permuted,  # known cap → skip the host sync
        )

    def permute_call():
        cpp_ext.permute(
            tokens,
            permuted_tokens,
            None,
            None,  # scaling_factor / output_scaling_factor
            None,
            None,  # probs / output_probs
            row_id_map,
            num_dispatched_t,
            0,  # pad_multiple
            E,
            H,
            0,  # scales_per_token
            False,  # use_fp8
            False,  # with_probs
            num_permuted,
        )

    def bwd_call():
        cpp_ext.unpermute(
            permuted_tokens,
            recovered,
            None,
            None,  # permuted_probs / output_probs
            row_id_map,
            num_dispatched_t,
            E,
            H,
            False,  # with_probs
        )

    preproc_ms = _bench_callable(preproc_call, warmup=warmup, iters=iters)
    permute_ms = _bench_callable(permute_call, warmup=warmup, iters=iters)
    bwd_ms = _bench_callable(bwd_call, warmup=warmup, iters=iters)
    fwd_ms = preproc_ms + permute_ms
    perm_bytes = (num_dispatched + num_permuted) * H * bytes_per_elem
    unperm_bytes = (num_permuted + num_dispatched) * H * bytes_per_elem
    fwd_gbps = perm_bytes / (fwd_ms * 1e-3) / 1e9 if fwd_ms > 0 else 0.0
    bwd_gbps = unperm_bytes / (bwd_ms * 1e-3) / 1e9 if bwd_ms > 0 else 0.0

    return BackendResult(
        correct=correct,
        preproc_ms=preproc_ms,
        permute_ms=permute_ms,
        fwd_ms=fwd_ms,
        fwd_gbps=fwd_gbps,
        bwd_ms=bwd_ms,
        bwd_gbps=bwd_gbps,
    )


def run_triton_backend(
    case: PermuteCase,
    *,
    topk_idx: torch.Tensor,
    tokens: torch.Tensor,
    num_dispatched: int,
    ref_perm: torch.Tensor,
    ref_unperm: torch.Tensor,
    warmup: int,
    iters: int,
) -> BackendResult:
    """Triton backend"""
    H = case.hidden_size
    E = case.num_local_experts
    bytes_per_elem = tokens.element_size()

    # Setup: topk_idx → routing_map (one-shot, NOT in the timed path).
    routing_map = routing_map_from_recv_topk_idx(topk_idx, E)

    # ---- correctness pass via the high-level autograd-aware API -----------
    # First call uses ``num_out_tokens=-1`` so ``TokenPermuteMaskMap`` derives
    # the actual permuted-token count from the routing_map (single host sync).
    permuted_tokens, _, row_id_map, tokens_per_expert = token_permute(
        tokens,
        num_out_tokens=-1,
        routing_map=routing_map,
        fused=True,
        return_tokens_per_expert=True,
    )
    torch.cuda.synchronize()
    num_permuted = int(tokens_per_expert.sum().item())
    ok_p = _check_close("triton/permute", ref_perm, permuted_tokens)

    recovered = token_unpermute(
        permuted_tokens,
        row_id_map,
        restore_shape=tokens.shape,
        fused=True,
    )
    torch.cuda.synchronize()
    ok_u = _check_close("triton/unpermute", ref_unperm, recovered)
    correct = ok_p and ok_u

    # ---- timing: split preproc / permute / unpermute via the low-level
    # triton kernels in ``primus_turbo.triton.moe.permutation`` so each cost
    # is attributable.
    def preproc_call():
        routing_map = routing_map_from_recv_topk_idx(topk_idx, E)
        triton_permute.make_row_id_map(routing_map, num_dispatched, E, return_tokens_per_expert=True)

    def permute_call():
        triton_permute.permute_with_mask_map(
            tokens,
            row_id_map,
            None,
            None,
            num_dispatched,
            E,
            num_permuted,
            H,
            0,
        )

    def bwd_call():
        triton_permute.unpermute_with_mask_map(
            permuted_tokens,
            row_id_map,
            None,
            None,
            num_dispatched,
            E,
            H,
        )

    preproc_ms = _bench_callable(preproc_call, warmup=warmup, iters=iters)
    permute_ms = _bench_callable(permute_call, warmup=warmup, iters=iters)
    bwd_ms = _bench_callable(bwd_call, warmup=warmup, iters=iters)
    fwd_ms = preproc_ms + permute_ms
    perm_bytes = (num_dispatched + num_permuted) * H * bytes_per_elem
    unperm_bytes = (num_permuted + num_dispatched) * H * bytes_per_elem
    fwd_gbps = perm_bytes / (fwd_ms * 1e-3) / 1e9 if fwd_ms > 0 else 0.0
    bwd_gbps = unperm_bytes / (bwd_ms * 1e-3) / 1e9 if bwd_ms > 0 else 0.0

    return BackendResult(
        correct=correct,
        preproc_ms=preproc_ms,
        permute_ms=permute_ms,
        fwd_ms=fwd_ms,
        fwd_gbps=fwd_gbps,
        bwd_ms=bwd_ms,
        bwd_gbps=bwd_gbps,
    )


def run_one(
    case: PermuteCase,
    backends: List[str],
    *,
    warmup: int,
    iters: int,
    seed: int = 0,
) -> "tuple[Dict[str, BackendResult], int]":
    device = torch.device("cuda")

    recv_topk_idx, recv_num_tokens = simulate_dispatch_recv(
        case.num_tokens_per_rank,
        case.num_experts,
        case.num_topk,
        case.ep,
        device=device,
        seed=seed,
    )
    if recv_num_tokens == 0:
        empty = BackendResult(False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        return {b: empty for b in backends}, 0

    topk_idx = recv_topk_idx.to(dtype=torch.int32)
    num_dispatched = recv_num_tokens
    num_dispatched_t = torch.tensor([num_dispatched], dtype=torch.int32, device=device)

    tokens = torch.randn((num_dispatched, case.hidden_size), dtype=torch.bfloat16, device=device)

    ref_routing_map = routing_map_from_recv_topk_idx(recv_topk_idx, case.num_local_experts)
    ref_perm = reference_permute_from_routing(tokens, ref_routing_map)
    ref_unperm = reference_unpermute_from_routing(ref_perm, ref_routing_map)
    num_permuted_ref = int(ref_perm.shape[0])

    results: Dict[str, BackendResult] = {}
    for name in backends:
        try:
            if name == "hip":
                results[name] = run_hip_backend(
                    case,
                    topk_idx=topk_idx,
                    tokens=tokens,
                    num_dispatched=num_dispatched,
                    num_dispatched_t=num_dispatched_t,
                    ref_perm=ref_perm,
                    ref_unperm=ref_unperm,
                    warmup=warmup,
                    iters=iters,
                )
            elif name == "triton":
                results[name] = run_triton_backend(
                    case,
                    topk_idx=topk_idx,
                    tokens=tokens,
                    num_dispatched=num_dispatched,
                    ref_perm=ref_perm,
                    ref_unperm=ref_unperm,
                    warmup=warmup,
                    iters=iters,
                )
            else:
                raise ValueError(f"unknown backend {name!r}")
        except Exception as exc:
            print(f"  [{name}] backend failed: {exc}")
            results[name] = BackendResult(False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    return results, num_permuted_ref


# -----------------------------------------------------------------------------
# Entry point.
# -----------------------------------------------------------------------------


def benchmark_permute(
    backends: List[str],
    *,
    warmup: int,
    iters: int,
    output_csv: Optional[str],
    seed: int,
    case_filter: Optional[str] = None,
    limit: Optional[int] = None,
):
    platform, gpu_name = get_platform_info()
    print(f"[bench] platform={platform}, gpu={gpu_name}, backends={backends}")

    cases = gen_permute_test_cases()
    if case_filter:
        keep = [k.strip() for k in case_filter.split(",") if k.strip()]
        cases = [c for c in cases if any(k in c.label for k in keep)]
    if limit is not None and limit > 0:
        cases = cases[:limit]
    print(f"[bench] {len(cases)} cases to run (filter={case_filter!r}, limit={limit})")

    rows: List[dict] = []
    for test_id, case in enumerate(cases, start=1):
        exp_recv = expected_recv_num_tokens(
            case.num_tokens_per_rank, case.num_experts, case.num_topk, case.ep
        )
        print(f"\n{'=' * 60}")
        print(
            f"TestID: {test_id}, Case: {case.label}, "
            f"N/rank: {case.num_tokens_per_rank}, hidden: {case.hidden_size}, "
            f"E: {case.num_experts}, K: {case.num_topk}, "
            f"local_E: {case.num_local_experts}, "
            f"E[recv_tokens]: {exp_recv:.0f}"
        )
        print(f"{'=' * 60}")

        row: dict = {
            "TestID": test_id,
            "Platform": platform,
            "GPU": gpu_name,
            "Case": case.model,
            "MBS": case.mbs,
            "EP": case.ep,
            "num_tokens_per_rank": case.num_tokens_per_rank,
            "hidden": case.hidden_size,
            "num_experts": case.num_experts,
            "num_topk": case.num_topk,
            "num_local_experts": case.num_local_experts,
            "expected_recv_tokens": int(round(exp_recv)),
        }

        try:
            results, num_permuted = run_one(case, backends, warmup=warmup, iters=iters, seed=seed)
            row["num_permuted"] = num_permuted
            for b in backends:
                r = results[b]
                row[f"{b}/Check"] = "PASS" if r.correct else "FAIL"
                row[f"{b}/Preproc Time (ms)"] = f"{r.preproc_ms:.3f}"
                row[f"{b}/Permute Time (ms)"] = f"{r.permute_ms:.3f}"
                row[f"{b}/Forward Time (ms)"] = f"{r.fwd_ms:.3f}"
                row[f"{b}/Forward Bandwidth (GB/s)"] = f"{r.fwd_gbps:.2f}"
                row[f"{b}/Backward Time (ms)"] = f"{r.bwd_ms:.3f}"
                row[f"{b}/Backward Bandwidth (GB/s)"] = f"{r.bwd_gbps:.2f}"
                print(
                    f"  [{b}] Forward {r.preproc_ms:.3f} + {r.permute_ms:.3f} = "
                    f"{r.fwd_ms:.3f} ms ({r.fwd_gbps:.1f} GB/s) | "
                    f"Backward {r.bwd_ms:.3f} ms ({r.bwd_gbps:.1f} GB/s) | "
                    f"{'PASS' if r.correct else 'FAIL'}"
                )

            if "hip" in backends and "triton" in backends:
                rc, rt = results["hip"], results["triton"]
                fwd_sp = (rt.fwd_ms / rc.fwd_ms) if rc.fwd_ms > 0 and rt.fwd_ms > 0 else 0.0
                bwd_sp = (rt.bwd_ms / rc.bwd_ms) if rc.bwd_ms > 0 and rt.bwd_ms > 0 else 0.0
                row["Speedup Forward (hip vs triton)"] = f"{fwd_sp:.2f}"
                row["Speedup Backward (hip vs triton)"] = f"{bwd_sp:.2f}"

        except Exception as exc:
            print(f"Failed: {exc}")
            row["num_permuted"] = "ERROR"
            for b in backends:
                row[f"{b}/Check"] = "ERROR"
                row[f"{b}/Preproc Time (ms)"] = "ERROR"
                row[f"{b}/Permute Time (ms)"] = "ERROR"
                row[f"{b}/Forward Time (ms)"] = "ERROR"
                row[f"{b}/Forward Bandwidth (GB/s)"] = "0.00"
                row[f"{b}/Backward Time (ms)"] = "ERROR"
                row[f"{b}/Backward Bandwidth (GB/s)"] = "0.00"

        rows.append(row)

    # Use the union of all keys to handle backend / case mismatches gracefully.
    all_keys: List[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                all_keys.append(k)
    results_df = pd.DataFrame(rows, columns=all_keys)
    print("\nFinal Results:")
    print(tabulate(results_df, headers="keys", tablefmt="grid", showindex=False))

    def _gmean(series):
        s = pd.to_numeric(series, errors="coerce").dropna()
        s = s[s > 0]
        if len(s) == 0:
            return 0.0
        return math.exp(s.apply(math.log).mean())

    for b in backends:
        col_fwd = f"{b}/Forward Bandwidth (GB/s)"
        col_bwd = f"{b}/Backward Bandwidth (GB/s)"
        if col_fwd in results_df.columns:
            avg = pd.to_numeric(results_df[col_fwd], errors="coerce").mean()
            gmean = _gmean(results_df[col_fwd])
            peak = pd.to_numeric(results_df[col_fwd], errors="coerce").max()
            print(
                f"Average {b} Forward Bandwidth (GB/s): {avg:.2f} " f"(geomean {gmean:.2f}, peak {peak:.2f})"
            )
        if col_bwd in results_df.columns:
            avg = pd.to_numeric(results_df[col_bwd], errors="coerce").mean()
            gmean = _gmean(results_df[col_bwd])
            peak = pd.to_numeric(results_df[col_bwd], errors="coerce").max()
            print(
                f"Average {b} Backward Bandwidth (GB/s): {avg:.2f} " f"(geomean {gmean:.2f}, peak {peak:.2f})"
            )

    if output_csv:
        filename = output_csv
    else:
        timestamp = datetime.now().strftime("%Y%m%d")
        tag = "_".join(backends)
        filename = f"permute_{tag}_{timestamp}_{gpu_name}.csv"
    results_df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--backends",
        type=str,
        default="hip,triton",
        help=f"comma-separated backends (supported: {','.join(SUPPORTED_BACKENDS)}; "
        f"default: hip,triton).",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", "-o", type=str, default=None, help="optional CSV output path")
    parser.add_argument(
        "--case-filter",
        type=str,
        default=None,
        help="comma-separated substrings to filter case labels "
        "(e.g. 'Mixtral-8x7B,DeepSeek-V2-Lite/MBS=1')",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="cap the number of cases (after filtering) to run",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA / ROCm not available; aborting.")
        sys.exit(1)

    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    for b in backends:
        if b not in SUPPORTED_BACKENDS:
            print(f"error: unknown backend {b!r}; supported: {SUPPORTED_BACKENDS}")
            sys.exit(1)
    if not backends:
        print("error: --backends must list at least one backend")
        sys.exit(1)

    benchmark_permute(
        backends,
        warmup=args.warmup,
        iters=args.iters,
        output_csv=args.output,
        seed=args.seed,
        case_filter=args.case_filter,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
