#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Compare the Triton and HIP MoE permutation backends (CUDA-event and wall time).

``*_kernel`` measurements exclude preprocessing but keep the output allocation.
"""

import argparse
import csv
import gc
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}

moe_permute: Any = None
moe_unpermute: Any = None
triton_permutation: Any = None
TURBO_BACKEND: Any = None


def load_backend_symbols() -> None:
    """Import Primus-Turbo only after argparse has handled ``--help``."""

    global moe_permute, moe_unpermute, triton_permutation, TURBO_BACKEND

    try:
        from primus_turbo.pytorch.core.backend import BackendType
        from primus_turbo.pytorch.ops.moe.moe_permute import (
            moe_permute as turbo_moe_permute,
        )
        from primus_turbo.pytorch.ops.moe.moe_permute import (
            moe_unpermute as turbo_moe_unpermute,
        )
        from primus_turbo.triton.moe import permutation as triton_backend
    except ImportError as error:
        raise RuntimeError(
            "Primus-Turbo with both Triton and HIP moe_permute backends must be installed"
        ) from error

    moe_permute = turbo_moe_permute
    moe_unpermute = turbo_moe_unpermute
    triton_permutation = triton_backend
    TURBO_BACKEND = BackendType.TURBO


@dataclass
class BackendState:
    """Precomputed tensors used by kernel-only and unpermute measurements."""

    row_id_map: torch.Tensor
    tokens_per_expert: torch.Tensor
    num_dispatched_tokens: Optional[torch.Tensor]
    permuted_tokens: torch.Tensor
    permuted_probs: Optional[torch.Tensor]


@dataclass
class BenchResult:
    """One timing result."""

    operation: str
    backend: str
    event_us: float
    wall_us: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--experts", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bf16")
    parser.add_argument("--backend", choices=("both", "triton", "hip"), default="both")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--with-probs", action="store_true")
    parser.add_argument("--skip-check", action="store_true")
    parser.add_argument("--output-csv", type=Path)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.tokens <= 0 or args.experts <= 0 or args.hidden <= 0:
        raise ValueError("tokens, experts, and hidden must be positive")
    if args.topk <= 0 or args.topk > args.experts:
        raise ValueError("topk must satisfy 0 < topk <= experts")
    if args.hidden % 8 != 0:
        raise ValueError("hidden must be divisible by 8 for the HIP int4 vector path")
    if args.warmup < 0 or args.iters <= 0:
        raise ValueError("warmup must be non-negative and iters must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA/HIP device is required")


def make_inputs(
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Create deterministic tokens, routing map, and optional routing probs."""

    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    tokens = torch.randn(
        (args.tokens, args.hidden),
        dtype=DTYPES[args.dtype],
        device="cuda",
        generator=generator,
    )

    scores = torch.rand(
        (args.tokens, args.experts),
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    topk_indices = torch.topk(scores, args.topk, dim=1, sorted=False).indices
    routing_map = torch.zeros(
        (args.tokens, args.experts),
        dtype=torch.bool,
        device="cuda",
    )
    routing_map.scatter_(1, topk_indices, True)

    probs = None
    if args.with_probs:
        probs = scores * routing_map
        probs = probs / probs.sum(dim=1, keepdim=True)

    return tokens, routing_map, probs


def pytorch_reference(
    tokens: torch.Tensor,
    routing_map: torch.Tensor,
    probs: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Return expert-major permuted tokens, roundtrip tokens, and probs."""

    num_tokens, hidden = tokens.shape
    num_experts = routing_map.shape[1]
    routing_map_t = routing_map.T.contiguous()
    token_indices = (
        torch.arange(num_tokens, device="cuda").unsqueeze(0).expand(num_experts, -1)
    )
    sorted_indices = token_indices.masked_select(routing_map_t)
    permuted = tokens.index_select(0, sorted_indices)

    roundtrip = torch.zeros_like(tokens)
    roundtrip.scatter_add_(0, sorted_indices.unsqueeze(1).expand(-1, hidden), permuted)

    permuted_probs = None
    if probs is not None:
        permuted_probs = probs.T.contiguous().masked_select(routing_map_t)

    return permuted, roundtrip, permuted_probs


def triton_preprocess(
    routing_map: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_tokens, num_experts = routing_map.shape
    return triton_permutation.make_row_id_map(
        routing_map,
        num_tokens,
        num_experts,
        return_tokens_per_expert=True,
    )


def hip_preprocess(
    routing_map: torch.Tensor,
    num_out_tokens: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_experts = routing_map.shape[1]
    row_id_map, tokens_per_expert, overflow_flag, num_dispatched_tokens = (
        torch.ops.primus_turbo_cpp_extension.permute_preprocessing(
            routing_map,
            num_experts,
            0,
            0,
            num_out_tokens,
            0,
        )
    )
    if int(overflow_flag.item()) != 0:
        raise RuntimeError("HIP preprocessing reported an unexpected overflow")
    return row_id_map, tokens_per_expert, num_dispatched_tokens


def triton_permute_kernel(
    tokens: torch.Tensor,
    row_id_map: torch.Tensor,
    probs: Optional[torch.Tensor],
    num_out_tokens: int,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    num_tokens, hidden = tokens.shape
    num_experts = (row_id_map.shape[1] - 1) // 2
    output, _, permuted_probs = triton_permutation.permute_with_mask_map(
        tokens,
        row_id_map,
        probs,
        None,
        num_tokens,
        num_experts,
        num_out_tokens,
        hidden,
        0,
    )
    return output, permuted_probs


def hip_permute_kernel(
    tokens: torch.Tensor,
    row_id_map: torch.Tensor,
    num_dispatched_tokens: torch.Tensor,
    probs: Optional[torch.Tensor],
    num_out_tokens: int,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    num_experts = (row_id_map.shape[1] - 1) // 2
    hidden = tokens.shape[1]
    output = torch.empty(
        (num_out_tokens, hidden),
        dtype=tokens.dtype,
        device=tokens.device,
    )
    permuted_probs = (
        torch.empty((num_out_tokens,), dtype=probs.dtype, device=probs.device)
        if probs is not None
        else None
    )
    torch.ops.primus_turbo_cpp_extension.permute(
        tokens,
        output,
        None,
        None,
        probs,
        permuted_probs,
        row_id_map,
        num_dispatched_tokens,
        0,
        num_experts,
        hidden,
        0,
        False,
        probs is not None,
        num_out_tokens,
        0,
    )
    return output, permuted_probs


def triton_unpermute_kernel(
    state: BackendState,
    tokens: torch.Tensor,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    num_tokens, hidden = tokens.shape
    num_experts = (state.row_id_map.shape[1] - 1) // 2
    return triton_permutation.unpermute_with_mask_map(
        state.permuted_tokens,
        state.row_id_map,
        None,
        state.permuted_probs,
        num_tokens,
        num_experts,
        hidden,
    )


def hip_unpermute_kernel(
    state: BackendState,
    tokens: torch.Tensor,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if state.num_dispatched_tokens is None:
        raise RuntimeError("HIP state is missing num_dispatched_tokens")
    num_tokens, hidden = tokens.shape
    num_experts = (state.row_id_map.shape[1] - 1) // 2
    output = torch.empty_like(tokens)
    output_probs = (
        torch.empty(
            (num_tokens, num_experts),
            dtype=state.permuted_probs.dtype,
            device=state.permuted_probs.device,
        )
        if state.permuted_probs is not None
        else None
    )
    torch.ops.primus_turbo_cpp_extension.unpermute(
        state.permuted_tokens,
        output,
        state.permuted_probs,
        output_probs,
        state.row_id_map,
        state.num_dispatched_tokens,
        num_experts,
        hidden,
        state.permuted_probs is not None,
        0,
    )
    return output, output_probs


def prepare_triton_state(
    tokens: torch.Tensor,
    routing_map: torch.Tensor,
    probs: Optional[torch.Tensor],
    num_out_tokens: int,
) -> BackendState:
    row_id_map, tokens_per_expert = triton_preprocess(routing_map)
    permuted_tokens, permuted_probs = triton_permute_kernel(
        tokens,
        row_id_map,
        probs,
        num_out_tokens,
    )
    return BackendState(
        row_id_map=row_id_map,
        tokens_per_expert=tokens_per_expert,
        num_dispatched_tokens=None,
        permuted_tokens=permuted_tokens,
        permuted_probs=permuted_probs,
    )


def prepare_hip_state(
    tokens: torch.Tensor,
    routing_map: torch.Tensor,
    probs: Optional[torch.Tensor],
    num_out_tokens: int,
) -> BackendState:
    row_id_map, tokens_per_expert, num_dispatched_tokens = hip_preprocess(
        routing_map,
        num_out_tokens,
    )
    permuted_tokens, permuted_probs = hip_permute_kernel(
        tokens,
        row_id_map,
        num_dispatched_tokens,
        probs,
        num_out_tokens,
    )
    return BackendState(
        row_id_map=row_id_map,
        tokens_per_expert=tokens_per_expert,
        num_dispatched_tokens=num_dispatched_tokens,
        permuted_tokens=permuted_tokens,
        permuted_probs=permuted_probs,
    )


def check_correctness(
    args: argparse.Namespace,
    tokens: torch.Tensor,
    routing_map: torch.Tensor,
    probs: Optional[torch.Tensor],
    states: Dict[str, BackendState],
) -> None:
    """Check each backend against the same PyTorch reference."""

    ref_permuted, ref_roundtrip, ref_permuted_probs = pytorch_reference(
        tokens,
        routing_map,
        probs,
    )
    expected_counts = routing_map.sum(dim=0, dtype=torch.int64)
    tolerances = {"atol": 1e-2, "rtol": 1e-2}

    for backend, state in states.items():
        torch.testing.assert_close(
            state.tokens_per_expert.to(torch.int64),
            expected_counts,
        )
        torch.testing.assert_close(
            state.permuted_tokens,
            ref_permuted,
            **tolerances,
        )
        if ref_permuted_probs is not None:
            if state.permuted_probs is None:
                raise AssertionError(f"{backend} did not return permuted probs")
            torch.testing.assert_close(
                state.permuted_probs,
                ref_permuted_probs,
                **tolerances,
            )

        if backend == "triton":
            roundtrip, unpermuted_probs = triton_unpermute_kernel(state, tokens)
        else:
            roundtrip, unpermuted_probs = hip_unpermute_kernel(state, tokens)

        torch.testing.assert_close(
            roundtrip,
            ref_roundtrip,
            **tolerances,
        )
        if probs is not None:
            if unpermuted_probs is None:
                raise AssertionError(f"{backend} did not return unpermuted probs")
            torch.testing.assert_close(
                unpermuted_probs,
                probs,
                **tolerances,
            )

    print("Correctness: PASS")


def benchmark(
    fn: Callable[[], object],
    warmup: int,
    iters: int,
) -> Tuple[float, float]:
    """Return average CUDA-event and wall-clock latency in microseconds."""

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    gc.collect()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    wall_start = time.perf_counter()
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    wall_s = time.perf_counter() - wall_start

    event_us = start.elapsed_time(end) * 1000.0 / iters
    wall_us = wall_s * 1e6 / iters
    return event_us, wall_us


def make_benchmarks(
    tokens: torch.Tensor,
    routing_map: torch.Tensor,
    probs: Optional[torch.Tensor],
    states: Dict[str, BackendState],
    num_out_tokens: int,
) -> Dict[str, Dict[str, Callable[[], object]]]:
    """Create operation callables for each requested backend."""

    benchmarks: Dict[str, Dict[str, Callable[[], object]]] = {}

    if "triton" in states:
        triton_state = states["triton"]

        def triton_e2e():
            row_id_map, _ = triton_preprocess(routing_map)
            return triton_permute_kernel(tokens, row_id_map, probs, num_out_tokens)

        def triton_roundtrip():
            state = prepare_triton_state(tokens, routing_map, probs, num_out_tokens)
            return triton_unpermute_kernel(state, tokens)

        benchmarks["triton"] = {
            "preprocess": lambda: triton_preprocess(routing_map),
            "permute_kernel": lambda: triton_permute_kernel(
                tokens,
                triton_state.row_id_map,
                probs,
                num_out_tokens,
            ),
            "unpermute_kernel": lambda: triton_unpermute_kernel(triton_state, tokens),
            "permute_e2e": triton_e2e,
            "roundtrip": triton_roundtrip,
        }

    if "hip" in states:
        hip_state = states["hip"]

        def hip_e2e():
            return moe_permute(
                tokens,
                routing_map=routing_map,
                num_local_experts=routing_map.shape[1],
                num_topk=0,
                num_permuted_tokens=num_out_tokens,
                probs=probs,
                probs_layout="routing_map",
                backend=TURBO_BACKEND,
            )

        def hip_roundtrip():
            output = hip_e2e()
            return moe_unpermute(
                output[0],
                output[1],
                restore_shape=tokens.shape,
                num_local_experts=routing_map.shape[1],
                permuted_probs=output[5],
                backend=TURBO_BACKEND,
            )

        benchmarks["hip"] = {
            "preprocess": lambda: hip_preprocess(routing_map, num_out_tokens),
            "permute_kernel": lambda: hip_permute_kernel(
                tokens,
                hip_state.row_id_map,
                hip_state.num_dispatched_tokens,
                probs,
                num_out_tokens,
            ),
            "unpermute_kernel": lambda: hip_unpermute_kernel(hip_state, tokens),
            "permute_e2e": hip_e2e,
            "roundtrip": hip_roundtrip,
        }

    return benchmarks


def print_results(results: List[BenchResult]) -> None:
    operation_width = max(len(row.operation) for row in results)
    header = (
        f"{'operation':<{operation_width}}  {'backend':<7}  "
        f"{'event_us':>12}  {'wall_us':>12}  {'HIP/Triton':>11}"
    )
    print(header)
    print("-" * len(header))

    by_operation: Dict[str, Dict[str, BenchResult]] = {}
    for row in results:
        by_operation.setdefault(row.operation, {})[row.backend] = row

    for operation, backend_rows in by_operation.items():
        ratio = None
        if "triton" in backend_rows and "hip" in backend_rows:
            ratio = backend_rows["hip"].event_us / backend_rows["triton"].event_us
        for backend in ("triton", "hip"):
            if backend not in backend_rows:
                continue
            row = backend_rows[backend]
            ratio_text = (
                f"{ratio:.3f}x" if ratio is not None and backend == "hip" else ""
            )
            print(
                f"{operation:<{operation_width}}  {backend:<7}  "
                f"{row.event_us:12.3f}  {row.wall_us:12.3f}  {ratio_text:>11}"
            )


def write_csv(path: Path, results: List[BenchResult], args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "tokens",
                "experts",
                "hidden",
                "topk",
                "dtype",
                "with_probs",
                "operation",
                "backend",
                "event_us",
                "wall_us",
            ],
        )
        writer.writeheader()
        for row in results:
            writer.writerow(
                {
                    "tokens": args.tokens,
                    "experts": args.experts,
                    "hidden": args.hidden,
                    "topk": args.topk,
                    "dtype": args.dtype,
                    "with_probs": args.with_probs,
                    "operation": row.operation,
                    "backend": row.backend,
                    "event_us": f"{row.event_us:.6f}",
                    "wall_us": f"{row.wall_us:.6f}",
                }
            )


def main() -> None:
    args = parse_args()
    validate_args(args)
    load_backend_symbols()

    tokens, routing_map, probs = make_inputs(args)
    num_out_tokens = args.tokens * args.topk

    requested_backends = (
        ("triton", "hip") if args.backend == "both" else (args.backend,)
    )
    states: Dict[str, BackendState] = {}
    if "triton" in requested_backends:
        states["triton"] = prepare_triton_state(
            tokens,
            routing_map,
            probs,
            num_out_tokens,
        )
    if "hip" in requested_backends:
        states["hip"] = prepare_hip_state(
            tokens,
            routing_map,
            probs,
            num_out_tokens,
        )
    torch.cuda.synchronize()

    print(
        f"GPU={torch.cuda.get_device_name()} torch={torch.__version__} "
        f"shape=({args.tokens}, {args.experts}, {args.hidden}, topk={args.topk}) "
        f"dtype={args.dtype} probs={args.with_probs}"
    )

    if not args.skip_check:
        check_correctness(args, tokens, routing_map, probs, states)

    callables = make_benchmarks(
        tokens,
        routing_map,
        probs,
        states,
        num_out_tokens,
    )
    operation_order = (
        "preprocess",
        "permute_kernel",
        "unpermute_kernel",
        "permute_e2e",
        "roundtrip",
    )
    results: List[BenchResult] = []
    for operation in operation_order:
        for backend in requested_backends:
            event_us, wall_us = benchmark(
                callables[backend][operation],
                args.warmup,
                args.iters,
            )
            results.append(
                BenchResult(
                    operation=operation,
                    backend=backend,
                    event_us=event_us,
                    wall_us=wall_us,
                )
            )

    print_results(results)
    if args.output_csv is not None:
        write_csv(args.output_csv, results, args)
        print(f"Wrote CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
