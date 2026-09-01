###############################################################################
# Copyright (c) 2025 DeepSeek. All rights reserved.
#
# Modification Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Derived from DeepEP (https://github.com/deepseek-ai/DeepEP), the
# `deep_ep/utils/{envs,math,testing}.py` modules merged into one file.
#
# See LICENSE for license information.
###############################################################################

import inspect
import json
import os
import random
import sys
import tempfile
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist

_local_rank = None
_local_seed = 0
_global_seed = 0


def get_deep_ep_backend(backend: str):
    if backend == "deep_ep":
        import deep_ep
    elif backend == "turbo":
        from primus_turbo.pytorch import deep_ep
    else:
        raise ValueError(f"Invalid backend: {backend}")

    return deep_ep


# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------
def init_seed(global_seed: int) -> None:
    """
    Initialize the random seed for reproducibility. The local seed is derived from the global seed plus rank.

    Arguments:
        global_seed: the global random seed.
    """
    global _local_seed, _global_seed
    _local_seed = global_seed + dist.get_rank()
    _global_seed = global_seed
    torch.manual_seed(_local_seed)
    random.seed(_local_seed)


def get_local_seed() -> int:
    """
    Get the local random seed.

    Returns:
        seed: the local random seed.
    """
    return _local_seed


def get_global_seed() -> int:
    """
    Get the global random seed.

    Returns:
        seed: the global random seed.
    """
    return _global_seed


def dist_print(s: str = "", once_in_node: bool = False) -> None:
    """
    Print a message from all ranks, or only from rank 0 of each node, followed by a barrier.

    Arguments:
        s: the message to print.
        once_in_node: if `True`, only the first local rank in each node prints.
    """
    global _local_rank
    assert _local_rank is not None
    if not once_in_node or _local_rank == 0:
        print(s, flush=True)
    dist.barrier()


def init_dist(local_rank: int, num_local_ranks: int, seed: int = 0) -> Tuple[int, int, dist.ProcessGroup]:
    """
    Initialize the distributed environment with NCCL backend.

    Arguments:
        local_rank: the local rank index.
        num_local_ranks: the number of local ranks.
        seed: the global random seed.

    Returns:
        rank: the global rank index.
        world_size: the total number of ranks.
        group: the communication group.
    """
    # NOTES: you may rewrite this function with your own cluster settings
    ip = os.getenv("MASTER_ADDR", "127.0.0.1")
    port = int(os.getenv("MASTER_PORT", "8361"))
    num_nodes = int(os.getenv("WORLD_SIZE", 1))
    node_rank = int(os.getenv("RANK", 0))

    # Set local rank
    global _local_rank
    _local_rank = local_rank

    sig = inspect.signature(dist.init_process_group)
    params = {
        "backend": "nccl",
        "init_method": f"tcp://{ip}:{port}",
        "world_size": num_nodes * num_local_ranks,
        "rank": node_rank * num_local_ranks + local_rank,
    }
    if "device_id" in sig.parameters:
        # noinspection PyTypeChecker
        params["device_id"] = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(**params)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.cuda.set_device(local_rank)

    init_seed(seed)
    return dist.get_rank(), dist.get_world_size(), dist.new_group(list(range(num_local_ranks * num_nodes)))


# -----------------------------------------------------------------------------
# Math
# -----------------------------------------------------------------------------
def calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    x, y = x.double() + 1, y.double() + 1
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return (1 - sim).item()


def safe_div(a, b) -> float:
    try:
        return a / b
    except ZeroDivisionError:
        if a == 0:
            return 0
        else:
            raise


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def align(x: int, y: int) -> int:
    return ceil_div(x, y) * y


@torch.compile(dynamic=True)
def per_token_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    aligned_n = align(n, 128)
    x_padded = torch.nn.functional.pad(x, (0, aligned_n - n), mode="constant", value=0)
    x_padded_view = x_padded.view(m, -1, 128)
    x_amax = x_padded_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    return (x_padded_view * (448.0 / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn).view(m, aligned_n)[
        :, :n
    ].contiguous(), (x_amax / 448.0).view(m, -1)


@torch.compile(dynamic=True)
def per_token_cast_back(x_fp8: torch.Tensor, x_scales: torch.Tensor) -> torch.Tensor:
    if x_fp8.numel() == 0:
        return x_fp8.to(torch.bfloat16)

    assert x_fp8.dim() == 2
    m, n = x_fp8.shape
    aligned_n = align(n, 128)
    x_fp8_padded = torch.nn.functional.pad(x_fp8, (0, aligned_n - n), mode="constant", value=0)
    if x_scales.dtype == torch.int:
        x_scales = x_scales.view(dtype=torch.uint8).to(torch.int) << 23
        x_scales = x_scales.view(dtype=torch.float)
    x_fp32_padded = x_fp8_padded.to(torch.float32).view(x_fp8.shape[0], -1, 128)
    x_scales = x_scales.view(x_fp8.shape[0], -1, 1)
    return (x_fp32_padded * x_scales).view(x_fp8_padded.shape).to(torch.bfloat16)[:, :n].contiguous()


def inplace_unique(x: torch.Tensor, num_slots: int) -> None:
    assert x.dim() == 2
    mask = x < 0
    x_padded = x.masked_fill(mask, num_slots)
    bin_count = torch.zeros((x.size(0), num_slots + 1), dtype=x.dtype, device=x.device)
    bin_count.scatter_add_(1, x_padded, torch.ones_like(x_padded))
    bin_count = bin_count[:, :num_slots]
    sorted_bin_count, sorted_bin_idx = torch.sort(bin_count, dim=-1, descending=True)
    sorted_bin_idx.masked_fill_(sorted_bin_count == 0, -1)
    sorted_bin_idx = torch.sort(sorted_bin_idx, descending=True, dim=-1).values
    x[:, :].fill_(-1)
    valid_len = min(num_slots, x.size(1))
    x[:, :valid_len] = sorted_bin_idx[:, :valid_len]


def create_grouped_scores(scores: torch.Tensor, group_idx: torch.Tensor, num_groups: int) -> torch.Tensor:
    num_tokens, num_experts = scores.shape
    scores = scores.view(num_tokens, num_groups, -1)
    mask = torch.zeros((num_tokens, num_groups), dtype=torch.bool, device=scores.device)
    mask = mask.scatter_(1, group_idx, True).unsqueeze(-1).expand_as(scores)
    return (scores * mask).view(num_tokens, num_experts)


def hash_tensor(t: torch.Tensor) -> int:
    return t.view(torch.int).sum().item()


def hash_tensors(*tensors) -> int:
    value = 0
    for t in tensors:
        if isinstance(t, (tuple, list)):
            value ^= hash_tensors(*t)
        elif t is not None and isinstance(t, torch.Tensor):
            value ^= hash_tensor(t)
    return value


def count_bytes(*tensors) -> int:
    total = 0
    for t in tensors:
        if isinstance(t, (tuple, list)):
            total += count_bytes(*t)
        elif t is not None:
            total += t.numel() * t.element_size()
    return total


# -----------------------------------------------------------------------------
# Testing
# -----------------------------------------------------------------------------
def flush_l2_cache(enabled: bool = True):
    """
    Flush the GPU L2 cache by writing a large zero-initialized tensor.

    Arguments:
        enabled: if `False`, does nothing.
    """
    l2_flush_cache_size = 256e6
    if enabled:
        torch.empty(int(l2_flush_cache_size // 4), dtype=torch.int, device="cuda").zero_()


def bench(
    fn, num_warmups: int = 50, num_tests: int = 50, post_fn: Optional[Callable] = None, flush_l2: bool = True
):
    """
    Benchmark a function using CUDA events.

    Arguments:
        fn: the function to benchmark.
        num_warmups: the number of warmup iterations.
        num_tests: the number of measurement iterations.
        post_fn: an optional function to call after each test iteration.
        flush_l2: whether to flush the L2 cache before each iteration.

    Returns:
        avg: the average execution time in seconds.
        min: the minimum execution time in seconds.
        max: the maximum execution time in seconds.
    """
    torch.cuda.synchronize()

    # Warmup
    for _ in range(num_warmups):
        fn()

    # Testing
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_tests)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_tests)]
    for i in range(num_tests):
        flush_l2_cache(flush_l2)
        start_events[i].record()
        fn()
        end_events[i].record()
        if post_fn is not None:
            post_fn()
    torch.cuda.synchronize()

    times = np.array([s.elapsed_time(e) / 1e3 for s, e in zip(start_events, end_events)])[1:]
    return np.average(times), np.min(times), np.max(times)


class empty_suppress:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


class suppress_stdout_stderr:
    """
    Context manager to suppress stdout and stderr output.
    """

    def __enter__(self):
        self.outnull_file = open(os.devnull, "w")
        self.errnull_file = open(os.devnull, "w")

        self.old_stdout_fileno_undup = sys.stdout.fileno()
        self.old_stderr_fileno_undup = sys.stderr.fileno()

        self.old_stdout_fileno = os.dup(sys.stdout.fileno())
        self.old_stderr_fileno = os.dup(sys.stderr.fileno())

        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr

        os.dup2(self.outnull_file.fileno(), self.old_stdout_fileno_undup)
        os.dup2(self.errnull_file.fileno(), self.old_stderr_fileno_undup)

        sys.stdout = self.outnull_file
        sys.stderr = self.errnull_file
        return self

    def __exit__(self, *_):
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr

        os.dup2(self.old_stdout_fileno, self.old_stdout_fileno_undup)
        os.dup2(self.old_stderr_fileno, self.old_stderr_fileno_undup)

        os.close(self.old_stdout_fileno)
        os.close(self.old_stderr_fileno)

        self.outnull_file.close()
        self.errnull_file.close()


def bench_kineto(
    fn,
    kernel_names: Union[str, tuple],
    num_tests: int = 30,
    suppress_kineto_output: bool = False,
    trace_path: Optional[str] = None,
    flush_l2: bool = True,
    barrier_comm_profiling: bool = False,
    num_kernels_per_period: int = 1,
    barrier: Optional[Callable] = None,
):
    """
    Benchmark a function using the PyTorch profiler (kineto) to get per-kernel timing.

    Arguments:
        fn: the function to benchmark.
        kernel_names: the CUDA kernel name(s) to profile.
        num_tests: the number of test iterations.
        suppress_kineto_output: whether to suppress profiler output.
        trace_path: the path to save the Chrome trace (`None` to skip).
        flush_l2: whether to flush the L2 cache before each iteration.
        barrier_comm_profiling: whether to insert a barrier before each iteration to reduce
            unbalanced CPU launch overhead.
        num_kernels_per_period: the number of kernels launched per test period.
        barrier: a custom barrier function to use instead of `dist.all_reduce`.

    Returns:
        durations: the average kernel duration(s) in seconds.
    """
    assert isinstance(kernel_names, (str, tuple))
    is_tuple = isinstance(kernel_names, tuple)

    # Skip profiling
    # Conflict with Nsight Systems, Nsight Compute and Compute Sanitizer
    if int(os.environ.get("EP_USE_NVIDIA_TOOLS", 0)):
        return (1,) * len(kernel_names) if is_tuple else 1

    # For some auto-tuning kernels with prints
    fn()
    torch.cuda.synchronize()

    # Profile
    suppress = suppress_stdout_stderr if suppress_kineto_output else empty_suppress
    barrier_comm_profiling &= int(os.environ.get("EP_DISABLE_BARRIER_PROFILING", 0)) == 0
    with suppress():
        schedule = torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=1)
        profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA], schedule=schedule, acc_events=True
        )
        dummy = torch.ones(1, dtype=torch.float, device="cuda")
        with profiler:
            for i in range(2):
                for _ in range(num_tests):
                    # Flush L2 cache
                    flush_l2_cache(flush_l2)

                    # NOTES: use a large kernel and a barrier to eliminate the unbalanced CPU launch overhead
                    if barrier_comm_profiling:
                        torch.cuda._sleep(int(2e7))  # ~10ms

                        # Some network may have ring-based implement, so be careful to use `all_reduce`
                        if barrier is None:
                            dist.all_reduce(dummy)
                        else:
                            barrier()
                    fn()
                torch.cuda.synchronize()
                profiler.step()

    # Parse the profiling table
    prof_lines = (
        profiler.key_averages().table(sort_by="cuda_time_total", max_name_column_width=100).split("\n")
    )
    kernel_names = (kernel_names,) if isinstance(kernel_names, str) else kernel_names
    assert all([isinstance(name, str) for name in kernel_names])
    for name in kernel_names:
        assert sum([name in line for line in prof_lines]) <= 1, (
            f"Errors of the kernel {name} in the profiling table: {prof_lines}"
        )

    # Save chrome traces
    if trace_path is not None:
        profiler.export_chrome_trace(trace_path)

    # Return average kernel durations
    units = {"ms": 1e3, "us": 1e6}
    kernel_durations = []
    for name in kernel_names:
        total_time = 0
        total_num = 0
        for line in prof_lines:
            if name in line:
                time_str = line.split()[-2]
                num_str = line.split()[-1]
                for unit, scale in units.items():
                    if unit in time_str:
                        total_time += float(time_str.replace(unit, "")) / scale * int(num_str)
                        total_num += int(num_str)
                        break
        kernel_durations.append(total_time / total_num if total_num > 0 else 0)

    # Expand the kernels by periods
    if num_kernels_per_period > 1:
        with tempfile.NamedTemporaryFile(suffix=".json") as tmp:
            profiler.export_chrome_trace(tmp.name)
            profile_data = json.loads(Path(tmp.name).read_text())

        for i, kernel_name in enumerate(kernel_names):
            events = [event for event in profile_data["traceEvents"] if f"::{kernel_name}" in event["name"]]
            events = sorted(events, key=lambda event: event["ts"])
            durations = [event["dur"] / 1e6 for event in events]
            assert len(durations) % num_kernels_per_period == 0
            num_kernel_patterns = len(durations) // num_kernels_per_period
            kernel_durations[i] = [
                sum(durations[j::num_kernels_per_period]) / num_kernel_patterns
                for j in range(num_kernels_per_period)
            ]

    # Return execution durations
    return kernel_durations if is_tuple else kernel_durations[0]
