import json
import os
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import torch
import torch.distributed as dist
from tabulate import tabulate

from primus_turbo.pytorch import deep_ep

# fmt: off
PROJECT_ROOT = Path(os.path.dirname(__file__)).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))


from benchmark.ops.pytorch.deep_ep.model_cfg import DeepEPModelCfg, get_model_cfg
from tests.pytorch.ref.deep_ep_ref import tune_and_verify_internode

# fmt: on


@dataclass
class DeepEPInterNodePerf(DeepEPModelCfg):
    mode: str
    nvl_chunk_size: int
    rdma_chunk_size: int
    num_sms: int
    rdma_bandwith: float
    nvl_bandwith: float
    notfy: float


def bench_kineto(
    fn,
    kernel_names: Union[str, tuple],
    num_tests: int = 30,
    trace_path: Optional[str] = None,
    barrier_comm_profiling: bool = False,
    num_kernels_per_period: int = 1,
):
    # Profile
    schedule = torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=1)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA], schedule=schedule) as prof:
        for i in range(2):
            # NOTES: use a large kernel and a barrier to eliminate the unbalanced CPU launch overhead
            if barrier_comm_profiling:
                lhs = torch.randn((8192, 8192), dtype=torch.float, device="cuda")
                rhs = torch.randn((8192, 8192), dtype=torch.float, device="cuda")
                lhs @ rhs
                dist.all_reduce(torch.ones(1, dtype=torch.float, device="cuda"))
            for _ in range(num_tests):
                fn()
            prof.step()

    # Parse the profiling table
    assert isinstance(kernel_names, str) or isinstance(kernel_names, tuple)
    is_tuple = isinstance(kernel_names, tuple)
    prof_lines = prof.key_averages().table(sort_by="cuda_time_total", max_name_column_width=100).split("\n")
    kernel_names = (kernel_names,) if isinstance(kernel_names, str) else kernel_names
    assert all([isinstance(name, str) for name in kernel_names])
    for name in kernel_names:
        assert (
            sum([name in line for line in prof_lines]) == 1
        ), f"Errors of the kernel {name} in the profiling table"

    # Save chrome traces
    if trace_path is not None:
        prof.export_chrome_trace(trace_path)

    # Return average kernel durations
    units = {"ms": 1e3, "us": 1e6}
    kernel_durations = []
    for name in kernel_names:
        for line in prof_lines:
            if name in line:
                time_str = line.split()[-2]
                for unit, scale in units.items():
                    if unit in time_str:
                        kernel_durations.append(float(time_str.replace(unit, "")) / scale)
                        break
                break

    # Expand the kernels by periods
    if num_kernels_per_period > 1:
        with tempfile.NamedTemporaryFile(suffix=".json") as tmp:
            prof.export_chrome_trace(tmp.name)
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


def bench_internode(
    cfg: DeepEPModelCfg,
    num_sms: int,
    local_rank: int,
    num_local_ranks: int,
    num_ranks: int,
    num_nodes: int,
    rank: int,
    buffer: deep_ep.Buffer,
    group: dist.ProcessGroup,
):
    # Settings
    num_tokens, hidden = cfg.num_tokens, cfg.hidden_size
    num_topk, num_experts = cfg.num_topk, cfg.num_experts

    num_topk_groups = min(4, num_nodes)
    (
        x,
        x_e4m3,
        (
            dispatch_bf16_rdma_send_bytes,
            combine_bf16_rdma_recv_bytes,
            dispatch_bf16_nvl_recv_bytes,
            combine_bf16_nvl_send_bytes,
            rdma_buffer_size,
            nvl_buffer_size,
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            is_token_in_rank,
            num_tokens_per_expert,
            handle,
        ),
    ) = tune_and_verify_internode(
        num_sms,
        num_tokens,
        hidden,
        num_topk,
        num_topk_groups,
        num_experts,
        local_rank,
        num_local_ranks,
        num_ranks,
        num_nodes,
        rank,
        buffer,
        group,
    )

    # Tune dispatch performance
    best_dispatch_results = None
    fp8_factor = (1 + 4 / 128) / 2
    for current_x in (x_e4m3, x):
        best_time, best_results = 1e10, None
        rdma_send_bytes = (
            (dispatch_bf16_rdma_send_bytes * fp8_factor)
            if isinstance(current_x, tuple)
            else dispatch_bf16_rdma_send_bytes
        )
        nvl_recv_bytes = (
            (dispatch_bf16_nvl_recv_bytes * fp8_factor)
            if isinstance(current_x, tuple)
            else dispatch_bf16_nvl_recv_bytes
        )
        for nvl_chunk_size in range(4, 45, 4):
            for rdma_chunk_size in range(4, 33, 4):
                config = deep_ep.Config(
                    num_sms, nvl_chunk_size, nvl_buffer_size, rdma_chunk_size, rdma_buffer_size
                )
                tune_args = {"x": current_x, "handle": handle, "config": config}
                t, notify_t = bench_kineto(lambda: buffer.dispatch(**tune_args), ("dispatch", "notify"))
                if t < best_time:
                    best_time, best_results = t, (num_sms, nvl_chunk_size, rdma_chunk_size, notify_t)
                if local_rank == 0:
                    print(
                        f"[{cfg.model_name}] [tuning] SMs {num_sms}, NVL chunk {nvl_chunk_size}, RDMA chunk {rdma_chunk_size}, transmit: {t * 1e6:.2f} us, notify: {notify_t * 1e6:.2f} us, BW: {rdma_send_bytes / 1e9 / t:.2f} GB/s (RDMA), {nvl_recv_bytes / 1e9 / t:.2f} GB/s (NVL) ",
                        flush=True,
                    )
        if local_rank == 0:
            print(
                f'[{cfg.model_name}] [tuning] Best dispatch ({"FP8" if isinstance(current_x, tuple) else "BF16"}): SMs {best_results[0]}, NVL chunk {best_results[1]}, RDMA chunk {best_results[2]}, transmit: {best_time * 1e6:.2f} us, notify: {best_results[3] * 1e6:.2f} us, BW: {rdma_send_bytes / 1e9 / best_time:.2f} GB/s (RDMA), {nvl_recv_bytes / 1e9 / best_time:.2f} GB/s (NVL)',
                flush=True,
            )
            print("", flush=True)

            dispatch_result = DeepEPInterNodePerf(
                **asdict(cfg),
                mode="dispatch",
                nvl_chunk_size=best_results[1],
                rdma_chunk_size=best_results[2],
                num_sms=best_results[0],
                rdma_bandwith=rdma_send_bytes / 1e9 / best_time,
                nvl_bandwith=nvl_recv_bytes / 1e9 / best_time,
                notfy=best_results[3] * 1e6,
            )

        if isinstance(current_x, tuple):
            # Gather FP8 the best config from rank 0
            best_dispatch_results = torch.tensor(
                [best_results[0], best_results[1], best_results[2]], dtype=torch.int32, device="cuda"
            )
            all_best_fp8_results_list = [
                torch.zeros_like(best_dispatch_results) for _ in range(torch.distributed.get_world_size())
            ]
            dist.all_gather(all_best_fp8_results_list, best_dispatch_results, group=group)
            best_dispatch_results = all_best_fp8_results_list[0].tolist()
    dispatch_config = deep_ep.Config(
        best_dispatch_results[0],
        best_dispatch_results[1],
        nvl_buffer_size,
        best_dispatch_results[2],
        rdma_buffer_size,
    )

    dispatch_args = {
        "x": x,
        "num_tokens_per_rank": num_tokens_per_rank,
        "num_tokens_per_rdma_rank": num_tokens_per_rdma_rank,
        "is_token_in_rank": is_token_in_rank,
        "num_tokens_per_expert": num_tokens_per_expert,
        "config": dispatch_config if dispatch_config is not None else config,
    }
    recv_x, _, _, _, handle, _ = buffer.dispatch(**dispatch_args)

    # Tune combine performance
    best_time, best_results = 1e10, None
    for nvl_chunk_size in range(1, 8, 1):
        for rdma_chunk_size in range(12 if num_nodes == 2 else 8, 33, 4):
            config = deep_ep.Config(
                num_sms, nvl_chunk_size, nvl_buffer_size, rdma_chunk_size, rdma_buffer_size
            )
            tune_args = {"x": recv_x, "handle": handle, "config": config}
            t, notify_t = bench_kineto(lambda: buffer.combine(**tune_args), ("combine", "notify"))
            if local_rank == 0:
                print(
                    f"[{cfg.model_name}] [tuning] SMs {num_sms}, NVL chunk {nvl_chunk_size}, RDMA chunk {rdma_chunk_size}, transmit: {t * 1e6:.2f} us, notify: {notify_t * 1e6:.2f} us, BW: {combine_bf16_rdma_recv_bytes / 1e9 / t:.2f} GB/s (RDMA), {combine_bf16_nvl_send_bytes / 1e9 / t:.2f} GB/s (NVL) ",
                    flush=True,
                )
                if t < best_time:
                    best_time, best_results = t, (num_sms, nvl_chunk_size, rdma_chunk_size, notify_t)

    if local_rank == 0:
        print(
            f"[{cfg.model_name}] [tuning] Best combine: SMs {best_results[0]}, NVL chunk {best_results[1]}, RDMA chunk {best_results[2]}, transmit: {best_time * 1e6:.2f} us, notify: {best_results[3] * 1e6:.2f} us, BW: {combine_bf16_rdma_recv_bytes / 1e9 / best_time:.2f} GB/s (RDMA), {combine_bf16_nvl_send_bytes / 1e9 / best_time:.2f} GB/s (NVL)",
            flush=True,
        )
        print("", flush=True)
        combine_result = DeepEPInterNodePerf(
            **asdict(cfg),
            mode="combine",
            nvl_chunk_size=best_results[1],
            rdma_chunk_size=best_results[2],
            num_sms=best_results[0],
            rdma_bandwith=combine_bf16_rdma_recv_bytes / 1e9 / best_time,
            nvl_bandwith=combine_bf16_nvl_send_bytes / 1e9 / best_time,
            notfy=best_results[3] * 1e6,
        )

    return (dispatch_result, combine_result) if local_rank == 0 else (None, None)


def init_dist(local_rank: int, num_local_ranks: int, backend: str = "nccl"):
    if backend == "nccl":
        ip = os.getenv("MASTER_ADDR", "127.0.0.1")
        port = int(os.getenv("MASTER_PORT", "8361"))
        node_rank = int(os.getenv("RANK", 0))
    num_nodes = int(os.getenv("WORLD_SIZE", 1))
    assert (num_local_ranks < 8 and num_nodes == 1) or num_local_ranks == 8

    if backend == "nccl":
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://{ip}:{port}",
            world_size=num_nodes * num_local_ranks,
            rank=node_rank * num_local_ranks + local_rank,
        )
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.cuda.set_device(local_rank)
    return dist.get_rank(), dist.get_world_size(), dist.new_group(list(range(num_local_ranks * num_nodes)))


def record_perf(local_rank: int, num_local_ranks: int):

    # DataFrame to store results
    best_result = []
    num_nodes = int(os.getenv("WORLD_SIZE", 1))
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    buffer = deep_ep.Buffer(
        group, int(2e9), int(1e9), low_latency_mode=False, num_qps_per_rank=32, explicitly_destroy=True
    )
    assert num_local_ranks == 8 and num_ranks > 8
    torch.manual_seed(rank)

    for cfg in get_model_cfg():
        for num_sms in (
            24,
            32,
        ):
            dispatch_result, combine_result = bench_internode(
                cfg, num_sms, local_rank, num_local_ranks, num_ranks, num_nodes, rank, buffer, group
            )
            if local_rank == 0:
                best_result.extend([dispatch_result, combine_result])
    df_result = pd.DataFrame(best_result)

    if rank == 0:
        # Print results
        print("\nFinal Results:")
        print(tabulate(df_result, headers="keys", tablefmt="grid", showindex=False))

        # Save to CSV
        df_result.to_csv("deep_ep_internode_benchmark_results.csv", index=False)
        print("Results saved to deep_ep_internode_benchmark_results.csv")

    buffer.destroy()
    dist.barrier()
    dist.destroy_process_group(group)


if __name__ == "__main__":
    torch.multiprocessing.spawn(record_perf, args=(8,), nprocs=8)
