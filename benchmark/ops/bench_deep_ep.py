###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import os
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from tabulate import tabulate

from primus_turbo.pytorch import deep_ep
from tests.pytorch.ref.deep_ep_ref import tune_and_verify_intranode


@dataclass
class DeepEPModelCfg:
    model_name: str
    hidden_size: int
    num_experts: int
    num_topk: int
    seqlen: int
    batch_size: int

    def __post_init__(self):
        self.num_tokens = self.batch_size * self.seqlen


@dataclass
class DeepEPPerf(DeepEPModelCfg):
    mode: str
    nvl_chunk_size: int
    num_sms: int
    bandwith: float
    avg_us: float


g_model_cfg = [
    DeepEPModelCfg(
        model_name="deepseekv3", hidden_size=7168, num_experts=256, num_topk=8, seqlen=4096, batch_size=1
    ),
    DeepEPModelCfg(
        model_name="deepseekv2", hidden_size=5120, num_experts=160, num_topk=6, seqlen=4096, batch_size=1
    ),
    DeepEPModelCfg(
        model_name="qwen3_235b", hidden_size=4096, num_experts=128, num_topk=6, seqlen=4096, batch_size=1
    ),
    DeepEPModelCfg(
        model_name="poolside-515B", hidden_size=8192, num_experts=112, num_topk=8, seqlen=4096, batch_size=1
    ),
]


def bench(fn, num_warmups: int = 20, num_tests: int = 30, post_fn=None):
    # Flush L2 cache with 256 MB data
    torch.cuda.synchronize()
    cache = torch.empty(int(256e6 // 4), dtype=torch.int, device="cuda")

    # Warmup
    for _ in range(num_warmups):
        fn()

    # Flush L2
    cache.zero_()

    # Testing
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_tests)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_tests)]
    for i in range(num_tests):
        # Record
        start_events[i].record()
        fn()
        end_events[i].record()
        if post_fn is not None:
            post_fn()
    torch.cuda.synchronize()

    times = np.array([s.elapsed_time(e) / 1e3 for s, e in zip(start_events, end_events)])[1:]
    return np.average(times), np.min(times), np.max(times)


def bench_intranode(
    cfg: DeepEPModelCfg,
    num_sms: int,
    local_rank: int,
    num_ranks: int,
    rank: int,
    buffer: deep_ep.Buffer,
    group: dist.ProcessGroup,
):
    # Settings
    num_tokens, hidden = cfg.num_tokens, cfg.hidden_size
    num_topk, num_experts = cfg.num_topk, cfg.num_experts

    (
        x,
        x_e4m3,
        (
            dispatch_bf16_nvl_recv_bytes,
            combine_bf16_nvl_send_bytes,
            nvl_buffer_size,
            num_tokens_per_rank,
            is_token_in_rank,
            num_tokens_per_expert,
            handle,
        ),
    ) = tune_and_verify_intranode(
        num_sms, num_tokens, hidden, num_topk, num_experts, local_rank, num_ranks, rank, buffer, group
    )

    # Random data

    # Tune dispatch performance
    best_dispatch_results = None
    fp8_factor = (1 + 4 / 128) / 2
    for current_x in filter(lambda elem: elem is not None, (x_e4m3, x)):
        best_time, best_results = 1e10, None
        nvl_recv_bytes = (
            (dispatch_bf16_nvl_recv_bytes * fp8_factor)
            if isinstance(current_x, tuple)
            else dispatch_bf16_nvl_recv_bytes
        )
        for nvl_chunk_size in tuple(range(4, 33, 2)) + (0,):
            if nvl_chunk_size > 0:
                config = deep_ep.Config(num_sms, nvl_chunk_size, nvl_buffer_size)
            else:
                # Test default config as well
                deep_ep.Buffer.set_num_sms(num_sms)
                config = deep_ep.Buffer.get_dispatch_config(num_ranks)
            tune_args = {"x": current_x, "handle": handle, "config": config}
            t = bench(lambda: buffer.dispatch(**tune_args))[0]
            if t < best_time and nvl_chunk_size > 0:
                best_time, best_results = t, (num_sms, nvl_chunk_size)
            if local_rank == 0:
                print(
                    f'[{cfg.model_name} tuning] SMs {num_sms}, NVL chunk {nvl_chunk_size if nvl_chunk_size else "default"}: '
                    f"{nvl_recv_bytes / 1e9 / t:.2f} GB/s (NVL), avg_t: {t * 1e6:.2f} us",
                    flush=True,
                )

        if local_rank == 0:
            print(
                f'[{cfg.model_name} tuning] Best dispatch ({"FP8" if isinstance(current_x, tuple) else "BF16"}): SMs {best_results[0]}, NVL chunk {best_results[1]}, {nvl_recv_bytes / 1e9 / best_time:.2f} GB/s (NVL), t: {best_time * 1e6:.2f} us',
                flush=True,
            )
            print("", flush=True)
            dispatch_result = DeepEPPerf(
                **asdict(cfg),
                mode="intranode-dispatch",
                nvl_chunk_size=best_results[1],
                num_sms=num_sms,
                bandwith=nvl_recv_bytes / 1e9 / best_time,
                avg_us=best_time * 1e6,
            )

        # Gather the best config from rank 0 and the first test setting
        if best_dispatch_results is None:
            best_dispatch_results = torch.tensor(
                [best_results[0], best_results[1]], dtype=torch.int32, device="cuda"
            )
            all_best_fp8_results_list = [
                torch.zeros_like(best_dispatch_results) for _ in range(torch.distributed.get_world_size())
            ]
            dist.all_gather(all_best_fp8_results_list, best_dispatch_results, group=group)
            best_dispatch_results = all_best_fp8_results_list[0].tolist()
    dispatch_config = deep_ep.Config(best_dispatch_results[0], best_dispatch_results[1], nvl_buffer_size)

    dispatch_args = {
        "x": x,
        "num_tokens_per_rank": num_tokens_per_rank,
        "is_token_in_rank": is_token_in_rank,
        "num_tokens_per_expert": num_tokens_per_expert,
        "config": dispatch_config if dispatch_config is not None else config,
    }
    recv_x, _, _, _, handle, _ = buffer.dispatch(**dispatch_args)

    # Tune combine performance
    best_time, best_results = 1e10, None
    for nvl_chunk_size in tuple(range(1, 17, 1)) + (0,):
        if nvl_chunk_size > 0:
            config = deep_ep.Config(num_sms, nvl_chunk_size, nvl_buffer_size)
        else:
            # Test default config as well
            deep_ep.Buffer.set_num_sms(num_sms)
            config = deep_ep.Buffer.get_combine_config(num_ranks)
        tune_args = {"x": recv_x, "handle": handle, "config": config}
        t = bench(lambda: buffer.combine(**tune_args))[0]
        if local_rank == 0:
            print(
                f'[{cfg.model_name} tuning] SMs {num_sms}, NVL chunk {nvl_chunk_size if nvl_chunk_size else "default"}: '
                f"{combine_bf16_nvl_send_bytes / 1e9 / t:.2f} GB/s (NVL), avg_t: {t * 1e6:.2f} us",
                flush=True,
            )
            if t < best_time and nvl_chunk_size > 0:
                best_time, best_results = t, (num_sms, nvl_chunk_size)

    if local_rank == 0:
        print(
            f"[{cfg.model_name} tuning] Best combine: SMs {best_results[0]}, NVL chunk {best_results[1]}: {combine_bf16_nvl_send_bytes / 1e9 / best_time:.2f} GB/s (NVL), t: {best_time * 1e6:.2f} us",
            flush=True,
        )
        print("", flush=True)

        combine_result = DeepEPPerf(
            **asdict(cfg),
            mode="intranode-combine",
            nvl_chunk_size=best_results[1],
            num_sms=num_sms,
            bandwith=combine_bf16_nvl_send_bytes / 1e9 / best_time,
            avg_us=best_time * 1e6,
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
    global g_model_cfg

    # DataFrame to store results
    best_result = []
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    buffer = deep_ep.Buffer(group, int(2e9), 0, False, 1, explicitly_destroy=True)
    torch.manual_seed(rank)

    for cfg in g_model_cfg:
        for num_sms in (24, 32, 64, 80):
            dispatch_result, combine_result = bench_intranode(
                cfg, num_sms, local_rank, num_ranks, rank, buffer, group
            )
            if local_rank == 0:
                best_result.extend([dispatch_result, combine_result])
    df_result = pd.DataFrame(best_result)

    if local_rank == 0:
        # Print results
        print("\nFinal Results:")
        print(tabulate(df_result, headers="keys", tablefmt="grid", showindex=False))

        # Save to CSV
        df_result.to_csv("deep_ep_benchmark_results.csv", index=False)
        print("Results saved to deep_ep_benchmark_result.csv")

    buffer.destroy()
    dist.barrier()
    dist.destroy_process_group(group)


if __name__ == "__main__":
    torch.multiprocessing.spawn(record_perf, args=(8,), nprocs=8)
