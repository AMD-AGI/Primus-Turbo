import inspect
import json
import tempfile
from pathlib import Path

import numpy as np
import os
import sys
import torch
import torch.distributed as dist
from typing import Optional, Union, Tuple


def init_dist(local_rank: int, num_local_ranks: int):
    # NOTES: you may rewrite this function with your own cluster settings
    ip = os.getenv('MASTER_ADDR', '127.0.0.1')
    port = int(os.getenv('MASTER_PORT', '8361'))
    num_nodes = int(os.getenv('WORLD_SIZE', 1))
    node_rank = int(os.getenv('RANK', 0))

    sig = inspect.signature(dist.init_process_group)
    params = {
        'backend': 'nccl',
        'init_method': f'tcp://{ip}:{port}',
        'world_size': num_nodes * num_local_ranks,
        'rank': node_rank * num_local_ranks + local_rank,
    }
    if 'device_id' in sig.parameters:
        # noinspection PyTypeChecker
        params['device_id'] = torch.device(f'cuda:{local_rank}')

    # #region agent log — hypothesis C: log init_process_group params
    import json as _json
    import time as _time
    _log_path = "/io/.cursor/debug.log"
    try:
        with open(_log_path, "a") as _f:
            _f.write(_json.dumps({"timestamp": int(_time.time()*1000), "location": "utils.py:init_dist", "message": "init_process_group params", "data": {"local_rank": local_rank, "has_device_id": 'device_id' in params,
                     "device_id": str(params.get('device_id', 'N/A')), "backend": params['backend'], "world_size": params['world_size'], "rank": params['rank']}, "hypothesisId": "C", "runId": "run1"}) + "\n")
    except Exception:
        pass
    # #endregion

    dist.init_process_group(**params)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device('cuda')
    torch.cuda.set_device(local_rank)

    return dist.get_rank(), dist.get_world_size(), dist.new_group(list[int](range(num_local_ranks * num_nodes)))


def calc_diff(x: torch.Tensor, y: torch.Tensor):
    x, y = x.double() + 1, y.double() + 1
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return (1 - sim).item()


def align_up(x, y):
    return (x + y - 1) // y * y


def per_token_cast_to_fp8(x: torch.Tensor):
    assert x.dim() == 2
    m, n = x.shape
    aligned_n = align_up(n, 128)
    x_padded = torch.nn.functional.pad(
        x, (0, aligned_n - n), mode='constant', value=0)
    x_padded_view = x_padded.view(m, -1, 128)
    x_amax = x_padded_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    return (x_padded_view * (448.0 / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn).view(
        m, aligned_n)[:, :n].contiguous(), (x_amax / 448.0).view(m, -1)


def per_token_cast_back(x_fp8: torch.Tensor, x_scales: torch.Tensor):
    if x_fp8.numel() == 0:
        return x_fp8.to(torch.bfloat16)

    assert x_fp8.dim() == 2
    m, n = x_fp8.shape
    aligned_n = align_up(n, 128)
    x_fp8_padded = torch.nn.functional.pad(
        x_fp8, (0, aligned_n - n), mode='constant', value=0)
    if x_scales.dtype == torch.int:
        x_scales = x_scales.view(dtype=torch.uint8).to(torch.int) << 23
        x_scales = x_scales.view(dtype=torch.float)
    x_fp32_padded = x_fp8_padded.to(torch.float32).view(x_fp8.size(0), -1, 128)
    x_scales = x_scales.view(x_fp8.size(0), -1, 1)
    return (x_fp32_padded * x_scales).view(x_fp8_padded.shape).to(torch.bfloat16)[:, :n].contiguous()


def inplace_unique(x: torch.Tensor, num_slots: int):
    assert x.dim() == 2
    mask = x < 0
    x_padded = x.masked_fill(mask, num_slots)
    bin_count = torch.zeros((x.size(0), num_slots + 1),
                            dtype=x.dtype, device=x.device)
    bin_count.scatter_add_(1, x_padded, torch.ones_like(x_padded))
    bin_count = bin_count[:, :num_slots]
    sorted_bin_count, sorted_bin_idx = torch.sort(
        bin_count, dim=-1, descending=True)
    sorted_bin_idx.masked_fill_(sorted_bin_count == 0, -1)
    sorted_bin_idx = torch.sort(sorted_bin_idx, descending=True, dim=-1).values
    x[:, :].fill_(-1)
    valid_len = min(num_slots, x.size(1))
    x[:, :valid_len] = sorted_bin_idx[:, :valid_len]


def create_grouped_scores(scores: torch.Tensor, group_idx: torch.Tensor, num_groups: int):
    num_tokens, num_experts = scores.shape
    scores = scores.view(num_tokens, num_groups, -1)
    mask = torch.zeros((num_tokens, num_groups),
                       dtype=torch.bool, device=scores.device)
    mask = mask.scatter_(1, group_idx, True).unsqueeze(-1).expand_as(scores)
    return (scores * mask).view(num_tokens, num_experts)


def bench(fn, num_warmups: int = 50, num_tests: int = 50, post_fn=None):
    # Flush L2 cache with 256 MB data
    torch.cuda.synchronize()
    cache = torch.empty(int(256e6 // 4), dtype=torch.int, device='cuda')

    # Warmup
    for _ in range(num_warmups):
        fn()

    # Flush L2
    cache.zero_()

    # Testing
    start_events = [torch.cuda.Event(enable_timing=True)
                    for _ in range(num_tests)]
    end_events = [torch.cuda.Event(enable_timing=True)
                  for _ in range(num_tests)]
    for i in range(num_tests):
        # Record
        start_events[i].record()
        fn()
        end_events[i].record()
        if post_fn is not None:
            post_fn()
    torch.cuda.synchronize()

    times = np.array([s.elapsed_time(e) / 1e3 for s,
                     e in zip(start_events, end_events)])[1:]
    return np.average(times), np.min(times), np.max(times)


class empty_suppress:

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


class suppress_stdout_stderr:

    def __enter__(self):
        self.outnull_file = open(os.devnull, 'w')
        self.errnull_file = open(os.devnull, 'w')

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


def bench_kineto(fn,
                 kernel_names: Union[str, tuple],
                 num_tests: int = 50,
                 suppress_kineto_output: bool = False,
                 trace_path: Optional[str] = None,
                 barrier_comm_profiling: bool = False,
                 num_kernels_per_period: int = 1):
    # Profile
    suppress = suppress_stdout_stderr if suppress_kineto_output else empty_suppress
    with suppress():
        schedule = torch.profiler.schedule(
            wait=1, warmup=0, active=1, repeat=1)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA], schedule=schedule) as prof:
            for _ in range(2):
                # NOTES: use a large kernel and a barrier to eliminate the unbalanced CPU launch overhead
                if barrier_comm_profiling:
                    lhs = torch.randn(
                        (8192, 8192), dtype=torch.float, device='cuda')
                    rhs = torch.randn(
                        (8192, 8192), dtype=torch.float, device='cuda')
                    lhs @ rhs
                    dist.all_reduce(torch.ones(
                        1, dtype=torch.float, device='cuda'))
                for _ in range(num_tests):
                    fn()
                torch.cuda.synchronize()
                prof.step()

    # Parse the profiling table
    assert isinstance(kernel_names, (str, tuple))
    is_tuple = isinstance(kernel_names, tuple)
    prof_lines = prof.key_averages().table(sort_by='cuda_time_total',
                                           max_name_column_width=100).split('\n')
    kernel_names = (kernel_names, ) if isinstance(
        kernel_names, str) else kernel_names
    assert all([isinstance(name, str) for name in kernel_names])
    for name in kernel_names:
        assert sum([name in line for line in prof_lines]
                   ) == 1, f'Errors of the kernel {name} in the profiling table'

    # Save chrome traces
    if trace_path is not None:
        prof.export_chrome_trace(trace_path)

    # Return average kernel durations
    units = {'ms': 1e3, 'us': 1e6}
    kernel_durations = []
    for name in kernel_names:
        for line in prof_lines:
            if name in line:
                time_str = line.split()[-2]
                for unit, scale in units.items():
                    if unit in time_str:
                        kernel_durations.append(
                            float(time_str.replace(unit, '')) / scale)
                        break
                break

    # Expand the kernels by periods
    if num_kernels_per_period > 1:
        with tempfile.NamedTemporaryFile(suffix='.json') as tmp:
            prof.export_chrome_trace(tmp.name)
            profile_data = json.loads(Path(tmp.name).read_text())

        for i, kernel_name in enumerate(kernel_names):
            events = [event for event in profile_data['traceEvents']
                      if f'::{kernel_name}' in event['name']]
            events = sorted(events, key=lambda event: event['ts'])
            durations = [event['dur'] / 1e6 for event in events]
            assert len(durations) % num_kernels_per_period == 0
            num_kernel_patterns = len(durations) // num_kernels_per_period
            kernel_durations[i] = [sum(durations[j::num_kernels_per_period]) /
                                   num_kernel_patterns for j in range(num_kernels_per_period)]

    # Return execution durations
    return kernel_durations if is_tuple else kernel_durations[0]


def hash_tensor(t: torch.Tensor):
    return t.view(torch.int).sum().item()


def permute(
    tokens: torch.Tensor,
    routing_map: torch.Tensor,
    probs: Optional[torch.Tensor] = None,
    num_out_tokens: Optional[int] = None,
    drop_and_pad: bool = False,
    tokens_per_expert: Optional[torch.Tensor] = None,
) -> Tuple[
    torch.Tensor,
    Optional[torch.Tensor],
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    """Permute the tokens and probs based on the mask.
    Tokens with the same designated expert will be grouped together.
    The shape of mask is [tokens, num_experts], it indicates which experts were selected
    by each token.

    When drop_and_pad=True, in routing_map, the number of non-zeros in each column equals to
    expert capacity. This function exploits this feature to use ops that support cuda graph.

    If the fused permute and pad kernel is available, it will pad the tokens to the align_size
    and return the padded permuted tokens, pad_offsets and padded tokens per expert.

    Args:
        tokens (torch.Tensor): The input token tensor, [num_tokens, hidden].
        routing_map (torch.Tensor): The sparse token to expert mapping, [num_tokens, num_experts].
        probs (torch.Tensor, optional): The probs tensor, [num_tokens, num_experts].
        num_out_tokens (int, optional): The number of output tokens. If None, it's set to
                                        the number of input tokens.
        fused (bool, optional): Whether use the fused permute function.
        drop_and_pad (bool, optional): Whether or not the token dispatcher uses token-drop
                                       and pads the number of tokens to the expert capacity.
                                       If set to true, routing_map has a fixed number of non-zeros
                                       in each column.
        tokens_per_expert (torch.Tensor, optional): Tensor of shape `[num_experts]` containing
                                                    actual token counts per expert.
        align_size (int, optional): The alignment size for the input tensor for fp8 or fp4.

    Returns:
        Tuple[
            torch.Tensor,
            Optional[torch.Tensor],
            torch.Tensor,
            Optional[torch.Tensor],
            Optional[torch.Tensor],
        ]:
            The permuted tokens, (optional) permuted probs, sorted indices,
            (optional) pad_offsets, (optional) padded_tokens_per_expert.
    """

    num_tokens, hidden = tokens.shape
    num_experts = routing_map.shape[1]
    permuted_probs = None
    if drop_and_pad and not (num_out_tokens is None):
        capacity = num_out_tokens // num_experts
        assert not routing_map.requires_grad
        # mask [num_tokens, num_experts] -> [num_experts, num_tokens]
        routing_map = routing_map.to(dtype=torch.int8).T.contiguous()
        # use argsort to put indices of all non-zeros in the beginning of list
        # and keep the first `capacity` number of indices
        sorted_indices = routing_map.argsort(dim=-1, descending=True, stable=True)[
            :, :capacity
        ].contiguous()
        # flatten from [num_experts, capacity] to 1D
        sorted_indices = sorted_indices.view(-1)

        if probs is not None:
            # [num_tokens, num_experts] -> num_experts * num_tokens
            probs_T_1D = probs.T.contiguous().view(-1)
            # get 1D indices of the probs selected by routing_map
            indices_dim0 = torch.arange(
                num_experts, device=routing_map.device).unsqueeze(-1)
            indices_dim1 = sorted_indices.view(num_experts, capacity)
            indices_1D = (indices_dim0 * num_tokens + indices_dim1).view(-1)
            # get probs from indices
            permuted_probs = probs_T_1D.index_select(0, indices_1D)
    else:
        assert (
            num_out_tokens is not None
        ), "num_out_tokens is required for the argsort-based permute"

        # mask [num_tokens, num_experts] -> [num_experts, num_tokens]
        routing_map = routing_map.bool().T.contiguous()

        # Use argsort to get indices of non-zero entries in row-major order.
        # This is equivalent to masked_select but produces fixed-shape output,
        # making it compatible with CUDA graph capture.
        flat_sorted = routing_map.reshape(
            -1).argsort(descending=True, stable=True)
        flat_sorted = flat_sorted[:num_out_tokens]
        sorted_indices = flat_sorted % num_tokens

        if probs is not None:
            permuted_probs = probs.T.contiguous().reshape(-1)[flat_sorted]

    # use the mapping to permute the tokens
    permuted_input = tokens.index_select(0, sorted_indices)

    return permuted_input, permuted_probs, sorted_indices, None, tokens_per_expert


def indices_to_map(
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    num_of_tokens: int,
    num_of_experts: int,
):
    """
    Map the map to the indices.
    """
    # Generate the routing map and the probs according to the topk_idx and topk_weights.
    assert topk_idx is not None
    routing_map = torch.zeros(
        num_of_tokens, num_of_experts, device="cuda", dtype=torch.bool
    )
    routing_map = routing_map.scatter(1, topk_idx.to(torch.int64), 1).bool()
    if topk_weights is not None:
        probs = torch.zeros(
            num_of_tokens, num_of_experts, device="cuda", dtype=torch.float32
        )
        probs = probs.scatter(1, topk_idx.to(torch.int64), topk_weights)
    else:
        probs = None
    return routing_map, probs