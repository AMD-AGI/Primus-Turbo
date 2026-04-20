"""
This is the same test_intranode.py test in DeepEP's repo.
OMP_NUM_THREADS=8 torchrun --standalone --nproc_per_node=8 bench/test_intranode.py \
    --num-tokens 4096 --hidden 7168 --num-topk 8 --num-experts 256
"""

import argparse
import ctypes
import ctypes.util
import math
import time

import torch
import torch.distributed as dist
from utils import (
    bench,
    bench_kineto,
    get_torch_prof_ctx,
    init_dist,
    inplace_unique,
    per_token_cast_to_fp8,
)

from primus_turbo.pytorch.cco import _fused_dispatch_permute
from primus_turbo.triton.grouped_gemm.grouped_gemm_kernel import (
    _get_num_cus,
    grouped_gemm_triton_kernel,
)

_HIP_RUNTIME = None
_NUM_XCDS = 8
_CUS_PER_XCD = 38


def _load_hip_runtime():
    global _HIP_RUNTIME
    if _HIP_RUNTIME is not None:
        return _HIP_RUNTIME

    candidates = [
        ctypes.util.find_library("amdhip64"),
        "libamdhip64.so",
        "libamdhip64.so.6",
        "libamdhip64.so.5",
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            hip_runtime = ctypes.CDLL(candidate)
            break
        except OSError:
            continue
    else:
        raise RuntimeError("Failed to load libamdhip64 for HIP masked streams")

    hip_runtime.hipExtStreamCreateWithCUMask.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint32),
    ]
    hip_runtime.hipExtStreamCreateWithCUMask.restype = ctypes.c_int
    hip_runtime.hipStreamDestroy.argtypes = [ctypes.c_void_p]
    hip_runtime.hipStreamDestroy.restype = ctypes.c_int

    _HIP_RUNTIME = hip_runtime
    return _HIP_RUNTIME


def _check_hip_error(code: int, op_name: str) -> None:
    if code != 0:
        raise RuntimeError(f"{op_name} failed with HIP error code {code}")


def _build_cu_mask_words(total_sms: int, cu_indices: list[int]) -> list[int]:
    assert total_sms > 0, f"total_sms must be positive, got {total_sms}"
    assert cu_indices, "cu_indices must not be empty"

    mask_words = [0] * math.ceil(total_sms / 32)
    for cu_idx in cu_indices:
        assert 0 <= cu_idx < total_sms, f"Invalid cu_idx={cu_idx} for total_sms={total_sms}"
        mask_words[cu_idx // 32] |= 1 << (cu_idx % 32)
    return mask_words


class _HipMaskedStream:
    def __init__(self, hip_runtime, stream_ptr: int, device: int):
        self._hip_runtime = hip_runtime
        self._stream_ptr = stream_ptr
        self.stream = torch.cuda.ExternalStream(stream_ptr, device=device)
        self._destroyed = False

    def destroy(self) -> None:
        if self._destroyed:
            return
        _check_hip_error(
            self._hip_runtime.hipStreamDestroy(ctypes.c_void_p(self._stream_ptr)),
            "hipStreamDestroy",
        )
        self._destroyed = True

    def __del__(self):
        try:
            self.destroy()
        except Exception:
            pass


def _build_round_robin_xcd_cu_indices(
    total_sms: int,
    cu_count: int,
    num_xcds: int = _NUM_XCDS,
    cu_stride_per_xcd: int = _CUS_PER_XCD,
) -> list[int]:
    assert total_sms > 0, f"total_sms must be positive, got {total_sms}"
    assert 0 < cu_count <= total_sms, f"cu_count must be in (0, {total_sms}], got {cu_count}"
    assert num_xcds > 0, f"num_xcds must be positive, got {num_xcds}"
    assert cu_stride_per_xcd > 0, f"cu_stride_per_xcd must be positive, got {cu_stride_per_xcd}"

    cu_indices: list[int] = []
    local_cu_idx = 0
    while len(cu_indices) < cu_count:
        made_progress = False
        for xcd in range(num_xcds):
            if len(cu_indices) >= cu_count:
                break
            cu_idx = xcd * cu_stride_per_xcd + local_cu_idx
            if cu_idx >= total_sms:
                continue
            cu_indices.append(cu_idx)
            made_progress = True
        local_cu_idx += 1
        assert made_progress, "Failed to build a round-robin XCD CU mask"
    return cu_indices


def _build_xcd_cu_groups(
    total_sms: int,
    num_xcds: int = _NUM_XCDS,
    cu_stride_per_xcd: int = _CUS_PER_XCD,
) -> list[list[int]]:
    assert total_sms > 0, f"total_sms must be positive, got {total_sms}"
    assert num_xcds > 0, f"num_xcds must be positive, got {num_xcds}"
    assert cu_stride_per_xcd > 0, f"cu_stride_per_xcd must be positive, got {cu_stride_per_xcd}"

    xcd_cu_groups: list[list[int]] = []
    for xcd in range(num_xcds):
        xcd_cu_group = []
        for local_cu_idx in range(cu_stride_per_xcd):
            cu_idx = xcd * cu_stride_per_xcd + local_cu_idx
            if cu_idx >= total_sms:
                continue
            xcd_cu_group.append(cu_idx)
        if xcd_cu_group:
            xcd_cu_groups.append(xcd_cu_group)

    assert len(xcd_cu_groups) >= 2, "Need at least two XCD groups to isolate comm and compute streams"
    return xcd_cu_groups


def _select_comm_xcd_group_ids(xcd_cu_groups: list[list[int]], requested_comm_sms: int) -> list[int]:
    assert xcd_cu_groups, "xcd_cu_groups must not be empty"
    assert 0 < requested_comm_sms, f"requested_comm_sms must be positive, got {requested_comm_sms}"

    prefix_totals = []
    running_total = 0
    # Keep at least one whole XCD group for the compute stream.
    for xcd_cu_group in xcd_cu_groups[:-1]:
        running_total += len(xcd_cu_group)
        prefix_totals.append(running_total)

    assert prefix_totals, "Need at least one candidate comm XCD group"
    chosen_num_groups = min(
        range(1, len(prefix_totals) + 1),
        key=lambda num_groups: (
            abs(prefix_totals[num_groups - 1] - requested_comm_sms),
            prefix_totals[num_groups - 1] < requested_comm_sms,
            num_groups,
        ),
    )
    return list(range(chosen_num_groups))


def _invert_cu_indices(total_sms: int, selected_cu_indices: list[int]) -> list[int]:
    selected = set(selected_cu_indices)
    return [cu_idx for cu_idx in range(total_sms) if cu_idx not in selected]


def _create_masked_stream(total_sms: int, cu_indices: list[int], device: int) -> _HipMaskedStream:
    hip_runtime = _load_hip_runtime()
    mask_words = _build_cu_mask_words(total_sms, cu_indices)
    mask_array = (ctypes.c_uint32 * len(mask_words))(*mask_words)
    stream_ptr = ctypes.c_void_p()
    _check_hip_error(
        hip_runtime.hipExtStreamCreateWithCUMask(
            ctypes.byref(stream_ptr),
            len(mask_words),
            mask_array,
        ),
        "hipExtStreamCreateWithCUMask",
    )
    assert stream_ptr.value is not None
    return _HipMaskedStream(hip_runtime, stream_ptr.value, device)


def _create_partitioned_masked_streams(num_comm_sms: int, total_sms: int, device: int):
    assert 0 < num_comm_sms < total_sms, f"num_comm_sms must be in (0, {total_sms}), got {num_comm_sms}"
    # Isolate comm/compute at whole-XCD granularity instead of splitting CUs within the same XCD.
    xcd_cu_groups = _build_xcd_cu_groups(total_sms)
    comm_xcd_group_ids = set(_select_comm_xcd_group_ids(xcd_cu_groups, num_comm_sms))
    comm_cu_indices = [
        cu_idx
        for xcd_group_id, xcd_cu_group in enumerate(xcd_cu_groups)
        if xcd_group_id in comm_xcd_group_ids
        for cu_idx in xcd_cu_group
    ]
    compute_cu_indices = [
        cu_idx
        for xcd_group_id, xcd_cu_group in enumerate(xcd_cu_groups)
        if xcd_group_id not in comm_xcd_group_ids
        for cu_idx in xcd_cu_group
    ]
    comm_stream = _create_masked_stream(total_sms, comm_cu_indices, device)
    compute_stream = _create_masked_stream(total_sms, compute_cu_indices, device)
    return comm_stream, compute_stream, len(comm_cu_indices), len(compute_cu_indices)


def _build_group_offs(expert_counts: torch.Tensor) -> torch.Tensor:
    group_offs = torch.empty((expert_counts.numel() + 1,), dtype=torch.int64, device=expert_counts.device)
    group_offs[0] = 0
    group_offs[1:] = torch.cumsum(expert_counts.to(torch.int64), dim=0)
    return group_offs


def benchmark_fused_dispatch_grouped_gemm_overlap(
    x: torch.Tensor,
    group: dist.ProcessGroup,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    groupedgemm_weight: torch.Tensor,
    num_experts: int,
    num_experts_per_rank: int,
    rank: int,
    num_recv_experts_count: torch.Tensor,
    num_comm_sms: int,
):
    total_sms = _get_num_cus()
    if num_comm_sms <= 0:
        return
    assert (
        num_comm_sms < total_sms
    ), f"overlap_comm_cus must be smaller than total_sms={total_sms}, got {num_comm_sms}"

    group_offs = _build_group_offs(num_recv_experts_count)

    device = torch.cuda.current_device()
    comm_stream_owner, compute_stream_owner, actual_comm_sms, actual_comp_sms = (
        _create_partitioned_masked_streams(num_comm_sms, total_sms, device)
    )
    comm_stream = comm_stream_owner.stream
    compute_stream = compute_stream_owner.stream

    try:

        def run_overlap():
            current_stream = torch.cuda.current_stream()

            # 1. Shared signal buffer on current stream (all zeros)
            group_tail_idx = torch.zeros((num_experts_per_rank,), dtype=torch.int32, device="cuda")

            # 2. comm_stream: dispatch writes group_tail_idx on the isolated comm XCD mask
            comm_stream.wait_stream(current_stream)
            with torch.cuda.stream(comm_stream):
                (ref_permuted_x, _, _, _, _, moe_recv_expert_counter, _, _, _, _), _ = (
                    _fused_dispatch_permute(
                        x, group, group_tail_idx, topk_idx, topk_weights, num_experts, num_sms=actual_comm_sms
                    )
                )

            # 3. compute_stream: GEMM polls group_tail_idx on the isolated compute XCD mask
            # ref_permuted_x.record_stream(compute_stream)
            # compute_stream.wait_stream(current_stream)
            with torch.cuda.stream(compute_stream):
                grouped_gemm_triton_kernel(
                    ref_permuted_x,
                    groupedgemm_weight,
                    group_tail_idx,
                    group_offs,
                    BLOCK_SIZE_M=128,
                    num_sms=actual_comp_sms,
                )

            # 4. Join both streams
            current_stream.wait_stream(comm_stream)
            current_stream.wait_stream(compute_stream)

        def run_separate():

            current_stream = torch.cuda.current_stream()
            comm_stream.wait_stream(current_stream)
            with torch.cuda.stream(comm_stream):
                group_tail_idx = torch.zeros((num_experts_per_rank,), dtype=torch.int32, device="cuda")

                (ref_permuted_x, _, _, _, _, moe_recv_expert_counter, _, _, _, _), _ = (
                    _fused_dispatch_permute(
                        x, group, group_tail_idx, topk_idx, topk_weights, num_experts, num_sms=actual_comm_sms
                    )
                )
            compute_stream.wait_stream(comm_stream)
            with torch.cuda.stream(compute_stream):
                grouped_gemm_triton_kernel(
                    ref_permuted_x,
                    groupedgemm_weight,
                    group_tail_idx,
                    group_offs,
                    BLOCK_SIZE_M=128,
                    num_sms=actual_comp_sms,
                )

            current_stream.wait_stream(compute_stream)

        overlap_ms = bench(run_overlap)[0]
        dist.barrier(group=group)

        separate_ms = bench(run_separate)[0]

        if rank == 0:
            print(
                f"[overlap] total_sms={total_sms}, requested_comm_sms={num_comm_sms}, actual_comm_sms={actual_comm_sms}, actual_comp_sms={actual_comp_sms}, overlap_ms={overlap_ms * 1e6:.2f} us",
                flush=True,
            )
            print(
                f"[separate] total_sms={total_sms}, requested_comm_sms={num_comm_sms}, actual_comm_sms={actual_comm_sms}, actual_comp_sms={actual_comp_sms}, separate_ms={separate_ms * 1e6:.2f} us",
                flush=True,
            )
    finally:
        group.barrier()
        torch.cuda.synchronize()
        comm_stream_owner.destroy()
        compute_stream_owner.destroy()


def compute_recv_and_permuted_x(
    rank_prefix_matrix, dispatch_to_expert_map, hidden_size, num_ranks, current_rank, num_out_tokens
):
    """
    Generate recv_x from rank_prefix_matrix, then compute permuted_x via dispatch_to_expert_map.

    recv_x generation: rank_prefix_matrix[i][current_rank] rows with value i, replicated hidden_size times.
    permuted_x is recv_x reordered according to dispatch_to_expert_map.
    """
    num_recv_tokens = rank_prefix_matrix[num_ranks - 1][current_rank].item()

    recv_x = torch.empty((num_recv_tokens, hidden_size), dtype=torch.bfloat16, device="cuda")
    start = 0
    for i in range(num_ranks):
        end = rank_prefix_matrix[i][current_rank].item()
        recv_x[start:end, :] = i
        start = end

    d2e = dispatch_to_expert_map[:num_recv_tokens]
    valid_mask = d2e >= 0

    permuted_x = torch.empty((num_out_tokens, hidden_size), dtype=torch.bfloat16, device="cuda")
    src_tokens = torch.arange(num_recv_tokens, device="cuda").unsqueeze(1).expand_as(d2e)
    valid_dst = d2e[valid_mask]
    valid_src = src_tokens[valid_mask]

    assert (
        valid_dst.max() < num_out_tokens
    ), f"valid_dst.max()={valid_dst.max()}, num_out_tokens={num_out_tokens}"
    assert valid_src.max() < num_recv_tokens
    permuted_x[valid_dst] = recv_x[valid_src]

    return recv_x, permuted_x


def test_main(
    args: argparse.Namespace,
    num_sms: int,
    local_rank: int,
    num_ranks: int,
    rank: int,
    group: dist.ProcessGroup,
):
    # Settings
    num_tokens, hidden = args.num_tokens, args.hidden
    num_topk, num_experts = args.num_topk, args.num_experts

    assert num_experts % num_ranks == 0

    num_experts_per_rank = num_experts // num_ranks
    if local_rank == 0:
        print(
            f"[config] num_tokens={num_tokens}, hidden={hidden}, num_topk={num_topk}",
            flush=True,
        )

    # Random data
    x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device="cuda") * rank
    x_pure_rand = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
    x_e4m3 = per_token_cast_to_fp8(x)
    x_e4m3 = (x_e4m3[0], x_e4m3[1].T.contiguous().T) if x_e4m3 is not None else None
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device="cuda").abs() + 1
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)[1]
    groupedgemm_weight = torch.randn(
        (num_experts_per_rank, hidden, 4096), dtype=torch.bfloat16, device="cuda"
    )

    group_tail_idx = torch.zeros((num_experts_per_rank), dtype=torch.int32, device="cuda")
    topk_weights = torch.ones((num_tokens, num_topk), dtype=torch.float32, device="cuda") * rank
    topk_weights_pure_rand = torch.randn((num_tokens, num_topk), dtype=torch.float32, device="cuda")
    rank_idx = topk_idx // (num_experts // num_ranks)
    rank_idx.masked_fill_(topk_idx == -1, -1)
    inplace_unique(rank_idx, num_ranks)

    # Expert meta
    num_tokens_per_expert = torch.zeros((num_experts,), dtype=torch.int, device="cuda")
    for i in range(num_experts):
        num_tokens_per_expert[i] = (topk_idx == i).sum()
    gbl_num_tokens_per_expert = num_tokens_per_expert.clone()
    dist.all_reduce(gbl_num_tokens_per_expert, group=group)

    # Rank layout meta
    num_tokens_per_rank = torch.empty((num_ranks,), dtype=torch.int, device="cuda")
    token_idx_in_rank = torch.full((num_ranks, num_tokens), -1, dtype=torch.long, device="cuda")
    for i in range(num_ranks):
        num_tokens_per_rank[i] = (rank_idx == i).sum()
        token_sel = (rank_idx == i).max(dim=-1)[0]
        count = token_sel.sum().item()
        tokens = torch.sort(token_sel.to(torch.int), descending=True)[1]
        tokens[:count] = torch.sort(tokens[:count])[0]
        token_idx_in_rank[i][tokens[:count]] = torch.arange(count, dtype=torch.long, device="cuda")
    token_idx_in_rank = token_idx_in_rank.T.contiguous().to(torch.int)
    is_token_in_rank = token_idx_in_rank >= 0
    gbl_num_tokens_per_rank = num_tokens_per_rank.clone()
    dist.all_reduce(gbl_num_tokens_per_rank, group=group)

    def check_data(check_x, rank_prefix_matrix):
        assert torch.allclose(check_x.amin(dim=1), check_x.amax(dim=1))
        check_start = 0
        for i in range(num_ranks):
            check_end = rank_prefix_matrix[i][rank].item()
            assert (check_x[check_start:check_end, :].int() - i).sum().item() == 0
            check_start = check_end

    (
        ref_permuted_x,
        ref_recv_x_scales,
        ref_recv_topk_idx,
        ref_recv_topk_weights,
        ref_moe_recv_counter,
        ref_moe_recv_expert_counter,
        ref_rank_prefix_matrix,
        ref_channel_prefix_matrix,
        ref_dispatch_to_expert_map,
        ref_send_head,
    ), handle = _fused_dispatch_permute(
        x, group, group_tail_idx, topk_idx, topk_weights, num_experts, num_sms=num_sms
    )
    group.barrier()

    time.sleep(1)

    (
        ref_recv_x,
        ref_num_tokens_per_rank,
        ref_num_tokens_per_expert,
        ref_rank_prefix_matrix,
        ref_channel_prefix_matrix,
        ref_is_token_in_rank,
        _,
        _,
    ) = handle

    # Check group_tail_idx
    assert torch.equal(ref_moe_recv_expert_counter, group_tail_idx)

    num_moe_recv_tokens = ref_moe_recv_counter.item()
    recv_num_tokens_per_expert_list = ref_moe_recv_expert_counter.tolist()

    # split out to num_moe_recv_tokens
    ref_recv_topk_idx = ref_recv_topk_idx[:num_moe_recv_tokens, :]
    ref_recv_topk_weights = ref_recv_topk_weights[:num_moe_recv_tokens, :]

    assert gbl_num_tokens_per_expert.view(num_ranks, -1)[rank].tolist() == recv_num_tokens_per_expert_list

    assert torch.allclose(ref_num_tokens_per_rank, num_tokens_per_rank)
    assert torch.allclose(ref_num_tokens_per_expert, num_tokens_per_expert)
    assert torch.allclose(ref_is_token_in_rank, is_token_in_rank)

    # Check `topk_idx`
    assert (
        ref_recv_topk_idx.eq(-1)
        | ((ref_recv_topk_idx >= 0) & (ref_recv_topk_idx < (num_experts // num_ranks)))
    ).sum().item() == ref_recv_topk_idx.numel()
    for i, count in enumerate(recv_num_tokens_per_expert_list):
        assert ref_recv_topk_idx.eq(i).sum().item() == count

    # Check `topk_weights`
    ref_recv_topk_weights[ref_recv_topk_idx.eq(-1)] = ref_recv_topk_weights.amax(
        dim=1, keepdim=True
    ).expand_as(ref_recv_topk_weights)[ref_recv_topk_idx.eq(-1)]
    check_data(ref_recv_topk_weights, ref_rank_prefix_matrix)

    num_out_tokens = sum(recv_num_tokens_per_expert_list)

    # compute permuted recv_x
    recv_x, permuted_x = compute_recv_and_permuted_x(
        ref_rank_prefix_matrix, ref_dispatch_to_expert_map, hidden, num_ranks, rank, num_out_tokens
    )

    # assert torch.allclose(ref_recv_x[:num_moe_recv_tokens, :], recv_x), f"{ref_recv_x[:num_moe_recv_tokens, 0]} {recv_x[:num_moe_recv_tokens, 0]}"
    assert torch.allclose(
        ref_permuted_x[:num_out_tokens, :], permuted_x
    ), f"{ref_permuted_x[:num_out_tokens, 0]} {permuted_x[:num_out_tokens, 0]}"

    group.barrier()
    if local_rank == 0:
        print("dispatch_permute correctness checks passed", flush=True)

    # local_expert_counts = gbl_num_tokens_per_expert.view(
    #     num_ranks, -1)[rank].to(torch.int32)

    # group_offs = _build_group_offs(ref_moe_recv_expert_counter)
    # group_tail_idx = group_offs[1:].to(torch.int32)
    # grouped_gemm_triton_kernel(
    #     ref_permuted_x, groupedgemm_weight, group_tail_idx, group_offs)
    ctx = get_torch_prof_ctx(do_prof=True)
    with ctx:
        benchmark_fused_dispatch_grouped_gemm_overlap(
            x,
            group,
            topk_idx,
            topk_weights,
            groupedgemm_weight,
            num_experts,
            num_experts_per_rank,
            rank,
            ref_moe_recv_expert_counter,
            num_sms,
        )
    if rank == 0:
        import os

        prof_dir = "prof/"
        os.makedirs(prof_dir, exist_ok=True)
        ctx.export_chrome_trace(f"{prof_dir}/dispatch_permute_grouped_gemm_overlap.json.gz")
    # benchmark performance
    nvl_recv_bytes = num_moe_recv_tokens * hidden * 2

    kwargs = {
        "x": x,
        "group": group,
        "group_tail_idx": group_tail_idx,
        "topk_idx": topk_idx,
        "topk_weights": topk_weights,
        "num_experts": num_experts,
        "num_sms": num_sms,
    }

    for num_max_send_tokens in range(2, 33, 2):
        t, notify_t = bench_kineto(
            lambda: _fused_dispatch_permute(**kwargs, num_max_send_tokens=num_max_send_tokens),
            ("fused_dispatch_permute", "notify_dispatch"),
            suppress_kineto_output=True,
        )

        if local_rank == 0:
            print(
                f"SMs {num_sms}: num_max_send_tokens={num_max_send_tokens}: {nvl_recv_bytes / 1e9 / t:.2f} GB/s (NVL), {t * 1e6:.2f} us + {notify_t * 1e6:.2f} us",
                flush=True,
            )


def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)

    torch.manual_seed(rank)

    for i in (48,):
        test_main(args, i, local_rank, num_ranks, rank, group)
        if local_rank == 0:
            print("", flush=True)

    # Destroy the buffer runtime and communication group
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test intranode EP kernels")
    parser.add_argument(
        "--num-processes", type=int, default=8, help="Number of processes to spawn (default: 8)"
    )
    parser.add_argument("--num-tokens", type=int, default=4096, help="Number of tokens (default: 4096)")
    parser.add_argument("--hidden", type=int, default=7168, help="Hidden dimension size (default: 7168)")
    parser.add_argument("--num-topk", type=int, default=8, help="Number of top-k experts (default: 8)")
    parser.add_argument("--num-experts", type=int, default=256, help="Number of experts (default: 256)")
    args = parser.parse_args()

    num_processes = args.num_processes
    torch.multiprocessing.spawn(test_loop, args=(num_processes, args), nprocs=num_processes)
