"""
Signal-based Pipeline EP Overlap using Triton + Symmetric Memory.

Tokens are grouped by expert groups and dispatched group-by-group.
Each group's completion is signaled via the symmetric memory signal pad,
allowing compute (GroupedGEMM) to overlap with dispatch of subsequent groups.

Pipeline timeline (per receiving rank):
  comm_stream:    dispatch_lg0 → sig_lg0 | dispatch_lg1 → sig_lg1 | ...
  compute_stream: ----wait_lg0 → recv_lg0 → GEMM_lg0 | wait_lg1 → recv_lg1 → GEMM_lg1

Expert grouping:
  expert_id → rank_id         = expert_id // experts_per_rank
  expert_id → global_group_id = expert_id // experts_per_group
  expert_id → local_group_id  = (expert_id % experts_per_rank) // experts_per_group

Buffer layout (per receiver rank):
  symm_mem buffer: [src_0 tokens | src_1 tokens | ... | src_{R-1} tokens]
  signal pad:      [uint32 counter per local_group]  (atomically incremented)
"""

import torch
import torch.distributed as dist
import triton
import triton.language as tl
from typing import List, Optional
from dataclasses import dataclass, field

from primus_turbo.pytorch.cco.symm_mem import get_symm_mem_workspace, SymmetricMemory


# ═══════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════

@dataclass
class PipelineEPConfig:
    num_experts: int
    num_groups: int
    expert_alignment: int = 1

    @property
    def experts_per_group(self) -> int:
        return self.num_experts // self.num_groups


# ═══════════════════════════════════════════════════════════════
#  Handle (precomputed metadata)
# ═══════════════════════════════════════════════════════════════

@dataclass
class PipelineEPHandle:
    config: PipelineEPConfig
    rank: int
    num_ranks: int
    num_tokens: int
    hidden: int
    num_topk: int
    groups_per_rank: int
    experts_per_rank: int

    send_token_ids: list = field(default_factory=list)
    send_buffer_rows: list = field(default_factory=list)

    recv_buffer_rows: list = field(default_factory=list)
    recv_output_rows: list = field(default_factory=list)

    recv_expert_counts: torch.Tensor = None
    recv_expert_offsets: torch.Tensor = None
    total_permuted_tokens: int = 0
    num_recv_tokens: int = 0
    max_recv_tokens: int = 0

    num_tokens_per_expert_group: torch.Tensor = None

    symm_mem: SymmetricMemory = None
    buffer_offset_bytes: int = 0

    recv_topk_idx: torch.Tensor = None
    recv_topk_weights: torch.Tensor = None

    # For unpermute_combine: maps permuted positions back to buffer positions
    # permute_to_buffer_map[permuted_idx] = buffer_row
    permute_to_buffer_map: torch.Tensor = None
    # permute_weights[permuted_idx] = routing weight for this expert assignment
    permute_weights: torch.Tensor = None


# ═══════════════════════════════════════════════════════════════
#  Triton Kernels
# ═══════════════════════════════════════════════════════════════

@triton.jit
def _indexed_copy_2d_kernel(
    src_ptr, dst_ptr,
    src_idx_ptr, dst_idx_ptr,
    count, hidden,
    src_stride_0, dst_stride_0,
    BLOCK_M: tl.constexpr, BLOCK_H: tl.constexpr,
):
    """dst[dst_idx[i], :] = src[src_idx[i], :] for i in [0, count)."""
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)

    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    h_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    m_mask = m_offs < count
    h_mask = h_offs < hidden

    src_row = tl.load(src_idx_ptr + m_offs, mask=m_mask, other=0).to(tl.int64)
    dst_row = tl.load(dst_idx_ptr + m_offs, mask=m_mask, other=0).to(tl.int64)

    mask_2d = m_mask[:, None] & h_mask[None, :]
    data = tl.load(
        src_ptr + src_row[:, None] * src_stride_0 + h_offs[None, :],
        mask=mask_2d, other=0.0,
    )
    tl.store(
        dst_ptr + dst_row[:, None] * dst_stride_0 + h_offs[None, :],
        data, mask=mask_2d,
    )


@triton.jit
def _atomic_inc_signal_kernel(signal_ptr, slot_id):
    """Atomically increment signal_ptr[slot_id] by 1."""
    tl.atomic_add(signal_ptr + slot_id, 1)


@triton.jit
def _spin_wait_signal_kernel(signal_ptr, slot_id, expected):
    """Spin until signal_ptr[slot_id] >= expected (uncached memory)."""
    val = tl.load(signal_ptr + slot_id).to(tl.int32)
    while val < expected:
        val = tl.load(signal_ptr + slot_id).to(tl.int32)


@triton.jit
def _memset_signal_kernel(signal_ptr, num_slots, BLOCK: tl.constexpr):
    """Zero out signal pad slots (BLOCK >= num_slots, power of 2)."""
    offs = tl.arange(0, BLOCK)
    mask = offs < num_slots
    tl.store(signal_ptr + offs, tl.zeros([BLOCK], dtype=tl.uint32), mask=mask)


# ═══════════════════════════════════════════════════════════════
#  Internal helpers
# ═══════════════════════════════════════════════════════════════

def _compute_mappings(all_topk_idx, num_ranks, num_groups, experts_per_rank, experts_per_group):
    """Compute token→rank and token→group boolean masks (vectorised)."""
    total_tokens, num_topk = all_topk_idx.shape
    device = all_topk_idx.device

    is_token_in_rank = torch.zeros(total_tokens, num_ranks, dtype=torch.bool, device=device)
    token_group_mask = torch.zeros(total_tokens, num_groups, dtype=torch.bool, device=device)

    for k in range(num_topk):
        ek = all_topk_idx[:, k]
        valid = ek >= 0
        rk = torch.clamp(ek // experts_per_rank, 0, num_ranks - 1)
        gk = torch.clamp(ek // experts_per_group, 0, num_groups - 1)
        is_token_in_rank[torch.arange(total_tokens, device=device)[valid], rk[valid]] = True
        token_group_mask[torch.arange(total_tokens, device=device)[valid], gk[valid]] = True

    return is_token_in_rank, token_group_mask


def _build_send_schedule(
    rank, is_token_in_rank, token_group_mask,
    num_tokens, num_ranks, groups_per_rank,
    send_base_offsets,
):
    """Build per-(local_group, target_rank) send lists for THIS rank as sender.

    Returns:
        send_token_ids:   list[G_per_R * R] of int64 tensors (local token IDs)
        send_buffer_rows: list[G_per_R * R] of int32 tensors (absolute buffer rows)
    """
    device = is_token_in_rank.device
    R = num_ranks
    G_per_R = groups_per_rank
    G = token_group_mask.shape[1]
    src_start = rank * num_tokens

    is_sent = torch.zeros(num_tokens, R, dtype=torch.bool, device=device)
    next_row = torch.zeros(R, dtype=torch.int32, device=device)
    send_token_ids, send_buffer_rows = [], []

    for lg in range(G_per_R):
        for r in range(R):
            g = r * G_per_R + lg
            if g >= G:
                send_token_ids.append(torch.tensor([], dtype=torch.int64, device=device))
                send_buffer_rows.append(torch.tensor([], dtype=torch.int32, device=device))
                continue

            eligible = (
                token_group_mask[src_start : src_start + num_tokens, g]
                & is_token_in_rank[src_start : src_start + num_tokens, r]
                & (~is_sent[:, r])
            )
            tids = torch.where(eligible)[0]

            if tids.numel() > 0:
                base = send_base_offsets[r].item() + next_row[r].item()
                buf_rows = torch.arange(
                    base, base + tids.numel(), dtype=torch.int32, device=device
                )
                is_sent[tids, r] = True
                next_row[r] += tids.numel()
            else:
                tids = torch.tensor([], dtype=torch.int64, device=device)
                buf_rows = torch.tensor([], dtype=torch.int32, device=device)

            send_token_ids.append(tids)
            send_buffer_rows.append(buf_rows)

    return send_token_ids, send_buffer_rows


def _simulate_sends_to_rank(
    target_rank, is_token_in_rank, token_group_mask,
    num_tokens_per_rank, num_ranks, groups_per_rank,
):
    """Simulate all senders' sends to *target_rank*.

    Returns global token IDs in buffer arrival order.
    """
    device = is_token_in_rank.device
    N = num_tokens_per_rank
    G_per_R = groups_per_rank
    G = token_group_mask.shape[1]
    result = []

    for src in range(num_ranks):
        src_start = src * N
        is_sent = torch.zeros(N, dtype=torch.bool, device=device)
        for lg in range(G_per_R):
            g = target_rank * G_per_R + lg
            if g >= G:
                continue
            eligible = (
                token_group_mask[src_start : src_start + N, g]
                & is_token_in_rank[src_start : src_start + N, target_rank]
                & (~is_sent)
            )
            tids = torch.where(eligible)[0]
            if tids.numel() > 0:
                is_sent[tids] = True
                result.append(src_start + tids)

    return torch.cat(result) if result else torch.tensor([], dtype=torch.int64, device=device)


def _build_recv_schedule(
    all_topk_idx, all_topk_weights, buffer_global_tids,
    rank, experts_per_rank, experts_per_group, groups_per_rank, num_topk,
    recv_expert_offsets, total_permuted_tokens,
):
    """Build per-local_group recv copy lists and unpermute mapping.

    Returns:
        recv_buffer_rows: list[G_per_R] of int32 tensors
        recv_output_rows: list[G_per_R] of int32 tensors
        permute_to_buffer_map: [total_permuted_tokens] int32, permuted_idx -> buffer_row
        permute_weights: [total_permuted_tokens] float32, permuted_idx -> routing weight
    """
    device = all_topk_idx.device
    E_per_R = experts_per_rank
    E_per_G = experts_per_group
    G_per_R = groups_per_rank
    local_base = rank * E_per_R

    next_pos = torch.zeros(E_per_R, dtype=torch.int32, device=device)
    buf_lists = [[] for _ in range(G_per_R)]
    out_lists = [[] for _ in range(G_per_R)]

    # For unpermute: track permuted_idx -> (buffer_row, weight)
    permute_to_buffer_map = torch.full((total_permuted_tokens,), -1, dtype=torch.int32, device=device)
    permute_weights = torch.zeros(total_permuted_tokens, dtype=torch.float32, device=device)

    for b in range(buffer_global_tids.shape[0]):
        gtid = buffer_global_tids[b].item()
        topk = all_topk_idx[gtid]
        weights = all_topk_weights[gtid]
        for k in range(num_topk):
            e = topk[k].item()
            if e < 0:
                continue
            le = e - local_base
            if 0 <= le < E_per_R:
                lg = le // E_per_G
                epos = next_pos[le].item()
                out_row = recv_expert_offsets[le].item() + epos
                next_pos[le] += 1
                buf_lists[lg].append(b)
                out_lists[lg].append(out_row)

                # Record mapping for unpermute
                if out_row < total_permuted_tokens:
                    permute_to_buffer_map[out_row] = b
                    permute_weights[out_row] = weights[k].item()

    to_t = lambda lst: torch.tensor(lst, dtype=torch.int32, device=device) if lst else torch.tensor([], dtype=torch.int32, device=device)
    return (
        [to_t(bl) for bl in buf_lists],
        [to_t(ol) for ol in out_lists],
        permute_to_buffer_map,
        permute_weights,
    )


def _launch_indexed_copy(src, dst, src_idx, dst_idx, hidden, *, BLOCK_M=4, BLOCK_H=256):
    """Launch _indexed_copy_2d_kernel with proper grid."""
    count = src_idx.shape[0]
    if count == 0:
        return
    grid = (triton.cdiv(count, BLOCK_M), triton.cdiv(hidden, BLOCK_H))
    _indexed_copy_2d_kernel[grid](
        src, dst, src_idx, dst_idx,
        count, hidden, src.stride(0), dst.stride(0),
        BLOCK_M=BLOCK_M, BLOCK_H=BLOCK_H,
    )


# ═══════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════

def pipeline_ep_preprocess(
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    x_shape: tuple,
    group: dist.ProcessGroup,
    config: PipelineEPConfig,
) -> PipelineEPHandle:
    """Compute all metadata for pipeline EP dispatch.

    Args:
        topk_idx:      [N, K] int64  per-token expert selections
        topk_weights:  [N, K] float32
        x_shape:       (N, H)  shape of token tensor
        group:         process group
        config:        PipelineEPConfig
    """
    rank = group.rank()
    num_ranks = group.size()
    num_tokens, num_topk = topk_idx.shape
    hidden = x_shape[1]
    device = topk_idx.device

    E = config.num_experts
    G = config.num_groups
    E_per_R = E // num_ranks
    E_per_G = config.experts_per_group
    G_per_R = E_per_R // E_per_G

    # ── 1. All-gather topk_idx & weights ──────────────────────
    all_topk_idx = torch.empty(num_tokens * num_ranks, num_topk, dtype=topk_idx.dtype, device=device)
    all_topk_weights = torch.empty(num_tokens * num_ranks, num_topk, dtype=topk_weights.dtype, device=device)
    dist.all_gather_into_tensor(all_topk_idx, topk_idx.contiguous(), group=group)
    dist.all_gather_into_tensor(all_topk_weights, topk_weights.contiguous(), group=group)

    total_tokens = num_tokens * num_ranks

    # ── 2. Token → rank / group masks ─────────────────────────
    is_token_in_rank, token_group_mask = _compute_mappings(
        all_topk_idx, num_ranks, G, E_per_R, E_per_G
    )

    # ── 3. Per-expert token counts ────────────────────────────
    num_tokens_per_expert = torch.zeros(E, dtype=torch.int32, device=device)
    for k in range(num_topk):
        ek = all_topk_idx[:, k]
        valid = ek >= 0
        if valid.any():
            num_tokens_per_expert.scatter_add_(
                0, ek[valid].to(torch.int64),
                torch.ones(valid.sum(), dtype=torch.int32, device=device),
            )

    num_tokens_per_expert_group = num_tokens_per_expert.view(G, E_per_G)

    # ── 4. Recv expert layout for this rank ───────────────────
    local_counts = num_tokens_per_expert[rank * E_per_R : (rank + 1) * E_per_R].clone()
    align = config.expert_alignment
    if align > 1:
        aligned = ((local_counts + align - 1) // align * align).to(torch.int32)
    else:
        aligned = local_counts.clone()

    recv_expert_offsets = torch.zeros(E_per_R, dtype=torch.int32, device=device)
    if E_per_R > 1:
        recv_expert_offsets[1:] = aligned[:-1].cumsum(0).to(torch.int32)
    total_permuted_tokens = aligned.sum().item()

    # ── 5. Send counts exchange ───────────────────────────────
    my_send_counts = is_token_in_rank[rank * num_tokens : (rank + 1) * num_tokens].sum(dim=0).to(torch.int32)

    all_send_counts = torch.empty(num_ranks, num_ranks, dtype=torch.int32, device=device)
    dist.all_gather_into_tensor(
        all_send_counts, my_send_counts.unsqueeze(0).contiguous(), group=group
    )

    send_offsets = all_send_counts.cumsum(dim=0) - all_send_counts
    my_send_offsets = send_offsets[rank]

    recv_from_rank = all_send_counts[:, rank]
    recv_rank_offsets = torch.zeros(num_ranks, dtype=torch.int32, device=device)
    if num_ranks > 1:
        recv_rank_offsets[1:] = recv_from_rank[:-1].cumsum(0).to(torch.int32)
    num_recv_tokens = recv_from_rank.sum().item()

    # ── 6. Allocate symmetric memory (all ranks agree on max size) ──
    max_recv_tensor = torch.tensor([num_recv_tokens], dtype=torch.int64, device=device)
    dist.all_reduce(max_recv_tensor, op=dist.ReduceOp.MAX, group=group)
    max_recv_tokens = max_recv_tensor.item()

    buf_bytes = max_recv_tokens * hidden * torch.bfloat16.itemsize
    total_bytes = buf_bytes + 4096
    symm_mem = get_symm_mem_workspace(group, max(total_bytes, 1024))

    # ── 7. Build send schedule ────────────────────────────────
    send_token_ids, send_buffer_rows = _build_send_schedule(
        rank, is_token_in_rank, token_group_mask,
        num_tokens, num_ranks, G_per_R, my_send_offsets,
    )

    # ── 8. Simulate recv & build recv schedule ────────────────
    buffer_global_tids = _simulate_sends_to_rank(
        rank, is_token_in_rank, token_group_mask,
        num_tokens, num_ranks, G_per_R,
    )

    recv_buffer_rows, recv_output_rows, permute_to_buffer_map, permute_weights = _build_recv_schedule(
        all_topk_idx, all_topk_weights, buffer_global_tids,
        rank, E_per_R, E_per_G, G_per_R, num_topk,
        recv_expert_offsets, total_permuted_tokens,
    )

    # ── 9. Build recv topk_idx / weights (local expert IDs) ──
    recv_topk_idx = torch.full((num_recv_tokens, num_topk), -1, dtype=torch.int64, device=device)
    recv_topk_weights = torch.zeros(num_recv_tokens, num_topk, dtype=torch.float32, device=device)
    local_base = rank * E_per_R
    for b in range(buffer_global_tids.shape[0]):
        gtid = buffer_global_tids[b].item()
        for k in range(num_topk):
            e = all_topk_idx[gtid, k].item()
            if e < 0:
                continue
            le = e - local_base
            if 0 <= le < E_per_R:
                recv_topk_idx[b, k] = le
                recv_topk_weights[b, k] = all_topk_weights[gtid, k].item()

    return PipelineEPHandle(
        config=config,
        rank=rank,
        num_ranks=num_ranks,
        num_tokens=num_tokens,
        hidden=hidden,
        num_topk=num_topk,
        groups_per_rank=G_per_R,
        experts_per_rank=E_per_R,
        send_token_ids=send_token_ids,
        send_buffer_rows=send_buffer_rows,
        recv_buffer_rows=recv_buffer_rows,
        recv_output_rows=recv_output_rows,
        recv_expert_counts=aligned,
        recv_expert_offsets=recv_expert_offsets,
        total_permuted_tokens=total_permuted_tokens,
        num_recv_tokens=num_recv_tokens,
        max_recv_tokens=max_recv_tokens,
        num_tokens_per_expert_group=num_tokens_per_expert_group,
        symm_mem=symm_mem,
        buffer_offset_bytes=0,
        recv_topk_idx=recv_topk_idx,
        recv_topk_weights=recv_topk_weights,
    )


def pipeline_ep_dispatch(
    x: torch.Tensor,
    handle: PipelineEPHandle,
    comm_stream: Optional[torch.cuda.Stream] = None,
) -> List[torch.cuda.Event]:
    """Dispatch tokens group-by-group via symmetric memory with signals.

    Returns a list of CUDA events, one per local_group. The event fires after
    the corresponding group's data has been copied to ALL target ranks' buffers
    and the signal has been written.
    """
    R = handle.num_ranks
    G_per_R = handle.groups_per_rank
    H = handle.hidden
    symm_mem = handle.symm_mem
    dtype = x.dtype
    use_comm_stream = comm_stream is not None

    events = []
    max_buf_rows = handle.max_recv_tokens

    for lg in range(G_per_R):
        if use_comm_stream:
            ctx = torch.cuda.stream(comm_stream)
            ctx.__enter__()

        for r in range(R):
            idx = lg * R + r
            tids = handle.send_token_ids[idx]
            buf_rows = handle.send_buffer_rows[idx]
            if tids.numel() == 0:
                continue

            remote_buf = symm_mem.get_buffer(r, [max_buf_rows, H], dtype)
            _launch_indexed_copy(x, remote_buf, tids, buf_rows, H)

        for r in range(R):
            remote_signal = symm_mem.get_signal_pad(r, [G_per_R], torch.uint32)
            _atomic_inc_signal_kernel[(1,)](remote_signal, lg)

        ev = torch.cuda.Event()
        ev.record(comm_stream if use_comm_stream else torch.cuda.current_stream())
        events.append(ev)

        if use_comm_stream:
            ctx.__exit__(None, None, None)

    return events


def pipeline_ep_recv(
    handle: PipelineEPHandle,
    expert_output: Optional[torch.Tensor] = None,
    wait_for_signal: bool = True,
) -> torch.Tensor:
    """Receive tokens group-by-group into expert output.

    If *wait_for_signal* is True, each local_group stage uses a GPU-side spin
    to wait for the signal counter before reading data.

    Returns:
        expert_output: [total_permuted_tokens, hidden] tensor.
    """
    R = handle.num_ranks
    G_per_R = handle.groups_per_rank
    H = handle.hidden
    symm_mem = handle.symm_mem
    dtype = torch.bfloat16

    if expert_output is None:
        expert_output = torch.zeros(handle.total_permuted_tokens, H, dtype=dtype, device="cuda")

    my_buffer = symm_mem.get_buffer(handle.rank, [handle.max_recv_tokens, H], dtype)
    my_signal = symm_mem.get_signal_pad(handle.rank, [G_per_R], torch.uint32)

    for lg in range(G_per_R):
        if wait_for_signal:
            _spin_wait_signal_kernel[(1,)](my_signal, lg, R)

        buf_rows = handle.recv_buffer_rows[lg]
        out_rows = handle.recv_output_rows[lg]
        if buf_rows.numel() > 0:
            _launch_indexed_copy(my_buffer, expert_output, buf_rows, out_rows, H)

    return expert_output


def pipeline_ep_full(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    group: dist.ProcessGroup,
    config: PipelineEPConfig,
) -> tuple:
    """End-to-end pipeline EP dispatch + receive with compute overlap.

    Returns:
        (expert_output, handle)
        expert_output: [total_permuted_tokens, hidden] ready for GroupedGEMM
        handle: PipelineEPHandle for later combine
    """
    handle = pipeline_ep_preprocess(topk_idx, topk_weights, x.shape, group, config)

    comm_stream = torch.cuda.Stream()
    compute_stream = torch.cuda.current_stream()

    _reset_signals(handle)
    dispatch_events = pipeline_ep_dispatch(x, handle, comm_stream=comm_stream)

    expert_output = torch.zeros(
        handle.total_permuted_tokens, handle.hidden,
        dtype=x.dtype, device=x.device,
    )

    my_buffer = handle.symm_mem.get_buffer(
        handle.rank, [handle.max_recv_tokens, handle.hidden], x.dtype
    )
    my_signal = handle.symm_mem.get_signal_pad(
        handle.rank, [handle.groups_per_rank], torch.uint32
    )

    recv_events = []
    for lg in range(handle.groups_per_rank):
        compute_stream.wait_event(dispatch_events[lg])
        _spin_wait_signal_kernel[(1,)](my_signal, lg, handle.num_ranks)

        buf_rows = handle.recv_buffer_rows[lg]
        out_rows = handle.recv_output_rows[lg]
        if buf_rows.numel() > 0:
            _launch_indexed_copy(my_buffer, expert_output, buf_rows, out_rows, handle.hidden)

        ev = torch.cuda.Event()
        ev.record(compute_stream)
        recv_events.append(ev)

    return expert_output, handle, recv_events


# ═══════════════════════════════════════════════════════════════
#  Utilities
# ═══════════════════════════════════════════════════════════════

def _reset_signals(handle: PipelineEPHandle):
    """Zero the signal pad on ALL ranks (local write only)."""
    G_per_R = handle.groups_per_rank
    pad_size = max(G_per_R, 1)
    BLOCK = triton.next_power_of_2(pad_size)
    my_signal = handle.symm_mem.get_signal_pad(
        handle.rank, [BLOCK], torch.uint32
    )
    _memset_signal_kernel[(1,)](my_signal, pad_size, BLOCK=BLOCK)
    handle.symm_mem.group.barrier()


def get_grouped_gemm_schedule(handle: PipelineEPHandle):
    """Return per-local-group (expert_counts, expert_offsets) for GroupedGEMM.

    Each entry covers the experts in one local group, so GroupedGEMM can be
    launched as soon as the corresponding recv_event fires.
    """
    E_per_G = handle.config.experts_per_group
    schedules = []
    for lg in range(handle.groups_per_rank):
        start = lg * E_per_G
        end = start + E_per_G
        counts = handle.recv_expert_counts[start:end]
        offsets = handle.recv_expert_offsets[start:end]
        schedules.append((counts, offsets))
    return schedules
