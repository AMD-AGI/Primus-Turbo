"""
Metadata preprocessing for Hybrid EP, implemented with Triton + PyTorch.
Single-node only (num_of_nodes=1), supports fuse_permute_dispatch.
Target: ROCm HIP platform.

This module replaces the CUDA scan kernel + permute_preprocessing with a
portable Triton + PyTorch implementation that produces identical outputs.
"""

import torch
import triton
import triton.language as tl
from typing import Optional, NamedTuple


# ---------------------------------------------------------------------------
# Data class mirroring C++ HandleImpl
# ---------------------------------------------------------------------------
class MetadataHandle(NamedTuple):
    sparse_to_dense_map: torch.Tensor          # [N*num_nodes, R] int32
    rdma_to_attn_map: torch.Tensor             # [padded_N, num_nodes] bool
    attn_to_rdma_map: torch.Tensor             # [N, num_nodes-1] bool
    num_dispatched_tokens_tensor: torch.Tensor  # [1] int32
    local_expert_routing_map: torch.Tensor      # [N*R*num_nodes, E] bool
    num_of_tokens_per_rank: int
    num_permuted_tokens: int
    row_id_map: Optional[torch.Tensor]                  # standalone permute
    tokens_per_expert: Optional[torch.Tensor]           # [E] int32
    padded_tokens_per_expert: Optional[torch.Tensor]    # [E] int64
    overflow_flag: Optional[torch.Tensor]               # [1] int32
    dense_chunk_layout: Optional[torch.Tensor]          # fuse path
    dense_to_expert_map: Optional[torch.Tensor]         # fuse path


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------

@triton.jit
def _build_sparse_to_dense_kernel(
    rank_routing_ptr,   # [total_tokens, R] bool
    excumsum_ptr,       # [total_tokens, R] int32
    s2d_ptr,            # output [N, R] int32
    rdma_ptr,           # output [padded_N] bool
    local_start,        # = local_rank * N
    N,
    R: tl.constexpr,
    padded_N,
    BLOCK: tl.constexpr,
):
    """Build sparse_to_dense_map and rdma_to_attn_map for local_rank tokens."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    global_offs = local_start + offs  # global token indices

    any_rank = tl.zeros([BLOCK], dtype=tl.int1)
    for r in tl.static_range(R):
        rr = tl.load(
            rank_routing_ptr + global_offs * R + r,
            mask=mask, other=False,
        ).to(tl.int1)
        ex = tl.load(
            excumsum_ptr + global_offs * R + r,
            mask=mask, other=0,
        )
        val = tl.where(rr, ex, -1)
        tl.store(s2d_ptr + offs * R + r, val, mask=mask)
        any_rank = any_rank | rr

    rdma_mask = mask & (offs < padded_N)
    tl.store(rdma_ptr + offs, any_rank, mask=rdma_mask)


@triton.jit
def _build_local_expert_routing_kernel(
    routing_ptr,        # [total_tokens, E*R] bool
    rank_routing_ptr,   # [total_tokens, R] bool
    excumsum_ptr,       # [total_tokens, R] int32
    out_ptr,            # output [N*R, E] bool
    local_rank,
    total_tokens,
    R: tl.constexpr,
    E: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Build local_expert_routing_map (non-fuse path)."""
    pid = tl.program_id(0)
    t = pid * BLOCK + tl.arange(0, BLOCK)
    mask = t < total_tokens

    goes = tl.load(rank_routing_ptr + t * R + local_rank, mask=mask, other=False).to(tl.int1)
    dense_pos = tl.load(excumsum_ptr + t * R + local_rank, mask=mask, other=0)

    write_mask = mask & goes
    for e in tl.static_range(E):
        val = tl.load(
            routing_ptr + t * (E * R) + local_rank * E + e,
            mask=write_mask, other=False,
        )
        tl.store(out_ptr + dense_pos * E + e, val, mask=write_mask)


@triton.jit
def _build_fuse_dense_to_expert_kernel(
    routing_ptr,         # [total_tokens, E*R] bool
    rank_routing_ptr,    # [total_tokens, R] bool
    rank_excumsum_ptr,   # [total_tokens, R] int32
    expert_excumsum_ptr, # [total_tokens, E] int32
    expert_offsets_ptr,  # [E] int32
    d2e_ptr,             # output [max_dense, E] int32
    local_rank,
    total_tokens,
    R: tl.constexpr,
    E: tl.constexpr,
    num_permuted_tokens_limit,  # -1 = no limit
    enable_token_drop: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Build dense_to_expert_map (fuse path)."""
    pid = tl.program_id(0)
    t = pid * BLOCK + tl.arange(0, BLOCK)
    mask = t < total_tokens

    goes = tl.load(rank_routing_ptr + t * R + local_rank, mask=mask, other=False).to(tl.int1)
    dense_pos = tl.load(rank_excumsum_ptr + t * R + local_rank, mask=mask, other=0)

    write_mask = mask & goes
    for e in tl.static_range(E):
        eroute = tl.load(
            routing_ptr + t * (E * R) + local_rank * E + e,
            mask=write_mask, other=False,
        ).to(tl.int1)
        eex = tl.load(expert_excumsum_ptr + t * E + e, mask=write_mask, other=0)
        eoff = tl.load(expert_offsets_ptr + e)
        val = tl.where(eroute, eoff + eex, -1)
        if enable_token_drop:
            val = tl.where((val >= 0) & (val >= num_permuted_tokens_limit), -1, val)
        tl.store(d2e_ptr + dense_pos * E + e, val, mask=write_mask)


@triton.jit
def _build_dense_chunk_layout_kernel(
    rank_excumsum_ptr,   # [total_tokens, R] int32
    rank_incumsum_ptr,   # [total_tokens, R] int32
    dcl_ptr,             # output [total_chunks] int32
    local_rank,
    total_tokens,
    N,
    R: tl.constexpr,
    chunk_size,
    chunks_per_rank,
    total_chunks,
    BLOCK: tl.constexpr,
):
    """Build dense_chunk_layout (fuse path)."""
    pid = tl.program_id(0)
    t = pid * BLOCK + tl.arange(0, BLOCK)
    mask = t < total_tokens

    token_rank = t // N
    token_local_id = t % N
    is_chunk_start = (token_local_id % chunk_size) == 0
    global_chunk_id = token_rank * chunks_per_rank + token_local_id // chunk_size

    store_mask = mask & is_chunk_start & (global_chunk_id > 0) & (global_chunk_id < total_chunks)
    ex_val = tl.load(rank_excumsum_ptr + t * R + local_rank, mask=store_mask, other=0)
    tl.store(dcl_ptr + global_chunk_id - 1, ex_val, mask=store_mask)

    # Last element = total tokens for local_rank (handled by the last token)
    is_last = (t == total_tokens - 1)
    last_mask = mask & is_last
    last_val = tl.load(rank_incumsum_ptr + t * R + local_rank, mask=last_mask, other=0)
    tl.store(dcl_ptr + total_chunks - 1, last_val, mask=last_mask)


@triton.jit
def _compute_tokens_per_expert_with_drop_kernel(
    all_expert_sum_ptr,     # [E] int32 - total tokens per expert (raw)
    out_ptr,                # [E] int32 - output tokens_per_expert
    overflow_ptr,           # [1] int32 - overflow flag
    E: tl.constexpr,
    pad_multiple,
    limit,                  # num_permuted_tokens limit
):
    """Compute tokens_per_expert with token-drop logic (fuse + non_blocking).

    Mirrors the HYBRID_EP_BUILD_TOKEN_DROP_ENABLE path in the CUDA scan kernel.
    For each expert, determine how many tokens fit within *limit* taking into
    account padding of all previous experts.
    """
    offs = tl.arange(0, E)
    raw = tl.load(all_expert_sum_ptr + offs).to(tl.int64)

    # Compute padded sizes per expert
    padded = raw
    if pad_multiple > 0:
        padded = ((raw + pad_multiple - 1) // pad_multiple) * pad_multiple

    # Exclusive prefix sum of padded sizes → offset of each expert in merged buffer
    inc_sum = tl.cumsum(padded, axis=0)
    exc_sum = inc_sum - padded           # previous_experts_acc

    prev_plus_cur = exc_sum + raw        # previous_experts_acc + current_expert_valid_tokens

    limit64 = tl.full([1], limit, dtype=tl.int64)

    # Per-expert valid count, matching the C++ branch:
    #   if limit > exc_sum:
    #       if limit >= prev_plus_cur: valid = raw
    #       else: valid = limit - exc_sum
    #   else: valid = 0
    valid = tl.where(
        limit64 > exc_sum,
        tl.where(limit64 >= prev_plus_cur, raw, tl.maximum(limit64 - exc_sum, 0)),
        tl.zeros_like(raw),
    )
    tl.store(out_ptr + offs, valid.to(tl.int32))

    # Overflow flag: 1 if any expert was clipped
    total_raw = tl.sum(raw)
    total_valid = tl.sum(valid)
    flag = tl.where(total_valid < total_raw, 1, 0).to(tl.int32)
    tl.store(overflow_ptr, flag)


@triton.jit
def _permute_preprocess_kernel(
    routing_ptr,        # [max_dispatched, E] bool
    ndt_ptr,            # [1] int32 (num dispatched tokens on GPU)
    expert_excumsum_ptr,    # [max_dispatched, E] int32
    expert_offsets_ptr,     # [E] int32
    row_id_ptr,         # output [max_dispatched + pad_multiple, E] int32
    tokens_per_expert_ptr,  # [E] int32 (raw counts)
    overflow_ptr,       # output [1] int32
    max_dispatched,
    E: tl.constexpr,
    pad_multiple,
    num_permuted_tokens_limit,
    BLOCK: tl.constexpr,
):
    """Build row_id_map for standalone permute path.

    row_id_map uses 1-based indexing:
      >0 : destination position (1-based) in merged expert output
      0  : token not routed to this expert
      <0 : padding slot (negative of 1-based position)
    """
    pid = tl.program_id(0)
    t = pid * BLOCK + tl.arange(0, BLOCK)
    mask = t < max_dispatched

    for e in tl.static_range(E):
        routed = tl.load(routing_ptr + t * E + e, mask=mask, other=False).to(tl.int1)
        ex = tl.load(expert_excumsum_ptr + t * E + e, mask=mask, other=0)
        eoff = tl.load(expert_offsets_ptr + e)
        pos1 = eoff + ex + 1  # 1-based

        val = tl.where(routed, pos1, 0)
        if num_permuted_tokens_limit > 0:
            val = tl.where((val > 0) & (val > num_permuted_tokens_limit), 0, val)
        tl.store(row_id_ptr + t * E + e, val, mask=mask)


@triton.jit
def _permute_preprocess_padding_kernel(
    tokens_per_expert_raw_ptr,  # [E] int32
    expert_offsets_ptr,         # [E] int32
    padded_counts_ptr,          # [E] int32  (padded - raw)
    row_id_ptr,                 # [max_dispatched + pad_multiple, E] int32
    overflow_ptr,               # [1] int32
    ndt_ptr,                    # [1] int32 (num dispatched tokens)
    max_dispatched,
    E: tl.constexpr,
    pad_multiple,
    num_permuted_tokens_limit,
    BLOCK_PAD: tl.constexpr,
):
    """Fill padding rows in row_id_map after the valid token rows."""
    pid = tl.program_id(0)
    pad_idx = pid * BLOCK_PAD + tl.arange(0, BLOCK_PAD)
    pad_mask = pad_idx < pad_multiple

    ndt = tl.load(ndt_ptr)
    row_offset = ndt + pad_idx  # row indices for padding

    for e in tl.static_range(E):
        raw_count = tl.load(tokens_per_expert_raw_ptr + e)
        eoff = tl.load(expert_offsets_ptr + e)
        pad_count = tl.load(padded_counts_ptr + e)

        need_pad = pad_idx < pad_count
        neg_val = -(raw_count + eoff + pad_idx + 1)

        store_mask = pad_mask & need_pad
        if num_permuted_tokens_limit > 0:
            abs_val = -neg_val
            overflow = abs_val > num_permuted_tokens_limit
            neg_val = tl.where(overflow, 0, neg_val)

        tl.store(row_id_ptr + row_offset * E + e, neg_val, mask=store_mask)

        no_pad_mask = pad_mask & (~need_pad)
        tl.store(row_id_ptr + row_offset * E + e, 0, mask=no_pad_mask)


@triton.jit
def _pad_tokens_per_expert_kernel(
    src_ptr,      # [E] int32
    dst_ptr,      # [E] int64
    E: tl.constexpr,
    pad_multiple,
):
    """Pad each element to nearest multiple of pad_multiple, cast to int64."""
    offs = tl.arange(0, E)
    val = tl.load(src_ptr + offs).to(tl.int64)
    if pad_multiple > 0:
        val = ((val + pad_multiple - 1) // pad_multiple) * pad_multiple
    tl.store(dst_ptr + offs, val)


# ---------------------------------------------------------------------------
# High-level PyTorch + Triton implementation
# ---------------------------------------------------------------------------

def allgather_routing_map(
    local_routing_map: torch.Tensor,
    group: torch.distributed.ProcessGroup,
) -> torch.Tensor:
    """All-gather routing map from all ranks in the group."""
    group_size = group.size()
    N = local_routing_map.shape[0]
    total_experts = local_routing_map.shape[1]

    global_routing_map = torch.empty(
        (N * group_size, total_experts),
        dtype=torch.bool, device=local_routing_map.device,
    )
    torch.distributed.all_gather_into_tensor(
        global_routing_map, local_routing_map, group,
    )
    return global_routing_map


def _compute_rank_routing(
    global_routing_map: torch.Tensor,
    num_ranks_per_node: int,
    num_experts_per_rank: int,
) -> torch.Tensor:
    """Reduce per-expert routing to per-rank routing.

    Args:
        global_routing_map: [total_tokens, E*R] bool (single-node slice)

    Returns:
        [total_tokens, R] bool: whether each rank needs each token
    """
    R = num_ranks_per_node
    E = num_experts_per_rank
    total_tokens = global_routing_map.shape[0]
    reshaped = global_routing_map.view(total_tokens, R, E)
    return reshaped.any(dim=-1)  # [total_tokens, R]


def metadata_preprocessing(
    local_routing_map: torch.Tensor,
    group: torch.distributed.ProcessGroup,
    num_of_tokens_per_rank: int,
    num_ranks_per_node: int,
    num_experts_per_rank: int,
    local_rank: int,
    num_of_tokens_per_chunk: int,
    max_num_of_tokens: int,
    num_permuted_tokens: int = -1,
    pad_multiple: int = 0,
    enable_permute: bool = True,
    fuse_permute_dispatch: bool = False,
    non_blocking: bool = False,
) -> MetadataHandle:
    """Metadata preprocessing for hybrid EP (single-node, ROCm/Triton).

    Replaces the C++ ``HybridEPBuffer::metadata_preprocessing`` +
    ``Executor::metadata_preprocess_core`` for the num_of_nodes==1 case.

    Args:
        local_routing_map: [N, E*R] bool  routing map for this rank
        group: process group for all-gather
        num_of_tokens_per_rank: N – tokens produced by each rank
        num_ranks_per_node: R – ranks in the NVLink domain
        num_experts_per_rank: E – experts per rank
        local_rank: this rank's index within the node
        num_of_tokens_per_chunk: chunk size for dispatch/combine
        max_num_of_tokens: upper-bound on tokens in the dense buffer
        num_permuted_tokens: pre-allocated output size (-1 = auto)
        pad_multiple: pad each expert's token count to this multiple
        enable_permute: whether to produce permute metadata
        fuse_permute_dispatch: fused permute-dispatch path
        non_blocking: if True, outputs stay on GPU (no sync)

    Returns:
        MetadataHandle with all fields populated.
    """
    if fuse_permute_dispatch:
        assert enable_permute, "fuse_permute_dispatch requires enable_permute"

    N = num_of_tokens_per_rank
    R = num_ranks_per_node
    E = num_experts_per_rank
    device = local_routing_map.device
    num_nodes = 1  # single-node only

    # ------------------------------------------------------------------
    # Step 1: All-gather routing map
    # ------------------------------------------------------------------
    global_routing_map = allgather_routing_map(local_routing_map, group)
    # For single-node: global_routing_map is [N*R, E*R], bool

    total_tokens = N * R
    node_routing = global_routing_map[:, :E * R]  # single-node slice

    # ------------------------------------------------------------------
    # Step 2: Per-rank routing + prefix sums  (PyTorch cumsum)
    # ------------------------------------------------------------------
    rank_routing = _compute_rank_routing(node_routing, R, E)  # [total_tokens, R]

    rank_int = rank_routing.to(torch.int32).contiguous()
    inclusive_cumsum = rank_int.cumsum(dim=0).contiguous()   # [total_tokens, R]
    exclusive_cumsum = (inclusive_cumsum - rank_int).contiguous()  # [total_tokens, R]

    # ------------------------------------------------------------------
    # Step 3: Build sparse_to_dense_map + rdma_to_attn_map   (Triton)
    # ------------------------------------------------------------------
    padded_N = ((N - 1) // 16 + 1) * 16
    sparse_to_dense_map = torch.empty(N, R, dtype=torch.int32, device=device)
    rdma_to_attn_map = torch.zeros(padded_N, num_nodes, dtype=torch.bool, device=device)

    local_start = local_rank * N
    BLOCK = 256
    grid = ((N + BLOCK - 1) // BLOCK,)
    _build_sparse_to_dense_kernel[grid](
        rank_routing, exclusive_cumsum,
        sparse_to_dense_map, rdma_to_attn_map,
        local_start, N, R, padded_N,
        BLOCK=BLOCK,
    )

    # attn_to_rdma_map: empty for single-node (num_nodes - 1 == 0)
    attn_to_rdma_map = torch.empty(N, 0, dtype=torch.bool, device=device)

    # ------------------------------------------------------------------
    # Step 4: num_dispatched_tokens
    # ------------------------------------------------------------------
    ndt_value = inclusive_cumsum[-1, local_rank]  # scalar on GPU
    if non_blocking:
        num_dispatched_tokens_tensor = ndt_value.unsqueeze(0).to(torch.int32).contiguous()
    else:
        num_dispatched_tokens_tensor = torch.empty(1, dtype=torch.int32, pin_memory=True)
        num_dispatched_tokens_tensor[0] = ndt_value.item()

    # ------------------------------------------------------------------
    # Step 5: local_expert_routing_map  (always allocated)
    # ------------------------------------------------------------------
    local_expert_routing_map = torch.zeros(
        total_tokens, E, dtype=torch.bool, device=device,
    )

    if not fuse_permute_dispatch:
        grid_ler = ((total_tokens + BLOCK - 1) // BLOCK,)
        _build_local_expert_routing_kernel[grid_ler](
            node_routing, rank_routing, exclusive_cumsum,
            local_expert_routing_map,
            local_rank, total_tokens, R, E,
            BLOCK=BLOCK,
        )

    # ------------------------------------------------------------------
    # Step 6: overflow_flag (always allocated)
    # ------------------------------------------------------------------
    overflow_flag = torch.zeros(1, dtype=torch.int32, device=device)

    # ------------------------------------------------------------------
    # Step 7: Fuse-permute-dispatch outputs
    # ------------------------------------------------------------------
    dense_chunk_layout = None
    dense_to_expert_map = None
    tokens_per_expert = None

    if fuse_permute_dispatch:
        # Per-local-expert routing & prefix sums
        reshaped = node_routing.view(total_tokens, R, E)
        local_expert_routing = reshaped[:, local_rank, :]  # [total_tokens, E]
        expert_int = local_expert_routing.to(torch.int32).contiguous()
        expert_inclusive = expert_int.cumsum(dim=0).contiguous()
        expert_exclusive = (expert_inclusive - expert_int).contiguous()  # [total_tokens, E]

        tokens_per_expert_raw = expert_inclusive[-1].to(torch.int32)  # [E]

        # Padded expert sizes + offsets
        if pad_multiple > 0:
            tpe_padded = ((tokens_per_expert_raw.to(torch.int64) + pad_multiple - 1)
                          // pad_multiple * pad_multiple).to(torch.int32)
        else:
            tpe_padded = tokens_per_expert_raw.clone()

        expert_offsets = torch.zeros(E, dtype=torch.int32, device=device)
        if E > 1:
            expert_offsets[1:] = tpe_padded[:-1].cumsum(0).to(torch.int32)

        # Token-drop mode (fuse + non_blocking)
        enable_token_drop = non_blocking
        npt_limit = num_permuted_tokens if enable_token_drop else -1

        # tokens_per_expert output
        if enable_token_drop and num_permuted_tokens > 0:
            tokens_per_expert = torch.empty(E, dtype=torch.int32, device=device)
            _compute_tokens_per_expert_with_drop_kernel[(1,)](
                tokens_per_expert_raw, tokens_per_expert, overflow_flag,
                E=E, pad_multiple=pad_multiple, limit=num_permuted_tokens,
            )
        else:
            tokens_per_expert = tokens_per_expert_raw.clone()

        # dense_to_expert_map
        dense_to_expert_map = torch.full(
            (total_tokens, E), -1, dtype=torch.int32, device=device,
        )
        grid_d2e = ((total_tokens + BLOCK - 1) // BLOCK,)
        _build_fuse_dense_to_expert_kernel[grid_d2e](
            node_routing, rank_routing, exclusive_cumsum, expert_exclusive,
            expert_offsets, dense_to_expert_map,
            local_rank, total_tokens, R, E,
            npt_limit,
            enable_token_drop=enable_token_drop,
            BLOCK=BLOCK,
        )

        # dense_chunk_layout
        chunks_per_rank = (N - 1) // num_of_tokens_per_chunk + 1
        total_chunks = chunks_per_rank * R
        dense_chunk_layout = torch.zeros(total_chunks, dtype=torch.int32, device=device)

        grid_dcl = ((total_tokens + BLOCK - 1) // BLOCK,)
        _build_dense_chunk_layout_kernel[grid_dcl](
            exclusive_cumsum, inclusive_cumsum, dense_chunk_layout,
            local_rank, total_tokens, N, R,
            num_of_tokens_per_chunk, chunks_per_rank, total_chunks,
            BLOCK=BLOCK,
        )

    # ------------------------------------------------------------------
    # Step 8: Standalone permute preprocessing (non-fuse path)
    # ------------------------------------------------------------------
    row_id_map = None

    if enable_permute and not fuse_permute_dispatch:
        # Triton needs GPU-resident tensor for num_dispatched_tokens
        if num_dispatched_tokens_tensor.is_cuda:
            ndt_gpu = num_dispatched_tokens_tensor
        else:
            ndt_gpu = ndt_value.unsqueeze(0).to(torch.int32).contiguous()
        row_id_map, tokens_per_expert, overflow_flag = _permute_preprocessing(
            local_expert_routing_map,
            ndt_gpu,
            max_num_of_tokens,
            E,
            pad_multiple,
            num_permuted_tokens,
            non_blocking,
        )

    # ------------------------------------------------------------------
    # Step 9: Pad tokens_per_expert + compute num_permuted_tokens
    # ------------------------------------------------------------------
    padded_tokens_per_expert = None

    if enable_permute and tokens_per_expert is not None:
        # Always compute on GPU first (Triton kernels write to device memory)
        padded_tpe_gpu = torch.empty(E, dtype=torch.int64, device=device)
        _pad_tokens_per_expert_kernel[(1,)](
            tokens_per_expert, padded_tpe_gpu,
            E=E, pad_multiple=pad_multiple,
        )

        if non_blocking:
            padded_tokens_per_expert = padded_tpe_gpu
        else:
            torch.cuda.current_stream().synchronize()
            padded_tokens_per_expert = padded_tpe_gpu.cpu().pin_memory()
            if num_permuted_tokens < 0:
                num_permuted_tokens = int(padded_tokens_per_expert.sum().item())

    return MetadataHandle(
        sparse_to_dense_map=sparse_to_dense_map,
        rdma_to_attn_map=rdma_to_attn_map,
        attn_to_rdma_map=attn_to_rdma_map,
        num_dispatched_tokens_tensor=num_dispatched_tokens_tensor,
        local_expert_routing_map=local_expert_routing_map,
        num_of_tokens_per_rank=N,
        num_permuted_tokens=num_permuted_tokens,
        row_id_map=row_id_map,
        tokens_per_expert=tokens_per_expert,
        padded_tokens_per_expert=padded_tokens_per_expert,
        overflow_flag=overflow_flag,
        dense_chunk_layout=dense_chunk_layout,
        dense_to_expert_map=dense_to_expert_map,
    )


# ---------------------------------------------------------------------------
# Standalone permute preprocessing  (non-fuse path)
# ---------------------------------------------------------------------------

def _permute_preprocessing(
    local_expert_routing_map: torch.Tensor,  # [max_dispatched, E] bool on GPU
    num_dispatched_tokens_tensor: torch.Tensor,  # [1] int32
    max_num_dispatched_tokens: int,
    num_experts: int,
    pad_multiple: int,
    num_permuted_tokens_limit: int,
    non_blocking: bool,
):
    """Compute row_id_map, tokens_per_expert, overflow_flag for standalone permute.

    Mirrors the CUDA ``permute_preprocessing_kernel`` cooperative kernel.

    Returns:
        (row_id_map, tokens_per_expert, overflow_flag)
    """
    E = num_experts
    device = local_expert_routing_map.device
    max_d = max_num_dispatched_tokens

    routing = local_expert_routing_map[:max_d].contiguous()  # [max_d, E]

    # Per-expert prefix sums
    expert_int = routing.to(torch.int32).contiguous()
    expert_inclusive = expert_int.cumsum(dim=0).contiguous()  # [max_d, E]
    expert_exclusive = (expert_inclusive - expert_int).contiguous()  # [max_d, E]

    tokens_per_expert_raw = expert_inclusive[-1].to(torch.int32)  # [E]

    # Padded sizes
    if pad_multiple > 0:
        tpe_padded = ((tokens_per_expert_raw.to(torch.int64) + pad_multiple - 1)
                      // pad_multiple * pad_multiple).to(torch.int32)
    else:
        tpe_padded = tokens_per_expert_raw.clone()

    # Expert offsets (exclusive prefix sum of padded sizes)
    expert_offsets = torch.zeros(E, dtype=torch.int32, device=device)
    if E > 1:
        expert_offsets[1:] = tpe_padded[:-1].cumsum(0).to(torch.int32)

    # Build row_id_map for valid tokens
    npt_limit = num_permuted_tokens_limit if num_permuted_tokens_limit > 0 else 0
    total_rows = max_d + pad_multiple
    row_id_map = torch.zeros(total_rows, E, dtype=torch.int32, device=device)

    BLOCK = 256
    grid = ((max_d + BLOCK - 1) // BLOCK,)
    _permute_preprocess_kernel[grid](
        routing, None,
        expert_exclusive, expert_offsets,
        row_id_map, tokens_per_expert_raw,
        torch.empty(1, dtype=torch.int32, device=device),  # placeholder overflow
        max_d, E=E,
        pad_multiple=pad_multiple,
        num_permuted_tokens_limit=npt_limit,
        BLOCK=BLOCK,
    )

    # Fill padding rows
    if pad_multiple > 0:
        padded_diff = (tpe_padded - tokens_per_expert_raw).to(torch.int32)
        BLOCK_PAD = min(256, triton.next_power_of_2(pad_multiple))
        grid_pad = ((pad_multiple + BLOCK_PAD - 1) // BLOCK_PAD,)
        _permute_preprocess_padding_kernel[grid_pad](
            tokens_per_expert_raw, expert_offsets, padded_diff,
            row_id_map,
            torch.empty(1, dtype=torch.int32, device=device),
            num_dispatched_tokens_tensor,
            max_d, E=E,
            pad_multiple=pad_multiple,
            num_permuted_tokens_limit=npt_limit,
            BLOCK_PAD=BLOCK_PAD,
        )

    # Compute overflow flag + final tokens_per_expert
    overflow_flag = torch.zeros(1, dtype=torch.int32, device=device)
    npt_eff = num_permuted_tokens_limit if num_permuted_tokens_limit > 0 else -1

    if npt_eff > 0:
        # Check if any routed token was zeroed (dropped)
        should_be_nonzero = routing[:max_d]
        is_zero = row_id_map[:max_d] == 0
        dropped = should_be_nonzero & is_zero
        if dropped.any():
            overflow_flag.fill_(1)
        # Vectorised overflow trimming
        padded_ends = tpe_padded + expert_offsets
        overflow_num = (padded_ends - npt_eff).clamp(min=0)
        tokens_per_expert = (tpe_padded - overflow_num).clamp(min=0).to(torch.int32)
    else:
        tokens_per_expert = tpe_padded.clone().to(torch.int32)

    return row_id_map, tokens_per_expert, overflow_flag
