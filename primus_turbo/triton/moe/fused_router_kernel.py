import triton
import triton.language as tl

from primus_turbo.triton.utils.argsort import argsort


@triton.jit
def fused_scaling_group_sum_routing_kernel(
    input_logit_ptr,  # [s, e]
    output_scores_ptr,  # [s, e]
    output_topk_probs_ptr,  # [s, k]
    output_topk_idx_ptr,  # [s, k]
    output_raw_topk_logits_ptr,  # [s, k]
    s: tl.constexpr,  # seq len
    e: tl.constexpr,  # how many experts
    g: tl.constexpr,  # how many groups
    k: tl.constexpr,  # topk
    selected_groups: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,  # cols
    num_stages: tl.constexpr,
    score_function: tl.constexpr,  # 0 sigmoid 1 softmax
    scaling_factor: float = 1.0,
):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    # offset and mask
    col_offsets = tl.arange(0, BLOCK_SIZE)
    col_mask = col_offsets < e

    k_mask = col_offsets < k
    sort_mask = col_offsets < selected_groups * (e // g)

    for row_idx in tl.range(row_start, s, row_step, num_stages=num_stages):
        # load row
        input_logit_row_ptr = input_logit_ptr + row_idx * e + col_offsets

        # cal score for aux loss
        if score_function == 0:
            input_logit_row = tl.load(input_logit_row_ptr, mask=col_mask, other=float(0.0))
            row_logit = tl.sigmoid(input_logit_row.to(tl.float32))
            row_sum = tl.sum(row_logit, dtype=tl.float32)
            row_scores = (row_logit / (row_sum + 1e-20)).to(input_logit_row.dtype)
        else:
            input_logit_row = tl.load(input_logit_row_ptr, mask=col_mask, other=-float("inf"))
            row_logit = tl.softmax(input_logit_row.to(tl.float32))
            row_scores = row_logit.to(input_logit_row.dtype)

        row_output_scores_ptr = output_scores_ptr + row_idx * e + col_offsets
        tl.store(row_output_scores_ptr, row_scores, mask=col_mask)

        # sort inner groups
        input_logit_groups = tl.reshape(row_logit, (g, e // g))  # [g, e // g]
        inner_groups_idx = tl.arange(0, e).reshape(g, e // g)
        sorted_groups_logits, sorted_inner_groups_idx = argsort(input_logit_groups, inner_groups_idx, 1, True)

        # gather inner groups top_(k // selected_groups)
        inner_group_gather_idx = (
            tl.arange(0, k // selected_groups)
            .reshape(1, k // selected_groups)
            .broadcast_to(g, k // selected_groups)
        )
        sorted_groups_logits = sorted_groups_logits.gather(inner_group_gather_idx, axis=1)
        sorted_inner_groups_idx = sorted_inner_groups_idx.gather(inner_group_gather_idx, axis=1)

        groups_topk_sum = tl.sum(sorted_groups_logits, axis=1)
        groups_idx = tl.arange(0, g)
        _, sorted_groups_idx = argsort(groups_topk_sum, groups_idx, 0, True)

        # gather topk
        sorted_groups_idx_for_gather = tl.broadcast_to(sorted_groups_idx.reshape(g, 1), (g, e // g))
        sorted_raw_topk_logits = tl.gather(input_logit_groups, sorted_groups_idx_for_gather, axis=0).reshape(
            e
        )
        sorted_topk_idxs = tl.gather(inner_groups_idx, sorted_groups_idx_for_gather, axis=0).reshape(e)

        minus_ones = tl.full(sorted_raw_topk_logits.shape, -1.0, dtype=sorted_raw_topk_logits.dtype)
        sorted_raw_topk_logits = tl.where(sort_mask, sorted_raw_topk_logits, minus_ones)
        sorted_raw_topk_logits, sorted_topk_idxs = argsort(sorted_raw_topk_logits, sorted_topk_idxs, 0, True)

        # cal scaled probs
        if score_function == 0:
            sorted_topk_logits = sorted_raw_topk_logits / (tl.sum(sorted_raw_topk_logits * k_mask) + 1e-20)
            row_output_raw_topk_logits = output_raw_topk_logits_ptr + row_idx * k + col_offsets
            tl.store(row_output_raw_topk_logits, sorted_raw_topk_logits, mask=k_mask)
        else:
            sorted_topk_logits = sorted_raw_topk_logits

        sorted_topk_logits = scaling_factor * sorted_topk_logits

        # save results
        row_output_sorted_groups_logits = output_topk_probs_ptr + row_idx * k + col_offsets
        row_output_sorted_inner_groups_idx = output_topk_idx_ptr + row_idx * k + col_offsets
        row_output_raw_topk_logits = output_raw_topk_logits_ptr + row_idx * k + col_offsets
        tl.store(row_output_sorted_groups_logits, sorted_topk_logits, mask=k_mask)
        tl.store(row_output_sorted_inner_groups_idx, sorted_topk_idxs, mask=k_mask)
