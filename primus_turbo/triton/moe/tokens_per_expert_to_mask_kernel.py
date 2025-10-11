import triton
import triton.language as tl


@triton.jit
def tokens_per_expert_to_mask_kernel(
    # pointers
    tokens_per_expert_ptr,
    mask_ptr,
    # sizes
    num_expert: tl.constexpr,
    # metas
    TOKENS_PER_EXPERT_LOAD_WIDTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    tokens_per_expert_off = tl.arange(0, TOKENS_PER_EXPERT_LOAD_WIDTH)
    real_num_tokens = tl.load(
        tokens_per_expert_ptr + tokens_per_expert_off, mask=(tokens_per_expert_off < num_expert)
    )
    real_num_tokens = tl.sum(real_num_tokens)

    ones = tl.full((BLOCK_SIZE,), 1, dtype=tl.int64)

    off = tl.arange(0, BLOCK_SIZE)
    mask = (pid * BLOCK_SIZE + off) < real_num_tokens

    tl.store(mask_ptr + pid * BLOCK_SIZE + off, ones, mask=mask)
