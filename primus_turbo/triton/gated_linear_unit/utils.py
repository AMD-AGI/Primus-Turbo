import triton
import triton.language as tl


@triton.jit
def _calc_and_valid_num_tokens(
    tokens_per_expert_ptr,
    dummy_num_tokens: tl.constexpr,
    num_expert: tl.constexpr,
    LOAD_WIDTH: tl.constexpr,
):
    tokens_per_expert_off = tl.arange(0, LOAD_WIDTH)
    num_tokens = tl.load(
        tokens_per_expert_ptr + tokens_per_expert_off, mask=(tokens_per_expert_off < num_expert)
    )
    num_tokens = tl.sum(num_tokens)

    tl.device_assert(num_tokens <= dummy_num_tokens, "num_tokens is invalid.")

    return num_tokens
