import torch
import triton
import triton.language as tl


@triton.jit
def fused_swiglu_with_probs_fwd_kernel(
    x_ptr,
    probs_ptr,
    tokens_per_expert_ptr,
    out_ptr,
    # sizes
    num_expert: tl.constexpr,
    # strides
    stride_x_token,
    stride_probs_token,
    stride_out_token,
    # metas
    LOAD_WIDTH_X: tl.constexpr,
    LOAD_WIDTH_TOKENS_PER_EXPERT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    compute_type = probs_ptr.dtype.element_ty
    data_type = x_ptr.dtype.element_ty
    idx_type = tl.int64

    tokens_per_expert_off = tl.arange(0, LOAD_WIDTH_TOKENS_PER_EXPERT)
    num_tokens = tl.load(
        tokens_per_expert_ptr + tokens_per_expert_off, mask=(tokens_per_expert_off < num_expert)
    )
    num_tokens = tl.sum(num_tokens)

    half_stride_x_token = stride_x_token // 2
    loop = (num_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE
    for i in range(0, loop):
        row_idx = (i * BLOCK_SIZE + pid).to(idx_type)
        row_mask = row_idx < num_tokens
        col_off = tl.arange(0, LOAD_WIDTH_X)
        col_mask = col_off < half_stride_x_token

        mask = row_mask & col_mask

        up_ptr = x_ptr + row_idx * stride_x_token
        down_ptr = up_ptr + half_stride_x_token

        up = tl.load(up_ptr + col_off, mask=mask).to(compute_type)
        down = tl.load(down_ptr + col_off, mask=mask).to(compute_type)

        up = tl.sigmoid(up) * up
        out = up * down

        probs = tl.load(probs_ptr + row_idx * stride_probs_token)
        out = out * probs

        tl.store(out_ptr + row_idx * stride_out_token + col_off, out.to(data_type), mask=mask)


def fused_swiglu_with_probs_fwd(x: torch.Tensor, probs: torch.Tensor, tokens_per_expert: torch.Tensor):
    assert x.size(0) == probs.size(0)
    assert x.ndim == 2
    assert probs.ndim == 1

    num_tokens, double_hidden_size = x.size()
    num_expert = tokens_per_expert.size(0)

    probs = probs.unsqueeze(-1)

    out = torch.empty(num_tokens, double_hidden_size // 2, dtype=x.dtype, device=x.device)

    BLOCK_SIZE = 8192
    grid = (BLOCK_SIZE,)
    fused_swiglu_with_probs_fwd_kernel[grid](
        x,
        probs,
        tokens_per_expert,
        out,
        num_expert=num_expert,
        stride_x_token=x.stride(0),
        stride_probs_token=probs.stride(0),
        stride_out_token=out.stride(0),
        LOAD_WIDTH_X=triton.next_power_of_2(double_hidden_size // 2),
        LOAD_WIDTH_TOKENS_PER_EXPERT=triton.next_power_of_2(num_expert),
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out


@triton.jit
def fused_swiglu_with_probs_bwd_kernel(
    # pointers
    grad_out_ptr,
    x_ptr,
    probs_ptr,
    tokens_per_expert_ptr,
    grad_x_ptr,
    grad_probs_ptr,
    # sizes
    num_expert: tl.constexpr,
    # strides
    stride_grad_out_token,
    stride_x_token,
    stride_probs_token,
    stride_grad_x_token,
    stride_grad_probs_token,
    # metas
    LOAD_WIDTH_X: tl.constexpr,
    LOAD_WIDTH_TOKENS_PER_EXPERT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    compute_type = probs_ptr.dtype.element_ty
    grad_x_data_type = grad_x_ptr.dtype.element_ty
    grad_probs_data_type = grad_probs_ptr.dtype.element_ty
    idx_type = tl.int64

    tokens_per_expert_off = tl.arange(0, LOAD_WIDTH_TOKENS_PER_EXPERT)
    num_tokens = tl.load(
        tokens_per_expert_ptr + tokens_per_expert_off, mask=(tokens_per_expert_off < num_expert)
    )
    num_tokens = tl.sum(num_tokens)

    half_stride_x_token = stride_x_token // 2
    loop = (num_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE
    for i in range(0, loop):
        row_idx = (i * BLOCK_SIZE + pid).to(idx_type)
        row_mask = row_idx < num_tokens
        col_off = tl.arange(0, LOAD_WIDTH_X)
        col_mask = col_off < half_stride_x_token

        mask = row_mask & col_mask

        up_ptr = x_ptr + row_idx * stride_x_token
        down_ptr = up_ptr + half_stride_x_token

        up = tl.load(up_ptr + col_off, mask=mask).to(compute_type)
        down = tl.load(down_ptr + col_off, mask=mask).to(compute_type)

        probs = tl.load(probs_ptr + row_idx * stride_probs_token).to(compute_type)

        sigmoid = tl.sigmoid(up)
        swiglu = sigmoid * up

        grad_out = tl.load(grad_out_ptr + row_idx * stride_grad_out_token + col_off, mask=mask).to(
            compute_type
        )

        grad_probs = grad_out * (swiglu * down)
        grad_probs_sum = tl.sum(grad_probs)

        grad_out_with_probs = grad_out * probs

        grad_down = grad_out_with_probs * swiglu

        grad_swiglu = sigmoid * (1.0 + up * (1.0 - sigmoid))
        grad_up = grad_out_with_probs * (down * grad_swiglu)

        tl.store(
            grad_probs_ptr + row_idx * stride_grad_probs_token,
            grad_probs_sum.to(grad_probs_data_type),
            mask=row_mask,
        )
        tl.store(
            grad_x_ptr + row_idx * stride_grad_x_token + col_off, grad_up.to(grad_x_data_type), mask=mask
        )
        tl.store(
            grad_x_ptr + row_idx * stride_grad_x_token + stride_grad_x_token // 2 + col_off,
            grad_down.to(grad_x_data_type),
            mask=mask,
        )


def fused_swiglu_with_probs_bwd(
    grad_out: torch.Tensor, x: torch.Tensor, probs: torch.Tensor, tokens_per_expert: torch.Tensor
):
    assert grad_out.ndim == 2
    assert x.size(0) == probs.size(0)
    assert x.ndim == 2
    assert probs.ndim == 1

    num_tokens, hidden_size = grad_out.size()
    num_expert = tokens_per_expert.size(0)

    grad_x = torch.empty_like(x)
    grad_probs = torch.empty_like(probs)

    probs = probs.unsqueeze(-1)

    BLOCK_SIZE = 8192
    grid = (BLOCK_SIZE,)
    fused_swiglu_with_probs_bwd_kernel[grid](
        grad_out,
        x,
        probs,
        tokens_per_expert,
        grad_x,
        grad_probs,
        num_expert=num_expert,
        stride_grad_out_token=grad_out.stride(0),
        stride_x_token=x.stride(0),
        stride_probs_token=probs.stride(0),
        stride_grad_x_token=grad_x.stride(0),
        stride_grad_probs_token=grad_probs.stride(0),
        LOAD_WIDTH_X=triton.next_power_of_2(hidden_size),
        LOAD_WIDTH_TOKENS_PER_EXPERT=triton.next_power_of_2(num_expert),
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return grad_x, grad_probs


@triton.jit
def fused_geglu_with_probs_fwd_kernel(
    x_ptr,
    probs_ptr,
    tokens_per_expert_ptr,
    out_ptr,
    # sizes
    num_expert: tl.constexpr,
    # strides
    stride_x_token,
    stride_probs_token,
    stride_out_token,
    # metas
    LOAD_WIDTH_X: tl.constexpr,
    LOAD_WIDTH_TOKENS_PER_EXPERT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    compute_type = probs_ptr.dtype.element_ty
    data_type = x_ptr.dtype.element_ty
    idx_type = tl.int64

    tokens_per_expert_off = tl.arange(0, LOAD_WIDTH_TOKENS_PER_EXPERT)
    num_tokens = tl.load(
        tokens_per_expert_ptr + tokens_per_expert_off, mask=(tokens_per_expert_off < num_expert)
    )
    num_tokens = tl.sum(num_tokens)

    half_stride_x_token = stride_x_token // 2
    loop = (num_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE
    for i in range(0, loop):
        row_idx = (i * BLOCK_SIZE + pid).to(idx_type)
        row_mask = row_idx < num_tokens
        col_off = tl.arange(0, LOAD_WIDTH_X)
        col_mask = col_off < half_stride_x_token

        mask = row_mask & col_mask

        up_ptr = x_ptr + row_idx * stride_x_token
        down_ptr = up_ptr + half_stride_x_token

        up = tl.load(up_ptr + col_off, mask=mask).to(compute_type)
        down = tl.load(down_ptr + col_off, mask=mask).to(compute_type)

        up = 0.5 * up * (1.0 + tl.erf(up / tl.sqrt(2.0)))
        out = up * down

        probs = tl.load(probs_ptr + row_idx * stride_probs_token)
        out = out * probs

        tl.store(out_ptr + row_idx * stride_out_token + col_off, out.to(data_type), mask=mask)


def fused_gelu_with_probs_fwd(x: torch.Tensor, probs: torch.Tensor, tokens_per_expert: torch.Tensor):
    assert x.size(0) == probs.size(0)
    assert x.ndim == 2
    assert probs.ndim == 1

    num_tokens, double_hidden_size = x.size()
    num_expert = tokens_per_expert.size(0)

    probs = probs.unsqueeze(-1)

    out = torch.empty(num_tokens, double_hidden_size // 2, dtype=x.dtype, device=x.device)

    BLOCK_SIZE = triton.next_power_of_2(num_tokens)
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8

    if BLOCK_SIZE >= 4096:
        num_warps = 16

    grid = (BLOCK_SIZE,)
    fused_geglu_with_probs_fwd_kernel[grid](
        x,
        probs,
        tokens_per_expert,
        out,
        num_expert=num_expert,
        stride_x_token=x.stride(0),
        stride_probs_token=probs.stride(0),
        stride_out_token=out.stride(0),
        LOAD_WIDTH_X=triton.next_power_of_2(double_hidden_size // 2),
        LOAD_WIDTH_TOKENS_PER_EXPERT=triton.next_power_of_2(num_expert),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )

    return out
