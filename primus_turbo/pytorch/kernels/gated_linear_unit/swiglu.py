import torch
import triton

from primus_turbo.triton.gated_linear_unit.swiglu import (
    swiglu_bwd_kernel,
    swiglu_bwd_with_tokens_per_expert_kernel,
    swiglu_fwd_kernel,
    swiglu_fwd_kernel_with_tokens_per_expert,
)


def swiglu_fwd_with_tokens_per_expert(x: torch.Tensor, probs: torch.Tensor, tokens_per_expert: torch.Tensor):
    num_tokens, double_hidden_size = x.size()
    num_expert = tokens_per_expert.size(0)

    probs = probs.unsqueeze(-1)

    out = torch.empty(num_tokens, double_hidden_size // 2, dtype=x.dtype, device=x.device)

    BLOCK_SIZE = 8192
    grid = (BLOCK_SIZE,)
    swiglu_fwd_kernel_with_tokens_per_expert[grid](
        x,
        probs,
        tokens_per_expert,
        out,
        dummy_num_tokens=num_tokens,
        num_expert=num_expert,
        stride_x_token=x.stride(0),
        stride_probs_token=probs.stride(0),
        stride_out_token=out.stride(0),
        LOAD_WIDTH_X=triton.next_power_of_2(double_hidden_size // 2),
        LOAD_WIDTH_TOKENS_PER_EXPERT=triton.next_power_of_2(num_expert),
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out


def swiglu_bwd_with_tokens_per_expert(
    grad_out: torch.Tensor, x: torch.Tensor, probs: torch.Tensor, tokens_per_expert: torch.Tensor
):
    num_tokens, hidden_size = grad_out.size()
    num_expert = tokens_per_expert.size(0)

    grad_x = torch.empty_like(x)
    grad_probs = torch.empty_like(probs)

    BLOCK_SIZE = 8192
    grid = (BLOCK_SIZE,)
    swiglu_bwd_with_tokens_per_expert_kernel[grid](
        grad_out,
        x,
        probs,
        tokens_per_expert,
        grad_x,
        grad_probs,
        dummy_num_tokens=num_tokens,
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


def swiglu_fwd(x: torch.Tensor, probs: torch.Tensor):
    num_tokens, double_hidden_size = x.size()

    probs = probs.unsqueeze(-1)
    out = torch.empty(num_tokens, double_hidden_size // 2, dtype=x.dtype, device=x.device)

    grid = (num_tokens,)
    swiglu_fwd_kernel[grid](
        x,
        probs,
        out,
        num_tokens=num_tokens,
        stride_x_token=x.stride(0),
        stride_probs_token=probs.stride(0),
        stride_out_token=out.stride(0),
        LOAD_WIDTH=triton.next_power_of_2(double_hidden_size // 2),
    )

    return out


def swiglu_bwd(grad_out: torch.Tensor, x: torch.Tensor, probs: torch.Tensor):
    num_tokens, hidden_size = grad_out.size()

    grad_x = torch.empty_like(x)
    grad_probs = torch.empty_like(probs)

    grid = (num_tokens,)
    swiglu_bwd_kernel[grid](
        grad_out,
        x,
        probs,
        grad_x,
        grad_probs,
        num_tokens=num_tokens,
        stride_grad_out_token=grad_out.stride(0),
        stride_x_token=x.stride(0),
        stride_probs_token=probs.stride(0),
        stride_grad_x_token=grad_x.stride(0),
        stride_grad_probs_token=grad_probs.stride(0),
        LOAD_WIDTH=triton.next_power_of_2(hidden_size),
    )

    return grad_x, grad_probs
