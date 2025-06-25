import triton
import triton.language as tl


# pertensor quantize
@triton.jit
def quant_fp8_pertensor_kernel(
    x_ptr,
    x_scale_ptr,
    x_fp8_ptr,
    n_elements,
    # metas
    TL_DTYPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    scale = tl.load(x_scale_ptr, mask=True)

    x_fp8 = x * scale

    tl.store(x_fp8_ptr + offsets, x_fp8.to(TL_DTYPE), mask=mask)


# pertensor dequantize
@triton.jit
def dequant_fp8_pertensor_kernel(
    x_fp8_ptr,
    x_scale_inv_ptr,
    x_ptr,
    n_elements,
    # metas
    TL_DTYPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    compute_type = tl.float32

    pid = tl.program_id(axis=0)

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x_fp8 = tl.load(x_fp8_ptr + offsets, mask=mask)
    scale_inv = tl.load(x_scale_inv_ptr, mask=True)

    x = x_fp8.to(compute_type) * scale_inv

    tl.store(x_ptr + offsets, x.to(TL_DTYPE), mask=mask)
