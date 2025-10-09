import triton
import triton.language as tl

tl_extra_shim = triton.language.extra.hip.libdevice


@triton.jit
def pow(x, exponent):
    if x.type.element_ty == tl.bfloat16:
        return tl_extra_shim.pow(x.to(tl.float32), exponent)
    elif x.type.element_ty == tl.float16:
        return tl_extra_shim.pow(x.to(tl.float32), exponent)
    else:
        return tl_extra_shim.pow(x.to(tl.float64), exponent)
