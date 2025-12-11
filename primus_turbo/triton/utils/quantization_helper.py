import triton
import triton.language as tl

FP32_EXPONENT_BIAS = tl.constexpr(127)
FP32_MANTISSA_BITS = tl.constexpr(23)


@triton.jit
def exp2f_rcp(biased_exp: tl.uint8) -> tl.float32:
    biased_exp_f32 = biased_exp.to(tl.float32)
    exp_val = FP32_EXPONENT_BIAS - biased_exp_f32
    result = tl.exp2(exp_val)
    final_result = tl.where(biased_exp == 0, 1.0, result)
    return final_result


@triton.jit
def calculate_e8m0_scale(x: tl.float32, axis: tl.constexpr) -> tl.uint8:
    if x.type.element_ty == tl.float32:
        hp_int_dtype = tl.int32
        hp_mbits = 23
        hp_ebits = 8
    else:
        hp_int_dtype = tl.int16
        hp_mbits = 7
        hp_ebits = 8
    mbits = 1
    sbits = 1
    target_max_pow2 = 2

    max_abs = tl.max(tl.abs(x), axis=axis, keep_dims=True)
    max_abs = max_abs.to(x.type.element_ty)

    # round even (adaptive)
    max_abs = max_abs.to(hp_int_dtype, bitcast=True)
    val_to_add = 1 << (hp_mbits - mbits - 1)
    mask = ((1 << (hp_ebits + sbits)) - 1) << hp_mbits
    max_abs = ((max_abs + val_to_add) & mask) >> hp_mbits
    scales = max_abs - target_max_pow2

    scales = tl.where(scales < 1, 1, scales)

    return scales.to(tl.uint8)
