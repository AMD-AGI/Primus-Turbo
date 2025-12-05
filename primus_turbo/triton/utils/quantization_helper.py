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
def float_to_e8m0(val: tl.float32) -> tl.uint8:
    is_nan = val != val
    is_inf = tl.abs(val) == float("inf")
    is_zero = val == 0.0

    result_e8m0 = tl.zeros(val.shape, dtype=tl.uint8)  # Placeholder
    val_u32 = tl.cast(val, tl.uint32, bitcast=True)

    # Extract exponent and mantissa
    exponent_raw = (val_u32 >> FP32_MANTISSA_BITS) & 0xFF
    mantissa = val_u32 & 0x7FFFFF

    # Round up exponent and deal with satfinite.
    # (mantissa > 0 && exponent != 0xFE) && !(exponent == 0 && mantissa <= 0x400000)
    cond1 = mantissa > 0
    cond2 = exponent_raw != 0xFE
    cond3_part1 = exponent_raw == 0
    cond3_part2 = mantissa <= 0x400000
    cond3 = cond3_part1 & cond3_part2

    round_up_condition = (cond1 & cond2) & ~cond3

    # Increment exponent if the condition is true
    calculated_exponent = tl.where(round_up_condition, exponent_raw + 1, exponent_raw)

    # Priority: NaN -> Inf -> Zero -> Calculated Exponent
    result_e8m0 = tl.where(is_nan, tl.full(val.shape, 0xFF, dtype=tl.uint8), result_e8m0)
    result_e8m0 = tl.where(~is_nan & is_inf, tl.full(val.shape, 0xFE, dtype=tl.uint8), result_e8m0)
    result_e8m0 = tl.where(~is_nan & ~is_inf & is_zero, tl.full(val.shape, 0x00, dtype=tl.uint8), result_e8m0)
    result_e8m0 = tl.where(~is_nan & ~is_inf & ~is_zero, calculated_exponent, result_e8m0)

    return result_e8m0