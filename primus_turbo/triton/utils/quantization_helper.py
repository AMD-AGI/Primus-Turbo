import triton
import triton.language as tl

FP32_MANTISSA_BITS = tl.constexpr(23)
FP32_EXPONENT_BITS = tl.constexpr(8)
FP32_EXPONENT_EXP_BIAS = tl.constexpr(127)

BF16_MANTISSA_BITS = tl.constexpr(7)
BF16_EXPONENT_BITS = tl.constexpr(8)
BF16_EXPONENT_EXP_BIAS = tl.constexpr(127)

FP8E5M2_MANTISSA_BITS = tl.constexpr(2)
FP8E5M2_EXPONENT_BITS = tl.constexpr(5)
FP8E5M2_TARGET_MAX_POW2 = tl.constexpr(15)

# NOTE: MXFP8 not support on MI300. Assuming fp8 e4m3 is not fnuz.
FP8E4M3_MANTISSA_BITS = tl.constexpr(3)
FP8E4M3_EXPONENT_BITS = tl.constexpr(4)
FP8E4M3_TARGET_MAX_POW2 = tl.constexpr(8)

FP4_MANTISSA_BITS = tl.constexpr(1)
FP4_EXPONENT_BITS = tl.constexpr(2)
FP4_TARGET_MAX_POW2 = tl.constexpr(2)

E8M0_EXPONENT_BIAS = tl.constexpr(127)


@triton.jit
def scale_e8m0_to_rcp(scale) -> tl.float32:
    exp_val = (E8M0_EXPONENT_BIAS - scale).to(tl.float32)
    rcp = tl.exp2(exp_val)
    rcp = tl.where(scale == 0, 1.0, rcp)

    return rcp


# NOTE(ruibin): The triton not support fp4 dtype. So we separate the fp4 and fp8 e8m0 scale calculation.
@triton.jit
def calculate_fp4_e8m0_scale(x, axis: tl.constexpr) -> tl.uint8:
    if x.type.element_ty == tl.float32:
        hp_int_dtype = tl.int32
        hp_mbits = FP32_MANTISSA_BITS
        hp_ebits = FP32_EXPONENT_BITS
        hp_exp_bias = FP32_EXPONENT_EXP_BIAS
    else:
        tl.device_assert(x.type.element_ty == tl.bfloat16, "Only float32 and bfloat16 are supported")
        hp_int_dtype = tl.int16
        hp_mbits = BF16_MANTISSA_BITS
        hp_ebits = BF16_EXPONENT_BITS
        hp_exp_bias = BF16_EXPONENT_EXP_BIAS
    mbits = FP4_MANTISSA_BITS
    target_max_pow2 = FP4_TARGET_MAX_POW2
    e8m0_exponent_bias = E8M0_EXPONENT_BIAS

    max_abs = tl.max(tl.abs(x), axis=axis, keep_dims=True)

    # round even (adaptive)
    max_abs = max_abs.to(hp_int_dtype, bitcast=True)
    val_to_add = 1 << (hp_mbits - mbits - 1)
    hp_exp_mask = (1 << (hp_ebits + 1)) - 1
    extracted_pow2 = (((max_abs + val_to_add) >> hp_mbits) & hp_exp_mask) - hp_exp_bias
    extracted_pow2 = extracted_pow2 - target_max_pow2

    scale_e8m0_unbiased = extracted_pow2.to(tl.bfloat16)

    # Clamp to exponents that can be represented in e8m0
    # Add 1 to capture NaNs
    scale_e8m0_unbiased = tl.clamp(scale_e8m0_unbiased, -1 * e8m0_exponent_bias, e8m0_exponent_bias + 1)

    scale_e8m0_biased = scale_e8m0_unbiased + e8m0_exponent_bias
    scale_e8m0_biased = scale_e8m0_biased.to(tl.uint8)

    return scale_e8m0_biased


@triton.jit
def calculate_fp8_e8m0_scale(x, axis: tl.constexpr, FP8_DTYPE: tl.dtype) -> tl.uint8:
    if x.type.element_ty == tl.float32:
        hp_int_dtype = tl.int32
        hp_mbits = FP32_MANTISSA_BITS
        hp_ebits = FP32_EXPONENT_BITS
        hp_exp_bias = FP32_EXPONENT_EXP_BIAS
    else:
        tl.device_assert(x.type.element_ty == tl.bfloat16, "Only float32 and bfloat16 are supported")
        hp_int_dtype = tl.int16
        hp_mbits = BF16_MANTISSA_BITS
        hp_ebits = BF16_EXPONENT_BITS
        hp_exp_bias = BF16_EXPONENT_EXP_BIAS

    if FP8_DTYPE == tl.float8e4nv:
        mbits = FP8E4M3_MANTISSA_BITS
        target_max_pow2 = FP8E4M3_TARGET_MAX_POW2
    else:
        tl.device_assert(FP8_DTYPE == tl.float8e5, "Unsupported FP8 dtype")

        mbits = FP8E5M2_MANTISSA_BITS
        target_max_pow2 = FP8E5M2_TARGET_MAX_POW2
    e8m0_exponent_bias = E8M0_EXPONENT_BIAS

    max_abs = tl.max(tl.abs(x), axis=axis, keep_dims=True)

    # round even (adaptive)
    max_abs = max_abs.to(hp_int_dtype, bitcast=True)
    val_to_add = 1 << (hp_mbits - mbits - 1)
    hp_exp_mask = (1 << (hp_ebits + 1)) - 1
    extracted_pow2 = (((max_abs + val_to_add) >> hp_mbits) & hp_exp_mask) - hp_exp_bias
    extracted_pow2 = extracted_pow2 - target_max_pow2

    scale_e8m0_unbiased = extracted_pow2.to(tl.bfloat16)

    # Clamp to exponents that can be represented in e8m0
    # Add 1 to capture NaNs
    scale_e8m0_unbiased = tl.clamp(scale_e8m0_unbiased, -1 * e8m0_exponent_bias, e8m0_exponent_bias + 1)

    scale_e8m0_biased = scale_e8m0_unbiased + e8m0_exponent_bias
    scale_e8m0_biased = scale_e8m0_biased.to(tl.uint8)

    return scale_e8m0_biased
