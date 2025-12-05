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


@triton.jit
def generate_randval_2x(m, n, philox_seed, philox_offset):
    ms = tl.arange(0, m)
    ns = tl.arange(0, n)
    rng_offsets = philox_offset + ms[:, None] * n + ns[None, :]
    r1, r2, _, _ = tl.randint4x(philox_seed, rng_offsets)

    return r1, r2


@triton.jit
def float_to_e2m1(val: tl.float32, randval: tl.uint32, USE_SR: tl.constexpr) -> tl.uint8:
    EXP_BIAS_FP32: tl.constexpr = 127
    EXP_BIAS_FP4: tl.constexpr = 1
    EBITS_F32: tl.constexpr = 8
    EBITS_FP4: tl.constexpr = 2
    MBITS_F32: tl.constexpr = 23
    MBITS_FP4: tl.constexpr = 1

    max_normal: tl.constexpr = 6
    min_normal: tl.constexpr = 1

    # Convert quantized fp32 tensor to uint32 before converting to mxfp4 format
    # Note: MXFP4  S:1-bit, E:2-bit, M:1-bit
    #   Zeros: S000 -> +/-0
    #   Denormal Numbers: S001 -> +/- 0.5
    #   Normal Numbers:
    #           S010 -> +/- 1.0
    #           S011 -> +/- 1.5
    #           S100 -> +/- 2.0
    #           S101 -> +/- 3.0
    #           S110 -> +/- 4.0
    #           S111 -> +/- 6.0
    val = val.to(tl.uint32, bitcast=True)

    # Extract sign
    s = val & 0x80000000
    # Set everything to positive, will add sign back at the end
    val = val ^ s

    qx_fp32 = val.to(tl.float32, bitcast=True)
    saturate_mask = qx_fp32 >= max_normal
    denormal_mask = (not saturate_mask) & (qx_fp32 < min_normal)
    normal_mask = not (saturate_mask | denormal_mask)

    # Denormal numbers
    if USE_SR:
        denorm_mask_low = denormal_mask & (qx_fp32 < 0.5)
        denorm_mask_high = denormal_mask & (not denorm_mask_low)
        randval_uint = randval.to(tl.uint32, bitcast=True)
        denormal_x = tl.zeros(val.type.get_block_shapes(), dtype=tl.uint8)

        threshold_low = (qx_fp32 * (2**33 - 2)).to(tl.uint32)
        denormal_x = tl.where(randval_uint <= threshold_low, 1, denormal_x)

        threshold_high = ((qx_fp32 * 2 - 1) * (2**32 - 1)).to(tl.uint32)
        mask_high = randval_uint <= threshold_high
        denormal_x = tl.where(denorm_mask_high & mask_high, 2, denormal_x)
        denormal_x = tl.where(denorm_mask_high & (not mask_high), 1, denormal_x)
    else:
        denorm_exp: tl.constexpr = ((EXP_BIAS_FP32 - EXP_BIAS_FP4) +
                                    (MBITS_F32 - MBITS_FP4) + 1)
        denorm_mask_int: tl.constexpr = denorm_exp << MBITS_F32
        denorm_mask_float: tl.constexpr = tl.cast(denorm_mask_int,
                                                tl.float32,
                                                bitcast=True)

        denormal_x = qx_fp32 + denorm_mask_float
        denormal_x = denormal_x.to(tl.uint32, bitcast=True)
        denormal_x -= denorm_mask_int
        denormal_x = denormal_x.to(tl.uint8)

    # Normal numbers
    normal_x = val
    # resulting mantissa is odd
    mant_odd = (normal_x >> (MBITS_F32 - MBITS_FP4)) & 1
    # update exponent
    val_to_add = ((EXP_BIAS_FP4 - EXP_BIAS_FP32) << MBITS_F32)
    if USE_SR:
        val_to_add += randval & ((1 << (MBITS_F32 - MBITS_FP4)) - 1)
    else:
        val_to_add += (1 << (MBITS_F32 - MBITS_FP4 - 1)) - 1 + mant_odd
    normal_x += val_to_add
    # take the bits!
    normal_x = normal_x >> (MBITS_F32 - MBITS_FP4)
    normal_x = normal_x.to(tl.uint8)

    # Merge results
    e2m1_value = tl.full(val.type.get_block_shapes(), 0x7, dtype=tl.uint8)
    e2m1_value = tl.where(normal_mask, normal_x, e2m1_value)
    e2m1_value = tl.where(denormal_mask, denormal_x, e2m1_value)
    # add sign back
    sign_lp = s >> (MBITS_F32 + EBITS_F32 - MBITS_FP4 - EBITS_FP4)
    sign_lp = sign_lp.to(tl.uint8)
    e2m1_value = e2m1_value | sign_lp

    return e2m1_value


@triton.jit
def pack_e2m1(val_h: tl.uint8, val_low: tl.uint8) -> tl.uint8:
    pass