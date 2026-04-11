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
