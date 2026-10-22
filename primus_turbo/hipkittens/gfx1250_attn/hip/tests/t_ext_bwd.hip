/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// t_ext_bwd.hip -- bit-exact tests of hk_attn/ext_bwd.h (the backward v0's added primitives), one workgroup
// per kernel, deterministic inputs, raw results through buffer stores; the host driver hk/lab/test_ext_bwd.py
// builds every expectation on the CPU (exact rational arithmetic, one fp32 rounding per operation).
//
//   t_softmax_rows  softmax_arg_rows (v_pk_fma, per-row bias pairs), exp2_acc at integer arguments,
//                   mask_col_gt_row(MASK_NEG) between them (-> +0), dsoftmax, scale_acc, cvt_pk_bf16 of P / dS
//   t_lds_small     ds_store_b64 / ds_store_b32 at lane-computed offsets, read back as b128
//   t_sig_tdm       4 waves: TDM tile A + B, protocol ds_store, 2 uncovered b128 loads, then
//                   split_signal_tdm_dscnt<1, 2> (A retired, B in flight), wait; read the other waves' A tiles
//                   and stores (the structural ISA check is the real test: 2 loads, no write in flight)
// Kernels take only pointers and ints; every global access is a buffer access within the extent passed.
#include "hk_attn/ext_bwd.h"
#include "hk_attn/prims.h"

using namespace hk_attn;

namespace {
HKA_DEV acc_t get_acc(rsrc_t r, int off, int lane) {
    const u32x4 a = buf_load_b128(r, off + 32 * lane, 0), b = buf_load_b128(r, off + 32 * lane + 16, 0);
    return __builtin_bit_cast(acc_t, __builtin_shufflevector(a, b, 0, 1, 2, 3, 4, 5, 6, 7));
}
HKA_DEV void put_acc(rsrc_t r, int off, acc_t c, int lane) {
    buf_store_b128(r, off + 32 * lane, 0, lo_u32(c));
    buf_store_b128(r, off + 32 * lane + 16, 0, hi_u32(c));
}
} // namespace

// ---- t_softmax_rows: src (fp32) = S [32 lanes][8] | E [32][8] (exp2 args) | P [32][8] | DP [32][8] | NL [16] |
// DE [16] | c1, sc (bits) at 4160, 4164; out = T | EXP | MASKED EXP | DS | SCALED | PU, DU (bf16) ------------------
// row m = 8h + i of lane (r, h) element i takes NL[m] / DE[m]; mask offset `off` (wave-uniform)
extern "C" __global__ __launch_bounds__(32) void t_softmax_rows(const void *src, int src_bytes, void *out,
                                                                int out_bytes, int off) {
    const int    l = lane_id();
    const rsrc_t rs = make_rsrc(src, src_bytes), ro = make_rsrc(out, out_bytes);
    const acc_t  s = get_acc(rs, 0, l), e = get_acc(rs, 1024, l), p = get_acc(rs, 2048, l), dp = get_acc(rs, 3072, l);
    const int    h = l / 16;
    const f32x4  nl_lo = buf_load_f32x4(rs, 4096 + 32 * h, 0), nl_hi = buf_load_f32x4(rs, 4096 + 32 * h + 16, 0);
    const f32x4  de_lo = buf_load_f32x4(rs, 4160 + 32 * h, 0), de_hi = buf_load_f32x4(rs, 4160 + 32 * h + 16, 0);
    const float  c1 = buf_load_f32(rs, 4224, 0), sc = buf_load_f32(rs, 4228, 0);
    const acc_t  t = softmax_arg_rows(s, c1, nl_lo, nl_hi);
    put_acc(ro, 0, t, l);
    put_acc(ro, 1024, exp2_acc(e), l);
    put_acc(ro, 2048, exp2_acc(mask_col_gt_row(e, l, off, MASK_NEG)), l);
    const acc_t d = dsoftmax(p, dp, de_lo, de_hi);
    put_acc(ro, 3072, d, l);
    put_acc(ro, 4096, scale_acc(s, sc), l);
    buf_store_b128(ro, 5120 + 16 * l, 0, cvt_pk_bf16(p));
    buf_store_b128(ro, 5632 + 16 * l, 0, cvt_pk_bf16(d));
}

// ---- t_lds_small: lane l stores u32x2 {A + l, B + l} at 8 * (31 - l) and u32 C + l at 256 + 4 * ((l * 7) % 32) of
// an LDS area pre-filled with FILL; out = the 384-B area -------------------------------------------------------------
extern "C" __global__ __launch_bounds__(32) void t_lds_small(void *out, int out_bytes) {
    __shared__ __attribute__((aligned(16))) unsigned char sm[384];
    const int      l = lane_id();
    const uint32_t a = lds_addr(sm);
    const rsrc_t   ro = make_rsrc(out, out_bytes);
    for (int o = 16 * l; o < 384; o += 512) ds_store_b128(lds_ptr(a), o, u32x4{0xF1F1F1F1u, 0xF1F1F1F1u, 0xF1F1F1F1u, 0xF1F1F1F1u});
    lds_barrier();
    ds_store_b64(lds_ptr(a), 8 * (31 - l), u32x2{0xA0000000u + l, 0xB0000000u + l});
    ds_store_b32(lds_ptr(a), 256 + 4 * ((l * 7) % 32), 0xC0000000u + l);
    lds_barrier();
    for (int o = 16 * l; o < 384; o += 512) buf_store_b128(ro, o, 0, ds_load_b128_u32(lds_ptr(a), o));
}

// ---- t_sig_tdm: 4 waves. src = [64 rows][64] bf16 (8 KB); wave w TDM-loads rows 16w.. (tile A -> LDS 2048 w) and
// rows 16 ((w + 1) % 4).. (tile B -> 8192 + 2048 w), stores u32x4 {w, l, 7, 9} at 16384 + 512 w + 16 l (protocol
// store), loads 2 b128 of the constant area 18432.. (uncovered), split_signal_tdm_dscnt<1, 2>, wait; then reads
// tile A and the stores of wave (w + 1) % 4. out: [w][l] 3 x 16 B (A row l, other wave's store, uncovered sum) ----
extern "C" __global__ __launch_bounds__(128) void t_sig_tdm(const void *src, int src_bytes, void *out, int out_bytes) {
    __shared__ __attribute__((aligned(1024))) unsigned char sm[20480];
    using T = tdm_tile<16, 64, 128>;
    const int      l = lane_id(), w = wave_id(), t = (int) __builtin_amdgcn_workitem_id_x();
    const uint32_t a = lds_addr(sm);
    const rsrc_t   rs = make_rsrc(src, src_bytes), ro = make_rsrc(out, out_bytes);
    (void) rs;
    // constant area (read uncovered later): 2048 B of 0x01010101 * k
    for (int o = 16 * t; o < 2048; o += 16 * 128) {
        const uint32_t v = 0x01010101u * (uint32_t) (o / 16 + 1);
        ds_store_b128(lds_ptr(a), 18432 + o, u32x4{v, v, v, v});
    }
    lds_barrier();
    const u32x8    g1 = tdm_g1<T>(64, 16);
    const uint64_t g = (uint64_t) (uintptr_t) src;
    tdm_load(tdm_g0(a + 2048 * w, g + (uint64_t) (16 * w) * 128), g1);
    tdm_load(tdm_g0(a + 8192 + 2048 * w, g + (uint64_t) (16 * ((w + 1) % 4)) * 128), g1);
    ds_store_b128(lds_ptr(a), 16384 + 512 * w + 16 * l, u32x4{(uint32_t) w, (uint32_t) l, 7u, 9u});
    order_fence();
    const u32x4 u0 = ds_load_b128_u32(lds_ptr(a), 18432 + 16 * l), u1 = ds_load_b128_u32(lds_ptr(a), 18432 + 1024 + 16 * l);
    split_signal_tdm_dscnt<1, 2>();
    split_wait_lds();
    const int   o = (w + 1) % 4;
    const u32x4 ra = ds_load_b128_u32(lds_ptr(a), 2048 * o + 128 * (l % 16) + 16 * (l / 16));
    const u32x4 rx = ds_load_b128_u32(lds_ptr(a), 16384 + 512 * o + 16 * l);
    buf_store_b128(ro, (w * 32 + l) * 48, 0, ra);
    buf_store_b128(ro, (w * 32 + l) * 48 + 16, 0, rx);
    buf_store_b128(ro, (w * 32 + l) * 48 + 32, 0, u0 + u1);
    split_signal_tdm_lds<0>();
    split_wait_lds();
}
