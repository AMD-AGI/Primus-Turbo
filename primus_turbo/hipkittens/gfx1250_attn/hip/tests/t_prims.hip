/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// t_prims.hip -- bit-exact layout tests of hk_attn/*.h: one extern "C" kernel per primitive group,
// one workgroup of 1..4 waves, deterministic inputs, raw results written through buffer stores.
// The host driver (hk/lab/test_prims.py) builds every expectation on the CPU from layout.h's maps
// and compares bytes. Kernels take only pointers and ints; every global access is a buffer access
// whose extent is the byte count the host passes (or a sub-range of it); TDM extents are the tile.
//
//   t_ld_operands  operand loaders (rows / tr16) at pitch 272 and 80, tr16 at 288: raw registers;
//                  C^T LDS store
//   t_wmma         WMMA 16x16x32 on every loader pairing at both pitches, reuse-hint runs in a fenced
//                  order (one adjacent, one interleaved)
//   t_chain        fwd S^T -> pack P^T -> O^T and bwd S -> pack P -> dV^T, dS^T LDS exchange -> dQ
//   t_mask         mask_col_gt_row / mask_row_gt_col over 48 diagonal offsets
//   t_reduce       lane / column / row max and sum, permlanex16 and DPP row_xmask maps
//   t_exp          exp2_hw, softmax_arg (v_fma_f32) vs exp2_tile (v_pk_fma_f32), cvt_pk_bf16 RNE
//   t_gmem         buffer b32/b64/b128 loads and stores at true extents (OOB load 0, store dropped),
//                  soffset and straddling-access range-check probes, rows8 / cols16 fp32 vectors, C^T
//                  bf16 row store
//   t_atomic       4 waves of dQ-style C-fragment fp32 atomics (private and contended lines, OOB)
//   t_tdm          TDM of padded 2-D tiles from a [rows, H*D] tensor, OOB-row zero fill, unpadded
//                  tile; split_signal_tdm_lds
//   t_sync         4-wave split-barrier LDS exchange with fp32 atomics in flight, lock_simd, setprio
//   t_tdm_null     NULL TDM descriptor (pred 0) over a readable tile; the driver runs it last
// -DT_PRIMS_MUTANT_NO_DSCNT drops t_sync's dscnt wait: test_prims.py --build compiles that negative
// control into a temporary file and requires its structural ISA check to flag it.
#include "hk_attn/prims.h"

using namespace hk_attn;

namespace {

// copy a dense row-major [ROWS][COLS] 16-bit matrix at byte src_off of r into an LDS tile of pitch
// PITCH (16-B chunks, lanes tid, tid + nthr, ...)
template <int ROWS, int COLS, int PITCH>
HKA_DEV void stage(rsrc_t r, int src_off, uint32_t tile, int tid, int nthr) {
    constexpr int CPR = COLS / 8;
    for (int c = tid; c < ROWS * CPR; c += nthr) {
        const int row = c / CPR, cc = c % CPR;
        ds_store_b128(lds_ptr(tile), row * PITCH + cc * 16,
                      buf_load_b128(r, src_off + (row * COLS + cc * 8) * 2, 0));
    }
}
HKA_DEV void fill_lds(uint32_t a, int nbytes, uint32_t v, int tid, int nthr) {
    for (int o = 16 * tid; o < nbytes; o += 16 * nthr) ds_store_b128(lds_ptr(a), o, u32x4{v, v, v, v});
}
HKA_DEV void dump_lds(rsrc_t r, int dst_off, uint32_t a, int nbytes, int tid, int nthr) {
    for (int o = 16 * tid; o < nbytes; o += 16 * nthr)
        buf_store_b128(r, dst_off + o, 0, ds_load_b128_u32(lds_ptr(a), o));
}
// raw register images: 32 B per lane at off + 32 * lane
HKA_DEV void put_acc(rsrc_t r, int off, acc_t c, int lane) {
    buf_store_b128(r, off + 32 * lane, 0, lo_u32(c));
    buf_store_b128(r, off + 32 * lane + 16, 0, hi_u32(c));
}
HKA_DEV void put_opnd(rsrc_t r, int off, opnd_t v, int lane) {
    const u32x8 w = __builtin_bit_cast(u32x8, v);
    buf_store_b128(r, off + 32 * lane, 0, __builtin_shufflevector(w, w, 0, 1, 2, 3));
    buf_store_b128(r, off + 32 * lane + 16, 0, __builtin_shufflevector(w, w, 4, 5, 6, 7));
}
HKA_DEV acc_t load_acc(rsrc_t r, int off, int lane) {
    const f32x4 a = buf_load_f32x4(r, off + 32 * lane, 0), b = buf_load_f32x4(r, off + 32 * lane + 16, 0);
    return __builtin_shufflevector(a, b, 0, 1, 2, 3, 4, 5, 6, 7);
}

constexpr int X272 = 272, X80 = 80, X288 = 288;

} // namespace

// ---- t_ld_operands: src = X272 [32][128] u16 (8192 B) | X80 [32][32] u16 (2048 B)
// out = o0 rows272(r0 16, k0 64) | o1 tr272(k0 0, r0 48) | o2 rows80(r0 16, k0 0) | o3 tr80(k0 0, r0 16)
//       (1024 B each: 32 lanes x 16 u16) | C^T image [16][80 B] of c[i] = lane * 8 + i stored at
//       (n0 0, m0 16) into a zeroed tile (1280 B) | o4 tr288(k0 0, r0 104) of X272 staged at pitch
//       288 (1024 B)                                                                -> 6400 B
extern "C" __global__ __launch_bounds__(32) void t_ld_operands(const void *src, int src_bytes,
                                                               void *out, int out_bytes) {
    constexpr int T272 = 0, T80 = 32 * X272, TCT = T80 + 32 * X80, T288 = TCT + 16 * X80;
    constexpr int NLDS = T288 + 32 * X288;
    __shared__ __attribute__((aligned(1024))) unsigned char sm[NLDS];
    const int      lane = lane_id();
    const uint32_t l0   = lds_addr(sm);
    const rsrc_t   rs = make_rsrc(src, src_bytes), ro = make_rsrc(out, out_bytes);
    stage<32, 128, X272>(rs, 0, l0 + T272, lane, 32);
    stage<32, 32, X80>(rs, 8192, l0 + T80, lane, 32);
    stage<32, 128, X288>(rs, 0, l0 + T288, lane, 32);
    fill_lds(l0 + TCT, 16 * X80, 0u, lane, 32);
    lds_barrier();
    const lds_u8 *r272 = rows_base<X272>(l0 + T272, lane), *t272 = tr_base<X272>(l0 + T272, lane);
    const lds_u8 *r80 = rows_base<X80>(l0 + T80, lane), *t80 = tr_base<X80>(l0 + T80, lane);
    const opnd_t  o0 = load_rows<X272>(r272, 16, 64);
    const opnd_t  o1 = load_tr<X272>(t272, 0, 48);
    const opnd_t  o2 = load_rows<X80>(r80, 16, 0);
    const opnd_t  o3 = load_tr<X80>(t80, 0, 16);
    const opnd_t  o4 = load_tr<X288>(tr_base<X288>(l0 + T288, lane), 0, 104);
    acc_t         c;
    static_for<8>([&](auto i) { c[(int) i] = (float) (lane * 8 + (int) i); });
    store_ct<X80>(rows_base<X80>(l0 + TCT, lane), 0, 16, c);
    put_opnd(ro, 0, o0, lane);
    put_opnd(ro, 1024, o1, lane);
    put_opnd(ro, 2048, o2, lane);
    put_opnd(ro, 3072, o3, lane);
    put_opnd(ro, 5376, o4, lane);
    lds_barrier();
    dump_lds(ro, 4096, l0 + TCT, 16 * X80, lane, 32);
}

// ---- t_wmma: src = P [32][128] | Q [32][128] | R [32][32] | S [32][32] bf16 (20480 B)
// out = 6 raw accumulators (1024 B each):
//   c0 = rows272(P, 0, 32) x rows272(Q, 16, 32)        c1 = tr272(P, 0, 16) x tr272(Q, 0, 112)
//   c2 = rows80(R, 16, 0) x tr80(S, 0, 16)             c3 = tr80(R, 0, 0) x rows80(S, 0, 0)
//   c4 = sum_dt rows272(P, 16, 32 dt) x rows272(Q, 0, 0)    (B fixed: reuse_b from dt 1)
//   c5 = sum_t  tr272(P, 0, 0) x tr272(Q, 0, 16 t)          (A fixed: reuse_a from t 1)
// Every operand is loaded first; then the WMMAs issue in a fenced order (sched_barrier between any
// two): c4's run adjacent (each matrix_b_reuse WMMA follows one with the same B registers), then c5
// interleaved with c0..c3 (each matrix_a_reuse WMMA follows one with a different A).
extern "C" __global__ __launch_bounds__(32) void t_wmma(const void *src, int src_bytes, void *out,
                                                        int out_bytes) {
    constexpr int TP = 0, TQ = 32 * X272, TR = 2 * 32 * X272, TS = TR + 32 * X80, NLDS = TS + 32 * X80;
    __shared__ __attribute__((aligned(1024))) unsigned char sm[NLDS];
    const int      lane = lane_id();
    const uint32_t l0   = lds_addr(sm);
    const rsrc_t   rs = make_rsrc(src, src_bytes), ro = make_rsrc(out, out_bytes);
    stage<32, 128, X272>(rs, 0, l0 + TP, lane, 32);
    stage<32, 128, X272>(rs, 8192, l0 + TQ, lane, 32);
    stage<32, 32, X80>(rs, 16384, l0 + TR, lane, 32);
    stage<32, 32, X80>(rs, 18432, l0 + TS, lane, 32);
    lds_barrier();
    const lds_u8 *rp = rows_base<X272>(l0 + TP, lane), *tp = tr_base<X272>(l0 + TP, lane);
    const lds_u8 *rq = rows_base<X272>(l0 + TQ, lane), *tq = tr_base<X272>(l0 + TQ, lane);
    const lds_u8 *rr = rows_base<X80>(l0 + TR, lane), *tr = tr_base<X80>(l0 + TR, lane);
    const lds_u8 *rs_ = rows_base<X80>(l0 + TS, lane), *ts = tr_base<X80>(l0 + TS, lane);
    const opnd_t  a0 = load_rows<X272>(rp, 0, 32), b0 = load_rows<X272>(rq, 16, 32);
    const opnd_t  a1 = load_tr<X272>(tp, 0, 16), b1 = load_tr<X272>(tq, 0, 112);
    const opnd_t  a2 = load_rows<X80>(rr, 16, 0), b2 = load_tr<X80>(ts, 0, 16);
    const opnd_t  a3 = load_tr<X80>(tr, 0, 0), b3 = load_rows<X80>(rs_, 0, 0);
    const opnd_t  bq = load_rows<X272>(rq, 0, 0), ap = load_tr<X272>(tp, 0, 0);
    opnd_t        a4[4], b5[4];
    static_for<4>([&](auto j) {
        a4[j] = load_rows<X272>(rp, 16, 32 * j);
        b5[j] = load_tr<X272>(tq, 0, 16 * j);
    });
    sched_barrier<0>();
    acc_t c4 = wmma0(a4[0], bq);
    sched_barrier<0>();
    c4 = wmma<false, true>(a4[1], bq, c4);
    sched_barrier<0>();
    c4 = wmma<false, true>(a4[2], bq, c4);
    sched_barrier<0>();
    c4 = wmma<false, true>(a4[3], bq, c4);
    sched_barrier<0>();
    acc_t c5 = wmma0(ap, b5[0]);
    sched_barrier<0>();
    const acc_t c0 = wmma0(a0, b0);
    sched_barrier<0>();
    c5 = wmma<true, false>(ap, b5[1], c5);
    sched_barrier<0>();
    const acc_t c1 = wmma0(a1, b1);
    sched_barrier<0>();
    c5 = wmma<true, false>(ap, b5[2], c5);
    sched_barrier<0>();
    const acc_t c2 = wmma0(a2, b2);
    sched_barrier<0>();
    c5 = wmma<true, false>(ap, b5[3], c5);
    sched_barrier<0>();
    const acc_t c3 = wmma0(a3, b3);
    sched_barrier<0>();
    put_acc(ro, 0, c0, lane);
    put_acc(ro, 1024, c1, lane);
    put_acc(ro, 2048, c2, lane);
    put_acc(ro, 3072, c3, lane);
    put_acc(ro, 4096, c4, lane);
    put_acc(ro, 5120, c5, lane);
}

// ---- t_chain: src (bf16, 32 columns each) = K [32] | Q [16] | V [32] | Q2 [32] | K2 [32] | dO [32] rows
// (11264 B); every tile at pitch 80.
// out (raw): fwd  st0, st1 (S^T kv 0..15 / 16..31, acc) | pt (P^T operand) | o0, o1 (O^T d 0..15 / 16..31)
//            bwd  s00, s10 (S q 0..15 / 16..31 x kv 0..15) | p0 (P operand) | dv0, dv1 (dV^T) |
//                 dq0 (dQ q 0..15 x d 0..15), dq1 (q 16..31 x d 16..31) | dS^T image [32][80 B]
//            -> 5 x 1024 + 7 x 1024 + 2560 = 14848 B
extern "C" __global__ __launch_bounds__(32) void t_chain(const void *src, int src_bytes, void *out,
                                                         int out_bytes) {
    constexpr int TK = 0, TQ = TK + 32 * X80, TV = TQ + 16 * X80, TQ2 = TV + 32 * X80;
    constexpr int TK2 = TQ2 + 32 * X80, TDO = TK2 + 32 * X80, TDS = TDO + 32 * X80, NLDS = TDS + 32 * X80;
    __shared__ __attribute__((aligned(1024))) unsigned char sm[NLDS];
    const int      lane = lane_id();
    const uint32_t l0   = lds_addr(sm);
    const rsrc_t   rs = make_rsrc(src, src_bytes), ro = make_rsrc(out, out_bytes);
    stage<32, 32, X80>(rs, 0, l0 + TK, lane, 32);
    stage<16, 32, X80>(rs, 32 * 64, l0 + TQ, lane, 32);
    stage<32, 32, X80>(rs, 48 * 64, l0 + TV, lane, 32);
    stage<32, 32, X80>(rs, 80 * 64, l0 + TQ2, lane, 32);
    stage<32, 32, X80>(rs, 112 * 64, l0 + TK2, lane, 32);
    stage<32, 32, X80>(rs, 144 * 64, l0 + TDO, lane, 32);
    fill_lds(l0 + TDS, 32 * X80, 0u, lane, 32);
    lds_barrier();
    // fwd: S^T = K Q^T (A = K rows, B = Q rows), P^T = pack(S^T kv 0..15, 16..31), O^T = V^T P^T
    const opnd_t bq  = load_rows<X80>(rows_base<X80>(l0 + TQ, lane), 0, 0);
    const lds_u8 *rk = rows_base<X80>(l0 + TK, lane), *tv = tr_base<X80>(l0 + TV, lane);
    const acc_t  st0 = wmma0(load_rows<X80>(rk, 0, 0), bq);
    const acc_t  st1 = wmma0(load_rows<X80>(rk, 16, 0), bq);
    const opnd_t pt  = pack_pt_fwd(st0, st1);
    const acc_t  o0  = wmma0(load_tr<X80>(tv, 0, 0), pt);
    const acc_t  o1  = wmma0(load_tr<X80>(tv, 0, 16), pt);
    // bwd: S = Q2 K2^T (A = Q2 rows, B = K2 rows), P = pack(S q 0..15, 16..31), dV^T = dO^T P
    const lds_u8 *rq2 = rows_base<X80>(l0 + TQ2, lane), *rk2 = rows_base<X80>(l0 + TK2, lane);
    const opnd_t  kb0 = load_rows<X80>(rk2, 0, 0), kb1 = load_rows<X80>(rk2, 16, 0);
    const opnd_t  qa0 = load_rows<X80>(rq2, 0, 0), qa1 = load_rows<X80>(rq2, 16, 0);
    const acc_t   s00 = wmma0(qa0, kb0), s10 = wmma0(qa1, kb0);
    const acc_t   s01 = wmma0(qa0, kb1), s11 = wmma0(qa1, kb1);
    const opnd_t  p0  = pack_p_bwd(s00, s10);
    const lds_u8 *tdo = tr_base<X80>(l0 + TDO, lane);
    const acc_t   dv0 = wmma0(load_tr<X80>(tdo, 0, 0), p0);
    const acc_t   dv1 = wmma0(load_tr<X80>(tdo, 0, 16), p0);
    // dS^T [kv][q] through LDS (C^T stores), dQ = dS K2 (A = tr16 of dS^T, B = tr16 of K2)
    lds_u8 *wds = rows_base<X80>(l0 + TDS, lane);
    store_ct<X80>(wds, 0, 0, s00);
    store_ct<X80>(wds, 0, 16, s10);
    store_ct<X80>(wds, 16, 0, s01);
    store_ct<X80>(wds, 16, 16, s11);
    lds_barrier();
    const lds_u8 *tds = tr_base<X80>(l0 + TDS, lane), *tk2 = tr_base<X80>(l0 + TK2, lane);
    const acc_t   dq0 = wmma0(load_tr<X80>(tds, 0, 0), load_tr<X80>(tk2, 0, 0));
    const acc_t   dq1 = wmma0(load_tr<X80>(tds, 0, 16), load_tr<X80>(tk2, 0, 16));
    put_acc(ro, 0, st0, lane);
    put_acc(ro, 1024, st1, lane);
    put_opnd(ro, 2048, pt, lane);
    put_acc(ro, 3072, o0, lane);
    put_acc(ro, 4096, o1, lane);
    put_acc(ro, 5120, s00, lane);
    put_acc(ro, 6144, s10, lane);
    put_opnd(ro, 7168, p0, lane);
    put_acc(ro, 8192, dv0, lane);
    put_acc(ro, 9216, dv1, lane);
    put_acc(ro, 10240, dq0, lane);
    put_acc(ro, 11264, dq1, lane);
    dump_lds(ro, 12288, l0 + TDS, 32 * X80, lane, 32);
}

// ---- t_mask: c[i] = lane * 8 + i + 1; for j in [0, 48): off = off0 + j (uniform)
// out[(2j + 0) * 256 + 8 lane + i] = mask_col_gt_row(c, off), out[(2j + 1) * 256 + ...] =
// mask_row_gt_col(c, off) (fp32, masked = -inf)                                       -> 98304 B
extern "C" __global__ __launch_bounds__(32) void t_mask(void *out, int out_bytes, int off0) {
    constexpr int NOFF = 48;
    const int     lane = lane_id();
    const rsrc_t  ro   = make_rsrc(out, out_bytes);
    acc_t         c;
    static_for<8>([&](auto i) { c[(int) i] = (float) (lane * 8 + (int) i + 1); });
    const int off = uniform(off0);
    static_for<NOFF>([&](auto j) {
        put_acc(ro, (2 * j + 0) * 1024, mask_col_gt_row(c, lane, off + (int) j), lane);
        put_acc(ro, (2 * j + 1) * 1024, mask_row_gt_col(c, lane, off + (int) j), lane);
    });
}

// ---- t_reduce: src = X [4 tiles][32 lanes][8] fp32 (4096 B)
// out = lane_max | col_max | lane_sum | col_sum | xhalf(X[0][l][0]) (128 B each) | row_max(X[0]) |
//       row_sum(X[1]) (1024 B each) | xor16<1, 2, 4, 8>(X[0][l][0]) (128 B each)        -> 3200 B
extern "C" __global__ __launch_bounds__(32) void t_reduce(const void *src, int src_bytes, void *out,
                                                          int out_bytes) {
    const int    lane = lane_id();
    const rsrc_t rs = make_rsrc(src, src_bytes), ro = make_rsrc(out, out_bytes);
    acc_t        x[4];
    static_for<4>([&](auto t) { x[t] = load_acc(rs, t * 1024, lane); });
    buf_store_f32(ro, 4 * lane, 0, lane_max(x));
    buf_store_f32(ro, 128 + 4 * lane, 0, col_max(x));
    buf_store_f32(ro, 256 + 4 * lane, 0, lane_sum(x));
    buf_store_f32(ro, 384 + 4 * lane, 0, col_sum(x));
    buf_store_f32(ro, 512 + 4 * lane, 0, xhalf(x[0][0]));
    put_acc(ro, 640, row_max(x[0]), lane);
    put_acc(ro, 1664, row_sum(x[1]), lane);
    buf_store_f32(ro, 2688 + 4 * lane, 0, xor16<1>(x[0][0]));
    buf_store_f32(ro, 2816 + 4 * lane, 0, xor16<2>(x[0][0]));
    buf_store_f32(ro, 2944 + 4 * lane, 0, xor16<4>(x[0][0]));
    buf_store_f32(ro, 3072 + 4 * lane, 0, xor16<8>(x[0][0]));
}

// ---- t_exp: src = S [32][8] | M [32] | E [32][8] | V [32][8] fp32 (3200 B); c = bits(c_bits)
// out = exp2_hw(softmax_arg(S, c, neg_scaled(M, c))) | exp2_tile(S, c, neg_scaled(M, c)) |
//       exp2_hw(E) (1024 B each) | cvt_pk_bf16(V) (512 B: 32 lanes x 4 dwords)        -> 3584 B
extern "C" __global__ __launch_bounds__(32) void t_exp(const void *src, int src_bytes, void *out,
                                                       int out_bytes, int c_bits) {
    const int    lane = lane_id();
    const rsrc_t rs = make_rsrc(src, src_bytes), ro = make_rsrc(out, out_bytes);
    const float  c  = __builtin_bit_cast(float, uniform(c_bits));
    const acc_t  s  = load_acc(rs, 0, lane);
    const float  nm = neg_scaled(buf_load_f32(rs, 1024 + 4 * lane, 0), c);
    const acc_t  e  = load_acc(rs, 1152, lane);
    const acc_t  v  = load_acc(rs, 2176, lane);
    acc_t        p1, pe;
    static_for<8>([&](auto i) {
        p1[(int) i] = exp2_hw(softmax_arg(s[(int) i], c, nm));
        pe[(int) i] = exp2_hw(e[(int) i]);
    });
    put_acc(ro, 0, p1, lane);
    put_acc(ro, 1024, exp2_tile(s, c, nm), lane);
    put_acc(ro, 2048, pe, lane);
    buf_store_b128(ro, 3072 + 16 * lane, 0, cvt_pk_bf16(v));
}

// ---- t_gmem: src (allocation src_alloc >= 2048 B) seen through extent src_bytes; dst (allocation
// dst_bytes >= 4096 B, prefilled) through four sub-range resources: [0, 400) b128, [1024, 1224)
// b64, [2048, 2148) b32, [3072, 3112) b128 (straddle probe).
//   a32  = buf_load_b32 (src, 896 + 4l)        a64  = buf_load_b64 (src, 768 + 8l)
//   a128 = buf_load_b128(src, 512 + 16l)       a32s = buf_load_b32 (src, voff 4l, soff soff_probe)
//   dst: b128 at 16l = {0x10000000 + 4l + j}, b64 at 8l = {0x20000000 + 2l + j}, b32 at 4l = 0x30000000 + l
//   rows8 = load_rows8_f32(src through src_alloc, voff_row0 = 32) (row0 8)
//   cols16: lanes 0..15 store 100 + r, lanes 16..31 hold 200 + l (must not be stored)
//   ct: c[i] = lane * 8 + i - 128 as bf16 rows of a [16][32] tile (pitch 64 B), columns 16 + 8h + i
//   a128p = buf_load_b128(src through src_bytes - 8, 512 + 16l): lane 30 straddles the extent
//   dst straddle: b128 at 3072 + 16l = {0x40000000 + 4l + j} through extent 40: lane 2 straddles
// out (prefilled by the host) = a32 128 | a64 256 | a128 512 | a32s 128 | rows8 1024 | cols16 128 |
//                               ct 1024 | a128p 512                                    -> 3712 B
extern "C" __global__ __launch_bounds__(32) void t_gmem(const void *src, int src_bytes,
                                                        int src_alloc, void *dst, int dst_bytes,
                                                        void *out, int out_bytes, int soff_probe) {
    const int    lane = lane_id();
    const rsrc_t rs = make_rsrc(src, src_bytes), ro = make_rsrc(out, out_bytes);
    const rsrc_t rsf = make_rsrc(src, src_alloc);
    // three sub-ranges of dst (each clamped to the allocation)
    const unsigned char *d8 = (const unsigned char *) dst;
    const rsrc_t r128 = make_rsrc(d8, dst_bytes < 400 ? dst_bytes : 400);
    const rsrc_t r64  = make_rsrc(d8 + 1024, dst_bytes < 1224 ? 0 : 200);
    const rsrc_t r32  = make_rsrc(d8 + 2048, dst_bytes < 2148 ? 0 : 100);
    const rsrc_t rp   = make_rsrc(d8 + 3072, dst_bytes < 3584 ? 0 : 40);
    const rsrc_t rsp  = make_rsrc(src, src_bytes < 8 ? 0 : src_bytes - 8);
    wait_kmcnt<0>();
    const uint32_t a32  = buf_load_b32(rs, 896 + 4 * lane, 0);
    const u32x2    a64  = buf_load_b64(rs, 768 + 8 * lane, 0);
    const u32x4    a128 = buf_load_b128(rs, 512 + 16 * lane, 0);
    const uint32_t a32s = buf_load_b32(rs, 4 * lane, uniform(soff_probe));
    const acc_t    r8   = load_rows8_f32(rsf, 32, 0, lane);
    const u32x4    a128p = buf_load_b128(rsp, 512 + 16 * lane, 0);
    wait_loadcnt<0>();
    buf_store_b32(ro, 4 * lane, 0, a32);
    buf_store_b64(ro, 128 + 8 * lane, 0, a64);
    buf_store_b128(ro, 384 + 16 * lane, 0, a128);
    buf_store_b32(ro, 896 + 4 * lane, 0, a32s);
    put_acc(ro, 1024, r8, lane);
    store_cols16_f32(ro, 2048, 0, (float) (lane < 16 ? 100 + lane : 200 + lane), lane);
    acc_t c;
    static_for<8>([&](auto i) { c[(int) i] = (float) (lane * 8 + (int) i - 128); });
    buf_store_ct_bf16(ro, 2176 + lay::ct_lane(lane, 64) + 2 * 16, 0, c);
    buf_store_b128(ro, 3200 + 16 * lane, 0, a128p);
    const uint32_t l4 = 4u * (uint32_t) lane;
    buf_store_b128(r128, 16 * lane, 0, u32x4{0x10000000u + l4, 0x10000001u + l4, 0x10000002u + l4,
                                            0x10000003u + l4});
    buf_store_b64(r64, 8 * lane, 0, u32x2{0x20000000u + 2u * lane, 0x20000001u + 2u * lane});
    buf_store_b32(r32, 4 * lane, 0, 0x30000000u + (uint32_t) lane);
    buf_store_b128(rp, 16 * lane, 0, u32x4{0x40000000u + l4, 0x40000001u + l4, 0x40000002u + l4,
                                          0x40000003u + l4});
    wait_storecnt<0>();
}

// ---- t_atomic (4 waves): acc (allocation >= acc_bytes + 512, prefilled 0.5) seen through extent
// acc_bytes (= 8192: two 4-KB regions). Iteration it < nrep targets region b = it % 2 (a per-step
// base in voffset, as a kernel's per-step dQ tile: the 8 element offsets then fold into the
// instruction offset): { private: element i of wave w lane l (value w * 256 + i * 32 + l) -> byte
// 4096 b + 512 i + 128 w + 4 l; contended: value i + 1 -> byte 4096 b + 512 i + 4 l (all waves) };
// then one out-of-extent atomic per lane at byte acc_bytes + 4 l (dropped).
extern "C" __global__ __launch_bounds__(128) void t_atomic(void *acc, int acc_bytes, int nrep) {
    const int    lane = lane_id(), wv = wave_id();
    const rsrc_t ra   = make_rsrc(acc, acc_bytes);
    acc_t        c, c2;
    static_for<8>([&](auto i) {
        c[(int) i]  = (float) (wv * 256 + (int) i * 32 + lane);
        c2[(int) i] = (float) ((int) i + 1);
    });
    const int n = uniform(nrep);
    for (int it = 0; it < n; ++it) {
        const int vb = lay::cfrag_lane(lane) + 4096 * (it & 1);
        atomic_add_c<512>(ra, vb + 128 * wv, 0, c);
        atomic_add_c<512>(ra, vb, 0, c2);
    }
    buf_atomic_add_f32(ra, uniform(acc_bytes) + 4 * lane, 0, 1.0f);
}

// ---- t_tdm (2 waves): src = X [nrows][512] u16 (H 4 x D 128); LDS prefilled 0xABCDABCD, then
//   wave 0: A <- rows [8, 24) cols [256, 384)  tile 16 x 128 pitch 272 at 0
//   wave 1: B <- rows [20, 36) cols [192, 224) tile 16 x 32 pitch 80 at 4352
//   wave 0: C <- rows [32, 48) cols [384, 512) tile 16 x 128 pitch 272 at 5632, rows_valid nrows - 32
//   D (1280 B at 9984): written by no op (must keep the LDS fill; the NULL probe is t_tdm_null)
//   wave 1: E <- rows [0, 8) cols [448, 512) tile 8 x 64 pitch 128 (no pad) at 11264
// out = the whole 12288-B LDS image after split_signal_tdm_lds<0> / split_wait_lds.
extern "C" __global__ __launch_bounds__(64) void t_tdm(const void *src, int nrows, void *out,
                                                       int out_bytes) {
    using TA = tdm_tile<16, 128, X272>;
    using TB = tdm_tile<16, 32, X80>;
    using TE = tdm_tile<8, 64, 128>;
    constexpr int A = 0, B = A + TA::bytes, C = B + TB::bytes, D = C + TA::bytes, E = D + TB::bytes;
    static_assert(D == 9984, "t_tdm LDS map: D is the untouched region");
    constexpr int NLDS = E + TE::bytes;
    static_assert(NLDS == 12288 && A % 128 == 0 && B % 128 == 0 && C % 128 == 0 && D % 128 == 0 &&
                      E % 128 == 0,
                  "t_tdm LDS map");
    __shared__ __attribute__((aligned(1024))) unsigned char sm[NLDS];
    const int      tid = (int) __builtin_amdgcn_workitem_id_x(), wv = wave_id();
    const uint32_t l0  = lds_addr(sm);
    const rsrc_t   ro  = make_rsrc(out, out_bytes);
    const int      nr  = uniform(nrows);
    constexpr int  ROW_EL = 4 * 128;
    fill_lds(l0, NLDS, 0xABCDABCDu, tid, 64);
    lds_barrier();
    if (wv == 0) {
        tdm_load(tdm_g0(l0 + A, tdm_gaddr(src, 8, ROW_EL, 2 * 128)), tdm_g1<TA>(ROW_EL, nr - 8));
        tdm_load(tdm_g0(l0 + C, tdm_gaddr(src, 32, ROW_EL, 3 * 128)), tdm_g1<TA>(ROW_EL, nr - 32));
    } else {
        tdm_load(tdm_g0(l0 + B, tdm_gaddr(src, 20, ROW_EL, 1 * 128 + 64)), tdm_g1<TB>(ROW_EL, nr - 20));
        tdm_load(tdm_g0(l0 + E, tdm_gaddr(src, 0, ROW_EL, 3 * 128 + 64)), tdm_g1<TE>(ROW_EL, nr));
    }
    split_signal_tdm_lds<0>();
    split_wait_lds();
    dump_lds(ro, 0, l0, NLDS, tid, 64);
}

// ---- t_sync (4 waves): NIT iterations; iteration t writes {t, w, l, 0xC0DE} into LDS slot
// [t % 2][w], adds 1.0 to cnt[l] (fp32 atomic, left in flight), signals (even t: split_signal_lds;
// odd t: order_fence, then one younger ds_load_b128 the protocol does not cover -- this wave's own
// slot [(t + 1) % 2][w], written in iteration t - 1 behind fences, so LLVM cannot forward it, and
// consumed only after the loop, so it stays a b128 load in flight at the signal -- then
// split_signal_dscnt<1>), waits, reads slot [t % 2][(w + 1) % 4] -> out[((t * 4 + w) * 32 + l) * 16].
// out = NIT x 4 x 32 x 16 B | XOR of the uncovered loads per lane (16 B: {0 ^ 2 ^ 4, w, l, 0xC0DE})
// (2048 B); cnt[l] = 4 NIT. The data checks are a smoke test (a missing wait rarely shows as a race);
// the structural ISA check (test_prims isa_ds_signal) is the real one.
extern "C" __global__ __launch_bounds__(128) void t_sync(void *out, int out_bytes, void *cnt,
                                                         int cnt_bytes) {
    constexpr int NIT = 6, SLOT = 32 * 16;
    __shared__ __attribute__((aligned(1024))) unsigned char sm[2 * 4 * SLOT];
    lock_simd();
    set_prio<1>();
    const int      lane = lane_id(), wv = wave_id();
    const uint32_t l0   = lds_addr(sm);
    const rsrc_t   ro = make_rsrc(out, out_bytes), rc = make_rsrc(cnt, cnt_bytes);
    lds_u8        *p  = lds_ptr(l0 + 16 * lane);
    u32x4          unc[NIT / 2];
    static_for<NIT>([&](auto t) {
        constexpr int s = t % 2;
        ds_store_b128(p, (s * 4 + wv) * SLOT,
                      u32x4{(uint32_t) t, (uint32_t) wv, (uint32_t) lane, 0xC0DEu});
        buf_atomic_add_f32(rc, 4 * lane, 0, 1.0f);
        if constexpr (t % 2 == 0) {
            split_signal_lds();
        } else {
            order_fence(); // the protocol store stays older than the uncovered load
            unc[t / 2] = ds_load_b128_u32(p, ((1 - s) * 4 + wv) * SLOT);
#if defined(T_PRIMS_MUTANT_NO_DSCNT)
            sched_barrier<0>(); // negative control: signal with the protocol store still in flight
            barrier_signal();
            sched_barrier<0>();
#else
            split_signal_dscnt<1>();
#endif
        }
        split_wait_lds();
        const u32x4 v = ds_load_b128_u32(p, (s * 4 + (wv + 1) % 4) * SLOT);
        buf_store_b128(ro, ((t * 4 + wv) * 32 + lane) * 16, 0, v);
    });
    u32x4 sink = unc[0];
    static_for<NIT / 2 - 1>([&](auto j) { sink ^= unc[j + 1]; });
    buf_store_b128(ro, NIT * 4 * 32 * 16 + 16 * (32 * wv + lane), 0, sink);
}

// ---- t_tdm_null (1 wave; the driver runs it last): src = X [nrows][512] u16 as t_tdm; LDS prefilled
// 0xABCDABCD, then
//   N <- NULL descriptor (pred 0) whose address and extents describe the READABLE tile rows [0, 16)
//        cols [0, 32) (if pred were ignored those bytes would land: a FAIL, never a fault)
//                                                              tile 16 x 32 pitch 80 at 0
//   R <- rows [16, 32) cols [32, 64)                            tile 16 x 32 pitch 80 at 1280
// then split_signal_tdm_lds<0> (s_wait_tensorcnt 0) / split_wait_lds; out = the 2560-B LDS image.
extern "C" __global__ __launch_bounds__(32) void t_tdm_null(const void *src, int nrows, void *out,
                                                            int out_bytes) {
    using TB = tdm_tile<16, 32, X80>;
    constexpr int N = 0, R = TB::bytes, NLDS = 2 * TB::bytes;
    __shared__ __attribute__((aligned(1024))) unsigned char sm[NLDS];
    const int      lane = lane_id();
    const uint32_t l0   = lds_addr(sm);
    const rsrc_t   ro   = make_rsrc(out, out_bytes);
    const int      nr   = uniform(nrows);
    constexpr int  ROW_EL = 4 * 128;
    fill_lds(l0, NLDS, 0xABCDABCDu, lane, 32);
    lds_barrier(); // the fill's DS stores complete before any TDM write
    tdm_load(tdm_g0_null(l0 + N, tdm_gaddr(src, 0, ROW_EL, 0)), tdm_g1<TB>(ROW_EL, nr));
    tdm_load(tdm_g0(l0 + R, tdm_gaddr(src, 16, ROW_EL, 32)), tdm_g1<TB>(ROW_EL, nr - 16));
    split_signal_tdm_lds<0>();
    split_wait_lds();
    dump_lds(ro, 0, l0, NLDS, lane, 32);
}
