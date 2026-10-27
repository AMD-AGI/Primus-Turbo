/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// t_ext_fwdpp -- the GPU checks k_attn_fwd_pp relies on before its first launch (driver: hk/lab/test_ext_fwdpp.py).
//
//   t_pp_simd     block 256, the kernel's 320-KB static LDS (1 WG per CU): every wave stores its SIMD
//                 (hk_attn::simd_id, HW_ID1[9:8]) and the raw HW_ID1. Expected [0, 3, 2, 1, 0, 3, 2, 1]: waves w and
//                 w + 4 share a SIMD (the lo / hi pairing of the ping-pong).
//   t_pp_stagger  block 256, 320-KB LDS, the barrier skeleton of k_attn_fwd_pp for n stages, from the same pp::
//                 schedule functions: B_0 (full), the hi group's lag barrier(s), one barrier window per cluster
//                 j = 1 .. 4n+1 (signal, a 2-op ds_load_b128 burst, wait), the lo group's exit barrier. In every
//                 barrier interval i (after its wait of B_i) a wave reads its partner's (w ^ 4) marker of interval
//                 i - 1 and writes its own (the cluster it runs, -1 for none) into a parity-double-buffered LDS slot, so
//                 every access is ordered by a barrier. Recorded: the partner's marker and s_memrealtime per interval.
//                 Expected (PP_STAGGER 1): lo in interval i sees i - 2, hi sees i - 1 (hi one cluster behind);
//                 (PP_STAGGER 0): both see i - 1.
// No TDM, no global reads; every store goes through a TRUE-extent buffer resource.
#include "fwdpp/idx.h"
#include "hk_attn/ext_fwdpp.h"
#include "hk_attn/prims.h"

using namespace hk_attn;

__shared__ __attribute__((aligned(1024))) unsigned char t_smem[pp::LDS_BYTES];

constexpr int MARK_OFF = 4 * pp::SEG + 1024;   // marker[2][8] ints (free segment 4, past the c2 cells)
constexpr int DATA_OFF = 4 * pp::SEG + 2048;   // 8 x 1 KB per-wave read-only burst data

extern "C" __global__ __attribute__((amdgpu_flat_work_group_size(256, 256))) void t_pp_simd(int *out,
                                                                                           long long out_bytes) {
    const int w = wave_id(), lane = lane_id();
    const int wg = (int) __builtin_amdgcn_workgroup_id_x();
    // touch the LDS (keeps the 320-KB allocation, so 1 WG per CU as in the kernel)
    typedef __attribute__((address_space(3))) volatile int lds_vint_;
    lds_vint_ *cell = (lds_vint_ *) lds_ptr(lds_addr(t_smem) + 16 * w);
    *cell           = w;
    lds_barrier();
    const int    rd = *cell;
    const rsrc_t r  = make_rsrc(out, out_bytes);
    const int    id = (int) __builtin_amdgcn_s_getreg((31 << 11) | 23); // HW_ID1, all 32 bits
    if (lane == 0) {
        buf_store_b32(r, 4 * (3 * (8 * wg + w) + 0), 0, (uint32_t) simd_id());
        buf_store_b32(r, 4 * (3 * (8 * wg + w) + 1), 0, (uint32_t) id);
        buf_store_b32(r, 4 * (3 * (8 * wg + w) + 2), 0, (uint32_t) rd);
    }
}

typedef __attribute__((address_space(3))) volatile int lds_vint;

template <int GRP> __device__ __forceinline__ void stagger_run(int *out, long long out_bytes, unsigned *ts,
                                                             long long ts_bytes, unsigned *ck, long long ck_bytes,
                                                             int n) {
    const int    w = wave_id(), lane = lane_id(), wg = (int) __builtin_amdgcn_workgroup_id_x();
    const int    J = pp::n_clusters(n), NB = pp::n_barriers(n), x = pp::lag_of(GRP), partner = w ^ 4;
    const rsrc_t ro = make_rsrc(out, out_bytes), rt = make_rsrc(ts, ts_bytes), rc = make_rsrc(ck, ck_bytes);
    const uint32_t lds0 = lds_addr(t_smem);
    lds_vint      *mark = (lds_vint *) lds_ptr(lds0 + MARK_OFF);
    lds_u8        *data = lds_ptr(lds0 + (uint32_t) (DATA_OFF + 1024 * w + 16 * lane));
    u32x4          acc  = {0, 0, 0, 0};
    // interval i (after the wait of B_i): record the partner's marker of interval i - 1, write ours (slot i % 2)
    auto interval = [&](int i) {
        const int got = mark[8 * ((i + 1) & 1) + partner];
        const int own = i - x <= J - 1 ? i - x : -1;
        if (lane == 0) {
            buf_store_b32(ro, 4 * ((8 * wg + w) * NB + i), 0, (uint32_t) got);
            buf_store_b32(rt, 4 * ((8 * wg + w) * NB + i), 0, (uint32_t) __builtin_readcyclecounter());
        }
        mark[8 * (i & 1) + w] = own;
    };
    // prologue: markers of "interval -1" = -1, the wave's 1-KB burst data, B_0
    mark[8 + w] = -1;
    ds_store_b128(data, 0, u32x4{(uint32_t) w, (uint32_t) lane, 1u, 2u});
    ds_store_b128(data, 512, u32x4{3u, (uint32_t) lane, (uint32_t) w, 4u});
    lds_barrier(); // B_0
    interval(0);
    for (int b = 1; b <= x; ++b) { // the lag barrier(s) of the hi group
        lds_barrier();
        interval(b);
    }
    for (int j = 1; j < J; ++j) { // one window per cluster: signal, a 2-op burst, wait
        bar_window_open();
        acc += ds_load_b128_u32(data, 0) + ds_load_b128_u32(data, 512);
        bar_window_close();
        interval(pp::bar_before(GRP, j));
    }
    for (int k = 0; k < pp::n_exit_barriers(GRP); ++k) {
        bar_window_open();
        bar_window_close();
        interval(J + x + k);
    }
    // checksum of the window bursts: (J - 1) x (w + 3 + lane + lane + 1 + w + 2 + 4) summed over lanes by the host
    buf_store_b32(rc, 4 * (32 * (8 * wg + w) + lane), 0, acc[0] + acc[1] + acc[2] + acc[3]);
}

extern "C" __global__ __attribute__((amdgpu_flat_work_group_size(256, 256))) void t_pp_stagger(
    int *out, long long out_bytes, unsigned *ts, long long ts_bytes, unsigned *ck, long long ck_bytes, int n) {
    if (wave_id() < pp::NPAIR) stagger_run<0>(out, out_bytes, ts, ts_bytes, ck, ck_bytes, n);
    else stagger_run<1>(out, out_bytes, ts, ts_bytes, ck, ck_bytes, n);
}
