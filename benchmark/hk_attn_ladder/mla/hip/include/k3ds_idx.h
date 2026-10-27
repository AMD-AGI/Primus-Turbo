/***************************************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

// k3ds_idx.h -- the index / address formulas of k_dkdv96ds's dS^T export (bwd_hk_k3ds), as
// constexpr
// __host__ __device__ functions of scalars. The kernel calls exactly these; bounds_proof.py
// compiles this file for the host and replays them over every launched shape. The layout is
// dslayout.h v1.
//
// k1ds_idx.h (bwd_hk_k1ds, byte-identical here) indexes a wave by (64-row block bid, wave wv), i.e.
// kv16 rows kt0 = 4 bid + 2 wv and kt0 + 1. A k_dkdv96ds wave owns nkv kv16 rows starting at kt0 =
// kv0 / 16 (kv0 = k3_kv0(kv0g, wv, nkv)): nkv 3 in a 96-row block (kt0 = 6 bid + 3 wv, odd for wave
// 1), nkv 2 in a 64-row tail block (the k1 body). So the k3ds functions take (kt0, nkv) instead of
// (bid, wv); they are built from k1ds_idx.h's branch-free row arithmetic, and bounds_proof.py ties
// them back: at nkv 2 and kt0 = k1ds_kt0(bid, wv) each equals its k1ds function (the tail blocks
// export exactly as k_dkdv64ds would at block kv0g / 64), and DT1 / DT2 replay every wave of both
// bodies against dslayout.h itself.
//
// One wave-iteration (query pair qt, q head gh of the group) exports the blocks (bh0 + gh, kt0 +
// kh, qt), kh < nkv, each with two buffer_store_b128 (k1ds's DPP regroup: 512 contiguous bytes = 4
// whole 128-B lines per store). Store address = descriptor base (64-bit, per wave) + soffset (SGPR:
// gh * hstride + 1024 qt + rowc(kh)) + voffset (VGPR, per lane, loop-invariant: k1ds_voff) + 512 t.
// A block is stored only if qt >= qp_first(kt0 + kh).
#pragma once
#include "dslayout.h"
#include "k1ds_idx.h"
#include <stdint.h>
#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif
#define K3DS_FN static inline constexpr __host__ __device__

// the wave's first kv16 row (kv0 is a multiple of 16: k3_kv0 = kv0g + 16 nkv wv, kv0g a multiple of
// 32)
K3DS_FN int k3ds_kt0(int kv0) {
    return kv0 / DSL_KV;
}
// the wave's descriptor base, bytes from the scratch start (int64): region bh0 = bat*Hq + hkv*G,
// first block of row kt0
K3DS_FN int64_t k3ds_base(int bat, int Hq, int hkv, int G, int kt0, int Sq, int Skv, int causal) {
    return (k1ds_bh0(bat, Hq, hkv, G) * k1ds_nblk_bh(Sq, Skv, causal) +
            k1ds_row_off(kt0, Sq / DSL_Q, k1ds_m(Skv - Sq), causal)) *
           DSL_BLK_B;
}
// the wave's TRUE extent (int64 bytes): rows kt0 .. kt0 + nkv - 1 of regions bh0 .. bh0 + G - 1 end
// it
K3DS_FN int64_t k3ds_ext(int G, int kt0, int nkv, int Sq, int Skv, int causal) {
    const int nqp = Sq / DSL_Q, m = k1ds_m(Skv - Sq);
    return ((int64_t) (G - 1) * k1ds_nblk_bh(Sq, Skv, causal) +
            k1ds_row_off(kt0 + nkv, nqp, m, causal) - k1ds_row_off(kt0, nqp, m, causal)) *
           DSL_BLK_B;
}
// num_records of the descriptor: the true extent (ds_mode != 0) or 0 (ds_mode 0, route dsw0: every
// store dropped)
K3DS_FN int64_t k3ds_nrec(int ds_mode, int G, int kt0, int nkv, int Sq, int Skv, int causal) {
    return ds_mode != 0 ? k3ds_ext(G, kt0, nkv, Sq, Skv, causal) : 0;
}
// first stored q pair of kv sub-tile kh (row kt0 + kh)
K3DS_FN int k3ds_qpf(int kh, int kt0, int Sq, int Skv, int causal) {
    return k1ds_qpf_kt(kt0 + kh, k1ds_m(Skv - Sq), causal);
}
// is (kh, qt) stored? (uniform per wave and iteration)
K3DS_FN bool k3ds_put(int kh, int qt, int kt0, int Sq, int Skv, int causal) {
    return qt >= k3ds_qpf(kh, kt0, Sq, Skv, causal);
}
// row constant of kv sub-tile kh: bytes from the wave base to row kt0 + kh, minus its qp_first
// blocks (may be negative; rowc + 1024 qt >= 0 whenever the block is stored)
K3DS_FN int k3ds_rowc(int kh, int kt0, int Sq, int Skv, int causal) {
    const int nqp = Sq / DSL_Q, m = k1ds_m(Skv - Sq);
    return (int) ((k1ds_row_off(kt0 + kh, nqp, m, causal) - k1ds_row_off(kt0, nqp, m, causal) -
                   (int64_t) k3ds_qpf(kh, kt0, Sq, Skv, causal)) *
                  DSL_BLK_B);
}
// q-pair / q-head part of a soffset (the kernel adds rowc(kh) at the store): gh * hstride + 1024 qt
K3DS_FN int k3ds_sq(int qt, int gh, int Sq, int Skv, int causal) {
    return gh * k1ds_hstride(Sq, Skv, causal) + qt * DSL_BLK_B;
}
// soffset of block (bh0 + gh, kt0 + kh, qt) relative to the wave base
K3DS_FN int k3ds_so(int kh, int qt, int gh, int kt0, int Sq, int Skv, int causal) {
    return k3ds_sq(qt, gh, Sq, Skv, causal) + k3ds_rowc(kh, kt0, Sq, Skv, causal);
}
