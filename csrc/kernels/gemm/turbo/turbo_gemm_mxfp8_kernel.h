// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.
//
// Hand-tuned MXFP8 GEMM kernel for GFX950 (MI350/MI355).
// 256x256x128 tile, 4-warp, shifted-LDG pipeline, pinned AGPR/VGPR.
// Generated from gemm_mxfp8_16x16x128_v3.hip

#pragma once

#include "primus_turbo/dtype.h"
#include <cassert>
#include <cstdint>
#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>
#include <type_traits>

namespace primus_turbo {
namespace turbo {

using fp8_t    = __hip_fp8_storage_t;
using fp8x32_t = __attribute__((vector_size(32 * sizeof(fp8_t)))) fp8_t;
using fp32x4_t = __attribute__((vector_size(4 * sizeof(float)))) float;

template <int CNT> __device__ __forceinline__ void wait_lgkmcnt() {
    asm volatile("s_waitcnt lgkmcnt(%0)" : : "n"(CNT) : "memory");
}

template <int CNT> __device__ __forceinline__ void wait_vmcnt() {
    asm volatile("s_waitcnt vmcnt(%0)" : : "n"(CNT) : "memory");
}

// ------------------------------------------------------------
// ── GMEM → SMEM direct load (gfx950 buffer_load_lds) ──
// ------------------------------------------------------------
using int32x4_t = int __attribute__((ext_vector_type(4)));

__device__ void llvm_amdgcn_raw_buffer_load_lds(int32x4_t,
                                                __attribute__((address_space(3))) uint32_t *,
                                                int32_t, int32_t, int32_t, int32_t,
                                                int32_t) __asm("llvm.amdgcn.raw.buffer.load.lds");

struct BufferSRD {
    int32x4_t srd;

    __device__ __forceinline__ BufferSRD() {}

    __device__ __forceinline__ explicit BufferSRD(const void *base_ptr,
                                                  uint32_t    num_bytes = 0xffffffffu) {
        struct __attribute__((packed)) {
            const void *p;
            uint32_t    r, c;
        } res{base_ptr, num_bytes, 0x00020000u};
        srd = __builtin_bit_cast(int32x4_t, res);
#pragma unroll
        for (int i = 0; i < 4; ++i)
            srd[i] = __builtin_amdgcn_readfirstlane(srd[i]);
    }
};

// ── Pinned AGPR helpers ──

template <int AC> __device__ __forceinline__ void zero_agpr() {
    asm volatile("v_accvgpr_write_b32 a[%0], 0" : : "n"(AC));
}

template <int START, int END> __device__ __forceinline__ void zero_agpr_range() {
    if constexpr (START <= END) {
        zero_agpr<START>();
        if constexpr (START < END)
            zero_agpr_range<START + 1, END>();
    }
}

template <int AC> __device__ __forceinline__ fp32x4_t read_agpr_4() {
    fp32x4_t out;
    asm volatile("v_accvgpr_read_b32 %0, a[%1]" : "=v"(out[0]) : "n"(AC));
    asm volatile("v_accvgpr_read_b32 %0, a[%1]" : "=v"(out[1]) : "n"(AC + 1));
    asm volatile("v_accvgpr_read_b32 %0, a[%1]" : "=v"(out[2]) : "n"(AC + 2));
    asm volatile("v_accvgpr_read_b32 %0, a[%1]" : "=v"(out[3]) : "n"(AC + 3));
    return out;
}

template <int AGPR> __device__ __forceinline__ void clobber_agpr_one() {
    static_assert(AGPR >= 0 && AGPR <= 255, "AGPR must be in [0, 255]");
    // clang-format off
#define CLOBBER_AREG_CASE(N) case N: asm volatile("" ::: "a" #N); break;
    switch (AGPR) {
        CLOBBER_AREG_CASE(0) CLOBBER_AREG_CASE(1) CLOBBER_AREG_CASE(2) CLOBBER_AREG_CASE(3) CLOBBER_AREG_CASE(4) CLOBBER_AREG_CASE(5) CLOBBER_AREG_CASE(6) CLOBBER_AREG_CASE(7)
        CLOBBER_AREG_CASE(8) CLOBBER_AREG_CASE(9) CLOBBER_AREG_CASE(10) CLOBBER_AREG_CASE(11) CLOBBER_AREG_CASE(12) CLOBBER_AREG_CASE(13) CLOBBER_AREG_CASE(14) CLOBBER_AREG_CASE(15)
        CLOBBER_AREG_CASE(16) CLOBBER_AREG_CASE(17) CLOBBER_AREG_CASE(18) CLOBBER_AREG_CASE(19) CLOBBER_AREG_CASE(20) CLOBBER_AREG_CASE(21) CLOBBER_AREG_CASE(22) CLOBBER_AREG_CASE(23)
        CLOBBER_AREG_CASE(24) CLOBBER_AREG_CASE(25) CLOBBER_AREG_CASE(26) CLOBBER_AREG_CASE(27) CLOBBER_AREG_CASE(28) CLOBBER_AREG_CASE(29) CLOBBER_AREG_CASE(30) CLOBBER_AREG_CASE(31)
        CLOBBER_AREG_CASE(32) CLOBBER_AREG_CASE(33) CLOBBER_AREG_CASE(34) CLOBBER_AREG_CASE(35) CLOBBER_AREG_CASE(36) CLOBBER_AREG_CASE(37) CLOBBER_AREG_CASE(38) CLOBBER_AREG_CASE(39)
        CLOBBER_AREG_CASE(40) CLOBBER_AREG_CASE(41) CLOBBER_AREG_CASE(42) CLOBBER_AREG_CASE(43) CLOBBER_AREG_CASE(44) CLOBBER_AREG_CASE(45) CLOBBER_AREG_CASE(46) CLOBBER_AREG_CASE(47)
        CLOBBER_AREG_CASE(48) CLOBBER_AREG_CASE(49) CLOBBER_AREG_CASE(50) CLOBBER_AREG_CASE(51) CLOBBER_AREG_CASE(52) CLOBBER_AREG_CASE(53) CLOBBER_AREG_CASE(54) CLOBBER_AREG_CASE(55)
        CLOBBER_AREG_CASE(56) CLOBBER_AREG_CASE(57) CLOBBER_AREG_CASE(58) CLOBBER_AREG_CASE(59) CLOBBER_AREG_CASE(60) CLOBBER_AREG_CASE(61) CLOBBER_AREG_CASE(62) CLOBBER_AREG_CASE(63)
        CLOBBER_AREG_CASE(64) CLOBBER_AREG_CASE(65) CLOBBER_AREG_CASE(66) CLOBBER_AREG_CASE(67) CLOBBER_AREG_CASE(68) CLOBBER_AREG_CASE(69) CLOBBER_AREG_CASE(70) CLOBBER_AREG_CASE(71)
        CLOBBER_AREG_CASE(72) CLOBBER_AREG_CASE(73) CLOBBER_AREG_CASE(74) CLOBBER_AREG_CASE(75) CLOBBER_AREG_CASE(76) CLOBBER_AREG_CASE(77) CLOBBER_AREG_CASE(78) CLOBBER_AREG_CASE(79)
        CLOBBER_AREG_CASE(80) CLOBBER_AREG_CASE(81) CLOBBER_AREG_CASE(82) CLOBBER_AREG_CASE(83) CLOBBER_AREG_CASE(84) CLOBBER_AREG_CASE(85) CLOBBER_AREG_CASE(86) CLOBBER_AREG_CASE(87)
        CLOBBER_AREG_CASE(88) CLOBBER_AREG_CASE(89) CLOBBER_AREG_CASE(90) CLOBBER_AREG_CASE(91) CLOBBER_AREG_CASE(92) CLOBBER_AREG_CASE(93) CLOBBER_AREG_CASE(94) CLOBBER_AREG_CASE(95)
        CLOBBER_AREG_CASE(96) CLOBBER_AREG_CASE(97) CLOBBER_AREG_CASE(98) CLOBBER_AREG_CASE(99) CLOBBER_AREG_CASE(100) CLOBBER_AREG_CASE(101) CLOBBER_AREG_CASE(102) CLOBBER_AREG_CASE(103)
        CLOBBER_AREG_CASE(104) CLOBBER_AREG_CASE(105) CLOBBER_AREG_CASE(106) CLOBBER_AREG_CASE(107) CLOBBER_AREG_CASE(108) CLOBBER_AREG_CASE(109) CLOBBER_AREG_CASE(110) CLOBBER_AREG_CASE(111)
        CLOBBER_AREG_CASE(112) CLOBBER_AREG_CASE(113) CLOBBER_AREG_CASE(114) CLOBBER_AREG_CASE(115) CLOBBER_AREG_CASE(116) CLOBBER_AREG_CASE(117) CLOBBER_AREG_CASE(118) CLOBBER_AREG_CASE(119)
        CLOBBER_AREG_CASE(120) CLOBBER_AREG_CASE(121) CLOBBER_AREG_CASE(122) CLOBBER_AREG_CASE(123) CLOBBER_AREG_CASE(124) CLOBBER_AREG_CASE(125) CLOBBER_AREG_CASE(126) CLOBBER_AREG_CASE(127)
        CLOBBER_AREG_CASE(128) CLOBBER_AREG_CASE(129) CLOBBER_AREG_CASE(130) CLOBBER_AREG_CASE(131) CLOBBER_AREG_CASE(132) CLOBBER_AREG_CASE(133) CLOBBER_AREG_CASE(134) CLOBBER_AREG_CASE(135)
        CLOBBER_AREG_CASE(136) CLOBBER_AREG_CASE(137) CLOBBER_AREG_CASE(138) CLOBBER_AREG_CASE(139) CLOBBER_AREG_CASE(140) CLOBBER_AREG_CASE(141) CLOBBER_AREG_CASE(142) CLOBBER_AREG_CASE(143)
        CLOBBER_AREG_CASE(144) CLOBBER_AREG_CASE(145) CLOBBER_AREG_CASE(146) CLOBBER_AREG_CASE(147) CLOBBER_AREG_CASE(148) CLOBBER_AREG_CASE(149) CLOBBER_AREG_CASE(150) CLOBBER_AREG_CASE(151)
        CLOBBER_AREG_CASE(152) CLOBBER_AREG_CASE(153) CLOBBER_AREG_CASE(154) CLOBBER_AREG_CASE(155) CLOBBER_AREG_CASE(156) CLOBBER_AREG_CASE(157) CLOBBER_AREG_CASE(158) CLOBBER_AREG_CASE(159)
        CLOBBER_AREG_CASE(160) CLOBBER_AREG_CASE(161) CLOBBER_AREG_CASE(162) CLOBBER_AREG_CASE(163) CLOBBER_AREG_CASE(164) CLOBBER_AREG_CASE(165) CLOBBER_AREG_CASE(166) CLOBBER_AREG_CASE(167)
        CLOBBER_AREG_CASE(168) CLOBBER_AREG_CASE(169) CLOBBER_AREG_CASE(170) CLOBBER_AREG_CASE(171) CLOBBER_AREG_CASE(172) CLOBBER_AREG_CASE(173) CLOBBER_AREG_CASE(174) CLOBBER_AREG_CASE(175)
        CLOBBER_AREG_CASE(176) CLOBBER_AREG_CASE(177) CLOBBER_AREG_CASE(178) CLOBBER_AREG_CASE(179) CLOBBER_AREG_CASE(180) CLOBBER_AREG_CASE(181) CLOBBER_AREG_CASE(182) CLOBBER_AREG_CASE(183)
        CLOBBER_AREG_CASE(184) CLOBBER_AREG_CASE(185) CLOBBER_AREG_CASE(186) CLOBBER_AREG_CASE(187) CLOBBER_AREG_CASE(188) CLOBBER_AREG_CASE(189) CLOBBER_AREG_CASE(190) CLOBBER_AREG_CASE(191)
        CLOBBER_AREG_CASE(192) CLOBBER_AREG_CASE(193) CLOBBER_AREG_CASE(194) CLOBBER_AREG_CASE(195) CLOBBER_AREG_CASE(196) CLOBBER_AREG_CASE(197) CLOBBER_AREG_CASE(198) CLOBBER_AREG_CASE(199)
        CLOBBER_AREG_CASE(200) CLOBBER_AREG_CASE(201) CLOBBER_AREG_CASE(202) CLOBBER_AREG_CASE(203) CLOBBER_AREG_CASE(204) CLOBBER_AREG_CASE(205) CLOBBER_AREG_CASE(206) CLOBBER_AREG_CASE(207)
        CLOBBER_AREG_CASE(208) CLOBBER_AREG_CASE(209) CLOBBER_AREG_CASE(210) CLOBBER_AREG_CASE(211) CLOBBER_AREG_CASE(212) CLOBBER_AREG_CASE(213) CLOBBER_AREG_CASE(214) CLOBBER_AREG_CASE(215)
        CLOBBER_AREG_CASE(216) CLOBBER_AREG_CASE(217) CLOBBER_AREG_CASE(218) CLOBBER_AREG_CASE(219) CLOBBER_AREG_CASE(220) CLOBBER_AREG_CASE(221) CLOBBER_AREG_CASE(222) CLOBBER_AREG_CASE(223)
        CLOBBER_AREG_CASE(224) CLOBBER_AREG_CASE(225) CLOBBER_AREG_CASE(226) CLOBBER_AREG_CASE(227) CLOBBER_AREG_CASE(228) CLOBBER_AREG_CASE(229) CLOBBER_AREG_CASE(230) CLOBBER_AREG_CASE(231)
        CLOBBER_AREG_CASE(232) CLOBBER_AREG_CASE(233) CLOBBER_AREG_CASE(234) CLOBBER_AREG_CASE(235) CLOBBER_AREG_CASE(236) CLOBBER_AREG_CASE(237) CLOBBER_AREG_CASE(238) CLOBBER_AREG_CASE(239)
        CLOBBER_AREG_CASE(240) CLOBBER_AREG_CASE(241) CLOBBER_AREG_CASE(242) CLOBBER_AREG_CASE(243) CLOBBER_AREG_CASE(244) CLOBBER_AREG_CASE(245) CLOBBER_AREG_CASE(246) CLOBBER_AREG_CASE(247)
        CLOBBER_AREG_CASE(248) CLOBBER_AREG_CASE(249) CLOBBER_AREG_CASE(250) CLOBBER_AREG_CASE(251) CLOBBER_AREG_CASE(252) CLOBBER_AREG_CASE(253) CLOBBER_AREG_CASE(254) CLOBBER_AREG_CASE(255)
    }
#undef CLOBBER_AREG_CASE
    // clang-format on
}

template <int START, int END> __device__ __forceinline__ void reserve_agpr_range() {
    if constexpr (START <= END) {
        clobber_agpr_one<START>();
        if constexpr (START < END)
            reserve_agpr_range<START + 1, END>();
    }
}

// ── Pinned VGPR helpers ──

template <int VGPR> __device__ __forceinline__ void clobber_vgpr_one() {
    static_assert(VGPR >= 0 && VGPR <= 255, "VGPR must be in [0, 255]");
    // clang-format off
#define CLOBBER_VREG_CASE(N) case N: asm volatile("" ::: "v" #N); break;
    switch (VGPR) {
        CLOBBER_VREG_CASE(0) CLOBBER_VREG_CASE(1) CLOBBER_VREG_CASE(2) CLOBBER_VREG_CASE(3) CLOBBER_VREG_CASE(4) CLOBBER_VREG_CASE(5) CLOBBER_VREG_CASE(6) CLOBBER_VREG_CASE(7)
        CLOBBER_VREG_CASE(8) CLOBBER_VREG_CASE(9) CLOBBER_VREG_CASE(10) CLOBBER_VREG_CASE(11) CLOBBER_VREG_CASE(12) CLOBBER_VREG_CASE(13) CLOBBER_VREG_CASE(14) CLOBBER_VREG_CASE(15)
        CLOBBER_VREG_CASE(16) CLOBBER_VREG_CASE(17) CLOBBER_VREG_CASE(18) CLOBBER_VREG_CASE(19) CLOBBER_VREG_CASE(20) CLOBBER_VREG_CASE(21) CLOBBER_VREG_CASE(22) CLOBBER_VREG_CASE(23)
        CLOBBER_VREG_CASE(24) CLOBBER_VREG_CASE(25) CLOBBER_VREG_CASE(26) CLOBBER_VREG_CASE(27) CLOBBER_VREG_CASE(28) CLOBBER_VREG_CASE(29) CLOBBER_VREG_CASE(30) CLOBBER_VREG_CASE(31)
        CLOBBER_VREG_CASE(32) CLOBBER_VREG_CASE(33) CLOBBER_VREG_CASE(34) CLOBBER_VREG_CASE(35) CLOBBER_VREG_CASE(36) CLOBBER_VREG_CASE(37) CLOBBER_VREG_CASE(38) CLOBBER_VREG_CASE(39)
        CLOBBER_VREG_CASE(40) CLOBBER_VREG_CASE(41) CLOBBER_VREG_CASE(42) CLOBBER_VREG_CASE(43) CLOBBER_VREG_CASE(44) CLOBBER_VREG_CASE(45) CLOBBER_VREG_CASE(46) CLOBBER_VREG_CASE(47)
        CLOBBER_VREG_CASE(48) CLOBBER_VREG_CASE(49) CLOBBER_VREG_CASE(50) CLOBBER_VREG_CASE(51) CLOBBER_VREG_CASE(52) CLOBBER_VREG_CASE(53) CLOBBER_VREG_CASE(54) CLOBBER_VREG_CASE(55)
        CLOBBER_VREG_CASE(56) CLOBBER_VREG_CASE(57) CLOBBER_VREG_CASE(58) CLOBBER_VREG_CASE(59) CLOBBER_VREG_CASE(60) CLOBBER_VREG_CASE(61) CLOBBER_VREG_CASE(62) CLOBBER_VREG_CASE(63)
        CLOBBER_VREG_CASE(64) CLOBBER_VREG_CASE(65) CLOBBER_VREG_CASE(66) CLOBBER_VREG_CASE(67) CLOBBER_VREG_CASE(68) CLOBBER_VREG_CASE(69) CLOBBER_VREG_CASE(70) CLOBBER_VREG_CASE(71)
        CLOBBER_VREG_CASE(72) CLOBBER_VREG_CASE(73) CLOBBER_VREG_CASE(74) CLOBBER_VREG_CASE(75) CLOBBER_VREG_CASE(76) CLOBBER_VREG_CASE(77) CLOBBER_VREG_CASE(78) CLOBBER_VREG_CASE(79)
        CLOBBER_VREG_CASE(80) CLOBBER_VREG_CASE(81) CLOBBER_VREG_CASE(82) CLOBBER_VREG_CASE(83) CLOBBER_VREG_CASE(84) CLOBBER_VREG_CASE(85) CLOBBER_VREG_CASE(86) CLOBBER_VREG_CASE(87)
        CLOBBER_VREG_CASE(88) CLOBBER_VREG_CASE(89) CLOBBER_VREG_CASE(90) CLOBBER_VREG_CASE(91) CLOBBER_VREG_CASE(92) CLOBBER_VREG_CASE(93) CLOBBER_VREG_CASE(94) CLOBBER_VREG_CASE(95)
        CLOBBER_VREG_CASE(96) CLOBBER_VREG_CASE(97) CLOBBER_VREG_CASE(98) CLOBBER_VREG_CASE(99) CLOBBER_VREG_CASE(100) CLOBBER_VREG_CASE(101) CLOBBER_VREG_CASE(102) CLOBBER_VREG_CASE(103)
        CLOBBER_VREG_CASE(104) CLOBBER_VREG_CASE(105) CLOBBER_VREG_CASE(106) CLOBBER_VREG_CASE(107) CLOBBER_VREG_CASE(108) CLOBBER_VREG_CASE(109) CLOBBER_VREG_CASE(110) CLOBBER_VREG_CASE(111)
        CLOBBER_VREG_CASE(112) CLOBBER_VREG_CASE(113) CLOBBER_VREG_CASE(114) CLOBBER_VREG_CASE(115) CLOBBER_VREG_CASE(116) CLOBBER_VREG_CASE(117) CLOBBER_VREG_CASE(118) CLOBBER_VREG_CASE(119)
        CLOBBER_VREG_CASE(120) CLOBBER_VREG_CASE(121) CLOBBER_VREG_CASE(122) CLOBBER_VREG_CASE(123) CLOBBER_VREG_CASE(124) CLOBBER_VREG_CASE(125) CLOBBER_VREG_CASE(126) CLOBBER_VREG_CASE(127)
        CLOBBER_VREG_CASE(128) CLOBBER_VREG_CASE(129) CLOBBER_VREG_CASE(130) CLOBBER_VREG_CASE(131) CLOBBER_VREG_CASE(132) CLOBBER_VREG_CASE(133) CLOBBER_VREG_CASE(134) CLOBBER_VREG_CASE(135)
        CLOBBER_VREG_CASE(136) CLOBBER_VREG_CASE(137) CLOBBER_VREG_CASE(138) CLOBBER_VREG_CASE(139) CLOBBER_VREG_CASE(140) CLOBBER_VREG_CASE(141) CLOBBER_VREG_CASE(142) CLOBBER_VREG_CASE(143)
        CLOBBER_VREG_CASE(144) CLOBBER_VREG_CASE(145) CLOBBER_VREG_CASE(146) CLOBBER_VREG_CASE(147) CLOBBER_VREG_CASE(148) CLOBBER_VREG_CASE(149) CLOBBER_VREG_CASE(150) CLOBBER_VREG_CASE(151)
        CLOBBER_VREG_CASE(152) CLOBBER_VREG_CASE(153) CLOBBER_VREG_CASE(154) CLOBBER_VREG_CASE(155) CLOBBER_VREG_CASE(156) CLOBBER_VREG_CASE(157) CLOBBER_VREG_CASE(158) CLOBBER_VREG_CASE(159)
        CLOBBER_VREG_CASE(160) CLOBBER_VREG_CASE(161) CLOBBER_VREG_CASE(162) CLOBBER_VREG_CASE(163) CLOBBER_VREG_CASE(164) CLOBBER_VREG_CASE(165) CLOBBER_VREG_CASE(166) CLOBBER_VREG_CASE(167)
        CLOBBER_VREG_CASE(168) CLOBBER_VREG_CASE(169) CLOBBER_VREG_CASE(170) CLOBBER_VREG_CASE(171) CLOBBER_VREG_CASE(172) CLOBBER_VREG_CASE(173) CLOBBER_VREG_CASE(174) CLOBBER_VREG_CASE(175)
        CLOBBER_VREG_CASE(176) CLOBBER_VREG_CASE(177) CLOBBER_VREG_CASE(178) CLOBBER_VREG_CASE(179) CLOBBER_VREG_CASE(180) CLOBBER_VREG_CASE(181) CLOBBER_VREG_CASE(182) CLOBBER_VREG_CASE(183)
        CLOBBER_VREG_CASE(184) CLOBBER_VREG_CASE(185) CLOBBER_VREG_CASE(186) CLOBBER_VREG_CASE(187) CLOBBER_VREG_CASE(188) CLOBBER_VREG_CASE(189) CLOBBER_VREG_CASE(190) CLOBBER_VREG_CASE(191)
        CLOBBER_VREG_CASE(192) CLOBBER_VREG_CASE(193) CLOBBER_VREG_CASE(194) CLOBBER_VREG_CASE(195) CLOBBER_VREG_CASE(196) CLOBBER_VREG_CASE(197) CLOBBER_VREG_CASE(198) CLOBBER_VREG_CASE(199)
        CLOBBER_VREG_CASE(200) CLOBBER_VREG_CASE(201) CLOBBER_VREG_CASE(202) CLOBBER_VREG_CASE(203) CLOBBER_VREG_CASE(204) CLOBBER_VREG_CASE(205) CLOBBER_VREG_CASE(206) CLOBBER_VREG_CASE(207)
        CLOBBER_VREG_CASE(208) CLOBBER_VREG_CASE(209) CLOBBER_VREG_CASE(210) CLOBBER_VREG_CASE(211) CLOBBER_VREG_CASE(212) CLOBBER_VREG_CASE(213) CLOBBER_VREG_CASE(214) CLOBBER_VREG_CASE(215)
        CLOBBER_VREG_CASE(216) CLOBBER_VREG_CASE(217) CLOBBER_VREG_CASE(218) CLOBBER_VREG_CASE(219) CLOBBER_VREG_CASE(220) CLOBBER_VREG_CASE(221) CLOBBER_VREG_CASE(222) CLOBBER_VREG_CASE(223)
        CLOBBER_VREG_CASE(224) CLOBBER_VREG_CASE(225) CLOBBER_VREG_CASE(226) CLOBBER_VREG_CASE(227) CLOBBER_VREG_CASE(228) CLOBBER_VREG_CASE(229) CLOBBER_VREG_CASE(230) CLOBBER_VREG_CASE(231)
        CLOBBER_VREG_CASE(232) CLOBBER_VREG_CASE(233) CLOBBER_VREG_CASE(234) CLOBBER_VREG_CASE(235) CLOBBER_VREG_CASE(236) CLOBBER_VREG_CASE(237) CLOBBER_VREG_CASE(238) CLOBBER_VREG_CASE(239)
        CLOBBER_VREG_CASE(240) CLOBBER_VREG_CASE(241) CLOBBER_VREG_CASE(242) CLOBBER_VREG_CASE(243) CLOBBER_VREG_CASE(244) CLOBBER_VREG_CASE(245) CLOBBER_VREG_CASE(246) CLOBBER_VREG_CASE(247)
        CLOBBER_VREG_CASE(248) CLOBBER_VREG_CASE(249) CLOBBER_VREG_CASE(250) CLOBBER_VREG_CASE(251) CLOBBER_VREG_CASE(252) CLOBBER_VREG_CASE(253) CLOBBER_VREG_CASE(254) CLOBBER_VREG_CASE(255)
    }
#undef CLOBBER_VREG_CASE
    // clang-format on
}

template <int START, int END> __device__ __forceinline__ void reserve_vgpr_range() {
    if constexpr (START <= END) {
        clobber_vgpr_one<START>();
        if constexpr (START < END)
            reserve_vgpr_range<START + 1, END>();
    }
}

template <int VDST, int IMM_OFFSET = 0>
__device__ __forceinline__ void ds_read_b128_pinned(uint32_t lds_addr) {
    asm volatile("ds_read_b128 v[%0:%1], %2 offset:%3"
                 :
                 : "n"(VDST), "n"(VDST + 3), "v"(lds_addr), "n"(IMM_OFFSET)
                 : "memory");
}

template <int VDST, int IMM_OFFSET = 0>
__device__ __forceinline__ void ds_read_b32_pinned(uint32_t lds_addr) {
    asm volatile("ds_read_b32 v[%0], %1 offset:%2"
                 :
                 : "n"(VDST), "v"(lds_addr), "n"(IMM_OFFSET)
                 : "memory");
}

template <int VSTART>
__device__ __forceinline__ void load_data_subtile_pinned(uint32_t subtile_addr,
                                                         uint32_t (&lds_offsets)[2]) {
    uint32_t addr0 = subtile_addr + lds_offsets[0];
    uint32_t addr1 = subtile_addr + lds_offsets[1];
    ds_read_b128_pinned<VSTART + 0, 0>(addr0);
    ds_read_b128_pinned<VSTART + 4, 0>(addr1);
    ds_read_b128_pinned<VSTART + 8, 2048>(addr0);
    ds_read_b128_pinned<VSTART + 12, 2048>(addr1);
    ds_read_b128_pinned<VSTART + 16, 4096>(addr0);
    ds_read_b128_pinned<VSTART + 20, 4096>(addr1);
    ds_read_b128_pinned<VSTART + 24, 6144>(addr0);
    ds_read_b128_pinned<VSTART + 28, 6144>(addr1);
}

template <int VSTART>
__device__ __forceinline__ void load_scale_subtile_pinned(uint32_t scale_subtile_addr,
                                                          uint32_t scale_lds_offset) {
    uint32_t base = scale_subtile_addr + scale_lds_offset * sizeof(uint32_t);
    ds_read_b32_pinned<VSTART, 0>(base);
    ds_read_b32_pinned<VSTART + 1, 256>(base);
    ds_read_b32_pinned<VSTART + 2, 512>(base);
    ds_read_b32_pinned<VSTART + 3, 768>(base);
}

template <int Bytes>
__device__ __forceinline__ void load_gmem_to_smem_srd(const BufferSRD &srd, uint32_t ldg_offset,
                                                      uint32_t lds_addr, int32_t soffset) {
    static_assert(Bytes == 4 || Bytes == 12 || Bytes == 16,
                  "gfx950 supports 1/3/4 DWORDs per thread.");
    using as3_uint32_ptr = __attribute__((address_space(3))) uint32_t *;
    auto lds             = reinterpret_cast<as3_uint32_ptr>((uintptr_t) lds_addr);
    llvm_amdgcn_raw_buffer_load_lds(srd.srd, lds, Bytes, ldg_offset, soffset, 0, 0);
}

// ------------------------------------------------------------
// ── Tile-index swizzle for MI355/GFX950 (8 XCDs × 32 CUs) ──
// ------------------------------------------------------------
template <uint32_t BLOCK_SIZE_M, uint32_t BLOCK_SIZE_N>
__device__ __forceinline__ void swizzle_pid_m_n(const int m, const int n, int &pid_m, int &pid_n) {
    const int NUM_WGS  = gridDim.x * gridDim.y;
    const int NUM_XCDS = 8;
    const int ntiles_n = (n + static_cast<int>(BLOCK_SIZE_N) - 1) / static_cast<int>(BLOCK_SIZE_N);
    const int WGM      = (ntiles_n > 32) ? 4 : 8;

    const int pid = static_cast<int>(blockIdx.x * gridDim.y + blockIdx.y);

    if (NUM_WGS < NUM_XCDS) {
        pid_m = static_cast<int>(blockIdx.x) * static_cast<int>(BLOCK_SIZE_M);
        pid_n = static_cast<int>(blockIdx.y) * static_cast<int>(BLOCK_SIZE_N);
        return;
    }

    const int q        = NUM_WGS / NUM_XCDS;
    const int r        = NUM_WGS % NUM_XCDS;
    const int xcd_id   = pid % NUM_XCDS;
    const int local_id = pid / NUM_XCDS;
    const int wgid     = xcd_id * q + local_id + min(xcd_id, r);

    const int num_pid_m = (m + static_cast<int>(BLOCK_SIZE_M) - 1) / static_cast<int>(BLOCK_SIZE_M);
    const int num_pid_n = (n + static_cast<int>(BLOCK_SIZE_N) - 1) / static_cast<int>(BLOCK_SIZE_N);
    const int num_wgid_in_group = WGM * num_pid_n;
    const int group_id          = int(wgid / num_wgid_in_group);
    const int first_pid_m       = group_id * WGM;
    const int group_size_m      = min(num_pid_m - first_pid_m, WGM);
    pid_m                       = first_pid_m + int((wgid % num_wgid_in_group) % group_size_m);
    pid_n                       = int((wgid % num_wgid_in_group) / group_size_m);
    pid_m *= static_cast<int>(BLOCK_SIZE_M);
    pid_n *= static_cast<int>(BLOCK_SIZE_N);
}

// ── GemmTile struct: v3-relevant methods ──
// ------------------------------------------------------------
template <typename AType, typename BType, typename CType, typename AccType>
struct GEMM_Tile_MXFP8_NT_256x256x128_16x16x128_4_WAVE_GFX950 {
public:
    static constexpr uint32_t WARP_SIZE = 64;
    static constexpr uint32_t NUM_WARPS = 4;

    static constexpr uint32_t BLOCK_SIZE_M = 256;
    static constexpr uint32_t BLOCK_SIZE_N = 256;
    static constexpr uint32_t BLOCK_SIZE_K = 128;

    static constexpr uint32_t MFMA_SIZE_M = 16;
    static constexpr uint32_t MFMA_SIZE_N = 16;
    static constexpr uint32_t MFMA_SIZE_K = 128;

    static constexpr uint32_t MX_BLOCK_SIZE   = 32;
    static constexpr uint32_t SCALE_FRAG_SIZE = MFMA_SIZE_M * MFMA_SIZE_K / MX_BLOCK_SIZE;

    // cbsz/blgp format codes for v_mfma_scale_f32_16x16x128_f8f6f4:
    //   0 = fp8 (e4m3), 1 = bf8 (e5m2)
    static constexpr int A_FMT =
        (std::is_same_v<AType, __hip_fp8_e5m2> || std::is_same_v<AType, dtype::float8_e5m2>) ? 1
                                                                                             : 0;
    static constexpr int B_FMT =
        (std::is_same_v<BType, __hip_fp8_e5m2> || std::is_same_v<BType, dtype::float8_e5m2>) ? 1
                                                                                             : 0;

    // Pinned register layout:
    //
    // VGPR (double-buffered A/B data + scale):
    //   v[0:111]   compiler-managed (addresses, pointers, loop vars)
    //   v[112:115] A scale buffer 0    (4 × uint32_t)
    //   v[116:119] A scale buffer 1    (4 × uint32_t)
    //   v[120:123] B scale buffer 0    (4 × uint32_t)
    //   v[124:127] B scale buffer 1    (4 × uint32_t)
    //   v[128:159] A data buffer 0     (4 frags × 8 VGPR = 32 VGPR)
    //   v[160:191] A data buffer 1     (4 frags × 8 VGPR = 32 VGPR)
    //   v[192:223] B data buffer 0     (4 frags × 8 VGPR = 32 VGPR)
    //   v[224:255] B data buffer 1     (4 frags × 8 VGPR = 32 VGPR)
    //
    // AGPR (C accumulator, 256 × fp32):
    //   a[0:255]   4 subtiles × 16 tiles × 4 fp32 = 256 AGPR
    static constexpr int PIN_AS0 = 112, PIN_AS1 = 116;
    static constexpr int PIN_BS0 = 120, PIN_BS1 = 124;
    static constexpr int PIN_A0 = 128, PIN_A1 = 160;
    static constexpr int PIN_B0 = 192, PIN_B1 = 224;

    template <typename T, uint32_t N> struct SmemTile {
        T                   data[N];
        __device__ uint32_t u32_ptr() { return reinterpret_cast<uintptr_t>(data); }
    };

    using ASmemSubtile      = SmemTile<AType, 64 * BLOCK_SIZE_K>;
    using BSmemSubtile      = SmemTile<BType, 64 * BLOCK_SIZE_K>;
    using AScaleSmemSubtile = SmemTile<uint32_t, 64 * BLOCK_SIZE_K / MX_BLOCK_SIZE>;
    using BScaleSmemSubtile = SmemTile<uint32_t, 64 * BLOCK_SIZE_K / MX_BLOCK_SIZE>;

    const uint32_t lane_id;
    const uint32_t warp_id;
    const uint32_t warp_m, warp_n;
    const uint32_t m, n, k;

public:
    __device__ __forceinline__ GEMM_Tile_MXFP8_NT_256x256x128_16x16x128_4_WAVE_GFX950(uint32_t tid,
                                                                                      uint32_t m,
                                                                                      uint32_t n,
                                                                                      uint32_t k)
        : lane_id(tid % WARP_SIZE), warp_id(tid / WARP_SIZE), warp_m(tid / WARP_SIZE / 2),
          warp_n(tid / WARP_SIZE % 2), m(m), n(n), k(k) {}

    template <uint32_t H>
    __device__ __forceinline__ void
    load_a_gmem_to_smem_half_srd(const BufferSRD &a_srd, const uint32_t (&ldg_offsets)[2],
                                 ASmemSubtile (&a_smem_tile)[4], const uint32_t (&sts_offsets)[2],
                                 int32_t extra_soffset = 0) {
        static_assert(H < 2, "H must be 0 or 1");
        const uint32_t sts_warp_base = warp_id * MFMA_SIZE_M * MFMA_SIZE_K;
#pragma unroll
        for (uint32_t i = H * 2; i < H * 2 + 2; ++i) {
            int32_t soff = __builtin_amdgcn_readfirstlane(
                (int32_t) ((i * 64 + warp_id * MFMA_SIZE_M) * k) + extra_soffset);
            load_gmem_to_smem_srd<16>(a_srd, ldg_offsets[0],
                                      a_smem_tile[i].u32_ptr() + sts_warp_base + sts_offsets[0],
                                      soff);
            load_gmem_to_smem_srd<16>(a_srd, ldg_offsets[1],
                                      a_smem_tile[i].u32_ptr() + sts_warp_base + sts_offsets[1],
                                      soff);
        }
    }

    template <uint32_t H>
    __device__ __forceinline__ void
    load_b_gmem_to_smem_half_srd(const BufferSRD &b_srd, const uint32_t (&ldg_offsets)[2],
                                 BSmemSubtile (&b_smem_tile)[4], const uint32_t (&sts_offsets)[2],
                                 int32_t extra_soffset = 0) {
        static_assert(H < 2, "H must be 0 or 1");
        const uint32_t sts_warp_base = warp_id * MFMA_SIZE_M * MFMA_SIZE_K;
#pragma unroll
        for (uint32_t i = H * 2; i < H * 2 + 2; ++i) {
            int32_t soff = __builtin_amdgcn_readfirstlane(
                (int32_t) ((i * 64 + warp_id * MFMA_SIZE_N) * k) + extra_soffset);
            load_gmem_to_smem_srd<16>(b_srd, ldg_offsets[0],
                                      b_smem_tile[i].u32_ptr() + sts_warp_base + sts_offsets[0],
                                      soff);
            load_gmem_to_smem_srd<16>(b_srd, ldg_offsets[1],
                                      b_smem_tile[i].u32_ptr() + sts_warp_base + sts_offsets[1],
                                      soff);
        }
    }

    template <uint32_t H>
    __device__ __forceinline__ void load_a_scale_gmem_to_smem_half_srd(
        const BufferSRD &a_s_srd, const uint32_t scale_ldg_offset, AScaleSmemSubtile (&a_s_smem)[4],
        const uint32_t scale_sts_offset, const uint32_t scale_cols, int32_t extra_soffset = 0) {
        static_assert(H < 2, "H must be 0 or 1");
        const uint32_t gmem_byte_offset = scale_ldg_offset * sizeof(uint32_t);
        const uint32_t smem_byte_offset = (warp_id * 64 + scale_sts_offset) * sizeof(uint32_t);
#pragma unroll
        for (uint32_t i = H * 2; i < H * 2 + 2; ++i) {
            int32_t soff = __builtin_amdgcn_readfirstlane(
                (int32_t) ((i * 4 + warp_id) * (16 * scale_cols) * sizeof(uint32_t)) +
                extra_soffset);
            load_gmem_to_smem_srd<4>(a_s_srd, gmem_byte_offset,
                                     a_s_smem[i].u32_ptr() + smem_byte_offset, soff);
        }
    }

    template <uint32_t H>
    __device__ __forceinline__ void load_b_scale_gmem_to_smem_half_srd(
        const BufferSRD &b_s_srd, const uint32_t scale_ldg_offset, BScaleSmemSubtile (&b_s_smem)[4],
        const uint32_t scale_sts_offset, const uint32_t scale_cols, int32_t extra_soffset = 0) {
        static_assert(H < 2, "H must be 0 or 1");
        const uint32_t gmem_byte_offset = scale_ldg_offset * sizeof(uint32_t);
        const uint32_t smem_byte_offset = (warp_id * 64 + scale_sts_offset) * sizeof(uint32_t);
#pragma unroll
        for (uint32_t i = H * 2; i < H * 2 + 2; ++i) {
            int32_t soff = __builtin_amdgcn_readfirstlane(
                (int32_t) ((i * 4 + warp_id) * (16 * scale_cols) * sizeof(uint32_t)) +
                extra_soffset);
            load_gmem_to_smem_srd<4>(b_s_srd, gmem_byte_offset,
                                     b_s_smem[i].u32_ptr() + smem_byte_offset, soff);
        }
    }

    __device__ __forceinline__ void precompute_base_soff(int32_t (&base_data_soff)[4],
                                                         int32_t (&base_scale_soff)[4],
                                                         uint32_t scale_cols) {
#pragma unroll
        for (int i = 0; i < 4; i++) {
            base_data_soff[i] =
                __builtin_amdgcn_readfirstlane((int32_t) ((i * 64 + warp_id * MFMA_SIZE_M) * k));
            base_scale_soff[i] = __builtin_amdgcn_readfirstlane(
                (int32_t) ((i * 4 + warp_id) * (16 * scale_cols) * (int32_t) sizeof(uint32_t)));
        }
    }

    __device__ __forceinline__ void zero_c_agpr() { zero_agpr_range<0, 255>(); }

    __device__ __forceinline__ void reserve_pinned_regs() {
        reserve_vgpr_range<PIN_AS0, 255>(); // v[112:255]: A/B data + scale double buffers
        reserve_agpr_range<0, 255>();       // a[0:255]:   C accumulator
    }

    template <int GR, int GC>
    __device__ __forceinline__ void read_c_subtile_from_agpr(fp32x4_t (&c_out)[4][4]) {
        constexpr int B = (GR * 2 + GC) * 64;
        c_out[0][0]     = read_agpr_4<B + 0>();
        c_out[0][1]     = read_agpr_4<B + 4>();
        c_out[0][2]     = read_agpr_4<B + 8>();
        c_out[0][3]     = read_agpr_4<B + 12>();
        c_out[1][0]     = read_agpr_4<B + 16>();
        c_out[1][1]     = read_agpr_4<B + 20>();
        c_out[1][2]     = read_agpr_4<B + 24>();
        c_out[1][3]     = read_agpr_4<B + 28>();
        c_out[2][0]     = read_agpr_4<B + 32>();
        c_out[2][1]     = read_agpr_4<B + 36>();
        c_out[2][2]     = read_agpr_4<B + 40>();
        c_out[2][3]     = read_agpr_4<B + 44>();
        c_out[3][0]     = read_agpr_4<B + 48>();
        c_out[3][1]     = read_agpr_4<B + 52>();
        c_out[3][2]     = read_agpr_4<B + 56>();
        c_out[3][3]     = read_agpr_4<B + 60>();
    }

    __device__ __forceinline__ void store_c_subtile(CType *c_stg_base_ptr, const int32_t n,
                                                    fp32x4_t (&c_frags)[4][4],
                                                    uint32_t (&c_stg_offsets)[4],
                                                    const int32_t valid_rows = 64,
                                                    const int32_t valid_cols = 64) {
#pragma unroll
        for (int tr = 0; tr < 4; ++tr) {
#pragma unroll
            for (int tc = 0; tc < 4; ++tc) {
                CType *c_stg_ptr = c_stg_base_ptr + tr * MFMA_SIZE_M * n + tc * MFMA_SIZE_N;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    int32_t row = tr * MFMA_SIZE_M + lane_id / 16 * 4 + i;
                    int32_t col = tc * MFMA_SIZE_N + lane_id % 16;
                    if (row < valid_rows && col < valid_cols)
                        c_stg_ptr[c_stg_offsets[i]] = CType(c_frags[tr][tc][i]);
                }
            }
        }
    }

    __device__ __forceinline__ void compute_ldg_offsets(uint32_t (&ldg_offsets)[2],
                                                        const uint32_t stride) {
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            uint32_t ldg_row = i * 8 + lane_id / 8;
            uint32_t ldg_col = swizzle_col_(ldg_row, lane_id % 8);
            ldg_offsets[i]   = ldg_row * stride + ldg_col * 16;
        }
    }

    __device__ __forceinline__ void compute_sts_offsets(uint32_t (&sts_offsets)[2]) {
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            sts_offsets[i] = i * 1024 + lane_id * 16;
        }
    }

    __device__ __forceinline__ void compute_lds_offsets(uint32_t (&lds_offsets)[2]) {
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            uint32_t lds_row = lane_id % 16;
            uint32_t lds_col = lane_id / 16 + i * 4;
            uint32_t swz_col = swizzle_col_(lds_row, lds_col);
            lds_offsets[i]   = lds_row * 128 + swz_col * 16;
        }
    }

    __device__ __forceinline__ void compute_stg_offsets(uint32_t (&c_stg_offsets)[4]) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            c_stg_offsets[i] = (lane_id / 16 * 4 + i) * n + lane_id % 16;
        }
    }

    __device__ __forceinline__ uint32_t swizzle_col_(const uint32_t row, const uint32_t col) {
        return col ^ (row >> 1);
    }

    // ── Static phase functions (pinned-register MFMA + memory scheduling) ──

    // cbsz selects A format, blgp selects B format:
    //   0=FP8(e4m3), 1=BF8(e5m2), 2=FP6(e2m3), 3=FP6(e3m2), 4=FP4(e2m1)
    // op_sel_hi controls scale source selection (not data format).
    template <int PIN_A, int PIN_B, int ACC, int PIN_SA, int PIN_SB>
    __device__ __forceinline__ static void mfma_scale_pinned() {
        if constexpr (A_FMT == 0 && B_FMT == 0)
            asm volatile("v_mfma_scale_f32_16x16x128_f8f6f4 a[%0:%1], v[%2:%3], v[%4:%5], "
                         "a[%0:%1], v[%6], v[%7] op_sel_hi:[0,0,0]"
                         :
                         : "n"(ACC), "n"(ACC + 3), "n"(PIN_A), "n"(PIN_A + 7), "n"(PIN_B),
                           "n"(PIN_B + 7), "n"(PIN_SA), "n"(PIN_SB));
        else if constexpr (A_FMT == 1 && B_FMT == 0)
            asm volatile("v_mfma_scale_f32_16x16x128_f8f6f4 a[%0:%1], v[%2:%3], v[%4:%5], "
                         "a[%0:%1], v[%6], v[%7] op_sel_hi:[0,0,0] cbsz:1"
                         :
                         : "n"(ACC), "n"(ACC + 3), "n"(PIN_A), "n"(PIN_A + 7), "n"(PIN_B),
                           "n"(PIN_B + 7), "n"(PIN_SA), "n"(PIN_SB));
        else if constexpr (A_FMT == 0 && B_FMT == 1)
            asm volatile("v_mfma_scale_f32_16x16x128_f8f6f4 a[%0:%1], v[%2:%3], v[%4:%5], "
                         "a[%0:%1], v[%6], v[%7] op_sel_hi:[0,0,0] blgp:1"
                         :
                         : "n"(ACC), "n"(ACC + 3), "n"(PIN_A), "n"(PIN_A + 7), "n"(PIN_B),
                           "n"(PIN_B + 7), "n"(PIN_SA), "n"(PIN_SB));
        else
            asm volatile("v_mfma_scale_f32_16x16x128_f8f6f4 a[%0:%1], v[%2:%3], v[%4:%5], "
                         "a[%0:%1], v[%6], v[%7] op_sel_hi:[0,0,0] cbsz:1 blgp:1"
                         :
                         : "n"(ACC), "n"(ACC + 3), "n"(PIN_A), "n"(PIN_A + 7), "n"(PIN_B),
                           "n"(PIN_B + 7), "n"(PIN_SA), "n"(PIN_SB));
    }

    // Same as phase_mfma_lds_ldg but without GMEM→SMEM prefetch (used in epilogue phases).
    template <int PIN_A, int PIN_SA, int PIN_B, int PIN_SB, int TILE_R, int TILE_C, int PIN_NEXT_D,
              int PIN_NEXT_S>
    __device__ __forceinline__ static void
    phase_mfma_lds(uint32_t lds_data_addr, uint32_t (&lds_offsets)[2], uint32_t lds_scale_addr,
                   uint32_t scale_lds_offset) {
        constexpr int ACC   = (TILE_R * 2 + TILE_C) * 64;
        uint32_t      da0   = lds_data_addr + lds_offsets[0];
        uint32_t      da1   = lds_data_addr + lds_offsets[1];
        uint32_t      sbase = lds_scale_addr + scale_lds_offset * sizeof(uint32_t);

        mfma_scale_pinned<PIN_A + 0, PIN_B + 0, ACC + 0, PIN_SA + 0, PIN_SB + 0>();
        ds_read_b128_pinned<PIN_NEXT_D + 0, 0>(da0);
        ds_read_b128_pinned<PIN_NEXT_D + 4, 0>(da1);
        mfma_scale_pinned<PIN_A + 0, PIN_B + 8, ACC + 4, PIN_SA + 0, PIN_SB + 1>();
        ds_read_b128_pinned<PIN_NEXT_D + 8, 2048>(da0);
        ds_read_b128_pinned<PIN_NEXT_D + 12, 2048>(da1);
        mfma_scale_pinned<PIN_A + 0, PIN_B + 16, ACC + 8, PIN_SA + 0, PIN_SB + 2>();
        ds_read_b128_pinned<PIN_NEXT_D + 16, 4096>(da0);
        ds_read_b128_pinned<PIN_NEXT_D + 20, 4096>(da1);
        mfma_scale_pinned<PIN_A + 0, PIN_B + 24, ACC + 12, PIN_SA + 0, PIN_SB + 3>();
        ds_read_b128_pinned<PIN_NEXT_D + 24, 6144>(da0);
        ds_read_b128_pinned<PIN_NEXT_D + 28, 6144>(da1);
        mfma_scale_pinned<PIN_A + 8, PIN_B + 0, ACC + 16, PIN_SA + 1, PIN_SB + 0>();
        ds_read_b32_pinned<PIN_NEXT_S + 0, 0>(sbase);
        ds_read_b32_pinned<PIN_NEXT_S + 1, 256>(sbase);
        mfma_scale_pinned<PIN_A + 8, PIN_B + 8, ACC + 20, PIN_SA + 1, PIN_SB + 1>();
        ds_read_b32_pinned<PIN_NEXT_S + 2, 512>(sbase);
        ds_read_b32_pinned<PIN_NEXT_S + 3, 768>(sbase);
        mfma_scale_pinned<PIN_A + 8, PIN_B + 16, ACC + 24, PIN_SA + 1, PIN_SB + 2>();
        mfma_scale_pinned<PIN_A + 8, PIN_B + 24, ACC + 28, PIN_SA + 1, PIN_SB + 3>();
        mfma_scale_pinned<PIN_A + 16, PIN_B + 0, ACC + 32, PIN_SA + 2, PIN_SB + 0>();
        mfma_scale_pinned<PIN_A + 16, PIN_B + 8, ACC + 36, PIN_SA + 2, PIN_SB + 1>();
        mfma_scale_pinned<PIN_A + 16, PIN_B + 16, ACC + 40, PIN_SA + 2, PIN_SB + 2>();
        mfma_scale_pinned<PIN_A + 16, PIN_B + 24, ACC + 44, PIN_SA + 2, PIN_SB + 3>();
        mfma_scale_pinned<PIN_A + 24, PIN_B + 0, ACC + 48, PIN_SA + 3, PIN_SB + 0>();
        mfma_scale_pinned<PIN_A + 24, PIN_B + 8, ACC + 52, PIN_SA + 3, PIN_SB + 1>();
        mfma_scale_pinned<PIN_A + 24, PIN_B + 16, ACC + 56, PIN_SA + 3, PIN_SB + 2>();
        mfma_scale_pinned<PIN_A + 24, PIN_B + 24, ACC + 60, PIN_SA + 3, PIN_SB + 3>();
    }

    // Pure MFMA phase — no LDS prefetch, no LDG. Used for the final epilogue phases.
    template <int PIN_A, int PIN_SA, int PIN_B, int PIN_SB, int TILE_R, int TILE_C>
    __device__ __forceinline__ static void phase_mfma_only() {
        constexpr int ACC = (TILE_R * 2 + TILE_C) * 64;
        mfma_scale_pinned<PIN_A + 0, PIN_B + 0, ACC + 0, PIN_SA + 0, PIN_SB + 0>();
        mfma_scale_pinned<PIN_A + 0, PIN_B + 8, ACC + 4, PIN_SA + 0, PIN_SB + 1>();
        mfma_scale_pinned<PIN_A + 0, PIN_B + 16, ACC + 8, PIN_SA + 0, PIN_SB + 2>();
        mfma_scale_pinned<PIN_A + 0, PIN_B + 24, ACC + 12, PIN_SA + 0, PIN_SB + 3>();
        mfma_scale_pinned<PIN_A + 8, PIN_B + 0, ACC + 16, PIN_SA + 1, PIN_SB + 0>();
        mfma_scale_pinned<PIN_A + 8, PIN_B + 8, ACC + 20, PIN_SA + 1, PIN_SB + 1>();
        mfma_scale_pinned<PIN_A + 8, PIN_B + 16, ACC + 24, PIN_SA + 1, PIN_SB + 2>();
        mfma_scale_pinned<PIN_A + 8, PIN_B + 24, ACC + 28, PIN_SA + 1, PIN_SB + 3>();
        mfma_scale_pinned<PIN_A + 16, PIN_B + 0, ACC + 32, PIN_SA + 2, PIN_SB + 0>();
        mfma_scale_pinned<PIN_A + 16, PIN_B + 8, ACC + 36, PIN_SA + 2, PIN_SB + 1>();
        mfma_scale_pinned<PIN_A + 16, PIN_B + 16, ACC + 40, PIN_SA + 2, PIN_SB + 2>();
        mfma_scale_pinned<PIN_A + 16, PIN_B + 24, ACC + 44, PIN_SA + 2, PIN_SB + 3>();
        mfma_scale_pinned<PIN_A + 24, PIN_B + 0, ACC + 48, PIN_SA + 3, PIN_SB + 0>();
        mfma_scale_pinned<PIN_A + 24, PIN_B + 8, ACC + 52, PIN_SA + 3, PIN_SB + 1>();
        mfma_scale_pinned<PIN_A + 24, PIN_B + 16, ACC + 56, PIN_SA + 3, PIN_SB + 2>();
        mfma_scale_pinned<PIN_A + 24, PIN_B + 24, ACC + 60, PIN_SA + 3, PIN_SB + 3>();
    }

    // One phase of the 64x64 subtile pipeline, executing three operations simultaneously:
    //   1. MFMA:  16 v_mfma_scale_f32_16x16x128 (4x4 tile grid), using data in
    //   PIN_A/PIN_B/PIN_SA/PIN_SB
    //   2. LDS:   prefetch next phase's data+scale from SMEM -> pinned VGPRs (PIN_NEXT_D,
    //   PIN_NEXT_S)
    //   3. LDG:   prefetch a future K-tile from GMEM -> SMEM (buffer_load_lds, bypassing VGPRs)
    //
    // Template parameters (all are pinned register start indices unless noted):
    //   PIN_A, PIN_SA   -- current A data / A scale VGPRs for MFMA input
    //   PIN_B, PIN_SB   -- current B data / B scale VGPRs for MFMA input
    //   TILE_R, TILE_C  -- C subtile row/col index (0 or 1), selects AGPR accumulator range
    //   PIN_NEXT_D      -- destination VGPRs for next phase's data (ds_read_b128 target)
    //   PIN_NEXT_S      -- destination VGPRs for next phase's scale (ds_read_b32 target)
    //
    // Instruction scheduling:
    //   MFMA #0-#5:  interleaved with ds_read (LDS->VGPR, no LDS port conflict with MFMA)
    //   MFMA #6-#11: interleaved with buffer_load_lds (GMEM->SMEM, placed in ds_read-free gap)
    //   MFMA #12-#15: pure compute, no memory ops
    template <int PIN_A, int PIN_SA, int PIN_B, int PIN_SB, int TILE_R, int TILE_C, int PIN_NEXT_D,
              int PIN_NEXT_S>
    __device__ __forceinline__ static void
    phase_mfma_lds_ldg(uint32_t lds_data_addr, uint32_t (&lds_offsets)[2], uint32_t lds_scale_addr,
                       uint32_t scale_lds_offset, const BufferSRD &data_srd,
                       const uint32_t (&ldg_offsets)[2], uint32_t data_m0_0, uint32_t data_m0_1,
                       int32_t data_soff_0, int32_t data_soff_1, const BufferSRD &scale_srd,
                       uint32_t scale_gmem_off, uint32_t scale_m0_0, uint32_t scale_m0_1,
                       int32_t scale_soff_0, int32_t scale_soff_1) {
        constexpr int ACC   = (TILE_R * 2 + TILE_C) * 64;
        uint32_t      da0   = lds_data_addr + lds_offsets[0];
        uint32_t      da1   = lds_data_addr + lds_offsets[1];
        uint32_t      sbase = lds_scale_addr + scale_lds_offset * sizeof(uint32_t);

        // MFMA #0-#5: interleaved with ds_read (data + scale prefetch for next phase)
        mfma_scale_pinned<PIN_A + 0, PIN_B + 0, ACC + 0, PIN_SA + 0, PIN_SB + 0>();
        ds_read_b128_pinned<PIN_NEXT_D + 0, 0>(da0);
        ds_read_b128_pinned<PIN_NEXT_D + 4, 0>(da1);
        mfma_scale_pinned<PIN_A + 0, PIN_B + 8, ACC + 4, PIN_SA + 0, PIN_SB + 1>();
        ds_read_b128_pinned<PIN_NEXT_D + 8, 2048>(da0);
        ds_read_b128_pinned<PIN_NEXT_D + 12, 2048>(da1);
        mfma_scale_pinned<PIN_A + 0, PIN_B + 16, ACC + 8, PIN_SA + 0, PIN_SB + 2>();
        ds_read_b128_pinned<PIN_NEXT_D + 16, 4096>(da0);
        ds_read_b128_pinned<PIN_NEXT_D + 20, 4096>(da1);
        mfma_scale_pinned<PIN_A + 0, PIN_B + 24, ACC + 12, PIN_SA + 0, PIN_SB + 3>();
        ds_read_b128_pinned<PIN_NEXT_D + 24, 6144>(da0);
        ds_read_b128_pinned<PIN_NEXT_D + 28, 6144>(da1);
        mfma_scale_pinned<PIN_A + 8, PIN_B + 0, ACC + 16, PIN_SA + 1, PIN_SB + 0>();
        ds_read_b32_pinned<PIN_NEXT_S + 0, 0>(sbase);
        ds_read_b32_pinned<PIN_NEXT_S + 1, 256>(sbase);
        mfma_scale_pinned<PIN_A + 8, PIN_B + 8, ACC + 20, PIN_SA + 1, PIN_SB + 1>();
        ds_read_b32_pinned<PIN_NEXT_S + 2, 512>(sbase);
        ds_read_b32_pinned<PIN_NEXT_S + 3, 768>(sbase);
        // MFMA #6-#15: GMEM->SMEM prefetch spread across MFMA gaps
        mfma_scale_pinned<PIN_A + 8, PIN_B + 16, ACC + 24, PIN_SA + 1, PIN_SB + 2>();
        load_gmem_to_smem_srd<16>(data_srd, ldg_offsets[0], data_m0_0, data_soff_0);
        mfma_scale_pinned<PIN_A + 8, PIN_B + 24, ACC + 28, PIN_SA + 1, PIN_SB + 3>();
        load_gmem_to_smem_srd<16>(data_srd, ldg_offsets[1], data_m0_0 + 1024, data_soff_0);
        mfma_scale_pinned<PIN_A + 16, PIN_B + 0, ACC + 32, PIN_SA + 2, PIN_SB + 0>();
        load_gmem_to_smem_srd<16>(data_srd, ldg_offsets[0], data_m0_1, data_soff_1);
        mfma_scale_pinned<PIN_A + 16, PIN_B + 8, ACC + 36, PIN_SA + 2, PIN_SB + 1>();
        load_gmem_to_smem_srd<16>(data_srd, ldg_offsets[1], data_m0_1 + 1024, data_soff_1);
        mfma_scale_pinned<PIN_A + 16, PIN_B + 16, ACC + 40, PIN_SA + 2, PIN_SB + 2>();
        load_gmem_to_smem_srd<4>(scale_srd, scale_gmem_off, scale_m0_0, scale_soff_0);
        mfma_scale_pinned<PIN_A + 16, PIN_B + 24, ACC + 44, PIN_SA + 2, PIN_SB + 3>();
        load_gmem_to_smem_srd<4>(scale_srd, scale_gmem_off, scale_m0_1, scale_soff_1);
        mfma_scale_pinned<PIN_A + 24, PIN_B + 0, ACC + 48, PIN_SA + 3, PIN_SB + 0>();
        mfma_scale_pinned<PIN_A + 24, PIN_B + 8, ACC + 52, PIN_SA + 3, PIN_SB + 1>();
        mfma_scale_pinned<PIN_A + 24, PIN_B + 16, ACC + 56, PIN_SA + 3, PIN_SB + 2>();
        mfma_scale_pinned<PIN_A + 24, PIN_B + 24, ACC + 60, PIN_SA + 3, PIN_SB + 3>();
    }
};

// ------------------------------------------------------------
// ── v3 Kernel: Shifted-LDG pipeline — zero mid-barrier, 1 barrier/iter ──
// ------------------------------------------------------------
//
// Each phase's LDG overwrites a location read by the PREVIOUS phase's LDS,
// giving ~512 cycles (16 MFMA × 32 cyc) of margin before DMA arrives.
// Phase 1 LDG: tile(ki+2).B0 → cur.B0  (deferred prev Phase 4 / prologue)
// Phase 2 LDG: tile(ki+2).B1 → cur.B1  (deferred Phase 1, overwrites Phase 1's LDS addr)
// Phase 3 LDG: tile(ki+2).A1 → cur.A1  (deferred Phase 2, overwrites Phase 2's LDS addr)
// Phase 4 LDG: tile(ki+3).A0 → next.A0 (deferred Phase 3, overwrites Phase 3's LDS addr)
// tile(ki+3).B0 → next.B0 deferred to next iter Phase 1.
//
template <typename AType, typename BType, typename CType, typename AccType = float>
__global__ __launch_bounds__(256, 1) void turbo_gemm_mxfp8_256x256x128_16x16x128_4wave_kernel(
    const AType *a_ptr, const BType *b_ptr, const uint32_t *a_s_ptr, const uint32_t *b_s_ptr,
    CType *c_ptr, const uint32_t m, const uint32_t n, const uint32_t k) {
#if !defined(__gfx950__)
    assert(false && "turbo_gemm_mxfp8 kernel requires gfx950");
    return;
#else
    using GemmTile =
        GEMM_Tile_MXFP8_NT_256x256x128_16x16x128_4_WAVE_GFX950<AType, BType, CType, AccType>;
    GemmTile tile(threadIdx.x, m, n, k);
    tile.reserve_pinned_regs();

    const uint32_t lane_id = tile.lane_id;
    const uint32_t warp_id = tile.warp_id;
    const uint32_t warp_m  = tile.warp_m;
    const uint32_t warp_n  = tile.warp_n;

    using ASmem                       = typename GemmTile::ASmemSubtile;
    using BSmem                       = typename GemmTile::BSmemSubtile;
    using ASSmem                      = typename GemmTile::AScaleSmemSubtile;
    using BSSmem                      = typename GemmTile::BScaleSmemSubtile;
    constexpr size_t SMEM_DATA_BYTES  = sizeof(ASmem) * 2 * 4 + sizeof(BSmem) * 2 * 4;
    constexpr size_t SMEM_SCALE_BYTES = sizeof(ASSmem) * 2 * 4 + sizeof(BSSmem) * 2 * 4;
    __shared__ char  smem_buf[SMEM_DATA_BYTES + SMEM_SCALE_BYTES];
    auto            *a_smem_tile = reinterpret_cast<ASmem(*)[4]>(smem_buf);
    auto            *b_smem_tile = reinterpret_cast<BSmem(*)[4]>(smem_buf + sizeof(ASmem) * 2 * 4);
    auto            *a_s_smem_tile = reinterpret_cast<ASSmem(*)[4]>(smem_buf + SMEM_DATA_BYTES);
    auto            *b_s_smem_tile =
        reinterpret_cast<BSSmem(*)[4]>(smem_buf + SMEM_DATA_BYTES + sizeof(ASSmem) * 2 * 4);

    int32_t pid_m, pid_n;
    swizzle_pid_m_n<GemmTile::BLOCK_SIZE_M, GemmTile::BLOCK_SIZE_N>(m, n, pid_m, pid_n);
    if (pid_m >= (int32_t) m || pid_n >= (int32_t) n)
        return;

    const AType    *a_base_ptr   = a_ptr + (int64_t) pid_m * k;
    const BType    *b_base_ptr   = b_ptr + (int64_t) pid_n * k;
    const uint32_t  scale_cols   = (k + GemmTile::MX_BLOCK_SIZE - 1) / GemmTile::MX_BLOCK_SIZE;
    const uint32_t *a_s_base_ptr = a_s_ptr + (int64_t) pid_m * scale_cols;
    const uint32_t *b_s_base_ptr = b_s_ptr + (int64_t) pid_n * scale_cols;

    uint32_t ldg_offsets[2];
    tile.compute_ldg_offsets(ldg_offsets, k);
    uint32_t sts_offsets[2];
    tile.compute_sts_offsets(sts_offsets);
    uint32_t lds_offsets[2];
    tile.compute_lds_offsets(lds_offsets);
    const uint32_t scale_ldg_offset = lane_id;
    const uint32_t scale_sts_offset = lane_id;
    const uint32_t scale_lds_offset = lane_id;

    const uint32_t  a_remaining  = (m - pid_m) * k * sizeof(AType);
    const uint32_t  b_remaining  = (n - pid_n) * k * sizeof(BType);
    const uint32_t  as_remaining = (m - pid_m) * scale_cols * sizeof(uint32_t);
    const uint32_t  bs_remaining = (n - pid_n) * scale_cols * sizeof(uint32_t);
    const BufferSRD a_srd(a_base_ptr, a_remaining);
    const BufferSRD b_srd(b_base_ptr, b_remaining);
    const BufferSRD a_s_srd(a_s_base_ptr, as_remaining);
    const BufferSRD b_s_srd(b_s_base_ptr, bs_remaining);

    constexpr int32_t DATA_STRIDE  = GemmTile::BLOCK_SIZE_K;
    constexpr int32_t SCALE_STRIDE = GemmTile::SCALE_FRAG_SIZE * sizeof(uint32_t);

    // ── Load tile 0 → smem[0], tile 1 → smem[1] ──
    tile.template load_a_gmem_to_smem_half_srd<0>(a_srd, ldg_offsets, a_smem_tile[0], sts_offsets);
    tile.template load_a_gmem_to_smem_half_srd<1>(a_srd, ldg_offsets, a_smem_tile[0], sts_offsets);
    tile.template load_b_gmem_to_smem_half_srd<0>(b_srd, ldg_offsets, b_smem_tile[0], sts_offsets);
    tile.template load_b_gmem_to_smem_half_srd<1>(b_srd, ldg_offsets, b_smem_tile[0], sts_offsets);
    tile.template load_a_scale_gmem_to_smem_half_srd<0>(a_s_srd, scale_ldg_offset, a_s_smem_tile[0],
                                                        scale_sts_offset, scale_cols);
    tile.template load_a_scale_gmem_to_smem_half_srd<1>(a_s_srd, scale_ldg_offset, a_s_smem_tile[0],
                                                        scale_sts_offset, scale_cols);
    tile.template load_b_scale_gmem_to_smem_half_srd<0>(b_s_srd, scale_ldg_offset, b_s_smem_tile[0],
                                                        scale_sts_offset, scale_cols);
    tile.template load_b_scale_gmem_to_smem_half_srd<1>(b_s_srd, scale_ldg_offset, b_s_smem_tile[0],
                                                        scale_sts_offset, scale_cols);

    tile.template load_a_gmem_to_smem_half_srd<0>(a_srd, ldg_offsets, a_smem_tile[1], sts_offsets,
                                                  DATA_STRIDE);
    tile.template load_a_gmem_to_smem_half_srd<1>(a_srd, ldg_offsets, a_smem_tile[1], sts_offsets,
                                                  DATA_STRIDE);
    tile.template load_b_gmem_to_smem_half_srd<0>(b_srd, ldg_offsets, b_smem_tile[1], sts_offsets,
                                                  DATA_STRIDE);
    tile.template load_b_gmem_to_smem_half_srd<1>(b_srd, ldg_offsets, b_smem_tile[1], sts_offsets,
                                                  DATA_STRIDE);
    tile.template load_a_scale_gmem_to_smem_half_srd<0>(a_s_srd, scale_ldg_offset, a_s_smem_tile[1],
                                                        scale_sts_offset, scale_cols, SCALE_STRIDE);
    tile.template load_a_scale_gmem_to_smem_half_srd<1>(a_s_srd, scale_ldg_offset, a_s_smem_tile[1],
                                                        scale_sts_offset, scale_cols, SCALE_STRIDE);
    tile.template load_b_scale_gmem_to_smem_half_srd<0>(b_s_srd, scale_ldg_offset, b_s_smem_tile[1],
                                                        scale_sts_offset, scale_cols, SCALE_STRIDE);
    tile.template load_b_scale_gmem_to_smem_half_srd<1>(b_s_srd, scale_ldg_offset, b_s_smem_tile[1],
                                                        scale_sts_offset, scale_cols, SCALE_STRIDE);

    tile.zero_c_agpr();
    wait_vmcnt<0>();
    __builtin_amdgcn_s_barrier();

    uint32_t       cur     = 0;
    uint32_t       next    = 1;
    const uint32_t k_iters = (k + GemmTile::BLOCK_SIZE_K - 1) / GemmTile::BLOCK_SIZE_K;

    // ── Prologue: issue LDS for A0/B0 ──
    load_data_subtile_pinned<GemmTile::PIN_A0>(a_smem_tile[cur][warp_m].u32_ptr(), lds_offsets);
    load_scale_subtile_pinned<GemmTile::PIN_AS0>(a_s_smem_tile[cur][warp_m].u32_ptr(),
                                                 scale_lds_offset);
    load_data_subtile_pinned<GemmTile::PIN_B0>(b_smem_tile[cur][warp_n].u32_ptr(), lds_offsets);
    load_scale_subtile_pinned<GemmTile::PIN_BS0>(b_s_smem_tile[cur][warp_n].u32_ptr(),
                                                 scale_lds_offset);

    if (k_iters > 2) {
        tile.template load_a_gmem_to_smem_half_srd<0>(a_srd, ldg_offsets, a_smem_tile[cur],
                                                      sts_offsets, 2 * DATA_STRIDE);
        tile.template load_a_scale_gmem_to_smem_half_srd<0>(a_s_srd, scale_ldg_offset,
                                                            a_s_smem_tile[cur], scale_sts_offset,
                                                            scale_cols, 2 * SCALE_STRIDE);
        tile.template load_b_gmem_to_smem_half_srd<0>(b_srd, ldg_offsets, b_smem_tile[cur],
                                                      sts_offsets, 2 * DATA_STRIDE);
        tile.template load_b_scale_gmem_to_smem_half_srd<0>(b_s_srd, scale_ldg_offset,
                                                            b_s_smem_tile[cur], scale_sts_offset,
                                                            scale_cols, 2 * SCALE_STRIDE);
    }
    wait_lgkmcnt<0>();

    int32_t base_data_soff[4], base_scale_soff[4];
    tile.precompute_base_soff(base_data_soff, base_scale_soff, scale_cols);

    const uint32_t sts_wb =
        __builtin_amdgcn_readfirstlane(warp_id * GemmTile::MFMA_SIZE_M * GemmTile::MFMA_SIZE_K);
    const uint32_t s_smem_off = __builtin_amdgcn_readfirstlane((warp_id * 64 + scale_sts_offset) *
                                                               (uint32_t) sizeof(uint32_t));
    const uint32_t scale_gmem_byte_off = scale_ldg_offset * (uint32_t) sizeof(uint32_t);

    // ── Main loop ──
    const uint32_t main_iters = k_iters > 3 ? k_iters - 3 : 0;
    int32_t        data_off = 2 * DATA_STRIDE, scale_off = 2 * SCALE_STRIDE;
    for (uint32_t ki = 0; ki < main_iters;
         ++ki, data_off += DATA_STRIDE, scale_off += SCALE_STRIDE) {
        const int32_t next_data_off  = data_off + DATA_STRIDE;
        const int32_t next_scale_off = scale_off + SCALE_STRIDE;

        // Phase 1: MFMA A0×B0, LDG B1→cur[2,3]
        {
            uint32_t dm0_0 = __builtin_amdgcn_readfirstlane(b_smem_tile[cur][2].u32_ptr()) + sts_wb;
            uint32_t dm0_1 = __builtin_amdgcn_readfirstlane(b_smem_tile[cur][3].u32_ptr()) + sts_wb;
            uint32_t sm0_0 =
                __builtin_amdgcn_readfirstlane(b_s_smem_tile[cur][2].u32_ptr()) + s_smem_off;
            uint32_t sm0_1 =
                __builtin_amdgcn_readfirstlane(b_s_smem_tile[cur][3].u32_ptr()) + s_smem_off;
            GemmTile::template phase_mfma_lds_ldg<GemmTile::PIN_A0, GemmTile::PIN_AS0,
                                                  GemmTile::PIN_B0, GemmTile::PIN_BS0, 0, 0,
                                                  GemmTile::PIN_B1, GemmTile::PIN_BS1>(
                b_smem_tile[cur][warp_n + 2].u32_ptr(), lds_offsets,
                b_s_smem_tile[cur][warp_n + 2].u32_ptr(), scale_lds_offset, b_srd, ldg_offsets,
                dm0_0, dm0_1, base_data_soff[2] + data_off, base_data_soff[3] + data_off, b_s_srd,
                scale_gmem_byte_off, sm0_0, sm0_1, base_scale_soff[2] + scale_off,
                base_scale_soff[3] + scale_off);
        }

        // Phase 2: MFMA A0×B1, LDG A1→cur[2,3]
        {
            uint32_t dm0_0 = __builtin_amdgcn_readfirstlane(a_smem_tile[cur][2].u32_ptr()) + sts_wb;
            uint32_t dm0_1 = __builtin_amdgcn_readfirstlane(a_smem_tile[cur][3].u32_ptr()) + sts_wb;
            uint32_t sm0_0 =
                __builtin_amdgcn_readfirstlane(a_s_smem_tile[cur][2].u32_ptr()) + s_smem_off;
            uint32_t sm0_1 =
                __builtin_amdgcn_readfirstlane(a_s_smem_tile[cur][3].u32_ptr()) + s_smem_off;
            GemmTile::template phase_mfma_lds_ldg<GemmTile::PIN_A0, GemmTile::PIN_AS0,
                                                  GemmTile::PIN_B1, GemmTile::PIN_BS1, 0, 1,
                                                  GemmTile::PIN_A1, GemmTile::PIN_AS1>(
                a_smem_tile[cur][warp_m + 2].u32_ptr(), lds_offsets,
                a_s_smem_tile[cur][warp_m + 2].u32_ptr(), scale_lds_offset, a_srd, ldg_offsets,
                dm0_0, dm0_1, base_data_soff[2] + data_off, base_data_soff[3] + data_off, a_s_srd,
                scale_gmem_byte_off, sm0_0, sm0_1, base_scale_soff[2] + scale_off,
                base_scale_soff[3] + scale_off);
        }

        // Phase 3: MFMA A1×B0, LDG A0→next[0,1]
        {
            uint32_t dm0_0 =
                __builtin_amdgcn_readfirstlane(a_smem_tile[next][0].u32_ptr()) + sts_wb;
            uint32_t dm0_1 =
                __builtin_amdgcn_readfirstlane(a_smem_tile[next][1].u32_ptr()) + sts_wb;
            uint32_t sm0_0 =
                __builtin_amdgcn_readfirstlane(a_s_smem_tile[next][0].u32_ptr()) + s_smem_off;
            uint32_t sm0_1 =
                __builtin_amdgcn_readfirstlane(a_s_smem_tile[next][1].u32_ptr()) + s_smem_off;
            GemmTile::template phase_mfma_lds_ldg<GemmTile::PIN_A1, GemmTile::PIN_AS1,
                                                  GemmTile::PIN_B0, GemmTile::PIN_BS0, 1, 0,
                                                  GemmTile::PIN_A0, GemmTile::PIN_AS0>(
                a_smem_tile[next][warp_m].u32_ptr(), lds_offsets,
                a_s_smem_tile[next][warp_m].u32_ptr(), scale_lds_offset, a_srd, ldg_offsets, dm0_0,
                dm0_1, base_data_soff[0] + next_data_off, base_data_soff[1] + next_data_off,
                a_s_srd, scale_gmem_byte_off, sm0_0, sm0_1, base_scale_soff[0] + next_scale_off,
                base_scale_soff[1] + next_scale_off);
        }

        // Phase 4: MFMA A1×B1, LDG B0→next[0,1]
        {
            uint32_t dm0_0 =
                __builtin_amdgcn_readfirstlane(b_smem_tile[next][0].u32_ptr()) + sts_wb;
            uint32_t dm0_1 =
                __builtin_amdgcn_readfirstlane(b_smem_tile[next][1].u32_ptr()) + sts_wb;
            uint32_t sm0_0 =
                __builtin_amdgcn_readfirstlane(b_s_smem_tile[next][0].u32_ptr()) + s_smem_off;
            uint32_t sm0_1 =
                __builtin_amdgcn_readfirstlane(b_s_smem_tile[next][1].u32_ptr()) + s_smem_off;
            GemmTile::template phase_mfma_lds_ldg<GemmTile::PIN_A1, GemmTile::PIN_AS1,
                                                  GemmTile::PIN_B1, GemmTile::PIN_BS1, 1, 1,
                                                  GemmTile::PIN_B0, GemmTile::PIN_BS0>(
                b_smem_tile[next][warp_n].u32_ptr(), lds_offsets,
                b_s_smem_tile[next][warp_n].u32_ptr(), scale_lds_offset, b_srd, ldg_offsets, dm0_0,
                dm0_1, base_data_soff[0] + next_data_off, base_data_soff[1] + next_data_off,
                b_s_srd, scale_gmem_byte_off, sm0_0, sm0_1, base_scale_soff[0] + next_scale_off,
                base_scale_soff[1] + next_scale_off);
        }

        wait_vmcnt<12>();
        __builtin_amdgcn_s_barrier();
        cur ^= 1;
        next ^= 1;
    }

    // ── Epilogue 1: last LDG tile — Phase 1+2 prefetch B1+A1, Phase 3+4 compute only ──
    {
        {
            uint32_t dm0_0 = __builtin_amdgcn_readfirstlane(b_smem_tile[cur][2].u32_ptr()) + sts_wb;
            uint32_t dm0_1 = __builtin_amdgcn_readfirstlane(b_smem_tile[cur][3].u32_ptr()) + sts_wb;
            uint32_t sm0_0 =
                __builtin_amdgcn_readfirstlane(b_s_smem_tile[cur][2].u32_ptr()) + s_smem_off;
            uint32_t sm0_1 =
                __builtin_amdgcn_readfirstlane(b_s_smem_tile[cur][3].u32_ptr()) + s_smem_off;
            GemmTile::template phase_mfma_lds_ldg<GemmTile::PIN_A0, GemmTile::PIN_AS0,
                                                  GemmTile::PIN_B0, GemmTile::PIN_BS0, 0, 0,
                                                  GemmTile::PIN_B1, GemmTile::PIN_BS1>(
                b_smem_tile[cur][warp_n + 2].u32_ptr(), lds_offsets,
                b_s_smem_tile[cur][warp_n + 2].u32_ptr(), scale_lds_offset, b_srd, ldg_offsets,
                dm0_0, dm0_1, base_data_soff[2] + data_off, base_data_soff[3] + data_off, b_s_srd,
                scale_gmem_byte_off, sm0_0, sm0_1, base_scale_soff[2] + scale_off,
                base_scale_soff[3] + scale_off);
        }

        {
            uint32_t dm0_0 = __builtin_amdgcn_readfirstlane(a_smem_tile[cur][2].u32_ptr()) + sts_wb;
            uint32_t dm0_1 = __builtin_amdgcn_readfirstlane(a_smem_tile[cur][3].u32_ptr()) + sts_wb;
            uint32_t sm0_0 =
                __builtin_amdgcn_readfirstlane(a_s_smem_tile[cur][2].u32_ptr()) + s_smem_off;
            uint32_t sm0_1 =
                __builtin_amdgcn_readfirstlane(a_s_smem_tile[cur][3].u32_ptr()) + s_smem_off;
            GemmTile::template phase_mfma_lds_ldg<GemmTile::PIN_A0, GemmTile::PIN_AS0,
                                                  GemmTile::PIN_B1, GemmTile::PIN_BS1, 0, 1,
                                                  GemmTile::PIN_A1, GemmTile::PIN_AS1>(
                a_smem_tile[cur][warp_m + 2].u32_ptr(), lds_offsets,
                a_s_smem_tile[cur][warp_m + 2].u32_ptr(), scale_lds_offset, a_srd, ldg_offsets,
                dm0_0, dm0_1, base_data_soff[2] + data_off, base_data_soff[3] + data_off, a_s_srd,
                scale_gmem_byte_off, sm0_0, sm0_1, base_scale_soff[2] + scale_off,
                base_scale_soff[3] + scale_off);
        }

        GemmTile::template phase_mfma_lds<GemmTile::PIN_A1, GemmTile::PIN_AS1, GemmTile::PIN_B0,
                                          GemmTile::PIN_BS0, 1, 0, GemmTile::PIN_A0,
                                          GemmTile::PIN_AS0>(
            a_smem_tile[next][warp_m].u32_ptr(), lds_offsets, a_s_smem_tile[next][warp_m].u32_ptr(),
            scale_lds_offset);
        GemmTile::template phase_mfma_lds<GemmTile::PIN_A1, GemmTile::PIN_AS1, GemmTile::PIN_B1,
                                          GemmTile::PIN_BS1, 1, 1, GemmTile::PIN_B0,
                                          GemmTile::PIN_BS0>(
            b_smem_tile[next][warp_n].u32_ptr(), lds_offsets, b_s_smem_tile[next][warp_n].u32_ptr(),
            scale_lds_offset);

        wait_vmcnt<6>();
        __builtin_amdgcn_s_barrier();
        cur ^= 1;
        next ^= 1;
    }

    // ── Epilogue 2: no LDG, LDS from both buffers ──
    {
        GemmTile::template phase_mfma_lds<GemmTile::PIN_A0, GemmTile::PIN_AS0, GemmTile::PIN_B0,
                                          GemmTile::PIN_BS0, 0, 0, GemmTile::PIN_B1,
                                          GemmTile::PIN_BS1>(
            b_smem_tile[cur][warp_n + 2].u32_ptr(), lds_offsets,
            b_s_smem_tile[cur][warp_n + 2].u32_ptr(), scale_lds_offset);

        GemmTile::template phase_mfma_lds<GemmTile::PIN_A0, GemmTile::PIN_AS0, GemmTile::PIN_B1,
                                          GemmTile::PIN_BS1, 0, 1, GemmTile::PIN_A1,
                                          GemmTile::PIN_AS1>(
            a_smem_tile[cur][warp_m + 2].u32_ptr(), lds_offsets,
            a_s_smem_tile[cur][warp_m + 2].u32_ptr(), scale_lds_offset);

        wait_vmcnt<0>();
        __builtin_amdgcn_s_barrier();

        GemmTile::template phase_mfma_lds<GemmTile::PIN_A1, GemmTile::PIN_AS1, GemmTile::PIN_B0,
                                          GemmTile::PIN_BS0, 1, 0, GemmTile::PIN_A0,
                                          GemmTile::PIN_AS0>(
            a_smem_tile[next][warp_m].u32_ptr(), lds_offsets, a_s_smem_tile[next][warp_m].u32_ptr(),
            scale_lds_offset);

        GemmTile::template phase_mfma_lds<GemmTile::PIN_A1, GemmTile::PIN_AS1, GemmTile::PIN_B1,
                                          GemmTile::PIN_BS1, 1, 1, GemmTile::PIN_B0,
                                          GemmTile::PIN_BS0>(
            b_smem_tile[next][warp_n].u32_ptr(), lds_offsets, b_s_smem_tile[next][warp_n].u32_ptr(),
            scale_lds_offset);

        cur ^= 1;
        next ^= 1;
    }

    // ── Epilogue 3: no LDG, no LDS from next ──
    {
        GemmTile::template phase_mfma_lds<GemmTile::PIN_A0, GemmTile::PIN_AS0, GemmTile::PIN_B0,
                                          GemmTile::PIN_BS0, 0, 0, GemmTile::PIN_B1,
                                          GemmTile::PIN_BS1>(
            b_smem_tile[cur][warp_n + 2].u32_ptr(), lds_offsets,
            b_s_smem_tile[cur][warp_n + 2].u32_ptr(), scale_lds_offset);

        GemmTile::template phase_mfma_lds<GemmTile::PIN_A0, GemmTile::PIN_AS0, GemmTile::PIN_B1,
                                          GemmTile::PIN_BS1, 0, 1, GemmTile::PIN_A1,
                                          GemmTile::PIN_AS1>(
            a_smem_tile[cur][warp_m + 2].u32_ptr(), lds_offsets,
            a_s_smem_tile[cur][warp_m + 2].u32_ptr(), scale_lds_offset);

        GemmTile::template phase_mfma_only<GemmTile::PIN_A1, GemmTile::PIN_AS1, GemmTile::PIN_B0,
                                           GemmTile::PIN_BS0, 1, 0>();
        GemmTile::template phase_mfma_only<GemmTile::PIN_A1, GemmTile::PIN_AS1, GemmTile::PIN_B1,
                                           GemmTile::PIN_BS1, 1, 1>();
    }

    // ── Store C ──
    __builtin_amdgcn_sched_barrier(0);
    uint32_t c_stg_offsets[4];
    tile.compute_stg_offsets(c_stg_offsets);
    CType *c_stg_base_ptr =
        c_ptr + (int64_t) pid_m * n + pid_n + warp_id / 2 * 64 * n + warp_id % 2 * 64;
    const bool is_boundary_tile = (pid_m + 256 > (int32_t) m) || (pid_n + 256 > (int32_t) n);

    if (!is_boundary_tile) {
        fp32x4_t c_tmp[4][4];
        tile.template read_c_subtile_from_agpr<0, 0>(c_tmp);
        tile.store_c_subtile(c_stg_base_ptr + 0 * 128 * n + 0 * 128, n, c_tmp, c_stg_offsets);
        tile.template read_c_subtile_from_agpr<0, 1>(c_tmp);
        tile.store_c_subtile(c_stg_base_ptr + 0 * 128 * n + 1 * 128, n, c_tmp, c_stg_offsets);
        tile.template read_c_subtile_from_agpr<1, 0>(c_tmp);
        tile.store_c_subtile(c_stg_base_ptr + 1 * 128 * n + 0 * 128, n, c_tmp, c_stg_offsets);
        tile.template read_c_subtile_from_agpr<1, 1>(c_tmp);
        tile.store_c_subtile(c_stg_base_ptr + 1 * 128 * n + 1 * 128, n, c_tmp, c_stg_offsets);
    } else {
        const int32_t warp_base_m  = warp_id / 2 * 64;
        const int32_t warp_base_n  = warp_id % 2 * 64;
        const int32_t tile_valid_m = min((int32_t) m - pid_m, 256) - warp_base_m;
        const int32_t tile_valid_n = min((int32_t) n - pid_n, 256) - warp_base_n;
        fp32x4_t      c_tmp[4][4];
        tile.template read_c_subtile_from_agpr<0, 0>(c_tmp);
        tile.store_c_subtile(c_stg_base_ptr + 0 * 128 * n + 0 * 128, n, c_tmp, c_stg_offsets,
                             tile_valid_m, tile_valid_n);
        tile.template read_c_subtile_from_agpr<0, 1>(c_tmp);
        tile.store_c_subtile(c_stg_base_ptr + 0 * 128 * n + 1 * 128, n, c_tmp, c_stg_offsets,
                             tile_valid_m, tile_valid_n - 128);
        tile.template read_c_subtile_from_agpr<1, 0>(c_tmp);
        tile.store_c_subtile(c_stg_base_ptr + 1 * 128 * n + 0 * 128, n, c_tmp, c_stg_offsets,
                             tile_valid_m - 128, tile_valid_n);
        tile.template read_c_subtile_from_agpr<1, 1>(c_tmp);
        tile.store_c_subtile(c_stg_base_ptr + 1 * 128 * n + 1 * 128, n, c_tmp, c_stg_offsets,
                             tile_valid_m - 128, tile_valid_n - 128);
    }
#endif // __gfx950__
}

// ── Pre-shuffle scale kernel (16×4 column-major reorder with optional type conversion) ──

template <typename InT, typename OutT>
__global__ void preshuffle_scale_16x4_kernel(const InT *in_scale_ptr, OutT *out_scale_ptr,
                                             const int rows, const int cols) {
    (void) rows;
    const int BLOCK_SIZE_ROW = 16;
    const int BLOCK_SIZE_COL = 4;
    const int tid            = threadIdx.x;
    const int bid            = blockIdx.x;

    in_scale_ptr  = in_scale_ptr + bid * BLOCK_SIZE_ROW * cols;
    out_scale_ptr = out_scale_ptr + bid * BLOCK_SIZE_ROW * cols;

    for (int i = 0; i < (cols / BLOCK_SIZE_COL); ++i) {
        const OutT val     = static_cast<OutT>(in_scale_ptr[tid % 16 * cols + tid / 16]);
        out_scale_ptr[tid] = val;
        in_scale_ptr += 4;
        out_scale_ptr += BLOCK_SIZE_ROW * BLOCK_SIZE_COL;
    }
}

} // namespace turbo
} // namespace primus_turbo
