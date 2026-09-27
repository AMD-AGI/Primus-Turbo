	.amdgcn_target "amdgcn-amd-amdhsa-unknown-gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	kn_fmha_fwd_prefill_a16w16_m32x8_bshd_0
	.p2align	8
	.type	kn_fmha_fwd_prefill_a16w16_m32x8_bshd_0,@function
kn_fmha_fwd_prefill_a16w16_m32x8_bshd_0:
	global_prefetch_b8 v0, s[0:1] scope:SCOPE_SE
	v_nop
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 0, 2), 2
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_load_b96 s[24:26], s[0:1], 0xa0 nv
	s_bfe_u32 s2, ttmp6, 0x40010
	s_and_b32 s3, ttmp7, 0xffff
	s_add_co_i32 s2, s2, 1
	s_bfe_u32 s5, ttmp6, 0x4000c
	s_mul_i32 s2, s3, s2
	s_bfe_u32 s4, ttmp6, 0x40004
	s_add_co_i32 s5, s5, 1
	s_bfe_u32 s6, ttmp6, 0x40014
	s_add_co_i32 s2, s4, s2
	s_and_b32 s4, ttmp6, 15
	s_mul_i32 s5, ttmp9, s5
	s_lshr_b32 s7, ttmp7, 16
	s_add_co_i32 s6, s6, 1
	s_add_co_i32 s4, s4, s5
	s_mul_i32 s5, s7, s6
	s_bfe_u32 s6, ttmp6, 0x40008
	s_getreg_b32 s8, hwreg(HW_REG_IB_STS2, 6, 4)
	s_add_co_i32 s6, s6, s5
	s_cmp_eq_u32 s8, 0
	s_set_vgpr_msb 64
	v_and_b32_e32 v141 /*v397*/, 31, v0
	s_cselect_b32 s5, s7, s6
	s_cselect_b32 s6, ttmp9, s4
	s_wait_kmcnt 0x0
	s_mul_i32 s4, s26, s25
	s_cselect_b32 s2, s3, s2
	s_abs_i32 s3, s4
	s_mul_i32 s5, s25, s5
	s_cvt_f32_u32 s7, s3
	s_add_co_i32 s2, s5, s2
	s_sub_co_i32 s5, 0, s3
	s_mul_i32 s2, s2, s24
	v_s_rcp_f32 s7, s7
	v_bfe_u32 v136 /*v392*/, v0, 4, 1
	s_set_vgpr_msb 0x4044
	v_lshlrev_b32_e32 v146 /*v402*/, 1, v141 /*v397*/
	s_mov_b32 s20, 1
	s_mov_b32 s48, 16
	s_mov_b32 s59, 0
	s_mov_b32 s37, 0xffff0000
	s_mov_b32 s40, 0x80004
	s_mul_f32 s7, s7, 0x4f7ffffe
	v_lshlrev_b32_e32 v143 /*v399*/, 3, v136 /*v392*/
	s_cvt_u32_f32 s7, s7
	s_mul_i32 s8, s5, s7
	s_add_co_i32 s5, s2, s6
	s_mul_hi_u32 s2, s7, s8
	s_abs_i32 s6, s5
	s_add_co_i32 s7, s7, s2
	s_mul_hi_u32 s2, s6, s7
	s_xor_b32 s7, s5, s4
	s_mul_i32 s8, s2, s3
	s_ashr_i32 s9, s7, 31
	s_sub_co_i32 s6, s6, s8
	s_add_co_i32 s8, s2, 1
	s_sub_co_i32 s10, s6, s3
	s_cmp_ge_u32 s6, s3
	s_cselect_b32 s2, s8, s2
	s_cselect_b32 s6, s10, s6
	s_add_co_i32 s8, s2, 1
	s_cmp_ge_u32 s6, s3
	s_cselect_b32 s2, s8, s2
	s_xor_b32 s2, s2, s9
	s_sub_co_i32 s3, s2, s9
	s_mul_i32 s3, s3, s4
	s_cmp_lg_u32 s5, s3
	s_cselect_b32 s3, -1, 0
	s_cmp_lt_i32 s7, 0
	s_cselect_b32 s6, -1, 0
	s_and_b32 s3, s3, s6
	s_sub_co_ci_u32 s2, s2, s9
	s_abs_i32 s21, s25
	s_mul_i32 s4, s2, s4
	s_cvt_f32_u32 s3, s21
	s_sub_co_i32 s7, 0, s21
	s_sub_co_i32 s27, s5, s4
	v_s_rcp_f32 s3, s3
	s_abs_i32 s22, s27
	s_xor_b32 s28, s27, s25
	s_ashr_i32 s29, s28, 31
	s_mul_f32 s6, s3, 0x4f7ffffe
	s_mov_b32 s3, -1
	s_cvt_u32_f32 s6, s6
	s_mul_i32 s7, s7, s6
	s_mul_hi_u32 s4, s6, s7
	s_add_co_i32 s23, s6, s4
	s_load_b512 s[4:19], s[0:1], 0x5c nv
	s_mul_hi_u32 s26, s22, s23
	s_mul_i32 s23, s26, s21
	s_add_co_i32 s31, s26, 1
	s_sub_co_i32 s30, s22, s23
	s_clause 0x3
	s_load_b64 s[22:23], s[0:1], 0x10 nv
	s_load_b64 s[66:67], s[0:1], 0x20 nv
	s_load_b64 s[68:69], s[0:1], 0x30 nv
	s_load_b64 s[70:71], s[0:1], 0x50 nv
	s_sub_co_i32 s33, s30, s21
	s_cmp_ge_u32 s30, s21
	s_cselect_b32 s26, s31, s26
	s_cselect_b32 s30, s33, s30
	s_add_co_i32 s31, s26, 1
	s_cmp_ge_u32 s30, s21
	s_cselect_b32 s21, s31, s26
	s_xor_b32 s21, s21, s29
	s_sub_co_i32 s26, s21, s29
	s_wait_kmcnt 0x0
	s_mov_b32 s62, s6
	s_mul_i32 s25, s26, s25
	s_mov_b32 s26, s5
	s_cmp_lg_u32 s27, s25
	s_mov_b32 s38, s11
	s_cselect_b32 s30, -1, 0
	s_cmp_lt_i32 s28, 0
	s_mov_b32 s28, s9
	s_cselect_b32 s31, -1, 0
	s_mov_b32 s64, s7
	s_and_b32 s30, s30, s31
	s_sub_co_ci_u32 s33, s21, s29
	s_bfe_u32 s102, ttmp8, 0x50019
	s_mul_i32 s91, s33, s18
	s_and_b32 s21, s102, 30
	s_lshr_b32 s36, s102, 1
	s_cmp_lg_u32 s102, s21
	s_mul_i32 s96, s102, 0x2200
	s_cselect_b32 s21, -1, 0
	s_cmp_lt_i32 s102, 0
	s_mov_b32 s30, s10
	s_cselect_b32 s29, -1, 0
	s_mul_i32 s97, s33, s19
	s_and_b32 s21, s29, s21
	v_cvt_pk_bf16_f32 v145 /*v401*/, s4, s4
	s_and_b32 s21, s21, exec_lo
	s_cselect_b32 s49, 1, 0
	s_not_b32 s2, s2
	s_sub_co_i32 s42, s27, s25
	s_add_co_i32 s31, s24, s2
	s_lshl_b32 s2, s102, 5
	s_lshl_b32 s39, s31, 7
	s_lshl_b32 s34, s42, 2
	s_add_co_i32 s95, s39, s2
	s_set_vgpr_msb 0x4440
	v_and_b32_e32 v140 /*v396*/, 15, v0
	s_cmp_lt_i32 s95, 0
	s_cselect_b32 s94, -1, 0
	s_ashr_i32 s2, s95, 31
	s_ashr_i32 s24, s95, 2
	s_lshr_b32 s2, s2, 30
	s_set_vgpr_msb 0x4000
	v_and_b32_e32 v1, 16, v0
	s_set_vgpr_msb 4
	v_or_b32_e32 v0, s95, v140 /*v396*/
	s_add_co_i32 s44, s91, s24
	s_ashr_i32 s27, s5, 31
	s_ashr_i32 s35, s34, 31
	s_ashr_i32 s29, s9, 31
	s_set_vgpr_msb 0x400
	v_or_b32_e32 v2, 16, v0
	s_ashr_i32 s45, s44, 31
	s_sub_co_i32 s25, s18, s24
	s_mul_u64 s[46:47], s[34:35], s[28:29]
	s_mul_u64 s[44:45], s[44:45], s[26:27]
	v_add_nc_u32_e32 v3, s2, v2
	s_set_vgpr_msb 0x44
	v_mad_u32_u24 v142 /*v398*/, 0x110, v140 /*v396*/, v1
	s_set_vgpr_msb 0x4400
	v_ashrrev_i32_e32 v1, 31, v0
	s_add_co_i32 s21, s96, 0x20000
	s_max_i32 s24, s25, 0
	v_and_b32_e32 v4, -4, v3
	s_lshl_b64 s[44:45], s[44:45], 1
	v_lshrrev_b32_e32 v1, 30, v1
	s_lshl_b64 s[46:47], s[46:47], 1
	s_add_nc_u64 s[22:23], s[22:23], s[44:45]
	v_cmp_ne_u32_e32 vcc_lo, v2, v4
	v_dual_ashrrev_i32 v2, 2, v3 :: v_dual_add_nc_u32 v1, v0, v1
	s_set_vgpr_msb 0x44
	v_add_nc_u32_e32 v147 /*v403*/, s21, v142 /*v398*/
	v_or_b32_e32 v144 /*v400*/, 0x20000, v142 /*v398*/
	s_and_b32 vcc_lo, s94, vcc_lo
	s_add_nc_u64 s[22:23], s[22:23], s[46:47]
	s_set_vgpr_msb 0x4400
	v_and_b32_e32 v5, -4, v1
	s_set_vgpr_msb 64
	v_subrev_co_ci_u32_e64 v137 /*v393*/, null, 0, v2, vcc_lo
	s_set_vgpr_msb 0x4000
	v_ashrrev_i32_e32 v1, 2, v1
	s_bitset1_b32 s23, 31
	v_cmp_ne_u32_e64 s2, v0, v5
	v_sub_nc_u32_e32 v0, v0, v5
	s_and_b32 vcc_lo, s94, s2
	s_cmp_lg_u32 s5, 0x80000000
	s_set_vgpr_msb 64
	v_add_nc_u32_e32 v139 /*v395*/, s34, v0
	s_cselect_b32 s27, s27, 0
	s_cselect_b32 s26, s5, 0x200
	s_cmp_lg_u32 s9, 0x80000000
	v_subrev_co_ci_u32_e64 v138 /*v394*/, null, 0, v1, vcc_lo
	s_cselect_b32 s41, s9, 0x80
	s_cselect_b32 s2, s29, 0
	s_bfe_i32 s9, s31, 0x10018
	s_ashr_i32 s63, s6, 31
	s_lshr_b32 s6, s9, 30
	s_lshl_b32 s5, s26, 16
	s_or_b32 s6, s6, s39
	s_lshr_b64 s[28:29], s[26:27], 16
	s_addk_co_i32 s6, 0x7f
	s_lshr_b32 s25, s26, 16
	s_ashr_i32 s6, s6, 2
	s_add_co_i32 s26, s18, -1
	s_add_co_i32 s6, s6, s9
	s_sub_co_i32 s99, s19, s18
	s_min_i32 s100, s6, s26
	s_ashr_i32 s35, s39, 2
	s_add_co_i32 s100, s100, s99
	s_ashr_i32 s43, s42, 31
	s_add_co_i32 s9, s17, s100
	s_ashr_i32 s31, s10, 31
	s_add_co_i32 s9, s9, 1
	s_ashr_i32 s39, s11, 31
	s_min_i32 s9, s9, s19
	s_and_b32 s2, s2, 0xffff
	s_add_co_i32 s101, s35, s99
	s_max_i32 s9, s9, 1
	s_ashr_i32 s65, s7, 31
	s_mul_u64 s[6:7], s[42:43], s[30:31]
	s_mul_u64 s[26:27], s[42:43], s[38:39]
	s_or_b32 s42, s2, s5
	s_sub_co_i32 s2, s101, s16
	s_add_co_i32 s5, s9, 63
	s_max_i32 s2, s2, 0
	s_lshr_b32 s10, s5, 6
	s_lshr_b32 s2, s2, 6
	s_add_co_i32 s98, s10, -1
	s_lshl_b64 s[72:73], s[6:7], 1
	s_min_u32 s60, s2, s98
	s_lshl_b64 s[74:75], s[26:27], 1
	s_lshl_b32 s2, s60, 6
	s_and_b32 s11, s28, 0xffff0000
	s_add_co_i32 s6, s2, s97
	s_sub_co_i32 s2, s9, s2
	s_ashr_i32 s7, s6, 31
	s_set_vgpr_msb 0x4000
	v_med3_i32 v0, s2, 0, 64
	s_mul_u64 s[26:27], s[6:7], s[62:63]
	s_mul_u64 s[6:7], s[6:7], s[64:65]
	s_lshl_b64 s[26:27], s[26:27], 1
	s_lshl_b64 s[6:7], s[6:7], 1
	v_readfirstlane_b32 s103, v0
	s_add_nc_u64 s[26:27], s[66:67], s[26:27]
	s_add_nc_u64 s[6:7], s[68:69], s[6:7]
	s_or_b32 s43, s11, s25
	s_cmp_lg_u32 s36, s49
	s_add_nc_u64 s[78:79], s[26:27], s[72:73]
	s_add_nc_u64 s[76:77], s[6:7], s[74:75]
	s_mov_b32 s39, 0x807fff
	s_mov_b32 s38, 0xffff7fff
	s_mov_b32 s36, 0x7510000
	s_cbranch_scc0 .LBB0_10
	s_mov_b32 s25, s59
	s_mov_b32 s26, s59
	s_mov_b32 s27, s59
	s_mov_b32 s56, s59
	s_mov_b32 s57, s59
	s_mov_b32 s58, s59
	s_cmp_lg_u32 s62, 0x80000000
	tensor_load_to_lds s[20:23], s[36:43], s[24:27], s[56:59]
	s_mov_b64 s[4:5], s[20:21]
	s_mov_b64 s[6:7], s[22:23]
	s_cselect_b32 s7, s63, 0
	s_cselect_b32 s57, s62, 0x80
	s_and_b32 s5, s102, 3
	s_and_b32 s58, s7, 0xffff
	s_lshl_b32 s25, s5, 4
	s_lshl_b32 s2, s5, 5
	s_sub_co_i32 s11, s103, s25
	s_mov_b32 s3, s59
	s_max_i32 s11, s11, 0
	s_mov_b64 s[30:31], s[22:23]
	s_lshl_b32 s11, s11, 16
	s_mov_b32 s6, s57
	s_or_b32 s54, s11, 0x7fff
	s_cmp_lg_u32 s64, 0x80000000
	s_mul_u64 s[26:27], s[6:7], s[2:3]
	s_cselect_b32 s49, s64, 0x80
	s_cselect_b32 s31, s65, 0
	s_mov_b32 s30, s49
	s_mul_i32 vcc_hi, s5, 0x1200
	s_mul_u64 s[80:81], s[30:31], s[2:3]
	s_mul_i32 s104, s5, 0x1100
	s_add_nc_u64 s[6:7], s[26:27], s[78:79]
	s_mov_b32 s55, 0x800000
	s_bitset1_b32 vcc_hi, 16
	s_add_nc_u64 s[2:3], s[80:81], s[76:77]
	s_mov_b32 s52, s36
	s_mov_b32 s53, s37
	s_mov_b64 s[28:29], s[20:21]
	s_mov_b32 s56, s48
	s_mov_b32 s5, s104
	s_bitset1_b32 s7, 31
	s_mov_b32 s44, 0xf510000
	s_mov_b32 s45, s37
	s_mov_b32 s51, s59
	s_mov_b32 s47, s55
	s_mov_b32 s46, s54
	s_mov_b32 s29, vcc_hi
	s_and_b32 s50, s31, 0xffff
	s_or_b32 s31, s3, 0x80000000
	s_mov_b32 s30, s2
	s_add_co_i32 s2, s101, s17
	s_set_vgpr_msb 0x44
	v_or_b32_e32 v152 /*v408*/, 0x20000, v142 /*v398*/
	s_max_i32 s2, s2, 0
	s_mov_b32 s89, s59
	s_add_co_i32 s2, s2, 1
	s_add_nc_u64 s[84:85], s[68:69], s[74:75]
	s_ashr_i32 s3, s2, 31
	s_add_nc_u64 s[86:87], s[66:67], s[72:73]
	s_lshr_b32 s3, s3, 26
	s_add_co_i32 s3, s2, s3
	s_wait_tensorcnt 0x0
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x4411
	ds_load_b128 v[0:3], v147 /*v403*/
	ds_load_b128 v[4:7], v147 /*v403*/ offset:32
	ds_load_b128 v[8:11], v147 /*v403*/ offset:64
	ds_load_b128 v[12:15], v147 /*v403*/ offset:96
	ds_load_b128 v[16:19], v147 /*v403*/ offset:128
	ds_load_b128 v[20:23], v147 /*v403*/ offset:160
	ds_load_b128 v[24:27], v147 /*v403*/ offset:192
	ds_load_b128 v[28:31], v147 /*v403*/ offset:224
	ds_load_b128 v[32:35], v147 /*v403*/ offset:4352
	ds_load_b128 v[36:39], v147 /*v403*/ offset:4384
	ds_load_b128 v[40:43], v147 /*v403*/ offset:4416
	ds_load_b128 v[44:47], v147 /*v403*/ offset:4448
	ds_load_b128 v[48:51], v147 /*v403*/ offset:4480
	ds_load_b128 v[52:55], v147 /*v403*/ offset:4512
	ds_load_b128 v[56:59], v147 /*v403*/ offset:4544
	ds_load_b128 v[60:63], v147 /*v403*/ offset:4576
	tensor_load_to_lds s[4:7], s[52:59]
	tensor_load_to_lds s[28:31], s[44:51]
	s_and_b32 s4, s3, 0xffffffc0
	s_ashr_i32 s3, s3, 6
	s_cmp_lg_u32 s2, s4
	s_wait_dscnt 0xf
	v_pk_mul_bf16 v128, v145 /*v401*/, v0
	s_cselect_b32 s4, -1, 0
	s_cmp_lt_i32 s2, 0
	v_and_or_b32 v0, v141 /*v397*/, 7, v143 /*v399*/
	s_cselect_b32 s2, -1, 0
	s_wait_dscnt 0xe
	v_pk_mul_bf16 v135, v145 /*v401*/, v7
	s_and_b32 s2, s2, s4
	s_sub_co_ci_u32 s2, s3, 0
	s_sub_co_i32 s3, s100, s16
	s_min_i32 s7, s2, s98
	s_max_i32 s3, s3, 0
	s_max_i32 s82, s7, s60
	s_add_co_i32 s3, s3, 63
	v_mul_u32_u24_e32 v0, 0x120, v0
	s_ashr_i32 s4, s3, 31
	v_pk_mul_bf16 v134, v145 /*v401*/, v6
	s_lshr_b32 s4, s4, 26
	v_pk_mul_bf16 v133, v145 /*v401*/, v5
	s_add_co_i32 s2, s3, s4
	s_set_vgpr_msb 0x1101
	v_and_or_b32 v0, v146 /*v402*/, 16, v0
	s_and_b32 s4, s2, 0xffffffc0
	s_ashr_i32 s2, s2, 6
	s_cmp_lg_u32 s3, s4
	v_pk_mul_bf16 v132, v145 /*v401*/, v4
	s_cselect_b32 s4, -1, 0
	s_cmp_lt_i32 s3, 0
	v_pk_mul_bf16 v131, v145 /*v401*/, v3
	s_cselect_b32 s3, -1, 0
	v_pk_mul_bf16 v130, v145 /*v401*/, v2
	s_and_b32 s3, s3, s4
	s_sub_co_ci_u32 s2, s2, 0
	v_pk_mul_bf16 v129, v145 /*v401*/, v1
	s_max_i32 s11, s2, s60
	s_wait_dscnt 0xc
	v_pk_mul_bf16 v143, v145 /*v401*/, v15
	v_pk_mul_bf16 v142, v145 /*v401*/, v14
	v_pk_mul_bf16 v141, v145 /*v401*/, v13
	v_pk_mul_bf16 v140, v145 /*v401*/, v12
	v_pk_mul_bf16 v139, v145 /*v401*/, v11
	v_pk_mul_bf16 v138, v145 /*v401*/, v10
	v_pk_mul_bf16 v137, v145 /*v401*/, v9
	v_pk_mul_bf16 v136, v145 /*v401*/, v8
	s_wait_dscnt 0xa
	v_pk_mul_bf16 v151, v145 /*v401*/, v23
	v_pk_mul_bf16 v150, v145 /*v401*/, v22
	v_pk_mul_bf16 v149, v145 /*v401*/, v21
	v_pk_mul_bf16 v148, v145 /*v401*/, v20
	v_pk_mul_bf16 v147, v145 /*v401*/, v19
	v_pk_mul_bf16 v146, v145 /*v401*/, v18
	v_pk_mul_bf16 v145, v145 /*v401*/, v17
	v_pk_mul_bf16 v144, v145 /*v401*/, v16
	s_wait_dscnt 0x8
	v_pk_mul_bf16 v159, v145 /*v401*/, v31
	v_pk_mul_bf16 v158, v145 /*v401*/, v30
	v_pk_mul_bf16 v157, v145 /*v401*/, v29
	v_pk_mul_bf16 v156, v145 /*v401*/, v28
	v_pk_mul_bf16 v155, v145 /*v401*/, v27
	v_pk_mul_bf16 v154, v145 /*v401*/, v26
	v_pk_mul_bf16 v153, v145 /*v401*/, v25
	v_pk_mul_bf16 v152, v145 /*v401*/, v24
	s_wait_dscnt 0x6
	v_pk_mul_bf16 v167, v145 /*v401*/, v39
	v_pk_mul_bf16 v166, v145 /*v401*/, v38
	v_pk_mul_bf16 v165, v145 /*v401*/, v37
	v_pk_mul_bf16 v164, v145 /*v401*/, v36
	v_pk_mul_bf16 v163, v145 /*v401*/, v35
	v_pk_mul_bf16 v162, v145 /*v401*/, v34
	v_pk_mul_bf16 v161, v145 /*v401*/, v33
	v_pk_mul_bf16 v160, v145 /*v401*/, v32
	s_wait_dscnt 0x4
	v_pk_mul_bf16 v175, v145 /*v401*/, v47
	v_pk_mul_bf16 v174, v145 /*v401*/, v46
	v_pk_mul_bf16 v173, v145 /*v401*/, v45
	v_pk_mul_bf16 v172, v145 /*v401*/, v44
	v_pk_mul_bf16 v171, v145 /*v401*/, v43
	v_pk_mul_bf16 v170, v145 /*v401*/, v42
	v_pk_mul_bf16 v169, v145 /*v401*/, v41
	v_pk_mul_bf16 v168, v145 /*v401*/, v40
	s_wait_dscnt 0x2
	v_pk_mul_bf16 v183, v145 /*v401*/, v55
	v_pk_mul_bf16 v182, v145 /*v401*/, v54
	v_pk_mul_bf16 v181, v145 /*v401*/, v53
	v_pk_mul_bf16 v180, v145 /*v401*/, v52
	v_pk_mul_bf16 v179, v145 /*v401*/, v51
	v_pk_mul_bf16 v178, v145 /*v401*/, v50
	v_pk_mul_bf16 v177, v145 /*v401*/, v49
	v_pk_mul_bf16 v176, v145 /*v401*/, v48
	s_wait_dscnt 0x0
	v_pk_mul_bf16 v191, v145 /*v401*/, v63
	v_pk_mul_bf16 v190, v145 /*v401*/, v62
	v_pk_mul_bf16 v189, v145 /*v401*/, v61
	v_pk_mul_bf16 v188, v145 /*v401*/, v60
	v_pk_mul_bf16 v187, v145 /*v401*/, v59
	v_pk_mul_bf16 v186, v145 /*v401*/, v58
	v_pk_mul_bf16 v185, v145 /*v401*/, v57
	v_pk_mul_bf16 v184, v145 /*v401*/, v56
	s_set_vgpr_msb 0x141
	v_or_b32_e32 v148 /*v404*/, 0x10000, v0
	v_or_b32_e32 v153 /*v409*/, 0x30000, v0
	s_min_i32 s88, s11, s82
	s_cmp_ge_u32 s60, s88
	s_wait_tensorcnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_load_b32 v135 /*v391*/, v139 /*v395*/, s[70:71] scale_offset
	s_set_vgpr_msb 0x4100
	s_cbranch_scc1 .LBB0_11
	s_set_vgpr_msb 5
	v_dual_add_nc_u32 v1, s99, v138 /*v394*/ :: v_dual_add_nc_u32 v2, s99, v137 /*v393*/
	v_dual_mov_b32 v0, 0 :: v_dual_mov_b32 v200, v153 /*v409*/
	s_set_vgpr_msb 0x541
	v_dual_mov_b32 v64 /*v320*/, 1.0 :: v_dual_mov_b32 v149 /*v405*/, v142 /*v398*/
	v_dual_add_nc_u32 v150 /*v406*/, s17, v1 :: v_dual_add_nc_u32 v151 /*v407*/, s17, v2
	v_subrev_nc_u32_e32 v154 /*v410*/, s16, v1
	v_subrev_nc_u32_e32 v155 /*v411*/, s16, v2
	s_set_vgpr_msb 0x4100
	v_dual_mov_b32 v1, v0 :: v_dual_mov_b32 v2, v0
	v_dual_mov_b32 v3, v0 :: v_dual_mov_b32 v4, v0
	v_dual_mov_b32 v5, v0 :: v_dual_mov_b32 v6, v0
	v_mov_b32_e32 v7, v0
	v_mov_b64_e32 v[10:11], v[2:3]
	v_mov_b64_e32 v[8:9], v[0:1]
	v_mov_b64_e32 v[12:13], v[4:5]
	v_mov_b64_e32 v[20:21], v[4:5]
	v_mov_b64_e32 v[14:15], v[6:7]
	v_mov_b64_e32 v[22:23], v[6:7]
	v_mov_b64_e32 v[18:19], v[2:3]
	v_mov_b64_e32 v[16:17], v[0:1]
	v_mov_b64_e32 v[30:31], v[6:7]
	v_mov_b64_e32 v[28:29], v[4:5]
	v_mov_b64_e32 v[26:27], v[2:3]
	v_mov_b64_e32 v[24:25], v[0:1]
	v_mov_b64_e32 v[38:39], v[6:7]
	v_mov_b64_e32 v[36:37], v[4:5]
	v_mov_b64_e32 v[34:35], v[2:3]
	v_mov_b64_e32 v[32:33], v[0:1]
	v_mov_b64_e32 v[46:47], v[6:7]
	v_mov_b64_e32 v[44:45], v[4:5]
	v_mov_b64_e32 v[42:43], v[2:3]
	v_mov_b64_e32 v[40:41], v[0:1]
	v_mov_b64_e32 v[54:55], v[6:7]
	v_mov_b64_e32 v[52:53], v[4:5]
	v_mov_b64_e32 v[50:51], v[2:3]
	v_mov_b64_e32 v[48:49], v[0:1]
	v_mov_b64_e32 v[62:63], v[6:7]
	v_mov_b64_e32 v[60:61], v[4:5]
	v_mov_b64_e32 v[58:59], v[2:3]
	v_mov_b64_e32 v[56:57], v[0:1]
	v_mov_b64_e32 v[70:71], v[6:7]
	v_mov_b64_e32 v[68:69], v[4:5]
	v_mov_b64_e32 v[66:67], v[2:3]
	v_mov_b64_e32 v[64:65], v[0:1]
	v_mov_b64_e32 v[78:79], v[6:7]
	v_mov_b64_e32 v[76:77], v[4:5]
	v_mov_b64_e32 v[74:75], v[2:3]
	v_mov_b64_e32 v[72:73], v[0:1]
	v_mov_b64_e32 v[86:87], v[6:7]
	v_mov_b64_e32 v[84:85], v[4:5]
	v_mov_b64_e32 v[82:83], v[2:3]
	v_mov_b64_e32 v[80:81], v[0:1]
	v_mov_b64_e32 v[94:95], v[6:7]
	v_mov_b64_e32 v[92:93], v[4:5]
	v_mov_b64_e32 v[90:91], v[2:3]
	v_mov_b64_e32 v[88:89], v[0:1]
	v_mov_b64_e32 v[102:103], v[6:7]
	v_mov_b64_e32 v[100:101], v[4:5]
	v_mov_b64_e32 v[98:99], v[2:3]
	v_mov_b64_e32 v[96:97], v[0:1]
	v_mov_b64_e32 v[110:111], v[6:7]
	v_mov_b64_e32 v[108:109], v[4:5]
	v_mov_b64_e32 v[106:107], v[2:3]
	v_mov_b64_e32 v[104:105], v[0:1]
	v_mov_b64_e32 v[118:119], v[6:7]
	v_mov_b64_e32 v[116:117], v[4:5]
	v_mov_b64_e32 v[114:115], v[2:3]
	v_mov_b64_e32 v[112:113], v[0:1]
	v_mov_b64_e32 v[126:127], v[6:7]
	v_mov_b64_e32 v[124:125], v[4:5]
	v_mov_b64_e32 v[122:123], v[2:3]
	v_mov_b64_e32 v[120:121], v[0:1]
	s_set_vgpr_msb 1
	v_mov_b32_e32 v201, v152 /*v408*/
	s_set_vgpr_msb 0x141
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v130 /*v386*/, v135 /*v391*/ :: v_dual_mov_b32 v65 /*v321*/, v64 /*v320*/
	s_mov_b32 s61, s59
	s_mov_b32 s28, 1
	s_sub_co_i32 s83, 1, s60
	s_mov_b32 s56, 16
	s_mov_b32 s53, 0xffff0000
	s_mov_b32 s52, 0x7510000
	s_mov_b32 s70, 0x76543210
	s_mov_b32 s90, 0x3fb8aa3b
	s_mov_b64 s[92:93], s[60:61]
	s_set_vgpr_msb 0x4100
	s_branch .LBB0_4
.LBB0_3:
	s_set_vgpr_msb 0x45
	v_cvt_pk_bf16_f32 v165 /*v421*/, v116 /*v372*/, v122 /*v378*/
	v_cvt_pk_bf16_f32 v164 /*v420*/, v108 /*v364*/, v114 /*v370*/
	v_cvt_pk_bf16_f32 v163 /*v419*/, v100 /*v356*/, v106 /*v362*/
	v_cvt_pk_bf16_f32 v162 /*v418*/, v92 /*v348*/, v98 /*v354*/
	v_cvt_pk_bf16_f32 v161 /*v417*/, v84 /*v340*/, v90 /*v346*/
	v_cvt_pk_bf16_f32 v160 /*v416*/, v76 /*v332*/, v82 /*v338*/
	v_cvt_pk_bf16_f32 v159 /*v415*/, v70 /*v326*/, v74 /*v330*/
	v_cvt_pk_bf16_f32 v158 /*v414*/, v66 /*v322*/, v68 /*v324*/
	v_cvt_pk_bf16_f32 v173 /*v429*/, v117 /*v373*/, v123 /*v379*/
	v_cvt_pk_bf16_f32 v172 /*v428*/, v109 /*v365*/, v115 /*v371*/
	v_cvt_pk_bf16_f32 v171 /*v427*/, v101 /*v357*/, v107 /*v363*/
	v_cvt_pk_bf16_f32 v170 /*v426*/, v93 /*v349*/, v99 /*v355*/
	v_cvt_pk_bf16_f32 v169 /*v425*/, v85 /*v341*/, v91 /*v347*/
	v_cvt_pk_bf16_f32 v168 /*v424*/, v77 /*v333*/, v83 /*v339*/
	v_cvt_pk_bf16_f32 v167 /*v423*/, v71 /*v327*/, v75 /*v331*/
	v_cvt_pk_bf16_f32 v166 /*v422*/, v67 /*v323*/, v69 /*v325*/
	s_set_vgpr_msb 0x4505
	s_wait_dscnt 0x1d
	v_wmma_f32_16x16x32_bf16 v[120:127], v[56:63] /*v[312:319]*/, v[158:165] /*v[414:421]*/, v[120:127]
	s_set_vgpr_msb 0x545
	v_cvt_pk_bf16_f32 v101 /*v357*/, v127 /*v383*/, v129 /*v385*/
	v_cvt_pk_bf16_f32 v100 /*v356*/, v121 /*v377*/, v125 /*v381*/
	v_cvt_pk_bf16_f32 v99 /*v355*/, v113 /*v369*/, v119 /*v375*/
	v_cvt_pk_bf16_f32 v98 /*v354*/, v105 /*v361*/, v111 /*v367*/
	v_cvt_pk_bf16_f32 v97 /*v353*/, v97 /*v353*/, v103 /*v359*/
	s_add_nc_u64 s[92:93], s[92:93], 1
	s_set_vgpr_msb 0x4505
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[56:63], v[56:63] /*v[312:319]*/, v[166:173] /*v[422:429]*/, v[56:63]
	v_cmp_lt_u64_e64 s2, s[92:93], s[88:89]
	s_and_b32 vcc_lo, exec_lo, s2
	v_wmma_f32_16x16x32_bf16 v[112:119], v[40:47] /*v[296:303]*/, v[158:165] /*v[414:421]*/, v[112:119]
	v_nop
	v_nop
	s_set_vgpr_msb 0x545
	v_cvt_pk_bf16_f32 v63 /*v319*/, v126 /*v382*/, v128 /*v384*/
	v_cvt_pk_bf16_f32 v62 /*v318*/, v120 /*v376*/, v124 /*v380*/
	v_cvt_pk_bf16_f32 v61 /*v317*/, v112 /*v368*/, v118 /*v374*/
	v_cvt_pk_bf16_f32 v60 /*v316*/, v104 /*v360*/, v110 /*v366*/
	v_cvt_pk_bf16_f32 v59 /*v315*/, v96 /*v352*/, v102 /*v358*/
	s_set_vgpr_msb 0x4505
	v_wmma_f32_16x16x32_bf16 v[48:55], v[40:47] /*v[296:303]*/, v[166:173] /*v[422:429]*/, v[48:55]
	s_set_vgpr_msb 0x545
	v_cvt_pk_bf16_f32 v58 /*v314*/, v88 /*v344*/, v94 /*v350*/
	v_cvt_pk_bf16_f32 v57 /*v313*/, v80 /*v336*/, v86 /*v342*/
	v_cvt_pk_bf16_f32 v56 /*v312*/, v72 /*v328*/, v78 /*v334*/
	v_cvt_pk_bf16_f32 v96 /*v352*/, v89 /*v345*/, v95 /*v351*/
	v_cvt_pk_bf16_f32 v95 /*v351*/, v81 /*v337*/, v87 /*v343*/
	v_cvt_pk_bf16_f32 v94 /*v350*/, v73 /*v329*/, v79 /*v335*/
	s_set_vgpr_msb 0x4505
	v_wmma_f32_16x16x32_bf16 v[104:111], v[24:31] /*v[280:287]*/, v[158:165] /*v[414:421]*/, v[104:111]
	v_wmma_f32_16x16x32_bf16 v[40:47], v[24:31] /*v[280:287]*/, v[166:173] /*v[422:429]*/, v[40:47]
	v_wmma_f32_16x16x32_bf16 v[96:103], v[8:15] /*v[264:271]*/, v[158:165] /*v[414:421]*/, v[96:103]
	v_wmma_f32_16x16x32_bf16 v[32:39], v[8:15] /*v[264:271]*/, v[166:173] /*v[422:429]*/, v[32:39]
	s_set_vgpr_msb 0x504
	v_wmma_f32_16x16x32_bf16 v[88:95], v[248:255], v[158:165] /*v[414:421]*/, v[88:95]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[248:255], v[166:173] /*v[422:429]*/, v[24:31]
	v_wmma_f32_16x16x32_bf16 v[80:87], v[232:239], v[158:165] /*v[414:421]*/, v[80:87]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[232:239], v[166:173] /*v[422:429]*/, v[16:23]
	v_wmma_f32_16x16x32_bf16 v[72:79], v[216:223], v[158:165] /*v[414:421]*/, v[72:79]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[216:223], v[166:173] /*v[422:429]*/, v[8:15]
	v_wmma_f32_16x16x32_bf16 v[64:71], v[200:207], v[158:165] /*v[414:421]*/, v[64:71]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[200:207], v[166:173] /*v[422:429]*/, v[0:7]
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x415
	v_pk_fma_f32 v[200:201], v[134:135] /*v[390:391]*/, v[64:65] /*v[320:321]*/, v[132:133] /*v[388:389]*/
	s_set_vgpr_msb 0x1541
	v_mov_b32_e32 v135 /*v391*/, v157 /*v413*/
	v_pk_add_f32 v[64:65] /*v[320:321]*/, v[130:131] /*v[386:387]*/, v[200:201]
	s_set_vgpr_msb 0x4105
	v_wmma_f32_16x16x32_bf16 v[120:127], v[48:55] /*v[304:311]*/, v[56:63] /*v[312:319]*/, v[120:127]
	v_dual_mov_b32 v200, v153 /*v409*/ :: v_dual_mov_b32 v201, v152 /*v408*/
	s_set_vgpr_msb 0x541
	v_mov_b32_e32 v130 /*v386*/, v156 /*v412*/
	s_set_vgpr_msb 0x4105
	v_wmma_f32_16x16x32_bf16 v[56:63], v[48:55] /*v[304:311]*/, v[94:101] /*v[350:357]*/, v[56:63]
	v_wmma_f32_16x16x32_bf16 v[112:119], v[32:39] /*v[288:295]*/, v[56:63] /*v[312:319]*/, v[112:119]
	v_wmma_f32_16x16x32_bf16 v[48:55], v[32:39] /*v[288:295]*/, v[94:101] /*v[350:357]*/, v[48:55]
	v_wmma_f32_16x16x32_bf16 v[104:111], v[16:23] /*v[272:279]*/, v[56:63] /*v[312:319]*/, v[104:111]
	v_wmma_f32_16x16x32_bf16 v[40:47], v[16:23] /*v[272:279]*/, v[94:101] /*v[350:357]*/, v[40:47]
	v_wmma_f32_16x16x32_bf16 v[96:103], v[0:7] /*v[256:263]*/, v[56:63] /*v[312:319]*/, v[96:103]
	v_wmma_f32_16x16x32_bf16 v[32:39], v[0:7] /*v[256:263]*/, v[94:101] /*v[350:357]*/, v[32:39]
	s_set_vgpr_msb 0x504
	v_wmma_f32_16x16x32_bf16 v[88:95], v[240:247], v[56:63] /*v[312:319]*/, v[88:95]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[240:247], v[94:101] /*v[350:357]*/, v[24:31]
	v_wmma_f32_16x16x32_bf16 v[80:87], v[224:231], v[56:63] /*v[312:319]*/, v[80:87]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[224:231], v[94:101] /*v[350:357]*/, v[16:23]
	v_wmma_f32_16x16x32_bf16 v[72:79], v[208:215], v[56:63] /*v[312:319]*/, v[72:79]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[208:215], v[94:101] /*v[350:357]*/, v[8:15]
	v_wmma_f32_16x16x32_bf16 v[64:71], v[192:199], v[56:63] /*v[312:319]*/, v[64:71]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[192:199], v[94:101] /*v[350:357]*/, v[0:7]
	s_set_vgpr_msb 0x400
	s_cbranch_vccz .LBB0_12
.LBB0_4:
	s_wait_alu depctr_vm_vsrc(0)
	s_set_vgpr_msb 0x41
	v_dual_mov_b32 v152 /*v408*/, v149 /*v405*/ :: v_dual_mov_b32 v153 /*v409*/, v148 /*v404*/
	s_set_vgpr_msb 0x4140
	v_dual_mov_b32 v149 /*v405*/, v201 :: v_dual_mov_b32 v148 /*v404*/, v200
	s_add_co_i32 s2, s92, 1
	s_wait_tensorcnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	s_cmp_ge_i32 s2, s10
	s_set_vgpr_msb 0x4000
	s_cbranch_scc1 .LBB0_6
	s_lshl_b32 s4, s2, 6
	s_add_co_i32 s6, s92, s83
	s_sub_co_i32 s30, s9, s4
	s_add_co_i32 s2, s4, s97
	v_nop
	v_nop
	v_med3_i32 v192, s30, 0, 64
	s_lshr_b32 s29, s6, 31
	s_ashr_i32 s3, s2, 31
	s_add_co_i32 s29, s6, s29
	s_mul_u64 s[4:5], s[2:3], s[64:65]
	v_readfirstlane_b32 s30, v192
	s_mul_u64 s[2:3], s[2:3], s[62:63]
	s_and_b32 s29, s29, 0x7ffe
	s_lshl_b64 s[2:3], s[2:3], 1
	s_sub_co_i32 s6, s6, s29
	s_add_nc_u64 s[2:3], s[86:87], s[2:3]
	s_sub_co_i32 s29, s30, s25
	s_add_nc_u64 s[30:31], s[26:27], s[2:3]
	s_max_i32 s2, s29, 0
	s_lshl_b64 s[4:5], s[4:5], 1
	s_lshl_b32 s6, s6, 17
	s_lshl_b32 s2, s2, 16
	s_add_nc_u64 s[4:5], s[84:85], s[4:5]
	s_or_b32 s29, s104, s6
	s_bitset1_b32 s31, 31
	s_or_b32 s54, s2, 0x7fff
	s_mov_b32 s45, s53
	tensor_load_to_lds s[28:31], s[52:59]
	s_add_nc_u64 s[30:31], s[80:81], s[4:5]
	s_or_b32 s29, vcc_hi, s6
	s_bitset1_b32 s31, 31
	s_mov_b32 s46, s54
	s_mov_b32 s47, s55
	s_mov_b32 s48, s56
	s_mov_b32 s51, s59
	tensor_load_to_lds s[28:31], s[44:51]
.LBB0_6:
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 1
	ds_load_b128 v[192:195], v152 /*v408*/
	ds_load_b128 v[196:199], v152 /*v408*/ offset:32
	ds_load_b128 v[200:203], v152 /*v408*/ offset:64
	ds_load_b128 v[204:207], v152 /*v408*/ offset:96
	ds_load_b128 v[208:211], v152 /*v408*/ offset:128
	ds_load_b128 v[212:215], v152 /*v408*/ offset:160
	ds_load_b128 v[216:219], v152 /*v408*/ offset:192
	ds_load_b128 v[220:223], v152 /*v408*/ offset:224
	ds_load_b128 v[224:227], v152 /*v408*/ offset:4352
	ds_load_b128 v[228:231], v152 /*v408*/ offset:4384
	ds_load_b128 v[232:235], v152 /*v408*/ offset:4416
	ds_load_b128 v[236:239], v152 /*v408*/ offset:4448
	ds_load_b128 v[240:243], v152 /*v408*/ offset:4480
	ds_load_b128 v[244:247], v152 /*v408*/ offset:4512
	ds_load_b128 v[248:251], v152 /*v408*/ offset:4544
	ds_load_b128 v[252:255], v152 /*v408*/ offset:4576
	s_set_vgpr_msb 0x141
	ds_load_b128 v[0:3] /*v[256:259]*/, v152 /*v408*/ offset:8704
	ds_load_b128 v[4:7] /*v[260:263]*/, v152 /*v408*/ offset:8736
	ds_load_b128 v[8:11] /*v[264:267]*/, v152 /*v408*/ offset:8768
	ds_load_b128 v[12:15] /*v[268:271]*/, v152 /*v408*/ offset:8800
	ds_load_b128 v[16:19] /*v[272:275]*/, v152 /*v408*/ offset:8832
	ds_load_b128 v[20:23] /*v[276:279]*/, v152 /*v408*/ offset:8864
	ds_load_b128 v[24:27] /*v[280:283]*/, v152 /*v408*/ offset:8896
	ds_load_b128 v[28:31] /*v[284:287]*/, v152 /*v408*/ offset:8928
	ds_load_b128 v[32:35] /*v[288:291]*/, v152 /*v408*/ offset:13056
	ds_load_b128 v[36:39] /*v[292:295]*/, v152 /*v408*/ offset:13088
	ds_load_b128 v[40:43] /*v[296:299]*/, v152 /*v408*/ offset:13120
	ds_load_b128 v[44:47] /*v[300:303]*/, v152 /*v408*/ offset:13152
	ds_load_b128 v[48:51] /*v[304:307]*/, v152 /*v408*/ offset:13184
	ds_load_b128 v[52:55] /*v[308:311]*/, v152 /*v408*/ offset:13216
	ds_load_b128 v[56:59] /*v[312:315]*/, v152 /*v408*/ offset:13248
	ds_load_b128 v[60:63] /*v[316:319]*/, v152 /*v408*/ offset:13280
	s_set_vgpr_msb 0x4140
	s_wait_dscnt 0x1e
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[322:329]*/, v[192:199], v[128:135], 0
	v_wmma_f32_16x16x32_bf16 v[74:81] /*v[330:337]*/, v[192:199], v[160:167], 0
	s_wait_dscnt 0x16
	v_wmma_f32_16x16x32_bf16 v[82:89] /*v[338:345]*/, v[224:231], v[128:135], 0
	v_wmma_f32_16x16x32_bf16 v[90:97] /*v[346:353]*/, v[224:231], v[160:167], 0
	s_set_vgpr_msb 0x4041
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x32_bf16 v[98:105] /*v[354:361]*/, v[0:7] /*v[256:263]*/, v[128:135], 0
	v_wmma_f32_16x16x32_bf16 v[106:113] /*v[362:369]*/, v[0:7] /*v[256:263]*/, v[160:167], 0
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x32_bf16 v[114:121] /*v[370:377]*/, v[32:39] /*v[288:295]*/, v[128:135], 0
	v_wmma_f32_16x16x32_bf16 v[122:129] /*v[378:385]*/, v[32:39] /*v[288:295]*/, v[160:167], 0
	s_set_vgpr_msb 0x4150
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[322:329]*/, v[200:207], v[136:143], v[66:73] /*v[322:329]*/
	v_wmma_f32_16x16x32_bf16 v[74:81] /*v[330:337]*/, v[200:207], v[168:175], v[74:81] /*v[330:337]*/
	v_wmma_f32_16x16x32_bf16 v[82:89] /*v[338:345]*/, v[232:239], v[136:143], v[82:89] /*v[338:345]*/
	v_wmma_f32_16x16x32_bf16 v[90:97] /*v[346:353]*/, v[232:239], v[168:175], v[90:97] /*v[346:353]*/
	s_set_vgpr_msb 0x5051
	v_wmma_f32_16x16x32_bf16 v[98:105] /*v[354:361]*/, v[8:15] /*v[264:271]*/, v[136:143], v[98:105] /*v[354:361]*/
	v_wmma_f32_16x16x32_bf16 v[106:113] /*v[362:369]*/, v[8:15] /*v[264:271]*/, v[168:175], v[106:113] /*v[362:369]*/
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x32_bf16 v[114:121] /*v[370:377]*/, v[40:47] /*v[296:303]*/, v[136:143], v[114:121] /*v[370:377]*/
	v_wmma_f32_16x16x32_bf16 v[122:129] /*v[378:385]*/, v[40:47] /*v[296:303]*/, v[168:175], v[122:129] /*v[378:385]*/
	s_set_vgpr_msb 0x5150
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[322:329]*/, v[208:215], v[144:151], v[66:73] /*v[322:329]*/
	v_wmma_f32_16x16x32_bf16 v[74:81] /*v[330:337]*/, v[208:215], v[176:183], v[74:81] /*v[330:337]*/
	v_wmma_f32_16x16x32_bf16 v[82:89] /*v[338:345]*/, v[240:247], v[144:151], v[82:89] /*v[338:345]*/
	v_wmma_f32_16x16x32_bf16 v[90:97] /*v[346:353]*/, v[240:247], v[176:183], v[90:97] /*v[346:353]*/
	s_set_vgpr_msb 0x5051
	v_wmma_f32_16x16x32_bf16 v[98:105] /*v[354:361]*/, v[16:23] /*v[272:279]*/, v[144:151], v[98:105] /*v[354:361]*/
	v_wmma_f32_16x16x32_bf16 v[106:113] /*v[362:369]*/, v[16:23] /*v[272:279]*/, v[176:183], v[106:113] /*v[362:369]*/
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[114:121] /*v[370:377]*/, v[48:55] /*v[304:311]*/, v[144:151], v[114:121] /*v[370:377]*/
	v_wmma_f32_16x16x32_bf16 v[122:129] /*v[378:385]*/, v[48:55] /*v[304:311]*/, v[176:183], v[122:129] /*v[378:385]*/
	s_set_vgpr_msb 0x5150
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[322:329]*/, v[216:223], v[152:159], v[66:73] /*v[322:329]*/
	v_wmma_f32_16x16x32_bf16 v[74:81] /*v[330:337]*/, v[216:223], v[184:191], v[74:81] /*v[330:337]*/
	v_wmma_f32_16x16x32_bf16 v[82:89] /*v[338:345]*/, v[248:255], v[152:159], v[82:89] /*v[338:345]*/
	v_wmma_f32_16x16x32_bf16 v[90:97] /*v[346:353]*/, v[248:255], v[184:191], v[90:97] /*v[346:353]*/
	s_set_vgpr_msb 0x5051
	v_wmma_f32_16x16x32_bf16 v[98:105] /*v[354:361]*/, v[24:31] /*v[280:287]*/, v[152:159], v[98:105] /*v[354:361]*/
	v_wmma_f32_16x16x32_bf16 v[106:113] /*v[362:369]*/, v[24:31] /*v[280:287]*/, v[184:191], v[106:113] /*v[362:369]*/
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[114:121] /*v[370:377]*/, v[56:63] /*v[312:319]*/, v[152:159], v[114:121] /*v[370:377]*/
	v_wmma_f32_16x16x32_bf16 v[122:129] /*v[378:385]*/, v[56:63] /*v[312:319]*/, v[184:191], v[122:129] /*v[378:385]*/
	s_wait_alu depctr_va_vdst(0)
	ds_load_tr16_b128 v[56:59] /*v[312:315]*/, v153 /*v409*/
	ds_load_tr16_b128 v[40:43] /*v[296:299]*/, v153 /*v409*/ offset:32
	ds_load_tr16_b128 v[60:63] /*v[316:319]*/, v153 /*v409*/ offset:4608
	ds_load_tr16_b128 v[44:47] /*v[300:303]*/, v153 /*v409*/ offset:4640
	ds_load_tr16_b128 v[48:51] /*v[304:307]*/, v153 /*v409*/ offset:9216
	ds_load_tr16_b128 v[32:35] /*v[288:291]*/, v153 /*v409*/ offset:9248
	ds_load_tr16_b128 v[52:55] /*v[308:311]*/, v153 /*v409*/ offset:13824
	ds_load_tr16_b128 v[36:39] /*v[292:295]*/, v153 /*v409*/ offset:13856
	ds_load_tr16_b128 v[24:27] /*v[280:283]*/, v153 /*v409*/ offset:64
	ds_load_tr16_b128 v[8:11] /*v[264:267]*/, v153 /*v409*/ offset:96
	ds_load_tr16_b128 v[28:31] /*v[284:287]*/, v153 /*v409*/ offset:4672
	ds_load_tr16_b128 v[12:15] /*v[268:271]*/, v153 /*v409*/ offset:4704
	ds_load_tr16_b128 v[16:19] /*v[272:275]*/, v153 /*v409*/ offset:9280
	ds_load_tr16_b128 v[0:3] /*v[256:259]*/, v153 /*v409*/ offset:9312
	ds_load_tr16_b128 v[20:23] /*v[276:279]*/, v153 /*v409*/ offset:13888
	ds_load_tr16_b128 v[4:7] /*v[260:263]*/, v153 /*v409*/ offset:13920
	s_set_vgpr_msb 0x5101
	ds_load_tr16_b128 v[248:251], v153 /*v409*/ offset:128
	ds_load_tr16_b128 v[232:235], v153 /*v409*/ offset:160
	ds_load_tr16_b128 v[252:255], v153 /*v409*/ offset:4736
	ds_load_tr16_b128 v[236:239], v153 /*v409*/ offset:4768
	ds_load_tr16_b128 v[240:243], v153 /*v409*/ offset:9344
	ds_load_tr16_b128 v[224:227], v153 /*v409*/ offset:9376
	ds_load_tr16_b128 v[244:247], v153 /*v409*/ offset:13952
	ds_load_tr16_b128 v[228:231], v153 /*v409*/ offset:13984
	ds_load_tr16_b128 v[216:219], v153 /*v409*/ offset:192
	ds_load_tr16_b128 v[200:203], v153 /*v409*/ offset:224
	ds_load_tr16_b128 v[220:223], v153 /*v409*/ offset:4800
	ds_load_tr16_b128 v[204:207], v153 /*v409*/ offset:4832
	ds_load_tr16_b128 v[208:211], v153 /*v409*/ offset:9408
	ds_load_tr16_b128 v[192:195], v153 /*v409*/ offset:9440
	ds_load_tr16_b128 v[212:215], v153 /*v409*/ offset:14016
	ds_load_tr16_b128 v[196:199], v153 /*v409*/ offset:14048
	s_set_vgpr_msb 0x155
	v_lshl_or_b32 v131 /*v387*/, s92, 6, v143 /*v399*/
	v_dual_add_nc_u32 v174 /*v430*/, 16, v131 /*v387*/ :: v_dual_bitop2_b32 v134 /*v390*/, 1, v131 /*v387*/ bitop3:0x54
	v_cmp_gt_i32_e32 vcc_lo, v131 /*v387*/, v150 /*v406*/
	v_cmp_lt_i32_e64 s2, v131 /*v387*/, v154 /*v410*/
	v_cmp_ge_i32_e64 s3, v131 /*v387*/, v150 /*v406*/
	v_dual_add_nc_u32 v175 /*v431*/, 17, v131 /*v387*/ :: v_dual_bitop2_b32 v156 /*v412*/, 2, v131 /*v387*/ bitop3:0x54
	v_cmp_lt_i32_e64 s4, v134 /*v390*/, v154 /*v410*/
	v_dual_add_nc_u32 v176 /*v432*/, 18, v131 /*v387*/ :: v_dual_bitop2_b32 v157 /*v413*/, 3, v131 /*v387*/ bitop3:0x54
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v156 /*v412*/, v150 /*v406*/
	v_cndmask_b32_e64 v66 /*v322*/, v66 /*v322*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v156 /*v412*/, v154 /*v410*/
	s_or_b32 s3, s4, s3
	v_dual_add_nc_u32 v177 /*v433*/, 19, v131 /*v387*/ :: v_dual_bitop2_b32 v168 /*v424*/, 4, v131 /*v387*/ bitop3:0x54
	v_cmp_gt_i32_e64 s5, v157 /*v413*/, v150 /*v406*/
	v_cndmask_b32_e64 v67 /*v323*/, v67 /*v323*/, 0xff800000, s3
	v_cmp_lt_i32_e64 s3, v157 /*v413*/, v154 /*v410*/
	v_dual_add_nc_u32 v178 /*v434*/, 20, v131 /*v387*/ :: v_dual_bitop2_b32 v171 /*v427*/, 5, v131 /*v387*/ bitop3:0x54
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v168 /*v424*/, v150 /*v406*/
	v_cndmask_b32_e64 v68 /*v324*/, v68 /*v324*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v168 /*v424*/, v154 /*v410*/
	s_or_b32 s3, s3, s5
	v_or_b32_e32 v172 /*v428*/, 6, v131 /*v387*/
	v_cndmask_b32_e64 v69 /*v325*/, v69 /*v325*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v171 /*v427*/, v150 /*v406*/
	v_cmp_lt_i32_e64 s4, v171 /*v427*/, v154 /*v410*/
	v_or_b32_e32 v173 /*v429*/, 7, v131 /*v387*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v172 /*v428*/, v150 /*v406*/
	v_cndmask_b32_e64 v70 /*v326*/, v70 /*v326*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v172 /*v428*/, v154 /*v410*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s4, v173 /*v429*/, v154 /*v410*/
	v_cndmask_b32_e64 v71 /*v327*/, v71 /*v327*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v173 /*v429*/, v150 /*v406*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v174 /*v430*/, v150 /*v406*/
	v_cndmask_b32_e64 v72 /*v328*/, v72 /*v328*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v174 /*v430*/, v154 /*v410*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s4, v175 /*v431*/, v154 /*v410*/
	v_cndmask_b32_e64 v73 /*v329*/, v73 /*v329*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v175 /*v431*/, v150 /*v406*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v176 /*v432*/, v150 /*v406*/
	v_cndmask_b32_e64 v82 /*v338*/, v82 /*v338*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v176 /*v432*/, v154 /*v410*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s4, v177 /*v433*/, v154 /*v410*/
	v_cndmask_b32_e64 v83 /*v339*/, v83 /*v339*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v177 /*v433*/, v150 /*v406*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v178 /*v434*/, v150 /*v406*/
	v_cndmask_b32_e64 v132 /*v388*/, v84 /*v340*/, 0xff800000, s2
	v_add_nc_u32_e32 v84 /*v340*/, 21, v131 /*v387*/
	v_cmp_lt_i32_e64 s2, v178 /*v434*/, v154 /*v410*/
	s_or_b32 s3, s4, s3
	v_dual_add_nc_u32 v180 /*v436*/, 23, v131 /*v387*/ :: v_dual_bitop2_b32 v181 /*v437*/, 32, v131 /*v387*/ bitop3:0x54
	v_cndmask_b32_e64 v133 /*v389*/, v85 /*v341*/, 0xff800000, s3
	v_add_nc_u32_e32 v85 /*v341*/, 22, v131 /*v387*/
	v_cmp_gt_i32_e64 s3, v84 /*v340*/, v150 /*v406*/
	v_cmp_lt_i32_e64 s4, v84 /*v340*/, v154 /*v410*/
	s_or_b32 s2, s2, vcc_lo
	v_dual_add_nc_u32 v190 /*v446*/, 48, v131 /*v387*/ :: v_dual_bitop2_b32 v183 /*v439*/, 33, v131 /*v387*/ bitop3:0x54
	v_cndmask_b32_e64 v86 /*v342*/, v86 /*v342*/, 0xff800000, s2
	v_cmp_gt_i32_e32 vcc_lo, v85 /*v341*/, v150 /*v406*/
	v_cmp_lt_i32_e64 s2, v85 /*v341*/, v154 /*v410*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s4, v180 /*v436*/, v154 /*v410*/
	v_cndmask_b32_e64 v87 /*v343*/, v87 /*v343*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v180 /*v436*/, v150 /*v406*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v181 /*v437*/, v150 /*v406*/
	v_cndmask_b32_e64 v88 /*v344*/, v88 /*v344*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v181 /*v437*/, v154 /*v410*/
	s_or_b32 s3, s4, s3
	v_dual_add_nc_u32 v191 /*v447*/, 49, v131 /*v387*/ :: v_dual_bitop2_b32 v184 /*v440*/, 34, v131 /*v387*/ bitop3:0x54
	v_cndmask_b32_e64 v89 /*v345*/, v89 /*v345*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v183 /*v439*/, v150 /*v406*/
	v_cmp_lt_i32_e64 s4, v183 /*v439*/, v154 /*v410*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v184 /*v440*/, v150 /*v406*/
	v_cndmask_b32_e64 v158 /*v414*/, v98 /*v354*/, 0xff800000, s2
	v_dual_add_nc_u32 v192 /*v448*/, 50, v131 /*v387*/ :: v_dual_bitop2_b32 v98 /*v354*/, 35, v131 /*v387*/ bitop3:0x54
	v_cmp_lt_i32_e64 s2, v184 /*v440*/, v154 /*v410*/
	s_or_b32 s3, s4, s3
	v_or_b32_e32 v189 /*v445*/, 39, v131 /*v387*/
	v_cndmask_b32_e64 v159 /*v415*/, v99 /*v355*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v98 /*v354*/, v150 /*v406*/
	v_or_b32_e32 v99 /*v355*/, 36, v131 /*v387*/
	v_cmp_lt_i32_e64 s4, v98 /*v354*/, v154 /*v410*/
	s_or_b32 s2, s2, vcc_lo
	v_add_nc_u32_e32 v195 /*v451*/, 55, v131 /*v387*/
	v_cndmask_b32_e64 v160 /*v416*/, v100 /*v356*/, 0xff800000, s2
	v_or_b32_e32 v100 /*v356*/, 37, v131 /*v387*/
	v_cmp_gt_i32_e32 vcc_lo, v99 /*v355*/, v150 /*v406*/
	v_cmp_lt_i32_e64 s2, v99 /*v355*/, v154 /*v410*/
	s_or_b32 s3, s4, s3
	v_cndmask_b32_e64 v161 /*v417*/, v101 /*v357*/, 0xff800000, s3
	v_or_b32_e32 v101 /*v357*/, 38, v131 /*v387*/
	v_cmp_gt_i32_e64 s3, v100 /*v356*/, v150 /*v406*/
	v_cmp_lt_i32_e64 s4, v100 /*v356*/, v154 /*v410*/
	s_or_b32 s2, s2, vcc_lo
	v_cndmask_b32_e64 v102 /*v358*/, v102 /*v358*/, 0xff800000, s2
	v_cmp_gt_i32_e32 vcc_lo, v101 /*v357*/, v150 /*v406*/
	v_cmp_lt_i32_e64 s2, v101 /*v357*/, v154 /*v410*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s4, v189 /*v445*/, v154 /*v410*/
	v_cndmask_b32_e64 v103 /*v359*/, v103 /*v359*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v189 /*v445*/, v150 /*v406*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v190 /*v446*/, v150 /*v406*/
	v_cndmask_b32_e64 v104 /*v360*/, v104 /*v360*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v190 /*v446*/, v154 /*v410*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s4, v191 /*v447*/, v154 /*v410*/
	v_cndmask_b32_e64 v105 /*v361*/, v105 /*v361*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v191 /*v447*/, v150 /*v406*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v192 /*v448*/, v150 /*v406*/
	v_cndmask_b32_e64 v162 /*v418*/, v114 /*v370*/, 0xff800000, s2
	v_add_nc_u32_e32 v114 /*v370*/, 51, v131 /*v387*/
	v_cmp_lt_i32_e64 s2, v192 /*v448*/, v154 /*v410*/
	s_or_b32 s3, s4, s3
	v_cndmask_b32_e64 v163 /*v419*/, v115 /*v371*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v114 /*v370*/, v150 /*v406*/
	v_cmp_lt_i32_e64 s4, v114 /*v370*/, v154 /*v410*/
	s_or_b32 s2, s2, vcc_lo
	v_add_nc_u32_e32 v115 /*v371*/, 52, v131 /*v387*/
	v_cndmask_b32_e64 v164 /*v420*/, v116 /*v372*/, 0xff800000, s2
	v_add_nc_u32_e32 v116 /*v372*/, 53, v131 /*v387*/
	s_or_b32 s2, s4, s3
	v_cndmask_b32_e64 v165 /*v421*/, v117 /*v373*/, 0xff800000, s2
	v_add_nc_u32_e32 v117 /*v373*/, 54, v131 /*v387*/
	v_cmp_gt_i32_e32 vcc_lo, v115 /*v371*/, v150 /*v406*/
	v_cmp_lt_i32_e64 s2, v115 /*v371*/, v154 /*v410*/
	v_cmp_gt_i32_e64 s3, v116 /*v372*/, v150 /*v406*/
	v_cmp_lt_i32_e64 s4, v116 /*v372*/, v154 /*v410*/
	v_cmp_gt_i32_e64 s5, v117 /*v373*/, v150 /*v406*/
	v_cmp_lt_i32_e64 s6, v117 /*v373*/, v154 /*v410*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v195 /*v451*/, v150 /*v406*/
	v_cndmask_b32_e64 v118 /*v374*/, v118 /*v374*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	v_cmp_gt_i32_e64 s3, v131 /*v387*/, v151 /*v407*/
	v_cndmask_b32_e64 v119 /*v375*/, v119 /*v375*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	v_cmp_lt_i32_e64 s4, v131 /*v387*/, v155 /*v411*/
	v_cndmask_b32_e64 v120 /*v376*/, v120 /*v376*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v195 /*v451*/, v154 /*v410*/
	v_cmp_ge_i32_e64 s5, v131 /*v387*/, v151 /*v407*/
	v_cmp_lt_i32_e64 s6, v134 /*v390*/, v155 /*v411*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v156 /*v412*/, v151 /*v407*/
	v_cndmask_b32_e64 v121 /*v377*/, v121 /*v377*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	v_cmp_gt_i32_e64 s3, v157 /*v413*/, v151 /*v407*/
	v_cndmask_b32_e64 v166 /*v422*/, v74 /*v330*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	v_cmp_lt_i32_e64 s4, v157 /*v413*/, v155 /*v411*/
	v_cndmask_b32_e64 v167 /*v423*/, v75 /*v331*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v156 /*v412*/, v155 /*v411*/
	v_cmp_gt_i32_e64 s5, v168 /*v424*/, v151 /*v407*/
	v_cmp_lt_i32_e64 s6, v168 /*v424*/, v155 /*v411*/
	v_max_num_f32_e32 v74 /*v330*/, v66 /*v322*/, v67 /*v323*/
	v_max_num_f32_e32 v75 /*v331*/, v166 /*v422*/, v167 /*v423*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v171 /*v427*/, v151 /*v407*/
	v_cndmask_b32_e64 v168 /*v424*/, v76 /*v332*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	v_cmp_gt_i32_e64 s3, v172 /*v428*/, v151 /*v407*/
	v_cndmask_b32_e64 v169 /*v425*/, v77 /*v333*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	v_cmp_lt_i32_e64 s4, v172 /*v428*/, v155 /*v411*/
	v_cndmask_b32_e64 v170 /*v426*/, v78 /*v334*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v171 /*v427*/, v155 /*v411*/
	v_cmp_gt_i32_e64 s5, v173 /*v429*/, v151 /*v407*/
	v_cmp_lt_i32_e64 s6, v173 /*v429*/, v155 /*v411*/
	v_max3_num_f32 v76 /*v332*/, v69 /*v325*/, v70 /*v326*/, v71 /*v327*/
	v_max3_num_f32 v78 /*v334*/, v72 /*v328*/, v73 /*v329*/, v82 /*v338*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v174 /*v430*/, v151 /*v407*/
	v_cndmask_b32_e64 v171 /*v427*/, v79 /*v335*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	v_cmp_gt_i32_e64 s3, v175 /*v431*/, v151 /*v407*/
	v_cndmask_b32_e64 v172 /*v428*/, v80 /*v336*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	v_cmp_lt_i32_e64 s4, v175 /*v431*/, v155 /*v411*/
	v_cndmask_b32_e64 v173 /*v429*/, v81 /*v337*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v174 /*v430*/, v155 /*v411*/
	v_cmp_gt_i32_e64 s5, v176 /*v432*/, v151 /*v407*/
	v_cmp_lt_i32_e64 s6, v176 /*v432*/, v155 /*v411*/
	v_max3_num_f32 v80 /*v336*/, v83 /*v339*/, v132 /*v388*/, v133 /*v389*/
	v_max3_num_f32 v74 /*v330*/, v74 /*v330*/, v68 /*v324*/, v76 /*v332*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v177 /*v433*/, v151 /*v407*/
	v_cndmask_b32_e64 v174 /*v430*/, v90 /*v346*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	v_cmp_gt_i32_e64 s3, v178 /*v434*/, v151 /*v407*/
	v_cndmask_b32_e64 v175 /*v431*/, v91 /*v347*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	v_cmp_lt_i32_e64 s4, v178 /*v434*/, v155 /*v411*/
	v_cndmask_b32_e64 v176 /*v432*/, v92 /*v348*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v177 /*v433*/, v155 /*v411*/
	v_cmp_gt_i32_e64 s5, v84 /*v340*/, v151 /*v407*/
	v_cmp_lt_i32_e64 s6, v84 /*v340*/, v155 /*v411*/
	v_max3_num_f32 v84 /*v340*/, v86 /*v342*/, v87 /*v343*/, v88 /*v344*/
	v_max3_num_f32 v90 /*v346*/, v89 /*v345*/, v158 /*v414*/, v159 /*v415*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v85 /*v341*/, v151 /*v407*/
	v_cndmask_b32_e64 v177 /*v433*/, v93 /*v349*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	v_cmp_gt_i32_e64 s3, v180 /*v436*/, v151 /*v407*/
	v_cndmask_b32_e64 v178 /*v434*/, v94 /*v350*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	v_cmp_lt_i32_e64 s4, v180 /*v436*/, v155 /*v411*/
	v_cndmask_b32_e64 v179 /*v435*/, v95 /*v351*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v85 /*v341*/, v155 /*v411*/
	v_cmp_gt_i32_e64 s5, v181 /*v437*/, v151 /*v407*/
	v_cmp_lt_i32_e64 s6, v181 /*v437*/, v155 /*v411*/
	v_max3_num_f32 v92 /*v348*/, v160 /*v416*/, v161 /*v417*/, v102 /*v358*/
	v_max3_num_f32 v93 /*v349*/, v103 /*v359*/, v104 /*v360*/, v105 /*v361*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v183 /*v439*/, v151 /*v407*/
	v_cndmask_b32_e64 v180 /*v436*/, v96 /*v352*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	v_cmp_gt_i32_e64 s3, v184 /*v440*/, v151 /*v407*/
	v_cndmask_b32_e64 v181 /*v437*/, v97 /*v353*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	v_cmp_lt_i32_e64 s4, v184 /*v440*/, v155 /*v411*/
	v_cndmask_b32_e64 v182 /*v438*/, v106 /*v362*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v183 /*v439*/, v155 /*v411*/
	v_cmp_gt_i32_e64 s5, v98 /*v354*/, v151 /*v407*/
	v_cmp_lt_i32_e64 s6, v98 /*v354*/, v155 /*v411*/
	v_max3_num_f32 v94 /*v350*/, v162 /*v418*/, v163 /*v419*/, v164 /*v420*/
	v_max3_num_f32 v95 /*v351*/, v165 /*v421*/, v118 /*v374*/, v119 /*v375*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v99 /*v355*/, v151 /*v407*/
	v_cndmask_b32_e64 v183 /*v439*/, v107 /*v363*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	v_cmp_gt_i32_e64 s3, v100 /*v356*/, v151 /*v407*/
	v_cndmask_b32_e64 v184 /*v440*/, v108 /*v364*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	v_cmp_lt_i32_e64 s4, v100 /*v356*/, v155 /*v411*/
	v_cndmask_b32_e64 v185 /*v441*/, v109 /*v365*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v99 /*v355*/, v155 /*v411*/
	v_cmp_gt_i32_e64 s5, v101 /*v357*/, v151 /*v407*/
	v_cmp_lt_i32_e64 s6, v101 /*v357*/, v155 /*v411*/
	v_max3_num_f32 v76 /*v332*/, v80 /*v336*/, v84 /*v340*/, v90 /*v346*/
	v_max3_num_f32 v77 /*v333*/, v169 /*v425*/, v170 /*v426*/, v171 /*v427*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v189 /*v445*/, v151 /*v407*/
	v_cndmask_b32_e64 v186 /*v442*/, v110 /*v366*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	v_cmp_gt_i32_e64 s3, v190 /*v446*/, v151 /*v407*/
	v_cndmask_b32_e64 v187 /*v443*/, v111 /*v367*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	v_cmp_lt_i32_e64 s4, v190 /*v446*/, v155 /*v411*/
	v_cndmask_b32_e64 v188 /*v444*/, v112 /*v368*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v189 /*v445*/, v155 /*v411*/
	v_cmp_gt_i32_e64 s5, v191 /*v447*/, v151 /*v407*/
	v_cmp_lt_i32_e64 s6, v191 /*v447*/, v155 /*v411*/
	v_max3_num_f32 v81 /*v337*/, v175 /*v431*/, v176 /*v432*/, v177 /*v433*/
	v_max3_num_f32 v85 /*v341*/, v178 /*v434*/, v179 /*v435*/, v180 /*v436*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v192 /*v448*/, v151 /*v407*/
	v_cndmask_b32_e64 v189 /*v445*/, v113 /*v369*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	v_cmp_gt_i32_e64 s3, v114 /*v370*/, v151 /*v407*/
	v_cndmask_b32_e64 v190 /*v446*/, v122 /*v378*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	v_cmp_lt_i32_e64 s4, v114 /*v370*/, v155 /*v411*/
	v_cndmask_b32_e64 v191 /*v447*/, v123 /*v379*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v192 /*v448*/, v155 /*v411*/
	v_cmp_gt_i32_e64 s5, v115 /*v371*/, v151 /*v407*/
	v_cmp_lt_i32_e64 s6, v115 /*v371*/, v155 /*v411*/
	v_max3_num_f32 v91 /*v347*/, v181 /*v437*/, v182 /*v438*/, v183 /*v439*/
	v_max3_num_f32 v90 /*v346*/, v92 /*v348*/, v93 /*v349*/, v94 /*v350*/
	s_or_b32 s2, s2, vcc_lo
	v_max3_num_f32 v92 /*v348*/, v95 /*v351*/, v120 /*v376*/, v121 /*v377*/
	v_max3_num_f32 v74 /*v330*/, v74 /*v330*/, v78 /*v334*/, v76 /*v332*/
	v_cndmask_b32_e64 v192 /*v448*/, v124 /*v380*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	v_cmp_gt_i32_e32 vcc_lo, v116 /*v372*/, v151 /*v407*/
	v_cndmask_b32_e64 v193 /*v449*/, v125 /*v381*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	v_max3_num_f32 v79 /*v335*/, v172 /*v428*/, v173 /*v429*/, v174 /*v430*/
	v_cndmask_b32_e64 v194 /*v450*/, v126 /*v382*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v116 /*v372*/, v155 /*v411*/
	v_max3_num_f32 v75 /*v331*/, v75 /*v331*/, v168 /*v424*/, v77 /*v333*/
	v_max3_num_f32 v77 /*v333*/, v81 /*v337*/, v85 /*v341*/, v91 /*v347*/
	v_max3_num_f32 v74 /*v330*/, v74 /*v330*/, v90 /*v346*/, v92 /*v348*/
	v_cmp_gt_i32_e64 s3, v117 /*v373*/, v151 /*v407*/
	v_cmp_lt_i32_e64 s4, v117 /*v373*/, v155 /*v411*/
	s_or_b32 s2, s2, vcc_lo
	v_max3_num_f32 v75 /*v331*/, v75 /*v331*/, v79 /*v335*/, v77 /*v333*/
	v_mov_b32_e32 v77 /*v333*/, v74 /*v330*/
	v_cmp_gt_i32_e64 s5, v195 /*v451*/, v151 /*v407*/
	v_cmp_lt_i32_e64 s6, v195 /*v451*/, v155 /*v411*/
	v_cndmask_b32_e64 v195 /*v451*/, v127 /*v383*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	v_max3_num_f32 v80 /*v336*/, v184 /*v440*/, v185 /*v441*/, v186 /*v442*/
	v_cndmask_b32_e64 v196 /*v452*/, v128 /*v384*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	v_max3_num_f32 v84 /*v340*/, v187 /*v443*/, v188 /*v444*/, v189 /*v445*/
	v_cndmask_b32_e64 v197 /*v453*/, v129 /*v385*/, 0xff800000, s2
	v_max3_num_f32 v76 /*v332*/, v190 /*v446*/, v191 /*v447*/, v192 /*v448*/
	v_max3_num_f32 v78 /*v334*/, v193 /*v449*/, v194 /*v450*/, v195 /*v451*/
	v_permlanex16_b32 v77 /*v333*/, v77 /*v333*/, s70, 0xfedcba98
	v_max3_num_f32 v76 /*v332*/, v80 /*v336*/, v84 /*v340*/, v76 /*v332*/
	v_max3_num_f32 v78 /*v334*/, v78 /*v334*/, v196 /*v452*/, v197 /*v453*/
	v_max3_num_f32 v75 /*v331*/, v75 /*v331*/, v76 /*v332*/, v78 /*v334*/
	v_mov_b32_e32 v76 /*v332*/, v75 /*v331*/
	v_permlanex16_b32 v76 /*v332*/, v76 /*v332*/, s70, 0xfedcba98
	v_dual_max_num_f32 v74 /*v330*/, v74 /*v330*/, v77 /*v333*/ :: v_dual_max_num_f32 v75 /*v331*/, v75 /*v331*/, v76 /*v332*/
	v_sub_f32_e32 v77 /*v333*/, v74 /*v330*/, v130 /*v386*/
	v_dual_max_num_f32 v74 /*v330*/, v130 /*v386*/, v74 /*v330*/ :: v_dual_sub_f32 v76 /*v332*/, v75 /*v331*/, v135 /*v391*/
	v_cmp_lt_f32_e32 vcc_lo, 0x41000000, v77 /*v333*/
	s_cmp_eq_u32 vcc_lo, 0
	s_cselect_b32 s2, -1, 0
	v_dual_cndmask_b32 v156 /*v412*/, v74 /*v330*/, v130 /*v386*/, s2 :: v_dual_max_num_f32 v74 /*v330*/, v135 /*v391*/, v75 /*v331*/
	v_cmp_lt_f32_e64 s2, 0x41000000, v76 /*v332*/
	v_mul_f32_e32 v124 /*v380*/, 0xbfb8aa3b, v156 /*v412*/
	s_cmp_lg_u32 s2, 0
	s_cselect_b32 s3, -1, 0
	s_cmp_eq_u32 s2, 0
	v_pk_fma_f32 v[72:73] /*v[328:329]*/, v[72:73] /*v[328:329]*/, s[90:91], v[124:125] /*v[380:381]*/ op_sel_hi:[1,0,0]
	s_cselect_b32 s2, -1, 0
	v_pk_fma_f32 v[80:81] /*v[336:337]*/, v[132:133] /*v[388:389]*/, s[90:91], v[124:125] /*v[380:381]*/ op_sel_hi:[1,0,0]
	v_cndmask_b32_e64 v157 /*v413*/, v74 /*v330*/, v135 /*v391*/, s2
	v_pk_fma_f32 v[66:67] /*v[322:323]*/, v[66:67] /*v[322:323]*/, s[90:91], v[124:125] /*v[380:381]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[74:75] /*v[330:331]*/, v[68:69] /*v[324:325]*/, s[90:91], v[124:125] /*v[380:381]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[76:77] /*v[332:333]*/, v[70:71] /*v[326:327]*/, s[90:91], v[124:125] /*v[380:381]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v84 /*v340*/, v72 /*v328*/
	v_mul_f32_e32 v134 /*v390*/, 0xbfb8aa3b, v157 /*v413*/
	v_exp_f32_e32 v90 /*v346*/, v73 /*v329*/
	v_nop
	v_pk_fma_f32 v[72:73] /*v[328:329]*/, v[86:87] /*v[342:343]*/, s[90:91], v[124:125] /*v[380:381]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v100 /*v356*/, v80 /*v336*/
	v_exp_f32_e32 v106 /*v362*/, v81 /*v337*/
	v_nop
	v_pk_fma_f32 v[80:81] /*v[336:337]*/, v[158:159] /*v[414:415]*/, s[90:91], v[124:125] /*v[380:381]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[86:87] /*v[342:343]*/, v[160:161] /*v[416:417]*/, s[90:91], v[124:125] /*v[380:381]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[132:133] /*v[388:389]*/, v[166:167] /*v[422:423]*/, s[90:91], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[158:159] /*v[414:415]*/, v[168:169] /*v[424:425]*/, s[90:91], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[160:161] /*v[416:417]*/, v[170:171] /*v[426:427]*/, s[90:91], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v68 /*v324*/, v67 /*v323*/
	v_exp_f32_e32 v70 /*v326*/, v74 /*v330*/
	v_exp_f32_e32 v74 /*v330*/, v75 /*v331*/
	v_pk_fma_f32 v[78:79] /*v[334:335]*/, v[82:83] /*v[338:339]*/, s[90:91], v[124:125] /*v[380:381]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v82 /*v338*/, v77 /*v333*/
	v_exp_f32_e32 v67 /*v323*/, v132 /*v388*/
	v_exp_f32_e32 v69 /*v325*/, v133 /*v389*/
	v_exp_f32_e32 v71 /*v327*/, v158 /*v414*/
	v_pk_fma_f32 v[132:133] /*v[388:389]*/, v[172:173] /*v[428:429]*/, s[90:91], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v75 /*v331*/, v159 /*v415*/
	v_exp_f32_e32 v77 /*v333*/, v160 /*v416*/
	v_pk_fma_f32 v[158:159] /*v[414:415]*/, v[174:175] /*v[430:431]*/, s[90:91], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v83 /*v339*/, v161 /*v417*/
	v_nop
	v_pk_fma_f32 v[160:161] /*v[416:417]*/, v[176:177] /*v[432:433]*/, s[90:91], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v92 /*v348*/, v78 /*v334*/
	v_exp_f32_e32 v98 /*v354*/, v79 /*v335*/
	v_nop
	v_pk_fma_f32 v[78:79] /*v[334:335]*/, v[88:89] /*v[344:345]*/, s[90:91], v[124:125] /*v[380:381]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v85 /*v341*/, v132 /*v388*/
	v_exp_f32_e32 v91 /*v347*/, v133 /*v389*/
	v_exp_f32_e32 v93 /*v349*/, v158 /*v414*/
	v_pk_fma_f32 v[132:133] /*v[388:389]*/, v[178:179] /*v[434:435]*/, s[90:91], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v99 /*v355*/, v159 /*v415*/
	v_exp_f32_e32 v101 /*v357*/, v160 /*v416*/
	v_pk_fma_f32 v[158:159] /*v[414:415]*/, v[180:181] /*v[436:437]*/, s[90:91], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v107 /*v363*/, v161 /*v417*/
	v_nop
	v_pk_fma_f32 v[160:161] /*v[416:417]*/, v[182:183] /*v[438:439]*/, s[90:91], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v114 /*v370*/, v73 /*v329*/
	v_exp_f32_e32 v122 /*v378*/, v79 /*v335*/
	v_pk_fma_f32 v[88:89] /*v[344:345]*/, v[102:103] /*v[358:359]*/, s[90:91], v[124:125] /*v[380:381]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[96:97] /*v[352:353]*/, v[104:105] /*v[360:361]*/, s[90:91], v[124:125] /*v[380:381]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v109 /*v365*/, v132 /*v388*/
	v_exp_f32_e32 v115 /*v371*/, v133 /*v389*/
	v_exp_f32_e32 v117 /*v373*/, v158 /*v414*/
	v_pk_fma_f32 v[132:133] /*v[388:389]*/, v[184:185] /*v[440:441]*/, s[90:91], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v123 /*v379*/, v159 /*v415*/
	v_exp_f32_e32 v73 /*v329*/, v160 /*v416*/
	v_pk_fma_f32 v[158:159] /*v[414:415]*/, v[186:187] /*v[442:443]*/, s[90:91], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v79 /*v335*/, v161 /*v417*/
	v_nop
	v_pk_fma_f32 v[160:161] /*v[416:417]*/, v[188:189] /*v[444:445]*/, s[90:91], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v66 /*v322*/, v66 /*v322*/
	v_exp_f32_e32 v76 /*v332*/, v76 /*v332*/
	v_exp_f32_e32 v108 /*v364*/, v72 /*v328*/
	v_exp_f32_e32 v116 /*v372*/, v78 /*v334*/
	v_exp_f32_e32 v72 /*v328*/, v80 /*v336*/
	v_exp_f32_e32 v78 /*v334*/, v81 /*v337*/
	v_exp_f32_e32 v80 /*v336*/, v86 /*v342*/
	v_exp_f32_e32 v86 /*v342*/, v87 /*v343*/
	v_pk_fma_f32 v[104:105] /*v[360:361]*/, v[162:163] /*v[418:419]*/, s[90:91], v[124:125] /*v[380:381]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v94 /*v350*/, v89 /*v345*/
	v_pk_fma_f32 v[112:113] /*v[368:369]*/, v[164:165] /*v[420:421]*/, s[90:91], v[124:125] /*v[380:381]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v102 /*v358*/, v97 /*v353*/
	v_exp_f32_e32 v81 /*v337*/, v132 /*v388*/
	v_exp_f32_e32 v87 /*v343*/, v133 /*v389*/
	v_exp_f32_e32 v89 /*v345*/, v158 /*v414*/
	v_pk_fma_f32 v[132:133] /*v[388:389]*/, v[190:191] /*v[446:447]*/, s[90:91], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v95 /*v351*/, v159 /*v415*/
	v_exp_f32_e32 v97 /*v353*/, v160 /*v416*/
	v_pk_fma_f32 v[158:159] /*v[414:415]*/, v[192:193] /*v[448:449]*/, s[90:91], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v103 /*v359*/, v161 /*v417*/
	v_nop
	v_pk_fma_f32 v[160:161] /*v[416:417]*/, v[194:195] /*v[450:451]*/, s[90:91], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v96 /*v352*/, v96 /*v352*/
	v_pk_fma_f32 v[126:127] /*v[382:383]*/, v[118:119] /*v[374:375]*/, s[90:91], v[124:125] /*v[380:381]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v110 /*v366*/, v105 /*v361*/
	v_pk_fma_f32 v[128:129] /*v[384:385]*/, v[120:121] /*v[376:377]*/, s[90:91], v[124:125] /*v[380:381]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v118 /*v374*/, v113 /*v369*/
	v_exp_f32_e32 v105 /*v361*/, v132 /*v388*/
	v_exp_f32_e32 v111 /*v367*/, v133 /*v389*/
	v_exp_f32_e32 v113 /*v369*/, v158 /*v414*/
	v_pk_fma_f32 v[132:133] /*v[388:389]*/, v[196:197] /*v[452:453]*/, s[90:91], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v119 /*v375*/, v159 /*v415*/
	v_exp_f32_e32 v121 /*v377*/, v160 /*v416*/
	v_exp_f32_e32 v125 /*v381*/, v161 /*v417*/
	v_pk_add_f32 v[158:159] /*v[414:415]*/, v[66:67] /*v[322:323]*/, v[68:69] /*v[324:325]*/
	v_pk_add_f32 v[160:161] /*v[416:417]*/, v[74:75] /*v[330:331]*/, v[76:77] /*v[332:333]*/
	v_pk_add_f32 v[162:163] /*v[418:419]*/, v[98:99] /*v[354:355]*/, v[100:101] /*v[356:357]*/
	v_pk_add_f32 v[164:165] /*v[420:421]*/, v[108:109] /*v[364:365]*/, v[114:115] /*v[370:371]*/
	v_exp_f32_e32 v88 /*v344*/, v88 /*v344*/
	v_exp_f32_e32 v104 /*v360*/, v104 /*v360*/
	v_exp_f32_e32 v120 /*v376*/, v126 /*v382*/
	v_exp_f32_e32 v124 /*v380*/, v127 /*v383*/
	v_exp_f32_e32 v126 /*v382*/, v128 /*v384*/
	v_exp_f32_e32 v128 /*v384*/, v129 /*v385*/
	v_exp_f32_e32 v127 /*v383*/, v132 /*v388*/
	v_exp_f32_e32 v129 /*v385*/, v133 /*v389*/
	v_nop
	v_pk_add_f32 v[132:133] /*v[388:389]*/, v[84:85] /*v[340:341]*/, v[90:91] /*v[346:347]*/
	v_pk_add_f32 v[158:159] /*v[414:415]*/, v[70:71] /*v[326:327]*/, v[158:159] /*v[414:415]*/
	v_pk_add_f32 v[160:161] /*v[416:417]*/, v[82:83] /*v[338:339]*/, v[160:161] /*v[416:417]*/
	v_pk_add_f32 v[166:167] /*v[422:423]*/, v[122:123] /*v[378:379]*/, v[72:73] /*v[328:329]*/
	v_pk_add_f32 v[162:163] /*v[418:419]*/, v[106:107] /*v[362:363]*/, v[162:163] /*v[418:419]*/
	v_pk_add_f32 v[168:169] /*v[424:425]*/, v[80:81] /*v[336:337]*/, v[86:87] /*v[342:343]*/
	v_pk_add_f32 v[164:165] /*v[420:421]*/, v[116:117] /*v[372:373]*/, v[164:165] /*v[420:421]*/
	v_pk_add_f32 v[170:171] /*v[426:427]*/, v[94:95] /*v[350:351]*/, v[96:97] /*v[352:353]*/
	v_exp_f32_e32 v112 /*v368*/, v112 /*v368*/
	v_pk_add_f32 v[132:133] /*v[388:389]*/, v[92:93] /*v[348:349]*/, v[132:133] /*v[388:389]*/
	v_pk_add_f32 v[166:167] /*v[422:423]*/, v[78:79] /*v[334:335]*/, v[166:167] /*v[422:423]*/
	v_pk_add_f32 v[172:173] /*v[428:429]*/, v[104:105] /*v[360:361]*/, v[110:111] /*v[366:367]*/
	v_pk_add_f32 v[168:169] /*v[424:425]*/, v[88:89] /*v[344:345]*/, v[168:169] /*v[424:425]*/
	v_pk_add_f32 v[158:159] /*v[414:415]*/, v[158:159] /*v[414:415]*/, v[160:161] /*v[416:417]*/
	v_pk_add_f32 v[160:161] /*v[416:417]*/, v[102:103] /*v[358:359]*/, v[170:171] /*v[426:427]*/
	v_pk_add_f32 v[162:163] /*v[418:419]*/, v[162:163] /*v[418:419]*/, v[164:165] /*v[420:421]*/
	v_pk_add_f32 v[164:165] /*v[420:421]*/, v[112:113] /*v[368:369]*/, v[172:173] /*v[428:429]*/
	v_pk_add_f32 v[170:171] /*v[426:427]*/, v[118:119] /*v[374:375]*/, v[120:121] /*v[376:377]*/
	v_pk_add_f32 v[132:133] /*v[388:389]*/, v[132:133] /*v[388:389]*/, v[158:159] /*v[414:415]*/
	v_pk_add_f32 v[158:159] /*v[414:415]*/, v[168:169] /*v[424:425]*/, v[160:161] /*v[416:417]*/
	v_pk_add_f32 v[160:161] /*v[416:417]*/, v[166:167] /*v[422:423]*/, v[162:163] /*v[418:419]*/
	v_pk_add_f32 v[166:167] /*v[422:423]*/, v[126:127] /*v[382:383]*/, v[128:129] /*v[384:385]*/
	v_pk_add_f32 v[162:163] /*v[418:419]*/, v[124:125] /*v[380:381]*/, v[170:171] /*v[426:427]*/
	v_sub_f32_e32 v134 /*v390*/, v130 /*v386*/, v156 /*v412*/
	v_pk_add_f32 v[158:159] /*v[414:415]*/, v[164:165] /*v[420:421]*/, v[158:159] /*v[414:415]*/
	v_pk_add_f32 v[132:133] /*v[388:389]*/, v[132:133] /*v[388:389]*/, v[160:161] /*v[416:417]*/
	v_pk_add_f32 v[160:161] /*v[416:417]*/, v[166:167] /*v[422:423]*/, v[162:163] /*v[418:419]*/
	v_mul_f32_e32 v134 /*v390*/, 0x3fb8aa3b, v134 /*v390*/
	v_pk_add_f32 v[132:133] /*v[388:389]*/, v[158:159] /*v[414:415]*/, v[132:133] /*v[388:389]*/
	v_exp_f32_e32 v134 /*v390*/, v134 /*v390*/
	v_pk_add_f32 v[130:131] /*v[386:387]*/, v[160:161] /*v[416:417]*/, v[132:133] /*v[388:389]*/
	v_dual_mov_b32 v132 /*v388*/, v130 /*v386*/ :: v_dual_mov_b32 v133 /*v389*/, v131 /*v387*/
	v_permlanex16_b32 v132 /*v388*/, v132 /*v388*/, s70, 0xfedcba98
	v_permlanex16_b32 v133 /*v389*/, v133 /*v389*/, s70, 0xfedcba98
	s_set_vgpr_msb 0x5500
	s_cbranch_vccz .LBB0_8
	s_set_vgpr_msb 4
	v_pk_mul_f32 v[126:127], v[126:127], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[124:125], v[124:125], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[122:123], v[122:123], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[120:121], v[120:121], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[118:119], v[118:119], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[116:117], v[116:117], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[114:115], v[114:115], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[112:113], v[112:113], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[110:111], v[110:111], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[108:109], v[108:109], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[106:107], v[106:107], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[104:105], v[104:105], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[102:103], v[102:103], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[100:101], v[100:101], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[98:99], v[98:99], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[96:97], v[96:97], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[94:95], v[94:95], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[92:93], v[92:93], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[90:91], v[90:91], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[88:89], v[88:89], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[86:87], v[86:87], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[84:85], v[84:85], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[82:83], v[82:83], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[80:81], v[80:81], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[78:79], v[78:79], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[76:77], v[76:77], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[74:75], v[74:75], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[72:73], v[72:73], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[70:71], v[70:71], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[68:69], v[68:69], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[66:67], v[66:67], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[64:65], v[64:65], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x400
.LBB0_8:
	s_set_vgpr_msb 0x45
	v_sub_f32_e32 v135 /*v391*/, v135 /*v391*/, v157 /*v413*/
	s_and_b32 s2, s3, exec_lo
	s_cselect_b32 s2, 1, 0
	s_cmp_lg_u32 s2, 1
	v_mul_f32_e32 v135 /*v391*/, 0x3fb8aa3b, v135 /*v391*/
	v_exp_f32_e32 v135 /*v391*/, v135 /*v391*/
	s_set_vgpr_msb 0x4500
	s_cbranch_scc1 .LBB0_3
	v_nop
	s_set_vgpr_msb 0x41
	v_mov_b32_e32 v158 /*v414*/, v135 /*v391*/
	s_set_vgpr_msb 0x4104
	v_pk_mul_f32 v[62:63], v[62:63], v[158:159] /*v[414:415]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[60:61], v[60:61], v[158:159] /*v[414:415]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59], v[58:59], v[158:159] /*v[414:415]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[56:57], v[56:57], v[158:159] /*v[414:415]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[54:55], v[54:55], v[158:159] /*v[414:415]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[52:53], v[52:53], v[158:159] /*v[414:415]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[50:51], v[50:51], v[158:159] /*v[414:415]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[48:49], v[48:49], v[158:159] /*v[414:415]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[46:47], v[46:47], v[158:159] /*v[414:415]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[44:45], v[44:45], v[158:159] /*v[414:415]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[42:43], v[42:43], v[158:159] /*v[414:415]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[40:41], v[40:41], v[158:159] /*v[414:415]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39], v[38:39], v[158:159] /*v[414:415]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[36:37], v[36:37], v[158:159] /*v[414:415]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35], v[34:35], v[158:159] /*v[414:415]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[32:33], v[32:33], v[158:159] /*v[414:415]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[30:31], v[30:31], v[158:159] /*v[414:415]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[28:29], v[28:29], v[158:159] /*v[414:415]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[26:27], v[26:27], v[158:159] /*v[414:415]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25], v[24:25], v[158:159] /*v[414:415]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[22:23], v[22:23], v[158:159] /*v[414:415]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21], v[20:21], v[158:159] /*v[414:415]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[18:19], v[18:19], v[158:159] /*v[414:415]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[16:17], v[16:17], v[158:159] /*v[414:415]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[14:15], v[158:159] /*v[414:415]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13], v[12:13], v[158:159] /*v[414:415]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[10:11], v[158:159] /*v[414:415]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[8:9], v[158:159] /*v[414:415]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[6:7], v[158:159] /*v[414:415]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[4:5], v[158:159] /*v[414:415]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[2:3], v[2:3], v[158:159] /*v[414:415]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[0:1], v[0:1], v[158:159] /*v[414:415]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x400
	s_branch .LBB0_3
.LBB0_10:
	s_and_b32 vcc_lo, exec_lo, s3
	s_cbranch_vccnz .LBB0_34
	s_branch .LBB0_65
.LBB0_11:
	v_mov_b32_e32 v0, 0
	s_set_vgpr_msb 0x41
	v_dual_mov_b32 v65 /*v321*/, 1.0 :: v_dual_mov_b32 v149 /*v405*/, v142 /*v398*/
	s_wait_loadcnt 0x0
	v_mov_b32_e32 v156 /*v412*/, v135 /*v391*/
	s_set_vgpr_msb 0x4100
	v_dual_mov_b32 v1, v0 :: v_dual_mov_b32 v2, v0
	v_dual_mov_b32 v3, v0 :: v_dual_mov_b32 v4, v0
	v_dual_mov_b32 v5, v0 :: v_dual_mov_b32 v6, v0
	v_mov_b32_e32 v7, v0
	s_set_vgpr_msb 0x41
	v_mov_b32_e32 v64 /*v320*/, v65 /*v321*/
	s_set_vgpr_msb 0x4100
	v_mov_b64_e32 v[10:11], v[2:3]
	v_mov_b64_e32 v[12:13], v[4:5]
	v_mov_b64_e32 v[8:9], v[0:1]
	v_mov_b64_e32 v[14:15], v[6:7]
	v_mov_b64_e32 v[22:23], v[6:7]
	v_mov_b64_e32 v[20:21], v[4:5]
	v_mov_b64_e32 v[18:19], v[2:3]
	v_mov_b64_e32 v[16:17], v[0:1]
	v_mov_b64_e32 v[30:31], v[6:7]
	v_mov_b64_e32 v[28:29], v[4:5]
	v_mov_b64_e32 v[26:27], v[2:3]
	v_mov_b64_e32 v[24:25], v[0:1]
	v_mov_b64_e32 v[38:39], v[6:7]
	v_mov_b64_e32 v[36:37], v[4:5]
	v_mov_b64_e32 v[34:35], v[2:3]
	v_mov_b64_e32 v[32:33], v[0:1]
	v_mov_b64_e32 v[46:47], v[6:7]
	v_mov_b64_e32 v[44:45], v[4:5]
	v_mov_b64_e32 v[42:43], v[2:3]
	v_mov_b64_e32 v[40:41], v[0:1]
	v_mov_b64_e32 v[54:55], v[6:7]
	v_mov_b64_e32 v[52:53], v[4:5]
	v_mov_b64_e32 v[50:51], v[2:3]
	v_mov_b64_e32 v[48:49], v[0:1]
	v_mov_b64_e32 v[62:63], v[6:7]
	v_mov_b64_e32 v[60:61], v[4:5]
	v_mov_b64_e32 v[58:59], v[2:3]
	v_mov_b64_e32 v[56:57], v[0:1]
	v_mov_b64_e32 v[70:71], v[6:7]
	v_mov_b64_e32 v[68:69], v[4:5]
	v_mov_b64_e32 v[66:67], v[2:3]
	v_mov_b64_e32 v[64:65], v[0:1]
	v_mov_b64_e32 v[78:79], v[6:7]
	v_mov_b64_e32 v[76:77], v[4:5]
	v_mov_b64_e32 v[74:75], v[2:3]
	v_mov_b64_e32 v[72:73], v[0:1]
	v_mov_b64_e32 v[86:87], v[6:7]
	v_mov_b64_e32 v[84:85], v[4:5]
	v_mov_b64_e32 v[82:83], v[2:3]
	v_mov_b64_e32 v[80:81], v[0:1]
	v_mov_b64_e32 v[94:95], v[6:7]
	v_mov_b64_e32 v[92:93], v[4:5]
	v_mov_b64_e32 v[90:91], v[2:3]
	v_mov_b64_e32 v[88:89], v[0:1]
	v_mov_b64_e32 v[102:103], v[6:7]
	v_mov_b64_e32 v[100:101], v[4:5]
	v_mov_b64_e32 v[98:99], v[2:3]
	v_mov_b64_e32 v[96:97], v[0:1]
	v_mov_b64_e32 v[110:111], v[6:7]
	v_mov_b64_e32 v[108:109], v[4:5]
	v_mov_b64_e32 v[106:107], v[2:3]
	v_mov_b64_e32 v[104:105], v[0:1]
	v_mov_b64_e32 v[118:119], v[6:7]
	v_mov_b64_e32 v[116:117], v[4:5]
	v_mov_b64_e32 v[114:115], v[2:3]
	v_mov_b64_e32 v[112:113], v[0:1]
	v_mov_b64_e32 v[126:127], v[6:7]
	v_mov_b64_e32 v[124:125], v[4:5]
	v_mov_b64_e32 v[122:123], v[2:3]
	v_mov_b64_e32 v[120:121], v[0:1]
	s_branch .LBB0_13
.LBB0_12:
	s_load_b64 s[70:71], s[0:1], 0x50 nv
	s_set_vgpr_msb 0x41
	v_mov_b32_e32 v135 /*v391*/, v157 /*v413*/
	s_set_vgpr_msb 0x4100
.LBB0_13:
	s_mov_b32 s59, 0
	s_cmp_le_i32 s7, s11
	s_mov_b32 s83, s59
	s_cbranch_scc1 .LBB0_22
	s_add_co_i32 s2, s19, s35
	s_lshl_b32 s3, s88, 6
	s_sub_co_i32 s2, s2, s18
	s_sub_co_i32 s5, s9, s3
	s_sub_co_i32 s2, s2, s16
	s_add_co_i32 s6, s97, s3
	s_max_i32 s2, s2, 0
	s_mov_b32 s4, 1
	s_lshr_b32 s2, s2, 6
	s_mov_b32 s56, 16
	s_min_u32 s2, s2, s98
	s_mov_b32 s55, 0x800000
	s_sub_co_i32 s2, 0, s2
	s_sub_co_i32 s11, s5, 64
	s_ashr_i32 s3, s2, 31
	s_add_co_i32 s28, s6, 64
	s_add_nc_u64 s[30:31], s[2:3], 1
	s_mov_b32 s53, 0xffff0000
	s_mov_b32 s52, 0x7510000
	s_mov_b32 s44, 0xf510000
	s_mov_b32 s31, 0x76543210
	s_mov_b32 s90, 0x3fb8aa3b
	s_branch .LBB0_16
.LBB0_15:
	s_set_vgpr_msb 0x45
	v_cvt_pk_bf16_f32 v167 /*v423*/, v116 /*v372*/, v122 /*v378*/
	v_cvt_pk_bf16_f32 v166 /*v422*/, v110 /*v366*/, v114 /*v370*/
	v_cvt_pk_bf16_f32 v165 /*v421*/, v102 /*v358*/, v106 /*v362*/
	v_cvt_pk_bf16_f32 v164 /*v420*/, v94 /*v350*/, v98 /*v354*/
	v_cvt_pk_bf16_f32 v163 /*v419*/, v84 /*v340*/, v90 /*v346*/
	v_cvt_pk_bf16_f32 v162 /*v418*/, v78 /*v334*/, v82 /*v338*/
	v_cvt_pk_bf16_f32 v161 /*v417*/, v72 /*v328*/, v74 /*v330*/
	v_cvt_pk_bf16_f32 v160 /*v416*/, v68 /*v324*/, v70 /*v326*/
	v_cvt_pk_bf16_f32 v175 /*v431*/, v117 /*v373*/, v123 /*v379*/
	v_cvt_pk_bf16_f32 v174 /*v430*/, v111 /*v367*/, v115 /*v371*/
	v_cvt_pk_bf16_f32 v173 /*v429*/, v103 /*v359*/, v107 /*v363*/
	v_cvt_pk_bf16_f32 v172 /*v428*/, v95 /*v351*/, v99 /*v355*/
	v_cvt_pk_bf16_f32 v171 /*v427*/, v85 /*v341*/, v91 /*v347*/
	v_cvt_pk_bf16_f32 v170 /*v426*/, v79 /*v335*/, v83 /*v339*/
	v_cvt_pk_bf16_f32 v169 /*v425*/, v73 /*v329*/, v75 /*v331*/
	v_cvt_pk_bf16_f32 v168 /*v424*/, v69 /*v325*/, v71 /*v327*/
	s_set_vgpr_msb 0x4505
	s_wait_dscnt 0x1d
	v_wmma_f32_16x16x32_bf16 v[120:127], v[56:63] /*v[312:319]*/, v[160:167] /*v[416:423]*/, v[120:127]
	s_set_vgpr_msb 0x545
	v_cvt_pk_bf16_f32 v75 /*v331*/, v127 /*v383*/, v129 /*v385*/
	v_cvt_pk_bf16_f32 v74 /*v330*/, v121 /*v377*/, v125 /*v381*/
	v_cvt_pk_bf16_f32 v73 /*v329*/, v113 /*v369*/, v119 /*v375*/
	v_cvt_pk_bf16_f32 v72 /*v328*/, v105 /*v361*/, v109 /*v365*/
	v_cvt_pk_bf16_f32 v71 /*v327*/, v97 /*v353*/, v101 /*v357*/
	v_cvt_pk_bf16_f32 v70 /*v326*/, v89 /*v345*/, v93 /*v349*/
	v_cvt_pk_bf16_f32 v69 /*v325*/, v81 /*v337*/, v87 /*v343*/
	s_set_vgpr_msb 0x4505
	v_wmma_f32_16x16x32_bf16 v[56:63], v[56:63] /*v[312:319]*/, v[168:175] /*v[424:431]*/, v[56:63]
	s_set_vgpr_msb 0x545
	v_cvt_pk_bf16_f32 v68 /*v324*/, v67 /*v323*/, v77 /*v333*/
	v_dual_fmac_f32 v152 /*v408*/, v132 /*v388*/, v64 /*v320*/ :: v_dual_fmac_f32 v133 /*v389*/, v134 /*v390*/, v65 /*v321*/
	v_cmp_lt_u64_e64 s2, s[92:93], s[82:83]
	v_nop
	v_cvt_pk_bf16_f32 v63 /*v319*/, v126 /*v382*/, v128 /*v384*/
	v_cvt_pk_bf16_f32 v62 /*v318*/, v120 /*v376*/, v124 /*v380*/
	v_cvt_pk_bf16_f32 v61 /*v317*/, v112 /*v368*/, v118 /*v374*/
	s_set_vgpr_msb 0x4505
	s_wait_dscnt 0x1c
	v_wmma_f32_16x16x32_bf16 v[112:119], v[40:47] /*v[296:303]*/, v[160:167] /*v[416:423]*/, v[112:119]
	s_set_vgpr_msb 0x545
	v_cvt_pk_bf16_f32 v60 /*v316*/, v104 /*v360*/, v108 /*v364*/
	v_cvt_pk_bf16_f32 v59 /*v315*/, v96 /*v352*/, v100 /*v356*/
	v_cvt_pk_bf16_f32 v58 /*v314*/, v88 /*v344*/, v92 /*v348*/
	v_cvt_pk_bf16_f32 v57 /*v313*/, v80 /*v336*/, v86 /*v342*/
	v_cvt_pk_bf16_f32 v56 /*v312*/, v66 /*v322*/, v76 /*v332*/
	v_dual_add_f32 v64 /*v320*/, v152 /*v408*/, v130 /*v386*/ :: v_dual_add_f32 v65 /*v321*/, v133 /*v389*/, v131 /*v387*/
	s_set_vgpr_msb 0x4505
	v_wmma_f32_16x16x32_bf16 v[48:55], v[40:47] /*v[296:303]*/, v[168:175] /*v[424:431]*/, v[48:55]
	s_set_vgpr_msb 0x541
	v_dual_mov_b32 v153 /*v409*/, v158 /*v414*/ :: v_dual_mov_b32 v152 /*v408*/, v157 /*v413*/
	v_dual_mov_b32 v135 /*v391*/, v150 /*v406*/ :: v_dual_mov_b32 v156 /*v412*/, v151 /*v407*/
	s_sub_co_i32 s11, s11, 64
	s_add_co_i32 s28, s28, 64
	s_and_b32 vcc_lo, exec_lo, s2
	s_set_vgpr_msb 0x4105
	s_wait_dscnt 0x15
	v_wmma_f32_16x16x32_bf16 v[104:111], v[24:31] /*v[280:287]*/, v[160:167] /*v[416:423]*/, v[104:111]
	s_mov_b64 s[88:89], s[92:93]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[40:47], v[24:31] /*v[280:287]*/, v[168:175] /*v[424:431]*/, v[40:47]
	v_wmma_f32_16x16x32_bf16 v[96:103], v[8:15] /*v[264:271]*/, v[160:167] /*v[416:423]*/, v[96:103]
	v_wmma_f32_16x16x32_bf16 v[32:39], v[8:15] /*v[264:271]*/, v[168:175] /*v[424:431]*/, v[32:39]
	s_set_vgpr_msb 0x504
	v_wmma_f32_16x16x32_bf16 v[88:95], v[248:255], v[160:167] /*v[416:423]*/, v[88:95]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[248:255], v[168:175] /*v[424:431]*/, v[24:31]
	v_wmma_f32_16x16x32_bf16 v[80:87], v[232:239], v[160:167] /*v[416:423]*/, v[80:87]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[232:239], v[168:175] /*v[424:431]*/, v[16:23]
	v_wmma_f32_16x16x32_bf16 v[72:79], v[216:223], v[160:167] /*v[416:423]*/, v[72:79]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[216:223], v[168:175] /*v[424:431]*/, v[8:15]
	v_wmma_f32_16x16x32_bf16 v[64:71], v[200:207], v[160:167] /*v[416:423]*/, v[64:71]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[200:207], v[168:175] /*v[424:431]*/, v[0:7]
	s_set_vgpr_msb 0x405
	v_wmma_f32_16x16x32_bf16 v[120:127], v[48:55] /*v[304:311]*/, v[56:63] /*v[312:319]*/, v[120:127]
	v_wmma_f32_16x16x32_bf16 v[56:63], v[48:55] /*v[304:311]*/, v[68:75] /*v[324:331]*/, v[56:63]
	v_wmma_f32_16x16x32_bf16 v[112:119], v[32:39] /*v[288:295]*/, v[56:63] /*v[312:319]*/, v[112:119]
	v_wmma_f32_16x16x32_bf16 v[48:55], v[32:39] /*v[288:295]*/, v[68:75] /*v[324:331]*/, v[48:55]
	v_wmma_f32_16x16x32_bf16 v[104:111], v[16:23] /*v[272:279]*/, v[56:63] /*v[312:319]*/, v[104:111]
	v_wmma_f32_16x16x32_bf16 v[40:47], v[16:23] /*v[272:279]*/, v[68:75] /*v[324:331]*/, v[40:47]
	v_wmma_f32_16x16x32_bf16 v[96:103], v[0:7] /*v[256:263]*/, v[56:63] /*v[312:319]*/, v[96:103]
	v_wmma_f32_16x16x32_bf16 v[32:39], v[0:7] /*v[256:263]*/, v[68:75] /*v[324:331]*/, v[32:39]
	s_set_vgpr_msb 0x504
	v_wmma_f32_16x16x32_bf16 v[88:95], v[240:247], v[56:63] /*v[312:319]*/, v[88:95]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[240:247], v[68:75] /*v[324:331]*/, v[24:31]
	v_wmma_f32_16x16x32_bf16 v[80:87], v[224:231], v[56:63] /*v[312:319]*/, v[80:87]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[224:231], v[68:75] /*v[324:331]*/, v[16:23]
	v_wmma_f32_16x16x32_bf16 v[72:79], v[208:215], v[56:63] /*v[312:319]*/, v[72:79]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[208:215], v[68:75] /*v[324:331]*/, v[8:15]
	v_wmma_f32_16x16x32_bf16 v[64:71], v[192:199], v[56:63] /*v[312:319]*/, v[64:71]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[192:199], v[68:75] /*v[324:331]*/, v[0:7]
	s_set_vgpr_msb 0x400
	s_cbranch_vccz .LBB0_23
.LBB0_16:
	s_set_vgpr_msb 0x41
	v_dual_mov_b32 v157 /*v413*/, v149 /*v405*/ :: v_dual_mov_b32 v149 /*v405*/, v152 /*v408*/
	v_dual_mov_b32 v158 /*v414*/, v148 /*v404*/ :: v_dual_mov_b32 v148 /*v404*/, v153 /*v409*/
	s_add_nc_u64 s[92:93], s[88:89], 1
	s_wait_tensorcnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	s_cmp_ge_i32 s92, s10
	s_set_vgpr_msb 0x4100
	s_cbranch_scc1 .LBB0_18
	s_add_co_i32 s5, s30, s88
	v_nop
	v_nop
	v_med3_i32 v192, s11, 0, 64
	s_lshr_b32 s6, s5, 31
	s_ashr_i32 s29, s28, 31
	s_add_co_i32 s6, s5, s6
	s_mul_u64 s[2:3], s[28:29], s[64:65]
	s_and_b32 s45, s6, 0x7ffe
	s_mul_u64 s[6:7], s[28:29], s[62:63]
	v_readfirstlane_b32 s29, v192
	s_sub_co_i32 s5, s5, s45
	s_lshl_b64 s[6:7], s[6:7], 1
	s_lshl_b32 s45, s5, 17
	s_add_nc_u64 s[6:7], s[86:87], s[6:7]
	s_sub_co_i32 s5, s29, s25
	s_lshl_b64 s[2:3], s[2:3], 1
	s_max_i32 s29, s5, 0
	s_add_nc_u64 s[6:7], s[26:27], s[6:7]
	s_lshl_b32 s29, s29, 16
	s_add_nc_u64 s[2:3], s[84:85], s[2:3]
	s_or_b32 s5, s104, s45
	s_bitset1_b32 s7, 31
	s_or_b32 s54, s29, 0x7fff
	s_mov_b32 s47, s55
	tensor_load_to_lds s[4:7], s[52:59]
	s_add_nc_u64 s[6:7], s[80:81], s[2:3]
	s_or_b32 s5, vcc_hi, s45
	s_bitset1_b32 s7, 31
	s_mov_b32 s45, s53
	s_mov_b32 s46, s54
	s_mov_b32 s48, s56
	s_mov_b32 s51, s59
	tensor_load_to_lds s[4:7], s[44:51]
.LBB0_18:
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 1
	ds_load_b128 v[192:195], v157 /*v413*/
	ds_load_b128 v[196:199], v157 /*v413*/ offset:32
	ds_load_b128 v[200:203], v157 /*v413*/ offset:64
	ds_load_b128 v[204:207], v157 /*v413*/ offset:96
	ds_load_b128 v[208:211], v157 /*v413*/ offset:128
	ds_load_b128 v[212:215], v157 /*v413*/ offset:160
	ds_load_b128 v[216:219], v157 /*v413*/ offset:192
	ds_load_b128 v[220:223], v157 /*v413*/ offset:224
	ds_load_b128 v[224:227], v157 /*v413*/ offset:4352
	ds_load_b128 v[228:231], v157 /*v413*/ offset:4384
	ds_load_b128 v[232:235], v157 /*v413*/ offset:4416
	ds_load_b128 v[236:239], v157 /*v413*/ offset:4448
	ds_load_b128 v[240:243], v157 /*v413*/ offset:4480
	ds_load_b128 v[244:247], v157 /*v413*/ offset:4512
	ds_load_b128 v[248:251], v157 /*v413*/ offset:4544
	ds_load_b128 v[252:255], v157 /*v413*/ offset:4576
	s_set_vgpr_msb 0x141
	ds_load_b128 v[0:3] /*v[256:259]*/, v157 /*v413*/ offset:8704
	ds_load_b128 v[4:7] /*v[260:263]*/, v157 /*v413*/ offset:8736
	ds_load_b128 v[8:11] /*v[264:267]*/, v157 /*v413*/ offset:8768
	ds_load_b128 v[12:15] /*v[268:271]*/, v157 /*v413*/ offset:8800
	ds_load_b128 v[16:19] /*v[272:275]*/, v157 /*v413*/ offset:8832
	ds_load_b128 v[20:23] /*v[276:279]*/, v157 /*v413*/ offset:8864
	ds_load_b128 v[24:27] /*v[280:283]*/, v157 /*v413*/ offset:8896
	ds_load_b128 v[28:31] /*v[284:287]*/, v157 /*v413*/ offset:8928
	ds_load_b128 v[32:35] /*v[288:291]*/, v157 /*v413*/ offset:13056
	ds_load_b128 v[36:39] /*v[292:295]*/, v157 /*v413*/ offset:13088
	ds_load_b128 v[40:43] /*v[296:299]*/, v157 /*v413*/ offset:13120
	ds_load_b128 v[44:47] /*v[300:303]*/, v157 /*v413*/ offset:13152
	ds_load_b128 v[48:51] /*v[304:307]*/, v157 /*v413*/ offset:13184
	ds_load_b128 v[52:55] /*v[308:311]*/, v157 /*v413*/ offset:13216
	ds_load_b128 v[56:59] /*v[312:315]*/, v157 /*v413*/ offset:13248
	ds_load_b128 v[60:63] /*v[316:319]*/, v157 /*v413*/ offset:13280
	s_set_vgpr_msb 0x4150
	s_wait_dscnt 0x1e
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[322:329]*/, v[192:199], v[128:135], 0
	v_wmma_f32_16x16x32_bf16 v[126:133] /*v[382:389]*/, v[192:199], v[160:167], 0
	s_wait_dscnt 0x16
	v_wmma_f32_16x16x32_bf16 v[94:101] /*v[350:357]*/, v[224:231], v[128:135], 0
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[322:329]*/, v[200:207], v[136:143], v[66:73] /*v[322:329]*/
	v_wmma_f32_16x16x32_bf16 v[126:133] /*v[382:389]*/, v[200:207], v[168:175], v[126:133] /*v[382:389]*/
	v_wmma_f32_16x16x32_bf16 v[160:167] /*v[416:423]*/, v[224:231], v[160:167], 0
	s_wait_dscnt 0x14
	v_wmma_f32_16x16x32_bf16 v[94:101] /*v[350:357]*/, v[232:239], v[136:143], v[94:101] /*v[350:357]*/
	s_set_vgpr_msb 0x5041
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x32_bf16 v[168:175] /*v[424:431]*/, v[0:7] /*v[256:263]*/, v[128:135], 0
	v_wmma_f32_16x16x32_bf16 v[176:183] /*v[432:439]*/, v[0:7] /*v[256:263]*/, v[160:167], 0
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x32_bf16 v[184:191] /*v[440:447]*/, v[32:39] /*v[288:295]*/, v[128:135], 0
	v_wmma_f32_16x16x32_bf16 v[192:199] /*v[448:455]*/, v[32:39] /*v[288:295]*/, v[160:167], 0
	s_set_vgpr_msb 0x4150
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[322:329]*/, v[208:215], v[144:151], v[66:73] /*v[322:329]*/
	v_wmma_f32_16x16x32_bf16 v[126:133] /*v[382:389]*/, v[208:215], v[176:183], v[126:133] /*v[382:389]*/
	v_wmma_f32_16x16x32_bf16 v[160:167] /*v[416:423]*/, v[232:239], v[168:175], v[160:167] /*v[416:423]*/
	v_wmma_f32_16x16x32_bf16 v[94:101] /*v[350:357]*/, v[240:247], v[144:151], v[94:101] /*v[350:357]*/
	s_set_vgpr_msb 0x5051
	v_wmma_f32_16x16x32_bf16 v[168:175] /*v[424:431]*/, v[8:15] /*v[264:271]*/, v[136:143], v[168:175] /*v[424:431]*/
	v_wmma_f32_16x16x32_bf16 v[176:183] /*v[432:439]*/, v[8:15] /*v[264:271]*/, v[168:175], v[176:183] /*v[432:439]*/
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x32_bf16 v[184:191] /*v[440:447]*/, v[40:47] /*v[296:303]*/, v[136:143], v[184:191] /*v[440:447]*/
	v_wmma_f32_16x16x32_bf16 v[192:199] /*v[448:455]*/, v[40:47] /*v[296:303]*/, v[168:175], v[192:199] /*v[448:455]*/
	s_set_vgpr_msb 0x5150
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[322:329]*/, v[216:223], v[152:159], v[66:73] /*v[322:329]*/
	v_wmma_f32_16x16x32_bf16 v[126:133] /*v[382:389]*/, v[216:223], v[184:191], v[126:133] /*v[382:389]*/
	v_wmma_f32_16x16x32_bf16 v[160:167] /*v[416:423]*/, v[240:247], v[176:183], v[160:167] /*v[416:423]*/
	v_wmma_f32_16x16x32_bf16 v[94:101] /*v[350:357]*/, v[248:255], v[152:159], v[94:101] /*v[350:357]*/
	s_set_vgpr_msb 0x5051
	v_wmma_f32_16x16x32_bf16 v[168:175] /*v[424:431]*/, v[16:23] /*v[272:279]*/, v[144:151], v[168:175] /*v[424:431]*/
	v_wmma_f32_16x16x32_bf16 v[176:183] /*v[432:439]*/, v[16:23] /*v[272:279]*/, v[176:183], v[176:183] /*v[432:439]*/
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[184:191] /*v[440:447]*/, v[48:55] /*v[304:311]*/, v[144:151], v[184:191] /*v[440:447]*/
	v_wmma_f32_16x16x32_bf16 v[192:199] /*v[448:455]*/, v[48:55] /*v[304:311]*/, v[176:183], v[192:199] /*v[448:455]*/
	s_set_vgpr_msb 0x5150
	v_wmma_f32_16x16x32_bf16 v[160:167] /*v[416:423]*/, v[248:255], v[184:191], v[160:167] /*v[416:423]*/
	s_set_vgpr_msb 0x5051
	v_wmma_f32_16x16x32_bf16 v[168:175] /*v[424:431]*/, v[24:31] /*v[280:287]*/, v[152:159], v[168:175] /*v[424:431]*/
	v_wmma_f32_16x16x32_bf16 v[176:183] /*v[432:439]*/, v[24:31] /*v[280:287]*/, v[184:191], v[176:183] /*v[432:439]*/
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[184:191] /*v[440:447]*/, v[56:63] /*v[312:319]*/, v[152:159], v[184:191] /*v[440:447]*/
	v_wmma_f32_16x16x32_bf16 v[192:199] /*v[448:455]*/, v[56:63] /*v[312:319]*/, v[184:191], v[192:199] /*v[448:455]*/
	s_wait_alu depctr_va_vdst(0)
	ds_load_tr16_b128 v[56:59] /*v[312:315]*/, v158 /*v414*/
	ds_load_tr16_b128 v[40:43] /*v[296:299]*/, v158 /*v414*/ offset:32
	ds_load_tr16_b128 v[60:63] /*v[316:319]*/, v158 /*v414*/ offset:4608
	ds_load_tr16_b128 v[44:47] /*v[300:303]*/, v158 /*v414*/ offset:4640
	ds_load_tr16_b128 v[48:51] /*v[304:307]*/, v158 /*v414*/ offset:9216
	ds_load_tr16_b128 v[32:35] /*v[288:291]*/, v158 /*v414*/ offset:9248
	ds_load_tr16_b128 v[52:55] /*v[308:311]*/, v158 /*v414*/ offset:13824
	ds_load_tr16_b128 v[36:39] /*v[292:295]*/, v158 /*v414*/ offset:13856
	ds_load_tr16_b128 v[24:27] /*v[280:283]*/, v158 /*v414*/ offset:64
	ds_load_tr16_b128 v[8:11] /*v[264:267]*/, v158 /*v414*/ offset:96
	ds_load_tr16_b128 v[28:31] /*v[284:287]*/, v158 /*v414*/ offset:4672
	ds_load_tr16_b128 v[12:15] /*v[268:271]*/, v158 /*v414*/ offset:4704
	ds_load_tr16_b128 v[16:19] /*v[272:275]*/, v158 /*v414*/ offset:9280
	ds_load_tr16_b128 v[0:3] /*v[256:259]*/, v158 /*v414*/ offset:9312
	ds_load_tr16_b128 v[20:23] /*v[276:279]*/, v158 /*v414*/ offset:13888
	ds_load_tr16_b128 v[4:7] /*v[260:263]*/, v158 /*v414*/ offset:13920
	s_set_vgpr_msb 0x5101
	ds_load_tr16_b128 v[248:251], v158 /*v414*/ offset:128
	ds_load_tr16_b128 v[232:235], v158 /*v414*/ offset:160
	ds_load_tr16_b128 v[252:255], v158 /*v414*/ offset:4736
	ds_load_tr16_b128 v[236:239], v158 /*v414*/ offset:4768
	ds_load_tr16_b128 v[240:243], v158 /*v414*/ offset:9344
	ds_load_tr16_b128 v[224:227], v158 /*v414*/ offset:9376
	ds_load_tr16_b128 v[244:247], v158 /*v414*/ offset:13952
	ds_load_tr16_b128 v[228:231], v158 /*v414*/ offset:13984
	ds_load_tr16_b128 v[216:219], v158 /*v414*/ offset:192
	ds_load_tr16_b128 v[200:203], v158 /*v414*/ offset:224
	ds_load_tr16_b128 v[220:223], v158 /*v414*/ offset:4800
	ds_load_tr16_b128 v[204:207], v158 /*v414*/ offset:4832
	ds_load_tr16_b128 v[208:211], v158 /*v414*/ offset:9408
	ds_load_tr16_b128 v[192:195], v158 /*v414*/ offset:9440
	ds_load_tr16_b128 v[212:215], v158 /*v414*/ offset:14016
	ds_load_tr16_b128 v[196:199], v158 /*v414*/ offset:14048
	s_set_vgpr_msb 0x155
	v_max_num_f32_e32 v74 /*v330*/, v66 /*v322*/, v67 /*v323*/
	v_max_num_f32_e32 v75 /*v331*/, v126 /*v382*/, v127 /*v383*/
	v_max3_num_f32 v76 /*v332*/, v69 /*v325*/, v70 /*v326*/, v71 /*v327*/
	v_max3_num_f32 v77 /*v333*/, v129 /*v385*/, v130 /*v386*/, v131 /*v387*/
	v_max3_num_f32 v80 /*v336*/, v95 /*v351*/, v96 /*v352*/, v97 /*v353*/
	v_max3_num_f32 v81 /*v337*/, v161 /*v417*/, v162 /*v418*/, v163 /*v419*/
	v_max3_num_f32 v82 /*v338*/, v98 /*v354*/, v99 /*v355*/, v100 /*v356*/
	v_max3_num_f32 v83 /*v339*/, v164 /*v420*/, v165 /*v421*/, v166 /*v422*/
	v_max3_num_f32 v84 /*v340*/, v101 /*v357*/, v168 /*v424*/, v169 /*v425*/
	v_max3_num_f32 v85 /*v341*/, v167 /*v423*/, v176 /*v432*/, v177 /*v433*/
	v_max3_num_f32 v78 /*v334*/, v72 /*v328*/, v73 /*v329*/, v94 /*v350*/
	v_max3_num_f32 v79 /*v335*/, v132 /*v388*/, v133 /*v389*/, v160 /*v416*/
	v_max3_num_f32 v87 /*v343*/, v178 /*v434*/, v179 /*v435*/, v180 /*v436*/
	v_max3_num_f32 v89 /*v345*/, v181 /*v437*/, v182 /*v438*/, v183 /*v439*/
	v_max3_num_f32 v91 /*v347*/, v192 /*v448*/, v193 /*v449*/, v194 /*v450*/
	v_max3_num_f32 v93 /*v349*/, v195 /*v451*/, v196 /*v452*/, v197 /*v453*/
	v_max3_num_f32 v74 /*v330*/, v74 /*v330*/, v68 /*v324*/, v76 /*v332*/
	v_max3_num_f32 v76 /*v332*/, v80 /*v336*/, v82 /*v338*/, v84 /*v340*/
	v_max3_num_f32 v75 /*v331*/, v75 /*v331*/, v128 /*v384*/, v77 /*v333*/
	v_max3_num_f32 v77 /*v333*/, v81 /*v337*/, v83 /*v339*/, v85 /*v341*/
	v_max3_num_f32 v86 /*v342*/, v170 /*v426*/, v171 /*v427*/, v172 /*v428*/
	v_max3_num_f32 v88 /*v344*/, v173 /*v429*/, v174 /*v430*/, v175 /*v431*/
	v_max3_num_f32 v90 /*v346*/, v184 /*v440*/, v185 /*v441*/, v186 /*v442*/
	v_max3_num_f32 v92 /*v348*/, v187 /*v443*/, v188 /*v444*/, v189 /*v445*/
	v_max3_num_f32 v74 /*v330*/, v74 /*v330*/, v78 /*v334*/, v76 /*v332*/
	v_max3_num_f32 v76 /*v332*/, v87 /*v343*/, v89 /*v345*/, v91 /*v347*/
	v_max3_num_f32 v78 /*v334*/, v93 /*v349*/, v198 /*v454*/, v199 /*v455*/
	v_max3_num_f32 v75 /*v331*/, v75 /*v331*/, v79 /*v335*/, v77 /*v333*/
	v_max3_num_f32 v80 /*v336*/, v86 /*v342*/, v88 /*v344*/, v90 /*v346*/
	v_max3_num_f32 v81 /*v337*/, v92 /*v348*/, v190 /*v446*/, v191 /*v447*/
	v_max3_num_f32 v75 /*v331*/, v75 /*v331*/, v76 /*v332*/, v78 /*v334*/
	v_max3_num_f32 v74 /*v330*/, v74 /*v330*/, v80 /*v336*/, v81 /*v337*/
	v_mov_b32_e32 v77 /*v333*/, v75 /*v331*/
	v_permlanex16_b32 v77 /*v333*/, v77 /*v333*/, s31, 0xfedcba98
	v_dual_mov_b32 v76 /*v332*/, v74 /*v330*/ :: v_dual_max_num_f32 v75 /*v331*/, v75 /*v331*/, v77 /*v333*/
	v_permlanex16_b32 v76 /*v332*/, v76 /*v332*/, s31, 0xfedcba98
	v_dual_sub_f32 v77 /*v333*/, v75 /*v331*/, v135 /*v391*/ :: v_dual_max_num_f32 v74 /*v330*/, v74 /*v330*/, v76 /*v332*/
	v_cmp_lt_f32_e64 s2, 0x41000000, v77 /*v333*/
	v_dual_sub_f32 v76 /*v332*/, v74 /*v330*/, v156 /*v412*/ :: v_dual_max_num_f32 v74 /*v330*/, v156 /*v412*/, v74 /*v330*/
	v_cmp_lt_f32_e32 vcc_lo, 0x41000000, v76 /*v332*/
	s_cmp_eq_u32 vcc_lo, 0
	s_cselect_b32 s3, -1, 0
	s_cmp_lg_u32 s2, 0
	v_dual_cndmask_b32 v151 /*v407*/, v74 /*v330*/, v156 /*v412*/, s3 :: v_dual_max_num_f32 v74 /*v330*/, v135 /*v391*/, v75 /*v331*/
	s_cselect_b32 s3, -1, 0
	s_cmp_eq_u32 s2, 0
	s_cselect_b32 s2, -1, 0
	v_mul_f32_e32 v118 /*v374*/, 0xbfb8aa3b, v151 /*v407*/
	v_cndmask_b32_e64 v150 /*v406*/, v74 /*v330*/, v135 /*v391*/, s2
	v_pk_fma_f32 v[76:77] /*v[332:333]*/, v[70:71] /*v[326:327]*/, s[90:91], v[118:119] /*v[374:375]*/ op_sel_hi:[1,0,0]
	v_mul_f32_e32 v134 /*v390*/, 0xbfb8aa3b, v150 /*v406*/
	v_pk_fma_f32 v[66:67] /*v[322:323]*/, v[66:67] /*v[322:323]*/, s[90:91], v[118:119] /*v[374:375]*/ op_sel_hi:[1,0,0]
	s_wait_alu depctr_vm_vsrc(0)
	v_pk_fma_f32 v[152:153] /*v[408:409]*/, v[190:191] /*v[446:447]*/, s[90:91], v[118:119] /*v[374:375]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[74:75] /*v[330:331]*/, v[68:69] /*v[324:325]*/, s[90:91], v[118:119] /*v[374:375]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v78 /*v334*/, v76 /*v332*/
	v_exp_f32_e32 v82 /*v338*/, v77 /*v333*/
	v_nop
	v_pk_fma_f32 v[76:77] /*v[332:333]*/, v[96:97] /*v[352:353]*/, s[90:91], v[118:119] /*v[374:375]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[130:131] /*v[386:387]*/, v[130:131] /*v[386:387]*/, s[90:91], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[80:81] /*v[336:337]*/, v[72:73] /*v[328:329]*/, s[90:91], v[118:119] /*v[374:375]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v68 /*v324*/, v66 /*v322*/
	v_exp_f32_e32 v70 /*v326*/, v67 /*v323*/
	v_nop
	v_pk_fma_f32 v[66:67] /*v[322:323]*/, v[94:95] /*v[350:351]*/, s[90:91], v[118:119] /*v[374:375]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v102 /*v358*/, v76 /*v332*/
	v_exp_f32_e32 v106 /*v362*/, v77 /*v333*/
	v_nop
	v_pk_fma_f32 v[76:77] /*v[332:333]*/, v[168:169] /*v[424:425]*/, s[90:91], v[118:119] /*v[374:375]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[154:155] /*v[410:411]*/, v[126:127] /*v[382:383]*/, s[90:91], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v126 /*v382*/, v152 /*v408*/
	v_pk_fma_f32 v[168:169] /*v[424:425]*/, v[128:129] /*v[384:385]*/, s[90:91], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v128 /*v384*/, v153 /*v409*/
	v_pk_fma_f32 v[132:133] /*v[388:389]*/, v[132:133] /*v[388:389]*/, s[90:91], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v79 /*v335*/, v130 /*v386*/
	v_pk_fma_f32 v[152:153] /*v[408:409]*/, v[160:161] /*v[416:417]*/, s[90:91], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v83 /*v339*/, v131 /*v387*/
	v_nop
	v_pk_fma_f32 v[130:131] /*v[386:387]*/, v[162:163] /*v[418:419]*/, s[90:91], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v84 /*v340*/, v80 /*v336*/
	v_exp_f32_e32 v90 /*v346*/, v81 /*v337*/
	v_exp_f32_e32 v94 /*v350*/, v66 /*v322*/
	v_pk_fma_f32 v[80:81] /*v[336:337]*/, v[98:99] /*v[354:355]*/, s[90:91], v[118:119] /*v[374:375]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v98 /*v354*/, v67 /*v323*/
	v_nop
	v_pk_fma_f32 v[66:67] /*v[322:323]*/, v[100:101] /*v[356:357]*/, s[90:91], v[118:119] /*v[374:375]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v85 /*v341*/, v132 /*v388*/
	v_exp_f32_e32 v91 /*v347*/, v133 /*v389*/
	v_exp_f32_e32 v95 /*v351*/, v152 /*v408*/
	v_pk_fma_f32 v[132:133] /*v[388:389]*/, v[164:165] /*v[420:421]*/, s[90:91], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v99 /*v355*/, v153 /*v409*/
	v_exp_f32_e32 v103 /*v359*/, v130 /*v386*/
	v_pk_fma_f32 v[152:153] /*v[408:409]*/, v[166:167] /*v[422:423]*/, s[90:91], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v107 /*v363*/, v131 /*v387*/
	v_nop
	v_pk_fma_f32 v[130:131] /*v[386:387]*/, v[176:177] /*v[432:433]*/, s[90:91], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v110 /*v366*/, v80 /*v336*/
	v_exp_f32_e32 v114 /*v370*/, v81 /*v337*/
	v_exp_f32_e32 v116 /*v372*/, v66 /*v322*/
	v_pk_fma_f32 v[80:81] /*v[336:337]*/, v[170:171] /*v[426:427]*/, s[90:91], v[118:119] /*v[374:375]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v122 /*v378*/, v67 /*v323*/
	v_exp_f32_e32 v66 /*v322*/, v76 /*v332*/
	v_pk_fma_f32 v[88:89] /*v[344:345]*/, v[172:173] /*v[428:429]*/, s[90:91], v[118:119] /*v[374:375]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v76 /*v332*/, v77 /*v333*/
	v_pk_fma_f32 v[96:97] /*v[352:353]*/, v[174:175] /*v[430:431]*/, s[90:91], v[118:119] /*v[374:375]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v111 /*v367*/, v132 /*v388*/
	v_exp_f32_e32 v115 /*v371*/, v133 /*v389*/
	v_exp_f32_e32 v117 /*v373*/, v152 /*v408*/
	v_pk_fma_f32 v[132:133] /*v[388:389]*/, v[178:179] /*v[434:435]*/, s[90:91], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v123 /*v379*/, v153 /*v409*/
	v_exp_f32_e32 v67 /*v323*/, v130 /*v386*/
	v_pk_fma_f32 v[152:153] /*v[408:409]*/, v[180:181] /*v[436:437]*/, s[90:91], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v77 /*v333*/, v131 /*v387*/
	v_nop
	v_pk_fma_f32 v[130:131] /*v[386:387]*/, v[182:183] /*v[438:439]*/, s[90:91], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v72 /*v328*/, v74 /*v330*/
	v_exp_f32_e32 v74 /*v330*/, v75 /*v331*/
	v_exp_f32_e32 v69 /*v325*/, v154 /*v410*/
	v_exp_f32_e32 v71 /*v327*/, v155 /*v411*/
	v_exp_f32_e32 v75 /*v331*/, v169 /*v425*/
	v_exp_f32_e32 v86 /*v342*/, v81 /*v337*/
	v_pk_fma_f32 v[104:105] /*v[360:361]*/, v[184:185] /*v[440:441]*/, s[90:91], v[118:119] /*v[374:375]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v92 /*v348*/, v89 /*v345*/
	v_pk_fma_f32 v[112:113] /*v[368:369]*/, v[186:187] /*v[442:443]*/, s[90:91], v[118:119] /*v[374:375]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v100 /*v356*/, v97 /*v353*/
	v_pk_fma_f32 v[120:121] /*v[376:377]*/, v[188:189] /*v[444:445]*/, s[90:91], v[118:119] /*v[374:375]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v81 /*v337*/, v132 /*v388*/
	v_exp_f32_e32 v87 /*v343*/, v133 /*v389*/
	v_exp_f32_e32 v89 /*v345*/, v152 /*v408*/
	v_pk_fma_f32 v[132:133] /*v[388:389]*/, v[192:193] /*v[448:449]*/, s[90:91], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v93 /*v349*/, v153 /*v409*/
	v_exp_f32_e32 v97 /*v353*/, v130 /*v386*/
	v_pk_fma_f32 v[152:153] /*v[408:409]*/, v[194:195] /*v[450:451]*/, s[90:91], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v101 /*v357*/, v131 /*v387*/
	v_nop
	v_pk_fma_f32 v[130:131] /*v[386:387]*/, v[196:197] /*v[452:453]*/, s[90:91], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v80 /*v336*/, v80 /*v336*/
	v_exp_f32_e32 v96 /*v352*/, v96 /*v352*/
	v_exp_f32_e32 v73 /*v329*/, v168 /*v424*/
	v_exp_f32_e32 v108 /*v364*/, v105 /*v361*/
	v_exp_f32_e32 v118 /*v374*/, v113 /*v369*/
	v_exp_f32_e32 v124 /*v380*/, v121 /*v377*/
	v_exp_f32_e32 v105 /*v361*/, v132 /*v388*/
	v_exp_f32_e32 v109 /*v365*/, v133 /*v389*/
	v_exp_f32_e32 v113 /*v369*/, v152 /*v408*/
	v_pk_fma_f32 v[132:133] /*v[388:389]*/, v[198:199] /*v[454:455]*/, s[90:91], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v119 /*v375*/, v153 /*v409*/
	v_exp_f32_e32 v121 /*v377*/, v130 /*v386*/
	v_exp_f32_e32 v125 /*v381*/, v131 /*v387*/
	v_nop
	v_pk_add_f32 v[130:131] /*v[386:387]*/, v[68:69] /*v[324:325]*/, v[70:71] /*v[326:327]*/
	v_pk_add_f32 v[152:153] /*v[408:409]*/, v[74:75] /*v[330:331]*/, v[78:79] /*v[334:335]*/
	v_pk_add_f32 v[154:155] /*v[410:411]*/, v[98:99] /*v[354:355]*/, v[102:103] /*v[358:359]*/
	v_pk_add_f32 v[160:161] /*v[416:417]*/, v[110:111] /*v[366:367]*/, v[114:115] /*v[370:371]*/
	v_exp_f32_e32 v88 /*v344*/, v88 /*v344*/
	v_exp_f32_e32 v104 /*v360*/, v104 /*v360*/
	v_exp_f32_e32 v127 /*v383*/, v132 /*v388*/
	v_exp_f32_e32 v129 /*v385*/, v133 /*v389*/
	v_nop
	v_pk_add_f32 v[132:133] /*v[388:389]*/, v[84:85] /*v[340:341]*/, v[90:91] /*v[346:347]*/
	v_pk_add_f32 v[130:131] /*v[386:387]*/, v[72:73] /*v[328:329]*/, v[130:131] /*v[386:387]*/
	v_pk_add_f32 v[152:153] /*v[408:409]*/, v[82:83] /*v[338:339]*/, v[152:153] /*v[408:409]*/
	v_pk_add_f32 v[162:163] /*v[418:419]*/, v[122:123] /*v[378:379]*/, v[66:67] /*v[322:323]*/
	v_pk_add_f32 v[154:155] /*v[410:411]*/, v[106:107] /*v[362:363]*/, v[154:155] /*v[410:411]*/
	v_pk_add_f32 v[164:165] /*v[420:421]*/, v[80:81] /*v[336:337]*/, v[86:87] /*v[342:343]*/
	v_pk_add_f32 v[160:161] /*v[416:417]*/, v[116:117] /*v[372:373]*/, v[160:161] /*v[416:417]*/
	v_pk_add_f32 v[166:167] /*v[422:423]*/, v[92:93] /*v[348:349]*/, v[96:97] /*v[352:353]*/
	v_exp_f32_e32 v112 /*v368*/, v112 /*v368*/
	v_exp_f32_e32 v120 /*v376*/, v120 /*v376*/
	v_pk_add_f32 v[132:133] /*v[388:389]*/, v[94:95] /*v[350:351]*/, v[132:133] /*v[388:389]*/
	v_pk_add_f32 v[162:163] /*v[418:419]*/, v[76:77] /*v[332:333]*/, v[162:163] /*v[418:419]*/
	v_pk_add_f32 v[168:169] /*v[424:425]*/, v[104:105] /*v[360:361]*/, v[108:109] /*v[364:365]*/
	v_pk_add_f32 v[164:165] /*v[420:421]*/, v[88:89] /*v[344:345]*/, v[164:165] /*v[420:421]*/
	v_pk_add_f32 v[130:131] /*v[386:387]*/, v[130:131] /*v[386:387]*/, v[152:153] /*v[408:409]*/
	v_pk_add_f32 v[152:153] /*v[408:409]*/, v[100:101] /*v[356:357]*/, v[166:167] /*v[422:423]*/
	v_pk_add_f32 v[154:155] /*v[410:411]*/, v[154:155] /*v[410:411]*/, v[160:161] /*v[416:417]*/
	v_pk_add_f32 v[160:161] /*v[416:417]*/, v[112:113] /*v[368:369]*/, v[168:169] /*v[424:425]*/
	v_pk_add_f32 v[166:167] /*v[422:423]*/, v[118:119] /*v[374:375]*/, v[120:121] /*v[376:377]*/
	v_pk_add_f32 v[130:131] /*v[386:387]*/, v[132:133] /*v[388:389]*/, v[130:131] /*v[386:387]*/
	v_pk_add_f32 v[132:133] /*v[388:389]*/, v[164:165] /*v[420:421]*/, v[152:153] /*v[408:409]*/
	v_pk_add_f32 v[152:153] /*v[408:409]*/, v[162:163] /*v[418:419]*/, v[154:155] /*v[410:411]*/
	v_pk_add_f32 v[162:163] /*v[418:419]*/, v[126:127] /*v[382:383]*/, v[128:129] /*v[384:385]*/
	v_pk_add_f32 v[154:155] /*v[410:411]*/, v[124:125] /*v[380:381]*/, v[166:167] /*v[422:423]*/
	v_pk_add_f32 v[132:133] /*v[388:389]*/, v[160:161] /*v[416:417]*/, v[132:133] /*v[388:389]*/
	v_pk_add_f32 v[130:131] /*v[386:387]*/, v[130:131] /*v[386:387]*/, v[152:153] /*v[408:409]*/
	v_pk_add_f32 v[152:153] /*v[408:409]*/, v[162:163] /*v[418:419]*/, v[154:155] /*v[410:411]*/
	v_pk_add_f32 v[130:131] /*v[386:387]*/, v[132:133] /*v[388:389]*/, v[130:131] /*v[386:387]*/
	v_sub_f32_e32 v132 /*v388*/, v156 /*v412*/, v151 /*v407*/
	v_pk_add_f32 v[130:131] /*v[386:387]*/, v[152:153] /*v[408:409]*/, v[130:131] /*v[386:387]*/
	v_mul_f32_e32 v132 /*v388*/, 0x3fb8aa3b, v132 /*v388*/
	v_dual_mov_b32 v152 /*v408*/, v130 /*v386*/ :: v_dual_mov_b32 v133 /*v389*/, v131 /*v387*/
	v_exp_f32_e32 v132 /*v388*/, v132 /*v388*/
	v_permlanex16_b32 v152 /*v408*/, v152 /*v408*/, s31, 0xfedcba98
	v_permlanex16_b32 v133 /*v389*/, v133 /*v389*/, s31, 0xfedcba98
	s_set_vgpr_msb 0x5500
	s_cbranch_vccz .LBB0_20
	s_set_vgpr_msb 4
	v_pk_mul_f32 v[126:127], v[126:127], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[124:125], v[124:125], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[122:123], v[122:123], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[120:121], v[120:121], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[118:119], v[118:119], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[116:117], v[116:117], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[114:115], v[114:115], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[112:113], v[112:113], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[110:111], v[110:111], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[108:109], v[108:109], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[106:107], v[106:107], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[104:105], v[104:105], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[102:103], v[102:103], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[100:101], v[100:101], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[98:99], v[98:99], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[96:97], v[96:97], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[94:95], v[94:95], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[92:93], v[92:93], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[90:91], v[90:91], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[88:89], v[88:89], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[86:87], v[86:87], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[84:85], v[84:85], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[82:83], v[82:83], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[80:81], v[80:81], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[78:79], v[78:79], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[76:77], v[76:77], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[74:75], v[74:75], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[72:73], v[72:73], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[70:71], v[70:71], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[68:69], v[68:69], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[66:67], v[66:67], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[64:65], v[64:65], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x400
.LBB0_20:
	s_set_vgpr_msb 0x45
	v_sub_f32_e32 v134 /*v390*/, v135 /*v391*/, v150 /*v406*/
	s_and_b32 s2, s3, exec_lo
	s_cselect_b32 s2, 1, 0
	s_cmp_lg_u32 s2, 1
	v_mul_f32_e32 v134 /*v390*/, 0x3fb8aa3b, v134 /*v390*/
	v_exp_f32_e32 v134 /*v390*/, v134 /*v390*/
	s_set_vgpr_msb 0x4500
	s_cbranch_scc1 .LBB0_15
	v_nop
	s_set_vgpr_msb 4
	v_pk_mul_f32 v[62:63], v[62:63], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[60:61], v[60:61], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59], v[58:59], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[56:57], v[56:57], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[54:55], v[54:55], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[52:53], v[52:53], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[50:51], v[50:51], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[48:49], v[48:49], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[46:47], v[46:47], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[44:45], v[44:45], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[42:43], v[42:43], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[40:41], v[40:41], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39], v[38:39], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[36:37], v[36:37], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35], v[34:35], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[32:33], v[32:33], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[30:31], v[30:31], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[28:29], v[28:29], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[26:27], v[26:27], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25], v[24:25], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[22:23], v[22:23], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21], v[20:21], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[18:19], v[18:19], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[16:17], v[16:17], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[14:15], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13], v[12:13], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[10:11], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[8:9], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[6:7], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[4:5], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[2:3], v[2:3], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[0:1], v[0:1], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x400
	s_branch .LBB0_15
.LBB0_22:
	s_set_vgpr_msb 0x41
	v_dual_mov_b32 v158 /*v414*/, v153 /*v409*/ :: v_dual_mov_b32 v157 /*v413*/, v152 /*v408*/
	v_dual_mov_b32 v150 /*v406*/, v135 /*v391*/ :: v_dual_mov_b32 v151 /*v407*/, v156 /*v412*/
	s_set_vgpr_msb 0x4100
.LBB0_23:
	s_cmp_ge_u32 s82, s10
	s_cbranch_scc1 .LBB0_32
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 4
	v_dual_add_nc_u32 v192, s99, v138 /*v394*/ :: v_dual_add_nc_u32 v193, s99, v137 /*v393*/
	s_add_co_i32 s2, s19, -1
	s_mov_b32 s59, 0
	s_mov_b32 s28, 1
	s_set_vgpr_msb 0x400
	v_dual_add_nc_u32 v194, s17, v192 :: v_dual_add_nc_u32 v195, s17, v193
	s_wait_alu depctr_vm_vsrc(0)
	s_set_vgpr_msb 64
	v_subrev_nc_u32_e32 v152 /*v408*/, s16, v192
	v_subrev_nc_u32_e32 v153 /*v409*/, s16, v193
	s_mov_b32 s11, s59
	v_min_i32_e32 v154 /*v410*/, s2, v194
	v_min_i32_e32 v155 /*v411*/, s2, v195
	s_sub_co_i32 s7, 1, s60
	s_mov_b32 s56, 16
	s_mov_b32 s55, 0x800000
	s_mov_b32 s53, 0xffff0000
	s_mov_b32 s52, 0x7510000
	s_mov_b32 s44, 0xf510000
	s_mov_b32 s61, 0x76543210
	s_mov_b32 s88, 0x3fb8aa3b
	s_set_vgpr_msb 0x4000
	s_branch .LBB0_26
.LBB0_25:
	s_set_vgpr_msb 0x45
	v_cvt_pk_bf16_f32 v167 /*v423*/, v116 /*v372*/, v122 /*v378*/
	v_cvt_pk_bf16_f32 v166 /*v422*/, v108 /*v364*/, v114 /*v370*/
	v_cvt_pk_bf16_f32 v165 /*v421*/, v100 /*v356*/, v106 /*v362*/
	v_cvt_pk_bf16_f32 v164 /*v420*/, v92 /*v348*/, v98 /*v354*/
	v_cvt_pk_bf16_f32 v163 /*v419*/, v84 /*v340*/, v90 /*v346*/
	v_cvt_pk_bf16_f32 v162 /*v418*/, v76 /*v332*/, v82 /*v338*/
	v_cvt_pk_bf16_f32 v161 /*v417*/, v70 /*v326*/, v74 /*v330*/
	v_cvt_pk_bf16_f32 v160 /*v416*/, v66 /*v322*/, v68 /*v324*/
	v_cvt_pk_bf16_f32 v175 /*v431*/, v117 /*v373*/, v123 /*v379*/
	v_cvt_pk_bf16_f32 v174 /*v430*/, v109 /*v365*/, v115 /*v371*/
	v_cvt_pk_bf16_f32 v173 /*v429*/, v101 /*v357*/, v107 /*v363*/
	v_cvt_pk_bf16_f32 v172 /*v428*/, v93 /*v349*/, v99 /*v355*/
	v_cvt_pk_bf16_f32 v171 /*v427*/, v85 /*v341*/, v91 /*v347*/
	v_cvt_pk_bf16_f32 v170 /*v426*/, v77 /*v333*/, v83 /*v339*/
	v_cvt_pk_bf16_f32 v169 /*v425*/, v71 /*v327*/, v75 /*v331*/
	v_cvt_pk_bf16_f32 v168 /*v424*/, v67 /*v323*/, v69 /*v325*/
	s_set_vgpr_msb 0x4505
	s_wait_dscnt 0x1d
	v_wmma_f32_16x16x32_bf16 v[120:127], v[56:63] /*v[312:319]*/, v[160:167] /*v[416:423]*/, v[120:127]
	s_set_vgpr_msb 0x545
	v_cvt_pk_bf16_f32 v101 /*v357*/, v127 /*v383*/, v129 /*v385*/
	v_cvt_pk_bf16_f32 v100 /*v356*/, v121 /*v377*/, v125 /*v381*/
	v_cvt_pk_bf16_f32 v99 /*v355*/, v113 /*v369*/, v119 /*v375*/
	v_cvt_pk_bf16_f32 v98 /*v354*/, v105 /*v361*/, v111 /*v367*/
	v_cvt_pk_bf16_f32 v97 /*v353*/, v97 /*v353*/, v103 /*v359*/
	v_dual_fmac_f32 v151 /*v407*/, v132 /*v388*/, v64 /*v320*/ :: v_dual_fmac_f32 v158 /*v414*/, v134 /*v390*/, v65 /*v321*/
	s_set_vgpr_msb 0x4505
	v_wmma_f32_16x16x32_bf16 v[56:63], v[56:63] /*v[312:319]*/, v[168:175] /*v[424:431]*/, v[56:63]
	s_add_nc_u64 s[82:83], s[82:83], 1
	s_set_vgpr_msb 0x545
	v_mov_b32_e32 v150 /*v406*/, v135 /*v391*/
	v_cmp_ge_u64_e64 s2, s[82:83], s[10:11]
	v_dual_add_f32 v64 /*v320*/, v151 /*v407*/, v130 /*v386*/ :: v_dual_add_f32 v65 /*v321*/, v158 /*v414*/, v131 /*v387*/
	v_nop
	v_cvt_pk_bf16_f32 v63 /*v319*/, v126 /*v382*/, v128 /*v384*/
	v_cvt_pk_bf16_f32 v62 /*v318*/, v120 /*v376*/, v124 /*v380*/
	s_set_vgpr_msb 0x4505
	s_wait_dscnt 0x1c
	v_wmma_f32_16x16x32_bf16 v[112:119], v[40:47] /*v[296:303]*/, v[160:167] /*v[416:423]*/, v[112:119]
	s_set_vgpr_msb 0x545
	v_cvt_pk_bf16_f32 v61 /*v317*/, v112 /*v368*/, v118 /*v374*/
	v_cvt_pk_bf16_f32 v60 /*v316*/, v104 /*v360*/, v110 /*v366*/
	v_cvt_pk_bf16_f32 v59 /*v315*/, v96 /*v352*/, v102 /*v358*/
	v_cvt_pk_bf16_f32 v58 /*v314*/, v88 /*v344*/, v94 /*v350*/
	v_cvt_pk_bf16_f32 v57 /*v313*/, v80 /*v336*/, v86 /*v342*/
	v_cvt_pk_bf16_f32 v56 /*v312*/, v72 /*v328*/, v78 /*v334*/
	v_cvt_pk_bf16_f32 v96 /*v352*/, v89 /*v345*/, v95 /*v351*/
	s_set_vgpr_msb 0x4505
	v_wmma_f32_16x16x32_bf16 v[48:55], v[40:47] /*v[296:303]*/, v[168:175] /*v[424:431]*/, v[48:55]
	s_set_vgpr_msb 0x545
	v_cvt_pk_bf16_f32 v95 /*v351*/, v81 /*v337*/, v87 /*v343*/
	v_cvt_pk_bf16_f32 v94 /*v350*/, v73 /*v329*/, v79 /*v335*/
	s_wait_alu depctr_vm_vsrc(0)
	v_dual_mov_b32 v158 /*v414*/, v148 /*v404*/ :: v_dual_mov_b32 v148 /*v404*/, v157 /*v413*/
	v_dual_mov_b32 v157 /*v413*/, v149 /*v405*/ :: v_dual_mov_b32 v149 /*v405*/, v156 /*v412*/
	v_mov_b32_e32 v151 /*v407*/, v133 /*v389*/
	s_set_vgpr_msb 0x4505
	s_wait_dscnt 0x15
	v_wmma_f32_16x16x32_bf16 v[104:111], v[24:31] /*v[280:287]*/, v[160:167] /*v[416:423]*/, v[104:111]
	s_and_b32 vcc_lo, exec_lo, s2
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[40:47], v[24:31] /*v[280:287]*/, v[168:175] /*v[424:431]*/, v[40:47]
	v_wmma_f32_16x16x32_bf16 v[96:103], v[8:15] /*v[264:271]*/, v[160:167] /*v[416:423]*/, v[96:103]
	v_wmma_f32_16x16x32_bf16 v[32:39], v[8:15] /*v[264:271]*/, v[168:175] /*v[424:431]*/, v[32:39]
	s_set_vgpr_msb 0x504
	v_wmma_f32_16x16x32_bf16 v[88:95], v[248:255], v[160:167] /*v[416:423]*/, v[88:95]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[248:255], v[168:175] /*v[424:431]*/, v[24:31]
	v_wmma_f32_16x16x32_bf16 v[80:87], v[232:239], v[160:167] /*v[416:423]*/, v[80:87]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[232:239], v[168:175] /*v[424:431]*/, v[16:23]
	v_wmma_f32_16x16x32_bf16 v[72:79], v[216:223], v[160:167] /*v[416:423]*/, v[72:79]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[216:223], v[168:175] /*v[424:431]*/, v[8:15]
	v_wmma_f32_16x16x32_bf16 v[64:71], v[200:207], v[160:167] /*v[416:423]*/, v[64:71]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[200:207], v[168:175] /*v[424:431]*/, v[0:7]
	s_set_vgpr_msb 0x405
	v_wmma_f32_16x16x32_bf16 v[120:127], v[48:55] /*v[304:311]*/, v[56:63] /*v[312:319]*/, v[120:127]
	v_wmma_f32_16x16x32_bf16 v[56:63], v[48:55] /*v[304:311]*/, v[94:101] /*v[350:357]*/, v[56:63]
	v_wmma_f32_16x16x32_bf16 v[112:119], v[32:39] /*v[288:295]*/, v[56:63] /*v[312:319]*/, v[112:119]
	v_wmma_f32_16x16x32_bf16 v[48:55], v[32:39] /*v[288:295]*/, v[94:101] /*v[350:357]*/, v[48:55]
	v_wmma_f32_16x16x32_bf16 v[104:111], v[16:23] /*v[272:279]*/, v[56:63] /*v[312:319]*/, v[104:111]
	v_wmma_f32_16x16x32_bf16 v[40:47], v[16:23] /*v[272:279]*/, v[94:101] /*v[350:357]*/, v[40:47]
	v_wmma_f32_16x16x32_bf16 v[96:103], v[0:7] /*v[256:263]*/, v[56:63] /*v[312:319]*/, v[96:103]
	v_wmma_f32_16x16x32_bf16 v[32:39], v[0:7] /*v[256:263]*/, v[94:101] /*v[350:357]*/, v[32:39]
	s_set_vgpr_msb 0x504
	v_wmma_f32_16x16x32_bf16 v[88:95], v[240:247], v[56:63] /*v[312:319]*/, v[88:95]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[240:247], v[94:101] /*v[350:357]*/, v[24:31]
	v_wmma_f32_16x16x32_bf16 v[80:87], v[224:231], v[56:63] /*v[312:319]*/, v[80:87]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[224:231], v[94:101] /*v[350:357]*/, v[16:23]
	v_wmma_f32_16x16x32_bf16 v[72:79], v[208:215], v[56:63] /*v[312:319]*/, v[72:79]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[208:215], v[94:101] /*v[350:357]*/, v[8:15]
	v_wmma_f32_16x16x32_bf16 v[64:71], v[192:199], v[56:63] /*v[312:319]*/, v[64:71]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[192:199], v[94:101] /*v[350:357]*/, v[0:7]
	s_set_vgpr_msb 0x400
	s_cbranch_vccnz .LBB0_33
.LBB0_26:
	s_set_vgpr_msb 0x41
	v_dual_mov_b32 v156 /*v412*/, v157 /*v413*/ :: v_dual_mov_b32 v157 /*v413*/, v158 /*v414*/
	s_add_co_i32 s2, s82, 1
	s_wait_tensorcnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	s_cmp_ge_i32 s2, s10
	s_set_vgpr_msb 0x4100
	s_cbranch_scc1 .LBB0_28
	s_lshl_b32 s4, s2, 6
	s_add_co_i32 s6, s82, s7
	s_sub_co_i32 s30, s9, s4
	s_add_co_i32 s2, s4, s97
	v_nop
	v_nop
	v_nop
	v_med3_i32 v192, s30, 0, 64
	s_lshr_b32 s29, s6, 31
	s_ashr_i32 s3, s2, 31
	s_add_co_i32 s29, s6, s29
	s_mul_u64 s[4:5], s[2:3], s[64:65]
	v_readfirstlane_b32 s30, v192
	s_mul_u64 s[2:3], s[2:3], s[62:63]
	s_and_b32 s29, s29, 0x7ffe
	s_lshl_b64 s[2:3], s[2:3], 1
	s_sub_co_i32 s6, s6, s29
	s_add_nc_u64 s[2:3], s[86:87], s[2:3]
	s_sub_co_i32 s29, s30, s25
	s_add_nc_u64 s[30:31], s[26:27], s[2:3]
	s_max_i32 s2, s29, 0
	s_lshl_b64 s[4:5], s[4:5], 1
	s_lshl_b32 s6, s6, 17
	s_lshl_b32 s2, s2, 16
	s_add_nc_u64 s[4:5], s[84:85], s[4:5]
	s_or_b32 s29, s104, s6
	s_bitset1_b32 s31, 31
	s_or_b32 s54, s2, 0x7fff
	s_mov_b32 s45, s53
	tensor_load_to_lds s[28:31], s[52:59]
	s_add_nc_u64 s[30:31], s[80:81], s[4:5]
	s_or_b32 s29, vcc_hi, s6
	s_bitset1_b32 s31, 31
	s_mov_b32 s46, s54
	s_mov_b32 s47, s55
	s_mov_b32 s48, s56
	s_mov_b32 s51, s59
	tensor_load_to_lds s[28:31], s[44:51]
.LBB0_28:
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 1
	ds_load_b128 v[192:195], v149 /*v405*/
	ds_load_b128 v[196:199], v149 /*v405*/ offset:32
	ds_load_b128 v[200:203], v149 /*v405*/ offset:64
	ds_load_b128 v[204:207], v149 /*v405*/ offset:96
	ds_load_b128 v[208:211], v149 /*v405*/ offset:128
	ds_load_b128 v[212:215], v149 /*v405*/ offset:160
	ds_load_b128 v[216:219], v149 /*v405*/ offset:192
	ds_load_b128 v[220:223], v149 /*v405*/ offset:224
	ds_load_b128 v[224:227], v149 /*v405*/ offset:4352
	ds_load_b128 v[228:231], v149 /*v405*/ offset:4384
	ds_load_b128 v[232:235], v149 /*v405*/ offset:4416
	ds_load_b128 v[236:239], v149 /*v405*/ offset:4448
	ds_load_b128 v[240:243], v149 /*v405*/ offset:4480
	ds_load_b128 v[244:247], v149 /*v405*/ offset:4512
	ds_load_b128 v[248:251], v149 /*v405*/ offset:4544
	ds_load_b128 v[252:255], v149 /*v405*/ offset:4576
	s_set_vgpr_msb 0x141
	ds_load_b128 v[0:3] /*v[256:259]*/, v149 /*v405*/ offset:8704
	ds_load_b128 v[4:7] /*v[260:263]*/, v149 /*v405*/ offset:8736
	ds_load_b128 v[8:11] /*v[264:267]*/, v149 /*v405*/ offset:8768
	ds_load_b128 v[12:15] /*v[268:271]*/, v149 /*v405*/ offset:8800
	ds_load_b128 v[16:19] /*v[272:275]*/, v149 /*v405*/ offset:8832
	ds_load_b128 v[20:23] /*v[276:279]*/, v149 /*v405*/ offset:8864
	ds_load_b128 v[24:27] /*v[280:283]*/, v149 /*v405*/ offset:8896
	ds_load_b128 v[28:31] /*v[284:287]*/, v149 /*v405*/ offset:8928
	ds_load_b128 v[32:35] /*v[288:291]*/, v149 /*v405*/ offset:13056
	ds_load_b128 v[36:39] /*v[292:295]*/, v149 /*v405*/ offset:13088
	ds_load_b128 v[40:43] /*v[296:299]*/, v149 /*v405*/ offset:13120
	ds_load_b128 v[44:47] /*v[300:303]*/, v149 /*v405*/ offset:13152
	ds_load_b128 v[48:51] /*v[304:307]*/, v149 /*v405*/ offset:13184
	ds_load_b128 v[52:55] /*v[308:311]*/, v149 /*v405*/ offset:13216
	ds_load_b128 v[56:59] /*v[312:315]*/, v149 /*v405*/ offset:13248
	ds_load_b128 v[60:63] /*v[316:319]*/, v149 /*v405*/ offset:13280
	s_set_vgpr_msb 0x4140
	s_wait_dscnt 0x1e
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[322:329]*/, v[192:199], v[128:135], 0
	v_wmma_f32_16x16x32_bf16 v[74:81] /*v[330:337]*/, v[192:199], v[160:167], 0
	s_wait_dscnt 0x16
	v_wmma_f32_16x16x32_bf16 v[82:89] /*v[338:345]*/, v[224:231], v[128:135], 0
	v_wmma_f32_16x16x32_bf16 v[90:97] /*v[346:353]*/, v[224:231], v[160:167], 0
	s_set_vgpr_msb 0x4041
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x32_bf16 v[98:105] /*v[354:361]*/, v[0:7] /*v[256:263]*/, v[128:135], 0
	v_wmma_f32_16x16x32_bf16 v[106:113] /*v[362:369]*/, v[0:7] /*v[256:263]*/, v[160:167], 0
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x32_bf16 v[114:121] /*v[370:377]*/, v[32:39] /*v[288:295]*/, v[128:135], 0
	v_wmma_f32_16x16x32_bf16 v[122:129] /*v[378:385]*/, v[32:39] /*v[288:295]*/, v[160:167], 0
	s_set_vgpr_msb 0x4150
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[322:329]*/, v[200:207], v[136:143], v[66:73] /*v[322:329]*/
	v_wmma_f32_16x16x32_bf16 v[74:81] /*v[330:337]*/, v[200:207], v[168:175], v[74:81] /*v[330:337]*/
	v_wmma_f32_16x16x32_bf16 v[82:89] /*v[338:345]*/, v[232:239], v[136:143], v[82:89] /*v[338:345]*/
	v_wmma_f32_16x16x32_bf16 v[90:97] /*v[346:353]*/, v[232:239], v[168:175], v[90:97] /*v[346:353]*/
	s_set_vgpr_msb 0x5051
	v_wmma_f32_16x16x32_bf16 v[98:105] /*v[354:361]*/, v[8:15] /*v[264:271]*/, v[136:143], v[98:105] /*v[354:361]*/
	v_wmma_f32_16x16x32_bf16 v[106:113] /*v[362:369]*/, v[8:15] /*v[264:271]*/, v[168:175], v[106:113] /*v[362:369]*/
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x32_bf16 v[114:121] /*v[370:377]*/, v[40:47] /*v[296:303]*/, v[136:143], v[114:121] /*v[370:377]*/
	v_wmma_f32_16x16x32_bf16 v[122:129] /*v[378:385]*/, v[40:47] /*v[296:303]*/, v[168:175], v[122:129] /*v[378:385]*/
	s_set_vgpr_msb 0x5150
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[322:329]*/, v[208:215], v[144:151], v[66:73] /*v[322:329]*/
	v_wmma_f32_16x16x32_bf16 v[74:81] /*v[330:337]*/, v[208:215], v[176:183], v[74:81] /*v[330:337]*/
	v_wmma_f32_16x16x32_bf16 v[82:89] /*v[338:345]*/, v[240:247], v[144:151], v[82:89] /*v[338:345]*/
	v_wmma_f32_16x16x32_bf16 v[90:97] /*v[346:353]*/, v[240:247], v[176:183], v[90:97] /*v[346:353]*/
	s_set_vgpr_msb 0x5051
	v_wmma_f32_16x16x32_bf16 v[98:105] /*v[354:361]*/, v[16:23] /*v[272:279]*/, v[144:151], v[98:105] /*v[354:361]*/
	v_wmma_f32_16x16x32_bf16 v[106:113] /*v[362:369]*/, v[16:23] /*v[272:279]*/, v[176:183], v[106:113] /*v[362:369]*/
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[114:121] /*v[370:377]*/, v[48:55] /*v[304:311]*/, v[144:151], v[114:121] /*v[370:377]*/
	v_wmma_f32_16x16x32_bf16 v[122:129] /*v[378:385]*/, v[48:55] /*v[304:311]*/, v[176:183], v[122:129] /*v[378:385]*/
	s_set_vgpr_msb 0x5150
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[322:329]*/, v[216:223], v[152:159], v[66:73] /*v[322:329]*/
	v_wmma_f32_16x16x32_bf16 v[74:81] /*v[330:337]*/, v[216:223], v[184:191], v[74:81] /*v[330:337]*/
	v_wmma_f32_16x16x32_bf16 v[82:89] /*v[338:345]*/, v[248:255], v[152:159], v[82:89] /*v[338:345]*/
	v_wmma_f32_16x16x32_bf16 v[90:97] /*v[346:353]*/, v[248:255], v[184:191], v[90:97] /*v[346:353]*/
	s_set_vgpr_msb 0x5051
	v_wmma_f32_16x16x32_bf16 v[98:105] /*v[354:361]*/, v[24:31] /*v[280:287]*/, v[152:159], v[98:105] /*v[354:361]*/
	v_wmma_f32_16x16x32_bf16 v[106:113] /*v[362:369]*/, v[24:31] /*v[280:287]*/, v[184:191], v[106:113] /*v[362:369]*/
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[114:121] /*v[370:377]*/, v[56:63] /*v[312:319]*/, v[152:159], v[114:121] /*v[370:377]*/
	v_wmma_f32_16x16x32_bf16 v[122:129] /*v[378:385]*/, v[56:63] /*v[312:319]*/, v[184:191], v[122:129] /*v[378:385]*/
	s_wait_alu depctr_va_vdst(0)
	ds_load_tr16_b128 v[56:59] /*v[312:315]*/, v148 /*v404*/
	ds_load_tr16_b128 v[40:43] /*v[296:299]*/, v148 /*v404*/ offset:32
	ds_load_tr16_b128 v[60:63] /*v[316:319]*/, v148 /*v404*/ offset:4608
	ds_load_tr16_b128 v[44:47] /*v[300:303]*/, v148 /*v404*/ offset:4640
	ds_load_tr16_b128 v[48:51] /*v[304:307]*/, v148 /*v404*/ offset:9216
	ds_load_tr16_b128 v[32:35] /*v[288:291]*/, v148 /*v404*/ offset:9248
	ds_load_tr16_b128 v[52:55] /*v[308:311]*/, v148 /*v404*/ offset:13824
	ds_load_tr16_b128 v[36:39] /*v[292:295]*/, v148 /*v404*/ offset:13856
	ds_load_tr16_b128 v[24:27] /*v[280:283]*/, v148 /*v404*/ offset:64
	ds_load_tr16_b128 v[8:11] /*v[264:267]*/, v148 /*v404*/ offset:96
	ds_load_tr16_b128 v[28:31] /*v[284:287]*/, v148 /*v404*/ offset:4672
	ds_load_tr16_b128 v[12:15] /*v[268:271]*/, v148 /*v404*/ offset:4704
	ds_load_tr16_b128 v[16:19] /*v[272:275]*/, v148 /*v404*/ offset:9280
	ds_load_tr16_b128 v[0:3] /*v[256:259]*/, v148 /*v404*/ offset:9312
	ds_load_tr16_b128 v[20:23] /*v[276:279]*/, v148 /*v404*/ offset:13888
	ds_load_tr16_b128 v[4:7] /*v[260:263]*/, v148 /*v404*/ offset:13920
	s_set_vgpr_msb 0x5101
	ds_load_tr16_b128 v[248:251], v148 /*v404*/ offset:128
	ds_load_tr16_b128 v[232:235], v148 /*v404*/ offset:160
	ds_load_tr16_b128 v[252:255], v148 /*v404*/ offset:4736
	ds_load_tr16_b128 v[236:239], v148 /*v404*/ offset:4768
	ds_load_tr16_b128 v[240:243], v148 /*v404*/ offset:9344
	ds_load_tr16_b128 v[224:227], v148 /*v404*/ offset:9376
	ds_load_tr16_b128 v[244:247], v148 /*v404*/ offset:13952
	ds_load_tr16_b128 v[228:231], v148 /*v404*/ offset:13984
	ds_load_tr16_b128 v[216:219], v148 /*v404*/ offset:192
	ds_load_tr16_b128 v[200:203], v148 /*v404*/ offset:224
	ds_load_tr16_b128 v[220:223], v148 /*v404*/ offset:4800
	ds_load_tr16_b128 v[204:207], v148 /*v404*/ offset:4832
	ds_load_tr16_b128 v[208:211], v148 /*v404*/ offset:9408
	ds_load_tr16_b128 v[192:195], v148 /*v404*/ offset:9440
	ds_load_tr16_b128 v[212:215], v148 /*v404*/ offset:14016
	ds_load_tr16_b128 v[196:199], v148 /*v404*/ offset:14048
	s_set_vgpr_msb 0x155
	v_lshl_or_b32 v132 /*v388*/, s82, 6, v143 /*v399*/
	v_dual_add_nc_u32 v174 /*v430*/, 16, v132 /*v388*/ :: v_dual_bitop2_b32 v133 /*v389*/, 1, v132 /*v388*/ bitop3:0x54
	v_cmp_gt_i32_e32 vcc_lo, v132 /*v388*/, v154 /*v410*/
	v_cmp_lt_i32_e64 s2, v132 /*v388*/, v152 /*v408*/
	v_cmp_ge_i32_e64 s3, v132 /*v388*/, v154 /*v410*/
	v_dual_add_nc_u32 v175 /*v431*/, 17, v132 /*v388*/ :: v_dual_bitop2_b32 v134 /*v390*/, 2, v132 /*v388*/ bitop3:0x54
	v_cmp_lt_i32_e64 s4, v133 /*v389*/, v152 /*v408*/
	v_dual_add_nc_u32 v176 /*v432*/, 18, v132 /*v388*/ :: v_dual_bitop2_b32 v135 /*v391*/, 3, v132 /*v388*/ bitop3:0x54
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v134 /*v390*/, v154 /*v410*/
	v_cndmask_b32_e64 v66 /*v322*/, v66 /*v322*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v134 /*v390*/, v152 /*v408*/
	s_or_b32 s3, s4, s3
	v_dual_add_nc_u32 v177 /*v433*/, 19, v132 /*v388*/ :: v_dual_bitop2_b32 v168 /*v424*/, 4, v132 /*v388*/ bitop3:0x54
	v_cmp_gt_i32_e64 s5, v135 /*v391*/, v154 /*v410*/
	v_cndmask_b32_e64 v67 /*v323*/, v67 /*v323*/, 0xff800000, s3
	v_cmp_lt_i32_e64 s3, v135 /*v391*/, v152 /*v408*/
	v_dual_add_nc_u32 v178 /*v434*/, 20, v132 /*v388*/ :: v_dual_bitop2_b32 v171 /*v427*/, 5, v132 /*v388*/ bitop3:0x54
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v168 /*v424*/, v154 /*v410*/
	v_cndmask_b32_e64 v68 /*v324*/, v68 /*v324*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v168 /*v424*/, v152 /*v408*/
	s_or_b32 s3, s3, s5
	v_or_b32_e32 v172 /*v428*/, 6, v132 /*v388*/
	v_cndmask_b32_e64 v69 /*v325*/, v69 /*v325*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v171 /*v427*/, v154 /*v410*/
	v_cmp_lt_i32_e64 s4, v171 /*v427*/, v152 /*v408*/
	v_or_b32_e32 v173 /*v429*/, 7, v132 /*v388*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v172 /*v428*/, v154 /*v410*/
	v_cndmask_b32_e64 v70 /*v326*/, v70 /*v326*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v172 /*v428*/, v152 /*v408*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s4, v173 /*v429*/, v152 /*v408*/
	v_cndmask_b32_e64 v71 /*v327*/, v71 /*v327*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v173 /*v429*/, v154 /*v410*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v174 /*v430*/, v154 /*v410*/
	v_cndmask_b32_e64 v72 /*v328*/, v72 /*v328*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v174 /*v430*/, v152 /*v408*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s4, v175 /*v431*/, v152 /*v408*/
	v_cndmask_b32_e64 v73 /*v329*/, v73 /*v329*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v175 /*v431*/, v154 /*v410*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v176 /*v432*/, v154 /*v410*/
	v_cndmask_b32_e64 v82 /*v338*/, v82 /*v338*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v176 /*v432*/, v152 /*v408*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s4, v177 /*v433*/, v152 /*v408*/
	v_cndmask_b32_e64 v83 /*v339*/, v83 /*v339*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v177 /*v433*/, v154 /*v410*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v178 /*v434*/, v154 /*v410*/
	v_cndmask_b32_e64 v130 /*v386*/, v84 /*v340*/, 0xff800000, s2
	v_add_nc_u32_e32 v84 /*v340*/, 21, v132 /*v388*/
	v_cmp_lt_i32_e64 s2, v178 /*v434*/, v152 /*v408*/
	s_or_b32 s3, s4, s3
	v_dual_add_nc_u32 v180 /*v436*/, 23, v132 /*v388*/ :: v_dual_bitop2_b32 v181 /*v437*/, 32, v132 /*v388*/ bitop3:0x54
	v_cndmask_b32_e64 v131 /*v387*/, v85 /*v341*/, 0xff800000, s3
	v_add_nc_u32_e32 v85 /*v341*/, 22, v132 /*v388*/
	v_cmp_gt_i32_e64 s3, v84 /*v340*/, v154 /*v410*/
	v_cmp_lt_i32_e64 s4, v84 /*v340*/, v152 /*v408*/
	s_or_b32 s2, s2, vcc_lo
	v_dual_add_nc_u32 v190 /*v446*/, 48, v132 /*v388*/ :: v_dual_bitop2_b32 v183 /*v439*/, 33, v132 /*v388*/ bitop3:0x54
	v_cndmask_b32_e64 v86 /*v342*/, v86 /*v342*/, 0xff800000, s2
	v_cmp_gt_i32_e32 vcc_lo, v85 /*v341*/, v154 /*v410*/
	v_cmp_lt_i32_e64 s2, v85 /*v341*/, v152 /*v408*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s4, v180 /*v436*/, v152 /*v408*/
	v_cndmask_b32_e64 v87 /*v343*/, v87 /*v343*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v180 /*v436*/, v154 /*v410*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v181 /*v437*/, v154 /*v410*/
	v_cndmask_b32_e64 v88 /*v344*/, v88 /*v344*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v181 /*v437*/, v152 /*v408*/
	s_or_b32 s3, s4, s3
	v_dual_add_nc_u32 v191 /*v447*/, 49, v132 /*v388*/ :: v_dual_bitop2_b32 v184 /*v440*/, 34, v132 /*v388*/ bitop3:0x54
	v_cndmask_b32_e64 v89 /*v345*/, v89 /*v345*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v183 /*v439*/, v154 /*v410*/
	v_cmp_lt_i32_e64 s4, v183 /*v439*/, v152 /*v408*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v184 /*v440*/, v154 /*v410*/
	v_cndmask_b32_e64 v158 /*v414*/, v98 /*v354*/, 0xff800000, s2
	v_dual_add_nc_u32 v192 /*v448*/, 50, v132 /*v388*/ :: v_dual_bitop2_b32 v98 /*v354*/, 35, v132 /*v388*/ bitop3:0x54
	v_cmp_lt_i32_e64 s2, v184 /*v440*/, v152 /*v408*/
	s_or_b32 s3, s4, s3
	v_or_b32_e32 v189 /*v445*/, 39, v132 /*v388*/
	v_cndmask_b32_e64 v159 /*v415*/, v99 /*v355*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v98 /*v354*/, v154 /*v410*/
	v_or_b32_e32 v99 /*v355*/, 36, v132 /*v388*/
	v_cmp_lt_i32_e64 s4, v98 /*v354*/, v152 /*v408*/
	s_or_b32 s2, s2, vcc_lo
	v_add_nc_u32_e32 v195 /*v451*/, 55, v132 /*v388*/
	v_cndmask_b32_e64 v160 /*v416*/, v100 /*v356*/, 0xff800000, s2
	v_or_b32_e32 v100 /*v356*/, 37, v132 /*v388*/
	v_cmp_gt_i32_e32 vcc_lo, v99 /*v355*/, v154 /*v410*/
	v_cmp_lt_i32_e64 s2, v99 /*v355*/, v152 /*v408*/
	s_or_b32 s3, s4, s3
	v_cndmask_b32_e64 v161 /*v417*/, v101 /*v357*/, 0xff800000, s3
	v_or_b32_e32 v101 /*v357*/, 38, v132 /*v388*/
	v_cmp_gt_i32_e64 s3, v100 /*v356*/, v154 /*v410*/
	v_cmp_lt_i32_e64 s4, v100 /*v356*/, v152 /*v408*/
	s_or_b32 s2, s2, vcc_lo
	v_cndmask_b32_e64 v102 /*v358*/, v102 /*v358*/, 0xff800000, s2
	v_cmp_gt_i32_e32 vcc_lo, v101 /*v357*/, v154 /*v410*/
	v_cmp_lt_i32_e64 s2, v101 /*v357*/, v152 /*v408*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s4, v189 /*v445*/, v152 /*v408*/
	v_cndmask_b32_e64 v103 /*v359*/, v103 /*v359*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v189 /*v445*/, v154 /*v410*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v190 /*v446*/, v154 /*v410*/
	v_cndmask_b32_e64 v104 /*v360*/, v104 /*v360*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v190 /*v446*/, v152 /*v408*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s4, v191 /*v447*/, v152 /*v408*/
	v_cndmask_b32_e64 v105 /*v361*/, v105 /*v361*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v191 /*v447*/, v154 /*v410*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v192 /*v448*/, v154 /*v410*/
	v_cndmask_b32_e64 v162 /*v418*/, v114 /*v370*/, 0xff800000, s2
	v_add_nc_u32_e32 v114 /*v370*/, 51, v132 /*v388*/
	v_cmp_lt_i32_e64 s2, v192 /*v448*/, v152 /*v408*/
	s_or_b32 s3, s4, s3
	v_cndmask_b32_e64 v163 /*v419*/, v115 /*v371*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v114 /*v370*/, v154 /*v410*/
	v_cmp_lt_i32_e64 s4, v114 /*v370*/, v152 /*v408*/
	s_or_b32 s2, s2, vcc_lo
	v_add_nc_u32_e32 v115 /*v371*/, 52, v132 /*v388*/
	v_cndmask_b32_e64 v164 /*v420*/, v116 /*v372*/, 0xff800000, s2
	v_add_nc_u32_e32 v116 /*v372*/, 53, v132 /*v388*/
	s_or_b32 s2, s4, s3
	v_cndmask_b32_e64 v165 /*v421*/, v117 /*v373*/, 0xff800000, s2
	v_add_nc_u32_e32 v117 /*v373*/, 54, v132 /*v388*/
	v_cmp_gt_i32_e32 vcc_lo, v115 /*v371*/, v154 /*v410*/
	v_cmp_lt_i32_e64 s2, v115 /*v371*/, v152 /*v408*/
	v_cmp_gt_i32_e64 s3, v116 /*v372*/, v154 /*v410*/
	v_cmp_lt_i32_e64 s4, v116 /*v372*/, v152 /*v408*/
	v_cmp_gt_i32_e64 s5, v117 /*v373*/, v154 /*v410*/
	v_cmp_lt_i32_e64 s6, v117 /*v373*/, v152 /*v408*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v195 /*v451*/, v154 /*v410*/
	v_cndmask_b32_e64 v118 /*v374*/, v118 /*v374*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	v_cmp_gt_i32_e64 s3, v132 /*v388*/, v155 /*v411*/
	v_cndmask_b32_e64 v119 /*v375*/, v119 /*v375*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	v_cmp_lt_i32_e64 s4, v132 /*v388*/, v153 /*v409*/
	v_cndmask_b32_e64 v120 /*v376*/, v120 /*v376*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v195 /*v451*/, v152 /*v408*/
	v_cmp_ge_i32_e64 s5, v132 /*v388*/, v155 /*v411*/
	v_cmp_lt_i32_e64 s6, v133 /*v389*/, v153 /*v409*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v134 /*v390*/, v155 /*v411*/
	v_cndmask_b32_e64 v121 /*v377*/, v121 /*v377*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	v_cmp_gt_i32_e64 s3, v135 /*v391*/, v155 /*v411*/
	v_cndmask_b32_e64 v166 /*v422*/, v74 /*v330*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	v_cmp_lt_i32_e64 s4, v135 /*v391*/, v153 /*v409*/
	v_cndmask_b32_e64 v167 /*v423*/, v75 /*v331*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v134 /*v390*/, v153 /*v409*/
	v_cmp_gt_i32_e64 s5, v168 /*v424*/, v155 /*v411*/
	v_cmp_lt_i32_e64 s6, v168 /*v424*/, v153 /*v409*/
	v_max_num_f32_e32 v74 /*v330*/, v66 /*v322*/, v67 /*v323*/
	v_max_num_f32_e32 v75 /*v331*/, v166 /*v422*/, v167 /*v423*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v171 /*v427*/, v155 /*v411*/
	v_cndmask_b32_e64 v168 /*v424*/, v76 /*v332*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	v_cmp_gt_i32_e64 s3, v172 /*v428*/, v155 /*v411*/
	v_cndmask_b32_e64 v169 /*v425*/, v77 /*v333*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	v_cmp_lt_i32_e64 s4, v172 /*v428*/, v153 /*v409*/
	v_cndmask_b32_e64 v170 /*v426*/, v78 /*v334*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v171 /*v427*/, v153 /*v409*/
	v_cmp_gt_i32_e64 s5, v173 /*v429*/, v155 /*v411*/
	v_cmp_lt_i32_e64 s6, v173 /*v429*/, v153 /*v409*/
	v_max3_num_f32 v76 /*v332*/, v69 /*v325*/, v70 /*v326*/, v71 /*v327*/
	v_max3_num_f32 v78 /*v334*/, v72 /*v328*/, v73 /*v329*/, v82 /*v338*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v174 /*v430*/, v155 /*v411*/
	v_cndmask_b32_e64 v171 /*v427*/, v79 /*v335*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	v_cmp_gt_i32_e64 s3, v175 /*v431*/, v155 /*v411*/
	v_cndmask_b32_e64 v172 /*v428*/, v80 /*v336*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	v_cmp_lt_i32_e64 s4, v175 /*v431*/, v153 /*v409*/
	v_cndmask_b32_e64 v173 /*v429*/, v81 /*v337*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v174 /*v430*/, v153 /*v409*/
	v_cmp_gt_i32_e64 s5, v176 /*v432*/, v155 /*v411*/
	v_cmp_lt_i32_e64 s6, v176 /*v432*/, v153 /*v409*/
	v_max3_num_f32 v80 /*v336*/, v83 /*v339*/, v130 /*v386*/, v131 /*v387*/
	v_max3_num_f32 v77 /*v333*/, v169 /*v425*/, v170 /*v426*/, v171 /*v427*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v177 /*v433*/, v155 /*v411*/
	v_cndmask_b32_e64 v174 /*v430*/, v90 /*v346*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	v_cmp_gt_i32_e64 s3, v178 /*v434*/, v155 /*v411*/
	v_cndmask_b32_e64 v175 /*v431*/, v91 /*v347*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	v_cmp_lt_i32_e64 s4, v178 /*v434*/, v153 /*v409*/
	v_cndmask_b32_e64 v176 /*v432*/, v92 /*v348*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v177 /*v433*/, v153 /*v409*/
	v_cmp_gt_i32_e64 s5, v84 /*v340*/, v155 /*v411*/
	v_cmp_lt_i32_e64 s6, v84 /*v340*/, v153 /*v409*/
	v_max3_num_f32 v84 /*v340*/, v86 /*v342*/, v87 /*v343*/, v88 /*v344*/
	v_max3_num_f32 v90 /*v346*/, v89 /*v345*/, v158 /*v414*/, v159 /*v415*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v85 /*v341*/, v155 /*v411*/
	v_cndmask_b32_e64 v177 /*v433*/, v93 /*v349*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	v_cmp_gt_i32_e64 s3, v180 /*v436*/, v155 /*v411*/
	v_cndmask_b32_e64 v178 /*v434*/, v94 /*v350*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	v_cmp_lt_i32_e64 s4, v180 /*v436*/, v153 /*v409*/
	v_cndmask_b32_e64 v179 /*v435*/, v95 /*v351*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v85 /*v341*/, v153 /*v409*/
	v_cmp_gt_i32_e64 s5, v181 /*v437*/, v155 /*v411*/
	v_cmp_lt_i32_e64 s6, v181 /*v437*/, v153 /*v409*/
	v_max3_num_f32 v81 /*v337*/, v175 /*v431*/, v176 /*v432*/, v177 /*v433*/
	v_max3_num_f32 v92 /*v348*/, v160 /*v416*/, v161 /*v417*/, v102 /*v358*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v183 /*v439*/, v155 /*v411*/
	v_cndmask_b32_e64 v180 /*v436*/, v96 /*v352*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	v_cmp_gt_i32_e64 s3, v184 /*v440*/, v155 /*v411*/
	v_cndmask_b32_e64 v181 /*v437*/, v97 /*v353*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	v_cmp_lt_i32_e64 s4, v184 /*v440*/, v153 /*v409*/
	v_cndmask_b32_e64 v182 /*v438*/, v106 /*v362*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v183 /*v439*/, v153 /*v409*/
	v_cmp_gt_i32_e64 s5, v98 /*v354*/, v155 /*v411*/
	v_cmp_lt_i32_e64 s6, v98 /*v354*/, v153 /*v409*/
	v_max3_num_f32 v85 /*v341*/, v178 /*v434*/, v179 /*v435*/, v180 /*v436*/
	v_max3_num_f32 v93 /*v349*/, v103 /*v359*/, v104 /*v360*/, v105 /*v361*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v99 /*v355*/, v155 /*v411*/
	v_cndmask_b32_e64 v183 /*v439*/, v107 /*v363*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	v_cmp_gt_i32_e64 s3, v100 /*v356*/, v155 /*v411*/
	v_cndmask_b32_e64 v184 /*v440*/, v108 /*v364*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	v_cmp_lt_i32_e64 s4, v100 /*v356*/, v153 /*v409*/
	v_cndmask_b32_e64 v185 /*v441*/, v109 /*v365*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v99 /*v355*/, v153 /*v409*/
	v_cmp_gt_i32_e64 s5, v101 /*v357*/, v155 /*v411*/
	v_cmp_lt_i32_e64 s6, v101 /*v357*/, v153 /*v409*/
	v_max3_num_f32 v91 /*v347*/, v181 /*v437*/, v182 /*v438*/, v183 /*v439*/
	v_max3_num_f32 v94 /*v350*/, v162 /*v418*/, v163 /*v419*/, v164 /*v420*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v189 /*v445*/, v155 /*v411*/
	v_cndmask_b32_e64 v186 /*v442*/, v110 /*v366*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	v_cmp_gt_i32_e64 s3, v190 /*v446*/, v155 /*v411*/
	v_cndmask_b32_e64 v187 /*v443*/, v111 /*v367*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	v_cmp_lt_i32_e64 s4, v190 /*v446*/, v153 /*v409*/
	v_cndmask_b32_e64 v188 /*v444*/, v112 /*v368*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v189 /*v445*/, v153 /*v409*/
	v_cmp_gt_i32_e64 s5, v191 /*v447*/, v155 /*v411*/
	v_cmp_lt_i32_e64 s6, v191 /*v447*/, v153 /*v409*/
	v_max3_num_f32 v95 /*v351*/, v165 /*v421*/, v118 /*v374*/, v119 /*v375*/
	v_max3_num_f32 v74 /*v330*/, v74 /*v330*/, v68 /*v324*/, v76 /*v332*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v192 /*v448*/, v155 /*v411*/
	v_cndmask_b32_e64 v189 /*v445*/, v113 /*v369*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	v_cmp_gt_i32_e64 s3, v114 /*v370*/, v155 /*v411*/
	v_cndmask_b32_e64 v190 /*v446*/, v122 /*v378*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	v_cmp_lt_i32_e64 s4, v114 /*v370*/, v153 /*v409*/
	v_cndmask_b32_e64 v191 /*v447*/, v123 /*v379*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v192 /*v448*/, v153 /*v409*/
	v_cmp_gt_i32_e64 s5, v115 /*v371*/, v155 /*v411*/
	v_cmp_lt_i32_e64 s6, v115 /*v371*/, v153 /*v409*/
	v_max3_num_f32 v76 /*v332*/, v80 /*v336*/, v84 /*v340*/, v90 /*v346*/
	v_max3_num_f32 v79 /*v335*/, v172 /*v428*/, v173 /*v429*/, v174 /*v430*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v116 /*v372*/, v155 /*v411*/
	v_cndmask_b32_e64 v192 /*v448*/, v124 /*v380*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	v_cmp_gt_i32_e64 s3, v117 /*v373*/, v155 /*v411*/
	v_cndmask_b32_e64 v193 /*v449*/, v125 /*v381*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	v_cmp_lt_i32_e64 s4, v117 /*v373*/, v153 /*v409*/
	v_cndmask_b32_e64 v194 /*v450*/, v126 /*v382*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v116 /*v372*/, v153 /*v409*/
	v_cmp_gt_i32_e64 s5, v195 /*v451*/, v155 /*v411*/
	v_cmp_lt_i32_e64 s6, v195 /*v451*/, v153 /*v409*/
	v_max3_num_f32 v80 /*v336*/, v184 /*v440*/, v185 /*v441*/, v186 /*v442*/
	v_max3_num_f32 v84 /*v340*/, v187 /*v443*/, v188 /*v444*/, v189 /*v445*/
	s_or_b32 s2, s2, vcc_lo
	v_max3_num_f32 v90 /*v346*/, v92 /*v348*/, v93 /*v349*/, v94 /*v350*/
	v_cndmask_b32_e64 v195 /*v451*/, v127 /*v383*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	v_max3_num_f32 v92 /*v348*/, v95 /*v351*/, v120 /*v376*/, v121 /*v377*/
	v_cndmask_b32_e64 v196 /*v452*/, v128 /*v384*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	v_max3_num_f32 v74 /*v330*/, v74 /*v330*/, v78 /*v334*/, v76 /*v332*/
	v_cndmask_b32_e64 v197 /*v453*/, v129 /*v385*/, 0xff800000, s2
	v_max3_num_f32 v76 /*v332*/, v190 /*v446*/, v191 /*v447*/, v192 /*v448*/
	v_max3_num_f32 v78 /*v334*/, v193 /*v449*/, v194 /*v450*/, v195 /*v451*/
	v_max3_num_f32 v75 /*v331*/, v75 /*v331*/, v168 /*v424*/, v77 /*v333*/
	v_max3_num_f32 v77 /*v333*/, v81 /*v337*/, v85 /*v341*/, v91 /*v347*/
	v_max3_num_f32 v74 /*v330*/, v74 /*v330*/, v90 /*v346*/, v92 /*v348*/
	v_max3_num_f32 v76 /*v332*/, v80 /*v336*/, v84 /*v340*/, v76 /*v332*/
	v_max3_num_f32 v78 /*v334*/, v78 /*v334*/, v196 /*v452*/, v197 /*v453*/
	v_max3_num_f32 v75 /*v331*/, v75 /*v331*/, v79 /*v335*/, v77 /*v333*/
	v_max3_num_f32 v75 /*v331*/, v75 /*v331*/, v76 /*v332*/, v78 /*v334*/
	v_dual_mov_b32 v77 /*v333*/, v74 /*v330*/ :: v_dual_mov_b32 v76 /*v332*/, v75 /*v331*/
	v_permlanex16_b32 v77 /*v333*/, v77 /*v333*/, s61, 0xfedcba98
	v_permlanex16_b32 v76 /*v332*/, v76 /*v332*/, s61, 0xfedcba98
	v_dual_max_num_f32 v74 /*v330*/, v74 /*v330*/, v77 /*v333*/ :: v_dual_max_num_f32 v75 /*v331*/, v75 /*v331*/, v76 /*v332*/
	v_dual_sub_f32 v77 /*v333*/, v74 /*v330*/, v151 /*v407*/ :: v_dual_max_num_f32 v74 /*v330*/, v151 /*v407*/, v74 /*v330*/
	v_sub_f32_e32 v76 /*v332*/, v75 /*v331*/, v150 /*v406*/
	v_cmp_lt_f32_e32 vcc_lo, 0x41000000, v77 /*v333*/
	s_cmp_eq_u32 vcc_lo, 0
	s_cselect_b32 s2, -1, 0
	v_cndmask_b32_e64 v133 /*v389*/, v74 /*v330*/, v151 /*v407*/, s2
	v_cmp_lt_f32_e64 s2, 0x41000000, v76 /*v332*/
	v_max_num_f32_e32 v74 /*v330*/, v150 /*v406*/, v75 /*v331*/
	s_cmp_lg_u32 s2, 0
	s_cselect_b32 s3, -1, 0
	s_cmp_eq_u32 s2, 0
	s_cselect_b32 s2, -1, 0
	v_cndmask_b32_e64 v135 /*v391*/, v74 /*v330*/, v150 /*v406*/, s2
	v_mul_f32_e32 v124 /*v380*/, 0xbfb8aa3b, v133 /*v389*/
	v_mul_f32_e32 v132 /*v388*/, 0xbfb8aa3b, v135 /*v391*/
	v_pk_fma_f32 v[72:73] /*v[328:329]*/, v[72:73] /*v[328:329]*/, s[88:89], v[124:125] /*v[380:381]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[80:81] /*v[336:337]*/, v[130:131] /*v[386:387]*/, s[88:89], v[124:125] /*v[380:381]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[66:67] /*v[322:323]*/, v[66:67] /*v[322:323]*/, s[88:89], v[124:125] /*v[380:381]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[74:75] /*v[330:331]*/, v[68:69] /*v[324:325]*/, s[88:89], v[124:125] /*v[380:381]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[76:77] /*v[332:333]*/, v[70:71] /*v[326:327]*/, s[88:89], v[124:125] /*v[380:381]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v84 /*v340*/, v72 /*v328*/
	v_exp_f32_e32 v90 /*v346*/, v73 /*v329*/
	v_nop
	v_pk_fma_f32 v[72:73] /*v[328:329]*/, v[86:87] /*v[342:343]*/, s[88:89], v[124:125] /*v[380:381]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v100 /*v356*/, v80 /*v336*/
	v_exp_f32_e32 v106 /*v362*/, v81 /*v337*/
	v_nop
	v_pk_fma_f32 v[80:81] /*v[336:337]*/, v[158:159] /*v[414:415]*/, s[88:89], v[124:125] /*v[380:381]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[86:87] /*v[342:343]*/, v[160:161] /*v[416:417]*/, s[88:89], v[124:125] /*v[380:381]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[130:131] /*v[386:387]*/, v[166:167] /*v[422:423]*/, s[88:89], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[158:159] /*v[414:415]*/, v[168:169] /*v[424:425]*/, s[88:89], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[160:161] /*v[416:417]*/, v[170:171] /*v[426:427]*/, s[88:89], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v68 /*v324*/, v67 /*v323*/
	v_exp_f32_e32 v70 /*v326*/, v74 /*v330*/
	v_exp_f32_e32 v74 /*v330*/, v75 /*v331*/
	v_pk_fma_f32 v[78:79] /*v[334:335]*/, v[82:83] /*v[338:339]*/, s[88:89], v[124:125] /*v[380:381]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v82 /*v338*/, v77 /*v333*/
	v_exp_f32_e32 v67 /*v323*/, v130 /*v386*/
	v_exp_f32_e32 v69 /*v325*/, v131 /*v387*/
	v_exp_f32_e32 v71 /*v327*/, v158 /*v414*/
	v_pk_fma_f32 v[130:131] /*v[386:387]*/, v[172:173] /*v[428:429]*/, s[88:89], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v75 /*v331*/, v159 /*v415*/
	v_exp_f32_e32 v77 /*v333*/, v160 /*v416*/
	v_pk_fma_f32 v[158:159] /*v[414:415]*/, v[174:175] /*v[430:431]*/, s[88:89], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v83 /*v339*/, v161 /*v417*/
	v_nop
	v_pk_fma_f32 v[160:161] /*v[416:417]*/, v[176:177] /*v[432:433]*/, s[88:89], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v92 /*v348*/, v78 /*v334*/
	v_exp_f32_e32 v98 /*v354*/, v79 /*v335*/
	v_nop
	v_pk_fma_f32 v[78:79] /*v[334:335]*/, v[88:89] /*v[344:345]*/, s[88:89], v[124:125] /*v[380:381]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v85 /*v341*/, v130 /*v386*/
	v_exp_f32_e32 v91 /*v347*/, v131 /*v387*/
	v_exp_f32_e32 v93 /*v349*/, v158 /*v414*/
	v_pk_fma_f32 v[130:131] /*v[386:387]*/, v[178:179] /*v[434:435]*/, s[88:89], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v99 /*v355*/, v159 /*v415*/
	v_exp_f32_e32 v101 /*v357*/, v160 /*v416*/
	v_pk_fma_f32 v[158:159] /*v[414:415]*/, v[180:181] /*v[436:437]*/, s[88:89], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v107 /*v363*/, v161 /*v417*/
	v_nop
	v_pk_fma_f32 v[160:161] /*v[416:417]*/, v[182:183] /*v[438:439]*/, s[88:89], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v114 /*v370*/, v73 /*v329*/
	v_exp_f32_e32 v122 /*v378*/, v79 /*v335*/
	v_pk_fma_f32 v[88:89] /*v[344:345]*/, v[102:103] /*v[358:359]*/, s[88:89], v[124:125] /*v[380:381]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[96:97] /*v[352:353]*/, v[104:105] /*v[360:361]*/, s[88:89], v[124:125] /*v[380:381]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v109 /*v365*/, v130 /*v386*/
	v_exp_f32_e32 v115 /*v371*/, v131 /*v387*/
	v_exp_f32_e32 v117 /*v373*/, v158 /*v414*/
	v_pk_fma_f32 v[130:131] /*v[386:387]*/, v[184:185] /*v[440:441]*/, s[88:89], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v123 /*v379*/, v159 /*v415*/
	v_exp_f32_e32 v73 /*v329*/, v160 /*v416*/
	v_pk_fma_f32 v[158:159] /*v[414:415]*/, v[186:187] /*v[442:443]*/, s[88:89], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v79 /*v335*/, v161 /*v417*/
	v_nop
	v_pk_fma_f32 v[160:161] /*v[416:417]*/, v[188:189] /*v[444:445]*/, s[88:89], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v66 /*v322*/, v66 /*v322*/
	v_exp_f32_e32 v76 /*v332*/, v76 /*v332*/
	v_exp_f32_e32 v108 /*v364*/, v72 /*v328*/
	v_exp_f32_e32 v116 /*v372*/, v78 /*v334*/
	v_exp_f32_e32 v72 /*v328*/, v80 /*v336*/
	v_exp_f32_e32 v78 /*v334*/, v81 /*v337*/
	v_exp_f32_e32 v80 /*v336*/, v86 /*v342*/
	v_exp_f32_e32 v86 /*v342*/, v87 /*v343*/
	v_pk_fma_f32 v[104:105] /*v[360:361]*/, v[162:163] /*v[418:419]*/, s[88:89], v[124:125] /*v[380:381]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v94 /*v350*/, v89 /*v345*/
	v_pk_fma_f32 v[112:113] /*v[368:369]*/, v[164:165] /*v[420:421]*/, s[88:89], v[124:125] /*v[380:381]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v102 /*v358*/, v97 /*v353*/
	v_exp_f32_e32 v81 /*v337*/, v130 /*v386*/
	v_exp_f32_e32 v87 /*v343*/, v131 /*v387*/
	v_exp_f32_e32 v89 /*v345*/, v158 /*v414*/
	v_pk_fma_f32 v[130:131] /*v[386:387]*/, v[190:191] /*v[446:447]*/, s[88:89], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v95 /*v351*/, v159 /*v415*/
	v_exp_f32_e32 v97 /*v353*/, v160 /*v416*/
	v_pk_fma_f32 v[158:159] /*v[414:415]*/, v[192:193] /*v[448:449]*/, s[88:89], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v103 /*v359*/, v161 /*v417*/
	v_nop
	v_pk_fma_f32 v[160:161] /*v[416:417]*/, v[194:195] /*v[450:451]*/, s[88:89], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v96 /*v352*/, v96 /*v352*/
	v_pk_fma_f32 v[126:127] /*v[382:383]*/, v[118:119] /*v[374:375]*/, s[88:89], v[124:125] /*v[380:381]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v110 /*v366*/, v105 /*v361*/
	v_pk_fma_f32 v[128:129] /*v[384:385]*/, v[120:121] /*v[376:377]*/, s[88:89], v[124:125] /*v[380:381]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v118 /*v374*/, v113 /*v369*/
	v_exp_f32_e32 v105 /*v361*/, v130 /*v386*/
	v_exp_f32_e32 v111 /*v367*/, v131 /*v387*/
	v_exp_f32_e32 v113 /*v369*/, v158 /*v414*/
	v_pk_fma_f32 v[130:131] /*v[386:387]*/, v[196:197] /*v[452:453]*/, s[88:89], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v119 /*v375*/, v159 /*v415*/
	v_exp_f32_e32 v121 /*v377*/, v160 /*v416*/
	v_exp_f32_e32 v125 /*v381*/, v161 /*v417*/
	v_pk_add_f32 v[158:159] /*v[414:415]*/, v[66:67] /*v[322:323]*/, v[68:69] /*v[324:325]*/
	v_pk_add_f32 v[160:161] /*v[416:417]*/, v[74:75] /*v[330:331]*/, v[76:77] /*v[332:333]*/
	v_pk_add_f32 v[162:163] /*v[418:419]*/, v[98:99] /*v[354:355]*/, v[100:101] /*v[356:357]*/
	v_pk_add_f32 v[164:165] /*v[420:421]*/, v[108:109] /*v[364:365]*/, v[114:115] /*v[370:371]*/
	v_exp_f32_e32 v88 /*v344*/, v88 /*v344*/
	v_exp_f32_e32 v104 /*v360*/, v104 /*v360*/
	v_exp_f32_e32 v120 /*v376*/, v126 /*v382*/
	v_exp_f32_e32 v124 /*v380*/, v127 /*v383*/
	v_exp_f32_e32 v126 /*v382*/, v128 /*v384*/
	v_exp_f32_e32 v128 /*v384*/, v129 /*v385*/
	v_exp_f32_e32 v127 /*v383*/, v130 /*v386*/
	v_exp_f32_e32 v129 /*v385*/, v131 /*v387*/
	v_nop
	v_pk_add_f32 v[130:131] /*v[386:387]*/, v[84:85] /*v[340:341]*/, v[90:91] /*v[346:347]*/
	v_pk_add_f32 v[158:159] /*v[414:415]*/, v[70:71] /*v[326:327]*/, v[158:159] /*v[414:415]*/
	v_pk_add_f32 v[160:161] /*v[416:417]*/, v[82:83] /*v[338:339]*/, v[160:161] /*v[416:417]*/
	v_pk_add_f32 v[166:167] /*v[422:423]*/, v[122:123] /*v[378:379]*/, v[72:73] /*v[328:329]*/
	v_pk_add_f32 v[162:163] /*v[418:419]*/, v[106:107] /*v[362:363]*/, v[162:163] /*v[418:419]*/
	v_pk_add_f32 v[168:169] /*v[424:425]*/, v[80:81] /*v[336:337]*/, v[86:87] /*v[342:343]*/
	v_pk_add_f32 v[164:165] /*v[420:421]*/, v[116:117] /*v[372:373]*/, v[164:165] /*v[420:421]*/
	v_pk_add_f32 v[170:171] /*v[426:427]*/, v[94:95] /*v[350:351]*/, v[96:97] /*v[352:353]*/
	v_exp_f32_e32 v112 /*v368*/, v112 /*v368*/
	v_pk_add_f32 v[130:131] /*v[386:387]*/, v[92:93] /*v[348:349]*/, v[130:131] /*v[386:387]*/
	v_pk_add_f32 v[166:167] /*v[422:423]*/, v[78:79] /*v[334:335]*/, v[166:167] /*v[422:423]*/
	v_pk_add_f32 v[172:173] /*v[428:429]*/, v[104:105] /*v[360:361]*/, v[110:111] /*v[366:367]*/
	v_pk_add_f32 v[168:169] /*v[424:425]*/, v[88:89] /*v[344:345]*/, v[168:169] /*v[424:425]*/
	v_pk_add_f32 v[158:159] /*v[414:415]*/, v[158:159] /*v[414:415]*/, v[160:161] /*v[416:417]*/
	v_pk_add_f32 v[160:161] /*v[416:417]*/, v[102:103] /*v[358:359]*/, v[170:171] /*v[426:427]*/
	v_pk_add_f32 v[162:163] /*v[418:419]*/, v[162:163] /*v[418:419]*/, v[164:165] /*v[420:421]*/
	v_pk_add_f32 v[164:165] /*v[420:421]*/, v[112:113] /*v[368:369]*/, v[172:173] /*v[428:429]*/
	v_pk_add_f32 v[170:171] /*v[426:427]*/, v[118:119] /*v[374:375]*/, v[120:121] /*v[376:377]*/
	v_pk_add_f32 v[130:131] /*v[386:387]*/, v[130:131] /*v[386:387]*/, v[158:159] /*v[414:415]*/
	v_pk_add_f32 v[158:159] /*v[414:415]*/, v[168:169] /*v[424:425]*/, v[160:161] /*v[416:417]*/
	v_pk_add_f32 v[160:161] /*v[416:417]*/, v[166:167] /*v[422:423]*/, v[162:163] /*v[418:419]*/
	v_pk_add_f32 v[166:167] /*v[422:423]*/, v[126:127] /*v[382:383]*/, v[128:129] /*v[384:385]*/
	v_pk_add_f32 v[162:163] /*v[418:419]*/, v[124:125] /*v[380:381]*/, v[170:171] /*v[426:427]*/
	v_sub_f32_e32 v132 /*v388*/, v151 /*v407*/, v133 /*v389*/
	v_pk_add_f32 v[158:159] /*v[414:415]*/, v[164:165] /*v[420:421]*/, v[158:159] /*v[414:415]*/
	v_pk_add_f32 v[130:131] /*v[386:387]*/, v[130:131] /*v[386:387]*/, v[160:161] /*v[416:417]*/
	v_pk_add_f32 v[160:161] /*v[416:417]*/, v[166:167] /*v[422:423]*/, v[162:163] /*v[418:419]*/
	v_mul_f32_e32 v132 /*v388*/, 0x3fb8aa3b, v132 /*v388*/
	v_pk_add_f32 v[130:131] /*v[386:387]*/, v[158:159] /*v[414:415]*/, v[130:131] /*v[386:387]*/
	v_exp_f32_e32 v132 /*v388*/, v132 /*v388*/
	v_pk_add_f32 v[130:131] /*v[386:387]*/, v[160:161] /*v[416:417]*/, v[130:131] /*v[386:387]*/
	v_dual_mov_b32 v151 /*v407*/, v130 /*v386*/ :: v_dual_mov_b32 v158 /*v414*/, v131 /*v387*/
	v_permlanex16_b32 v151 /*v407*/, v151 /*v407*/, s61, 0xfedcba98
	v_permlanex16_b32 v158 /*v414*/, v158 /*v414*/, s61, 0xfedcba98
	s_set_vgpr_msb 0x5500
	s_cbranch_vccz .LBB0_30
	s_set_vgpr_msb 4
	v_pk_mul_f32 v[126:127], v[126:127], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[124:125], v[124:125], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[122:123], v[122:123], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[120:121], v[120:121], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[118:119], v[118:119], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[116:117], v[116:117], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[114:115], v[114:115], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[112:113], v[112:113], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[110:111], v[110:111], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[108:109], v[108:109], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[106:107], v[106:107], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[104:105], v[104:105], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[102:103], v[102:103], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[100:101], v[100:101], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[98:99], v[98:99], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[96:97], v[96:97], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[94:95], v[94:95], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[92:93], v[92:93], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[90:91], v[90:91], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[88:89], v[88:89], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[86:87], v[86:87], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[84:85], v[84:85], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[82:83], v[82:83], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[80:81], v[80:81], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[78:79], v[78:79], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[76:77], v[76:77], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[74:75], v[74:75], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[72:73], v[72:73], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[70:71], v[70:71], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[68:69], v[68:69], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[66:67], v[66:67], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[64:65], v[64:65], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x400
.LBB0_30:
	s_set_vgpr_msb 0x45
	v_sub_f32_e32 v134 /*v390*/, v150 /*v406*/, v135 /*v391*/
	s_and_b32 s2, s3, exec_lo
	s_cselect_b32 s2, 1, 0
	s_cmp_lg_u32 s2, 1
	v_mul_f32_e32 v134 /*v390*/, 0x3fb8aa3b, v134 /*v390*/
	v_exp_f32_e32 v134 /*v390*/, v134 /*v390*/
	s_set_vgpr_msb 0x4500
	s_cbranch_scc1 .LBB0_25
	v_nop
	s_set_vgpr_msb 4
	v_pk_mul_f32 v[62:63], v[62:63], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[60:61], v[60:61], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59], v[58:59], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[56:57], v[56:57], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[54:55], v[54:55], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[52:53], v[52:53], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[50:51], v[50:51], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[48:49], v[48:49], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[46:47], v[46:47], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[44:45], v[44:45], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[42:43], v[42:43], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[40:41], v[40:41], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39], v[38:39], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[36:37], v[36:37], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35], v[34:35], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[32:33], v[32:33], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[30:31], v[30:31], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[28:29], v[28:29], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[26:27], v[26:27], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25], v[24:25], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[22:23], v[22:23], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21], v[20:21], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[18:19], v[18:19], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[16:17], v[16:17], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[14:15], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13], v[12:13], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[10:11], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[8:9], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[6:7], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[4:5], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[2:3], v[2:3], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[0:1], v[0:1], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x400
	s_branch .LBB0_25
.LBB0_32:
	s_set_vgpr_msb 0x41
	v_dual_mov_b32 v135 /*v391*/, v150 /*v406*/ :: v_dual_mov_b32 v133 /*v389*/, v151 /*v407*/
	s_set_vgpr_msb 0x4100
.LBB0_33:
	s_branch .LBB0_65
.LBB0_34:
	s_mov_b32 s27, 0
	s_cmp_lg_u32 s62, 0x80000000
	s_mov_b32 s25, s27
	s_mov_b32 s26, s27
	s_mov_b32 s4, s27
	s_mov_b32 s5, s27
	s_mov_b32 s6, s27
	s_mov_b32 s7, s27
	s_cselect_b32 s3, s63, 0
	tensor_load_to_lds s[20:23], s[36:43], s[24:27], s[4:7]
	s_cselect_b32 s25, s62, 0x80
	s_and_b32 s6, s102, 3
	s_mov_b32 s2, s25
	s_lshl_b32 s80, s6, 4
	s_lshl_b32 s4, s6, 5
	s_and_b32 s26, s3, 0xffff
	s_mul_u64 s[44:45], s[2:3], s[4:5]
	s_sub_co_i32 s2, s103, s80
	s_mul_i32 s81, s6, 0x1100
	s_max_i32 s2, s2, 0
	s_add_nc_u64 s[30:31], s[44:45], s[78:79]
	s_lshl_b32 s2, s2, 16
	s_mov_b32 s28, 1
	s_or_b32 s22, s2, 0x7fff
	s_cmp_lg_u32 s64, 0x80000000
	s_mov_b32 s23, 0x800000
	s_cselect_b32 s41, s64, 0x80
	s_cselect_b32 s3, s65, 0
	s_mov_b32 s2, s41
	s_mov_b32 s21, 0xffff0000
	s_mov_b32 s20, 0x7510000
	s_mov_b32 s24, 16
	s_mulk_i32 s6, 0x1200
	s_mov_b32 s29, s81
	s_bitset1_b32 s31, 31
	s_mul_u64 s[46:47], s[2:3], s[4:5]
	s_or_b32 s78, s6, 0x10000
	s_mov_b32 s36, 0xf510000
	s_mov_b32 s37, s21
	s_mov_b32 s39, s23
	s_mov_b32 s43, s27
	s_mov_b32 s40, s24
	s_mov_b32 s38, s22
	s_and_b32 s42, s3, 0xffff
	s_add_co_i32 s101, s101, s17
	s_mov_b32 s55, s27
	s_max_i32 s2, s101, 0
	s_add_nc_u64 s[50:51], s[68:69], s[74:75]
	s_add_co_i32 s2, s2, 1
	s_add_nc_u64 s[52:53], s[66:67], s[72:73]
	s_ashr_i32 s3, s2, 31
	s_lshr_b32 s3, s3, 26
	s_add_co_i32 s3, s2, s3
	s_and_b32 s4, s3, 0xffffffc0
	s_ashr_i32 s3, s3, 6
	s_cmp_lg_u32 s2, s4
	s_cselect_b32 s4, -1, 0
	s_cmp_lt_i32 s2, 0
	s_cselect_b32 s2, -1, 0
	s_and_b32 s2, s2, s4
	s_sub_co_ci_u32 s2, s3, 0
	s_sub_co_i32 s3, s100, s16
	s_min_i32 s7, s2, s98
	s_max_i32 s3, s3, 0
	s_max_i32 s48, s7, s60
	s_add_co_i32 s3, s3, 63
	s_ashr_i32 s4, s3, 31
	s_lshr_b32 s4, s4, 26
	s_add_co_i32 s2, s3, s4
	s_and_b32 s4, s2, 0xffffffc0
	s_ashr_i32 s2, s2, 6
	s_cmp_lg_u32 s3, s4
	s_cselect_b32 s4, -1, 0
	s_cmp_lt_i32 s3, 0
	s_cselect_b32 s3, -1, 0
	s_and_b32 s3, s3, s4
	s_sub_co_ci_u32 s2, s2, 0
	s_max_i32 s11, s2, s60
	s_min_i32 s54, s11, s48
	s_cmp_ge_u32 s60, s54
	s_wait_tensorcnt 0x0
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 17
	ds_load_b128 v[0:3], v147 /*v403*/
	ds_load_b128 v[4:7], v147 /*v403*/ offset:32
	ds_load_b128 v[8:11], v147 /*v403*/ offset:64
	ds_load_b128 v[12:15], v147 /*v403*/ offset:96
	ds_load_b128 v[16:19], v147 /*v403*/ offset:128
	ds_load_b128 v[20:23], v147 /*v403*/ offset:160
	ds_load_b128 v[24:27], v147 /*v403*/ offset:192
	ds_load_b128 v[28:31], v147 /*v403*/ offset:224
	ds_load_b128 v[32:35], v147 /*v403*/ offset:4352
	ds_load_b128 v[36:39], v147 /*v403*/ offset:4384
	ds_load_b128 v[40:43], v147 /*v403*/ offset:4416
	ds_load_b128 v[44:47], v147 /*v403*/ offset:4448
	ds_load_b128 v[48:51], v147 /*v403*/ offset:4480
	ds_load_b128 v[52:55], v147 /*v403*/ offset:4512
	ds_load_b128 v[56:59], v147 /*v403*/ offset:4544
	ds_load_b128 v[60:63], v147 /*v403*/ offset:4576
	tensor_load_to_lds s[28:31], s[20:27]
	s_add_nc_u64 s[30:31], s[46:47], s[76:77]
	s_mov_b32 s29, s78
	s_bitset1_b32 s31, 31
	s_wait_dscnt 0xf
	v_pk_mul_bf16 v128, v145 /*v401*/, v0
	tensor_load_to_lds s[28:31], s[36:43]
	v_and_or_b32 v0, v141 /*v397*/, 7, v143 /*v399*/
	v_pk_mul_bf16 v130, v145 /*v401*/, v2
	v_pk_mul_bf16 v129, v145 /*v401*/, v1
	s_set_vgpr_msb 0x1104
	v_dual_add_nc_u32 v1, s99, v138 /*v394*/ :: v_dual_add_nc_u32 v2, s99, v137 /*v393*/
	s_set_vgpr_msb 0x401
	v_mul_u32_u24_e32 v0, 0x120, v0
	s_wait_dscnt 0xe
	v_pk_mul_bf16 v135, v145 /*v401*/, v7
	v_pk_mul_bf16 v134, v145 /*v401*/, v6
	v_pk_mul_bf16 v133, v145 /*v401*/, v5
	v_pk_mul_bf16 v132, v145 /*v401*/, v4
	v_and_or_b32 v0, v146 /*v402*/, 16, v0
	v_pk_mul_bf16 v131, v145 /*v401*/, v3
	s_wait_dscnt 0xc
	v_pk_mul_bf16 v143, v145 /*v401*/, v15
	v_pk_mul_bf16 v142, v145 /*v401*/, v14
	v_pk_mul_bf16 v141, v145 /*v401*/, v13
	v_pk_mul_bf16 v140, v145 /*v401*/, v12
	v_pk_mul_bf16 v139, v145 /*v401*/, v11
	v_pk_mul_bf16 v138, v145 /*v401*/, v10
	v_pk_mul_bf16 v137, v145 /*v401*/, v9
	v_pk_mul_bf16 v136, v145 /*v401*/, v8
	s_wait_dscnt 0xa
	v_pk_mul_bf16 v151, v145 /*v401*/, v23
	v_pk_mul_bf16 v150, v145 /*v401*/, v22
	v_pk_mul_bf16 v149, v145 /*v401*/, v21
	v_pk_mul_bf16 v148, v145 /*v401*/, v20
	v_pk_mul_bf16 v147, v145 /*v401*/, v19
	v_pk_mul_bf16 v146, v145 /*v401*/, v18
	v_pk_mul_bf16 v145, v145 /*v401*/, v17
	v_pk_mul_bf16 v144, v145 /*v401*/, v16
	s_wait_dscnt 0x8
	v_pk_mul_bf16 v159, v145 /*v401*/, v31
	v_pk_mul_bf16 v158, v145 /*v401*/, v30
	v_pk_mul_bf16 v157, v145 /*v401*/, v29
	v_pk_mul_bf16 v156, v145 /*v401*/, v28
	v_pk_mul_bf16 v155, v145 /*v401*/, v27
	v_pk_mul_bf16 v154, v145 /*v401*/, v26
	v_pk_mul_bf16 v153, v145 /*v401*/, v25
	v_pk_mul_bf16 v152, v145 /*v401*/, v24
	s_wait_dscnt 0x6
	v_pk_mul_bf16 v167, v145 /*v401*/, v39
	v_pk_mul_bf16 v166, v145 /*v401*/, v38
	v_pk_mul_bf16 v165, v145 /*v401*/, v37
	v_pk_mul_bf16 v164, v145 /*v401*/, v36
	v_pk_mul_bf16 v163, v145 /*v401*/, v35
	v_pk_mul_bf16 v162, v145 /*v401*/, v34
	v_pk_mul_bf16 v161, v145 /*v401*/, v33
	v_pk_mul_bf16 v160, v145 /*v401*/, v32
	s_wait_dscnt 0x4
	v_pk_mul_bf16 v175, v145 /*v401*/, v47
	v_pk_mul_bf16 v174, v145 /*v401*/, v46
	v_pk_mul_bf16 v173, v145 /*v401*/, v45
	v_pk_mul_bf16 v172, v145 /*v401*/, v44
	v_pk_mul_bf16 v171, v145 /*v401*/, v43
	v_pk_mul_bf16 v170, v145 /*v401*/, v42
	v_pk_mul_bf16 v169, v145 /*v401*/, v41
	v_pk_mul_bf16 v168, v145 /*v401*/, v40
	s_wait_dscnt 0x2
	v_pk_mul_bf16 v183, v145 /*v401*/, v55
	v_pk_mul_bf16 v182, v145 /*v401*/, v54
	v_pk_mul_bf16 v181, v145 /*v401*/, v53
	v_pk_mul_bf16 v180, v145 /*v401*/, v52
	v_pk_mul_bf16 v179, v145 /*v401*/, v51
	v_pk_mul_bf16 v178, v145 /*v401*/, v50
	v_pk_mul_bf16 v177, v145 /*v401*/, v49
	v_pk_mul_bf16 v176, v145 /*v401*/, v48
	s_wait_dscnt 0x0
	v_pk_mul_bf16 v191, v145 /*v401*/, v63
	v_pk_mul_bf16 v190, v145 /*v401*/, v62
	v_pk_mul_bf16 v189, v145 /*v401*/, v61
	v_pk_mul_bf16 v188, v145 /*v401*/, v60
	v_pk_mul_bf16 v187, v145 /*v401*/, v59
	v_pk_mul_bf16 v186, v145 /*v401*/, v58
	v_pk_mul_bf16 v185, v145 /*v401*/, v57
	v_pk_mul_bf16 v184, v145 /*v401*/, v56
	s_set_vgpr_msb 0x140
	v_or_b32_e32 v145 /*v401*/, 0x10000, v0
	s_wait_alu depctr_vm_vsrc(0)
	v_or_b32_e32 v153 /*v409*/, 0x30000, v0
	v_dual_add_nc_u32 v149 /*v405*/, s17, v1 :: v_dual_add_nc_u32 v150 /*v406*/, s17, v2
	v_subrev_nc_u32_e32 v146 /*v402*/, s16, v1
	v_subrev_nc_u32_e32 v147 /*v403*/, s16, v2
	s_set_vgpr_msb 0x4000
	v_mov_b32_e32 v0, 0
	s_wait_tensorcnt 0x0
	s_set_vgpr_msb 0x41
	s_barrier_signal -1
	s_barrier_wait -1
	s_wait_kmcnt 0x0
	global_load_b32 v135 /*v391*/, v139 /*v395*/, s[70:71] scale_offset
	s_set_vgpr_msb 0x4100
	s_cbranch_scc1 .LBB0_43
	v_dual_mov_b32 v1, v0 :: v_dual_mov_b32 v2, v0
	v_dual_mov_b32 v3, v0 :: v_dual_mov_b32 v4, v0
	v_dual_mov_b32 v5, v0 :: v_dual_mov_b32 v6, v0
	v_mov_b32_e32 v7, v0
	s_set_vgpr_msb 0x41
	v_dual_mov_b32 v64 /*v320*/, 1.0 :: v_dual_mov_b32 v148 /*v404*/, v142 /*v398*/
	s_set_vgpr_msb 0x4100
	v_mov_b64_e32 v[12:13], v[4:5]
	v_mov_b64_e32 v[10:11], v[2:3]
	v_mov_b64_e32 v[14:15], v[6:7]
	v_mov_b64_e32 v[8:9], v[0:1]
	v_mov_b64_e32 v[22:23], v[6:7]
	v_mov_b64_e32 v[20:21], v[4:5]
	v_mov_b64_e32 v[18:19], v[2:3]
	v_mov_b64_e32 v[16:17], v[0:1]
	v_mov_b64_e32 v[30:31], v[6:7]
	v_mov_b64_e32 v[28:29], v[4:5]
	v_mov_b64_e32 v[26:27], v[2:3]
	v_mov_b64_e32 v[24:25], v[0:1]
	v_mov_b64_e32 v[38:39], v[6:7]
	v_mov_b64_e32 v[36:37], v[4:5]
	v_mov_b64_e32 v[34:35], v[2:3]
	v_mov_b64_e32 v[32:33], v[0:1]
	v_mov_b64_e32 v[46:47], v[6:7]
	v_mov_b64_e32 v[44:45], v[4:5]
	v_mov_b64_e32 v[42:43], v[2:3]
	v_mov_b64_e32 v[40:41], v[0:1]
	v_mov_b64_e32 v[54:55], v[6:7]
	v_mov_b64_e32 v[52:53], v[4:5]
	v_mov_b64_e32 v[50:51], v[2:3]
	v_mov_b64_e32 v[48:49], v[0:1]
	v_mov_b64_e32 v[62:63], v[6:7]
	v_mov_b64_e32 v[60:61], v[4:5]
	v_mov_b64_e32 v[58:59], v[2:3]
	v_mov_b64_e32 v[56:57], v[0:1]
	v_mov_b64_e32 v[70:71], v[6:7]
	v_mov_b64_e32 v[68:69], v[4:5]
	v_mov_b64_e32 v[66:67], v[2:3]
	v_mov_b64_e32 v[64:65], v[0:1]
	v_mov_b64_e32 v[78:79], v[6:7]
	v_mov_b64_e32 v[76:77], v[4:5]
	v_mov_b64_e32 v[74:75], v[2:3]
	v_mov_b64_e32 v[72:73], v[0:1]
	v_mov_b64_e32 v[86:87], v[6:7]
	v_mov_b64_e32 v[84:85], v[4:5]
	v_mov_b64_e32 v[82:83], v[2:3]
	v_mov_b64_e32 v[80:81], v[0:1]
	v_mov_b64_e32 v[94:95], v[6:7]
	v_mov_b64_e32 v[92:93], v[4:5]
	v_mov_b64_e32 v[90:91], v[2:3]
	v_mov_b64_e32 v[88:89], v[0:1]
	v_mov_b64_e32 v[102:103], v[6:7]
	v_mov_b64_e32 v[100:101], v[4:5]
	v_mov_b64_e32 v[98:99], v[2:3]
	v_mov_b64_e32 v[96:97], v[0:1]
	v_mov_b64_e32 v[110:111], v[6:7]
	v_mov_b64_e32 v[108:109], v[4:5]
	v_mov_b64_e32 v[106:107], v[2:3]
	v_mov_b64_e32 v[104:105], v[0:1]
	v_mov_b64_e32 v[118:119], v[6:7]
	v_mov_b64_e32 v[116:117], v[4:5]
	v_mov_b64_e32 v[114:115], v[2:3]
	v_mov_b64_e32 v[112:113], v[0:1]
	v_mov_b64_e32 v[126:127], v[6:7]
	v_mov_b64_e32 v[124:125], v[4:5]
	v_mov_b64_e32 v[122:123], v[2:3]
	v_mov_b64_e32 v[120:121], v[0:1]
	s_set_vgpr_msb 1
	v_dual_mov_b32 v200, v153 /*v409*/ :: v_dual_mov_b32 v201, v144 /*v400*/
	s_set_vgpr_msb 0x141
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v130 /*v386*/, v135 /*v391*/ :: v_dual_mov_b32 v65 /*v321*/, v64 /*v320*/
	s_mov_b32 s61, s27
	s_sub_co_i32 s17, 1, s60
	s_mov_b32 s49, 0x76543210
	s_mov_b32 s56, 0x3fb8aa3b
	s_mov_b64 s[58:59], s[60:61]
	s_set_vgpr_msb 0x4100
	s_branch .LBB0_37
.LBB0_36:
	s_set_vgpr_msb 0x45
	v_cvt_pk_bf16_f32 v163 /*v419*/, v116 /*v372*/, v122 /*v378*/
	v_cvt_pk_bf16_f32 v162 /*v418*/, v108 /*v364*/, v114 /*v370*/
	v_cvt_pk_bf16_f32 v161 /*v417*/, v100 /*v356*/, v106 /*v362*/
	v_cvt_pk_bf16_f32 v160 /*v416*/, v92 /*v348*/, v98 /*v354*/
	v_cvt_pk_bf16_f32 v159 /*v415*/, v84 /*v340*/, v90 /*v346*/
	v_cvt_pk_bf16_f32 v158 /*v414*/, v76 /*v332*/, v82 /*v338*/
	v_cvt_pk_bf16_f32 v157 /*v413*/, v70 /*v326*/, v74 /*v330*/
	v_cvt_pk_bf16_f32 v156 /*v412*/, v66 /*v322*/, v68 /*v324*/
	v_cvt_pk_bf16_f32 v171 /*v427*/, v117 /*v373*/, v123 /*v379*/
	v_cvt_pk_bf16_f32 v170 /*v426*/, v109 /*v365*/, v115 /*v371*/
	v_cvt_pk_bf16_f32 v169 /*v425*/, v101 /*v357*/, v107 /*v363*/
	v_cvt_pk_bf16_f32 v168 /*v424*/, v93 /*v349*/, v99 /*v355*/
	v_cvt_pk_bf16_f32 v167 /*v423*/, v85 /*v341*/, v91 /*v347*/
	v_cvt_pk_bf16_f32 v166 /*v422*/, v77 /*v333*/, v83 /*v339*/
	v_cvt_pk_bf16_f32 v165 /*v421*/, v71 /*v327*/, v75 /*v331*/
	v_cvt_pk_bf16_f32 v164 /*v420*/, v67 /*v323*/, v69 /*v325*/
	s_set_vgpr_msb 0x4505
	s_wait_dscnt 0x1d
	v_wmma_f32_16x16x32_bf16 v[120:127], v[56:63] /*v[312:319]*/, v[156:163] /*v[412:419]*/, v[120:127]
	s_set_vgpr_msb 0x545
	v_cvt_pk_bf16_f32 v101 /*v357*/, v127 /*v383*/, v129 /*v385*/
	v_cvt_pk_bf16_f32 v100 /*v356*/, v121 /*v377*/, v125 /*v381*/
	v_cvt_pk_bf16_f32 v99 /*v355*/, v113 /*v369*/, v119 /*v375*/
	v_cvt_pk_bf16_f32 v98 /*v354*/, v105 /*v361*/, v111 /*v367*/
	v_cvt_pk_bf16_f32 v97 /*v353*/, v97 /*v353*/, v103 /*v359*/
	s_add_nc_u64 s[58:59], s[58:59], 1
	s_set_vgpr_msb 0x4505
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[56:63], v[56:63] /*v[312:319]*/, v[164:171] /*v[420:427]*/, v[56:63]
	v_cmp_lt_u64_e64 s2, s[58:59], s[54:55]
	s_and_b32 vcc_lo, exec_lo, s2
	v_wmma_f32_16x16x32_bf16 v[112:119], v[40:47] /*v[296:303]*/, v[156:163] /*v[412:419]*/, v[112:119]
	v_nop
	v_nop
	s_set_vgpr_msb 0x545
	v_cvt_pk_bf16_f32 v63 /*v319*/, v126 /*v382*/, v128 /*v384*/
	v_cvt_pk_bf16_f32 v62 /*v318*/, v120 /*v376*/, v124 /*v380*/
	v_cvt_pk_bf16_f32 v61 /*v317*/, v112 /*v368*/, v118 /*v374*/
	v_cvt_pk_bf16_f32 v60 /*v316*/, v104 /*v360*/, v110 /*v366*/
	v_cvt_pk_bf16_f32 v59 /*v315*/, v96 /*v352*/, v102 /*v358*/
	s_set_vgpr_msb 0x4505
	v_wmma_f32_16x16x32_bf16 v[48:55], v[40:47] /*v[296:303]*/, v[164:171] /*v[420:427]*/, v[48:55]
	s_set_vgpr_msb 0x545
	v_cvt_pk_bf16_f32 v58 /*v314*/, v88 /*v344*/, v94 /*v350*/
	v_cvt_pk_bf16_f32 v57 /*v313*/, v80 /*v336*/, v86 /*v342*/
	v_cvt_pk_bf16_f32 v56 /*v312*/, v72 /*v328*/, v78 /*v334*/
	v_cvt_pk_bf16_f32 v96 /*v352*/, v89 /*v345*/, v95 /*v351*/
	v_cvt_pk_bf16_f32 v95 /*v351*/, v81 /*v337*/, v87 /*v343*/
	v_cvt_pk_bf16_f32 v94 /*v350*/, v73 /*v329*/, v79 /*v335*/
	s_set_vgpr_msb 0x4505
	v_wmma_f32_16x16x32_bf16 v[104:111], v[24:31] /*v[280:287]*/, v[156:163] /*v[412:419]*/, v[104:111]
	v_wmma_f32_16x16x32_bf16 v[40:47], v[24:31] /*v[280:287]*/, v[164:171] /*v[420:427]*/, v[40:47]
	v_wmma_f32_16x16x32_bf16 v[96:103], v[8:15] /*v[264:271]*/, v[156:163] /*v[412:419]*/, v[96:103]
	v_wmma_f32_16x16x32_bf16 v[32:39], v[8:15] /*v[264:271]*/, v[164:171] /*v[420:427]*/, v[32:39]
	s_set_vgpr_msb 0x504
	v_wmma_f32_16x16x32_bf16 v[88:95], v[248:255], v[156:163] /*v[412:419]*/, v[88:95]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[248:255], v[164:171] /*v[420:427]*/, v[24:31]
	v_wmma_f32_16x16x32_bf16 v[80:87], v[232:239], v[156:163] /*v[412:419]*/, v[80:87]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[232:239], v[164:171] /*v[420:427]*/, v[16:23]
	v_wmma_f32_16x16x32_bf16 v[72:79], v[216:223], v[156:163] /*v[412:419]*/, v[72:79]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[216:223], v[164:171] /*v[420:427]*/, v[8:15]
	v_wmma_f32_16x16x32_bf16 v[64:71], v[200:207], v[156:163] /*v[412:419]*/, v[64:71]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[200:207], v[164:171] /*v[420:427]*/, v[0:7]
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x415
	v_pk_fma_f32 v[200:201], v[134:135] /*v[390:391]*/, v[64:65] /*v[320:321]*/, v[132:133] /*v[388:389]*/
	s_set_vgpr_msb 0x1541
	v_mov_b32_e32 v135 /*v391*/, v151 /*v407*/
	v_pk_add_f32 v[64:65] /*v[320:321]*/, v[130:131] /*v[386:387]*/, v[200:201]
	s_set_vgpr_msb 0x4105
	v_wmma_f32_16x16x32_bf16 v[120:127], v[48:55] /*v[304:311]*/, v[56:63] /*v[312:319]*/, v[120:127]
	v_dual_mov_b32 v200, v153 /*v409*/ :: v_dual_mov_b32 v201, v144 /*v400*/
	s_set_vgpr_msb 0x541
	v_mov_b32_e32 v130 /*v386*/, v154 /*v410*/
	s_set_vgpr_msb 0x4105
	v_wmma_f32_16x16x32_bf16 v[56:63], v[48:55] /*v[304:311]*/, v[94:101] /*v[350:357]*/, v[56:63]
	v_wmma_f32_16x16x32_bf16 v[112:119], v[32:39] /*v[288:295]*/, v[56:63] /*v[312:319]*/, v[112:119]
	v_wmma_f32_16x16x32_bf16 v[48:55], v[32:39] /*v[288:295]*/, v[94:101] /*v[350:357]*/, v[48:55]
	v_wmma_f32_16x16x32_bf16 v[104:111], v[16:23] /*v[272:279]*/, v[56:63] /*v[312:319]*/, v[104:111]
	v_wmma_f32_16x16x32_bf16 v[40:47], v[16:23] /*v[272:279]*/, v[94:101] /*v[350:357]*/, v[40:47]
	v_wmma_f32_16x16x32_bf16 v[96:103], v[0:7] /*v[256:263]*/, v[56:63] /*v[312:319]*/, v[96:103]
	v_wmma_f32_16x16x32_bf16 v[32:39], v[0:7] /*v[256:263]*/, v[94:101] /*v[350:357]*/, v[32:39]
	s_set_vgpr_msb 0x504
	v_wmma_f32_16x16x32_bf16 v[88:95], v[240:247], v[56:63] /*v[312:319]*/, v[88:95]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[240:247], v[94:101] /*v[350:357]*/, v[24:31]
	v_wmma_f32_16x16x32_bf16 v[80:87], v[224:231], v[56:63] /*v[312:319]*/, v[80:87]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[224:231], v[94:101] /*v[350:357]*/, v[16:23]
	v_wmma_f32_16x16x32_bf16 v[72:79], v[208:215], v[56:63] /*v[312:319]*/, v[72:79]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[208:215], v[94:101] /*v[350:357]*/, v[8:15]
	v_wmma_f32_16x16x32_bf16 v[64:71], v[192:199], v[56:63] /*v[312:319]*/, v[64:71]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[192:199], v[94:101] /*v[350:357]*/, v[0:7]
	s_set_vgpr_msb 0x400
	s_cbranch_vccz .LBB0_44
.LBB0_37:
	s_wait_alu depctr_vm_vsrc(0)
	s_set_vgpr_msb 0x41
	v_dual_mov_b32 v144 /*v400*/, v148 /*v404*/ :: v_dual_mov_b32 v153 /*v409*/, v145 /*v401*/
	s_set_vgpr_msb 0x4140
	v_dual_mov_b32 v148 /*v404*/, v201 :: v_dual_mov_b32 v145 /*v401*/, v200
	s_add_co_i32 s2, s58, 1
	s_wait_tensorcnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x4041
	ds_load_b128 v[56:59] /*v[312:315]*/, v144 /*v400*/
	ds_load_b128 v[60:63] /*v[316:319]*/, v144 /*v400*/ offset:32
	ds_load_b128 v[48:51] /*v[304:307]*/, v144 /*v400*/ offset:64
	ds_load_b128 v[52:55] /*v[308:311]*/, v144 /*v400*/ offset:96
	ds_load_b128 v[40:43] /*v[296:299]*/, v144 /*v400*/ offset:128
	ds_load_b128 v[44:47] /*v[300:303]*/, v144 /*v400*/ offset:160
	ds_load_b128 v[32:35] /*v[288:291]*/, v144 /*v400*/ offset:192
	ds_load_b128 v[36:39] /*v[292:295]*/, v144 /*v400*/ offset:224
	ds_load_b128 v[24:27] /*v[280:283]*/, v144 /*v400*/ offset:4352
	ds_load_b128 v[28:31] /*v[284:287]*/, v144 /*v400*/ offset:4384
	ds_load_b128 v[16:19] /*v[272:275]*/, v144 /*v400*/ offset:4416
	ds_load_b128 v[20:23] /*v[276:279]*/, v144 /*v400*/ offset:4448
	ds_load_b128 v[8:11] /*v[264:267]*/, v144 /*v400*/ offset:4480
	ds_load_b128 v[12:15] /*v[268:271]*/, v144 /*v400*/ offset:4512
	ds_load_b128 v[0:3] /*v[256:259]*/, v144 /*v400*/ offset:4544
	ds_load_b128 v[4:7] /*v[260:263]*/, v144 /*v400*/ offset:4576
	s_set_vgpr_msb 0x4101
	ds_load_b128 v[248:251], v144 /*v400*/ offset:8704
	ds_load_b128 v[252:255], v144 /*v400*/ offset:8736
	ds_load_b128 v[240:243], v144 /*v400*/ offset:8768
	ds_load_b128 v[244:247], v144 /*v400*/ offset:8800
	ds_load_b128 v[232:235], v144 /*v400*/ offset:8832
	ds_load_b128 v[236:239], v144 /*v400*/ offset:8864
	ds_load_b128 v[224:227], v144 /*v400*/ offset:8896
	ds_load_b128 v[228:231], v144 /*v400*/ offset:8928
	ds_load_b128 v[216:219], v144 /*v400*/ offset:13056
	ds_load_b128 v[220:223], v144 /*v400*/ offset:13088
	ds_load_b128 v[208:211], v144 /*v400*/ offset:13120
	ds_load_b128 v[212:215], v144 /*v400*/ offset:13152
	ds_load_b128 v[200:203], v144 /*v400*/ offset:13184
	ds_load_b128 v[204:207], v144 /*v400*/ offset:13216
	ds_load_b128 v[192:195], v144 /*v400*/ offset:13248
	ds_load_b128 v[196:199], v144 /*v400*/ offset:13280
	s_cmp_ge_i32 s2, s10
	s_set_vgpr_msb 0x100
	s_cbranch_scc1 .LBB0_39
	s_lshl_b32 s4, s2, 6
	s_add_co_i32 s6, s58, s17
	s_sub_co_i32 s29, s9, s4
	s_add_co_i32 s2, s4, s97
	s_set_vgpr_msb 0x41
	v_med3_i32 v66 /*v322*/, s29, 0, 64
	s_lshr_b32 s22, s6, 31
	s_ashr_i32 s3, s2, 31
	s_add_co_i32 s22, s6, s22
	s_mul_u64 s[4:5], s[2:3], s[64:65]
	v_readfirstlane_b32 s29, v66 /*v322*/
	s_mul_u64 s[2:3], s[2:3], s[62:63]
	s_and_b32 s22, s22, 0x7ffe
	s_lshl_b64 s[2:3], s[2:3], 1
	s_sub_co_i32 s6, s6, s22
	s_add_nc_u64 s[2:3], s[52:53], s[2:3]
	s_sub_co_i32 s22, s29, s80
	s_add_nc_u64 s[30:31], s[44:45], s[2:3]
	s_max_i32 s2, s22, 0
	s_lshl_b64 s[4:5], s[4:5], 1
	s_lshl_b32 s6, s6, 17
	s_lshl_b32 s2, s2, 16
	s_add_nc_u64 s[4:5], s[50:51], s[4:5]
	s_or_b32 s29, s81, s6
	s_bitset1_b32 s31, 31
	s_or_b32 s22, s2, 0x7fff
	s_mov_b32 s37, s21
	tensor_load_to_lds s[28:31], s[20:27]
	s_add_nc_u64 s[30:31], s[46:47], s[4:5]
	s_or_b32 s29, s78, s6
	s_bitset1_b32 s31, 31
	s_mov_b32 s38, s22
	s_mov_b32 s39, s23
	s_mov_b32 s40, s24
	s_mov_b32 s43, s27
	tensor_load_to_lds s[28:31], s[36:43]
	s_set_vgpr_msb 0x4100
.LBB0_39:
	s_set_vgpr_msb 0x41
	s_wait_dscnt 0x1e
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[322:329]*/, v[56:63] /*v[312:319]*/, v[128:135], 0
	v_wmma_f32_16x16x32_bf16 v[74:81] /*v[330:337]*/, v[56:63] /*v[312:319]*/, v[160:167], 0
	s_wait_dscnt 0x16
	v_wmma_f32_16x16x32_bf16 v[82:89] /*v[338:345]*/, v[24:31] /*v[280:287]*/, v[128:135], 0
	v_wmma_f32_16x16x32_bf16 v[90:97] /*v[346:353]*/, v[24:31] /*v[280:287]*/, v[160:167], 0
	s_set_vgpr_msb 0x4140
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x32_bf16 v[98:105] /*v[354:361]*/, v[248:255], v[128:135], 0
	v_wmma_f32_16x16x32_bf16 v[106:113] /*v[362:369]*/, v[248:255], v[160:167], 0
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x32_bf16 v[114:121] /*v[370:377]*/, v[216:223], v[128:135], 0
	v_wmma_f32_16x16x32_bf16 v[122:129] /*v[378:385]*/, v[216:223], v[160:167], 0
	s_set_vgpr_msb 0x4051
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[322:329]*/, v[48:55] /*v[304:311]*/, v[136:143], v[66:73] /*v[322:329]*/
	v_wmma_f32_16x16x32_bf16 v[74:81] /*v[330:337]*/, v[48:55] /*v[304:311]*/, v[168:175], v[74:81] /*v[330:337]*/
	v_wmma_f32_16x16x32_bf16 v[82:89] /*v[338:345]*/, v[16:23] /*v[272:279]*/, v[136:143], v[82:89] /*v[338:345]*/
	v_wmma_f32_16x16x32_bf16 v[90:97] /*v[346:353]*/, v[16:23] /*v[272:279]*/, v[168:175], v[90:97] /*v[346:353]*/
	s_set_vgpr_msb 0x5150
	v_wmma_f32_16x16x32_bf16 v[98:105] /*v[354:361]*/, v[240:247], v[136:143], v[98:105] /*v[354:361]*/
	v_wmma_f32_16x16x32_bf16 v[106:113] /*v[362:369]*/, v[240:247], v[168:175], v[106:113] /*v[362:369]*/
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x32_bf16 v[114:121] /*v[370:377]*/, v[208:215], v[136:143], v[114:121] /*v[370:377]*/
	v_wmma_f32_16x16x32_bf16 v[122:129] /*v[378:385]*/, v[208:215], v[168:175], v[122:129] /*v[378:385]*/
	s_set_vgpr_msb 0x5051
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[322:329]*/, v[40:47] /*v[296:303]*/, v[144:151], v[66:73] /*v[322:329]*/
	v_wmma_f32_16x16x32_bf16 v[74:81] /*v[330:337]*/, v[40:47] /*v[296:303]*/, v[176:183], v[74:81] /*v[330:337]*/
	v_wmma_f32_16x16x32_bf16 v[82:89] /*v[338:345]*/, v[8:15] /*v[264:271]*/, v[144:151], v[82:89] /*v[338:345]*/
	v_wmma_f32_16x16x32_bf16 v[90:97] /*v[346:353]*/, v[8:15] /*v[264:271]*/, v[176:183], v[90:97] /*v[346:353]*/
	s_set_vgpr_msb 0x5150
	v_wmma_f32_16x16x32_bf16 v[98:105] /*v[354:361]*/, v[232:239], v[144:151], v[98:105] /*v[354:361]*/
	v_wmma_f32_16x16x32_bf16 v[106:113] /*v[362:369]*/, v[232:239], v[176:183], v[106:113] /*v[362:369]*/
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[114:121] /*v[370:377]*/, v[200:207], v[144:151], v[114:121] /*v[370:377]*/
	v_wmma_f32_16x16x32_bf16 v[122:129] /*v[378:385]*/, v[200:207], v[176:183], v[122:129] /*v[378:385]*/
	s_set_vgpr_msb 0x5051
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[322:329]*/, v[32:39] /*v[288:295]*/, v[152:159], v[66:73] /*v[322:329]*/
	v_wmma_f32_16x16x32_bf16 v[74:81] /*v[330:337]*/, v[32:39] /*v[288:295]*/, v[184:191], v[74:81] /*v[330:337]*/
	v_wmma_f32_16x16x32_bf16 v[82:89] /*v[338:345]*/, v[0:7] /*v[256:263]*/, v[152:159], v[82:89] /*v[338:345]*/
	v_wmma_f32_16x16x32_bf16 v[90:97] /*v[346:353]*/, v[0:7] /*v[256:263]*/, v[184:191], v[90:97] /*v[346:353]*/
	s_set_vgpr_msb 0x5150
	v_wmma_f32_16x16x32_bf16 v[98:105] /*v[354:361]*/, v[224:231], v[152:159], v[98:105] /*v[354:361]*/
	v_wmma_f32_16x16x32_bf16 v[106:113] /*v[362:369]*/, v[224:231], v[184:191], v[106:113] /*v[362:369]*/
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[114:121] /*v[370:377]*/, v[192:199], v[152:159], v[114:121] /*v[370:377]*/
	v_wmma_f32_16x16x32_bf16 v[122:129] /*v[378:385]*/, v[192:199], v[184:191], v[122:129] /*v[378:385]*/
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x5041
	ds_load_tr16_b128 v[56:59] /*v[312:315]*/, v153 /*v409*/
	ds_load_tr16_b128 v[40:43] /*v[296:299]*/, v153 /*v409*/ offset:32
	ds_load_tr16_b128 v[60:63] /*v[316:319]*/, v153 /*v409*/ offset:4608
	ds_load_tr16_b128 v[44:47] /*v[300:303]*/, v153 /*v409*/ offset:4640
	ds_load_tr16_b128 v[48:51] /*v[304:307]*/, v153 /*v409*/ offset:9216
	ds_load_tr16_b128 v[32:35] /*v[288:291]*/, v153 /*v409*/ offset:9248
	ds_load_tr16_b128 v[52:55] /*v[308:311]*/, v153 /*v409*/ offset:13824
	ds_load_tr16_b128 v[36:39] /*v[292:295]*/, v153 /*v409*/ offset:13856
	ds_load_tr16_b128 v[24:27] /*v[280:283]*/, v153 /*v409*/ offset:64
	ds_load_tr16_b128 v[8:11] /*v[264:267]*/, v153 /*v409*/ offset:96
	ds_load_tr16_b128 v[28:31] /*v[284:287]*/, v153 /*v409*/ offset:4672
	ds_load_tr16_b128 v[12:15] /*v[268:271]*/, v153 /*v409*/ offset:4704
	ds_load_tr16_b128 v[16:19] /*v[272:275]*/, v153 /*v409*/ offset:9280
	ds_load_tr16_b128 v[0:3] /*v[256:259]*/, v153 /*v409*/ offset:9312
	ds_load_tr16_b128 v[20:23] /*v[276:279]*/, v153 /*v409*/ offset:13888
	ds_load_tr16_b128 v[4:7] /*v[260:263]*/, v153 /*v409*/ offset:13920
	s_set_vgpr_msb 0x4101
	ds_load_tr16_b128 v[248:251], v153 /*v409*/ offset:128
	ds_load_tr16_b128 v[232:235], v153 /*v409*/ offset:160
	ds_load_tr16_b128 v[252:255], v153 /*v409*/ offset:4736
	ds_load_tr16_b128 v[236:239], v153 /*v409*/ offset:4768
	ds_load_tr16_b128 v[240:243], v153 /*v409*/ offset:9344
	ds_load_tr16_b128 v[224:227], v153 /*v409*/ offset:9376
	ds_load_tr16_b128 v[244:247], v153 /*v409*/ offset:13952
	ds_load_tr16_b128 v[228:231], v153 /*v409*/ offset:13984
	ds_load_tr16_b128 v[216:219], v153 /*v409*/ offset:192
	ds_load_tr16_b128 v[200:203], v153 /*v409*/ offset:224
	ds_load_tr16_b128 v[220:223], v153 /*v409*/ offset:4800
	ds_load_tr16_b128 v[204:207], v153 /*v409*/ offset:4832
	ds_load_tr16_b128 v[208:211], v153 /*v409*/ offset:9408
	ds_load_tr16_b128 v[192:195], v153 /*v409*/ offset:9440
	ds_load_tr16_b128 v[212:215], v153 /*v409*/ offset:14016
	ds_load_tr16_b128 v[196:199], v153 /*v409*/ offset:14048
	s_set_vgpr_msb 0x155
	v_lshl_or_b32 v131 /*v387*/, s58, 6, v143 /*v399*/
	v_dual_add_nc_u32 v172 /*v428*/, 16, v131 /*v387*/ :: v_dual_bitop2_b32 v134 /*v390*/, 1, v131 /*v387*/ bitop3:0x54
	v_cmp_gt_i32_e32 vcc_lo, v131 /*v387*/, v149 /*v405*/
	v_cmp_lt_i32_e64 s2, v131 /*v387*/, v146 /*v402*/
	v_cmp_ge_i32_e64 s3, v131 /*v387*/, v149 /*v405*/
	v_dual_add_nc_u32 v173 /*v429*/, 17, v131 /*v387*/ :: v_dual_bitop2_b32 v151 /*v407*/, 2, v131 /*v387*/ bitop3:0x54
	v_cmp_lt_i32_e64 s4, v134 /*v390*/, v146 /*v402*/
	v_dual_add_nc_u32 v174 /*v430*/, 18, v131 /*v387*/ :: v_dual_bitop2_b32 v152 /*v408*/, 3, v131 /*v387*/ bitop3:0x54
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v151 /*v407*/, v149 /*v405*/
	v_cndmask_b32_e64 v66 /*v322*/, v66 /*v322*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v151 /*v407*/, v146 /*v402*/
	s_or_b32 s3, s4, s3
	v_dual_add_nc_u32 v175 /*v431*/, 19, v131 /*v387*/ :: v_dual_bitop2_b32 v154 /*v410*/, 4, v131 /*v387*/ bitop3:0x54
	v_cmp_gt_i32_e64 s5, v152 /*v408*/, v149 /*v405*/
	v_cndmask_b32_e64 v67 /*v323*/, v67 /*v323*/, 0xff800000, s3
	v_cmp_lt_i32_e64 s3, v152 /*v408*/, v146 /*v402*/
	v_dual_add_nc_u32 v176 /*v432*/, 20, v131 /*v387*/ :: v_dual_bitop2_b32 v155 /*v411*/, 5, v131 /*v387*/ bitop3:0x54
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v154 /*v410*/, v149 /*v405*/
	v_cndmask_b32_e64 v68 /*v324*/, v68 /*v324*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v154 /*v410*/, v146 /*v402*/
	s_or_b32 s3, s3, s5
	v_or_b32_e32 v169 /*v425*/, 6, v131 /*v387*/
	v_cndmask_b32_e64 v69 /*v325*/, v69 /*v325*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v155 /*v411*/, v149 /*v405*/
	v_cmp_lt_i32_e64 s4, v155 /*v411*/, v146 /*v402*/
	v_or_b32_e32 v170 /*v426*/, 7, v131 /*v387*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v169 /*v425*/, v149 /*v405*/
	v_cndmask_b32_e64 v70 /*v326*/, v70 /*v326*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v169 /*v425*/, v146 /*v402*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s4, v170 /*v426*/, v146 /*v402*/
	v_cndmask_b32_e64 v71 /*v327*/, v71 /*v327*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v170 /*v426*/, v149 /*v405*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v172 /*v428*/, v149 /*v405*/
	v_cndmask_b32_e64 v72 /*v328*/, v72 /*v328*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v172 /*v428*/, v146 /*v402*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s4, v173 /*v429*/, v146 /*v402*/
	v_cndmask_b32_e64 v73 /*v329*/, v73 /*v329*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v173 /*v429*/, v149 /*v405*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v174 /*v430*/, v149 /*v405*/
	v_cndmask_b32_e64 v82 /*v338*/, v82 /*v338*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v174 /*v430*/, v146 /*v402*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s4, v175 /*v431*/, v146 /*v402*/
	v_cndmask_b32_e64 v83 /*v339*/, v83 /*v339*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v175 /*v431*/, v149 /*v405*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v176 /*v432*/, v149 /*v405*/
	v_cndmask_b32_e64 v132 /*v388*/, v84 /*v340*/, 0xff800000, s2
	v_add_nc_u32_e32 v84 /*v340*/, 21, v131 /*v387*/
	v_cmp_lt_i32_e64 s2, v176 /*v432*/, v146 /*v402*/
	s_or_b32 s3, s4, s3
	v_dual_add_nc_u32 v178 /*v434*/, 23, v131 /*v387*/ :: v_dual_bitop2_b32 v179 /*v435*/, 32, v131 /*v387*/ bitop3:0x54
	v_cndmask_b32_e64 v133 /*v389*/, v85 /*v341*/, 0xff800000, s3
	v_add_nc_u32_e32 v85 /*v341*/, 22, v131 /*v387*/
	v_cmp_gt_i32_e64 s3, v84 /*v340*/, v149 /*v405*/
	v_cmp_lt_i32_e64 s4, v84 /*v340*/, v146 /*v402*/
	s_or_b32 s2, s2, vcc_lo
	v_dual_add_nc_u32 v188 /*v444*/, 48, v131 /*v387*/ :: v_dual_bitop2_b32 v181 /*v437*/, 33, v131 /*v387*/ bitop3:0x54
	v_cndmask_b32_e64 v86 /*v342*/, v86 /*v342*/, 0xff800000, s2
	v_cmp_gt_i32_e32 vcc_lo, v85 /*v341*/, v149 /*v405*/
	v_cmp_lt_i32_e64 s2, v85 /*v341*/, v146 /*v402*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s4, v178 /*v434*/, v146 /*v402*/
	v_cndmask_b32_e64 v87 /*v343*/, v87 /*v343*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v178 /*v434*/, v149 /*v405*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v179 /*v435*/, v149 /*v405*/
	v_cndmask_b32_e64 v88 /*v344*/, v88 /*v344*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v179 /*v435*/, v146 /*v402*/
	s_or_b32 s3, s4, s3
	v_dual_add_nc_u32 v189 /*v445*/, 49, v131 /*v387*/ :: v_dual_bitop2_b32 v182 /*v438*/, 34, v131 /*v387*/ bitop3:0x54
	v_cndmask_b32_e64 v89 /*v345*/, v89 /*v345*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v181 /*v437*/, v149 /*v405*/
	v_cmp_lt_i32_e64 s4, v181 /*v437*/, v146 /*v402*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v182 /*v438*/, v149 /*v405*/
	v_cndmask_b32_e64 v156 /*v412*/, v98 /*v354*/, 0xff800000, s2
	v_dual_add_nc_u32 v190 /*v446*/, 50, v131 /*v387*/ :: v_dual_bitop2_b32 v98 /*v354*/, 35, v131 /*v387*/ bitop3:0x54
	v_cmp_lt_i32_e64 s2, v182 /*v438*/, v146 /*v402*/
	s_or_b32 s3, s4, s3
	v_or_b32_e32 v187 /*v443*/, 39, v131 /*v387*/
	v_cndmask_b32_e64 v157 /*v413*/, v99 /*v355*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v98 /*v354*/, v149 /*v405*/
	v_or_b32_e32 v99 /*v355*/, 36, v131 /*v387*/
	v_cmp_lt_i32_e64 s4, v98 /*v354*/, v146 /*v402*/
	s_or_b32 s2, s2, vcc_lo
	v_add_nc_u32_e32 v193 /*v449*/, 55, v131 /*v387*/
	v_cndmask_b32_e64 v158 /*v414*/, v100 /*v356*/, 0xff800000, s2
	v_or_b32_e32 v100 /*v356*/, 37, v131 /*v387*/
	v_cmp_gt_i32_e32 vcc_lo, v99 /*v355*/, v149 /*v405*/
	v_cmp_lt_i32_e64 s2, v99 /*v355*/, v146 /*v402*/
	s_or_b32 s3, s4, s3
	v_cndmask_b32_e64 v159 /*v415*/, v101 /*v357*/, 0xff800000, s3
	v_or_b32_e32 v101 /*v357*/, 38, v131 /*v387*/
	v_cmp_gt_i32_e64 s3, v100 /*v356*/, v149 /*v405*/
	v_cmp_lt_i32_e64 s4, v100 /*v356*/, v146 /*v402*/
	s_or_b32 s2, s2, vcc_lo
	v_cndmask_b32_e64 v102 /*v358*/, v102 /*v358*/, 0xff800000, s2
	v_cmp_gt_i32_e32 vcc_lo, v101 /*v357*/, v149 /*v405*/
	v_cmp_lt_i32_e64 s2, v101 /*v357*/, v146 /*v402*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s4, v187 /*v443*/, v146 /*v402*/
	v_cndmask_b32_e64 v103 /*v359*/, v103 /*v359*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v187 /*v443*/, v149 /*v405*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v188 /*v444*/, v149 /*v405*/
	v_cndmask_b32_e64 v104 /*v360*/, v104 /*v360*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v188 /*v444*/, v146 /*v402*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s4, v189 /*v445*/, v146 /*v402*/
	v_cndmask_b32_e64 v105 /*v361*/, v105 /*v361*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v189 /*v445*/, v149 /*v405*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v190 /*v446*/, v149 /*v405*/
	v_cndmask_b32_e64 v160 /*v416*/, v114 /*v370*/, 0xff800000, s2
	v_add_nc_u32_e32 v114 /*v370*/, 51, v131 /*v387*/
	v_cmp_lt_i32_e64 s2, v190 /*v446*/, v146 /*v402*/
	s_or_b32 s3, s4, s3
	v_cndmask_b32_e64 v161 /*v417*/, v115 /*v371*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v114 /*v370*/, v149 /*v405*/
	v_cmp_lt_i32_e64 s4, v114 /*v370*/, v146 /*v402*/
	s_or_b32 s2, s2, vcc_lo
	v_add_nc_u32_e32 v115 /*v371*/, 52, v131 /*v387*/
	v_cndmask_b32_e64 v162 /*v418*/, v116 /*v372*/, 0xff800000, s2
	v_add_nc_u32_e32 v116 /*v372*/, 53, v131 /*v387*/
	s_or_b32 s2, s4, s3
	v_cndmask_b32_e64 v163 /*v419*/, v117 /*v373*/, 0xff800000, s2
	v_add_nc_u32_e32 v117 /*v373*/, 54, v131 /*v387*/
	v_cmp_gt_i32_e32 vcc_lo, v115 /*v371*/, v149 /*v405*/
	v_cmp_lt_i32_e64 s2, v115 /*v371*/, v146 /*v402*/
	v_cmp_gt_i32_e64 s3, v116 /*v372*/, v149 /*v405*/
	v_cmp_lt_i32_e64 s4, v116 /*v372*/, v146 /*v402*/
	v_cmp_gt_i32_e64 s5, v117 /*v373*/, v149 /*v405*/
	v_cmp_lt_i32_e64 s6, v117 /*v373*/, v146 /*v402*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v193 /*v449*/, v149 /*v405*/
	v_cndmask_b32_e64 v118 /*v374*/, v118 /*v374*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	v_cmp_gt_i32_e64 s3, v131 /*v387*/, v150 /*v406*/
	v_cndmask_b32_e64 v119 /*v375*/, v119 /*v375*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	v_cmp_lt_i32_e64 s4, v131 /*v387*/, v147 /*v403*/
	v_cndmask_b32_e64 v120 /*v376*/, v120 /*v376*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v193 /*v449*/, v146 /*v402*/
	v_cmp_ge_i32_e64 s5, v131 /*v387*/, v150 /*v406*/
	v_cmp_lt_i32_e64 s6, v134 /*v390*/, v147 /*v403*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v151 /*v407*/, v150 /*v406*/
	v_cndmask_b32_e64 v121 /*v377*/, v121 /*v377*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	v_cmp_gt_i32_e64 s3, v152 /*v408*/, v150 /*v406*/
	v_cndmask_b32_e64 v164 /*v420*/, v74 /*v330*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	v_cmp_lt_i32_e64 s4, v152 /*v408*/, v147 /*v403*/
	v_cndmask_b32_e64 v165 /*v421*/, v75 /*v331*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v151 /*v407*/, v147 /*v403*/
	v_cmp_gt_i32_e64 s5, v154 /*v410*/, v150 /*v406*/
	v_cmp_lt_i32_e64 s6, v154 /*v410*/, v147 /*v403*/
	v_dual_max_num_f32 v74 /*v330*/, v66 /*v322*/, v67 /*v323*/ :: v_dual_max_num_f32 v75 /*v331*/, v164 /*v420*/, v165 /*v421*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v155 /*v411*/, v150 /*v406*/
	v_cndmask_b32_e64 v166 /*v422*/, v76 /*v332*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	v_cmp_gt_i32_e64 s3, v169 /*v425*/, v150 /*v406*/
	v_cndmask_b32_e64 v167 /*v423*/, v77 /*v333*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	v_cmp_lt_i32_e64 s4, v169 /*v425*/, v147 /*v403*/
	v_cndmask_b32_e64 v168 /*v424*/, v78 /*v334*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v155 /*v411*/, v147 /*v403*/
	v_cmp_gt_i32_e64 s5, v170 /*v426*/, v150 /*v406*/
	v_cmp_lt_i32_e64 s6, v170 /*v426*/, v147 /*v403*/
	v_max3_num_f32 v76 /*v332*/, v69 /*v325*/, v70 /*v326*/, v71 /*v327*/
	v_max3_num_f32 v78 /*v334*/, v72 /*v328*/, v73 /*v329*/, v82 /*v338*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v172 /*v428*/, v150 /*v406*/
	v_cndmask_b32_e64 v169 /*v425*/, v79 /*v335*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	v_cmp_gt_i32_e64 s3, v173 /*v429*/, v150 /*v406*/
	v_cndmask_b32_e64 v170 /*v426*/, v80 /*v336*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	v_cmp_lt_i32_e64 s4, v173 /*v429*/, v147 /*v403*/
	v_cndmask_b32_e64 v171 /*v427*/, v81 /*v337*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v172 /*v428*/, v147 /*v403*/
	v_cmp_gt_i32_e64 s5, v174 /*v430*/, v150 /*v406*/
	v_cmp_lt_i32_e64 s6, v174 /*v430*/, v147 /*v403*/
	v_max3_num_f32 v80 /*v336*/, v83 /*v339*/, v132 /*v388*/, v133 /*v389*/
	v_max3_num_f32 v77 /*v333*/, v167 /*v423*/, v168 /*v424*/, v169 /*v425*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v175 /*v431*/, v150 /*v406*/
	v_cndmask_b32_e64 v172 /*v428*/, v90 /*v346*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	v_cmp_gt_i32_e64 s3, v176 /*v432*/, v150 /*v406*/
	v_cndmask_b32_e64 v173 /*v429*/, v91 /*v347*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	v_cmp_lt_i32_e64 s4, v176 /*v432*/, v147 /*v403*/
	v_cndmask_b32_e64 v174 /*v430*/, v92 /*v348*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v175 /*v431*/, v147 /*v403*/
	v_cmp_gt_i32_e64 s5, v84 /*v340*/, v150 /*v406*/
	v_cmp_lt_i32_e64 s6, v84 /*v340*/, v147 /*v403*/
	v_max3_num_f32 v84 /*v340*/, v86 /*v342*/, v87 /*v343*/, v88 /*v344*/
	v_max3_num_f32 v90 /*v346*/, v89 /*v345*/, v156 /*v412*/, v157 /*v413*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v85 /*v341*/, v150 /*v406*/
	v_cndmask_b32_e64 v175 /*v431*/, v93 /*v349*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	v_cmp_gt_i32_e64 s3, v178 /*v434*/, v150 /*v406*/
	v_cndmask_b32_e64 v176 /*v432*/, v94 /*v350*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	v_cmp_lt_i32_e64 s4, v178 /*v434*/, v147 /*v403*/
	v_cndmask_b32_e64 v177 /*v433*/, v95 /*v351*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v85 /*v341*/, v147 /*v403*/
	v_cmp_gt_i32_e64 s5, v179 /*v435*/, v150 /*v406*/
	v_cmp_lt_i32_e64 s6, v179 /*v435*/, v147 /*v403*/
	v_max3_num_f32 v81 /*v337*/, v173 /*v429*/, v174 /*v430*/, v175 /*v431*/
	v_max3_num_f32 v74 /*v330*/, v74 /*v330*/, v68 /*v324*/, v76 /*v332*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v181 /*v437*/, v150 /*v406*/
	v_cndmask_b32_e64 v178 /*v434*/, v96 /*v352*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	v_cmp_gt_i32_e64 s3, v182 /*v438*/, v150 /*v406*/
	v_cndmask_b32_e64 v179 /*v435*/, v97 /*v353*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	v_cmp_lt_i32_e64 s4, v182 /*v438*/, v147 /*v403*/
	v_cndmask_b32_e64 v180 /*v436*/, v106 /*v362*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v181 /*v437*/, v147 /*v403*/
	v_cmp_gt_i32_e64 s5, v98 /*v354*/, v150 /*v406*/
	v_cmp_lt_i32_e64 s6, v98 /*v354*/, v147 /*v403*/
	v_max3_num_f32 v85 /*v341*/, v176 /*v432*/, v177 /*v433*/, v178 /*v434*/
	v_max3_num_f32 v76 /*v332*/, v80 /*v336*/, v84 /*v340*/, v90 /*v346*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v99 /*v355*/, v150 /*v406*/
	v_cndmask_b32_e64 v181 /*v437*/, v107 /*v363*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	v_cmp_gt_i32_e64 s3, v100 /*v356*/, v150 /*v406*/
	v_cndmask_b32_e64 v182 /*v438*/, v108 /*v364*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	v_cmp_lt_i32_e64 s4, v100 /*v356*/, v147 /*v403*/
	v_cndmask_b32_e64 v183 /*v439*/, v109 /*v365*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v99 /*v355*/, v147 /*v403*/
	v_cmp_gt_i32_e64 s5, v101 /*v357*/, v150 /*v406*/
	v_cmp_lt_i32_e64 s6, v101 /*v357*/, v147 /*v403*/
	v_max3_num_f32 v91 /*v347*/, v179 /*v435*/, v180 /*v436*/, v181 /*v437*/
	v_max3_num_f32 v79 /*v335*/, v170 /*v426*/, v171 /*v427*/, v172 /*v428*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v187 /*v443*/, v150 /*v406*/
	v_cndmask_b32_e64 v184 /*v440*/, v110 /*v366*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	v_cmp_gt_i32_e64 s3, v188 /*v444*/, v150 /*v406*/
	v_cndmask_b32_e64 v185 /*v441*/, v111 /*v367*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	v_cmp_lt_i32_e64 s4, v188 /*v444*/, v147 /*v403*/
	v_cndmask_b32_e64 v186 /*v442*/, v112 /*v368*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v187 /*v443*/, v147 /*v403*/
	v_cmp_gt_i32_e64 s5, v189 /*v445*/, v150 /*v406*/
	v_cmp_lt_i32_e64 s6, v189 /*v445*/, v147 /*v403*/
	v_max3_num_f32 v80 /*v336*/, v182 /*v438*/, v183 /*v439*/, v184 /*v440*/
	v_max3_num_f32 v74 /*v330*/, v74 /*v330*/, v78 /*v334*/, v76 /*v332*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v190 /*v446*/, v150 /*v406*/
	v_cndmask_b32_e64 v187 /*v443*/, v113 /*v369*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	v_cmp_gt_i32_e64 s3, v114 /*v370*/, v150 /*v406*/
	v_cndmask_b32_e64 v188 /*v444*/, v122 /*v378*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	v_cmp_lt_i32_e64 s4, v114 /*v370*/, v147 /*v403*/
	v_cndmask_b32_e64 v189 /*v445*/, v123 /*v379*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v190 /*v446*/, v147 /*v403*/
	v_cmp_gt_i32_e64 s5, v115 /*v371*/, v150 /*v406*/
	v_cmp_lt_i32_e64 s6, v115 /*v371*/, v147 /*v403*/
	v_max3_num_f32 v84 /*v340*/, v185 /*v441*/, v186 /*v442*/, v187 /*v443*/
	v_max3_num_f32 v75 /*v331*/, v75 /*v331*/, v166 /*v422*/, v77 /*v333*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v116 /*v372*/, v150 /*v406*/
	v_cndmask_b32_e64 v190 /*v446*/, v124 /*v380*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	v_cmp_gt_i32_e64 s3, v117 /*v373*/, v150 /*v406*/
	v_cndmask_b32_e64 v191 /*v447*/, v125 /*v381*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	v_cmp_lt_i32_e64 s4, v117 /*v373*/, v147 /*v403*/
	v_cndmask_b32_e64 v192 /*v448*/, v126 /*v382*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v116 /*v372*/, v147 /*v403*/
	v_cmp_gt_i32_e64 s5, v193 /*v449*/, v150 /*v406*/
	v_cmp_lt_i32_e64 s6, v193 /*v449*/, v147 /*v403*/
	v_max3_num_f32 v76 /*v332*/, v188 /*v444*/, v189 /*v445*/, v190 /*v446*/
	v_max3_num_f32 v77 /*v333*/, v81 /*v337*/, v85 /*v341*/, v91 /*v347*/
	s_or_b32 s2, s2, vcc_lo
	v_max3_num_f32 v92 /*v348*/, v158 /*v414*/, v159 /*v415*/, v102 /*v358*/
	v_cndmask_b32_e64 v193 /*v449*/, v127 /*v383*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	v_max3_num_f32 v93 /*v349*/, v103 /*v359*/, v104 /*v360*/, v105 /*v361*/
	v_cndmask_b32_e64 v194 /*v450*/, v128 /*v384*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	v_max3_num_f32 v78 /*v334*/, v191 /*v447*/, v192 /*v448*/, v193 /*v449*/
	v_cndmask_b32_e64 v195 /*v451*/, v129 /*v385*/, 0xff800000, s2
	v_max3_num_f32 v94 /*v350*/, v160 /*v416*/, v161 /*v417*/, v162 /*v418*/
	v_max3_num_f32 v95 /*v351*/, v163 /*v419*/, v118 /*v374*/, v119 /*v375*/
	v_max3_num_f32 v76 /*v332*/, v80 /*v336*/, v84 /*v340*/, v76 /*v332*/
	v_max3_num_f32 v75 /*v331*/, v75 /*v331*/, v79 /*v335*/, v77 /*v333*/
	v_max3_num_f32 v78 /*v334*/, v78 /*v334*/, v194 /*v450*/, v195 /*v451*/
	v_max3_num_f32 v90 /*v346*/, v92 /*v348*/, v93 /*v349*/, v94 /*v350*/
	v_max3_num_f32 v92 /*v348*/, v95 /*v351*/, v120 /*v376*/, v121 /*v377*/
	v_max3_num_f32 v75 /*v331*/, v75 /*v331*/, v76 /*v332*/, v78 /*v334*/
	v_max3_num_f32 v74 /*v330*/, v74 /*v330*/, v90 /*v346*/, v92 /*v348*/
	v_mov_b32_e32 v76 /*v332*/, v75 /*v331*/
	v_permlanex16_b32 v76 /*v332*/, v76 /*v332*/, s49, 0xfedcba98
	v_dual_mov_b32 v77 /*v333*/, v74 /*v330*/ :: v_dual_max_num_f32 v75 /*v331*/, v75 /*v331*/, v76 /*v332*/
	v_permlanex16_b32 v77 /*v333*/, v77 /*v333*/, s49, 0xfedcba98
	v_dual_sub_f32 v76 /*v332*/, v75 /*v331*/, v135 /*v391*/ :: v_dual_max_num_f32 v74 /*v330*/, v74 /*v330*/, v77 /*v333*/
	v_sub_f32_e32 v77 /*v333*/, v74 /*v330*/, v130 /*v386*/
	v_max_num_f32_e32 v74 /*v330*/, v130 /*v386*/, v74 /*v330*/
	v_cmp_lt_f32_e32 vcc_lo, 0x41000000, v77 /*v333*/
	s_cmp_eq_u32 vcc_lo, 0
	s_cselect_b32 s2, -1, 0
	v_cndmask_b32_e64 v154 /*v410*/, v74 /*v330*/, v130 /*v386*/, s2
	v_cmp_lt_f32_e64 s2, 0x41000000, v76 /*v332*/
	v_max_num_f32_e32 v74 /*v330*/, v135 /*v391*/, v75 /*v331*/
	v_mul_f32_e32 v124 /*v380*/, 0xbfb8aa3b, v154 /*v410*/
	s_cmp_lg_u32 s2, 0
	s_cselect_b32 s3, -1, 0
	s_cmp_eq_u32 s2, 0
	v_pk_fma_f32 v[72:73] /*v[328:329]*/, v[72:73] /*v[328:329]*/, s[56:57], v[124:125] /*v[380:381]*/ op_sel_hi:[1,0,0]
	s_cselect_b32 s2, -1, 0
	v_pk_fma_f32 v[80:81] /*v[336:337]*/, v[132:133] /*v[388:389]*/, s[56:57], v[124:125] /*v[380:381]*/ op_sel_hi:[1,0,0]
	v_cndmask_b32_e64 v151 /*v407*/, v74 /*v330*/, v135 /*v391*/, s2
	v_pk_fma_f32 v[66:67] /*v[322:323]*/, v[66:67] /*v[322:323]*/, s[56:57], v[124:125] /*v[380:381]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[74:75] /*v[330:331]*/, v[68:69] /*v[324:325]*/, s[56:57], v[124:125] /*v[380:381]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[76:77] /*v[332:333]*/, v[70:71] /*v[326:327]*/, s[56:57], v[124:125] /*v[380:381]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v84 /*v340*/, v72 /*v328*/
	v_mul_f32_e32 v134 /*v390*/, 0xbfb8aa3b, v151 /*v407*/
	v_exp_f32_e32 v90 /*v346*/, v73 /*v329*/
	v_nop
	v_pk_fma_f32 v[72:73] /*v[328:329]*/, v[86:87] /*v[342:343]*/, s[56:57], v[124:125] /*v[380:381]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v100 /*v356*/, v80 /*v336*/
	v_exp_f32_e32 v106 /*v362*/, v81 /*v337*/
	v_nop
	v_pk_fma_f32 v[80:81] /*v[336:337]*/, v[156:157] /*v[412:413]*/, s[56:57], v[124:125] /*v[380:381]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[86:87] /*v[342:343]*/, v[158:159] /*v[414:415]*/, s[56:57], v[124:125] /*v[380:381]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[132:133] /*v[388:389]*/, v[164:165] /*v[420:421]*/, s[56:57], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[156:157] /*v[412:413]*/, v[166:167] /*v[422:423]*/, s[56:57], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[158:159] /*v[414:415]*/, v[168:169] /*v[424:425]*/, s[56:57], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v68 /*v324*/, v67 /*v323*/
	v_exp_f32_e32 v70 /*v326*/, v74 /*v330*/
	v_exp_f32_e32 v74 /*v330*/, v75 /*v331*/
	v_pk_fma_f32 v[78:79] /*v[334:335]*/, v[82:83] /*v[338:339]*/, s[56:57], v[124:125] /*v[380:381]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v82 /*v338*/, v77 /*v333*/
	v_exp_f32_e32 v67 /*v323*/, v132 /*v388*/
	v_exp_f32_e32 v69 /*v325*/, v133 /*v389*/
	v_exp_f32_e32 v71 /*v327*/, v156 /*v412*/
	v_pk_fma_f32 v[132:133] /*v[388:389]*/, v[170:171] /*v[426:427]*/, s[56:57], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v75 /*v331*/, v157 /*v413*/
	v_exp_f32_e32 v77 /*v333*/, v158 /*v414*/
	v_pk_fma_f32 v[156:157] /*v[412:413]*/, v[172:173] /*v[428:429]*/, s[56:57], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v83 /*v339*/, v159 /*v415*/
	v_nop
	v_pk_fma_f32 v[158:159] /*v[414:415]*/, v[174:175] /*v[430:431]*/, s[56:57], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v92 /*v348*/, v78 /*v334*/
	v_exp_f32_e32 v98 /*v354*/, v79 /*v335*/
	v_nop
	v_pk_fma_f32 v[78:79] /*v[334:335]*/, v[88:89] /*v[344:345]*/, s[56:57], v[124:125] /*v[380:381]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v85 /*v341*/, v132 /*v388*/
	v_exp_f32_e32 v91 /*v347*/, v133 /*v389*/
	v_exp_f32_e32 v93 /*v349*/, v156 /*v412*/
	v_pk_fma_f32 v[132:133] /*v[388:389]*/, v[176:177] /*v[432:433]*/, s[56:57], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v99 /*v355*/, v157 /*v413*/
	v_exp_f32_e32 v101 /*v357*/, v158 /*v414*/
	v_pk_fma_f32 v[156:157] /*v[412:413]*/, v[178:179] /*v[434:435]*/, s[56:57], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v107 /*v363*/, v159 /*v415*/
	v_nop
	v_pk_fma_f32 v[158:159] /*v[414:415]*/, v[180:181] /*v[436:437]*/, s[56:57], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v114 /*v370*/, v73 /*v329*/
	v_exp_f32_e32 v122 /*v378*/, v79 /*v335*/
	v_pk_fma_f32 v[88:89] /*v[344:345]*/, v[102:103] /*v[358:359]*/, s[56:57], v[124:125] /*v[380:381]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[96:97] /*v[352:353]*/, v[104:105] /*v[360:361]*/, s[56:57], v[124:125] /*v[380:381]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v109 /*v365*/, v132 /*v388*/
	v_exp_f32_e32 v115 /*v371*/, v133 /*v389*/
	v_exp_f32_e32 v117 /*v373*/, v156 /*v412*/
	v_pk_fma_f32 v[132:133] /*v[388:389]*/, v[182:183] /*v[438:439]*/, s[56:57], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v123 /*v379*/, v157 /*v413*/
	v_exp_f32_e32 v73 /*v329*/, v158 /*v414*/
	v_pk_fma_f32 v[156:157] /*v[412:413]*/, v[184:185] /*v[440:441]*/, s[56:57], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v79 /*v335*/, v159 /*v415*/
	v_nop
	v_pk_fma_f32 v[158:159] /*v[414:415]*/, v[186:187] /*v[442:443]*/, s[56:57], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v66 /*v322*/, v66 /*v322*/
	v_exp_f32_e32 v76 /*v332*/, v76 /*v332*/
	v_exp_f32_e32 v108 /*v364*/, v72 /*v328*/
	v_exp_f32_e32 v116 /*v372*/, v78 /*v334*/
	v_exp_f32_e32 v72 /*v328*/, v80 /*v336*/
	v_exp_f32_e32 v78 /*v334*/, v81 /*v337*/
	v_exp_f32_e32 v80 /*v336*/, v86 /*v342*/
	v_exp_f32_e32 v86 /*v342*/, v87 /*v343*/
	v_pk_fma_f32 v[104:105] /*v[360:361]*/, v[160:161] /*v[416:417]*/, s[56:57], v[124:125] /*v[380:381]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v94 /*v350*/, v89 /*v345*/
	v_pk_fma_f32 v[112:113] /*v[368:369]*/, v[162:163] /*v[418:419]*/, s[56:57], v[124:125] /*v[380:381]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v102 /*v358*/, v97 /*v353*/
	v_exp_f32_e32 v81 /*v337*/, v132 /*v388*/
	v_exp_f32_e32 v87 /*v343*/, v133 /*v389*/
	v_exp_f32_e32 v89 /*v345*/, v156 /*v412*/
	v_pk_fma_f32 v[132:133] /*v[388:389]*/, v[188:189] /*v[444:445]*/, s[56:57], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v95 /*v351*/, v157 /*v413*/
	v_exp_f32_e32 v97 /*v353*/, v158 /*v414*/
	v_pk_fma_f32 v[156:157] /*v[412:413]*/, v[190:191] /*v[446:447]*/, s[56:57], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v103 /*v359*/, v159 /*v415*/
	v_nop
	v_pk_fma_f32 v[158:159] /*v[414:415]*/, v[192:193] /*v[448:449]*/, s[56:57], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v96 /*v352*/, v96 /*v352*/
	v_pk_fma_f32 v[126:127] /*v[382:383]*/, v[118:119] /*v[374:375]*/, s[56:57], v[124:125] /*v[380:381]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v110 /*v366*/, v105 /*v361*/
	v_pk_fma_f32 v[128:129] /*v[384:385]*/, v[120:121] /*v[376:377]*/, s[56:57], v[124:125] /*v[380:381]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v118 /*v374*/, v113 /*v369*/
	v_exp_f32_e32 v105 /*v361*/, v132 /*v388*/
	v_exp_f32_e32 v111 /*v367*/, v133 /*v389*/
	v_exp_f32_e32 v113 /*v369*/, v156 /*v412*/
	v_pk_fma_f32 v[132:133] /*v[388:389]*/, v[194:195] /*v[450:451]*/, s[56:57], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v119 /*v375*/, v157 /*v413*/
	v_exp_f32_e32 v121 /*v377*/, v158 /*v414*/
	v_exp_f32_e32 v125 /*v381*/, v159 /*v415*/
	v_pk_add_f32 v[156:157] /*v[412:413]*/, v[66:67] /*v[322:323]*/, v[68:69] /*v[324:325]*/
	v_pk_add_f32 v[158:159] /*v[414:415]*/, v[74:75] /*v[330:331]*/, v[76:77] /*v[332:333]*/
	v_pk_add_f32 v[160:161] /*v[416:417]*/, v[98:99] /*v[354:355]*/, v[100:101] /*v[356:357]*/
	v_pk_add_f32 v[162:163] /*v[418:419]*/, v[108:109] /*v[364:365]*/, v[114:115] /*v[370:371]*/
	v_exp_f32_e32 v88 /*v344*/, v88 /*v344*/
	v_exp_f32_e32 v104 /*v360*/, v104 /*v360*/
	v_exp_f32_e32 v120 /*v376*/, v126 /*v382*/
	v_exp_f32_e32 v124 /*v380*/, v127 /*v383*/
	v_exp_f32_e32 v126 /*v382*/, v128 /*v384*/
	v_exp_f32_e32 v128 /*v384*/, v129 /*v385*/
	v_exp_f32_e32 v127 /*v383*/, v132 /*v388*/
	v_exp_f32_e32 v129 /*v385*/, v133 /*v389*/
	v_nop
	v_pk_add_f32 v[132:133] /*v[388:389]*/, v[84:85] /*v[340:341]*/, v[90:91] /*v[346:347]*/
	v_pk_add_f32 v[156:157] /*v[412:413]*/, v[70:71] /*v[326:327]*/, v[156:157] /*v[412:413]*/
	v_pk_add_f32 v[158:159] /*v[414:415]*/, v[82:83] /*v[338:339]*/, v[158:159] /*v[414:415]*/
	v_pk_add_f32 v[164:165] /*v[420:421]*/, v[122:123] /*v[378:379]*/, v[72:73] /*v[328:329]*/
	v_pk_add_f32 v[160:161] /*v[416:417]*/, v[106:107] /*v[362:363]*/, v[160:161] /*v[416:417]*/
	v_pk_add_f32 v[166:167] /*v[422:423]*/, v[80:81] /*v[336:337]*/, v[86:87] /*v[342:343]*/
	v_pk_add_f32 v[162:163] /*v[418:419]*/, v[116:117] /*v[372:373]*/, v[162:163] /*v[418:419]*/
	v_pk_add_f32 v[168:169] /*v[424:425]*/, v[94:95] /*v[350:351]*/, v[96:97] /*v[352:353]*/
	v_exp_f32_e32 v112 /*v368*/, v112 /*v368*/
	v_pk_add_f32 v[132:133] /*v[388:389]*/, v[92:93] /*v[348:349]*/, v[132:133] /*v[388:389]*/
	v_pk_add_f32 v[164:165] /*v[420:421]*/, v[78:79] /*v[334:335]*/, v[164:165] /*v[420:421]*/
	v_pk_add_f32 v[170:171] /*v[426:427]*/, v[104:105] /*v[360:361]*/, v[110:111] /*v[366:367]*/
	v_pk_add_f32 v[166:167] /*v[422:423]*/, v[88:89] /*v[344:345]*/, v[166:167] /*v[422:423]*/
	v_pk_add_f32 v[156:157] /*v[412:413]*/, v[156:157] /*v[412:413]*/, v[158:159] /*v[414:415]*/
	v_pk_add_f32 v[158:159] /*v[414:415]*/, v[102:103] /*v[358:359]*/, v[168:169] /*v[424:425]*/
	v_pk_add_f32 v[160:161] /*v[416:417]*/, v[160:161] /*v[416:417]*/, v[162:163] /*v[418:419]*/
	v_pk_add_f32 v[162:163] /*v[418:419]*/, v[112:113] /*v[368:369]*/, v[170:171] /*v[426:427]*/
	v_pk_add_f32 v[168:169] /*v[424:425]*/, v[118:119] /*v[374:375]*/, v[120:121] /*v[376:377]*/
	v_pk_add_f32 v[132:133] /*v[388:389]*/, v[132:133] /*v[388:389]*/, v[156:157] /*v[412:413]*/
	v_pk_add_f32 v[156:157] /*v[412:413]*/, v[166:167] /*v[422:423]*/, v[158:159] /*v[414:415]*/
	v_pk_add_f32 v[158:159] /*v[414:415]*/, v[164:165] /*v[420:421]*/, v[160:161] /*v[416:417]*/
	v_pk_add_f32 v[164:165] /*v[420:421]*/, v[126:127] /*v[382:383]*/, v[128:129] /*v[384:385]*/
	v_pk_add_f32 v[160:161] /*v[416:417]*/, v[124:125] /*v[380:381]*/, v[168:169] /*v[424:425]*/
	v_sub_f32_e32 v134 /*v390*/, v130 /*v386*/, v154 /*v410*/
	v_pk_add_f32 v[156:157] /*v[412:413]*/, v[162:163] /*v[418:419]*/, v[156:157] /*v[412:413]*/
	v_pk_add_f32 v[132:133] /*v[388:389]*/, v[132:133] /*v[388:389]*/, v[158:159] /*v[414:415]*/
	v_pk_add_f32 v[158:159] /*v[414:415]*/, v[164:165] /*v[420:421]*/, v[160:161] /*v[416:417]*/
	v_mul_f32_e32 v134 /*v390*/, 0x3fb8aa3b, v134 /*v390*/
	v_pk_add_f32 v[132:133] /*v[388:389]*/, v[156:157] /*v[412:413]*/, v[132:133] /*v[388:389]*/
	v_exp_f32_e32 v134 /*v390*/, v134 /*v390*/
	v_pk_add_f32 v[130:131] /*v[386:387]*/, v[158:159] /*v[414:415]*/, v[132:133] /*v[388:389]*/
	v_dual_mov_b32 v132 /*v388*/, v130 /*v386*/ :: v_dual_mov_b32 v133 /*v389*/, v131 /*v387*/
	v_permlanex16_b32 v132 /*v388*/, v132 /*v388*/, s49, 0xfedcba98
	v_permlanex16_b32 v133 /*v389*/, v133 /*v389*/, s49, 0xfedcba98
	s_set_vgpr_msb 0x5500
	s_cbranch_vccz .LBB0_41
	s_set_vgpr_msb 4
	v_pk_mul_f32 v[126:127], v[126:127], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[124:125], v[124:125], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[122:123], v[122:123], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[120:121], v[120:121], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[118:119], v[118:119], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[116:117], v[116:117], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[114:115], v[114:115], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[112:113], v[112:113], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[110:111], v[110:111], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[108:109], v[108:109], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[106:107], v[106:107], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[104:105], v[104:105], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[102:103], v[102:103], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[100:101], v[100:101], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[98:99], v[98:99], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[96:97], v[96:97], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[94:95], v[94:95], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[92:93], v[92:93], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[90:91], v[90:91], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[88:89], v[88:89], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[86:87], v[86:87], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[84:85], v[84:85], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[82:83], v[82:83], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[80:81], v[80:81], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[78:79], v[78:79], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[76:77], v[76:77], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[74:75], v[74:75], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[72:73], v[72:73], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[70:71], v[70:71], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[68:69], v[68:69], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[66:67], v[66:67], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[64:65], v[64:65], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x400
.LBB0_41:
	s_set_vgpr_msb 0x45
	v_sub_f32_e32 v135 /*v391*/, v135 /*v391*/, v151 /*v407*/
	s_and_b32 s2, s3, exec_lo
	s_cselect_b32 s2, 1, 0
	s_cmp_lg_u32 s2, 1
	v_mul_f32_e32 v135 /*v391*/, 0x3fb8aa3b, v135 /*v391*/
	v_exp_f32_e32 v135 /*v391*/, v135 /*v391*/
	s_set_vgpr_msb 0x4500
	s_cbranch_scc1 .LBB0_36
	v_nop
	s_set_vgpr_msb 0x41
	v_mov_b32_e32 v152 /*v408*/, v135 /*v391*/
	s_set_vgpr_msb 0x4104
	v_pk_mul_f32 v[62:63], v[62:63], v[152:153] /*v[408:409]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[60:61], v[60:61], v[152:153] /*v[408:409]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59], v[58:59], v[152:153] /*v[408:409]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[56:57], v[56:57], v[152:153] /*v[408:409]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[54:55], v[54:55], v[152:153] /*v[408:409]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[52:53], v[52:53], v[152:153] /*v[408:409]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[50:51], v[50:51], v[152:153] /*v[408:409]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[48:49], v[48:49], v[152:153] /*v[408:409]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[46:47], v[46:47], v[152:153] /*v[408:409]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[44:45], v[44:45], v[152:153] /*v[408:409]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[42:43], v[42:43], v[152:153] /*v[408:409]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[40:41], v[40:41], v[152:153] /*v[408:409]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39], v[38:39], v[152:153] /*v[408:409]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[36:37], v[36:37], v[152:153] /*v[408:409]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35], v[34:35], v[152:153] /*v[408:409]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[32:33], v[32:33], v[152:153] /*v[408:409]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[30:31], v[30:31], v[152:153] /*v[408:409]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[28:29], v[28:29], v[152:153] /*v[408:409]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[26:27], v[26:27], v[152:153] /*v[408:409]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25], v[24:25], v[152:153] /*v[408:409]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[22:23], v[22:23], v[152:153] /*v[408:409]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21], v[20:21], v[152:153] /*v[408:409]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[18:19], v[18:19], v[152:153] /*v[408:409]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[16:17], v[16:17], v[152:153] /*v[408:409]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[14:15], v[152:153] /*v[408:409]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13], v[12:13], v[152:153] /*v[408:409]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[10:11], v[152:153] /*v[408:409]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[8:9], v[152:153] /*v[408:409]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[6:7], v[152:153] /*v[408:409]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[4:5], v[152:153] /*v[408:409]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[2:3], v[2:3], v[152:153] /*v[408:409]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[0:1], v[0:1], v[152:153] /*v[408:409]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x400
	s_branch .LBB0_36
.LBB0_43:
	v_dual_mov_b32 v1, v0 :: v_dual_mov_b32 v2, v0
	v_dual_mov_b32 v3, v0 :: v_dual_mov_b32 v4, v0
	v_dual_mov_b32 v5, v0 :: v_dual_mov_b32 v6, v0
	s_set_vgpr_msb 0x41
	v_dual_mov_b32 v65 /*v321*/, 1.0 :: v_dual_mov_b32 v148 /*v404*/, v142 /*v398*/
	s_set_vgpr_msb 0x4100
	v_mov_b32_e32 v7, v0
	v_mov_b64_e32 v[12:13], v[4:5]
	v_mov_b64_e32 v[10:11], v[2:3]
	s_set_vgpr_msb 0x41
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v64 /*v320*/, v65 /*v321*/ :: v_dual_mov_b32 v154 /*v410*/, v135 /*v391*/
	s_set_vgpr_msb 0x4100
	v_mov_b64_e32 v[14:15], v[6:7]
	v_mov_b64_e32 v[8:9], v[0:1]
	v_mov_b64_e32 v[22:23], v[6:7]
	v_mov_b64_e32 v[20:21], v[4:5]
	v_mov_b64_e32 v[18:19], v[2:3]
	v_mov_b64_e32 v[16:17], v[0:1]
	v_mov_b64_e32 v[30:31], v[6:7]
	v_mov_b64_e32 v[28:29], v[4:5]
	v_mov_b64_e32 v[26:27], v[2:3]
	v_mov_b64_e32 v[24:25], v[0:1]
	v_mov_b64_e32 v[38:39], v[6:7]
	v_mov_b64_e32 v[36:37], v[4:5]
	v_mov_b64_e32 v[34:35], v[2:3]
	v_mov_b64_e32 v[32:33], v[0:1]
	v_mov_b64_e32 v[46:47], v[6:7]
	v_mov_b64_e32 v[44:45], v[4:5]
	v_mov_b64_e32 v[42:43], v[2:3]
	v_mov_b64_e32 v[40:41], v[0:1]
	v_mov_b64_e32 v[54:55], v[6:7]
	v_mov_b64_e32 v[52:53], v[4:5]
	v_mov_b64_e32 v[50:51], v[2:3]
	v_mov_b64_e32 v[48:49], v[0:1]
	v_mov_b64_e32 v[62:63], v[6:7]
	v_mov_b64_e32 v[60:61], v[4:5]
	v_mov_b64_e32 v[58:59], v[2:3]
	v_mov_b64_e32 v[56:57], v[0:1]
	v_mov_b64_e32 v[70:71], v[6:7]
	v_mov_b64_e32 v[68:69], v[4:5]
	v_mov_b64_e32 v[66:67], v[2:3]
	v_mov_b64_e32 v[64:65], v[0:1]
	v_mov_b64_e32 v[78:79], v[6:7]
	v_mov_b64_e32 v[76:77], v[4:5]
	v_mov_b64_e32 v[74:75], v[2:3]
	v_mov_b64_e32 v[72:73], v[0:1]
	v_mov_b64_e32 v[86:87], v[6:7]
	v_mov_b64_e32 v[84:85], v[4:5]
	v_mov_b64_e32 v[82:83], v[2:3]
	v_mov_b64_e32 v[80:81], v[0:1]
	v_mov_b64_e32 v[94:95], v[6:7]
	v_mov_b64_e32 v[92:93], v[4:5]
	v_mov_b64_e32 v[90:91], v[2:3]
	v_mov_b64_e32 v[88:89], v[0:1]
	v_mov_b64_e32 v[102:103], v[6:7]
	v_mov_b64_e32 v[100:101], v[4:5]
	v_mov_b64_e32 v[98:99], v[2:3]
	v_mov_b64_e32 v[96:97], v[0:1]
	v_mov_b64_e32 v[110:111], v[6:7]
	v_mov_b64_e32 v[108:109], v[4:5]
	v_mov_b64_e32 v[106:107], v[2:3]
	v_mov_b64_e32 v[104:105], v[0:1]
	v_mov_b64_e32 v[118:119], v[6:7]
	v_mov_b64_e32 v[116:117], v[4:5]
	v_mov_b64_e32 v[114:115], v[2:3]
	v_mov_b64_e32 v[112:113], v[0:1]
	v_mov_b64_e32 v[126:127], v[6:7]
	v_mov_b64_e32 v[124:125], v[4:5]
	v_mov_b64_e32 v[122:123], v[2:3]
	v_mov_b64_e32 v[120:121], v[0:1]
	s_branch .LBB0_45
.LBB0_44:
	s_set_vgpr_msb 0x41
	v_mov_b32_e32 v135 /*v391*/, v151 /*v407*/
	s_set_vgpr_msb 0x4100
.LBB0_45:
	s_mov_b32 s27, 0
	s_cmp_le_i32 s7, s11
	s_mov_b32 s49, s27
	s_cbranch_scc1 .LBB0_54
	s_add_co_i32 s2, s19, s35
	s_lshl_b32 s3, s54, 6
	s_sub_co_i32 s2, s2, s18
	s_sub_co_i32 s5, s9, s3
	s_sub_co_i32 s2, s2, s16
	s_add_co_i32 s6, s97, s3
	s_max_i32 s2, s2, 0
	s_mov_b32 s4, 1
	s_lshr_b32 s2, s2, 6
	s_mov_b32 s24, 16
	s_min_u32 s2, s2, s98
	s_mov_b32 s23, 0x800000
	s_sub_co_i32 s2, 0, s2
	s_sub_co_i32 s11, s5, 64
	s_ashr_i32 s3, s2, 31
	s_add_co_i32 s16, s6, 64
	s_add_nc_u64 s[28:29], s[2:3], 1
	s_mov_b32 s21, 0xffff0000
	s_mov_b32 s20, 0x7510000
	s_mov_b32 s36, 0xf510000
	s_mov_b32 s29, 0x76543210
	s_mov_b32 s30, 0x3fb8aa3b
	s_branch .LBB0_48
.LBB0_47:
	s_set_vgpr_msb 0x45
	v_cvt_pk_bf16_f32 v165 /*v421*/, v116 /*v372*/, v122 /*v378*/
	v_cvt_pk_bf16_f32 v164 /*v420*/, v110 /*v366*/, v114 /*v370*/
	v_cvt_pk_bf16_f32 v163 /*v419*/, v102 /*v358*/, v106 /*v362*/
	v_cvt_pk_bf16_f32 v162 /*v418*/, v94 /*v350*/, v98 /*v354*/
	v_cvt_pk_bf16_f32 v161 /*v417*/, v84 /*v340*/, v90 /*v346*/
	v_cvt_pk_bf16_f32 v160 /*v416*/, v78 /*v334*/, v82 /*v338*/
	v_cvt_pk_bf16_f32 v159 /*v415*/, v72 /*v328*/, v74 /*v330*/
	v_cvt_pk_bf16_f32 v158 /*v414*/, v68 /*v324*/, v70 /*v326*/
	v_cvt_pk_bf16_f32 v173 /*v429*/, v117 /*v373*/, v123 /*v379*/
	v_cvt_pk_bf16_f32 v172 /*v428*/, v111 /*v367*/, v115 /*v371*/
	v_cvt_pk_bf16_f32 v171 /*v427*/, v103 /*v359*/, v107 /*v363*/
	v_cvt_pk_bf16_f32 v170 /*v426*/, v95 /*v351*/, v99 /*v355*/
	v_cvt_pk_bf16_f32 v169 /*v425*/, v85 /*v341*/, v91 /*v347*/
	v_cvt_pk_bf16_f32 v168 /*v424*/, v79 /*v335*/, v83 /*v339*/
	v_cvt_pk_bf16_f32 v167 /*v423*/, v73 /*v329*/, v75 /*v331*/
	v_cvt_pk_bf16_f32 v166 /*v422*/, v69 /*v325*/, v71 /*v327*/
	s_set_vgpr_msb 0x4505
	s_wait_dscnt 0x1d
	v_wmma_f32_16x16x32_bf16 v[120:127], v[56:63] /*v[312:319]*/, v[158:165] /*v[414:421]*/, v[120:127]
	s_set_vgpr_msb 0x545
	v_cvt_pk_bf16_f32 v75 /*v331*/, v127 /*v383*/, v129 /*v385*/
	v_cvt_pk_bf16_f32 v74 /*v330*/, v121 /*v377*/, v125 /*v381*/
	v_cvt_pk_bf16_f32 v73 /*v329*/, v113 /*v369*/, v119 /*v375*/
	v_cvt_pk_bf16_f32 v72 /*v328*/, v105 /*v361*/, v109 /*v365*/
	v_cvt_pk_bf16_f32 v71 /*v327*/, v97 /*v353*/, v101 /*v357*/
	v_cvt_pk_bf16_f32 v70 /*v326*/, v89 /*v345*/, v93 /*v349*/
	v_cvt_pk_bf16_f32 v69 /*v325*/, v81 /*v337*/, v87 /*v343*/
	s_set_vgpr_msb 0x4505
	v_wmma_f32_16x16x32_bf16 v[56:63], v[56:63] /*v[312:319]*/, v[166:173] /*v[422:429]*/, v[56:63]
	s_set_vgpr_msb 0x545
	v_cvt_pk_bf16_f32 v68 /*v324*/, v67 /*v323*/, v77 /*v333*/
	v_dual_fmac_f32 v144 /*v400*/, v132 /*v388*/, v64 /*v320*/ :: v_dual_fmac_f32 v133 /*v389*/, v134 /*v390*/, v65 /*v321*/
	v_cmp_lt_u64_e64 s2, s[56:57], s[48:49]
	v_nop
	v_cvt_pk_bf16_f32 v63 /*v319*/, v126 /*v382*/, v128 /*v384*/
	v_cvt_pk_bf16_f32 v62 /*v318*/, v120 /*v376*/, v124 /*v380*/
	v_cvt_pk_bf16_f32 v61 /*v317*/, v112 /*v368*/, v118 /*v374*/
	s_set_vgpr_msb 0x4505
	s_wait_dscnt 0x1c
	v_wmma_f32_16x16x32_bf16 v[112:119], v[40:47] /*v[296:303]*/, v[158:165] /*v[414:421]*/, v[112:119]
	s_set_vgpr_msb 0x545
	v_cvt_pk_bf16_f32 v60 /*v316*/, v104 /*v360*/, v108 /*v364*/
	v_cvt_pk_bf16_f32 v59 /*v315*/, v96 /*v352*/, v100 /*v356*/
	v_cvt_pk_bf16_f32 v58 /*v314*/, v88 /*v344*/, v92 /*v348*/
	v_cvt_pk_bf16_f32 v57 /*v313*/, v80 /*v336*/, v86 /*v342*/
	v_cvt_pk_bf16_f32 v56 /*v312*/, v66 /*v322*/, v76 /*v332*/
	v_dual_add_f32 v64 /*v320*/, v144 /*v400*/, v130 /*v386*/ :: v_dual_add_f32 v65 /*v321*/, v133 /*v389*/, v131 /*v387*/
	s_set_vgpr_msb 0x4505
	v_wmma_f32_16x16x32_bf16 v[48:55], v[40:47] /*v[296:303]*/, v[166:173] /*v[422:429]*/, v[48:55]
	s_set_vgpr_msb 0x541
	v_dual_mov_b32 v153 /*v409*/, v155 /*v411*/ :: v_dual_mov_b32 v144 /*v400*/, v156 /*v412*/
	v_dual_mov_b32 v135 /*v391*/, v151 /*v407*/ :: v_dual_mov_b32 v154 /*v410*/, v152 /*v408*/
	s_sub_co_i32 s11, s11, 64
	s_add_co_i32 s16, s16, 64
	s_and_b32 vcc_lo, exec_lo, s2
	s_set_vgpr_msb 0x4105
	s_wait_dscnt 0x15
	v_wmma_f32_16x16x32_bf16 v[104:111], v[24:31] /*v[280:287]*/, v[158:165] /*v[414:421]*/, v[104:111]
	s_mov_b64 s[54:55], s[56:57]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[40:47], v[24:31] /*v[280:287]*/, v[166:173] /*v[422:429]*/, v[40:47]
	v_wmma_f32_16x16x32_bf16 v[96:103], v[8:15] /*v[264:271]*/, v[158:165] /*v[414:421]*/, v[96:103]
	v_wmma_f32_16x16x32_bf16 v[32:39], v[8:15] /*v[264:271]*/, v[166:173] /*v[422:429]*/, v[32:39]
	s_set_vgpr_msb 0x504
	v_wmma_f32_16x16x32_bf16 v[88:95], v[248:255], v[158:165] /*v[414:421]*/, v[88:95]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[248:255], v[166:173] /*v[422:429]*/, v[24:31]
	v_wmma_f32_16x16x32_bf16 v[80:87], v[232:239], v[158:165] /*v[414:421]*/, v[80:87]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[232:239], v[166:173] /*v[422:429]*/, v[16:23]
	v_wmma_f32_16x16x32_bf16 v[72:79], v[216:223], v[158:165] /*v[414:421]*/, v[72:79]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[216:223], v[166:173] /*v[422:429]*/, v[8:15]
	v_wmma_f32_16x16x32_bf16 v[64:71], v[200:207], v[158:165] /*v[414:421]*/, v[64:71]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[200:207], v[166:173] /*v[422:429]*/, v[0:7]
	s_set_vgpr_msb 0x405
	v_wmma_f32_16x16x32_bf16 v[120:127], v[48:55] /*v[304:311]*/, v[56:63] /*v[312:319]*/, v[120:127]
	v_wmma_f32_16x16x32_bf16 v[56:63], v[48:55] /*v[304:311]*/, v[68:75] /*v[324:331]*/, v[56:63]
	v_wmma_f32_16x16x32_bf16 v[112:119], v[32:39] /*v[288:295]*/, v[56:63] /*v[312:319]*/, v[112:119]
	v_wmma_f32_16x16x32_bf16 v[48:55], v[32:39] /*v[288:295]*/, v[68:75] /*v[324:331]*/, v[48:55]
	v_wmma_f32_16x16x32_bf16 v[104:111], v[16:23] /*v[272:279]*/, v[56:63] /*v[312:319]*/, v[104:111]
	v_wmma_f32_16x16x32_bf16 v[40:47], v[16:23] /*v[272:279]*/, v[68:75] /*v[324:331]*/, v[40:47]
	v_wmma_f32_16x16x32_bf16 v[96:103], v[0:7] /*v[256:263]*/, v[56:63] /*v[312:319]*/, v[96:103]
	v_wmma_f32_16x16x32_bf16 v[32:39], v[0:7] /*v[256:263]*/, v[68:75] /*v[324:331]*/, v[32:39]
	s_set_vgpr_msb 0x504
	v_wmma_f32_16x16x32_bf16 v[88:95], v[240:247], v[56:63] /*v[312:319]*/, v[88:95]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[240:247], v[68:75] /*v[324:331]*/, v[24:31]
	v_wmma_f32_16x16x32_bf16 v[80:87], v[224:231], v[56:63] /*v[312:319]*/, v[80:87]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[224:231], v[68:75] /*v[324:331]*/, v[16:23]
	v_wmma_f32_16x16x32_bf16 v[72:79], v[208:215], v[56:63] /*v[312:319]*/, v[72:79]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[208:215], v[68:75] /*v[324:331]*/, v[8:15]
	v_wmma_f32_16x16x32_bf16 v[64:71], v[192:199], v[56:63] /*v[312:319]*/, v[64:71]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[192:199], v[68:75] /*v[324:331]*/, v[0:7]
	s_set_vgpr_msb 0x400
	s_cbranch_vccz .LBB0_55
.LBB0_48:
	s_set_vgpr_msb 0x41
	v_dual_mov_b32 v156 /*v412*/, v148 /*v404*/ :: v_dual_mov_b32 v148 /*v404*/, v144 /*v400*/
	v_dual_mov_b32 v155 /*v411*/, v145 /*v401*/ :: v_dual_mov_b32 v145 /*v401*/, v153 /*v409*/
	s_add_nc_u64 s[56:57], s[54:55], 1
	s_wait_tensorcnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	s_wait_alu depctr_va_vdst(0)
	ds_load_b128 v[56:59] /*v[312:315]*/, v156 /*v412*/
	ds_load_b128 v[60:63] /*v[316:319]*/, v156 /*v412*/ offset:32
	ds_load_b128 v[48:51] /*v[304:307]*/, v156 /*v412*/ offset:64
	ds_load_b128 v[52:55] /*v[308:311]*/, v156 /*v412*/ offset:96
	ds_load_b128 v[40:43] /*v[296:299]*/, v156 /*v412*/ offset:128
	ds_load_b128 v[44:47] /*v[300:303]*/, v156 /*v412*/ offset:160
	ds_load_b128 v[32:35] /*v[288:291]*/, v156 /*v412*/ offset:192
	ds_load_b128 v[36:39] /*v[292:295]*/, v156 /*v412*/ offset:224
	ds_load_b128 v[24:27] /*v[280:283]*/, v156 /*v412*/ offset:4352
	ds_load_b128 v[28:31] /*v[284:287]*/, v156 /*v412*/ offset:4384
	ds_load_b128 v[16:19] /*v[272:275]*/, v156 /*v412*/ offset:4416
	ds_load_b128 v[20:23] /*v[276:279]*/, v156 /*v412*/ offset:4448
	ds_load_b128 v[8:11] /*v[264:267]*/, v156 /*v412*/ offset:4480
	ds_load_b128 v[12:15] /*v[268:271]*/, v156 /*v412*/ offset:4512
	ds_load_b128 v[0:3] /*v[256:259]*/, v156 /*v412*/ offset:4544
	ds_load_b128 v[4:7] /*v[260:263]*/, v156 /*v412*/ offset:4576
	s_set_vgpr_msb 0x4101
	ds_load_b128 v[248:251], v156 /*v412*/ offset:8704
	ds_load_b128 v[252:255], v156 /*v412*/ offset:8736
	ds_load_b128 v[240:243], v156 /*v412*/ offset:8768
	ds_load_b128 v[244:247], v156 /*v412*/ offset:8800
	ds_load_b128 v[232:235], v156 /*v412*/ offset:8832
	ds_load_b128 v[236:239], v156 /*v412*/ offset:8864
	ds_load_b128 v[224:227], v156 /*v412*/ offset:8896
	ds_load_b128 v[228:231], v156 /*v412*/ offset:8928
	ds_load_b128 v[216:219], v156 /*v412*/ offset:13056
	ds_load_b128 v[220:223], v156 /*v412*/ offset:13088
	ds_load_b128 v[208:211], v156 /*v412*/ offset:13120
	ds_load_b128 v[212:215], v156 /*v412*/ offset:13152
	ds_load_b128 v[200:203], v156 /*v412*/ offset:13184
	ds_load_b128 v[204:207], v156 /*v412*/ offset:13216
	ds_load_b128 v[192:195], v156 /*v412*/ offset:13248
	ds_load_b128 v[196:199], v156 /*v412*/ offset:13280
	s_cmp_ge_i32 s56, s10
	s_set_vgpr_msb 0x100
	s_cbranch_scc1 .LBB0_50
	s_add_co_i32 s5, s28, s54
	s_set_vgpr_msb 0x41
	v_med3_i32 v66 /*v322*/, s11, 0, 64
	s_lshr_b32 s6, s5, 31
	s_ashr_i32 s17, s16, 31
	s_add_co_i32 s6, s5, s6
	s_mul_u64 s[2:3], s[16:17], s[64:65]
	s_and_b32 s22, s6, 0x7ffe
	s_mul_u64 s[6:7], s[16:17], s[62:63]
	v_readfirstlane_b32 s17, v66 /*v322*/
	s_sub_co_i32 s5, s5, s22
	s_lshl_b64 s[6:7], s[6:7], 1
	s_lshl_b32 s31, s5, 17
	s_add_nc_u64 s[6:7], s[52:53], s[6:7]
	s_sub_co_i32 s5, s17, s80
	s_lshl_b64 s[2:3], s[2:3], 1
	s_max_i32 s17, s5, 0
	s_add_nc_u64 s[6:7], s[44:45], s[6:7]
	s_lshl_b32 s17, s17, 16
	s_add_nc_u64 s[2:3], s[50:51], s[2:3]
	s_or_b32 s5, s81, s31
	s_bitset1_b32 s7, 31
	s_or_b32 s22, s17, 0x7fff
	s_mov_b32 s37, s21
	tensor_load_to_lds s[4:7], s[20:27]
	s_add_nc_u64 s[6:7], s[46:47], s[2:3]
	s_or_b32 s5, s78, s31
	s_bitset1_b32 s7, 31
	s_mov_b32 s38, s22
	s_mov_b32 s39, s23
	s_mov_b32 s40, s24
	s_mov_b32 s43, s27
	tensor_load_to_lds s[4:7], s[36:43]
	s_set_vgpr_msb 0x4100
.LBB0_50:
	s_set_vgpr_msb 0x51
	s_wait_dscnt 0x1e
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[322:329]*/, v[56:63] /*v[312:319]*/, v[128:135], 0
	v_wmma_f32_16x16x32_bf16 v[126:133] /*v[382:389]*/, v[56:63] /*v[312:319]*/, v[160:167], 0
	s_wait_dscnt 0x16
	v_wmma_f32_16x16x32_bf16 v[94:101] /*v[350:357]*/, v[24:31] /*v[280:287]*/, v[128:135], 0
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[322:329]*/, v[48:55] /*v[304:311]*/, v[136:143], v[66:73] /*v[322:329]*/
	v_wmma_f32_16x16x32_bf16 v[126:133] /*v[382:389]*/, v[48:55] /*v[304:311]*/, v[168:175], v[126:133] /*v[382:389]*/
	v_wmma_f32_16x16x32_bf16 v[158:165] /*v[414:421]*/, v[24:31] /*v[280:287]*/, v[160:167], 0
	s_wait_dscnt 0x14
	v_wmma_f32_16x16x32_bf16 v[94:101] /*v[350:357]*/, v[16:23] /*v[272:279]*/, v[136:143], v[94:101] /*v[350:357]*/
	s_set_vgpr_msb 0x5140
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x32_bf16 v[166:173] /*v[422:429]*/, v[248:255], v[128:135], 0
	v_wmma_f32_16x16x32_bf16 v[174:181] /*v[430:437]*/, v[248:255], v[160:167], 0
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x32_bf16 v[182:189] /*v[438:445]*/, v[216:223], v[128:135], 0
	v_wmma_f32_16x16x32_bf16 v[190:197] /*v[446:453]*/, v[216:223], v[160:167], 0
	s_set_vgpr_msb 0x4051
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[322:329]*/, v[40:47] /*v[296:303]*/, v[144:151], v[66:73] /*v[322:329]*/
	v_wmma_f32_16x16x32_bf16 v[126:133] /*v[382:389]*/, v[40:47] /*v[296:303]*/, v[176:183], v[126:133] /*v[382:389]*/
	v_wmma_f32_16x16x32_bf16 v[158:165] /*v[414:421]*/, v[16:23] /*v[272:279]*/, v[168:175], v[158:165] /*v[414:421]*/
	v_wmma_f32_16x16x32_bf16 v[94:101] /*v[350:357]*/, v[8:15] /*v[264:271]*/, v[144:151], v[94:101] /*v[350:357]*/
	s_set_vgpr_msb 0x5150
	v_wmma_f32_16x16x32_bf16 v[166:173] /*v[422:429]*/, v[240:247], v[136:143], v[166:173] /*v[422:429]*/
	v_wmma_f32_16x16x32_bf16 v[174:181] /*v[430:437]*/, v[240:247], v[168:175], v[174:181] /*v[430:437]*/
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x32_bf16 v[182:189] /*v[438:445]*/, v[208:215], v[136:143], v[182:189] /*v[438:445]*/
	v_wmma_f32_16x16x32_bf16 v[190:197] /*v[446:453]*/, v[208:215], v[168:175], v[190:197] /*v[446:453]*/
	s_set_vgpr_msb 0x5051
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[322:329]*/, v[32:39] /*v[288:295]*/, v[152:159], v[66:73] /*v[322:329]*/
	v_wmma_f32_16x16x32_bf16 v[126:133] /*v[382:389]*/, v[32:39] /*v[288:295]*/, v[184:191], v[126:133] /*v[382:389]*/
	v_wmma_f32_16x16x32_bf16 v[158:165] /*v[414:421]*/, v[8:15] /*v[264:271]*/, v[176:183], v[158:165] /*v[414:421]*/
	v_wmma_f32_16x16x32_bf16 v[94:101] /*v[350:357]*/, v[0:7] /*v[256:263]*/, v[152:159], v[94:101] /*v[350:357]*/
	s_set_vgpr_msb 0x5150
	v_wmma_f32_16x16x32_bf16 v[166:173] /*v[422:429]*/, v[232:239], v[144:151], v[166:173] /*v[422:429]*/
	v_wmma_f32_16x16x32_bf16 v[174:181] /*v[430:437]*/, v[232:239], v[176:183], v[174:181] /*v[430:437]*/
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[182:189] /*v[438:445]*/, v[200:207], v[144:151], v[182:189] /*v[438:445]*/
	v_wmma_f32_16x16x32_bf16 v[190:197] /*v[446:453]*/, v[200:207], v[176:183], v[190:197] /*v[446:453]*/
	s_set_vgpr_msb 0x5051
	v_wmma_f32_16x16x32_bf16 v[158:165] /*v[414:421]*/, v[0:7] /*v[256:263]*/, v[184:191], v[158:165] /*v[414:421]*/
	s_set_vgpr_msb 0x5150
	v_wmma_f32_16x16x32_bf16 v[166:173] /*v[422:429]*/, v[224:231], v[152:159], v[166:173] /*v[422:429]*/
	v_wmma_f32_16x16x32_bf16 v[174:181] /*v[430:437]*/, v[224:231], v[184:191], v[174:181] /*v[430:437]*/
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[182:189] /*v[438:445]*/, v[192:199], v[152:159], v[182:189] /*v[438:445]*/
	v_wmma_f32_16x16x32_bf16 v[190:197] /*v[446:453]*/, v[192:199], v[184:191], v[190:197] /*v[446:453]*/
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x5041
	ds_load_tr16_b128 v[56:59] /*v[312:315]*/, v155 /*v411*/
	ds_load_tr16_b128 v[40:43] /*v[296:299]*/, v155 /*v411*/ offset:32
	ds_load_tr16_b128 v[60:63] /*v[316:319]*/, v155 /*v411*/ offset:4608
	ds_load_tr16_b128 v[44:47] /*v[300:303]*/, v155 /*v411*/ offset:4640
	ds_load_tr16_b128 v[48:51] /*v[304:307]*/, v155 /*v411*/ offset:9216
	ds_load_tr16_b128 v[32:35] /*v[288:291]*/, v155 /*v411*/ offset:9248
	ds_load_tr16_b128 v[52:55] /*v[308:311]*/, v155 /*v411*/ offset:13824
	ds_load_tr16_b128 v[36:39] /*v[292:295]*/, v155 /*v411*/ offset:13856
	ds_load_tr16_b128 v[24:27] /*v[280:283]*/, v155 /*v411*/ offset:64
	ds_load_tr16_b128 v[8:11] /*v[264:267]*/, v155 /*v411*/ offset:96
	ds_load_tr16_b128 v[28:31] /*v[284:287]*/, v155 /*v411*/ offset:4672
	ds_load_tr16_b128 v[12:15] /*v[268:271]*/, v155 /*v411*/ offset:4704
	ds_load_tr16_b128 v[16:19] /*v[272:275]*/, v155 /*v411*/ offset:9280
	ds_load_tr16_b128 v[0:3] /*v[256:259]*/, v155 /*v411*/ offset:9312
	ds_load_tr16_b128 v[20:23] /*v[276:279]*/, v155 /*v411*/ offset:13888
	ds_load_tr16_b128 v[4:7] /*v[260:263]*/, v155 /*v411*/ offset:13920
	s_set_vgpr_msb 0x4101
	ds_load_tr16_b128 v[248:251], v155 /*v411*/ offset:128
	ds_load_tr16_b128 v[232:235], v155 /*v411*/ offset:160
	ds_load_tr16_b128 v[252:255], v155 /*v411*/ offset:4736
	ds_load_tr16_b128 v[236:239], v155 /*v411*/ offset:4768
	ds_load_tr16_b128 v[240:243], v155 /*v411*/ offset:9344
	ds_load_tr16_b128 v[224:227], v155 /*v411*/ offset:9376
	ds_load_tr16_b128 v[244:247], v155 /*v411*/ offset:13952
	ds_load_tr16_b128 v[228:231], v155 /*v411*/ offset:13984
	ds_load_tr16_b128 v[216:219], v155 /*v411*/ offset:192
	ds_load_tr16_b128 v[200:203], v155 /*v411*/ offset:224
	ds_load_tr16_b128 v[220:223], v155 /*v411*/ offset:4800
	ds_load_tr16_b128 v[204:207], v155 /*v411*/ offset:4832
	ds_load_tr16_b128 v[208:211], v155 /*v411*/ offset:9408
	ds_load_tr16_b128 v[192:195], v155 /*v411*/ offset:9440
	ds_load_tr16_b128 v[212:215], v155 /*v411*/ offset:14016
	ds_load_tr16_b128 v[196:199], v155 /*v411*/ offset:14048
	s_set_vgpr_msb 0x155
	v_max_num_f32_e32 v74 /*v330*/, v66 /*v322*/, v67 /*v323*/
	v_max_num_f32_e32 v75 /*v331*/, v126 /*v382*/, v127 /*v383*/
	v_max3_num_f32 v76 /*v332*/, v69 /*v325*/, v70 /*v326*/, v71 /*v327*/
	v_max3_num_f32 v77 /*v333*/, v129 /*v385*/, v130 /*v386*/, v131 /*v387*/
	v_max3_num_f32 v80 /*v336*/, v95 /*v351*/, v96 /*v352*/, v97 /*v353*/
	v_max3_num_f32 v81 /*v337*/, v159 /*v415*/, v160 /*v416*/, v161 /*v417*/
	v_max3_num_f32 v82 /*v338*/, v98 /*v354*/, v99 /*v355*/, v100 /*v356*/
	v_max3_num_f32 v83 /*v339*/, v162 /*v418*/, v163 /*v419*/, v164 /*v420*/
	v_max3_num_f32 v84 /*v340*/, v101 /*v357*/, v166 /*v422*/, v167 /*v423*/
	v_max3_num_f32 v85 /*v341*/, v165 /*v421*/, v174 /*v430*/, v175 /*v431*/
	v_max3_num_f32 v78 /*v334*/, v72 /*v328*/, v73 /*v329*/, v94 /*v350*/
	v_max3_num_f32 v79 /*v335*/, v132 /*v388*/, v133 /*v389*/, v158 /*v414*/
	v_max3_num_f32 v87 /*v343*/, v176 /*v432*/, v177 /*v433*/, v178 /*v434*/
	v_max3_num_f32 v89 /*v345*/, v179 /*v435*/, v180 /*v436*/, v181 /*v437*/
	v_max3_num_f32 v91 /*v347*/, v190 /*v446*/, v191 /*v447*/, v192 /*v448*/
	v_max3_num_f32 v93 /*v349*/, v193 /*v449*/, v194 /*v450*/, v195 /*v451*/
	v_max3_num_f32 v74 /*v330*/, v74 /*v330*/, v68 /*v324*/, v76 /*v332*/
	v_max3_num_f32 v76 /*v332*/, v80 /*v336*/, v82 /*v338*/, v84 /*v340*/
	v_max3_num_f32 v75 /*v331*/, v75 /*v331*/, v128 /*v384*/, v77 /*v333*/
	v_max3_num_f32 v77 /*v333*/, v81 /*v337*/, v83 /*v339*/, v85 /*v341*/
	v_max3_num_f32 v86 /*v342*/, v168 /*v424*/, v169 /*v425*/, v170 /*v426*/
	v_max3_num_f32 v88 /*v344*/, v171 /*v427*/, v172 /*v428*/, v173 /*v429*/
	v_max3_num_f32 v90 /*v346*/, v182 /*v438*/, v183 /*v439*/, v184 /*v440*/
	v_max3_num_f32 v92 /*v348*/, v185 /*v441*/, v186 /*v442*/, v187 /*v443*/
	v_max3_num_f32 v74 /*v330*/, v74 /*v330*/, v78 /*v334*/, v76 /*v332*/
	v_max3_num_f32 v76 /*v332*/, v87 /*v343*/, v89 /*v345*/, v91 /*v347*/
	v_max3_num_f32 v78 /*v334*/, v93 /*v349*/, v196 /*v452*/, v197 /*v453*/
	v_max3_num_f32 v75 /*v331*/, v75 /*v331*/, v79 /*v335*/, v77 /*v333*/
	v_max3_num_f32 v80 /*v336*/, v86 /*v342*/, v88 /*v344*/, v90 /*v346*/
	v_max3_num_f32 v81 /*v337*/, v92 /*v348*/, v188 /*v444*/, v189 /*v445*/
	v_max3_num_f32 v75 /*v331*/, v75 /*v331*/, v76 /*v332*/, v78 /*v334*/
	v_max3_num_f32 v74 /*v330*/, v74 /*v330*/, v80 /*v336*/, v81 /*v337*/
	v_mov_b32_e32 v77 /*v333*/, v75 /*v331*/
	v_permlanex16_b32 v77 /*v333*/, v77 /*v333*/, s29, 0xfedcba98
	v_dual_mov_b32 v76 /*v332*/, v74 /*v330*/ :: v_dual_max_num_f32 v75 /*v331*/, v75 /*v331*/, v77 /*v333*/
	v_permlanex16_b32 v76 /*v332*/, v76 /*v332*/, s29, 0xfedcba98
	v_dual_sub_f32 v77 /*v333*/, v75 /*v331*/, v135 /*v391*/ :: v_dual_max_num_f32 v74 /*v330*/, v74 /*v330*/, v76 /*v332*/
	v_cmp_lt_f32_e64 s2, 0x41000000, v77 /*v333*/
	v_sub_f32_e32 v76 /*v332*/, v74 /*v330*/, v154 /*v410*/
	v_max_num_f32_e32 v74 /*v330*/, v154 /*v410*/, v74 /*v330*/
	v_cmp_lt_f32_e32 vcc_lo, 0x41000000, v76 /*v332*/
	s_cmp_eq_u32 vcc_lo, 0
	s_cselect_b32 s3, -1, 0
	s_cmp_lg_u32 s2, 0
	v_dual_cndmask_b32 v152 /*v408*/, v74 /*v330*/, v154 /*v410*/, s3 :: v_dual_max_num_f32 v74 /*v330*/, v135 /*v391*/, v75 /*v331*/
	s_cselect_b32 s3, -1, 0
	s_cmp_eq_u32 s2, 0
	s_cselect_b32 s2, -1, 0
	v_mul_f32_e32 v118 /*v374*/, 0xbfb8aa3b, v152 /*v408*/
	v_cndmask_b32_e64 v151 /*v407*/, v74 /*v330*/, v135 /*v391*/, s2
	v_pk_fma_f32 v[66:67] /*v[322:323]*/, v[66:67] /*v[322:323]*/, s[30:31], v[118:119] /*v[374:375]*/ op_sel_hi:[1,0,0]
	v_mul_f32_e32 v134 /*v390*/, 0xbfb8aa3b, v151 /*v407*/
	v_pk_fma_f32 v[76:77] /*v[332:333]*/, v[70:71] /*v[326:327]*/, s[30:31], v[118:119] /*v[374:375]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[80:81] /*v[336:337]*/, v[72:73] /*v[328:329]*/, s[30:31], v[118:119] /*v[374:375]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[74:75] /*v[330:331]*/, v[68:69] /*v[324:325]*/, s[30:31], v[118:119] /*v[374:375]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v68 /*v324*/, v66 /*v322*/
	v_pk_fma_f32 v[130:131] /*v[386:387]*/, v[130:131] /*v[386:387]*/, s[30:31], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v70 /*v326*/, v67 /*v323*/
	v_exp_f32_e32 v78 /*v334*/, v76 /*v332*/
	v_pk_fma_f32 v[66:67] /*v[322:323]*/, v[94:95] /*v[350:351]*/, s[30:31], v[118:119] /*v[374:375]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v82 /*v338*/, v77 /*v333*/
	v_nop
	v_pk_fma_f32 v[76:77] /*v[332:333]*/, v[96:97] /*v[352:353]*/, s[30:31], v[118:119] /*v[374:375]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[132:133] /*v[388:389]*/, v[132:133] /*v[388:389]*/, s[30:31], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v79 /*v335*/, v130 /*v386*/
	v_pk_fma_f32 v[158:159] /*v[414:415]*/, v[158:159] /*v[414:415]*/, s[30:31], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v83 /*v339*/, v131 /*v387*/
	v_nop
	v_pk_fma_f32 v[130:131] /*v[386:387]*/, v[160:161] /*v[416:417]*/, s[30:31], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v84 /*v340*/, v80 /*v336*/
	v_exp_f32_e32 v90 /*v346*/, v81 /*v337*/
	v_nop
	v_pk_fma_f32 v[80:81] /*v[336:337]*/, v[98:99] /*v[354:355]*/, s[30:31], v[118:119] /*v[374:375]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v94 /*v350*/, v66 /*v322*/
	v_exp_f32_e32 v98 /*v354*/, v67 /*v323*/
	v_exp_f32_e32 v102 /*v358*/, v76 /*v332*/
	v_pk_fma_f32 v[66:67] /*v[322:323]*/, v[100:101] /*v[356:357]*/, s[30:31], v[118:119] /*v[374:375]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v106 /*v362*/, v77 /*v333*/
	v_nop
	v_pk_fma_f32 v[76:77] /*v[332:333]*/, v[166:167] /*v[422:423]*/, s[30:31], v[118:119] /*v[374:375]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v85 /*v341*/, v132 /*v388*/
	v_exp_f32_e32 v91 /*v347*/, v133 /*v389*/
	v_exp_f32_e32 v95 /*v351*/, v158 /*v414*/
	v_pk_fma_f32 v[132:133] /*v[388:389]*/, v[162:163] /*v[418:419]*/, s[30:31], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v99 /*v355*/, v159 /*v415*/
	v_exp_f32_e32 v103 /*v359*/, v130 /*v386*/
	v_pk_fma_f32 v[158:159] /*v[414:415]*/, v[164:165] /*v[420:421]*/, s[30:31], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v107 /*v363*/, v131 /*v387*/
	v_nop
	v_pk_fma_f32 v[130:131] /*v[386:387]*/, v[174:175] /*v[430:431]*/, s[30:31], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v110 /*v366*/, v80 /*v336*/
	v_exp_f32_e32 v114 /*v370*/, v81 /*v337*/
	v_nop
	v_pk_fma_f32 v[80:81] /*v[336:337]*/, v[168:169] /*v[424:425]*/, s[30:31], v[118:119] /*v[374:375]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[88:89] /*v[344:345]*/, v[170:171] /*v[426:427]*/, s[30:31], v[118:119] /*v[374:375]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[168:169] /*v[424:425]*/, v[126:127] /*v[382:383]*/, s[30:31], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[170:171] /*v[426:427]*/, v[128:129] /*v[384:385]*/, s[30:31], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v116 /*v372*/, v66 /*v322*/
	v_exp_f32_e32 v122 /*v378*/, v67 /*v323*/
	v_exp_f32_e32 v66 /*v322*/, v76 /*v332*/
	v_exp_f32_e32 v76 /*v332*/, v77 /*v333*/
	v_pk_fma_f32 v[96:97] /*v[352:353]*/, v[172:173] /*v[428:429]*/, s[30:31], v[118:119] /*v[374:375]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v111 /*v367*/, v132 /*v388*/
	v_exp_f32_e32 v115 /*v371*/, v133 /*v389*/
	v_exp_f32_e32 v117 /*v373*/, v158 /*v414*/
	v_pk_fma_f32 v[132:133] /*v[388:389]*/, v[176:177] /*v[432:433]*/, s[30:31], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v123 /*v379*/, v159 /*v415*/
	v_exp_f32_e32 v67 /*v323*/, v130 /*v386*/
	v_pk_fma_f32 v[158:159] /*v[414:415]*/, v[178:179] /*v[434:435]*/, s[30:31], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v77 /*v333*/, v131 /*v387*/
	v_nop
	v_pk_fma_f32 v[130:131] /*v[386:387]*/, v[180:181] /*v[436:437]*/, s[30:31], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v72 /*v328*/, v74 /*v330*/
	v_exp_f32_e32 v74 /*v330*/, v75 /*v331*/
	v_exp_f32_e32 v69 /*v325*/, v168 /*v424*/
	v_exp_f32_e32 v71 /*v327*/, v169 /*v425*/
	v_exp_f32_e32 v75 /*v331*/, v171 /*v427*/
	v_exp_f32_e32 v86 /*v342*/, v81 /*v337*/
	v_pk_fma_f32 v[104:105] /*v[360:361]*/, v[182:183] /*v[438:439]*/, s[30:31], v[118:119] /*v[374:375]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v92 /*v348*/, v89 /*v345*/
	v_pk_fma_f32 v[112:113] /*v[368:369]*/, v[184:185] /*v[440:441]*/, s[30:31], v[118:119] /*v[374:375]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v100 /*v356*/, v97 /*v353*/
	v_pk_fma_f32 v[120:121] /*v[376:377]*/, v[186:187] /*v[442:443]*/, s[30:31], v[118:119] /*v[374:375]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v81 /*v337*/, v132 /*v388*/
	v_exp_f32_e32 v87 /*v343*/, v133 /*v389*/
	v_exp_f32_e32 v89 /*v345*/, v158 /*v414*/
	v_pk_fma_f32 v[132:133] /*v[388:389]*/, v[190:191] /*v[446:447]*/, s[30:31], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v93 /*v349*/, v159 /*v415*/
	v_exp_f32_e32 v97 /*v353*/, v130 /*v386*/
	v_pk_fma_f32 v[158:159] /*v[414:415]*/, v[192:193] /*v[448:449]*/, s[30:31], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v101 /*v357*/, v131 /*v387*/
	v_nop
	v_pk_fma_f32 v[130:131] /*v[386:387]*/, v[194:195] /*v[450:451]*/, s[30:31], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v80 /*v336*/, v80 /*v336*/
	v_exp_f32_e32 v96 /*v352*/, v96 /*v352*/
	v_exp_f32_e32 v73 /*v329*/, v170 /*v426*/
	v_exp_f32_e32 v108 /*v364*/, v105 /*v361*/
	v_pk_fma_f32 v[166:167] /*v[422:423]*/, v[188:189] /*v[444:445]*/, s[30:31], v[118:119] /*v[374:375]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v118 /*v374*/, v113 /*v369*/
	v_exp_f32_e32 v124 /*v380*/, v121 /*v377*/
	v_exp_f32_e32 v105 /*v361*/, v132 /*v388*/
	v_exp_f32_e32 v109 /*v365*/, v133 /*v389*/
	v_exp_f32_e32 v113 /*v369*/, v158 /*v414*/
	v_pk_fma_f32 v[132:133] /*v[388:389]*/, v[196:197] /*v[452:453]*/, s[30:31], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v119 /*v375*/, v159 /*v415*/
	v_exp_f32_e32 v121 /*v377*/, v130 /*v386*/
	v_exp_f32_e32 v125 /*v381*/, v131 /*v387*/
	v_nop
	v_pk_add_f32 v[130:131] /*v[386:387]*/, v[68:69] /*v[324:325]*/, v[70:71] /*v[326:327]*/
	v_pk_add_f32 v[158:159] /*v[414:415]*/, v[74:75] /*v[330:331]*/, v[78:79] /*v[334:335]*/
	v_pk_add_f32 v[160:161] /*v[416:417]*/, v[98:99] /*v[354:355]*/, v[102:103] /*v[358:359]*/
	v_pk_add_f32 v[162:163] /*v[418:419]*/, v[110:111] /*v[366:367]*/, v[114:115] /*v[370:371]*/
	v_exp_f32_e32 v88 /*v344*/, v88 /*v344*/
	v_exp_f32_e32 v104 /*v360*/, v104 /*v360*/
	v_exp_f32_e32 v126 /*v382*/, v166 /*v422*/
	v_exp_f32_e32 v128 /*v384*/, v167 /*v423*/
	v_exp_f32_e32 v127 /*v383*/, v132 /*v388*/
	v_exp_f32_e32 v129 /*v385*/, v133 /*v389*/
	v_nop
	v_pk_add_f32 v[132:133] /*v[388:389]*/, v[84:85] /*v[340:341]*/, v[90:91] /*v[346:347]*/
	v_pk_add_f32 v[130:131] /*v[386:387]*/, v[72:73] /*v[328:329]*/, v[130:131] /*v[386:387]*/
	v_pk_add_f32 v[158:159] /*v[414:415]*/, v[82:83] /*v[338:339]*/, v[158:159] /*v[414:415]*/
	v_pk_add_f32 v[164:165] /*v[420:421]*/, v[122:123] /*v[378:379]*/, v[66:67] /*v[322:323]*/
	v_pk_add_f32 v[160:161] /*v[416:417]*/, v[106:107] /*v[362:363]*/, v[160:161] /*v[416:417]*/
	v_pk_add_f32 v[166:167] /*v[422:423]*/, v[80:81] /*v[336:337]*/, v[86:87] /*v[342:343]*/
	v_pk_add_f32 v[162:163] /*v[418:419]*/, v[116:117] /*v[372:373]*/, v[162:163] /*v[418:419]*/
	v_pk_add_f32 v[168:169] /*v[424:425]*/, v[92:93] /*v[348:349]*/, v[96:97] /*v[352:353]*/
	v_exp_f32_e32 v112 /*v368*/, v112 /*v368*/
	v_exp_f32_e32 v120 /*v376*/, v120 /*v376*/
	v_pk_add_f32 v[132:133] /*v[388:389]*/, v[94:95] /*v[350:351]*/, v[132:133] /*v[388:389]*/
	v_pk_add_f32 v[164:165] /*v[420:421]*/, v[76:77] /*v[332:333]*/, v[164:165] /*v[420:421]*/
	v_pk_add_f32 v[170:171] /*v[426:427]*/, v[104:105] /*v[360:361]*/, v[108:109] /*v[364:365]*/
	v_pk_add_f32 v[166:167] /*v[422:423]*/, v[88:89] /*v[344:345]*/, v[166:167] /*v[422:423]*/
	v_pk_add_f32 v[130:131] /*v[386:387]*/, v[130:131] /*v[386:387]*/, v[158:159] /*v[414:415]*/
	v_pk_add_f32 v[158:159] /*v[414:415]*/, v[100:101] /*v[356:357]*/, v[168:169] /*v[424:425]*/
	v_pk_add_f32 v[160:161] /*v[416:417]*/, v[160:161] /*v[416:417]*/, v[162:163] /*v[418:419]*/
	v_pk_add_f32 v[162:163] /*v[418:419]*/, v[112:113] /*v[368:369]*/, v[170:171] /*v[426:427]*/
	v_pk_add_f32 v[168:169] /*v[424:425]*/, v[118:119] /*v[374:375]*/, v[120:121] /*v[376:377]*/
	v_pk_add_f32 v[130:131] /*v[386:387]*/, v[132:133] /*v[388:389]*/, v[130:131] /*v[386:387]*/
	v_pk_add_f32 v[132:133] /*v[388:389]*/, v[166:167] /*v[422:423]*/, v[158:159] /*v[414:415]*/
	v_pk_add_f32 v[158:159] /*v[414:415]*/, v[164:165] /*v[420:421]*/, v[160:161] /*v[416:417]*/
	v_pk_add_f32 v[164:165] /*v[420:421]*/, v[126:127] /*v[382:383]*/, v[128:129] /*v[384:385]*/
	v_pk_add_f32 v[160:161] /*v[416:417]*/, v[124:125] /*v[380:381]*/, v[168:169] /*v[424:425]*/
	v_pk_add_f32 v[132:133] /*v[388:389]*/, v[162:163] /*v[418:419]*/, v[132:133] /*v[388:389]*/
	v_pk_add_f32 v[130:131] /*v[386:387]*/, v[130:131] /*v[386:387]*/, v[158:159] /*v[414:415]*/
	v_pk_add_f32 v[158:159] /*v[414:415]*/, v[164:165] /*v[420:421]*/, v[160:161] /*v[416:417]*/
	v_pk_add_f32 v[130:131] /*v[386:387]*/, v[132:133] /*v[388:389]*/, v[130:131] /*v[386:387]*/
	v_sub_f32_e32 v132 /*v388*/, v154 /*v410*/, v152 /*v408*/
	v_pk_add_f32 v[130:131] /*v[386:387]*/, v[158:159] /*v[414:415]*/, v[130:131] /*v[386:387]*/
	v_mul_f32_e32 v132 /*v388*/, 0x3fb8aa3b, v132 /*v388*/
	s_wait_alu depctr_vm_vsrc(0)
	v_dual_mov_b32 v144 /*v400*/, v130 /*v386*/ :: v_dual_mov_b32 v133 /*v389*/, v131 /*v387*/
	v_exp_f32_e32 v132 /*v388*/, v132 /*v388*/
	v_permlanex16_b32 v144 /*v400*/, v144 /*v400*/, s29, 0xfedcba98
	v_permlanex16_b32 v133 /*v389*/, v133 /*v389*/, s29, 0xfedcba98
	s_set_vgpr_msb 0x5500
	s_cbranch_vccz .LBB0_52
	s_set_vgpr_msb 4
	v_pk_mul_f32 v[126:127], v[126:127], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[124:125], v[124:125], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[122:123], v[122:123], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[120:121], v[120:121], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[118:119], v[118:119], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[116:117], v[116:117], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[114:115], v[114:115], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[112:113], v[112:113], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[110:111], v[110:111], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[108:109], v[108:109], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[106:107], v[106:107], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[104:105], v[104:105], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[102:103], v[102:103], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[100:101], v[100:101], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[98:99], v[98:99], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[96:97], v[96:97], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[94:95], v[94:95], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[92:93], v[92:93], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[90:91], v[90:91], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[88:89], v[88:89], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[86:87], v[86:87], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[84:85], v[84:85], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[82:83], v[82:83], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[80:81], v[80:81], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[78:79], v[78:79], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[76:77], v[76:77], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[74:75], v[74:75], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[72:73], v[72:73], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[70:71], v[70:71], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[68:69], v[68:69], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[66:67], v[66:67], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[64:65], v[64:65], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x400
.LBB0_52:
	s_set_vgpr_msb 0x45
	v_sub_f32_e32 v134 /*v390*/, v135 /*v391*/, v151 /*v407*/
	s_and_b32 s2, s3, exec_lo
	s_cselect_b32 s2, 1, 0
	s_cmp_lg_u32 s2, 1
	v_mul_f32_e32 v134 /*v390*/, 0x3fb8aa3b, v134 /*v390*/
	v_exp_f32_e32 v134 /*v390*/, v134 /*v390*/
	s_set_vgpr_msb 0x4500
	s_cbranch_scc1 .LBB0_47
	v_nop
	s_set_vgpr_msb 4
	v_pk_mul_f32 v[62:63], v[62:63], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[60:61], v[60:61], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59], v[58:59], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[56:57], v[56:57], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[54:55], v[54:55], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[52:53], v[52:53], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[50:51], v[50:51], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[48:49], v[48:49], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[46:47], v[46:47], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[44:45], v[44:45], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[42:43], v[42:43], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[40:41], v[40:41], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39], v[38:39], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[36:37], v[36:37], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35], v[34:35], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[32:33], v[32:33], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[30:31], v[30:31], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[28:29], v[28:29], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[26:27], v[26:27], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25], v[24:25], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[22:23], v[22:23], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21], v[20:21], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[18:19], v[18:19], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[16:17], v[16:17], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[14:15], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13], v[12:13], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[10:11], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[8:9], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[6:7], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[4:5], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[2:3], v[2:3], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[0:1], v[0:1], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x400
	s_branch .LBB0_47
.LBB0_54:
	s_set_vgpr_msb 0x41
	v_dual_mov_b32 v155 /*v411*/, v153 /*v409*/ :: v_dual_mov_b32 v156 /*v412*/, v144 /*v400*/
	v_dual_mov_b32 v151 /*v407*/, v135 /*v391*/ :: v_dual_mov_b32 v152 /*v408*/, v154 /*v410*/
	s_set_vgpr_msb 0x4100
.LBB0_55:
	s_cmp_ge_u32 s48, s10
	s_cbranch_scc1 .LBB0_64
	s_add_co_i32 s2, s19, -1
	s_mov_b32 s27, 0
	s_wait_alu depctr_vm_vsrc(0)
	s_set_vgpr_msb 0x44
	v_min_i32_e32 v144 /*v400*/, s2, v149 /*v405*/
	v_min_i32_e32 v149 /*v405*/, s2, v150 /*v406*/
	s_mov_b32 s11, s27
	s_mov_b32 s28, 1
	s_sub_co_i32 s7, 1, s60
	s_mov_b32 s24, 16
	s_mov_b32 s23, 0x800000
	s_mov_b32 s21, 0xffff0000
	s_mov_b32 s20, 0x7510000
	s_mov_b32 s36, 0xf510000
	s_mov_b32 s17, 0x76543210
	s_mov_b32 s16, 0x3fb8aa3b
	s_set_vgpr_msb 0x4400
	s_branch .LBB0_58
.LBB0_57:
	s_set_vgpr_msb 0x45
	v_cvt_pk_bf16_f32 v163 /*v419*/, v116 /*v372*/, v122 /*v378*/
	v_cvt_pk_bf16_f32 v162 /*v418*/, v108 /*v364*/, v114 /*v370*/
	v_cvt_pk_bf16_f32 v161 /*v417*/, v100 /*v356*/, v106 /*v362*/
	v_cvt_pk_bf16_f32 v160 /*v416*/, v92 /*v348*/, v98 /*v354*/
	v_cvt_pk_bf16_f32 v159 /*v415*/, v84 /*v340*/, v90 /*v346*/
	v_cvt_pk_bf16_f32 v158 /*v414*/, v76 /*v332*/, v82 /*v338*/
	v_cvt_pk_bf16_f32 v157 /*v413*/, v70 /*v326*/, v74 /*v330*/
	v_cvt_pk_bf16_f32 v156 /*v412*/, v66 /*v322*/, v68 /*v324*/
	v_cvt_pk_bf16_f32 v171 /*v427*/, v117 /*v373*/, v123 /*v379*/
	v_cvt_pk_bf16_f32 v170 /*v426*/, v109 /*v365*/, v115 /*v371*/
	v_cvt_pk_bf16_f32 v169 /*v425*/, v101 /*v357*/, v107 /*v363*/
	v_cvt_pk_bf16_f32 v168 /*v424*/, v93 /*v349*/, v99 /*v355*/
	v_cvt_pk_bf16_f32 v167 /*v423*/, v85 /*v341*/, v91 /*v347*/
	v_cvt_pk_bf16_f32 v166 /*v422*/, v77 /*v333*/, v83 /*v339*/
	v_cvt_pk_bf16_f32 v165 /*v421*/, v71 /*v327*/, v75 /*v331*/
	v_cvt_pk_bf16_f32 v164 /*v420*/, v67 /*v323*/, v69 /*v325*/
	s_set_vgpr_msb 0x4505
	s_wait_dscnt 0x1d
	v_wmma_f32_16x16x32_bf16 v[120:127], v[56:63] /*v[312:319]*/, v[156:163] /*v[412:419]*/, v[120:127]
	s_set_vgpr_msb 0x545
	v_cvt_pk_bf16_f32 v101 /*v357*/, v127 /*v383*/, v129 /*v385*/
	v_cvt_pk_bf16_f32 v100 /*v356*/, v121 /*v377*/, v125 /*v381*/
	v_cvt_pk_bf16_f32 v99 /*v355*/, v113 /*v369*/, v119 /*v375*/
	v_cvt_pk_bf16_f32 v98 /*v354*/, v105 /*v361*/, v111 /*v367*/
	v_cvt_pk_bf16_f32 v97 /*v353*/, v97 /*v353*/, v103 /*v359*/
	v_dual_fmac_f32 v152 /*v408*/, v132 /*v388*/, v64 /*v320*/ :: v_dual_fmac_f32 v154 /*v410*/, v134 /*v390*/, v65 /*v321*/
	s_set_vgpr_msb 0x4505
	v_wmma_f32_16x16x32_bf16 v[56:63], v[56:63] /*v[312:319]*/, v[164:171] /*v[420:427]*/, v[56:63]
	s_add_nc_u64 s[48:49], s[48:49], 1
	s_wait_alu depctr_vm_vsrc(0)
	s_set_vgpr_msb 0x545
	v_dual_mov_b32 v155 /*v411*/, v145 /*v401*/ :: v_dual_mov_b32 v145 /*v401*/, v153 /*v409*/
	v_cmp_ge_u64_e64 s2, s[48:49], s[10:11]
	v_dual_add_f32 v64 /*v320*/, v152 /*v408*/, v130 /*v386*/ :: v_dual_add_f32 v65 /*v321*/, v154 /*v410*/, v131 /*v387*/
	v_nop
	v_cvt_pk_bf16_f32 v63 /*v319*/, v126 /*v382*/, v128 /*v384*/
	s_set_vgpr_msb 0x4505
	s_wait_dscnt 0x1c
	v_wmma_f32_16x16x32_bf16 v[112:119], v[40:47] /*v[296:303]*/, v[156:163] /*v[412:419]*/, v[112:119]
	s_set_vgpr_msb 0x545
	v_cvt_pk_bf16_f32 v62 /*v318*/, v120 /*v376*/, v124 /*v380*/
	v_cvt_pk_bf16_f32 v61 /*v317*/, v112 /*v368*/, v118 /*v374*/
	v_cvt_pk_bf16_f32 v60 /*v316*/, v104 /*v360*/, v110 /*v366*/
	v_cvt_pk_bf16_f32 v59 /*v315*/, v96 /*v352*/, v102 /*v358*/
	v_cvt_pk_bf16_f32 v58 /*v314*/, v88 /*v344*/, v94 /*v350*/
	v_cvt_pk_bf16_f32 v57 /*v313*/, v80 /*v336*/, v86 /*v342*/
	v_cvt_pk_bf16_f32 v56 /*v312*/, v72 /*v328*/, v78 /*v334*/
	s_set_vgpr_msb 0x4505
	v_wmma_f32_16x16x32_bf16 v[48:55], v[40:47] /*v[296:303]*/, v[164:171] /*v[420:427]*/, v[48:55]
	s_set_vgpr_msb 0x545
	v_cvt_pk_bf16_f32 v96 /*v352*/, v89 /*v345*/, v95 /*v351*/
	v_cvt_pk_bf16_f32 v95 /*v351*/, v81 /*v337*/, v87 /*v343*/
	v_cvt_pk_bf16_f32 v94 /*v350*/, v73 /*v329*/, v79 /*v335*/
	v_dual_mov_b32 v151 /*v407*/, v135 /*v391*/ :: v_dual_mov_b32 v152 /*v408*/, v133 /*v389*/
	s_and_b32 vcc_lo, exec_lo, s2
	s_set_vgpr_msb 0x4505
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[104:111], v[24:31] /*v[280:287]*/, v[156:163] /*v[412:419]*/, v[104:111]
	v_wmma_f32_16x16x32_bf16 v[40:47], v[24:31] /*v[280:287]*/, v[164:171] /*v[420:427]*/, v[40:47]
	v_wmma_f32_16x16x32_bf16 v[96:103], v[8:15] /*v[264:271]*/, v[156:163] /*v[412:419]*/, v[96:103]
	v_wmma_f32_16x16x32_bf16 v[32:39], v[8:15] /*v[264:271]*/, v[164:171] /*v[420:427]*/, v[32:39]
	s_set_vgpr_msb 0x504
	v_wmma_f32_16x16x32_bf16 v[88:95], v[248:255], v[156:163] /*v[412:419]*/, v[88:95]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[248:255], v[164:171] /*v[420:427]*/, v[24:31]
	v_wmma_f32_16x16x32_bf16 v[80:87], v[232:239], v[156:163] /*v[412:419]*/, v[80:87]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[232:239], v[164:171] /*v[420:427]*/, v[16:23]
	v_wmma_f32_16x16x32_bf16 v[72:79], v[216:223], v[156:163] /*v[412:419]*/, v[72:79]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[216:223], v[164:171] /*v[420:427]*/, v[8:15]
	v_wmma_f32_16x16x32_bf16 v[64:71], v[200:207], v[156:163] /*v[412:419]*/, v[64:71]
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x441
	v_dual_mov_b32 v156 /*v412*/, v148 /*v404*/ :: v_dual_mov_b32 v148 /*v404*/, v150 /*v406*/
	s_set_vgpr_msb 0x4104
	v_wmma_f32_16x16x32_bf16 v[0:7], v[200:207], v[164:171] /*v[420:427]*/, v[0:7]
	s_set_vgpr_msb 0x405
	v_wmma_f32_16x16x32_bf16 v[120:127], v[48:55] /*v[304:311]*/, v[56:63] /*v[312:319]*/, v[120:127]
	v_wmma_f32_16x16x32_bf16 v[56:63], v[48:55] /*v[304:311]*/, v[94:101] /*v[350:357]*/, v[56:63]
	v_wmma_f32_16x16x32_bf16 v[112:119], v[32:39] /*v[288:295]*/, v[56:63] /*v[312:319]*/, v[112:119]
	v_wmma_f32_16x16x32_bf16 v[48:55], v[32:39] /*v[288:295]*/, v[94:101] /*v[350:357]*/, v[48:55]
	v_wmma_f32_16x16x32_bf16 v[104:111], v[16:23] /*v[272:279]*/, v[56:63] /*v[312:319]*/, v[104:111]
	v_wmma_f32_16x16x32_bf16 v[40:47], v[16:23] /*v[272:279]*/, v[94:101] /*v[350:357]*/, v[40:47]
	v_wmma_f32_16x16x32_bf16 v[96:103], v[0:7] /*v[256:263]*/, v[56:63] /*v[312:319]*/, v[96:103]
	v_wmma_f32_16x16x32_bf16 v[32:39], v[0:7] /*v[256:263]*/, v[94:101] /*v[350:357]*/, v[32:39]
	s_set_vgpr_msb 0x504
	v_wmma_f32_16x16x32_bf16 v[88:95], v[240:247], v[56:63] /*v[312:319]*/, v[88:95]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[240:247], v[94:101] /*v[350:357]*/, v[24:31]
	v_wmma_f32_16x16x32_bf16 v[80:87], v[224:231], v[56:63] /*v[312:319]*/, v[80:87]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[224:231], v[94:101] /*v[350:357]*/, v[16:23]
	v_wmma_f32_16x16x32_bf16 v[72:79], v[208:215], v[56:63] /*v[312:319]*/, v[72:79]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[208:215], v[94:101] /*v[350:357]*/, v[8:15]
	v_wmma_f32_16x16x32_bf16 v[64:71], v[192:199], v[56:63] /*v[312:319]*/, v[64:71]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[192:199], v[94:101] /*v[350:357]*/, v[0:7]
	s_set_vgpr_msb 0x400
	s_cbranch_vccnz .LBB0_65
.LBB0_58:
	s_set_vgpr_msb 0x41
	v_dual_mov_b32 v150 /*v406*/, v156 /*v412*/ :: v_dual_mov_b32 v153 /*v409*/, v155 /*v411*/
	s_add_co_i32 s2, s48, 1
	s_wait_tensorcnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	s_wait_alu depctr_va_vdst(0)
	ds_load_b128 v[56:59] /*v[312:315]*/, v148 /*v404*/
	ds_load_b128 v[60:63] /*v[316:319]*/, v148 /*v404*/ offset:32
	ds_load_b128 v[48:51] /*v[304:307]*/, v148 /*v404*/ offset:64
	ds_load_b128 v[52:55] /*v[308:311]*/, v148 /*v404*/ offset:96
	ds_load_b128 v[40:43] /*v[296:299]*/, v148 /*v404*/ offset:128
	ds_load_b128 v[44:47] /*v[300:303]*/, v148 /*v404*/ offset:160
	ds_load_b128 v[32:35] /*v[288:291]*/, v148 /*v404*/ offset:192
	ds_load_b128 v[36:39] /*v[292:295]*/, v148 /*v404*/ offset:224
	ds_load_b128 v[24:27] /*v[280:283]*/, v148 /*v404*/ offset:4352
	ds_load_b128 v[28:31] /*v[284:287]*/, v148 /*v404*/ offset:4384
	ds_load_b128 v[16:19] /*v[272:275]*/, v148 /*v404*/ offset:4416
	ds_load_b128 v[20:23] /*v[276:279]*/, v148 /*v404*/ offset:4448
	ds_load_b128 v[8:11] /*v[264:267]*/, v148 /*v404*/ offset:4480
	ds_load_b128 v[12:15] /*v[268:271]*/, v148 /*v404*/ offset:4512
	ds_load_b128 v[0:3] /*v[256:259]*/, v148 /*v404*/ offset:4544
	ds_load_b128 v[4:7] /*v[260:263]*/, v148 /*v404*/ offset:4576
	s_set_vgpr_msb 0x4101
	ds_load_b128 v[248:251], v148 /*v404*/ offset:8704
	ds_load_b128 v[252:255], v148 /*v404*/ offset:8736
	ds_load_b128 v[240:243], v148 /*v404*/ offset:8768
	ds_load_b128 v[244:247], v148 /*v404*/ offset:8800
	ds_load_b128 v[232:235], v148 /*v404*/ offset:8832
	ds_load_b128 v[236:239], v148 /*v404*/ offset:8864
	ds_load_b128 v[224:227], v148 /*v404*/ offset:8896
	ds_load_b128 v[228:231], v148 /*v404*/ offset:8928
	ds_load_b128 v[216:219], v148 /*v404*/ offset:13056
	ds_load_b128 v[220:223], v148 /*v404*/ offset:13088
	ds_load_b128 v[208:211], v148 /*v404*/ offset:13120
	ds_load_b128 v[212:215], v148 /*v404*/ offset:13152
	ds_load_b128 v[200:203], v148 /*v404*/ offset:13184
	ds_load_b128 v[204:207], v148 /*v404*/ offset:13216
	ds_load_b128 v[192:195], v148 /*v404*/ offset:13248
	ds_load_b128 v[196:199], v148 /*v404*/ offset:13280
	s_cmp_ge_i32 s2, s10
	s_set_vgpr_msb 0x100
	s_cbranch_scc1 .LBB0_60
	s_lshl_b32 s4, s2, 6
	s_add_co_i32 s6, s48, s7
	s_sub_co_i32 s22, s9, s4
	s_add_co_i32 s2, s4, s97
	s_set_vgpr_msb 0x41
	v_med3_i32 v66 /*v322*/, s22, 0, 64
	s_lshr_b32 s19, s6, 31
	s_ashr_i32 s3, s2, 31
	s_add_co_i32 s19, s6, s19
	s_mul_u64 s[4:5], s[2:3], s[64:65]
	v_readfirstlane_b32 s22, v66 /*v322*/
	s_mul_u64 s[2:3], s[2:3], s[62:63]
	s_and_b32 s19, s19, 0x7ffe
	s_lshl_b64 s[2:3], s[2:3], 1
	s_sub_co_i32 s6, s6, s19
	s_add_nc_u64 s[2:3], s[52:53], s[2:3]
	s_sub_co_i32 s19, s22, s80
	s_add_nc_u64 s[30:31], s[44:45], s[2:3]
	s_max_i32 s2, s19, 0
	s_lshl_b64 s[4:5], s[4:5], 1
	s_lshl_b32 s6, s6, 17
	s_lshl_b32 s2, s2, 16
	s_add_nc_u64 s[4:5], s[50:51], s[4:5]
	s_or_b32 s29, s81, s6
	s_bitset1_b32 s31, 31
	s_or_b32 s22, s2, 0x7fff
	s_mov_b32 s37, s21
	tensor_load_to_lds s[28:31], s[20:27]
	s_add_nc_u64 s[30:31], s[46:47], s[4:5]
	s_or_b32 s29, s78, s6
	s_bitset1_b32 s31, 31
	s_mov_b32 s38, s22
	s_mov_b32 s39, s23
	s_mov_b32 s40, s24
	s_mov_b32 s43, s27
	tensor_load_to_lds s[28:31], s[36:43]
	s_set_vgpr_msb 0x4100
.LBB0_60:
	s_set_vgpr_msb 0x41
	s_wait_dscnt 0x1e
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[322:329]*/, v[56:63] /*v[312:319]*/, v[128:135], 0
	v_wmma_f32_16x16x32_bf16 v[74:81] /*v[330:337]*/, v[56:63] /*v[312:319]*/, v[160:167], 0
	s_wait_dscnt 0x16
	v_wmma_f32_16x16x32_bf16 v[82:89] /*v[338:345]*/, v[24:31] /*v[280:287]*/, v[128:135], 0
	v_wmma_f32_16x16x32_bf16 v[90:97] /*v[346:353]*/, v[24:31] /*v[280:287]*/, v[160:167], 0
	s_set_vgpr_msb 0x4140
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x32_bf16 v[98:105] /*v[354:361]*/, v[248:255], v[128:135], 0
	v_wmma_f32_16x16x32_bf16 v[106:113] /*v[362:369]*/, v[248:255], v[160:167], 0
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x32_bf16 v[114:121] /*v[370:377]*/, v[216:223], v[128:135], 0
	v_wmma_f32_16x16x32_bf16 v[122:129] /*v[378:385]*/, v[216:223], v[160:167], 0
	s_set_vgpr_msb 0x4051
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[322:329]*/, v[48:55] /*v[304:311]*/, v[136:143], v[66:73] /*v[322:329]*/
	v_wmma_f32_16x16x32_bf16 v[74:81] /*v[330:337]*/, v[48:55] /*v[304:311]*/, v[168:175], v[74:81] /*v[330:337]*/
	v_wmma_f32_16x16x32_bf16 v[82:89] /*v[338:345]*/, v[16:23] /*v[272:279]*/, v[136:143], v[82:89] /*v[338:345]*/
	v_wmma_f32_16x16x32_bf16 v[90:97] /*v[346:353]*/, v[16:23] /*v[272:279]*/, v[168:175], v[90:97] /*v[346:353]*/
	s_set_vgpr_msb 0x5150
	v_wmma_f32_16x16x32_bf16 v[98:105] /*v[354:361]*/, v[240:247], v[136:143], v[98:105] /*v[354:361]*/
	v_wmma_f32_16x16x32_bf16 v[106:113] /*v[362:369]*/, v[240:247], v[168:175], v[106:113] /*v[362:369]*/
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x32_bf16 v[114:121] /*v[370:377]*/, v[208:215], v[136:143], v[114:121] /*v[370:377]*/
	v_wmma_f32_16x16x32_bf16 v[122:129] /*v[378:385]*/, v[208:215], v[168:175], v[122:129] /*v[378:385]*/
	s_set_vgpr_msb 0x5051
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[322:329]*/, v[40:47] /*v[296:303]*/, v[144:151], v[66:73] /*v[322:329]*/
	v_wmma_f32_16x16x32_bf16 v[74:81] /*v[330:337]*/, v[40:47] /*v[296:303]*/, v[176:183], v[74:81] /*v[330:337]*/
	v_wmma_f32_16x16x32_bf16 v[82:89] /*v[338:345]*/, v[8:15] /*v[264:271]*/, v[144:151], v[82:89] /*v[338:345]*/
	v_wmma_f32_16x16x32_bf16 v[90:97] /*v[346:353]*/, v[8:15] /*v[264:271]*/, v[176:183], v[90:97] /*v[346:353]*/
	s_set_vgpr_msb 0x5150
	v_wmma_f32_16x16x32_bf16 v[98:105] /*v[354:361]*/, v[232:239], v[144:151], v[98:105] /*v[354:361]*/
	v_wmma_f32_16x16x32_bf16 v[106:113] /*v[362:369]*/, v[232:239], v[176:183], v[106:113] /*v[362:369]*/
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[114:121] /*v[370:377]*/, v[200:207], v[144:151], v[114:121] /*v[370:377]*/
	v_wmma_f32_16x16x32_bf16 v[122:129] /*v[378:385]*/, v[200:207], v[176:183], v[122:129] /*v[378:385]*/
	s_set_vgpr_msb 0x5051
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[322:329]*/, v[32:39] /*v[288:295]*/, v[152:159], v[66:73] /*v[322:329]*/
	v_wmma_f32_16x16x32_bf16 v[74:81] /*v[330:337]*/, v[32:39] /*v[288:295]*/, v[184:191], v[74:81] /*v[330:337]*/
	v_wmma_f32_16x16x32_bf16 v[82:89] /*v[338:345]*/, v[0:7] /*v[256:263]*/, v[152:159], v[82:89] /*v[338:345]*/
	v_wmma_f32_16x16x32_bf16 v[90:97] /*v[346:353]*/, v[0:7] /*v[256:263]*/, v[184:191], v[90:97] /*v[346:353]*/
	s_set_vgpr_msb 0x5150
	v_wmma_f32_16x16x32_bf16 v[98:105] /*v[354:361]*/, v[224:231], v[152:159], v[98:105] /*v[354:361]*/
	v_wmma_f32_16x16x32_bf16 v[106:113] /*v[362:369]*/, v[224:231], v[184:191], v[106:113] /*v[362:369]*/
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[114:121] /*v[370:377]*/, v[192:199], v[152:159], v[114:121] /*v[370:377]*/
	v_wmma_f32_16x16x32_bf16 v[122:129] /*v[378:385]*/, v[192:199], v[184:191], v[122:129] /*v[378:385]*/
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x5041
	ds_load_tr16_b128 v[56:59] /*v[312:315]*/, v145 /*v401*/
	ds_load_tr16_b128 v[40:43] /*v[296:299]*/, v145 /*v401*/ offset:32
	ds_load_tr16_b128 v[60:63] /*v[316:319]*/, v145 /*v401*/ offset:4608
	ds_load_tr16_b128 v[44:47] /*v[300:303]*/, v145 /*v401*/ offset:4640
	ds_load_tr16_b128 v[48:51] /*v[304:307]*/, v145 /*v401*/ offset:9216
	ds_load_tr16_b128 v[32:35] /*v[288:291]*/, v145 /*v401*/ offset:9248
	ds_load_tr16_b128 v[52:55] /*v[308:311]*/, v145 /*v401*/ offset:13824
	ds_load_tr16_b128 v[36:39] /*v[292:295]*/, v145 /*v401*/ offset:13856
	ds_load_tr16_b128 v[24:27] /*v[280:283]*/, v145 /*v401*/ offset:64
	ds_load_tr16_b128 v[8:11] /*v[264:267]*/, v145 /*v401*/ offset:96
	ds_load_tr16_b128 v[28:31] /*v[284:287]*/, v145 /*v401*/ offset:4672
	ds_load_tr16_b128 v[12:15] /*v[268:271]*/, v145 /*v401*/ offset:4704
	ds_load_tr16_b128 v[16:19] /*v[272:275]*/, v145 /*v401*/ offset:9280
	ds_load_tr16_b128 v[0:3] /*v[256:259]*/, v145 /*v401*/ offset:9312
	ds_load_tr16_b128 v[20:23] /*v[276:279]*/, v145 /*v401*/ offset:13888
	ds_load_tr16_b128 v[4:7] /*v[260:263]*/, v145 /*v401*/ offset:13920
	s_set_vgpr_msb 0x4101
	ds_load_tr16_b128 v[248:251], v145 /*v401*/ offset:128
	ds_load_tr16_b128 v[232:235], v145 /*v401*/ offset:160
	ds_load_tr16_b128 v[252:255], v145 /*v401*/ offset:4736
	ds_load_tr16_b128 v[236:239], v145 /*v401*/ offset:4768
	ds_load_tr16_b128 v[240:243], v145 /*v401*/ offset:9344
	ds_load_tr16_b128 v[224:227], v145 /*v401*/ offset:9376
	ds_load_tr16_b128 v[244:247], v145 /*v401*/ offset:13952
	ds_load_tr16_b128 v[228:231], v145 /*v401*/ offset:13984
	ds_load_tr16_b128 v[216:219], v145 /*v401*/ offset:192
	ds_load_tr16_b128 v[200:203], v145 /*v401*/ offset:224
	ds_load_tr16_b128 v[220:223], v145 /*v401*/ offset:4800
	ds_load_tr16_b128 v[204:207], v145 /*v401*/ offset:4832
	ds_load_tr16_b128 v[208:211], v145 /*v401*/ offset:9408
	ds_load_tr16_b128 v[192:195], v145 /*v401*/ offset:9440
	ds_load_tr16_b128 v[212:215], v145 /*v401*/ offset:14016
	ds_load_tr16_b128 v[196:199], v145 /*v401*/ offset:14048
	s_set_vgpr_msb 0x155
	v_lshl_or_b32 v132 /*v388*/, s48, 6, v143 /*v399*/
	v_dual_add_nc_u32 v170 /*v426*/, 16, v132 /*v388*/ :: v_dual_bitop2_b32 v133 /*v389*/, 1, v132 /*v388*/ bitop3:0x54
	v_cmp_gt_i32_e32 vcc_lo, v132 /*v388*/, v144 /*v400*/
	v_cmp_lt_i32_e64 s2, v132 /*v388*/, v146 /*v402*/
	v_cmp_ge_i32_e64 s3, v132 /*v388*/, v144 /*v400*/
	v_dual_add_nc_u32 v171 /*v427*/, 17, v132 /*v388*/ :: v_dual_bitop2_b32 v134 /*v390*/, 2, v132 /*v388*/ bitop3:0x54
	v_cmp_lt_i32_e64 s4, v133 /*v389*/, v146 /*v402*/
	v_dual_add_nc_u32 v172 /*v428*/, 18, v132 /*v388*/ :: v_dual_bitop2_b32 v135 /*v391*/, 3, v132 /*v388*/ bitop3:0x54
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v134 /*v390*/, v144 /*v400*/
	v_cndmask_b32_e64 v66 /*v322*/, v66 /*v322*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v134 /*v390*/, v146 /*v402*/
	s_or_b32 s3, s4, s3
	v_dual_add_nc_u32 v173 /*v429*/, 19, v132 /*v388*/ :: v_dual_bitop2_b32 v164 /*v420*/, 4, v132 /*v388*/ bitop3:0x54
	v_cmp_gt_i32_e64 s5, v135 /*v391*/, v144 /*v400*/
	v_cndmask_b32_e64 v67 /*v323*/, v67 /*v323*/, 0xff800000, s3
	v_cmp_lt_i32_e64 s3, v135 /*v391*/, v146 /*v402*/
	v_dual_add_nc_u32 v174 /*v430*/, 20, v132 /*v388*/ :: v_dual_bitop2_b32 v167 /*v423*/, 5, v132 /*v388*/ bitop3:0x54
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v164 /*v420*/, v144 /*v400*/
	v_cndmask_b32_e64 v68 /*v324*/, v68 /*v324*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v164 /*v420*/, v146 /*v402*/
	s_or_b32 s3, s3, s5
	v_or_b32_e32 v168 /*v424*/, 6, v132 /*v388*/
	v_cndmask_b32_e64 v69 /*v325*/, v69 /*v325*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v167 /*v423*/, v144 /*v400*/
	v_cmp_lt_i32_e64 s4, v167 /*v423*/, v146 /*v402*/
	v_or_b32_e32 v169 /*v425*/, 7, v132 /*v388*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v168 /*v424*/, v144 /*v400*/
	v_cndmask_b32_e64 v70 /*v326*/, v70 /*v326*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v168 /*v424*/, v146 /*v402*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s4, v169 /*v425*/, v146 /*v402*/
	v_cndmask_b32_e64 v71 /*v327*/, v71 /*v327*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v169 /*v425*/, v144 /*v400*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v170 /*v426*/, v144 /*v400*/
	v_cndmask_b32_e64 v72 /*v328*/, v72 /*v328*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v170 /*v426*/, v146 /*v402*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s4, v171 /*v427*/, v146 /*v402*/
	v_cndmask_b32_e64 v73 /*v329*/, v73 /*v329*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v171 /*v427*/, v144 /*v400*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v172 /*v428*/, v144 /*v400*/
	v_cndmask_b32_e64 v82 /*v338*/, v82 /*v338*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v172 /*v428*/, v146 /*v402*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s4, v173 /*v429*/, v146 /*v402*/
	v_cndmask_b32_e64 v83 /*v339*/, v83 /*v339*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v173 /*v429*/, v144 /*v400*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v174 /*v430*/, v144 /*v400*/
	v_cndmask_b32_e64 v130 /*v386*/, v84 /*v340*/, 0xff800000, s2
	v_add_nc_u32_e32 v84 /*v340*/, 21, v132 /*v388*/
	v_cmp_lt_i32_e64 s2, v174 /*v430*/, v146 /*v402*/
	s_or_b32 s3, s4, s3
	v_dual_add_nc_u32 v176 /*v432*/, 23, v132 /*v388*/ :: v_dual_bitop2_b32 v177 /*v433*/, 32, v132 /*v388*/ bitop3:0x54
	v_cndmask_b32_e64 v131 /*v387*/, v85 /*v341*/, 0xff800000, s3
	v_add_nc_u32_e32 v85 /*v341*/, 22, v132 /*v388*/
	v_cmp_gt_i32_e64 s3, v84 /*v340*/, v144 /*v400*/
	v_cmp_lt_i32_e64 s4, v84 /*v340*/, v146 /*v402*/
	s_or_b32 s2, s2, vcc_lo
	v_dual_add_nc_u32 v186 /*v442*/, 48, v132 /*v388*/ :: v_dual_bitop2_b32 v179 /*v435*/, 33, v132 /*v388*/ bitop3:0x54
	v_cndmask_b32_e64 v86 /*v342*/, v86 /*v342*/, 0xff800000, s2
	v_cmp_gt_i32_e32 vcc_lo, v85 /*v341*/, v144 /*v400*/
	v_cmp_lt_i32_e64 s2, v85 /*v341*/, v146 /*v402*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s4, v176 /*v432*/, v146 /*v402*/
	v_cndmask_b32_e64 v87 /*v343*/, v87 /*v343*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v176 /*v432*/, v144 /*v400*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v177 /*v433*/, v144 /*v400*/
	v_cndmask_b32_e64 v88 /*v344*/, v88 /*v344*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v177 /*v433*/, v146 /*v402*/
	s_or_b32 s3, s4, s3
	v_dual_add_nc_u32 v187 /*v443*/, 49, v132 /*v388*/ :: v_dual_bitop2_b32 v180 /*v436*/, 34, v132 /*v388*/ bitop3:0x54
	v_cndmask_b32_e64 v89 /*v345*/, v89 /*v345*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v179 /*v435*/, v144 /*v400*/
	v_cmp_lt_i32_e64 s4, v179 /*v435*/, v146 /*v402*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v180 /*v436*/, v144 /*v400*/
	v_cndmask_b32_e64 v154 /*v410*/, v98 /*v354*/, 0xff800000, s2
	v_dual_add_nc_u32 v188 /*v444*/, 50, v132 /*v388*/ :: v_dual_bitop2_b32 v98 /*v354*/, 35, v132 /*v388*/ bitop3:0x54
	v_cmp_lt_i32_e64 s2, v180 /*v436*/, v146 /*v402*/
	s_or_b32 s3, s4, s3
	v_or_b32_e32 v185 /*v441*/, 39, v132 /*v388*/
	v_cndmask_b32_e64 v155 /*v411*/, v99 /*v355*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v98 /*v354*/, v144 /*v400*/
	v_or_b32_e32 v99 /*v355*/, 36, v132 /*v388*/
	v_cmp_lt_i32_e64 s4, v98 /*v354*/, v146 /*v402*/
	s_or_b32 s2, s2, vcc_lo
	v_add_nc_u32_e32 v191 /*v447*/, 55, v132 /*v388*/
	v_cndmask_b32_e64 v156 /*v412*/, v100 /*v356*/, 0xff800000, s2
	v_or_b32_e32 v100 /*v356*/, 37, v132 /*v388*/
	v_cmp_gt_i32_e32 vcc_lo, v99 /*v355*/, v144 /*v400*/
	v_cmp_lt_i32_e64 s2, v99 /*v355*/, v146 /*v402*/
	s_or_b32 s3, s4, s3
	v_cndmask_b32_e64 v157 /*v413*/, v101 /*v357*/, 0xff800000, s3
	v_or_b32_e32 v101 /*v357*/, 38, v132 /*v388*/
	v_cmp_gt_i32_e64 s3, v100 /*v356*/, v144 /*v400*/
	v_cmp_lt_i32_e64 s4, v100 /*v356*/, v146 /*v402*/
	s_or_b32 s2, s2, vcc_lo
	v_cndmask_b32_e64 v102 /*v358*/, v102 /*v358*/, 0xff800000, s2
	v_cmp_gt_i32_e32 vcc_lo, v101 /*v357*/, v144 /*v400*/
	v_cmp_lt_i32_e64 s2, v101 /*v357*/, v146 /*v402*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s4, v185 /*v441*/, v146 /*v402*/
	v_cndmask_b32_e64 v103 /*v359*/, v103 /*v359*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v185 /*v441*/, v144 /*v400*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v186 /*v442*/, v144 /*v400*/
	v_cndmask_b32_e64 v104 /*v360*/, v104 /*v360*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v186 /*v442*/, v146 /*v402*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s4, v187 /*v443*/, v146 /*v402*/
	v_cndmask_b32_e64 v105 /*v361*/, v105 /*v361*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v187 /*v443*/, v144 /*v400*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v188 /*v444*/, v144 /*v400*/
	v_cndmask_b32_e64 v158 /*v414*/, v114 /*v370*/, 0xff800000, s2
	v_add_nc_u32_e32 v114 /*v370*/, 51, v132 /*v388*/
	v_cmp_lt_i32_e64 s2, v188 /*v444*/, v146 /*v402*/
	s_or_b32 s3, s4, s3
	v_cndmask_b32_e64 v159 /*v415*/, v115 /*v371*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v114 /*v370*/, v144 /*v400*/
	v_cmp_lt_i32_e64 s4, v114 /*v370*/, v146 /*v402*/
	s_or_b32 s2, s2, vcc_lo
	v_add_nc_u32_e32 v115 /*v371*/, 52, v132 /*v388*/
	v_cndmask_b32_e64 v160 /*v416*/, v116 /*v372*/, 0xff800000, s2
	v_add_nc_u32_e32 v116 /*v372*/, 53, v132 /*v388*/
	s_or_b32 s2, s4, s3
	v_cndmask_b32_e64 v161 /*v417*/, v117 /*v373*/, 0xff800000, s2
	v_add_nc_u32_e32 v117 /*v373*/, 54, v132 /*v388*/
	v_cmp_gt_i32_e32 vcc_lo, v115 /*v371*/, v144 /*v400*/
	v_cmp_lt_i32_e64 s2, v115 /*v371*/, v146 /*v402*/
	v_cmp_gt_i32_e64 s3, v116 /*v372*/, v144 /*v400*/
	v_cmp_lt_i32_e64 s4, v116 /*v372*/, v146 /*v402*/
	v_cmp_gt_i32_e64 s5, v117 /*v373*/, v144 /*v400*/
	v_cmp_lt_i32_e64 s6, v117 /*v373*/, v146 /*v402*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v191 /*v447*/, v144 /*v400*/
	v_cndmask_b32_e64 v118 /*v374*/, v118 /*v374*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	v_cmp_gt_i32_e64 s3, v132 /*v388*/, v149 /*v405*/
	v_cndmask_b32_e64 v119 /*v375*/, v119 /*v375*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	v_cmp_lt_i32_e64 s4, v132 /*v388*/, v147 /*v403*/
	v_cndmask_b32_e64 v120 /*v376*/, v120 /*v376*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v191 /*v447*/, v146 /*v402*/
	v_cmp_ge_i32_e64 s5, v132 /*v388*/, v149 /*v405*/
	v_cmp_lt_i32_e64 s6, v133 /*v389*/, v147 /*v403*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v134 /*v390*/, v149 /*v405*/
	v_cndmask_b32_e64 v121 /*v377*/, v121 /*v377*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	v_cmp_gt_i32_e64 s3, v135 /*v391*/, v149 /*v405*/
	v_cndmask_b32_e64 v162 /*v418*/, v74 /*v330*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	v_cmp_lt_i32_e64 s4, v135 /*v391*/, v147 /*v403*/
	v_cndmask_b32_e64 v163 /*v419*/, v75 /*v331*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v134 /*v390*/, v147 /*v403*/
	v_cmp_gt_i32_e64 s5, v164 /*v420*/, v149 /*v405*/
	v_cmp_lt_i32_e64 s6, v164 /*v420*/, v147 /*v403*/
	v_max_num_f32_e32 v74 /*v330*/, v66 /*v322*/, v67 /*v323*/
	v_max_num_f32_e32 v75 /*v331*/, v162 /*v418*/, v163 /*v419*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v167 /*v423*/, v149 /*v405*/
	v_cndmask_b32_e64 v164 /*v420*/, v76 /*v332*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	v_cmp_gt_i32_e64 s3, v168 /*v424*/, v149 /*v405*/
	v_cndmask_b32_e64 v165 /*v421*/, v77 /*v333*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	v_cmp_lt_i32_e64 s4, v168 /*v424*/, v147 /*v403*/
	v_cndmask_b32_e64 v166 /*v422*/, v78 /*v334*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v167 /*v423*/, v147 /*v403*/
	v_cmp_gt_i32_e64 s5, v169 /*v425*/, v149 /*v405*/
	v_cmp_lt_i32_e64 s6, v169 /*v425*/, v147 /*v403*/
	v_max3_num_f32 v76 /*v332*/, v69 /*v325*/, v70 /*v326*/, v71 /*v327*/
	v_max3_num_f32 v78 /*v334*/, v72 /*v328*/, v73 /*v329*/, v82 /*v338*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v170 /*v426*/, v149 /*v405*/
	v_cndmask_b32_e64 v167 /*v423*/, v79 /*v335*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	v_cmp_gt_i32_e64 s3, v171 /*v427*/, v149 /*v405*/
	v_cndmask_b32_e64 v168 /*v424*/, v80 /*v336*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	v_cmp_lt_i32_e64 s4, v171 /*v427*/, v147 /*v403*/
	v_cndmask_b32_e64 v169 /*v425*/, v81 /*v337*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v170 /*v426*/, v147 /*v403*/
	v_cmp_gt_i32_e64 s5, v172 /*v428*/, v149 /*v405*/
	v_cmp_lt_i32_e64 s6, v172 /*v428*/, v147 /*v403*/
	v_max3_num_f32 v80 /*v336*/, v83 /*v339*/, v130 /*v386*/, v131 /*v387*/
	v_max3_num_f32 v77 /*v333*/, v165 /*v421*/, v166 /*v422*/, v167 /*v423*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v173 /*v429*/, v149 /*v405*/
	v_cndmask_b32_e64 v170 /*v426*/, v90 /*v346*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	v_cmp_gt_i32_e64 s3, v174 /*v430*/, v149 /*v405*/
	v_cndmask_b32_e64 v171 /*v427*/, v91 /*v347*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	v_cmp_lt_i32_e64 s4, v174 /*v430*/, v147 /*v403*/
	v_cndmask_b32_e64 v172 /*v428*/, v92 /*v348*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v173 /*v429*/, v147 /*v403*/
	v_cmp_gt_i32_e64 s5, v84 /*v340*/, v149 /*v405*/
	v_cmp_lt_i32_e64 s6, v84 /*v340*/, v147 /*v403*/
	v_max3_num_f32 v84 /*v340*/, v86 /*v342*/, v87 /*v343*/, v88 /*v344*/
	v_max3_num_f32 v90 /*v346*/, v89 /*v345*/, v154 /*v410*/, v155 /*v411*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v85 /*v341*/, v149 /*v405*/
	v_cndmask_b32_e64 v173 /*v429*/, v93 /*v349*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	v_cmp_gt_i32_e64 s3, v176 /*v432*/, v149 /*v405*/
	v_cndmask_b32_e64 v174 /*v430*/, v94 /*v350*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	v_cmp_lt_i32_e64 s4, v176 /*v432*/, v147 /*v403*/
	v_cndmask_b32_e64 v175 /*v431*/, v95 /*v351*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v85 /*v341*/, v147 /*v403*/
	v_cmp_gt_i32_e64 s5, v177 /*v433*/, v149 /*v405*/
	v_cmp_lt_i32_e64 s6, v177 /*v433*/, v147 /*v403*/
	v_max3_num_f32 v81 /*v337*/, v171 /*v427*/, v172 /*v428*/, v173 /*v429*/
	v_max3_num_f32 v92 /*v348*/, v156 /*v412*/, v157 /*v413*/, v102 /*v358*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v179 /*v435*/, v149 /*v405*/
	v_cndmask_b32_e64 v176 /*v432*/, v96 /*v352*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	v_cmp_gt_i32_e64 s3, v180 /*v436*/, v149 /*v405*/
	v_cndmask_b32_e64 v177 /*v433*/, v97 /*v353*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	v_cmp_lt_i32_e64 s4, v180 /*v436*/, v147 /*v403*/
	v_cndmask_b32_e64 v178 /*v434*/, v106 /*v362*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v179 /*v435*/, v147 /*v403*/
	v_cmp_gt_i32_e64 s5, v98 /*v354*/, v149 /*v405*/
	v_cmp_lt_i32_e64 s6, v98 /*v354*/, v147 /*v403*/
	v_max3_num_f32 v85 /*v341*/, v174 /*v430*/, v175 /*v431*/, v176 /*v432*/
	v_max3_num_f32 v93 /*v349*/, v103 /*v359*/, v104 /*v360*/, v105 /*v361*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v99 /*v355*/, v149 /*v405*/
	v_cndmask_b32_e64 v179 /*v435*/, v107 /*v363*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	v_cmp_gt_i32_e64 s3, v100 /*v356*/, v149 /*v405*/
	v_cndmask_b32_e64 v180 /*v436*/, v108 /*v364*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	v_cmp_lt_i32_e64 s4, v100 /*v356*/, v147 /*v403*/
	v_cndmask_b32_e64 v181 /*v437*/, v109 /*v365*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v99 /*v355*/, v147 /*v403*/
	v_cmp_gt_i32_e64 s5, v101 /*v357*/, v149 /*v405*/
	v_cmp_lt_i32_e64 s6, v101 /*v357*/, v147 /*v403*/
	v_max3_num_f32 v91 /*v347*/, v177 /*v433*/, v178 /*v434*/, v179 /*v435*/
	v_max3_num_f32 v94 /*v350*/, v158 /*v414*/, v159 /*v415*/, v160 /*v416*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v185 /*v441*/, v149 /*v405*/
	v_cndmask_b32_e64 v182 /*v438*/, v110 /*v366*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	v_cmp_gt_i32_e64 s3, v186 /*v442*/, v149 /*v405*/
	v_cndmask_b32_e64 v183 /*v439*/, v111 /*v367*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	v_cmp_lt_i32_e64 s4, v186 /*v442*/, v147 /*v403*/
	v_cndmask_b32_e64 v184 /*v440*/, v112 /*v368*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v185 /*v441*/, v147 /*v403*/
	v_cmp_gt_i32_e64 s5, v187 /*v443*/, v149 /*v405*/
	v_cmp_lt_i32_e64 s6, v187 /*v443*/, v147 /*v403*/
	v_max3_num_f32 v95 /*v351*/, v161 /*v417*/, v118 /*v374*/, v119 /*v375*/
	v_max3_num_f32 v74 /*v330*/, v74 /*v330*/, v68 /*v324*/, v76 /*v332*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v188 /*v444*/, v149 /*v405*/
	v_cndmask_b32_e64 v185 /*v441*/, v113 /*v369*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	v_cmp_gt_i32_e64 s3, v114 /*v370*/, v149 /*v405*/
	v_cndmask_b32_e64 v186 /*v442*/, v122 /*v378*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	v_cmp_lt_i32_e64 s4, v114 /*v370*/, v147 /*v403*/
	v_cndmask_b32_e64 v187 /*v443*/, v123 /*v379*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v188 /*v444*/, v147 /*v403*/
	v_cmp_gt_i32_e64 s5, v115 /*v371*/, v149 /*v405*/
	v_cmp_lt_i32_e64 s6, v115 /*v371*/, v147 /*v403*/
	v_max3_num_f32 v76 /*v332*/, v80 /*v336*/, v84 /*v340*/, v90 /*v346*/
	v_max3_num_f32 v79 /*v335*/, v168 /*v424*/, v169 /*v425*/, v170 /*v426*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v116 /*v372*/, v149 /*v405*/
	v_cndmask_b32_e64 v188 /*v444*/, v124 /*v380*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	v_cmp_gt_i32_e64 s3, v117 /*v373*/, v149 /*v405*/
	v_cndmask_b32_e64 v189 /*v445*/, v125 /*v381*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	v_cmp_lt_i32_e64 s4, v117 /*v373*/, v147 /*v403*/
	v_cndmask_b32_e64 v190 /*v446*/, v126 /*v382*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v116 /*v372*/, v147 /*v403*/
	v_cmp_gt_i32_e64 s5, v191 /*v447*/, v149 /*v405*/
	v_cmp_lt_i32_e64 s6, v191 /*v447*/, v147 /*v403*/
	v_max3_num_f32 v80 /*v336*/, v180 /*v436*/, v181 /*v437*/, v182 /*v438*/
	v_max3_num_f32 v84 /*v340*/, v183 /*v439*/, v184 /*v440*/, v185 /*v441*/
	s_or_b32 s2, s2, vcc_lo
	v_max3_num_f32 v90 /*v346*/, v92 /*v348*/, v93 /*v349*/, v94 /*v350*/
	v_cndmask_b32_e64 v191 /*v447*/, v127 /*v383*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	v_max3_num_f32 v92 /*v348*/, v95 /*v351*/, v120 /*v376*/, v121 /*v377*/
	v_cndmask_b32_e64 v192 /*v448*/, v128 /*v384*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	v_max3_num_f32 v74 /*v330*/, v74 /*v330*/, v78 /*v334*/, v76 /*v332*/
	v_cndmask_b32_e64 v193 /*v449*/, v129 /*v385*/, 0xff800000, s2
	v_max3_num_f32 v76 /*v332*/, v186 /*v442*/, v187 /*v443*/, v188 /*v444*/
	v_max3_num_f32 v78 /*v334*/, v189 /*v445*/, v190 /*v446*/, v191 /*v447*/
	v_max3_num_f32 v75 /*v331*/, v75 /*v331*/, v164 /*v420*/, v77 /*v333*/
	v_max3_num_f32 v77 /*v333*/, v81 /*v337*/, v85 /*v341*/, v91 /*v347*/
	v_max3_num_f32 v74 /*v330*/, v74 /*v330*/, v90 /*v346*/, v92 /*v348*/
	v_max3_num_f32 v76 /*v332*/, v80 /*v336*/, v84 /*v340*/, v76 /*v332*/
	v_max3_num_f32 v78 /*v334*/, v78 /*v334*/, v192 /*v448*/, v193 /*v449*/
	v_max3_num_f32 v75 /*v331*/, v75 /*v331*/, v79 /*v335*/, v77 /*v333*/
	v_max3_num_f32 v75 /*v331*/, v75 /*v331*/, v76 /*v332*/, v78 /*v334*/
	v_dual_mov_b32 v77 /*v333*/, v74 /*v330*/ :: v_dual_mov_b32 v76 /*v332*/, v75 /*v331*/
	v_permlanex16_b32 v77 /*v333*/, v77 /*v333*/, s17, 0xfedcba98
	v_permlanex16_b32 v76 /*v332*/, v76 /*v332*/, s17, 0xfedcba98
	v_dual_max_num_f32 v74 /*v330*/, v74 /*v330*/, v77 /*v333*/ :: v_dual_max_num_f32 v75 /*v331*/, v75 /*v331*/, v76 /*v332*/
	v_dual_sub_f32 v77 /*v333*/, v74 /*v330*/, v152 /*v408*/ :: v_dual_max_num_f32 v74 /*v330*/, v152 /*v408*/, v74 /*v330*/
	v_sub_f32_e32 v76 /*v332*/, v75 /*v331*/, v151 /*v407*/
	v_cmp_lt_f32_e32 vcc_lo, 0x41000000, v77 /*v333*/
	s_cmp_eq_u32 vcc_lo, 0
	s_cselect_b32 s2, -1, 0
	v_dual_cndmask_b32 v133 /*v389*/, v74 /*v330*/, v152 /*v408*/, s2 :: v_dual_max_num_f32 v74 /*v330*/, v151 /*v407*/, v75 /*v331*/
	v_cmp_lt_f32_e64 s2, 0x41000000, v76 /*v332*/
	v_mul_f32_e32 v124 /*v380*/, 0xbfb8aa3b, v133 /*v389*/
	s_cmp_lg_u32 s2, 0
	s_cselect_b32 s3, -1, 0
	s_cmp_eq_u32 s2, 0
	v_pk_fma_f32 v[72:73] /*v[328:329]*/, v[72:73] /*v[328:329]*/, s[16:17], v[124:125] /*v[380:381]*/ op_sel_hi:[1,0,0]
	s_cselect_b32 s2, -1, 0
	v_pk_fma_f32 v[80:81] /*v[336:337]*/, v[130:131] /*v[386:387]*/, s[16:17], v[124:125] /*v[380:381]*/ op_sel_hi:[1,0,0]
	v_cndmask_b32_e64 v135 /*v391*/, v74 /*v330*/, v151 /*v407*/, s2
	v_pk_fma_f32 v[66:67] /*v[322:323]*/, v[66:67] /*v[322:323]*/, s[16:17], v[124:125] /*v[380:381]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[74:75] /*v[330:331]*/, v[68:69] /*v[324:325]*/, s[16:17], v[124:125] /*v[380:381]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[76:77] /*v[332:333]*/, v[70:71] /*v[326:327]*/, s[16:17], v[124:125] /*v[380:381]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v84 /*v340*/, v72 /*v328*/
	v_mul_f32_e32 v132 /*v388*/, 0xbfb8aa3b, v135 /*v391*/
	v_exp_f32_e32 v90 /*v346*/, v73 /*v329*/
	v_nop
	v_pk_fma_f32 v[72:73] /*v[328:329]*/, v[86:87] /*v[342:343]*/, s[16:17], v[124:125] /*v[380:381]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v100 /*v356*/, v80 /*v336*/
	v_exp_f32_e32 v106 /*v362*/, v81 /*v337*/
	v_nop
	v_pk_fma_f32 v[80:81] /*v[336:337]*/, v[154:155] /*v[410:411]*/, s[16:17], v[124:125] /*v[380:381]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[86:87] /*v[342:343]*/, v[156:157] /*v[412:413]*/, s[16:17], v[124:125] /*v[380:381]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[130:131] /*v[386:387]*/, v[162:163] /*v[418:419]*/, s[16:17], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[154:155] /*v[410:411]*/, v[164:165] /*v[420:421]*/, s[16:17], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[156:157] /*v[412:413]*/, v[166:167] /*v[422:423]*/, s[16:17], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v68 /*v324*/, v67 /*v323*/
	v_exp_f32_e32 v70 /*v326*/, v74 /*v330*/
	v_exp_f32_e32 v74 /*v330*/, v75 /*v331*/
	v_pk_fma_f32 v[78:79] /*v[334:335]*/, v[82:83] /*v[338:339]*/, s[16:17], v[124:125] /*v[380:381]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v82 /*v338*/, v77 /*v333*/
	v_exp_f32_e32 v67 /*v323*/, v130 /*v386*/
	v_exp_f32_e32 v69 /*v325*/, v131 /*v387*/
	v_exp_f32_e32 v71 /*v327*/, v154 /*v410*/
	v_pk_fma_f32 v[130:131] /*v[386:387]*/, v[168:169] /*v[424:425]*/, s[16:17], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v75 /*v331*/, v155 /*v411*/
	v_exp_f32_e32 v77 /*v333*/, v156 /*v412*/
	v_pk_fma_f32 v[154:155] /*v[410:411]*/, v[170:171] /*v[426:427]*/, s[16:17], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v83 /*v339*/, v157 /*v413*/
	v_nop
	v_pk_fma_f32 v[156:157] /*v[412:413]*/, v[172:173] /*v[428:429]*/, s[16:17], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v92 /*v348*/, v78 /*v334*/
	v_exp_f32_e32 v98 /*v354*/, v79 /*v335*/
	v_nop
	v_pk_fma_f32 v[78:79] /*v[334:335]*/, v[88:89] /*v[344:345]*/, s[16:17], v[124:125] /*v[380:381]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v85 /*v341*/, v130 /*v386*/
	v_exp_f32_e32 v91 /*v347*/, v131 /*v387*/
	v_exp_f32_e32 v93 /*v349*/, v154 /*v410*/
	v_pk_fma_f32 v[130:131] /*v[386:387]*/, v[174:175] /*v[430:431]*/, s[16:17], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v99 /*v355*/, v155 /*v411*/
	v_exp_f32_e32 v101 /*v357*/, v156 /*v412*/
	v_pk_fma_f32 v[154:155] /*v[410:411]*/, v[176:177] /*v[432:433]*/, s[16:17], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v107 /*v363*/, v157 /*v413*/
	v_nop
	v_pk_fma_f32 v[156:157] /*v[412:413]*/, v[178:179] /*v[434:435]*/, s[16:17], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v114 /*v370*/, v73 /*v329*/
	v_exp_f32_e32 v122 /*v378*/, v79 /*v335*/
	v_pk_fma_f32 v[88:89] /*v[344:345]*/, v[102:103] /*v[358:359]*/, s[16:17], v[124:125] /*v[380:381]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[96:97] /*v[352:353]*/, v[104:105] /*v[360:361]*/, s[16:17], v[124:125] /*v[380:381]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v109 /*v365*/, v130 /*v386*/
	v_exp_f32_e32 v115 /*v371*/, v131 /*v387*/
	v_exp_f32_e32 v117 /*v373*/, v154 /*v410*/
	v_pk_fma_f32 v[130:131] /*v[386:387]*/, v[180:181] /*v[436:437]*/, s[16:17], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v123 /*v379*/, v155 /*v411*/
	v_exp_f32_e32 v73 /*v329*/, v156 /*v412*/
	v_pk_fma_f32 v[154:155] /*v[410:411]*/, v[182:183] /*v[438:439]*/, s[16:17], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v79 /*v335*/, v157 /*v413*/
	v_nop
	v_pk_fma_f32 v[156:157] /*v[412:413]*/, v[184:185] /*v[440:441]*/, s[16:17], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v66 /*v322*/, v66 /*v322*/
	v_exp_f32_e32 v76 /*v332*/, v76 /*v332*/
	v_exp_f32_e32 v108 /*v364*/, v72 /*v328*/
	v_exp_f32_e32 v116 /*v372*/, v78 /*v334*/
	v_exp_f32_e32 v72 /*v328*/, v80 /*v336*/
	v_exp_f32_e32 v78 /*v334*/, v81 /*v337*/
	v_exp_f32_e32 v80 /*v336*/, v86 /*v342*/
	v_exp_f32_e32 v86 /*v342*/, v87 /*v343*/
	v_pk_fma_f32 v[104:105] /*v[360:361]*/, v[158:159] /*v[414:415]*/, s[16:17], v[124:125] /*v[380:381]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v94 /*v350*/, v89 /*v345*/
	v_pk_fma_f32 v[112:113] /*v[368:369]*/, v[160:161] /*v[416:417]*/, s[16:17], v[124:125] /*v[380:381]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v102 /*v358*/, v97 /*v353*/
	v_exp_f32_e32 v81 /*v337*/, v130 /*v386*/
	v_exp_f32_e32 v87 /*v343*/, v131 /*v387*/
	v_exp_f32_e32 v89 /*v345*/, v154 /*v410*/
	v_pk_fma_f32 v[130:131] /*v[386:387]*/, v[186:187] /*v[442:443]*/, s[16:17], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v95 /*v351*/, v155 /*v411*/
	v_exp_f32_e32 v97 /*v353*/, v156 /*v412*/
	v_pk_fma_f32 v[154:155] /*v[410:411]*/, v[188:189] /*v[444:445]*/, s[16:17], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v103 /*v359*/, v157 /*v413*/
	v_nop
	v_pk_fma_f32 v[156:157] /*v[412:413]*/, v[190:191] /*v[446:447]*/, s[16:17], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v96 /*v352*/, v96 /*v352*/
	v_pk_fma_f32 v[126:127] /*v[382:383]*/, v[118:119] /*v[374:375]*/, s[16:17], v[124:125] /*v[380:381]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v110 /*v366*/, v105 /*v361*/
	v_pk_fma_f32 v[128:129] /*v[384:385]*/, v[120:121] /*v[376:377]*/, s[16:17], v[124:125] /*v[380:381]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v118 /*v374*/, v113 /*v369*/
	v_exp_f32_e32 v105 /*v361*/, v130 /*v386*/
	v_exp_f32_e32 v111 /*v367*/, v131 /*v387*/
	v_exp_f32_e32 v113 /*v369*/, v154 /*v410*/
	v_pk_fma_f32 v[130:131] /*v[386:387]*/, v[192:193] /*v[448:449]*/, s[16:17], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v119 /*v375*/, v155 /*v411*/
	v_exp_f32_e32 v121 /*v377*/, v156 /*v412*/
	v_exp_f32_e32 v125 /*v381*/, v157 /*v413*/
	v_pk_add_f32 v[154:155] /*v[410:411]*/, v[66:67] /*v[322:323]*/, v[68:69] /*v[324:325]*/
	v_pk_add_f32 v[156:157] /*v[412:413]*/, v[74:75] /*v[330:331]*/, v[76:77] /*v[332:333]*/
	v_pk_add_f32 v[158:159] /*v[414:415]*/, v[98:99] /*v[354:355]*/, v[100:101] /*v[356:357]*/
	v_pk_add_f32 v[160:161] /*v[416:417]*/, v[108:109] /*v[364:365]*/, v[114:115] /*v[370:371]*/
	v_exp_f32_e32 v88 /*v344*/, v88 /*v344*/
	v_exp_f32_e32 v104 /*v360*/, v104 /*v360*/
	v_exp_f32_e32 v120 /*v376*/, v126 /*v382*/
	v_exp_f32_e32 v124 /*v380*/, v127 /*v383*/
	v_exp_f32_e32 v126 /*v382*/, v128 /*v384*/
	v_exp_f32_e32 v128 /*v384*/, v129 /*v385*/
	v_exp_f32_e32 v127 /*v383*/, v130 /*v386*/
	v_exp_f32_e32 v129 /*v385*/, v131 /*v387*/
	v_nop
	v_pk_add_f32 v[130:131] /*v[386:387]*/, v[84:85] /*v[340:341]*/, v[90:91] /*v[346:347]*/
	v_pk_add_f32 v[154:155] /*v[410:411]*/, v[70:71] /*v[326:327]*/, v[154:155] /*v[410:411]*/
	v_pk_add_f32 v[156:157] /*v[412:413]*/, v[82:83] /*v[338:339]*/, v[156:157] /*v[412:413]*/
	v_pk_add_f32 v[162:163] /*v[418:419]*/, v[122:123] /*v[378:379]*/, v[72:73] /*v[328:329]*/
	v_pk_add_f32 v[158:159] /*v[414:415]*/, v[106:107] /*v[362:363]*/, v[158:159] /*v[414:415]*/
	v_pk_add_f32 v[164:165] /*v[420:421]*/, v[80:81] /*v[336:337]*/, v[86:87] /*v[342:343]*/
	v_pk_add_f32 v[160:161] /*v[416:417]*/, v[116:117] /*v[372:373]*/, v[160:161] /*v[416:417]*/
	v_pk_add_f32 v[166:167] /*v[422:423]*/, v[94:95] /*v[350:351]*/, v[96:97] /*v[352:353]*/
	v_exp_f32_e32 v112 /*v368*/, v112 /*v368*/
	v_pk_add_f32 v[130:131] /*v[386:387]*/, v[92:93] /*v[348:349]*/, v[130:131] /*v[386:387]*/
	v_pk_add_f32 v[162:163] /*v[418:419]*/, v[78:79] /*v[334:335]*/, v[162:163] /*v[418:419]*/
	v_pk_add_f32 v[168:169] /*v[424:425]*/, v[104:105] /*v[360:361]*/, v[110:111] /*v[366:367]*/
	v_pk_add_f32 v[164:165] /*v[420:421]*/, v[88:89] /*v[344:345]*/, v[164:165] /*v[420:421]*/
	v_pk_add_f32 v[154:155] /*v[410:411]*/, v[154:155] /*v[410:411]*/, v[156:157] /*v[412:413]*/
	v_pk_add_f32 v[156:157] /*v[412:413]*/, v[102:103] /*v[358:359]*/, v[166:167] /*v[422:423]*/
	v_pk_add_f32 v[158:159] /*v[414:415]*/, v[158:159] /*v[414:415]*/, v[160:161] /*v[416:417]*/
	v_pk_add_f32 v[160:161] /*v[416:417]*/, v[112:113] /*v[368:369]*/, v[168:169] /*v[424:425]*/
	v_pk_add_f32 v[166:167] /*v[422:423]*/, v[118:119] /*v[374:375]*/, v[120:121] /*v[376:377]*/
	v_pk_add_f32 v[130:131] /*v[386:387]*/, v[130:131] /*v[386:387]*/, v[154:155] /*v[410:411]*/
	v_pk_add_f32 v[154:155] /*v[410:411]*/, v[164:165] /*v[420:421]*/, v[156:157] /*v[412:413]*/
	v_pk_add_f32 v[156:157] /*v[412:413]*/, v[162:163] /*v[418:419]*/, v[158:159] /*v[414:415]*/
	v_pk_add_f32 v[162:163] /*v[418:419]*/, v[126:127] /*v[382:383]*/, v[128:129] /*v[384:385]*/
	v_pk_add_f32 v[158:159] /*v[414:415]*/, v[124:125] /*v[380:381]*/, v[166:167] /*v[422:423]*/
	v_sub_f32_e32 v132 /*v388*/, v152 /*v408*/, v133 /*v389*/
	v_pk_add_f32 v[154:155] /*v[410:411]*/, v[160:161] /*v[416:417]*/, v[154:155] /*v[410:411]*/
	v_pk_add_f32 v[130:131] /*v[386:387]*/, v[130:131] /*v[386:387]*/, v[156:157] /*v[412:413]*/
	v_pk_add_f32 v[156:157] /*v[412:413]*/, v[162:163] /*v[418:419]*/, v[158:159] /*v[414:415]*/
	v_mul_f32_e32 v132 /*v388*/, 0x3fb8aa3b, v132 /*v388*/
	v_pk_add_f32 v[130:131] /*v[386:387]*/, v[154:155] /*v[410:411]*/, v[130:131] /*v[386:387]*/
	v_exp_f32_e32 v132 /*v388*/, v132 /*v388*/
	v_pk_add_f32 v[130:131] /*v[386:387]*/, v[156:157] /*v[412:413]*/, v[130:131] /*v[386:387]*/
	v_dual_mov_b32 v152 /*v408*/, v130 /*v386*/ :: v_dual_mov_b32 v154 /*v410*/, v131 /*v387*/
	v_permlanex16_b32 v152 /*v408*/, v152 /*v408*/, s17, 0xfedcba98
	v_permlanex16_b32 v154 /*v410*/, v154 /*v410*/, s17, 0xfedcba98
	s_set_vgpr_msb 0x5500
	s_cbranch_vccz .LBB0_62
	s_set_vgpr_msb 4
	v_pk_mul_f32 v[126:127], v[126:127], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[124:125], v[124:125], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[122:123], v[122:123], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[120:121], v[120:121], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[118:119], v[118:119], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[116:117], v[116:117], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[114:115], v[114:115], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[112:113], v[112:113], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[110:111], v[110:111], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[108:109], v[108:109], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[106:107], v[106:107], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[104:105], v[104:105], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[102:103], v[102:103], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[100:101], v[100:101], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[98:99], v[98:99], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[96:97], v[96:97], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[94:95], v[94:95], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[92:93], v[92:93], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[90:91], v[90:91], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[88:89], v[88:89], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[86:87], v[86:87], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[84:85], v[84:85], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[82:83], v[82:83], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[80:81], v[80:81], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[78:79], v[78:79], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[76:77], v[76:77], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[74:75], v[74:75], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[72:73], v[72:73], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[70:71], v[70:71], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[68:69], v[68:69], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[66:67], v[66:67], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[64:65], v[64:65], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x400
.LBB0_62:
	s_set_vgpr_msb 0x45
	v_sub_f32_e32 v134 /*v390*/, v151 /*v407*/, v135 /*v391*/
	s_and_b32 s2, s3, exec_lo
	s_cselect_b32 s2, 1, 0
	s_cmp_lg_u32 s2, 1
	v_mul_f32_e32 v134 /*v390*/, 0x3fb8aa3b, v134 /*v390*/
	v_exp_f32_e32 v134 /*v390*/, v134 /*v390*/
	s_set_vgpr_msb 0x4500
	s_cbranch_scc1 .LBB0_57
	v_nop
	s_set_vgpr_msb 4
	v_pk_mul_f32 v[62:63], v[62:63], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[60:61], v[60:61], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59], v[58:59], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[56:57], v[56:57], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[54:55], v[54:55], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[52:53], v[52:53], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[50:51], v[50:51], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[48:49], v[48:49], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[46:47], v[46:47], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[44:45], v[44:45], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[42:43], v[42:43], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[40:41], v[40:41], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39], v[38:39], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[36:37], v[36:37], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35], v[34:35], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[32:33], v[32:33], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[30:31], v[30:31], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[28:29], v[28:29], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[26:27], v[26:27], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25], v[24:25], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[22:23], v[22:23], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21], v[20:21], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[18:19], v[18:19], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[16:17], v[16:17], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[14:15], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13], v[12:13], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[10:11], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[8:9], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[6:7], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[4:5], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[2:3], v[2:3], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[0:1], v[0:1], v[134:135] /*v[390:391]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x400
	s_branch .LBB0_57
.LBB0_64:
	s_set_vgpr_msb 0x41
	v_dual_mov_b32 v135 /*v391*/, v151 /*v407*/ :: v_dual_mov_b32 v133 /*v389*/, v152 /*v408*/
	s_set_vgpr_msb 0x4100
.LBB0_65:
	s_set_vgpr_msb 5
	v_div_scale_f32 v129, null, v64 /*v320*/, v64 /*v320*/, 1.0
	v_div_scale_f32 v132, vcc_lo, 1.0, v64 /*v320*/, 1.0
	v_div_scale_f32 v150, null, v65 /*v321*/, v65 /*v321*/, 1.0
	s_sub_co_i32 s2, s10, s60
	v_mul_u32_u24_e32 v128, 0x110, v140 /*v396*/
	s_set_vgpr_msb 0x500
	v_rcp_f32_e32 v130, v129
	s_lshr_b32 s3, s2, 31
	v_rcp_f32_e32 v151, v150
	s_add_co_i32 s3, s2, s3
	s_mov_b32 s4, 0
	s_and_b32 s3, s3, -2
	s_wait_dscnt 0x0
	s_sub_co_i32 s2, s2, s3
	v_fma_f32 v131, -v129, v130, 1.0
	s_lshl_b32 s2, s2, 17
	s_add_co_i32 s2, s2, s96
	s_set_vgpr_msb 4
	v_add_nc_u32_e32 v152, s2, v142 /*v398*/
	s_set_vgpr_msb 0x400
	v_fmac_f32_e32 v130, v131, v130
	v_mul_f32_e32 v131, v132, v130
	v_fma_f32 v133, -v129, v131, v132
	v_fmac_f32_e32 v131, v133, v130
	v_fma_f32 v129, -v129, v131, v132
	v_div_fmas_f32 v129, v129, v130, v131
	s_set_vgpr_msb 4
	v_cmp_lt_f32_e32 vcc_lo, 0, v64 /*v320*/
	s_set_vgpr_msb 0x400
	v_fma_f32 v131, -v150, v151, 1.0
	s_set_vgpr_msb 4
	v_div_fixup_f32 v129, v129, v64 /*v320*/, 1.0
	s_set_vgpr_msb 0x400
	v_cndmask_b32_e32 v130, 0, v129, vcc_lo
	s_set_vgpr_msb 4
	v_div_scale_f32 v129, vcc_lo, 1.0, v65 /*v321*/, 1.0
	s_set_vgpr_msb 0x400
	v_fmac_f32_e32 v151, v131, v151
	v_pk_mul_f32 v[114:115], v[114:115], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[148:149], v[130:131], v[68:69] op_sel_hi:[0,1]
	v_pk_mul_f32 v[140:141], v[130:131], v[76:77] op_sel_hi:[0,1]
	v_pk_mul_f32 v[90:91], v[130:131], v[90:91] op_sel_hi:[0,1]
	v_pk_mul_f32 v[88:89], v[130:131], v[88:89] op_sel_hi:[0,1]
	v_cvt_pk_bf16_f32 v69, v114, v115
	v_mul_f32_e32 v114, v129, v151
	v_pk_mul_f32 v[132:133], v[130:131], v[80:81] op_sel_hi:[0,1]
	v_cvt_pk_bf16_f32 v81, v90, v91
	v_pk_mul_f32 v[92:93], v[130:131], v[92:93] op_sel_hi:[0,1]
	v_cvt_pk_bf16_f32 v80, v88, v89
	v_fma_f32 v76, -v150, v114, v129
	v_pk_mul_f32 v[96:97], v[96:97], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[134:135], v[130:131], v[82:83] op_sel_hi:[0,1]
	v_cvt_pk_bf16_f32 v82, v92, v93
	v_pk_mul_f32 v[112:113], v[112:113], v[130:131] op_sel_hi:[1,0]
	v_fmac_f32_e32 v114, v76, v151
	v_pk_mul_f32 v[104:105], v[104:105], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[106:107], v[106:107], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[108:109], v[108:109], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[110:111], v[110:111], v[130:131] op_sel_hi:[1,0]
	v_fma_f32 v90, -v150, v114, v129
	v_pk_mul_f32 v[98:99], v[98:99], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[100:101], v[100:101], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[102:103], v[102:103], v[130:131] op_sel_hi:[1,0]
	v_cvt_pk_bf16_f32 v76, v96, v97
	v_div_fmas_f32 v88, v90, v151, v114
	s_set_vgpr_msb 4
	v_cmp_lt_f32_e32 vcc_lo, 0, v65 /*v321*/
	s_set_vgpr_msb 0x400
	v_pk_mul_f32 v[120:121], v[120:121], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[122:123], v[122:123], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[124:125], v[124:125], v[130:131] op_sel_hi:[1,0]
	s_set_vgpr_msb 4
	v_div_fixup_f32 v92, v88, v65 /*v321*/, 1.0
	s_set_vgpr_msb 0x400
	v_pk_mul_f32 v[126:127], v[126:127], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[116:117], v[116:117], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[118:119], v[118:119], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[94:95], v[130:131], v[94:95] op_sel_hi:[0,1]
	v_cndmask_b32_e32 v96, 0, v92, vcc_lo
	v_pk_mul_f32 v[84:85], v[130:131], v[84:85] op_sel_hi:[0,1]
	v_pk_mul_f32 v[86:87], v[130:131], v[86:87] op_sel_hi:[0,1]
	v_pk_mul_f32 v[136:137], v[130:131], v[72:73] op_sel_hi:[0,1]
	v_pk_mul_f32 v[138:139], v[130:131], v[74:75] op_sel_hi:[0,1]
	v_pk_mul_f32 v[142:143], v[130:131], v[78:79] op_sel_hi:[0,1]
	v_pk_mul_f32 v[144:145], v[130:131], v[64:65] op_sel_hi:[0,1]
	v_pk_mul_f32 v[146:147], v[130:131], v[66:67] op_sel_hi:[0,1]
	v_pk_mul_f32 v[130:131], v[130:131], v[70:71] op_sel_hi:[0,1]
	v_cvt_pk_bf16_f32 v68, v112, v113
	v_cvt_pk_bf16_f32 v75, v110, v111
	v_cvt_pk_bf16_f32 v74, v108, v109
	v_cvt_pk_bf16_f32 v73, v106, v107
	v_cvt_pk_bf16_f32 v72, v104, v105
	v_cvt_pk_bf16_f32 v79, v102, v103
	v_cvt_pk_bf16_f32 v78, v100, v101
	v_cvt_pk_bf16_f32 v77, v98, v99
	v_pk_mul_f32 v[56:57], v[56:57], v[96:97] op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59], v[58:59], v[96:97] op_sel_hi:[1,0]
	v_pk_mul_f32 v[60:61], v[60:61], v[96:97] op_sel_hi:[1,0]
	v_pk_mul_f32 v[62:63], v[62:63], v[96:97] op_sel_hi:[1,0]
	v_pk_mul_f32 v[48:49], v[48:49], v[96:97] op_sel_hi:[1,0]
	v_pk_mul_f32 v[50:51], v[50:51], v[96:97] op_sel_hi:[1,0]
	v_pk_mul_f32 v[52:53], v[52:53], v[96:97] op_sel_hi:[1,0]
	v_pk_mul_f32 v[54:55], v[54:55], v[96:97] op_sel_hi:[1,0]
	v_pk_mul_f32 v[40:41], v[40:41], v[96:97] op_sel_hi:[1,0]
	v_pk_mul_f32 v[42:43], v[42:43], v[96:97] op_sel_hi:[1,0]
	v_pk_mul_f32 v[44:45], v[44:45], v[96:97] op_sel_hi:[1,0]
	v_pk_mul_f32 v[46:47], v[46:47], v[96:97] op_sel_hi:[1,0]
	v_pk_mul_f32 v[32:33], v[32:33], v[96:97] op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35], v[34:35], v[96:97] op_sel_hi:[1,0]
	v_pk_mul_f32 v[36:37], v[36:37], v[96:97] op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39], v[38:39], v[96:97] op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25], v[96:97], v[24:25] op_sel_hi:[0,1]
	v_pk_mul_f32 v[26:27], v[96:97], v[26:27] op_sel_hi:[0,1]
	v_pk_mul_f32 v[28:29], v[96:97], v[28:29] op_sel_hi:[0,1]
	v_pk_mul_f32 v[30:31], v[96:97], v[30:31] op_sel_hi:[0,1]
	v_pk_mul_f32 v[98:99], v[96:97], v[16:17] op_sel_hi:[0,1]
	v_pk_mul_f32 v[100:101], v[96:97], v[18:19] op_sel_hi:[0,1]
	v_pk_mul_f32 v[20:21], v[96:97], v[20:21] op_sel_hi:[0,1]
	v_pk_mul_f32 v[22:23], v[96:97], v[22:23] op_sel_hi:[0,1]
	v_pk_mul_f32 v[102:103], v[96:97], v[8:9] op_sel_hi:[0,1]
	v_pk_mul_f32 v[104:105], v[96:97], v[10:11] op_sel_hi:[0,1]
	v_pk_mul_f32 v[106:107], v[96:97], v[12:13] op_sel_hi:[0,1]
	v_pk_mul_f32 v[108:109], v[96:97], v[14:15] op_sel_hi:[0,1]
	v_pk_mul_f32 v[110:111], v[96:97], v[0:1] op_sel_hi:[0,1]
	v_pk_mul_f32 v[112:113], v[96:97], v[2:3] op_sel_hi:[0,1]
	v_pk_mul_f32 v[114:115], v[96:97], v[4:5] op_sel_hi:[0,1]
	v_pk_mul_f32 v[96:97], v[96:97], v[6:7] op_sel_hi:[0,1]
	v_cvt_pk_bf16_f32 v67, v126, v127
	v_cvt_pk_bf16_f32 v66, v124, v125
	v_cvt_pk_bf16_f32 v65, v122, v123
	v_cvt_pk_bf16_f32 v64, v120, v121
	v_cvt_pk_bf16_f32 v71, v118, v119
	v_cvt_pk_bf16_f32 v70, v116, v117
	v_cvt_pk_bf16_f32 v83, v94, v95
	v_cvt_pk_bf16_f32 v87, v86, v87
	v_cvt_pk_bf16_f32 v86, v84, v85
	v_cvt_pk_bf16_f32 v85, v134, v135
	v_cvt_pk_bf16_f32 v84, v132, v133
	v_cvt_pk_bf16_f32 v91, v142, v143
	v_cvt_pk_bf16_f32 v90, v140, v141
	v_cvt_pk_bf16_f32 v89, v138, v139
	v_cvt_pk_bf16_f32 v88, v136, v137
	v_cvt_pk_bf16_f32 v95, v130, v131
	v_cvt_pk_bf16_f32 v94, v148, v149
	v_cvt_pk_bf16_f32 v93, v146, v147
	v_cvt_pk_bf16_f32 v92, v144, v145
	v_cvt_pk_bf16_f32 v3, v62, v63
	v_cvt_pk_bf16_f32 v2, v60, v61
	v_cvt_pk_bf16_f32 v1, v58, v59
	v_cvt_pk_bf16_f32 v0, v56, v57
	v_cvt_pk_bf16_f32 v7, v54, v55
	v_cvt_pk_bf16_f32 v6, v52, v53
	v_cvt_pk_bf16_f32 v5, v50, v51
	v_cvt_pk_bf16_f32 v4, v48, v49
	v_cvt_pk_bf16_f32 v11, v46, v47
	v_cvt_pk_bf16_f32 v10, v44, v45
	v_cvt_pk_bf16_f32 v9, v42, v43
	v_cvt_pk_bf16_f32 v8, v40, v41
	v_cvt_pk_bf16_f32 v15, v38, v39
	v_cvt_pk_bf16_f32 v14, v36, v37
	v_cvt_pk_bf16_f32 v13, v34, v35
	v_cvt_pk_bf16_f32 v12, v32, v33
	v_cvt_pk_bf16_f32 v19, v30, v31
	v_cvt_pk_bf16_f32 v18, v28, v29
	v_cvt_pk_bf16_f32 v17, v26, v27
	v_cvt_pk_bf16_f32 v16, v24, v25
	v_cvt_pk_bf16_f32 v23, v22, v23
	v_cvt_pk_bf16_f32 v22, v20, v21
	v_cvt_pk_bf16_f32 v21, v100, v101
	v_cvt_pk_bf16_f32 v20, v98, v99
	v_cvt_pk_bf16_f32 v27, v108, v109
	v_cvt_pk_bf16_f32 v26, v106, v107
	v_cvt_pk_bf16_f32 v25, v104, v105
	v_cvt_pk_bf16_f32 v24, v102, v103
	v_cvt_pk_bf16_f32 v31, v96, v97
	v_cvt_pk_bf16_f32 v30, v114, v115
	v_cvt_pk_bf16_f32 v29, v112, v113
	v_cvt_pk_bf16_f32 v28, v110, v111
	s_lshl_b32 s3, s18, 2
	s_wait_alu depctr_va_vdst(0)
	ds_store_b128 v152, v[64:67]
	ds_store_b128 v152, v[68:71] offset:32
	ds_store_b128 v152, v[72:75] offset:64
	ds_store_b128 v152, v[76:79] offset:96
	ds_store_b128 v152, v[80:83] offset:128
	ds_store_b128 v152, v[84:87] offset:160
	ds_store_b128 v152, v[88:91] offset:192
	ds_store_b128 v152, v[92:95] offset:224
	ds_store_b128 v152, v[0:3] offset:4352
	ds_store_b128 v152, v[4:7] offset:4384
	ds_store_b128 v152, v[8:11] offset:4416
	ds_store_b128 v152, v[12:15] offset:4448
	s_sub_co_i32 s3, s3, s95
	ds_store_b128 v152, v[16:19] offset:4480
	ds_store_b128 v152, v[20:23] offset:4512
	ds_store_b128 v152, v[24:27] offset:4544
	ds_store_b128 v152, v[28:31] offset:4576
	s_cmp_lt_i32 s3, 1
	s_wait_dscnt 0x0
	s_cbranch_scc1 .LBB0_67
	s_wait_alu depctr_vm_vsrc(0)
	s_set_vgpr_msb 4
	v_dual_lshrrev_b32 v26, 4, v141 /*v397*/ :: v_dual_bitop2_b32 v0, 28, v136 /*v392*/ bitop3:0x54
	s_add_co_i32 s3, s3, -1
	s_set_vgpr_msb 0x400
	v_add_nc_u32_e32 v2, s2, v128
	s_min_u32 s3, s3, 31
	v_or_b32_e32 v1, 30, v26
	v_min_u32_e32 v3, s3, v0
	s_set_vgpr_msb 4
	v_dual_lshlrev_b32 v4, 8, v140 /*v396*/ :: v_dual_lshlrev_b32 v0, 3, v140 /*v396*/
	s_set_vgpr_msb 0x400
	v_or_b32_e32 v5, 26, v26
	v_min_u32_e32 v6, s3, v1
	v_dual_sub_nc_u32 v32, v2, v4 :: v_dual_bitop2_b32 v7, s95, v3 bitop3:0x54
	v_mov_b32_e32 v1, 0
	v_min_u32_e32 v9, s3, v5
	v_dual_ashrrev_i32 v10, 31, v7 :: v_dual_bitop2_b32 v8, s95, v6 bitop3:0x54
	s_set_vgpr_msb 4
	v_or_b32_e32 v5, 24, v136 /*v392*/
	s_set_vgpr_msb 0x400
	v_mad_u32_u24 v33, 0x110, v3, v32
	v_mad_u32_u24 v34, 0x110, v6, v32
	v_dual_ashrrev_i32 v2, 31, v8 :: v_dual_bitop2_b32 v11, s95, v9 bitop3:0x54
	v_lshrrev_b32_e32 v4, 30, v10
	v_min_u32_e32 v14, s3, v5
	v_mad_u32_u24 v35, 0x110, v9, v32
	s_set_vgpr_msb 4
	v_or_b32_e32 v21, 16, v136 /*v392*/
	s_set_vgpr_msb 0x400
	v_dual_ashrrev_i32 v10, 31, v11 :: v_dual_bitop2_b32 v5, 22, v26 bitop3:0x54
	v_dual_add_nc_u32 v3, v7, v4 :: v_dual_lshrrev_b32 v2, 30, v2
	v_or_b32_e32 v12, s95, v14
	v_lshrrev_b32_e32 v4, 30, v10
	v_min_u32_e32 v15, s3, v5
	v_dual_add_nc_u32 v2, v8, v2 :: v_dual_bitop2_b32 v5, -4, v3 bitop3:0x40
	v_dual_ashrrev_i32 v3, 2, v3 :: v_dual_add_nc_u32 v13, v11, v4
	v_min_u32_e32 v21, s3, v21
	v_dual_sub_nc_u32 v4, v7, v5 :: v_dual_bitop2_b32 v10, -4, v2 bitop3:0x40
	v_ashrrev_i32_e32 v2, 2, v2
	v_cmp_ne_u32_e32 vcc_lo, v7, v5
	v_add_nc_u32_e32 v7, s91, v3
	v_mad_u32_u24 v36, 0x110, v14, v32
	v_cmp_ne_u32_e64 s2, v8, v10
	v_sub_nc_u32_e32 v5, v8, v10
	v_dual_add_nc_u32 v8, s91, v2 :: v_dual_add_nc_u32 v2, s34, v4
	s_and_b32 s5, s94, vcc_lo
	s_and_b32 s2, s94, s2
	v_cndmask_b32_e64 v10, 0, 1, s5
	v_cndmask_b32_e64 v16, 0, 1, s2
	v_mad_nc_i64_i32 v[2:3], v2, s12, v[0:1]
	v_add_nc_u32_e32 v4, s34, v5
	v_mad_u32_u24 v37, 0x110, v15, v32
	v_dual_sub_nc_u32 v6, v7, v10 :: v_dual_sub_nc_u32 v7, v8, v16
	v_dual_ashrrev_i32 v10, 31, v12 :: v_dual_bitop2_b32 v8, -4, v13 bitop3:0x40
	v_ashrrev_i32_e32 v13, 2, v13
	v_mad_nc_i64_i32 v[2:3], v6, s8, v[2:3]
	v_mad_nc_i64_i32 v[4:5], v4, s12, v[0:1]
	v_dual_sub_nc_u32 v6, v11, v8 :: v_dual_bitop2_b32 v16, s95, v15 bitop3:0x54
	v_cmp_ne_u32_e32 vcc_lo, v11, v8
	v_dual_lshrrev_b32 v10, 30, v10 :: v_dual_add_nc_u32 v8, s91, v13
	v_dual_add_nc_u32 v6, s34, v6 :: v_dual_ashrrev_i32 v17, 31, v16
	s_and_b32 s2, s94, vcc_lo
	s_set_vgpr_msb 4
	v_or_b32_e32 v13, 20, v136 /*v392*/
	v_cndmask_b32_e64 v11, 0, 1, s2
	s_set_vgpr_msb 0x400
	v_add_nc_u32_e32 v10, v12, v10
	v_mad_nc_i64_i32 v[4:5], v7, s8, v[4:5]
	v_mad_nc_i64_i32 v[6:7], v6, s12, v[0:1]
	v_min_u32_e32 v18, s3, v13
	v_dual_sub_nc_u32 v8, v8, v11 :: v_dual_lshrrev_b32 v11, 30, v17
	v_and_b32_e32 v9, -4, v10
	v_mad_u32_u24 v40, 0x110, v21, v32
	v_or_b32_e32 v13, s95, v18
	v_mad_u32_u24 v38, 0x110, v18, v32
	v_mad_nc_i64_i32 v[6:7], v8, s8, v[6:7]
	v_dual_ashrrev_i32 v8, 2, v10 :: v_dual_sub_nc_u32 v10, v12, v9
	v_cmp_ne_u32_e32 vcc_lo, v12, v9
	v_dual_add_nc_u32 v11, v16, v11 :: v_dual_ashrrev_i32 v9, 31, v13
	v_dual_add_nc_u32 v12, s91, v8 :: v_dual_add_nc_u32 v8, s34, v10
	s_and_b32 s2, s94, vcc_lo
	v_and_b32_e32 v10, -4, v11
	v_cndmask_b32_e64 v17, 0, 1, s2
	v_dual_ashrrev_i32 v11, 2, v11 :: v_dual_lshrrev_b32 v19, 30, v9
	v_mad_nc_i64_i32 v[8:9], v8, s12, v[0:1]
	v_cmp_ne_u32_e32 vcc_lo, v16, v10
	v_dual_sub_nc_u32 v12, v12, v17 :: v_dual_add_nc_u32 v11, s91, v11
	v_dual_add_nc_u32 v17, v13, v19 :: v_dual_sub_nc_u32 v10, v16, v10
	s_and_b32 s2, s94, vcc_lo
	s_set_vgpr_msb 4
	v_or_b32_e32 v30, 8, v136 /*v392*/
	v_mad_nc_i64_i32 v[8:9], v12, s8, v[8:9]
	s_set_vgpr_msb 0x400
	v_dual_add_nc_u32 v10, s34, v10 :: v_dual_bitop2_b32 v12, -4, v17 bitop3:0x40
	v_cndmask_b32_e64 v16, 0, 1, s2
	v_dual_ashrrev_i32 v17, 2, v17 :: v_dual_bitop2_b32 v19, 18, v26 bitop3:0x54
	v_sub_nc_u32_e32 v20, v13, v12
	v_cmp_ne_u32_e32 vcc_lo, v13, v12
	v_sub_nc_u32_e32 v16, v11, v16
	v_min_u32_e32 v19, s3, v19
	v_mad_nc_i64_i32 v[10:11], v10, s12, v[0:1]
	v_dual_add_nc_u32 v17, s91, v17 :: v_dual_add_nc_u32 v12, s34, v20
	s_and_b32 s2, s94, vcc_lo
	v_mad_u32_u24 v39, 0x110, v19, v32
	v_cndmask_b32_e64 v22, 0, 1, s2
	v_or_b32_e32 v20, s95, v19
	v_mad_nc_i64_i32 v[12:13], v12, s12, v[0:1]
	v_mad_nc_i64_i32 v[10:11], v16, s8, v[10:11]
	v_min_u32_e32 v41, s3, v30
	v_dual_sub_nc_u32 v16, v17, v22 :: v_dual_ashrrev_i32 v23, 31, v20
	v_or_b32_e32 v22, s95, v21
	s_load_b64 s[6:7], s[0:1], 0x0 nv
	v_or_b32_e32 v30, s95, v41
	v_mad_nc_i64_i32 v[12:13], v16, s8, v[12:13]
	v_dual_lshrrev_b32 v17, 30, v23 :: v_dual_ashrrev_i32 v16, 31, v22
	v_mad_u32_u24 v41, 0x110, v41, v32
	v_dual_ashrrev_i32 v31, 31, v30 :: v_dual_add_nc_u32 v14, v20, v17
	v_dual_lshrrev_b32 v16, 30, v16 :: v_dual_bitop2_b32 v15, -4, v14 bitop3:0x40
	v_dual_add_nc_u32 v16, v22, v16 :: v_dual_bitop2_b32 v17, 14, v26 bitop3:0x54
	v_dual_ashrrev_i32 v14, 2, v14 :: v_dual_sub_nc_u32 v18, v20, v15
	v_cmp_ne_u32_e32 vcc_lo, v20, v15
	v_min_u32_e32 v23, s3, v17
	v_dual_add_nc_u32 v20, s91, v14 :: v_dual_bitop2_b32 v17, -4, v16 bitop3:0x40
	v_dual_add_nc_u32 v14, s34, v18 :: v_dual_ashrrev_i32 v16, 2, v16
	v_dual_sub_nc_u32 v25, v22, v17 :: v_dual_bitop2_b32 v18, s95, v23 bitop3:0x54
	s_and_b32 s2, s94, vcc_lo
	v_cmp_ne_u32_e32 vcc_lo, v22, v17
	v_cndmask_b32_e64 v24, 0, 1, s2
	v_dual_ashrrev_i32 v27, 31, v18 :: v_dual_add_nc_u32 v22, s91, v16
	v_add_nc_u32_e32 v16, s34, v25
	s_and_b32 s2, s94, vcc_lo
	v_dual_sub_nc_u32 v20, v20, v24 :: v_dual_lshrrev_b32 v25, 30, v27
	s_set_vgpr_msb 4
	v_or_b32_e32 v27, 12, v136 /*v392*/
	v_cndmask_b32_e64 v28, 0, 1, s2
	v_mad_nc_i64_i32 v[14:15], v14, s12, v[0:1]
	v_mad_nc_i64_i32 v[16:17], v16, s12, v[0:1]
	s_set_vgpr_msb 0x400
	v_mad_u32_u24 v42, 0x110, v23, v32
	v_min_u32_e32 v27, s3, v27
	v_add_nc_u32_e32 v25, v18, v25
	s_wait_kmcnt 0x0
	v_lshl_add_u64 v[2:3], v[2:3], 1, s[6:7]
	v_lshl_add_u64 v[4:5], v[4:5], 1, s[6:7]
	v_lshl_add_u64 v[6:7], v[6:7], 1, s[6:7]
	v_dual_sub_nc_u32 v22, v22, v28 :: v_dual_bitop2_b32 v24, s95, v27 bitop3:0x54
	v_and_b32_e32 v19, -4, v25
	v_mad_nc_i64_i32 v[14:15], v20, s8, v[14:15]
	v_dual_ashrrev_i32 v20, 2, v25 :: v_dual_ashrrev_i32 v25, 31, v24
	v_or_b32_e32 v28, 10, v26
	v_mad_nc_i64_i32 v[16:17], v22, s8, v[16:17]
	v_sub_nc_u32_e32 v22, v18, v19
	v_cmp_ne_u32_e32 vcc_lo, v18, v19
	v_add_nc_u32_e32 v20, s91, v20
	v_mad_u32_u24 v44, 0x110, v27, v32
	v_lshl_add_u64 v[8:9], v[8:9], 1, s[6:7]
	v_dual_add_nc_u32 v18, s34, v22 :: v_dual_lshrrev_b32 v22, 30, v25
	v_min_u32_e32 v25, s3, v28
	s_and_b32 s2, s94, vcc_lo
	v_lshl_add_u64 v[10:11], v[10:11], 1, s[6:7]
	v_cndmask_b32_e64 v28, 0, 1, s2
	v_mad_nc_i64_i32 v[18:19], v18, s12, v[0:1]
	v_dual_add_nc_u32 v22, v24, v22 :: v_dual_bitop2_b32 v29, s95, v25 bitop3:0x54
	v_mad_u32_u24 v45, 0x110, v25, v32
	v_sub_nc_u32_e32 v20, v20, v28
	v_lshl_add_u64 v[12:13], v[12:13], 1, s[6:7]
	v_dual_ashrrev_i32 v28, 31, v29 :: v_dual_bitop2_b32 v21, -4, v22 bitop3:0x40
	v_lshl_add_u64 v[14:15], v[14:15], 1, s[6:7]
	v_mad_nc_i64_i32 v[18:19], v20, s8, v[18:19]
	v_ashrrev_i32_e32 v20, 2, v22
	v_lshl_add_u64 v[16:17], v[16:17], 1, s[6:7]
	v_dual_sub_nc_u32 v22, v24, v21 :: v_dual_lshrrev_b32 v28, 30, v28
	v_cmp_ne_u32_e32 vcc_lo, v24, v21
	v_dual_add_nc_u32 v24, s91, v20 :: v_dual_add_nc_u32 v20, s34, v22
	v_add_nc_u32_e32 v22, v29, v28
	s_and_b32 s2, s94, vcc_lo
	v_lshl_add_u64 v[18:19], v[18:19], 1, s[6:7]
	v_cndmask_b32_e64 v28, 0, 1, s2
	v_mad_nc_i64_i32 v[20:21], v20, s12, v[0:1]
	v_and_b32_e32 v23, -4, v22
	v_dual_ashrrev_i32 v22, 2, v22 :: v_dual_sub_nc_u32 v24, v24, v28
	v_sub_nc_u32_e32 v28, v29, v23
	v_cmp_ne_u32_e32 vcc_lo, v29, v23
	v_or_b32_e32 v29, 6, v26
	v_mad_nc_i64_i32 v[20:21], v24, s8, v[20:21]
	v_dual_add_nc_u32 v24, s91, v22 :: v_dual_add_nc_u32 v22, s34, v28
	s_and_b32 s2, s94, vcc_lo
	v_lshrrev_b32_e32 v28, 30, v31
	v_cndmask_b32_e64 v31, 0, 1, s2
	v_min_u32_e32 v43, s3, v29
	v_mad_nc_i64_i32 v[22:23], v22, s12, v[0:1]
	v_lshl_add_u64 v[20:21], v[20:21], 1, s[6:7]
	v_dual_add_nc_u32 v28, v30, v28 :: v_dual_sub_nc_u32 v24, v24, v31
	s_set_vgpr_msb 4
	v_or_b32_e32 v31, 4, v136 /*v392*/
	s_set_vgpr_msb 0x400
	v_and_b32_e32 v27, -4, v28
	v_mad_nc_i64_i32 v[22:23], v24, s8, v[22:23]
	v_dual_ashrrev_i32 v24, 2, v28 :: v_dual_bitop2_b32 v29, s95, v43 bitop3:0x54
	v_min_u32_e32 v46, s3, v31
	v_sub_nc_u32_e32 v25, v30, v27
	v_cmp_ne_u32_e32 vcc_lo, v30, v27
	v_dual_add_nc_u32 v27, s91, v24 :: v_dual_ashrrev_i32 v28, 31, v29
	v_dual_add_nc_u32 v24, s34, v25 :: v_dual_bitop2_b32 v31, s95, v46 bitop3:0x54
	s_and_b32 s2, s94, vcc_lo
	v_mad_u32_u24 v43, 0x110, v43, v32
	v_lshrrev_b32_e32 v28, 30, v28
	v_cndmask_b32_e64 v30, 0, 1, s2
	v_mad_nc_i64_i32 v[24:25], v24, s12, v[0:1]
	v_mad_u32_u24 v46, 0x110, v46, v32
	v_lshl_add_u64 v[22:23], v[22:23], 1, s[6:7]
	v_dual_add_nc_u32 v28, v29, v28 :: v_dual_sub_nc_u32 v27, v27, v30
	v_or_b32_e32 v26, 2, v26
	v_and_b32_e32 v30, -4, v28
	v_dual_ashrrev_i32 v28, 2, v28 :: v_dual_ashrrev_i32 v47, 31, v31
	v_mad_nc_i64_i32 v[24:25], v27, s8, v[24:25]
	v_min_u32_e32 v48, s3, v26
	v_cmp_ne_u32_e32 vcc_lo, v29, v30
	v_dual_add_nc_u32 v26, s91, v28 :: v_dual_lshrrev_b32 v27, 30, v47
	s_set_vgpr_msb 4
	v_min_i32_e32 v47, s3, v136 /*v392*/
	s_set_vgpr_msb 0x400
	v_or_b32_e32 v28, s95, v48
	s_and_b32 s2, s94, vcc_lo
	v_sub_nc_u32_e32 v29, v29, v30
	v_cndmask_b32_e64 v49, 0, 1, s2
	v_dual_ashrrev_i32 v50, 31, v28 :: v_dual_bitop2_b32 v30, s95, v47 bitop3:0x54
	v_add_nc_u32_e32 v27, v31, v27
	v_mad_u32_u24 v47, 0x110, v47, v32
	v_sub_nc_u32_e32 v49, v26, v49
	v_dual_add_nc_u32 v26, s34, v29 :: v_dual_lshrrev_b32 v50, 30, v50
	v_dual_ashrrev_i32 v29, 31, v30 :: v_dual_bitop2_b32 v51, -4, v27 bitop3:0x40
	v_ashrrev_i32_e32 v52, 2, v27
	v_mad_nc_i64_i32 v[26:27], v26, s12, v[0:1]
	v_dual_add_nc_u32 v50, v28, v50 :: v_dual_lshrrev_b32 v29, 30, v29
	v_cmp_ne_u32_e32 vcc_lo, v31, v51
	v_dual_add_nc_u32 v52, s91, v52 :: v_dual_sub_nc_u32 v31, v31, v51
	v_dual_add_nc_u32 v29, v30, v29 :: v_dual_bitop2_b32 v51, -4, v50 bitop3:0x40
	s_and_b32 s2, s94, vcc_lo
	v_dual_ashrrev_i32 v50, 2, v50 :: v_dual_add_nc_u32 v31, s34, v31
	v_cndmask_b32_e64 v53, 0, 1, s2
	v_and_b32_e32 v54, -4, v29
	v_cmp_ne_u32_e32 vcc_lo, v28, v51
	v_dual_sub_nc_u32 v28, v28, v51 :: v_dual_ashrrev_i32 v29, 2, v29
	v_add_nc_u32_e32 v50, s91, v50
	v_sub_nc_u32_e32 v51, v30, v54
	v_cmp_ne_u32_e64 s2, v30, v54
	v_dual_add_nc_u32 v54, s34, v28 :: v_dual_add_nc_u32 v55, s91, v29
	v_mad_nc_i64_i32 v[30:31], v31, s12, v[0:1]
	v_add_nc_u32_e32 v28, s34, v51
	s_and_b32 s2, s94, s2
	v_mad_nc_i64_i32 v[26:27], v49, s8, v[26:27]
	v_cndmask_b32_e64 v51, 0, 1, s2
	s_and_b32 s2, s94, vcc_lo
	v_mad_nc_i64_i32 v[28:29], v28, s12, v[0:1]
	v_cndmask_b32_e64 v56, 0, 1, s2
	v_mad_nc_i64_i32 v[0:1], v54, s12, v[0:1]
	v_sub_nc_u32_e32 v51, v55, v51
	v_mad_u32_u24 v32, 0x110, v48, v32
	v_lshl_add_u64 v[26:27], v[26:27], 1, s[6:7]
	v_dual_sub_nc_u32 v49, v50, v56 :: v_dual_sub_nc_u32 v50, v52, v53
	v_mad_nc_i64_i32 v[28:29], v51, s8, v[28:29]
	v_lshl_add_u64 v[24:25], v[24:25], 1, s[6:7]
	v_mad_nc_i64_i32 v[0:1], v49, s8, v[0:1]
	v_mad_nc_i64_i32 v[30:31], v50, s8, v[30:31]
	v_lshl_add_u64 v[28:29], v[28:29], 1, s[6:7]
	v_lshl_add_u64 v[0:1], v[0:1], 1, s[6:7]
	v_lshl_add_u64 v[30:31], v[30:31], 1, s[6:7]
	s_wait_alu depctr_va_vdst(2)
	global_store_async_from_lds_b128 v[28:29], v47, off
	s_wait_alu depctr_va_vdst(1)
	global_store_async_from_lds_b128 v[0:1], v32, off
	s_wait_alu depctr_va_vdst(0)
	global_store_async_from_lds_b128 v[30:31], v46, off
	global_store_async_from_lds_b128 v[26:27], v43, off
	global_store_async_from_lds_b128 v[24:25], v41, off
	global_store_async_from_lds_b128 v[22:23], v45, off
	global_store_async_from_lds_b128 v[20:21], v44, off
	global_store_async_from_lds_b128 v[18:19], v42, off
	global_store_async_from_lds_b128 v[16:17], v40, off
	global_store_async_from_lds_b128 v[14:15], v39, off
	global_store_async_from_lds_b128 v[12:13], v38, off
	global_store_async_from_lds_b128 v[10:11], v37, off
	global_store_async_from_lds_b128 v[8:9], v36, off
	global_store_async_from_lds_b128 v[6:7], v35, off
	global_store_async_from_lds_b128 v[2:3], v33, off
	global_store_async_from_lds_b128 v[4:5], v34, off
.LBB0_67:
	s_set_vgpr_msb 4
	v_cmp_gt_f32_e32 vcc_lo, 0x800000, v64 /*v320*/
	v_cmp_gt_f32_e64 s2, 0x800000, v65 /*v321*/
	s_wait_alu depctr_vm_vsrc(0)
	v_mul_lo_u32 v4, s14, v139 /*v395*/
	s_load_b64 s[6:7], s[0:1], 0x40 nv
	s_mul_i32 s8, s33, s15
	v_cndmask_b32_e64 v2, 0, 32, vcc_lo
	v_cndmask_b32_e64 v3, 0, 32, s2
	v_cndmask_b32_e64 v0, 0, 0x42000000, vcc_lo
	v_cndmask_b32_e64 v1, 0, 0x42000000, s2
	v_mad_u32 v5, s13, v138 /*v394*/, s8
	s_set_vgpr_msb 0x401
	v_ldexp_f32 v2, v64 /*v320*/, v2
	v_ldexp_f32 v3, v65 /*v321*/, v3
	v_mul_lo_u32 v6, v137 /*v393*/, s13
	s_set_vgpr_msb 0x104
	v_cmp_eq_u32_e32 vcc_lo, 0, v136 /*v392*/
	v_cmp_gt_i32_e64 s0, s18, v138 /*v394*/
	v_log_f32_e32 v2, v2
	v_log_f32_e32 v3, v3
	s_set_vgpr_msb 0x400
	v_add_nc_u32_e32 v7, s8, v4
	s_set_vgpr_msb 4
	v_cmp_gt_i32_e64 s1, s18, v137 /*v393*/
	s_add_co_i32 s2, s8, s15
	s_and_b32 s0, vcc_lo, s0
	s_ashr_i32 s3, s2, 31
	s_lshl_b32 s5, s2, 27
	s_and_b32 vcc_lo, vcc_lo, s1
	s_set_vgpr_msb 0x400
	v_dual_sub_f32 v1, v3, v1 :: v_dual_sub_f32 v0, v2, v0
	v_add_lshl_u32 v2, v5, v4, 2
	v_add_lshl_u32 v3, v7, v6, 2
	s_lshr_b64 s[2:3], s[2:3], 5
	v_dual_mul_f32 v1, 0x3f317218, v1 :: v_dual_mul_f32 v0, 0x3f317218, v0
	v_cndmask_b32_e64 v2, 0x7fffffff, v2, s0
	v_cndmask_b32_e32 v3, 0x7fffffff, v3, vcc_lo
	s_and_b64 s[2:3], s[2:3], 0x1ffffffffffffff
	s_set_vgpr_msb 4
	v_dual_add_f32 v1, v1, v135 /*v391*/ :: v_dual_add_f32 v0, v0, v133 /*v389*/
	s_wait_kmcnt 0x0
	s_or_b64 s[0:1], s[6:7], s[4:5]
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	buffer_store_b32 v0, v2, s[0:3], null offen
	buffer_store_b32 v1, v3, s[0:3], null offen
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
.Lfunc_end0:
	.size	kn_fmha_fwd_prefill_a16w16_m32x8_bshd_0, .Lfunc_end0-kn_fmha_fwd_prefill_a16w16_m32x8_bshd_0
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel kn_fmha_fwd_prefill_a16w16_m32x8_bshd_0
		.amdhsa_group_segment_fixed_size 327680
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 416
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 0
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 1
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 513
		.amdhsa_next_free_sgpr 105
		.amdhsa_named_barrier_count 0
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end0-kn_fmha_fwd_prefill_a16w16_m32x8_bshd_0)<<4)&4080)>>4
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text

	.set .Lkn_fmha_fwd_prefill_a16w16_m32x8_bshd_0.num_vgpr, 456
	.set .Lkn_fmha_fwd_prefill_a16w16_m32x8_bshd_0.num_agpr, 0
	.set .Lkn_fmha_fwd_prefill_a16w16_m32x8_bshd_0.numbered_sgpr, 105
	.set .Lkn_fmha_fwd_prefill_a16w16_m32x8_bshd_0.num_named_barrier, 0
	.set .Lkn_fmha_fwd_prefill_a16w16_m32x8_bshd_0.private_seg_size, 0
	.set .Lkn_fmha_fwd_prefill_a16w16_m32x8_bshd_0.uses_vcc, 1
	.set .Lkn_fmha_fwd_prefill_a16w16_m32x8_bshd_0.uses_flat_scratch, 0
	.set .Lkn_fmha_fwd_prefill_a16w16_m32x8_bshd_0.has_dyn_sized_stack, 0
	.set .Lkn_fmha_fwd_prefill_a16w16_m32x8_bshd_0.has_recursion, 0
	.set .Lkn_fmha_fwd_prefill_a16w16_m32x8_bshd_0.has_indirect_call, 0
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.set amdgpu.max_num_named_barrier, 0
	.text
	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .offset:         8
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .offset:         24
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         32
        .size:           8
        .value_kind:     global_buffer
      - .offset:         40
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         48
        .size:           8
        .value_kind:     global_buffer
      - .offset:         56
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         64
        .size:           8
        .value_kind:     global_buffer
      - .offset:         72
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         80
        .size:           8
        .value_kind:     global_buffer
      - .offset:         88
        .size:           4
        .value_kind:     by_value
      - .offset:         92
        .size:           4
        .value_kind:     by_value
      - .offset:         96
        .size:           4
        .value_kind:     by_value
      - .offset:         100
        .size:           4
        .value_kind:     by_value
      - .offset:         104
        .size:           4
        .value_kind:     by_value
      - .offset:         108
        .size:           4
        .value_kind:     by_value
      - .offset:         112
        .size:           4
        .value_kind:     by_value
      - .offset:         116
        .size:           4
        .value_kind:     by_value
      - .offset:         120
        .size:           4
        .value_kind:     by_value
      - .offset:         124
        .size:           4
        .value_kind:     by_value
      - .offset:         128
        .size:           4
        .value_kind:     by_value
      - .offset:         132
        .size:           4
        .value_kind:     by_value
      - .offset:         136
        .size:           4
        .value_kind:     by_value
      - .offset:         140
        .size:           4
        .value_kind:     by_value
      - .offset:         144
        .size:           4
        .value_kind:     by_value
      - .offset:         148
        .size:           4
        .value_kind:     by_value
      - .offset:         152
        .size:           4
        .value_kind:     by_value
      - .offset:         160
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         164
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         168
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         172
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         174
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         176
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         178
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         180
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         182
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         200
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         208
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         216
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         224
        .size:           2
        .value_kind:     hidden_grid_dims
    .group_segment_fixed_size: 327680
    .kernarg_segment_align: 8
    .kernarg_segment_size: 416
    .max_flat_workgroup_size: 128
    .name:           kn_fmha_fwd_prefill_a16w16_m32x8_bshd_0
    .private_segment_fixed_size: 0
    .reqd_workgroup_size:
      - 128
      - 1
      - 1
    .sgpr_count:     107
    .sgpr_spill_count: 0
    .symbol:         kn_fmha_fwd_prefill_a16w16_m32x8_bshd_0.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     456
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa-unknown-gfx1250
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
