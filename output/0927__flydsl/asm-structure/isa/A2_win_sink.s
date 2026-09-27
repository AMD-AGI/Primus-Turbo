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
	s_set_vgpr_msb 0x80
	v_and_b32_e32 v77 /*v589*/, 31, v0
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
	v_bfe_u32 v72 /*v584*/, v0, 4, 1
	s_set_vgpr_msb 0x8088
	v_lshlrev_b32_e32 v82 /*v594*/, 1, v77 /*v589*/
	s_mov_b32 s20, 1
	s_mov_b32 s51, 0
	s_mov_b32 s37, 0xffff0000
	s_mov_b32 s26, -1
	s_mov_b32 s40, 0x80004
	s_mul_f32 s7, s7, 0x4f7ffffe
	v_lshlrev_b32_e32 v79 /*v591*/, 3, v72 /*v584*/
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
	s_sub_co_ci_u32 s21, s2, s9
	s_abs_i32 s22, s25
	s_mul_i32 s4, s21, s4
	s_cvt_f32_u32 s2, s22
	s_sub_co_i32 s7, 0, s22
	s_sub_co_i32 s23, s5, s4
	v_s_rcp_f32 s2, s2
	s_abs_i32 s27, s23
	s_xor_b32 s29, s23, s25
	s_ashr_i32 s31, s29, 31
	s_mul_f32 s2, s2, 0x4f7ffffe
	s_cvt_u32_f32 s6, s2
	s_clause 0x3
	s_load_b64 s[2:3], s[0:1], 0x10 nv
	s_load_b64 s[66:67], s[0:1], 0x20 nv
	s_load_b64 s[68:69], s[0:1], 0x30 nv
	s_load_b64 s[70:71], s[0:1], 0x50 nv
	s_mul_i32 s7, s7, s6
	s_mul_hi_u32 s4, s6, s7
	s_add_co_i32 s28, s6, s4
	s_load_b512 s[4:19], s[0:1], 0x5c nv
	s_mul_hi_u32 s28, s27, s28
	s_mul_i32 s30, s28, s22
	s_sub_co_i32 s27, s27, s30
	s_add_co_i32 s30, s28, 1
	s_sub_co_i32 s33, s27, s22
	s_cmp_ge_u32 s27, s22
	s_cselect_b32 s28, s30, s28
	s_cselect_b32 s27, s33, s27
	s_add_co_i32 s30, s28, 1
	s_cmp_ge_u32 s27, s22
	s_cselect_b32 s22, s30, s28
	s_xor_b32 s22, s22, s31
	s_sub_co_i32 s27, s22, s31
	s_wait_kmcnt 0x0
	v_cvt_pk_bf16_f32 v81 /*v593*/, s4, s4
	s_mul_i32 s27, s27, s25
	s_mov_b32 s28, s5
	s_cmp_lg_u32 s23, s27
	s_mov_b32 s30, s9
	s_cselect_b32 s25, -1, 0
	s_cmp_lt_i32 s29, 0
	s_mov_b32 s62, s6
	s_cselect_b32 s29, -1, 0
	s_mov_b32 s64, s7
	s_and_b32 s25, s25, s29
	s_sub_co_ci_u32 s33, s22, s31
	s_bfe_u32 s102, ttmp8, 0x50019
	s_mul_i32 s91, s33, s18
	s_and_b32 s22, s102, 30
	s_lshr_b32 s25, s102, 1
	s_cmp_lg_u32 s102, s22
	s_mul_i32 s96, s102, 0x2200
	s_cselect_b32 s22, -1, 0
	s_cmp_lt_i32 s102, 0
	s_mov_b32 s38, s10
	s_cselect_b32 s4, -1, 0
	s_mov_b32 s42, s11
	s_and_b32 s4, s4, s22
	s_mul_i32 s97, s33, s19
	s_and_b32 s4, s4, exec_lo
	s_cselect_b32 s36, 1, 0
	s_not_b32 s4, s21
	s_lshl_b32 s21, s102, 5
	s_add_co_i32 s39, s24, s4
	s_sub_co_i32 s4, s23, s27
	s_lshl_b32 s43, s39, 7
	s_lshl_b32 s34, s4, 2
	s_add_co_i32 s95, s43, s21
	s_set_vgpr_msb 0x8880
	v_and_b32_e32 v76 /*v588*/, 15, v0
	s_cmp_lt_i32 s95, 0
	s_cselect_b32 s94, -1, 0
	s_ashr_i32 s22, s95, 31
	s_ashr_i32 s23, s95, 2
	s_lshr_b32 s24, s22, 30
	s_set_vgpr_msb 0x8000
	v_and_b32_e32 v1, 16, v0
	s_set_vgpr_msb 8
	v_or_b32_e32 v0, s95, v76 /*v588*/
	s_add_co_i32 s22, s91, s23
	s_ashr_i32 s29, s5, 31
	s_sub_co_i32 s27, s18, s23
	s_ashr_i32 s23, s22, 31
	s_set_vgpr_msb 0x800
	v_or_b32_e32 v2, 16, v0
	s_ashr_i32 s35, s34, 31
	s_ashr_i32 s31, s9, 31
	s_mul_u64 s[22:23], s[22:23], s[28:29]
	s_mul_u64 s[44:45], s[34:35], s[30:31]
	v_add_nc_u32_e32 v3, s24, v2
	s_set_vgpr_msb 0x88
	v_mad_u32_u24 v78 /*v590*/, 0x110, v76 /*v588*/, v1
	s_set_vgpr_msb 0x8800
	v_ashrrev_i32_e32 v1, 31, v0
	s_lshl_b64 s[22:23], s[22:23], 1
	s_lshl_b64 s[44:45], s[44:45], 1
	v_and_b32_e32 v4, -4, v3
	s_add_nc_u64 s[2:3], s[2:3], s[22:23]
	v_lshrrev_b32_e32 v1, 30, v1
	s_add_nc_u64 s[22:23], s[2:3], s[44:45]
	s_add_co_i32 s21, s96, 0x20000
	v_cmp_ne_u32_e32 vcc_lo, v2, v4
	v_dual_ashrrev_i32 v2, 2, v3 :: v_dual_add_nc_u32 v1, v0, v1
	s_max_i32 s24, s27, 0
	s_set_vgpr_msb 0x88
	v_add_nc_u32_e32 v83 /*v595*/, s21, v78 /*v590*/
	s_and_b32 vcc_lo, s94, vcc_lo
	v_or_b32_e32 v80 /*v592*/, 0x20000, v78 /*v590*/
	s_set_vgpr_msb 0x8800
	v_and_b32_e32 v5, -4, v1
	s_set_vgpr_msb 0x80
	v_subrev_co_ci_u32_e64 v73 /*v585*/, null, 0, v2, vcc_lo
	s_set_vgpr_msb 0x8000
	v_ashrrev_i32_e32 v1, 2, v1
	s_bitset1_b32 s23, 31
	v_cmp_ne_u32_e64 s2, v0, v5
	v_sub_nc_u32_e32 v0, v0, v5
	s_and_b32 vcc_lo, s94, s2
	s_cmp_lg_u32 s5, 0x80000000
	s_set_vgpr_msb 0x80
	v_add_nc_u32_e32 v75 /*v587*/, s34, v0
	s_cselect_b32 s3, s29, 0
	s_cselect_b32 s2, s5, 0x200
	s_cmp_lg_u32 s9, 0x80000000
	v_subrev_co_ci_u32_e64 v74 /*v586*/, null, 0, v1, vcc_lo
	s_cselect_b32 s41, s9, 0x80
	s_cselect_b32 s9, s31, 0
	s_lshr_b64 s[28:29], s[2:3], 16
	s_lshl_b32 s27, s2, 16
	s_lshr_b32 s29, s2, 16
	s_bfe_i32 s2, s39, 0x10018
	s_ashr_i32 s63, s6, 31
	s_lshr_b32 s6, s2, 30
	s_add_co_i32 s3, s18, -1
	s_or_b32 s6, s6, s43
	s_sub_co_i32 s99, s19, s18
	s_addk_co_i32 s6, 0x7f
	s_ashr_i32 s65, s7, 31
	s_ashr_i32 s6, s6, 2
	s_and_b32 s7, s9, 0xffff
	s_add_co_i32 s6, s6, s2
	s_ashr_i32 s35, s43, 2
	s_min_i32 s100, s6, s3
	s_ashr_i32 s5, s4, 31
	s_add_co_i32 s100, s100, s99
	s_ashr_i32 s39, s10, 31
	s_add_co_i32 s9, s17, s100
	s_ashr_i32 s43, s11, 31
	s_add_co_i32 s9, s9, 1
	s_add_co_i32 s101, s35, s99
	s_min_i32 s9, s9, s19
	s_mul_u64 s[2:3], s[4:5], s[38:39]
	s_max_i32 s9, s9, 1
	s_mul_u64 s[4:5], s[4:5], s[42:43]
	s_or_b32 s42, s7, s27
	s_sub_co_i32 s7, s101, s16
	s_add_co_i32 s10, s9, 0x7f
	s_max_i32 s7, s7, 0
	s_lshr_b32 s10, s10, 7
	s_lshr_b32 s7, s7, 7
	s_add_co_i32 s98, s10, -1
	s_lshl_b64 s[72:73], s[2:3], 1
	s_min_u32 s60, s7, s98
	s_lshl_b64 s[74:75], s[4:5], 1
	s_lshl_b32 s3, s60, 7
	s_and_b32 s6, s28, 0xffff0000
	s_add_co_i32 s2, s3, s97
	s_sub_co_i32 s4, s9, s3
	s_ashr_i32 s3, s2, 31
	s_set_vgpr_msb 0x8000
	v_med3_i32 v0, s4, 0, 0x80
	s_mul_u64 s[4:5], s[2:3], s[62:63]
	s_mul_u64 s[2:3], s[2:3], s[64:65]
	s_lshl_b64 s[4:5], s[4:5], 1
	s_lshl_b64 s[2:3], s[2:3], 1
	v_readfirstlane_b32 s103, v0
	s_add_nc_u64 s[4:5], s[66:67], s[4:5]
	s_add_nc_u64 s[2:3], s[68:69], s[2:3]
	s_or_b32 s43, s6, s29
	s_cmp_lg_u32 s25, s36
	s_add_nc_u64 s[78:79], s[4:5], s[72:73]
	s_add_nc_u64 s[76:77], s[2:3], s[74:75]
	s_mov_b32 s39, 0x807fff
	s_mov_b32 s38, 0xffff7fff
	s_mov_b32 s36, 0x7510000
	s_cbranch_scc0 .LBB0_10
	s_mov_b32 s25, s51
	s_mov_b32 s26, s51
	s_mov_b32 s27, s51
	s_mov_b32 s48, s51
	s_mov_b32 s49, s51
	s_mov_b32 s50, s51
	s_cmp_lg_u32 s62, 0x80000000
	tensor_load_to_lds s[20:23], s[36:43], s[24:27], s[48:51]
	s_mov_b64 s[4:5], s[20:21]
	s_mov_b64 s[6:7], s[22:23]
	s_cselect_b32 s7, s63, 0
	s_cselect_b32 s49, s62, 0x80
	s_and_b32 s5, s102, 3
	s_and_b32 s50, s7, 0xffff
	s_lshl_b32 s25, s5, 5
	s_lshl_b32 s2, s5, 6
	s_sub_co_i32 s11, s103, s25
	s_mov_b32 s3, s51
	s_max_i32 s11, s11, 0
	s_mov_b64 s[30:31], s[22:23]
	s_lshl_b32 s11, s11, 16
	s_mov_b32 s6, s49
	s_or_b32 s46, s11, 0x7fff
	s_cmp_lg_u32 s64, 0x80000000
	s_mul_u64 s[26:27], s[6:7], s[2:3]
	s_cselect_b32 s57, s64, 0x80
	s_cselect_b32 s31, s65, 0
	s_mov_b32 s30, s57
	s_mul_i32 vcc_hi, s5, 0x2400
	s_mul_u64 s[80:81], s[30:31], s[2:3]
	s_mul_i32 s104, s5, 0x2200
	s_add_nc_u64 s[6:7], s[26:27], s[78:79]
	s_mov_b32 s47, 0x800000
	s_mov_b32 s48, 32
	s_bitset1_b32 vcc_hi, 16
	s_add_nc_u64 s[2:3], s[80:81], s[76:77]
	s_mov_b32 s44, s36
	s_mov_b32 s45, s37
	s_mov_b64 s[28:29], s[20:21]
	s_mov_b32 s5, s104
	s_bitset1_b32 s7, 31
	s_mov_b32 s52, 0xf510000
	s_mov_b32 s53, s37
	s_mov_b32 s59, s51
	s_mov_b32 s55, s47
	s_mov_b32 s56, s48
	s_mov_b32 s54, s46
	s_mov_b32 s29, vcc_hi
	s_and_b32 s58, s31, 0xffff
	s_or_b32 s31, s3, 0x80000000
	s_mov_b32 s30, s2
	s_add_co_i32 s2, s101, s17
	s_set_vgpr_msb 0x88
	v_or_b32_e32 v88 /*v600*/, 0x20000, v78 /*v590*/
	s_max_i32 s2, s2, 0
	s_mov_b32 s89, s51
	s_add_co_i32 s2, s2, 1
	s_add_nc_u64 s[84:85], s[68:69], s[74:75]
	s_ashr_i32 s3, s2, 31
	s_add_nc_u64 s[86:87], s[66:67], s[72:73]
	s_lshr_b32 s3, s3, 25
	s_add_co_i32 s3, s2, s3
	s_wait_tensorcnt 0x0
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x8822
	ds_load_b128 v[0:3], v83 /*v595*/
	ds_load_b128 v[4:7], v83 /*v595*/ offset:32
	ds_load_b128 v[8:11], v83 /*v595*/ offset:64
	ds_load_b128 v[12:15], v83 /*v595*/ offset:96
	ds_load_b128 v[16:19], v83 /*v595*/ offset:128
	ds_load_b128 v[20:23], v83 /*v595*/ offset:160
	ds_load_b128 v[24:27], v83 /*v595*/ offset:192
	ds_load_b128 v[28:31], v83 /*v595*/ offset:224
	ds_load_b128 v[32:35], v83 /*v595*/ offset:4352
	ds_load_b128 v[36:39], v83 /*v595*/ offset:4384
	ds_load_b128 v[40:43], v83 /*v595*/ offset:4416
	ds_load_b128 v[44:47], v83 /*v595*/ offset:4448
	ds_load_b128 v[48:51], v83 /*v595*/ offset:4480
	ds_load_b128 v[52:55], v83 /*v595*/ offset:4512
	ds_load_b128 v[56:59], v83 /*v595*/ offset:4544
	ds_load_b128 v[60:63], v83 /*v595*/ offset:4576
	tensor_load_to_lds s[4:7], s[44:51]
	tensor_load_to_lds s[28:31], s[52:59]
	s_and_b32 s4, s3, 0xffffff80
	s_ashr_i32 s3, s3, 7
	s_cmp_lg_u32 s2, s4
	s_wait_dscnt 0xf
	v_pk_mul_bf16 v128, v81 /*v593*/, v0
	s_cselect_b32 s4, -1, 0
	s_cmp_lt_i32 s2, 0
	v_and_or_b32 v0, v77 /*v589*/, 7, v79 /*v591*/
	s_cselect_b32 s2, -1, 0
	s_wait_dscnt 0xe
	v_pk_mul_bf16 v135, v81 /*v593*/, v7
	s_and_b32 s2, s2, s4
	s_sub_co_ci_u32 s2, s3, 0
	s_sub_co_i32 s3, s100, s16
	s_min_i32 s7, s2, s98
	s_max_i32 s3, s3, 0
	s_max_i32 s82, s7, s60
	s_addk_co_i32 s3, 0x7f
	v_mul_u32_u24_e32 v0, 0x120, v0
	s_ashr_i32 s4, s3, 31
	v_pk_mul_bf16 v134, v81 /*v593*/, v6
	s_lshr_b32 s4, s4, 25
	v_pk_mul_bf16 v133, v81 /*v593*/, v5
	s_add_co_i32 s2, s3, s4
	s_set_vgpr_msb 0x2202
	v_and_or_b32 v0, v82 /*v594*/, 16, v0
	s_and_b32 s4, s2, 0xffffff80
	s_ashr_i32 s2, s2, 7
	s_cmp_lg_u32 s3, s4
	v_pk_mul_bf16 v132, v81 /*v593*/, v4
	s_cselect_b32 s4, -1, 0
	s_cmp_lt_i32 s3, 0
	v_pk_mul_bf16 v131, v81 /*v593*/, v3
	s_cselect_b32 s3, -1, 0
	v_pk_mul_bf16 v130, v81 /*v593*/, v2
	s_and_b32 s3, s3, s4
	s_sub_co_ci_u32 s2, s2, 0
	v_pk_mul_bf16 v129, v81 /*v593*/, v1
	s_max_i32 s11, s2, s60
	s_wait_dscnt 0xc
	v_pk_mul_bf16 v143, v81 /*v593*/, v15
	v_pk_mul_bf16 v142, v81 /*v593*/, v14
	v_pk_mul_bf16 v141, v81 /*v593*/, v13
	v_pk_mul_bf16 v140, v81 /*v593*/, v12
	v_pk_mul_bf16 v139, v81 /*v593*/, v11
	v_pk_mul_bf16 v138, v81 /*v593*/, v10
	v_pk_mul_bf16 v137, v81 /*v593*/, v9
	v_pk_mul_bf16 v136, v81 /*v593*/, v8
	s_wait_dscnt 0xa
	v_pk_mul_bf16 v151, v81 /*v593*/, v23
	v_pk_mul_bf16 v150, v81 /*v593*/, v22
	v_pk_mul_bf16 v149, v81 /*v593*/, v21
	v_pk_mul_bf16 v148, v81 /*v593*/, v20
	v_pk_mul_bf16 v147, v81 /*v593*/, v19
	v_pk_mul_bf16 v146, v81 /*v593*/, v18
	v_pk_mul_bf16 v145, v81 /*v593*/, v17
	v_pk_mul_bf16 v144, v81 /*v593*/, v16
	s_wait_dscnt 0x8
	v_pk_mul_bf16 v159, v81 /*v593*/, v31
	v_pk_mul_bf16 v158, v81 /*v593*/, v30
	v_pk_mul_bf16 v157, v81 /*v593*/, v29
	v_pk_mul_bf16 v156, v81 /*v593*/, v28
	v_pk_mul_bf16 v155, v81 /*v593*/, v27
	v_pk_mul_bf16 v154, v81 /*v593*/, v26
	v_pk_mul_bf16 v153, v81 /*v593*/, v25
	v_pk_mul_bf16 v152, v81 /*v593*/, v24
	s_wait_dscnt 0x6
	v_pk_mul_bf16 v167, v81 /*v593*/, v39
	v_pk_mul_bf16 v166, v81 /*v593*/, v38
	v_pk_mul_bf16 v165, v81 /*v593*/, v37
	v_pk_mul_bf16 v164, v81 /*v593*/, v36
	v_pk_mul_bf16 v163, v81 /*v593*/, v35
	v_pk_mul_bf16 v162, v81 /*v593*/, v34
	v_pk_mul_bf16 v161, v81 /*v593*/, v33
	v_pk_mul_bf16 v160, v81 /*v593*/, v32
	s_wait_dscnt 0x4
	v_pk_mul_bf16 v175, v81 /*v593*/, v47
	v_pk_mul_bf16 v174, v81 /*v593*/, v46
	v_pk_mul_bf16 v173, v81 /*v593*/, v45
	v_pk_mul_bf16 v172, v81 /*v593*/, v44
	v_pk_mul_bf16 v171, v81 /*v593*/, v43
	v_pk_mul_bf16 v170, v81 /*v593*/, v42
	v_pk_mul_bf16 v169, v81 /*v593*/, v41
	v_pk_mul_bf16 v168, v81 /*v593*/, v40
	s_wait_dscnt 0x2
	v_pk_mul_bf16 v183, v81 /*v593*/, v55
	v_pk_mul_bf16 v182, v81 /*v593*/, v54
	v_pk_mul_bf16 v181, v81 /*v593*/, v53
	v_pk_mul_bf16 v180, v81 /*v593*/, v52
	v_pk_mul_bf16 v179, v81 /*v593*/, v51
	v_pk_mul_bf16 v178, v81 /*v593*/, v50
	v_pk_mul_bf16 v177, v81 /*v593*/, v49
	v_pk_mul_bf16 v176, v81 /*v593*/, v48
	s_wait_dscnt 0x0
	v_pk_mul_bf16 v191, v81 /*v593*/, v63
	v_pk_mul_bf16 v190, v81 /*v593*/, v62
	v_pk_mul_bf16 v189, v81 /*v593*/, v61
	v_pk_mul_bf16 v188, v81 /*v593*/, v60
	v_pk_mul_bf16 v187, v81 /*v593*/, v59
	v_pk_mul_bf16 v186, v81 /*v593*/, v58
	v_pk_mul_bf16 v185, v81 /*v593*/, v57
	v_pk_mul_bf16 v184, v81 /*v593*/, v56
	s_set_vgpr_msb 0x282
	v_or_b32_e32 v84 /*v596*/, 0x10000, v0
	v_or_b32_e32 v89 /*v601*/, 0x30000, v0
	s_min_i32 s88, s11, s82
	s_cmp_ge_u32 s60, s88
	s_wait_tensorcnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_load_b32 v71 /*v583*/, v75 /*v587*/, s[70:71] scale_offset
	s_set_vgpr_msb 0x8200
	s_cbranch_scc1 .LBB0_11
	s_set_vgpr_msb 10
	v_dual_add_nc_u32 v1, s99, v74 /*v586*/ :: v_dual_add_nc_u32 v2, s99, v73 /*v585*/
	v_dual_mov_b32 v0, 0 :: v_dual_mov_b32 v200, v89 /*v601*/
	s_set_vgpr_msb 0xa40
	v_mov_b32_e32 v248 /*v504*/, 1.0
	s_set_vgpr_msb 0x4080
	v_dual_add_nc_u32 v86 /*v598*/, s17, v1 :: v_dual_add_nc_u32 v87 /*v599*/, s17, v2
	v_subrev_nc_u32_e32 v90 /*v602*/, s16, v1
	v_subrev_nc_u32_e32 v91 /*v603*/, s16, v2
	s_set_vgpr_msb 0x8000
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
	s_set_vgpr_msb 2
	v_mov_b32_e32 v201, v88 /*v600*/
	s_set_vgpr_msb 0x282
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v85 /*v597*/, v78 /*v590*/ :: v_dual_mov_b32 v66 /*v578*/, v71 /*v583*/
	s_set_vgpr_msb 0x8241
	v_mov_b32_e32 v249 /*v505*/, v248 /*v504*/
	s_mov_b32 s61, s51
	s_mov_b32 s28, 1
	s_sub_co_i32 s83, 1, s60
	s_mov_b32 s45, 0xffff0000
	s_mov_b32 s44, 0x7510000
	s_mov_b32 s70, 0x76543210
	s_mov_b32 s90, 0x3fb8aa3b
	s_mov_b64 s[92:93], s[60:61]
	s_set_vgpr_msb 0x4100
	s_branch .LBB0_4
.LBB0_3:
	s_set_vgpr_msb 0x8a
	v_cvt_pk_bf16_f32 v101 /*v613*/, v14 /*v526*/, v26 /*v538*/
	s_set_vgpr_msb 0x8a89
	v_cvt_pk_bf16_f32 v100 /*v612*/, v254 /*v510*/, v10 /*v522*/
	s_set_vgpr_msb 0x8985
	v_cvt_pk_bf16_f32 v99 /*v611*/, v236 /*v492*/, v250 /*v506*/
	v_cvt_pk_bf16_f32 v98 /*v610*/, v222 /*v478*/, v232 /*v488*/
	v_cvt_pk_bf16_f32 v97 /*v609*/, v210 /*v466*/, v218 /*v474*/
	v_cvt_pk_bf16_f32 v96 /*v608*/, v202 /*v458*/, v208 /*v464*/
	v_cvt_pk_bf16_f32 v95 /*v607*/, v196 /*v452*/, v200 /*v456*/
	v_cvt_pk_bf16_f32 v94 /*v606*/, v192 /*v448*/, v194 /*v450*/
	s_set_vgpr_msb 0x858a
	v_cvt_pk_bf16_f32 v109 /*v621*/, v15 /*v527*/, v27 /*v539*/
	s_set_vgpr_msb 0x8a89
	v_cvt_pk_bf16_f32 v108 /*v620*/, v255 /*v511*/, v11 /*v523*/
	s_set_vgpr_msb 0x8985
	v_cvt_pk_bf16_f32 v107 /*v619*/, v237 /*v493*/, v251 /*v507*/
	v_cvt_pk_bf16_f32 v106 /*v618*/, v223 /*v479*/, v233 /*v489*/
	v_cvt_pk_bf16_f32 v105 /*v617*/, v211 /*v467*/, v219 /*v475*/
	v_cvt_pk_bf16_f32 v104 /*v616*/, v203 /*v459*/, v209 /*v465*/
	v_cvt_pk_bf16_f32 v103 /*v615*/, v197 /*v453*/, v201 /*v457*/
	v_cvt_pk_bf16_f32 v102 /*v614*/, v193 /*v449*/, v195 /*v451*/
	s_set_vgpr_msb 0x8509
	s_wait_dscnt 0x3d
	v_wmma_f32_16x16x32_bf16 v[120:127], v[184:191] /*v[440:447]*/, v[94:101] /*v[606:613]*/, v[120:127]
	s_set_vgpr_msb 0x98a
	v_cvt_pk_bf16_f32 v117 /*v629*/, v37 /*v549*/, v45 /*v557*/
	v_cvt_pk_bf16_f32 v116 /*v628*/, v23 /*v535*/, v33 /*v545*/
	v_cvt_pk_bf16_f32 v115 /*v627*/, v7 /*v519*/, v19 /*v531*/
	s_set_vgpr_msb 0x8a89
	v_cvt_pk_bf16_f32 v114 /*v626*/, v245 /*v501*/, v3 /*v515*/
	s_set_vgpr_msb 0x8985
	v_cvt_pk_bf16_f32 v113 /*v625*/, v229 /*v485*/, v241 /*v497*/
	v_cvt_pk_bf16_f32 v112 /*v624*/, v217 /*v473*/, v227 /*v483*/
	v_cvt_pk_bf16_f32 v111 /*v623*/, v207 /*v463*/, v215 /*v471*/
	s_set_vgpr_msb 0x8509
	v_wmma_f32_16x16x32_bf16 v[56:63], v[184:191] /*v[440:447]*/, v[102:109] /*v[614:621]*/, v[56:63]
	s_set_vgpr_msb 0x985
	v_cvt_pk_bf16_f32 v110 /*v622*/, v199 /*v455*/, v205 /*v461*/
	s_set_vgpr_msb 0x854a
	v_cvt_pk_bf16_f32 v199 /*v455*/, v53 /*v565*/, v59 /*v571*/
	v_cvt_pk_bf16_f32 v197 /*v453*/, v31 /*v543*/, v41 /*v553*/
	v_cvt_pk_bf16_f32 v196 /*v452*/, v17 /*v529*/, v29 /*v541*/
	v_cvt_pk_bf16_f32 v191 /*v447*/, v36 /*v548*/, v44 /*v556*/
	v_cvt_pk_bf16_f32 v190 /*v446*/, v22 /*v534*/, v32 /*v544*/
	v_cvt_pk_bf16_f32 v189 /*v445*/, v6 /*v518*/, v18 /*v530*/
	s_set_vgpr_msb 0x4a09
	s_wait_dscnt 0x3c
	v_wmma_f32_16x16x32_bf16 v[112:119], v[152:159] /*v[408:415]*/, v[94:101] /*v[606:613]*/, v[112:119]
	s_set_vgpr_msb 0x949
	v_cvt_pk_bf16_f32 v188 /*v444*/, v244 /*v500*/, v2 /*v514*/
	s_set_vgpr_msb 0x4945
	v_cvt_pk_bf16_f32 v187 /*v443*/, v228 /*v484*/, v240 /*v496*/
	v_cvt_pk_bf16_f32 v186 /*v442*/, v216 /*v472*/, v226 /*v482*/
	v_cvt_pk_bf16_f32 v185 /*v441*/, v206 /*v462*/, v214 /*v470*/
	v_cvt_pk_bf16_f32 v184 /*v440*/, v198 /*v454*/, v204 /*v460*/
	s_set_vgpr_msb 0x454a
	v_cvt_pk_bf16_f32 v198 /*v454*/, v43 /*v555*/, v51 /*v563*/
	v_cvt_pk_bf16_f32 v195 /*v451*/, v1 /*v513*/, v13 /*v525*/
	s_set_vgpr_msb 0x4a09
	v_wmma_f32_16x16x32_bf16 v[48:55], v[152:159] /*v[408:415]*/, v[102:109] /*v[614:621]*/, v[48:55]
	s_set_vgpr_msb 0x945
	v_cvt_pk_bf16_f32 v194 /*v450*/, v239 /*v495*/, v253 /*v509*/
	v_cvt_pk_bf16_f32 v193 /*v449*/, v225 /*v481*/, v235 /*v491*/
	v_cvt_pk_bf16_f32 v192 /*v448*/, v213 /*v469*/, v221 /*v477*/
	s_set_vgpr_msb 0x454a
	v_cvt_pk_bf16_f32 v207 /*v463*/, v63 /*v575*/, v65 /*v577*/
	v_cvt_pk_bf16_f32 v206 /*v462*/, v57 /*v569*/, v61 /*v573*/
	v_cvt_pk_bf16_f32 v205 /*v461*/, v49 /*v561*/, v55 /*v567*/
	v_cvt_pk_bf16_f32 v204 /*v460*/, v39 /*v551*/, v47 /*v559*/
	s_set_vgpr_msb 0x4a09
	s_wait_dscnt 0x2d
	v_wmma_f32_16x16x32_bf16 v[104:111], v[120:127] /*v[376:383]*/, v[94:101] /*v[606:613]*/, v[104:111]
	s_set_vgpr_msb 0x94a
	v_cvt_pk_bf16_f32 v203 /*v459*/, v25 /*v537*/, v35 /*v547*/
	v_cvt_pk_bf16_f32 v202 /*v458*/, v9 /*v521*/, v21 /*v533*/
	s_set_vgpr_msb 0x4a49
	v_cvt_pk_bf16_f32 v201 /*v457*/, v247 /*v503*/, v5 /*v517*/
	s_set_vgpr_msb 0x4945
	v_cvt_pk_bf16_f32 v200 /*v456*/, v231 /*v487*/, v243 /*v499*/
	s_add_nc_u64 s[92:93], s[92:93], 1
	s_wait_dscnt 0x0
	v_cmp_lt_u64_e64 s2, s[92:93], s[88:89]
	s_set_vgpr_msb 0x4509
	v_wmma_f32_16x16x32_bf16 v[40:47], v[120:127] /*v[376:383]*/, v[102:109] /*v[614:621]*/, v[40:47]
	s_and_b32 vcc_lo, exec_lo, s2
	v_wmma_f32_16x16x32_bf16 v[96:103], v[88:95] /*v[344:351]*/, v[94:101] /*v[606:613]*/, v[96:103]
	v_wmma_f32_16x16x32_bf16 v[32:39], v[88:95] /*v[344:351]*/, v[102:109] /*v[614:621]*/, v[32:39]
	v_wmma_f32_16x16x32_bf16 v[88:95], v[56:63] /*v[312:319]*/, v[94:101] /*v[606:613]*/, v[88:95]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[56:63] /*v[312:319]*/, v[102:109] /*v[614:621]*/, v[24:31]
	v_wmma_f32_16x16x32_bf16 v[80:87], v[24:31] /*v[280:287]*/, v[94:101] /*v[606:613]*/, v[80:87]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[24:31] /*v[280:287]*/, v[102:109] /*v[614:621]*/, v[16:23]
	s_set_vgpr_msb 0x908
	v_wmma_f32_16x16x32_bf16 v[72:79], v[248:255], v[94:101] /*v[606:613]*/, v[72:79]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[248:255], v[102:109] /*v[614:621]*/, v[8:15]
	v_wmma_f32_16x16x32_bf16 v[64:71], v[216:223], v[94:101] /*v[606:613]*/, v[64:71]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[216:223], v[102:109] /*v[614:621]*/, v[0:7]
	s_set_vgpr_msb 0x805
	v_wmma_f32_16x16x32_bf16 v[120:127], v[176:183] /*v[432:439]*/, v[184:191] /*v[440:447]*/, v[120:127]
	s_set_vgpr_msb 0x509
	v_wmma_f32_16x16x32_bf16 v[56:63], v[176:183] /*v[432:439]*/, v[110:117] /*v[622:629]*/, v[56:63]
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x94a
	v_cvt_pk_bf16_f32 v183 /*v439*/, v52 /*v564*/, v58 /*v570*/
	v_cvt_pk_bf16_f32 v182 /*v438*/, v42 /*v554*/, v50 /*v562*/
	v_cvt_pk_bf16_f32 v181 /*v437*/, v30 /*v542*/, v40 /*v552*/
	s_set_vgpr_msb 0x4a05
	v_wmma_f32_16x16x32_bf16 v[112:119], v[144:151] /*v[400:407]*/, v[184:191] /*v[440:447]*/, v[112:119]
	s_set_vgpr_msb 0x54a
	v_cvt_pk_bf16_f32 v180 /*v436*/, v16 /*v528*/, v28 /*v540*/
	v_cvt_pk_bf16_f32 v179 /*v435*/, v0 /*v512*/, v12 /*v524*/
	s_set_vgpr_msb 0x4a45
	v_cvt_pk_bf16_f32 v178 /*v434*/, v238 /*v494*/, v252 /*v508*/
	v_cvt_pk_bf16_f32 v177 /*v433*/, v224 /*v480*/, v234 /*v490*/
	v_cvt_pk_bf16_f32 v176 /*v432*/, v212 /*v468*/, v220 /*v476*/
	s_set_vgpr_msb 0x4509
	v_wmma_f32_16x16x32_bf16 v[48:55], v[144:151] /*v[400:407]*/, v[110:117] /*v[622:629]*/, v[48:55]
	s_set_vgpr_msb 0x905
	v_wmma_f32_16x16x32_bf16 v[104:111], v[112:119] /*v[368:375]*/, v[184:191] /*v[440:447]*/, v[104:111]
	s_set_vgpr_msb 0x509
	v_wmma_f32_16x16x32_bf16 v[40:47], v[112:119] /*v[368:375]*/, v[110:117] /*v[622:629]*/, v[40:47]
	s_set_vgpr_msb 0x905
	v_wmma_f32_16x16x32_bf16 v[96:103], v[80:87] /*v[336:343]*/, v[184:191] /*v[440:447]*/, v[96:103]
	s_set_vgpr_msb 0x509
	v_wmma_f32_16x16x32_bf16 v[32:39], v[80:87] /*v[336:343]*/, v[110:117] /*v[622:629]*/, v[32:39]
	s_set_vgpr_msb 0x905
	v_wmma_f32_16x16x32_bf16 v[88:95], v[48:55] /*v[304:311]*/, v[184:191] /*v[440:447]*/, v[88:95]
	s_set_vgpr_msb 0x509
	v_wmma_f32_16x16x32_bf16 v[24:31], v[48:55] /*v[304:311]*/, v[110:117] /*v[622:629]*/, v[24:31]
	s_set_vgpr_msb 0x905
	v_wmma_f32_16x16x32_bf16 v[80:87], v[16:23] /*v[272:279]*/, v[184:191] /*v[440:447]*/, v[80:87]
	s_set_vgpr_msb 0x509
	v_wmma_f32_16x16x32_bf16 v[16:23], v[16:23] /*v[272:279]*/, v[110:117] /*v[622:629]*/, v[16:23]
	s_set_vgpr_msb 0x904
	v_wmma_f32_16x16x32_bf16 v[72:79], v[240:247], v[184:191] /*v[440:447]*/, v[72:79]
	s_set_vgpr_msb 0x408
	v_wmma_f32_16x16x32_bf16 v[8:15], v[240:247], v[110:117] /*v[622:629]*/, v[8:15]
	s_set_vgpr_msb 0x804
	v_wmma_f32_16x16x32_bf16 v[64:71], v[208:215], v[184:191] /*v[440:447]*/, v[64:71]
	s_set_vgpr_msb 0x408
	v_wmma_f32_16x16x32_bf16 v[0:7], v[208:215], v[110:117] /*v[622:629]*/, v[0:7]
	s_set_vgpr_msb 0x805
	v_wmma_f32_16x16x32_bf16 v[120:127], v[168:175] /*v[424:431]*/, v[176:183] /*v[432:439]*/, v[120:127]
	v_wmma_f32_16x16x32_bf16 v[56:63], v[168:175] /*v[424:431]*/, v[192:199] /*v[448:455]*/, v[56:63]
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x54a
	v_cvt_pk_bf16_f32 v175 /*v431*/, v62 /*v574*/, v64 /*v576*/
	v_cvt_pk_bf16_f32 v174 /*v430*/, v56 /*v568*/, v60 /*v572*/
	v_cvt_pk_bf16_f32 v173 /*v429*/, v48 /*v560*/, v54 /*v566*/
	s_set_vgpr_msb 0x4a05
	v_wmma_f32_16x16x32_bf16 v[112:119], v[136:143] /*v[392:399]*/, v[176:183] /*v[432:439]*/, v[112:119]
	s_set_vgpr_msb 0x54a
	v_cvt_pk_bf16_f32 v172 /*v428*/, v38 /*v550*/, v46 /*v558*/
	v_cvt_pk_bf16_f32 v171 /*v427*/, v24 /*v536*/, v34 /*v546*/
	v_cvt_pk_bf16_f32 v170 /*v426*/, v8 /*v520*/, v20 /*v532*/
	s_set_vgpr_msb 0x4a49
	v_cvt_pk_bf16_f32 v169 /*v425*/, v246 /*v502*/, v4 /*v516*/
	s_set_vgpr_msb 0x4945
	v_cvt_pk_bf16_f32 v168 /*v424*/, v230 /*v486*/, v242 /*v498*/
	s_set_vgpr_msb 0x4505
	v_wmma_f32_16x16x32_bf16 v[48:55], v[136:143] /*v[392:399]*/, v[192:199] /*v[448:455]*/, v[48:55]
	v_wmma_f32_16x16x32_bf16 v[104:111], v[104:111] /*v[360:367]*/, v[176:183] /*v[432:439]*/, v[104:111]
	v_wmma_f32_16x16x32_bf16 v[40:47], v[104:111] /*v[360:367]*/, v[192:199] /*v[448:455]*/, v[40:47]
	v_wmma_f32_16x16x32_bf16 v[96:103], v[72:79] /*v[328:335]*/, v[176:183] /*v[432:439]*/, v[96:103]
	v_wmma_f32_16x16x32_bf16 v[32:39], v[72:79] /*v[328:335]*/, v[192:199] /*v[448:455]*/, v[32:39]
	v_wmma_f32_16x16x32_bf16 v[88:95], v[40:47] /*v[296:303]*/, v[176:183] /*v[432:439]*/, v[88:95]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[40:47] /*v[296:303]*/, v[192:199] /*v[448:455]*/, v[24:31]
	v_wmma_f32_16x16x32_bf16 v[80:87], v[8:15] /*v[264:271]*/, v[176:183] /*v[432:439]*/, v[80:87]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[8:15] /*v[264:271]*/, v[192:199] /*v[448:455]*/, v[16:23]
	s_set_vgpr_msb 0x504
	v_wmma_f32_16x16x32_bf16 v[72:79], v[232:239], v[176:183] /*v[432:439]*/, v[72:79]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[232:239], v[192:199] /*v[448:455]*/, v[8:15]
	v_wmma_f32_16x16x32_bf16 v[64:71], v[200:207], v[176:183] /*v[432:439]*/, v[64:71]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[200:207], v[192:199] /*v[448:455]*/, v[0:7]
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x426
	v_pk_fma_f32 v[200:201], v[70:71] /*v[582:583]*/, v[248:249] /*v[504:505]*/, v[68:69] /*v[580:581]*/
	s_set_vgpr_msb 0x2682
	v_mov_b32_e32 v71 /*v583*/, v93 /*v605*/
	s_set_vgpr_msb 0x8248
	v_pk_add_f32 v[248:249] /*v[504:505]*/, v[200:201], v[66:67] /*v[578:579]*/
	s_set_vgpr_msb 0x4805
	v_wmma_f32_16x16x32_bf16 v[120:127], v[160:167] /*v[416:423]*/, v[168:175] /*v[424:431]*/, v[120:127]
	s_set_vgpr_msb 0x502
	v_dual_mov_b32 v200, v89 /*v601*/ :: v_dual_mov_b32 v201, v88 /*v600*/
	s_set_vgpr_msb 0x282
	v_mov_b32_e32 v66 /*v578*/, v92 /*v604*/
	s_set_vgpr_msb 0x8205
	v_wmma_f32_16x16x32_bf16 v[56:63], v[160:167] /*v[416:423]*/, v[200:207] /*v[456:463]*/, v[56:63]
	v_wmma_f32_16x16x32_bf16 v[112:119], v[128:135] /*v[384:391]*/, v[168:175] /*v[424:431]*/, v[112:119]
	v_wmma_f32_16x16x32_bf16 v[48:55], v[128:135] /*v[384:391]*/, v[200:207] /*v[456:463]*/, v[48:55]
	v_wmma_f32_16x16x32_bf16 v[104:111], v[96:103] /*v[352:359]*/, v[168:175] /*v[424:431]*/, v[104:111]
	v_wmma_f32_16x16x32_bf16 v[40:47], v[96:103] /*v[352:359]*/, v[200:207] /*v[456:463]*/, v[40:47]
	v_wmma_f32_16x16x32_bf16 v[96:103], v[64:71] /*v[320:327]*/, v[168:175] /*v[424:431]*/, v[96:103]
	v_wmma_f32_16x16x32_bf16 v[32:39], v[64:71] /*v[320:327]*/, v[200:207] /*v[456:463]*/, v[32:39]
	v_wmma_f32_16x16x32_bf16 v[88:95], v[32:39] /*v[288:295]*/, v[168:175] /*v[424:431]*/, v[88:95]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[32:39] /*v[288:295]*/, v[200:207] /*v[456:463]*/, v[24:31]
	v_wmma_f32_16x16x32_bf16 v[80:87], v[0:7] /*v[256:263]*/, v[168:175] /*v[424:431]*/, v[80:87]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[0:7] /*v[256:263]*/, v[200:207] /*v[456:463]*/, v[16:23]
	s_set_vgpr_msb 0x504
	v_wmma_f32_16x16x32_bf16 v[72:79], v[224:231], v[168:175] /*v[424:431]*/, v[72:79]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[224:231], v[200:207] /*v[456:463]*/, v[8:15]
	v_wmma_f32_16x16x32_bf16 v[64:71], v[192:199], v[168:175] /*v[424:431]*/, v[64:71]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[192:199], v[200:207] /*v[456:463]*/, v[0:7]
	s_set_vgpr_msb 0x400
	s_cbranch_vccz .LBB0_12
.LBB0_4:
	s_wait_alu depctr_vm_vsrc(0)
	s_set_vgpr_msb 0x82
	v_dual_mov_b32 v88 /*v600*/, v85 /*v597*/ :: v_dual_mov_b32 v89 /*v601*/, v84 /*v596*/
	s_set_vgpr_msb 0x8280
	v_dual_mov_b32 v85 /*v597*/, v201 :: v_dual_mov_b32 v84 /*v596*/, v200
	s_add_co_i32 s2, s92, 1
	s_wait_tensorcnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	s_cmp_ge_i32 s2, s10
	s_set_vgpr_msb 0x8000
	s_cbranch_scc1 .LBB0_6
	s_lshl_b32 s4, s2, 7
	s_add_co_i32 s6, s92, s83
	s_sub_co_i32 s30, s9, s4
	s_add_co_i32 s2, s4, s97
	v_nop
	v_nop
	v_med3_i32 v192, s30, 0, 0x80
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
	s_or_b32 s46, s2, 0x7fff
	s_mov_b32 s53, s45
	tensor_load_to_lds s[28:31], s[44:51]
	s_add_nc_u64 s[30:31], s[80:81], s[4:5]
	s_or_b32 s29, vcc_hi, s6
	s_bitset1_b32 s31, 31
	s_mov_b32 s54, s46
	s_mov_b32 s55, s47
	s_mov_b32 s56, s48
	s_mov_b32 s59, s51
	tensor_load_to_lds s[28:31], s[52:59]
.LBB0_6:
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 2
	ds_load_b128 v[192:195], v88 /*v600*/
	ds_load_b128 v[196:199], v88 /*v600*/ offset:32
	ds_load_b128 v[200:203], v88 /*v600*/ offset:64
	ds_load_b128 v[204:207], v88 /*v600*/ offset:96
	ds_load_b128 v[208:211], v88 /*v600*/ offset:128
	ds_load_b128 v[212:215], v88 /*v600*/ offset:160
	ds_load_b128 v[216:219], v88 /*v600*/ offset:192
	ds_load_b128 v[220:223], v88 /*v600*/ offset:224
	ds_load_b128 v[224:227], v88 /*v600*/ offset:4352
	ds_load_b128 v[228:231], v88 /*v600*/ offset:4384
	ds_load_b128 v[232:235], v88 /*v600*/ offset:4416
	ds_load_b128 v[236:239], v88 /*v600*/ offset:4448
	ds_load_b128 v[240:243], v88 /*v600*/ offset:4480
	ds_load_b128 v[244:247], v88 /*v600*/ offset:4512
	ds_load_b128 v[248:251], v88 /*v600*/ offset:4544
	ds_load_b128 v[252:255], v88 /*v600*/ offset:4576
	s_set_vgpr_msb 0x242
	ds_load_b128 v[0:3] /*v[256:259]*/, v88 /*v600*/ offset:8704
	ds_load_b128 v[4:7] /*v[260:263]*/, v88 /*v600*/ offset:8736
	ds_load_b128 v[8:11] /*v[264:267]*/, v88 /*v600*/ offset:8768
	ds_load_b128 v[12:15] /*v[268:271]*/, v88 /*v600*/ offset:8800
	ds_load_b128 v[16:19] /*v[272:275]*/, v88 /*v600*/ offset:8832
	ds_load_b128 v[20:23] /*v[276:279]*/, v88 /*v600*/ offset:8864
	ds_load_b128 v[24:27] /*v[280:283]*/, v88 /*v600*/ offset:8896
	ds_load_b128 v[28:31] /*v[284:287]*/, v88 /*v600*/ offset:8928
	ds_load_b128 v[32:35] /*v[288:291]*/, v88 /*v600*/ offset:13056
	ds_load_b128 v[36:39] /*v[292:295]*/, v88 /*v600*/ offset:13088
	ds_load_b128 v[40:43] /*v[296:299]*/, v88 /*v600*/ offset:13120
	ds_load_b128 v[44:47] /*v[300:303]*/, v88 /*v600*/ offset:13152
	ds_load_b128 v[48:51] /*v[304:307]*/, v88 /*v600*/ offset:13184
	ds_load_b128 v[52:55] /*v[308:311]*/, v88 /*v600*/ offset:13216
	ds_load_b128 v[56:59] /*v[312:315]*/, v88 /*v600*/ offset:13248
	ds_load_b128 v[60:63] /*v[316:319]*/, v88 /*v600*/ offset:13280
	ds_load_b128 v[64:67] /*v[320:323]*/, v88 /*v600*/ offset:17408
	ds_load_b128 v[68:71] /*v[324:327]*/, v88 /*v600*/ offset:17440
	ds_load_b128 v[72:75] /*v[328:331]*/, v88 /*v600*/ offset:17472
	ds_load_b128 v[76:79] /*v[332:335]*/, v88 /*v600*/ offset:17504
	ds_load_b128 v[80:83] /*v[336:339]*/, v88 /*v600*/ offset:17536
	ds_load_b128 v[84:87] /*v[340:343]*/, v88 /*v600*/ offset:17568
	ds_load_b128 v[88:91] /*v[344:347]*/, v88 /*v600*/ offset:17600
	ds_load_b128 v[92:95] /*v[348:351]*/, v88 /*v600*/ offset:17632
	ds_load_b128 v[96:99] /*v[352:355]*/, v88 /*v600*/ offset:21760
	ds_load_b128 v[100:103] /*v[356:359]*/, v88 /*v600*/ offset:21792
	ds_load_b128 v[104:107] /*v[360:363]*/, v88 /*v600*/ offset:21824
	ds_load_b128 v[108:111] /*v[364:367]*/, v88 /*v600*/ offset:21856
	ds_load_b128 v[112:115] /*v[368:371]*/, v88 /*v600*/ offset:21888
	ds_load_b128 v[116:119] /*v[372:375]*/, v88 /*v600*/ offset:21920
	ds_load_b128 v[120:123] /*v[376:379]*/, v88 /*v600*/ offset:21952
	ds_load_b128 v[124:127] /*v[380:383]*/, v88 /*v600*/ offset:21984
	ds_load_b128 v[128:131] /*v[384:387]*/, v88 /*v600*/ offset:26112
	ds_load_b128 v[132:135] /*v[388:391]*/, v88 /*v600*/ offset:26144
	ds_load_b128 v[136:139] /*v[392:395]*/, v88 /*v600*/ offset:26176
	ds_load_b128 v[140:143] /*v[396:399]*/, v88 /*v600*/ offset:26208
	ds_load_b128 v[144:147] /*v[400:403]*/, v88 /*v600*/ offset:26240
	ds_load_b128 v[148:151] /*v[404:407]*/, v88 /*v600*/ offset:26272
	ds_load_b128 v[152:155] /*v[408:411]*/, v88 /*v600*/ offset:26304
	ds_load_b128 v[156:159] /*v[412:415]*/, v88 /*v600*/ offset:26336
	ds_load_b128 v[160:163] /*v[416:419]*/, v88 /*v600*/ offset:30464
	ds_load_b128 v[164:167] /*v[420:423]*/, v88 /*v600*/ offset:30496
	ds_load_b128 v[168:171] /*v[424:427]*/, v88 /*v600*/ offset:30528
	ds_load_b128 v[172:175] /*v[428:431]*/, v88 /*v600*/ offset:30560
	ds_load_b128 v[176:179] /*v[432:435]*/, v88 /*v600*/ offset:30592
	ds_load_b128 v[180:183] /*v[436:439]*/, v88 /*v600*/ offset:30624
	ds_load_b128 v[184:187] /*v[440:443]*/, v88 /*v600*/ offset:30656
	ds_load_b128 v[188:191] /*v[444:447]*/, v88 /*v600*/ offset:30688
	s_set_vgpr_msb 0x4240
	s_wait_dscnt 0x3e
	v_wmma_f32_16x16x32_bf16 v[250:257] /*v[506:513]*/, v[192:199], v[128:135], 0
	v_wmma_f32_16x16x32_bf16 v[192:199] /*v[448:455]*/, v[192:199], v[160:167], 0
	s_set_vgpr_msb 0x4080
	s_wait_dscnt 0x36
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[514:521]*/, v[224:231], v[128:135], 0
	s_set_vgpr_msb 0x8040
	v_wmma_f32_16x16x32_bf16 v[200:207] /*v[456:463]*/, v[224:231], v[160:167], 0
	s_set_vgpr_msb 0x4081
	s_wait_dscnt 0x2e
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[522:529]*/, v[0:7] /*v[256:263]*/, v[128:135], 0
	s_set_vgpr_msb 0x8141
	v_wmma_f32_16x16x32_bf16 v[208:215] /*v[464:471]*/, v[0:7] /*v[256:263]*/, v[160:167], 0
	s_set_vgpr_msb 0x4181
	s_wait_dscnt 0x26
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[530:537]*/, v[32:39] /*v[288:295]*/, v[128:135], 0
	s_set_vgpr_msb 0x8141
	v_wmma_f32_16x16x32_bf16 v[216:223] /*v[472:479]*/, v[32:39] /*v[288:295]*/, v[160:167], 0
	s_set_vgpr_msb 0x4181
	s_wait_dscnt 0x1e
	v_wmma_f32_16x16x32_bf16 v[26:33] /*v[538:545]*/, v[64:71] /*v[320:327]*/, v[128:135], 0
	s_set_vgpr_msb 0x8141
	v_wmma_f32_16x16x32_bf16 v[224:231] /*v[480:487]*/, v[64:71] /*v[320:327]*/, v[160:167], 0
	s_set_vgpr_msb 0x4181
	s_wait_dscnt 0x16
	v_wmma_f32_16x16x32_bf16 v[34:41] /*v[546:553]*/, v[96:103] /*v[352:359]*/, v[128:135], 0
	s_set_vgpr_msb 0x8141
	v_wmma_f32_16x16x32_bf16 v[232:239] /*v[488:495]*/, v[96:103] /*v[352:359]*/, v[160:167], 0
	s_set_vgpr_msb 0x4181
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[554:561]*/, v[128:135] /*v[384:391]*/, v[128:135], 0
	s_set_vgpr_msb 0x8141
	v_wmma_f32_16x16x32_bf16 v[240:247] /*v[496:503]*/, v[128:135] /*v[384:391]*/, v[160:167], 0
	s_set_vgpr_msb 0x4181
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[562:569]*/, v[160:167] /*v[416:423]*/, v[128:135], 0
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[570:577]*/, v[160:167] /*v[416:423]*/, v[160:167], 0
	s_set_vgpr_msb 0x8150
	v_wmma_f32_16x16x32_bf16 v[250:257] /*v[506:513]*/, v[200:207], v[136:143], v[250:257] /*v[506:513]*/
	v_wmma_f32_16x16x32_bf16 v[192:199] /*v[448:455]*/, v[200:207], v[168:175], v[192:199] /*v[448:455]*/
	s_set_vgpr_msb 0x50a0
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[514:521]*/, v[232:239], v[136:143], v[2:9] /*v[514:521]*/
	s_set_vgpr_msb 0xa050
	v_wmma_f32_16x16x32_bf16 v[200:207] /*v[456:463]*/, v[232:239], v[168:175], v[200:207] /*v[456:463]*/
	s_set_vgpr_msb 0x50a1
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[522:529]*/, v[8:15] /*v[264:271]*/, v[136:143], v[10:17] /*v[522:529]*/
	s_set_vgpr_msb 0xa151
	v_wmma_f32_16x16x32_bf16 v[208:215] /*v[464:471]*/, v[8:15] /*v[264:271]*/, v[168:175], v[208:215] /*v[464:471]*/
	s_set_vgpr_msb 0x51a1
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[530:537]*/, v[40:47] /*v[296:303]*/, v[136:143], v[18:25] /*v[530:537]*/
	s_set_vgpr_msb 0xa151
	v_wmma_f32_16x16x32_bf16 v[216:223] /*v[472:479]*/, v[40:47] /*v[296:303]*/, v[168:175], v[216:223] /*v[472:479]*/
	s_set_vgpr_msb 0x51a1
	v_wmma_f32_16x16x32_bf16 v[26:33] /*v[538:545]*/, v[72:79] /*v[328:335]*/, v[136:143], v[26:33] /*v[538:545]*/
	s_set_vgpr_msb 0xa151
	v_wmma_f32_16x16x32_bf16 v[224:231] /*v[480:487]*/, v[72:79] /*v[328:335]*/, v[168:175], v[224:231] /*v[480:487]*/
	s_set_vgpr_msb 0x51a1
	v_wmma_f32_16x16x32_bf16 v[34:41] /*v[546:553]*/, v[104:111] /*v[360:367]*/, v[136:143], v[34:41] /*v[546:553]*/
	s_set_vgpr_msb 0xa151
	v_wmma_f32_16x16x32_bf16 v[232:239] /*v[488:495]*/, v[104:111] /*v[360:367]*/, v[168:175], v[232:239] /*v[488:495]*/
	s_set_vgpr_msb 0x51a1
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[554:561]*/, v[136:143] /*v[392:399]*/, v[136:143], v[42:49] /*v[554:561]*/
	s_set_vgpr_msb 0xa151
	v_wmma_f32_16x16x32_bf16 v[240:247] /*v[496:503]*/, v[136:143] /*v[392:399]*/, v[168:175], v[240:247] /*v[496:503]*/
	s_set_vgpr_msb 0x51a1
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[562:569]*/, v[168:175] /*v[424:431]*/, v[136:143], v[50:57] /*v[562:569]*/
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[570:577]*/, v[168:175] /*v[424:431]*/, v[168:175], v[58:65] /*v[570:577]*/
	s_set_vgpr_msb 0xa150
	v_wmma_f32_16x16x32_bf16 v[250:257] /*v[506:513]*/, v[208:215], v[144:151], v[250:257] /*v[506:513]*/
	v_wmma_f32_16x16x32_bf16 v[192:199] /*v[448:455]*/, v[208:215], v[176:183], v[192:199] /*v[448:455]*/
	s_set_vgpr_msb 0x50a0
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[514:521]*/, v[240:247], v[144:151], v[2:9] /*v[514:521]*/
	s_set_vgpr_msb 0xa050
	v_wmma_f32_16x16x32_bf16 v[200:207] /*v[456:463]*/, v[240:247], v[176:183], v[200:207] /*v[456:463]*/
	s_set_vgpr_msb 0x50a1
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[522:529]*/, v[16:23] /*v[272:279]*/, v[144:151], v[10:17] /*v[522:529]*/
	s_set_vgpr_msb 0xa151
	v_wmma_f32_16x16x32_bf16 v[208:215] /*v[464:471]*/, v[16:23] /*v[272:279]*/, v[176:183], v[208:215] /*v[464:471]*/
	s_set_vgpr_msb 0x51a1
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[530:537]*/, v[48:55] /*v[304:311]*/, v[144:151], v[18:25] /*v[530:537]*/
	s_set_vgpr_msb 0xa151
	v_wmma_f32_16x16x32_bf16 v[216:223] /*v[472:479]*/, v[48:55] /*v[304:311]*/, v[176:183], v[216:223] /*v[472:479]*/
	s_set_vgpr_msb 0x51a1
	v_wmma_f32_16x16x32_bf16 v[26:33] /*v[538:545]*/, v[80:87] /*v[336:343]*/, v[144:151], v[26:33] /*v[538:545]*/
	s_set_vgpr_msb 0xa151
	v_wmma_f32_16x16x32_bf16 v[224:231] /*v[480:487]*/, v[80:87] /*v[336:343]*/, v[176:183], v[224:231] /*v[480:487]*/
	s_set_vgpr_msb 0x51a1
	v_wmma_f32_16x16x32_bf16 v[34:41] /*v[546:553]*/, v[112:119] /*v[368:375]*/, v[144:151], v[34:41] /*v[546:553]*/
	s_set_vgpr_msb 0xa151
	v_wmma_f32_16x16x32_bf16 v[232:239] /*v[488:495]*/, v[112:119] /*v[368:375]*/, v[176:183], v[232:239] /*v[488:495]*/
	s_set_vgpr_msb 0x51a1
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[554:561]*/, v[144:151] /*v[400:407]*/, v[144:151], v[42:49] /*v[554:561]*/
	s_set_vgpr_msb 0xa151
	v_wmma_f32_16x16x32_bf16 v[240:247] /*v[496:503]*/, v[144:151] /*v[400:407]*/, v[176:183], v[240:247] /*v[496:503]*/
	s_set_vgpr_msb 0x51a1
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[562:569]*/, v[176:183] /*v[432:439]*/, v[144:151], v[50:57] /*v[562:569]*/
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[570:577]*/, v[176:183] /*v[432:439]*/, v[176:183], v[58:65] /*v[570:577]*/
	s_set_vgpr_msb 0xa150
	v_wmma_f32_16x16x32_bf16 v[250:257] /*v[506:513]*/, v[216:223], v[152:159], v[250:257] /*v[506:513]*/
	v_wmma_f32_16x16x32_bf16 v[192:199] /*v[448:455]*/, v[216:223], v[184:191], v[192:199] /*v[448:455]*/
	s_set_vgpr_msb 0x50a0
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[514:521]*/, v[248:255], v[152:159], v[2:9] /*v[514:521]*/
	s_set_vgpr_msb 0xa050
	v_wmma_f32_16x16x32_bf16 v[200:207] /*v[456:463]*/, v[248:255], v[184:191], v[200:207] /*v[456:463]*/
	s_set_vgpr_msb 0x50a1
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[522:529]*/, v[24:31] /*v[280:287]*/, v[152:159], v[10:17] /*v[522:529]*/
	s_set_vgpr_msb 0xa151
	v_wmma_f32_16x16x32_bf16 v[208:215] /*v[464:471]*/, v[24:31] /*v[280:287]*/, v[184:191], v[208:215] /*v[464:471]*/
	s_set_vgpr_msb 0x51a1
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[530:537]*/, v[56:63] /*v[312:319]*/, v[152:159], v[18:25] /*v[530:537]*/
	s_set_vgpr_msb 0xa151
	v_wmma_f32_16x16x32_bf16 v[216:223] /*v[472:479]*/, v[56:63] /*v[312:319]*/, v[184:191], v[216:223] /*v[472:479]*/
	s_set_vgpr_msb 0x51a1
	v_wmma_f32_16x16x32_bf16 v[26:33] /*v[538:545]*/, v[88:95] /*v[344:351]*/, v[152:159], v[26:33] /*v[538:545]*/
	s_set_vgpr_msb 0xa151
	v_wmma_f32_16x16x32_bf16 v[224:231] /*v[480:487]*/, v[88:95] /*v[344:351]*/, v[184:191], v[224:231] /*v[480:487]*/
	s_set_vgpr_msb 0x51a1
	v_wmma_f32_16x16x32_bf16 v[34:41] /*v[546:553]*/, v[120:127] /*v[376:383]*/, v[152:159], v[34:41] /*v[546:553]*/
	s_set_vgpr_msb 0xa151
	v_wmma_f32_16x16x32_bf16 v[232:239] /*v[488:495]*/, v[120:127] /*v[376:383]*/, v[184:191], v[232:239] /*v[488:495]*/
	s_set_vgpr_msb 0x51a1
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[554:561]*/, v[152:159] /*v[408:415]*/, v[152:159], v[42:49] /*v[554:561]*/
	s_set_vgpr_msb 0xa151
	v_wmma_f32_16x16x32_bf16 v[240:247] /*v[496:503]*/, v[152:159] /*v[408:415]*/, v[184:191], v[240:247] /*v[496:503]*/
	s_set_vgpr_msb 0x51a1
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[562:569]*/, v[184:191] /*v[440:447]*/, v[152:159], v[50:57] /*v[562:569]*/
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[570:577]*/, v[184:191] /*v[440:447]*/, v[184:191], v[58:65] /*v[570:577]*/
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0xa142
	ds_load_tr16_b128 v[184:187] /*v[440:443]*/, v89 /*v601*/
	ds_load_tr16_b128 v[152:155] /*v[408:411]*/, v89 /*v601*/ offset:32
	ds_load_tr16_b128 v[188:191] /*v[444:447]*/, v89 /*v601*/ offset:4608
	ds_load_tr16_b128 v[156:159] /*v[412:415]*/, v89 /*v601*/ offset:4640
	ds_load_tr16_b128 v[176:179] /*v[432:435]*/, v89 /*v601*/ offset:9216
	ds_load_tr16_b128 v[144:147] /*v[400:403]*/, v89 /*v601*/ offset:9248
	ds_load_tr16_b128 v[180:183] /*v[436:439]*/, v89 /*v601*/ offset:13824
	ds_load_tr16_b128 v[148:151] /*v[404:407]*/, v89 /*v601*/ offset:13856
	ds_load_tr16_b128 v[168:171] /*v[424:427]*/, v89 /*v601*/ offset:18432
	ds_load_tr16_b128 v[136:139] /*v[392:395]*/, v89 /*v601*/ offset:18464
	ds_load_tr16_b128 v[172:175] /*v[428:431]*/, v89 /*v601*/ offset:23040
	ds_load_tr16_b128 v[140:143] /*v[396:399]*/, v89 /*v601*/ offset:23072
	ds_load_tr16_b128 v[160:163] /*v[416:419]*/, v89 /*v601*/ offset:27648
	ds_load_tr16_b128 v[128:131] /*v[384:387]*/, v89 /*v601*/ offset:27680
	ds_load_tr16_b128 v[164:167] /*v[420:423]*/, v89 /*v601*/ offset:32256
	ds_load_tr16_b128 v[132:135] /*v[388:391]*/, v89 /*v601*/ offset:32288
	ds_load_tr16_b128 v[120:123] /*v[376:379]*/, v89 /*v601*/ offset:64
	ds_load_tr16_b128 v[88:91] /*v[344:347]*/, v89 /*v601*/ offset:96
	ds_load_tr16_b128 v[124:127] /*v[380:383]*/, v89 /*v601*/ offset:4672
	ds_load_tr16_b128 v[92:95] /*v[348:351]*/, v89 /*v601*/ offset:4704
	ds_load_tr16_b128 v[112:115] /*v[368:371]*/, v89 /*v601*/ offset:9280
	ds_load_tr16_b128 v[80:83] /*v[336:339]*/, v89 /*v601*/ offset:9312
	ds_load_tr16_b128 v[116:119] /*v[372:375]*/, v89 /*v601*/ offset:13888
	ds_load_tr16_b128 v[84:87] /*v[340:343]*/, v89 /*v601*/ offset:13920
	ds_load_tr16_b128 v[104:107] /*v[360:363]*/, v89 /*v601*/ offset:18496
	ds_load_tr16_b128 v[72:75] /*v[328:331]*/, v89 /*v601*/ offset:18528
	ds_load_tr16_b128 v[108:111] /*v[364:367]*/, v89 /*v601*/ offset:23104
	ds_load_tr16_b128 v[76:79] /*v[332:335]*/, v89 /*v601*/ offset:23136
	ds_load_tr16_b128 v[96:99] /*v[352:355]*/, v89 /*v601*/ offset:27712
	ds_load_tr16_b128 v[64:67] /*v[320:323]*/, v89 /*v601*/ offset:27744
	ds_load_tr16_b128 v[100:103] /*v[356:359]*/, v89 /*v601*/ offset:32320
	ds_load_tr16_b128 v[68:71] /*v[324:327]*/, v89 /*v601*/ offset:32352
	ds_load_tr16_b128 v[56:59] /*v[312:315]*/, v89 /*v601*/ offset:128
	ds_load_tr16_b128 v[24:27] /*v[280:283]*/, v89 /*v601*/ offset:160
	ds_load_tr16_b128 v[60:63] /*v[316:319]*/, v89 /*v601*/ offset:4736
	ds_load_tr16_b128 v[28:31] /*v[284:287]*/, v89 /*v601*/ offset:4768
	ds_load_tr16_b128 v[48:51] /*v[304:307]*/, v89 /*v601*/ offset:9344
	ds_load_tr16_b128 v[16:19] /*v[272:275]*/, v89 /*v601*/ offset:9376
	ds_load_tr16_b128 v[52:55] /*v[308:311]*/, v89 /*v601*/ offset:13952
	ds_load_tr16_b128 v[20:23] /*v[276:279]*/, v89 /*v601*/ offset:13984
	ds_load_tr16_b128 v[40:43] /*v[296:299]*/, v89 /*v601*/ offset:18560
	ds_load_tr16_b128 v[8:11] /*v[264:267]*/, v89 /*v601*/ offset:18592
	ds_load_tr16_b128 v[44:47] /*v[300:303]*/, v89 /*v601*/ offset:23168
	ds_load_tr16_b128 v[12:15] /*v[268:271]*/, v89 /*v601*/ offset:23200
	ds_load_tr16_b128 v[32:35] /*v[288:291]*/, v89 /*v601*/ offset:27776
	ds_load_tr16_b128 v[0:3] /*v[256:259]*/, v89 /*v601*/ offset:27808
	ds_load_tr16_b128 v[36:39] /*v[292:295]*/, v89 /*v601*/ offset:32384
	ds_load_tr16_b128 v[4:7] /*v[260:263]*/, v89 /*v601*/ offset:32416
	s_set_vgpr_msb 0x4202
	ds_load_tr16_b128 v[248:251], v89 /*v601*/ offset:192
	ds_load_tr16_b128 v[216:219], v89 /*v601*/ offset:224
	ds_load_tr16_b128 v[252:255], v89 /*v601*/ offset:4800
	ds_load_tr16_b128 v[220:223], v89 /*v601*/ offset:4832
	ds_load_tr16_b128 v[240:243], v89 /*v601*/ offset:9408
	ds_load_tr16_b128 v[208:211], v89 /*v601*/ offset:9440
	ds_load_tr16_b128 v[244:247], v89 /*v601*/ offset:14016
	ds_load_tr16_b128 v[212:215], v89 /*v601*/ offset:14048
	ds_load_tr16_b128 v[232:235], v89 /*v601*/ offset:18624
	ds_load_tr16_b128 v[200:203], v89 /*v601*/ offset:18656
	ds_load_tr16_b128 v[236:239], v89 /*v601*/ offset:23232
	ds_load_tr16_b128 v[204:207], v89 /*v601*/ offset:23264
	ds_load_tr16_b128 v[224:227], v89 /*v601*/ offset:27840
	ds_load_tr16_b128 v[192:195], v89 /*v601*/ offset:27872
	ds_load_tr16_b128 v[228:231], v89 /*v601*/ offset:32448
	ds_load_tr16_b128 v[196:199], v89 /*v601*/ offset:32480
	s_set_vgpr_msb 0x2aa
	v_lshl_or_b32 v67 /*v579*/, s92, 7, v79 /*v591*/
	v_cmp_gt_i32_e32 vcc_lo, v67 /*v579*/, v86 /*v598*/
	v_cmp_lt_i32_e64 s2, v67 /*v579*/, v90 /*v602*/
	v_dual_add_nc_u32 v116 /*v628*/, 16, v67 /*v579*/ :: v_dual_bitop2_b32 v70 /*v582*/, 1, v67 /*v579*/ bitop3:0x54
	v_dual_add_nc_u32 v117 /*v629*/, 17, v67 /*v579*/ :: v_dual_bitop2_b32 v92 /*v604*/, 2, v67 /*v579*/ bitop3:0x54
	v_cmp_ge_i32_e64 s3, v67 /*v579*/, v86 /*v598*/
	v_cmp_lt_i32_e64 s4, v70 /*v582*/, v90 /*v602*/
	s_or_b32 s2, s2, vcc_lo
	v_dual_add_nc_u32 v118 /*v630*/, 18, v67 /*v579*/ :: v_dual_bitop2_b32 v93 /*v605*/, 3, v67 /*v579*/ bitop3:0x54
	s_set_vgpr_msb 0xaa41
	v_cndmask_b32_e64 v250 /*v506*/, v250 /*v506*/, 0xff800000, s2
	s_set_vgpr_msb 0x418a
	v_cmp_gt_i32_e32 vcc_lo, v92 /*v604*/, v86 /*v598*/
	v_cmp_lt_i32_e64 s2, v92 /*v604*/, v90 /*v602*/
	v_dual_add_nc_u32 v119 /*v631*/, 19, v67 /*v579*/ :: v_dual_bitop2_b32 v112 /*v624*/, 4, v67 /*v579*/ bitop3:0x54
	s_or_b32 s3, s4, s3
	v_cmp_gt_i32_e64 s5, v93 /*v605*/, v86 /*v598*/
	s_set_vgpr_msb 0x8a41
	v_cndmask_b32_e64 v251 /*v507*/, v251 /*v507*/, 0xff800000, s3
	s_set_vgpr_msb 0x418a
	v_cmp_lt_i32_e64 s3, v93 /*v605*/, v90 /*v602*/
	s_or_b32 s2, s2, vcc_lo
	v_dual_add_nc_u32 v120 /*v632*/, 20, v67 /*v579*/ :: v_dual_bitop2_b32 v113 /*v625*/, 5, v67 /*v579*/ bitop3:0x54
	s_set_vgpr_msb 0x8a41
	v_cndmask_b32_e64 v252 /*v508*/, v252 /*v508*/, 0xff800000, s2
	s_set_vgpr_msb 0x418a
	v_cmp_gt_i32_e32 vcc_lo, v112 /*v624*/, v86 /*v598*/
	v_cmp_lt_i32_e64 s2, v112 /*v624*/, v90 /*v602*/
	v_dual_add_nc_u32 v121 /*v633*/, 21, v67 /*v579*/ :: v_dual_bitop2_b32 v114 /*v626*/, 6, v67 /*v579*/ bitop3:0x54
	s_or_b32 s3, s3, s5
	v_cmp_lt_i32_e64 s4, v113 /*v625*/, v90 /*v602*/
	s_set_vgpr_msb 0x8a41
	v_cndmask_b32_e64 v253 /*v509*/, v253 /*v509*/, 0xff800000, s3
	s_set_vgpr_msb 0x418a
	v_cmp_gt_i32_e64 s3, v113 /*v625*/, v86 /*v598*/
	s_or_b32 s2, s2, vcc_lo
	v_dual_add_nc_u32 v122 /*v634*/, 22, v67 /*v579*/ :: v_dual_bitop2_b32 v115 /*v627*/, 7, v67 /*v579*/ bitop3:0x54
	s_set_vgpr_msb 0x8a41
	v_cndmask_b32_e64 v254 /*v510*/, v254 /*v510*/, 0xff800000, s2
	s_set_vgpr_msb 0x410a
	v_cmp_gt_i32_e32 vcc_lo, v114 /*v626*/, v86 /*v598*/
	v_cmp_lt_i32_e64 s2, v114 /*v626*/, v90 /*v602*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s4, v115 /*v627*/, v90 /*v602*/
	s_set_vgpr_msb 0xa41
	v_cndmask_b32_e64 v255 /*v511*/, v255 /*v511*/, 0xff800000, s3
	s_set_vgpr_msb 0x418a
	v_cmp_gt_i32_e64 s3, v115 /*v627*/, v86 /*v598*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v116 /*v628*/, v86 /*v598*/
	v_cndmask_b32_e64 v0 /*v512*/, v0 /*v512*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v116 /*v628*/, v90 /*v602*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s4, v117 /*v629*/, v90 /*v602*/
	v_cndmask_b32_e64 v1 /*v513*/, v1 /*v513*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v117 /*v629*/, v86 /*v598*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v118 /*v630*/, v86 /*v598*/
	v_cndmask_b32_e64 v2 /*v514*/, v2 /*v514*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v118 /*v630*/, v90 /*v602*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s4, v119 /*v631*/, v90 /*v602*/
	v_cndmask_b32_e64 v3 /*v515*/, v3 /*v515*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v119 /*v631*/, v86 /*v598*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v120 /*v632*/, v86 /*v598*/
	v_cndmask_b32_e64 v4 /*v516*/, v4 /*v516*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v120 /*v632*/, v90 /*v602*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s4, v121 /*v633*/, v90 /*v602*/
	v_cndmask_b32_e64 v5 /*v517*/, v5 /*v517*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v121 /*v633*/, v86 /*v598*/
	s_or_b32 s2, s2, vcc_lo
	v_dual_add_nc_u32 v123 /*v635*/, 23, v67 /*v579*/ :: v_dual_bitop2_b32 v124 /*v636*/, 32, v67 /*v579*/ bitop3:0x54
	v_cndmask_b32_e64 v6 /*v518*/, v6 /*v518*/, 0xff800000, s2
	v_cmp_gt_i32_e32 vcc_lo, v122 /*v634*/, v86 /*v598*/
	v_cmp_lt_i32_e64 s2, v122 /*v634*/, v90 /*v602*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s4, v123 /*v635*/, v90 /*v602*/
	v_cndmask_b32_e64 v7 /*v519*/, v7 /*v519*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v123 /*v635*/, v86 /*v598*/
	s_or_b32 s2, s2, vcc_lo
	v_or_b32_e32 v125 /*v637*/, 33, v67 /*v579*/
	v_cndmask_b32_e64 v8 /*v520*/, v8 /*v520*/, 0xff800000, s2
	v_cmp_gt_i32_e32 vcc_lo, v124 /*v636*/, v86 /*v598*/
	v_cmp_lt_i32_e64 s2, v124 /*v636*/, v90 /*v602*/
	v_dual_add_nc_u32 v133 /*v645*/, 49, v67 /*v579*/ :: v_dual_bitop2_b32 v126 /*v638*/, 34, v67 /*v579*/ bitop3:0x54
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s4, v125 /*v637*/, v90 /*v602*/
	v_cndmask_b32_e64 v9 /*v521*/, v9 /*v521*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v125 /*v637*/, v86 /*v598*/
	s_or_b32 s2, s2, vcc_lo
	v_dual_add_nc_u32 v134 /*v646*/, 50, v67 /*v579*/ :: v_dual_bitop2_b32 v127 /*v639*/, 35, v67 /*v579*/ bitop3:0x54
	v_cndmask_b32_e64 v10 /*v522*/, v10 /*v522*/, 0xff800000, s2
	v_cmp_gt_i32_e32 vcc_lo, v126 /*v638*/, v86 /*v598*/
	v_cmp_lt_i32_e64 s2, v126 /*v638*/, v90 /*v602*/
	v_dual_add_nc_u32 v135 /*v647*/, 51, v67 /*v579*/ :: v_dual_bitop2_b32 v128 /*v640*/, 36, v67 /*v579*/ bitop3:0x54
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s4, v127 /*v639*/, v90 /*v602*/
	v_cndmask_b32_e64 v11 /*v523*/, v11 /*v523*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v127 /*v639*/, v86 /*v598*/
	s_or_b32 s2, s2, vcc_lo
	v_dual_add_nc_u32 v136 /*v648*/, 52, v67 /*v579*/ :: v_dual_bitop2_b32 v129 /*v641*/, 37, v67 /*v579*/ bitop3:0x54
	v_cndmask_b32_e64 v12 /*v524*/, v12 /*v524*/, 0xff800000, s2
	v_cmp_gt_i32_e32 vcc_lo, v128 /*v640*/, v86 /*v598*/
	v_cmp_lt_i32_e64 s2, v128 /*v640*/, v90 /*v602*/
	s_or_b32 s3, s4, s3
	v_dual_add_nc_u32 v137 /*v649*/, 53, v67 /*v579*/ :: v_dual_bitop2_b32 v130 /*v642*/, 38, v67 /*v579*/ bitop3:0x54
	v_cndmask_b32_e64 v13 /*v525*/, v13 /*v525*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v129 /*v641*/, v86 /*v598*/
	v_cmp_lt_i32_e64 s4, v129 /*v641*/, v90 /*v602*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v130 /*v642*/, v86 /*v598*/
	v_cndmask_b32_e64 v68 /*v580*/, v14 /*v526*/, 0xff800000, s2
	v_dual_add_nc_u32 v138 /*v650*/, 54, v67 /*v579*/ :: v_dual_bitop2_b32 v14 /*v526*/, 39, v67 /*v579*/ bitop3:0x54
	v_cmp_lt_i32_e64 s2, v130 /*v642*/, v90 /*v602*/
	s_or_b32 s3, s4, s3
	v_dual_add_nc_u32 v139 /*v651*/, 55, v67 /*v579*/ :: v_dual_bitop2_b32 v140 /*v652*/, 64, v67 /*v579*/ bitop3:0x54
	v_cndmask_b32_e64 v69 /*v581*/, v15 /*v527*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v14 /*v526*/, v86 /*v598*/
	v_add_nc_u32_e32 v15 /*v527*/, 48, v67 /*v579*/
	v_cmp_lt_i32_e64 s4, v14 /*v526*/, v90 /*v602*/
	s_or_b32 s2, s2, vcc_lo
	v_or_b32_e32 v141 /*v653*/, 0x41, v67 /*v579*/
	v_cndmask_b32_e64 v16 /*v528*/, v16 /*v528*/, 0xff800000, s2
	v_cmp_gt_i32_e32 vcc_lo, v15 /*v527*/, v86 /*v598*/
	v_cmp_lt_i32_e64 s2, v15 /*v527*/, v90 /*v602*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s4, v133 /*v645*/, v90 /*v602*/
	v_cndmask_b32_e64 v17 /*v529*/, v17 /*v529*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v133 /*v645*/, v86 /*v598*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v134 /*v646*/, v86 /*v598*/
	v_cndmask_b32_e64 v18 /*v530*/, v18 /*v530*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v134 /*v646*/, v90 /*v602*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s4, v135 /*v647*/, v90 /*v602*/
	v_cndmask_b32_e64 v19 /*v531*/, v19 /*v531*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v135 /*v647*/, v86 /*v598*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v136 /*v648*/, v86 /*v598*/
	v_cndmask_b32_e64 v20 /*v532*/, v20 /*v532*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v136 /*v648*/, v90 /*v602*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s4, v137 /*v649*/, v90 /*v602*/
	v_cndmask_b32_e64 v21 /*v533*/, v21 /*v533*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v137 /*v649*/, v86 /*v598*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v138 /*v650*/, v86 /*v598*/
	v_cndmask_b32_e64 v22 /*v534*/, v22 /*v534*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v138 /*v650*/, v90 /*v602*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s4, v139 /*v651*/, v90 /*v602*/
	v_cndmask_b32_e64 v23 /*v535*/, v23 /*v535*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v139 /*v651*/, v86 /*v598*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v140 /*v652*/, v86 /*v598*/
	v_cndmask_b32_e64 v24 /*v536*/, v24 /*v536*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v140 /*v652*/, v90 /*v602*/
	s_or_b32 s3, s4, s3
	v_or_b32_e32 v142 /*v654*/, 0x42, v67 /*v579*/
	v_cndmask_b32_e64 v25 /*v537*/, v25 /*v537*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v141 /*v653*/, v86 /*v598*/
	v_cmp_lt_i32_e64 s4, v141 /*v653*/, v90 /*v602*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v142 /*v654*/, v86 /*v598*/
	v_cndmask_b32_e64 v94 /*v606*/, v26 /*v538*/, 0xff800000, s2
	v_or_b32_e32 v26 /*v538*/, 0x43, v67 /*v579*/
	v_cmp_lt_i32_e64 s2, v142 /*v654*/, v90 /*v602*/
	s_or_b32 s3, s4, s3
	v_or_b32_e32 v145 /*v657*/, 0x45, v67 /*v579*/
	v_cndmask_b32_e64 v95 /*v607*/, v27 /*v539*/, 0xff800000, s3
	v_or_b32_e32 v27 /*v539*/, 0x44, v67 /*v579*/
	v_cmp_gt_i32_e64 s3, v26 /*v538*/, v86 /*v598*/
	v_cmp_lt_i32_e64 s4, v26 /*v538*/, v90 /*v602*/
	s_or_b32 s2, s2, vcc_lo
	v_or_b32_e32 v146 /*v658*/, 0x46, v67 /*v579*/
	v_cndmask_b32_e64 v28 /*v540*/, v28 /*v540*/, 0xff800000, s2
	v_cmp_gt_i32_e32 vcc_lo, v27 /*v539*/, v86 /*v598*/
	v_cmp_lt_i32_e64 s2, v27 /*v539*/, v90 /*v602*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s4, v145 /*v657*/, v90 /*v602*/
	v_cndmask_b32_e64 v29 /*v541*/, v29 /*v541*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v145 /*v657*/, v86 /*v598*/
	s_or_b32 s2, s2, vcc_lo
	v_or_b32_e32 v147 /*v659*/, 0x47, v67 /*v579*/
	v_cndmask_b32_e64 v30 /*v542*/, v30 /*v542*/, 0xff800000, s2
	v_cmp_gt_i32_e32 vcc_lo, v146 /*v658*/, v86 /*v598*/
	v_cmp_lt_i32_e64 s2, v146 /*v658*/, v90 /*v602*/
	s_or_b32 s3, s4, s3
	v_add_nc_u32_e32 v148 /*v660*/, 0x50, v67 /*v579*/
	v_cndmask_b32_e64 v31 /*v543*/, v31 /*v543*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v147 /*v659*/, v86 /*v598*/
	v_cmp_lt_i32_e64 s4, v147 /*v659*/, v90 /*v602*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v148 /*v660*/, v86 /*v598*/
	v_cndmask_b32_e64 v96 /*v608*/, v32 /*v544*/, 0xff800000, s2
	v_add_nc_u32_e32 v32 /*v544*/, 0x51, v67 /*v579*/
	v_cmp_lt_i32_e64 s2, v148 /*v660*/, v90 /*v602*/
	s_or_b32 s3, s4, s3
	v_add_nc_u32_e32 v151 /*v663*/, 0x53, v67 /*v579*/
	v_cndmask_b32_e64 v97 /*v609*/, v33 /*v545*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v32 /*v544*/, v86 /*v598*/
	v_add_nc_u32_e32 v33 /*v545*/, 0x52, v67 /*v579*/
	v_cmp_lt_i32_e64 s4, v32 /*v544*/, v90 /*v602*/
	s_or_b32 s2, s2, vcc_lo
	v_add_nc_u32_e32 v152 /*v664*/, 0x54, v67 /*v579*/
	v_cndmask_b32_e64 v34 /*v546*/, v34 /*v546*/, 0xff800000, s2
	v_cmp_gt_i32_e32 vcc_lo, v33 /*v545*/, v86 /*v598*/
	v_cmp_lt_i32_e64 s2, v33 /*v545*/, v90 /*v602*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s4, v151 /*v663*/, v90 /*v602*/
	v_cndmask_b32_e64 v35 /*v547*/, v35 /*v547*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v151 /*v663*/, v86 /*v598*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v152 /*v664*/, v86 /*v598*/
	v_cndmask_b32_e64 v98 /*v610*/, v36 /*v548*/, 0xff800000, s2
	v_add_nc_u32_e32 v36 /*v548*/, 0x55, v67 /*v579*/
	v_cmp_lt_i32_e64 s2, v152 /*v664*/, v90 /*v602*/
	s_or_b32 s3, s4, s3
	v_add_nc_u32_e32 v154 /*v666*/, 0x57, v67 /*v579*/
	v_cndmask_b32_e64 v99 /*v611*/, v37 /*v549*/, 0xff800000, s3
	v_add_nc_u32_e32 v37 /*v549*/, 0x56, v67 /*v579*/
	v_cmp_gt_i32_e64 s3, v36 /*v548*/, v86 /*v598*/
	v_cmp_lt_i32_e64 s4, v36 /*v548*/, v90 /*v602*/
	s_or_b32 s2, s2, vcc_lo
	v_or_b32_e32 v155 /*v667*/, 0x60, v67 /*v579*/
	v_cndmask_b32_e64 v38 /*v550*/, v38 /*v550*/, 0xff800000, s2
	v_cmp_gt_i32_e32 vcc_lo, v37 /*v549*/, v86 /*v598*/
	v_cmp_lt_i32_e64 s2, v37 /*v549*/, v90 /*v602*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s4, v154 /*v666*/, v90 /*v602*/
	v_cndmask_b32_e64 v39 /*v551*/, v39 /*v551*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v154 /*v666*/, v86 /*v598*/
	s_or_b32 s2, s2, vcc_lo
	v_or_b32_e32 v157 /*v669*/, 0x61, v67 /*v579*/
	v_cndmask_b32_e64 v40 /*v552*/, v40 /*v552*/, 0xff800000, s2
	v_cmp_gt_i32_e32 vcc_lo, v155 /*v667*/, v86 /*v598*/
	v_cmp_lt_i32_e64 s2, v155 /*v667*/, v90 /*v602*/
	s_or_b32 s3, s4, s3
	v_or_b32_e32 v158 /*v670*/, 0x62, v67 /*v579*/
	v_cndmask_b32_e64 v41 /*v553*/, v41 /*v553*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v157 /*v669*/, v86 /*v598*/
	v_cmp_lt_i32_e64 s4, v157 /*v669*/, v90 /*v602*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v158 /*v670*/, v86 /*v598*/
	v_cndmask_b32_e64 v100 /*v612*/, v42 /*v554*/, 0xff800000, s2
	v_or_b32_e32 v42 /*v554*/, 0x63, v67 /*v579*/
	v_cmp_lt_i32_e64 s2, v158 /*v670*/, v90 /*v602*/
	s_or_b32 s3, s4, s3
	v_or_b32_e32 v163 /*v675*/, 0x67, v67 /*v579*/
	v_cndmask_b32_e64 v101 /*v613*/, v43 /*v555*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v42 /*v554*/, v86 /*v598*/
	v_or_b32_e32 v43 /*v555*/, 0x64, v67 /*v579*/
	v_cmp_lt_i32_e64 s4, v42 /*v554*/, v90 /*v602*/
	s_or_b32 s2, s2, vcc_lo
	v_add_nc_u32_e32 v164 /*v676*/, 0x70, v67 /*v579*/
	v_cndmask_b32_e64 v102 /*v614*/, v44 /*v556*/, 0xff800000, s2
	v_or_b32_e32 v44 /*v556*/, 0x65, v67 /*v579*/
	v_cmp_gt_i32_e32 vcc_lo, v43 /*v555*/, v86 /*v598*/
	v_cmp_lt_i32_e64 s2, v43 /*v555*/, v90 /*v602*/
	s_or_b32 s3, s4, s3
	v_add_nc_u32_e32 v165 /*v677*/, 0x71, v67 /*v579*/
	v_cndmask_b32_e64 v103 /*v615*/, v45 /*v557*/, 0xff800000, s3
	v_or_b32_e32 v45 /*v557*/, 0x66, v67 /*v579*/
	v_cmp_gt_i32_e64 s3, v44 /*v556*/, v86 /*v598*/
	v_cmp_lt_i32_e64 s4, v44 /*v556*/, v90 /*v602*/
	s_or_b32 s2, s2, vcc_lo
	v_add_nc_u32_e32 v166 /*v678*/, 0x72, v67 /*v579*/
	v_cndmask_b32_e64 v46 /*v558*/, v46 /*v558*/, 0xff800000, s2
	v_cmp_gt_i32_e32 vcc_lo, v45 /*v557*/, v86 /*v598*/
	v_cmp_lt_i32_e64 s2, v45 /*v557*/, v90 /*v602*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s4, v163 /*v675*/, v90 /*v602*/
	v_cndmask_b32_e64 v47 /*v559*/, v47 /*v559*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v163 /*v675*/, v86 /*v598*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v164 /*v676*/, v86 /*v598*/
	v_cndmask_b32_e64 v48 /*v560*/, v48 /*v560*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v164 /*v676*/, v90 /*v602*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s4, v165 /*v677*/, v90 /*v602*/
	v_cndmask_b32_e64 v49 /*v561*/, v49 /*v561*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v165 /*v677*/, v86 /*v598*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v166 /*v678*/, v86 /*v598*/
	v_cndmask_b32_e64 v104 /*v616*/, v50 /*v562*/, 0xff800000, s2
	v_add_nc_u32_e32 v50 /*v562*/, 0x73, v67 /*v579*/
	v_cmp_lt_i32_e64 s2, v166 /*v678*/, v90 /*v602*/
	s_or_b32 s3, s4, s3
	v_add_nc_u32_e32 v169 /*v681*/, 0x77, v67 /*v579*/
	v_cndmask_b32_e64 v105 /*v617*/, v51 /*v563*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v50 /*v562*/, v86 /*v598*/
	v_add_nc_u32_e32 v51 /*v563*/, 0x74, v67 /*v579*/
	v_cmp_lt_i32_e64 s4, v50 /*v562*/, v90 /*v602*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e64 s5, v67 /*v579*/, v87 /*v599*/
	v_cndmask_b32_e64 v106 /*v618*/, v52 /*v564*/, 0xff800000, s2
	v_add_nc_u32_e32 v52 /*v564*/, 0x75, v67 /*v579*/
	v_cmp_gt_i32_e32 vcc_lo, v51 /*v563*/, v86 /*v598*/
	v_cmp_lt_i32_e64 s2, v51 /*v563*/, v90 /*v602*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s6, v67 /*v579*/, v91 /*v603*/
	v_cndmask_b32_e64 v107 /*v619*/, v53 /*v565*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v52 /*v564*/, v86 /*v598*/
	v_cmp_lt_i32_e64 s4, v52 /*v564*/, v90 /*v602*/
	v_add_nc_u32_e32 v53 /*v565*/, 0x76, v67 /*v579*/
	s_or_b32 s2, s2, vcc_lo
	v_cndmask_b32_e64 v54 /*v566*/, v54 /*v566*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	v_cmp_gt_i32_e32 vcc_lo, v53 /*v565*/, v86 /*v598*/
	v_cndmask_b32_e64 v55 /*v567*/, v55 /*v567*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v53 /*v565*/, v90 /*v602*/
	v_cmp_gt_i32_e64 s3, v169 /*v681*/, v86 /*v598*/
	v_cmp_lt_i32_e64 s4, v169 /*v681*/, v90 /*v602*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_ge_i32_e32 vcc_lo, v67 /*v579*/, v87 /*v599*/
	v_cndmask_b32_e64 v56 /*v568*/, v56 /*v568*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	v_cmp_gt_i32_e64 s3, v92 /*v604*/, v87 /*v599*/
	v_cndmask_b32_e64 v57 /*v569*/, v57 /*v569*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	v_cmp_lt_i32_e64 s4, v92 /*v604*/, v91 /*v603*/
	s_set_vgpr_msb 0x8a81
	v_cndmask_b32_e64 v108 /*v620*/, v192 /*v448*/, 0xff800000, s2
	s_set_vgpr_msb 0x810a
	v_cmp_lt_i32_e64 s2, v70 /*v582*/, v91 /*v603*/
	v_cmp_gt_i32_e64 s5, v93 /*v605*/, v87 /*v599*/
	v_cmp_lt_i32_e64 s6, v93 /*v605*/, v91 /*v603*/
	s_set_vgpr_msb 0xa55
	v_max3_num_f32 v192 /*v448*/, v250 /*v506*/, v251 /*v507*/, v252 /*v508*/
	s_or_b32 s2, s2, vcc_lo
	s_set_vgpr_msb 0x550a
	v_cmp_gt_i32_e32 vcc_lo, v112 /*v624*/, v87 /*v599*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v109 /*v621*/, v193 /*v449*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	s_set_vgpr_msb 0x810a
	v_cmp_gt_i32_e64 s3, v113 /*v625*/, v87 /*v599*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v110 /*v622*/, v194 /*v450*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	s_set_vgpr_msb 0x810a
	v_cmp_lt_i32_e64 s4, v113 /*v625*/, v91 /*v603*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v111 /*v623*/, v195 /*v451*/, 0xff800000, s2
	s_set_vgpr_msb 0x816a
	v_cmp_lt_i32_e64 s2, v112 /*v624*/, v91 /*v603*/
	v_cmp_gt_i32_e64 s5, v114 /*v626*/, v87 /*v599*/
	v_cmp_lt_i32_e64 s6, v114 /*v626*/, v91 /*v603*/
	v_max3_num_f32 v193 /*v449*/, v108 /*v620*/, v109 /*v621*/, v110 /*v622*/
	s_set_vgpr_msb 0x6a55
	v_max3_num_f32 v194 /*v450*/, v253 /*v509*/, v254 /*v510*/, v255 /*v511*/
	s_or_b32 s2, s2, vcc_lo
	s_set_vgpr_msb 0x550a
	v_cmp_gt_i32_e32 vcc_lo, v115 /*v627*/, v87 /*v599*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v112 /*v624*/, v196 /*v452*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	s_set_vgpr_msb 0x810a
	v_cmp_gt_i32_e64 s3, v116 /*v628*/, v87 /*v599*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v113 /*v625*/, v197 /*v453*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	s_set_vgpr_msb 0x810a
	v_cmp_lt_i32_e64 s4, v116 /*v628*/, v91 /*v603*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v114 /*v626*/, v198 /*v454*/, 0xff800000, s2
	s_set_vgpr_msb 0x816a
	v_cmp_lt_i32_e64 s2, v115 /*v627*/, v91 /*v603*/
	v_cmp_gt_i32_e64 s5, v117 /*v629*/, v87 /*v599*/
	v_cmp_lt_i32_e64 s6, v117 /*v629*/, v91 /*v603*/
	v_max3_num_f32 v195 /*v451*/, v111 /*v623*/, v112 /*v624*/, v113 /*v625*/
	v_max3_num_f32 v196 /*v452*/, v0 /*v512*/, v1 /*v513*/, v2 /*v514*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v118 /*v630*/, v87 /*v599*/
	s_set_vgpr_msb 0x6a81
	v_cndmask_b32_e64 v115 /*v627*/, v199 /*v455*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	s_set_vgpr_msb 0x810a
	v_cmp_gt_i32_e64 s3, v119 /*v631*/, v87 /*v599*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v116 /*v628*/, v200 /*v456*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	s_set_vgpr_msb 0x810a
	v_cmp_lt_i32_e64 s4, v119 /*v631*/, v91 /*v603*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v117 /*v629*/, v201 /*v457*/, 0xff800000, s2
	s_set_vgpr_msb 0x816a
	v_cmp_lt_i32_e64 s2, v118 /*v630*/, v91 /*v603*/
	v_cmp_gt_i32_e64 s5, v120 /*v632*/, v87 /*v599*/
	v_cmp_lt_i32_e64 s6, v120 /*v632*/, v91 /*v603*/
	v_max3_num_f32 v197 /*v453*/, v114 /*v626*/, v115 /*v627*/, v116 /*v628*/
	v_max3_num_f32 v198 /*v454*/, v3 /*v515*/, v4 /*v516*/, v5 /*v517*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v121 /*v633*/, v87 /*v599*/
	s_set_vgpr_msb 0x6a81
	v_cndmask_b32_e64 v118 /*v630*/, v202 /*v458*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	s_set_vgpr_msb 0x810a
	v_cmp_gt_i32_e64 s3, v122 /*v634*/, v87 /*v599*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v119 /*v631*/, v203 /*v459*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	s_set_vgpr_msb 0x810a
	v_cmp_lt_i32_e64 s4, v122 /*v634*/, v91 /*v603*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v120 /*v632*/, v204 /*v460*/, 0xff800000, s2
	s_set_vgpr_msb 0x816a
	v_cmp_lt_i32_e64 s2, v121 /*v633*/, v91 /*v603*/
	v_cmp_gt_i32_e64 s5, v123 /*v635*/, v87 /*v599*/
	v_cmp_lt_i32_e64 s6, v123 /*v635*/, v91 /*v603*/
	v_max3_num_f32 v200 /*v456*/, v6 /*v518*/, v7 /*v519*/, v8 /*v520*/
	v_max3_num_f32 v202 /*v458*/, v9 /*v521*/, v10 /*v522*/, v11 /*v523*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v124 /*v636*/, v87 /*v599*/
	s_set_vgpr_msb 0x6a81
	v_cndmask_b32_e64 v121 /*v633*/, v205 /*v461*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	s_set_vgpr_msb 0x810a
	v_cmp_gt_i32_e64 s3, v125 /*v637*/, v87 /*v599*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v122 /*v634*/, v206 /*v462*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	s_set_vgpr_msb 0x810a
	v_cmp_lt_i32_e64 s4, v125 /*v637*/, v91 /*v603*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v123 /*v635*/, v207 /*v463*/, 0xff800000, s2
	s_set_vgpr_msb 0x816a
	v_cmp_lt_i32_e64 s2, v124 /*v636*/, v91 /*v603*/
	v_cmp_gt_i32_e64 s5, v126 /*v638*/, v87 /*v599*/
	v_cmp_lt_i32_e64 s6, v126 /*v638*/, v91 /*v603*/
	v_max3_num_f32 v204 /*v460*/, v12 /*v524*/, v13 /*v525*/, v68 /*v580*/
	v_max3_num_f32 v206 /*v462*/, v69 /*v581*/, v16 /*v528*/, v17 /*v529*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v127 /*v639*/, v87 /*v599*/
	s_set_vgpr_msb 0x6a81
	v_cndmask_b32_e64 v124 /*v636*/, v208 /*v464*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	s_set_vgpr_msb 0x810a
	v_cmp_gt_i32_e64 s3, v128 /*v640*/, v87 /*v599*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v125 /*v637*/, v209 /*v465*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	s_set_vgpr_msb 0x810a
	v_cmp_lt_i32_e64 s4, v128 /*v640*/, v91 /*v603*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v126 /*v638*/, v210 /*v466*/, 0xff800000, s2
	s_set_vgpr_msb 0x816a
	v_cmp_lt_i32_e64 s2, v127 /*v639*/, v91 /*v603*/
	v_cmp_gt_i32_e64 s5, v129 /*v641*/, v87 /*v599*/
	v_cmp_lt_i32_e64 s6, v129 /*v641*/, v91 /*v603*/
	v_max3_num_f32 v208 /*v464*/, v18 /*v530*/, v19 /*v531*/, v20 /*v532*/
	v_max3_num_f32 v210 /*v466*/, v21 /*v533*/, v22 /*v534*/, v23 /*v535*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v130 /*v642*/, v87 /*v599*/
	s_set_vgpr_msb 0x6a81
	v_cndmask_b32_e64 v127 /*v639*/, v211 /*v467*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	s_set_vgpr_msb 0x810a
	v_cmp_gt_i32_e64 s3, v14 /*v526*/, v87 /*v599*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v128 /*v640*/, v212 /*v468*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	s_set_vgpr_msb 0x810a
	v_cmp_lt_i32_e64 s4, v14 /*v526*/, v91 /*v603*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v129 /*v641*/, v213 /*v469*/, 0xff800000, s2
	s_set_vgpr_msb 0x816a
	v_cmp_lt_i32_e64 s2, v130 /*v642*/, v91 /*v603*/
	v_cmp_gt_i32_e64 s5, v15 /*v527*/, v87 /*v599*/
	v_cmp_lt_i32_e64 s6, v15 /*v527*/, v91 /*v603*/
	v_max3_num_f32 v212 /*v468*/, v24 /*v536*/, v25 /*v537*/, v94 /*v606*/
	v_max3_num_f32 v199 /*v455*/, v117 /*v629*/, v118 /*v630*/, v119 /*v631*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v133 /*v645*/, v87 /*v599*/
	s_set_vgpr_msb 0x6a81
	v_cndmask_b32_e64 v130 /*v642*/, v214 /*v470*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	s_set_vgpr_msb 0x810a
	v_cmp_gt_i32_e64 s3, v134 /*v646*/, v87 /*v599*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v131 /*v643*/, v215 /*v471*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	s_set_vgpr_msb 0x810a
	v_cmp_lt_i32_e64 s4, v134 /*v646*/, v91 /*v603*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v132 /*v644*/, v216 /*v472*/, 0xff800000, s2
	s_set_vgpr_msb 0x816a
	v_cmp_lt_i32_e64 s2, v133 /*v645*/, v91 /*v603*/
	v_cmp_gt_i32_e64 s5, v135 /*v647*/, v87 /*v599*/
	v_cmp_lt_i32_e64 s6, v135 /*v647*/, v91 /*v603*/
	v_max3_num_f32 v214 /*v470*/, v95 /*v607*/, v28 /*v540*/, v29 /*v541*/
	v_max3_num_f32 v216 /*v472*/, v30 /*v542*/, v31 /*v543*/, v96 /*v608*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v136 /*v648*/, v87 /*v599*/
	s_set_vgpr_msb 0x6a81
	v_cndmask_b32_e64 v133 /*v645*/, v217 /*v473*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	s_set_vgpr_msb 0x810a
	v_cmp_gt_i32_e64 s3, v137 /*v649*/, v87 /*v599*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v134 /*v646*/, v218 /*v474*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	s_set_vgpr_msb 0x810a
	v_cmp_lt_i32_e64 s4, v137 /*v649*/, v91 /*v603*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v135 /*v647*/, v219 /*v475*/, 0xff800000, s2
	s_set_vgpr_msb 0x816a
	v_cmp_lt_i32_e64 s2, v136 /*v648*/, v91 /*v603*/
	v_cmp_gt_i32_e64 s5, v138 /*v650*/, v87 /*v599*/
	v_cmp_lt_i32_e64 s6, v138 /*v650*/, v91 /*v603*/
	v_max3_num_f32 v218 /*v474*/, v97 /*v609*/, v34 /*v546*/, v35 /*v547*/
	v_max3_num_f32 v201 /*v457*/, v120 /*v632*/, v121 /*v633*/, v122 /*v634*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v139 /*v651*/, v87 /*v599*/
	s_set_vgpr_msb 0x6a81
	v_cndmask_b32_e64 v136 /*v648*/, v220 /*v476*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	s_set_vgpr_msb 0x810a
	v_cmp_gt_i32_e64 s3, v140 /*v652*/, v87 /*v599*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v137 /*v649*/, v221 /*v477*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	s_set_vgpr_msb 0x810a
	v_cmp_lt_i32_e64 s4, v140 /*v652*/, v91 /*v603*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v138 /*v650*/, v222 /*v478*/, 0xff800000, s2
	s_set_vgpr_msb 0x816a
	v_cmp_lt_i32_e64 s2, v139 /*v651*/, v91 /*v603*/
	v_cmp_gt_i32_e64 s5, v141 /*v653*/, v87 /*v599*/
	v_cmp_lt_i32_e64 s6, v141 /*v653*/, v91 /*v603*/
	v_max3_num_f32 v220 /*v476*/, v98 /*v610*/, v99 /*v611*/, v38 /*v550*/
	v_max3_num_f32 v222 /*v478*/, v39 /*v551*/, v40 /*v552*/, v41 /*v553*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v142 /*v654*/, v87 /*v599*/
	s_set_vgpr_msb 0x6a81
	v_cndmask_b32_e64 v139 /*v651*/, v223 /*v479*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	s_set_vgpr_msb 0x810a
	v_cmp_gt_i32_e64 s3, v26 /*v538*/, v87 /*v599*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v140 /*v652*/, v224 /*v480*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	s_set_vgpr_msb 0x810a
	v_cmp_lt_i32_e64 s4, v26 /*v538*/, v91 /*v603*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v141 /*v653*/, v225 /*v481*/, 0xff800000, s2
	s_set_vgpr_msb 0x816a
	v_cmp_lt_i32_e64 s2, v142 /*v654*/, v91 /*v603*/
	v_cmp_gt_i32_e64 s5, v27 /*v539*/, v87 /*v599*/
	v_cmp_lt_i32_e64 s6, v27 /*v539*/, v91 /*v603*/
	v_max3_num_f32 v224 /*v480*/, v100 /*v612*/, v101 /*v613*/, v102 /*v614*/
	v_max3_num_f32 v203 /*v459*/, v123 /*v635*/, v124 /*v636*/, v125 /*v637*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v145 /*v657*/, v87 /*v599*/
	s_set_vgpr_msb 0x6a81
	v_cndmask_b32_e64 v142 /*v654*/, v226 /*v482*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	s_set_vgpr_msb 0x810a
	v_cmp_gt_i32_e64 s3, v146 /*v658*/, v87 /*v599*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v143 /*v655*/, v227 /*v483*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	s_set_vgpr_msb 0x810a
	v_cmp_lt_i32_e64 s4, v146 /*v658*/, v91 /*v603*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v144 /*v656*/, v228 /*v484*/, 0xff800000, s2
	s_set_vgpr_msb 0x816a
	v_cmp_lt_i32_e64 s2, v145 /*v657*/, v91 /*v603*/
	v_cmp_gt_i32_e64 s5, v147 /*v659*/, v87 /*v599*/
	v_cmp_lt_i32_e64 s6, v147 /*v659*/, v91 /*v603*/
	v_max3_num_f32 v226 /*v482*/, v103 /*v615*/, v46 /*v558*/, v47 /*v559*/
	v_max3_num_f32 v205 /*v461*/, v126 /*v638*/, v127 /*v639*/, v128 /*v640*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v148 /*v660*/, v87 /*v599*/
	s_set_vgpr_msb 0x6a81
	v_cndmask_b32_e64 v145 /*v657*/, v229 /*v485*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	s_set_vgpr_msb 0x810a
	v_cmp_gt_i32_e64 s3, v32 /*v544*/, v87 /*v599*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v146 /*v658*/, v230 /*v486*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	s_set_vgpr_msb 0x810a
	v_cmp_lt_i32_e64 s4, v32 /*v544*/, v91 /*v603*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v147 /*v659*/, v231 /*v487*/, 0xff800000, s2
	s_set_vgpr_msb 0x816a
	v_cmp_lt_i32_e64 s2, v148 /*v660*/, v91 /*v603*/
	v_cmp_gt_i32_e64 s5, v33 /*v545*/, v87 /*v599*/
	v_cmp_lt_i32_e64 s6, v33 /*v545*/, v91 /*v603*/
	v_max3_num_f32 v230 /*v486*/, v105 /*v617*/, v106 /*v618*/, v107 /*v619*/
	v_max3_num_f32 v207 /*v463*/, v129 /*v641*/, v130 /*v642*/, v131 /*v643*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v151 /*v663*/, v87 /*v599*/
	s_set_vgpr_msb 0x6a81
	v_cndmask_b32_e64 v148 /*v660*/, v232 /*v488*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	s_set_vgpr_msb 0x810a
	v_cmp_gt_i32_e64 s3, v152 /*v664*/, v87 /*v599*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v149 /*v661*/, v233 /*v489*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	s_set_vgpr_msb 0x810a
	v_cmp_lt_i32_e64 s4, v152 /*v664*/, v91 /*v603*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v150 /*v662*/, v234 /*v490*/, 0xff800000, s2
	s_set_vgpr_msb 0x816a
	v_cmp_lt_i32_e64 s2, v151 /*v663*/, v91 /*v603*/
	v_cmp_gt_i32_e64 s5, v36 /*v548*/, v87 /*v599*/
	v_cmp_lt_i32_e64 s6, v36 /*v548*/, v91 /*v603*/
	v_max3_num_f32 v209 /*v465*/, v132 /*v644*/, v133 /*v645*/, v134 /*v646*/
	v_max3_num_f32 v211 /*v467*/, v135 /*v647*/, v136 /*v648*/, v137 /*v649*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v37 /*v549*/, v87 /*v599*/
	s_set_vgpr_msb 0x6a81
	v_cndmask_b32_e64 v151 /*v663*/, v235 /*v491*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	s_set_vgpr_msb 0x810a
	v_cmp_gt_i32_e64 s3, v154 /*v666*/, v87 /*v599*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v152 /*v664*/, v236 /*v492*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	s_set_vgpr_msb 0x810a
	v_cmp_lt_i32_e64 s4, v154 /*v666*/, v91 /*v603*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v153 /*v665*/, v237 /*v493*/, 0xff800000, s2
	s_set_vgpr_msb 0x816a
	v_cmp_lt_i32_e64 s2, v37 /*v549*/, v91 /*v603*/
	v_cmp_gt_i32_e64 s5, v155 /*v667*/, v87 /*v599*/
	v_cmp_lt_i32_e64 s6, v155 /*v667*/, v91 /*v603*/
	v_max3_num_f32 v213 /*v469*/, v138 /*v650*/, v139 /*v651*/, v140 /*v652*/
	v_max3_num_f32 v215 /*v471*/, v141 /*v653*/, v142 /*v654*/, v143 /*v655*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v157 /*v669*/, v87 /*v599*/
	s_set_vgpr_msb 0x6a81
	v_cndmask_b32_e64 v154 /*v666*/, v238 /*v494*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	s_set_vgpr_msb 0x810a
	v_cmp_gt_i32_e64 s3, v158 /*v670*/, v87 /*v599*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v155 /*v667*/, v239 /*v495*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	s_set_vgpr_msb 0x810a
	v_cmp_lt_i32_e64 s4, v158 /*v670*/, v91 /*v603*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v156 /*v668*/, v240 /*v496*/, 0xff800000, s2
	s_set_vgpr_msb 0x816a
	v_cmp_lt_i32_e64 s2, v157 /*v669*/, v91 /*v603*/
	v_cmp_gt_i32_e64 s5, v42 /*v554*/, v87 /*v599*/
	v_cmp_lt_i32_e64 s6, v42 /*v554*/, v91 /*v603*/
	v_max3_num_f32 v217 /*v473*/, v144 /*v656*/, v145 /*v657*/, v146 /*v658*/
	v_max3_num_f32 v219 /*v475*/, v147 /*v659*/, v148 /*v660*/, v149 /*v661*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v43 /*v555*/, v87 /*v599*/
	s_set_vgpr_msb 0x6a81
	v_cndmask_b32_e64 v157 /*v669*/, v241 /*v497*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	s_set_vgpr_msb 0x810a
	v_cmp_gt_i32_e64 s3, v44 /*v556*/, v87 /*v599*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v158 /*v670*/, v242 /*v498*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	s_set_vgpr_msb 0x810a
	v_cmp_lt_i32_e64 s4, v44 /*v556*/, v91 /*v603*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v159 /*v671*/, v243 /*v499*/, 0xff800000, s2
	s_set_vgpr_msb 0x816a
	v_cmp_lt_i32_e64 s2, v43 /*v555*/, v91 /*v603*/
	v_cmp_gt_i32_e64 s5, v45 /*v557*/, v87 /*v599*/
	v_cmp_lt_i32_e64 s6, v45 /*v557*/, v91 /*v603*/
	v_max3_num_f32 v221 /*v477*/, v150 /*v662*/, v151 /*v663*/, v152 /*v664*/
	v_max3_num_f32 v223 /*v479*/, v153 /*v665*/, v154 /*v666*/, v155 /*v667*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v163 /*v675*/, v87 /*v599*/
	s_set_vgpr_msb 0x6a81
	v_cndmask_b32_e64 v160 /*v672*/, v244 /*v500*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	s_set_vgpr_msb 0x810a
	v_cmp_gt_i32_e64 s3, v164 /*v676*/, v87 /*v599*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v161 /*v673*/, v245 /*v501*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	s_set_vgpr_msb 0x810a
	v_cmp_lt_i32_e64 s4, v164 /*v676*/, v91 /*v603*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v162 /*v674*/, v246 /*v502*/, 0xff800000, s2
	s_set_vgpr_msb 0x816a
	v_cmp_lt_i32_e64 s2, v163 /*v675*/, v91 /*v603*/
	v_cmp_gt_i32_e64 s5, v165 /*v677*/, v87 /*v599*/
	v_cmp_lt_i32_e64 s6, v165 /*v677*/, v91 /*v603*/
	v_max3_num_f32 v225 /*v481*/, v156 /*v668*/, v157 /*v669*/, v158 /*v670*/
	v_max3_num_f32 v227 /*v483*/, v159 /*v671*/, v160 /*v672*/, v161 /*v673*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v166 /*v678*/, v87 /*v599*/
	s_set_vgpr_msb 0x6a81
	v_cndmask_b32_e64 v163 /*v675*/, v247 /*v503*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	s_set_vgpr_msb 0x818a
	v_cmp_gt_i32_e64 s3, v50 /*v562*/, v87 /*v599*/
	v_cndmask_b32_e64 v164 /*v676*/, v58 /*v570*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	v_cmp_lt_i32_e64 s4, v50 /*v562*/, v91 /*v603*/
	v_cndmask_b32_e64 v165 /*v677*/, v59 /*v571*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v166 /*v678*/, v91 /*v603*/
	v_cmp_gt_i32_e64 s5, v51 /*v563*/, v87 /*v599*/
	v_cmp_lt_i32_e64 s6, v51 /*v563*/, v91 /*v603*/
	s_set_vgpr_msb 0x8a4a
	v_dual_max_num_f32 v228 /*v484*/, v48 /*v560*/, v49 /*v561*/ :: v_dual_max_num_f32 v229 /*v485*/, v162 /*v674*/, v163 /*v675*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v52 /*v564*/, v87 /*v599*/
	s_set_vgpr_msb 0x4a8a
	v_cndmask_b32_e64 v166 /*v678*/, v60 /*v572*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	v_cmp_gt_i32_e64 s3, v53 /*v565*/, v87 /*v599*/
	v_cndmask_b32_e64 v167 /*v679*/, v61 /*v573*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	v_cmp_lt_i32_e64 s4, v53 /*v565*/, v91 /*v603*/
	v_cndmask_b32_e64 v168 /*v680*/, v62 /*v574*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v52 /*v564*/, v91 /*v603*/
	v_cmp_gt_i32_e64 s5, v169 /*v681*/, v87 /*v599*/
	v_cmp_lt_i32_e64 s6, v169 /*v681*/, v91 /*v603*/
	s_set_vgpr_msb 0x8a6a
	v_max3_num_f32 v231 /*v487*/, v165 /*v677*/, v166 /*v678*/, v167 /*v679*/
	v_max3_num_f32 v232 /*v488*/, v54 /*v566*/, v55 /*v567*/, v56 /*v568*/
	s_or_b32 s2, s2, vcc_lo
	s_set_vgpr_msb 0x6a55
	v_max3_num_f32 v192 /*v448*/, v192 /*v448*/, v194 /*v450*/, v196 /*v452*/
	s_set_vgpr_msb 0x5582
	v_cndmask_b32_e64 v169 /*v681*/, v63 /*v575*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	s_set_vgpr_msb 0x8255
	v_max3_num_f32 v193 /*v449*/, v193 /*v449*/, v195 /*v451*/, v197 /*v453*/
	s_set_vgpr_msb 0x5582
	v_cndmask_b32_e64 v170 /*v682*/, v64 /*v576*/, 0xff800000, s2
	s_set_vgpr_msb 0x8255
	v_max3_num_f32 v194 /*v450*/, v198 /*v454*/, v200 /*v456*/, v202 /*v458*/
	v_max3_num_f32 v195 /*v451*/, v204 /*v460*/, v206 /*v462*/, v208 /*v464*/
	v_max3_num_f32 v196 /*v452*/, v210 /*v466*/, v212 /*v468*/, v214 /*v470*/
	v_max3_num_f32 v197 /*v453*/, v216 /*v472*/, v218 /*v474*/, v220 /*v476*/
	v_max3_num_f32 v198 /*v454*/, v222 /*v478*/, v224 /*v480*/, v226 /*v482*/
	s_set_vgpr_msb 0x5559
	v_max3_num_f32 v200 /*v456*/, v228 /*v484*/, v104 /*v616*/, v230 /*v486*/
	s_or_b32 s2, s6, s5
	s_set_vgpr_msb 0x596a
	v_max3_num_f32 v233 /*v489*/, v168 /*v680*/, v169 /*v681*/, v170 /*v682*/
	s_set_vgpr_msb 0x6a82
	v_cndmask_b32_e64 v171 /*v683*/, v65 /*v577*/, 0xff800000, s2
	s_set_vgpr_msb 0x8255
	v_max3_num_f32 v199 /*v455*/, v199 /*v455*/, v201 /*v457*/, v203 /*v459*/
	v_max3_num_f32 v201 /*v457*/, v205 /*v461*/, v207 /*v463*/, v209 /*v465*/
	v_max3_num_f32 v192 /*v448*/, v192 /*v448*/, v194 /*v450*/, v195 /*v451*/
	v_max3_num_f32 v194 /*v450*/, v196 /*v452*/, v197 /*v453*/, v198 /*v454*/
	s_set_vgpr_msb 0x5565
	v_max3_num_f32 v195 /*v451*/, v200 /*v456*/, v232 /*v488*/, v57 /*v569*/
	s_set_vgpr_msb 0x6555
	v_max3_num_f32 v196 /*v452*/, v211 /*v467*/, v213 /*v469*/, v215 /*v471*/
	v_max3_num_f32 v197 /*v453*/, v217 /*v473*/, v219 /*v475*/, v221 /*v477*/
	v_max3_num_f32 v198 /*v454*/, v223 /*v479*/, v225 /*v481*/, v227 /*v483*/
	s_set_vgpr_msb 0x5559
	v_max3_num_f32 v200 /*v456*/, v229 /*v485*/, v164 /*v676*/, v231 /*v487*/
	s_set_vgpr_msb 0x5955
	v_max3_num_f32 v192 /*v448*/, v192 /*v448*/, v194 /*v450*/, v195 /*v451*/
	v_max3_num_f32 v193 /*v449*/, v193 /*v449*/, v199 /*v455*/, v201 /*v457*/
	v_max3_num_f32 v194 /*v450*/, v196 /*v452*/, v197 /*v453*/, v198 /*v454*/
	s_set_vgpr_msb 0x5565
	v_max3_num_f32 v195 /*v451*/, v200 /*v456*/, v233 /*v489*/, v171 /*v683*/
	s_set_vgpr_msb 0x6555
	v_max3_num_f32 v193 /*v449*/, v193 /*v449*/, v194 /*v450*/, v195 /*v451*/
	v_dual_mov_b32 v196 /*v452*/, v192 /*v448*/ :: v_dual_mov_b32 v194 /*v450*/, v193 /*v449*/
	v_permlanex16_b32 v196 /*v452*/, v196 /*v452*/, s70, 0xfedcba98
	v_permlanex16_b32 v194 /*v450*/, v194 /*v450*/, s70, 0xfedcba98
	v_dual_max_num_f32 v192 /*v448*/, v192 /*v448*/, v196 /*v452*/ :: v_dual_max_num_f32 v193 /*v449*/, v193 /*v449*/, v194 /*v450*/
	s_set_vgpr_msb 0x5549
	v_sub_f32_e32 v195 /*v451*/, v192 /*v448*/, v66 /*v578*/
	v_max_num_f32_e32 v192 /*v448*/, v192 /*v448*/, v66 /*v578*/
	v_sub_f32_e32 v194 /*v450*/, v193 /*v449*/, v71 /*v583*/
	s_set_vgpr_msb 0x4904
	v_cmp_lt_f32_e32 vcc_lo, 0x41000000, v195 /*v451*/
	s_cmp_eq_u32 vcc_lo, 0
	s_cselect_b32 s2, -1, 0
	s_set_vgpr_msb 0x489
	v_cndmask_b32_e64 v92 /*v604*/, v192 /*v448*/, v66 /*v578*/, s2
	s_set_vgpr_msb 0x8946
	v_cmp_lt_f32_e64 s2, 0x41000000, v194 /*v450*/
	v_max_num_f32_e32 v192 /*v448*/, v71 /*v583*/, v193 /*v449*/
	s_cmp_lg_u32 s2, 0
	s_cselect_b32 s3, -1, 0
	s_cmp_eq_u32 s2, 0
	s_cselect_b32 s2, -1, 0
	s_set_vgpr_msb 0x4689
	v_cndmask_b32_e64 v93 /*v605*/, v192 /*v448*/, v71 /*v583*/, s2
	v_mul_f32_e32 v60 /*v572*/, 0xbfb8aa3b, v92 /*v604*/
	v_mul_f32_e32 v70 /*v582*/, 0xbfb8aa3b, v93 /*v605*/
	s_set_vgpr_msb 0x8962
	v_pk_fma_f32 v[206:207] /*v[462:463]*/, v[2:3] /*v[514:515]*/, s[90:91], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6261
	v_pk_fma_f32 v[192:193] /*v[448:449]*/, v[250:251] /*v[506:507]*/, s[90:91], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[198:199] /*v[454:455]*/, v[254:255] /*v[510:511]*/, s[90:91], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6162
	v_pk_fma_f32 v[216:217] /*v[472:473]*/, v[68:69] /*v[580:581]*/, s[90:91], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[242:243] /*v[498:499]*/, v[96:97] /*v[608:609]*/, s[90:91], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x62a2
	v_pk_fma_f32 v[68:69] /*v[580:581]*/, v[108:109] /*v[620:621]*/, s[90:91], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[96:97] /*v[608:609]*/, v[112:113] /*v[624:625]*/, s[90:91], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa262
	v_pk_fma_f32 v[204:205] /*v[460:461]*/, v[0:1] /*v[512:513]*/, s[90:91], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6241
	v_exp_f32_e32 v222 /*v478*/, v206 /*v462*/
	v_exp_f32_e32 v232 /*v488*/, v207 /*v463*/
	v_nop
	s_set_vgpr_msb 0x4162
	v_pk_fma_f32 v[206:207] /*v[462:463]*/, v[8:9] /*v[520:521]*/, s[90:91], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[212:213] /*v[468:469]*/, v[12:13] /*v[524:525]*/, s[90:91], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[224:225] /*v[480:481]*/, v[20:21] /*v[532:533]*/, s[90:91], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6241
	v_exp_f32_e32 v194 /*v450*/, v193 /*v449*/
	v_exp_f32_e32 v202 /*v458*/, v198 /*v454*/
	v_exp_f32_e32 v208 /*v464*/, v199 /*v455*/
	v_nop
	s_set_vgpr_msb 0x4162
	v_pk_fma_f32 v[198:199] /*v[454:455]*/, v[4:5] /*v[516:517]*/, s[90:91], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v193 /*v449*/, v68 /*v580*/
	v_exp_f32_e32 v195 /*v451*/, v69 /*v581*/
	v_nop
	s_set_vgpr_msb 0x62a2
	v_pk_fma_f32 v[68:69] /*v[580:581]*/, v[114:115] /*v[626:627]*/, s[90:91], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa242
	v_exp_f32_e32 v203 /*v459*/, v96 /*v608*/
	v_exp_f32_e32 v209 /*v465*/, v97 /*v609*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[96:97] /*v[608:609]*/, v[118:119] /*v[630:631]*/, s[90:91], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa261
	v_pk_fma_f32 v[196:197] /*v[452:453]*/, v[252:253] /*v[508:509]*/, s[90:91], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v210 /*v466*/, v204 /*v460*/
	v_exp_f32_e32 v218 /*v474*/, v205 /*v461*/
	v_nop
	s_set_vgpr_msb 0x6162
	v_pk_fma_f32 v[204:205] /*v[460:461]*/, v[6:7] /*v[518:519]*/, s[90:91], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6281
	v_exp_f32_e32 v14 /*v526*/, v206 /*v462*/
	s_set_vgpr_msb 0x8141
	v_exp_f32_e32 v206 /*v462*/, v212 /*v468*/
	v_exp_f32_e32 v214 /*v470*/, v213 /*v469*/
	v_nop
	s_set_vgpr_msb 0x4162
	v_pk_fma_f32 v[212:213] /*v[468:469]*/, v[18:19] /*v[530:531]*/, s[90:91], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6281
	v_exp_f32_e32 v6 /*v518*/, v224 /*v480*/
	v_exp_f32_e32 v18 /*v530*/, v225 /*v481*/
	v_nop
	s_set_vgpr_msb 0x8162
	v_pk_fma_f32 v[224:225] /*v[480:481]*/, v[94:95] /*v[606:607]*/, s[90:91], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x62a2
	v_pk_fma_f32 v[94:95] /*v[606:607]*/, v[110:111] /*v[622:623]*/, s[90:91], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa241
	v_exp_f32_e32 v236 /*v492*/, v198 /*v454*/
	v_exp_f32_e32 v250 /*v506*/, v199 /*v455*/
	v_nop
	s_set_vgpr_msb 0x4162
	v_pk_fma_f32 v[198:199] /*v[454:455]*/, v[10:11] /*v[522:523]*/, s[90:91], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v211 /*v467*/, v68 /*v580*/
	v_exp_f32_e32 v219 /*v475*/, v69 /*v581*/
	v_nop
	s_set_vgpr_msb 0x62a2
	v_pk_fma_f32 v[68:69] /*v[580:581]*/, v[120:121] /*v[632:633]*/, s[90:91], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa242
	v_exp_f32_e32 v237 /*v493*/, v96 /*v608*/
	v_exp_f32_e32 v251 /*v507*/, v97 /*v609*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[96:97] /*v[608:609]*/, v[124:125] /*v[636:637]*/, s[90:91], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa241
	v_exp_f32_e32 v200 /*v456*/, v197 /*v453*/
	s_set_vgpr_msb 0x4142
	v_exp_f32_e32 v197 /*v453*/, v94 /*v606*/
	v_exp_f32_e32 v201 /*v457*/, v95 /*v607*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[94:95] /*v[606:607]*/, v[116:117] /*v[628:629]*/, s[90:91], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa241
	v_exp_f32_e32 v254 /*v510*/, v204 /*v460*/
	s_set_vgpr_msb 0x4181
	v_exp_f32_e32 v10 /*v522*/, v205 /*v461*/
	s_set_vgpr_msb 0x8141
	v_exp_f32_e32 v204 /*v460*/, v199 /*v455*/
	s_set_vgpr_msb 0x4142
	v_exp_f32_e32 v255 /*v511*/, v68 /*v580*/
	s_set_vgpr_msb 0x42a2
	v_exp_f32_e32 v11 /*v523*/, v69 /*v581*/
	v_nop
	v_pk_fma_f32 v[68:69] /*v[580:581]*/, v[126:127] /*v[638:639]*/, s[90:91], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa242
	v_exp_f32_e32 v199 /*v455*/, v96 /*v608*/
	v_exp_f32_e32 v205 /*v461*/, v97 /*v609*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[96:97] /*v[608:609]*/, v[130:131] /*v[642:643]*/, s[90:91], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa242
	v_exp_f32_e32 v223 /*v479*/, v94 /*v606*/
	v_exp_f32_e32 v233 /*v489*/, v95 /*v607*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[94:95] /*v[606:607]*/, v[122:123] /*v[634:635]*/, s[90:91], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa281
	v_exp_f32_e32 v26 /*v538*/, v207 /*v463*/
	s_set_vgpr_msb 0x8162
	v_pk_fma_f32 v[220:221] /*v[476:477]*/, v[16:17] /*v[528:529]*/, s[90:91], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v207 /*v463*/, v68 /*v580*/
	v_exp_f32_e32 v215 /*v471*/, v69 /*v581*/
	v_nop
	s_set_vgpr_msb 0x62a2
	v_pk_fma_f32 v[68:69] /*v[580:581]*/, v[132:133] /*v[644:645]*/, s[90:91], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa242
	v_exp_f32_e32 v229 /*v485*/, v96 /*v608*/
	v_exp_f32_e32 v241 /*v497*/, v97 /*v609*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[96:97] /*v[608:609]*/, v[136:137] /*v[648:649]*/, s[90:91], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v15 /*v527*/, v94 /*v606*/
	v_exp_f32_e32 v27 /*v539*/, v95 /*v607*/
	v_nop
	v_pk_fma_f32 v[94:95] /*v[606:607]*/, v[128:129] /*v[640:641]*/, s[90:91], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa241
	v_exp_f32_e32 v228 /*v484*/, v220 /*v476*/
	v_exp_f32_e32 v240 /*v496*/, v221 /*v477*/
	v_nop
	s_set_vgpr_msb 0x4162
	v_pk_fma_f32 v[220:221] /*v[476:477]*/, v[22:23] /*v[534:535]*/, s[90:91], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v245 /*v501*/, v68 /*v580*/
	s_set_vgpr_msb 0x62a2
	v_exp_f32_e32 v3 /*v515*/, v69 /*v581*/
	v_nop
	v_pk_fma_f32 v[68:69] /*v[580:581]*/, v[138:139] /*v[650:651]*/, s[90:91], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v23 /*v535*/, v96 /*v608*/
	v_exp_f32_e32 v33 /*v545*/, v97 /*v609*/
	v_nop
	v_pk_fma_f32 v[96:97] /*v[608:609]*/, v[142:143] /*v[654:655]*/, s[90:91], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa241
	v_exp_f32_e32 v226 /*v482*/, v217 /*v473*/
	s_set_vgpr_msb 0x4142
	v_exp_f32_e32 v217 /*v473*/, v94 /*v606*/
	v_exp_f32_e32 v227 /*v483*/, v95 /*v607*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[94:95] /*v[606:607]*/, v[134:135] /*v[646:647]*/, s[90:91], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa241
	v_exp_f32_e32 v244 /*v500*/, v212 /*v468*/
	s_set_vgpr_msb 0x4181
	v_exp_f32_e32 v2 /*v514*/, v213 /*v469*/
	v_nop
	s_set_vgpr_msb 0x8162
	v_pk_fma_f32 v[212:213] /*v[468:469]*/, v[24:25] /*v[536:537]*/, s[90:91], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6281
	v_exp_f32_e32 v22 /*v534*/, v220 /*v476*/
	s_set_vgpr_msb 0x8162
	v_pk_fma_f32 v[230:231] /*v[486:487]*/, v[28:29] /*v[540:541]*/, s[90:91], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[238:239] /*v[494:495]*/, v[30:31] /*v[542:543]*/, s[90:91], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6241
	v_exp_f32_e32 v220 /*v476*/, v225 /*v481*/
	s_set_vgpr_msb 0x41a2
	v_exp_f32_e32 v37 /*v549*/, v68 /*v580*/
	v_exp_f32_e32 v45 /*v557*/, v69 /*v581*/
	v_nop
	v_pk_fma_f32 v[68:69] /*v[580:581]*/, v[144:145] /*v[656:657]*/, s[90:91], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa242
	v_exp_f32_e32 v225 /*v481*/, v96 /*v608*/
	v_exp_f32_e32 v235 /*v491*/, v97 /*v609*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[96:97] /*v[608:609]*/, v[148:149] /*v[660:661]*/, s[90:91], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v7 /*v519*/, v94 /*v606*/
	v_exp_f32_e32 v19 /*v531*/, v95 /*v607*/
	v_nop
	v_pk_fma_f32 v[94:95] /*v[606:607]*/, v[140:141] /*v[652:653]*/, s[90:91], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa281
	v_exp_f32_e32 v36 /*v548*/, v212 /*v468*/
	s_set_vgpr_msb 0x8141
	v_exp_f32_e32 v212 /*v468*/, v224 /*v480*/
	v_exp_f32_e32 v224 /*v480*/, v230 /*v486*/
	v_exp_f32_e32 v234 /*v490*/, v231 /*v487*/
	v_nop
	s_set_vgpr_msb 0x4162
	v_pk_fma_f32 v[230:231] /*v[486:487]*/, v[34:35] /*v[546:547]*/, s[90:91], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6241
	v_exp_f32_e32 v252 /*v508*/, v239 /*v495*/
	s_set_vgpr_msb 0x4142
	v_exp_f32_e32 v239 /*v495*/, v68 /*v580*/
	v_exp_f32_e32 v253 /*v509*/, v69 /*v581*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[68:69] /*v[580:581]*/, v[150:151] /*v[662:663]*/, s[90:91], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v17 /*v529*/, v96 /*v608*/
	v_exp_f32_e32 v29 /*v541*/, v97 /*v609*/
	v_nop
	v_pk_fma_f32 v[96:97] /*v[608:609]*/, v[154:155] /*v[666:667]*/, s[90:91], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa281
	v_exp_f32_e32 v32 /*v544*/, v221 /*v477*/
	v_exp_f32_e32 v44 /*v556*/, v213 /*v469*/
	s_set_vgpr_msb 0x8142
	v_exp_f32_e32 v213 /*v469*/, v94 /*v606*/
	v_exp_f32_e32 v221 /*v477*/, v95 /*v607*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[94:95] /*v[606:607]*/, v[146:147] /*v[658:659]*/, s[90:91], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa281
	v_exp_f32_e32 v0 /*v512*/, v242 /*v498*/
	v_exp_f32_e32 v12 /*v524*/, v243 /*v499*/
	v_exp_f32_e32 v16 /*v528*/, v230 /*v486*/
	s_set_vgpr_msb 0x8162
	v_pk_fma_f32 v[242:243] /*v[498:499]*/, v[38:39] /*v[550:551]*/, s[90:91], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6281
	v_exp_f32_e32 v28 /*v540*/, v231 /*v487*/
	v_nop
	s_set_vgpr_msb 0x8162
	v_pk_fma_f32 v[230:231] /*v[486:487]*/, v[40:41] /*v[552:553]*/, s[90:91], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x62a2
	v_pk_fma_f32 v[8:9] /*v[520:521]*/, v[46:47] /*v[558:559]*/, s[90:91], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v31 /*v543*/, v68 /*v580*/
	v_exp_f32_e32 v41 /*v553*/, v69 /*v581*/
	v_nop
	v_pk_fma_f32 v[68:69] /*v[580:581]*/, v[156:157] /*v[668:669]*/, s[90:91], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v53 /*v565*/, v96 /*v608*/
	v_exp_f32_e32 v59 /*v571*/, v97 /*v609*/
	v_nop
	v_pk_fma_f32 v[96:97] /*v[608:609]*/, v[160:161] /*v[672:673]*/, s[90:91], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa241
	v_exp_f32_e32 v192 /*v448*/, v192 /*v448*/
	s_set_vgpr_msb 0x4162
	v_pk_fma_f32 v[246:247] /*v[502:503]*/, v[98:99] /*v[610:611]*/, s[90:91], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x62a2
	v_exp_f32_e32 v1 /*v513*/, v94 /*v606*/
	v_exp_f32_e32 v13 /*v525*/, v95 /*v607*/
	v_nop
	v_pk_fma_f32 v[94:95] /*v[606:607]*/, v[152:153] /*v[664:665]*/, s[90:91], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa281
	v_exp_f32_e32 v50 /*v562*/, v243 /*v499*/
	v_exp_f32_e32 v58 /*v570*/, v231 /*v487*/
	s_set_vgpr_msb 0x81a2
	v_pk_fma_f32 v[24:25] /*v[536:537]*/, v[48:49] /*v[560:561]*/, s[90:91], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v20 /*v532*/, v9 /*v521*/
	v_pk_fma_f32 v[48:49] /*v[560:561]*/, v[106:107] /*v[618:619]*/, s[90:91], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa242
	v_exp_f32_e32 v231 /*v487*/, v68 /*v580*/
	v_exp_f32_e32 v243 /*v499*/, v69 /*v581*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[68:69] /*v[580:581]*/, v[162:163] /*v[674:675]*/, s[90:91], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v9 /*v521*/, v96 /*v608*/
	v_exp_f32_e32 v21 /*v533*/, v97 /*v609*/
	v_nop
	v_pk_fma_f32 v[96:97] /*v[608:609]*/, v[166:167] /*v[678:679]*/, s[90:91], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa241
	v_exp_f32_e32 v196 /*v452*/, v196 /*v452*/
	v_exp_f32_e32 v238 /*v494*/, v238 /*v494*/
	s_set_vgpr_msb 0x4181
	v_exp_f32_e32 v30 /*v542*/, v246 /*v502*/
	v_exp_f32_e32 v40 /*v552*/, v247 /*v503*/
	v_nop
	s_set_vgpr_msb 0x8162
	v_pk_fma_f32 v[246:247] /*v[502:503]*/, v[100:101] /*v[612:613]*/, s[90:91], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x62a2
	v_pk_fma_f32 v[4:5] /*v[516:517]*/, v[102:103] /*v[614:615]*/, s[90:91], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v43 /*v555*/, v94 /*v606*/
	v_exp_f32_e32 v51 /*v563*/, v95 /*v607*/
	v_nop
	v_pk_fma_f32 v[94:95] /*v[606:607]*/, v[158:159] /*v[670:671]*/, s[90:91], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v34 /*v546*/, v25 /*v537*/
	v_pk_fma_f32 v[62:63] /*v[574:575]*/, v[54:55] /*v[566:567]*/, s[90:91], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v54 /*v566*/, v49 /*v561*/
	v_exp_f32_e32 v25 /*v537*/, v68 /*v580*/
	v_exp_f32_e32 v35 /*v547*/, v69 /*v581*/
	v_exp_f32_e32 v49 /*v561*/, v96 /*v608*/
	v_pk_fma_f32 v[68:69] /*v[580:581]*/, v[168:169] /*v[680:681]*/, s[90:91], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v55 /*v567*/, v97 /*v609*/
	v_nop
	s_set_vgpr_msb 0xa285
	v_pk_add_f32 v[96:97] /*v[608:609]*/, v[192:193] /*v[448:449]*/, v[194:195] /*v[450:451]*/
	v_pk_add_f32 v[98:99] /*v[610:611]*/, v[200:201] /*v[456:457]*/, v[202:203] /*v[458:459]*/
	s_set_vgpr_msb 0x8541
	v_exp_f32_e32 v198 /*v454*/, v198 /*v454*/
	s_set_vgpr_msb 0x4181
	v_exp_f32_e32 v42 /*v554*/, v242 /*v498*/
	v_exp_f32_e32 v52 /*v564*/, v230 /*v486*/
	s_set_vgpr_msb 0x8141
	v_exp_f32_e32 v230 /*v486*/, v246 /*v502*/
	v_exp_f32_e32 v242 /*v498*/, v247 /*v503*/
	s_set_vgpr_msb 0x4142
	v_exp_f32_e32 v246 /*v502*/, v4 /*v516*/
	s_set_vgpr_msb 0x42a2
	v_exp_f32_e32 v4 /*v516*/, v5 /*v517*/
	v_pk_fma_f32 v[38:39] /*v[550:551]*/, v[104:105] /*v[616:617]*/, s[90:91], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa242
	v_exp_f32_e32 v247 /*v503*/, v94 /*v606*/
	s_set_vgpr_msb 0x42a2
	v_exp_f32_e32 v5 /*v517*/, v95 /*v607*/
	v_nop
	v_pk_fma_f32 v[94:95] /*v[606:607]*/, v[164:165] /*v[676:677]*/, s[90:91], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[64:65] /*v[576:577]*/, v[56:57] /*v[568:569]*/, s[90:91], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v57 /*v569*/, v68 /*v580*/
	v_exp_f32_e32 v61 /*v573*/, v69 /*v581*/
	v_nop
	s_set_vgpr_msb 0xa289
	v_pk_add_f32 v[68:69] /*v[580:581]*/, v[196:197] /*v[452:453]*/, v[96:97] /*v[608:609]*/
	v_pk_add_f32 v[96:97] /*v[608:609]*/, v[208:209] /*v[464:465]*/, v[98:99] /*v[610:611]*/
	s_set_vgpr_msb 0x8985
	v_pk_add_f32 v[98:99] /*v[610:611]*/, v[210:211] /*v[466:467]*/, v[218:219] /*v[474:475]*/
	v_pk_add_f32 v[100:101] /*v[612:613]*/, v[232:233] /*v[488:489]*/, v[236:237] /*v[492:493]*/
	s_set_vgpr_msb 0x8589
	v_pk_add_f32 v[102:103] /*v[614:615]*/, v[254:255] /*v[510:511]*/, v[10:11] /*v[522:523]*/
	s_set_vgpr_msb 0x898a
	v_pk_add_f32 v[112:113] /*v[624:625]*/, v[18:19] /*v[530:531]*/, v[22:23] /*v[534:535]*/
	v_pk_add_f32 v[114:115] /*v[626:627]*/, v[36:37] /*v[548:549]*/, v[44:45] /*v[556:557]*/
	s_set_vgpr_msb 0x8a85
	v_pk_add_f32 v[118:119] /*v[630:631]*/, v[238:239] /*v[494:495]*/, v[252:253] /*v[508:509]*/
	s_set_vgpr_msb 0x858a
	v_pk_add_f32 v[120:121] /*v[632:633]*/, v[12:13] /*v[524:525]*/, v[16:17] /*v[528:529]*/
	s_set_vgpr_msb 0x8a41
	v_exp_f32_e32 v216 /*v472*/, v216 /*v472*/
	s_set_vgpr_msb 0x4186
	v_exp_f32_e32 v8 /*v520*/, v8 /*v520*/
	v_exp_f32_e32 v24 /*v536*/, v24 /*v536*/
	v_exp_f32_e32 v46 /*v558*/, v39 /*v551*/
	v_exp_f32_e32 v48 /*v560*/, v48 /*v560*/
	v_exp_f32_e32 v47 /*v559*/, v95 /*v607*/
	v_pk_add_f32 v[104:105] /*v[616:617]*/, v[26:27] /*v[538:539]*/, v[198:199] /*v[454:455]*/
	s_set_vgpr_msb 0x8685
	v_pk_add_f32 v[106:107] /*v[618:619]*/, v[206:207] /*v[462:463]*/, v[214:215] /*v[470:471]*/
	s_set_vgpr_msb 0x8589
	v_pk_add_f32 v[98:99] /*v[610:611]*/, v[222:223] /*v[478:479]*/, v[98:99] /*v[610:611]*/
	v_pk_add_f32 v[100:101] /*v[612:613]*/, v[250:251] /*v[506:507]*/, v[100:101] /*v[612:613]*/
	s_set_vgpr_msb 0x898a
	v_pk_add_f32 v[102:103] /*v[614:615]*/, v[14:15] /*v[526:527]*/, v[102:103] /*v[614:615]*/
	s_set_vgpr_msb 0x8a85
	v_pk_add_f32 v[108:109] /*v[620:621]*/, v[226:227] /*v[482:483]*/, v[228:229] /*v[484:485]*/
	v_pk_add_f32 v[116:117] /*v[628:629]*/, v[220:221] /*v[476:477]*/, v[224:225] /*v[480:481]*/
	s_set_vgpr_msb 0x858a
	v_pk_add_f32 v[112:113] /*v[624:625]*/, v[32:33] /*v[544:545]*/, v[112:113] /*v[624:625]*/
	s_set_vgpr_msb 0x8a89
	v_pk_add_f32 v[114:115] /*v[626:627]*/, v[212:213] /*v[468:469]*/, v[114:115] /*v[626:627]*/
	s_set_vgpr_msb 0x898a
	v_pk_add_f32 v[122:123] /*v[634:635]*/, v[30:31] /*v[542:543]*/, v[40:41] /*v[552:553]*/
	v_pk_add_f32 v[124:125] /*v[636:637]*/, v[50:51] /*v[562:563]*/, v[52:53] /*v[564:565]*/
	s_set_vgpr_msb 0x8a85
	v_pk_add_f32 v[126:127] /*v[638:639]*/, v[230:231] /*v[486:487]*/, v[242:243] /*v[498:499]*/
	s_set_vgpr_msb 0x85aa
	v_pk_add_f32 v[118:119] /*v[630:631]*/, v[0:1] /*v[512:513]*/, v[118:119] /*v[630:631]*/
	v_pk_add_f32 v[120:121] /*v[632:633]*/, v[28:29] /*v[540:541]*/, v[120:121] /*v[632:633]*/
	v_pk_add_f32 v[68:69] /*v[580:581]*/, v[68:69] /*v[580:581]*/, v[96:97] /*v[608:609]*/
	v_exp_f32_e32 v38 /*v550*/, v38 /*v550*/
	v_exp_f32_e32 v56 /*v568*/, v62 /*v574*/
	v_exp_f32_e32 v60 /*v572*/, v63 /*v575*/
	v_exp_f32_e32 v39 /*v551*/, v94 /*v606*/
	v_nop
	v_pk_fma_f32 v[94:95] /*v[606:607]*/, v[170:171] /*v[682:683]*/, s[90:91], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xaa89
	v_pk_add_f32 v[104:105] /*v[616:617]*/, v[204:205] /*v[460:461]*/, v[104:105] /*v[616:617]*/
	v_pk_add_f32 v[106:107] /*v[618:619]*/, v[216:217] /*v[472:473]*/, v[106:107] /*v[618:619]*/
	v_pk_add_f32 v[110:111] /*v[622:623]*/, v[244:245] /*v[500:501]*/, v[2:3] /*v[514:515]*/
	v_pk_add_f32 v[108:109] /*v[620:621]*/, v[240:241] /*v[496:497]*/, v[108:109] /*v[620:621]*/
	v_pk_add_f32 v[116:117] /*v[628:629]*/, v[234:235] /*v[490:491]*/, v[116:117] /*v[628:629]*/
	s_set_vgpr_msb 0x898a
	v_pk_add_f32 v[122:123] /*v[634:635]*/, v[42:43] /*v[554:555]*/, v[122:123] /*v[634:635]*/
	v_pk_add_f32 v[124:125] /*v[636:637]*/, v[58:59] /*v[570:571]*/, v[124:125] /*v[636:637]*/
	s_set_vgpr_msb 0x8a89
	v_pk_add_f32 v[126:127] /*v[638:639]*/, v[246:247] /*v[502:503]*/, v[126:127] /*v[638:639]*/
	s_set_vgpr_msb 0x898a
	v_pk_add_f32 v[128:129] /*v[640:641]*/, v[4:5] /*v[516:517]*/, v[8:9] /*v[520:521]*/
	v_pk_add_f32 v[130:131] /*v[642:643]*/, v[24:25] /*v[536:537]*/, v[34:35] /*v[546:547]*/
	v_pk_add_f32 v[132:133] /*v[644:645]*/, v[46:47] /*v[558:559]*/, v[48:49] /*v[560:561]*/
	v_pk_add_f32 v[68:69] /*v[580:581]*/, v[98:99] /*v[610:611]*/, v[68:69] /*v[580:581]*/
	v_pk_add_f32 v[98:99] /*v[610:611]*/, v[100:101] /*v[612:613]*/, v[102:103] /*v[614:615]*/
	v_pk_add_f32 v[100:101] /*v[612:613]*/, v[112:113] /*v[624:625]*/, v[114:115] /*v[626:627]*/
	v_pk_add_f32 v[102:103] /*v[614:615]*/, v[118:119] /*v[630:631]*/, v[120:121] /*v[632:633]*/
	v_exp_f32_e32 v62 /*v574*/, v64 /*v576*/
	v_exp_f32_e32 v63 /*v575*/, v94 /*v606*/
	v_pk_add_f32 v[110:111] /*v[622:623]*/, v[6:7] /*v[518:519]*/, v[110:111] /*v[622:623]*/
	v_pk_add_f32 v[134:135] /*v[646:647]*/, v[56:57] /*v[568:569]*/, v[60:61] /*v[572:573]*/
	v_pk_add_f32 v[96:97] /*v[608:609]*/, v[20:21] /*v[532:533]*/, v[128:129] /*v[640:641]*/
	v_pk_add_f32 v[128:129] /*v[640:641]*/, v[38:39] /*v[550:551]*/, v[130:131] /*v[642:643]*/
	v_pk_add_f32 v[130:131] /*v[642:643]*/, v[54:55] /*v[566:567]*/, v[132:133] /*v[644:645]*/
	v_pk_add_f32 v[106:107] /*v[618:619]*/, v[106:107] /*v[618:619]*/, v[108:109] /*v[620:621]*/
	v_pk_add_f32 v[108:109] /*v[620:621]*/, v[124:125] /*v[636:637]*/, v[126:127] /*v[638:639]*/
	v_pk_add_f32 v[98:99] /*v[610:611]*/, v[104:105] /*v[616:617]*/, v[98:99] /*v[610:611]*/
	v_pk_add_f32 v[100:101] /*v[612:613]*/, v[116:117] /*v[628:629]*/, v[100:101] /*v[612:613]*/
	v_pk_add_f32 v[102:103] /*v[614:615]*/, v[122:123] /*v[634:635]*/, v[102:103] /*v[614:615]*/
	v_pk_add_f32 v[132:133] /*v[644:645]*/, v[62:63] /*v[574:575]*/, v[134:135] /*v[646:647]*/
	v_pk_add_f32 v[104:105] /*v[616:617]*/, v[110:111] /*v[622:623]*/, v[106:107] /*v[618:619]*/
	v_pk_add_f32 v[96:97] /*v[608:609]*/, v[96:97] /*v[608:609]*/, v[108:109] /*v[620:621]*/
	v_pk_add_f32 v[106:107] /*v[618:619]*/, v[128:129] /*v[640:641]*/, v[130:131] /*v[642:643]*/
	v_pk_add_f32 v[68:69] /*v[580:581]*/, v[68:69] /*v[580:581]*/, v[98:99] /*v[610:611]*/
	v_pk_add_f32 v[98:99] /*v[610:611]*/, v[100:101] /*v[612:613]*/, v[102:103] /*v[614:615]*/
	v_exp_f32_e32 v64 /*v576*/, v65 /*v577*/
	v_exp_f32_e32 v65 /*v577*/, v95 /*v607*/
	v_nop
	v_pk_add_f32 v[94:95] /*v[606:607]*/, v[132:133] /*v[644:645]*/, v[106:107] /*v[618:619]*/
	v_pk_add_f32 v[68:69] /*v[580:581]*/, v[104:105] /*v[616:617]*/, v[68:69] /*v[580:581]*/
	v_pk_add_f32 v[96:97] /*v[608:609]*/, v[96:97] /*v[608:609]*/, v[98:99] /*v[610:611]*/
	v_sub_f32_e32 v70 /*v582*/, v66 /*v578*/, v92 /*v604*/
	v_pk_add_f32 v[94:95] /*v[606:607]*/, v[64:65] /*v[576:577]*/, v[94:95] /*v[606:607]*/
	v_pk_add_f32 v[68:69] /*v[580:581]*/, v[68:69] /*v[580:581]*/, v[96:97] /*v[608:609]*/
	v_mul_f32_e32 v70 /*v582*/, 0x3fb8aa3b, v70 /*v582*/
	v_pk_add_f32 v[66:67] /*v[578:579]*/, v[94:95] /*v[606:607]*/, v[68:69] /*v[580:581]*/
	v_exp_f32_e32 v70 /*v582*/, v70 /*v582*/
	v_dual_mov_b32 v68 /*v580*/, v66 /*v578*/ :: v_dual_mov_b32 v69 /*v581*/, v67 /*v579*/
	v_permlanex16_b32 v68 /*v580*/, v68 /*v580*/, s70, 0xfedcba98
	v_permlanex16_b32 v69 /*v581*/, v69 /*v581*/, s70, 0xfedcba98
	s_set_vgpr_msb 0x8a00
	s_cbranch_vccz .LBB0_8
	s_set_vgpr_msb 8
	v_pk_mul_f32 v[126:127], v[126:127], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[124:125], v[124:125], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[122:123], v[122:123], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[120:121], v[120:121], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[118:119], v[118:119], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[116:117], v[116:117], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[114:115], v[114:115], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[112:113], v[112:113], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[110:111], v[110:111], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[108:109], v[108:109], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[106:107], v[106:107], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[104:105], v[104:105], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[102:103], v[102:103], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[100:101], v[100:101], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[98:99], v[98:99], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[96:97], v[96:97], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[94:95], v[94:95], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[92:93], v[92:93], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[90:91], v[90:91], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[88:89], v[88:89], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[86:87], v[86:87], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[84:85], v[84:85], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[82:83], v[82:83], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[80:81], v[80:81], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[78:79], v[78:79], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[76:77], v[76:77], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[74:75], v[74:75], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[72:73], v[72:73], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[70:71], v[70:71], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[68:69], v[68:69], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[66:67], v[66:67], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[64:65], v[64:65], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x800
.LBB0_8:
	s_set_vgpr_msb 0x8a
	v_sub_f32_e32 v71 /*v583*/, v71 /*v583*/, v93 /*v605*/
	s_and_b32 s2, s3, exec_lo
	s_cselect_b32 s2, 1, 0
	s_cmp_lg_u32 s2, 1
	v_mul_f32_e32 v71 /*v583*/, 0x3fb8aa3b, v71 /*v583*/
	v_exp_f32_e32 v71 /*v583*/, v71 /*v583*/
	s_set_vgpr_msb 0x8a00
	s_cbranch_scc1 .LBB0_3
	v_nop
	s_set_vgpr_msb 0x82
	v_mov_b32_e32 v94 /*v606*/, v71 /*v583*/
	s_set_vgpr_msb 0x8208
	v_pk_mul_f32 v[62:63], v[62:63], v[94:95] /*v[606:607]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[60:61], v[60:61], v[94:95] /*v[606:607]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59], v[58:59], v[94:95] /*v[606:607]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[56:57], v[56:57], v[94:95] /*v[606:607]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[54:55], v[54:55], v[94:95] /*v[606:607]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[52:53], v[52:53], v[94:95] /*v[606:607]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[50:51], v[50:51], v[94:95] /*v[606:607]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[48:49], v[48:49], v[94:95] /*v[606:607]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[46:47], v[46:47], v[94:95] /*v[606:607]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[44:45], v[44:45], v[94:95] /*v[606:607]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[42:43], v[42:43], v[94:95] /*v[606:607]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[40:41], v[40:41], v[94:95] /*v[606:607]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39], v[38:39], v[94:95] /*v[606:607]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[36:37], v[36:37], v[94:95] /*v[606:607]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35], v[34:35], v[94:95] /*v[606:607]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[32:33], v[32:33], v[94:95] /*v[606:607]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[30:31], v[30:31], v[94:95] /*v[606:607]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[28:29], v[28:29], v[94:95] /*v[606:607]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[26:27], v[26:27], v[94:95] /*v[606:607]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25], v[24:25], v[94:95] /*v[606:607]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[22:23], v[22:23], v[94:95] /*v[606:607]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21], v[20:21], v[94:95] /*v[606:607]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[18:19], v[18:19], v[94:95] /*v[606:607]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[16:17], v[16:17], v[94:95] /*v[606:607]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[14:15], v[94:95] /*v[606:607]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13], v[12:13], v[94:95] /*v[606:607]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[10:11], v[94:95] /*v[606:607]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[8:9], v[94:95] /*v[606:607]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[6:7], v[94:95] /*v[606:607]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[4:5], v[94:95] /*v[606:607]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[2:3], v[2:3], v[94:95] /*v[606:607]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[0:1], v[0:1], v[94:95] /*v[606:607]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x800
	s_branch .LBB0_3
.LBB0_10:
	s_and_b32 vcc_lo, exec_lo, s26
	s_cbranch_vccnz .LBB0_34
	s_branch .LBB0_65
.LBB0_11:
	v_mov_b32_e32 v0, 0
	s_set_vgpr_msb 64
	v_mov_b32_e32 v249 /*v505*/, 1.0
	s_set_vgpr_msb 0x4082
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v85 /*v597*/, v78 /*v590*/ :: v_dual_mov_b32 v92 /*v604*/, v71 /*v583*/
	s_set_vgpr_msb 0x8200
	v_dual_mov_b32 v1, v0 :: v_dual_mov_b32 v2, v0
	v_dual_mov_b32 v3, v0 :: v_dual_mov_b32 v4, v0
	v_dual_mov_b32 v5, v0 :: v_dual_mov_b32 v6, v0
	v_mov_b32_e32 v7, v0
	s_set_vgpr_msb 0x41
	v_mov_b32_e32 v248 /*v504*/, v249 /*v505*/
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
	s_set_vgpr_msb 0x82
	v_mov_b32_e32 v71 /*v583*/, v93 /*v605*/
	s_set_vgpr_msb 0x8200
.LBB0_13:
	s_mov_b32 s51, 0
	s_cmp_le_i32 s7, s11
	s_mov_b32 s83, s51
	s_cbranch_scc1 .LBB0_22
	s_add_co_i32 s2, s19, s35
	s_lshl_b32 s3, s88, 7
	s_sub_co_i32 s2, s2, s18
	s_sub_co_i32 s5, s9, s3
	s_sub_co_i32 s2, s2, s16
	s_add_co_i32 s11, s97, s3
	s_max_i32 s2, s2, 0
	s_mov_b32 s4, 1
	s_lshr_b32 s2, s2, 7
	s_mov_b32 s48, 32
	s_min_u32 s2, s2, s98
	s_mov_b32 s47, 0x800000
	s_sub_co_i32 s6, 0, s2
	s_add_co_i32 s3, s5, 0xffffff80
	s_ashr_i32 s7, s6, 31
	s_add_co_i32 s28, s11, 0x80
	s_add_nc_u64 s[30:31], s[6:7], 1
	s_mov_b32 s45, 0xffff0000
	s_mov_b32 s44, 0x7510000
	s_mov_b32 s52, 0xf510000
	s_mov_b32 s11, 0x76543210
	s_mov_b32 s90, 0x3fb8aa3b
	s_branch .LBB0_16
.LBB0_15:
	s_set_vgpr_msb 0x8a
	v_cvt_pk_bf16_f32 v103 /*v615*/, v20 /*v532*/, v34 /*v546*/
	v_cvt_pk_bf16_f32 v102 /*v614*/, v10 /*v522*/, v18 /*v530*/
	s_set_vgpr_msb 0x8a89
	v_cvt_pk_bf16_f32 v101 /*v613*/, v246 /*v502*/, v2 /*v514*/
	s_set_vgpr_msb 0x8985
	v_cvt_pk_bf16_f32 v100 /*v612*/, v228 /*v484*/, v240 /*v496*/
	v_cvt_pk_bf16_f32 v99 /*v611*/, v212 /*v468*/, v224 /*v480*/
	v_cvt_pk_bf16_f32 v98 /*v610*/, v204 /*v460*/, v210 /*v466*/
	v_cvt_pk_bf16_f32 v97 /*v609*/, v196 /*v452*/, v200 /*v456*/
	v_cvt_pk_bf16_f32 v96 /*v608*/, v192 /*v448*/, v194 /*v450*/
	s_set_vgpr_msb 0x858a
	v_cvt_pk_bf16_f32 v111 /*v623*/, v21 /*v533*/, v35 /*v547*/
	v_cvt_pk_bf16_f32 v110 /*v622*/, v11 /*v523*/, v19 /*v531*/
	s_set_vgpr_msb 0x8a89
	v_cvt_pk_bf16_f32 v109 /*v621*/, v247 /*v503*/, v3 /*v515*/
	s_set_vgpr_msb 0x8985
	v_cvt_pk_bf16_f32 v108 /*v620*/, v229 /*v485*/, v241 /*v497*/
	v_cvt_pk_bf16_f32 v107 /*v619*/, v213 /*v469*/, v225 /*v481*/
	v_cvt_pk_bf16_f32 v106 /*v618*/, v205 /*v461*/, v211 /*v467*/
	v_cvt_pk_bf16_f32 v105 /*v617*/, v197 /*v453*/, v201 /*v457*/
	v_cvt_pk_bf16_f32 v104 /*v616*/, v193 /*v449*/, v195 /*v451*/
	s_set_vgpr_msb 0x8509
	s_wait_dscnt 0x3d
	v_wmma_f32_16x16x32_bf16 v[120:127], v[184:191] /*v[440:447]*/, v[96:103] /*v[608:615]*/, v[120:127]
	s_set_vgpr_msb 0x98a
	v_cvt_pk_bf16_f32 v119 /*v631*/, v39 /*v551*/, v49 /*v561*/
	v_cvt_pk_bf16_f32 v118 /*v630*/, v29 /*v541*/, v37 /*v549*/
	v_cvt_pk_bf16_f32 v117 /*v629*/, v13 /*v525*/, v23 /*v535*/
	s_set_vgpr_msb 0x8a89
	v_cvt_pk_bf16_f32 v116 /*v628*/, v251 /*v507*/, v5 /*v517*/
	s_set_vgpr_msb 0x8985
	v_cvt_pk_bf16_f32 v115 /*v627*/, v231 /*v487*/, v243 /*v499*/
	v_cvt_pk_bf16_f32 v114 /*v626*/, v219 /*v475*/, v227 /*v483*/
	v_cvt_pk_bf16_f32 v113 /*v625*/, v207 /*v463*/, v215 /*v471*/
	s_set_vgpr_msb 0x8509
	v_wmma_f32_16x16x32_bf16 v[56:63], v[184:191] /*v[440:447]*/, v[104:111] /*v[616:623]*/, v[56:63]
	s_set_vgpr_msb 0x985
	v_cvt_pk_bf16_f32 v112 /*v624*/, v199 /*v455*/, v203 /*v459*/
	s_set_vgpr_msb 0x854a
	v_cvt_pk_bf16_f32 v199 /*v455*/, v53 /*v565*/, v59 /*v571*/
	v_cvt_pk_bf16_f32 v197 /*v453*/, v31 /*v543*/, v41 /*v553*/
	v_cvt_pk_bf16_f32 v196 /*v452*/, v15 /*v527*/, v25 /*v537*/
	v_cvt_pk_bf16_f32 v191 /*v447*/, v38 /*v550*/, v48 /*v560*/
	v_cvt_pk_bf16_f32 v190 /*v446*/, v28 /*v540*/, v36 /*v548*/
	v_cvt_pk_bf16_f32 v189 /*v445*/, v12 /*v524*/, v22 /*v534*/
	s_set_vgpr_msb 0x4a09
	s_wait_dscnt 0x3c
	v_wmma_f32_16x16x32_bf16 v[112:119], v[152:159] /*v[408:415]*/, v[96:103] /*v[608:615]*/, v[112:119]
	s_set_vgpr_msb 0x949
	v_cvt_pk_bf16_f32 v188 /*v444*/, v250 /*v506*/, v4 /*v516*/
	s_set_vgpr_msb 0x4945
	v_cvt_pk_bf16_f32 v187 /*v443*/, v230 /*v486*/, v242 /*v498*/
	v_cvt_pk_bf16_f32 v186 /*v442*/, v218 /*v474*/, v226 /*v482*/
	v_cvt_pk_bf16_f32 v185 /*v441*/, v206 /*v462*/, v214 /*v470*/
	v_cvt_pk_bf16_f32 v184 /*v440*/, v198 /*v454*/, v202 /*v458*/
	s_set_vgpr_msb 0x454a
	v_cvt_pk_bf16_f32 v198 /*v454*/, v45 /*v557*/, v51 /*v563*/
	s_set_vgpr_msb 0x4a49
	v_cvt_pk_bf16_f32 v195 /*v451*/, v253 /*v509*/, v7 /*v519*/
	s_set_vgpr_msb 0x4909
	v_wmma_f32_16x16x32_bf16 v[48:55], v[152:159] /*v[408:415]*/, v[104:111] /*v[616:623]*/, v[48:55]
	s_set_vgpr_msb 0x945
	v_cvt_pk_bf16_f32 v194 /*v450*/, v237 /*v493*/, v245 /*v501*/
	v_cvt_pk_bf16_f32 v193 /*v449*/, v221 /*v477*/, v233 /*v489*/
	v_cvt_pk_bf16_f32 v192 /*v448*/, v209 /*v465*/, v217 /*v473*/
	s_set_vgpr_msb 0x454a
	v_cvt_pk_bf16_f32 v207 /*v463*/, v63 /*v575*/, v65 /*v577*/
	v_cvt_pk_bf16_f32 v206 /*v462*/, v57 /*v569*/, v61 /*v573*/
	v_cvt_pk_bf16_f32 v205 /*v461*/, v47 /*v559*/, v55 /*v567*/
	v_cvt_pk_bf16_f32 v204 /*v460*/, v33 /*v545*/, v43 /*v555*/
	s_set_vgpr_msb 0x4a09
	s_wait_dscnt 0x2d
	v_wmma_f32_16x16x32_bf16 v[104:111], v[120:127] /*v[376:383]*/, v[96:103] /*v[608:615]*/, v[104:111]
	s_set_vgpr_msb 0x94a
	v_cvt_pk_bf16_f32 v203 /*v459*/, v17 /*v529*/, v27 /*v539*/
	v_cvt_pk_bf16_f32 v202 /*v458*/, v1 /*v513*/, v9 /*v521*/
	s_set_vgpr_msb 0x4a45
	v_cvt_pk_bf16_f32 v201 /*v457*/, v239 /*v495*/, v255 /*v511*/
	v_cvt_pk_bf16_f32 v200 /*v456*/, v223 /*v479*/, v235 /*v491*/
	s_set_vgpr_msb 0x4586
	v_dual_fmac_f32 v88 /*v600*/, v68 /*v580*/, v248 /*v504*/ :: v_dual_fmac_f32 v69 /*v581*/, v70 /*v582*/, v249 /*v505*/
	v_cmp_lt_u64_e64 s2, s[92:93], s[82:83]
	s_set_vgpr_msb 0x8609
	v_wmma_f32_16x16x32_bf16 v[40:47], v[120:127] /*v[376:383]*/, v[104:111] /*v[616:623]*/, v[40:47]
	s_set_vgpr_msb 0x982
	v_mov_b32_e32 v89 /*v601*/, v94 /*v606*/
	s_set_vgpr_msb 0x824a
	v_dual_add_f32 v248 /*v504*/, v88 /*v600*/, v66 /*v578*/ :: v_dual_add_f32 v249 /*v505*/, v69 /*v581*/, v67 /*v579*/
	s_set_vgpr_msb 0x4a82
	v_dual_mov_b32 v88 /*v600*/, v93 /*v605*/ :: v_dual_mov_b32 v71 /*v583*/, v86 /*v598*/
	v_mov_b32_e32 v92 /*v604*/, v87 /*v599*/
	s_addk_co_i32 s3, 0xff80
	s_set_vgpr_msb 0x8209
	s_wait_dscnt 0x2c
	v_wmma_f32_16x16x32_bf16 v[96:103], v[88:95] /*v[344:351]*/, v[96:103] /*v[608:615]*/, v[96:103]
	s_addk_co_i32 s28, 0x80
	s_and_b32 vcc_lo, exec_lo, s2
	s_mov_b64 s[88:89], s[92:93]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[32:39], v[88:95] /*v[344:351]*/, v[104:111] /*v[616:623]*/, v[32:39]
	v_wmma_f32_16x16x32_bf16 v[88:95], v[56:63] /*v[312:319]*/, v[96:103] /*v[608:615]*/, v[88:95]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[56:63] /*v[312:319]*/, v[104:111] /*v[616:623]*/, v[24:31]
	v_wmma_f32_16x16x32_bf16 v[80:87], v[24:31] /*v[280:287]*/, v[96:103] /*v[608:615]*/, v[80:87]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[24:31] /*v[280:287]*/, v[104:111] /*v[616:623]*/, v[16:23]
	s_set_vgpr_msb 0x908
	v_wmma_f32_16x16x32_bf16 v[72:79], v[248:255], v[96:103] /*v[608:615]*/, v[72:79]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[248:255], v[104:111] /*v[616:623]*/, v[8:15]
	v_wmma_f32_16x16x32_bf16 v[64:71], v[216:223], v[96:103] /*v[608:615]*/, v[64:71]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[216:223], v[104:111] /*v[616:623]*/, v[0:7]
	s_set_vgpr_msb 0x805
	v_wmma_f32_16x16x32_bf16 v[120:127], v[176:183] /*v[432:439]*/, v[184:191] /*v[440:447]*/, v[120:127]
	s_set_vgpr_msb 0x509
	v_wmma_f32_16x16x32_bf16 v[56:63], v[176:183] /*v[432:439]*/, v[112:119] /*v[624:631]*/, v[56:63]
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x94a
	v_cvt_pk_bf16_f32 v183 /*v439*/, v52 /*v564*/, v58 /*v570*/
	v_cvt_pk_bf16_f32 v182 /*v438*/, v44 /*v556*/, v50 /*v562*/
	v_cvt_pk_bf16_f32 v181 /*v437*/, v30 /*v542*/, v40 /*v552*/
	s_set_vgpr_msb 0x4a05
	v_wmma_f32_16x16x32_bf16 v[112:119], v[144:151] /*v[400:407]*/, v[184:191] /*v[440:447]*/, v[112:119]
	s_set_vgpr_msb 0x54a
	v_cvt_pk_bf16_f32 v180 /*v436*/, v14 /*v526*/, v24 /*v536*/
	s_set_vgpr_msb 0x4a49
	v_cvt_pk_bf16_f32 v179 /*v435*/, v252 /*v508*/, v6 /*v518*/
	s_set_vgpr_msb 0x4945
	v_cvt_pk_bf16_f32 v178 /*v434*/, v236 /*v492*/, v244 /*v500*/
	v_cvt_pk_bf16_f32 v177 /*v433*/, v220 /*v476*/, v232 /*v488*/
	v_cvt_pk_bf16_f32 v176 /*v432*/, v208 /*v464*/, v216 /*v472*/
	s_set_vgpr_msb 0x4509
	v_wmma_f32_16x16x32_bf16 v[48:55], v[144:151] /*v[400:407]*/, v[112:119] /*v[624:631]*/, v[48:55]
	s_set_vgpr_msb 0x905
	v_wmma_f32_16x16x32_bf16 v[104:111], v[112:119] /*v[368:375]*/, v[184:191] /*v[440:447]*/, v[104:111]
	s_set_vgpr_msb 0x509
	v_wmma_f32_16x16x32_bf16 v[40:47], v[112:119] /*v[368:375]*/, v[112:119] /*v[624:631]*/, v[40:47]
	s_set_vgpr_msb 0x905
	v_wmma_f32_16x16x32_bf16 v[96:103], v[80:87] /*v[336:343]*/, v[184:191] /*v[440:447]*/, v[96:103]
	s_set_vgpr_msb 0x509
	v_wmma_f32_16x16x32_bf16 v[32:39], v[80:87] /*v[336:343]*/, v[112:119] /*v[624:631]*/, v[32:39]
	s_set_vgpr_msb 0x905
	v_wmma_f32_16x16x32_bf16 v[88:95], v[48:55] /*v[304:311]*/, v[184:191] /*v[440:447]*/, v[88:95]
	s_set_vgpr_msb 0x509
	v_wmma_f32_16x16x32_bf16 v[24:31], v[48:55] /*v[304:311]*/, v[112:119] /*v[624:631]*/, v[24:31]
	s_set_vgpr_msb 0x905
	v_wmma_f32_16x16x32_bf16 v[80:87], v[16:23] /*v[272:279]*/, v[184:191] /*v[440:447]*/, v[80:87]
	s_set_vgpr_msb 0x509
	v_wmma_f32_16x16x32_bf16 v[16:23], v[16:23] /*v[272:279]*/, v[112:119] /*v[624:631]*/, v[16:23]
	s_set_vgpr_msb 0x904
	v_wmma_f32_16x16x32_bf16 v[72:79], v[240:247], v[184:191] /*v[440:447]*/, v[72:79]
	s_set_vgpr_msb 0x408
	v_wmma_f32_16x16x32_bf16 v[8:15], v[240:247], v[112:119] /*v[624:631]*/, v[8:15]
	s_set_vgpr_msb 0x804
	v_wmma_f32_16x16x32_bf16 v[64:71], v[208:215], v[184:191] /*v[440:447]*/, v[64:71]
	s_set_vgpr_msb 0x408
	v_wmma_f32_16x16x32_bf16 v[0:7], v[208:215], v[112:119] /*v[624:631]*/, v[0:7]
	s_set_vgpr_msb 0x805
	v_wmma_f32_16x16x32_bf16 v[120:127], v[168:175] /*v[424:431]*/, v[176:183] /*v[432:439]*/, v[120:127]
	v_wmma_f32_16x16x32_bf16 v[56:63], v[168:175] /*v[424:431]*/, v[192:199] /*v[448:455]*/, v[56:63]
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x54a
	v_cvt_pk_bf16_f32 v175 /*v431*/, v62 /*v574*/, v64 /*v576*/
	v_cvt_pk_bf16_f32 v174 /*v430*/, v56 /*v568*/, v60 /*v572*/
	v_cvt_pk_bf16_f32 v173 /*v429*/, v46 /*v558*/, v54 /*v566*/
	s_set_vgpr_msb 0x4a05
	v_wmma_f32_16x16x32_bf16 v[112:119], v[136:143] /*v[392:399]*/, v[176:183] /*v[432:439]*/, v[112:119]
	s_set_vgpr_msb 0x54a
	v_cvt_pk_bf16_f32 v172 /*v428*/, v32 /*v544*/, v42 /*v554*/
	v_cvt_pk_bf16_f32 v171 /*v427*/, v16 /*v528*/, v26 /*v538*/
	v_cvt_pk_bf16_f32 v170 /*v426*/, v0 /*v512*/, v8 /*v520*/
	s_set_vgpr_msb 0x4a45
	v_cvt_pk_bf16_f32 v169 /*v425*/, v238 /*v494*/, v254 /*v510*/
	v_cvt_pk_bf16_f32 v168 /*v424*/, v222 /*v478*/, v234 /*v490*/
	s_set_vgpr_msb 0x4505
	v_wmma_f32_16x16x32_bf16 v[48:55], v[136:143] /*v[392:399]*/, v[192:199] /*v[448:455]*/, v[48:55]
	v_wmma_f32_16x16x32_bf16 v[104:111], v[104:111] /*v[360:367]*/, v[176:183] /*v[432:439]*/, v[104:111]
	v_wmma_f32_16x16x32_bf16 v[40:47], v[104:111] /*v[360:367]*/, v[192:199] /*v[448:455]*/, v[40:47]
	v_wmma_f32_16x16x32_bf16 v[96:103], v[72:79] /*v[328:335]*/, v[176:183] /*v[432:439]*/, v[96:103]
	v_wmma_f32_16x16x32_bf16 v[32:39], v[72:79] /*v[328:335]*/, v[192:199] /*v[448:455]*/, v[32:39]
	v_wmma_f32_16x16x32_bf16 v[88:95], v[40:47] /*v[296:303]*/, v[176:183] /*v[432:439]*/, v[88:95]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[40:47] /*v[296:303]*/, v[192:199] /*v[448:455]*/, v[24:31]
	v_wmma_f32_16x16x32_bf16 v[80:87], v[8:15] /*v[264:271]*/, v[176:183] /*v[432:439]*/, v[80:87]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[8:15] /*v[264:271]*/, v[192:199] /*v[448:455]*/, v[16:23]
	s_set_vgpr_msb 0x504
	v_wmma_f32_16x16x32_bf16 v[72:79], v[232:239], v[176:183] /*v[432:439]*/, v[72:79]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[232:239], v[192:199] /*v[448:455]*/, v[8:15]
	v_wmma_f32_16x16x32_bf16 v[64:71], v[200:207], v[176:183] /*v[432:439]*/, v[64:71]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[200:207], v[192:199] /*v[448:455]*/, v[0:7]
	s_set_vgpr_msb 0x405
	v_wmma_f32_16x16x32_bf16 v[120:127], v[160:167] /*v[416:423]*/, v[168:175] /*v[424:431]*/, v[120:127]
	v_wmma_f32_16x16x32_bf16 v[56:63], v[160:167] /*v[416:423]*/, v[200:207] /*v[456:463]*/, v[56:63]
	v_wmma_f32_16x16x32_bf16 v[112:119], v[128:135] /*v[384:391]*/, v[168:175] /*v[424:431]*/, v[112:119]
	v_wmma_f32_16x16x32_bf16 v[48:55], v[128:135] /*v[384:391]*/, v[200:207] /*v[456:463]*/, v[48:55]
	v_wmma_f32_16x16x32_bf16 v[104:111], v[96:103] /*v[352:359]*/, v[168:175] /*v[424:431]*/, v[104:111]
	v_wmma_f32_16x16x32_bf16 v[40:47], v[96:103] /*v[352:359]*/, v[200:207] /*v[456:463]*/, v[40:47]
	v_wmma_f32_16x16x32_bf16 v[96:103], v[64:71] /*v[320:327]*/, v[168:175] /*v[424:431]*/, v[96:103]
	v_wmma_f32_16x16x32_bf16 v[32:39], v[64:71] /*v[320:327]*/, v[200:207] /*v[456:463]*/, v[32:39]
	v_wmma_f32_16x16x32_bf16 v[88:95], v[32:39] /*v[288:295]*/, v[168:175] /*v[424:431]*/, v[88:95]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[32:39] /*v[288:295]*/, v[200:207] /*v[456:463]*/, v[24:31]
	v_wmma_f32_16x16x32_bf16 v[80:87], v[0:7] /*v[256:263]*/, v[168:175] /*v[424:431]*/, v[80:87]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[0:7] /*v[256:263]*/, v[200:207] /*v[456:463]*/, v[16:23]
	s_set_vgpr_msb 0x504
	v_wmma_f32_16x16x32_bf16 v[72:79], v[224:231], v[168:175] /*v[424:431]*/, v[72:79]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[224:231], v[200:207] /*v[456:463]*/, v[8:15]
	v_wmma_f32_16x16x32_bf16 v[64:71], v[192:199], v[168:175] /*v[424:431]*/, v[64:71]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[192:199], v[200:207] /*v[456:463]*/, v[0:7]
	s_set_vgpr_msb 0x400
	s_cbranch_vccz .LBB0_23
.LBB0_16:
	s_set_vgpr_msb 0x82
	v_dual_mov_b32 v93 /*v605*/, v85 /*v597*/ :: v_dual_mov_b32 v85 /*v597*/, v88 /*v600*/
	v_dual_mov_b32 v94 /*v606*/, v84 /*v596*/ :: v_dual_mov_b32 v84 /*v596*/, v89 /*v601*/
	s_add_nc_u64 s[92:93], s[88:89], 1
	s_wait_tensorcnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	s_cmp_ge_i32 s92, s10
	s_set_vgpr_msb 0x8200
	s_cbranch_scc1 .LBB0_18
	s_ashr_i32 s29, s28, 31
	v_nop
	v_nop
	v_med3_i32 v192, s3, 0, 0x80
	s_mul_u64 s[6:7], s[28:29], s[64:65]
	s_add_co_i32 s2, s30, s88
	s_lshl_b64 s[6:7], s[6:7], 1
	s_lshr_b32 s5, s2, 31
	s_add_nc_u64 s[54:55], s[84:85], s[6:7]
	s_mul_u64 s[6:7], s[28:29], s[62:63]
	v_readfirstlane_b32 s29, v192
	s_add_co_i32 s5, s2, s5
	s_lshl_b64 s[6:7], s[6:7], 1
	s_and_b32 s5, s5, 0x7ffe
	s_add_nc_u64 s[6:7], s[86:87], s[6:7]
	s_sub_co_i32 s2, s2, s5
	s_sub_co_i32 s5, s29, s25
	s_lshl_b32 s2, s2, 17
	s_max_i32 s29, s5, 0
	s_add_nc_u64 s[6:7], s[26:27], s[6:7]
	s_lshl_b32 s29, s29, 16
	s_or_b32 s5, s104, s2
	s_bitset1_b32 s7, 31
	s_or_b32 s46, s29, 0x7fff
	s_mov_b32 s53, s45
	tensor_load_to_lds s[4:7], s[44:51]
	s_add_nc_u64 s[6:7], s[80:81], s[54:55]
	s_or_b32 s5, vcc_hi, s2
	s_bitset1_b32 s7, 31
	s_mov_b32 s54, s46
	s_mov_b32 s55, s47
	s_mov_b32 s56, s48
	s_mov_b32 s59, s51
	tensor_load_to_lds s[4:7], s[52:59]
.LBB0_18:
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 2
	ds_load_b128 v[192:195], v93 /*v605*/
	ds_load_b128 v[196:199], v93 /*v605*/ offset:32
	ds_load_b128 v[200:203], v93 /*v605*/ offset:64
	ds_load_b128 v[204:207], v93 /*v605*/ offset:96
	ds_load_b128 v[208:211], v93 /*v605*/ offset:128
	ds_load_b128 v[212:215], v93 /*v605*/ offset:160
	ds_load_b128 v[216:219], v93 /*v605*/ offset:192
	ds_load_b128 v[220:223], v93 /*v605*/ offset:224
	ds_load_b128 v[224:227], v93 /*v605*/ offset:4352
	ds_load_b128 v[228:231], v93 /*v605*/ offset:4384
	ds_load_b128 v[232:235], v93 /*v605*/ offset:4416
	ds_load_b128 v[236:239], v93 /*v605*/ offset:4448
	ds_load_b128 v[240:243], v93 /*v605*/ offset:4480
	ds_load_b128 v[244:247], v93 /*v605*/ offset:4512
	ds_load_b128 v[248:251], v93 /*v605*/ offset:4544
	ds_load_b128 v[252:255], v93 /*v605*/ offset:4576
	s_set_vgpr_msb 0x242
	ds_load_b128 v[0:3] /*v[256:259]*/, v93 /*v605*/ offset:8704
	ds_load_b128 v[4:7] /*v[260:263]*/, v93 /*v605*/ offset:8736
	ds_load_b128 v[8:11] /*v[264:267]*/, v93 /*v605*/ offset:8768
	ds_load_b128 v[12:15] /*v[268:271]*/, v93 /*v605*/ offset:8800
	ds_load_b128 v[16:19] /*v[272:275]*/, v93 /*v605*/ offset:8832
	ds_load_b128 v[20:23] /*v[276:279]*/, v93 /*v605*/ offset:8864
	ds_load_b128 v[24:27] /*v[280:283]*/, v93 /*v605*/ offset:8896
	ds_load_b128 v[28:31] /*v[284:287]*/, v93 /*v605*/ offset:8928
	ds_load_b128 v[32:35] /*v[288:291]*/, v93 /*v605*/ offset:13056
	ds_load_b128 v[36:39] /*v[292:295]*/, v93 /*v605*/ offset:13088
	ds_load_b128 v[40:43] /*v[296:299]*/, v93 /*v605*/ offset:13120
	ds_load_b128 v[44:47] /*v[300:303]*/, v93 /*v605*/ offset:13152
	ds_load_b128 v[48:51] /*v[304:307]*/, v93 /*v605*/ offset:13184
	ds_load_b128 v[52:55] /*v[308:311]*/, v93 /*v605*/ offset:13216
	ds_load_b128 v[56:59] /*v[312:315]*/, v93 /*v605*/ offset:13248
	ds_load_b128 v[60:63] /*v[316:319]*/, v93 /*v605*/ offset:13280
	ds_load_b128 v[64:67] /*v[320:323]*/, v93 /*v605*/ offset:17408
	ds_load_b128 v[68:71] /*v[324:327]*/, v93 /*v605*/ offset:17440
	ds_load_b128 v[72:75] /*v[328:331]*/, v93 /*v605*/ offset:17472
	ds_load_b128 v[76:79] /*v[332:335]*/, v93 /*v605*/ offset:17504
	ds_load_b128 v[80:83] /*v[336:339]*/, v93 /*v605*/ offset:17536
	ds_load_b128 v[84:87] /*v[340:343]*/, v93 /*v605*/ offset:17568
	ds_load_b128 v[88:91] /*v[344:347]*/, v93 /*v605*/ offset:17600
	ds_load_b128 v[92:95] /*v[348:351]*/, v93 /*v605*/ offset:17632
	ds_load_b128 v[96:99] /*v[352:355]*/, v93 /*v605*/ offset:21760
	ds_load_b128 v[100:103] /*v[356:359]*/, v93 /*v605*/ offset:21792
	ds_load_b128 v[104:107] /*v[360:363]*/, v93 /*v605*/ offset:21824
	ds_load_b128 v[108:111] /*v[364:367]*/, v93 /*v605*/ offset:21856
	ds_load_b128 v[112:115] /*v[368:371]*/, v93 /*v605*/ offset:21888
	ds_load_b128 v[116:119] /*v[372:375]*/, v93 /*v605*/ offset:21920
	ds_load_b128 v[120:123] /*v[376:379]*/, v93 /*v605*/ offset:21952
	ds_load_b128 v[124:127] /*v[380:383]*/, v93 /*v605*/ offset:21984
	ds_load_b128 v[128:131] /*v[384:387]*/, v93 /*v605*/ offset:26112
	ds_load_b128 v[132:135] /*v[388:391]*/, v93 /*v605*/ offset:26144
	ds_load_b128 v[136:139] /*v[392:395]*/, v93 /*v605*/ offset:26176
	ds_load_b128 v[140:143] /*v[396:399]*/, v93 /*v605*/ offset:26208
	ds_load_b128 v[144:147] /*v[400:403]*/, v93 /*v605*/ offset:26240
	ds_load_b128 v[148:151] /*v[404:407]*/, v93 /*v605*/ offset:26272
	ds_load_b128 v[152:155] /*v[408:411]*/, v93 /*v605*/ offset:26304
	ds_load_b128 v[156:159] /*v[412:415]*/, v93 /*v605*/ offset:26336
	ds_load_b128 v[160:163] /*v[416:419]*/, v93 /*v605*/ offset:30464
	ds_load_b128 v[164:167] /*v[420:423]*/, v93 /*v605*/ offset:30496
	ds_load_b128 v[168:171] /*v[424:427]*/, v93 /*v605*/ offset:30528
	ds_load_b128 v[172:175] /*v[428:431]*/, v93 /*v605*/ offset:30560
	ds_load_b128 v[176:179] /*v[432:435]*/, v93 /*v605*/ offset:30592
	ds_load_b128 v[180:183] /*v[436:439]*/, v93 /*v605*/ offset:30624
	ds_load_b128 v[184:187] /*v[440:443]*/, v93 /*v605*/ offset:30656
	ds_load_b128 v[188:191] /*v[444:447]*/, v93 /*v605*/ offset:30688
	s_set_vgpr_msb 0x4240
	s_wait_dscnt 0x3e
	v_wmma_f32_16x16x32_bf16 v[192:199] /*v[448:455]*/, v[192:199], v[128:135], 0
	s_set_vgpr_msb 0x4080
	v_wmma_f32_16x16x32_bf16 v[62:69] /*v[574:581]*/, v[192:199], v[160:167], 0
	s_set_vgpr_msb 0x8040
	s_wait_dscnt 0x36
	v_wmma_f32_16x16x32_bf16 v[212:219] /*v[468:475]*/, v[224:231], v[128:135], 0
	s_set_vgpr_msb 0x4041
	s_wait_dscnt 0x2e
	v_wmma_f32_16x16x32_bf16 v[230:237] /*v[486:493]*/, v[0:7] /*v[256:263]*/, v[128:135], 0
	s_wait_dscnt 0x26
	v_wmma_f32_16x16x32_bf16 v[250:257] /*v[506:513]*/, v[32:39] /*v[288:295]*/, v[128:135], 0
	s_set_vgpr_msb 0x4181
	s_wait_dscnt 0x1e
	v_wmma_f32_16x16x32_bf16 v[38:45] /*v[550:557]*/, v[64:71] /*v[320:327]*/, v[128:135], 0
	s_wait_dscnt 0x16
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[562:569]*/, v[96:103] /*v[352:359]*/, v[128:135], 0
	s_set_vgpr_msb 0x8150
	v_wmma_f32_16x16x32_bf16 v[192:199] /*v[448:455]*/, v[200:207], v[136:143], v[192:199] /*v[448:455]*/
	s_set_vgpr_msb 0x50a0
	v_wmma_f32_16x16x32_bf16 v[62:69] /*v[574:581]*/, v[200:207], v[168:175], v[62:69] /*v[574:581]*/
	v_wmma_f32_16x16x32_bf16 v[96:103] /*v[608:615]*/, v[224:231], v[160:167], 0
	s_set_vgpr_msb 0xa050
	v_wmma_f32_16x16x32_bf16 v[212:219] /*v[468:475]*/, v[232:239], v[136:143], v[212:219] /*v[468:475]*/
	s_set_vgpr_msb 0x5081
	v_wmma_f32_16x16x32_bf16 v[104:111] /*v[616:623]*/, v[0:7] /*v[256:263]*/, v[160:167], 0
	s_set_vgpr_msb 0x8151
	v_wmma_f32_16x16x32_bf16 v[230:237] /*v[486:493]*/, v[8:15] /*v[264:271]*/, v[136:143], v[230:237] /*v[486:493]*/
	s_set_vgpr_msb 0x5181
	v_wmma_f32_16x16x32_bf16 v[112:119] /*v[624:631]*/, v[32:39] /*v[288:295]*/, v[160:167], 0
	s_set_vgpr_msb 0x8151
	v_wmma_f32_16x16x32_bf16 v[250:257] /*v[506:513]*/, v[40:47] /*v[296:303]*/, v[136:143], v[250:257] /*v[506:513]*/
	s_set_vgpr_msb 0x51a1
	v_wmma_f32_16x16x32_bf16 v[120:127] /*v[632:639]*/, v[64:71] /*v[320:327]*/, v[160:167], 0
	v_wmma_f32_16x16x32_bf16 v[38:45] /*v[550:557]*/, v[72:79] /*v[328:335]*/, v[136:143], v[38:45] /*v[550:557]*/
	v_wmma_f32_16x16x32_bf16 v[128:135] /*v[640:647]*/, v[96:103] /*v[352:359]*/, v[160:167], 0
	s_wait_dscnt 0x14
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[562:569]*/, v[104:111] /*v[360:367]*/, v[136:143], v[50:57] /*v[562:569]*/
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x32_bf16 v[136:143] /*v[648:655]*/, v[128:135] /*v[384:391]*/, v[128:135], 0
	v_wmma_f32_16x16x32_bf16 v[144:151] /*v[656:663]*/, v[128:135] /*v[384:391]*/, v[160:167], 0
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x32_bf16 v[152:159] /*v[664:671]*/, v[160:167] /*v[416:423]*/, v[128:135], 0
	v_wmma_f32_16x16x32_bf16 v[160:167] /*v[672:679]*/, v[160:167] /*v[416:423]*/, v[160:167], 0
	s_set_vgpr_msb 0xa150
	v_wmma_f32_16x16x32_bf16 v[192:199] /*v[448:455]*/, v[208:215], v[144:151], v[192:199] /*v[448:455]*/
	s_set_vgpr_msb 0x50a0
	v_wmma_f32_16x16x32_bf16 v[62:69] /*v[574:581]*/, v[208:215], v[176:183], v[62:69] /*v[574:581]*/
	v_wmma_f32_16x16x32_bf16 v[96:103] /*v[608:615]*/, v[232:239], v[168:175], v[96:103] /*v[608:615]*/
	s_set_vgpr_msb 0xa050
	v_wmma_f32_16x16x32_bf16 v[212:219] /*v[468:475]*/, v[240:247], v[144:151], v[212:219] /*v[468:475]*/
	s_set_vgpr_msb 0x50a1
	v_wmma_f32_16x16x32_bf16 v[104:111] /*v[616:623]*/, v[8:15] /*v[264:271]*/, v[168:175], v[104:111] /*v[616:623]*/
	s_set_vgpr_msb 0xa151
	v_wmma_f32_16x16x32_bf16 v[230:237] /*v[486:493]*/, v[16:23] /*v[272:279]*/, v[144:151], v[230:237] /*v[486:493]*/
	s_set_vgpr_msb 0x51a1
	v_wmma_f32_16x16x32_bf16 v[112:119] /*v[624:631]*/, v[40:47] /*v[296:303]*/, v[168:175], v[112:119] /*v[624:631]*/
	s_set_vgpr_msb 0xa151
	v_wmma_f32_16x16x32_bf16 v[250:257] /*v[506:513]*/, v[48:55] /*v[304:311]*/, v[144:151], v[250:257] /*v[506:513]*/
	s_set_vgpr_msb 0x51a1
	v_wmma_f32_16x16x32_bf16 v[120:127] /*v[632:639]*/, v[72:79] /*v[328:335]*/, v[168:175], v[120:127] /*v[632:639]*/
	v_wmma_f32_16x16x32_bf16 v[38:45] /*v[550:557]*/, v[80:87] /*v[336:343]*/, v[144:151], v[38:45] /*v[550:557]*/
	v_wmma_f32_16x16x32_bf16 v[128:135] /*v[640:647]*/, v[104:111] /*v[360:367]*/, v[168:175], v[128:135] /*v[640:647]*/
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[562:569]*/, v[112:119] /*v[368:375]*/, v[144:151], v[50:57] /*v[562:569]*/
	v_wmma_f32_16x16x32_bf16 v[136:143] /*v[648:655]*/, v[136:143] /*v[392:399]*/, v[136:143], v[136:143] /*v[648:655]*/
	v_wmma_f32_16x16x32_bf16 v[144:151] /*v[656:663]*/, v[136:143] /*v[392:399]*/, v[168:175], v[144:151] /*v[656:663]*/
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x32_bf16 v[152:159] /*v[664:671]*/, v[168:175] /*v[424:431]*/, v[136:143], v[152:159] /*v[664:671]*/
	v_wmma_f32_16x16x32_bf16 v[160:167] /*v[672:679]*/, v[168:175] /*v[424:431]*/, v[168:175], v[160:167] /*v[672:679]*/
	s_set_vgpr_msb 0xa150
	v_wmma_f32_16x16x32_bf16 v[192:199] /*v[448:455]*/, v[216:223], v[152:159], v[192:199] /*v[448:455]*/
	s_set_vgpr_msb 0x50a0
	v_wmma_f32_16x16x32_bf16 v[62:69] /*v[574:581]*/, v[216:223], v[184:191], v[62:69] /*v[574:581]*/
	v_wmma_f32_16x16x32_bf16 v[96:103] /*v[608:615]*/, v[240:247], v[176:183], v[96:103] /*v[608:615]*/
	s_set_vgpr_msb 0xa050
	v_wmma_f32_16x16x32_bf16 v[212:219] /*v[468:475]*/, v[248:255], v[152:159], v[212:219] /*v[468:475]*/
	s_set_vgpr_msb 0x50a1
	v_wmma_f32_16x16x32_bf16 v[104:111] /*v[616:623]*/, v[16:23] /*v[272:279]*/, v[176:183], v[104:111] /*v[616:623]*/
	s_set_vgpr_msb 0xa151
	v_wmma_f32_16x16x32_bf16 v[230:237] /*v[486:493]*/, v[24:31] /*v[280:287]*/, v[152:159], v[230:237] /*v[486:493]*/
	s_set_vgpr_msb 0x51a1
	v_wmma_f32_16x16x32_bf16 v[112:119] /*v[624:631]*/, v[48:55] /*v[304:311]*/, v[176:183], v[112:119] /*v[624:631]*/
	s_set_vgpr_msb 0xa151
	v_wmma_f32_16x16x32_bf16 v[250:257] /*v[506:513]*/, v[56:63] /*v[312:319]*/, v[152:159], v[250:257] /*v[506:513]*/
	s_set_vgpr_msb 0x51a1
	v_wmma_f32_16x16x32_bf16 v[120:127] /*v[632:639]*/, v[80:87] /*v[336:343]*/, v[176:183], v[120:127] /*v[632:639]*/
	v_wmma_f32_16x16x32_bf16 v[38:45] /*v[550:557]*/, v[88:95] /*v[344:351]*/, v[152:159], v[38:45] /*v[550:557]*/
	v_wmma_f32_16x16x32_bf16 v[128:135] /*v[640:647]*/, v[112:119] /*v[368:375]*/, v[176:183], v[128:135] /*v[640:647]*/
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[562:569]*/, v[120:127] /*v[376:383]*/, v[152:159], v[50:57] /*v[562:569]*/
	v_wmma_f32_16x16x32_bf16 v[136:143] /*v[648:655]*/, v[144:151] /*v[400:407]*/, v[144:151], v[136:143] /*v[648:655]*/
	v_wmma_f32_16x16x32_bf16 v[144:151] /*v[656:663]*/, v[144:151] /*v[400:407]*/, v[176:183], v[144:151] /*v[656:663]*/
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[152:159] /*v[664:671]*/, v[176:183] /*v[432:439]*/, v[144:151], v[152:159] /*v[664:671]*/
	v_wmma_f32_16x16x32_bf16 v[160:167] /*v[672:679]*/, v[176:183] /*v[432:439]*/, v[176:183], v[160:167] /*v[672:679]*/
	s_set_vgpr_msb 0xa1a0
	v_wmma_f32_16x16x32_bf16 v[96:103] /*v[608:615]*/, v[248:255], v[184:191], v[96:103] /*v[608:615]*/
	s_set_vgpr_msb 0xa0a1
	v_wmma_f32_16x16x32_bf16 v[104:111] /*v[616:623]*/, v[24:31] /*v[280:287]*/, v[184:191], v[104:111] /*v[616:623]*/
	v_wmma_f32_16x16x32_bf16 v[112:119] /*v[624:631]*/, v[56:63] /*v[312:319]*/, v[184:191], v[112:119] /*v[624:631]*/
	v_wmma_f32_16x16x32_bf16 v[120:127] /*v[632:639]*/, v[88:95] /*v[344:351]*/, v[184:191], v[120:127] /*v[632:639]*/
	v_wmma_f32_16x16x32_bf16 v[128:135] /*v[640:647]*/, v[120:127] /*v[376:383]*/, v[184:191], v[128:135] /*v[640:647]*/
	v_wmma_f32_16x16x32_bf16 v[136:143] /*v[648:655]*/, v[152:159] /*v[408:415]*/, v[152:159], v[136:143] /*v[648:655]*/
	v_wmma_f32_16x16x32_bf16 v[144:151] /*v[656:663]*/, v[152:159] /*v[408:415]*/, v[184:191], v[144:151] /*v[656:663]*/
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[152:159] /*v[664:671]*/, v[184:191] /*v[440:447]*/, v[152:159], v[152:159] /*v[664:671]*/
	v_wmma_f32_16x16x32_bf16 v[160:167] /*v[672:679]*/, v[184:191] /*v[440:447]*/, v[184:191], v[160:167] /*v[672:679]*/
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0xa142
	ds_load_tr16_b128 v[184:187] /*v[440:443]*/, v94 /*v606*/
	ds_load_tr16_b128 v[152:155] /*v[408:411]*/, v94 /*v606*/ offset:32
	ds_load_tr16_b128 v[188:191] /*v[444:447]*/, v94 /*v606*/ offset:4608
	ds_load_tr16_b128 v[156:159] /*v[412:415]*/, v94 /*v606*/ offset:4640
	ds_load_tr16_b128 v[176:179] /*v[432:435]*/, v94 /*v606*/ offset:9216
	ds_load_tr16_b128 v[144:147] /*v[400:403]*/, v94 /*v606*/ offset:9248
	ds_load_tr16_b128 v[180:183] /*v[436:439]*/, v94 /*v606*/ offset:13824
	ds_load_tr16_b128 v[148:151] /*v[404:407]*/, v94 /*v606*/ offset:13856
	ds_load_tr16_b128 v[168:171] /*v[424:427]*/, v94 /*v606*/ offset:18432
	ds_load_tr16_b128 v[136:139] /*v[392:395]*/, v94 /*v606*/ offset:18464
	ds_load_tr16_b128 v[172:175] /*v[428:431]*/, v94 /*v606*/ offset:23040
	ds_load_tr16_b128 v[140:143] /*v[396:399]*/, v94 /*v606*/ offset:23072
	ds_load_tr16_b128 v[160:163] /*v[416:419]*/, v94 /*v606*/ offset:27648
	ds_load_tr16_b128 v[128:131] /*v[384:387]*/, v94 /*v606*/ offset:27680
	ds_load_tr16_b128 v[164:167] /*v[420:423]*/, v94 /*v606*/ offset:32256
	ds_load_tr16_b128 v[132:135] /*v[388:391]*/, v94 /*v606*/ offset:32288
	ds_load_tr16_b128 v[120:123] /*v[376:379]*/, v94 /*v606*/ offset:64
	ds_load_tr16_b128 v[88:91] /*v[344:347]*/, v94 /*v606*/ offset:96
	ds_load_tr16_b128 v[124:127] /*v[380:383]*/, v94 /*v606*/ offset:4672
	ds_load_tr16_b128 v[92:95] /*v[348:351]*/, v94 /*v606*/ offset:4704
	ds_load_tr16_b128 v[112:115] /*v[368:371]*/, v94 /*v606*/ offset:9280
	ds_load_tr16_b128 v[80:83] /*v[336:339]*/, v94 /*v606*/ offset:9312
	ds_load_tr16_b128 v[116:119] /*v[372:375]*/, v94 /*v606*/ offset:13888
	ds_load_tr16_b128 v[84:87] /*v[340:343]*/, v94 /*v606*/ offset:13920
	ds_load_tr16_b128 v[104:107] /*v[360:363]*/, v94 /*v606*/ offset:18496
	ds_load_tr16_b128 v[72:75] /*v[328:331]*/, v94 /*v606*/ offset:18528
	ds_load_tr16_b128 v[108:111] /*v[364:367]*/, v94 /*v606*/ offset:23104
	ds_load_tr16_b128 v[76:79] /*v[332:335]*/, v94 /*v606*/ offset:23136
	ds_load_tr16_b128 v[96:99] /*v[352:355]*/, v94 /*v606*/ offset:27712
	ds_load_tr16_b128 v[64:67] /*v[320:323]*/, v94 /*v606*/ offset:27744
	ds_load_tr16_b128 v[100:103] /*v[356:359]*/, v94 /*v606*/ offset:32320
	ds_load_tr16_b128 v[68:71] /*v[324:327]*/, v94 /*v606*/ offset:32352
	ds_load_tr16_b128 v[56:59] /*v[312:315]*/, v94 /*v606*/ offset:128
	ds_load_tr16_b128 v[24:27] /*v[280:283]*/, v94 /*v606*/ offset:160
	ds_load_tr16_b128 v[60:63] /*v[316:319]*/, v94 /*v606*/ offset:4736
	ds_load_tr16_b128 v[28:31] /*v[284:287]*/, v94 /*v606*/ offset:4768
	ds_load_tr16_b128 v[48:51] /*v[304:307]*/, v94 /*v606*/ offset:9344
	ds_load_tr16_b128 v[16:19] /*v[272:275]*/, v94 /*v606*/ offset:9376
	ds_load_tr16_b128 v[52:55] /*v[308:311]*/, v94 /*v606*/ offset:13952
	ds_load_tr16_b128 v[20:23] /*v[276:279]*/, v94 /*v606*/ offset:13984
	ds_load_tr16_b128 v[40:43] /*v[296:299]*/, v94 /*v606*/ offset:18560
	ds_load_tr16_b128 v[8:11] /*v[264:267]*/, v94 /*v606*/ offset:18592
	ds_load_tr16_b128 v[44:47] /*v[300:303]*/, v94 /*v606*/ offset:23168
	ds_load_tr16_b128 v[12:15] /*v[268:271]*/, v94 /*v606*/ offset:23200
	ds_load_tr16_b128 v[32:35] /*v[288:291]*/, v94 /*v606*/ offset:27776
	ds_load_tr16_b128 v[0:3] /*v[256:259]*/, v94 /*v606*/ offset:27808
	ds_load_tr16_b128 v[36:39] /*v[292:295]*/, v94 /*v606*/ offset:32384
	ds_load_tr16_b128 v[4:7] /*v[260:263]*/, v94 /*v606*/ offset:32416
	s_set_vgpr_msb 0x4202
	ds_load_tr16_b128 v[248:251], v94 /*v606*/ offset:192
	ds_load_tr16_b128 v[216:219], v94 /*v606*/ offset:224
	ds_load_tr16_b128 v[252:255], v94 /*v606*/ offset:4800
	ds_load_tr16_b128 v[220:223], v94 /*v606*/ offset:4832
	ds_load_tr16_b128 v[240:243], v94 /*v606*/ offset:9408
	ds_load_tr16_b128 v[208:211], v94 /*v606*/ offset:9440
	ds_load_tr16_b128 v[244:247], v94 /*v606*/ offset:14016
	ds_load_tr16_b128 v[212:215], v94 /*v606*/ offset:14048
	ds_load_tr16_b128 v[232:235], v94 /*v606*/ offset:18624
	ds_load_tr16_b128 v[200:203], v94 /*v606*/ offset:18656
	ds_load_tr16_b128 v[236:239], v94 /*v606*/ offset:23232
	ds_load_tr16_b128 v[204:207], v94 /*v606*/ offset:23264
	ds_load_tr16_b128 v[224:227], v94 /*v606*/ offset:27840
	ds_load_tr16_b128 v[192:195], v94 /*v606*/ offset:27872
	ds_load_tr16_b128 v[228:231], v94 /*v606*/ offset:32448
	ds_load_tr16_b128 v[196:199], v94 /*v606*/ offset:32480
	s_set_vgpr_msb 0x255
	v_max3_num_f32 v200 /*v456*/, v192 /*v448*/, v193 /*v449*/, v194 /*v450*/
	s_set_vgpr_msb 0x556a
	v_max3_num_f32 v201 /*v457*/, v62 /*v574*/, v63 /*v575*/, v64 /*v576*/
	s_set_vgpr_msb 0x6a55
	v_max3_num_f32 v202 /*v458*/, v195 /*v451*/, v196 /*v452*/, v197 /*v453*/
	s_set_vgpr_msb 0x556a
	v_max3_num_f32 v203 /*v459*/, v65 /*v577*/, v66 /*v578*/, v67 /*v579*/
	s_set_vgpr_msb 0x6a55
	v_max3_num_f32 v204 /*v460*/, v198 /*v454*/, v199 /*v455*/, v212 /*v468*/
	s_set_vgpr_msb 0x556a
	v_max3_num_f32 v205 /*v461*/, v68 /*v580*/, v69 /*v581*/, v96 /*v608*/
	s_set_vgpr_msb 0x6a55
	v_max3_num_f32 v206 /*v462*/, v213 /*v469*/, v214 /*v470*/, v215 /*v471*/
	v_max3_num_f32 v208 /*v464*/, v216 /*v472*/, v217 /*v473*/, v218 /*v474*/
	v_max3_num_f32 v210 /*v466*/, v219 /*v475*/, v230 /*v486*/, v231 /*v487*/
	v_max3_num_f32 v220 /*v476*/, v232 /*v488*/, v233 /*v489*/, v234 /*v490*/
	v_max3_num_f32 v222 /*v478*/, v235 /*v491*/, v236 /*v492*/, v237 /*v493*/
	v_max3_num_f32 v224 /*v480*/, v250 /*v506*/, v251 /*v507*/, v252 /*v508*/
	v_max3_num_f32 v226 /*v482*/, v253 /*v509*/, v254 /*v510*/, v255 /*v511*/
	s_set_vgpr_msb 0x556a
	v_max3_num_f32 v228 /*v484*/, v0 /*v512*/, v1 /*v513*/, v38 /*v550*/
	v_max3_num_f32 v238 /*v494*/, v39 /*v551*/, v40 /*v552*/, v41 /*v553*/
	v_max3_num_f32 v240 /*v496*/, v42 /*v554*/, v43 /*v555*/, v44 /*v556*/
	v_max3_num_f32 v242 /*v498*/, v45 /*v557*/, v50 /*v562*/, v51 /*v563*/
	v_max3_num_f32 v244 /*v500*/, v52 /*v564*/, v53 /*v565*/, v54 /*v566*/
	v_max3_num_f32 v246 /*v502*/, v55 /*v567*/, v56 /*v568*/, v57 /*v569*/
	s_set_vgpr_msb 0x6aaa
	v_max3_num_f32 v2 /*v514*/, v136 /*v648*/, v137 /*v649*/, v138 /*v650*/
	v_max3_num_f32 v4 /*v516*/, v139 /*v651*/, v140 /*v652*/, v141 /*v653*/
	v_max_num_f32_e32 v6 /*v518*/, v142 /*v654*/, v143 /*v655*/
	v_max3_num_f32 v8 /*v520*/, v153 /*v665*/, v154 /*v666*/, v155 /*v667*/
	s_set_vgpr_msb 0xaa6a
	v_max3_num_f32 v207 /*v463*/, v97 /*v609*/, v98 /*v610*/, v99 /*v611*/
	v_max3_num_f32 v209 /*v465*/, v100 /*v612*/, v101 /*v613*/, v102 /*v614*/
	v_max3_num_f32 v211 /*v467*/, v103 /*v615*/, v104 /*v616*/, v105 /*v617*/
	v_max3_num_f32 v221 /*v477*/, v106 /*v618*/, v107 /*v619*/, v108 /*v620*/
	v_max3_num_f32 v223 /*v479*/, v109 /*v621*/, v110 /*v622*/, v111 /*v623*/
	v_max3_num_f32 v225 /*v481*/, v112 /*v624*/, v113 /*v625*/, v114 /*v626*/
	v_max3_num_f32 v227 /*v483*/, v115 /*v627*/, v116 /*v628*/, v117 /*v629*/
	v_max3_num_f32 v229 /*v485*/, v118 /*v630*/, v119 /*v631*/, v120 /*v632*/
	v_max3_num_f32 v239 /*v495*/, v121 /*v633*/, v122 /*v634*/, v123 /*v635*/
	v_max3_num_f32 v241 /*v497*/, v124 /*v636*/, v125 /*v637*/, v126 /*v638*/
	v_max3_num_f32 v243 /*v499*/, v127 /*v639*/, v128 /*v640*/, v129 /*v641*/
	v_max3_num_f32 v245 /*v501*/, v130 /*v642*/, v131 /*v643*/, v132 /*v644*/
	v_max3_num_f32 v247 /*v503*/, v133 /*v645*/, v134 /*v646*/, v135 /*v647*/
	s_set_vgpr_msb 0x6aaa
	v_max3_num_f32 v3 /*v515*/, v144 /*v656*/, v145 /*v657*/, v146 /*v658*/
	v_max3_num_f32 v5 /*v517*/, v147 /*v659*/, v148 /*v660*/, v149 /*v661*/
	v_max_num_f32_e32 v7 /*v519*/, v150 /*v662*/, v151 /*v663*/
	v_max3_num_f32 v9 /*v521*/, v161 /*v673*/, v162 /*v674*/, v163 /*v675*/
	v_max3_num_f32 v10 /*v522*/, v156 /*v668*/, v157 /*v669*/, v158 /*v670*/
	s_set_vgpr_msb 0xaa55
	v_max3_num_f32 v200 /*v456*/, v200 /*v456*/, v202 /*v458*/, v204 /*v460*/
	v_max3_num_f32 v201 /*v457*/, v201 /*v457*/, v203 /*v459*/, v205 /*v461*/
	v_max3_num_f32 v202 /*v458*/, v206 /*v462*/, v208 /*v464*/, v210 /*v466*/
	v_max3_num_f32 v203 /*v459*/, v220 /*v476*/, v222 /*v478*/, v224 /*v480*/
	v_max3_num_f32 v204 /*v460*/, v226 /*v482*/, v228 /*v484*/, v238 /*v494*/
	v_max3_num_f32 v205 /*v461*/, v240 /*v496*/, v242 /*v498*/, v244 /*v500*/
	s_set_vgpr_msb 0x5569
	v_max3_num_f32 v206 /*v462*/, v246 /*v502*/, v2 /*v514*/, v4 /*v516*/
	s_set_vgpr_msb 0x696a
	v_max3_num_f32 v208 /*v464*/, v6 /*v518*/, v152 /*v664*/, v8 /*v520*/
	s_set_vgpr_msb 0x6aaa
	v_max3_num_f32 v11 /*v523*/, v164 /*v676*/, v165 /*v677*/, v166 /*v678*/
	s_set_vgpr_msb 0xaa55
	v_max3_num_f32 v207 /*v463*/, v207 /*v463*/, v209 /*v465*/, v211 /*v467*/
	v_max3_num_f32 v209 /*v465*/, v221 /*v477*/, v223 /*v479*/, v225 /*v481*/
	v_max3_num_f32 v200 /*v456*/, v200 /*v456*/, v202 /*v458*/, v203 /*v459*/
	v_max3_num_f32 v202 /*v458*/, v204 /*v460*/, v205 /*v461*/, v206 /*v462*/
	s_set_vgpr_msb 0x5569
	v_max3_num_f32 v203 /*v459*/, v208 /*v464*/, v10 /*v522*/, v159 /*v671*/
	s_set_vgpr_msb 0x6955
	v_max3_num_f32 v204 /*v460*/, v227 /*v483*/, v229 /*v485*/, v239 /*v495*/
	v_max3_num_f32 v205 /*v461*/, v241 /*v497*/, v243 /*v499*/, v245 /*v501*/
	s_set_vgpr_msb 0x5569
	v_max3_num_f32 v206 /*v462*/, v247 /*v503*/, v3 /*v515*/, v5 /*v517*/
	s_set_vgpr_msb 0x696a
	v_max3_num_f32 v208 /*v464*/, v7 /*v519*/, v160 /*v672*/, v9 /*v521*/
	s_set_vgpr_msb 0x6a55
	v_max3_num_f32 v200 /*v456*/, v200 /*v456*/, v202 /*v458*/, v203 /*v459*/
	v_max3_num_f32 v201 /*v457*/, v201 /*v457*/, v207 /*v463*/, v209 /*v465*/
	v_max3_num_f32 v202 /*v458*/, v204 /*v460*/, v205 /*v461*/, v206 /*v462*/
	s_set_vgpr_msb 0x5569
	v_max3_num_f32 v203 /*v459*/, v208 /*v464*/, v11 /*v523*/, v167 /*v679*/
	s_set_vgpr_msb 0x6955
	v_max3_num_f32 v201 /*v457*/, v201 /*v457*/, v202 /*v458*/, v203 /*v459*/
	v_dual_mov_b32 v204 /*v460*/, v200 /*v456*/ :: v_dual_mov_b32 v202 /*v458*/, v201 /*v457*/
	v_permlanex16_b32 v204 /*v460*/, v204 /*v460*/, s11, 0xfedcba98
	v_permlanex16_b32 v202 /*v458*/, v202 /*v458*/, s11, 0xfedcba98
	v_dual_max_num_f32 v200 /*v456*/, v200 /*v456*/, v204 /*v460*/ :: v_dual_max_num_f32 v201 /*v457*/, v201 /*v457*/, v202 /*v458*/
	s_set_vgpr_msb 0x5549
	v_sub_f32_e32 v203 /*v459*/, v200 /*v456*/, v92 /*v604*/
	v_max_num_f32_e32 v200 /*v456*/, v200 /*v456*/, v92 /*v604*/
	v_sub_f32_e32 v202 /*v458*/, v201 /*v457*/, v71 /*v583*/
	s_set_vgpr_msb 0x4904
	v_cmp_lt_f32_e32 vcc_lo, 0x41000000, v203 /*v459*/
	s_cmp_eq_u32 vcc_lo, 0
	s_cselect_b32 s2, -1, 0
	s_set_vgpr_msb 0x489
	v_cndmask_b32_e64 v87 /*v599*/, v200 /*v456*/, v92 /*v604*/, s2
	s_set_vgpr_msb 0x8946
	v_cmp_lt_f32_e64 s2, 0x41000000, v202 /*v458*/
	v_max_num_f32_e32 v200 /*v456*/, v71 /*v583*/, v201 /*v457*/
	s_cmp_lg_u32 s2, 0
	s_cselect_b32 s5, -1, 0
	s_cmp_eq_u32 s2, 0
	s_cselect_b32 s2, -1, 0
	s_set_vgpr_msb 0x4689
	v_cndmask_b32_e64 v86 /*v598*/, v200 /*v456*/, v71 /*v583*/, s2
	v_mul_f32_e32 v60 /*v572*/, 0xbfb8aa3b, v87 /*v599*/
	v_mul_f32_e32 v70 /*v582*/, 0xbfb8aa3b, v86 /*v598*/
	s_set_vgpr_msb 0x8961
	v_pk_fma_f32 v[202:203] /*v[458:459]*/, v[196:197] /*v[452:453]*/, s[90:91], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[220:221] /*v[476:477]*/, v[236:237] /*v[492:493]*/, s[90:91], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6162
	v_pk_fma_f32 v[222:223] /*v[478:479]*/, v[42:43] /*v[554:555]*/, s[90:91], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_wait_alu depctr_vm_vsrc(0)
	s_set_vgpr_msb 0x62a2
	v_pk_fma_f32 v[88:89] /*v[600:601]*/, v[158:159] /*v[670:671]*/, s[90:91], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[90:91] /*v[602:603]*/, v[62:63] /*v[574:575]*/, s[90:91], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa261
	v_exp_f32_e32 v204 /*v460*/, v202 /*v458*/
	v_exp_f32_e32 v210 /*v466*/, v203 /*v459*/
	v_nop
	v_pk_fma_f32 v[202:203] /*v[458:459]*/, v[214:215] /*v[470:471]*/, s[90:91], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v242 /*v498*/, v221 /*v477*/
	v_exp_f32_e32 v236 /*v492*/, v222 /*v478*/
	v_exp_f32_e32 v244 /*v500*/, v223 /*v479*/
	v_nop
	s_set_vgpr_msb 0x6162
	v_pk_fma_f32 v[222:223] /*v[478:479]*/, v[52:53] /*v[564:565]*/, s[90:91], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6241
	v_exp_f32_e32 v246 /*v502*/, v202 /*v458*/
	s_set_vgpr_msb 0x4181
	v_exp_f32_e32 v2 /*v514*/, v203 /*v459*/
	v_nop
	s_set_vgpr_msb 0x8161
	v_pk_fma_f32 v[202:203] /*v[458:459]*/, v[230:231] /*v[486:487]*/, s[90:91], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v230 /*v486*/, v220 /*v476*/
	v_nop
	v_pk_fma_f32 v[220:221] /*v[476:477]*/, v[254:255] /*v[510:511]*/, s[90:91], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6181
	v_exp_f32_e32 v30 /*v542*/, v222 /*v478*/
	s_set_vgpr_msb 0x81a2
	v_exp_f32_e32 v62 /*v574*/, v88 /*v600*/
	v_pk_fma_f32 v[68:69] /*v[580:581]*/, v[68:69] /*v[580:581]*/, s[90:91], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[66:67] /*v[578:579]*/, v[66:67] /*v[578:579]*/, s[90:91], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa281
	v_exp_f32_e32 v28 /*v540*/, v220 /*v476*/
	v_exp_f32_e32 v36 /*v548*/, v221 /*v477*/
	v_nop
	s_set_vgpr_msb 0x8162
	v_pk_fma_f32 v[220:221] /*v[476:477]*/, v[40:41] /*v[552:553]*/, s[90:91], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6281
	v_exp_f32_e32 v40 /*v552*/, v223 /*v479*/
	v_nop
	s_set_vgpr_msb 0x8162
	v_pk_fma_f32 v[222:223] /*v[478:479]*/, v[136:137] /*v[648:649]*/, s[90:91], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x62a2
	v_pk_fma_f32 v[136:137] /*v[648:649]*/, v[64:65] /*v[576:577]*/, s[90:91], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v64 /*v576*/, v89 /*v601*/
	v_nop
	v_pk_fma_f32 v[88:89] /*v[600:601]*/, v[96:97] /*v[608:609]*/, s[90:91], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa261
	v_pk_fma_f32 v[198:199] /*v[454:455]*/, v[198:199] /*v[454:455]*/, s[90:91], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[206:207] /*v[462:463]*/, v[212:213] /*v[468:469]*/, s[90:91], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6142
	v_exp_f32_e32 v213 /*v469*/, v68 /*v580*/
	v_exp_f32_e32 v225 /*v481*/, v69 /*v581*/
	v_exp_f32_e32 v229 /*v485*/, v88 /*v600*/
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[68:69] /*v[580:581]*/, v[100:101] /*v[612:613]*/, s[90:91], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa242
	v_exp_f32_e32 v241 /*v497*/, v89 /*v601*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[88:89] /*v[600:601]*/, v[102:103] /*v[614:615]*/, s[90:91], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa242
	v_exp_f32_e32 v205 /*v461*/, v66 /*v578*/
	v_exp_f32_e32 v211 /*v467*/, v67 /*v579*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[66:67] /*v[578:579]*/, v[98:99] /*v[610:611]*/, s[90:91], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa261
	v_exp_f32_e32 v212 /*v468*/, v198 /*v454*/
	v_exp_f32_e32 v224 /*v480*/, v199 /*v455*/
	v_exp_f32_e32 v228 /*v484*/, v206 /*v462*/
	v_pk_fma_f32 v[198:199] /*v[454:455]*/, v[216:217] /*v[472:473]*/, s[90:91], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v240 /*v496*/, v207 /*v463*/
	v_nop
	v_pk_fma_f32 v[206:207] /*v[462:463]*/, v[218:219] /*v[474:475]*/, s[90:91], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[208:209] /*v[464:465]*/, v[232:233] /*v[488:489]*/, s[90:91], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[216:217] /*v[472:473]*/, v[234:235] /*v[490:491]*/, s[90:91], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x61a2
	v_exp_f32_e32 v11 /*v523*/, v68 /*v580*/
	v_exp_f32_e32 v19 /*v531*/, v69 /*v581*/
	v_exp_f32_e32 v21 /*v533*/, v88 /*v600*/
	v_pk_fma_f32 v[68:69] /*v[580:581]*/, v[106:107] /*v[618:619]*/, s[90:91], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v35 /*v547*/, v89 /*v601*/
	v_nop
	v_pk_fma_f32 v[88:89] /*v[600:601]*/, v[108:109] /*v[620:621]*/, s[90:91], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa242
	v_exp_f32_e32 v247 /*v503*/, v66 /*v578*/
	s_set_vgpr_msb 0x42a2
	v_exp_f32_e32 v3 /*v515*/, v67 /*v579*/
	v_nop
	v_pk_fma_f32 v[66:67] /*v[578:579]*/, v[104:105] /*v[616:617]*/, s[90:91], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa281
	v_exp_f32_e32 v20 /*v532*/, v206 /*v462*/
	v_exp_f32_e32 v34 /*v546*/, v207 /*v463*/
	s_set_vgpr_msb 0x8161
	v_exp_f32_e32 v206 /*v462*/, v208 /*v464*/
	v_exp_f32_e32 v214 /*v470*/, v209 /*v465*/
	v_exp_f32_e32 v218 /*v474*/, v216 /*v472*/
	v_pk_fma_f32 v[208:209] /*v[464:465]*/, v[250:251] /*v[506:507]*/, s[90:91], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v226 /*v482*/, v217 /*v473*/
	v_nop
	v_pk_fma_f32 v[216:217] /*v[472:473]*/, v[252:253] /*v[508:509]*/, s[90:91], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6142
	v_exp_f32_e32 v207 /*v463*/, v68 /*v580*/
	v_exp_f32_e32 v215 /*v471*/, v69 /*v581*/
	v_exp_f32_e32 v219 /*v475*/, v88 /*v600*/
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[68:69] /*v[580:581]*/, v[112:113] /*v[624:625]*/, s[90:91], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa242
	v_exp_f32_e32 v227 /*v483*/, v89 /*v601*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[88:89] /*v[600:601]*/, v[114:115] /*v[626:627]*/, s[90:91], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa281
	v_exp_f32_e32 v10 /*v522*/, v198 /*v454*/
	v_exp_f32_e32 v18 /*v530*/, v199 /*v455*/
	s_set_vgpr_msb 0x8141
	v_exp_f32_e32 v198 /*v454*/, v202 /*v458*/
	v_exp_f32_e32 v202 /*v458*/, v203 /*v459*/
	s_set_vgpr_msb 0x4142
	v_exp_f32_e32 v199 /*v455*/, v66 /*v578*/
	v_exp_f32_e32 v203 /*v459*/, v67 /*v579*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[66:67] /*v[578:579]*/, v[110:111] /*v[622:623]*/, s[90:91], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa241
	v_exp_f32_e32 v250 /*v506*/, v208 /*v464*/
	s_set_vgpr_msb 0x4181
	v_exp_f32_e32 v4 /*v516*/, v209 /*v465*/
	v_exp_f32_e32 v12 /*v524*/, v216 /*v472*/
	s_set_vgpr_msb 0x8162
	v_pk_fma_f32 v[208:209] /*v[464:465]*/, v[0:1] /*v[512:513]*/, s[90:91], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6281
	v_exp_f32_e32 v22 /*v534*/, v217 /*v473*/
	v_nop
	s_set_vgpr_msb 0x8162
	v_pk_fma_f32 v[216:217] /*v[472:473]*/, v[38:39] /*v[550:551]*/, s[90:91], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v251 /*v507*/, v68 /*v580*/
	s_set_vgpr_msb 0x62a2
	v_exp_f32_e32 v5 /*v517*/, v69 /*v581*/
	v_exp_f32_e32 v13 /*v525*/, v88 /*v600*/
	v_pk_fma_f32 v[68:69] /*v[580:581]*/, v[118:119] /*v[630:631]*/, s[90:91], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v23 /*v535*/, v89 /*v601*/
	v_nop
	v_pk_fma_f32 v[88:89] /*v[600:601]*/, v[120:121] /*v[632:633]*/, s[90:91], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa242
	v_exp_f32_e32 v231 /*v487*/, v66 /*v578*/
	v_exp_f32_e32 v243 /*v499*/, v67 /*v579*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[66:67] /*v[578:579]*/, v[116:117] /*v[628:629]*/, s[90:91], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa281
	v_exp_f32_e32 v38 /*v550*/, v208 /*v464*/
	v_exp_f32_e32 v48 /*v560*/, v209 /*v465*/
	s_set_vgpr_msb 0x8141
	v_exp_f32_e32 v208 /*v464*/, v216 /*v472*/
	v_exp_f32_e32 v216 /*v472*/, v217 /*v473*/
	s_set_vgpr_msb 0x4182
	v_exp_f32_e32 v39 /*v551*/, v68 /*v580*/
	v_exp_f32_e32 v49 /*v561*/, v69 /*v581*/
	s_set_vgpr_msb 0x8242
	v_exp_f32_e32 v209 /*v465*/, v88 /*v600*/
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[68:69] /*v[580:581]*/, v[124:125] /*v[636:637]*/, s[90:91], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa242
	v_exp_f32_e32 v217 /*v473*/, v89 /*v601*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[88:89] /*v[600:601]*/, v[126:127] /*v[638:639]*/, s[90:91], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v29 /*v541*/, v66 /*v578*/
	v_exp_f32_e32 v37 /*v549*/, v67 /*v579*/
	v_nop
	v_pk_fma_f32 v[66:67] /*v[578:579]*/, v[122:123] /*v[634:635]*/, s[90:91], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa262
	v_pk_fma_f32 v[234:235] /*v[490:491]*/, v[44:45] /*v[556:557]*/, s[90:91], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[238:239] /*v[494:495]*/, v[50:51] /*v[562:563]*/, s[90:91], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v237 /*v493*/, v68 /*v580*/
	v_exp_f32_e32 v245 /*v501*/, v69 /*v581*/
	v_exp_f32_e32 v253 /*v509*/, v88 /*v600*/
	s_set_vgpr_msb 0x62a2
	v_pk_fma_f32 v[68:69] /*v[580:581]*/, v[130:131] /*v[642:643]*/, s[90:91], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v7 /*v519*/, v89 /*v601*/
	v_nop
	v_pk_fma_f32 v[88:89] /*v[600:601]*/, v[132:133] /*v[644:645]*/, s[90:91], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa261
	v_pk_fma_f32 v[192:193] /*v[448:449]*/, v[192:193] /*v[448:449]*/, s[90:91], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[200:201] /*v[456:457]*/, v[194:195] /*v[450:451]*/, s[90:91], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v232 /*v488*/, v221 /*v477*/
	s_set_vgpr_msb 0x6142
	v_exp_f32_e32 v221 /*v477*/, v66 /*v578*/
	v_exp_f32_e32 v233 /*v489*/, v67 /*v579*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[66:67] /*v[578:579]*/, v[128:129] /*v[640:641]*/, s[90:91], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa241
	v_exp_f32_e32 v252 /*v508*/, v234 /*v490*/
	s_set_vgpr_msb 0x4181
	v_exp_f32_e32 v6 /*v518*/, v235 /*v491*/
	v_exp_f32_e32 v14 /*v526*/, v238 /*v494*/
	s_set_vgpr_msb 0x8162
	v_pk_fma_f32 v[234:235] /*v[490:491]*/, v[54:55] /*v[566:567]*/, s[90:91], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6281
	v_exp_f32_e32 v24 /*v536*/, v239 /*v495*/
	v_nop
	s_set_vgpr_msb 0x8162
	v_pk_fma_f32 v[238:239] /*v[494:495]*/, v[56:57] /*v[568:569]*/, s[90:91], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[254:255] /*v[510:511]*/, v[138:139] /*v[650:651]*/, s[90:91], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x62a2
	v_exp_f32_e32 v31 /*v543*/, v68 /*v580*/
	v_exp_f32_e32 v41 /*v553*/, v69 /*v581*/
	v_exp_f32_e32 v45 /*v557*/, v88 /*v600*/
	v_pk_fma_f32 v[68:69] /*v[580:581]*/, v[144:145] /*v[656:657]*/, s[90:91], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v51 /*v563*/, v89 /*v601*/
	v_nop
	v_pk_fma_f32 v[88:89] /*v[600:601]*/, v[146:147] /*v[658:659]*/, s[90:91], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa241
	v_exp_f32_e32 v192 /*v448*/, v192 /*v448*/
	v_exp_f32_e32 v194 /*v450*/, v193 /*v449*/
	v_exp_f32_e32 v196 /*v452*/, v200 /*v456*/
	v_exp_f32_e32 v200 /*v456*/, v201 /*v457*/
	s_set_vgpr_msb 0x4142
	v_exp_f32_e32 v193 /*v449*/, v90 /*v602*/
	v_exp_f32_e32 v195 /*v451*/, v91 /*v603*/
	v_exp_f32_e32 v201 /*v457*/, v137 /*v649*/
	s_set_vgpr_msb 0x42a2
	v_exp_f32_e32 v15 /*v527*/, v66 /*v578*/
	v_exp_f32_e32 v25 /*v537*/, v67 /*v579*/
	v_nop
	v_pk_fma_f32 v[66:67] /*v[578:579]*/, v[134:135] /*v[646:647]*/, s[90:91], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa281
	v_exp_f32_e32 v44 /*v556*/, v234 /*v490*/
	v_exp_f32_e32 v50 /*v562*/, v235 /*v491*/
	v_exp_f32_e32 v52 /*v564*/, v238 /*v494*/
	v_exp_f32_e32 v58 /*v570*/, v239 /*v495*/
	s_set_vgpr_msb 0x8141
	v_exp_f32_e32 v234 /*v490*/, v223 /*v479*/
	v_exp_f32_e32 v238 /*v494*/, v254 /*v510*/
	s_set_vgpr_msb 0x41a2
	v_pk_fma_f32 v[16:17] /*v[528:529]*/, v[142:143] /*v[654:655]*/, s[90:91], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa241
	v_exp_f32_e32 v254 /*v510*/, v255 /*v511*/
	s_set_vgpr_msb 0x41a2
	v_pk_fma_f32 v[32:33] /*v[544:545]*/, v[152:153] /*v[664:665]*/, s[90:91], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa242
	v_exp_f32_e32 v223 /*v479*/, v68 /*v580*/
	v_exp_f32_e32 v235 /*v491*/, v69 /*v581*/
	v_exp_f32_e32 v239 /*v495*/, v88 /*v600*/
	v_exp_f32_e32 v255 /*v511*/, v89 /*v601*/
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[68:69] /*v[580:581]*/, v[150:151] /*v[662:663]*/, s[90:91], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[88:89] /*v[600:601]*/, v[160:161] /*v[672:673]*/, s[90:91], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[0:1] /*v[512:513]*/, v[140:141] /*v[652:653]*/, s[90:91], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa242
	v_exp_f32_e32 v197 /*v453*/, v136 /*v648*/
	s_set_vgpr_msb 0x42a2
	v_exp_f32_e32 v53 /*v565*/, v66 /*v578*/
	v_exp_f32_e32 v59 /*v571*/, v67 /*v579*/
	v_nop
	v_pk_fma_f32 v[66:67] /*v[578:579]*/, v[148:149] /*v[660:661]*/, s[90:91], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v26 /*v538*/, v17 /*v529*/
	v_pk_fma_f32 v[56:57] /*v[568:569]*/, v[156:157] /*v[668:669]*/, s[90:91], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v42 /*v554*/, v33 /*v545*/
	v_exp_f32_e32 v17 /*v529*/, v68 /*v580*/
	v_exp_f32_e32 v27 /*v539*/, v69 /*v581*/
	v_exp_f32_e32 v33 /*v545*/, v88 /*v600*/
	v_exp_f32_e32 v43 /*v555*/, v89 /*v601*/
	v_pk_fma_f32 v[68:69] /*v[580:581]*/, v[164:165] /*v[676:677]*/, s[90:91], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa285
	v_pk_add_f32 v[88:89] /*v[600:601]*/, v[192:193] /*v[448:449]*/, v[194:195] /*v[450:451]*/
	v_pk_add_f32 v[90:91] /*v[602:603]*/, v[200:201] /*v[456:457]*/, v[204:205] /*v[460:461]*/
	s_set_vgpr_msb 0x8541
	v_exp_f32_e32 v220 /*v476*/, v220 /*v476*/
	v_exp_f32_e32 v222 /*v478*/, v222 /*v478*/
	s_set_vgpr_msb 0x41a2
	v_exp_f32_e32 v8 /*v520*/, v1 /*v513*/
	v_pk_fma_f32 v[46:47] /*v[558:559]*/, v[154:155] /*v[666:667]*/, s[90:91], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v1 /*v513*/, v66 /*v578*/
	v_exp_f32_e32 v9 /*v521*/, v67 /*v579*/
	v_nop
	v_pk_fma_f32 v[66:67] /*v[578:579]*/, v[162:163] /*v[674:675]*/, s[90:91], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v60 /*v572*/, v57 /*v569*/
	v_exp_f32_e32 v57 /*v569*/, v68 /*v580*/
	v_exp_f32_e32 v61 /*v573*/, v69 /*v581*/
	v_nop
	s_set_vgpr_msb 0xa289
	v_pk_add_f32 v[68:69] /*v[580:581]*/, v[196:197] /*v[452:453]*/, v[88:89] /*v[600:601]*/
	v_pk_add_f32 v[88:89] /*v[600:601]*/, v[210:211] /*v[466:467]*/, v[90:91] /*v[602:603]*/
	s_set_vgpr_msb 0x8985
	v_pk_add_f32 v[90:91] /*v[602:603]*/, v[212:213] /*v[468:469]*/, v[224:225] /*v[480:481]*/
	v_pk_add_f32 v[96:97] /*v[608:609]*/, v[240:241] /*v[496:497]*/, v[246:247] /*v[502:503]*/
	s_set_vgpr_msb 0x858a
	v_pk_add_f32 v[98:99] /*v[610:611]*/, v[10:11] /*v[522:523]*/, v[18:19] /*v[530:531]*/
	v_pk_add_f32 v[108:109] /*v[620:621]*/, v[22:23] /*v[534:535]*/, v[28:29] /*v[540:541]*/
	v_pk_add_f32 v[110:111] /*v[622:623]*/, v[38:39] /*v[550:551]*/, v[48:49] /*v[560:561]*/
	s_set_vgpr_msb 0x8a85
	v_pk_add_f32 v[114:115] /*v[626:627]*/, v[236:237] /*v[492:493]*/, v[244:245] /*v[500:501]*/
	s_set_vgpr_msb 0x858a
	v_pk_add_f32 v[116:117] /*v[628:629]*/, v[6:7] /*v[518:519]*/, v[14:15] /*v[526:527]*/
	v_exp_f32_e32 v0 /*v512*/, v0 /*v512*/
	v_exp_f32_e32 v16 /*v528*/, v16 /*v528*/
	v_exp_f32_e32 v46 /*v558*/, v46 /*v558*/
	v_exp_f32_e32 v54 /*v566*/, v47 /*v559*/
	v_exp_f32_e32 v47 /*v559*/, v66 /*v578*/
	s_set_vgpr_msb 0x8a86
	v_pk_add_f32 v[100:101] /*v[612:613]*/, v[34:35] /*v[546:547]*/, v[198:199] /*v[454:455]*/
	s_set_vgpr_msb 0x8685
	v_pk_add_f32 v[102:103] /*v[614:615]*/, v[206:207] /*v[462:463]*/, v[214:215] /*v[470:471]*/
	s_set_vgpr_msb 0x8589
	v_pk_add_f32 v[90:91] /*v[602:603]*/, v[228:229] /*v[484:485]*/, v[90:91] /*v[602:603]*/
	s_set_vgpr_msb 0x898a
	v_pk_add_f32 v[96:97] /*v[608:609]*/, v[2:3] /*v[514:515]*/, v[96:97] /*v[608:609]*/
	v_pk_add_f32 v[98:99] /*v[610:611]*/, v[20:21] /*v[532:533]*/, v[98:99] /*v[610:611]*/
	s_set_vgpr_msb 0x8a85
	v_pk_add_f32 v[104:105] /*v[616:617]*/, v[226:227] /*v[482:483]*/, v[230:231] /*v[486:487]*/
	v_pk_add_f32 v[112:113] /*v[624:625]*/, v[216:217] /*v[472:473]*/, v[220:221] /*v[476:477]*/
	s_set_vgpr_msb 0x858a
	v_pk_add_f32 v[108:109] /*v[620:621]*/, v[36:37] /*v[548:549]*/, v[108:109] /*v[620:621]*/
	s_set_vgpr_msb 0x8a89
	v_pk_add_f32 v[110:111] /*v[622:623]*/, v[208:209] /*v[464:465]*/, v[110:111] /*v[622:623]*/
	s_set_vgpr_msb 0x898a
	v_pk_add_f32 v[118:119] /*v[630:631]*/, v[30:31] /*v[542:543]*/, v[40:41] /*v[552:553]*/
	v_pk_add_f32 v[120:121] /*v[632:633]*/, v[50:51] /*v[562:563]*/, v[52:53] /*v[564:565]*/
	s_set_vgpr_msb 0x8a85
	v_pk_add_f32 v[122:123] /*v[634:635]*/, v[222:223] /*v[478:479]*/, v[234:235] /*v[490:491]*/
	s_set_vgpr_msb 0x8589
	v_pk_add_f32 v[114:115] /*v[626:627]*/, v[252:253] /*v[508:509]*/, v[114:115] /*v[626:627]*/
	s_set_vgpr_msb 0x89aa
	v_pk_add_f32 v[116:117] /*v[628:629]*/, v[24:25] /*v[536:537]*/, v[116:117] /*v[628:629]*/
	v_pk_add_f32 v[68:69] /*v[580:581]*/, v[68:69] /*v[580:581]*/, v[88:89] /*v[600:601]*/
	v_exp_f32_e32 v32 /*v544*/, v32 /*v544*/
	v_exp_f32_e32 v56 /*v568*/, v56 /*v568*/
	v_exp_f32_e32 v55 /*v567*/, v67 /*v579*/
	v_nop
	v_pk_fma_f32 v[66:67] /*v[578:579]*/, v[166:167] /*v[678:679]*/, s[90:91], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xaa89
	v_pk_add_f32 v[100:101] /*v[612:613]*/, v[202:203] /*v[458:459]*/, v[100:101] /*v[612:613]*/
	v_pk_add_f32 v[102:103] /*v[614:615]*/, v[218:219] /*v[474:475]*/, v[102:103] /*v[614:615]*/
	v_pk_add_f32 v[106:107] /*v[618:619]*/, v[250:251] /*v[506:507]*/, v[4:5] /*v[516:517]*/
	v_pk_add_f32 v[104:105] /*v[616:617]*/, v[242:243] /*v[498:499]*/, v[104:105] /*v[616:617]*/
	v_pk_add_f32 v[112:113] /*v[624:625]*/, v[232:233] /*v[488:489]*/, v[112:113] /*v[624:625]*/
	s_set_vgpr_msb 0x898a
	v_pk_add_f32 v[118:119] /*v[630:631]*/, v[44:45] /*v[556:557]*/, v[118:119] /*v[630:631]*/
	v_pk_add_f32 v[120:121] /*v[632:633]*/, v[58:59] /*v[570:571]*/, v[120:121] /*v[632:633]*/
	s_set_vgpr_msb 0x8a89
	v_pk_add_f32 v[122:123] /*v[634:635]*/, v[238:239] /*v[494:495]*/, v[122:123] /*v[634:635]*/
	v_pk_add_f32 v[124:125] /*v[636:637]*/, v[254:255] /*v[510:511]*/, v[0:1] /*v[512:513]*/
	s_set_vgpr_msb 0x898a
	v_pk_add_f32 v[126:127] /*v[638:639]*/, v[16:17] /*v[528:529]*/, v[26:27] /*v[538:539]*/
	v_pk_add_f32 v[128:129] /*v[640:641]*/, v[42:43] /*v[554:555]*/, v[46:47] /*v[558:559]*/
	v_pk_add_f32 v[68:69] /*v[580:581]*/, v[90:91] /*v[602:603]*/, v[68:69] /*v[580:581]*/
	v_pk_add_f32 v[90:91] /*v[602:603]*/, v[96:97] /*v[608:609]*/, v[98:99] /*v[610:611]*/
	v_pk_add_f32 v[96:97] /*v[608:609]*/, v[108:109] /*v[620:621]*/, v[110:111] /*v[622:623]*/
	v_pk_add_f32 v[98:99] /*v[610:611]*/, v[114:115] /*v[626:627]*/, v[116:117] /*v[628:629]*/
	v_exp_f32_e32 v63 /*v575*/, v66 /*v578*/
	v_pk_add_f32 v[106:107] /*v[618:619]*/, v[12:13] /*v[524:525]*/, v[106:107] /*v[618:619]*/
	v_pk_add_f32 v[130:131] /*v[642:643]*/, v[56:57] /*v[568:569]*/, v[60:61] /*v[572:573]*/
	v_pk_add_f32 v[88:89] /*v[600:601]*/, v[8:9] /*v[520:521]*/, v[124:125] /*v[636:637]*/
	v_pk_add_f32 v[124:125] /*v[636:637]*/, v[32:33] /*v[544:545]*/, v[126:127] /*v[638:639]*/
	v_pk_add_f32 v[126:127] /*v[638:639]*/, v[54:55] /*v[566:567]*/, v[128:129] /*v[640:641]*/
	v_pk_add_f32 v[102:103] /*v[614:615]*/, v[102:103] /*v[614:615]*/, v[104:105] /*v[616:617]*/
	v_pk_add_f32 v[104:105] /*v[616:617]*/, v[120:121] /*v[632:633]*/, v[122:123] /*v[634:635]*/
	v_pk_add_f32 v[90:91] /*v[602:603]*/, v[100:101] /*v[612:613]*/, v[90:91] /*v[602:603]*/
	v_pk_add_f32 v[96:97] /*v[608:609]*/, v[112:113] /*v[624:625]*/, v[96:97] /*v[608:609]*/
	v_pk_add_f32 v[98:99] /*v[610:611]*/, v[118:119] /*v[630:631]*/, v[98:99] /*v[610:611]*/
	v_pk_add_f32 v[128:129] /*v[640:641]*/, v[62:63] /*v[574:575]*/, v[130:131] /*v[642:643]*/
	v_pk_add_f32 v[100:101] /*v[612:613]*/, v[106:107] /*v[618:619]*/, v[102:103] /*v[614:615]*/
	v_pk_add_f32 v[88:89] /*v[600:601]*/, v[88:89] /*v[600:601]*/, v[104:105] /*v[616:617]*/
	v_pk_add_f32 v[102:103] /*v[614:615]*/, v[124:125] /*v[636:637]*/, v[126:127] /*v[638:639]*/
	v_pk_add_f32 v[68:69] /*v[580:581]*/, v[68:69] /*v[580:581]*/, v[90:91] /*v[602:603]*/
	v_pk_add_f32 v[90:91] /*v[602:603]*/, v[96:97] /*v[608:609]*/, v[98:99] /*v[610:611]*/
	v_exp_f32_e32 v65 /*v577*/, v67 /*v579*/
	v_nop
	v_pk_add_f32 v[66:67] /*v[578:579]*/, v[128:129] /*v[640:641]*/, v[102:103] /*v[614:615]*/
	v_pk_add_f32 v[68:69] /*v[580:581]*/, v[100:101] /*v[612:613]*/, v[68:69] /*v[580:581]*/
	v_pk_add_f32 v[88:89] /*v[600:601]*/, v[88:89] /*v[600:601]*/, v[90:91] /*v[602:603]*/
	v_pk_add_f32 v[66:67] /*v[578:579]*/, v[64:65] /*v[576:577]*/, v[66:67] /*v[578:579]*/
	v_pk_add_f32 v[68:69] /*v[580:581]*/, v[68:69] /*v[580:581]*/, v[88:89] /*v[600:601]*/
	v_pk_add_f32 v[66:67] /*v[578:579]*/, v[66:67] /*v[578:579]*/, v[68:69] /*v[580:581]*/
	v_dual_sub_f32 v70 /*v582*/, v92 /*v604*/, v87 /*v599*/ :: v_dual_mov_b32 v88 /*v600*/, v66 /*v578*/
	v_dual_mul_f32 v68 /*v580*/, 0x3fb8aa3b, v70 /*v582*/ :: v_dual_mov_b32 v69 /*v581*/, v67 /*v579*/
	v_permlanex16_b32 v88 /*v600*/, v88 /*v600*/, s11, 0xfedcba98
	v_exp_f32_e32 v68 /*v580*/, v68 /*v580*/
	v_permlanex16_b32 v69 /*v581*/, v69 /*v581*/, s11, 0xfedcba98
	s_set_vgpr_msb 0x8a00
	s_cbranch_vccz .LBB0_20
	s_set_vgpr_msb 8
	v_pk_mul_f32 v[126:127], v[126:127], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[124:125], v[124:125], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[122:123], v[122:123], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[120:121], v[120:121], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[118:119], v[118:119], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[116:117], v[116:117], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[114:115], v[114:115], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[112:113], v[112:113], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[110:111], v[110:111], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[108:109], v[108:109], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[106:107], v[106:107], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[104:105], v[104:105], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[102:103], v[102:103], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[100:101], v[100:101], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[98:99], v[98:99], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[96:97], v[96:97], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[94:95], v[94:95], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[92:93], v[92:93], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[90:91], v[90:91], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[88:89], v[88:89], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[86:87], v[86:87], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[84:85], v[84:85], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[82:83], v[82:83], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[80:81], v[80:81], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[78:79], v[78:79], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[76:77], v[76:77], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[74:75], v[74:75], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[72:73], v[72:73], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[70:71], v[70:71], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[68:69], v[68:69], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[66:67], v[66:67], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[64:65], v[64:65], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x800
.LBB0_20:
	s_set_vgpr_msb 0x8a
	v_sub_f32_e32 v70 /*v582*/, v71 /*v583*/, v86 /*v598*/
	s_and_b32 s2, s5, exec_lo
	s_cselect_b32 s2, 1, 0
	s_cmp_lg_u32 s2, 1
	v_mul_f32_e32 v70 /*v582*/, 0x3fb8aa3b, v70 /*v582*/
	v_exp_f32_e32 v70 /*v582*/, v70 /*v582*/
	s_set_vgpr_msb 0x8a00
	s_cbranch_scc1 .LBB0_15
	v_nop
	s_set_vgpr_msb 8
	v_pk_mul_f32 v[62:63], v[62:63], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[60:61], v[60:61], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59], v[58:59], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[56:57], v[56:57], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[54:55], v[54:55], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[52:53], v[52:53], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[50:51], v[50:51], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[48:49], v[48:49], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[46:47], v[46:47], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[44:45], v[44:45], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[42:43], v[42:43], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[40:41], v[40:41], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39], v[38:39], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[36:37], v[36:37], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35], v[34:35], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[32:33], v[32:33], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[30:31], v[30:31], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[28:29], v[28:29], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[26:27], v[26:27], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25], v[24:25], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[22:23], v[22:23], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21], v[20:21], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[18:19], v[18:19], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[16:17], v[16:17], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[14:15], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13], v[12:13], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[10:11], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[8:9], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[6:7], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[4:5], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[2:3], v[2:3], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[0:1], v[0:1], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x800
	s_branch .LBB0_15
.LBB0_22:
	s_set_vgpr_msb 0x82
	v_dual_mov_b32 v94 /*v606*/, v89 /*v601*/ :: v_dual_mov_b32 v93 /*v605*/, v88 /*v600*/
	v_dual_mov_b32 v86 /*v598*/, v71 /*v583*/ :: v_dual_mov_b32 v87 /*v599*/, v92 /*v604*/
	s_set_vgpr_msb 0x8200
.LBB0_23:
	s_cmp_ge_u32 s82, s10
	s_cbranch_scc1 .LBB0_32
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 8
	v_dual_add_nc_u32 v192, s99, v74 /*v586*/ :: v_dual_add_nc_u32 v193, s99, v73 /*v585*/
	s_add_co_i32 s2, s19, -1
	s_mov_b32 s51, 0
	s_mov_b32 s28, 1
	s_set_vgpr_msb 0x800
	v_dual_add_nc_u32 v194, s17, v192 :: v_dual_add_nc_u32 v195, s17, v193
	s_wait_alu depctr_vm_vsrc(0)
	s_set_vgpr_msb 0x80
	v_subrev_nc_u32_e32 v88 /*v600*/, s16, v192
	v_subrev_nc_u32_e32 v89 /*v601*/, s16, v193
	s_mov_b32 s11, s51
	v_min_i32_e32 v90 /*v602*/, s2, v194
	v_min_i32_e32 v91 /*v603*/, s2, v195
	s_sub_co_i32 s7, 1, s60
	s_mov_b32 s48, 32
	s_mov_b32 s47, 0x800000
	s_mov_b32 s45, 0xffff0000
	s_mov_b32 s44, 0x7510000
	s_mov_b32 s52, 0xf510000
	s_mov_b32 s61, 0x76543210
	s_mov_b32 s88, 0x3fb8aa3b
	s_set_vgpr_msb 0x8000
	s_branch .LBB0_26
.LBB0_25:
	s_set_vgpr_msb 0x8a
	v_cvt_pk_bf16_f32 v103 /*v615*/, v14 /*v526*/, v26 /*v538*/
	s_set_vgpr_msb 0x8a89
	v_cvt_pk_bf16_f32 v102 /*v614*/, v254 /*v510*/, v10 /*v522*/
	s_set_vgpr_msb 0x8985
	v_cvt_pk_bf16_f32 v101 /*v613*/, v236 /*v492*/, v250 /*v506*/
	v_cvt_pk_bf16_f32 v100 /*v612*/, v222 /*v478*/, v232 /*v488*/
	v_cvt_pk_bf16_f32 v99 /*v611*/, v210 /*v466*/, v218 /*v474*/
	v_cvt_pk_bf16_f32 v98 /*v610*/, v202 /*v458*/, v208 /*v464*/
	v_cvt_pk_bf16_f32 v97 /*v609*/, v196 /*v452*/, v200 /*v456*/
	v_cvt_pk_bf16_f32 v96 /*v608*/, v192 /*v448*/, v194 /*v450*/
	s_set_vgpr_msb 0x858a
	v_cvt_pk_bf16_f32 v111 /*v623*/, v15 /*v527*/, v27 /*v539*/
	s_set_vgpr_msb 0x8a89
	v_cvt_pk_bf16_f32 v110 /*v622*/, v255 /*v511*/, v11 /*v523*/
	s_set_vgpr_msb 0x8985
	v_cvt_pk_bf16_f32 v109 /*v621*/, v237 /*v493*/, v251 /*v507*/
	v_cvt_pk_bf16_f32 v108 /*v620*/, v223 /*v479*/, v233 /*v489*/
	v_cvt_pk_bf16_f32 v107 /*v619*/, v211 /*v467*/, v219 /*v475*/
	v_cvt_pk_bf16_f32 v106 /*v618*/, v203 /*v459*/, v209 /*v465*/
	v_cvt_pk_bf16_f32 v105 /*v617*/, v197 /*v453*/, v201 /*v457*/
	v_cvt_pk_bf16_f32 v104 /*v616*/, v193 /*v449*/, v195 /*v451*/
	s_set_vgpr_msb 0x8509
	s_wait_dscnt 0x3d
	v_wmma_f32_16x16x32_bf16 v[120:127], v[184:191] /*v[440:447]*/, v[96:103] /*v[608:615]*/, v[120:127]
	s_set_vgpr_msb 0x98a
	v_cvt_pk_bf16_f32 v119 /*v631*/, v37 /*v549*/, v45 /*v557*/
	v_cvt_pk_bf16_f32 v118 /*v630*/, v23 /*v535*/, v33 /*v545*/
	v_cvt_pk_bf16_f32 v117 /*v629*/, v7 /*v519*/, v19 /*v531*/
	s_set_vgpr_msb 0x8a89
	v_cvt_pk_bf16_f32 v116 /*v628*/, v245 /*v501*/, v3 /*v515*/
	s_set_vgpr_msb 0x8985
	v_cvt_pk_bf16_f32 v115 /*v627*/, v229 /*v485*/, v241 /*v497*/
	v_cvt_pk_bf16_f32 v114 /*v626*/, v217 /*v473*/, v227 /*v483*/
	v_cvt_pk_bf16_f32 v113 /*v625*/, v207 /*v463*/, v215 /*v471*/
	s_set_vgpr_msb 0x8509
	v_wmma_f32_16x16x32_bf16 v[56:63], v[184:191] /*v[440:447]*/, v[104:111] /*v[616:623]*/, v[56:63]
	s_set_vgpr_msb 0x985
	v_cvt_pk_bf16_f32 v112 /*v624*/, v199 /*v455*/, v205 /*v461*/
	s_set_vgpr_msb 0x854a
	v_cvt_pk_bf16_f32 v199 /*v455*/, v53 /*v565*/, v59 /*v571*/
	v_cvt_pk_bf16_f32 v197 /*v453*/, v31 /*v543*/, v41 /*v553*/
	v_cvt_pk_bf16_f32 v196 /*v452*/, v17 /*v529*/, v29 /*v541*/
	v_cvt_pk_bf16_f32 v191 /*v447*/, v36 /*v548*/, v44 /*v556*/
	v_cvt_pk_bf16_f32 v190 /*v446*/, v22 /*v534*/, v32 /*v544*/
	v_cvt_pk_bf16_f32 v189 /*v445*/, v6 /*v518*/, v18 /*v530*/
	s_set_vgpr_msb 0x4a09
	s_wait_dscnt 0x3c
	v_wmma_f32_16x16x32_bf16 v[112:119], v[152:159] /*v[408:415]*/, v[96:103] /*v[608:615]*/, v[112:119]
	s_set_vgpr_msb 0x949
	v_cvt_pk_bf16_f32 v188 /*v444*/, v244 /*v500*/, v2 /*v514*/
	s_set_vgpr_msb 0x4945
	v_cvt_pk_bf16_f32 v187 /*v443*/, v228 /*v484*/, v240 /*v496*/
	v_cvt_pk_bf16_f32 v186 /*v442*/, v216 /*v472*/, v226 /*v482*/
	v_cvt_pk_bf16_f32 v185 /*v441*/, v206 /*v462*/, v214 /*v470*/
	v_cvt_pk_bf16_f32 v184 /*v440*/, v198 /*v454*/, v204 /*v460*/
	s_set_vgpr_msb 0x454a
	v_cvt_pk_bf16_f32 v198 /*v454*/, v43 /*v555*/, v51 /*v563*/
	v_cvt_pk_bf16_f32 v195 /*v451*/, v1 /*v513*/, v13 /*v525*/
	s_set_vgpr_msb 0x4a09
	v_wmma_f32_16x16x32_bf16 v[48:55], v[152:159] /*v[408:415]*/, v[104:111] /*v[616:623]*/, v[48:55]
	s_set_vgpr_msb 0x945
	v_cvt_pk_bf16_f32 v194 /*v450*/, v239 /*v495*/, v253 /*v509*/
	v_cvt_pk_bf16_f32 v193 /*v449*/, v225 /*v481*/, v235 /*v491*/
	v_cvt_pk_bf16_f32 v192 /*v448*/, v213 /*v469*/, v221 /*v477*/
	s_set_vgpr_msb 0x454a
	v_cvt_pk_bf16_f32 v207 /*v463*/, v63 /*v575*/, v65 /*v577*/
	v_cvt_pk_bf16_f32 v206 /*v462*/, v57 /*v569*/, v61 /*v573*/
	v_cvt_pk_bf16_f32 v205 /*v461*/, v49 /*v561*/, v55 /*v567*/
	v_cvt_pk_bf16_f32 v204 /*v460*/, v39 /*v551*/, v47 /*v559*/
	s_set_vgpr_msb 0x4a09
	s_wait_dscnt 0x2d
	v_wmma_f32_16x16x32_bf16 v[104:111], v[120:127] /*v[376:383]*/, v[96:103] /*v[608:615]*/, v[104:111]
	s_set_vgpr_msb 0x94a
	v_cvt_pk_bf16_f32 v203 /*v459*/, v25 /*v537*/, v35 /*v547*/
	v_cvt_pk_bf16_f32 v202 /*v458*/, v9 /*v521*/, v21 /*v533*/
	s_set_vgpr_msb 0x4a49
	v_cvt_pk_bf16_f32 v201 /*v457*/, v247 /*v503*/, v5 /*v517*/
	s_set_vgpr_msb 0x4945
	v_cvt_pk_bf16_f32 v200 /*v456*/, v231 /*v487*/, v243 /*v499*/
	s_set_vgpr_msb 0x4586
	v_dual_fmac_f32 v87 /*v599*/, v68 /*v580*/, v248 /*v504*/ :: v_dual_fmac_f32 v94 /*v606*/, v70 /*v582*/, v249 /*v505*/
	s_add_nc_u64 s[82:83], s[82:83], 1
	s_set_vgpr_msb 0x8609
	v_wmma_f32_16x16x32_bf16 v[40:47], v[120:127] /*v[376:383]*/, v[104:111] /*v[616:623]*/, v[40:47]
	v_cmp_ge_u64_e64 s2, s[82:83], s[10:11]
	s_set_vgpr_msb 0x94a
	v_dual_add_f32 v248 /*v504*/, v87 /*v599*/, v66 /*v578*/ :: v_dual_add_f32 v249 /*v505*/, v94 /*v606*/, v67 /*v579*/
	s_wait_alu depctr_vm_vsrc(0)
	s_set_vgpr_msb 0x4a82
	v_dual_mov_b32 v94 /*v606*/, v84 /*v596*/ :: v_dual_mov_b32 v84 /*v596*/, v93 /*v605*/
	v_dual_mov_b32 v93 /*v605*/, v85 /*v597*/ :: v_dual_mov_b32 v85 /*v597*/, v92 /*v604*/
	s_set_vgpr_msb 0x8209
	s_wait_dscnt 0x2c
	v_wmma_f32_16x16x32_bf16 v[96:103], v[88:95] /*v[344:351]*/, v[96:103] /*v[608:615]*/, v[96:103]
	s_set_vgpr_msb 0x982
	v_dual_mov_b32 v86 /*v598*/, v71 /*v583*/ :: v_dual_mov_b32 v87 /*v599*/, v69 /*v581*/
	s_and_b32 vcc_lo, exec_lo, s2
	s_set_vgpr_msb 0x8209
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[32:39], v[88:95] /*v[344:351]*/, v[104:111] /*v[616:623]*/, v[32:39]
	v_wmma_f32_16x16x32_bf16 v[88:95], v[56:63] /*v[312:319]*/, v[96:103] /*v[608:615]*/, v[88:95]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[56:63] /*v[312:319]*/, v[104:111] /*v[616:623]*/, v[24:31]
	v_wmma_f32_16x16x32_bf16 v[80:87], v[24:31] /*v[280:287]*/, v[96:103] /*v[608:615]*/, v[80:87]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[24:31] /*v[280:287]*/, v[104:111] /*v[616:623]*/, v[16:23]
	s_set_vgpr_msb 0x908
	v_wmma_f32_16x16x32_bf16 v[72:79], v[248:255], v[96:103] /*v[608:615]*/, v[72:79]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[248:255], v[104:111] /*v[616:623]*/, v[8:15]
	v_wmma_f32_16x16x32_bf16 v[64:71], v[216:223], v[96:103] /*v[608:615]*/, v[64:71]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[216:223], v[104:111] /*v[616:623]*/, v[0:7]
	s_set_vgpr_msb 0x805
	v_wmma_f32_16x16x32_bf16 v[120:127], v[176:183] /*v[432:439]*/, v[184:191] /*v[440:447]*/, v[120:127]
	s_set_vgpr_msb 0x509
	v_wmma_f32_16x16x32_bf16 v[56:63], v[176:183] /*v[432:439]*/, v[112:119] /*v[624:631]*/, v[56:63]
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x94a
	v_cvt_pk_bf16_f32 v183 /*v439*/, v52 /*v564*/, v58 /*v570*/
	v_cvt_pk_bf16_f32 v182 /*v438*/, v42 /*v554*/, v50 /*v562*/
	v_cvt_pk_bf16_f32 v181 /*v437*/, v30 /*v542*/, v40 /*v552*/
	s_set_vgpr_msb 0x4a05
	v_wmma_f32_16x16x32_bf16 v[112:119], v[144:151] /*v[400:407]*/, v[184:191] /*v[440:447]*/, v[112:119]
	s_set_vgpr_msb 0x54a
	v_cvt_pk_bf16_f32 v180 /*v436*/, v16 /*v528*/, v28 /*v540*/
	v_cvt_pk_bf16_f32 v179 /*v435*/, v0 /*v512*/, v12 /*v524*/
	s_set_vgpr_msb 0x4a45
	v_cvt_pk_bf16_f32 v178 /*v434*/, v238 /*v494*/, v252 /*v508*/
	v_cvt_pk_bf16_f32 v177 /*v433*/, v224 /*v480*/, v234 /*v490*/
	v_cvt_pk_bf16_f32 v176 /*v432*/, v212 /*v468*/, v220 /*v476*/
	s_set_vgpr_msb 0x4509
	v_wmma_f32_16x16x32_bf16 v[48:55], v[144:151] /*v[400:407]*/, v[112:119] /*v[624:631]*/, v[48:55]
	s_set_vgpr_msb 0x905
	v_wmma_f32_16x16x32_bf16 v[104:111], v[112:119] /*v[368:375]*/, v[184:191] /*v[440:447]*/, v[104:111]
	s_set_vgpr_msb 0x509
	v_wmma_f32_16x16x32_bf16 v[40:47], v[112:119] /*v[368:375]*/, v[112:119] /*v[624:631]*/, v[40:47]
	s_set_vgpr_msb 0x905
	v_wmma_f32_16x16x32_bf16 v[96:103], v[80:87] /*v[336:343]*/, v[184:191] /*v[440:447]*/, v[96:103]
	s_set_vgpr_msb 0x509
	v_wmma_f32_16x16x32_bf16 v[32:39], v[80:87] /*v[336:343]*/, v[112:119] /*v[624:631]*/, v[32:39]
	s_set_vgpr_msb 0x905
	v_wmma_f32_16x16x32_bf16 v[88:95], v[48:55] /*v[304:311]*/, v[184:191] /*v[440:447]*/, v[88:95]
	s_set_vgpr_msb 0x509
	v_wmma_f32_16x16x32_bf16 v[24:31], v[48:55] /*v[304:311]*/, v[112:119] /*v[624:631]*/, v[24:31]
	s_set_vgpr_msb 0x905
	v_wmma_f32_16x16x32_bf16 v[80:87], v[16:23] /*v[272:279]*/, v[184:191] /*v[440:447]*/, v[80:87]
	s_set_vgpr_msb 0x509
	v_wmma_f32_16x16x32_bf16 v[16:23], v[16:23] /*v[272:279]*/, v[112:119] /*v[624:631]*/, v[16:23]
	s_set_vgpr_msb 0x904
	v_wmma_f32_16x16x32_bf16 v[72:79], v[240:247], v[184:191] /*v[440:447]*/, v[72:79]
	s_set_vgpr_msb 0x408
	v_wmma_f32_16x16x32_bf16 v[8:15], v[240:247], v[112:119] /*v[624:631]*/, v[8:15]
	s_set_vgpr_msb 0x804
	v_wmma_f32_16x16x32_bf16 v[64:71], v[208:215], v[184:191] /*v[440:447]*/, v[64:71]
	s_set_vgpr_msb 0x408
	v_wmma_f32_16x16x32_bf16 v[0:7], v[208:215], v[112:119] /*v[624:631]*/, v[0:7]
	s_set_vgpr_msb 0x805
	v_wmma_f32_16x16x32_bf16 v[120:127], v[168:175] /*v[424:431]*/, v[176:183] /*v[432:439]*/, v[120:127]
	v_wmma_f32_16x16x32_bf16 v[56:63], v[168:175] /*v[424:431]*/, v[192:199] /*v[448:455]*/, v[56:63]
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x54a
	v_cvt_pk_bf16_f32 v175 /*v431*/, v62 /*v574*/, v64 /*v576*/
	v_cvt_pk_bf16_f32 v174 /*v430*/, v56 /*v568*/, v60 /*v572*/
	v_cvt_pk_bf16_f32 v173 /*v429*/, v48 /*v560*/, v54 /*v566*/
	s_set_vgpr_msb 0x4a05
	v_wmma_f32_16x16x32_bf16 v[112:119], v[136:143] /*v[392:399]*/, v[176:183] /*v[432:439]*/, v[112:119]
	s_set_vgpr_msb 0x54a
	v_cvt_pk_bf16_f32 v172 /*v428*/, v38 /*v550*/, v46 /*v558*/
	v_cvt_pk_bf16_f32 v171 /*v427*/, v24 /*v536*/, v34 /*v546*/
	v_cvt_pk_bf16_f32 v170 /*v426*/, v8 /*v520*/, v20 /*v532*/
	s_set_vgpr_msb 0x4a49
	v_cvt_pk_bf16_f32 v169 /*v425*/, v246 /*v502*/, v4 /*v516*/
	s_set_vgpr_msb 0x4945
	v_cvt_pk_bf16_f32 v168 /*v424*/, v230 /*v486*/, v242 /*v498*/
	s_set_vgpr_msb 0x4505
	v_wmma_f32_16x16x32_bf16 v[48:55], v[136:143] /*v[392:399]*/, v[192:199] /*v[448:455]*/, v[48:55]
	v_wmma_f32_16x16x32_bf16 v[104:111], v[104:111] /*v[360:367]*/, v[176:183] /*v[432:439]*/, v[104:111]
	v_wmma_f32_16x16x32_bf16 v[40:47], v[104:111] /*v[360:367]*/, v[192:199] /*v[448:455]*/, v[40:47]
	v_wmma_f32_16x16x32_bf16 v[96:103], v[72:79] /*v[328:335]*/, v[176:183] /*v[432:439]*/, v[96:103]
	v_wmma_f32_16x16x32_bf16 v[32:39], v[72:79] /*v[328:335]*/, v[192:199] /*v[448:455]*/, v[32:39]
	v_wmma_f32_16x16x32_bf16 v[88:95], v[40:47] /*v[296:303]*/, v[176:183] /*v[432:439]*/, v[88:95]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[40:47] /*v[296:303]*/, v[192:199] /*v[448:455]*/, v[24:31]
	v_wmma_f32_16x16x32_bf16 v[80:87], v[8:15] /*v[264:271]*/, v[176:183] /*v[432:439]*/, v[80:87]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[8:15] /*v[264:271]*/, v[192:199] /*v[448:455]*/, v[16:23]
	s_set_vgpr_msb 0x504
	v_wmma_f32_16x16x32_bf16 v[72:79], v[232:239], v[176:183] /*v[432:439]*/, v[72:79]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[232:239], v[192:199] /*v[448:455]*/, v[8:15]
	v_wmma_f32_16x16x32_bf16 v[64:71], v[200:207], v[176:183] /*v[432:439]*/, v[64:71]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[200:207], v[192:199] /*v[448:455]*/, v[0:7]
	s_set_vgpr_msb 0x405
	v_wmma_f32_16x16x32_bf16 v[120:127], v[160:167] /*v[416:423]*/, v[168:175] /*v[424:431]*/, v[120:127]
	v_wmma_f32_16x16x32_bf16 v[56:63], v[160:167] /*v[416:423]*/, v[200:207] /*v[456:463]*/, v[56:63]
	v_wmma_f32_16x16x32_bf16 v[112:119], v[128:135] /*v[384:391]*/, v[168:175] /*v[424:431]*/, v[112:119]
	v_wmma_f32_16x16x32_bf16 v[48:55], v[128:135] /*v[384:391]*/, v[200:207] /*v[456:463]*/, v[48:55]
	v_wmma_f32_16x16x32_bf16 v[104:111], v[96:103] /*v[352:359]*/, v[168:175] /*v[424:431]*/, v[104:111]
	v_wmma_f32_16x16x32_bf16 v[40:47], v[96:103] /*v[352:359]*/, v[200:207] /*v[456:463]*/, v[40:47]
	v_wmma_f32_16x16x32_bf16 v[96:103], v[64:71] /*v[320:327]*/, v[168:175] /*v[424:431]*/, v[96:103]
	v_wmma_f32_16x16x32_bf16 v[32:39], v[64:71] /*v[320:327]*/, v[200:207] /*v[456:463]*/, v[32:39]
	v_wmma_f32_16x16x32_bf16 v[88:95], v[32:39] /*v[288:295]*/, v[168:175] /*v[424:431]*/, v[88:95]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[32:39] /*v[288:295]*/, v[200:207] /*v[456:463]*/, v[24:31]
	v_wmma_f32_16x16x32_bf16 v[80:87], v[0:7] /*v[256:263]*/, v[168:175] /*v[424:431]*/, v[80:87]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[0:7] /*v[256:263]*/, v[200:207] /*v[456:463]*/, v[16:23]
	s_set_vgpr_msb 0x504
	v_wmma_f32_16x16x32_bf16 v[72:79], v[224:231], v[168:175] /*v[424:431]*/, v[72:79]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[224:231], v[200:207] /*v[456:463]*/, v[8:15]
	v_wmma_f32_16x16x32_bf16 v[64:71], v[192:199], v[168:175] /*v[424:431]*/, v[64:71]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[192:199], v[200:207] /*v[456:463]*/, v[0:7]
	s_set_vgpr_msb 0x400
	s_cbranch_vccnz .LBB0_33
.LBB0_26:
	s_set_vgpr_msb 0x82
	v_dual_mov_b32 v92 /*v604*/, v93 /*v605*/ :: v_dual_mov_b32 v93 /*v605*/, v94 /*v606*/
	s_add_co_i32 s2, s82, 1
	s_wait_tensorcnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	s_cmp_ge_i32 s2, s10
	s_set_vgpr_msb 0x8200
	s_cbranch_scc1 .LBB0_28
	s_lshl_b32 s4, s2, 7
	s_add_co_i32 s6, s82, s7
	s_sub_co_i32 s30, s9, s4
	s_add_co_i32 s2, s4, s97
	v_nop
	v_nop
	v_nop
	v_med3_i32 v192, s30, 0, 0x80
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
	s_or_b32 s46, s2, 0x7fff
	s_mov_b32 s53, s45
	tensor_load_to_lds s[28:31], s[44:51]
	s_add_nc_u64 s[30:31], s[80:81], s[4:5]
	s_or_b32 s29, vcc_hi, s6
	s_bitset1_b32 s31, 31
	s_mov_b32 s54, s46
	s_mov_b32 s55, s47
	s_mov_b32 s56, s48
	s_mov_b32 s59, s51
	tensor_load_to_lds s[28:31], s[52:59]
.LBB0_28:
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 2
	ds_load_b128 v[192:195], v85 /*v597*/
	ds_load_b128 v[196:199], v85 /*v597*/ offset:32
	ds_load_b128 v[200:203], v85 /*v597*/ offset:64
	ds_load_b128 v[204:207], v85 /*v597*/ offset:96
	ds_load_b128 v[208:211], v85 /*v597*/ offset:128
	ds_load_b128 v[212:215], v85 /*v597*/ offset:160
	ds_load_b128 v[216:219], v85 /*v597*/ offset:192
	ds_load_b128 v[220:223], v85 /*v597*/ offset:224
	ds_load_b128 v[224:227], v85 /*v597*/ offset:4352
	ds_load_b128 v[228:231], v85 /*v597*/ offset:4384
	ds_load_b128 v[232:235], v85 /*v597*/ offset:4416
	ds_load_b128 v[236:239], v85 /*v597*/ offset:4448
	ds_load_b128 v[240:243], v85 /*v597*/ offset:4480
	ds_load_b128 v[244:247], v85 /*v597*/ offset:4512
	ds_load_b128 v[248:251], v85 /*v597*/ offset:4544
	ds_load_b128 v[252:255], v85 /*v597*/ offset:4576
	s_set_vgpr_msb 0x242
	ds_load_b128 v[0:3] /*v[256:259]*/, v85 /*v597*/ offset:8704
	ds_load_b128 v[4:7] /*v[260:263]*/, v85 /*v597*/ offset:8736
	ds_load_b128 v[8:11] /*v[264:267]*/, v85 /*v597*/ offset:8768
	ds_load_b128 v[12:15] /*v[268:271]*/, v85 /*v597*/ offset:8800
	ds_load_b128 v[16:19] /*v[272:275]*/, v85 /*v597*/ offset:8832
	ds_load_b128 v[20:23] /*v[276:279]*/, v85 /*v597*/ offset:8864
	ds_load_b128 v[24:27] /*v[280:283]*/, v85 /*v597*/ offset:8896
	ds_load_b128 v[28:31] /*v[284:287]*/, v85 /*v597*/ offset:8928
	ds_load_b128 v[32:35] /*v[288:291]*/, v85 /*v597*/ offset:13056
	ds_load_b128 v[36:39] /*v[292:295]*/, v85 /*v597*/ offset:13088
	ds_load_b128 v[40:43] /*v[296:299]*/, v85 /*v597*/ offset:13120
	ds_load_b128 v[44:47] /*v[300:303]*/, v85 /*v597*/ offset:13152
	ds_load_b128 v[48:51] /*v[304:307]*/, v85 /*v597*/ offset:13184
	ds_load_b128 v[52:55] /*v[308:311]*/, v85 /*v597*/ offset:13216
	ds_load_b128 v[56:59] /*v[312:315]*/, v85 /*v597*/ offset:13248
	ds_load_b128 v[60:63] /*v[316:319]*/, v85 /*v597*/ offset:13280
	ds_load_b128 v[64:67] /*v[320:323]*/, v85 /*v597*/ offset:17408
	ds_load_b128 v[68:71] /*v[324:327]*/, v85 /*v597*/ offset:17440
	ds_load_b128 v[72:75] /*v[328:331]*/, v85 /*v597*/ offset:17472
	ds_load_b128 v[76:79] /*v[332:335]*/, v85 /*v597*/ offset:17504
	ds_load_b128 v[80:83] /*v[336:339]*/, v85 /*v597*/ offset:17536
	ds_load_b128 v[84:87] /*v[340:343]*/, v85 /*v597*/ offset:17568
	ds_load_b128 v[88:91] /*v[344:347]*/, v85 /*v597*/ offset:17600
	ds_load_b128 v[92:95] /*v[348:351]*/, v85 /*v597*/ offset:17632
	ds_load_b128 v[96:99] /*v[352:355]*/, v85 /*v597*/ offset:21760
	ds_load_b128 v[100:103] /*v[356:359]*/, v85 /*v597*/ offset:21792
	ds_load_b128 v[104:107] /*v[360:363]*/, v85 /*v597*/ offset:21824
	ds_load_b128 v[108:111] /*v[364:367]*/, v85 /*v597*/ offset:21856
	ds_load_b128 v[112:115] /*v[368:371]*/, v85 /*v597*/ offset:21888
	ds_load_b128 v[116:119] /*v[372:375]*/, v85 /*v597*/ offset:21920
	ds_load_b128 v[120:123] /*v[376:379]*/, v85 /*v597*/ offset:21952
	ds_load_b128 v[124:127] /*v[380:383]*/, v85 /*v597*/ offset:21984
	ds_load_b128 v[128:131] /*v[384:387]*/, v85 /*v597*/ offset:26112
	ds_load_b128 v[132:135] /*v[388:391]*/, v85 /*v597*/ offset:26144
	ds_load_b128 v[136:139] /*v[392:395]*/, v85 /*v597*/ offset:26176
	ds_load_b128 v[140:143] /*v[396:399]*/, v85 /*v597*/ offset:26208
	ds_load_b128 v[144:147] /*v[400:403]*/, v85 /*v597*/ offset:26240
	ds_load_b128 v[148:151] /*v[404:407]*/, v85 /*v597*/ offset:26272
	ds_load_b128 v[152:155] /*v[408:411]*/, v85 /*v597*/ offset:26304
	ds_load_b128 v[156:159] /*v[412:415]*/, v85 /*v597*/ offset:26336
	ds_load_b128 v[160:163] /*v[416:419]*/, v85 /*v597*/ offset:30464
	ds_load_b128 v[164:167] /*v[420:423]*/, v85 /*v597*/ offset:30496
	ds_load_b128 v[168:171] /*v[424:427]*/, v85 /*v597*/ offset:30528
	ds_load_b128 v[172:175] /*v[428:431]*/, v85 /*v597*/ offset:30560
	ds_load_b128 v[176:179] /*v[432:435]*/, v85 /*v597*/ offset:30592
	ds_load_b128 v[180:183] /*v[436:439]*/, v85 /*v597*/ offset:30624
	ds_load_b128 v[184:187] /*v[440:443]*/, v85 /*v597*/ offset:30656
	ds_load_b128 v[188:191] /*v[444:447]*/, v85 /*v597*/ offset:30688
	s_set_vgpr_msb 0x4240
	s_wait_dscnt 0x3e
	v_wmma_f32_16x16x32_bf16 v[250:257] /*v[506:513]*/, v[192:199], v[128:135], 0
	v_wmma_f32_16x16x32_bf16 v[192:199] /*v[448:455]*/, v[192:199], v[160:167], 0
	s_set_vgpr_msb 0x4080
	s_wait_dscnt 0x36
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[514:521]*/, v[224:231], v[128:135], 0
	s_set_vgpr_msb 0x8040
	v_wmma_f32_16x16x32_bf16 v[200:207] /*v[456:463]*/, v[224:231], v[160:167], 0
	s_set_vgpr_msb 0x4081
	s_wait_dscnt 0x2e
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[522:529]*/, v[0:7] /*v[256:263]*/, v[128:135], 0
	s_set_vgpr_msb 0x8141
	v_wmma_f32_16x16x32_bf16 v[208:215] /*v[464:471]*/, v[0:7] /*v[256:263]*/, v[160:167], 0
	s_set_vgpr_msb 0x4181
	s_wait_dscnt 0x26
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[530:537]*/, v[32:39] /*v[288:295]*/, v[128:135], 0
	s_set_vgpr_msb 0x8141
	v_wmma_f32_16x16x32_bf16 v[216:223] /*v[472:479]*/, v[32:39] /*v[288:295]*/, v[160:167], 0
	s_set_vgpr_msb 0x4181
	s_wait_dscnt 0x1e
	v_wmma_f32_16x16x32_bf16 v[26:33] /*v[538:545]*/, v[64:71] /*v[320:327]*/, v[128:135], 0
	s_set_vgpr_msb 0x8141
	v_wmma_f32_16x16x32_bf16 v[224:231] /*v[480:487]*/, v[64:71] /*v[320:327]*/, v[160:167], 0
	s_set_vgpr_msb 0x4181
	s_wait_dscnt 0x16
	v_wmma_f32_16x16x32_bf16 v[34:41] /*v[546:553]*/, v[96:103] /*v[352:359]*/, v[128:135], 0
	s_set_vgpr_msb 0x8141
	v_wmma_f32_16x16x32_bf16 v[232:239] /*v[488:495]*/, v[96:103] /*v[352:359]*/, v[160:167], 0
	s_set_vgpr_msb 0x4181
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[554:561]*/, v[128:135] /*v[384:391]*/, v[128:135], 0
	s_set_vgpr_msb 0x8141
	v_wmma_f32_16x16x32_bf16 v[240:247] /*v[496:503]*/, v[128:135] /*v[384:391]*/, v[160:167], 0
	s_set_vgpr_msb 0x4181
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[562:569]*/, v[160:167] /*v[416:423]*/, v[128:135], 0
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[570:577]*/, v[160:167] /*v[416:423]*/, v[160:167], 0
	s_set_vgpr_msb 0x8150
	v_wmma_f32_16x16x32_bf16 v[250:257] /*v[506:513]*/, v[200:207], v[136:143], v[250:257] /*v[506:513]*/
	v_wmma_f32_16x16x32_bf16 v[192:199] /*v[448:455]*/, v[200:207], v[168:175], v[192:199] /*v[448:455]*/
	s_set_vgpr_msb 0x50a0
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[514:521]*/, v[232:239], v[136:143], v[2:9] /*v[514:521]*/
	s_set_vgpr_msb 0xa050
	v_wmma_f32_16x16x32_bf16 v[200:207] /*v[456:463]*/, v[232:239], v[168:175], v[200:207] /*v[456:463]*/
	s_set_vgpr_msb 0x50a1
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[522:529]*/, v[8:15] /*v[264:271]*/, v[136:143], v[10:17] /*v[522:529]*/
	s_set_vgpr_msb 0xa151
	v_wmma_f32_16x16x32_bf16 v[208:215] /*v[464:471]*/, v[8:15] /*v[264:271]*/, v[168:175], v[208:215] /*v[464:471]*/
	s_set_vgpr_msb 0x51a1
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[530:537]*/, v[40:47] /*v[296:303]*/, v[136:143], v[18:25] /*v[530:537]*/
	s_set_vgpr_msb 0xa151
	v_wmma_f32_16x16x32_bf16 v[216:223] /*v[472:479]*/, v[40:47] /*v[296:303]*/, v[168:175], v[216:223] /*v[472:479]*/
	s_set_vgpr_msb 0x51a1
	v_wmma_f32_16x16x32_bf16 v[26:33] /*v[538:545]*/, v[72:79] /*v[328:335]*/, v[136:143], v[26:33] /*v[538:545]*/
	s_set_vgpr_msb 0xa151
	v_wmma_f32_16x16x32_bf16 v[224:231] /*v[480:487]*/, v[72:79] /*v[328:335]*/, v[168:175], v[224:231] /*v[480:487]*/
	s_set_vgpr_msb 0x51a1
	v_wmma_f32_16x16x32_bf16 v[34:41] /*v[546:553]*/, v[104:111] /*v[360:367]*/, v[136:143], v[34:41] /*v[546:553]*/
	s_set_vgpr_msb 0xa151
	v_wmma_f32_16x16x32_bf16 v[232:239] /*v[488:495]*/, v[104:111] /*v[360:367]*/, v[168:175], v[232:239] /*v[488:495]*/
	s_set_vgpr_msb 0x51a1
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[554:561]*/, v[136:143] /*v[392:399]*/, v[136:143], v[42:49] /*v[554:561]*/
	s_set_vgpr_msb 0xa151
	v_wmma_f32_16x16x32_bf16 v[240:247] /*v[496:503]*/, v[136:143] /*v[392:399]*/, v[168:175], v[240:247] /*v[496:503]*/
	s_set_vgpr_msb 0x51a1
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[562:569]*/, v[168:175] /*v[424:431]*/, v[136:143], v[50:57] /*v[562:569]*/
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[570:577]*/, v[168:175] /*v[424:431]*/, v[168:175], v[58:65] /*v[570:577]*/
	s_set_vgpr_msb 0xa150
	v_wmma_f32_16x16x32_bf16 v[250:257] /*v[506:513]*/, v[208:215], v[144:151], v[250:257] /*v[506:513]*/
	v_wmma_f32_16x16x32_bf16 v[192:199] /*v[448:455]*/, v[208:215], v[176:183], v[192:199] /*v[448:455]*/
	s_set_vgpr_msb 0x50a0
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[514:521]*/, v[240:247], v[144:151], v[2:9] /*v[514:521]*/
	s_set_vgpr_msb 0xa050
	v_wmma_f32_16x16x32_bf16 v[200:207] /*v[456:463]*/, v[240:247], v[176:183], v[200:207] /*v[456:463]*/
	s_set_vgpr_msb 0x50a1
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[522:529]*/, v[16:23] /*v[272:279]*/, v[144:151], v[10:17] /*v[522:529]*/
	s_set_vgpr_msb 0xa151
	v_wmma_f32_16x16x32_bf16 v[208:215] /*v[464:471]*/, v[16:23] /*v[272:279]*/, v[176:183], v[208:215] /*v[464:471]*/
	s_set_vgpr_msb 0x51a1
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[530:537]*/, v[48:55] /*v[304:311]*/, v[144:151], v[18:25] /*v[530:537]*/
	s_set_vgpr_msb 0xa151
	v_wmma_f32_16x16x32_bf16 v[216:223] /*v[472:479]*/, v[48:55] /*v[304:311]*/, v[176:183], v[216:223] /*v[472:479]*/
	s_set_vgpr_msb 0x51a1
	v_wmma_f32_16x16x32_bf16 v[26:33] /*v[538:545]*/, v[80:87] /*v[336:343]*/, v[144:151], v[26:33] /*v[538:545]*/
	s_set_vgpr_msb 0xa151
	v_wmma_f32_16x16x32_bf16 v[224:231] /*v[480:487]*/, v[80:87] /*v[336:343]*/, v[176:183], v[224:231] /*v[480:487]*/
	s_set_vgpr_msb 0x51a1
	v_wmma_f32_16x16x32_bf16 v[34:41] /*v[546:553]*/, v[112:119] /*v[368:375]*/, v[144:151], v[34:41] /*v[546:553]*/
	s_set_vgpr_msb 0xa151
	v_wmma_f32_16x16x32_bf16 v[232:239] /*v[488:495]*/, v[112:119] /*v[368:375]*/, v[176:183], v[232:239] /*v[488:495]*/
	s_set_vgpr_msb 0x51a1
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[554:561]*/, v[144:151] /*v[400:407]*/, v[144:151], v[42:49] /*v[554:561]*/
	s_set_vgpr_msb 0xa151
	v_wmma_f32_16x16x32_bf16 v[240:247] /*v[496:503]*/, v[144:151] /*v[400:407]*/, v[176:183], v[240:247] /*v[496:503]*/
	s_set_vgpr_msb 0x51a1
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[562:569]*/, v[176:183] /*v[432:439]*/, v[144:151], v[50:57] /*v[562:569]*/
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[570:577]*/, v[176:183] /*v[432:439]*/, v[176:183], v[58:65] /*v[570:577]*/
	s_set_vgpr_msb 0xa150
	v_wmma_f32_16x16x32_bf16 v[250:257] /*v[506:513]*/, v[216:223], v[152:159], v[250:257] /*v[506:513]*/
	v_wmma_f32_16x16x32_bf16 v[192:199] /*v[448:455]*/, v[216:223], v[184:191], v[192:199] /*v[448:455]*/
	s_set_vgpr_msb 0x50a0
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[514:521]*/, v[248:255], v[152:159], v[2:9] /*v[514:521]*/
	s_set_vgpr_msb 0xa050
	v_wmma_f32_16x16x32_bf16 v[200:207] /*v[456:463]*/, v[248:255], v[184:191], v[200:207] /*v[456:463]*/
	s_set_vgpr_msb 0x50a1
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[522:529]*/, v[24:31] /*v[280:287]*/, v[152:159], v[10:17] /*v[522:529]*/
	s_set_vgpr_msb 0xa151
	v_wmma_f32_16x16x32_bf16 v[208:215] /*v[464:471]*/, v[24:31] /*v[280:287]*/, v[184:191], v[208:215] /*v[464:471]*/
	s_set_vgpr_msb 0x51a1
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[530:537]*/, v[56:63] /*v[312:319]*/, v[152:159], v[18:25] /*v[530:537]*/
	s_set_vgpr_msb 0xa151
	v_wmma_f32_16x16x32_bf16 v[216:223] /*v[472:479]*/, v[56:63] /*v[312:319]*/, v[184:191], v[216:223] /*v[472:479]*/
	s_set_vgpr_msb 0x51a1
	v_wmma_f32_16x16x32_bf16 v[26:33] /*v[538:545]*/, v[88:95] /*v[344:351]*/, v[152:159], v[26:33] /*v[538:545]*/
	s_set_vgpr_msb 0xa151
	v_wmma_f32_16x16x32_bf16 v[224:231] /*v[480:487]*/, v[88:95] /*v[344:351]*/, v[184:191], v[224:231] /*v[480:487]*/
	s_set_vgpr_msb 0x51a1
	v_wmma_f32_16x16x32_bf16 v[34:41] /*v[546:553]*/, v[120:127] /*v[376:383]*/, v[152:159], v[34:41] /*v[546:553]*/
	s_set_vgpr_msb 0xa151
	v_wmma_f32_16x16x32_bf16 v[232:239] /*v[488:495]*/, v[120:127] /*v[376:383]*/, v[184:191], v[232:239] /*v[488:495]*/
	s_set_vgpr_msb 0x51a1
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[554:561]*/, v[152:159] /*v[408:415]*/, v[152:159], v[42:49] /*v[554:561]*/
	s_set_vgpr_msb 0xa151
	v_wmma_f32_16x16x32_bf16 v[240:247] /*v[496:503]*/, v[152:159] /*v[408:415]*/, v[184:191], v[240:247] /*v[496:503]*/
	s_set_vgpr_msb 0x51a1
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[562:569]*/, v[184:191] /*v[440:447]*/, v[152:159], v[50:57] /*v[562:569]*/
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[570:577]*/, v[184:191] /*v[440:447]*/, v[184:191], v[58:65] /*v[570:577]*/
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0xa142
	ds_load_tr16_b128 v[184:187] /*v[440:443]*/, v84 /*v596*/
	ds_load_tr16_b128 v[152:155] /*v[408:411]*/, v84 /*v596*/ offset:32
	ds_load_tr16_b128 v[188:191] /*v[444:447]*/, v84 /*v596*/ offset:4608
	ds_load_tr16_b128 v[156:159] /*v[412:415]*/, v84 /*v596*/ offset:4640
	ds_load_tr16_b128 v[176:179] /*v[432:435]*/, v84 /*v596*/ offset:9216
	ds_load_tr16_b128 v[144:147] /*v[400:403]*/, v84 /*v596*/ offset:9248
	ds_load_tr16_b128 v[180:183] /*v[436:439]*/, v84 /*v596*/ offset:13824
	ds_load_tr16_b128 v[148:151] /*v[404:407]*/, v84 /*v596*/ offset:13856
	ds_load_tr16_b128 v[168:171] /*v[424:427]*/, v84 /*v596*/ offset:18432
	ds_load_tr16_b128 v[136:139] /*v[392:395]*/, v84 /*v596*/ offset:18464
	ds_load_tr16_b128 v[172:175] /*v[428:431]*/, v84 /*v596*/ offset:23040
	ds_load_tr16_b128 v[140:143] /*v[396:399]*/, v84 /*v596*/ offset:23072
	ds_load_tr16_b128 v[160:163] /*v[416:419]*/, v84 /*v596*/ offset:27648
	ds_load_tr16_b128 v[128:131] /*v[384:387]*/, v84 /*v596*/ offset:27680
	ds_load_tr16_b128 v[164:167] /*v[420:423]*/, v84 /*v596*/ offset:32256
	ds_load_tr16_b128 v[132:135] /*v[388:391]*/, v84 /*v596*/ offset:32288
	ds_load_tr16_b128 v[120:123] /*v[376:379]*/, v84 /*v596*/ offset:64
	ds_load_tr16_b128 v[88:91] /*v[344:347]*/, v84 /*v596*/ offset:96
	ds_load_tr16_b128 v[124:127] /*v[380:383]*/, v84 /*v596*/ offset:4672
	ds_load_tr16_b128 v[92:95] /*v[348:351]*/, v84 /*v596*/ offset:4704
	ds_load_tr16_b128 v[112:115] /*v[368:371]*/, v84 /*v596*/ offset:9280
	ds_load_tr16_b128 v[80:83] /*v[336:339]*/, v84 /*v596*/ offset:9312
	ds_load_tr16_b128 v[116:119] /*v[372:375]*/, v84 /*v596*/ offset:13888
	ds_load_tr16_b128 v[84:87] /*v[340:343]*/, v84 /*v596*/ offset:13920
	ds_load_tr16_b128 v[104:107] /*v[360:363]*/, v84 /*v596*/ offset:18496
	ds_load_tr16_b128 v[72:75] /*v[328:331]*/, v84 /*v596*/ offset:18528
	ds_load_tr16_b128 v[108:111] /*v[364:367]*/, v84 /*v596*/ offset:23104
	ds_load_tr16_b128 v[76:79] /*v[332:335]*/, v84 /*v596*/ offset:23136
	ds_load_tr16_b128 v[96:99] /*v[352:355]*/, v84 /*v596*/ offset:27712
	ds_load_tr16_b128 v[64:67] /*v[320:323]*/, v84 /*v596*/ offset:27744
	ds_load_tr16_b128 v[100:103] /*v[356:359]*/, v84 /*v596*/ offset:32320
	ds_load_tr16_b128 v[68:71] /*v[324:327]*/, v84 /*v596*/ offset:32352
	ds_load_tr16_b128 v[56:59] /*v[312:315]*/, v84 /*v596*/ offset:128
	ds_load_tr16_b128 v[24:27] /*v[280:283]*/, v84 /*v596*/ offset:160
	ds_load_tr16_b128 v[60:63] /*v[316:319]*/, v84 /*v596*/ offset:4736
	ds_load_tr16_b128 v[28:31] /*v[284:287]*/, v84 /*v596*/ offset:4768
	ds_load_tr16_b128 v[48:51] /*v[304:307]*/, v84 /*v596*/ offset:9344
	ds_load_tr16_b128 v[16:19] /*v[272:275]*/, v84 /*v596*/ offset:9376
	ds_load_tr16_b128 v[52:55] /*v[308:311]*/, v84 /*v596*/ offset:13952
	ds_load_tr16_b128 v[20:23] /*v[276:279]*/, v84 /*v596*/ offset:13984
	ds_load_tr16_b128 v[40:43] /*v[296:299]*/, v84 /*v596*/ offset:18560
	ds_load_tr16_b128 v[8:11] /*v[264:267]*/, v84 /*v596*/ offset:18592
	ds_load_tr16_b128 v[44:47] /*v[300:303]*/, v84 /*v596*/ offset:23168
	ds_load_tr16_b128 v[12:15] /*v[268:271]*/, v84 /*v596*/ offset:23200
	ds_load_tr16_b128 v[32:35] /*v[288:291]*/, v84 /*v596*/ offset:27776
	ds_load_tr16_b128 v[0:3] /*v[256:259]*/, v84 /*v596*/ offset:27808
	ds_load_tr16_b128 v[36:39] /*v[292:295]*/, v84 /*v596*/ offset:32384
	ds_load_tr16_b128 v[4:7] /*v[260:263]*/, v84 /*v596*/ offset:32416
	s_set_vgpr_msb 0x4202
	ds_load_tr16_b128 v[248:251], v84 /*v596*/ offset:192
	ds_load_tr16_b128 v[216:219], v84 /*v596*/ offset:224
	ds_load_tr16_b128 v[252:255], v84 /*v596*/ offset:4800
	ds_load_tr16_b128 v[220:223], v84 /*v596*/ offset:4832
	ds_load_tr16_b128 v[240:243], v84 /*v596*/ offset:9408
	ds_load_tr16_b128 v[208:211], v84 /*v596*/ offset:9440
	ds_load_tr16_b128 v[244:247], v84 /*v596*/ offset:14016
	ds_load_tr16_b128 v[212:215], v84 /*v596*/ offset:14048
	ds_load_tr16_b128 v[232:235], v84 /*v596*/ offset:18624
	ds_load_tr16_b128 v[200:203], v84 /*v596*/ offset:18656
	ds_load_tr16_b128 v[236:239], v84 /*v596*/ offset:23232
	ds_load_tr16_b128 v[204:207], v84 /*v596*/ offset:23264
	ds_load_tr16_b128 v[224:227], v84 /*v596*/ offset:27840
	ds_load_tr16_b128 v[192:195], v84 /*v596*/ offset:27872
	ds_load_tr16_b128 v[228:231], v84 /*v596*/ offset:32448
	ds_load_tr16_b128 v[196:199], v84 /*v596*/ offset:32480
	s_set_vgpr_msb 0x2aa
	v_lshl_or_b32 v68 /*v580*/, s82, 7, v79 /*v591*/
	v_cmp_gt_i32_e32 vcc_lo, v68 /*v580*/, v90 /*v602*/
	v_cmp_lt_i32_e64 s2, v68 /*v580*/, v88 /*v600*/
	v_dual_add_nc_u32 v116 /*v628*/, 16, v68 /*v580*/ :: v_dual_bitop2_b32 v69 /*v581*/, 1, v68 /*v580*/ bitop3:0x54
	v_dual_add_nc_u32 v117 /*v629*/, 17, v68 /*v580*/ :: v_dual_bitop2_b32 v70 /*v582*/, 2, v68 /*v580*/ bitop3:0x54
	v_cmp_ge_i32_e64 s3, v68 /*v580*/, v90 /*v602*/
	v_cmp_lt_i32_e64 s4, v69 /*v581*/, v88 /*v600*/
	s_or_b32 s2, s2, vcc_lo
	v_dual_add_nc_u32 v118 /*v630*/, 18, v68 /*v580*/ :: v_dual_bitop2_b32 v71 /*v583*/, 3, v68 /*v580*/ bitop3:0x54
	s_set_vgpr_msb 0xaa41
	v_cndmask_b32_e64 v250 /*v506*/, v250 /*v506*/, 0xff800000, s2
	s_set_vgpr_msb 0x418a
	v_cmp_gt_i32_e32 vcc_lo, v70 /*v582*/, v90 /*v602*/
	v_cmp_lt_i32_e64 s2, v70 /*v582*/, v88 /*v600*/
	v_dual_add_nc_u32 v119 /*v631*/, 19, v68 /*v580*/ :: v_dual_bitop2_b32 v112 /*v624*/, 4, v68 /*v580*/ bitop3:0x54
	s_or_b32 s3, s4, s3
	v_cmp_gt_i32_e64 s5, v71 /*v583*/, v90 /*v602*/
	s_set_vgpr_msb 0x8a41
	v_cndmask_b32_e64 v251 /*v507*/, v251 /*v507*/, 0xff800000, s3
	s_set_vgpr_msb 0x418a
	v_cmp_lt_i32_e64 s3, v71 /*v583*/, v88 /*v600*/
	s_or_b32 s2, s2, vcc_lo
	v_dual_add_nc_u32 v120 /*v632*/, 20, v68 /*v580*/ :: v_dual_bitop2_b32 v113 /*v625*/, 5, v68 /*v580*/ bitop3:0x54
	s_set_vgpr_msb 0x8a41
	v_cndmask_b32_e64 v252 /*v508*/, v252 /*v508*/, 0xff800000, s2
	s_set_vgpr_msb 0x418a
	v_cmp_gt_i32_e32 vcc_lo, v112 /*v624*/, v90 /*v602*/
	v_cmp_lt_i32_e64 s2, v112 /*v624*/, v88 /*v600*/
	v_dual_add_nc_u32 v121 /*v633*/, 21, v68 /*v580*/ :: v_dual_bitop2_b32 v114 /*v626*/, 6, v68 /*v580*/ bitop3:0x54
	s_or_b32 s3, s3, s5
	v_cmp_lt_i32_e64 s4, v113 /*v625*/, v88 /*v600*/
	s_set_vgpr_msb 0x8a41
	v_cndmask_b32_e64 v253 /*v509*/, v253 /*v509*/, 0xff800000, s3
	s_set_vgpr_msb 0x418a
	v_cmp_gt_i32_e64 s3, v113 /*v625*/, v90 /*v602*/
	s_or_b32 s2, s2, vcc_lo
	v_dual_add_nc_u32 v122 /*v634*/, 22, v68 /*v580*/ :: v_dual_bitop2_b32 v115 /*v627*/, 7, v68 /*v580*/ bitop3:0x54
	s_set_vgpr_msb 0x8a41
	v_cndmask_b32_e64 v254 /*v510*/, v254 /*v510*/, 0xff800000, s2
	s_set_vgpr_msb 0x410a
	v_cmp_gt_i32_e32 vcc_lo, v114 /*v626*/, v90 /*v602*/
	v_cmp_lt_i32_e64 s2, v114 /*v626*/, v88 /*v600*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s4, v115 /*v627*/, v88 /*v600*/
	s_set_vgpr_msb 0xa41
	v_cndmask_b32_e64 v255 /*v511*/, v255 /*v511*/, 0xff800000, s3
	s_set_vgpr_msb 0x418a
	v_cmp_gt_i32_e64 s3, v115 /*v627*/, v90 /*v602*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v116 /*v628*/, v90 /*v602*/
	v_cndmask_b32_e64 v0 /*v512*/, v0 /*v512*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v116 /*v628*/, v88 /*v600*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s4, v117 /*v629*/, v88 /*v600*/
	v_cndmask_b32_e64 v1 /*v513*/, v1 /*v513*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v117 /*v629*/, v90 /*v602*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v118 /*v630*/, v90 /*v602*/
	v_cndmask_b32_e64 v2 /*v514*/, v2 /*v514*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v118 /*v630*/, v88 /*v600*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s4, v119 /*v631*/, v88 /*v600*/
	v_cndmask_b32_e64 v3 /*v515*/, v3 /*v515*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v119 /*v631*/, v90 /*v602*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v120 /*v632*/, v90 /*v602*/
	v_cndmask_b32_e64 v4 /*v516*/, v4 /*v516*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v120 /*v632*/, v88 /*v600*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s4, v121 /*v633*/, v88 /*v600*/
	v_cndmask_b32_e64 v5 /*v517*/, v5 /*v517*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v121 /*v633*/, v90 /*v602*/
	s_or_b32 s2, s2, vcc_lo
	v_dual_add_nc_u32 v123 /*v635*/, 23, v68 /*v580*/ :: v_dual_bitop2_b32 v124 /*v636*/, 32, v68 /*v580*/ bitop3:0x54
	v_cndmask_b32_e64 v6 /*v518*/, v6 /*v518*/, 0xff800000, s2
	v_cmp_gt_i32_e32 vcc_lo, v122 /*v634*/, v90 /*v602*/
	v_cmp_lt_i32_e64 s2, v122 /*v634*/, v88 /*v600*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s4, v123 /*v635*/, v88 /*v600*/
	v_cndmask_b32_e64 v7 /*v519*/, v7 /*v519*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v123 /*v635*/, v90 /*v602*/
	s_or_b32 s2, s2, vcc_lo
	v_or_b32_e32 v125 /*v637*/, 33, v68 /*v580*/
	v_cndmask_b32_e64 v8 /*v520*/, v8 /*v520*/, 0xff800000, s2
	v_cmp_gt_i32_e32 vcc_lo, v124 /*v636*/, v90 /*v602*/
	v_cmp_lt_i32_e64 s2, v124 /*v636*/, v88 /*v600*/
	v_dual_add_nc_u32 v133 /*v645*/, 49, v68 /*v580*/ :: v_dual_bitop2_b32 v126 /*v638*/, 34, v68 /*v580*/ bitop3:0x54
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s4, v125 /*v637*/, v88 /*v600*/
	v_cndmask_b32_e64 v9 /*v521*/, v9 /*v521*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v125 /*v637*/, v90 /*v602*/
	s_or_b32 s2, s2, vcc_lo
	v_dual_add_nc_u32 v134 /*v646*/, 50, v68 /*v580*/ :: v_dual_bitop2_b32 v127 /*v639*/, 35, v68 /*v580*/ bitop3:0x54
	v_cndmask_b32_e64 v10 /*v522*/, v10 /*v522*/, 0xff800000, s2
	v_cmp_gt_i32_e32 vcc_lo, v126 /*v638*/, v90 /*v602*/
	v_cmp_lt_i32_e64 s2, v126 /*v638*/, v88 /*v600*/
	v_dual_add_nc_u32 v135 /*v647*/, 51, v68 /*v580*/ :: v_dual_bitop2_b32 v128 /*v640*/, 36, v68 /*v580*/ bitop3:0x54
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s4, v127 /*v639*/, v88 /*v600*/
	v_cndmask_b32_e64 v11 /*v523*/, v11 /*v523*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v127 /*v639*/, v90 /*v602*/
	s_or_b32 s2, s2, vcc_lo
	v_dual_add_nc_u32 v136 /*v648*/, 52, v68 /*v580*/ :: v_dual_bitop2_b32 v129 /*v641*/, 37, v68 /*v580*/ bitop3:0x54
	v_cndmask_b32_e64 v12 /*v524*/, v12 /*v524*/, 0xff800000, s2
	v_cmp_gt_i32_e32 vcc_lo, v128 /*v640*/, v90 /*v602*/
	v_cmp_lt_i32_e64 s2, v128 /*v640*/, v88 /*v600*/
	s_or_b32 s3, s4, s3
	v_dual_add_nc_u32 v137 /*v649*/, 53, v68 /*v580*/ :: v_dual_bitop2_b32 v130 /*v642*/, 38, v68 /*v580*/ bitop3:0x54
	v_cndmask_b32_e64 v13 /*v525*/, v13 /*v525*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v129 /*v641*/, v90 /*v602*/
	v_cmp_lt_i32_e64 s4, v129 /*v641*/, v88 /*v600*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v130 /*v642*/, v90 /*v602*/
	v_cndmask_b32_e64 v66 /*v578*/, v14 /*v526*/, 0xff800000, s2
	v_dual_add_nc_u32 v138 /*v650*/, 54, v68 /*v580*/ :: v_dual_bitop2_b32 v14 /*v526*/, 39, v68 /*v580*/ bitop3:0x54
	v_cmp_lt_i32_e64 s2, v130 /*v642*/, v88 /*v600*/
	s_or_b32 s3, s4, s3
	v_dual_add_nc_u32 v139 /*v651*/, 55, v68 /*v580*/ :: v_dual_bitop2_b32 v140 /*v652*/, 64, v68 /*v580*/ bitop3:0x54
	v_cndmask_b32_e64 v67 /*v579*/, v15 /*v527*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v14 /*v526*/, v90 /*v602*/
	v_add_nc_u32_e32 v15 /*v527*/, 48, v68 /*v580*/
	v_cmp_lt_i32_e64 s4, v14 /*v526*/, v88 /*v600*/
	s_or_b32 s2, s2, vcc_lo
	v_or_b32_e32 v141 /*v653*/, 0x41, v68 /*v580*/
	v_cndmask_b32_e64 v16 /*v528*/, v16 /*v528*/, 0xff800000, s2
	v_cmp_gt_i32_e32 vcc_lo, v15 /*v527*/, v90 /*v602*/
	v_cmp_lt_i32_e64 s2, v15 /*v527*/, v88 /*v600*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s4, v133 /*v645*/, v88 /*v600*/
	v_cndmask_b32_e64 v17 /*v529*/, v17 /*v529*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v133 /*v645*/, v90 /*v602*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v134 /*v646*/, v90 /*v602*/
	v_cndmask_b32_e64 v18 /*v530*/, v18 /*v530*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v134 /*v646*/, v88 /*v600*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s4, v135 /*v647*/, v88 /*v600*/
	v_cndmask_b32_e64 v19 /*v531*/, v19 /*v531*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v135 /*v647*/, v90 /*v602*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v136 /*v648*/, v90 /*v602*/
	v_cndmask_b32_e64 v20 /*v532*/, v20 /*v532*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v136 /*v648*/, v88 /*v600*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s4, v137 /*v649*/, v88 /*v600*/
	v_cndmask_b32_e64 v21 /*v533*/, v21 /*v533*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v137 /*v649*/, v90 /*v602*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v138 /*v650*/, v90 /*v602*/
	v_cndmask_b32_e64 v22 /*v534*/, v22 /*v534*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v138 /*v650*/, v88 /*v600*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s4, v139 /*v651*/, v88 /*v600*/
	v_cndmask_b32_e64 v23 /*v535*/, v23 /*v535*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v139 /*v651*/, v90 /*v602*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v140 /*v652*/, v90 /*v602*/
	v_cndmask_b32_e64 v24 /*v536*/, v24 /*v536*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v140 /*v652*/, v88 /*v600*/
	s_or_b32 s3, s4, s3
	v_or_b32_e32 v142 /*v654*/, 0x42, v68 /*v580*/
	v_cndmask_b32_e64 v25 /*v537*/, v25 /*v537*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v141 /*v653*/, v90 /*v602*/
	v_cmp_lt_i32_e64 s4, v141 /*v653*/, v88 /*v600*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v142 /*v654*/, v90 /*v602*/
	v_cndmask_b32_e64 v94 /*v606*/, v26 /*v538*/, 0xff800000, s2
	v_or_b32_e32 v26 /*v538*/, 0x43, v68 /*v580*/
	v_cmp_lt_i32_e64 s2, v142 /*v654*/, v88 /*v600*/
	s_or_b32 s3, s4, s3
	v_or_b32_e32 v145 /*v657*/, 0x45, v68 /*v580*/
	v_cndmask_b32_e64 v95 /*v607*/, v27 /*v539*/, 0xff800000, s3
	v_or_b32_e32 v27 /*v539*/, 0x44, v68 /*v580*/
	v_cmp_gt_i32_e64 s3, v26 /*v538*/, v90 /*v602*/
	v_cmp_lt_i32_e64 s4, v26 /*v538*/, v88 /*v600*/
	s_or_b32 s2, s2, vcc_lo
	v_or_b32_e32 v146 /*v658*/, 0x46, v68 /*v580*/
	v_cndmask_b32_e64 v28 /*v540*/, v28 /*v540*/, 0xff800000, s2
	v_cmp_gt_i32_e32 vcc_lo, v27 /*v539*/, v90 /*v602*/
	v_cmp_lt_i32_e64 s2, v27 /*v539*/, v88 /*v600*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s4, v145 /*v657*/, v88 /*v600*/
	v_cndmask_b32_e64 v29 /*v541*/, v29 /*v541*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v145 /*v657*/, v90 /*v602*/
	s_or_b32 s2, s2, vcc_lo
	v_or_b32_e32 v147 /*v659*/, 0x47, v68 /*v580*/
	v_cndmask_b32_e64 v30 /*v542*/, v30 /*v542*/, 0xff800000, s2
	v_cmp_gt_i32_e32 vcc_lo, v146 /*v658*/, v90 /*v602*/
	v_cmp_lt_i32_e64 s2, v146 /*v658*/, v88 /*v600*/
	s_or_b32 s3, s4, s3
	v_add_nc_u32_e32 v148 /*v660*/, 0x50, v68 /*v580*/
	v_cndmask_b32_e64 v31 /*v543*/, v31 /*v543*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v147 /*v659*/, v90 /*v602*/
	v_cmp_lt_i32_e64 s4, v147 /*v659*/, v88 /*v600*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v148 /*v660*/, v90 /*v602*/
	v_cndmask_b32_e64 v96 /*v608*/, v32 /*v544*/, 0xff800000, s2
	v_add_nc_u32_e32 v32 /*v544*/, 0x51, v68 /*v580*/
	v_cmp_lt_i32_e64 s2, v148 /*v660*/, v88 /*v600*/
	s_or_b32 s3, s4, s3
	v_add_nc_u32_e32 v151 /*v663*/, 0x53, v68 /*v580*/
	v_cndmask_b32_e64 v97 /*v609*/, v33 /*v545*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v32 /*v544*/, v90 /*v602*/
	v_add_nc_u32_e32 v33 /*v545*/, 0x52, v68 /*v580*/
	v_cmp_lt_i32_e64 s4, v32 /*v544*/, v88 /*v600*/
	s_or_b32 s2, s2, vcc_lo
	v_add_nc_u32_e32 v152 /*v664*/, 0x54, v68 /*v580*/
	v_cndmask_b32_e64 v34 /*v546*/, v34 /*v546*/, 0xff800000, s2
	v_cmp_gt_i32_e32 vcc_lo, v33 /*v545*/, v90 /*v602*/
	v_cmp_lt_i32_e64 s2, v33 /*v545*/, v88 /*v600*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s4, v151 /*v663*/, v88 /*v600*/
	v_cndmask_b32_e64 v35 /*v547*/, v35 /*v547*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v151 /*v663*/, v90 /*v602*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v152 /*v664*/, v90 /*v602*/
	v_cndmask_b32_e64 v98 /*v610*/, v36 /*v548*/, 0xff800000, s2
	v_add_nc_u32_e32 v36 /*v548*/, 0x55, v68 /*v580*/
	v_cmp_lt_i32_e64 s2, v152 /*v664*/, v88 /*v600*/
	s_or_b32 s3, s4, s3
	v_add_nc_u32_e32 v154 /*v666*/, 0x57, v68 /*v580*/
	v_cndmask_b32_e64 v99 /*v611*/, v37 /*v549*/, 0xff800000, s3
	v_add_nc_u32_e32 v37 /*v549*/, 0x56, v68 /*v580*/
	v_cmp_gt_i32_e64 s3, v36 /*v548*/, v90 /*v602*/
	v_cmp_lt_i32_e64 s4, v36 /*v548*/, v88 /*v600*/
	s_or_b32 s2, s2, vcc_lo
	v_or_b32_e32 v155 /*v667*/, 0x60, v68 /*v580*/
	v_cndmask_b32_e64 v38 /*v550*/, v38 /*v550*/, 0xff800000, s2
	v_cmp_gt_i32_e32 vcc_lo, v37 /*v549*/, v90 /*v602*/
	v_cmp_lt_i32_e64 s2, v37 /*v549*/, v88 /*v600*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s4, v154 /*v666*/, v88 /*v600*/
	v_cndmask_b32_e64 v39 /*v551*/, v39 /*v551*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v154 /*v666*/, v90 /*v602*/
	s_or_b32 s2, s2, vcc_lo
	v_or_b32_e32 v157 /*v669*/, 0x61, v68 /*v580*/
	v_cndmask_b32_e64 v40 /*v552*/, v40 /*v552*/, 0xff800000, s2
	v_cmp_gt_i32_e32 vcc_lo, v155 /*v667*/, v90 /*v602*/
	v_cmp_lt_i32_e64 s2, v155 /*v667*/, v88 /*v600*/
	s_or_b32 s3, s4, s3
	v_or_b32_e32 v158 /*v670*/, 0x62, v68 /*v580*/
	v_cndmask_b32_e64 v41 /*v553*/, v41 /*v553*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v157 /*v669*/, v90 /*v602*/
	v_cmp_lt_i32_e64 s4, v157 /*v669*/, v88 /*v600*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v158 /*v670*/, v90 /*v602*/
	v_cndmask_b32_e64 v100 /*v612*/, v42 /*v554*/, 0xff800000, s2
	v_or_b32_e32 v42 /*v554*/, 0x63, v68 /*v580*/
	v_cmp_lt_i32_e64 s2, v158 /*v670*/, v88 /*v600*/
	s_or_b32 s3, s4, s3
	v_or_b32_e32 v163 /*v675*/, 0x67, v68 /*v580*/
	v_cndmask_b32_e64 v101 /*v613*/, v43 /*v555*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v42 /*v554*/, v90 /*v602*/
	v_or_b32_e32 v43 /*v555*/, 0x64, v68 /*v580*/
	v_cmp_lt_i32_e64 s4, v42 /*v554*/, v88 /*v600*/
	s_or_b32 s2, s2, vcc_lo
	v_add_nc_u32_e32 v164 /*v676*/, 0x70, v68 /*v580*/
	v_cndmask_b32_e64 v102 /*v614*/, v44 /*v556*/, 0xff800000, s2
	v_or_b32_e32 v44 /*v556*/, 0x65, v68 /*v580*/
	v_cmp_gt_i32_e32 vcc_lo, v43 /*v555*/, v90 /*v602*/
	v_cmp_lt_i32_e64 s2, v43 /*v555*/, v88 /*v600*/
	s_or_b32 s3, s4, s3
	v_add_nc_u32_e32 v165 /*v677*/, 0x71, v68 /*v580*/
	v_cndmask_b32_e64 v103 /*v615*/, v45 /*v557*/, 0xff800000, s3
	v_or_b32_e32 v45 /*v557*/, 0x66, v68 /*v580*/
	v_cmp_gt_i32_e64 s3, v44 /*v556*/, v90 /*v602*/
	v_cmp_lt_i32_e64 s4, v44 /*v556*/, v88 /*v600*/
	s_or_b32 s2, s2, vcc_lo
	v_add_nc_u32_e32 v166 /*v678*/, 0x72, v68 /*v580*/
	v_cndmask_b32_e64 v46 /*v558*/, v46 /*v558*/, 0xff800000, s2
	v_cmp_gt_i32_e32 vcc_lo, v45 /*v557*/, v90 /*v602*/
	v_cmp_lt_i32_e64 s2, v45 /*v557*/, v88 /*v600*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s4, v163 /*v675*/, v88 /*v600*/
	v_cndmask_b32_e64 v47 /*v559*/, v47 /*v559*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v163 /*v675*/, v90 /*v602*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v164 /*v676*/, v90 /*v602*/
	v_cndmask_b32_e64 v48 /*v560*/, v48 /*v560*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v164 /*v676*/, v88 /*v600*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s4, v165 /*v677*/, v88 /*v600*/
	v_cndmask_b32_e64 v49 /*v561*/, v49 /*v561*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v165 /*v677*/, v90 /*v602*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v166 /*v678*/, v90 /*v602*/
	v_cndmask_b32_e64 v104 /*v616*/, v50 /*v562*/, 0xff800000, s2
	v_add_nc_u32_e32 v50 /*v562*/, 0x73, v68 /*v580*/
	v_cmp_lt_i32_e64 s2, v166 /*v678*/, v88 /*v600*/
	s_or_b32 s3, s4, s3
	v_add_nc_u32_e32 v169 /*v681*/, 0x77, v68 /*v580*/
	v_cndmask_b32_e64 v105 /*v617*/, v51 /*v563*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v50 /*v562*/, v90 /*v602*/
	v_add_nc_u32_e32 v51 /*v563*/, 0x74, v68 /*v580*/
	v_cmp_lt_i32_e64 s4, v50 /*v562*/, v88 /*v600*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e64 s5, v68 /*v580*/, v91 /*v603*/
	v_cndmask_b32_e64 v106 /*v618*/, v52 /*v564*/, 0xff800000, s2
	v_add_nc_u32_e32 v52 /*v564*/, 0x75, v68 /*v580*/
	v_cmp_gt_i32_e32 vcc_lo, v51 /*v563*/, v90 /*v602*/
	v_cmp_lt_i32_e64 s2, v51 /*v563*/, v88 /*v600*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s6, v68 /*v580*/, v89 /*v601*/
	v_cndmask_b32_e64 v107 /*v619*/, v53 /*v565*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v52 /*v564*/, v90 /*v602*/
	v_cmp_lt_i32_e64 s4, v52 /*v564*/, v88 /*v600*/
	v_add_nc_u32_e32 v53 /*v565*/, 0x76, v68 /*v580*/
	s_or_b32 s2, s2, vcc_lo
	v_cndmask_b32_e64 v54 /*v566*/, v54 /*v566*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	v_cmp_gt_i32_e32 vcc_lo, v53 /*v565*/, v90 /*v602*/
	v_cndmask_b32_e64 v55 /*v567*/, v55 /*v567*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v53 /*v565*/, v88 /*v600*/
	v_cmp_gt_i32_e64 s3, v169 /*v681*/, v90 /*v602*/
	v_cmp_lt_i32_e64 s4, v169 /*v681*/, v88 /*v600*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_ge_i32_e32 vcc_lo, v68 /*v580*/, v91 /*v603*/
	v_cndmask_b32_e64 v56 /*v568*/, v56 /*v568*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	v_cmp_gt_i32_e64 s3, v70 /*v582*/, v91 /*v603*/
	v_cndmask_b32_e64 v57 /*v569*/, v57 /*v569*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	v_cmp_lt_i32_e64 s4, v70 /*v582*/, v89 /*v601*/
	s_set_vgpr_msb 0x8a81
	v_cndmask_b32_e64 v108 /*v620*/, v192 /*v448*/, 0xff800000, s2
	s_set_vgpr_msb 0x810a
	v_cmp_lt_i32_e64 s2, v69 /*v581*/, v89 /*v601*/
	v_cmp_gt_i32_e64 s5, v71 /*v583*/, v91 /*v603*/
	v_cmp_lt_i32_e64 s6, v71 /*v583*/, v89 /*v601*/
	s_set_vgpr_msb 0xa55
	v_max3_num_f32 v192 /*v448*/, v250 /*v506*/, v251 /*v507*/, v252 /*v508*/
	s_or_b32 s2, s2, vcc_lo
	s_set_vgpr_msb 0x550a
	v_cmp_gt_i32_e32 vcc_lo, v112 /*v624*/, v91 /*v603*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v109 /*v621*/, v193 /*v449*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	s_set_vgpr_msb 0x810a
	v_cmp_gt_i32_e64 s3, v113 /*v625*/, v91 /*v603*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v110 /*v622*/, v194 /*v450*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	s_set_vgpr_msb 0x810a
	v_cmp_lt_i32_e64 s4, v113 /*v625*/, v89 /*v601*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v111 /*v623*/, v195 /*v451*/, 0xff800000, s2
	s_set_vgpr_msb 0x816a
	v_cmp_lt_i32_e64 s2, v112 /*v624*/, v89 /*v601*/
	v_cmp_gt_i32_e64 s5, v114 /*v626*/, v91 /*v603*/
	v_cmp_lt_i32_e64 s6, v114 /*v626*/, v89 /*v601*/
	v_max3_num_f32 v193 /*v449*/, v108 /*v620*/, v109 /*v621*/, v110 /*v622*/
	s_set_vgpr_msb 0x6a55
	v_max3_num_f32 v194 /*v450*/, v253 /*v509*/, v254 /*v510*/, v255 /*v511*/
	s_or_b32 s2, s2, vcc_lo
	s_set_vgpr_msb 0x550a
	v_cmp_gt_i32_e32 vcc_lo, v115 /*v627*/, v91 /*v603*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v112 /*v624*/, v196 /*v452*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	s_set_vgpr_msb 0x810a
	v_cmp_gt_i32_e64 s3, v116 /*v628*/, v91 /*v603*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v113 /*v625*/, v197 /*v453*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	s_set_vgpr_msb 0x810a
	v_cmp_lt_i32_e64 s4, v116 /*v628*/, v89 /*v601*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v114 /*v626*/, v198 /*v454*/, 0xff800000, s2
	s_set_vgpr_msb 0x816a
	v_cmp_lt_i32_e64 s2, v115 /*v627*/, v89 /*v601*/
	v_cmp_gt_i32_e64 s5, v117 /*v629*/, v91 /*v603*/
	v_cmp_lt_i32_e64 s6, v117 /*v629*/, v89 /*v601*/
	v_max3_num_f32 v195 /*v451*/, v111 /*v623*/, v112 /*v624*/, v113 /*v625*/
	v_max3_num_f32 v196 /*v452*/, v0 /*v512*/, v1 /*v513*/, v2 /*v514*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v118 /*v630*/, v91 /*v603*/
	s_set_vgpr_msb 0x6a81
	v_cndmask_b32_e64 v115 /*v627*/, v199 /*v455*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	s_set_vgpr_msb 0x810a
	v_cmp_gt_i32_e64 s3, v119 /*v631*/, v91 /*v603*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v116 /*v628*/, v200 /*v456*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	s_set_vgpr_msb 0x810a
	v_cmp_lt_i32_e64 s4, v119 /*v631*/, v89 /*v601*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v117 /*v629*/, v201 /*v457*/, 0xff800000, s2
	s_set_vgpr_msb 0x816a
	v_cmp_lt_i32_e64 s2, v118 /*v630*/, v89 /*v601*/
	v_cmp_gt_i32_e64 s5, v120 /*v632*/, v91 /*v603*/
	v_cmp_lt_i32_e64 s6, v120 /*v632*/, v89 /*v601*/
	v_max3_num_f32 v197 /*v453*/, v114 /*v626*/, v115 /*v627*/, v116 /*v628*/
	v_max3_num_f32 v198 /*v454*/, v3 /*v515*/, v4 /*v516*/, v5 /*v517*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v121 /*v633*/, v91 /*v603*/
	s_set_vgpr_msb 0x6a81
	v_cndmask_b32_e64 v118 /*v630*/, v202 /*v458*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	s_set_vgpr_msb 0x810a
	v_cmp_gt_i32_e64 s3, v122 /*v634*/, v91 /*v603*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v119 /*v631*/, v203 /*v459*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	s_set_vgpr_msb 0x810a
	v_cmp_lt_i32_e64 s4, v122 /*v634*/, v89 /*v601*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v120 /*v632*/, v204 /*v460*/, 0xff800000, s2
	s_set_vgpr_msb 0x816a
	v_cmp_lt_i32_e64 s2, v121 /*v633*/, v89 /*v601*/
	v_cmp_gt_i32_e64 s5, v123 /*v635*/, v91 /*v603*/
	v_cmp_lt_i32_e64 s6, v123 /*v635*/, v89 /*v601*/
	v_max3_num_f32 v200 /*v456*/, v6 /*v518*/, v7 /*v519*/, v8 /*v520*/
	v_max3_num_f32 v202 /*v458*/, v9 /*v521*/, v10 /*v522*/, v11 /*v523*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v124 /*v636*/, v91 /*v603*/
	s_set_vgpr_msb 0x6a81
	v_cndmask_b32_e64 v121 /*v633*/, v205 /*v461*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	s_set_vgpr_msb 0x810a
	v_cmp_gt_i32_e64 s3, v125 /*v637*/, v91 /*v603*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v122 /*v634*/, v206 /*v462*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	s_set_vgpr_msb 0x810a
	v_cmp_lt_i32_e64 s4, v125 /*v637*/, v89 /*v601*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v123 /*v635*/, v207 /*v463*/, 0xff800000, s2
	s_set_vgpr_msb 0x816a
	v_cmp_lt_i32_e64 s2, v124 /*v636*/, v89 /*v601*/
	v_cmp_gt_i32_e64 s5, v126 /*v638*/, v91 /*v603*/
	v_cmp_lt_i32_e64 s6, v126 /*v638*/, v89 /*v601*/
	v_max3_num_f32 v204 /*v460*/, v12 /*v524*/, v13 /*v525*/, v66 /*v578*/
	v_max3_num_f32 v206 /*v462*/, v67 /*v579*/, v16 /*v528*/, v17 /*v529*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v127 /*v639*/, v91 /*v603*/
	s_set_vgpr_msb 0x6a81
	v_cndmask_b32_e64 v124 /*v636*/, v208 /*v464*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	s_set_vgpr_msb 0x810a
	v_cmp_gt_i32_e64 s3, v128 /*v640*/, v91 /*v603*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v125 /*v637*/, v209 /*v465*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	s_set_vgpr_msb 0x810a
	v_cmp_lt_i32_e64 s4, v128 /*v640*/, v89 /*v601*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v126 /*v638*/, v210 /*v466*/, 0xff800000, s2
	s_set_vgpr_msb 0x816a
	v_cmp_lt_i32_e64 s2, v127 /*v639*/, v89 /*v601*/
	v_cmp_gt_i32_e64 s5, v129 /*v641*/, v91 /*v603*/
	v_cmp_lt_i32_e64 s6, v129 /*v641*/, v89 /*v601*/
	v_max3_num_f32 v208 /*v464*/, v18 /*v530*/, v19 /*v531*/, v20 /*v532*/
	v_max3_num_f32 v210 /*v466*/, v21 /*v533*/, v22 /*v534*/, v23 /*v535*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v130 /*v642*/, v91 /*v603*/
	s_set_vgpr_msb 0x6a81
	v_cndmask_b32_e64 v127 /*v639*/, v211 /*v467*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	s_set_vgpr_msb 0x810a
	v_cmp_gt_i32_e64 s3, v14 /*v526*/, v91 /*v603*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v128 /*v640*/, v212 /*v468*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	s_set_vgpr_msb 0x810a
	v_cmp_lt_i32_e64 s4, v14 /*v526*/, v89 /*v601*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v129 /*v641*/, v213 /*v469*/, 0xff800000, s2
	s_set_vgpr_msb 0x816a
	v_cmp_lt_i32_e64 s2, v130 /*v642*/, v89 /*v601*/
	v_cmp_gt_i32_e64 s5, v15 /*v527*/, v91 /*v603*/
	v_cmp_lt_i32_e64 s6, v15 /*v527*/, v89 /*v601*/
	v_max3_num_f32 v212 /*v468*/, v24 /*v536*/, v25 /*v537*/, v94 /*v606*/
	v_max3_num_f32 v199 /*v455*/, v117 /*v629*/, v118 /*v630*/, v119 /*v631*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v133 /*v645*/, v91 /*v603*/
	s_set_vgpr_msb 0x6a81
	v_cndmask_b32_e64 v130 /*v642*/, v214 /*v470*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	s_set_vgpr_msb 0x810a
	v_cmp_gt_i32_e64 s3, v134 /*v646*/, v91 /*v603*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v131 /*v643*/, v215 /*v471*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	s_set_vgpr_msb 0x810a
	v_cmp_lt_i32_e64 s4, v134 /*v646*/, v89 /*v601*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v132 /*v644*/, v216 /*v472*/, 0xff800000, s2
	s_set_vgpr_msb 0x816a
	v_cmp_lt_i32_e64 s2, v133 /*v645*/, v89 /*v601*/
	v_cmp_gt_i32_e64 s5, v135 /*v647*/, v91 /*v603*/
	v_cmp_lt_i32_e64 s6, v135 /*v647*/, v89 /*v601*/
	v_max3_num_f32 v214 /*v470*/, v95 /*v607*/, v28 /*v540*/, v29 /*v541*/
	v_max3_num_f32 v216 /*v472*/, v30 /*v542*/, v31 /*v543*/, v96 /*v608*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v136 /*v648*/, v91 /*v603*/
	s_set_vgpr_msb 0x6a81
	v_cndmask_b32_e64 v133 /*v645*/, v217 /*v473*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	s_set_vgpr_msb 0x810a
	v_cmp_gt_i32_e64 s3, v137 /*v649*/, v91 /*v603*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v134 /*v646*/, v218 /*v474*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	s_set_vgpr_msb 0x810a
	v_cmp_lt_i32_e64 s4, v137 /*v649*/, v89 /*v601*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v135 /*v647*/, v219 /*v475*/, 0xff800000, s2
	s_set_vgpr_msb 0x816a
	v_cmp_lt_i32_e64 s2, v136 /*v648*/, v89 /*v601*/
	v_cmp_gt_i32_e64 s5, v138 /*v650*/, v91 /*v603*/
	v_cmp_lt_i32_e64 s6, v138 /*v650*/, v89 /*v601*/
	v_max3_num_f32 v218 /*v474*/, v97 /*v609*/, v34 /*v546*/, v35 /*v547*/
	v_max3_num_f32 v201 /*v457*/, v120 /*v632*/, v121 /*v633*/, v122 /*v634*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v139 /*v651*/, v91 /*v603*/
	s_set_vgpr_msb 0x6a81
	v_cndmask_b32_e64 v136 /*v648*/, v220 /*v476*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	s_set_vgpr_msb 0x810a
	v_cmp_gt_i32_e64 s3, v140 /*v652*/, v91 /*v603*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v137 /*v649*/, v221 /*v477*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	s_set_vgpr_msb 0x810a
	v_cmp_lt_i32_e64 s4, v140 /*v652*/, v89 /*v601*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v138 /*v650*/, v222 /*v478*/, 0xff800000, s2
	s_set_vgpr_msb 0x816a
	v_cmp_lt_i32_e64 s2, v139 /*v651*/, v89 /*v601*/
	v_cmp_gt_i32_e64 s5, v141 /*v653*/, v91 /*v603*/
	v_cmp_lt_i32_e64 s6, v141 /*v653*/, v89 /*v601*/
	v_max3_num_f32 v220 /*v476*/, v98 /*v610*/, v99 /*v611*/, v38 /*v550*/
	v_max3_num_f32 v222 /*v478*/, v39 /*v551*/, v40 /*v552*/, v41 /*v553*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v142 /*v654*/, v91 /*v603*/
	s_set_vgpr_msb 0x6a81
	v_cndmask_b32_e64 v139 /*v651*/, v223 /*v479*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	s_set_vgpr_msb 0x810a
	v_cmp_gt_i32_e64 s3, v26 /*v538*/, v91 /*v603*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v140 /*v652*/, v224 /*v480*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	s_set_vgpr_msb 0x810a
	v_cmp_lt_i32_e64 s4, v26 /*v538*/, v89 /*v601*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v141 /*v653*/, v225 /*v481*/, 0xff800000, s2
	s_set_vgpr_msb 0x816a
	v_cmp_lt_i32_e64 s2, v142 /*v654*/, v89 /*v601*/
	v_cmp_gt_i32_e64 s5, v27 /*v539*/, v91 /*v603*/
	v_cmp_lt_i32_e64 s6, v27 /*v539*/, v89 /*v601*/
	v_max3_num_f32 v224 /*v480*/, v100 /*v612*/, v101 /*v613*/, v102 /*v614*/
	v_max3_num_f32 v203 /*v459*/, v123 /*v635*/, v124 /*v636*/, v125 /*v637*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v145 /*v657*/, v91 /*v603*/
	s_set_vgpr_msb 0x6a81
	v_cndmask_b32_e64 v142 /*v654*/, v226 /*v482*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	s_set_vgpr_msb 0x810a
	v_cmp_gt_i32_e64 s3, v146 /*v658*/, v91 /*v603*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v143 /*v655*/, v227 /*v483*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	s_set_vgpr_msb 0x810a
	v_cmp_lt_i32_e64 s4, v146 /*v658*/, v89 /*v601*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v144 /*v656*/, v228 /*v484*/, 0xff800000, s2
	s_set_vgpr_msb 0x816a
	v_cmp_lt_i32_e64 s2, v145 /*v657*/, v89 /*v601*/
	v_cmp_gt_i32_e64 s5, v147 /*v659*/, v91 /*v603*/
	v_cmp_lt_i32_e64 s6, v147 /*v659*/, v89 /*v601*/
	v_max3_num_f32 v226 /*v482*/, v103 /*v615*/, v46 /*v558*/, v47 /*v559*/
	v_max3_num_f32 v205 /*v461*/, v126 /*v638*/, v127 /*v639*/, v128 /*v640*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v148 /*v660*/, v91 /*v603*/
	s_set_vgpr_msb 0x6a81
	v_cndmask_b32_e64 v145 /*v657*/, v229 /*v485*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	s_set_vgpr_msb 0x810a
	v_cmp_gt_i32_e64 s3, v32 /*v544*/, v91 /*v603*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v146 /*v658*/, v230 /*v486*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	s_set_vgpr_msb 0x810a
	v_cmp_lt_i32_e64 s4, v32 /*v544*/, v89 /*v601*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v147 /*v659*/, v231 /*v487*/, 0xff800000, s2
	s_set_vgpr_msb 0x816a
	v_cmp_lt_i32_e64 s2, v148 /*v660*/, v89 /*v601*/
	v_cmp_gt_i32_e64 s5, v33 /*v545*/, v91 /*v603*/
	v_cmp_lt_i32_e64 s6, v33 /*v545*/, v89 /*v601*/
	v_max3_num_f32 v230 /*v486*/, v105 /*v617*/, v106 /*v618*/, v107 /*v619*/
	v_max3_num_f32 v207 /*v463*/, v129 /*v641*/, v130 /*v642*/, v131 /*v643*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v151 /*v663*/, v91 /*v603*/
	s_set_vgpr_msb 0x6a81
	v_cndmask_b32_e64 v148 /*v660*/, v232 /*v488*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	s_set_vgpr_msb 0x810a
	v_cmp_gt_i32_e64 s3, v152 /*v664*/, v91 /*v603*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v149 /*v661*/, v233 /*v489*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	s_set_vgpr_msb 0x810a
	v_cmp_lt_i32_e64 s4, v152 /*v664*/, v89 /*v601*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v150 /*v662*/, v234 /*v490*/, 0xff800000, s2
	s_set_vgpr_msb 0x816a
	v_cmp_lt_i32_e64 s2, v151 /*v663*/, v89 /*v601*/
	v_cmp_gt_i32_e64 s5, v36 /*v548*/, v91 /*v603*/
	v_cmp_lt_i32_e64 s6, v36 /*v548*/, v89 /*v601*/
	v_max3_num_f32 v209 /*v465*/, v132 /*v644*/, v133 /*v645*/, v134 /*v646*/
	v_max3_num_f32 v211 /*v467*/, v135 /*v647*/, v136 /*v648*/, v137 /*v649*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v37 /*v549*/, v91 /*v603*/
	s_set_vgpr_msb 0x6a81
	v_cndmask_b32_e64 v151 /*v663*/, v235 /*v491*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	s_set_vgpr_msb 0x810a
	v_cmp_gt_i32_e64 s3, v154 /*v666*/, v91 /*v603*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v152 /*v664*/, v236 /*v492*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	s_set_vgpr_msb 0x810a
	v_cmp_lt_i32_e64 s4, v154 /*v666*/, v89 /*v601*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v153 /*v665*/, v237 /*v493*/, 0xff800000, s2
	s_set_vgpr_msb 0x816a
	v_cmp_lt_i32_e64 s2, v37 /*v549*/, v89 /*v601*/
	v_cmp_gt_i32_e64 s5, v155 /*v667*/, v91 /*v603*/
	v_cmp_lt_i32_e64 s6, v155 /*v667*/, v89 /*v601*/
	v_max3_num_f32 v213 /*v469*/, v138 /*v650*/, v139 /*v651*/, v140 /*v652*/
	v_max3_num_f32 v215 /*v471*/, v141 /*v653*/, v142 /*v654*/, v143 /*v655*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v157 /*v669*/, v91 /*v603*/
	s_set_vgpr_msb 0x6a81
	v_cndmask_b32_e64 v154 /*v666*/, v238 /*v494*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	s_set_vgpr_msb 0x810a
	v_cmp_gt_i32_e64 s3, v158 /*v670*/, v91 /*v603*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v155 /*v667*/, v239 /*v495*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	s_set_vgpr_msb 0x810a
	v_cmp_lt_i32_e64 s4, v158 /*v670*/, v89 /*v601*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v156 /*v668*/, v240 /*v496*/, 0xff800000, s2
	s_set_vgpr_msb 0x816a
	v_cmp_lt_i32_e64 s2, v157 /*v669*/, v89 /*v601*/
	v_cmp_gt_i32_e64 s5, v42 /*v554*/, v91 /*v603*/
	v_cmp_lt_i32_e64 s6, v42 /*v554*/, v89 /*v601*/
	v_max3_num_f32 v217 /*v473*/, v144 /*v656*/, v145 /*v657*/, v146 /*v658*/
	v_max3_num_f32 v219 /*v475*/, v147 /*v659*/, v148 /*v660*/, v149 /*v661*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v43 /*v555*/, v91 /*v603*/
	s_set_vgpr_msb 0x6a81
	v_cndmask_b32_e64 v157 /*v669*/, v241 /*v497*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	s_set_vgpr_msb 0x810a
	v_cmp_gt_i32_e64 s3, v44 /*v556*/, v91 /*v603*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v158 /*v670*/, v242 /*v498*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	s_set_vgpr_msb 0x810a
	v_cmp_lt_i32_e64 s4, v44 /*v556*/, v89 /*v601*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v159 /*v671*/, v243 /*v499*/, 0xff800000, s2
	s_set_vgpr_msb 0x816a
	v_cmp_lt_i32_e64 s2, v43 /*v555*/, v89 /*v601*/
	v_cmp_gt_i32_e64 s5, v45 /*v557*/, v91 /*v603*/
	v_cmp_lt_i32_e64 s6, v45 /*v557*/, v89 /*v601*/
	v_max3_num_f32 v221 /*v477*/, v150 /*v662*/, v151 /*v663*/, v152 /*v664*/
	v_max3_num_f32 v223 /*v479*/, v153 /*v665*/, v154 /*v666*/, v155 /*v667*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v163 /*v675*/, v91 /*v603*/
	s_set_vgpr_msb 0x6a81
	v_cndmask_b32_e64 v160 /*v672*/, v244 /*v500*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	s_set_vgpr_msb 0x810a
	v_cmp_gt_i32_e64 s3, v164 /*v676*/, v91 /*v603*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v161 /*v673*/, v245 /*v501*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	s_set_vgpr_msb 0x810a
	v_cmp_lt_i32_e64 s4, v164 /*v676*/, v89 /*v601*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v162 /*v674*/, v246 /*v502*/, 0xff800000, s2
	s_set_vgpr_msb 0x816a
	v_cmp_lt_i32_e64 s2, v163 /*v675*/, v89 /*v601*/
	v_cmp_gt_i32_e64 s5, v165 /*v677*/, v91 /*v603*/
	v_cmp_lt_i32_e64 s6, v165 /*v677*/, v89 /*v601*/
	v_max3_num_f32 v225 /*v481*/, v156 /*v668*/, v157 /*v669*/, v158 /*v670*/
	v_max3_num_f32 v227 /*v483*/, v159 /*v671*/, v160 /*v672*/, v161 /*v673*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v166 /*v678*/, v91 /*v603*/
	s_set_vgpr_msb 0x6a81
	v_cndmask_b32_e64 v163 /*v675*/, v247 /*v503*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	s_set_vgpr_msb 0x818a
	v_cmp_gt_i32_e64 s3, v50 /*v562*/, v91 /*v603*/
	v_cndmask_b32_e64 v164 /*v676*/, v58 /*v570*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	v_cmp_lt_i32_e64 s4, v50 /*v562*/, v89 /*v601*/
	v_cndmask_b32_e64 v165 /*v677*/, v59 /*v571*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v166 /*v678*/, v89 /*v601*/
	v_cmp_gt_i32_e64 s5, v51 /*v563*/, v91 /*v603*/
	v_cmp_lt_i32_e64 s6, v51 /*v563*/, v89 /*v601*/
	s_set_vgpr_msb 0x8a4a
	v_dual_max_num_f32 v228 /*v484*/, v48 /*v560*/, v49 /*v561*/ :: v_dual_max_num_f32 v229 /*v485*/, v162 /*v674*/, v163 /*v675*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v52 /*v564*/, v91 /*v603*/
	s_set_vgpr_msb 0x4a8a
	v_cndmask_b32_e64 v166 /*v678*/, v60 /*v572*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	v_cmp_gt_i32_e64 s3, v53 /*v565*/, v91 /*v603*/
	v_cndmask_b32_e64 v167 /*v679*/, v61 /*v573*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	v_cmp_lt_i32_e64 s4, v53 /*v565*/, v89 /*v601*/
	v_cndmask_b32_e64 v168 /*v680*/, v62 /*v574*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v52 /*v564*/, v89 /*v601*/
	v_cmp_gt_i32_e64 s5, v169 /*v681*/, v91 /*v603*/
	v_cmp_lt_i32_e64 s6, v169 /*v681*/, v89 /*v601*/
	s_set_vgpr_msb 0x8a6a
	v_max3_num_f32 v231 /*v487*/, v165 /*v677*/, v166 /*v678*/, v167 /*v679*/
	v_max3_num_f32 v232 /*v488*/, v54 /*v566*/, v55 /*v567*/, v56 /*v568*/
	s_or_b32 s2, s2, vcc_lo
	s_set_vgpr_msb 0x6a55
	v_max3_num_f32 v192 /*v448*/, v192 /*v448*/, v194 /*v450*/, v196 /*v452*/
	s_set_vgpr_msb 0x5582
	v_cndmask_b32_e64 v169 /*v681*/, v63 /*v575*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	s_set_vgpr_msb 0x8255
	v_max3_num_f32 v193 /*v449*/, v193 /*v449*/, v195 /*v451*/, v197 /*v453*/
	s_set_vgpr_msb 0x5582
	v_cndmask_b32_e64 v170 /*v682*/, v64 /*v576*/, 0xff800000, s2
	s_set_vgpr_msb 0x8255
	v_max3_num_f32 v194 /*v450*/, v198 /*v454*/, v200 /*v456*/, v202 /*v458*/
	v_max3_num_f32 v195 /*v451*/, v204 /*v460*/, v206 /*v462*/, v208 /*v464*/
	v_max3_num_f32 v196 /*v452*/, v210 /*v466*/, v212 /*v468*/, v214 /*v470*/
	v_max3_num_f32 v197 /*v453*/, v216 /*v472*/, v218 /*v474*/, v220 /*v476*/
	v_max3_num_f32 v198 /*v454*/, v222 /*v478*/, v224 /*v480*/, v226 /*v482*/
	s_set_vgpr_msb 0x5559
	v_max3_num_f32 v200 /*v456*/, v228 /*v484*/, v104 /*v616*/, v230 /*v486*/
	s_or_b32 s2, s6, s5
	s_set_vgpr_msb 0x596a
	v_max3_num_f32 v233 /*v489*/, v168 /*v680*/, v169 /*v681*/, v170 /*v682*/
	s_set_vgpr_msb 0x6a82
	v_cndmask_b32_e64 v171 /*v683*/, v65 /*v577*/, 0xff800000, s2
	s_set_vgpr_msb 0x8255
	v_max3_num_f32 v199 /*v455*/, v199 /*v455*/, v201 /*v457*/, v203 /*v459*/
	v_max3_num_f32 v201 /*v457*/, v205 /*v461*/, v207 /*v463*/, v209 /*v465*/
	v_max3_num_f32 v192 /*v448*/, v192 /*v448*/, v194 /*v450*/, v195 /*v451*/
	v_max3_num_f32 v194 /*v450*/, v196 /*v452*/, v197 /*v453*/, v198 /*v454*/
	s_set_vgpr_msb 0x5565
	v_max3_num_f32 v195 /*v451*/, v200 /*v456*/, v232 /*v488*/, v57 /*v569*/
	s_set_vgpr_msb 0x6555
	v_max3_num_f32 v196 /*v452*/, v211 /*v467*/, v213 /*v469*/, v215 /*v471*/
	v_max3_num_f32 v197 /*v453*/, v217 /*v473*/, v219 /*v475*/, v221 /*v477*/
	v_max3_num_f32 v198 /*v454*/, v223 /*v479*/, v225 /*v481*/, v227 /*v483*/
	s_set_vgpr_msb 0x5559
	v_max3_num_f32 v200 /*v456*/, v229 /*v485*/, v164 /*v676*/, v231 /*v487*/
	s_set_vgpr_msb 0x5955
	v_max3_num_f32 v192 /*v448*/, v192 /*v448*/, v194 /*v450*/, v195 /*v451*/
	v_max3_num_f32 v193 /*v449*/, v193 /*v449*/, v199 /*v455*/, v201 /*v457*/
	v_max3_num_f32 v194 /*v450*/, v196 /*v452*/, v197 /*v453*/, v198 /*v454*/
	s_set_vgpr_msb 0x5565
	v_max3_num_f32 v195 /*v451*/, v200 /*v456*/, v233 /*v489*/, v171 /*v683*/
	s_set_vgpr_msb 0x6555
	v_max3_num_f32 v193 /*v449*/, v193 /*v449*/, v194 /*v450*/, v195 /*v451*/
	v_dual_mov_b32 v196 /*v452*/, v192 /*v448*/ :: v_dual_mov_b32 v194 /*v450*/, v193 /*v449*/
	v_permlanex16_b32 v196 /*v452*/, v196 /*v452*/, s61, 0xfedcba98
	v_permlanex16_b32 v194 /*v450*/, v194 /*v450*/, s61, 0xfedcba98
	v_dual_max_num_f32 v192 /*v448*/, v192 /*v448*/, v196 /*v452*/ :: v_dual_max_num_f32 v193 /*v449*/, v193 /*v449*/, v194 /*v450*/
	s_set_vgpr_msb 0x5549
	v_sub_f32_e32 v195 /*v451*/, v192 /*v448*/, v87 /*v599*/
	v_max_num_f32_e32 v192 /*v448*/, v192 /*v448*/, v87 /*v599*/
	v_sub_f32_e32 v194 /*v450*/, v193 /*v449*/, v86 /*v598*/
	s_set_vgpr_msb 0x4904
	v_cmp_lt_f32_e32 vcc_lo, 0x41000000, v195 /*v451*/
	s_cmp_eq_u32 vcc_lo, 0
	s_cselect_b32 s2, -1, 0
	s_set_vgpr_msb 0x489
	v_cndmask_b32_e64 v69 /*v581*/, v192 /*v448*/, v87 /*v599*/, s2
	s_set_vgpr_msb 0x8946
	v_cmp_lt_f32_e64 s2, 0x41000000, v194 /*v450*/
	v_max_num_f32_e32 v192 /*v448*/, v86 /*v598*/, v193 /*v449*/
	s_cmp_lg_u32 s2, 0
	s_cselect_b32 s3, -1, 0
	s_cmp_eq_u32 s2, 0
	s_cselect_b32 s2, -1, 0
	s_set_vgpr_msb 0x4689
	v_cndmask_b32_e64 v71 /*v583*/, v192 /*v448*/, v86 /*v598*/, s2
	v_mul_f32_e32 v60 /*v572*/, 0xbfb8aa3b, v69 /*v581*/
	v_mul_f32_e32 v68 /*v580*/, 0xbfb8aa3b, v71 /*v583*/
	s_set_vgpr_msb 0x8962
	v_pk_fma_f32 v[206:207] /*v[462:463]*/, v[2:3] /*v[514:515]*/, s[88:89], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6261
	v_pk_fma_f32 v[192:193] /*v[448:449]*/, v[250:251] /*v[506:507]*/, s[88:89], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[198:199] /*v[454:455]*/, v[254:255] /*v[510:511]*/, s[88:89], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6162
	v_pk_fma_f32 v[216:217] /*v[472:473]*/, v[66:67] /*v[578:579]*/, s[88:89], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[242:243] /*v[498:499]*/, v[96:97] /*v[608:609]*/, s[88:89], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x62a2
	v_pk_fma_f32 v[66:67] /*v[578:579]*/, v[108:109] /*v[620:621]*/, s[88:89], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[96:97] /*v[608:609]*/, v[112:113] /*v[624:625]*/, s[88:89], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa262
	v_pk_fma_f32 v[204:205] /*v[460:461]*/, v[0:1] /*v[512:513]*/, s[88:89], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6241
	v_exp_f32_e32 v222 /*v478*/, v206 /*v462*/
	v_exp_f32_e32 v232 /*v488*/, v207 /*v463*/
	v_nop
	s_set_vgpr_msb 0x4162
	v_pk_fma_f32 v[206:207] /*v[462:463]*/, v[8:9] /*v[520:521]*/, s[88:89], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[212:213] /*v[468:469]*/, v[12:13] /*v[524:525]*/, s[88:89], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[224:225] /*v[480:481]*/, v[20:21] /*v[532:533]*/, s[88:89], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6241
	v_exp_f32_e32 v194 /*v450*/, v193 /*v449*/
	v_exp_f32_e32 v202 /*v458*/, v198 /*v454*/
	v_exp_f32_e32 v208 /*v464*/, v199 /*v455*/
	v_nop
	s_set_vgpr_msb 0x4162
	v_pk_fma_f32 v[198:199] /*v[454:455]*/, v[4:5] /*v[516:517]*/, s[88:89], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v193 /*v449*/, v66 /*v578*/
	v_exp_f32_e32 v195 /*v451*/, v67 /*v579*/
	v_nop
	s_set_vgpr_msb 0x62a2
	v_pk_fma_f32 v[66:67] /*v[578:579]*/, v[114:115] /*v[626:627]*/, s[88:89], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa242
	v_exp_f32_e32 v203 /*v459*/, v96 /*v608*/
	v_exp_f32_e32 v209 /*v465*/, v97 /*v609*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[96:97] /*v[608:609]*/, v[118:119] /*v[630:631]*/, s[88:89], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa261
	v_pk_fma_f32 v[196:197] /*v[452:453]*/, v[252:253] /*v[508:509]*/, s[88:89], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v210 /*v466*/, v204 /*v460*/
	v_exp_f32_e32 v218 /*v474*/, v205 /*v461*/
	v_nop
	s_set_vgpr_msb 0x6162
	v_pk_fma_f32 v[204:205] /*v[460:461]*/, v[6:7] /*v[518:519]*/, s[88:89], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6281
	v_exp_f32_e32 v14 /*v526*/, v206 /*v462*/
	s_set_vgpr_msb 0x8141
	v_exp_f32_e32 v206 /*v462*/, v212 /*v468*/
	v_exp_f32_e32 v214 /*v470*/, v213 /*v469*/
	v_nop
	s_set_vgpr_msb 0x4162
	v_pk_fma_f32 v[212:213] /*v[468:469]*/, v[18:19] /*v[530:531]*/, s[88:89], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6281
	v_exp_f32_e32 v6 /*v518*/, v224 /*v480*/
	v_exp_f32_e32 v18 /*v530*/, v225 /*v481*/
	v_nop
	s_set_vgpr_msb 0x8162
	v_pk_fma_f32 v[224:225] /*v[480:481]*/, v[94:95] /*v[606:607]*/, s[88:89], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x62a2
	v_pk_fma_f32 v[94:95] /*v[606:607]*/, v[110:111] /*v[622:623]*/, s[88:89], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa241
	v_exp_f32_e32 v236 /*v492*/, v198 /*v454*/
	v_exp_f32_e32 v250 /*v506*/, v199 /*v455*/
	v_nop
	s_set_vgpr_msb 0x4162
	v_pk_fma_f32 v[198:199] /*v[454:455]*/, v[10:11] /*v[522:523]*/, s[88:89], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v211 /*v467*/, v66 /*v578*/
	v_exp_f32_e32 v219 /*v475*/, v67 /*v579*/
	v_nop
	s_set_vgpr_msb 0x62a2
	v_pk_fma_f32 v[66:67] /*v[578:579]*/, v[120:121] /*v[632:633]*/, s[88:89], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa242
	v_exp_f32_e32 v237 /*v493*/, v96 /*v608*/
	v_exp_f32_e32 v251 /*v507*/, v97 /*v609*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[96:97] /*v[608:609]*/, v[124:125] /*v[636:637]*/, s[88:89], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa241
	v_exp_f32_e32 v200 /*v456*/, v197 /*v453*/
	s_set_vgpr_msb 0x4142
	v_exp_f32_e32 v197 /*v453*/, v94 /*v606*/
	v_exp_f32_e32 v201 /*v457*/, v95 /*v607*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[94:95] /*v[606:607]*/, v[116:117] /*v[628:629]*/, s[88:89], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa241
	v_exp_f32_e32 v254 /*v510*/, v204 /*v460*/
	s_set_vgpr_msb 0x4181
	v_exp_f32_e32 v10 /*v522*/, v205 /*v461*/
	s_set_vgpr_msb 0x8141
	v_exp_f32_e32 v204 /*v460*/, v199 /*v455*/
	s_set_vgpr_msb 0x4142
	v_exp_f32_e32 v255 /*v511*/, v66 /*v578*/
	s_set_vgpr_msb 0x42a2
	v_exp_f32_e32 v11 /*v523*/, v67 /*v579*/
	v_nop
	v_pk_fma_f32 v[66:67] /*v[578:579]*/, v[126:127] /*v[638:639]*/, s[88:89], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa242
	v_exp_f32_e32 v199 /*v455*/, v96 /*v608*/
	v_exp_f32_e32 v205 /*v461*/, v97 /*v609*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[96:97] /*v[608:609]*/, v[130:131] /*v[642:643]*/, s[88:89], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa242
	v_exp_f32_e32 v223 /*v479*/, v94 /*v606*/
	v_exp_f32_e32 v233 /*v489*/, v95 /*v607*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[94:95] /*v[606:607]*/, v[122:123] /*v[634:635]*/, s[88:89], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa281
	v_exp_f32_e32 v26 /*v538*/, v207 /*v463*/
	s_set_vgpr_msb 0x8162
	v_pk_fma_f32 v[220:221] /*v[476:477]*/, v[16:17] /*v[528:529]*/, s[88:89], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v207 /*v463*/, v66 /*v578*/
	v_exp_f32_e32 v215 /*v471*/, v67 /*v579*/
	v_nop
	s_set_vgpr_msb 0x62a2
	v_pk_fma_f32 v[66:67] /*v[578:579]*/, v[132:133] /*v[644:645]*/, s[88:89], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa242
	v_exp_f32_e32 v229 /*v485*/, v96 /*v608*/
	v_exp_f32_e32 v241 /*v497*/, v97 /*v609*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[96:97] /*v[608:609]*/, v[136:137] /*v[648:649]*/, s[88:89], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v15 /*v527*/, v94 /*v606*/
	v_exp_f32_e32 v27 /*v539*/, v95 /*v607*/
	v_nop
	v_pk_fma_f32 v[94:95] /*v[606:607]*/, v[128:129] /*v[640:641]*/, s[88:89], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa241
	v_exp_f32_e32 v228 /*v484*/, v220 /*v476*/
	v_exp_f32_e32 v240 /*v496*/, v221 /*v477*/
	v_nop
	s_set_vgpr_msb 0x4162
	v_pk_fma_f32 v[220:221] /*v[476:477]*/, v[22:23] /*v[534:535]*/, s[88:89], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v245 /*v501*/, v66 /*v578*/
	s_set_vgpr_msb 0x62a2
	v_exp_f32_e32 v3 /*v515*/, v67 /*v579*/
	v_nop
	v_pk_fma_f32 v[66:67] /*v[578:579]*/, v[138:139] /*v[650:651]*/, s[88:89], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v23 /*v535*/, v96 /*v608*/
	v_exp_f32_e32 v33 /*v545*/, v97 /*v609*/
	v_nop
	v_pk_fma_f32 v[96:97] /*v[608:609]*/, v[142:143] /*v[654:655]*/, s[88:89], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa241
	v_exp_f32_e32 v226 /*v482*/, v217 /*v473*/
	s_set_vgpr_msb 0x4142
	v_exp_f32_e32 v217 /*v473*/, v94 /*v606*/
	v_exp_f32_e32 v227 /*v483*/, v95 /*v607*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[94:95] /*v[606:607]*/, v[134:135] /*v[646:647]*/, s[88:89], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa241
	v_exp_f32_e32 v244 /*v500*/, v212 /*v468*/
	s_set_vgpr_msb 0x4181
	v_exp_f32_e32 v2 /*v514*/, v213 /*v469*/
	v_nop
	s_set_vgpr_msb 0x8162
	v_pk_fma_f32 v[212:213] /*v[468:469]*/, v[24:25] /*v[536:537]*/, s[88:89], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6281
	v_exp_f32_e32 v22 /*v534*/, v220 /*v476*/
	s_set_vgpr_msb 0x8162
	v_pk_fma_f32 v[230:231] /*v[486:487]*/, v[28:29] /*v[540:541]*/, s[88:89], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[238:239] /*v[494:495]*/, v[30:31] /*v[542:543]*/, s[88:89], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6241
	v_exp_f32_e32 v220 /*v476*/, v225 /*v481*/
	s_set_vgpr_msb 0x41a2
	v_exp_f32_e32 v37 /*v549*/, v66 /*v578*/
	v_exp_f32_e32 v45 /*v557*/, v67 /*v579*/
	v_nop
	v_pk_fma_f32 v[66:67] /*v[578:579]*/, v[144:145] /*v[656:657]*/, s[88:89], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa242
	v_exp_f32_e32 v225 /*v481*/, v96 /*v608*/
	v_exp_f32_e32 v235 /*v491*/, v97 /*v609*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[96:97] /*v[608:609]*/, v[148:149] /*v[660:661]*/, s[88:89], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v7 /*v519*/, v94 /*v606*/
	v_exp_f32_e32 v19 /*v531*/, v95 /*v607*/
	v_nop
	v_pk_fma_f32 v[94:95] /*v[606:607]*/, v[140:141] /*v[652:653]*/, s[88:89], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa281
	v_exp_f32_e32 v36 /*v548*/, v212 /*v468*/
	s_set_vgpr_msb 0x8141
	v_exp_f32_e32 v212 /*v468*/, v224 /*v480*/
	v_exp_f32_e32 v224 /*v480*/, v230 /*v486*/
	v_exp_f32_e32 v234 /*v490*/, v231 /*v487*/
	v_nop
	s_set_vgpr_msb 0x4162
	v_pk_fma_f32 v[230:231] /*v[486:487]*/, v[34:35] /*v[546:547]*/, s[88:89], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6241
	v_exp_f32_e32 v252 /*v508*/, v239 /*v495*/
	s_set_vgpr_msb 0x4142
	v_exp_f32_e32 v239 /*v495*/, v66 /*v578*/
	v_exp_f32_e32 v253 /*v509*/, v67 /*v579*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[66:67] /*v[578:579]*/, v[150:151] /*v[662:663]*/, s[88:89], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v17 /*v529*/, v96 /*v608*/
	v_exp_f32_e32 v29 /*v541*/, v97 /*v609*/
	v_nop
	v_pk_fma_f32 v[96:97] /*v[608:609]*/, v[154:155] /*v[666:667]*/, s[88:89], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa281
	v_exp_f32_e32 v32 /*v544*/, v221 /*v477*/
	v_exp_f32_e32 v44 /*v556*/, v213 /*v469*/
	s_set_vgpr_msb 0x8142
	v_exp_f32_e32 v213 /*v469*/, v94 /*v606*/
	v_exp_f32_e32 v221 /*v477*/, v95 /*v607*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[94:95] /*v[606:607]*/, v[146:147] /*v[658:659]*/, s[88:89], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa281
	v_exp_f32_e32 v0 /*v512*/, v242 /*v498*/
	v_exp_f32_e32 v12 /*v524*/, v243 /*v499*/
	v_exp_f32_e32 v16 /*v528*/, v230 /*v486*/
	s_set_vgpr_msb 0x8162
	v_pk_fma_f32 v[242:243] /*v[498:499]*/, v[38:39] /*v[550:551]*/, s[88:89], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6281
	v_exp_f32_e32 v28 /*v540*/, v231 /*v487*/
	v_nop
	s_set_vgpr_msb 0x8162
	v_pk_fma_f32 v[230:231] /*v[486:487]*/, v[40:41] /*v[552:553]*/, s[88:89], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x62a2
	v_pk_fma_f32 v[8:9] /*v[520:521]*/, v[46:47] /*v[558:559]*/, s[88:89], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v31 /*v543*/, v66 /*v578*/
	v_exp_f32_e32 v41 /*v553*/, v67 /*v579*/
	v_nop
	v_pk_fma_f32 v[66:67] /*v[578:579]*/, v[156:157] /*v[668:669]*/, s[88:89], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v53 /*v565*/, v96 /*v608*/
	v_exp_f32_e32 v59 /*v571*/, v97 /*v609*/
	v_nop
	v_pk_fma_f32 v[96:97] /*v[608:609]*/, v[160:161] /*v[672:673]*/, s[88:89], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa241
	v_exp_f32_e32 v192 /*v448*/, v192 /*v448*/
	s_set_vgpr_msb 0x4162
	v_pk_fma_f32 v[246:247] /*v[502:503]*/, v[98:99] /*v[610:611]*/, s[88:89], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x62a2
	v_exp_f32_e32 v1 /*v513*/, v94 /*v606*/
	v_exp_f32_e32 v13 /*v525*/, v95 /*v607*/
	v_nop
	v_pk_fma_f32 v[94:95] /*v[606:607]*/, v[152:153] /*v[664:665]*/, s[88:89], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa281
	v_exp_f32_e32 v50 /*v562*/, v243 /*v499*/
	v_exp_f32_e32 v58 /*v570*/, v231 /*v487*/
	s_set_vgpr_msb 0x81a2
	v_pk_fma_f32 v[24:25] /*v[536:537]*/, v[48:49] /*v[560:561]*/, s[88:89], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v20 /*v532*/, v9 /*v521*/
	v_pk_fma_f32 v[48:49] /*v[560:561]*/, v[106:107] /*v[618:619]*/, s[88:89], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa242
	v_exp_f32_e32 v231 /*v487*/, v66 /*v578*/
	v_exp_f32_e32 v243 /*v499*/, v67 /*v579*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[66:67] /*v[578:579]*/, v[162:163] /*v[674:675]*/, s[88:89], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v9 /*v521*/, v96 /*v608*/
	v_exp_f32_e32 v21 /*v533*/, v97 /*v609*/
	v_nop
	v_pk_fma_f32 v[96:97] /*v[608:609]*/, v[166:167] /*v[678:679]*/, s[88:89], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa241
	v_exp_f32_e32 v196 /*v452*/, v196 /*v452*/
	v_exp_f32_e32 v238 /*v494*/, v238 /*v494*/
	s_set_vgpr_msb 0x4181
	v_exp_f32_e32 v30 /*v542*/, v246 /*v502*/
	v_exp_f32_e32 v40 /*v552*/, v247 /*v503*/
	v_nop
	s_set_vgpr_msb 0x8162
	v_pk_fma_f32 v[246:247] /*v[502:503]*/, v[100:101] /*v[612:613]*/, s[88:89], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x62a2
	v_pk_fma_f32 v[4:5] /*v[516:517]*/, v[102:103] /*v[614:615]*/, s[88:89], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v43 /*v555*/, v94 /*v606*/
	v_exp_f32_e32 v51 /*v563*/, v95 /*v607*/
	v_nop
	v_pk_fma_f32 v[94:95] /*v[606:607]*/, v[158:159] /*v[670:671]*/, s[88:89], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v34 /*v546*/, v25 /*v537*/
	v_pk_fma_f32 v[62:63] /*v[574:575]*/, v[54:55] /*v[566:567]*/, s[88:89], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v54 /*v566*/, v49 /*v561*/
	v_exp_f32_e32 v25 /*v537*/, v66 /*v578*/
	v_exp_f32_e32 v35 /*v547*/, v67 /*v579*/
	v_exp_f32_e32 v49 /*v561*/, v96 /*v608*/
	v_pk_fma_f32 v[66:67] /*v[578:579]*/, v[168:169] /*v[680:681]*/, s[88:89], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v55 /*v567*/, v97 /*v609*/
	v_nop
	s_set_vgpr_msb 0xa285
	v_pk_add_f32 v[96:97] /*v[608:609]*/, v[192:193] /*v[448:449]*/, v[194:195] /*v[450:451]*/
	v_pk_add_f32 v[98:99] /*v[610:611]*/, v[200:201] /*v[456:457]*/, v[202:203] /*v[458:459]*/
	s_set_vgpr_msb 0x8541
	v_exp_f32_e32 v198 /*v454*/, v198 /*v454*/
	s_set_vgpr_msb 0x4181
	v_exp_f32_e32 v42 /*v554*/, v242 /*v498*/
	v_exp_f32_e32 v52 /*v564*/, v230 /*v486*/
	s_set_vgpr_msb 0x8141
	v_exp_f32_e32 v230 /*v486*/, v246 /*v502*/
	v_exp_f32_e32 v242 /*v498*/, v247 /*v503*/
	s_set_vgpr_msb 0x4142
	v_exp_f32_e32 v246 /*v502*/, v4 /*v516*/
	s_set_vgpr_msb 0x42a2
	v_exp_f32_e32 v4 /*v516*/, v5 /*v517*/
	v_pk_fma_f32 v[38:39] /*v[550:551]*/, v[104:105] /*v[616:617]*/, s[88:89], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa242
	v_exp_f32_e32 v247 /*v503*/, v94 /*v606*/
	s_set_vgpr_msb 0x42a2
	v_exp_f32_e32 v5 /*v517*/, v95 /*v607*/
	v_nop
	v_pk_fma_f32 v[94:95] /*v[606:607]*/, v[164:165] /*v[676:677]*/, s[88:89], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[64:65] /*v[576:577]*/, v[56:57] /*v[568:569]*/, s[88:89], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v57 /*v569*/, v66 /*v578*/
	v_exp_f32_e32 v61 /*v573*/, v67 /*v579*/
	v_nop
	s_set_vgpr_msb 0xa289
	v_pk_add_f32 v[66:67] /*v[578:579]*/, v[196:197] /*v[452:453]*/, v[96:97] /*v[608:609]*/
	v_pk_add_f32 v[96:97] /*v[608:609]*/, v[208:209] /*v[464:465]*/, v[98:99] /*v[610:611]*/
	s_set_vgpr_msb 0x8985
	v_pk_add_f32 v[98:99] /*v[610:611]*/, v[210:211] /*v[466:467]*/, v[218:219] /*v[474:475]*/
	v_pk_add_f32 v[100:101] /*v[612:613]*/, v[232:233] /*v[488:489]*/, v[236:237] /*v[492:493]*/
	s_set_vgpr_msb 0x8589
	v_pk_add_f32 v[102:103] /*v[614:615]*/, v[254:255] /*v[510:511]*/, v[10:11] /*v[522:523]*/
	s_set_vgpr_msb 0x898a
	v_pk_add_f32 v[112:113] /*v[624:625]*/, v[18:19] /*v[530:531]*/, v[22:23] /*v[534:535]*/
	v_pk_add_f32 v[114:115] /*v[626:627]*/, v[36:37] /*v[548:549]*/, v[44:45] /*v[556:557]*/
	s_set_vgpr_msb 0x8a85
	v_pk_add_f32 v[118:119] /*v[630:631]*/, v[238:239] /*v[494:495]*/, v[252:253] /*v[508:509]*/
	s_set_vgpr_msb 0x858a
	v_pk_add_f32 v[120:121] /*v[632:633]*/, v[12:13] /*v[524:525]*/, v[16:17] /*v[528:529]*/
	s_set_vgpr_msb 0x8a41
	v_exp_f32_e32 v216 /*v472*/, v216 /*v472*/
	s_set_vgpr_msb 0x4186
	v_exp_f32_e32 v8 /*v520*/, v8 /*v520*/
	v_exp_f32_e32 v24 /*v536*/, v24 /*v536*/
	v_exp_f32_e32 v46 /*v558*/, v39 /*v551*/
	v_exp_f32_e32 v48 /*v560*/, v48 /*v560*/
	v_exp_f32_e32 v47 /*v559*/, v95 /*v607*/
	v_pk_add_f32 v[104:105] /*v[616:617]*/, v[26:27] /*v[538:539]*/, v[198:199] /*v[454:455]*/
	s_set_vgpr_msb 0x8685
	v_pk_add_f32 v[106:107] /*v[618:619]*/, v[206:207] /*v[462:463]*/, v[214:215] /*v[470:471]*/
	s_set_vgpr_msb 0x8589
	v_pk_add_f32 v[98:99] /*v[610:611]*/, v[222:223] /*v[478:479]*/, v[98:99] /*v[610:611]*/
	v_pk_add_f32 v[100:101] /*v[612:613]*/, v[250:251] /*v[506:507]*/, v[100:101] /*v[612:613]*/
	s_set_vgpr_msb 0x898a
	v_pk_add_f32 v[102:103] /*v[614:615]*/, v[14:15] /*v[526:527]*/, v[102:103] /*v[614:615]*/
	s_set_vgpr_msb 0x8a85
	v_pk_add_f32 v[108:109] /*v[620:621]*/, v[226:227] /*v[482:483]*/, v[228:229] /*v[484:485]*/
	v_pk_add_f32 v[116:117] /*v[628:629]*/, v[220:221] /*v[476:477]*/, v[224:225] /*v[480:481]*/
	s_set_vgpr_msb 0x858a
	v_pk_add_f32 v[112:113] /*v[624:625]*/, v[32:33] /*v[544:545]*/, v[112:113] /*v[624:625]*/
	s_set_vgpr_msb 0x8a89
	v_pk_add_f32 v[114:115] /*v[626:627]*/, v[212:213] /*v[468:469]*/, v[114:115] /*v[626:627]*/
	s_set_vgpr_msb 0x898a
	v_pk_add_f32 v[122:123] /*v[634:635]*/, v[30:31] /*v[542:543]*/, v[40:41] /*v[552:553]*/
	v_pk_add_f32 v[124:125] /*v[636:637]*/, v[50:51] /*v[562:563]*/, v[52:53] /*v[564:565]*/
	s_set_vgpr_msb 0x8a85
	v_pk_add_f32 v[126:127] /*v[638:639]*/, v[230:231] /*v[486:487]*/, v[242:243] /*v[498:499]*/
	s_set_vgpr_msb 0x85aa
	v_pk_add_f32 v[118:119] /*v[630:631]*/, v[0:1] /*v[512:513]*/, v[118:119] /*v[630:631]*/
	v_pk_add_f32 v[120:121] /*v[632:633]*/, v[28:29] /*v[540:541]*/, v[120:121] /*v[632:633]*/
	v_pk_add_f32 v[66:67] /*v[578:579]*/, v[66:67] /*v[578:579]*/, v[96:97] /*v[608:609]*/
	v_exp_f32_e32 v38 /*v550*/, v38 /*v550*/
	v_exp_f32_e32 v56 /*v568*/, v62 /*v574*/
	v_exp_f32_e32 v60 /*v572*/, v63 /*v575*/
	v_exp_f32_e32 v39 /*v551*/, v94 /*v606*/
	v_nop
	v_pk_fma_f32 v[94:95] /*v[606:607]*/, v[170:171] /*v[682:683]*/, s[88:89], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xaa89
	v_pk_add_f32 v[104:105] /*v[616:617]*/, v[204:205] /*v[460:461]*/, v[104:105] /*v[616:617]*/
	v_pk_add_f32 v[106:107] /*v[618:619]*/, v[216:217] /*v[472:473]*/, v[106:107] /*v[618:619]*/
	v_pk_add_f32 v[110:111] /*v[622:623]*/, v[244:245] /*v[500:501]*/, v[2:3] /*v[514:515]*/
	v_pk_add_f32 v[108:109] /*v[620:621]*/, v[240:241] /*v[496:497]*/, v[108:109] /*v[620:621]*/
	v_pk_add_f32 v[116:117] /*v[628:629]*/, v[234:235] /*v[490:491]*/, v[116:117] /*v[628:629]*/
	s_set_vgpr_msb 0x898a
	v_pk_add_f32 v[122:123] /*v[634:635]*/, v[42:43] /*v[554:555]*/, v[122:123] /*v[634:635]*/
	v_pk_add_f32 v[124:125] /*v[636:637]*/, v[58:59] /*v[570:571]*/, v[124:125] /*v[636:637]*/
	s_set_vgpr_msb 0x8a89
	v_pk_add_f32 v[126:127] /*v[638:639]*/, v[246:247] /*v[502:503]*/, v[126:127] /*v[638:639]*/
	s_set_vgpr_msb 0x898a
	v_pk_add_f32 v[128:129] /*v[640:641]*/, v[4:5] /*v[516:517]*/, v[8:9] /*v[520:521]*/
	v_pk_add_f32 v[130:131] /*v[642:643]*/, v[24:25] /*v[536:537]*/, v[34:35] /*v[546:547]*/
	v_pk_add_f32 v[132:133] /*v[644:645]*/, v[46:47] /*v[558:559]*/, v[48:49] /*v[560:561]*/
	v_pk_add_f32 v[66:67] /*v[578:579]*/, v[98:99] /*v[610:611]*/, v[66:67] /*v[578:579]*/
	v_pk_add_f32 v[98:99] /*v[610:611]*/, v[100:101] /*v[612:613]*/, v[102:103] /*v[614:615]*/
	v_pk_add_f32 v[100:101] /*v[612:613]*/, v[112:113] /*v[624:625]*/, v[114:115] /*v[626:627]*/
	v_pk_add_f32 v[102:103] /*v[614:615]*/, v[118:119] /*v[630:631]*/, v[120:121] /*v[632:633]*/
	v_exp_f32_e32 v62 /*v574*/, v64 /*v576*/
	v_exp_f32_e32 v63 /*v575*/, v94 /*v606*/
	v_pk_add_f32 v[110:111] /*v[622:623]*/, v[6:7] /*v[518:519]*/, v[110:111] /*v[622:623]*/
	v_pk_add_f32 v[134:135] /*v[646:647]*/, v[56:57] /*v[568:569]*/, v[60:61] /*v[572:573]*/
	v_pk_add_f32 v[96:97] /*v[608:609]*/, v[20:21] /*v[532:533]*/, v[128:129] /*v[640:641]*/
	v_pk_add_f32 v[128:129] /*v[640:641]*/, v[38:39] /*v[550:551]*/, v[130:131] /*v[642:643]*/
	v_pk_add_f32 v[130:131] /*v[642:643]*/, v[54:55] /*v[566:567]*/, v[132:133] /*v[644:645]*/
	v_pk_add_f32 v[106:107] /*v[618:619]*/, v[106:107] /*v[618:619]*/, v[108:109] /*v[620:621]*/
	v_pk_add_f32 v[108:109] /*v[620:621]*/, v[124:125] /*v[636:637]*/, v[126:127] /*v[638:639]*/
	v_pk_add_f32 v[98:99] /*v[610:611]*/, v[104:105] /*v[616:617]*/, v[98:99] /*v[610:611]*/
	v_pk_add_f32 v[100:101] /*v[612:613]*/, v[116:117] /*v[628:629]*/, v[100:101] /*v[612:613]*/
	v_pk_add_f32 v[102:103] /*v[614:615]*/, v[122:123] /*v[634:635]*/, v[102:103] /*v[614:615]*/
	v_pk_add_f32 v[132:133] /*v[644:645]*/, v[62:63] /*v[574:575]*/, v[134:135] /*v[646:647]*/
	v_pk_add_f32 v[104:105] /*v[616:617]*/, v[110:111] /*v[622:623]*/, v[106:107] /*v[618:619]*/
	v_pk_add_f32 v[96:97] /*v[608:609]*/, v[96:97] /*v[608:609]*/, v[108:109] /*v[620:621]*/
	v_pk_add_f32 v[106:107] /*v[618:619]*/, v[128:129] /*v[640:641]*/, v[130:131] /*v[642:643]*/
	v_pk_add_f32 v[66:67] /*v[578:579]*/, v[66:67] /*v[578:579]*/, v[98:99] /*v[610:611]*/
	v_pk_add_f32 v[98:99] /*v[610:611]*/, v[100:101] /*v[612:613]*/, v[102:103] /*v[614:615]*/
	v_exp_f32_e32 v64 /*v576*/, v65 /*v577*/
	v_exp_f32_e32 v65 /*v577*/, v95 /*v607*/
	v_nop
	v_pk_add_f32 v[94:95] /*v[606:607]*/, v[132:133] /*v[644:645]*/, v[106:107] /*v[618:619]*/
	v_pk_add_f32 v[66:67] /*v[578:579]*/, v[104:105] /*v[616:617]*/, v[66:67] /*v[578:579]*/
	v_pk_add_f32 v[96:97] /*v[608:609]*/, v[96:97] /*v[608:609]*/, v[98:99] /*v[610:611]*/
	v_sub_f32_e32 v68 /*v580*/, v87 /*v599*/, v69 /*v581*/
	v_pk_add_f32 v[94:95] /*v[606:607]*/, v[64:65] /*v[576:577]*/, v[94:95] /*v[606:607]*/
	v_pk_add_f32 v[66:67] /*v[578:579]*/, v[66:67] /*v[578:579]*/, v[96:97] /*v[608:609]*/
	v_mul_f32_e32 v68 /*v580*/, 0x3fb8aa3b, v68 /*v580*/
	v_pk_add_f32 v[66:67] /*v[578:579]*/, v[94:95] /*v[606:607]*/, v[66:67] /*v[578:579]*/
	v_exp_f32_e32 v68 /*v580*/, v68 /*v580*/
	v_dual_mov_b32 v87 /*v599*/, v66 /*v578*/ :: v_dual_mov_b32 v94 /*v606*/, v67 /*v579*/
	v_permlanex16_b32 v87 /*v599*/, v87 /*v599*/, s61, 0xfedcba98
	v_permlanex16_b32 v94 /*v606*/, v94 /*v606*/, s61, 0xfedcba98
	s_set_vgpr_msb 0x8a00
	s_cbranch_vccz .LBB0_30
	s_set_vgpr_msb 8
	v_pk_mul_f32 v[126:127], v[126:127], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[124:125], v[124:125], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[122:123], v[122:123], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[120:121], v[120:121], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[118:119], v[118:119], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[116:117], v[116:117], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[114:115], v[114:115], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[112:113], v[112:113], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[110:111], v[110:111], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[108:109], v[108:109], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[106:107], v[106:107], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[104:105], v[104:105], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[102:103], v[102:103], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[100:101], v[100:101], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[98:99], v[98:99], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[96:97], v[96:97], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[94:95], v[94:95], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[92:93], v[92:93], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[90:91], v[90:91], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[88:89], v[88:89], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[86:87], v[86:87], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[84:85], v[84:85], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[82:83], v[82:83], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[80:81], v[80:81], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[78:79], v[78:79], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[76:77], v[76:77], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[74:75], v[74:75], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[72:73], v[72:73], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[70:71], v[70:71], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[68:69], v[68:69], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[66:67], v[66:67], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[64:65], v[64:65], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x800
.LBB0_30:
	s_set_vgpr_msb 0x8a
	v_sub_f32_e32 v70 /*v582*/, v86 /*v598*/, v71 /*v583*/
	s_and_b32 s2, s3, exec_lo
	s_cselect_b32 s2, 1, 0
	s_cmp_lg_u32 s2, 1
	v_mul_f32_e32 v70 /*v582*/, 0x3fb8aa3b, v70 /*v582*/
	v_exp_f32_e32 v70 /*v582*/, v70 /*v582*/
	s_set_vgpr_msb 0x8a00
	s_cbranch_scc1 .LBB0_25
	v_nop
	s_set_vgpr_msb 8
	v_pk_mul_f32 v[62:63], v[62:63], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[60:61], v[60:61], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59], v[58:59], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[56:57], v[56:57], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[54:55], v[54:55], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[52:53], v[52:53], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[50:51], v[50:51], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[48:49], v[48:49], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[46:47], v[46:47], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[44:45], v[44:45], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[42:43], v[42:43], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[40:41], v[40:41], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39], v[38:39], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[36:37], v[36:37], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35], v[34:35], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[32:33], v[32:33], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[30:31], v[30:31], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[28:29], v[28:29], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[26:27], v[26:27], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25], v[24:25], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[22:23], v[22:23], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21], v[20:21], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[18:19], v[18:19], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[16:17], v[16:17], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[14:15], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13], v[12:13], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[10:11], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[8:9], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[6:7], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[4:5], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[2:3], v[2:3], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[0:1], v[0:1], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x800
	s_branch .LBB0_25
.LBB0_32:
	s_set_vgpr_msb 0x82
	v_dual_mov_b32 v71 /*v583*/, v86 /*v598*/ :: v_dual_mov_b32 v69 /*v581*/, v87 /*v599*/
	s_set_vgpr_msb 0x8200
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
	s_lshl_b32 s80, s6, 5
	s_lshl_b32 s4, s6, 6
	s_and_b32 s26, s3, 0xffff
	s_mul_u64 s[44:45], s[2:3], s[4:5]
	s_sub_co_i32 s2, s103, s80
	s_mul_i32 s81, s6, 0x2200
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
	s_mov_b32 s24, 32
	s_mulk_i32 s6, 0x2400
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
	s_lshr_b32 s3, s3, 25
	s_add_co_i32 s3, s2, s3
	s_and_b32 s4, s3, 0xffffff80
	s_ashr_i32 s3, s3, 7
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
	s_addk_co_i32 s3, 0x7f
	s_ashr_i32 s4, s3, 31
	s_lshr_b32 s4, s4, 25
	s_add_co_i32 s2, s3, s4
	s_and_b32 s4, s2, 0xffffff80
	s_ashr_i32 s2, s2, 7
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
	s_set_vgpr_msb 34
	ds_load_b128 v[0:3], v83 /*v595*/
	ds_load_b128 v[4:7], v83 /*v595*/ offset:32
	ds_load_b128 v[8:11], v83 /*v595*/ offset:64
	ds_load_b128 v[12:15], v83 /*v595*/ offset:96
	ds_load_b128 v[16:19], v83 /*v595*/ offset:128
	ds_load_b128 v[20:23], v83 /*v595*/ offset:160
	ds_load_b128 v[24:27], v83 /*v595*/ offset:192
	ds_load_b128 v[28:31], v83 /*v595*/ offset:224
	ds_load_b128 v[32:35], v83 /*v595*/ offset:4352
	ds_load_b128 v[36:39], v83 /*v595*/ offset:4384
	ds_load_b128 v[40:43], v83 /*v595*/ offset:4416
	ds_load_b128 v[44:47], v83 /*v595*/ offset:4448
	ds_load_b128 v[48:51], v83 /*v595*/ offset:4480
	ds_load_b128 v[52:55], v83 /*v595*/ offset:4512
	ds_load_b128 v[56:59], v83 /*v595*/ offset:4544
	ds_load_b128 v[60:63], v83 /*v595*/ offset:4576
	tensor_load_to_lds s[28:31], s[20:27]
	s_add_nc_u64 s[30:31], s[46:47], s[76:77]
	s_mov_b32 s29, s78
	s_bitset1_b32 s31, 31
	s_wait_dscnt 0xf
	v_pk_mul_bf16 v128, v81 /*v593*/, v0
	tensor_load_to_lds s[28:31], s[36:43]
	v_and_or_b32 v0, v77 /*v589*/, 7, v79 /*v591*/
	v_pk_mul_bf16 v130, v81 /*v593*/, v2
	v_pk_mul_bf16 v129, v81 /*v593*/, v1
	s_set_vgpr_msb 0x2208
	v_dual_add_nc_u32 v1, s99, v74 /*v586*/ :: v_dual_add_nc_u32 v2, s99, v73 /*v585*/
	s_set_vgpr_msb 0x802
	v_mul_u32_u24_e32 v0, 0x120, v0
	s_wait_dscnt 0xe
	v_pk_mul_bf16 v135, v81 /*v593*/, v7
	v_pk_mul_bf16 v134, v81 /*v593*/, v6
	v_pk_mul_bf16 v133, v81 /*v593*/, v5
	v_pk_mul_bf16 v132, v81 /*v593*/, v4
	v_and_or_b32 v0, v82 /*v594*/, 16, v0
	v_pk_mul_bf16 v131, v81 /*v593*/, v3
	s_wait_dscnt 0xc
	v_pk_mul_bf16 v143, v81 /*v593*/, v15
	v_pk_mul_bf16 v142, v81 /*v593*/, v14
	v_pk_mul_bf16 v141, v81 /*v593*/, v13
	v_pk_mul_bf16 v140, v81 /*v593*/, v12
	v_pk_mul_bf16 v139, v81 /*v593*/, v11
	v_pk_mul_bf16 v138, v81 /*v593*/, v10
	v_pk_mul_bf16 v137, v81 /*v593*/, v9
	v_pk_mul_bf16 v136, v81 /*v593*/, v8
	s_wait_dscnt 0xa
	v_pk_mul_bf16 v151, v81 /*v593*/, v23
	v_pk_mul_bf16 v150, v81 /*v593*/, v22
	v_pk_mul_bf16 v149, v81 /*v593*/, v21
	v_pk_mul_bf16 v148, v81 /*v593*/, v20
	v_pk_mul_bf16 v147, v81 /*v593*/, v19
	v_pk_mul_bf16 v146, v81 /*v593*/, v18
	v_pk_mul_bf16 v145, v81 /*v593*/, v17
	v_pk_mul_bf16 v144, v81 /*v593*/, v16
	s_wait_dscnt 0x8
	v_pk_mul_bf16 v159, v81 /*v593*/, v31
	v_pk_mul_bf16 v158, v81 /*v593*/, v30
	v_pk_mul_bf16 v157, v81 /*v593*/, v29
	v_pk_mul_bf16 v156, v81 /*v593*/, v28
	v_pk_mul_bf16 v155, v81 /*v593*/, v27
	v_pk_mul_bf16 v154, v81 /*v593*/, v26
	v_pk_mul_bf16 v153, v81 /*v593*/, v25
	v_pk_mul_bf16 v152, v81 /*v593*/, v24
	s_wait_dscnt 0x6
	v_pk_mul_bf16 v167, v81 /*v593*/, v39
	v_pk_mul_bf16 v166, v81 /*v593*/, v38
	v_pk_mul_bf16 v165, v81 /*v593*/, v37
	v_pk_mul_bf16 v164, v81 /*v593*/, v36
	v_pk_mul_bf16 v163, v81 /*v593*/, v35
	v_pk_mul_bf16 v162, v81 /*v593*/, v34
	v_pk_mul_bf16 v161, v81 /*v593*/, v33
	v_pk_mul_bf16 v160, v81 /*v593*/, v32
	s_wait_dscnt 0x4
	v_pk_mul_bf16 v175, v81 /*v593*/, v47
	v_pk_mul_bf16 v174, v81 /*v593*/, v46
	v_pk_mul_bf16 v173, v81 /*v593*/, v45
	v_pk_mul_bf16 v172, v81 /*v593*/, v44
	v_pk_mul_bf16 v171, v81 /*v593*/, v43
	v_pk_mul_bf16 v170, v81 /*v593*/, v42
	v_pk_mul_bf16 v169, v81 /*v593*/, v41
	v_pk_mul_bf16 v168, v81 /*v593*/, v40
	s_wait_dscnt 0x2
	v_pk_mul_bf16 v183, v81 /*v593*/, v55
	v_pk_mul_bf16 v182, v81 /*v593*/, v54
	v_pk_mul_bf16 v181, v81 /*v593*/, v53
	v_pk_mul_bf16 v180, v81 /*v593*/, v52
	v_pk_mul_bf16 v179, v81 /*v593*/, v51
	v_pk_mul_bf16 v178, v81 /*v593*/, v50
	v_pk_mul_bf16 v177, v81 /*v593*/, v49
	v_pk_mul_bf16 v176, v81 /*v593*/, v48
	s_wait_dscnt 0x0
	v_pk_mul_bf16 v191, v81 /*v593*/, v63
	v_pk_mul_bf16 v190, v81 /*v593*/, v62
	v_pk_mul_bf16 v189, v81 /*v593*/, v61
	v_pk_mul_bf16 v188, v81 /*v593*/, v60
	v_pk_mul_bf16 v187, v81 /*v593*/, v59
	v_pk_mul_bf16 v186, v81 /*v593*/, v58
	v_pk_mul_bf16 v185, v81 /*v593*/, v57
	v_pk_mul_bf16 v184, v81 /*v593*/, v56
	s_set_vgpr_msb 0x280
	v_or_b32_e32 v81 /*v593*/, 0x10000, v0
	s_wait_alu depctr_vm_vsrc(0)
	v_or_b32_e32 v89 /*v601*/, 0x30000, v0
	v_dual_add_nc_u32 v85 /*v597*/, s17, v1 :: v_dual_add_nc_u32 v86 /*v598*/, s17, v2
	v_subrev_nc_u32_e32 v82 /*v594*/, s16, v1
	v_subrev_nc_u32_e32 v83 /*v595*/, s16, v2
	s_set_vgpr_msb 0x8000
	v_mov_b32_e32 v0, 0
	s_wait_tensorcnt 0x0
	s_set_vgpr_msb 0x82
	s_barrier_signal -1
	s_barrier_wait -1
	s_wait_kmcnt 0x0
	global_load_b32 v71 /*v583*/, v75 /*v587*/, s[70:71] scale_offset
	s_set_vgpr_msb 0x8200
	s_cbranch_scc1 .LBB0_43
	v_dual_mov_b32 v1, v0 :: v_dual_mov_b32 v2, v0
	v_dual_mov_b32 v3, v0 :: v_dual_mov_b32 v4, v0
	v_dual_mov_b32 v5, v0 :: v_dual_mov_b32 v6, v0
	v_mov_b32_e32 v7, v0
	s_set_vgpr_msb 64
	v_mov_b32_e32 v248 /*v504*/, 1.0
	s_set_vgpr_msb 0x4000
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
	s_set_vgpr_msb 2
	v_dual_mov_b32 v200, v89 /*v601*/ :: v_dual_mov_b32 v201, v80 /*v592*/
	s_set_vgpr_msb 0x282
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v84 /*v596*/, v78 /*v590*/ :: v_dual_mov_b32 v66 /*v578*/, v71 /*v583*/
	s_set_vgpr_msb 0x8241
	v_mov_b32_e32 v249 /*v505*/, v248 /*v504*/
	s_mov_b32 s61, s27
	s_sub_co_i32 s17, 1, s60
	s_mov_b32 s49, 0x76543210
	s_mov_b32 s56, 0x3fb8aa3b
	s_mov_b64 s[58:59], s[60:61]
	s_set_vgpr_msb 0x4100
	s_branch .LBB0_37
.LBB0_36:
	s_set_vgpr_msb 0x8a
	v_cvt_pk_bf16_f32 v99 /*v611*/, v14 /*v526*/, v26 /*v538*/
	s_set_vgpr_msb 0x8a89
	v_cvt_pk_bf16_f32 v98 /*v610*/, v254 /*v510*/, v10 /*v522*/
	s_set_vgpr_msb 0x8985
	v_cvt_pk_bf16_f32 v97 /*v609*/, v236 /*v492*/, v250 /*v506*/
	v_cvt_pk_bf16_f32 v96 /*v608*/, v222 /*v478*/, v232 /*v488*/
	v_cvt_pk_bf16_f32 v95 /*v607*/, v210 /*v466*/, v218 /*v474*/
	v_cvt_pk_bf16_f32 v94 /*v606*/, v202 /*v458*/, v208 /*v464*/
	v_cvt_pk_bf16_f32 v93 /*v605*/, v196 /*v452*/, v200 /*v456*/
	v_cvt_pk_bf16_f32 v92 /*v604*/, v192 /*v448*/, v194 /*v450*/
	s_set_vgpr_msb 0x858a
	v_cvt_pk_bf16_f32 v107 /*v619*/, v15 /*v527*/, v27 /*v539*/
	s_set_vgpr_msb 0x8a89
	v_cvt_pk_bf16_f32 v106 /*v618*/, v255 /*v511*/, v11 /*v523*/
	s_set_vgpr_msb 0x8985
	v_cvt_pk_bf16_f32 v105 /*v617*/, v237 /*v493*/, v251 /*v507*/
	v_cvt_pk_bf16_f32 v104 /*v616*/, v223 /*v479*/, v233 /*v489*/
	v_cvt_pk_bf16_f32 v103 /*v615*/, v211 /*v467*/, v219 /*v475*/
	v_cvt_pk_bf16_f32 v102 /*v614*/, v203 /*v459*/, v209 /*v465*/
	v_cvt_pk_bf16_f32 v101 /*v613*/, v197 /*v453*/, v201 /*v457*/
	v_cvt_pk_bf16_f32 v100 /*v612*/, v193 /*v449*/, v195 /*v451*/
	s_set_vgpr_msb 0x8509
	s_wait_dscnt 0x3d
	v_wmma_f32_16x16x32_bf16 v[120:127], v[184:191] /*v[440:447]*/, v[92:99] /*v[604:611]*/, v[120:127]
	s_set_vgpr_msb 0x98a
	v_cvt_pk_bf16_f32 v115 /*v627*/, v37 /*v549*/, v45 /*v557*/
	v_cvt_pk_bf16_f32 v114 /*v626*/, v23 /*v535*/, v33 /*v545*/
	v_cvt_pk_bf16_f32 v113 /*v625*/, v7 /*v519*/, v19 /*v531*/
	s_set_vgpr_msb 0x8a89
	v_cvt_pk_bf16_f32 v112 /*v624*/, v245 /*v501*/, v3 /*v515*/
	s_set_vgpr_msb 0x8985
	v_cvt_pk_bf16_f32 v111 /*v623*/, v229 /*v485*/, v241 /*v497*/
	v_cvt_pk_bf16_f32 v110 /*v622*/, v217 /*v473*/, v227 /*v483*/
	v_cvt_pk_bf16_f32 v109 /*v621*/, v207 /*v463*/, v215 /*v471*/
	s_set_vgpr_msb 0x8509
	v_wmma_f32_16x16x32_bf16 v[56:63], v[184:191] /*v[440:447]*/, v[100:107] /*v[612:619]*/, v[56:63]
	s_set_vgpr_msb 0x985
	v_cvt_pk_bf16_f32 v108 /*v620*/, v199 /*v455*/, v205 /*v461*/
	s_set_vgpr_msb 0x854a
	v_cvt_pk_bf16_f32 v199 /*v455*/, v53 /*v565*/, v59 /*v571*/
	v_cvt_pk_bf16_f32 v197 /*v453*/, v31 /*v543*/, v41 /*v553*/
	v_cvt_pk_bf16_f32 v196 /*v452*/, v17 /*v529*/, v29 /*v541*/
	v_cvt_pk_bf16_f32 v191 /*v447*/, v36 /*v548*/, v44 /*v556*/
	v_cvt_pk_bf16_f32 v190 /*v446*/, v22 /*v534*/, v32 /*v544*/
	v_cvt_pk_bf16_f32 v189 /*v445*/, v6 /*v518*/, v18 /*v530*/
	s_set_vgpr_msb 0x4a09
	s_wait_dscnt 0x3c
	v_wmma_f32_16x16x32_bf16 v[112:119], v[152:159] /*v[408:415]*/, v[92:99] /*v[604:611]*/, v[112:119]
	s_set_vgpr_msb 0x949
	v_cvt_pk_bf16_f32 v188 /*v444*/, v244 /*v500*/, v2 /*v514*/
	s_set_vgpr_msb 0x4945
	v_cvt_pk_bf16_f32 v187 /*v443*/, v228 /*v484*/, v240 /*v496*/
	v_cvt_pk_bf16_f32 v186 /*v442*/, v216 /*v472*/, v226 /*v482*/
	v_cvt_pk_bf16_f32 v185 /*v441*/, v206 /*v462*/, v214 /*v470*/
	v_cvt_pk_bf16_f32 v184 /*v440*/, v198 /*v454*/, v204 /*v460*/
	s_set_vgpr_msb 0x454a
	v_cvt_pk_bf16_f32 v198 /*v454*/, v43 /*v555*/, v51 /*v563*/
	v_cvt_pk_bf16_f32 v195 /*v451*/, v1 /*v513*/, v13 /*v525*/
	s_set_vgpr_msb 0x4a09
	v_wmma_f32_16x16x32_bf16 v[48:55], v[152:159] /*v[408:415]*/, v[100:107] /*v[612:619]*/, v[48:55]
	s_set_vgpr_msb 0x945
	v_cvt_pk_bf16_f32 v194 /*v450*/, v239 /*v495*/, v253 /*v509*/
	v_cvt_pk_bf16_f32 v193 /*v449*/, v225 /*v481*/, v235 /*v491*/
	v_cvt_pk_bf16_f32 v192 /*v448*/, v213 /*v469*/, v221 /*v477*/
	s_set_vgpr_msb 0x454a
	v_cvt_pk_bf16_f32 v207 /*v463*/, v63 /*v575*/, v65 /*v577*/
	v_cvt_pk_bf16_f32 v206 /*v462*/, v57 /*v569*/, v61 /*v573*/
	v_cvt_pk_bf16_f32 v205 /*v461*/, v49 /*v561*/, v55 /*v567*/
	v_cvt_pk_bf16_f32 v204 /*v460*/, v39 /*v551*/, v47 /*v559*/
	s_set_vgpr_msb 0x4a09
	s_wait_dscnt 0x2d
	v_wmma_f32_16x16x32_bf16 v[104:111], v[120:127] /*v[376:383]*/, v[92:99] /*v[604:611]*/, v[104:111]
	s_set_vgpr_msb 0x94a
	v_cvt_pk_bf16_f32 v203 /*v459*/, v25 /*v537*/, v35 /*v547*/
	v_cvt_pk_bf16_f32 v202 /*v458*/, v9 /*v521*/, v21 /*v533*/
	s_set_vgpr_msb 0x4a49
	v_cvt_pk_bf16_f32 v201 /*v457*/, v247 /*v503*/, v5 /*v517*/
	s_set_vgpr_msb 0x4945
	v_cvt_pk_bf16_f32 v200 /*v456*/, v231 /*v487*/, v243 /*v499*/
	s_add_nc_u64 s[58:59], s[58:59], 1
	s_wait_dscnt 0x0
	v_cmp_lt_u64_e64 s2, s[58:59], s[54:55]
	s_set_vgpr_msb 0x4509
	v_wmma_f32_16x16x32_bf16 v[40:47], v[120:127] /*v[376:383]*/, v[100:107] /*v[612:619]*/, v[40:47]
	s_and_b32 vcc_lo, exec_lo, s2
	v_wmma_f32_16x16x32_bf16 v[96:103], v[88:95] /*v[344:351]*/, v[92:99] /*v[604:611]*/, v[96:103]
	v_wmma_f32_16x16x32_bf16 v[32:39], v[88:95] /*v[344:351]*/, v[100:107] /*v[612:619]*/, v[32:39]
	v_wmma_f32_16x16x32_bf16 v[88:95], v[56:63] /*v[312:319]*/, v[92:99] /*v[604:611]*/, v[88:95]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[56:63] /*v[312:319]*/, v[100:107] /*v[612:619]*/, v[24:31]
	v_wmma_f32_16x16x32_bf16 v[80:87], v[24:31] /*v[280:287]*/, v[92:99] /*v[604:611]*/, v[80:87]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[24:31] /*v[280:287]*/, v[100:107] /*v[612:619]*/, v[16:23]
	s_set_vgpr_msb 0x908
	v_wmma_f32_16x16x32_bf16 v[72:79], v[248:255], v[92:99] /*v[604:611]*/, v[72:79]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[248:255], v[100:107] /*v[612:619]*/, v[8:15]
	v_wmma_f32_16x16x32_bf16 v[64:71], v[216:223], v[92:99] /*v[604:611]*/, v[64:71]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[216:223], v[100:107] /*v[612:619]*/, v[0:7]
	s_set_vgpr_msb 0x805
	v_wmma_f32_16x16x32_bf16 v[120:127], v[176:183] /*v[432:439]*/, v[184:191] /*v[440:447]*/, v[120:127]
	s_set_vgpr_msb 0x509
	v_wmma_f32_16x16x32_bf16 v[56:63], v[176:183] /*v[432:439]*/, v[108:115] /*v[620:627]*/, v[56:63]
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x94a
	v_cvt_pk_bf16_f32 v183 /*v439*/, v52 /*v564*/, v58 /*v570*/
	v_cvt_pk_bf16_f32 v182 /*v438*/, v42 /*v554*/, v50 /*v562*/
	v_cvt_pk_bf16_f32 v181 /*v437*/, v30 /*v542*/, v40 /*v552*/
	s_set_vgpr_msb 0x4a05
	v_wmma_f32_16x16x32_bf16 v[112:119], v[144:151] /*v[400:407]*/, v[184:191] /*v[440:447]*/, v[112:119]
	s_set_vgpr_msb 0x54a
	v_cvt_pk_bf16_f32 v180 /*v436*/, v16 /*v528*/, v28 /*v540*/
	v_cvt_pk_bf16_f32 v179 /*v435*/, v0 /*v512*/, v12 /*v524*/
	s_set_vgpr_msb 0x4a45
	v_cvt_pk_bf16_f32 v178 /*v434*/, v238 /*v494*/, v252 /*v508*/
	v_cvt_pk_bf16_f32 v177 /*v433*/, v224 /*v480*/, v234 /*v490*/
	v_cvt_pk_bf16_f32 v176 /*v432*/, v212 /*v468*/, v220 /*v476*/
	s_set_vgpr_msb 0x4509
	v_wmma_f32_16x16x32_bf16 v[48:55], v[144:151] /*v[400:407]*/, v[108:115] /*v[620:627]*/, v[48:55]
	s_set_vgpr_msb 0x905
	v_wmma_f32_16x16x32_bf16 v[104:111], v[112:119] /*v[368:375]*/, v[184:191] /*v[440:447]*/, v[104:111]
	s_set_vgpr_msb 0x509
	v_wmma_f32_16x16x32_bf16 v[40:47], v[112:119] /*v[368:375]*/, v[108:115] /*v[620:627]*/, v[40:47]
	s_set_vgpr_msb 0x905
	v_wmma_f32_16x16x32_bf16 v[96:103], v[80:87] /*v[336:343]*/, v[184:191] /*v[440:447]*/, v[96:103]
	s_set_vgpr_msb 0x509
	v_wmma_f32_16x16x32_bf16 v[32:39], v[80:87] /*v[336:343]*/, v[108:115] /*v[620:627]*/, v[32:39]
	s_set_vgpr_msb 0x905
	v_wmma_f32_16x16x32_bf16 v[88:95], v[48:55] /*v[304:311]*/, v[184:191] /*v[440:447]*/, v[88:95]
	s_set_vgpr_msb 0x509
	v_wmma_f32_16x16x32_bf16 v[24:31], v[48:55] /*v[304:311]*/, v[108:115] /*v[620:627]*/, v[24:31]
	s_set_vgpr_msb 0x905
	v_wmma_f32_16x16x32_bf16 v[80:87], v[16:23] /*v[272:279]*/, v[184:191] /*v[440:447]*/, v[80:87]
	s_set_vgpr_msb 0x509
	v_wmma_f32_16x16x32_bf16 v[16:23], v[16:23] /*v[272:279]*/, v[108:115] /*v[620:627]*/, v[16:23]
	s_set_vgpr_msb 0x904
	v_wmma_f32_16x16x32_bf16 v[72:79], v[240:247], v[184:191] /*v[440:447]*/, v[72:79]
	s_set_vgpr_msb 0x408
	v_wmma_f32_16x16x32_bf16 v[8:15], v[240:247], v[108:115] /*v[620:627]*/, v[8:15]
	s_set_vgpr_msb 0x804
	v_wmma_f32_16x16x32_bf16 v[64:71], v[208:215], v[184:191] /*v[440:447]*/, v[64:71]
	s_set_vgpr_msb 0x408
	v_wmma_f32_16x16x32_bf16 v[0:7], v[208:215], v[108:115] /*v[620:627]*/, v[0:7]
	s_set_vgpr_msb 0x805
	v_wmma_f32_16x16x32_bf16 v[120:127], v[168:175] /*v[424:431]*/, v[176:183] /*v[432:439]*/, v[120:127]
	v_wmma_f32_16x16x32_bf16 v[56:63], v[168:175] /*v[424:431]*/, v[192:199] /*v[448:455]*/, v[56:63]
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x54a
	v_cvt_pk_bf16_f32 v175 /*v431*/, v62 /*v574*/, v64 /*v576*/
	v_cvt_pk_bf16_f32 v174 /*v430*/, v56 /*v568*/, v60 /*v572*/
	v_cvt_pk_bf16_f32 v173 /*v429*/, v48 /*v560*/, v54 /*v566*/
	s_set_vgpr_msb 0x4a05
	v_wmma_f32_16x16x32_bf16 v[112:119], v[136:143] /*v[392:399]*/, v[176:183] /*v[432:439]*/, v[112:119]
	s_set_vgpr_msb 0x54a
	v_cvt_pk_bf16_f32 v172 /*v428*/, v38 /*v550*/, v46 /*v558*/
	v_cvt_pk_bf16_f32 v171 /*v427*/, v24 /*v536*/, v34 /*v546*/
	v_cvt_pk_bf16_f32 v170 /*v426*/, v8 /*v520*/, v20 /*v532*/
	s_set_vgpr_msb 0x4a49
	v_cvt_pk_bf16_f32 v169 /*v425*/, v246 /*v502*/, v4 /*v516*/
	s_set_vgpr_msb 0x4945
	v_cvt_pk_bf16_f32 v168 /*v424*/, v230 /*v486*/, v242 /*v498*/
	s_set_vgpr_msb 0x4505
	v_wmma_f32_16x16x32_bf16 v[48:55], v[136:143] /*v[392:399]*/, v[192:199] /*v[448:455]*/, v[48:55]
	v_wmma_f32_16x16x32_bf16 v[104:111], v[104:111] /*v[360:367]*/, v[176:183] /*v[432:439]*/, v[104:111]
	v_wmma_f32_16x16x32_bf16 v[40:47], v[104:111] /*v[360:367]*/, v[192:199] /*v[448:455]*/, v[40:47]
	v_wmma_f32_16x16x32_bf16 v[96:103], v[72:79] /*v[328:335]*/, v[176:183] /*v[432:439]*/, v[96:103]
	v_wmma_f32_16x16x32_bf16 v[32:39], v[72:79] /*v[328:335]*/, v[192:199] /*v[448:455]*/, v[32:39]
	v_wmma_f32_16x16x32_bf16 v[88:95], v[40:47] /*v[296:303]*/, v[176:183] /*v[432:439]*/, v[88:95]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[40:47] /*v[296:303]*/, v[192:199] /*v[448:455]*/, v[24:31]
	v_wmma_f32_16x16x32_bf16 v[80:87], v[8:15] /*v[264:271]*/, v[176:183] /*v[432:439]*/, v[80:87]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[8:15] /*v[264:271]*/, v[192:199] /*v[448:455]*/, v[16:23]
	s_set_vgpr_msb 0x504
	v_wmma_f32_16x16x32_bf16 v[72:79], v[232:239], v[176:183] /*v[432:439]*/, v[72:79]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[232:239], v[192:199] /*v[448:455]*/, v[8:15]
	v_wmma_f32_16x16x32_bf16 v[64:71], v[200:207], v[176:183] /*v[432:439]*/, v[64:71]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[200:207], v[192:199] /*v[448:455]*/, v[0:7]
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x426
	v_pk_fma_f32 v[200:201], v[70:71] /*v[582:583]*/, v[248:249] /*v[504:505]*/, v[68:69] /*v[580:581]*/
	s_set_vgpr_msb 0x2682
	v_mov_b32_e32 v71 /*v583*/, v87 /*v599*/
	s_set_vgpr_msb 0x8248
	v_pk_add_f32 v[248:249] /*v[504:505]*/, v[200:201], v[66:67] /*v[578:579]*/
	s_set_vgpr_msb 0x4805
	v_wmma_f32_16x16x32_bf16 v[120:127], v[160:167] /*v[416:423]*/, v[168:175] /*v[424:431]*/, v[120:127]
	s_set_vgpr_msb 0x502
	v_dual_mov_b32 v200, v89 /*v601*/ :: v_dual_mov_b32 v201, v80 /*v592*/
	s_set_vgpr_msb 0x282
	v_mov_b32_e32 v66 /*v578*/, v90 /*v602*/
	s_set_vgpr_msb 0x8205
	v_wmma_f32_16x16x32_bf16 v[56:63], v[160:167] /*v[416:423]*/, v[200:207] /*v[456:463]*/, v[56:63]
	v_wmma_f32_16x16x32_bf16 v[112:119], v[128:135] /*v[384:391]*/, v[168:175] /*v[424:431]*/, v[112:119]
	v_wmma_f32_16x16x32_bf16 v[48:55], v[128:135] /*v[384:391]*/, v[200:207] /*v[456:463]*/, v[48:55]
	v_wmma_f32_16x16x32_bf16 v[104:111], v[96:103] /*v[352:359]*/, v[168:175] /*v[424:431]*/, v[104:111]
	v_wmma_f32_16x16x32_bf16 v[40:47], v[96:103] /*v[352:359]*/, v[200:207] /*v[456:463]*/, v[40:47]
	v_wmma_f32_16x16x32_bf16 v[96:103], v[64:71] /*v[320:327]*/, v[168:175] /*v[424:431]*/, v[96:103]
	v_wmma_f32_16x16x32_bf16 v[32:39], v[64:71] /*v[320:327]*/, v[200:207] /*v[456:463]*/, v[32:39]
	v_wmma_f32_16x16x32_bf16 v[88:95], v[32:39] /*v[288:295]*/, v[168:175] /*v[424:431]*/, v[88:95]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[32:39] /*v[288:295]*/, v[200:207] /*v[456:463]*/, v[24:31]
	v_wmma_f32_16x16x32_bf16 v[80:87], v[0:7] /*v[256:263]*/, v[168:175] /*v[424:431]*/, v[80:87]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[0:7] /*v[256:263]*/, v[200:207] /*v[456:463]*/, v[16:23]
	s_set_vgpr_msb 0x504
	v_wmma_f32_16x16x32_bf16 v[72:79], v[224:231], v[168:175] /*v[424:431]*/, v[72:79]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[224:231], v[200:207] /*v[456:463]*/, v[8:15]
	v_wmma_f32_16x16x32_bf16 v[64:71], v[192:199], v[168:175] /*v[424:431]*/, v[64:71]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[192:199], v[200:207] /*v[456:463]*/, v[0:7]
	s_set_vgpr_msb 0x400
	s_cbranch_vccz .LBB0_44
.LBB0_37:
	s_wait_alu depctr_vm_vsrc(0)
	s_set_vgpr_msb 0x82
	v_dual_mov_b32 v80 /*v592*/, v84 /*v596*/ :: v_dual_mov_b32 v89 /*v601*/, v81 /*v593*/
	s_set_vgpr_msb 0x8280
	v_dual_mov_b32 v84 /*v596*/, v201 :: v_dual_mov_b32 v81 /*v593*/, v200
	s_add_co_i32 s2, s58, 1
	s_wait_tensorcnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x8042
	ds_load_b128 v[184:187] /*v[440:443]*/, v80 /*v592*/
	ds_load_b128 v[188:191] /*v[444:447]*/, v80 /*v592*/ offset:32
	ds_load_b128 v[176:179] /*v[432:435]*/, v80 /*v592*/ offset:64
	ds_load_b128 v[180:183] /*v[436:439]*/, v80 /*v592*/ offset:96
	ds_load_b128 v[168:171] /*v[424:427]*/, v80 /*v592*/ offset:128
	ds_load_b128 v[172:175] /*v[428:431]*/, v80 /*v592*/ offset:160
	ds_load_b128 v[160:163] /*v[416:419]*/, v80 /*v592*/ offset:192
	ds_load_b128 v[164:167] /*v[420:423]*/, v80 /*v592*/ offset:224
	ds_load_b128 v[152:155] /*v[408:411]*/, v80 /*v592*/ offset:4352
	ds_load_b128 v[156:159] /*v[412:415]*/, v80 /*v592*/ offset:4384
	ds_load_b128 v[144:147] /*v[400:403]*/, v80 /*v592*/ offset:4416
	ds_load_b128 v[148:151] /*v[404:407]*/, v80 /*v592*/ offset:4448
	ds_load_b128 v[136:139] /*v[392:395]*/, v80 /*v592*/ offset:4480
	ds_load_b128 v[140:143] /*v[396:399]*/, v80 /*v592*/ offset:4512
	ds_load_b128 v[128:131] /*v[384:387]*/, v80 /*v592*/ offset:4544
	ds_load_b128 v[132:135] /*v[388:391]*/, v80 /*v592*/ offset:4576
	ds_load_b128 v[120:123] /*v[376:379]*/, v80 /*v592*/ offset:8704
	ds_load_b128 v[124:127] /*v[380:383]*/, v80 /*v592*/ offset:8736
	ds_load_b128 v[112:115] /*v[368:371]*/, v80 /*v592*/ offset:8768
	ds_load_b128 v[116:119] /*v[372:375]*/, v80 /*v592*/ offset:8800
	ds_load_b128 v[104:107] /*v[360:363]*/, v80 /*v592*/ offset:8832
	ds_load_b128 v[108:111] /*v[364:367]*/, v80 /*v592*/ offset:8864
	ds_load_b128 v[96:99] /*v[352:355]*/, v80 /*v592*/ offset:8896
	ds_load_b128 v[100:103] /*v[356:359]*/, v80 /*v592*/ offset:8928
	ds_load_b128 v[88:91] /*v[344:347]*/, v80 /*v592*/ offset:13056
	ds_load_b128 v[92:95] /*v[348:351]*/, v80 /*v592*/ offset:13088
	ds_load_b128 v[80:83] /*v[336:339]*/, v80 /*v592*/ offset:13120
	ds_load_b128 v[84:87] /*v[340:343]*/, v80 /*v592*/ offset:13152
	ds_load_b128 v[72:75] /*v[328:331]*/, v80 /*v592*/ offset:13184
	ds_load_b128 v[76:79] /*v[332:335]*/, v80 /*v592*/ offset:13216
	ds_load_b128 v[64:67] /*v[320:323]*/, v80 /*v592*/ offset:13248
	ds_load_b128 v[68:71] /*v[324:327]*/, v80 /*v592*/ offset:13280
	ds_load_b128 v[56:59] /*v[312:315]*/, v80 /*v592*/ offset:17408
	ds_load_b128 v[60:63] /*v[316:319]*/, v80 /*v592*/ offset:17440
	ds_load_b128 v[48:51] /*v[304:307]*/, v80 /*v592*/ offset:17472
	ds_load_b128 v[52:55] /*v[308:311]*/, v80 /*v592*/ offset:17504
	ds_load_b128 v[40:43] /*v[296:299]*/, v80 /*v592*/ offset:17536
	ds_load_b128 v[44:47] /*v[300:303]*/, v80 /*v592*/ offset:17568
	ds_load_b128 v[32:35] /*v[288:291]*/, v80 /*v592*/ offset:17600
	ds_load_b128 v[36:39] /*v[292:295]*/, v80 /*v592*/ offset:17632
	ds_load_b128 v[24:27] /*v[280:283]*/, v80 /*v592*/ offset:21760
	ds_load_b128 v[28:31] /*v[284:287]*/, v80 /*v592*/ offset:21792
	ds_load_b128 v[16:19] /*v[272:275]*/, v80 /*v592*/ offset:21824
	ds_load_b128 v[20:23] /*v[276:279]*/, v80 /*v592*/ offset:21856
	ds_load_b128 v[8:11] /*v[264:267]*/, v80 /*v592*/ offset:21888
	ds_load_b128 v[12:15] /*v[268:271]*/, v80 /*v592*/ offset:21920
	ds_load_b128 v[0:3] /*v[256:259]*/, v80 /*v592*/ offset:21952
	ds_load_b128 v[4:7] /*v[260:263]*/, v80 /*v592*/ offset:21984
	s_set_vgpr_msb 0x4202
	ds_load_b128 v[248:251], v80 /*v592*/ offset:26112
	ds_load_b128 v[252:255], v80 /*v592*/ offset:26144
	ds_load_b128 v[240:243], v80 /*v592*/ offset:26176
	ds_load_b128 v[244:247], v80 /*v592*/ offset:26208
	ds_load_b128 v[232:235], v80 /*v592*/ offset:26240
	ds_load_b128 v[236:239], v80 /*v592*/ offset:26272
	ds_load_b128 v[224:227], v80 /*v592*/ offset:26304
	ds_load_b128 v[228:231], v80 /*v592*/ offset:26336
	ds_load_b128 v[216:219], v80 /*v592*/ offset:30464
	ds_load_b128 v[220:223], v80 /*v592*/ offset:30496
	ds_load_b128 v[208:211], v80 /*v592*/ offset:30528
	ds_load_b128 v[212:215], v80 /*v592*/ offset:30560
	ds_load_b128 v[200:203], v80 /*v592*/ offset:30592
	ds_load_b128 v[204:207], v80 /*v592*/ offset:30624
	ds_load_b128 v[192:195], v80 /*v592*/ offset:30656
	ds_load_b128 v[196:199], v80 /*v592*/ offset:30688
	s_cmp_ge_i32 s2, s10
	s_set_vgpr_msb 0x200
	s_cbranch_scc1 .LBB0_39
	s_lshl_b32 s4, s2, 7
	s_add_co_i32 s6, s58, s17
	s_sub_co_i32 s29, s9, s4
	s_add_co_i32 s2, s4, s97
	s_set_vgpr_msb 0x41
	v_med3_i32 v192 /*v448*/, s29, 0, 0x80
	s_lshr_b32 s22, s6, 31
	s_ashr_i32 s3, s2, 31
	s_add_co_i32 s22, s6, s22
	s_mul_u64 s[4:5], s[2:3], s[64:65]
	v_readfirstlane_b32 s29, v192 /*v448*/
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
	s_wait_dscnt 0x3e
	v_wmma_f32_16x16x32_bf16 v[250:257] /*v[506:513]*/, v[184:191] /*v[440:447]*/, v[128:135], 0
	v_wmma_f32_16x16x32_bf16 v[192:199] /*v[448:455]*/, v[184:191] /*v[440:447]*/, v[160:167], 0
	s_set_vgpr_msb 0x4181
	s_wait_dscnt 0x36
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[514:521]*/, v[152:159] /*v[408:415]*/, v[128:135], 0
	s_set_vgpr_msb 0x8141
	v_wmma_f32_16x16x32_bf16 v[200:207] /*v[456:463]*/, v[152:159] /*v[408:415]*/, v[160:167], 0
	s_set_vgpr_msb 0x4181
	s_wait_dscnt 0x2e
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[522:529]*/, v[120:127] /*v[376:383]*/, v[128:135], 0
	s_set_vgpr_msb 0x8141
	v_wmma_f32_16x16x32_bf16 v[208:215] /*v[464:471]*/, v[120:127] /*v[376:383]*/, v[160:167], 0
	s_set_vgpr_msb 0x4181
	s_wait_dscnt 0x26
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[530:537]*/, v[88:95] /*v[344:351]*/, v[128:135], 0
	s_set_vgpr_msb 0x8141
	v_wmma_f32_16x16x32_bf16 v[216:223] /*v[472:479]*/, v[88:95] /*v[344:351]*/, v[160:167], 0
	s_set_vgpr_msb 0x4181
	s_wait_dscnt 0x1e
	v_wmma_f32_16x16x32_bf16 v[26:33] /*v[538:545]*/, v[56:63] /*v[312:319]*/, v[128:135], 0
	s_set_vgpr_msb 0x8141
	v_wmma_f32_16x16x32_bf16 v[224:231] /*v[480:487]*/, v[56:63] /*v[312:319]*/, v[160:167], 0
	s_set_vgpr_msb 0x4181
	s_wait_dscnt 0x16
	v_wmma_f32_16x16x32_bf16 v[34:41] /*v[546:553]*/, v[24:31] /*v[280:287]*/, v[128:135], 0
	s_set_vgpr_msb 0x8141
	v_wmma_f32_16x16x32_bf16 v[232:239] /*v[488:495]*/, v[24:31] /*v[280:287]*/, v[160:167], 0
	s_set_vgpr_msb 0x4180
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[554:561]*/, v[248:255], v[128:135], 0
	s_set_vgpr_msb 0x8040
	v_wmma_f32_16x16x32_bf16 v[240:247] /*v[496:503]*/, v[248:255], v[160:167], 0
	s_set_vgpr_msb 0x4080
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[562:569]*/, v[216:223], v[128:135], 0
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[570:577]*/, v[216:223], v[160:167], 0
	s_set_vgpr_msb 0x8051
	v_wmma_f32_16x16x32_bf16 v[250:257] /*v[506:513]*/, v[176:183] /*v[432:439]*/, v[136:143], v[250:257] /*v[506:513]*/
	v_wmma_f32_16x16x32_bf16 v[192:199] /*v[448:455]*/, v[176:183] /*v[432:439]*/, v[168:175], v[192:199] /*v[448:455]*/
	s_set_vgpr_msb 0x51a1
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[514:521]*/, v[144:151] /*v[400:407]*/, v[136:143], v[2:9] /*v[514:521]*/
	s_set_vgpr_msb 0xa151
	v_wmma_f32_16x16x32_bf16 v[200:207] /*v[456:463]*/, v[144:151] /*v[400:407]*/, v[168:175], v[200:207] /*v[456:463]*/
	s_set_vgpr_msb 0x51a1
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[522:529]*/, v[112:119] /*v[368:375]*/, v[136:143], v[10:17] /*v[522:529]*/
	s_set_vgpr_msb 0xa151
	v_wmma_f32_16x16x32_bf16 v[208:215] /*v[464:471]*/, v[112:119] /*v[368:375]*/, v[168:175], v[208:215] /*v[464:471]*/
	s_set_vgpr_msb 0x51a1
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[530:537]*/, v[80:87] /*v[336:343]*/, v[136:143], v[18:25] /*v[530:537]*/
	s_set_vgpr_msb 0xa151
	v_wmma_f32_16x16x32_bf16 v[216:223] /*v[472:479]*/, v[80:87] /*v[336:343]*/, v[168:175], v[216:223] /*v[472:479]*/
	s_set_vgpr_msb 0x51a1
	v_wmma_f32_16x16x32_bf16 v[26:33] /*v[538:545]*/, v[48:55] /*v[304:311]*/, v[136:143], v[26:33] /*v[538:545]*/
	s_set_vgpr_msb 0xa151
	v_wmma_f32_16x16x32_bf16 v[224:231] /*v[480:487]*/, v[48:55] /*v[304:311]*/, v[168:175], v[224:231] /*v[480:487]*/
	s_set_vgpr_msb 0x51a1
	v_wmma_f32_16x16x32_bf16 v[34:41] /*v[546:553]*/, v[16:23] /*v[272:279]*/, v[136:143], v[34:41] /*v[546:553]*/
	s_set_vgpr_msb 0xa151
	v_wmma_f32_16x16x32_bf16 v[232:239] /*v[488:495]*/, v[16:23] /*v[272:279]*/, v[168:175], v[232:239] /*v[488:495]*/
	s_set_vgpr_msb 0x51a0
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[554:561]*/, v[240:247], v[136:143], v[42:49] /*v[554:561]*/
	s_set_vgpr_msb 0xa050
	v_wmma_f32_16x16x32_bf16 v[240:247] /*v[496:503]*/, v[240:247], v[168:175], v[240:247] /*v[496:503]*/
	s_set_vgpr_msb 0x50a0
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[562:569]*/, v[208:215], v[136:143], v[50:57] /*v[562:569]*/
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[570:577]*/, v[208:215], v[168:175], v[58:65] /*v[570:577]*/
	s_set_vgpr_msb 0xa051
	v_wmma_f32_16x16x32_bf16 v[250:257] /*v[506:513]*/, v[168:175] /*v[424:431]*/, v[144:151], v[250:257] /*v[506:513]*/
	v_wmma_f32_16x16x32_bf16 v[192:199] /*v[448:455]*/, v[168:175] /*v[424:431]*/, v[176:183], v[192:199] /*v[448:455]*/
	s_set_vgpr_msb 0x51a1
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[514:521]*/, v[136:143] /*v[392:399]*/, v[144:151], v[2:9] /*v[514:521]*/
	s_set_vgpr_msb 0xa151
	v_wmma_f32_16x16x32_bf16 v[200:207] /*v[456:463]*/, v[136:143] /*v[392:399]*/, v[176:183], v[200:207] /*v[456:463]*/
	s_set_vgpr_msb 0x51a1
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[522:529]*/, v[104:111] /*v[360:367]*/, v[144:151], v[10:17] /*v[522:529]*/
	s_set_vgpr_msb 0xa151
	v_wmma_f32_16x16x32_bf16 v[208:215] /*v[464:471]*/, v[104:111] /*v[360:367]*/, v[176:183], v[208:215] /*v[464:471]*/
	s_set_vgpr_msb 0x51a1
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[530:537]*/, v[72:79] /*v[328:335]*/, v[144:151], v[18:25] /*v[530:537]*/
	s_set_vgpr_msb 0xa151
	v_wmma_f32_16x16x32_bf16 v[216:223] /*v[472:479]*/, v[72:79] /*v[328:335]*/, v[176:183], v[216:223] /*v[472:479]*/
	s_set_vgpr_msb 0x51a1
	v_wmma_f32_16x16x32_bf16 v[26:33] /*v[538:545]*/, v[40:47] /*v[296:303]*/, v[144:151], v[26:33] /*v[538:545]*/
	s_set_vgpr_msb 0xa151
	v_wmma_f32_16x16x32_bf16 v[224:231] /*v[480:487]*/, v[40:47] /*v[296:303]*/, v[176:183], v[224:231] /*v[480:487]*/
	s_set_vgpr_msb 0x51a1
	v_wmma_f32_16x16x32_bf16 v[34:41] /*v[546:553]*/, v[8:15] /*v[264:271]*/, v[144:151], v[34:41] /*v[546:553]*/
	s_set_vgpr_msb 0xa151
	v_wmma_f32_16x16x32_bf16 v[232:239] /*v[488:495]*/, v[8:15] /*v[264:271]*/, v[176:183], v[232:239] /*v[488:495]*/
	s_set_vgpr_msb 0x51a0
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[554:561]*/, v[232:239], v[144:151], v[42:49] /*v[554:561]*/
	s_set_vgpr_msb 0xa050
	v_wmma_f32_16x16x32_bf16 v[240:247] /*v[496:503]*/, v[232:239], v[176:183], v[240:247] /*v[496:503]*/
	s_set_vgpr_msb 0x50a0
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[562:569]*/, v[200:207], v[144:151], v[50:57] /*v[562:569]*/
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[570:577]*/, v[200:207], v[176:183], v[58:65] /*v[570:577]*/
	s_set_vgpr_msb 0xa051
	v_wmma_f32_16x16x32_bf16 v[250:257] /*v[506:513]*/, v[160:167] /*v[416:423]*/, v[152:159], v[250:257] /*v[506:513]*/
	v_wmma_f32_16x16x32_bf16 v[192:199] /*v[448:455]*/, v[160:167] /*v[416:423]*/, v[184:191], v[192:199] /*v[448:455]*/
	s_set_vgpr_msb 0x51a1
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[514:521]*/, v[128:135] /*v[384:391]*/, v[152:159], v[2:9] /*v[514:521]*/
	s_set_vgpr_msb 0xa151
	v_wmma_f32_16x16x32_bf16 v[200:207] /*v[456:463]*/, v[128:135] /*v[384:391]*/, v[184:191], v[200:207] /*v[456:463]*/
	s_set_vgpr_msb 0x51a1
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[522:529]*/, v[96:103] /*v[352:359]*/, v[152:159], v[10:17] /*v[522:529]*/
	s_set_vgpr_msb 0xa151
	v_wmma_f32_16x16x32_bf16 v[208:215] /*v[464:471]*/, v[96:103] /*v[352:359]*/, v[184:191], v[208:215] /*v[464:471]*/
	s_set_vgpr_msb 0x51a1
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[530:537]*/, v[64:71] /*v[320:327]*/, v[152:159], v[18:25] /*v[530:537]*/
	s_set_vgpr_msb 0xa151
	v_wmma_f32_16x16x32_bf16 v[216:223] /*v[472:479]*/, v[64:71] /*v[320:327]*/, v[184:191], v[216:223] /*v[472:479]*/
	s_set_vgpr_msb 0x51a1
	v_wmma_f32_16x16x32_bf16 v[26:33] /*v[538:545]*/, v[32:39] /*v[288:295]*/, v[152:159], v[26:33] /*v[538:545]*/
	s_set_vgpr_msb 0xa151
	v_wmma_f32_16x16x32_bf16 v[224:231] /*v[480:487]*/, v[32:39] /*v[288:295]*/, v[184:191], v[224:231] /*v[480:487]*/
	s_set_vgpr_msb 0x51a1
	v_wmma_f32_16x16x32_bf16 v[34:41] /*v[546:553]*/, v[0:7] /*v[256:263]*/, v[152:159], v[34:41] /*v[546:553]*/
	s_set_vgpr_msb 0xa151
	v_wmma_f32_16x16x32_bf16 v[232:239] /*v[488:495]*/, v[0:7] /*v[256:263]*/, v[184:191], v[232:239] /*v[488:495]*/
	s_set_vgpr_msb 0x51a0
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[554:561]*/, v[224:231], v[152:159], v[42:49] /*v[554:561]*/
	s_set_vgpr_msb 0xa050
	v_wmma_f32_16x16x32_bf16 v[240:247] /*v[496:503]*/, v[224:231], v[184:191], v[240:247] /*v[496:503]*/
	s_set_vgpr_msb 0x50a0
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[562:569]*/, v[192:199], v[152:159], v[50:57] /*v[562:569]*/
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[570:577]*/, v[192:199], v[184:191], v[58:65] /*v[570:577]*/
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0xa042
	ds_load_tr16_b128 v[184:187] /*v[440:443]*/, v89 /*v601*/
	ds_load_tr16_b128 v[152:155] /*v[408:411]*/, v89 /*v601*/ offset:32
	ds_load_tr16_b128 v[188:191] /*v[444:447]*/, v89 /*v601*/ offset:4608
	ds_load_tr16_b128 v[156:159] /*v[412:415]*/, v89 /*v601*/ offset:4640
	ds_load_tr16_b128 v[176:179] /*v[432:435]*/, v89 /*v601*/ offset:9216
	ds_load_tr16_b128 v[144:147] /*v[400:403]*/, v89 /*v601*/ offset:9248
	ds_load_tr16_b128 v[180:183] /*v[436:439]*/, v89 /*v601*/ offset:13824
	ds_load_tr16_b128 v[148:151] /*v[404:407]*/, v89 /*v601*/ offset:13856
	ds_load_tr16_b128 v[168:171] /*v[424:427]*/, v89 /*v601*/ offset:18432
	ds_load_tr16_b128 v[136:139] /*v[392:395]*/, v89 /*v601*/ offset:18464
	ds_load_tr16_b128 v[172:175] /*v[428:431]*/, v89 /*v601*/ offset:23040
	ds_load_tr16_b128 v[140:143] /*v[396:399]*/, v89 /*v601*/ offset:23072
	ds_load_tr16_b128 v[160:163] /*v[416:419]*/, v89 /*v601*/ offset:27648
	ds_load_tr16_b128 v[128:131] /*v[384:387]*/, v89 /*v601*/ offset:27680
	ds_load_tr16_b128 v[164:167] /*v[420:423]*/, v89 /*v601*/ offset:32256
	ds_load_tr16_b128 v[132:135] /*v[388:391]*/, v89 /*v601*/ offset:32288
	ds_load_tr16_b128 v[120:123] /*v[376:379]*/, v89 /*v601*/ offset:64
	ds_load_tr16_b128 v[88:91] /*v[344:347]*/, v89 /*v601*/ offset:96
	ds_load_tr16_b128 v[124:127] /*v[380:383]*/, v89 /*v601*/ offset:4672
	ds_load_tr16_b128 v[92:95] /*v[348:351]*/, v89 /*v601*/ offset:4704
	ds_load_tr16_b128 v[112:115] /*v[368:371]*/, v89 /*v601*/ offset:9280
	ds_load_tr16_b128 v[80:83] /*v[336:339]*/, v89 /*v601*/ offset:9312
	ds_load_tr16_b128 v[116:119] /*v[372:375]*/, v89 /*v601*/ offset:13888
	ds_load_tr16_b128 v[84:87] /*v[340:343]*/, v89 /*v601*/ offset:13920
	ds_load_tr16_b128 v[104:107] /*v[360:363]*/, v89 /*v601*/ offset:18496
	ds_load_tr16_b128 v[72:75] /*v[328:331]*/, v89 /*v601*/ offset:18528
	ds_load_tr16_b128 v[108:111] /*v[364:367]*/, v89 /*v601*/ offset:23104
	ds_load_tr16_b128 v[76:79] /*v[332:335]*/, v89 /*v601*/ offset:23136
	ds_load_tr16_b128 v[96:99] /*v[352:355]*/, v89 /*v601*/ offset:27712
	ds_load_tr16_b128 v[64:67] /*v[320:323]*/, v89 /*v601*/ offset:27744
	ds_load_tr16_b128 v[100:103] /*v[356:359]*/, v89 /*v601*/ offset:32320
	ds_load_tr16_b128 v[68:71] /*v[324:327]*/, v89 /*v601*/ offset:32352
	ds_load_tr16_b128 v[56:59] /*v[312:315]*/, v89 /*v601*/ offset:128
	ds_load_tr16_b128 v[24:27] /*v[280:283]*/, v89 /*v601*/ offset:160
	ds_load_tr16_b128 v[60:63] /*v[316:319]*/, v89 /*v601*/ offset:4736
	ds_load_tr16_b128 v[28:31] /*v[284:287]*/, v89 /*v601*/ offset:4768
	ds_load_tr16_b128 v[48:51] /*v[304:307]*/, v89 /*v601*/ offset:9344
	ds_load_tr16_b128 v[16:19] /*v[272:275]*/, v89 /*v601*/ offset:9376
	ds_load_tr16_b128 v[52:55] /*v[308:311]*/, v89 /*v601*/ offset:13952
	ds_load_tr16_b128 v[20:23] /*v[276:279]*/, v89 /*v601*/ offset:13984
	ds_load_tr16_b128 v[40:43] /*v[296:299]*/, v89 /*v601*/ offset:18560
	ds_load_tr16_b128 v[8:11] /*v[264:267]*/, v89 /*v601*/ offset:18592
	ds_load_tr16_b128 v[44:47] /*v[300:303]*/, v89 /*v601*/ offset:23168
	ds_load_tr16_b128 v[12:15] /*v[268:271]*/, v89 /*v601*/ offset:23200
	ds_load_tr16_b128 v[32:35] /*v[288:291]*/, v89 /*v601*/ offset:27776
	ds_load_tr16_b128 v[0:3] /*v[256:259]*/, v89 /*v601*/ offset:27808
	ds_load_tr16_b128 v[36:39] /*v[292:295]*/, v89 /*v601*/ offset:32384
	ds_load_tr16_b128 v[4:7] /*v[260:263]*/, v89 /*v601*/ offset:32416
	s_set_vgpr_msb 0x4202
	ds_load_tr16_b128 v[248:251], v89 /*v601*/ offset:192
	ds_load_tr16_b128 v[216:219], v89 /*v601*/ offset:224
	ds_load_tr16_b128 v[252:255], v89 /*v601*/ offset:4800
	ds_load_tr16_b128 v[220:223], v89 /*v601*/ offset:4832
	ds_load_tr16_b128 v[240:243], v89 /*v601*/ offset:9408
	ds_load_tr16_b128 v[208:211], v89 /*v601*/ offset:9440
	ds_load_tr16_b128 v[244:247], v89 /*v601*/ offset:14016
	ds_load_tr16_b128 v[212:215], v89 /*v601*/ offset:14048
	ds_load_tr16_b128 v[232:235], v89 /*v601*/ offset:18624
	ds_load_tr16_b128 v[200:203], v89 /*v601*/ offset:18656
	ds_load_tr16_b128 v[236:239], v89 /*v601*/ offset:23232
	ds_load_tr16_b128 v[204:207], v89 /*v601*/ offset:23264
	ds_load_tr16_b128 v[224:227], v89 /*v601*/ offset:27840
	ds_load_tr16_b128 v[192:195], v89 /*v601*/ offset:27872
	ds_load_tr16_b128 v[228:231], v89 /*v601*/ offset:32448
	ds_load_tr16_b128 v[196:199], v89 /*v601*/ offset:32480
	s_set_vgpr_msb 0x2aa
	v_lshl_or_b32 v67 /*v579*/, s58, 7, v79 /*v591*/
	v_cmp_gt_i32_e32 vcc_lo, v67 /*v579*/, v85 /*v597*/
	v_cmp_lt_i32_e64 s2, v67 /*v579*/, v82 /*v594*/
	v_dual_add_nc_u32 v114 /*v626*/, 16, v67 /*v579*/ :: v_dual_bitop2_b32 v70 /*v582*/, 1, v67 /*v579*/ bitop3:0x54
	v_dual_add_nc_u32 v115 /*v627*/, 17, v67 /*v579*/ :: v_dual_bitop2_b32 v87 /*v599*/, 2, v67 /*v579*/ bitop3:0x54
	v_cmp_ge_i32_e64 s3, v67 /*v579*/, v85 /*v597*/
	v_cmp_lt_i32_e64 s4, v70 /*v582*/, v82 /*v594*/
	s_or_b32 s2, s2, vcc_lo
	v_dual_add_nc_u32 v116 /*v628*/, 18, v67 /*v579*/ :: v_dual_bitop2_b32 v88 /*v600*/, 3, v67 /*v579*/ bitop3:0x54
	s_set_vgpr_msb 0xaa41
	v_cndmask_b32_e64 v250 /*v506*/, v250 /*v506*/, 0xff800000, s2
	s_set_vgpr_msb 0x418a
	v_cmp_gt_i32_e32 vcc_lo, v87 /*v599*/, v85 /*v597*/
	v_cmp_lt_i32_e64 s2, v87 /*v599*/, v82 /*v594*/
	v_dual_add_nc_u32 v117 /*v629*/, 19, v67 /*v579*/ :: v_dual_bitop2_b32 v90 /*v602*/, 4, v67 /*v579*/ bitop3:0x54
	s_or_b32 s3, s4, s3
	v_cmp_gt_i32_e64 s5, v88 /*v600*/, v85 /*v597*/
	s_set_vgpr_msb 0x8a41
	v_cndmask_b32_e64 v251 /*v507*/, v251 /*v507*/, 0xff800000, s3
	s_set_vgpr_msb 0x418a
	v_cmp_lt_i32_e64 s3, v88 /*v600*/, v82 /*v594*/
	s_or_b32 s2, s2, vcc_lo
	v_dual_add_nc_u32 v118 /*v630*/, 20, v67 /*v579*/ :: v_dual_bitop2_b32 v91 /*v603*/, 5, v67 /*v579*/ bitop3:0x54
	s_set_vgpr_msb 0x8a41
	v_cndmask_b32_e64 v252 /*v508*/, v252 /*v508*/, 0xff800000, s2
	s_set_vgpr_msb 0x418a
	v_cmp_gt_i32_e32 vcc_lo, v90 /*v602*/, v85 /*v597*/
	v_cmp_lt_i32_e64 s2, v90 /*v602*/, v82 /*v594*/
	v_dual_add_nc_u32 v119 /*v631*/, 21, v67 /*v579*/ :: v_dual_bitop2_b32 v110 /*v622*/, 6, v67 /*v579*/ bitop3:0x54
	s_or_b32 s3, s3, s5
	v_cmp_lt_i32_e64 s4, v91 /*v603*/, v82 /*v594*/
	s_set_vgpr_msb 0x8a41
	v_cndmask_b32_e64 v253 /*v509*/, v253 /*v509*/, 0xff800000, s3
	s_set_vgpr_msb 0x418a
	v_cmp_gt_i32_e64 s3, v91 /*v603*/, v85 /*v597*/
	s_or_b32 s2, s2, vcc_lo
	v_dual_add_nc_u32 v120 /*v632*/, 22, v67 /*v579*/ :: v_dual_bitop2_b32 v113 /*v625*/, 7, v67 /*v579*/ bitop3:0x54
	s_set_vgpr_msb 0x8a41
	v_cndmask_b32_e64 v254 /*v510*/, v254 /*v510*/, 0xff800000, s2
	s_set_vgpr_msb 0x410a
	v_cmp_gt_i32_e32 vcc_lo, v110 /*v622*/, v85 /*v597*/
	v_cmp_lt_i32_e64 s2, v110 /*v622*/, v82 /*v594*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s4, v113 /*v625*/, v82 /*v594*/
	s_set_vgpr_msb 0xa41
	v_cndmask_b32_e64 v255 /*v511*/, v255 /*v511*/, 0xff800000, s3
	s_set_vgpr_msb 0x418a
	v_cmp_gt_i32_e64 s3, v113 /*v625*/, v85 /*v597*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v114 /*v626*/, v85 /*v597*/
	v_cndmask_b32_e64 v0 /*v512*/, v0 /*v512*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v114 /*v626*/, v82 /*v594*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s4, v115 /*v627*/, v82 /*v594*/
	v_cndmask_b32_e64 v1 /*v513*/, v1 /*v513*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v115 /*v627*/, v85 /*v597*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v116 /*v628*/, v85 /*v597*/
	v_cndmask_b32_e64 v2 /*v514*/, v2 /*v514*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v116 /*v628*/, v82 /*v594*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s4, v117 /*v629*/, v82 /*v594*/
	v_cndmask_b32_e64 v3 /*v515*/, v3 /*v515*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v117 /*v629*/, v85 /*v597*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v118 /*v630*/, v85 /*v597*/
	v_cndmask_b32_e64 v4 /*v516*/, v4 /*v516*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v118 /*v630*/, v82 /*v594*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s4, v119 /*v631*/, v82 /*v594*/
	v_cndmask_b32_e64 v5 /*v517*/, v5 /*v517*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v119 /*v631*/, v85 /*v597*/
	s_or_b32 s2, s2, vcc_lo
	v_dual_add_nc_u32 v121 /*v633*/, 23, v67 /*v579*/ :: v_dual_bitop2_b32 v122 /*v634*/, 32, v67 /*v579*/ bitop3:0x54
	v_cndmask_b32_e64 v6 /*v518*/, v6 /*v518*/, 0xff800000, s2
	v_cmp_gt_i32_e32 vcc_lo, v120 /*v632*/, v85 /*v597*/
	v_cmp_lt_i32_e64 s2, v120 /*v632*/, v82 /*v594*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s4, v121 /*v633*/, v82 /*v594*/
	v_cndmask_b32_e64 v7 /*v519*/, v7 /*v519*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v121 /*v633*/, v85 /*v597*/
	s_or_b32 s2, s2, vcc_lo
	v_or_b32_e32 v123 /*v635*/, 33, v67 /*v579*/
	v_cndmask_b32_e64 v8 /*v520*/, v8 /*v520*/, 0xff800000, s2
	v_cmp_gt_i32_e32 vcc_lo, v122 /*v634*/, v85 /*v597*/
	v_cmp_lt_i32_e64 s2, v122 /*v634*/, v82 /*v594*/
	v_dual_add_nc_u32 v131 /*v643*/, 49, v67 /*v579*/ :: v_dual_bitop2_b32 v124 /*v636*/, 34, v67 /*v579*/ bitop3:0x54
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s4, v123 /*v635*/, v82 /*v594*/
	v_cndmask_b32_e64 v9 /*v521*/, v9 /*v521*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v123 /*v635*/, v85 /*v597*/
	s_or_b32 s2, s2, vcc_lo
	v_dual_add_nc_u32 v132 /*v644*/, 50, v67 /*v579*/ :: v_dual_bitop2_b32 v125 /*v637*/, 35, v67 /*v579*/ bitop3:0x54
	v_cndmask_b32_e64 v10 /*v522*/, v10 /*v522*/, 0xff800000, s2
	v_cmp_gt_i32_e32 vcc_lo, v124 /*v636*/, v85 /*v597*/
	v_cmp_lt_i32_e64 s2, v124 /*v636*/, v82 /*v594*/
	v_dual_add_nc_u32 v133 /*v645*/, 51, v67 /*v579*/ :: v_dual_bitop2_b32 v126 /*v638*/, 36, v67 /*v579*/ bitop3:0x54
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s4, v125 /*v637*/, v82 /*v594*/
	v_cndmask_b32_e64 v11 /*v523*/, v11 /*v523*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v125 /*v637*/, v85 /*v597*/
	s_or_b32 s2, s2, vcc_lo
	v_dual_add_nc_u32 v134 /*v646*/, 52, v67 /*v579*/ :: v_dual_bitop2_b32 v127 /*v639*/, 37, v67 /*v579*/ bitop3:0x54
	v_cndmask_b32_e64 v12 /*v524*/, v12 /*v524*/, 0xff800000, s2
	v_cmp_gt_i32_e32 vcc_lo, v126 /*v638*/, v85 /*v597*/
	v_cmp_lt_i32_e64 s2, v126 /*v638*/, v82 /*v594*/
	s_or_b32 s3, s4, s3
	v_dual_add_nc_u32 v135 /*v647*/, 53, v67 /*v579*/ :: v_dual_bitop2_b32 v128 /*v640*/, 38, v67 /*v579*/ bitop3:0x54
	v_cndmask_b32_e64 v13 /*v525*/, v13 /*v525*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v127 /*v639*/, v85 /*v597*/
	v_cmp_lt_i32_e64 s4, v127 /*v639*/, v82 /*v594*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v128 /*v640*/, v85 /*v597*/
	v_cndmask_b32_e64 v68 /*v580*/, v14 /*v526*/, 0xff800000, s2
	v_dual_add_nc_u32 v136 /*v648*/, 54, v67 /*v579*/ :: v_dual_bitop2_b32 v14 /*v526*/, 39, v67 /*v579*/ bitop3:0x54
	v_cmp_lt_i32_e64 s2, v128 /*v640*/, v82 /*v594*/
	s_or_b32 s3, s4, s3
	v_dual_add_nc_u32 v137 /*v649*/, 55, v67 /*v579*/ :: v_dual_bitop2_b32 v138 /*v650*/, 64, v67 /*v579*/ bitop3:0x54
	v_cndmask_b32_e64 v69 /*v581*/, v15 /*v527*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v14 /*v526*/, v85 /*v597*/
	v_add_nc_u32_e32 v15 /*v527*/, 48, v67 /*v579*/
	v_cmp_lt_i32_e64 s4, v14 /*v526*/, v82 /*v594*/
	s_or_b32 s2, s2, vcc_lo
	v_or_b32_e32 v139 /*v651*/, 0x41, v67 /*v579*/
	v_cndmask_b32_e64 v16 /*v528*/, v16 /*v528*/, 0xff800000, s2
	v_cmp_gt_i32_e32 vcc_lo, v15 /*v527*/, v85 /*v597*/
	v_cmp_lt_i32_e64 s2, v15 /*v527*/, v82 /*v594*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s4, v131 /*v643*/, v82 /*v594*/
	v_cndmask_b32_e64 v17 /*v529*/, v17 /*v529*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v131 /*v643*/, v85 /*v597*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v132 /*v644*/, v85 /*v597*/
	v_cndmask_b32_e64 v18 /*v530*/, v18 /*v530*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v132 /*v644*/, v82 /*v594*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s4, v133 /*v645*/, v82 /*v594*/
	v_cndmask_b32_e64 v19 /*v531*/, v19 /*v531*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v133 /*v645*/, v85 /*v597*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v134 /*v646*/, v85 /*v597*/
	v_cndmask_b32_e64 v20 /*v532*/, v20 /*v532*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v134 /*v646*/, v82 /*v594*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s4, v135 /*v647*/, v82 /*v594*/
	v_cndmask_b32_e64 v21 /*v533*/, v21 /*v533*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v135 /*v647*/, v85 /*v597*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v136 /*v648*/, v85 /*v597*/
	v_cndmask_b32_e64 v22 /*v534*/, v22 /*v534*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v136 /*v648*/, v82 /*v594*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s4, v137 /*v649*/, v82 /*v594*/
	v_cndmask_b32_e64 v23 /*v535*/, v23 /*v535*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v137 /*v649*/, v85 /*v597*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v138 /*v650*/, v85 /*v597*/
	v_cndmask_b32_e64 v24 /*v536*/, v24 /*v536*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v138 /*v650*/, v82 /*v594*/
	s_or_b32 s3, s4, s3
	v_or_b32_e32 v140 /*v652*/, 0x42, v67 /*v579*/
	v_cndmask_b32_e64 v25 /*v537*/, v25 /*v537*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v139 /*v651*/, v85 /*v597*/
	v_cmp_lt_i32_e64 s4, v139 /*v651*/, v82 /*v594*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v140 /*v652*/, v85 /*v597*/
	v_cndmask_b32_e64 v92 /*v604*/, v26 /*v538*/, 0xff800000, s2
	v_or_b32_e32 v26 /*v538*/, 0x43, v67 /*v579*/
	v_cmp_lt_i32_e64 s2, v140 /*v652*/, v82 /*v594*/
	s_or_b32 s3, s4, s3
	v_or_b32_e32 v143 /*v655*/, 0x45, v67 /*v579*/
	v_cndmask_b32_e64 v93 /*v605*/, v27 /*v539*/, 0xff800000, s3
	v_or_b32_e32 v27 /*v539*/, 0x44, v67 /*v579*/
	v_cmp_gt_i32_e64 s3, v26 /*v538*/, v85 /*v597*/
	v_cmp_lt_i32_e64 s4, v26 /*v538*/, v82 /*v594*/
	s_or_b32 s2, s2, vcc_lo
	v_or_b32_e32 v144 /*v656*/, 0x46, v67 /*v579*/
	v_cndmask_b32_e64 v28 /*v540*/, v28 /*v540*/, 0xff800000, s2
	v_cmp_gt_i32_e32 vcc_lo, v27 /*v539*/, v85 /*v597*/
	v_cmp_lt_i32_e64 s2, v27 /*v539*/, v82 /*v594*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s4, v143 /*v655*/, v82 /*v594*/
	v_cndmask_b32_e64 v29 /*v541*/, v29 /*v541*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v143 /*v655*/, v85 /*v597*/
	s_or_b32 s2, s2, vcc_lo
	v_or_b32_e32 v145 /*v657*/, 0x47, v67 /*v579*/
	v_cndmask_b32_e64 v30 /*v542*/, v30 /*v542*/, 0xff800000, s2
	v_cmp_gt_i32_e32 vcc_lo, v144 /*v656*/, v85 /*v597*/
	v_cmp_lt_i32_e64 s2, v144 /*v656*/, v82 /*v594*/
	s_or_b32 s3, s4, s3
	v_add_nc_u32_e32 v146 /*v658*/, 0x50, v67 /*v579*/
	v_cndmask_b32_e64 v31 /*v543*/, v31 /*v543*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v145 /*v657*/, v85 /*v597*/
	v_cmp_lt_i32_e64 s4, v145 /*v657*/, v82 /*v594*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v146 /*v658*/, v85 /*v597*/
	v_cndmask_b32_e64 v94 /*v606*/, v32 /*v544*/, 0xff800000, s2
	v_add_nc_u32_e32 v32 /*v544*/, 0x51, v67 /*v579*/
	v_cmp_lt_i32_e64 s2, v146 /*v658*/, v82 /*v594*/
	s_or_b32 s3, s4, s3
	v_add_nc_u32_e32 v149 /*v661*/, 0x53, v67 /*v579*/
	v_cndmask_b32_e64 v95 /*v607*/, v33 /*v545*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v32 /*v544*/, v85 /*v597*/
	v_add_nc_u32_e32 v33 /*v545*/, 0x52, v67 /*v579*/
	v_cmp_lt_i32_e64 s4, v32 /*v544*/, v82 /*v594*/
	s_or_b32 s2, s2, vcc_lo
	v_add_nc_u32_e32 v150 /*v662*/, 0x54, v67 /*v579*/
	v_cndmask_b32_e64 v34 /*v546*/, v34 /*v546*/, 0xff800000, s2
	v_cmp_gt_i32_e32 vcc_lo, v33 /*v545*/, v85 /*v597*/
	v_cmp_lt_i32_e64 s2, v33 /*v545*/, v82 /*v594*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s4, v149 /*v661*/, v82 /*v594*/
	v_cndmask_b32_e64 v35 /*v547*/, v35 /*v547*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v149 /*v661*/, v85 /*v597*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v150 /*v662*/, v85 /*v597*/
	v_cndmask_b32_e64 v96 /*v608*/, v36 /*v548*/, 0xff800000, s2
	v_add_nc_u32_e32 v36 /*v548*/, 0x55, v67 /*v579*/
	v_cmp_lt_i32_e64 s2, v150 /*v662*/, v82 /*v594*/
	s_or_b32 s3, s4, s3
	v_add_nc_u32_e32 v152 /*v664*/, 0x57, v67 /*v579*/
	v_cndmask_b32_e64 v97 /*v609*/, v37 /*v549*/, 0xff800000, s3
	v_add_nc_u32_e32 v37 /*v549*/, 0x56, v67 /*v579*/
	v_cmp_gt_i32_e64 s3, v36 /*v548*/, v85 /*v597*/
	v_cmp_lt_i32_e64 s4, v36 /*v548*/, v82 /*v594*/
	s_or_b32 s2, s2, vcc_lo
	v_or_b32_e32 v153 /*v665*/, 0x60, v67 /*v579*/
	v_cndmask_b32_e64 v38 /*v550*/, v38 /*v550*/, 0xff800000, s2
	v_cmp_gt_i32_e32 vcc_lo, v37 /*v549*/, v85 /*v597*/
	v_cmp_lt_i32_e64 s2, v37 /*v549*/, v82 /*v594*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s4, v152 /*v664*/, v82 /*v594*/
	v_cndmask_b32_e64 v39 /*v551*/, v39 /*v551*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v152 /*v664*/, v85 /*v597*/
	s_or_b32 s2, s2, vcc_lo
	v_or_b32_e32 v155 /*v667*/, 0x61, v67 /*v579*/
	v_cndmask_b32_e64 v40 /*v552*/, v40 /*v552*/, 0xff800000, s2
	v_cmp_gt_i32_e32 vcc_lo, v153 /*v665*/, v85 /*v597*/
	v_cmp_lt_i32_e64 s2, v153 /*v665*/, v82 /*v594*/
	s_or_b32 s3, s4, s3
	v_or_b32_e32 v156 /*v668*/, 0x62, v67 /*v579*/
	v_cndmask_b32_e64 v41 /*v553*/, v41 /*v553*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v155 /*v667*/, v85 /*v597*/
	v_cmp_lt_i32_e64 s4, v155 /*v667*/, v82 /*v594*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v156 /*v668*/, v85 /*v597*/
	v_cndmask_b32_e64 v98 /*v610*/, v42 /*v554*/, 0xff800000, s2
	v_or_b32_e32 v42 /*v554*/, 0x63, v67 /*v579*/
	v_cmp_lt_i32_e64 s2, v156 /*v668*/, v82 /*v594*/
	s_or_b32 s3, s4, s3
	v_or_b32_e32 v161 /*v673*/, 0x67, v67 /*v579*/
	v_cndmask_b32_e64 v99 /*v611*/, v43 /*v555*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v42 /*v554*/, v85 /*v597*/
	v_or_b32_e32 v43 /*v555*/, 0x64, v67 /*v579*/
	v_cmp_lt_i32_e64 s4, v42 /*v554*/, v82 /*v594*/
	s_or_b32 s2, s2, vcc_lo
	v_add_nc_u32_e32 v162 /*v674*/, 0x70, v67 /*v579*/
	v_cndmask_b32_e64 v100 /*v612*/, v44 /*v556*/, 0xff800000, s2
	v_or_b32_e32 v44 /*v556*/, 0x65, v67 /*v579*/
	v_cmp_gt_i32_e32 vcc_lo, v43 /*v555*/, v85 /*v597*/
	v_cmp_lt_i32_e64 s2, v43 /*v555*/, v82 /*v594*/
	s_or_b32 s3, s4, s3
	v_add_nc_u32_e32 v163 /*v675*/, 0x71, v67 /*v579*/
	v_cndmask_b32_e64 v101 /*v613*/, v45 /*v557*/, 0xff800000, s3
	v_or_b32_e32 v45 /*v557*/, 0x66, v67 /*v579*/
	v_cmp_gt_i32_e64 s3, v44 /*v556*/, v85 /*v597*/
	v_cmp_lt_i32_e64 s4, v44 /*v556*/, v82 /*v594*/
	s_or_b32 s2, s2, vcc_lo
	v_add_nc_u32_e32 v164 /*v676*/, 0x72, v67 /*v579*/
	v_cndmask_b32_e64 v46 /*v558*/, v46 /*v558*/, 0xff800000, s2
	v_cmp_gt_i32_e32 vcc_lo, v45 /*v557*/, v85 /*v597*/
	v_cmp_lt_i32_e64 s2, v45 /*v557*/, v82 /*v594*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s4, v161 /*v673*/, v82 /*v594*/
	v_cndmask_b32_e64 v47 /*v559*/, v47 /*v559*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v161 /*v673*/, v85 /*v597*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v162 /*v674*/, v85 /*v597*/
	v_cndmask_b32_e64 v48 /*v560*/, v48 /*v560*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v162 /*v674*/, v82 /*v594*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s4, v163 /*v675*/, v82 /*v594*/
	v_cndmask_b32_e64 v49 /*v561*/, v49 /*v561*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v163 /*v675*/, v85 /*v597*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v164 /*v676*/, v85 /*v597*/
	v_cndmask_b32_e64 v102 /*v614*/, v50 /*v562*/, 0xff800000, s2
	v_add_nc_u32_e32 v50 /*v562*/, 0x73, v67 /*v579*/
	v_cmp_lt_i32_e64 s2, v164 /*v676*/, v82 /*v594*/
	s_or_b32 s3, s4, s3
	v_add_nc_u32_e32 v167 /*v679*/, 0x77, v67 /*v579*/
	v_cndmask_b32_e64 v103 /*v615*/, v51 /*v563*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v50 /*v562*/, v85 /*v597*/
	v_add_nc_u32_e32 v51 /*v563*/, 0x74, v67 /*v579*/
	v_cmp_lt_i32_e64 s4, v50 /*v562*/, v82 /*v594*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e64 s5, v67 /*v579*/, v86 /*v598*/
	v_cndmask_b32_e64 v104 /*v616*/, v52 /*v564*/, 0xff800000, s2
	v_add_nc_u32_e32 v52 /*v564*/, 0x75, v67 /*v579*/
	v_cmp_gt_i32_e32 vcc_lo, v51 /*v563*/, v85 /*v597*/
	v_cmp_lt_i32_e64 s2, v51 /*v563*/, v82 /*v594*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s6, v67 /*v579*/, v83 /*v595*/
	v_cndmask_b32_e64 v105 /*v617*/, v53 /*v565*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v52 /*v564*/, v85 /*v597*/
	v_cmp_lt_i32_e64 s4, v52 /*v564*/, v82 /*v594*/
	v_add_nc_u32_e32 v53 /*v565*/, 0x76, v67 /*v579*/
	s_or_b32 s2, s2, vcc_lo
	v_cndmask_b32_e64 v54 /*v566*/, v54 /*v566*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	v_cmp_gt_i32_e32 vcc_lo, v53 /*v565*/, v85 /*v597*/
	v_cndmask_b32_e64 v55 /*v567*/, v55 /*v567*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v53 /*v565*/, v82 /*v594*/
	v_cmp_gt_i32_e64 s3, v167 /*v679*/, v85 /*v597*/
	v_cmp_lt_i32_e64 s4, v167 /*v679*/, v82 /*v594*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_ge_i32_e32 vcc_lo, v67 /*v579*/, v86 /*v598*/
	v_cndmask_b32_e64 v56 /*v568*/, v56 /*v568*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	v_cmp_gt_i32_e64 s3, v87 /*v599*/, v86 /*v598*/
	v_cndmask_b32_e64 v57 /*v569*/, v57 /*v569*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	v_cmp_lt_i32_e64 s4, v87 /*v599*/, v83 /*v595*/
	s_set_vgpr_msb 0x8a81
	v_cndmask_b32_e64 v106 /*v618*/, v192 /*v448*/, 0xff800000, s2
	s_set_vgpr_msb 0x810a
	v_cmp_lt_i32_e64 s2, v70 /*v582*/, v83 /*v595*/
	v_cmp_gt_i32_e64 s5, v88 /*v600*/, v86 /*v598*/
	v_cmp_lt_i32_e64 s6, v88 /*v600*/, v83 /*v595*/
	s_set_vgpr_msb 0xa55
	v_max3_num_f32 v192 /*v448*/, v250 /*v506*/, v251 /*v507*/, v252 /*v508*/
	s_or_b32 s2, s2, vcc_lo
	s_set_vgpr_msb 0x550a
	v_cmp_gt_i32_e32 vcc_lo, v90 /*v602*/, v86 /*v598*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v107 /*v619*/, v193 /*v449*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	s_set_vgpr_msb 0x810a
	v_cmp_gt_i32_e64 s3, v91 /*v603*/, v86 /*v598*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v108 /*v620*/, v194 /*v450*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	s_set_vgpr_msb 0x810a
	v_cmp_lt_i32_e64 s4, v91 /*v603*/, v83 /*v595*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v109 /*v621*/, v195 /*v451*/, 0xff800000, s2
	s_set_vgpr_msb 0x816a
	v_cmp_lt_i32_e64 s2, v90 /*v602*/, v83 /*v595*/
	v_cmp_gt_i32_e64 s5, v110 /*v622*/, v86 /*v598*/
	v_cmp_lt_i32_e64 s6, v110 /*v622*/, v83 /*v595*/
	v_max3_num_f32 v193 /*v449*/, v106 /*v618*/, v107 /*v619*/, v108 /*v620*/
	s_set_vgpr_msb 0x6a55
	v_max3_num_f32 v194 /*v450*/, v253 /*v509*/, v254 /*v510*/, v255 /*v511*/
	s_or_b32 s2, s2, vcc_lo
	s_set_vgpr_msb 0x550a
	v_cmp_gt_i32_e32 vcc_lo, v113 /*v625*/, v86 /*v598*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v110 /*v622*/, v196 /*v452*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	s_set_vgpr_msb 0x810a
	v_cmp_gt_i32_e64 s3, v114 /*v626*/, v86 /*v598*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v111 /*v623*/, v197 /*v453*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	s_set_vgpr_msb 0x810a
	v_cmp_lt_i32_e64 s4, v114 /*v626*/, v83 /*v595*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v112 /*v624*/, v198 /*v454*/, 0xff800000, s2
	s_set_vgpr_msb 0x816a
	v_cmp_lt_i32_e64 s2, v113 /*v625*/, v83 /*v595*/
	v_cmp_gt_i32_e64 s5, v115 /*v627*/, v86 /*v598*/
	v_cmp_lt_i32_e64 s6, v115 /*v627*/, v83 /*v595*/
	v_max3_num_f32 v195 /*v451*/, v109 /*v621*/, v110 /*v622*/, v111 /*v623*/
	v_max3_num_f32 v196 /*v452*/, v0 /*v512*/, v1 /*v513*/, v2 /*v514*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v116 /*v628*/, v86 /*v598*/
	s_set_vgpr_msb 0x6a81
	v_cndmask_b32_e64 v113 /*v625*/, v199 /*v455*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	s_set_vgpr_msb 0x810a
	v_cmp_gt_i32_e64 s3, v117 /*v629*/, v86 /*v598*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v114 /*v626*/, v200 /*v456*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	s_set_vgpr_msb 0x810a
	v_cmp_lt_i32_e64 s4, v117 /*v629*/, v83 /*v595*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v115 /*v627*/, v201 /*v457*/, 0xff800000, s2
	s_set_vgpr_msb 0x816a
	v_cmp_lt_i32_e64 s2, v116 /*v628*/, v83 /*v595*/
	v_cmp_gt_i32_e64 s5, v118 /*v630*/, v86 /*v598*/
	v_cmp_lt_i32_e64 s6, v118 /*v630*/, v83 /*v595*/
	v_max3_num_f32 v197 /*v453*/, v112 /*v624*/, v113 /*v625*/, v114 /*v626*/
	v_max3_num_f32 v198 /*v454*/, v3 /*v515*/, v4 /*v516*/, v5 /*v517*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v119 /*v631*/, v86 /*v598*/
	s_set_vgpr_msb 0x6a81
	v_cndmask_b32_e64 v116 /*v628*/, v202 /*v458*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	s_set_vgpr_msb 0x810a
	v_cmp_gt_i32_e64 s3, v120 /*v632*/, v86 /*v598*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v117 /*v629*/, v203 /*v459*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	s_set_vgpr_msb 0x810a
	v_cmp_lt_i32_e64 s4, v120 /*v632*/, v83 /*v595*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v118 /*v630*/, v204 /*v460*/, 0xff800000, s2
	s_set_vgpr_msb 0x816a
	v_cmp_lt_i32_e64 s2, v119 /*v631*/, v83 /*v595*/
	v_cmp_gt_i32_e64 s5, v121 /*v633*/, v86 /*v598*/
	v_cmp_lt_i32_e64 s6, v121 /*v633*/, v83 /*v595*/
	v_max3_num_f32 v200 /*v456*/, v6 /*v518*/, v7 /*v519*/, v8 /*v520*/
	v_max3_num_f32 v202 /*v458*/, v9 /*v521*/, v10 /*v522*/, v11 /*v523*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v122 /*v634*/, v86 /*v598*/
	s_set_vgpr_msb 0x6a81
	v_cndmask_b32_e64 v119 /*v631*/, v205 /*v461*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	s_set_vgpr_msb 0x810a
	v_cmp_gt_i32_e64 s3, v123 /*v635*/, v86 /*v598*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v120 /*v632*/, v206 /*v462*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	s_set_vgpr_msb 0x810a
	v_cmp_lt_i32_e64 s4, v123 /*v635*/, v83 /*v595*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v121 /*v633*/, v207 /*v463*/, 0xff800000, s2
	s_set_vgpr_msb 0x816a
	v_cmp_lt_i32_e64 s2, v122 /*v634*/, v83 /*v595*/
	v_cmp_gt_i32_e64 s5, v124 /*v636*/, v86 /*v598*/
	v_cmp_lt_i32_e64 s6, v124 /*v636*/, v83 /*v595*/
	v_max3_num_f32 v204 /*v460*/, v12 /*v524*/, v13 /*v525*/, v68 /*v580*/
	v_max3_num_f32 v206 /*v462*/, v69 /*v581*/, v16 /*v528*/, v17 /*v529*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v125 /*v637*/, v86 /*v598*/
	s_set_vgpr_msb 0x6a81
	v_cndmask_b32_e64 v122 /*v634*/, v208 /*v464*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	s_set_vgpr_msb 0x810a
	v_cmp_gt_i32_e64 s3, v126 /*v638*/, v86 /*v598*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v123 /*v635*/, v209 /*v465*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	s_set_vgpr_msb 0x810a
	v_cmp_lt_i32_e64 s4, v126 /*v638*/, v83 /*v595*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v124 /*v636*/, v210 /*v466*/, 0xff800000, s2
	s_set_vgpr_msb 0x816a
	v_cmp_lt_i32_e64 s2, v125 /*v637*/, v83 /*v595*/
	v_cmp_gt_i32_e64 s5, v127 /*v639*/, v86 /*v598*/
	v_cmp_lt_i32_e64 s6, v127 /*v639*/, v83 /*v595*/
	v_max3_num_f32 v208 /*v464*/, v18 /*v530*/, v19 /*v531*/, v20 /*v532*/
	v_max3_num_f32 v210 /*v466*/, v21 /*v533*/, v22 /*v534*/, v23 /*v535*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v128 /*v640*/, v86 /*v598*/
	s_set_vgpr_msb 0x6a81
	v_cndmask_b32_e64 v125 /*v637*/, v211 /*v467*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	s_set_vgpr_msb 0x810a
	v_cmp_gt_i32_e64 s3, v14 /*v526*/, v86 /*v598*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v126 /*v638*/, v212 /*v468*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	s_set_vgpr_msb 0x810a
	v_cmp_lt_i32_e64 s4, v14 /*v526*/, v83 /*v595*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v127 /*v639*/, v213 /*v469*/, 0xff800000, s2
	s_set_vgpr_msb 0x816a
	v_cmp_lt_i32_e64 s2, v128 /*v640*/, v83 /*v595*/
	v_cmp_gt_i32_e64 s5, v15 /*v527*/, v86 /*v598*/
	v_cmp_lt_i32_e64 s6, v15 /*v527*/, v83 /*v595*/
	v_max3_num_f32 v212 /*v468*/, v24 /*v536*/, v25 /*v537*/, v92 /*v604*/
	v_max3_num_f32 v199 /*v455*/, v115 /*v627*/, v116 /*v628*/, v117 /*v629*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v131 /*v643*/, v86 /*v598*/
	s_set_vgpr_msb 0x6a81
	v_cndmask_b32_e64 v128 /*v640*/, v214 /*v470*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	s_set_vgpr_msb 0x810a
	v_cmp_gt_i32_e64 s3, v132 /*v644*/, v86 /*v598*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v129 /*v641*/, v215 /*v471*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	s_set_vgpr_msb 0x810a
	v_cmp_lt_i32_e64 s4, v132 /*v644*/, v83 /*v595*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v130 /*v642*/, v216 /*v472*/, 0xff800000, s2
	s_set_vgpr_msb 0x816a
	v_cmp_lt_i32_e64 s2, v131 /*v643*/, v83 /*v595*/
	v_cmp_gt_i32_e64 s5, v133 /*v645*/, v86 /*v598*/
	v_cmp_lt_i32_e64 s6, v133 /*v645*/, v83 /*v595*/
	v_max3_num_f32 v214 /*v470*/, v93 /*v605*/, v28 /*v540*/, v29 /*v541*/
	v_max3_num_f32 v216 /*v472*/, v30 /*v542*/, v31 /*v543*/, v94 /*v606*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v134 /*v646*/, v86 /*v598*/
	s_set_vgpr_msb 0x6a81
	v_cndmask_b32_e64 v131 /*v643*/, v217 /*v473*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	s_set_vgpr_msb 0x810a
	v_cmp_gt_i32_e64 s3, v135 /*v647*/, v86 /*v598*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v132 /*v644*/, v218 /*v474*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	s_set_vgpr_msb 0x810a
	v_cmp_lt_i32_e64 s4, v135 /*v647*/, v83 /*v595*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v133 /*v645*/, v219 /*v475*/, 0xff800000, s2
	s_set_vgpr_msb 0x816a
	v_cmp_lt_i32_e64 s2, v134 /*v646*/, v83 /*v595*/
	v_cmp_gt_i32_e64 s5, v136 /*v648*/, v86 /*v598*/
	v_cmp_lt_i32_e64 s6, v136 /*v648*/, v83 /*v595*/
	v_max3_num_f32 v218 /*v474*/, v95 /*v607*/, v34 /*v546*/, v35 /*v547*/
	v_max3_num_f32 v201 /*v457*/, v118 /*v630*/, v119 /*v631*/, v120 /*v632*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v137 /*v649*/, v86 /*v598*/
	s_set_vgpr_msb 0x6a81
	v_cndmask_b32_e64 v134 /*v646*/, v220 /*v476*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	s_set_vgpr_msb 0x810a
	v_cmp_gt_i32_e64 s3, v138 /*v650*/, v86 /*v598*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v135 /*v647*/, v221 /*v477*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	s_set_vgpr_msb 0x810a
	v_cmp_lt_i32_e64 s4, v138 /*v650*/, v83 /*v595*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v136 /*v648*/, v222 /*v478*/, 0xff800000, s2
	s_set_vgpr_msb 0x816a
	v_cmp_lt_i32_e64 s2, v137 /*v649*/, v83 /*v595*/
	v_cmp_gt_i32_e64 s5, v139 /*v651*/, v86 /*v598*/
	v_cmp_lt_i32_e64 s6, v139 /*v651*/, v83 /*v595*/
	v_max3_num_f32 v220 /*v476*/, v96 /*v608*/, v97 /*v609*/, v38 /*v550*/
	v_max3_num_f32 v222 /*v478*/, v39 /*v551*/, v40 /*v552*/, v41 /*v553*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v140 /*v652*/, v86 /*v598*/
	s_set_vgpr_msb 0x6a81
	v_cndmask_b32_e64 v137 /*v649*/, v223 /*v479*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	s_set_vgpr_msb 0x810a
	v_cmp_gt_i32_e64 s3, v26 /*v538*/, v86 /*v598*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v138 /*v650*/, v224 /*v480*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	s_set_vgpr_msb 0x810a
	v_cmp_lt_i32_e64 s4, v26 /*v538*/, v83 /*v595*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v139 /*v651*/, v225 /*v481*/, 0xff800000, s2
	s_set_vgpr_msb 0x816a
	v_cmp_lt_i32_e64 s2, v140 /*v652*/, v83 /*v595*/
	v_cmp_gt_i32_e64 s5, v27 /*v539*/, v86 /*v598*/
	v_cmp_lt_i32_e64 s6, v27 /*v539*/, v83 /*v595*/
	v_max3_num_f32 v224 /*v480*/, v98 /*v610*/, v99 /*v611*/, v100 /*v612*/
	v_max3_num_f32 v203 /*v459*/, v121 /*v633*/, v122 /*v634*/, v123 /*v635*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v143 /*v655*/, v86 /*v598*/
	s_set_vgpr_msb 0x6a81
	v_cndmask_b32_e64 v140 /*v652*/, v226 /*v482*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	s_set_vgpr_msb 0x810a
	v_cmp_gt_i32_e64 s3, v144 /*v656*/, v86 /*v598*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v141 /*v653*/, v227 /*v483*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	s_set_vgpr_msb 0x810a
	v_cmp_lt_i32_e64 s4, v144 /*v656*/, v83 /*v595*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v142 /*v654*/, v228 /*v484*/, 0xff800000, s2
	s_set_vgpr_msb 0x816a
	v_cmp_lt_i32_e64 s2, v143 /*v655*/, v83 /*v595*/
	v_cmp_gt_i32_e64 s5, v145 /*v657*/, v86 /*v598*/
	v_cmp_lt_i32_e64 s6, v145 /*v657*/, v83 /*v595*/
	v_max3_num_f32 v226 /*v482*/, v101 /*v613*/, v46 /*v558*/, v47 /*v559*/
	v_max_num_f32_e32 v228 /*v484*/, v48 /*v560*/, v49 /*v561*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v146 /*v658*/, v86 /*v598*/
	s_set_vgpr_msb 0x6a81
	v_cndmask_b32_e64 v143 /*v655*/, v229 /*v485*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	s_set_vgpr_msb 0x810a
	v_cmp_gt_i32_e64 s3, v32 /*v544*/, v86 /*v598*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v144 /*v656*/, v230 /*v486*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	s_set_vgpr_msb 0x810a
	v_cmp_lt_i32_e64 s4, v32 /*v544*/, v83 /*v595*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v145 /*v657*/, v231 /*v487*/, 0xff800000, s2
	s_set_vgpr_msb 0x816a
	v_cmp_lt_i32_e64 s2, v146 /*v658*/, v83 /*v595*/
	v_cmp_gt_i32_e64 s5, v33 /*v545*/, v86 /*v598*/
	v_cmp_lt_i32_e64 s6, v33 /*v545*/, v83 /*v595*/
	v_max3_num_f32 v230 /*v486*/, v103 /*v615*/, v104 /*v616*/, v105 /*v617*/
	v_max3_num_f32 v205 /*v461*/, v124 /*v636*/, v125 /*v637*/, v126 /*v638*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v149 /*v661*/, v86 /*v598*/
	s_set_vgpr_msb 0x6a81
	v_cndmask_b32_e64 v146 /*v658*/, v232 /*v488*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	s_set_vgpr_msb 0x810a
	v_cmp_gt_i32_e64 s3, v150 /*v662*/, v86 /*v598*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v147 /*v659*/, v233 /*v489*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	s_set_vgpr_msb 0x810a
	v_cmp_lt_i32_e64 s4, v150 /*v662*/, v83 /*v595*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v148 /*v660*/, v234 /*v490*/, 0xff800000, s2
	s_set_vgpr_msb 0x816a
	v_cmp_lt_i32_e64 s2, v149 /*v661*/, v83 /*v595*/
	v_cmp_gt_i32_e64 s5, v36 /*v548*/, v86 /*v598*/
	v_cmp_lt_i32_e64 s6, v36 /*v548*/, v83 /*v595*/
	v_max3_num_f32 v207 /*v463*/, v127 /*v639*/, v128 /*v640*/, v129 /*v641*/
	v_max3_num_f32 v209 /*v465*/, v130 /*v642*/, v131 /*v643*/, v132 /*v644*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v37 /*v549*/, v86 /*v598*/
	s_set_vgpr_msb 0x6a81
	v_cndmask_b32_e64 v149 /*v661*/, v235 /*v491*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	s_set_vgpr_msb 0x810a
	v_cmp_gt_i32_e64 s3, v152 /*v664*/, v86 /*v598*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v150 /*v662*/, v236 /*v492*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	s_set_vgpr_msb 0x810a
	v_cmp_lt_i32_e64 s4, v152 /*v664*/, v83 /*v595*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v151 /*v663*/, v237 /*v493*/, 0xff800000, s2
	s_set_vgpr_msb 0x816a
	v_cmp_lt_i32_e64 s2, v37 /*v549*/, v83 /*v595*/
	v_cmp_gt_i32_e64 s5, v153 /*v665*/, v86 /*v598*/
	v_cmp_lt_i32_e64 s6, v153 /*v665*/, v83 /*v595*/
	v_max3_num_f32 v211 /*v467*/, v133 /*v645*/, v134 /*v646*/, v135 /*v647*/
	v_max3_num_f32 v213 /*v469*/, v136 /*v648*/, v137 /*v649*/, v138 /*v650*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v155 /*v667*/, v86 /*v598*/
	s_set_vgpr_msb 0x6a81
	v_cndmask_b32_e64 v152 /*v664*/, v238 /*v494*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	s_set_vgpr_msb 0x810a
	v_cmp_gt_i32_e64 s3, v156 /*v668*/, v86 /*v598*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v153 /*v665*/, v239 /*v495*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	s_set_vgpr_msb 0x810a
	v_cmp_lt_i32_e64 s4, v156 /*v668*/, v83 /*v595*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v154 /*v666*/, v240 /*v496*/, 0xff800000, s2
	s_set_vgpr_msb 0x816a
	v_cmp_lt_i32_e64 s2, v155 /*v667*/, v83 /*v595*/
	v_cmp_gt_i32_e64 s5, v42 /*v554*/, v86 /*v598*/
	v_cmp_lt_i32_e64 s6, v42 /*v554*/, v83 /*v595*/
	v_max3_num_f32 v215 /*v471*/, v139 /*v651*/, v140 /*v652*/, v141 /*v653*/
	v_max3_num_f32 v217 /*v473*/, v142 /*v654*/, v143 /*v655*/, v144 /*v656*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v43 /*v555*/, v86 /*v598*/
	s_set_vgpr_msb 0x6a81
	v_cndmask_b32_e64 v155 /*v667*/, v241 /*v497*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	s_set_vgpr_msb 0x810a
	v_cmp_gt_i32_e64 s3, v44 /*v556*/, v86 /*v598*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v156 /*v668*/, v242 /*v498*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	s_set_vgpr_msb 0x810a
	v_cmp_lt_i32_e64 s4, v44 /*v556*/, v83 /*v595*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v157 /*v669*/, v243 /*v499*/, 0xff800000, s2
	s_set_vgpr_msb 0x816a
	v_cmp_lt_i32_e64 s2, v43 /*v555*/, v83 /*v595*/
	v_cmp_gt_i32_e64 s5, v45 /*v557*/, v86 /*v598*/
	v_cmp_lt_i32_e64 s6, v45 /*v557*/, v83 /*v595*/
	v_max3_num_f32 v219 /*v475*/, v145 /*v657*/, v146 /*v658*/, v147 /*v659*/
	v_max3_num_f32 v221 /*v477*/, v148 /*v660*/, v149 /*v661*/, v150 /*v662*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v161 /*v673*/, v86 /*v598*/
	s_set_vgpr_msb 0x6a81
	v_cndmask_b32_e64 v158 /*v670*/, v244 /*v500*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	s_set_vgpr_msb 0x810a
	v_cmp_gt_i32_e64 s3, v162 /*v674*/, v86 /*v598*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v159 /*v671*/, v245 /*v501*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	s_set_vgpr_msb 0x810a
	v_cmp_lt_i32_e64 s4, v162 /*v674*/, v83 /*v595*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v160 /*v672*/, v246 /*v502*/, 0xff800000, s2
	s_set_vgpr_msb 0x816a
	v_cmp_lt_i32_e64 s2, v161 /*v673*/, v83 /*v595*/
	v_cmp_gt_i32_e64 s5, v163 /*v675*/, v86 /*v598*/
	v_cmp_lt_i32_e64 s6, v163 /*v675*/, v83 /*v595*/
	v_max3_num_f32 v223 /*v479*/, v151 /*v663*/, v152 /*v664*/, v153 /*v665*/
	v_max3_num_f32 v225 /*v481*/, v154 /*v666*/, v155 /*v667*/, v156 /*v668*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v164 /*v676*/, v86 /*v598*/
	s_set_vgpr_msb 0x6a81
	v_cndmask_b32_e64 v161 /*v673*/, v247 /*v503*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	s_set_vgpr_msb 0x818a
	v_cmp_gt_i32_e64 s3, v50 /*v562*/, v86 /*v598*/
	v_cndmask_b32_e64 v162 /*v674*/, v58 /*v570*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	v_cmp_lt_i32_e64 s4, v50 /*v562*/, v83 /*v595*/
	v_cndmask_b32_e64 v163 /*v675*/, v59 /*v571*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v164 /*v676*/, v83 /*v595*/
	v_cmp_gt_i32_e64 s5, v51 /*v563*/, v86 /*v598*/
	v_cmp_lt_i32_e64 s6, v51 /*v563*/, v83 /*v595*/
	s_set_vgpr_msb 0x8a6a
	v_max3_num_f32 v227 /*v483*/, v157 /*v669*/, v158 /*v670*/, v159 /*v671*/
	v_max_num_f32_e32 v229 /*v485*/, v160 /*v672*/, v161 /*v673*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v52 /*v564*/, v86 /*v598*/
	s_set_vgpr_msb 0x6a8a
	v_cndmask_b32_e64 v164 /*v676*/, v60 /*v572*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	v_cmp_gt_i32_e64 s3, v53 /*v565*/, v86 /*v598*/
	v_cndmask_b32_e64 v165 /*v677*/, v61 /*v573*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	v_cmp_lt_i32_e64 s4, v53 /*v565*/, v83 /*v595*/
	v_cndmask_b32_e64 v166 /*v678*/, v62 /*v574*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v52 /*v564*/, v83 /*v595*/
	v_cmp_gt_i32_e64 s5, v167 /*v679*/, v86 /*v598*/
	v_cmp_lt_i32_e64 s6, v167 /*v679*/, v83 /*v595*/
	s_set_vgpr_msb 0x8a6a
	v_max3_num_f32 v231 /*v487*/, v163 /*v675*/, v164 /*v676*/, v165 /*v677*/
	v_max3_num_f32 v232 /*v488*/, v54 /*v566*/, v55 /*v567*/, v56 /*v568*/
	s_or_b32 s2, s2, vcc_lo
	s_set_vgpr_msb 0x6a55
	v_max3_num_f32 v192 /*v448*/, v192 /*v448*/, v194 /*v450*/, v196 /*v452*/
	s_set_vgpr_msb 0x5582
	v_cndmask_b32_e64 v167 /*v679*/, v63 /*v575*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	s_set_vgpr_msb 0x8255
	v_max3_num_f32 v193 /*v449*/, v193 /*v449*/, v195 /*v451*/, v197 /*v453*/
	s_set_vgpr_msb 0x5582
	v_cndmask_b32_e64 v168 /*v680*/, v64 /*v576*/, 0xff800000, s2
	s_set_vgpr_msb 0x8255
	v_max3_num_f32 v194 /*v450*/, v198 /*v454*/, v200 /*v456*/, v202 /*v458*/
	v_max3_num_f32 v195 /*v451*/, v204 /*v460*/, v206 /*v462*/, v208 /*v464*/
	v_max3_num_f32 v196 /*v452*/, v210 /*v466*/, v212 /*v468*/, v214 /*v470*/
	v_max3_num_f32 v197 /*v453*/, v216 /*v472*/, v218 /*v474*/, v220 /*v476*/
	v_max3_num_f32 v198 /*v454*/, v222 /*v478*/, v224 /*v480*/, v226 /*v482*/
	s_set_vgpr_msb 0x5559
	v_max3_num_f32 v200 /*v456*/, v228 /*v484*/, v102 /*v614*/, v230 /*v486*/
	s_or_b32 s2, s6, s5
	s_set_vgpr_msb 0x596a
	v_max3_num_f32 v233 /*v489*/, v166 /*v678*/, v167 /*v679*/, v168 /*v680*/
	s_set_vgpr_msb 0x6a82
	v_cndmask_b32_e64 v169 /*v681*/, v65 /*v577*/, 0xff800000, s2
	s_set_vgpr_msb 0x8255
	v_max3_num_f32 v199 /*v455*/, v199 /*v455*/, v201 /*v457*/, v203 /*v459*/
	v_max3_num_f32 v201 /*v457*/, v205 /*v461*/, v207 /*v463*/, v209 /*v465*/
	v_max3_num_f32 v192 /*v448*/, v192 /*v448*/, v194 /*v450*/, v195 /*v451*/
	v_max3_num_f32 v194 /*v450*/, v196 /*v452*/, v197 /*v453*/, v198 /*v454*/
	s_set_vgpr_msb 0x5565
	v_max3_num_f32 v195 /*v451*/, v200 /*v456*/, v232 /*v488*/, v57 /*v569*/
	s_set_vgpr_msb 0x6555
	v_max3_num_f32 v196 /*v452*/, v211 /*v467*/, v213 /*v469*/, v215 /*v471*/
	v_max3_num_f32 v197 /*v453*/, v217 /*v473*/, v219 /*v475*/, v221 /*v477*/
	v_max3_num_f32 v198 /*v454*/, v223 /*v479*/, v225 /*v481*/, v227 /*v483*/
	s_set_vgpr_msb 0x5559
	v_max3_num_f32 v200 /*v456*/, v229 /*v485*/, v162 /*v674*/, v231 /*v487*/
	s_set_vgpr_msb 0x5955
	v_max3_num_f32 v192 /*v448*/, v192 /*v448*/, v194 /*v450*/, v195 /*v451*/
	v_max3_num_f32 v193 /*v449*/, v193 /*v449*/, v199 /*v455*/, v201 /*v457*/
	v_max3_num_f32 v194 /*v450*/, v196 /*v452*/, v197 /*v453*/, v198 /*v454*/
	s_set_vgpr_msb 0x5565
	v_max3_num_f32 v195 /*v451*/, v200 /*v456*/, v233 /*v489*/, v169 /*v681*/
	s_set_vgpr_msb 0x6555
	v_max3_num_f32 v193 /*v449*/, v193 /*v449*/, v194 /*v450*/, v195 /*v451*/
	v_dual_mov_b32 v196 /*v452*/, v192 /*v448*/ :: v_dual_mov_b32 v194 /*v450*/, v193 /*v449*/
	v_permlanex16_b32 v196 /*v452*/, v196 /*v452*/, s49, 0xfedcba98
	v_permlanex16_b32 v194 /*v450*/, v194 /*v450*/, s49, 0xfedcba98
	v_dual_max_num_f32 v192 /*v448*/, v192 /*v448*/, v196 /*v452*/ :: v_dual_max_num_f32 v193 /*v449*/, v193 /*v449*/, v194 /*v450*/
	s_set_vgpr_msb 0x5549
	v_sub_f32_e32 v195 /*v451*/, v192 /*v448*/, v66 /*v578*/
	v_max_num_f32_e32 v192 /*v448*/, v192 /*v448*/, v66 /*v578*/
	v_sub_f32_e32 v194 /*v450*/, v193 /*v449*/, v71 /*v583*/
	s_set_vgpr_msb 0x4904
	v_cmp_lt_f32_e32 vcc_lo, 0x41000000, v195 /*v451*/
	s_cmp_eq_u32 vcc_lo, 0
	s_cselect_b32 s2, -1, 0
	s_set_vgpr_msb 0x489
	v_cndmask_b32_e64 v90 /*v602*/, v192 /*v448*/, v66 /*v578*/, s2
	s_set_vgpr_msb 0x8946
	v_cmp_lt_f32_e64 s2, 0x41000000, v194 /*v450*/
	v_max_num_f32_e32 v192 /*v448*/, v71 /*v583*/, v193 /*v449*/
	s_cmp_lg_u32 s2, 0
	s_cselect_b32 s3, -1, 0
	s_cmp_eq_u32 s2, 0
	s_cselect_b32 s2, -1, 0
	s_set_vgpr_msb 0x4689
	v_cndmask_b32_e64 v87 /*v599*/, v192 /*v448*/, v71 /*v583*/, s2
	v_mul_f32_e32 v60 /*v572*/, 0xbfb8aa3b, v90 /*v602*/
	v_mul_f32_e32 v70 /*v582*/, 0xbfb8aa3b, v87 /*v599*/
	s_set_vgpr_msb 0x8962
	v_pk_fma_f32 v[206:207] /*v[462:463]*/, v[2:3] /*v[514:515]*/, s[56:57], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6261
	v_pk_fma_f32 v[192:193] /*v[448:449]*/, v[250:251] /*v[506:507]*/, s[56:57], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[198:199] /*v[454:455]*/, v[254:255] /*v[510:511]*/, s[56:57], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6162
	v_pk_fma_f32 v[216:217] /*v[472:473]*/, v[68:69] /*v[580:581]*/, s[56:57], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[242:243] /*v[498:499]*/, v[94:95] /*v[606:607]*/, s[56:57], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x62a2
	v_pk_fma_f32 v[68:69] /*v[580:581]*/, v[106:107] /*v[618:619]*/, s[56:57], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[94:95] /*v[606:607]*/, v[110:111] /*v[622:623]*/, s[56:57], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa262
	v_pk_fma_f32 v[204:205] /*v[460:461]*/, v[0:1] /*v[512:513]*/, s[56:57], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6241
	v_exp_f32_e32 v222 /*v478*/, v206 /*v462*/
	v_exp_f32_e32 v232 /*v488*/, v207 /*v463*/
	v_nop
	s_set_vgpr_msb 0x4162
	v_pk_fma_f32 v[206:207] /*v[462:463]*/, v[8:9] /*v[520:521]*/, s[56:57], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[212:213] /*v[468:469]*/, v[12:13] /*v[524:525]*/, s[56:57], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[224:225] /*v[480:481]*/, v[20:21] /*v[532:533]*/, s[56:57], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6241
	v_exp_f32_e32 v194 /*v450*/, v193 /*v449*/
	v_exp_f32_e32 v202 /*v458*/, v198 /*v454*/
	v_exp_f32_e32 v208 /*v464*/, v199 /*v455*/
	v_nop
	s_set_vgpr_msb 0x4162
	v_pk_fma_f32 v[198:199] /*v[454:455]*/, v[4:5] /*v[516:517]*/, s[56:57], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v193 /*v449*/, v68 /*v580*/
	v_exp_f32_e32 v195 /*v451*/, v69 /*v581*/
	v_nop
	s_set_vgpr_msb 0x62a2
	v_pk_fma_f32 v[68:69] /*v[580:581]*/, v[112:113] /*v[624:625]*/, s[56:57], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa242
	v_exp_f32_e32 v203 /*v459*/, v94 /*v606*/
	v_exp_f32_e32 v209 /*v465*/, v95 /*v607*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[94:95] /*v[606:607]*/, v[116:117] /*v[628:629]*/, s[56:57], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa261
	v_pk_fma_f32 v[196:197] /*v[452:453]*/, v[252:253] /*v[508:509]*/, s[56:57], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v210 /*v466*/, v204 /*v460*/
	v_exp_f32_e32 v218 /*v474*/, v205 /*v461*/
	v_nop
	s_set_vgpr_msb 0x6162
	v_pk_fma_f32 v[204:205] /*v[460:461]*/, v[6:7] /*v[518:519]*/, s[56:57], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6281
	v_exp_f32_e32 v14 /*v526*/, v206 /*v462*/
	s_set_vgpr_msb 0x8141
	v_exp_f32_e32 v206 /*v462*/, v212 /*v468*/
	v_exp_f32_e32 v214 /*v470*/, v213 /*v469*/
	v_nop
	s_set_vgpr_msb 0x4162
	v_pk_fma_f32 v[212:213] /*v[468:469]*/, v[18:19] /*v[530:531]*/, s[56:57], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6281
	v_exp_f32_e32 v6 /*v518*/, v224 /*v480*/
	v_exp_f32_e32 v18 /*v530*/, v225 /*v481*/
	v_nop
	s_set_vgpr_msb 0x8162
	v_pk_fma_f32 v[224:225] /*v[480:481]*/, v[92:93] /*v[604:605]*/, s[56:57], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x62a2
	v_pk_fma_f32 v[92:93] /*v[604:605]*/, v[108:109] /*v[620:621]*/, s[56:57], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa241
	v_exp_f32_e32 v236 /*v492*/, v198 /*v454*/
	v_exp_f32_e32 v250 /*v506*/, v199 /*v455*/
	v_nop
	s_set_vgpr_msb 0x4162
	v_pk_fma_f32 v[198:199] /*v[454:455]*/, v[10:11] /*v[522:523]*/, s[56:57], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v211 /*v467*/, v68 /*v580*/
	v_exp_f32_e32 v219 /*v475*/, v69 /*v581*/
	v_nop
	s_set_vgpr_msb 0x62a2
	v_pk_fma_f32 v[68:69] /*v[580:581]*/, v[118:119] /*v[630:631]*/, s[56:57], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa242
	v_exp_f32_e32 v237 /*v493*/, v94 /*v606*/
	v_exp_f32_e32 v251 /*v507*/, v95 /*v607*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[94:95] /*v[606:607]*/, v[122:123] /*v[634:635]*/, s[56:57], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa241
	v_exp_f32_e32 v200 /*v456*/, v197 /*v453*/
	s_set_vgpr_msb 0x4142
	v_exp_f32_e32 v197 /*v453*/, v92 /*v604*/
	v_exp_f32_e32 v201 /*v457*/, v93 /*v605*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[92:93] /*v[604:605]*/, v[114:115] /*v[626:627]*/, s[56:57], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa241
	v_exp_f32_e32 v254 /*v510*/, v204 /*v460*/
	s_set_vgpr_msb 0x4181
	v_exp_f32_e32 v10 /*v522*/, v205 /*v461*/
	s_set_vgpr_msb 0x8141
	v_exp_f32_e32 v204 /*v460*/, v199 /*v455*/
	s_set_vgpr_msb 0x4142
	v_exp_f32_e32 v255 /*v511*/, v68 /*v580*/
	s_set_vgpr_msb 0x42a2
	v_exp_f32_e32 v11 /*v523*/, v69 /*v581*/
	v_nop
	v_pk_fma_f32 v[68:69] /*v[580:581]*/, v[124:125] /*v[636:637]*/, s[56:57], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa242
	v_exp_f32_e32 v199 /*v455*/, v94 /*v606*/
	v_exp_f32_e32 v205 /*v461*/, v95 /*v607*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[94:95] /*v[606:607]*/, v[128:129] /*v[640:641]*/, s[56:57], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa242
	v_exp_f32_e32 v223 /*v479*/, v92 /*v604*/
	v_exp_f32_e32 v233 /*v489*/, v93 /*v605*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[92:93] /*v[604:605]*/, v[120:121] /*v[632:633]*/, s[56:57], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa281
	v_exp_f32_e32 v26 /*v538*/, v207 /*v463*/
	s_set_vgpr_msb 0x8162
	v_pk_fma_f32 v[220:221] /*v[476:477]*/, v[16:17] /*v[528:529]*/, s[56:57], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v207 /*v463*/, v68 /*v580*/
	v_exp_f32_e32 v215 /*v471*/, v69 /*v581*/
	v_nop
	s_set_vgpr_msb 0x62a2
	v_pk_fma_f32 v[68:69] /*v[580:581]*/, v[130:131] /*v[642:643]*/, s[56:57], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa242
	v_exp_f32_e32 v229 /*v485*/, v94 /*v606*/
	v_exp_f32_e32 v241 /*v497*/, v95 /*v607*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[94:95] /*v[606:607]*/, v[134:135] /*v[646:647]*/, s[56:57], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v15 /*v527*/, v92 /*v604*/
	v_exp_f32_e32 v27 /*v539*/, v93 /*v605*/
	v_nop
	v_pk_fma_f32 v[92:93] /*v[604:605]*/, v[126:127] /*v[638:639]*/, s[56:57], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa241
	v_exp_f32_e32 v228 /*v484*/, v220 /*v476*/
	v_exp_f32_e32 v240 /*v496*/, v221 /*v477*/
	v_nop
	s_set_vgpr_msb 0x4162
	v_pk_fma_f32 v[220:221] /*v[476:477]*/, v[22:23] /*v[534:535]*/, s[56:57], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v245 /*v501*/, v68 /*v580*/
	s_set_vgpr_msb 0x62a2
	v_exp_f32_e32 v3 /*v515*/, v69 /*v581*/
	v_nop
	v_pk_fma_f32 v[68:69] /*v[580:581]*/, v[136:137] /*v[648:649]*/, s[56:57], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v23 /*v535*/, v94 /*v606*/
	v_exp_f32_e32 v33 /*v545*/, v95 /*v607*/
	v_nop
	v_pk_fma_f32 v[94:95] /*v[606:607]*/, v[140:141] /*v[652:653]*/, s[56:57], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa241
	v_exp_f32_e32 v226 /*v482*/, v217 /*v473*/
	s_set_vgpr_msb 0x4142
	v_exp_f32_e32 v217 /*v473*/, v92 /*v604*/
	v_exp_f32_e32 v227 /*v483*/, v93 /*v605*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[92:93] /*v[604:605]*/, v[132:133] /*v[644:645]*/, s[56:57], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa241
	v_exp_f32_e32 v244 /*v500*/, v212 /*v468*/
	s_set_vgpr_msb 0x4181
	v_exp_f32_e32 v2 /*v514*/, v213 /*v469*/
	v_nop
	s_set_vgpr_msb 0x8162
	v_pk_fma_f32 v[212:213] /*v[468:469]*/, v[24:25] /*v[536:537]*/, s[56:57], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6281
	v_exp_f32_e32 v22 /*v534*/, v220 /*v476*/
	s_set_vgpr_msb 0x8162
	v_pk_fma_f32 v[230:231] /*v[486:487]*/, v[28:29] /*v[540:541]*/, s[56:57], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[238:239] /*v[494:495]*/, v[30:31] /*v[542:543]*/, s[56:57], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6241
	v_exp_f32_e32 v220 /*v476*/, v225 /*v481*/
	s_set_vgpr_msb 0x41a2
	v_exp_f32_e32 v37 /*v549*/, v68 /*v580*/
	v_exp_f32_e32 v45 /*v557*/, v69 /*v581*/
	v_nop
	v_pk_fma_f32 v[68:69] /*v[580:581]*/, v[142:143] /*v[654:655]*/, s[56:57], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa242
	v_exp_f32_e32 v225 /*v481*/, v94 /*v606*/
	v_exp_f32_e32 v235 /*v491*/, v95 /*v607*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[94:95] /*v[606:607]*/, v[146:147] /*v[658:659]*/, s[56:57], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v7 /*v519*/, v92 /*v604*/
	v_exp_f32_e32 v19 /*v531*/, v93 /*v605*/
	v_nop
	v_pk_fma_f32 v[92:93] /*v[604:605]*/, v[138:139] /*v[650:651]*/, s[56:57], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa281
	v_exp_f32_e32 v36 /*v548*/, v212 /*v468*/
	s_set_vgpr_msb 0x8141
	v_exp_f32_e32 v212 /*v468*/, v224 /*v480*/
	v_exp_f32_e32 v224 /*v480*/, v230 /*v486*/
	v_exp_f32_e32 v234 /*v490*/, v231 /*v487*/
	v_nop
	s_set_vgpr_msb 0x4162
	v_pk_fma_f32 v[230:231] /*v[486:487]*/, v[34:35] /*v[546:547]*/, s[56:57], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6241
	v_exp_f32_e32 v252 /*v508*/, v239 /*v495*/
	s_set_vgpr_msb 0x4142
	v_exp_f32_e32 v239 /*v495*/, v68 /*v580*/
	v_exp_f32_e32 v253 /*v509*/, v69 /*v581*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[68:69] /*v[580:581]*/, v[148:149] /*v[660:661]*/, s[56:57], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v17 /*v529*/, v94 /*v606*/
	v_exp_f32_e32 v29 /*v541*/, v95 /*v607*/
	v_nop
	v_pk_fma_f32 v[94:95] /*v[606:607]*/, v[152:153] /*v[664:665]*/, s[56:57], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa281
	v_exp_f32_e32 v32 /*v544*/, v221 /*v477*/
	v_exp_f32_e32 v44 /*v556*/, v213 /*v469*/
	s_set_vgpr_msb 0x8142
	v_exp_f32_e32 v213 /*v469*/, v92 /*v604*/
	v_exp_f32_e32 v221 /*v477*/, v93 /*v605*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[92:93] /*v[604:605]*/, v[144:145] /*v[656:657]*/, s[56:57], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa281
	v_exp_f32_e32 v0 /*v512*/, v242 /*v498*/
	v_exp_f32_e32 v12 /*v524*/, v243 /*v499*/
	v_exp_f32_e32 v16 /*v528*/, v230 /*v486*/
	s_set_vgpr_msb 0x8162
	v_pk_fma_f32 v[242:243] /*v[498:499]*/, v[38:39] /*v[550:551]*/, s[56:57], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6281
	v_exp_f32_e32 v28 /*v540*/, v231 /*v487*/
	v_nop
	s_set_vgpr_msb 0x8162
	v_pk_fma_f32 v[230:231] /*v[486:487]*/, v[40:41] /*v[552:553]*/, s[56:57], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x62a2
	v_pk_fma_f32 v[8:9] /*v[520:521]*/, v[46:47] /*v[558:559]*/, s[56:57], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v31 /*v543*/, v68 /*v580*/
	v_exp_f32_e32 v41 /*v553*/, v69 /*v581*/
	v_nop
	v_pk_fma_f32 v[68:69] /*v[580:581]*/, v[154:155] /*v[666:667]*/, s[56:57], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v53 /*v565*/, v94 /*v606*/
	v_exp_f32_e32 v59 /*v571*/, v95 /*v607*/
	v_nop
	v_pk_fma_f32 v[94:95] /*v[606:607]*/, v[158:159] /*v[670:671]*/, s[56:57], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa241
	v_exp_f32_e32 v192 /*v448*/, v192 /*v448*/
	s_set_vgpr_msb 0x4162
	v_pk_fma_f32 v[246:247] /*v[502:503]*/, v[96:97] /*v[608:609]*/, s[56:57], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x62a2
	v_exp_f32_e32 v1 /*v513*/, v92 /*v604*/
	v_exp_f32_e32 v13 /*v525*/, v93 /*v605*/
	v_nop
	v_pk_fma_f32 v[92:93] /*v[604:605]*/, v[150:151] /*v[662:663]*/, s[56:57], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa281
	v_exp_f32_e32 v50 /*v562*/, v243 /*v499*/
	v_exp_f32_e32 v58 /*v570*/, v231 /*v487*/
	s_set_vgpr_msb 0x81a2
	v_pk_fma_f32 v[24:25] /*v[536:537]*/, v[48:49] /*v[560:561]*/, s[56:57], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v20 /*v532*/, v9 /*v521*/
	v_pk_fma_f32 v[48:49] /*v[560:561]*/, v[104:105] /*v[616:617]*/, s[56:57], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa242
	v_exp_f32_e32 v231 /*v487*/, v68 /*v580*/
	v_exp_f32_e32 v243 /*v499*/, v69 /*v581*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[68:69] /*v[580:581]*/, v[160:161] /*v[672:673]*/, s[56:57], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v9 /*v521*/, v94 /*v606*/
	v_exp_f32_e32 v21 /*v533*/, v95 /*v607*/
	v_nop
	v_pk_fma_f32 v[94:95] /*v[606:607]*/, v[164:165] /*v[676:677]*/, s[56:57], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa241
	v_exp_f32_e32 v196 /*v452*/, v196 /*v452*/
	v_exp_f32_e32 v238 /*v494*/, v238 /*v494*/
	s_set_vgpr_msb 0x4181
	v_exp_f32_e32 v30 /*v542*/, v246 /*v502*/
	v_exp_f32_e32 v40 /*v552*/, v247 /*v503*/
	v_nop
	s_set_vgpr_msb 0x8162
	v_pk_fma_f32 v[246:247] /*v[502:503]*/, v[98:99] /*v[610:611]*/, s[56:57], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x62a2
	v_pk_fma_f32 v[4:5] /*v[516:517]*/, v[100:101] /*v[612:613]*/, s[56:57], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v43 /*v555*/, v92 /*v604*/
	v_exp_f32_e32 v51 /*v563*/, v93 /*v605*/
	v_nop
	v_pk_fma_f32 v[92:93] /*v[604:605]*/, v[156:157] /*v[668:669]*/, s[56:57], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v34 /*v546*/, v25 /*v537*/
	v_pk_fma_f32 v[62:63] /*v[574:575]*/, v[54:55] /*v[566:567]*/, s[56:57], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v54 /*v566*/, v49 /*v561*/
	v_exp_f32_e32 v25 /*v537*/, v68 /*v580*/
	v_exp_f32_e32 v35 /*v547*/, v69 /*v581*/
	v_exp_f32_e32 v49 /*v561*/, v94 /*v606*/
	v_pk_fma_f32 v[68:69] /*v[580:581]*/, v[166:167] /*v[678:679]*/, s[56:57], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v55 /*v567*/, v95 /*v607*/
	v_nop
	s_set_vgpr_msb 0xa285
	v_pk_add_f32 v[94:95] /*v[606:607]*/, v[192:193] /*v[448:449]*/, v[194:195] /*v[450:451]*/
	v_pk_add_f32 v[96:97] /*v[608:609]*/, v[200:201] /*v[456:457]*/, v[202:203] /*v[458:459]*/
	s_set_vgpr_msb 0x8541
	v_exp_f32_e32 v198 /*v454*/, v198 /*v454*/
	s_set_vgpr_msb 0x4181
	v_exp_f32_e32 v42 /*v554*/, v242 /*v498*/
	v_exp_f32_e32 v52 /*v564*/, v230 /*v486*/
	s_set_vgpr_msb 0x8141
	v_exp_f32_e32 v230 /*v486*/, v246 /*v502*/
	v_exp_f32_e32 v242 /*v498*/, v247 /*v503*/
	s_set_vgpr_msb 0x4142
	v_exp_f32_e32 v246 /*v502*/, v4 /*v516*/
	s_set_vgpr_msb 0x42a2
	v_exp_f32_e32 v4 /*v516*/, v5 /*v517*/
	v_pk_fma_f32 v[38:39] /*v[550:551]*/, v[102:103] /*v[614:615]*/, s[56:57], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa242
	v_exp_f32_e32 v247 /*v503*/, v92 /*v604*/
	s_set_vgpr_msb 0x42a2
	v_exp_f32_e32 v5 /*v517*/, v93 /*v605*/
	v_nop
	v_pk_fma_f32 v[92:93] /*v[604:605]*/, v[162:163] /*v[674:675]*/, s[56:57], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[64:65] /*v[576:577]*/, v[56:57] /*v[568:569]*/, s[56:57], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v57 /*v569*/, v68 /*v580*/
	v_exp_f32_e32 v61 /*v573*/, v69 /*v581*/
	v_nop
	s_set_vgpr_msb 0xa289
	v_pk_add_f32 v[68:69] /*v[580:581]*/, v[196:197] /*v[452:453]*/, v[94:95] /*v[606:607]*/
	v_pk_add_f32 v[94:95] /*v[606:607]*/, v[208:209] /*v[464:465]*/, v[96:97] /*v[608:609]*/
	s_set_vgpr_msb 0x8985
	v_pk_add_f32 v[96:97] /*v[608:609]*/, v[210:211] /*v[466:467]*/, v[218:219] /*v[474:475]*/
	v_pk_add_f32 v[98:99] /*v[610:611]*/, v[232:233] /*v[488:489]*/, v[236:237] /*v[492:493]*/
	s_set_vgpr_msb 0x8589
	v_pk_add_f32 v[100:101] /*v[612:613]*/, v[254:255] /*v[510:511]*/, v[10:11] /*v[522:523]*/
	s_set_vgpr_msb 0x898a
	v_pk_add_f32 v[110:111] /*v[622:623]*/, v[18:19] /*v[530:531]*/, v[22:23] /*v[534:535]*/
	v_pk_add_f32 v[112:113] /*v[624:625]*/, v[36:37] /*v[548:549]*/, v[44:45] /*v[556:557]*/
	s_set_vgpr_msb 0x8a85
	v_pk_add_f32 v[116:117] /*v[628:629]*/, v[238:239] /*v[494:495]*/, v[252:253] /*v[508:509]*/
	s_set_vgpr_msb 0x858a
	v_pk_add_f32 v[118:119] /*v[630:631]*/, v[12:13] /*v[524:525]*/, v[16:17] /*v[528:529]*/
	s_set_vgpr_msb 0x8a41
	v_exp_f32_e32 v216 /*v472*/, v216 /*v472*/
	s_set_vgpr_msb 0x4186
	v_exp_f32_e32 v8 /*v520*/, v8 /*v520*/
	v_exp_f32_e32 v24 /*v536*/, v24 /*v536*/
	v_exp_f32_e32 v46 /*v558*/, v39 /*v551*/
	v_exp_f32_e32 v48 /*v560*/, v48 /*v560*/
	v_exp_f32_e32 v47 /*v559*/, v93 /*v605*/
	v_pk_add_f32 v[102:103] /*v[614:615]*/, v[26:27] /*v[538:539]*/, v[198:199] /*v[454:455]*/
	s_set_vgpr_msb 0x8685
	v_pk_add_f32 v[104:105] /*v[616:617]*/, v[206:207] /*v[462:463]*/, v[214:215] /*v[470:471]*/
	s_set_vgpr_msb 0x8589
	v_pk_add_f32 v[96:97] /*v[608:609]*/, v[222:223] /*v[478:479]*/, v[96:97] /*v[608:609]*/
	v_pk_add_f32 v[98:99] /*v[610:611]*/, v[250:251] /*v[506:507]*/, v[98:99] /*v[610:611]*/
	s_set_vgpr_msb 0x898a
	v_pk_add_f32 v[100:101] /*v[612:613]*/, v[14:15] /*v[526:527]*/, v[100:101] /*v[612:613]*/
	s_set_vgpr_msb 0x8a85
	v_pk_add_f32 v[106:107] /*v[618:619]*/, v[226:227] /*v[482:483]*/, v[228:229] /*v[484:485]*/
	v_pk_add_f32 v[114:115] /*v[626:627]*/, v[220:221] /*v[476:477]*/, v[224:225] /*v[480:481]*/
	s_set_vgpr_msb 0x858a
	v_pk_add_f32 v[110:111] /*v[622:623]*/, v[32:33] /*v[544:545]*/, v[110:111] /*v[622:623]*/
	s_set_vgpr_msb 0x8a89
	v_pk_add_f32 v[112:113] /*v[624:625]*/, v[212:213] /*v[468:469]*/, v[112:113] /*v[624:625]*/
	s_set_vgpr_msb 0x898a
	v_pk_add_f32 v[120:121] /*v[632:633]*/, v[30:31] /*v[542:543]*/, v[40:41] /*v[552:553]*/
	v_pk_add_f32 v[122:123] /*v[634:635]*/, v[50:51] /*v[562:563]*/, v[52:53] /*v[564:565]*/
	s_set_vgpr_msb 0x8a85
	v_pk_add_f32 v[124:125] /*v[636:637]*/, v[230:231] /*v[486:487]*/, v[242:243] /*v[498:499]*/
	s_set_vgpr_msb 0x85aa
	v_pk_add_f32 v[116:117] /*v[628:629]*/, v[0:1] /*v[512:513]*/, v[116:117] /*v[628:629]*/
	v_pk_add_f32 v[118:119] /*v[630:631]*/, v[28:29] /*v[540:541]*/, v[118:119] /*v[630:631]*/
	v_pk_add_f32 v[68:69] /*v[580:581]*/, v[68:69] /*v[580:581]*/, v[94:95] /*v[606:607]*/
	v_exp_f32_e32 v38 /*v550*/, v38 /*v550*/
	v_exp_f32_e32 v56 /*v568*/, v62 /*v574*/
	v_exp_f32_e32 v60 /*v572*/, v63 /*v575*/
	v_exp_f32_e32 v39 /*v551*/, v92 /*v604*/
	v_nop
	v_pk_fma_f32 v[92:93] /*v[604:605]*/, v[168:169] /*v[680:681]*/, s[56:57], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xaa89
	v_pk_add_f32 v[102:103] /*v[614:615]*/, v[204:205] /*v[460:461]*/, v[102:103] /*v[614:615]*/
	v_pk_add_f32 v[104:105] /*v[616:617]*/, v[216:217] /*v[472:473]*/, v[104:105] /*v[616:617]*/
	v_pk_add_f32 v[108:109] /*v[620:621]*/, v[244:245] /*v[500:501]*/, v[2:3] /*v[514:515]*/
	v_pk_add_f32 v[106:107] /*v[618:619]*/, v[240:241] /*v[496:497]*/, v[106:107] /*v[618:619]*/
	v_pk_add_f32 v[114:115] /*v[626:627]*/, v[234:235] /*v[490:491]*/, v[114:115] /*v[626:627]*/
	s_set_vgpr_msb 0x898a
	v_pk_add_f32 v[120:121] /*v[632:633]*/, v[42:43] /*v[554:555]*/, v[120:121] /*v[632:633]*/
	v_pk_add_f32 v[122:123] /*v[634:635]*/, v[58:59] /*v[570:571]*/, v[122:123] /*v[634:635]*/
	s_set_vgpr_msb 0x8a89
	v_pk_add_f32 v[124:125] /*v[636:637]*/, v[246:247] /*v[502:503]*/, v[124:125] /*v[636:637]*/
	s_set_vgpr_msb 0x898a
	v_pk_add_f32 v[126:127] /*v[638:639]*/, v[4:5] /*v[516:517]*/, v[8:9] /*v[520:521]*/
	v_pk_add_f32 v[128:129] /*v[640:641]*/, v[24:25] /*v[536:537]*/, v[34:35] /*v[546:547]*/
	v_pk_add_f32 v[130:131] /*v[642:643]*/, v[46:47] /*v[558:559]*/, v[48:49] /*v[560:561]*/
	v_pk_add_f32 v[68:69] /*v[580:581]*/, v[96:97] /*v[608:609]*/, v[68:69] /*v[580:581]*/
	v_pk_add_f32 v[96:97] /*v[608:609]*/, v[98:99] /*v[610:611]*/, v[100:101] /*v[612:613]*/
	v_pk_add_f32 v[98:99] /*v[610:611]*/, v[110:111] /*v[622:623]*/, v[112:113] /*v[624:625]*/
	v_pk_add_f32 v[100:101] /*v[612:613]*/, v[116:117] /*v[628:629]*/, v[118:119] /*v[630:631]*/
	v_exp_f32_e32 v62 /*v574*/, v64 /*v576*/
	v_exp_f32_e32 v63 /*v575*/, v92 /*v604*/
	v_pk_add_f32 v[108:109] /*v[620:621]*/, v[6:7] /*v[518:519]*/, v[108:109] /*v[620:621]*/
	v_pk_add_f32 v[132:133] /*v[644:645]*/, v[56:57] /*v[568:569]*/, v[60:61] /*v[572:573]*/
	v_pk_add_f32 v[94:95] /*v[606:607]*/, v[20:21] /*v[532:533]*/, v[126:127] /*v[638:639]*/
	v_pk_add_f32 v[126:127] /*v[638:639]*/, v[38:39] /*v[550:551]*/, v[128:129] /*v[640:641]*/
	v_pk_add_f32 v[128:129] /*v[640:641]*/, v[54:55] /*v[566:567]*/, v[130:131] /*v[642:643]*/
	v_pk_add_f32 v[104:105] /*v[616:617]*/, v[104:105] /*v[616:617]*/, v[106:107] /*v[618:619]*/
	v_pk_add_f32 v[106:107] /*v[618:619]*/, v[122:123] /*v[634:635]*/, v[124:125] /*v[636:637]*/
	v_pk_add_f32 v[96:97] /*v[608:609]*/, v[102:103] /*v[614:615]*/, v[96:97] /*v[608:609]*/
	v_pk_add_f32 v[98:99] /*v[610:611]*/, v[114:115] /*v[626:627]*/, v[98:99] /*v[610:611]*/
	v_pk_add_f32 v[100:101] /*v[612:613]*/, v[120:121] /*v[632:633]*/, v[100:101] /*v[612:613]*/
	v_pk_add_f32 v[130:131] /*v[642:643]*/, v[62:63] /*v[574:575]*/, v[132:133] /*v[644:645]*/
	v_pk_add_f32 v[102:103] /*v[614:615]*/, v[108:109] /*v[620:621]*/, v[104:105] /*v[616:617]*/
	v_pk_add_f32 v[94:95] /*v[606:607]*/, v[94:95] /*v[606:607]*/, v[106:107] /*v[618:619]*/
	v_pk_add_f32 v[104:105] /*v[616:617]*/, v[126:127] /*v[638:639]*/, v[128:129] /*v[640:641]*/
	v_pk_add_f32 v[68:69] /*v[580:581]*/, v[68:69] /*v[580:581]*/, v[96:97] /*v[608:609]*/
	v_pk_add_f32 v[96:97] /*v[608:609]*/, v[98:99] /*v[610:611]*/, v[100:101] /*v[612:613]*/
	v_exp_f32_e32 v64 /*v576*/, v65 /*v577*/
	v_exp_f32_e32 v65 /*v577*/, v93 /*v605*/
	v_nop
	v_pk_add_f32 v[92:93] /*v[604:605]*/, v[130:131] /*v[642:643]*/, v[104:105] /*v[616:617]*/
	v_pk_add_f32 v[68:69] /*v[580:581]*/, v[102:103] /*v[614:615]*/, v[68:69] /*v[580:581]*/
	v_pk_add_f32 v[94:95] /*v[606:607]*/, v[94:95] /*v[606:607]*/, v[96:97] /*v[608:609]*/
	v_sub_f32_e32 v70 /*v582*/, v66 /*v578*/, v90 /*v602*/
	v_pk_add_f32 v[92:93] /*v[604:605]*/, v[64:65] /*v[576:577]*/, v[92:93] /*v[604:605]*/
	v_pk_add_f32 v[68:69] /*v[580:581]*/, v[68:69] /*v[580:581]*/, v[94:95] /*v[606:607]*/
	v_mul_f32_e32 v70 /*v582*/, 0x3fb8aa3b, v70 /*v582*/
	v_pk_add_f32 v[66:67] /*v[578:579]*/, v[92:93] /*v[604:605]*/, v[68:69] /*v[580:581]*/
	v_exp_f32_e32 v70 /*v582*/, v70 /*v582*/
	v_dual_mov_b32 v68 /*v580*/, v66 /*v578*/ :: v_dual_mov_b32 v69 /*v581*/, v67 /*v579*/
	v_permlanex16_b32 v68 /*v580*/, v68 /*v580*/, s49, 0xfedcba98
	v_permlanex16_b32 v69 /*v581*/, v69 /*v581*/, s49, 0xfedcba98
	s_set_vgpr_msb 0x8a00
	s_cbranch_vccz .LBB0_41
	s_set_vgpr_msb 8
	v_pk_mul_f32 v[126:127], v[126:127], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[124:125], v[124:125], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[122:123], v[122:123], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[120:121], v[120:121], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[118:119], v[118:119], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[116:117], v[116:117], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[114:115], v[114:115], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[112:113], v[112:113], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[110:111], v[110:111], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[108:109], v[108:109], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[106:107], v[106:107], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[104:105], v[104:105], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[102:103], v[102:103], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[100:101], v[100:101], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[98:99], v[98:99], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[96:97], v[96:97], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[94:95], v[94:95], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[92:93], v[92:93], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[90:91], v[90:91], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[88:89], v[88:89], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[86:87], v[86:87], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[84:85], v[84:85], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[82:83], v[82:83], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[80:81], v[80:81], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[78:79], v[78:79], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[76:77], v[76:77], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[74:75], v[74:75], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[72:73], v[72:73], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[70:71], v[70:71], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[68:69], v[68:69], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[66:67], v[66:67], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[64:65], v[64:65], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x800
.LBB0_41:
	s_set_vgpr_msb 0x8a
	v_sub_f32_e32 v71 /*v583*/, v71 /*v583*/, v87 /*v599*/
	s_and_b32 s2, s3, exec_lo
	s_cselect_b32 s2, 1, 0
	s_cmp_lg_u32 s2, 1
	v_mul_f32_e32 v71 /*v583*/, 0x3fb8aa3b, v71 /*v583*/
	v_exp_f32_e32 v71 /*v583*/, v71 /*v583*/
	s_set_vgpr_msb 0x8a00
	s_cbranch_scc1 .LBB0_36
	v_nop
	s_set_vgpr_msb 0x82
	v_mov_b32_e32 v88 /*v600*/, v71 /*v583*/
	s_set_vgpr_msb 0x8208
	v_pk_mul_f32 v[62:63], v[62:63], v[88:89] /*v[600:601]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[60:61], v[60:61], v[88:89] /*v[600:601]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59], v[58:59], v[88:89] /*v[600:601]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[56:57], v[56:57], v[88:89] /*v[600:601]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[54:55], v[54:55], v[88:89] /*v[600:601]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[52:53], v[52:53], v[88:89] /*v[600:601]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[50:51], v[50:51], v[88:89] /*v[600:601]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[48:49], v[48:49], v[88:89] /*v[600:601]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[46:47], v[46:47], v[88:89] /*v[600:601]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[44:45], v[44:45], v[88:89] /*v[600:601]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[42:43], v[42:43], v[88:89] /*v[600:601]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[40:41], v[40:41], v[88:89] /*v[600:601]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39], v[38:39], v[88:89] /*v[600:601]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[36:37], v[36:37], v[88:89] /*v[600:601]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35], v[34:35], v[88:89] /*v[600:601]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[32:33], v[32:33], v[88:89] /*v[600:601]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[30:31], v[30:31], v[88:89] /*v[600:601]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[28:29], v[28:29], v[88:89] /*v[600:601]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[26:27], v[26:27], v[88:89] /*v[600:601]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25], v[24:25], v[88:89] /*v[600:601]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[22:23], v[22:23], v[88:89] /*v[600:601]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21], v[20:21], v[88:89] /*v[600:601]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[18:19], v[18:19], v[88:89] /*v[600:601]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[16:17], v[16:17], v[88:89] /*v[600:601]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[14:15], v[88:89] /*v[600:601]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13], v[12:13], v[88:89] /*v[600:601]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[10:11], v[88:89] /*v[600:601]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[8:9], v[88:89] /*v[600:601]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[6:7], v[88:89] /*v[600:601]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[4:5], v[88:89] /*v[600:601]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[2:3], v[2:3], v[88:89] /*v[600:601]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[0:1], v[0:1], v[88:89] /*v[600:601]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x800
	s_branch .LBB0_36
.LBB0_43:
	v_dual_mov_b32 v1, v0 :: v_dual_mov_b32 v2, v0
	v_dual_mov_b32 v3, v0 :: v_dual_mov_b32 v4, v0
	v_dual_mov_b32 v5, v0 :: v_dual_mov_b32 v6, v0
	s_set_vgpr_msb 64
	v_mov_b32_e32 v249 /*v505*/, 1.0
	s_set_vgpr_msb 0x4000
	v_mov_b32_e32 v7, v0
	s_set_vgpr_msb 0x82
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v84 /*v596*/, v78 /*v590*/ :: v_dual_mov_b32 v90 /*v602*/, v71 /*v583*/
	s_set_vgpr_msb 0x8200
	v_mov_b64_e32 v[12:13], v[4:5]
	s_set_vgpr_msb 0x41
	v_mov_b32_e32 v248 /*v504*/, v249 /*v505*/
	s_set_vgpr_msb 0x4100
	v_mov_b64_e32 v[14:15], v[6:7]
	v_mov_b64_e32 v[10:11], v[2:3]
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
	s_set_vgpr_msb 0x82
	v_mov_b32_e32 v71 /*v583*/, v87 /*v599*/
	s_set_vgpr_msb 0x8200
.LBB0_45:
	s_mov_b32 s27, 0
	s_cmp_le_i32 s7, s11
	s_mov_b32 s49, s27
	s_cbranch_scc1 .LBB0_54
	s_add_co_i32 s2, s19, s35
	s_lshl_b32 s3, s54, 7
	s_sub_co_i32 s2, s2, s18
	s_sub_co_i32 s5, s9, s3
	s_sub_co_i32 s2, s2, s16
	s_add_co_i32 s11, s97, s3
	s_max_i32 s2, s2, 0
	s_mov_b32 s4, 1
	s_lshr_b32 s2, s2, 7
	s_mov_b32 s24, 32
	s_min_u32 s2, s2, s98
	s_mov_b32 s23, 0x800000
	s_sub_co_i32 s6, 0, s2
	s_add_co_i32 s3, s5, 0xffffff80
	s_ashr_i32 s7, s6, 31
	s_add_co_i32 s16, s11, 0x80
	s_add_nc_u64 s[28:29], s[6:7], 1
	s_mov_b32 s21, 0xffff0000
	s_mov_b32 s20, 0x7510000
	s_mov_b32 s36, 0xf510000
	s_mov_b32 s11, 0x76543210
	s_mov_b32 s30, 0x3fb8aa3b
	s_branch .LBB0_48
.LBB0_47:
	s_set_vgpr_msb 0x8a
	v_cvt_pk_bf16_f32 v101 /*v613*/, v20 /*v532*/, v34 /*v546*/
	v_cvt_pk_bf16_f32 v100 /*v612*/, v10 /*v522*/, v18 /*v530*/
	s_set_vgpr_msb 0x8a89
	v_cvt_pk_bf16_f32 v99 /*v611*/, v246 /*v502*/, v2 /*v514*/
	s_set_vgpr_msb 0x8985
	v_cvt_pk_bf16_f32 v98 /*v610*/, v228 /*v484*/, v240 /*v496*/
	v_cvt_pk_bf16_f32 v97 /*v609*/, v212 /*v468*/, v224 /*v480*/
	v_cvt_pk_bf16_f32 v96 /*v608*/, v204 /*v460*/, v210 /*v466*/
	v_cvt_pk_bf16_f32 v95 /*v607*/, v196 /*v452*/, v200 /*v456*/
	v_cvt_pk_bf16_f32 v94 /*v606*/, v192 /*v448*/, v194 /*v450*/
	s_set_vgpr_msb 0x858a
	v_cvt_pk_bf16_f32 v109 /*v621*/, v21 /*v533*/, v35 /*v547*/
	v_cvt_pk_bf16_f32 v108 /*v620*/, v11 /*v523*/, v19 /*v531*/
	s_set_vgpr_msb 0x8a89
	v_cvt_pk_bf16_f32 v107 /*v619*/, v247 /*v503*/, v3 /*v515*/
	s_set_vgpr_msb 0x8985
	v_cvt_pk_bf16_f32 v106 /*v618*/, v229 /*v485*/, v241 /*v497*/
	v_cvt_pk_bf16_f32 v105 /*v617*/, v213 /*v469*/, v225 /*v481*/
	v_cvt_pk_bf16_f32 v104 /*v616*/, v205 /*v461*/, v211 /*v467*/
	v_cvt_pk_bf16_f32 v103 /*v615*/, v197 /*v453*/, v201 /*v457*/
	v_cvt_pk_bf16_f32 v102 /*v614*/, v193 /*v449*/, v195 /*v451*/
	s_set_vgpr_msb 0x8509
	s_wait_dscnt 0x3d
	v_wmma_f32_16x16x32_bf16 v[120:127], v[184:191] /*v[440:447]*/, v[94:101] /*v[606:613]*/, v[120:127]
	s_set_vgpr_msb 0x98a
	v_cvt_pk_bf16_f32 v117 /*v629*/, v39 /*v551*/, v49 /*v561*/
	v_cvt_pk_bf16_f32 v116 /*v628*/, v29 /*v541*/, v37 /*v549*/
	v_cvt_pk_bf16_f32 v115 /*v627*/, v13 /*v525*/, v23 /*v535*/
	s_set_vgpr_msb 0x8a89
	v_cvt_pk_bf16_f32 v114 /*v626*/, v251 /*v507*/, v5 /*v517*/
	s_set_vgpr_msb 0x8985
	v_cvt_pk_bf16_f32 v113 /*v625*/, v231 /*v487*/, v243 /*v499*/
	v_cvt_pk_bf16_f32 v112 /*v624*/, v219 /*v475*/, v227 /*v483*/
	v_cvt_pk_bf16_f32 v111 /*v623*/, v207 /*v463*/, v215 /*v471*/
	s_set_vgpr_msb 0x8509
	v_wmma_f32_16x16x32_bf16 v[56:63], v[184:191] /*v[440:447]*/, v[102:109] /*v[614:621]*/, v[56:63]
	s_set_vgpr_msb 0x985
	v_cvt_pk_bf16_f32 v110 /*v622*/, v199 /*v455*/, v203 /*v459*/
	s_set_vgpr_msb 0x854a
	v_cvt_pk_bf16_f32 v199 /*v455*/, v53 /*v565*/, v59 /*v571*/
	v_cvt_pk_bf16_f32 v197 /*v453*/, v31 /*v543*/, v41 /*v553*/
	v_cvt_pk_bf16_f32 v196 /*v452*/, v15 /*v527*/, v25 /*v537*/
	v_cvt_pk_bf16_f32 v191 /*v447*/, v38 /*v550*/, v48 /*v560*/
	v_cvt_pk_bf16_f32 v190 /*v446*/, v28 /*v540*/, v36 /*v548*/
	v_cvt_pk_bf16_f32 v189 /*v445*/, v12 /*v524*/, v22 /*v534*/
	s_set_vgpr_msb 0x4a09
	s_wait_dscnt 0x3c
	v_wmma_f32_16x16x32_bf16 v[112:119], v[152:159] /*v[408:415]*/, v[94:101] /*v[606:613]*/, v[112:119]
	s_set_vgpr_msb 0x949
	v_cvt_pk_bf16_f32 v188 /*v444*/, v250 /*v506*/, v4 /*v516*/
	s_set_vgpr_msb 0x4945
	v_cvt_pk_bf16_f32 v187 /*v443*/, v230 /*v486*/, v242 /*v498*/
	v_cvt_pk_bf16_f32 v186 /*v442*/, v218 /*v474*/, v226 /*v482*/
	v_cvt_pk_bf16_f32 v185 /*v441*/, v206 /*v462*/, v214 /*v470*/
	v_cvt_pk_bf16_f32 v184 /*v440*/, v198 /*v454*/, v202 /*v458*/
	s_set_vgpr_msb 0x454a
	v_cvt_pk_bf16_f32 v198 /*v454*/, v45 /*v557*/, v51 /*v563*/
	s_set_vgpr_msb 0x4a49
	v_cvt_pk_bf16_f32 v195 /*v451*/, v253 /*v509*/, v7 /*v519*/
	s_set_vgpr_msb 0x4909
	v_wmma_f32_16x16x32_bf16 v[48:55], v[152:159] /*v[408:415]*/, v[102:109] /*v[614:621]*/, v[48:55]
	s_set_vgpr_msb 0x945
	v_cvt_pk_bf16_f32 v194 /*v450*/, v237 /*v493*/, v245 /*v501*/
	v_cvt_pk_bf16_f32 v193 /*v449*/, v221 /*v477*/, v233 /*v489*/
	v_cvt_pk_bf16_f32 v192 /*v448*/, v209 /*v465*/, v217 /*v473*/
	s_set_vgpr_msb 0x454a
	v_cvt_pk_bf16_f32 v207 /*v463*/, v63 /*v575*/, v65 /*v577*/
	v_cvt_pk_bf16_f32 v206 /*v462*/, v57 /*v569*/, v61 /*v573*/
	v_cvt_pk_bf16_f32 v205 /*v461*/, v47 /*v559*/, v55 /*v567*/
	v_cvt_pk_bf16_f32 v204 /*v460*/, v33 /*v545*/, v43 /*v555*/
	s_set_vgpr_msb 0x4a09
	s_wait_dscnt 0x2d
	v_wmma_f32_16x16x32_bf16 v[104:111], v[120:127] /*v[376:383]*/, v[94:101] /*v[606:613]*/, v[104:111]
	s_set_vgpr_msb 0x94a
	v_cvt_pk_bf16_f32 v203 /*v459*/, v17 /*v529*/, v27 /*v539*/
	v_cvt_pk_bf16_f32 v202 /*v458*/, v1 /*v513*/, v9 /*v521*/
	s_set_vgpr_msb 0x4a45
	v_cvt_pk_bf16_f32 v201 /*v457*/, v239 /*v495*/, v255 /*v511*/
	v_cvt_pk_bf16_f32 v200 /*v456*/, v223 /*v479*/, v235 /*v491*/
	s_set_vgpr_msb 0x4586
	v_dual_fmac_f32 v80 /*v592*/, v68 /*v580*/, v248 /*v504*/ :: v_dual_fmac_f32 v69 /*v581*/, v70 /*v582*/, v249 /*v505*/
	v_cmp_lt_u64_e64 s2, s[56:57], s[48:49]
	s_set_vgpr_msb 0x8609
	v_wmma_f32_16x16x32_bf16 v[40:47], v[120:127] /*v[376:383]*/, v[102:109] /*v[614:621]*/, v[40:47]
	s_set_vgpr_msb 0x982
	v_mov_b32_e32 v89 /*v601*/, v91 /*v603*/
	s_set_vgpr_msb 0x824a
	v_dual_add_f32 v248 /*v504*/, v80 /*v592*/, v66 /*v578*/ :: v_dual_add_f32 v249 /*v505*/, v69 /*v581*/, v67 /*v579*/
	s_set_vgpr_msb 0x4a82
	v_dual_mov_b32 v80 /*v592*/, v92 /*v604*/ :: v_dual_mov_b32 v71 /*v583*/, v87 /*v599*/
	v_mov_b32_e32 v90 /*v602*/, v88 /*v600*/
	s_addk_co_i32 s3, 0xff80
	s_set_vgpr_msb 0x8209
	s_wait_dscnt 0x2c
	v_wmma_f32_16x16x32_bf16 v[96:103], v[88:95] /*v[344:351]*/, v[94:101] /*v[606:613]*/, v[96:103]
	s_addk_co_i32 s16, 0x80
	s_and_b32 vcc_lo, exec_lo, s2
	s_mov_b64 s[54:55], s[56:57]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[32:39], v[88:95] /*v[344:351]*/, v[102:109] /*v[614:621]*/, v[32:39]
	v_wmma_f32_16x16x32_bf16 v[88:95], v[56:63] /*v[312:319]*/, v[94:101] /*v[606:613]*/, v[88:95]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[56:63] /*v[312:319]*/, v[102:109] /*v[614:621]*/, v[24:31]
	v_wmma_f32_16x16x32_bf16 v[80:87], v[24:31] /*v[280:287]*/, v[94:101] /*v[606:613]*/, v[80:87]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[24:31] /*v[280:287]*/, v[102:109] /*v[614:621]*/, v[16:23]
	s_set_vgpr_msb 0x908
	v_wmma_f32_16x16x32_bf16 v[72:79], v[248:255], v[94:101] /*v[606:613]*/, v[72:79]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[248:255], v[102:109] /*v[614:621]*/, v[8:15]
	v_wmma_f32_16x16x32_bf16 v[64:71], v[216:223], v[94:101] /*v[606:613]*/, v[64:71]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[216:223], v[102:109] /*v[614:621]*/, v[0:7]
	s_set_vgpr_msb 0x805
	v_wmma_f32_16x16x32_bf16 v[120:127], v[176:183] /*v[432:439]*/, v[184:191] /*v[440:447]*/, v[120:127]
	s_set_vgpr_msb 0x509
	v_wmma_f32_16x16x32_bf16 v[56:63], v[176:183] /*v[432:439]*/, v[110:117] /*v[622:629]*/, v[56:63]
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x94a
	v_cvt_pk_bf16_f32 v183 /*v439*/, v52 /*v564*/, v58 /*v570*/
	v_cvt_pk_bf16_f32 v182 /*v438*/, v44 /*v556*/, v50 /*v562*/
	v_cvt_pk_bf16_f32 v181 /*v437*/, v30 /*v542*/, v40 /*v552*/
	s_set_vgpr_msb 0x4a05
	v_wmma_f32_16x16x32_bf16 v[112:119], v[144:151] /*v[400:407]*/, v[184:191] /*v[440:447]*/, v[112:119]
	s_set_vgpr_msb 0x54a
	v_cvt_pk_bf16_f32 v180 /*v436*/, v14 /*v526*/, v24 /*v536*/
	s_set_vgpr_msb 0x4a49
	v_cvt_pk_bf16_f32 v179 /*v435*/, v252 /*v508*/, v6 /*v518*/
	s_set_vgpr_msb 0x4945
	v_cvt_pk_bf16_f32 v178 /*v434*/, v236 /*v492*/, v244 /*v500*/
	v_cvt_pk_bf16_f32 v177 /*v433*/, v220 /*v476*/, v232 /*v488*/
	v_cvt_pk_bf16_f32 v176 /*v432*/, v208 /*v464*/, v216 /*v472*/
	s_set_vgpr_msb 0x4509
	v_wmma_f32_16x16x32_bf16 v[48:55], v[144:151] /*v[400:407]*/, v[110:117] /*v[622:629]*/, v[48:55]
	s_set_vgpr_msb 0x905
	v_wmma_f32_16x16x32_bf16 v[104:111], v[112:119] /*v[368:375]*/, v[184:191] /*v[440:447]*/, v[104:111]
	s_set_vgpr_msb 0x509
	v_wmma_f32_16x16x32_bf16 v[40:47], v[112:119] /*v[368:375]*/, v[110:117] /*v[622:629]*/, v[40:47]
	s_set_vgpr_msb 0x905
	v_wmma_f32_16x16x32_bf16 v[96:103], v[80:87] /*v[336:343]*/, v[184:191] /*v[440:447]*/, v[96:103]
	s_set_vgpr_msb 0x509
	v_wmma_f32_16x16x32_bf16 v[32:39], v[80:87] /*v[336:343]*/, v[110:117] /*v[622:629]*/, v[32:39]
	s_set_vgpr_msb 0x905
	v_wmma_f32_16x16x32_bf16 v[88:95], v[48:55] /*v[304:311]*/, v[184:191] /*v[440:447]*/, v[88:95]
	s_set_vgpr_msb 0x509
	v_wmma_f32_16x16x32_bf16 v[24:31], v[48:55] /*v[304:311]*/, v[110:117] /*v[622:629]*/, v[24:31]
	s_set_vgpr_msb 0x905
	v_wmma_f32_16x16x32_bf16 v[80:87], v[16:23] /*v[272:279]*/, v[184:191] /*v[440:447]*/, v[80:87]
	s_set_vgpr_msb 0x509
	v_wmma_f32_16x16x32_bf16 v[16:23], v[16:23] /*v[272:279]*/, v[110:117] /*v[622:629]*/, v[16:23]
	s_set_vgpr_msb 0x904
	v_wmma_f32_16x16x32_bf16 v[72:79], v[240:247], v[184:191] /*v[440:447]*/, v[72:79]
	s_set_vgpr_msb 0x408
	v_wmma_f32_16x16x32_bf16 v[8:15], v[240:247], v[110:117] /*v[622:629]*/, v[8:15]
	s_set_vgpr_msb 0x804
	v_wmma_f32_16x16x32_bf16 v[64:71], v[208:215], v[184:191] /*v[440:447]*/, v[64:71]
	s_set_vgpr_msb 0x408
	v_wmma_f32_16x16x32_bf16 v[0:7], v[208:215], v[110:117] /*v[622:629]*/, v[0:7]
	s_set_vgpr_msb 0x805
	v_wmma_f32_16x16x32_bf16 v[120:127], v[168:175] /*v[424:431]*/, v[176:183] /*v[432:439]*/, v[120:127]
	v_wmma_f32_16x16x32_bf16 v[56:63], v[168:175] /*v[424:431]*/, v[192:199] /*v[448:455]*/, v[56:63]
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x54a
	v_cvt_pk_bf16_f32 v175 /*v431*/, v62 /*v574*/, v64 /*v576*/
	v_cvt_pk_bf16_f32 v174 /*v430*/, v56 /*v568*/, v60 /*v572*/
	v_cvt_pk_bf16_f32 v173 /*v429*/, v46 /*v558*/, v54 /*v566*/
	s_set_vgpr_msb 0x4a05
	v_wmma_f32_16x16x32_bf16 v[112:119], v[136:143] /*v[392:399]*/, v[176:183] /*v[432:439]*/, v[112:119]
	s_set_vgpr_msb 0x54a
	v_cvt_pk_bf16_f32 v172 /*v428*/, v32 /*v544*/, v42 /*v554*/
	v_cvt_pk_bf16_f32 v171 /*v427*/, v16 /*v528*/, v26 /*v538*/
	v_cvt_pk_bf16_f32 v170 /*v426*/, v0 /*v512*/, v8 /*v520*/
	s_set_vgpr_msb 0x4a45
	v_cvt_pk_bf16_f32 v169 /*v425*/, v238 /*v494*/, v254 /*v510*/
	v_cvt_pk_bf16_f32 v168 /*v424*/, v222 /*v478*/, v234 /*v490*/
	s_set_vgpr_msb 0x4505
	v_wmma_f32_16x16x32_bf16 v[48:55], v[136:143] /*v[392:399]*/, v[192:199] /*v[448:455]*/, v[48:55]
	v_wmma_f32_16x16x32_bf16 v[104:111], v[104:111] /*v[360:367]*/, v[176:183] /*v[432:439]*/, v[104:111]
	v_wmma_f32_16x16x32_bf16 v[40:47], v[104:111] /*v[360:367]*/, v[192:199] /*v[448:455]*/, v[40:47]
	v_wmma_f32_16x16x32_bf16 v[96:103], v[72:79] /*v[328:335]*/, v[176:183] /*v[432:439]*/, v[96:103]
	v_wmma_f32_16x16x32_bf16 v[32:39], v[72:79] /*v[328:335]*/, v[192:199] /*v[448:455]*/, v[32:39]
	v_wmma_f32_16x16x32_bf16 v[88:95], v[40:47] /*v[296:303]*/, v[176:183] /*v[432:439]*/, v[88:95]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[40:47] /*v[296:303]*/, v[192:199] /*v[448:455]*/, v[24:31]
	v_wmma_f32_16x16x32_bf16 v[80:87], v[8:15] /*v[264:271]*/, v[176:183] /*v[432:439]*/, v[80:87]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[8:15] /*v[264:271]*/, v[192:199] /*v[448:455]*/, v[16:23]
	s_set_vgpr_msb 0x504
	v_wmma_f32_16x16x32_bf16 v[72:79], v[232:239], v[176:183] /*v[432:439]*/, v[72:79]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[232:239], v[192:199] /*v[448:455]*/, v[8:15]
	v_wmma_f32_16x16x32_bf16 v[64:71], v[200:207], v[176:183] /*v[432:439]*/, v[64:71]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[200:207], v[192:199] /*v[448:455]*/, v[0:7]
	s_set_vgpr_msb 0x405
	v_wmma_f32_16x16x32_bf16 v[120:127], v[160:167] /*v[416:423]*/, v[168:175] /*v[424:431]*/, v[120:127]
	v_wmma_f32_16x16x32_bf16 v[56:63], v[160:167] /*v[416:423]*/, v[200:207] /*v[456:463]*/, v[56:63]
	v_wmma_f32_16x16x32_bf16 v[112:119], v[128:135] /*v[384:391]*/, v[168:175] /*v[424:431]*/, v[112:119]
	v_wmma_f32_16x16x32_bf16 v[48:55], v[128:135] /*v[384:391]*/, v[200:207] /*v[456:463]*/, v[48:55]
	v_wmma_f32_16x16x32_bf16 v[104:111], v[96:103] /*v[352:359]*/, v[168:175] /*v[424:431]*/, v[104:111]
	v_wmma_f32_16x16x32_bf16 v[40:47], v[96:103] /*v[352:359]*/, v[200:207] /*v[456:463]*/, v[40:47]
	v_wmma_f32_16x16x32_bf16 v[96:103], v[64:71] /*v[320:327]*/, v[168:175] /*v[424:431]*/, v[96:103]
	v_wmma_f32_16x16x32_bf16 v[32:39], v[64:71] /*v[320:327]*/, v[200:207] /*v[456:463]*/, v[32:39]
	v_wmma_f32_16x16x32_bf16 v[88:95], v[32:39] /*v[288:295]*/, v[168:175] /*v[424:431]*/, v[88:95]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[32:39] /*v[288:295]*/, v[200:207] /*v[456:463]*/, v[24:31]
	v_wmma_f32_16x16x32_bf16 v[80:87], v[0:7] /*v[256:263]*/, v[168:175] /*v[424:431]*/, v[80:87]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[0:7] /*v[256:263]*/, v[200:207] /*v[456:463]*/, v[16:23]
	s_set_vgpr_msb 0x504
	v_wmma_f32_16x16x32_bf16 v[72:79], v[224:231], v[168:175] /*v[424:431]*/, v[72:79]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[224:231], v[200:207] /*v[456:463]*/, v[8:15]
	v_wmma_f32_16x16x32_bf16 v[64:71], v[192:199], v[168:175] /*v[424:431]*/, v[64:71]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[192:199], v[200:207] /*v[456:463]*/, v[0:7]
	s_set_vgpr_msb 0x400
	s_cbranch_vccz .LBB0_55
.LBB0_48:
	s_set_vgpr_msb 0x82
	v_dual_mov_b32 v92 /*v604*/, v84 /*v596*/ :: v_dual_mov_b32 v84 /*v596*/, v80 /*v592*/
	v_dual_mov_b32 v91 /*v603*/, v81 /*v593*/ :: v_dual_mov_b32 v81 /*v593*/, v89 /*v601*/
	s_add_nc_u64 s[56:57], s[54:55], 1
	s_wait_tensorcnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x8242
	ds_load_b128 v[184:187] /*v[440:443]*/, v92 /*v604*/
	ds_load_b128 v[188:191] /*v[444:447]*/, v92 /*v604*/ offset:32
	ds_load_b128 v[176:179] /*v[432:435]*/, v92 /*v604*/ offset:64
	ds_load_b128 v[180:183] /*v[436:439]*/, v92 /*v604*/ offset:96
	ds_load_b128 v[168:171] /*v[424:427]*/, v92 /*v604*/ offset:128
	ds_load_b128 v[172:175] /*v[428:431]*/, v92 /*v604*/ offset:160
	ds_load_b128 v[160:163] /*v[416:419]*/, v92 /*v604*/ offset:192
	ds_load_b128 v[164:167] /*v[420:423]*/, v92 /*v604*/ offset:224
	ds_load_b128 v[152:155] /*v[408:411]*/, v92 /*v604*/ offset:4352
	ds_load_b128 v[156:159] /*v[412:415]*/, v92 /*v604*/ offset:4384
	ds_load_b128 v[144:147] /*v[400:403]*/, v92 /*v604*/ offset:4416
	ds_load_b128 v[148:151] /*v[404:407]*/, v92 /*v604*/ offset:4448
	ds_load_b128 v[136:139] /*v[392:395]*/, v92 /*v604*/ offset:4480
	ds_load_b128 v[140:143] /*v[396:399]*/, v92 /*v604*/ offset:4512
	ds_load_b128 v[128:131] /*v[384:387]*/, v92 /*v604*/ offset:4544
	ds_load_b128 v[132:135] /*v[388:391]*/, v92 /*v604*/ offset:4576
	ds_load_b128 v[120:123] /*v[376:379]*/, v92 /*v604*/ offset:8704
	ds_load_b128 v[124:127] /*v[380:383]*/, v92 /*v604*/ offset:8736
	ds_load_b128 v[112:115] /*v[368:371]*/, v92 /*v604*/ offset:8768
	ds_load_b128 v[116:119] /*v[372:375]*/, v92 /*v604*/ offset:8800
	ds_load_b128 v[104:107] /*v[360:363]*/, v92 /*v604*/ offset:8832
	ds_load_b128 v[108:111] /*v[364:367]*/, v92 /*v604*/ offset:8864
	ds_load_b128 v[96:99] /*v[352:355]*/, v92 /*v604*/ offset:8896
	ds_load_b128 v[100:103] /*v[356:359]*/, v92 /*v604*/ offset:8928
	ds_load_b128 v[88:91] /*v[344:347]*/, v92 /*v604*/ offset:13056
	ds_load_b128 v[92:95] /*v[348:351]*/, v92 /*v604*/ offset:13088
	ds_load_b128 v[80:83] /*v[336:339]*/, v92 /*v604*/ offset:13120
	ds_load_b128 v[84:87] /*v[340:343]*/, v92 /*v604*/ offset:13152
	ds_load_b128 v[72:75] /*v[328:331]*/, v92 /*v604*/ offset:13184
	ds_load_b128 v[76:79] /*v[332:335]*/, v92 /*v604*/ offset:13216
	ds_load_b128 v[64:67] /*v[320:323]*/, v92 /*v604*/ offset:13248
	ds_load_b128 v[68:71] /*v[324:327]*/, v92 /*v604*/ offset:13280
	ds_load_b128 v[56:59] /*v[312:315]*/, v92 /*v604*/ offset:17408
	ds_load_b128 v[60:63] /*v[316:319]*/, v92 /*v604*/ offset:17440
	ds_load_b128 v[48:51] /*v[304:307]*/, v92 /*v604*/ offset:17472
	ds_load_b128 v[52:55] /*v[308:311]*/, v92 /*v604*/ offset:17504
	ds_load_b128 v[40:43] /*v[296:299]*/, v92 /*v604*/ offset:17536
	ds_load_b128 v[44:47] /*v[300:303]*/, v92 /*v604*/ offset:17568
	ds_load_b128 v[32:35] /*v[288:291]*/, v92 /*v604*/ offset:17600
	ds_load_b128 v[36:39] /*v[292:295]*/, v92 /*v604*/ offset:17632
	ds_load_b128 v[24:27] /*v[280:283]*/, v92 /*v604*/ offset:21760
	ds_load_b128 v[28:31] /*v[284:287]*/, v92 /*v604*/ offset:21792
	ds_load_b128 v[16:19] /*v[272:275]*/, v92 /*v604*/ offset:21824
	ds_load_b128 v[20:23] /*v[276:279]*/, v92 /*v604*/ offset:21856
	ds_load_b128 v[8:11] /*v[264:267]*/, v92 /*v604*/ offset:21888
	ds_load_b128 v[12:15] /*v[268:271]*/, v92 /*v604*/ offset:21920
	ds_load_b128 v[0:3] /*v[256:259]*/, v92 /*v604*/ offset:21952
	ds_load_b128 v[4:7] /*v[260:263]*/, v92 /*v604*/ offset:21984
	s_set_vgpr_msb 0x4202
	ds_load_b128 v[248:251], v92 /*v604*/ offset:26112
	ds_load_b128 v[252:255], v92 /*v604*/ offset:26144
	ds_load_b128 v[240:243], v92 /*v604*/ offset:26176
	ds_load_b128 v[244:247], v92 /*v604*/ offset:26208
	ds_load_b128 v[232:235], v92 /*v604*/ offset:26240
	ds_load_b128 v[236:239], v92 /*v604*/ offset:26272
	ds_load_b128 v[224:227], v92 /*v604*/ offset:26304
	ds_load_b128 v[228:231], v92 /*v604*/ offset:26336
	ds_load_b128 v[216:219], v92 /*v604*/ offset:30464
	ds_load_b128 v[220:223], v92 /*v604*/ offset:30496
	ds_load_b128 v[208:211], v92 /*v604*/ offset:30528
	ds_load_b128 v[212:215], v92 /*v604*/ offset:30560
	ds_load_b128 v[200:203], v92 /*v604*/ offset:30592
	ds_load_b128 v[204:207], v92 /*v604*/ offset:30624
	ds_load_b128 v[192:195], v92 /*v604*/ offset:30656
	ds_load_b128 v[196:199], v92 /*v604*/ offset:30688
	s_cmp_ge_i32 s56, s10
	s_set_vgpr_msb 0x200
	s_cbranch_scc1 .LBB0_50
	s_ashr_i32 s17, s16, 31
	s_set_vgpr_msb 0x41
	v_med3_i32 v192 /*v448*/, s3, 0, 0x80
	s_mul_u64 s[6:7], s[16:17], s[64:65]
	s_add_co_i32 s2, s28, s54
	s_lshl_b64 s[6:7], s[6:7], 1
	s_lshr_b32 s5, s2, 31
	s_add_nc_u64 s[38:39], s[50:51], s[6:7]
	s_mul_u64 s[6:7], s[16:17], s[62:63]
	v_readfirstlane_b32 s17, v192 /*v448*/
	s_add_co_i32 s5, s2, s5
	s_lshl_b64 s[6:7], s[6:7], 1
	s_and_b32 s5, s5, 0x7ffe
	s_add_nc_u64 s[6:7], s[52:53], s[6:7]
	s_sub_co_i32 s2, s2, s5
	s_sub_co_i32 s5, s17, s80
	s_lshl_b32 s2, s2, 17
	s_max_i32 s17, s5, 0
	s_add_nc_u64 s[6:7], s[44:45], s[6:7]
	s_lshl_b32 s17, s17, 16
	s_or_b32 s5, s81, s2
	s_bitset1_b32 s7, 31
	s_or_b32 s22, s17, 0x7fff
	s_mov_b32 s37, s21
	tensor_load_to_lds s[4:7], s[20:27]
	s_add_nc_u64 s[6:7], s[46:47], s[38:39]
	s_or_b32 s5, s78, s2
	s_bitset1_b32 s7, 31
	s_mov_b32 s38, s22
	s_mov_b32 s39, s23
	s_mov_b32 s40, s24
	s_mov_b32 s43, s27
	tensor_load_to_lds s[4:7], s[36:43]
	s_set_vgpr_msb 0x4100
.LBB0_50:
	s_set_vgpr_msb 0x41
	s_wait_dscnt 0x3e
	v_wmma_f32_16x16x32_bf16 v[192:199] /*v[448:455]*/, v[184:191] /*v[440:447]*/, v[128:135], 0
	s_set_vgpr_msb 0x4181
	v_wmma_f32_16x16x32_bf16 v[62:69] /*v[574:581]*/, v[184:191] /*v[440:447]*/, v[160:167], 0
	s_set_vgpr_msb 0x8141
	s_wait_dscnt 0x36
	v_wmma_f32_16x16x32_bf16 v[212:219] /*v[468:475]*/, v[152:159] /*v[408:415]*/, v[128:135], 0
	s_wait_dscnt 0x2e
	v_wmma_f32_16x16x32_bf16 v[230:237] /*v[486:493]*/, v[120:127] /*v[376:383]*/, v[128:135], 0
	s_wait_dscnt 0x26
	v_wmma_f32_16x16x32_bf16 v[250:257] /*v[506:513]*/, v[88:95] /*v[344:351]*/, v[128:135], 0
	s_set_vgpr_msb 0x4181
	s_wait_dscnt 0x1e
	v_wmma_f32_16x16x32_bf16 v[38:45] /*v[550:557]*/, v[56:63] /*v[312:319]*/, v[128:135], 0
	s_wait_dscnt 0x16
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[562:569]*/, v[24:31] /*v[280:287]*/, v[128:135], 0
	s_set_vgpr_msb 0x8151
	v_wmma_f32_16x16x32_bf16 v[192:199] /*v[448:455]*/, v[176:183] /*v[432:439]*/, v[136:143], v[192:199] /*v[448:455]*/
	s_set_vgpr_msb 0x51a1
	v_wmma_f32_16x16x32_bf16 v[62:69] /*v[574:581]*/, v[176:183] /*v[432:439]*/, v[168:175], v[62:69] /*v[574:581]*/
	v_wmma_f32_16x16x32_bf16 v[94:101] /*v[606:613]*/, v[152:159] /*v[408:415]*/, v[160:167], 0
	s_set_vgpr_msb 0xa151
	v_wmma_f32_16x16x32_bf16 v[212:219] /*v[468:475]*/, v[144:151] /*v[400:407]*/, v[136:143], v[212:219] /*v[468:475]*/
	s_set_vgpr_msb 0x5181
	v_wmma_f32_16x16x32_bf16 v[102:109] /*v[614:621]*/, v[120:127] /*v[376:383]*/, v[160:167], 0
	s_set_vgpr_msb 0x8151
	v_wmma_f32_16x16x32_bf16 v[230:237] /*v[486:493]*/, v[112:119] /*v[368:375]*/, v[136:143], v[230:237] /*v[486:493]*/
	s_set_vgpr_msb 0x5181
	v_wmma_f32_16x16x32_bf16 v[110:117] /*v[622:629]*/, v[88:95] /*v[344:351]*/, v[160:167], 0
	s_set_vgpr_msb 0x8151
	v_wmma_f32_16x16x32_bf16 v[250:257] /*v[506:513]*/, v[80:87] /*v[336:343]*/, v[136:143], v[250:257] /*v[506:513]*/
	s_set_vgpr_msb 0x51a1
	v_wmma_f32_16x16x32_bf16 v[118:125] /*v[630:637]*/, v[56:63] /*v[312:319]*/, v[160:167], 0
	v_wmma_f32_16x16x32_bf16 v[38:45] /*v[550:557]*/, v[48:55] /*v[304:311]*/, v[136:143], v[38:45] /*v[550:557]*/
	v_wmma_f32_16x16x32_bf16 v[126:133] /*v[638:645]*/, v[24:31] /*v[280:287]*/, v[160:167], 0
	s_wait_dscnt 0x14
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[562:569]*/, v[16:23] /*v[272:279]*/, v[136:143], v[50:57] /*v[562:569]*/
	s_set_vgpr_msb 0xa180
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x32_bf16 v[134:141] /*v[646:653]*/, v[248:255], v[128:135], 0
	v_wmma_f32_16x16x32_bf16 v[142:149] /*v[654:661]*/, v[248:255], v[160:167], 0
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x32_bf16 v[150:157] /*v[662:669]*/, v[216:223], v[128:135], 0
	v_wmma_f32_16x16x32_bf16 v[158:165] /*v[670:677]*/, v[216:223], v[160:167], 0
	s_set_vgpr_msb 0x8051
	v_wmma_f32_16x16x32_bf16 v[192:199] /*v[448:455]*/, v[168:175] /*v[424:431]*/, v[144:151], v[192:199] /*v[448:455]*/
	s_set_vgpr_msb 0x51a1
	v_wmma_f32_16x16x32_bf16 v[62:69] /*v[574:581]*/, v[168:175] /*v[424:431]*/, v[176:183], v[62:69] /*v[574:581]*/
	v_wmma_f32_16x16x32_bf16 v[94:101] /*v[606:613]*/, v[144:151] /*v[400:407]*/, v[168:175], v[94:101] /*v[606:613]*/
	s_set_vgpr_msb 0xa151
	v_wmma_f32_16x16x32_bf16 v[212:219] /*v[468:475]*/, v[136:143] /*v[392:399]*/, v[144:151], v[212:219] /*v[468:475]*/
	s_set_vgpr_msb 0x51a1
	v_wmma_f32_16x16x32_bf16 v[102:109] /*v[614:621]*/, v[112:119] /*v[368:375]*/, v[168:175], v[102:109] /*v[614:621]*/
	s_set_vgpr_msb 0xa151
	v_wmma_f32_16x16x32_bf16 v[230:237] /*v[486:493]*/, v[104:111] /*v[360:367]*/, v[144:151], v[230:237] /*v[486:493]*/
	s_set_vgpr_msb 0x51a1
	v_wmma_f32_16x16x32_bf16 v[110:117] /*v[622:629]*/, v[80:87] /*v[336:343]*/, v[168:175], v[110:117] /*v[622:629]*/
	s_set_vgpr_msb 0xa151
	v_wmma_f32_16x16x32_bf16 v[250:257] /*v[506:513]*/, v[72:79] /*v[328:335]*/, v[144:151], v[250:257] /*v[506:513]*/
	s_set_vgpr_msb 0x51a1
	v_wmma_f32_16x16x32_bf16 v[118:125] /*v[630:637]*/, v[48:55] /*v[304:311]*/, v[168:175], v[118:125] /*v[630:637]*/
	v_wmma_f32_16x16x32_bf16 v[38:45] /*v[550:557]*/, v[40:47] /*v[296:303]*/, v[144:151], v[38:45] /*v[550:557]*/
	v_wmma_f32_16x16x32_bf16 v[126:133] /*v[638:645]*/, v[16:23] /*v[272:279]*/, v[168:175], v[126:133] /*v[638:645]*/
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[562:569]*/, v[8:15] /*v[264:271]*/, v[144:151], v[50:57] /*v[562:569]*/
	s_set_vgpr_msb 0xa1a0
	v_wmma_f32_16x16x32_bf16 v[134:141] /*v[646:653]*/, v[240:247], v[136:143], v[134:141] /*v[646:653]*/
	v_wmma_f32_16x16x32_bf16 v[142:149] /*v[654:661]*/, v[240:247], v[168:175], v[142:149] /*v[654:661]*/
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x32_bf16 v[150:157] /*v[662:669]*/, v[208:215], v[136:143], v[150:157] /*v[662:669]*/
	v_wmma_f32_16x16x32_bf16 v[158:165] /*v[670:677]*/, v[208:215], v[168:175], v[158:165] /*v[670:677]*/
	s_set_vgpr_msb 0xa051
	v_wmma_f32_16x16x32_bf16 v[192:199] /*v[448:455]*/, v[160:167] /*v[416:423]*/, v[152:159], v[192:199] /*v[448:455]*/
	s_set_vgpr_msb 0x51a1
	v_wmma_f32_16x16x32_bf16 v[62:69] /*v[574:581]*/, v[160:167] /*v[416:423]*/, v[184:191], v[62:69] /*v[574:581]*/
	v_wmma_f32_16x16x32_bf16 v[94:101] /*v[606:613]*/, v[136:143] /*v[392:399]*/, v[176:183], v[94:101] /*v[606:613]*/
	s_set_vgpr_msb 0xa151
	v_wmma_f32_16x16x32_bf16 v[212:219] /*v[468:475]*/, v[128:135] /*v[384:391]*/, v[152:159], v[212:219] /*v[468:475]*/
	s_set_vgpr_msb 0x51a1
	v_wmma_f32_16x16x32_bf16 v[102:109] /*v[614:621]*/, v[104:111] /*v[360:367]*/, v[176:183], v[102:109] /*v[614:621]*/
	s_set_vgpr_msb 0xa151
	v_wmma_f32_16x16x32_bf16 v[230:237] /*v[486:493]*/, v[96:103] /*v[352:359]*/, v[152:159], v[230:237] /*v[486:493]*/
	s_set_vgpr_msb 0x51a1
	v_wmma_f32_16x16x32_bf16 v[110:117] /*v[622:629]*/, v[72:79] /*v[328:335]*/, v[176:183], v[110:117] /*v[622:629]*/
	s_set_vgpr_msb 0xa151
	v_wmma_f32_16x16x32_bf16 v[250:257] /*v[506:513]*/, v[64:71] /*v[320:327]*/, v[152:159], v[250:257] /*v[506:513]*/
	s_set_vgpr_msb 0x51a1
	v_wmma_f32_16x16x32_bf16 v[118:125] /*v[630:637]*/, v[40:47] /*v[296:303]*/, v[176:183], v[118:125] /*v[630:637]*/
	v_wmma_f32_16x16x32_bf16 v[38:45] /*v[550:557]*/, v[32:39] /*v[288:295]*/, v[152:159], v[38:45] /*v[550:557]*/
	v_wmma_f32_16x16x32_bf16 v[126:133] /*v[638:645]*/, v[8:15] /*v[264:271]*/, v[176:183], v[126:133] /*v[638:645]*/
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[562:569]*/, v[0:7] /*v[256:263]*/, v[152:159], v[50:57] /*v[562:569]*/
	s_set_vgpr_msb 0xa1a0
	v_wmma_f32_16x16x32_bf16 v[134:141] /*v[646:653]*/, v[232:239], v[144:151], v[134:141] /*v[646:653]*/
	v_wmma_f32_16x16x32_bf16 v[142:149] /*v[654:661]*/, v[232:239], v[176:183], v[142:149] /*v[654:661]*/
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[150:157] /*v[662:669]*/, v[200:207], v[144:151], v[150:157] /*v[662:669]*/
	v_wmma_f32_16x16x32_bf16 v[158:165] /*v[670:677]*/, v[200:207], v[176:183], v[158:165] /*v[670:677]*/
	s_set_vgpr_msb 0xa0a1
	v_wmma_f32_16x16x32_bf16 v[94:101] /*v[606:613]*/, v[128:135] /*v[384:391]*/, v[184:191], v[94:101] /*v[606:613]*/
	v_wmma_f32_16x16x32_bf16 v[102:109] /*v[614:621]*/, v[96:103] /*v[352:359]*/, v[184:191], v[102:109] /*v[614:621]*/
	v_wmma_f32_16x16x32_bf16 v[110:117] /*v[622:629]*/, v[64:71] /*v[320:327]*/, v[184:191], v[110:117] /*v[622:629]*/
	v_wmma_f32_16x16x32_bf16 v[118:125] /*v[630:637]*/, v[32:39] /*v[288:295]*/, v[184:191], v[118:125] /*v[630:637]*/
	v_wmma_f32_16x16x32_bf16 v[126:133] /*v[638:645]*/, v[0:7] /*v[256:263]*/, v[184:191], v[126:133] /*v[638:645]*/
	s_set_vgpr_msb 0xa1a0
	v_wmma_f32_16x16x32_bf16 v[134:141] /*v[646:653]*/, v[224:231], v[152:159], v[134:141] /*v[646:653]*/
	v_wmma_f32_16x16x32_bf16 v[142:149] /*v[654:661]*/, v[224:231], v[184:191], v[142:149] /*v[654:661]*/
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[150:157] /*v[662:669]*/, v[192:199], v[152:159], v[150:157] /*v[662:669]*/
	v_wmma_f32_16x16x32_bf16 v[158:165] /*v[670:677]*/, v[192:199], v[184:191], v[158:165] /*v[670:677]*/
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0xa042
	ds_load_tr16_b128 v[184:187] /*v[440:443]*/, v91 /*v603*/
	ds_load_tr16_b128 v[152:155] /*v[408:411]*/, v91 /*v603*/ offset:32
	ds_load_tr16_b128 v[188:191] /*v[444:447]*/, v91 /*v603*/ offset:4608
	ds_load_tr16_b128 v[156:159] /*v[412:415]*/, v91 /*v603*/ offset:4640
	ds_load_tr16_b128 v[176:179] /*v[432:435]*/, v91 /*v603*/ offset:9216
	ds_load_tr16_b128 v[144:147] /*v[400:403]*/, v91 /*v603*/ offset:9248
	ds_load_tr16_b128 v[180:183] /*v[436:439]*/, v91 /*v603*/ offset:13824
	ds_load_tr16_b128 v[148:151] /*v[404:407]*/, v91 /*v603*/ offset:13856
	ds_load_tr16_b128 v[168:171] /*v[424:427]*/, v91 /*v603*/ offset:18432
	ds_load_tr16_b128 v[136:139] /*v[392:395]*/, v91 /*v603*/ offset:18464
	ds_load_tr16_b128 v[172:175] /*v[428:431]*/, v91 /*v603*/ offset:23040
	ds_load_tr16_b128 v[140:143] /*v[396:399]*/, v91 /*v603*/ offset:23072
	ds_load_tr16_b128 v[160:163] /*v[416:419]*/, v91 /*v603*/ offset:27648
	ds_load_tr16_b128 v[128:131] /*v[384:387]*/, v91 /*v603*/ offset:27680
	ds_load_tr16_b128 v[164:167] /*v[420:423]*/, v91 /*v603*/ offset:32256
	ds_load_tr16_b128 v[132:135] /*v[388:391]*/, v91 /*v603*/ offset:32288
	ds_load_tr16_b128 v[120:123] /*v[376:379]*/, v91 /*v603*/ offset:64
	ds_load_tr16_b128 v[88:91] /*v[344:347]*/, v91 /*v603*/ offset:96
	ds_load_tr16_b128 v[124:127] /*v[380:383]*/, v91 /*v603*/ offset:4672
	ds_load_tr16_b128 v[92:95] /*v[348:351]*/, v91 /*v603*/ offset:4704
	ds_load_tr16_b128 v[112:115] /*v[368:371]*/, v91 /*v603*/ offset:9280
	ds_load_tr16_b128 v[80:83] /*v[336:339]*/, v91 /*v603*/ offset:9312
	ds_load_tr16_b128 v[116:119] /*v[372:375]*/, v91 /*v603*/ offset:13888
	ds_load_tr16_b128 v[84:87] /*v[340:343]*/, v91 /*v603*/ offset:13920
	ds_load_tr16_b128 v[104:107] /*v[360:363]*/, v91 /*v603*/ offset:18496
	ds_load_tr16_b128 v[72:75] /*v[328:331]*/, v91 /*v603*/ offset:18528
	ds_load_tr16_b128 v[108:111] /*v[364:367]*/, v91 /*v603*/ offset:23104
	ds_load_tr16_b128 v[76:79] /*v[332:335]*/, v91 /*v603*/ offset:23136
	ds_load_tr16_b128 v[96:99] /*v[352:355]*/, v91 /*v603*/ offset:27712
	ds_load_tr16_b128 v[64:67] /*v[320:323]*/, v91 /*v603*/ offset:27744
	ds_load_tr16_b128 v[100:103] /*v[356:359]*/, v91 /*v603*/ offset:32320
	ds_load_tr16_b128 v[68:71] /*v[324:327]*/, v91 /*v603*/ offset:32352
	ds_load_tr16_b128 v[56:59] /*v[312:315]*/, v91 /*v603*/ offset:128
	ds_load_tr16_b128 v[24:27] /*v[280:283]*/, v91 /*v603*/ offset:160
	ds_load_tr16_b128 v[60:63] /*v[316:319]*/, v91 /*v603*/ offset:4736
	ds_load_tr16_b128 v[28:31] /*v[284:287]*/, v91 /*v603*/ offset:4768
	ds_load_tr16_b128 v[48:51] /*v[304:307]*/, v91 /*v603*/ offset:9344
	ds_load_tr16_b128 v[16:19] /*v[272:275]*/, v91 /*v603*/ offset:9376
	ds_load_tr16_b128 v[52:55] /*v[308:311]*/, v91 /*v603*/ offset:13952
	ds_load_tr16_b128 v[20:23] /*v[276:279]*/, v91 /*v603*/ offset:13984
	ds_load_tr16_b128 v[40:43] /*v[296:299]*/, v91 /*v603*/ offset:18560
	ds_load_tr16_b128 v[8:11] /*v[264:267]*/, v91 /*v603*/ offset:18592
	ds_load_tr16_b128 v[44:47] /*v[300:303]*/, v91 /*v603*/ offset:23168
	ds_load_tr16_b128 v[12:15] /*v[268:271]*/, v91 /*v603*/ offset:23200
	ds_load_tr16_b128 v[32:35] /*v[288:291]*/, v91 /*v603*/ offset:27776
	ds_load_tr16_b128 v[0:3] /*v[256:259]*/, v91 /*v603*/ offset:27808
	ds_load_tr16_b128 v[36:39] /*v[292:295]*/, v91 /*v603*/ offset:32384
	ds_load_tr16_b128 v[4:7] /*v[260:263]*/, v91 /*v603*/ offset:32416
	s_set_vgpr_msb 0x4202
	ds_load_tr16_b128 v[248:251], v91 /*v603*/ offset:192
	ds_load_tr16_b128 v[216:219], v91 /*v603*/ offset:224
	ds_load_tr16_b128 v[252:255], v91 /*v603*/ offset:4800
	ds_load_tr16_b128 v[220:223], v91 /*v603*/ offset:4832
	ds_load_tr16_b128 v[240:243], v91 /*v603*/ offset:9408
	ds_load_tr16_b128 v[208:211], v91 /*v603*/ offset:9440
	ds_load_tr16_b128 v[244:247], v91 /*v603*/ offset:14016
	ds_load_tr16_b128 v[212:215], v91 /*v603*/ offset:14048
	ds_load_tr16_b128 v[232:235], v91 /*v603*/ offset:18624
	ds_load_tr16_b128 v[200:203], v91 /*v603*/ offset:18656
	ds_load_tr16_b128 v[236:239], v91 /*v603*/ offset:23232
	ds_load_tr16_b128 v[204:207], v91 /*v603*/ offset:23264
	ds_load_tr16_b128 v[224:227], v91 /*v603*/ offset:27840
	ds_load_tr16_b128 v[192:195], v91 /*v603*/ offset:27872
	ds_load_tr16_b128 v[228:231], v91 /*v603*/ offset:32448
	ds_load_tr16_b128 v[196:199], v91 /*v603*/ offset:32480
	s_set_vgpr_msb 0x255
	v_max3_num_f32 v200 /*v456*/, v192 /*v448*/, v193 /*v449*/, v194 /*v450*/
	s_set_vgpr_msb 0x556a
	v_max3_num_f32 v201 /*v457*/, v62 /*v574*/, v63 /*v575*/, v64 /*v576*/
	s_set_vgpr_msb 0x6a55
	v_max3_num_f32 v202 /*v458*/, v195 /*v451*/, v196 /*v452*/, v197 /*v453*/
	s_set_vgpr_msb 0x556a
	v_max3_num_f32 v203 /*v459*/, v65 /*v577*/, v66 /*v578*/, v67 /*v579*/
	s_set_vgpr_msb 0x6a55
	v_max3_num_f32 v204 /*v460*/, v198 /*v454*/, v199 /*v455*/, v212 /*v468*/
	s_set_vgpr_msb 0x556a
	v_max3_num_f32 v205 /*v461*/, v68 /*v580*/, v69 /*v581*/, v94 /*v606*/
	s_set_vgpr_msb 0x6a55
	v_max3_num_f32 v206 /*v462*/, v213 /*v469*/, v214 /*v470*/, v215 /*v471*/
	v_max3_num_f32 v208 /*v464*/, v216 /*v472*/, v217 /*v473*/, v218 /*v474*/
	v_max3_num_f32 v210 /*v466*/, v219 /*v475*/, v230 /*v486*/, v231 /*v487*/
	v_max3_num_f32 v220 /*v476*/, v232 /*v488*/, v233 /*v489*/, v234 /*v490*/
	v_max3_num_f32 v222 /*v478*/, v235 /*v491*/, v236 /*v492*/, v237 /*v493*/
	v_max3_num_f32 v224 /*v480*/, v250 /*v506*/, v251 /*v507*/, v252 /*v508*/
	v_max3_num_f32 v226 /*v482*/, v253 /*v509*/, v254 /*v510*/, v255 /*v511*/
	s_set_vgpr_msb 0x556a
	v_max3_num_f32 v228 /*v484*/, v0 /*v512*/, v1 /*v513*/, v38 /*v550*/
	v_max3_num_f32 v238 /*v494*/, v39 /*v551*/, v40 /*v552*/, v41 /*v553*/
	v_max3_num_f32 v240 /*v496*/, v42 /*v554*/, v43 /*v555*/, v44 /*v556*/
	v_max3_num_f32 v242 /*v498*/, v45 /*v557*/, v50 /*v562*/, v51 /*v563*/
	v_max3_num_f32 v244 /*v500*/, v52 /*v564*/, v53 /*v565*/, v54 /*v566*/
	v_max3_num_f32 v246 /*v502*/, v55 /*v567*/, v56 /*v568*/, v57 /*v569*/
	s_set_vgpr_msb 0x6aaa
	v_max3_num_f32 v2 /*v514*/, v134 /*v646*/, v135 /*v647*/, v136 /*v648*/
	v_max3_num_f32 v4 /*v516*/, v137 /*v649*/, v138 /*v650*/, v139 /*v651*/
	v_max_num_f32_e32 v6 /*v518*/, v140 /*v652*/, v141 /*v653*/
	v_max3_num_f32 v8 /*v520*/, v151 /*v663*/, v152 /*v664*/, v153 /*v665*/
	s_set_vgpr_msb 0xaa6a
	v_max3_num_f32 v207 /*v463*/, v95 /*v607*/, v96 /*v608*/, v97 /*v609*/
	v_max3_num_f32 v209 /*v465*/, v98 /*v610*/, v99 /*v611*/, v100 /*v612*/
	v_max3_num_f32 v211 /*v467*/, v101 /*v613*/, v102 /*v614*/, v103 /*v615*/
	v_max3_num_f32 v221 /*v477*/, v104 /*v616*/, v105 /*v617*/, v106 /*v618*/
	v_max3_num_f32 v223 /*v479*/, v107 /*v619*/, v108 /*v620*/, v109 /*v621*/
	v_max3_num_f32 v225 /*v481*/, v110 /*v622*/, v111 /*v623*/, v112 /*v624*/
	v_max3_num_f32 v227 /*v483*/, v113 /*v625*/, v114 /*v626*/, v115 /*v627*/
	v_max3_num_f32 v229 /*v485*/, v116 /*v628*/, v117 /*v629*/, v118 /*v630*/
	v_max3_num_f32 v239 /*v495*/, v119 /*v631*/, v120 /*v632*/, v121 /*v633*/
	v_max3_num_f32 v241 /*v497*/, v122 /*v634*/, v123 /*v635*/, v124 /*v636*/
	v_max3_num_f32 v243 /*v499*/, v125 /*v637*/, v126 /*v638*/, v127 /*v639*/
	v_max3_num_f32 v245 /*v501*/, v128 /*v640*/, v129 /*v641*/, v130 /*v642*/
	v_max3_num_f32 v247 /*v503*/, v131 /*v643*/, v132 /*v644*/, v133 /*v645*/
	s_set_vgpr_msb 0x6aaa
	v_max3_num_f32 v3 /*v515*/, v142 /*v654*/, v143 /*v655*/, v144 /*v656*/
	v_max3_num_f32 v5 /*v517*/, v145 /*v657*/, v146 /*v658*/, v147 /*v659*/
	v_max_num_f32_e32 v7 /*v519*/, v148 /*v660*/, v149 /*v661*/
	v_max3_num_f32 v9 /*v521*/, v159 /*v671*/, v160 /*v672*/, v161 /*v673*/
	v_max3_num_f32 v10 /*v522*/, v154 /*v666*/, v155 /*v667*/, v156 /*v668*/
	s_set_vgpr_msb 0xaa55
	v_max3_num_f32 v200 /*v456*/, v200 /*v456*/, v202 /*v458*/, v204 /*v460*/
	v_max3_num_f32 v201 /*v457*/, v201 /*v457*/, v203 /*v459*/, v205 /*v461*/
	v_max3_num_f32 v202 /*v458*/, v206 /*v462*/, v208 /*v464*/, v210 /*v466*/
	v_max3_num_f32 v203 /*v459*/, v220 /*v476*/, v222 /*v478*/, v224 /*v480*/
	v_max3_num_f32 v204 /*v460*/, v226 /*v482*/, v228 /*v484*/, v238 /*v494*/
	v_max3_num_f32 v205 /*v461*/, v240 /*v496*/, v242 /*v498*/, v244 /*v500*/
	s_set_vgpr_msb 0x5569
	v_max3_num_f32 v206 /*v462*/, v246 /*v502*/, v2 /*v514*/, v4 /*v516*/
	s_set_vgpr_msb 0x696a
	v_max3_num_f32 v208 /*v464*/, v6 /*v518*/, v150 /*v662*/, v8 /*v520*/
	s_set_vgpr_msb 0x6aaa
	v_max3_num_f32 v11 /*v523*/, v162 /*v674*/, v163 /*v675*/, v164 /*v676*/
	s_set_vgpr_msb 0xaa55
	v_max3_num_f32 v207 /*v463*/, v207 /*v463*/, v209 /*v465*/, v211 /*v467*/
	v_max3_num_f32 v209 /*v465*/, v221 /*v477*/, v223 /*v479*/, v225 /*v481*/
	v_max3_num_f32 v200 /*v456*/, v200 /*v456*/, v202 /*v458*/, v203 /*v459*/
	v_max3_num_f32 v202 /*v458*/, v204 /*v460*/, v205 /*v461*/, v206 /*v462*/
	s_set_vgpr_msb 0x5569
	v_max3_num_f32 v203 /*v459*/, v208 /*v464*/, v10 /*v522*/, v157 /*v669*/
	s_set_vgpr_msb 0x6955
	v_max3_num_f32 v204 /*v460*/, v227 /*v483*/, v229 /*v485*/, v239 /*v495*/
	v_max3_num_f32 v205 /*v461*/, v241 /*v497*/, v243 /*v499*/, v245 /*v501*/
	s_set_vgpr_msb 0x5569
	v_max3_num_f32 v206 /*v462*/, v247 /*v503*/, v3 /*v515*/, v5 /*v517*/
	s_set_vgpr_msb 0x696a
	v_max3_num_f32 v208 /*v464*/, v7 /*v519*/, v158 /*v670*/, v9 /*v521*/
	s_set_vgpr_msb 0x6a55
	v_max3_num_f32 v200 /*v456*/, v200 /*v456*/, v202 /*v458*/, v203 /*v459*/
	v_max3_num_f32 v201 /*v457*/, v201 /*v457*/, v207 /*v463*/, v209 /*v465*/
	v_max3_num_f32 v202 /*v458*/, v204 /*v460*/, v205 /*v461*/, v206 /*v462*/
	s_set_vgpr_msb 0x5569
	v_max3_num_f32 v203 /*v459*/, v208 /*v464*/, v11 /*v523*/, v165 /*v677*/
	s_set_vgpr_msb 0x6955
	v_max3_num_f32 v201 /*v457*/, v201 /*v457*/, v202 /*v458*/, v203 /*v459*/
	v_dual_mov_b32 v204 /*v460*/, v200 /*v456*/ :: v_dual_mov_b32 v202 /*v458*/, v201 /*v457*/
	v_permlanex16_b32 v204 /*v460*/, v204 /*v460*/, s11, 0xfedcba98
	v_permlanex16_b32 v202 /*v458*/, v202 /*v458*/, s11, 0xfedcba98
	v_dual_max_num_f32 v200 /*v456*/, v200 /*v456*/, v204 /*v460*/ :: v_dual_max_num_f32 v201 /*v457*/, v201 /*v457*/, v202 /*v458*/
	s_set_vgpr_msb 0x5549
	v_sub_f32_e32 v203 /*v459*/, v200 /*v456*/, v90 /*v602*/
	v_max_num_f32_e32 v200 /*v456*/, v200 /*v456*/, v90 /*v602*/
	v_sub_f32_e32 v202 /*v458*/, v201 /*v457*/, v71 /*v583*/
	s_set_vgpr_msb 0x4904
	v_cmp_lt_f32_e32 vcc_lo, 0x41000000, v203 /*v459*/
	s_cmp_eq_u32 vcc_lo, 0
	s_cselect_b32 s2, -1, 0
	s_set_vgpr_msb 0x489
	v_cndmask_b32_e64 v88 /*v600*/, v200 /*v456*/, v90 /*v602*/, s2
	s_set_vgpr_msb 0x8946
	v_cmp_lt_f32_e64 s2, 0x41000000, v202 /*v458*/
	v_max_num_f32_e32 v200 /*v456*/, v71 /*v583*/, v201 /*v457*/
	s_cmp_lg_u32 s2, 0
	s_cselect_b32 s5, -1, 0
	s_cmp_eq_u32 s2, 0
	s_cselect_b32 s2, -1, 0
	s_set_vgpr_msb 0x4689
	v_cndmask_b32_e64 v87 /*v599*/, v200 /*v456*/, v71 /*v583*/, s2
	v_mul_f32_e32 v60 /*v572*/, 0xbfb8aa3b, v88 /*v600*/
	v_mul_f32_e32 v70 /*v582*/, 0xbfb8aa3b, v87 /*v599*/
	s_set_vgpr_msb 0x8961
	v_pk_fma_f32 v[202:203] /*v[458:459]*/, v[196:197] /*v[452:453]*/, s[30:31], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[198:199] /*v[454:455]*/, v[198:199] /*v[454:455]*/, s[30:31], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[206:207] /*v[462:463]*/, v[212:213] /*v[468:469]*/, s[30:31], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[208:209] /*v[464:465]*/, v[232:233] /*v[488:489]*/, s[30:31], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x61a2
	v_pk_fma_f32 v[68:69] /*v[580:581]*/, v[68:69] /*v[580:581]*/, s[30:31], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[94:95] /*v[606:607]*/, v[94:95] /*v[606:607]*/, s[30:31], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[66:67] /*v[578:579]*/, v[66:67] /*v[578:579]*/, s[30:31], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa241
	v_exp_f32_e32 v204 /*v460*/, v202 /*v458*/
	v_exp_f32_e32 v210 /*v466*/, v203 /*v459*/
	s_set_vgpr_msb 0x4142
	v_exp_f32_e32 v213 /*v469*/, v68 /*v580*/
	v_exp_f32_e32 v225 /*v481*/, v69 /*v581*/
	v_exp_f32_e32 v229 /*v485*/, v94 /*v606*/
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[68:69] /*v[580:581]*/, v[98:99] /*v[610:611]*/, s[30:31], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa242
	v_exp_f32_e32 v241 /*v497*/, v95 /*v607*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[94:95] /*v[606:607]*/, v[100:101] /*v[612:613]*/, s[30:31], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa261
	v_pk_fma_f32 v[202:203] /*v[458:459]*/, v[214:215] /*v[470:471]*/, s[30:31], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6142
	v_exp_f32_e32 v205 /*v461*/, v66 /*v578*/
	v_exp_f32_e32 v211 /*v467*/, v67 /*v579*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[66:67] /*v[578:579]*/, v[96:97] /*v[608:609]*/, s[30:31], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa261
	v_exp_f32_e32 v212 /*v468*/, v198 /*v454*/
	v_exp_f32_e32 v224 /*v480*/, v199 /*v455*/
	v_exp_f32_e32 v228 /*v484*/, v206 /*v462*/
	v_pk_fma_f32 v[198:199] /*v[454:455]*/, v[216:217] /*v[472:473]*/, s[30:31], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v240 /*v496*/, v207 /*v463*/
	v_nop
	v_pk_fma_f32 v[206:207] /*v[462:463]*/, v[218:219] /*v[474:475]*/, s[30:31], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[216:217] /*v[472:473]*/, v[234:235] /*v[490:491]*/, s[30:31], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x61a2
	v_exp_f32_e32 v11 /*v523*/, v68 /*v580*/
	v_exp_f32_e32 v19 /*v531*/, v69 /*v581*/
	v_exp_f32_e32 v21 /*v533*/, v94 /*v606*/
	v_pk_fma_f32 v[68:69] /*v[580:581]*/, v[104:105] /*v[616:617]*/, s[30:31], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v35 /*v547*/, v95 /*v607*/
	v_nop
	v_pk_fma_f32 v[94:95] /*v[606:607]*/, v[106:107] /*v[618:619]*/, s[30:31], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa241
	v_exp_f32_e32 v246 /*v502*/, v202 /*v458*/
	s_set_vgpr_msb 0x4181
	v_exp_f32_e32 v2 /*v514*/, v203 /*v459*/
	v_nop
	s_set_vgpr_msb 0x8161
	v_pk_fma_f32 v[202:203] /*v[458:459]*/, v[230:231] /*v[486:487]*/, s[30:31], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6142
	v_exp_f32_e32 v247 /*v503*/, v66 /*v578*/
	s_set_vgpr_msb 0x42a2
	v_exp_f32_e32 v3 /*v515*/, v67 /*v579*/
	v_nop
	v_pk_fma_f32 v[66:67] /*v[578:579]*/, v[102:103] /*v[614:615]*/, s[30:31], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa281
	v_exp_f32_e32 v20 /*v532*/, v206 /*v462*/
	v_exp_f32_e32 v34 /*v546*/, v207 /*v463*/
	s_set_vgpr_msb 0x8161
	v_exp_f32_e32 v206 /*v462*/, v208 /*v464*/
	v_exp_f32_e32 v214 /*v470*/, v209 /*v465*/
	v_exp_f32_e32 v218 /*v474*/, v216 /*v472*/
	v_pk_fma_f32 v[208:209] /*v[464:465]*/, v[250:251] /*v[506:507]*/, s[30:31], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v226 /*v482*/, v217 /*v473*/
	v_nop
	v_pk_fma_f32 v[216:217] /*v[472:473]*/, v[252:253] /*v[508:509]*/, s[30:31], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6142
	v_exp_f32_e32 v207 /*v463*/, v68 /*v580*/
	v_exp_f32_e32 v215 /*v471*/, v69 /*v581*/
	v_exp_f32_e32 v219 /*v475*/, v94 /*v606*/
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[68:69] /*v[580:581]*/, v[110:111] /*v[622:623]*/, s[30:31], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa242
	v_exp_f32_e32 v227 /*v483*/, v95 /*v607*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[94:95] /*v[606:607]*/, v[112:113] /*v[624:625]*/, s[30:31], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa281
	v_exp_f32_e32 v10 /*v522*/, v198 /*v454*/
	v_exp_f32_e32 v18 /*v530*/, v199 /*v455*/
	s_set_vgpr_msb 0x8161
	v_exp_f32_e32 v198 /*v454*/, v202 /*v458*/
	v_exp_f32_e32 v202 /*v458*/, v203 /*v459*/
	v_pk_fma_f32 v[220:221] /*v[476:477]*/, v[236:237] /*v[492:493]*/, s[30:31], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6142
	v_exp_f32_e32 v199 /*v455*/, v66 /*v578*/
	v_exp_f32_e32 v203 /*v459*/, v67 /*v579*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[66:67] /*v[578:579]*/, v[108:109] /*v[620:621]*/, s[30:31], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa241
	v_exp_f32_e32 v250 /*v506*/, v208 /*v464*/
	s_set_vgpr_msb 0x4181
	v_exp_f32_e32 v4 /*v516*/, v209 /*v465*/
	v_exp_f32_e32 v12 /*v524*/, v216 /*v472*/
	s_set_vgpr_msb 0x8162
	v_pk_fma_f32 v[208:209] /*v[464:465]*/, v[0:1] /*v[512:513]*/, s[30:31], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6281
	v_exp_f32_e32 v22 /*v534*/, v217 /*v473*/
	v_nop
	s_set_vgpr_msb 0x8162
	v_pk_fma_f32 v[216:217] /*v[472:473]*/, v[38:39] /*v[550:551]*/, s[30:31], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v251 /*v507*/, v68 /*v580*/
	s_set_vgpr_msb 0x62a2
	v_exp_f32_e32 v5 /*v517*/, v69 /*v581*/
	v_exp_f32_e32 v13 /*v525*/, v94 /*v606*/
	v_pk_fma_f32 v[68:69] /*v[580:581]*/, v[116:117] /*v[628:629]*/, s[30:31], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v23 /*v535*/, v95 /*v607*/
	v_nop
	v_pk_fma_f32 v[94:95] /*v[606:607]*/, v[118:119] /*v[630:631]*/, s[30:31], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa261
	v_exp_f32_e32 v230 /*v486*/, v220 /*v476*/
	v_exp_f32_e32 v242 /*v498*/, v221 /*v477*/
	v_nop
	v_pk_fma_f32 v[220:221] /*v[476:477]*/, v[254:255] /*v[510:511]*/, s[30:31], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6142
	v_exp_f32_e32 v231 /*v487*/, v66 /*v578*/
	v_exp_f32_e32 v243 /*v499*/, v67 /*v579*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[66:67] /*v[578:579]*/, v[114:115] /*v[626:627]*/, s[30:31], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa281
	v_exp_f32_e32 v38 /*v550*/, v208 /*v464*/
	v_exp_f32_e32 v48 /*v560*/, v209 /*v465*/
	s_set_vgpr_msb 0x8141
	v_exp_f32_e32 v208 /*v464*/, v216 /*v472*/
	s_set_vgpr_msb 0x4162
	v_pk_fma_f32 v[222:223] /*v[478:479]*/, v[42:43] /*v[554:555]*/, s[30:31], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6241
	v_exp_f32_e32 v216 /*v472*/, v217 /*v473*/
	s_set_vgpr_msb 0x4182
	v_exp_f32_e32 v39 /*v551*/, v68 /*v580*/
	v_exp_f32_e32 v49 /*v561*/, v69 /*v581*/
	s_set_vgpr_msb 0x8242
	v_exp_f32_e32 v209 /*v465*/, v94 /*v606*/
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[68:69] /*v[580:581]*/, v[122:123] /*v[634:635]*/, s[30:31], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa242
	v_exp_f32_e32 v217 /*v473*/, v95 /*v607*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[94:95] /*v[606:607]*/, v[124:125] /*v[636:637]*/, s[30:31], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa281
	v_exp_f32_e32 v28 /*v540*/, v220 /*v476*/
	v_exp_f32_e32 v36 /*v548*/, v221 /*v477*/
	v_nop
	s_set_vgpr_msb 0x8162
	v_pk_fma_f32 v[220:221] /*v[476:477]*/, v[40:41] /*v[552:553]*/, s[30:31], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x62a2
	v_exp_f32_e32 v29 /*v541*/, v66 /*v578*/
	v_exp_f32_e32 v37 /*v549*/, v67 /*v579*/
	v_nop
	v_pk_fma_f32 v[66:67] /*v[578:579]*/, v[120:121] /*v[632:633]*/, s[30:31], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa262
	v_pk_fma_f32 v[234:235] /*v[490:491]*/, v[44:45] /*v[556:557]*/, s[30:31], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6241
	v_exp_f32_e32 v236 /*v492*/, v222 /*v478*/
	s_set_vgpr_msb 0x4162
	v_pk_fma_f32 v[238:239] /*v[494:495]*/, v[50:51] /*v[562:563]*/, s[30:31], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6241
	v_exp_f32_e32 v244 /*v500*/, v223 /*v479*/
	v_nop
	s_set_vgpr_msb 0x4162
	v_pk_fma_f32 v[222:223] /*v[478:479]*/, v[52:53] /*v[564:565]*/, s[30:31], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v237 /*v493*/, v68 /*v580*/
	v_exp_f32_e32 v245 /*v501*/, v69 /*v581*/
	v_exp_f32_e32 v253 /*v509*/, v94 /*v606*/
	s_set_vgpr_msb 0x62a2
	v_pk_fma_f32 v[68:69] /*v[580:581]*/, v[128:129] /*v[640:641]*/, s[30:31], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v7 /*v519*/, v95 /*v607*/
	v_nop
	v_pk_fma_f32 v[94:95] /*v[606:607]*/, v[130:131] /*v[642:643]*/, s[30:31], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa261
	v_pk_fma_f32 v[192:193] /*v[448:449]*/, v[192:193] /*v[448:449]*/, s[30:31], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[200:201] /*v[456:457]*/, v[194:195] /*v[450:451]*/, s[30:31], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v232 /*v488*/, v221 /*v477*/
	s_set_vgpr_msb 0x6162
	v_pk_fma_f32 v[254:255] /*v[510:511]*/, v[136:137] /*v[648:649]*/, s[30:31], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x62a2
	v_pk_fma_f32 v[0:1] /*v[512:513]*/, v[138:139] /*v[650:651]*/, s[30:31], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[136:137] /*v[648:649]*/, v[62:63] /*v[574:575]*/, s[30:31], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[138:139] /*v[650:651]*/, v[64:65] /*v[576:577]*/, s[30:31], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa242
	v_exp_f32_e32 v221 /*v477*/, v66 /*v578*/
	v_exp_f32_e32 v233 /*v489*/, v67 /*v579*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[66:67] /*v[578:579]*/, v[126:127] /*v[638:639]*/, s[30:31], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa241
	v_exp_f32_e32 v252 /*v508*/, v234 /*v490*/
	s_set_vgpr_msb 0x4181
	v_exp_f32_e32 v6 /*v518*/, v235 /*v491*/
	v_exp_f32_e32 v14 /*v526*/, v238 /*v494*/
	s_set_vgpr_msb 0x8162
	v_pk_fma_f32 v[234:235] /*v[490:491]*/, v[54:55] /*v[566:567]*/, s[30:31], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6281
	v_exp_f32_e32 v24 /*v536*/, v239 /*v495*/
	v_exp_f32_e32 v30 /*v542*/, v222 /*v478*/
	s_set_vgpr_msb 0x8162
	v_pk_fma_f32 v[238:239] /*v[494:495]*/, v[56:57] /*v[568:569]*/, s[30:31], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6281
	v_exp_f32_e32 v40 /*v552*/, v223 /*v479*/
	v_nop
	s_set_vgpr_msb 0x8162
	v_pk_fma_f32 v[222:223] /*v[478:479]*/, v[134:135] /*v[646:647]*/, s[30:31], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x62a2
	v_exp_f32_e32 v31 /*v543*/, v68 /*v580*/
	v_exp_f32_e32 v41 /*v553*/, v69 /*v581*/
	v_exp_f32_e32 v45 /*v557*/, v94 /*v606*/
	v_pk_fma_f32 v[68:69] /*v[580:581]*/, v[142:143] /*v[654:655]*/, s[30:31], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v51 /*v563*/, v95 /*v607*/
	v_nop
	v_pk_fma_f32 v[94:95] /*v[606:607]*/, v[144:145] /*v[656:657]*/, s[30:31], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa241
	v_exp_f32_e32 v192 /*v448*/, v192 /*v448*/
	v_exp_f32_e32 v194 /*v450*/, v193 /*v449*/
	v_exp_f32_e32 v196 /*v452*/, v200 /*v456*/
	v_exp_f32_e32 v200 /*v456*/, v201 /*v457*/
	s_set_vgpr_msb 0x4142
	v_exp_f32_e32 v193 /*v449*/, v136 /*v648*/
	v_exp_f32_e32 v195 /*v451*/, v137 /*v649*/
	v_exp_f32_e32 v201 /*v457*/, v139 /*v651*/
	s_set_vgpr_msb 0x42a2
	v_exp_f32_e32 v15 /*v527*/, v66 /*v578*/
	v_exp_f32_e32 v25 /*v537*/, v67 /*v579*/
	v_nop
	v_pk_fma_f32 v[66:67] /*v[578:579]*/, v[132:133] /*v[644:645]*/, s[30:31], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa281
	v_exp_f32_e32 v44 /*v556*/, v234 /*v490*/
	v_exp_f32_e32 v50 /*v562*/, v235 /*v491*/
	v_exp_f32_e32 v52 /*v564*/, v238 /*v494*/
	v_exp_f32_e32 v58 /*v570*/, v239 /*v495*/
	s_set_vgpr_msb 0x8141
	v_exp_f32_e32 v234 /*v490*/, v223 /*v479*/
	v_exp_f32_e32 v238 /*v494*/, v254 /*v510*/
	s_set_vgpr_msb 0x41a2
	v_pk_fma_f32 v[16:17] /*v[528:529]*/, v[140:141] /*v[652:653]*/, s[30:31], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa241
	v_exp_f32_e32 v254 /*v510*/, v255 /*v511*/
	s_set_vgpr_msb 0x41a2
	v_pk_fma_f32 v[32:33] /*v[544:545]*/, v[150:151] /*v[662:663]*/, s[30:31], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa242
	v_exp_f32_e32 v223 /*v479*/, v68 /*v580*/
	v_exp_f32_e32 v235 /*v491*/, v69 /*v581*/
	v_exp_f32_e32 v239 /*v495*/, v94 /*v606*/
	v_exp_f32_e32 v255 /*v511*/, v95 /*v607*/
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[68:69] /*v[580:581]*/, v[148:149] /*v[660:661]*/, s[30:31], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[94:95] /*v[606:607]*/, v[158:159] /*v[670:671]*/, s[30:31], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa242
	v_exp_f32_e32 v197 /*v453*/, v138 /*v650*/
	s_set_vgpr_msb 0x42a2
	v_exp_f32_e32 v53 /*v565*/, v66 /*v578*/
	v_exp_f32_e32 v59 /*v571*/, v67 /*v579*/
	v_nop
	v_pk_fma_f32 v[66:67] /*v[578:579]*/, v[146:147] /*v[658:659]*/, s[30:31], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v26 /*v538*/, v17 /*v529*/
	v_pk_fma_f32 v[56:57] /*v[568:569]*/, v[154:155] /*v[666:667]*/, s[30:31], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v42 /*v554*/, v33 /*v545*/
	v_exp_f32_e32 v17 /*v529*/, v68 /*v580*/
	v_exp_f32_e32 v27 /*v539*/, v69 /*v581*/
	v_exp_f32_e32 v33 /*v545*/, v94 /*v606*/
	v_exp_f32_e32 v43 /*v555*/, v95 /*v607*/
	v_pk_fma_f32 v[68:69] /*v[580:581]*/, v[162:163] /*v[674:675]*/, s[30:31], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa285
	v_pk_add_f32 v[94:95] /*v[606:607]*/, v[192:193] /*v[448:449]*/, v[194:195] /*v[450:451]*/
	v_pk_add_f32 v[96:97] /*v[608:609]*/, v[200:201] /*v[456:457]*/, v[204:205] /*v[460:461]*/
	s_set_vgpr_msb 0x8541
	v_exp_f32_e32 v220 /*v476*/, v220 /*v476*/
	v_exp_f32_e32 v222 /*v478*/, v222 /*v478*/
	s_set_vgpr_msb 0x41a2
	v_exp_f32_e32 v8 /*v520*/, v1 /*v513*/
	v_pk_fma_f32 v[46:47] /*v[558:559]*/, v[152:153] /*v[664:665]*/, s[30:31], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v1 /*v513*/, v66 /*v578*/
	v_exp_f32_e32 v9 /*v521*/, v67 /*v579*/
	v_nop
	v_pk_fma_f32 v[66:67] /*v[578:579]*/, v[160:161] /*v[672:673]*/, s[30:31], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[134:135] /*v[646:647]*/, v[156:157] /*v[668:669]*/, s[30:31], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v60 /*v572*/, v57 /*v569*/
	v_exp_f32_e32 v57 /*v569*/, v68 /*v580*/
	v_exp_f32_e32 v61 /*v573*/, v69 /*v581*/
	v_nop
	s_set_vgpr_msb 0xa289
	v_pk_add_f32 v[68:69] /*v[580:581]*/, v[196:197] /*v[452:453]*/, v[94:95] /*v[606:607]*/
	v_pk_add_f32 v[94:95] /*v[606:607]*/, v[210:211] /*v[466:467]*/, v[96:97] /*v[608:609]*/
	s_set_vgpr_msb 0x8985
	v_pk_add_f32 v[96:97] /*v[608:609]*/, v[212:213] /*v[468:469]*/, v[224:225] /*v[480:481]*/
	v_pk_add_f32 v[98:99] /*v[610:611]*/, v[240:241] /*v[496:497]*/, v[246:247] /*v[502:503]*/
	s_set_vgpr_msb 0x858a
	v_pk_add_f32 v[100:101] /*v[612:613]*/, v[10:11] /*v[522:523]*/, v[18:19] /*v[530:531]*/
	v_pk_add_f32 v[110:111] /*v[622:623]*/, v[22:23] /*v[534:535]*/, v[28:29] /*v[540:541]*/
	v_pk_add_f32 v[112:113] /*v[624:625]*/, v[38:39] /*v[550:551]*/, v[48:49] /*v[560:561]*/
	s_set_vgpr_msb 0x8a85
	v_pk_add_f32 v[116:117] /*v[628:629]*/, v[236:237] /*v[492:493]*/, v[244:245] /*v[500:501]*/
	s_set_vgpr_msb 0x858a
	v_pk_add_f32 v[118:119] /*v[630:631]*/, v[6:7] /*v[518:519]*/, v[14:15] /*v[526:527]*/
	v_exp_f32_e32 v0 /*v512*/, v0 /*v512*/
	v_exp_f32_e32 v16 /*v528*/, v16 /*v528*/
	v_exp_f32_e32 v46 /*v558*/, v46 /*v558*/
	v_exp_f32_e32 v54 /*v566*/, v47 /*v559*/
	v_exp_f32_e32 v47 /*v559*/, v66 /*v578*/
	s_set_vgpr_msb 0x8a86
	v_pk_add_f32 v[102:103] /*v[614:615]*/, v[34:35] /*v[546:547]*/, v[198:199] /*v[454:455]*/
	s_set_vgpr_msb 0x8685
	v_pk_add_f32 v[104:105] /*v[616:617]*/, v[206:207] /*v[462:463]*/, v[214:215] /*v[470:471]*/
	s_set_vgpr_msb 0x8589
	v_pk_add_f32 v[96:97] /*v[608:609]*/, v[228:229] /*v[484:485]*/, v[96:97] /*v[608:609]*/
	s_set_vgpr_msb 0x898a
	v_pk_add_f32 v[98:99] /*v[610:611]*/, v[2:3] /*v[514:515]*/, v[98:99] /*v[610:611]*/
	v_pk_add_f32 v[100:101] /*v[612:613]*/, v[20:21] /*v[532:533]*/, v[100:101] /*v[612:613]*/
	s_set_vgpr_msb 0x8a85
	v_pk_add_f32 v[106:107] /*v[618:619]*/, v[226:227] /*v[482:483]*/, v[230:231] /*v[486:487]*/
	v_pk_add_f32 v[114:115] /*v[626:627]*/, v[216:217] /*v[472:473]*/, v[220:221] /*v[476:477]*/
	s_set_vgpr_msb 0x858a
	v_pk_add_f32 v[110:111] /*v[622:623]*/, v[36:37] /*v[548:549]*/, v[110:111] /*v[622:623]*/
	s_set_vgpr_msb 0x8a89
	v_pk_add_f32 v[112:113] /*v[624:625]*/, v[208:209] /*v[464:465]*/, v[112:113] /*v[624:625]*/
	s_set_vgpr_msb 0x898a
	v_pk_add_f32 v[120:121] /*v[632:633]*/, v[30:31] /*v[542:543]*/, v[40:41] /*v[552:553]*/
	v_pk_add_f32 v[122:123] /*v[634:635]*/, v[50:51] /*v[562:563]*/, v[52:53] /*v[564:565]*/
	s_set_vgpr_msb 0x8a85
	v_pk_add_f32 v[124:125] /*v[636:637]*/, v[222:223] /*v[478:479]*/, v[234:235] /*v[490:491]*/
	s_set_vgpr_msb 0x8589
	v_pk_add_f32 v[116:117] /*v[628:629]*/, v[252:253] /*v[508:509]*/, v[116:117] /*v[628:629]*/
	s_set_vgpr_msb 0x89aa
	v_pk_add_f32 v[118:119] /*v[630:631]*/, v[24:25] /*v[536:537]*/, v[118:119] /*v[630:631]*/
	v_pk_add_f32 v[68:69] /*v[580:581]*/, v[68:69] /*v[580:581]*/, v[94:95] /*v[606:607]*/
	v_exp_f32_e32 v32 /*v544*/, v32 /*v544*/
	v_exp_f32_e32 v56 /*v568*/, v56 /*v568*/
	v_exp_f32_e32 v55 /*v567*/, v67 /*v579*/
	v_nop
	v_pk_fma_f32 v[66:67] /*v[578:579]*/, v[164:165] /*v[676:677]*/, s[30:31], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xaa89
	v_pk_add_f32 v[102:103] /*v[614:615]*/, v[202:203] /*v[458:459]*/, v[102:103] /*v[614:615]*/
	v_pk_add_f32 v[104:105] /*v[616:617]*/, v[218:219] /*v[474:475]*/, v[104:105] /*v[616:617]*/
	v_pk_add_f32 v[108:109] /*v[620:621]*/, v[250:251] /*v[506:507]*/, v[4:5] /*v[516:517]*/
	v_pk_add_f32 v[106:107] /*v[618:619]*/, v[242:243] /*v[498:499]*/, v[106:107] /*v[618:619]*/
	v_pk_add_f32 v[114:115] /*v[626:627]*/, v[232:233] /*v[488:489]*/, v[114:115] /*v[626:627]*/
	s_set_vgpr_msb 0x898a
	v_pk_add_f32 v[120:121] /*v[632:633]*/, v[44:45] /*v[556:557]*/, v[120:121] /*v[632:633]*/
	v_pk_add_f32 v[122:123] /*v[634:635]*/, v[58:59] /*v[570:571]*/, v[122:123] /*v[634:635]*/
	s_set_vgpr_msb 0x8a89
	v_pk_add_f32 v[124:125] /*v[636:637]*/, v[238:239] /*v[494:495]*/, v[124:125] /*v[636:637]*/
	v_pk_add_f32 v[126:127] /*v[638:639]*/, v[254:255] /*v[510:511]*/, v[0:1] /*v[512:513]*/
	s_set_vgpr_msb 0x898a
	v_pk_add_f32 v[128:129] /*v[640:641]*/, v[16:17] /*v[528:529]*/, v[26:27] /*v[538:539]*/
	v_pk_add_f32 v[130:131] /*v[642:643]*/, v[42:43] /*v[554:555]*/, v[46:47] /*v[558:559]*/
	v_pk_add_f32 v[68:69] /*v[580:581]*/, v[96:97] /*v[608:609]*/, v[68:69] /*v[580:581]*/
	v_pk_add_f32 v[96:97] /*v[608:609]*/, v[98:99] /*v[610:611]*/, v[100:101] /*v[612:613]*/
	v_pk_add_f32 v[98:99] /*v[610:611]*/, v[110:111] /*v[622:623]*/, v[112:113] /*v[624:625]*/
	v_pk_add_f32 v[100:101] /*v[612:613]*/, v[116:117] /*v[628:629]*/, v[118:119] /*v[630:631]*/
	v_exp_f32_e32 v62 /*v574*/, v134 /*v646*/
	v_exp_f32_e32 v63 /*v575*/, v66 /*v578*/
	v_pk_add_f32 v[108:109] /*v[620:621]*/, v[12:13] /*v[524:525]*/, v[108:109] /*v[620:621]*/
	v_pk_add_f32 v[132:133] /*v[644:645]*/, v[56:57] /*v[568:569]*/, v[60:61] /*v[572:573]*/
	v_pk_add_f32 v[94:95] /*v[606:607]*/, v[8:9] /*v[520:521]*/, v[126:127] /*v[638:639]*/
	v_pk_add_f32 v[126:127] /*v[638:639]*/, v[32:33] /*v[544:545]*/, v[128:129] /*v[640:641]*/
	v_pk_add_f32 v[128:129] /*v[640:641]*/, v[54:55] /*v[566:567]*/, v[130:131] /*v[642:643]*/
	v_pk_add_f32 v[104:105] /*v[616:617]*/, v[104:105] /*v[616:617]*/, v[106:107] /*v[618:619]*/
	v_pk_add_f32 v[106:107] /*v[618:619]*/, v[122:123] /*v[634:635]*/, v[124:125] /*v[636:637]*/
	v_pk_add_f32 v[96:97] /*v[608:609]*/, v[102:103] /*v[614:615]*/, v[96:97] /*v[608:609]*/
	v_pk_add_f32 v[98:99] /*v[610:611]*/, v[114:115] /*v[626:627]*/, v[98:99] /*v[610:611]*/
	v_pk_add_f32 v[100:101] /*v[612:613]*/, v[120:121] /*v[632:633]*/, v[100:101] /*v[612:613]*/
	v_pk_add_f32 v[130:131] /*v[642:643]*/, v[62:63] /*v[574:575]*/, v[132:133] /*v[644:645]*/
	v_pk_add_f32 v[102:103] /*v[614:615]*/, v[108:109] /*v[620:621]*/, v[104:105] /*v[616:617]*/
	v_pk_add_f32 v[94:95] /*v[606:607]*/, v[94:95] /*v[606:607]*/, v[106:107] /*v[618:619]*/
	v_pk_add_f32 v[104:105] /*v[616:617]*/, v[126:127] /*v[638:639]*/, v[128:129] /*v[640:641]*/
	v_pk_add_f32 v[68:69] /*v[580:581]*/, v[68:69] /*v[580:581]*/, v[96:97] /*v[608:609]*/
	v_pk_add_f32 v[96:97] /*v[608:609]*/, v[98:99] /*v[610:611]*/, v[100:101] /*v[612:613]*/
	v_exp_f32_e32 v64 /*v576*/, v135 /*v647*/
	v_exp_f32_e32 v65 /*v577*/, v67 /*v579*/
	v_nop
	v_pk_add_f32 v[66:67] /*v[578:579]*/, v[130:131] /*v[642:643]*/, v[104:105] /*v[616:617]*/
	v_pk_add_f32 v[68:69] /*v[580:581]*/, v[102:103] /*v[614:615]*/, v[68:69] /*v[580:581]*/
	v_pk_add_f32 v[94:95] /*v[606:607]*/, v[94:95] /*v[606:607]*/, v[96:97] /*v[608:609]*/
	v_sub_f32_e32 v70 /*v582*/, v90 /*v602*/, v88 /*v600*/
	v_pk_add_f32 v[66:67] /*v[578:579]*/, v[64:65] /*v[576:577]*/, v[66:67] /*v[578:579]*/
	v_pk_add_f32 v[68:69] /*v[580:581]*/, v[68:69] /*v[580:581]*/, v[94:95] /*v[606:607]*/
	v_pk_add_f32 v[66:67] /*v[578:579]*/, v[66:67] /*v[578:579]*/, v[68:69] /*v[580:581]*/
	v_mul_f32_e32 v68 /*v580*/, 0x3fb8aa3b, v70 /*v582*/
	s_wait_alu depctr_vm_vsrc(0)
	v_dual_mov_b32 v80 /*v592*/, v66 /*v578*/ :: v_dual_mov_b32 v69 /*v581*/, v67 /*v579*/
	v_exp_f32_e32 v68 /*v580*/, v68 /*v580*/
	v_permlanex16_b32 v80 /*v592*/, v80 /*v592*/, s11, 0xfedcba98
	v_permlanex16_b32 v69 /*v581*/, v69 /*v581*/, s11, 0xfedcba98
	s_set_vgpr_msb 0x8a00
	s_cbranch_vccz .LBB0_52
	s_set_vgpr_msb 8
	v_pk_mul_f32 v[126:127], v[126:127], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[124:125], v[124:125], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[122:123], v[122:123], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[120:121], v[120:121], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[118:119], v[118:119], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[116:117], v[116:117], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[114:115], v[114:115], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[112:113], v[112:113], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[110:111], v[110:111], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[108:109], v[108:109], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[106:107], v[106:107], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[104:105], v[104:105], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[102:103], v[102:103], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[100:101], v[100:101], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[98:99], v[98:99], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[96:97], v[96:97], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[94:95], v[94:95], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[92:93], v[92:93], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[90:91], v[90:91], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[88:89], v[88:89], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[86:87], v[86:87], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[84:85], v[84:85], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[82:83], v[82:83], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[80:81], v[80:81], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[78:79], v[78:79], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[76:77], v[76:77], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[74:75], v[74:75], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[72:73], v[72:73], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[70:71], v[70:71], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[68:69], v[68:69], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[66:67], v[66:67], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[64:65], v[64:65], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x800
.LBB0_52:
	s_set_vgpr_msb 0x8a
	v_sub_f32_e32 v70 /*v582*/, v71 /*v583*/, v87 /*v599*/
	s_and_b32 s2, s5, exec_lo
	s_cselect_b32 s2, 1, 0
	s_cmp_lg_u32 s2, 1
	v_mul_f32_e32 v70 /*v582*/, 0x3fb8aa3b, v70 /*v582*/
	v_exp_f32_e32 v70 /*v582*/, v70 /*v582*/
	s_set_vgpr_msb 0x8a00
	s_cbranch_scc1 .LBB0_47
	v_nop
	s_set_vgpr_msb 8
	v_pk_mul_f32 v[62:63], v[62:63], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[60:61], v[60:61], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59], v[58:59], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[56:57], v[56:57], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[54:55], v[54:55], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[52:53], v[52:53], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[50:51], v[50:51], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[48:49], v[48:49], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[46:47], v[46:47], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[44:45], v[44:45], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[42:43], v[42:43], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[40:41], v[40:41], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39], v[38:39], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[36:37], v[36:37], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35], v[34:35], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[32:33], v[32:33], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[30:31], v[30:31], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[28:29], v[28:29], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[26:27], v[26:27], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25], v[24:25], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[22:23], v[22:23], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21], v[20:21], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[18:19], v[18:19], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[16:17], v[16:17], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[14:15], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13], v[12:13], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[10:11], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[8:9], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[6:7], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[4:5], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[2:3], v[2:3], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[0:1], v[0:1], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x800
	s_branch .LBB0_47
.LBB0_54:
	s_set_vgpr_msb 0x82
	v_dual_mov_b32 v91 /*v603*/, v89 /*v601*/ :: v_dual_mov_b32 v92 /*v604*/, v80 /*v592*/
	v_dual_mov_b32 v87 /*v599*/, v71 /*v583*/ :: v_dual_mov_b32 v88 /*v600*/, v90 /*v602*/
	s_set_vgpr_msb 0x8200
.LBB0_55:
	s_cmp_ge_u32 s48, s10
	s_cbranch_scc1 .LBB0_64
	s_add_co_i32 s2, s19, -1
	s_mov_b32 s27, 0
	s_wait_alu depctr_vm_vsrc(0)
	s_set_vgpr_msb 0x88
	v_min_i32_e32 v80 /*v592*/, s2, v85 /*v597*/
	v_min_i32_e32 v85 /*v597*/, s2, v86 /*v598*/
	s_mov_b32 s11, s27
	s_mov_b32 s28, 1
	s_sub_co_i32 s7, 1, s60
	s_mov_b32 s24, 32
	s_mov_b32 s23, 0x800000
	s_mov_b32 s21, 0xffff0000
	s_mov_b32 s20, 0x7510000
	s_mov_b32 s36, 0xf510000
	s_mov_b32 s17, 0x76543210
	s_mov_b32 s16, 0x3fb8aa3b
	s_set_vgpr_msb 0x8800
	s_branch .LBB0_58
.LBB0_57:
	s_set_vgpr_msb 0x8a
	v_cvt_pk_bf16_f32 v99 /*v611*/, v14 /*v526*/, v26 /*v538*/
	s_set_vgpr_msb 0x8a89
	v_cvt_pk_bf16_f32 v98 /*v610*/, v254 /*v510*/, v10 /*v522*/
	s_set_vgpr_msb 0x8985
	v_cvt_pk_bf16_f32 v97 /*v609*/, v236 /*v492*/, v250 /*v506*/
	v_cvt_pk_bf16_f32 v96 /*v608*/, v222 /*v478*/, v232 /*v488*/
	v_cvt_pk_bf16_f32 v95 /*v607*/, v210 /*v466*/, v218 /*v474*/
	v_cvt_pk_bf16_f32 v94 /*v606*/, v202 /*v458*/, v208 /*v464*/
	v_cvt_pk_bf16_f32 v93 /*v605*/, v196 /*v452*/, v200 /*v456*/
	v_cvt_pk_bf16_f32 v92 /*v604*/, v192 /*v448*/, v194 /*v450*/
	s_set_vgpr_msb 0x858a
	v_cvt_pk_bf16_f32 v107 /*v619*/, v15 /*v527*/, v27 /*v539*/
	s_set_vgpr_msb 0x8a89
	v_cvt_pk_bf16_f32 v106 /*v618*/, v255 /*v511*/, v11 /*v523*/
	s_set_vgpr_msb 0x8985
	v_cvt_pk_bf16_f32 v105 /*v617*/, v237 /*v493*/, v251 /*v507*/
	v_cvt_pk_bf16_f32 v104 /*v616*/, v223 /*v479*/, v233 /*v489*/
	v_cvt_pk_bf16_f32 v103 /*v615*/, v211 /*v467*/, v219 /*v475*/
	v_cvt_pk_bf16_f32 v102 /*v614*/, v203 /*v459*/, v209 /*v465*/
	v_cvt_pk_bf16_f32 v101 /*v613*/, v197 /*v453*/, v201 /*v457*/
	v_cvt_pk_bf16_f32 v100 /*v612*/, v193 /*v449*/, v195 /*v451*/
	s_set_vgpr_msb 0x8509
	s_wait_dscnt 0x3d
	v_wmma_f32_16x16x32_bf16 v[120:127], v[184:191] /*v[440:447]*/, v[92:99] /*v[604:611]*/, v[120:127]
	s_set_vgpr_msb 0x98a
	v_cvt_pk_bf16_f32 v115 /*v627*/, v37 /*v549*/, v45 /*v557*/
	v_cvt_pk_bf16_f32 v114 /*v626*/, v23 /*v535*/, v33 /*v545*/
	v_cvt_pk_bf16_f32 v113 /*v625*/, v7 /*v519*/, v19 /*v531*/
	s_set_vgpr_msb 0x8a89
	v_cvt_pk_bf16_f32 v112 /*v624*/, v245 /*v501*/, v3 /*v515*/
	s_set_vgpr_msb 0x8985
	v_cvt_pk_bf16_f32 v111 /*v623*/, v229 /*v485*/, v241 /*v497*/
	v_cvt_pk_bf16_f32 v110 /*v622*/, v217 /*v473*/, v227 /*v483*/
	v_cvt_pk_bf16_f32 v109 /*v621*/, v207 /*v463*/, v215 /*v471*/
	s_set_vgpr_msb 0x8509
	v_wmma_f32_16x16x32_bf16 v[56:63], v[184:191] /*v[440:447]*/, v[100:107] /*v[612:619]*/, v[56:63]
	s_set_vgpr_msb 0x985
	v_cvt_pk_bf16_f32 v108 /*v620*/, v199 /*v455*/, v205 /*v461*/
	s_set_vgpr_msb 0x854a
	v_cvt_pk_bf16_f32 v199 /*v455*/, v53 /*v565*/, v59 /*v571*/
	v_cvt_pk_bf16_f32 v197 /*v453*/, v31 /*v543*/, v41 /*v553*/
	v_cvt_pk_bf16_f32 v196 /*v452*/, v17 /*v529*/, v29 /*v541*/
	v_cvt_pk_bf16_f32 v191 /*v447*/, v36 /*v548*/, v44 /*v556*/
	v_cvt_pk_bf16_f32 v190 /*v446*/, v22 /*v534*/, v32 /*v544*/
	v_cvt_pk_bf16_f32 v189 /*v445*/, v6 /*v518*/, v18 /*v530*/
	s_set_vgpr_msb 0x4a09
	s_wait_dscnt 0x3c
	v_wmma_f32_16x16x32_bf16 v[112:119], v[152:159] /*v[408:415]*/, v[92:99] /*v[604:611]*/, v[112:119]
	s_set_vgpr_msb 0x949
	v_cvt_pk_bf16_f32 v188 /*v444*/, v244 /*v500*/, v2 /*v514*/
	s_set_vgpr_msb 0x4945
	v_cvt_pk_bf16_f32 v187 /*v443*/, v228 /*v484*/, v240 /*v496*/
	v_cvt_pk_bf16_f32 v186 /*v442*/, v216 /*v472*/, v226 /*v482*/
	v_cvt_pk_bf16_f32 v185 /*v441*/, v206 /*v462*/, v214 /*v470*/
	v_cvt_pk_bf16_f32 v184 /*v440*/, v198 /*v454*/, v204 /*v460*/
	s_set_vgpr_msb 0x454a
	v_cvt_pk_bf16_f32 v198 /*v454*/, v43 /*v555*/, v51 /*v563*/
	v_cvt_pk_bf16_f32 v195 /*v451*/, v1 /*v513*/, v13 /*v525*/
	s_set_vgpr_msb 0x4a09
	v_wmma_f32_16x16x32_bf16 v[48:55], v[152:159] /*v[408:415]*/, v[100:107] /*v[612:619]*/, v[48:55]
	s_set_vgpr_msb 0x945
	v_cvt_pk_bf16_f32 v194 /*v450*/, v239 /*v495*/, v253 /*v509*/
	v_cvt_pk_bf16_f32 v193 /*v449*/, v225 /*v481*/, v235 /*v491*/
	v_cvt_pk_bf16_f32 v192 /*v448*/, v213 /*v469*/, v221 /*v477*/
	s_set_vgpr_msb 0x454a
	v_cvt_pk_bf16_f32 v207 /*v463*/, v63 /*v575*/, v65 /*v577*/
	v_cvt_pk_bf16_f32 v206 /*v462*/, v57 /*v569*/, v61 /*v573*/
	v_cvt_pk_bf16_f32 v205 /*v461*/, v49 /*v561*/, v55 /*v567*/
	v_cvt_pk_bf16_f32 v204 /*v460*/, v39 /*v551*/, v47 /*v559*/
	s_set_vgpr_msb 0x4a09
	s_wait_dscnt 0x2d
	v_wmma_f32_16x16x32_bf16 v[104:111], v[120:127] /*v[376:383]*/, v[92:99] /*v[604:611]*/, v[104:111]
	s_set_vgpr_msb 0x94a
	v_cvt_pk_bf16_f32 v203 /*v459*/, v25 /*v537*/, v35 /*v547*/
	v_cvt_pk_bf16_f32 v202 /*v458*/, v9 /*v521*/, v21 /*v533*/
	s_set_vgpr_msb 0x4a49
	v_cvt_pk_bf16_f32 v201 /*v457*/, v247 /*v503*/, v5 /*v517*/
	s_set_vgpr_msb 0x4945
	v_cvt_pk_bf16_f32 v200 /*v456*/, v231 /*v487*/, v243 /*v499*/
	s_set_vgpr_msb 0x4586
	v_dual_fmac_f32 v88 /*v600*/, v68 /*v580*/, v248 /*v504*/ :: v_dual_fmac_f32 v90 /*v602*/, v70 /*v582*/, v249 /*v505*/
	s_add_nc_u64 s[48:49], s[48:49], 1
	s_set_vgpr_msb 0x8609
	v_wmma_f32_16x16x32_bf16 v[40:47], v[120:127] /*v[376:383]*/, v[100:107] /*v[612:619]*/, v[40:47]
	v_cmp_ge_u64_e64 s2, s[48:49], s[10:11]
	s_set_vgpr_msb 0x94a
	v_dual_add_f32 v248 /*v504*/, v88 /*v600*/, v66 /*v578*/ :: v_dual_add_f32 v249 /*v505*/, v90 /*v602*/, v67 /*v579*/
	s_wait_alu depctr_vm_vsrc(0)
	s_set_vgpr_msb 0x4a82
	v_dual_mov_b32 v91 /*v603*/, v81 /*v593*/ :: v_dual_mov_b32 v81 /*v593*/, v89 /*v601*/
	v_dual_mov_b32 v87 /*v599*/, v71 /*v583*/ :: v_dual_mov_b32 v88 /*v600*/, v69 /*v581*/
	s_set_vgpr_msb 0x8209
	s_wait_dscnt 0x2c
	v_wmma_f32_16x16x32_bf16 v[96:103], v[88:95] /*v[344:351]*/, v[92:99] /*v[604:611]*/, v[96:103]
	s_and_b32 vcc_lo, exec_lo, s2
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[32:39], v[88:95] /*v[344:351]*/, v[100:107] /*v[612:619]*/, v[32:39]
	v_wmma_f32_16x16x32_bf16 v[88:95], v[56:63] /*v[312:319]*/, v[92:99] /*v[604:611]*/, v[88:95]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[56:63] /*v[312:319]*/, v[100:107] /*v[612:619]*/, v[24:31]
	v_wmma_f32_16x16x32_bf16 v[80:87], v[24:31] /*v[280:287]*/, v[92:99] /*v[604:611]*/, v[80:87]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[24:31] /*v[280:287]*/, v[100:107] /*v[612:619]*/, v[16:23]
	s_set_vgpr_msb 0x908
	v_wmma_f32_16x16x32_bf16 v[72:79], v[248:255], v[92:99] /*v[604:611]*/, v[72:79]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[248:255], v[100:107] /*v[612:619]*/, v[8:15]
	v_wmma_f32_16x16x32_bf16 v[64:71], v[216:223], v[92:99] /*v[604:611]*/, v[64:71]
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x882
	v_dual_mov_b32 v92 /*v604*/, v84 /*v596*/ :: v_dual_mov_b32 v84 /*v596*/, v86 /*v598*/
	s_set_vgpr_msb 0x8208
	v_wmma_f32_16x16x32_bf16 v[0:7], v[216:223], v[100:107] /*v[612:619]*/, v[0:7]
	s_set_vgpr_msb 0x805
	v_wmma_f32_16x16x32_bf16 v[120:127], v[176:183] /*v[432:439]*/, v[184:191] /*v[440:447]*/, v[120:127]
	s_set_vgpr_msb 0x509
	v_wmma_f32_16x16x32_bf16 v[56:63], v[176:183] /*v[432:439]*/, v[108:115] /*v[620:627]*/, v[56:63]
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x94a
	v_cvt_pk_bf16_f32 v183 /*v439*/, v52 /*v564*/, v58 /*v570*/
	v_cvt_pk_bf16_f32 v182 /*v438*/, v42 /*v554*/, v50 /*v562*/
	v_cvt_pk_bf16_f32 v181 /*v437*/, v30 /*v542*/, v40 /*v552*/
	s_set_vgpr_msb 0x4a05
	v_wmma_f32_16x16x32_bf16 v[112:119], v[144:151] /*v[400:407]*/, v[184:191] /*v[440:447]*/, v[112:119]
	s_set_vgpr_msb 0x54a
	v_cvt_pk_bf16_f32 v180 /*v436*/, v16 /*v528*/, v28 /*v540*/
	v_cvt_pk_bf16_f32 v179 /*v435*/, v0 /*v512*/, v12 /*v524*/
	s_set_vgpr_msb 0x4a45
	v_cvt_pk_bf16_f32 v178 /*v434*/, v238 /*v494*/, v252 /*v508*/
	v_cvt_pk_bf16_f32 v177 /*v433*/, v224 /*v480*/, v234 /*v490*/
	v_cvt_pk_bf16_f32 v176 /*v432*/, v212 /*v468*/, v220 /*v476*/
	s_set_vgpr_msb 0x4509
	v_wmma_f32_16x16x32_bf16 v[48:55], v[144:151] /*v[400:407]*/, v[108:115] /*v[620:627]*/, v[48:55]
	s_set_vgpr_msb 0x905
	v_wmma_f32_16x16x32_bf16 v[104:111], v[112:119] /*v[368:375]*/, v[184:191] /*v[440:447]*/, v[104:111]
	s_set_vgpr_msb 0x509
	v_wmma_f32_16x16x32_bf16 v[40:47], v[112:119] /*v[368:375]*/, v[108:115] /*v[620:627]*/, v[40:47]
	s_set_vgpr_msb 0x905
	v_wmma_f32_16x16x32_bf16 v[96:103], v[80:87] /*v[336:343]*/, v[184:191] /*v[440:447]*/, v[96:103]
	s_set_vgpr_msb 0x509
	v_wmma_f32_16x16x32_bf16 v[32:39], v[80:87] /*v[336:343]*/, v[108:115] /*v[620:627]*/, v[32:39]
	s_set_vgpr_msb 0x905
	v_wmma_f32_16x16x32_bf16 v[88:95], v[48:55] /*v[304:311]*/, v[184:191] /*v[440:447]*/, v[88:95]
	s_set_vgpr_msb 0x509
	v_wmma_f32_16x16x32_bf16 v[24:31], v[48:55] /*v[304:311]*/, v[108:115] /*v[620:627]*/, v[24:31]
	s_set_vgpr_msb 0x905
	v_wmma_f32_16x16x32_bf16 v[80:87], v[16:23] /*v[272:279]*/, v[184:191] /*v[440:447]*/, v[80:87]
	s_set_vgpr_msb 0x509
	v_wmma_f32_16x16x32_bf16 v[16:23], v[16:23] /*v[272:279]*/, v[108:115] /*v[620:627]*/, v[16:23]
	s_set_vgpr_msb 0x904
	v_wmma_f32_16x16x32_bf16 v[72:79], v[240:247], v[184:191] /*v[440:447]*/, v[72:79]
	s_set_vgpr_msb 0x408
	v_wmma_f32_16x16x32_bf16 v[8:15], v[240:247], v[108:115] /*v[620:627]*/, v[8:15]
	s_set_vgpr_msb 0x804
	v_wmma_f32_16x16x32_bf16 v[64:71], v[208:215], v[184:191] /*v[440:447]*/, v[64:71]
	s_set_vgpr_msb 0x408
	v_wmma_f32_16x16x32_bf16 v[0:7], v[208:215], v[108:115] /*v[620:627]*/, v[0:7]
	s_set_vgpr_msb 0x805
	v_wmma_f32_16x16x32_bf16 v[120:127], v[168:175] /*v[424:431]*/, v[176:183] /*v[432:439]*/, v[120:127]
	v_wmma_f32_16x16x32_bf16 v[56:63], v[168:175] /*v[424:431]*/, v[192:199] /*v[448:455]*/, v[56:63]
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x54a
	v_cvt_pk_bf16_f32 v175 /*v431*/, v62 /*v574*/, v64 /*v576*/
	v_cvt_pk_bf16_f32 v174 /*v430*/, v56 /*v568*/, v60 /*v572*/
	v_cvt_pk_bf16_f32 v173 /*v429*/, v48 /*v560*/, v54 /*v566*/
	s_set_vgpr_msb 0x4a05
	v_wmma_f32_16x16x32_bf16 v[112:119], v[136:143] /*v[392:399]*/, v[176:183] /*v[432:439]*/, v[112:119]
	s_set_vgpr_msb 0x54a
	v_cvt_pk_bf16_f32 v172 /*v428*/, v38 /*v550*/, v46 /*v558*/
	v_cvt_pk_bf16_f32 v171 /*v427*/, v24 /*v536*/, v34 /*v546*/
	v_cvt_pk_bf16_f32 v170 /*v426*/, v8 /*v520*/, v20 /*v532*/
	s_set_vgpr_msb 0x4a49
	v_cvt_pk_bf16_f32 v169 /*v425*/, v246 /*v502*/, v4 /*v516*/
	s_set_vgpr_msb 0x4945
	v_cvt_pk_bf16_f32 v168 /*v424*/, v230 /*v486*/, v242 /*v498*/
	s_set_vgpr_msb 0x4505
	v_wmma_f32_16x16x32_bf16 v[48:55], v[136:143] /*v[392:399]*/, v[192:199] /*v[448:455]*/, v[48:55]
	v_wmma_f32_16x16x32_bf16 v[104:111], v[104:111] /*v[360:367]*/, v[176:183] /*v[432:439]*/, v[104:111]
	v_wmma_f32_16x16x32_bf16 v[40:47], v[104:111] /*v[360:367]*/, v[192:199] /*v[448:455]*/, v[40:47]
	v_wmma_f32_16x16x32_bf16 v[96:103], v[72:79] /*v[328:335]*/, v[176:183] /*v[432:439]*/, v[96:103]
	v_wmma_f32_16x16x32_bf16 v[32:39], v[72:79] /*v[328:335]*/, v[192:199] /*v[448:455]*/, v[32:39]
	v_wmma_f32_16x16x32_bf16 v[88:95], v[40:47] /*v[296:303]*/, v[176:183] /*v[432:439]*/, v[88:95]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[40:47] /*v[296:303]*/, v[192:199] /*v[448:455]*/, v[24:31]
	v_wmma_f32_16x16x32_bf16 v[80:87], v[8:15] /*v[264:271]*/, v[176:183] /*v[432:439]*/, v[80:87]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[8:15] /*v[264:271]*/, v[192:199] /*v[448:455]*/, v[16:23]
	s_set_vgpr_msb 0x504
	v_wmma_f32_16x16x32_bf16 v[72:79], v[232:239], v[176:183] /*v[432:439]*/, v[72:79]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[232:239], v[192:199] /*v[448:455]*/, v[8:15]
	v_wmma_f32_16x16x32_bf16 v[64:71], v[200:207], v[176:183] /*v[432:439]*/, v[64:71]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[200:207], v[192:199] /*v[448:455]*/, v[0:7]
	s_set_vgpr_msb 0x405
	v_wmma_f32_16x16x32_bf16 v[120:127], v[160:167] /*v[416:423]*/, v[168:175] /*v[424:431]*/, v[120:127]
	v_wmma_f32_16x16x32_bf16 v[56:63], v[160:167] /*v[416:423]*/, v[200:207] /*v[456:463]*/, v[56:63]
	v_wmma_f32_16x16x32_bf16 v[112:119], v[128:135] /*v[384:391]*/, v[168:175] /*v[424:431]*/, v[112:119]
	v_wmma_f32_16x16x32_bf16 v[48:55], v[128:135] /*v[384:391]*/, v[200:207] /*v[456:463]*/, v[48:55]
	v_wmma_f32_16x16x32_bf16 v[104:111], v[96:103] /*v[352:359]*/, v[168:175] /*v[424:431]*/, v[104:111]
	v_wmma_f32_16x16x32_bf16 v[40:47], v[96:103] /*v[352:359]*/, v[200:207] /*v[456:463]*/, v[40:47]
	v_wmma_f32_16x16x32_bf16 v[96:103], v[64:71] /*v[320:327]*/, v[168:175] /*v[424:431]*/, v[96:103]
	v_wmma_f32_16x16x32_bf16 v[32:39], v[64:71] /*v[320:327]*/, v[200:207] /*v[456:463]*/, v[32:39]
	v_wmma_f32_16x16x32_bf16 v[88:95], v[32:39] /*v[288:295]*/, v[168:175] /*v[424:431]*/, v[88:95]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[32:39] /*v[288:295]*/, v[200:207] /*v[456:463]*/, v[24:31]
	v_wmma_f32_16x16x32_bf16 v[80:87], v[0:7] /*v[256:263]*/, v[168:175] /*v[424:431]*/, v[80:87]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[0:7] /*v[256:263]*/, v[200:207] /*v[456:463]*/, v[16:23]
	s_set_vgpr_msb 0x504
	v_wmma_f32_16x16x32_bf16 v[72:79], v[224:231], v[168:175] /*v[424:431]*/, v[72:79]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[224:231], v[200:207] /*v[456:463]*/, v[8:15]
	v_wmma_f32_16x16x32_bf16 v[64:71], v[192:199], v[168:175] /*v[424:431]*/, v[64:71]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[192:199], v[200:207] /*v[456:463]*/, v[0:7]
	s_set_vgpr_msb 0x400
	s_cbranch_vccnz .LBB0_65
.LBB0_58:
	s_set_vgpr_msb 0x82
	v_dual_mov_b32 v86 /*v598*/, v92 /*v604*/ :: v_dual_mov_b32 v89 /*v601*/, v91 /*v603*/
	s_add_co_i32 s2, s48, 1
	s_wait_tensorcnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x8242
	ds_load_b128 v[184:187] /*v[440:443]*/, v84 /*v596*/
	ds_load_b128 v[188:191] /*v[444:447]*/, v84 /*v596*/ offset:32
	ds_load_b128 v[176:179] /*v[432:435]*/, v84 /*v596*/ offset:64
	ds_load_b128 v[180:183] /*v[436:439]*/, v84 /*v596*/ offset:96
	ds_load_b128 v[168:171] /*v[424:427]*/, v84 /*v596*/ offset:128
	ds_load_b128 v[172:175] /*v[428:431]*/, v84 /*v596*/ offset:160
	ds_load_b128 v[160:163] /*v[416:419]*/, v84 /*v596*/ offset:192
	ds_load_b128 v[164:167] /*v[420:423]*/, v84 /*v596*/ offset:224
	ds_load_b128 v[152:155] /*v[408:411]*/, v84 /*v596*/ offset:4352
	ds_load_b128 v[156:159] /*v[412:415]*/, v84 /*v596*/ offset:4384
	ds_load_b128 v[144:147] /*v[400:403]*/, v84 /*v596*/ offset:4416
	ds_load_b128 v[148:151] /*v[404:407]*/, v84 /*v596*/ offset:4448
	ds_load_b128 v[136:139] /*v[392:395]*/, v84 /*v596*/ offset:4480
	ds_load_b128 v[140:143] /*v[396:399]*/, v84 /*v596*/ offset:4512
	ds_load_b128 v[128:131] /*v[384:387]*/, v84 /*v596*/ offset:4544
	ds_load_b128 v[132:135] /*v[388:391]*/, v84 /*v596*/ offset:4576
	ds_load_b128 v[120:123] /*v[376:379]*/, v84 /*v596*/ offset:8704
	ds_load_b128 v[124:127] /*v[380:383]*/, v84 /*v596*/ offset:8736
	ds_load_b128 v[112:115] /*v[368:371]*/, v84 /*v596*/ offset:8768
	ds_load_b128 v[116:119] /*v[372:375]*/, v84 /*v596*/ offset:8800
	ds_load_b128 v[104:107] /*v[360:363]*/, v84 /*v596*/ offset:8832
	ds_load_b128 v[108:111] /*v[364:367]*/, v84 /*v596*/ offset:8864
	ds_load_b128 v[96:99] /*v[352:355]*/, v84 /*v596*/ offset:8896
	ds_load_b128 v[100:103] /*v[356:359]*/, v84 /*v596*/ offset:8928
	ds_load_b128 v[88:91] /*v[344:347]*/, v84 /*v596*/ offset:13056
	ds_load_b128 v[92:95] /*v[348:351]*/, v84 /*v596*/ offset:13088
	ds_load_b128 v[80:83] /*v[336:339]*/, v84 /*v596*/ offset:13120
	ds_load_b128 v[84:87] /*v[340:343]*/, v84 /*v596*/ offset:13152
	ds_load_b128 v[72:75] /*v[328:331]*/, v84 /*v596*/ offset:13184
	ds_load_b128 v[76:79] /*v[332:335]*/, v84 /*v596*/ offset:13216
	ds_load_b128 v[64:67] /*v[320:323]*/, v84 /*v596*/ offset:13248
	ds_load_b128 v[68:71] /*v[324:327]*/, v84 /*v596*/ offset:13280
	ds_load_b128 v[56:59] /*v[312:315]*/, v84 /*v596*/ offset:17408
	ds_load_b128 v[60:63] /*v[316:319]*/, v84 /*v596*/ offset:17440
	ds_load_b128 v[48:51] /*v[304:307]*/, v84 /*v596*/ offset:17472
	ds_load_b128 v[52:55] /*v[308:311]*/, v84 /*v596*/ offset:17504
	ds_load_b128 v[40:43] /*v[296:299]*/, v84 /*v596*/ offset:17536
	ds_load_b128 v[44:47] /*v[300:303]*/, v84 /*v596*/ offset:17568
	ds_load_b128 v[32:35] /*v[288:291]*/, v84 /*v596*/ offset:17600
	ds_load_b128 v[36:39] /*v[292:295]*/, v84 /*v596*/ offset:17632
	ds_load_b128 v[24:27] /*v[280:283]*/, v84 /*v596*/ offset:21760
	ds_load_b128 v[28:31] /*v[284:287]*/, v84 /*v596*/ offset:21792
	ds_load_b128 v[16:19] /*v[272:275]*/, v84 /*v596*/ offset:21824
	ds_load_b128 v[20:23] /*v[276:279]*/, v84 /*v596*/ offset:21856
	ds_load_b128 v[8:11] /*v[264:267]*/, v84 /*v596*/ offset:21888
	ds_load_b128 v[12:15] /*v[268:271]*/, v84 /*v596*/ offset:21920
	ds_load_b128 v[0:3] /*v[256:259]*/, v84 /*v596*/ offset:21952
	ds_load_b128 v[4:7] /*v[260:263]*/, v84 /*v596*/ offset:21984
	s_set_vgpr_msb 0x4202
	ds_load_b128 v[248:251], v84 /*v596*/ offset:26112
	ds_load_b128 v[252:255], v84 /*v596*/ offset:26144
	ds_load_b128 v[240:243], v84 /*v596*/ offset:26176
	ds_load_b128 v[244:247], v84 /*v596*/ offset:26208
	ds_load_b128 v[232:235], v84 /*v596*/ offset:26240
	ds_load_b128 v[236:239], v84 /*v596*/ offset:26272
	ds_load_b128 v[224:227], v84 /*v596*/ offset:26304
	ds_load_b128 v[228:231], v84 /*v596*/ offset:26336
	ds_load_b128 v[216:219], v84 /*v596*/ offset:30464
	ds_load_b128 v[220:223], v84 /*v596*/ offset:30496
	ds_load_b128 v[208:211], v84 /*v596*/ offset:30528
	ds_load_b128 v[212:215], v84 /*v596*/ offset:30560
	ds_load_b128 v[200:203], v84 /*v596*/ offset:30592
	ds_load_b128 v[204:207], v84 /*v596*/ offset:30624
	ds_load_b128 v[192:195], v84 /*v596*/ offset:30656
	ds_load_b128 v[196:199], v84 /*v596*/ offset:30688
	s_cmp_ge_i32 s2, s10
	s_set_vgpr_msb 0x200
	s_cbranch_scc1 .LBB0_60
	s_lshl_b32 s4, s2, 7
	s_add_co_i32 s6, s48, s7
	s_sub_co_i32 s22, s9, s4
	s_add_co_i32 s2, s4, s97
	s_set_vgpr_msb 0x41
	v_med3_i32 v192 /*v448*/, s22, 0, 0x80
	s_lshr_b32 s19, s6, 31
	s_ashr_i32 s3, s2, 31
	s_add_co_i32 s19, s6, s19
	s_mul_u64 s[4:5], s[2:3], s[64:65]
	v_readfirstlane_b32 s22, v192 /*v448*/
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
	s_wait_dscnt 0x3e
	v_wmma_f32_16x16x32_bf16 v[250:257] /*v[506:513]*/, v[184:191] /*v[440:447]*/, v[128:135], 0
	v_wmma_f32_16x16x32_bf16 v[192:199] /*v[448:455]*/, v[184:191] /*v[440:447]*/, v[160:167], 0
	s_set_vgpr_msb 0x4181
	s_wait_dscnt 0x36
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[514:521]*/, v[152:159] /*v[408:415]*/, v[128:135], 0
	s_set_vgpr_msb 0x8141
	v_wmma_f32_16x16x32_bf16 v[200:207] /*v[456:463]*/, v[152:159] /*v[408:415]*/, v[160:167], 0
	s_set_vgpr_msb 0x4181
	s_wait_dscnt 0x2e
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[522:529]*/, v[120:127] /*v[376:383]*/, v[128:135], 0
	s_set_vgpr_msb 0x8141
	v_wmma_f32_16x16x32_bf16 v[208:215] /*v[464:471]*/, v[120:127] /*v[376:383]*/, v[160:167], 0
	s_set_vgpr_msb 0x4181
	s_wait_dscnt 0x26
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[530:537]*/, v[88:95] /*v[344:351]*/, v[128:135], 0
	s_set_vgpr_msb 0x8141
	v_wmma_f32_16x16x32_bf16 v[216:223] /*v[472:479]*/, v[88:95] /*v[344:351]*/, v[160:167], 0
	s_set_vgpr_msb 0x4181
	s_wait_dscnt 0x1e
	v_wmma_f32_16x16x32_bf16 v[26:33] /*v[538:545]*/, v[56:63] /*v[312:319]*/, v[128:135], 0
	s_set_vgpr_msb 0x8141
	v_wmma_f32_16x16x32_bf16 v[224:231] /*v[480:487]*/, v[56:63] /*v[312:319]*/, v[160:167], 0
	s_set_vgpr_msb 0x4181
	s_wait_dscnt 0x16
	v_wmma_f32_16x16x32_bf16 v[34:41] /*v[546:553]*/, v[24:31] /*v[280:287]*/, v[128:135], 0
	s_set_vgpr_msb 0x8141
	v_wmma_f32_16x16x32_bf16 v[232:239] /*v[488:495]*/, v[24:31] /*v[280:287]*/, v[160:167], 0
	s_set_vgpr_msb 0x4180
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[554:561]*/, v[248:255], v[128:135], 0
	s_set_vgpr_msb 0x8040
	v_wmma_f32_16x16x32_bf16 v[240:247] /*v[496:503]*/, v[248:255], v[160:167], 0
	s_set_vgpr_msb 0x4080
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[562:569]*/, v[216:223], v[128:135], 0
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[570:577]*/, v[216:223], v[160:167], 0
	s_set_vgpr_msb 0x8051
	v_wmma_f32_16x16x32_bf16 v[250:257] /*v[506:513]*/, v[176:183] /*v[432:439]*/, v[136:143], v[250:257] /*v[506:513]*/
	v_wmma_f32_16x16x32_bf16 v[192:199] /*v[448:455]*/, v[176:183] /*v[432:439]*/, v[168:175], v[192:199] /*v[448:455]*/
	s_set_vgpr_msb 0x51a1
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[514:521]*/, v[144:151] /*v[400:407]*/, v[136:143], v[2:9] /*v[514:521]*/
	s_set_vgpr_msb 0xa151
	v_wmma_f32_16x16x32_bf16 v[200:207] /*v[456:463]*/, v[144:151] /*v[400:407]*/, v[168:175], v[200:207] /*v[456:463]*/
	s_set_vgpr_msb 0x51a1
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[522:529]*/, v[112:119] /*v[368:375]*/, v[136:143], v[10:17] /*v[522:529]*/
	s_set_vgpr_msb 0xa151
	v_wmma_f32_16x16x32_bf16 v[208:215] /*v[464:471]*/, v[112:119] /*v[368:375]*/, v[168:175], v[208:215] /*v[464:471]*/
	s_set_vgpr_msb 0x51a1
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[530:537]*/, v[80:87] /*v[336:343]*/, v[136:143], v[18:25] /*v[530:537]*/
	s_set_vgpr_msb 0xa151
	v_wmma_f32_16x16x32_bf16 v[216:223] /*v[472:479]*/, v[80:87] /*v[336:343]*/, v[168:175], v[216:223] /*v[472:479]*/
	s_set_vgpr_msb 0x51a1
	v_wmma_f32_16x16x32_bf16 v[26:33] /*v[538:545]*/, v[48:55] /*v[304:311]*/, v[136:143], v[26:33] /*v[538:545]*/
	s_set_vgpr_msb 0xa151
	v_wmma_f32_16x16x32_bf16 v[224:231] /*v[480:487]*/, v[48:55] /*v[304:311]*/, v[168:175], v[224:231] /*v[480:487]*/
	s_set_vgpr_msb 0x51a1
	v_wmma_f32_16x16x32_bf16 v[34:41] /*v[546:553]*/, v[16:23] /*v[272:279]*/, v[136:143], v[34:41] /*v[546:553]*/
	s_set_vgpr_msb 0xa151
	v_wmma_f32_16x16x32_bf16 v[232:239] /*v[488:495]*/, v[16:23] /*v[272:279]*/, v[168:175], v[232:239] /*v[488:495]*/
	s_set_vgpr_msb 0x51a0
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[554:561]*/, v[240:247], v[136:143], v[42:49] /*v[554:561]*/
	s_set_vgpr_msb 0xa050
	v_wmma_f32_16x16x32_bf16 v[240:247] /*v[496:503]*/, v[240:247], v[168:175], v[240:247] /*v[496:503]*/
	s_set_vgpr_msb 0x50a0
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[562:569]*/, v[208:215], v[136:143], v[50:57] /*v[562:569]*/
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[570:577]*/, v[208:215], v[168:175], v[58:65] /*v[570:577]*/
	s_set_vgpr_msb 0xa051
	v_wmma_f32_16x16x32_bf16 v[250:257] /*v[506:513]*/, v[168:175] /*v[424:431]*/, v[144:151], v[250:257] /*v[506:513]*/
	v_wmma_f32_16x16x32_bf16 v[192:199] /*v[448:455]*/, v[168:175] /*v[424:431]*/, v[176:183], v[192:199] /*v[448:455]*/
	s_set_vgpr_msb 0x51a1
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[514:521]*/, v[136:143] /*v[392:399]*/, v[144:151], v[2:9] /*v[514:521]*/
	s_set_vgpr_msb 0xa151
	v_wmma_f32_16x16x32_bf16 v[200:207] /*v[456:463]*/, v[136:143] /*v[392:399]*/, v[176:183], v[200:207] /*v[456:463]*/
	s_set_vgpr_msb 0x51a1
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[522:529]*/, v[104:111] /*v[360:367]*/, v[144:151], v[10:17] /*v[522:529]*/
	s_set_vgpr_msb 0xa151
	v_wmma_f32_16x16x32_bf16 v[208:215] /*v[464:471]*/, v[104:111] /*v[360:367]*/, v[176:183], v[208:215] /*v[464:471]*/
	s_set_vgpr_msb 0x51a1
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[530:537]*/, v[72:79] /*v[328:335]*/, v[144:151], v[18:25] /*v[530:537]*/
	s_set_vgpr_msb 0xa151
	v_wmma_f32_16x16x32_bf16 v[216:223] /*v[472:479]*/, v[72:79] /*v[328:335]*/, v[176:183], v[216:223] /*v[472:479]*/
	s_set_vgpr_msb 0x51a1
	v_wmma_f32_16x16x32_bf16 v[26:33] /*v[538:545]*/, v[40:47] /*v[296:303]*/, v[144:151], v[26:33] /*v[538:545]*/
	s_set_vgpr_msb 0xa151
	v_wmma_f32_16x16x32_bf16 v[224:231] /*v[480:487]*/, v[40:47] /*v[296:303]*/, v[176:183], v[224:231] /*v[480:487]*/
	s_set_vgpr_msb 0x51a1
	v_wmma_f32_16x16x32_bf16 v[34:41] /*v[546:553]*/, v[8:15] /*v[264:271]*/, v[144:151], v[34:41] /*v[546:553]*/
	s_set_vgpr_msb 0xa151
	v_wmma_f32_16x16x32_bf16 v[232:239] /*v[488:495]*/, v[8:15] /*v[264:271]*/, v[176:183], v[232:239] /*v[488:495]*/
	s_set_vgpr_msb 0x51a0
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[554:561]*/, v[232:239], v[144:151], v[42:49] /*v[554:561]*/
	s_set_vgpr_msb 0xa050
	v_wmma_f32_16x16x32_bf16 v[240:247] /*v[496:503]*/, v[232:239], v[176:183], v[240:247] /*v[496:503]*/
	s_set_vgpr_msb 0x50a0
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[562:569]*/, v[200:207], v[144:151], v[50:57] /*v[562:569]*/
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[570:577]*/, v[200:207], v[176:183], v[58:65] /*v[570:577]*/
	s_set_vgpr_msb 0xa051
	v_wmma_f32_16x16x32_bf16 v[250:257] /*v[506:513]*/, v[160:167] /*v[416:423]*/, v[152:159], v[250:257] /*v[506:513]*/
	v_wmma_f32_16x16x32_bf16 v[192:199] /*v[448:455]*/, v[160:167] /*v[416:423]*/, v[184:191], v[192:199] /*v[448:455]*/
	s_set_vgpr_msb 0x51a1
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[514:521]*/, v[128:135] /*v[384:391]*/, v[152:159], v[2:9] /*v[514:521]*/
	s_set_vgpr_msb 0xa151
	v_wmma_f32_16x16x32_bf16 v[200:207] /*v[456:463]*/, v[128:135] /*v[384:391]*/, v[184:191], v[200:207] /*v[456:463]*/
	s_set_vgpr_msb 0x51a1
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[522:529]*/, v[96:103] /*v[352:359]*/, v[152:159], v[10:17] /*v[522:529]*/
	s_set_vgpr_msb 0xa151
	v_wmma_f32_16x16x32_bf16 v[208:215] /*v[464:471]*/, v[96:103] /*v[352:359]*/, v[184:191], v[208:215] /*v[464:471]*/
	s_set_vgpr_msb 0x51a1
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[530:537]*/, v[64:71] /*v[320:327]*/, v[152:159], v[18:25] /*v[530:537]*/
	s_set_vgpr_msb 0xa151
	v_wmma_f32_16x16x32_bf16 v[216:223] /*v[472:479]*/, v[64:71] /*v[320:327]*/, v[184:191], v[216:223] /*v[472:479]*/
	s_set_vgpr_msb 0x51a1
	v_wmma_f32_16x16x32_bf16 v[26:33] /*v[538:545]*/, v[32:39] /*v[288:295]*/, v[152:159], v[26:33] /*v[538:545]*/
	s_set_vgpr_msb 0xa151
	v_wmma_f32_16x16x32_bf16 v[224:231] /*v[480:487]*/, v[32:39] /*v[288:295]*/, v[184:191], v[224:231] /*v[480:487]*/
	s_set_vgpr_msb 0x51a1
	v_wmma_f32_16x16x32_bf16 v[34:41] /*v[546:553]*/, v[0:7] /*v[256:263]*/, v[152:159], v[34:41] /*v[546:553]*/
	s_set_vgpr_msb 0xa151
	v_wmma_f32_16x16x32_bf16 v[232:239] /*v[488:495]*/, v[0:7] /*v[256:263]*/, v[184:191], v[232:239] /*v[488:495]*/
	s_set_vgpr_msb 0x51a0
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[554:561]*/, v[224:231], v[152:159], v[42:49] /*v[554:561]*/
	s_set_vgpr_msb 0xa050
	v_wmma_f32_16x16x32_bf16 v[240:247] /*v[496:503]*/, v[224:231], v[184:191], v[240:247] /*v[496:503]*/
	s_set_vgpr_msb 0x50a0
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[562:569]*/, v[192:199], v[152:159], v[50:57] /*v[562:569]*/
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[570:577]*/, v[192:199], v[184:191], v[58:65] /*v[570:577]*/
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0xa042
	ds_load_tr16_b128 v[184:187] /*v[440:443]*/, v81 /*v593*/
	ds_load_tr16_b128 v[152:155] /*v[408:411]*/, v81 /*v593*/ offset:32
	ds_load_tr16_b128 v[188:191] /*v[444:447]*/, v81 /*v593*/ offset:4608
	ds_load_tr16_b128 v[156:159] /*v[412:415]*/, v81 /*v593*/ offset:4640
	ds_load_tr16_b128 v[176:179] /*v[432:435]*/, v81 /*v593*/ offset:9216
	ds_load_tr16_b128 v[144:147] /*v[400:403]*/, v81 /*v593*/ offset:9248
	ds_load_tr16_b128 v[180:183] /*v[436:439]*/, v81 /*v593*/ offset:13824
	ds_load_tr16_b128 v[148:151] /*v[404:407]*/, v81 /*v593*/ offset:13856
	ds_load_tr16_b128 v[168:171] /*v[424:427]*/, v81 /*v593*/ offset:18432
	ds_load_tr16_b128 v[136:139] /*v[392:395]*/, v81 /*v593*/ offset:18464
	ds_load_tr16_b128 v[172:175] /*v[428:431]*/, v81 /*v593*/ offset:23040
	ds_load_tr16_b128 v[140:143] /*v[396:399]*/, v81 /*v593*/ offset:23072
	ds_load_tr16_b128 v[160:163] /*v[416:419]*/, v81 /*v593*/ offset:27648
	ds_load_tr16_b128 v[128:131] /*v[384:387]*/, v81 /*v593*/ offset:27680
	ds_load_tr16_b128 v[164:167] /*v[420:423]*/, v81 /*v593*/ offset:32256
	ds_load_tr16_b128 v[132:135] /*v[388:391]*/, v81 /*v593*/ offset:32288
	ds_load_tr16_b128 v[120:123] /*v[376:379]*/, v81 /*v593*/ offset:64
	ds_load_tr16_b128 v[88:91] /*v[344:347]*/, v81 /*v593*/ offset:96
	ds_load_tr16_b128 v[124:127] /*v[380:383]*/, v81 /*v593*/ offset:4672
	ds_load_tr16_b128 v[92:95] /*v[348:351]*/, v81 /*v593*/ offset:4704
	ds_load_tr16_b128 v[112:115] /*v[368:371]*/, v81 /*v593*/ offset:9280
	ds_load_tr16_b128 v[80:83] /*v[336:339]*/, v81 /*v593*/ offset:9312
	ds_load_tr16_b128 v[116:119] /*v[372:375]*/, v81 /*v593*/ offset:13888
	ds_load_tr16_b128 v[84:87] /*v[340:343]*/, v81 /*v593*/ offset:13920
	ds_load_tr16_b128 v[104:107] /*v[360:363]*/, v81 /*v593*/ offset:18496
	ds_load_tr16_b128 v[72:75] /*v[328:331]*/, v81 /*v593*/ offset:18528
	ds_load_tr16_b128 v[108:111] /*v[364:367]*/, v81 /*v593*/ offset:23104
	ds_load_tr16_b128 v[76:79] /*v[332:335]*/, v81 /*v593*/ offset:23136
	ds_load_tr16_b128 v[96:99] /*v[352:355]*/, v81 /*v593*/ offset:27712
	ds_load_tr16_b128 v[64:67] /*v[320:323]*/, v81 /*v593*/ offset:27744
	ds_load_tr16_b128 v[100:103] /*v[356:359]*/, v81 /*v593*/ offset:32320
	ds_load_tr16_b128 v[68:71] /*v[324:327]*/, v81 /*v593*/ offset:32352
	ds_load_tr16_b128 v[56:59] /*v[312:315]*/, v81 /*v593*/ offset:128
	ds_load_tr16_b128 v[24:27] /*v[280:283]*/, v81 /*v593*/ offset:160
	ds_load_tr16_b128 v[60:63] /*v[316:319]*/, v81 /*v593*/ offset:4736
	ds_load_tr16_b128 v[28:31] /*v[284:287]*/, v81 /*v593*/ offset:4768
	ds_load_tr16_b128 v[48:51] /*v[304:307]*/, v81 /*v593*/ offset:9344
	ds_load_tr16_b128 v[16:19] /*v[272:275]*/, v81 /*v593*/ offset:9376
	ds_load_tr16_b128 v[52:55] /*v[308:311]*/, v81 /*v593*/ offset:13952
	ds_load_tr16_b128 v[20:23] /*v[276:279]*/, v81 /*v593*/ offset:13984
	ds_load_tr16_b128 v[40:43] /*v[296:299]*/, v81 /*v593*/ offset:18560
	ds_load_tr16_b128 v[8:11] /*v[264:267]*/, v81 /*v593*/ offset:18592
	ds_load_tr16_b128 v[44:47] /*v[300:303]*/, v81 /*v593*/ offset:23168
	ds_load_tr16_b128 v[12:15] /*v[268:271]*/, v81 /*v593*/ offset:23200
	ds_load_tr16_b128 v[32:35] /*v[288:291]*/, v81 /*v593*/ offset:27776
	ds_load_tr16_b128 v[0:3] /*v[256:259]*/, v81 /*v593*/ offset:27808
	ds_load_tr16_b128 v[36:39] /*v[292:295]*/, v81 /*v593*/ offset:32384
	ds_load_tr16_b128 v[4:7] /*v[260:263]*/, v81 /*v593*/ offset:32416
	s_set_vgpr_msb 0x4202
	ds_load_tr16_b128 v[248:251], v81 /*v593*/ offset:192
	ds_load_tr16_b128 v[216:219], v81 /*v593*/ offset:224
	ds_load_tr16_b128 v[252:255], v81 /*v593*/ offset:4800
	ds_load_tr16_b128 v[220:223], v81 /*v593*/ offset:4832
	ds_load_tr16_b128 v[240:243], v81 /*v593*/ offset:9408
	ds_load_tr16_b128 v[208:211], v81 /*v593*/ offset:9440
	ds_load_tr16_b128 v[244:247], v81 /*v593*/ offset:14016
	ds_load_tr16_b128 v[212:215], v81 /*v593*/ offset:14048
	ds_load_tr16_b128 v[232:235], v81 /*v593*/ offset:18624
	ds_load_tr16_b128 v[200:203], v81 /*v593*/ offset:18656
	ds_load_tr16_b128 v[236:239], v81 /*v593*/ offset:23232
	ds_load_tr16_b128 v[204:207], v81 /*v593*/ offset:23264
	ds_load_tr16_b128 v[224:227], v81 /*v593*/ offset:27840
	ds_load_tr16_b128 v[192:195], v81 /*v593*/ offset:27872
	ds_load_tr16_b128 v[228:231], v81 /*v593*/ offset:32448
	ds_load_tr16_b128 v[196:199], v81 /*v593*/ offset:32480
	s_set_vgpr_msb 0x2aa
	v_lshl_or_b32 v68 /*v580*/, s48, 7, v79 /*v591*/
	v_cmp_gt_i32_e32 vcc_lo, v68 /*v580*/, v80 /*v592*/
	v_cmp_lt_i32_e64 s2, v68 /*v580*/, v82 /*v594*/
	v_dual_add_nc_u32 v112 /*v624*/, 16, v68 /*v580*/ :: v_dual_bitop2_b32 v69 /*v581*/, 1, v68 /*v580*/ bitop3:0x54
	v_dual_add_nc_u32 v113 /*v625*/, 17, v68 /*v580*/ :: v_dual_bitop2_b32 v70 /*v582*/, 2, v68 /*v580*/ bitop3:0x54
	v_cmp_ge_i32_e64 s3, v68 /*v580*/, v80 /*v592*/
	v_cmp_lt_i32_e64 s4, v69 /*v581*/, v82 /*v594*/
	s_or_b32 s2, s2, vcc_lo
	v_dual_add_nc_u32 v114 /*v626*/, 18, v68 /*v580*/ :: v_dual_bitop2_b32 v71 /*v583*/, 3, v68 /*v580*/ bitop3:0x54
	s_set_vgpr_msb 0xaa41
	v_cndmask_b32_e64 v250 /*v506*/, v250 /*v506*/, 0xff800000, s2
	s_set_vgpr_msb 0x418a
	v_cmp_gt_i32_e32 vcc_lo, v70 /*v582*/, v80 /*v592*/
	v_cmp_lt_i32_e64 s2, v70 /*v582*/, v82 /*v594*/
	v_dual_add_nc_u32 v115 /*v627*/, 19, v68 /*v580*/ :: v_dual_bitop2_b32 v108 /*v620*/, 4, v68 /*v580*/ bitop3:0x54
	s_or_b32 s3, s4, s3
	v_cmp_gt_i32_e64 s5, v71 /*v583*/, v80 /*v592*/
	s_set_vgpr_msb 0x8a41
	v_cndmask_b32_e64 v251 /*v507*/, v251 /*v507*/, 0xff800000, s3
	s_set_vgpr_msb 0x418a
	v_cmp_lt_i32_e64 s3, v71 /*v583*/, v82 /*v594*/
	s_or_b32 s2, s2, vcc_lo
	v_dual_add_nc_u32 v116 /*v628*/, 20, v68 /*v580*/ :: v_dual_bitop2_b32 v109 /*v621*/, 5, v68 /*v580*/ bitop3:0x54
	s_set_vgpr_msb 0x8a41
	v_cndmask_b32_e64 v252 /*v508*/, v252 /*v508*/, 0xff800000, s2
	s_set_vgpr_msb 0x418a
	v_cmp_gt_i32_e32 vcc_lo, v108 /*v620*/, v80 /*v592*/
	v_cmp_lt_i32_e64 s2, v108 /*v620*/, v82 /*v594*/
	v_dual_add_nc_u32 v117 /*v629*/, 21, v68 /*v580*/ :: v_dual_bitop2_b32 v110 /*v622*/, 6, v68 /*v580*/ bitop3:0x54
	s_or_b32 s3, s3, s5
	v_cmp_lt_i32_e64 s4, v109 /*v621*/, v82 /*v594*/
	s_set_vgpr_msb 0x8a41
	v_cndmask_b32_e64 v253 /*v509*/, v253 /*v509*/, 0xff800000, s3
	s_set_vgpr_msb 0x418a
	v_cmp_gt_i32_e64 s3, v109 /*v621*/, v80 /*v592*/
	s_or_b32 s2, s2, vcc_lo
	v_dual_add_nc_u32 v118 /*v630*/, 22, v68 /*v580*/ :: v_dual_bitop2_b32 v111 /*v623*/, 7, v68 /*v580*/ bitop3:0x54
	s_set_vgpr_msb 0x8a41
	v_cndmask_b32_e64 v254 /*v510*/, v254 /*v510*/, 0xff800000, s2
	s_set_vgpr_msb 0x410a
	v_cmp_gt_i32_e32 vcc_lo, v110 /*v622*/, v80 /*v592*/
	v_cmp_lt_i32_e64 s2, v110 /*v622*/, v82 /*v594*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s4, v111 /*v623*/, v82 /*v594*/
	s_set_vgpr_msb 0xa41
	v_cndmask_b32_e64 v255 /*v511*/, v255 /*v511*/, 0xff800000, s3
	s_set_vgpr_msb 0x418a
	v_cmp_gt_i32_e64 s3, v111 /*v623*/, v80 /*v592*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v112 /*v624*/, v80 /*v592*/
	v_cndmask_b32_e64 v0 /*v512*/, v0 /*v512*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v112 /*v624*/, v82 /*v594*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s4, v113 /*v625*/, v82 /*v594*/
	v_cndmask_b32_e64 v1 /*v513*/, v1 /*v513*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v113 /*v625*/, v80 /*v592*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v114 /*v626*/, v80 /*v592*/
	v_cndmask_b32_e64 v2 /*v514*/, v2 /*v514*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v114 /*v626*/, v82 /*v594*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s4, v115 /*v627*/, v82 /*v594*/
	v_cndmask_b32_e64 v3 /*v515*/, v3 /*v515*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v115 /*v627*/, v80 /*v592*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v116 /*v628*/, v80 /*v592*/
	v_cndmask_b32_e64 v4 /*v516*/, v4 /*v516*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v116 /*v628*/, v82 /*v594*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s4, v117 /*v629*/, v82 /*v594*/
	v_cndmask_b32_e64 v5 /*v517*/, v5 /*v517*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v117 /*v629*/, v80 /*v592*/
	s_or_b32 s2, s2, vcc_lo
	v_dual_add_nc_u32 v119 /*v631*/, 23, v68 /*v580*/ :: v_dual_bitop2_b32 v120 /*v632*/, 32, v68 /*v580*/ bitop3:0x54
	v_cndmask_b32_e64 v6 /*v518*/, v6 /*v518*/, 0xff800000, s2
	v_cmp_gt_i32_e32 vcc_lo, v118 /*v630*/, v80 /*v592*/
	v_cmp_lt_i32_e64 s2, v118 /*v630*/, v82 /*v594*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s4, v119 /*v631*/, v82 /*v594*/
	v_cndmask_b32_e64 v7 /*v519*/, v7 /*v519*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v119 /*v631*/, v80 /*v592*/
	s_or_b32 s2, s2, vcc_lo
	v_or_b32_e32 v121 /*v633*/, 33, v68 /*v580*/
	v_cndmask_b32_e64 v8 /*v520*/, v8 /*v520*/, 0xff800000, s2
	v_cmp_gt_i32_e32 vcc_lo, v120 /*v632*/, v80 /*v592*/
	v_cmp_lt_i32_e64 s2, v120 /*v632*/, v82 /*v594*/
	v_dual_add_nc_u32 v129 /*v641*/, 49, v68 /*v580*/ :: v_dual_bitop2_b32 v122 /*v634*/, 34, v68 /*v580*/ bitop3:0x54
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s4, v121 /*v633*/, v82 /*v594*/
	v_cndmask_b32_e64 v9 /*v521*/, v9 /*v521*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v121 /*v633*/, v80 /*v592*/
	s_or_b32 s2, s2, vcc_lo
	v_dual_add_nc_u32 v130 /*v642*/, 50, v68 /*v580*/ :: v_dual_bitop2_b32 v123 /*v635*/, 35, v68 /*v580*/ bitop3:0x54
	v_cndmask_b32_e64 v10 /*v522*/, v10 /*v522*/, 0xff800000, s2
	v_cmp_gt_i32_e32 vcc_lo, v122 /*v634*/, v80 /*v592*/
	v_cmp_lt_i32_e64 s2, v122 /*v634*/, v82 /*v594*/
	v_dual_add_nc_u32 v131 /*v643*/, 51, v68 /*v580*/ :: v_dual_bitop2_b32 v124 /*v636*/, 36, v68 /*v580*/ bitop3:0x54
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s4, v123 /*v635*/, v82 /*v594*/
	v_cndmask_b32_e64 v11 /*v523*/, v11 /*v523*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v123 /*v635*/, v80 /*v592*/
	s_or_b32 s2, s2, vcc_lo
	v_dual_add_nc_u32 v132 /*v644*/, 52, v68 /*v580*/ :: v_dual_bitop2_b32 v125 /*v637*/, 37, v68 /*v580*/ bitop3:0x54
	v_cndmask_b32_e64 v12 /*v524*/, v12 /*v524*/, 0xff800000, s2
	v_cmp_gt_i32_e32 vcc_lo, v124 /*v636*/, v80 /*v592*/
	v_cmp_lt_i32_e64 s2, v124 /*v636*/, v82 /*v594*/
	s_or_b32 s3, s4, s3
	v_dual_add_nc_u32 v133 /*v645*/, 53, v68 /*v580*/ :: v_dual_bitop2_b32 v126 /*v638*/, 38, v68 /*v580*/ bitop3:0x54
	v_cndmask_b32_e64 v13 /*v525*/, v13 /*v525*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v125 /*v637*/, v80 /*v592*/
	v_cmp_lt_i32_e64 s4, v125 /*v637*/, v82 /*v594*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v126 /*v638*/, v80 /*v592*/
	v_cndmask_b32_e64 v66 /*v578*/, v14 /*v526*/, 0xff800000, s2
	v_dual_add_nc_u32 v134 /*v646*/, 54, v68 /*v580*/ :: v_dual_bitop2_b32 v14 /*v526*/, 39, v68 /*v580*/ bitop3:0x54
	v_cmp_lt_i32_e64 s2, v126 /*v638*/, v82 /*v594*/
	s_or_b32 s3, s4, s3
	v_dual_add_nc_u32 v135 /*v647*/, 55, v68 /*v580*/ :: v_dual_bitop2_b32 v136 /*v648*/, 64, v68 /*v580*/ bitop3:0x54
	v_cndmask_b32_e64 v67 /*v579*/, v15 /*v527*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v14 /*v526*/, v80 /*v592*/
	v_add_nc_u32_e32 v15 /*v527*/, 48, v68 /*v580*/
	v_cmp_lt_i32_e64 s4, v14 /*v526*/, v82 /*v594*/
	s_or_b32 s2, s2, vcc_lo
	v_or_b32_e32 v137 /*v649*/, 0x41, v68 /*v580*/
	v_cndmask_b32_e64 v16 /*v528*/, v16 /*v528*/, 0xff800000, s2
	v_cmp_gt_i32_e32 vcc_lo, v15 /*v527*/, v80 /*v592*/
	v_cmp_lt_i32_e64 s2, v15 /*v527*/, v82 /*v594*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s4, v129 /*v641*/, v82 /*v594*/
	v_cndmask_b32_e64 v17 /*v529*/, v17 /*v529*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v129 /*v641*/, v80 /*v592*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v130 /*v642*/, v80 /*v592*/
	v_cndmask_b32_e64 v18 /*v530*/, v18 /*v530*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v130 /*v642*/, v82 /*v594*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s4, v131 /*v643*/, v82 /*v594*/
	v_cndmask_b32_e64 v19 /*v531*/, v19 /*v531*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v131 /*v643*/, v80 /*v592*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v132 /*v644*/, v80 /*v592*/
	v_cndmask_b32_e64 v20 /*v532*/, v20 /*v532*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v132 /*v644*/, v82 /*v594*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s4, v133 /*v645*/, v82 /*v594*/
	v_cndmask_b32_e64 v21 /*v533*/, v21 /*v533*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v133 /*v645*/, v80 /*v592*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v134 /*v646*/, v80 /*v592*/
	v_cndmask_b32_e64 v22 /*v534*/, v22 /*v534*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v134 /*v646*/, v82 /*v594*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s4, v135 /*v647*/, v82 /*v594*/
	v_cndmask_b32_e64 v23 /*v535*/, v23 /*v535*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v135 /*v647*/, v80 /*v592*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v136 /*v648*/, v80 /*v592*/
	v_cndmask_b32_e64 v24 /*v536*/, v24 /*v536*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v136 /*v648*/, v82 /*v594*/
	s_or_b32 s3, s4, s3
	v_or_b32_e32 v138 /*v650*/, 0x42, v68 /*v580*/
	v_cndmask_b32_e64 v25 /*v537*/, v25 /*v537*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v137 /*v649*/, v80 /*v592*/
	v_cmp_lt_i32_e64 s4, v137 /*v649*/, v82 /*v594*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v138 /*v650*/, v80 /*v592*/
	v_cndmask_b32_e64 v90 /*v602*/, v26 /*v538*/, 0xff800000, s2
	v_or_b32_e32 v26 /*v538*/, 0x43, v68 /*v580*/
	v_cmp_lt_i32_e64 s2, v138 /*v650*/, v82 /*v594*/
	s_or_b32 s3, s4, s3
	v_or_b32_e32 v141 /*v653*/, 0x45, v68 /*v580*/
	v_cndmask_b32_e64 v91 /*v603*/, v27 /*v539*/, 0xff800000, s3
	v_or_b32_e32 v27 /*v539*/, 0x44, v68 /*v580*/
	v_cmp_gt_i32_e64 s3, v26 /*v538*/, v80 /*v592*/
	v_cmp_lt_i32_e64 s4, v26 /*v538*/, v82 /*v594*/
	s_or_b32 s2, s2, vcc_lo
	v_or_b32_e32 v142 /*v654*/, 0x46, v68 /*v580*/
	v_cndmask_b32_e64 v28 /*v540*/, v28 /*v540*/, 0xff800000, s2
	v_cmp_gt_i32_e32 vcc_lo, v27 /*v539*/, v80 /*v592*/
	v_cmp_lt_i32_e64 s2, v27 /*v539*/, v82 /*v594*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s4, v141 /*v653*/, v82 /*v594*/
	v_cndmask_b32_e64 v29 /*v541*/, v29 /*v541*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v141 /*v653*/, v80 /*v592*/
	s_or_b32 s2, s2, vcc_lo
	v_or_b32_e32 v143 /*v655*/, 0x47, v68 /*v580*/
	v_cndmask_b32_e64 v30 /*v542*/, v30 /*v542*/, 0xff800000, s2
	v_cmp_gt_i32_e32 vcc_lo, v142 /*v654*/, v80 /*v592*/
	v_cmp_lt_i32_e64 s2, v142 /*v654*/, v82 /*v594*/
	s_or_b32 s3, s4, s3
	v_add_nc_u32_e32 v144 /*v656*/, 0x50, v68 /*v580*/
	v_cndmask_b32_e64 v31 /*v543*/, v31 /*v543*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v143 /*v655*/, v80 /*v592*/
	v_cmp_lt_i32_e64 s4, v143 /*v655*/, v82 /*v594*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v144 /*v656*/, v80 /*v592*/
	v_cndmask_b32_e64 v92 /*v604*/, v32 /*v544*/, 0xff800000, s2
	v_add_nc_u32_e32 v32 /*v544*/, 0x51, v68 /*v580*/
	v_cmp_lt_i32_e64 s2, v144 /*v656*/, v82 /*v594*/
	s_or_b32 s3, s4, s3
	v_add_nc_u32_e32 v147 /*v659*/, 0x53, v68 /*v580*/
	v_cndmask_b32_e64 v93 /*v605*/, v33 /*v545*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v32 /*v544*/, v80 /*v592*/
	v_add_nc_u32_e32 v33 /*v545*/, 0x52, v68 /*v580*/
	v_cmp_lt_i32_e64 s4, v32 /*v544*/, v82 /*v594*/
	s_or_b32 s2, s2, vcc_lo
	v_add_nc_u32_e32 v148 /*v660*/, 0x54, v68 /*v580*/
	v_cndmask_b32_e64 v34 /*v546*/, v34 /*v546*/, 0xff800000, s2
	v_cmp_gt_i32_e32 vcc_lo, v33 /*v545*/, v80 /*v592*/
	v_cmp_lt_i32_e64 s2, v33 /*v545*/, v82 /*v594*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s4, v147 /*v659*/, v82 /*v594*/
	v_cndmask_b32_e64 v35 /*v547*/, v35 /*v547*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v147 /*v659*/, v80 /*v592*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v148 /*v660*/, v80 /*v592*/
	v_cndmask_b32_e64 v94 /*v606*/, v36 /*v548*/, 0xff800000, s2
	v_add_nc_u32_e32 v36 /*v548*/, 0x55, v68 /*v580*/
	v_cmp_lt_i32_e64 s2, v148 /*v660*/, v82 /*v594*/
	s_or_b32 s3, s4, s3
	v_add_nc_u32_e32 v150 /*v662*/, 0x57, v68 /*v580*/
	v_cndmask_b32_e64 v95 /*v607*/, v37 /*v549*/, 0xff800000, s3
	v_add_nc_u32_e32 v37 /*v549*/, 0x56, v68 /*v580*/
	v_cmp_gt_i32_e64 s3, v36 /*v548*/, v80 /*v592*/
	v_cmp_lt_i32_e64 s4, v36 /*v548*/, v82 /*v594*/
	s_or_b32 s2, s2, vcc_lo
	v_or_b32_e32 v151 /*v663*/, 0x60, v68 /*v580*/
	v_cndmask_b32_e64 v38 /*v550*/, v38 /*v550*/, 0xff800000, s2
	v_cmp_gt_i32_e32 vcc_lo, v37 /*v549*/, v80 /*v592*/
	v_cmp_lt_i32_e64 s2, v37 /*v549*/, v82 /*v594*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s4, v150 /*v662*/, v82 /*v594*/
	v_cndmask_b32_e64 v39 /*v551*/, v39 /*v551*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v150 /*v662*/, v80 /*v592*/
	s_or_b32 s2, s2, vcc_lo
	v_or_b32_e32 v153 /*v665*/, 0x61, v68 /*v580*/
	v_cndmask_b32_e64 v40 /*v552*/, v40 /*v552*/, 0xff800000, s2
	v_cmp_gt_i32_e32 vcc_lo, v151 /*v663*/, v80 /*v592*/
	v_cmp_lt_i32_e64 s2, v151 /*v663*/, v82 /*v594*/
	s_or_b32 s3, s4, s3
	v_or_b32_e32 v154 /*v666*/, 0x62, v68 /*v580*/
	v_cndmask_b32_e64 v41 /*v553*/, v41 /*v553*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v153 /*v665*/, v80 /*v592*/
	v_cmp_lt_i32_e64 s4, v153 /*v665*/, v82 /*v594*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v154 /*v666*/, v80 /*v592*/
	v_cndmask_b32_e64 v96 /*v608*/, v42 /*v554*/, 0xff800000, s2
	v_or_b32_e32 v42 /*v554*/, 0x63, v68 /*v580*/
	v_cmp_lt_i32_e64 s2, v154 /*v666*/, v82 /*v594*/
	s_or_b32 s3, s4, s3
	v_or_b32_e32 v159 /*v671*/, 0x67, v68 /*v580*/
	v_cndmask_b32_e64 v97 /*v609*/, v43 /*v555*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v42 /*v554*/, v80 /*v592*/
	v_or_b32_e32 v43 /*v555*/, 0x64, v68 /*v580*/
	v_cmp_lt_i32_e64 s4, v42 /*v554*/, v82 /*v594*/
	s_or_b32 s2, s2, vcc_lo
	v_add_nc_u32_e32 v160 /*v672*/, 0x70, v68 /*v580*/
	v_cndmask_b32_e64 v98 /*v610*/, v44 /*v556*/, 0xff800000, s2
	v_or_b32_e32 v44 /*v556*/, 0x65, v68 /*v580*/
	v_cmp_gt_i32_e32 vcc_lo, v43 /*v555*/, v80 /*v592*/
	v_cmp_lt_i32_e64 s2, v43 /*v555*/, v82 /*v594*/
	s_or_b32 s3, s4, s3
	v_add_nc_u32_e32 v161 /*v673*/, 0x71, v68 /*v580*/
	v_cndmask_b32_e64 v99 /*v611*/, v45 /*v557*/, 0xff800000, s3
	v_or_b32_e32 v45 /*v557*/, 0x66, v68 /*v580*/
	v_cmp_gt_i32_e64 s3, v44 /*v556*/, v80 /*v592*/
	v_cmp_lt_i32_e64 s4, v44 /*v556*/, v82 /*v594*/
	s_or_b32 s2, s2, vcc_lo
	v_add_nc_u32_e32 v162 /*v674*/, 0x72, v68 /*v580*/
	v_cndmask_b32_e64 v46 /*v558*/, v46 /*v558*/, 0xff800000, s2
	v_cmp_gt_i32_e32 vcc_lo, v45 /*v557*/, v80 /*v592*/
	v_cmp_lt_i32_e64 s2, v45 /*v557*/, v82 /*v594*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s4, v159 /*v671*/, v82 /*v594*/
	v_cndmask_b32_e64 v47 /*v559*/, v47 /*v559*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v159 /*v671*/, v80 /*v592*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v160 /*v672*/, v80 /*v592*/
	v_cndmask_b32_e64 v48 /*v560*/, v48 /*v560*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v160 /*v672*/, v82 /*v594*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s4, v161 /*v673*/, v82 /*v594*/
	v_cndmask_b32_e64 v49 /*v561*/, v49 /*v561*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v161 /*v673*/, v80 /*v592*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v162 /*v674*/, v80 /*v592*/
	v_cndmask_b32_e64 v100 /*v612*/, v50 /*v562*/, 0xff800000, s2
	v_add_nc_u32_e32 v50 /*v562*/, 0x73, v68 /*v580*/
	v_cmp_lt_i32_e64 s2, v162 /*v674*/, v82 /*v594*/
	s_or_b32 s3, s4, s3
	v_add_nc_u32_e32 v165 /*v677*/, 0x77, v68 /*v580*/
	v_cndmask_b32_e64 v101 /*v613*/, v51 /*v563*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v50 /*v562*/, v80 /*v592*/
	v_add_nc_u32_e32 v51 /*v563*/, 0x74, v68 /*v580*/
	v_cmp_lt_i32_e64 s4, v50 /*v562*/, v82 /*v594*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e64 s5, v68 /*v580*/, v85 /*v597*/
	v_cndmask_b32_e64 v102 /*v614*/, v52 /*v564*/, 0xff800000, s2
	v_add_nc_u32_e32 v52 /*v564*/, 0x75, v68 /*v580*/
	v_cmp_gt_i32_e32 vcc_lo, v51 /*v563*/, v80 /*v592*/
	v_cmp_lt_i32_e64 s2, v51 /*v563*/, v82 /*v594*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s6, v68 /*v580*/, v83 /*v595*/
	v_cndmask_b32_e64 v103 /*v615*/, v53 /*v565*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v52 /*v564*/, v80 /*v592*/
	v_cmp_lt_i32_e64 s4, v52 /*v564*/, v82 /*v594*/
	v_add_nc_u32_e32 v53 /*v565*/, 0x76, v68 /*v580*/
	s_or_b32 s2, s2, vcc_lo
	v_cndmask_b32_e64 v54 /*v566*/, v54 /*v566*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	v_cmp_gt_i32_e32 vcc_lo, v53 /*v565*/, v80 /*v592*/
	v_cndmask_b32_e64 v55 /*v567*/, v55 /*v567*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v53 /*v565*/, v82 /*v594*/
	v_cmp_gt_i32_e64 s3, v165 /*v677*/, v80 /*v592*/
	v_cmp_lt_i32_e64 s4, v165 /*v677*/, v82 /*v594*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_ge_i32_e32 vcc_lo, v68 /*v580*/, v85 /*v597*/
	v_cndmask_b32_e64 v56 /*v568*/, v56 /*v568*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	v_cmp_gt_i32_e64 s3, v70 /*v582*/, v85 /*v597*/
	v_cndmask_b32_e64 v57 /*v569*/, v57 /*v569*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	v_cmp_lt_i32_e64 s4, v70 /*v582*/, v83 /*v595*/
	s_set_vgpr_msb 0x8a81
	v_cndmask_b32_e64 v104 /*v616*/, v192 /*v448*/, 0xff800000, s2
	s_set_vgpr_msb 0x810a
	v_cmp_lt_i32_e64 s2, v69 /*v581*/, v83 /*v595*/
	v_cmp_gt_i32_e64 s5, v71 /*v583*/, v85 /*v597*/
	v_cmp_lt_i32_e64 s6, v71 /*v583*/, v83 /*v595*/
	s_set_vgpr_msb 0xa55
	v_max3_num_f32 v192 /*v448*/, v250 /*v506*/, v251 /*v507*/, v252 /*v508*/
	s_or_b32 s2, s2, vcc_lo
	s_set_vgpr_msb 0x550a
	v_cmp_gt_i32_e32 vcc_lo, v108 /*v620*/, v85 /*v597*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v105 /*v617*/, v193 /*v449*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	s_set_vgpr_msb 0x810a
	v_cmp_gt_i32_e64 s3, v109 /*v621*/, v85 /*v597*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v106 /*v618*/, v194 /*v450*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	s_set_vgpr_msb 0x810a
	v_cmp_lt_i32_e64 s4, v109 /*v621*/, v83 /*v595*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v107 /*v619*/, v195 /*v451*/, 0xff800000, s2
	s_set_vgpr_msb 0x816a
	v_cmp_lt_i32_e64 s2, v108 /*v620*/, v83 /*v595*/
	v_cmp_gt_i32_e64 s5, v110 /*v622*/, v85 /*v597*/
	v_cmp_lt_i32_e64 s6, v110 /*v622*/, v83 /*v595*/
	v_max3_num_f32 v193 /*v449*/, v104 /*v616*/, v105 /*v617*/, v106 /*v618*/
	s_set_vgpr_msb 0x6a55
	v_max3_num_f32 v194 /*v450*/, v253 /*v509*/, v254 /*v510*/, v255 /*v511*/
	s_or_b32 s2, s2, vcc_lo
	s_set_vgpr_msb 0x550a
	v_cmp_gt_i32_e32 vcc_lo, v111 /*v623*/, v85 /*v597*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v108 /*v620*/, v196 /*v452*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	s_set_vgpr_msb 0x810a
	v_cmp_gt_i32_e64 s3, v112 /*v624*/, v85 /*v597*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v109 /*v621*/, v197 /*v453*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	s_set_vgpr_msb 0x810a
	v_cmp_lt_i32_e64 s4, v112 /*v624*/, v83 /*v595*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v110 /*v622*/, v198 /*v454*/, 0xff800000, s2
	s_set_vgpr_msb 0x816a
	v_cmp_lt_i32_e64 s2, v111 /*v623*/, v83 /*v595*/
	v_cmp_gt_i32_e64 s5, v113 /*v625*/, v85 /*v597*/
	v_cmp_lt_i32_e64 s6, v113 /*v625*/, v83 /*v595*/
	v_max3_num_f32 v195 /*v451*/, v107 /*v619*/, v108 /*v620*/, v109 /*v621*/
	v_max3_num_f32 v196 /*v452*/, v0 /*v512*/, v1 /*v513*/, v2 /*v514*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v114 /*v626*/, v85 /*v597*/
	s_set_vgpr_msb 0x6a81
	v_cndmask_b32_e64 v111 /*v623*/, v199 /*v455*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	s_set_vgpr_msb 0x810a
	v_cmp_gt_i32_e64 s3, v115 /*v627*/, v85 /*v597*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v112 /*v624*/, v200 /*v456*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	s_set_vgpr_msb 0x810a
	v_cmp_lt_i32_e64 s4, v115 /*v627*/, v83 /*v595*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v113 /*v625*/, v201 /*v457*/, 0xff800000, s2
	s_set_vgpr_msb 0x816a
	v_cmp_lt_i32_e64 s2, v114 /*v626*/, v83 /*v595*/
	v_cmp_gt_i32_e64 s5, v116 /*v628*/, v85 /*v597*/
	v_cmp_lt_i32_e64 s6, v116 /*v628*/, v83 /*v595*/
	v_max3_num_f32 v197 /*v453*/, v110 /*v622*/, v111 /*v623*/, v112 /*v624*/
	v_max3_num_f32 v198 /*v454*/, v3 /*v515*/, v4 /*v516*/, v5 /*v517*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v117 /*v629*/, v85 /*v597*/
	s_set_vgpr_msb 0x6a81
	v_cndmask_b32_e64 v114 /*v626*/, v202 /*v458*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	s_set_vgpr_msb 0x810a
	v_cmp_gt_i32_e64 s3, v118 /*v630*/, v85 /*v597*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v115 /*v627*/, v203 /*v459*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	s_set_vgpr_msb 0x810a
	v_cmp_lt_i32_e64 s4, v118 /*v630*/, v83 /*v595*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v116 /*v628*/, v204 /*v460*/, 0xff800000, s2
	s_set_vgpr_msb 0x816a
	v_cmp_lt_i32_e64 s2, v117 /*v629*/, v83 /*v595*/
	v_cmp_gt_i32_e64 s5, v119 /*v631*/, v85 /*v597*/
	v_cmp_lt_i32_e64 s6, v119 /*v631*/, v83 /*v595*/
	v_max3_num_f32 v200 /*v456*/, v6 /*v518*/, v7 /*v519*/, v8 /*v520*/
	v_max3_num_f32 v202 /*v458*/, v9 /*v521*/, v10 /*v522*/, v11 /*v523*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v120 /*v632*/, v85 /*v597*/
	s_set_vgpr_msb 0x6a81
	v_cndmask_b32_e64 v117 /*v629*/, v205 /*v461*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	s_set_vgpr_msb 0x810a
	v_cmp_gt_i32_e64 s3, v121 /*v633*/, v85 /*v597*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v118 /*v630*/, v206 /*v462*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	s_set_vgpr_msb 0x810a
	v_cmp_lt_i32_e64 s4, v121 /*v633*/, v83 /*v595*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v119 /*v631*/, v207 /*v463*/, 0xff800000, s2
	s_set_vgpr_msb 0x816a
	v_cmp_lt_i32_e64 s2, v120 /*v632*/, v83 /*v595*/
	v_cmp_gt_i32_e64 s5, v122 /*v634*/, v85 /*v597*/
	v_cmp_lt_i32_e64 s6, v122 /*v634*/, v83 /*v595*/
	v_max3_num_f32 v204 /*v460*/, v12 /*v524*/, v13 /*v525*/, v66 /*v578*/
	v_max3_num_f32 v206 /*v462*/, v67 /*v579*/, v16 /*v528*/, v17 /*v529*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v123 /*v635*/, v85 /*v597*/
	s_set_vgpr_msb 0x6a81
	v_cndmask_b32_e64 v120 /*v632*/, v208 /*v464*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	s_set_vgpr_msb 0x810a
	v_cmp_gt_i32_e64 s3, v124 /*v636*/, v85 /*v597*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v121 /*v633*/, v209 /*v465*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	s_set_vgpr_msb 0x810a
	v_cmp_lt_i32_e64 s4, v124 /*v636*/, v83 /*v595*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v122 /*v634*/, v210 /*v466*/, 0xff800000, s2
	s_set_vgpr_msb 0x816a
	v_cmp_lt_i32_e64 s2, v123 /*v635*/, v83 /*v595*/
	v_cmp_gt_i32_e64 s5, v125 /*v637*/, v85 /*v597*/
	v_cmp_lt_i32_e64 s6, v125 /*v637*/, v83 /*v595*/
	v_max3_num_f32 v208 /*v464*/, v18 /*v530*/, v19 /*v531*/, v20 /*v532*/
	v_max3_num_f32 v210 /*v466*/, v21 /*v533*/, v22 /*v534*/, v23 /*v535*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v126 /*v638*/, v85 /*v597*/
	s_set_vgpr_msb 0x6a81
	v_cndmask_b32_e64 v123 /*v635*/, v211 /*v467*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	s_set_vgpr_msb 0x810a
	v_cmp_gt_i32_e64 s3, v14 /*v526*/, v85 /*v597*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v124 /*v636*/, v212 /*v468*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	s_set_vgpr_msb 0x810a
	v_cmp_lt_i32_e64 s4, v14 /*v526*/, v83 /*v595*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v125 /*v637*/, v213 /*v469*/, 0xff800000, s2
	s_set_vgpr_msb 0x816a
	v_cmp_lt_i32_e64 s2, v126 /*v638*/, v83 /*v595*/
	v_cmp_gt_i32_e64 s5, v15 /*v527*/, v85 /*v597*/
	v_cmp_lt_i32_e64 s6, v15 /*v527*/, v83 /*v595*/
	v_max3_num_f32 v212 /*v468*/, v24 /*v536*/, v25 /*v537*/, v90 /*v602*/
	v_max3_num_f32 v199 /*v455*/, v113 /*v625*/, v114 /*v626*/, v115 /*v627*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v129 /*v641*/, v85 /*v597*/
	s_set_vgpr_msb 0x6a81
	v_cndmask_b32_e64 v126 /*v638*/, v214 /*v470*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	s_set_vgpr_msb 0x810a
	v_cmp_gt_i32_e64 s3, v130 /*v642*/, v85 /*v597*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v127 /*v639*/, v215 /*v471*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	s_set_vgpr_msb 0x810a
	v_cmp_lt_i32_e64 s4, v130 /*v642*/, v83 /*v595*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v128 /*v640*/, v216 /*v472*/, 0xff800000, s2
	s_set_vgpr_msb 0x816a
	v_cmp_lt_i32_e64 s2, v129 /*v641*/, v83 /*v595*/
	v_cmp_gt_i32_e64 s5, v131 /*v643*/, v85 /*v597*/
	v_cmp_lt_i32_e64 s6, v131 /*v643*/, v83 /*v595*/
	v_max3_num_f32 v214 /*v470*/, v91 /*v603*/, v28 /*v540*/, v29 /*v541*/
	v_max3_num_f32 v216 /*v472*/, v30 /*v542*/, v31 /*v543*/, v92 /*v604*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v132 /*v644*/, v85 /*v597*/
	s_set_vgpr_msb 0x6a81
	v_cndmask_b32_e64 v129 /*v641*/, v217 /*v473*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	s_set_vgpr_msb 0x810a
	v_cmp_gt_i32_e64 s3, v133 /*v645*/, v85 /*v597*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v130 /*v642*/, v218 /*v474*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	s_set_vgpr_msb 0x810a
	v_cmp_lt_i32_e64 s4, v133 /*v645*/, v83 /*v595*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v131 /*v643*/, v219 /*v475*/, 0xff800000, s2
	s_set_vgpr_msb 0x816a
	v_cmp_lt_i32_e64 s2, v132 /*v644*/, v83 /*v595*/
	v_cmp_gt_i32_e64 s5, v134 /*v646*/, v85 /*v597*/
	v_cmp_lt_i32_e64 s6, v134 /*v646*/, v83 /*v595*/
	v_max3_num_f32 v218 /*v474*/, v93 /*v605*/, v34 /*v546*/, v35 /*v547*/
	v_max3_num_f32 v201 /*v457*/, v116 /*v628*/, v117 /*v629*/, v118 /*v630*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v135 /*v647*/, v85 /*v597*/
	s_set_vgpr_msb 0x6a81
	v_cndmask_b32_e64 v132 /*v644*/, v220 /*v476*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	s_set_vgpr_msb 0x810a
	v_cmp_gt_i32_e64 s3, v136 /*v648*/, v85 /*v597*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v133 /*v645*/, v221 /*v477*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	s_set_vgpr_msb 0x810a
	v_cmp_lt_i32_e64 s4, v136 /*v648*/, v83 /*v595*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v134 /*v646*/, v222 /*v478*/, 0xff800000, s2
	s_set_vgpr_msb 0x816a
	v_cmp_lt_i32_e64 s2, v135 /*v647*/, v83 /*v595*/
	v_cmp_gt_i32_e64 s5, v137 /*v649*/, v85 /*v597*/
	v_cmp_lt_i32_e64 s6, v137 /*v649*/, v83 /*v595*/
	v_max3_num_f32 v220 /*v476*/, v94 /*v606*/, v95 /*v607*/, v38 /*v550*/
	v_max3_num_f32 v222 /*v478*/, v39 /*v551*/, v40 /*v552*/, v41 /*v553*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v138 /*v650*/, v85 /*v597*/
	s_set_vgpr_msb 0x6a81
	v_cndmask_b32_e64 v135 /*v647*/, v223 /*v479*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	s_set_vgpr_msb 0x810a
	v_cmp_gt_i32_e64 s3, v26 /*v538*/, v85 /*v597*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v136 /*v648*/, v224 /*v480*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	s_set_vgpr_msb 0x810a
	v_cmp_lt_i32_e64 s4, v26 /*v538*/, v83 /*v595*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v137 /*v649*/, v225 /*v481*/, 0xff800000, s2
	s_set_vgpr_msb 0x816a
	v_cmp_lt_i32_e64 s2, v138 /*v650*/, v83 /*v595*/
	v_cmp_gt_i32_e64 s5, v27 /*v539*/, v85 /*v597*/
	v_cmp_lt_i32_e64 s6, v27 /*v539*/, v83 /*v595*/
	v_max3_num_f32 v224 /*v480*/, v96 /*v608*/, v97 /*v609*/, v98 /*v610*/
	v_max3_num_f32 v203 /*v459*/, v119 /*v631*/, v120 /*v632*/, v121 /*v633*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v141 /*v653*/, v85 /*v597*/
	s_set_vgpr_msb 0x6a81
	v_cndmask_b32_e64 v138 /*v650*/, v226 /*v482*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	s_set_vgpr_msb 0x810a
	v_cmp_gt_i32_e64 s3, v142 /*v654*/, v85 /*v597*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v139 /*v651*/, v227 /*v483*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	s_set_vgpr_msb 0x810a
	v_cmp_lt_i32_e64 s4, v142 /*v654*/, v83 /*v595*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v140 /*v652*/, v228 /*v484*/, 0xff800000, s2
	s_set_vgpr_msb 0x816a
	v_cmp_lt_i32_e64 s2, v141 /*v653*/, v83 /*v595*/
	v_cmp_gt_i32_e64 s5, v143 /*v655*/, v85 /*v597*/
	v_cmp_lt_i32_e64 s6, v143 /*v655*/, v83 /*v595*/
	v_max3_num_f32 v226 /*v482*/, v99 /*v611*/, v46 /*v558*/, v47 /*v559*/
	v_max3_num_f32 v205 /*v461*/, v122 /*v634*/, v123 /*v635*/, v124 /*v636*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v144 /*v656*/, v85 /*v597*/
	s_set_vgpr_msb 0x6a81
	v_cndmask_b32_e64 v141 /*v653*/, v229 /*v485*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	s_set_vgpr_msb 0x810a
	v_cmp_gt_i32_e64 s3, v32 /*v544*/, v85 /*v597*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v142 /*v654*/, v230 /*v486*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	s_set_vgpr_msb 0x810a
	v_cmp_lt_i32_e64 s4, v32 /*v544*/, v83 /*v595*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v143 /*v655*/, v231 /*v487*/, 0xff800000, s2
	s_set_vgpr_msb 0x816a
	v_cmp_lt_i32_e64 s2, v144 /*v656*/, v83 /*v595*/
	v_cmp_gt_i32_e64 s5, v33 /*v545*/, v85 /*v597*/
	v_cmp_lt_i32_e64 s6, v33 /*v545*/, v83 /*v595*/
	v_max3_num_f32 v230 /*v486*/, v101 /*v613*/, v102 /*v614*/, v103 /*v615*/
	v_max3_num_f32 v207 /*v463*/, v125 /*v637*/, v126 /*v638*/, v127 /*v639*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v147 /*v659*/, v85 /*v597*/
	s_set_vgpr_msb 0x6a81
	v_cndmask_b32_e64 v144 /*v656*/, v232 /*v488*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	s_set_vgpr_msb 0x810a
	v_cmp_gt_i32_e64 s3, v148 /*v660*/, v85 /*v597*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v145 /*v657*/, v233 /*v489*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	s_set_vgpr_msb 0x810a
	v_cmp_lt_i32_e64 s4, v148 /*v660*/, v83 /*v595*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v146 /*v658*/, v234 /*v490*/, 0xff800000, s2
	s_set_vgpr_msb 0x816a
	v_cmp_lt_i32_e64 s2, v147 /*v659*/, v83 /*v595*/
	v_cmp_gt_i32_e64 s5, v36 /*v548*/, v85 /*v597*/
	v_cmp_lt_i32_e64 s6, v36 /*v548*/, v83 /*v595*/
	v_max3_num_f32 v209 /*v465*/, v128 /*v640*/, v129 /*v641*/, v130 /*v642*/
	v_max3_num_f32 v211 /*v467*/, v131 /*v643*/, v132 /*v644*/, v133 /*v645*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v37 /*v549*/, v85 /*v597*/
	s_set_vgpr_msb 0x6a81
	v_cndmask_b32_e64 v147 /*v659*/, v235 /*v491*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	s_set_vgpr_msb 0x810a
	v_cmp_gt_i32_e64 s3, v150 /*v662*/, v85 /*v597*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v148 /*v660*/, v236 /*v492*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	s_set_vgpr_msb 0x810a
	v_cmp_lt_i32_e64 s4, v150 /*v662*/, v83 /*v595*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v149 /*v661*/, v237 /*v493*/, 0xff800000, s2
	s_set_vgpr_msb 0x816a
	v_cmp_lt_i32_e64 s2, v37 /*v549*/, v83 /*v595*/
	v_cmp_gt_i32_e64 s5, v151 /*v663*/, v85 /*v597*/
	v_cmp_lt_i32_e64 s6, v151 /*v663*/, v83 /*v595*/
	v_max3_num_f32 v213 /*v469*/, v134 /*v646*/, v135 /*v647*/, v136 /*v648*/
	v_max3_num_f32 v215 /*v471*/, v137 /*v649*/, v138 /*v650*/, v139 /*v651*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v153 /*v665*/, v85 /*v597*/
	s_set_vgpr_msb 0x6a81
	v_cndmask_b32_e64 v150 /*v662*/, v238 /*v494*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	s_set_vgpr_msb 0x810a
	v_cmp_gt_i32_e64 s3, v154 /*v666*/, v85 /*v597*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v151 /*v663*/, v239 /*v495*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	s_set_vgpr_msb 0x810a
	v_cmp_lt_i32_e64 s4, v154 /*v666*/, v83 /*v595*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v152 /*v664*/, v240 /*v496*/, 0xff800000, s2
	s_set_vgpr_msb 0x816a
	v_cmp_lt_i32_e64 s2, v153 /*v665*/, v83 /*v595*/
	v_cmp_gt_i32_e64 s5, v42 /*v554*/, v85 /*v597*/
	v_cmp_lt_i32_e64 s6, v42 /*v554*/, v83 /*v595*/
	v_max3_num_f32 v217 /*v473*/, v140 /*v652*/, v141 /*v653*/, v142 /*v654*/
	v_max3_num_f32 v219 /*v475*/, v143 /*v655*/, v144 /*v656*/, v145 /*v657*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v43 /*v555*/, v85 /*v597*/
	s_set_vgpr_msb 0x6a81
	v_cndmask_b32_e64 v153 /*v665*/, v241 /*v497*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	s_set_vgpr_msb 0x810a
	v_cmp_gt_i32_e64 s3, v44 /*v556*/, v85 /*v597*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v154 /*v666*/, v242 /*v498*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	s_set_vgpr_msb 0x810a
	v_cmp_lt_i32_e64 s4, v44 /*v556*/, v83 /*v595*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v155 /*v667*/, v243 /*v499*/, 0xff800000, s2
	s_set_vgpr_msb 0x816a
	v_cmp_lt_i32_e64 s2, v43 /*v555*/, v83 /*v595*/
	v_cmp_gt_i32_e64 s5, v45 /*v557*/, v85 /*v597*/
	v_cmp_lt_i32_e64 s6, v45 /*v557*/, v83 /*v595*/
	v_max3_num_f32 v221 /*v477*/, v146 /*v658*/, v147 /*v659*/, v148 /*v660*/
	v_max3_num_f32 v223 /*v479*/, v149 /*v661*/, v150 /*v662*/, v151 /*v663*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v159 /*v671*/, v85 /*v597*/
	s_set_vgpr_msb 0x6a81
	v_cndmask_b32_e64 v156 /*v668*/, v244 /*v500*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	s_set_vgpr_msb 0x810a
	v_cmp_gt_i32_e64 s3, v160 /*v672*/, v85 /*v597*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v157 /*v669*/, v245 /*v501*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	s_set_vgpr_msb 0x810a
	v_cmp_lt_i32_e64 s4, v160 /*v672*/, v83 /*v595*/
	s_set_vgpr_msb 0xa81
	v_cndmask_b32_e64 v158 /*v670*/, v246 /*v502*/, 0xff800000, s2
	s_set_vgpr_msb 0x816a
	v_cmp_lt_i32_e64 s2, v159 /*v671*/, v83 /*v595*/
	v_cmp_gt_i32_e64 s5, v161 /*v673*/, v85 /*v597*/
	v_cmp_lt_i32_e64 s6, v161 /*v673*/, v83 /*v595*/
	v_max3_num_f32 v225 /*v481*/, v152 /*v664*/, v153 /*v665*/, v154 /*v666*/
	v_max3_num_f32 v227 /*v483*/, v155 /*v667*/, v156 /*v668*/, v157 /*v669*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v162 /*v674*/, v85 /*v597*/
	s_set_vgpr_msb 0x6a81
	v_cndmask_b32_e64 v159 /*v671*/, v247 /*v503*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	s_set_vgpr_msb 0x818a
	v_cmp_gt_i32_e64 s3, v50 /*v562*/, v85 /*v597*/
	v_cndmask_b32_e64 v160 /*v672*/, v58 /*v570*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	v_cmp_lt_i32_e64 s4, v50 /*v562*/, v83 /*v595*/
	v_cndmask_b32_e64 v161 /*v673*/, v59 /*v571*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v162 /*v674*/, v83 /*v595*/
	v_cmp_gt_i32_e64 s5, v51 /*v563*/, v85 /*v597*/
	v_cmp_lt_i32_e64 s6, v51 /*v563*/, v83 /*v595*/
	s_set_vgpr_msb 0x8a4a
	v_dual_max_num_f32 v228 /*v484*/, v48 /*v560*/, v49 /*v561*/ :: v_dual_max_num_f32 v229 /*v485*/, v158 /*v670*/, v159 /*v671*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v52 /*v564*/, v85 /*v597*/
	s_set_vgpr_msb 0x4a8a
	v_cndmask_b32_e64 v162 /*v674*/, v60 /*v572*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	v_cmp_gt_i32_e64 s3, v53 /*v565*/, v85 /*v597*/
	v_cndmask_b32_e64 v163 /*v675*/, v61 /*v573*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	v_cmp_lt_i32_e64 s4, v53 /*v565*/, v83 /*v595*/
	v_cndmask_b32_e64 v164 /*v676*/, v62 /*v574*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v52 /*v564*/, v83 /*v595*/
	v_cmp_gt_i32_e64 s5, v165 /*v677*/, v85 /*v597*/
	v_cmp_lt_i32_e64 s6, v165 /*v677*/, v83 /*v595*/
	s_set_vgpr_msb 0x8a6a
	v_max3_num_f32 v231 /*v487*/, v161 /*v673*/, v162 /*v674*/, v163 /*v675*/
	v_max3_num_f32 v232 /*v488*/, v54 /*v566*/, v55 /*v567*/, v56 /*v568*/
	s_or_b32 s2, s2, vcc_lo
	s_set_vgpr_msb 0x6a55
	v_max3_num_f32 v192 /*v448*/, v192 /*v448*/, v194 /*v450*/, v196 /*v452*/
	s_set_vgpr_msb 0x5582
	v_cndmask_b32_e64 v165 /*v677*/, v63 /*v575*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	s_set_vgpr_msb 0x8255
	v_max3_num_f32 v193 /*v449*/, v193 /*v449*/, v195 /*v451*/, v197 /*v453*/
	s_set_vgpr_msb 0x5582
	v_cndmask_b32_e64 v166 /*v678*/, v64 /*v576*/, 0xff800000, s2
	s_set_vgpr_msb 0x8255
	v_max3_num_f32 v194 /*v450*/, v198 /*v454*/, v200 /*v456*/, v202 /*v458*/
	v_max3_num_f32 v195 /*v451*/, v204 /*v460*/, v206 /*v462*/, v208 /*v464*/
	v_max3_num_f32 v196 /*v452*/, v210 /*v466*/, v212 /*v468*/, v214 /*v470*/
	v_max3_num_f32 v197 /*v453*/, v216 /*v472*/, v218 /*v474*/, v220 /*v476*/
	v_max3_num_f32 v198 /*v454*/, v222 /*v478*/, v224 /*v480*/, v226 /*v482*/
	s_set_vgpr_msb 0x5559
	v_max3_num_f32 v200 /*v456*/, v228 /*v484*/, v100 /*v612*/, v230 /*v486*/
	s_or_b32 s2, s6, s5
	s_set_vgpr_msb 0x596a
	v_max3_num_f32 v233 /*v489*/, v164 /*v676*/, v165 /*v677*/, v166 /*v678*/
	s_set_vgpr_msb 0x6a82
	v_cndmask_b32_e64 v167 /*v679*/, v65 /*v577*/, 0xff800000, s2
	s_set_vgpr_msb 0x8255
	v_max3_num_f32 v199 /*v455*/, v199 /*v455*/, v201 /*v457*/, v203 /*v459*/
	v_max3_num_f32 v201 /*v457*/, v205 /*v461*/, v207 /*v463*/, v209 /*v465*/
	v_max3_num_f32 v192 /*v448*/, v192 /*v448*/, v194 /*v450*/, v195 /*v451*/
	v_max3_num_f32 v194 /*v450*/, v196 /*v452*/, v197 /*v453*/, v198 /*v454*/
	s_set_vgpr_msb 0x5565
	v_max3_num_f32 v195 /*v451*/, v200 /*v456*/, v232 /*v488*/, v57 /*v569*/
	s_set_vgpr_msb 0x6555
	v_max3_num_f32 v196 /*v452*/, v211 /*v467*/, v213 /*v469*/, v215 /*v471*/
	v_max3_num_f32 v197 /*v453*/, v217 /*v473*/, v219 /*v475*/, v221 /*v477*/
	v_max3_num_f32 v198 /*v454*/, v223 /*v479*/, v225 /*v481*/, v227 /*v483*/
	s_set_vgpr_msb 0x5559
	v_max3_num_f32 v200 /*v456*/, v229 /*v485*/, v160 /*v672*/, v231 /*v487*/
	s_set_vgpr_msb 0x5955
	v_max3_num_f32 v192 /*v448*/, v192 /*v448*/, v194 /*v450*/, v195 /*v451*/
	v_max3_num_f32 v193 /*v449*/, v193 /*v449*/, v199 /*v455*/, v201 /*v457*/
	v_max3_num_f32 v194 /*v450*/, v196 /*v452*/, v197 /*v453*/, v198 /*v454*/
	s_set_vgpr_msb 0x5565
	v_max3_num_f32 v195 /*v451*/, v200 /*v456*/, v233 /*v489*/, v167 /*v679*/
	s_set_vgpr_msb 0x6555
	v_max3_num_f32 v193 /*v449*/, v193 /*v449*/, v194 /*v450*/, v195 /*v451*/
	v_dual_mov_b32 v196 /*v452*/, v192 /*v448*/ :: v_dual_mov_b32 v194 /*v450*/, v193 /*v449*/
	v_permlanex16_b32 v196 /*v452*/, v196 /*v452*/, s17, 0xfedcba98
	v_permlanex16_b32 v194 /*v450*/, v194 /*v450*/, s17, 0xfedcba98
	v_dual_max_num_f32 v192 /*v448*/, v192 /*v448*/, v196 /*v452*/ :: v_dual_max_num_f32 v193 /*v449*/, v193 /*v449*/, v194 /*v450*/
	s_set_vgpr_msb 0x5549
	v_sub_f32_e32 v195 /*v451*/, v192 /*v448*/, v88 /*v600*/
	v_max_num_f32_e32 v192 /*v448*/, v192 /*v448*/, v88 /*v600*/
	v_sub_f32_e32 v194 /*v450*/, v193 /*v449*/, v87 /*v599*/
	s_set_vgpr_msb 0x4904
	v_cmp_lt_f32_e32 vcc_lo, 0x41000000, v195 /*v451*/
	s_cmp_eq_u32 vcc_lo, 0
	s_cselect_b32 s2, -1, 0
	s_set_vgpr_msb 0x489
	v_cndmask_b32_e64 v69 /*v581*/, v192 /*v448*/, v88 /*v600*/, s2
	s_set_vgpr_msb 0x8946
	v_cmp_lt_f32_e64 s2, 0x41000000, v194 /*v450*/
	v_max_num_f32_e32 v192 /*v448*/, v87 /*v599*/, v193 /*v449*/
	s_cmp_lg_u32 s2, 0
	s_cselect_b32 s3, -1, 0
	s_cmp_eq_u32 s2, 0
	s_cselect_b32 s2, -1, 0
	s_set_vgpr_msb 0x4689
	v_cndmask_b32_e64 v71 /*v583*/, v192 /*v448*/, v87 /*v599*/, s2
	v_mul_f32_e32 v60 /*v572*/, 0xbfb8aa3b, v69 /*v581*/
	v_mul_f32_e32 v68 /*v580*/, 0xbfb8aa3b, v71 /*v583*/
	s_set_vgpr_msb 0x8962
	v_pk_fma_f32 v[206:207] /*v[462:463]*/, v[2:3] /*v[514:515]*/, s[16:17], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6261
	v_pk_fma_f32 v[192:193] /*v[448:449]*/, v[250:251] /*v[506:507]*/, s[16:17], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[198:199] /*v[454:455]*/, v[254:255] /*v[510:511]*/, s[16:17], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6162
	v_pk_fma_f32 v[216:217] /*v[472:473]*/, v[66:67] /*v[578:579]*/, s[16:17], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[242:243] /*v[498:499]*/, v[92:93] /*v[604:605]*/, s[16:17], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x62a2
	v_pk_fma_f32 v[66:67] /*v[578:579]*/, v[104:105] /*v[616:617]*/, s[16:17], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[92:93] /*v[604:605]*/, v[108:109] /*v[620:621]*/, s[16:17], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa262
	v_pk_fma_f32 v[204:205] /*v[460:461]*/, v[0:1] /*v[512:513]*/, s[16:17], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6241
	v_exp_f32_e32 v222 /*v478*/, v206 /*v462*/
	v_exp_f32_e32 v232 /*v488*/, v207 /*v463*/
	v_nop
	s_set_vgpr_msb 0x4162
	v_pk_fma_f32 v[206:207] /*v[462:463]*/, v[8:9] /*v[520:521]*/, s[16:17], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[212:213] /*v[468:469]*/, v[12:13] /*v[524:525]*/, s[16:17], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[224:225] /*v[480:481]*/, v[20:21] /*v[532:533]*/, s[16:17], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6241
	v_exp_f32_e32 v194 /*v450*/, v193 /*v449*/
	v_exp_f32_e32 v202 /*v458*/, v198 /*v454*/
	v_exp_f32_e32 v208 /*v464*/, v199 /*v455*/
	v_nop
	s_set_vgpr_msb 0x4162
	v_pk_fma_f32 v[198:199] /*v[454:455]*/, v[4:5] /*v[516:517]*/, s[16:17], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v193 /*v449*/, v66 /*v578*/
	v_exp_f32_e32 v195 /*v451*/, v67 /*v579*/
	v_nop
	s_set_vgpr_msb 0x62a2
	v_pk_fma_f32 v[66:67] /*v[578:579]*/, v[110:111] /*v[622:623]*/, s[16:17], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa242
	v_exp_f32_e32 v203 /*v459*/, v92 /*v604*/
	v_exp_f32_e32 v209 /*v465*/, v93 /*v605*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[92:93] /*v[604:605]*/, v[114:115] /*v[626:627]*/, s[16:17], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa261
	v_pk_fma_f32 v[196:197] /*v[452:453]*/, v[252:253] /*v[508:509]*/, s[16:17], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v210 /*v466*/, v204 /*v460*/
	v_exp_f32_e32 v218 /*v474*/, v205 /*v461*/
	v_nop
	s_set_vgpr_msb 0x6162
	v_pk_fma_f32 v[204:205] /*v[460:461]*/, v[6:7] /*v[518:519]*/, s[16:17], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6281
	v_exp_f32_e32 v14 /*v526*/, v206 /*v462*/
	s_set_vgpr_msb 0x8141
	v_exp_f32_e32 v206 /*v462*/, v212 /*v468*/
	v_exp_f32_e32 v214 /*v470*/, v213 /*v469*/
	v_nop
	s_set_vgpr_msb 0x4162
	v_pk_fma_f32 v[212:213] /*v[468:469]*/, v[18:19] /*v[530:531]*/, s[16:17], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6281
	v_exp_f32_e32 v6 /*v518*/, v224 /*v480*/
	v_exp_f32_e32 v18 /*v530*/, v225 /*v481*/
	v_nop
	s_set_vgpr_msb 0x8162
	v_pk_fma_f32 v[224:225] /*v[480:481]*/, v[90:91] /*v[602:603]*/, s[16:17], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x62a2
	v_pk_fma_f32 v[90:91] /*v[602:603]*/, v[106:107] /*v[618:619]*/, s[16:17], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa241
	v_exp_f32_e32 v236 /*v492*/, v198 /*v454*/
	v_exp_f32_e32 v250 /*v506*/, v199 /*v455*/
	v_nop
	s_set_vgpr_msb 0x4162
	v_pk_fma_f32 v[198:199] /*v[454:455]*/, v[10:11] /*v[522:523]*/, s[16:17], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v211 /*v467*/, v66 /*v578*/
	v_exp_f32_e32 v219 /*v475*/, v67 /*v579*/
	v_nop
	s_set_vgpr_msb 0x62a2
	v_pk_fma_f32 v[66:67] /*v[578:579]*/, v[116:117] /*v[628:629]*/, s[16:17], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa242
	v_exp_f32_e32 v237 /*v493*/, v92 /*v604*/
	v_exp_f32_e32 v251 /*v507*/, v93 /*v605*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[92:93] /*v[604:605]*/, v[120:121] /*v[632:633]*/, s[16:17], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa241
	v_exp_f32_e32 v200 /*v456*/, v197 /*v453*/
	s_set_vgpr_msb 0x4142
	v_exp_f32_e32 v197 /*v453*/, v90 /*v602*/
	v_exp_f32_e32 v201 /*v457*/, v91 /*v603*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[90:91] /*v[602:603]*/, v[112:113] /*v[624:625]*/, s[16:17], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa241
	v_exp_f32_e32 v254 /*v510*/, v204 /*v460*/
	s_set_vgpr_msb 0x4181
	v_exp_f32_e32 v10 /*v522*/, v205 /*v461*/
	s_set_vgpr_msb 0x8141
	v_exp_f32_e32 v204 /*v460*/, v199 /*v455*/
	s_set_vgpr_msb 0x4142
	v_exp_f32_e32 v255 /*v511*/, v66 /*v578*/
	s_set_vgpr_msb 0x42a2
	v_exp_f32_e32 v11 /*v523*/, v67 /*v579*/
	v_nop
	v_pk_fma_f32 v[66:67] /*v[578:579]*/, v[122:123] /*v[634:635]*/, s[16:17], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa242
	v_exp_f32_e32 v199 /*v455*/, v92 /*v604*/
	v_exp_f32_e32 v205 /*v461*/, v93 /*v605*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[92:93] /*v[604:605]*/, v[126:127] /*v[638:639]*/, s[16:17], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa242
	v_exp_f32_e32 v223 /*v479*/, v90 /*v602*/
	v_exp_f32_e32 v233 /*v489*/, v91 /*v603*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[90:91] /*v[602:603]*/, v[118:119] /*v[630:631]*/, s[16:17], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa281
	v_exp_f32_e32 v26 /*v538*/, v207 /*v463*/
	s_set_vgpr_msb 0x8162
	v_pk_fma_f32 v[220:221] /*v[476:477]*/, v[16:17] /*v[528:529]*/, s[16:17], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v207 /*v463*/, v66 /*v578*/
	v_exp_f32_e32 v215 /*v471*/, v67 /*v579*/
	v_nop
	s_set_vgpr_msb 0x62a2
	v_pk_fma_f32 v[66:67] /*v[578:579]*/, v[128:129] /*v[640:641]*/, s[16:17], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa242
	v_exp_f32_e32 v229 /*v485*/, v92 /*v604*/
	v_exp_f32_e32 v241 /*v497*/, v93 /*v605*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[92:93] /*v[604:605]*/, v[132:133] /*v[644:645]*/, s[16:17], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v15 /*v527*/, v90 /*v602*/
	v_exp_f32_e32 v27 /*v539*/, v91 /*v603*/
	v_nop
	v_pk_fma_f32 v[90:91] /*v[602:603]*/, v[124:125] /*v[636:637]*/, s[16:17], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa241
	v_exp_f32_e32 v228 /*v484*/, v220 /*v476*/
	v_exp_f32_e32 v240 /*v496*/, v221 /*v477*/
	v_nop
	s_set_vgpr_msb 0x4162
	v_pk_fma_f32 v[220:221] /*v[476:477]*/, v[22:23] /*v[534:535]*/, s[16:17], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v245 /*v501*/, v66 /*v578*/
	s_set_vgpr_msb 0x62a2
	v_exp_f32_e32 v3 /*v515*/, v67 /*v579*/
	v_nop
	v_pk_fma_f32 v[66:67] /*v[578:579]*/, v[134:135] /*v[646:647]*/, s[16:17], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v23 /*v535*/, v92 /*v604*/
	v_exp_f32_e32 v33 /*v545*/, v93 /*v605*/
	v_nop
	v_pk_fma_f32 v[92:93] /*v[604:605]*/, v[138:139] /*v[650:651]*/, s[16:17], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa241
	v_exp_f32_e32 v226 /*v482*/, v217 /*v473*/
	s_set_vgpr_msb 0x4142
	v_exp_f32_e32 v217 /*v473*/, v90 /*v602*/
	v_exp_f32_e32 v227 /*v483*/, v91 /*v603*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[90:91] /*v[602:603]*/, v[130:131] /*v[642:643]*/, s[16:17], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa241
	v_exp_f32_e32 v244 /*v500*/, v212 /*v468*/
	s_set_vgpr_msb 0x4181
	v_exp_f32_e32 v2 /*v514*/, v213 /*v469*/
	v_nop
	s_set_vgpr_msb 0x8162
	v_pk_fma_f32 v[212:213] /*v[468:469]*/, v[24:25] /*v[536:537]*/, s[16:17], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6281
	v_exp_f32_e32 v22 /*v534*/, v220 /*v476*/
	s_set_vgpr_msb 0x8162
	v_pk_fma_f32 v[230:231] /*v[486:487]*/, v[28:29] /*v[540:541]*/, s[16:17], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[238:239] /*v[494:495]*/, v[30:31] /*v[542:543]*/, s[16:17], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6241
	v_exp_f32_e32 v220 /*v476*/, v225 /*v481*/
	s_set_vgpr_msb 0x41a2
	v_exp_f32_e32 v37 /*v549*/, v66 /*v578*/
	v_exp_f32_e32 v45 /*v557*/, v67 /*v579*/
	v_nop
	v_pk_fma_f32 v[66:67] /*v[578:579]*/, v[140:141] /*v[652:653]*/, s[16:17], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa242
	v_exp_f32_e32 v225 /*v481*/, v92 /*v604*/
	v_exp_f32_e32 v235 /*v491*/, v93 /*v605*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[92:93] /*v[604:605]*/, v[144:145] /*v[656:657]*/, s[16:17], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v7 /*v519*/, v90 /*v602*/
	v_exp_f32_e32 v19 /*v531*/, v91 /*v603*/
	v_nop
	v_pk_fma_f32 v[90:91] /*v[602:603]*/, v[136:137] /*v[648:649]*/, s[16:17], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa281
	v_exp_f32_e32 v36 /*v548*/, v212 /*v468*/
	s_set_vgpr_msb 0x8141
	v_exp_f32_e32 v212 /*v468*/, v224 /*v480*/
	v_exp_f32_e32 v224 /*v480*/, v230 /*v486*/
	v_exp_f32_e32 v234 /*v490*/, v231 /*v487*/
	v_nop
	s_set_vgpr_msb 0x4162
	v_pk_fma_f32 v[230:231] /*v[486:487]*/, v[34:35] /*v[546:547]*/, s[16:17], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6241
	v_exp_f32_e32 v252 /*v508*/, v239 /*v495*/
	s_set_vgpr_msb 0x4142
	v_exp_f32_e32 v239 /*v495*/, v66 /*v578*/
	v_exp_f32_e32 v253 /*v509*/, v67 /*v579*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[66:67] /*v[578:579]*/, v[146:147] /*v[658:659]*/, s[16:17], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v17 /*v529*/, v92 /*v604*/
	v_exp_f32_e32 v29 /*v541*/, v93 /*v605*/
	v_nop
	v_pk_fma_f32 v[92:93] /*v[604:605]*/, v[150:151] /*v[662:663]*/, s[16:17], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa281
	v_exp_f32_e32 v32 /*v544*/, v221 /*v477*/
	v_exp_f32_e32 v44 /*v556*/, v213 /*v469*/
	s_set_vgpr_msb 0x8142
	v_exp_f32_e32 v213 /*v469*/, v90 /*v602*/
	v_exp_f32_e32 v221 /*v477*/, v91 /*v603*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[90:91] /*v[602:603]*/, v[142:143] /*v[654:655]*/, s[16:17], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa281
	v_exp_f32_e32 v0 /*v512*/, v242 /*v498*/
	v_exp_f32_e32 v12 /*v524*/, v243 /*v499*/
	v_exp_f32_e32 v16 /*v528*/, v230 /*v486*/
	s_set_vgpr_msb 0x8162
	v_pk_fma_f32 v[242:243] /*v[498:499]*/, v[38:39] /*v[550:551]*/, s[16:17], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6281
	v_exp_f32_e32 v28 /*v540*/, v231 /*v487*/
	v_nop
	s_set_vgpr_msb 0x8162
	v_pk_fma_f32 v[230:231] /*v[486:487]*/, v[40:41] /*v[552:553]*/, s[16:17], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x62a2
	v_pk_fma_f32 v[8:9] /*v[520:521]*/, v[46:47] /*v[558:559]*/, s[16:17], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v31 /*v543*/, v66 /*v578*/
	v_exp_f32_e32 v41 /*v553*/, v67 /*v579*/
	v_nop
	v_pk_fma_f32 v[66:67] /*v[578:579]*/, v[152:153] /*v[664:665]*/, s[16:17], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v53 /*v565*/, v92 /*v604*/
	v_exp_f32_e32 v59 /*v571*/, v93 /*v605*/
	v_nop
	v_pk_fma_f32 v[92:93] /*v[604:605]*/, v[156:157] /*v[668:669]*/, s[16:17], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa241
	v_exp_f32_e32 v192 /*v448*/, v192 /*v448*/
	s_set_vgpr_msb 0x4162
	v_pk_fma_f32 v[246:247] /*v[502:503]*/, v[94:95] /*v[606:607]*/, s[16:17], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x62a2
	v_exp_f32_e32 v1 /*v513*/, v90 /*v602*/
	v_exp_f32_e32 v13 /*v525*/, v91 /*v603*/
	v_nop
	v_pk_fma_f32 v[90:91] /*v[602:603]*/, v[148:149] /*v[660:661]*/, s[16:17], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa281
	v_exp_f32_e32 v50 /*v562*/, v243 /*v499*/
	v_exp_f32_e32 v58 /*v570*/, v231 /*v487*/
	s_set_vgpr_msb 0x81a2
	v_pk_fma_f32 v[24:25] /*v[536:537]*/, v[48:49] /*v[560:561]*/, s[16:17], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v20 /*v532*/, v9 /*v521*/
	v_pk_fma_f32 v[48:49] /*v[560:561]*/, v[102:103] /*v[614:615]*/, s[16:17], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa242
	v_exp_f32_e32 v231 /*v487*/, v66 /*v578*/
	v_exp_f32_e32 v243 /*v499*/, v67 /*v579*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[66:67] /*v[578:579]*/, v[158:159] /*v[670:671]*/, s[16:17], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v9 /*v521*/, v92 /*v604*/
	v_exp_f32_e32 v21 /*v533*/, v93 /*v605*/
	v_nop
	v_pk_fma_f32 v[92:93] /*v[604:605]*/, v[162:163] /*v[674:675]*/, s[16:17], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa241
	v_exp_f32_e32 v196 /*v452*/, v196 /*v452*/
	v_exp_f32_e32 v238 /*v494*/, v238 /*v494*/
	s_set_vgpr_msb 0x4181
	v_exp_f32_e32 v30 /*v542*/, v246 /*v502*/
	v_exp_f32_e32 v40 /*v552*/, v247 /*v503*/
	v_nop
	s_set_vgpr_msb 0x8162
	v_pk_fma_f32 v[246:247] /*v[502:503]*/, v[96:97] /*v[608:609]*/, s[16:17], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x62a2
	v_pk_fma_f32 v[4:5] /*v[516:517]*/, v[98:99] /*v[610:611]*/, s[16:17], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v43 /*v555*/, v90 /*v602*/
	v_exp_f32_e32 v51 /*v563*/, v91 /*v603*/
	v_nop
	v_pk_fma_f32 v[90:91] /*v[602:603]*/, v[154:155] /*v[666:667]*/, s[16:17], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v34 /*v546*/, v25 /*v537*/
	v_pk_fma_f32 v[62:63] /*v[574:575]*/, v[54:55] /*v[566:567]*/, s[16:17], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v54 /*v566*/, v49 /*v561*/
	v_exp_f32_e32 v25 /*v537*/, v66 /*v578*/
	v_exp_f32_e32 v35 /*v547*/, v67 /*v579*/
	v_exp_f32_e32 v49 /*v561*/, v92 /*v604*/
	v_pk_fma_f32 v[66:67] /*v[578:579]*/, v[164:165] /*v[676:677]*/, s[16:17], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v55 /*v567*/, v93 /*v605*/
	v_nop
	s_set_vgpr_msb 0xa285
	v_pk_add_f32 v[92:93] /*v[604:605]*/, v[192:193] /*v[448:449]*/, v[194:195] /*v[450:451]*/
	v_pk_add_f32 v[94:95] /*v[606:607]*/, v[200:201] /*v[456:457]*/, v[202:203] /*v[458:459]*/
	s_set_vgpr_msb 0x8541
	v_exp_f32_e32 v198 /*v454*/, v198 /*v454*/
	s_set_vgpr_msb 0x4181
	v_exp_f32_e32 v42 /*v554*/, v242 /*v498*/
	v_exp_f32_e32 v52 /*v564*/, v230 /*v486*/
	s_set_vgpr_msb 0x8141
	v_exp_f32_e32 v230 /*v486*/, v246 /*v502*/
	v_exp_f32_e32 v242 /*v498*/, v247 /*v503*/
	s_set_vgpr_msb 0x4142
	v_exp_f32_e32 v246 /*v502*/, v4 /*v516*/
	s_set_vgpr_msb 0x42a2
	v_exp_f32_e32 v4 /*v516*/, v5 /*v517*/
	v_pk_fma_f32 v[38:39] /*v[550:551]*/, v[100:101] /*v[612:613]*/, s[16:17], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa242
	v_exp_f32_e32 v247 /*v503*/, v90 /*v602*/
	s_set_vgpr_msb 0x42a2
	v_exp_f32_e32 v5 /*v517*/, v91 /*v603*/
	v_nop
	v_pk_fma_f32 v[90:91] /*v[602:603]*/, v[160:161] /*v[672:673]*/, s[16:17], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[64:65] /*v[576:577]*/, v[56:57] /*v[568:569]*/, s[16:17], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v57 /*v569*/, v66 /*v578*/
	v_exp_f32_e32 v61 /*v573*/, v67 /*v579*/
	v_nop
	s_set_vgpr_msb 0xa289
	v_pk_add_f32 v[66:67] /*v[578:579]*/, v[196:197] /*v[452:453]*/, v[92:93] /*v[604:605]*/
	v_pk_add_f32 v[92:93] /*v[604:605]*/, v[208:209] /*v[464:465]*/, v[94:95] /*v[606:607]*/
	s_set_vgpr_msb 0x8985
	v_pk_add_f32 v[94:95] /*v[606:607]*/, v[210:211] /*v[466:467]*/, v[218:219] /*v[474:475]*/
	v_pk_add_f32 v[96:97] /*v[608:609]*/, v[232:233] /*v[488:489]*/, v[236:237] /*v[492:493]*/
	s_set_vgpr_msb 0x8589
	v_pk_add_f32 v[98:99] /*v[610:611]*/, v[254:255] /*v[510:511]*/, v[10:11] /*v[522:523]*/
	s_set_vgpr_msb 0x898a
	v_pk_add_f32 v[108:109] /*v[620:621]*/, v[18:19] /*v[530:531]*/, v[22:23] /*v[534:535]*/
	v_pk_add_f32 v[110:111] /*v[622:623]*/, v[36:37] /*v[548:549]*/, v[44:45] /*v[556:557]*/
	s_set_vgpr_msb 0x8a85
	v_pk_add_f32 v[114:115] /*v[626:627]*/, v[238:239] /*v[494:495]*/, v[252:253] /*v[508:509]*/
	s_set_vgpr_msb 0x858a
	v_pk_add_f32 v[116:117] /*v[628:629]*/, v[12:13] /*v[524:525]*/, v[16:17] /*v[528:529]*/
	s_set_vgpr_msb 0x8a41
	v_exp_f32_e32 v216 /*v472*/, v216 /*v472*/
	s_set_vgpr_msb 0x4186
	v_exp_f32_e32 v8 /*v520*/, v8 /*v520*/
	v_exp_f32_e32 v24 /*v536*/, v24 /*v536*/
	v_exp_f32_e32 v46 /*v558*/, v39 /*v551*/
	v_exp_f32_e32 v48 /*v560*/, v48 /*v560*/
	v_exp_f32_e32 v47 /*v559*/, v91 /*v603*/
	v_pk_add_f32 v[100:101] /*v[612:613]*/, v[26:27] /*v[538:539]*/, v[198:199] /*v[454:455]*/
	s_set_vgpr_msb 0x8685
	v_pk_add_f32 v[102:103] /*v[614:615]*/, v[206:207] /*v[462:463]*/, v[214:215] /*v[470:471]*/
	s_set_vgpr_msb 0x8589
	v_pk_add_f32 v[94:95] /*v[606:607]*/, v[222:223] /*v[478:479]*/, v[94:95] /*v[606:607]*/
	v_pk_add_f32 v[96:97] /*v[608:609]*/, v[250:251] /*v[506:507]*/, v[96:97] /*v[608:609]*/
	s_set_vgpr_msb 0x898a
	v_pk_add_f32 v[98:99] /*v[610:611]*/, v[14:15] /*v[526:527]*/, v[98:99] /*v[610:611]*/
	s_set_vgpr_msb 0x8a85
	v_pk_add_f32 v[104:105] /*v[616:617]*/, v[226:227] /*v[482:483]*/, v[228:229] /*v[484:485]*/
	v_pk_add_f32 v[112:113] /*v[624:625]*/, v[220:221] /*v[476:477]*/, v[224:225] /*v[480:481]*/
	s_set_vgpr_msb 0x858a
	v_pk_add_f32 v[108:109] /*v[620:621]*/, v[32:33] /*v[544:545]*/, v[108:109] /*v[620:621]*/
	s_set_vgpr_msb 0x8a89
	v_pk_add_f32 v[110:111] /*v[622:623]*/, v[212:213] /*v[468:469]*/, v[110:111] /*v[622:623]*/
	s_set_vgpr_msb 0x898a
	v_pk_add_f32 v[118:119] /*v[630:631]*/, v[30:31] /*v[542:543]*/, v[40:41] /*v[552:553]*/
	v_pk_add_f32 v[120:121] /*v[632:633]*/, v[50:51] /*v[562:563]*/, v[52:53] /*v[564:565]*/
	s_set_vgpr_msb 0x8a85
	v_pk_add_f32 v[122:123] /*v[634:635]*/, v[230:231] /*v[486:487]*/, v[242:243] /*v[498:499]*/
	s_set_vgpr_msb 0x85aa
	v_pk_add_f32 v[114:115] /*v[626:627]*/, v[0:1] /*v[512:513]*/, v[114:115] /*v[626:627]*/
	v_pk_add_f32 v[116:117] /*v[628:629]*/, v[28:29] /*v[540:541]*/, v[116:117] /*v[628:629]*/
	v_pk_add_f32 v[66:67] /*v[578:579]*/, v[66:67] /*v[578:579]*/, v[92:93] /*v[604:605]*/
	v_exp_f32_e32 v38 /*v550*/, v38 /*v550*/
	v_exp_f32_e32 v56 /*v568*/, v62 /*v574*/
	v_exp_f32_e32 v60 /*v572*/, v63 /*v575*/
	v_exp_f32_e32 v39 /*v551*/, v90 /*v602*/
	v_nop
	v_pk_fma_f32 v[90:91] /*v[602:603]*/, v[166:167] /*v[678:679]*/, s[16:17], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xaa89
	v_pk_add_f32 v[100:101] /*v[612:613]*/, v[204:205] /*v[460:461]*/, v[100:101] /*v[612:613]*/
	v_pk_add_f32 v[102:103] /*v[614:615]*/, v[216:217] /*v[472:473]*/, v[102:103] /*v[614:615]*/
	v_pk_add_f32 v[106:107] /*v[618:619]*/, v[244:245] /*v[500:501]*/, v[2:3] /*v[514:515]*/
	v_pk_add_f32 v[104:105] /*v[616:617]*/, v[240:241] /*v[496:497]*/, v[104:105] /*v[616:617]*/
	v_pk_add_f32 v[112:113] /*v[624:625]*/, v[234:235] /*v[490:491]*/, v[112:113] /*v[624:625]*/
	s_set_vgpr_msb 0x898a
	v_pk_add_f32 v[118:119] /*v[630:631]*/, v[42:43] /*v[554:555]*/, v[118:119] /*v[630:631]*/
	v_pk_add_f32 v[120:121] /*v[632:633]*/, v[58:59] /*v[570:571]*/, v[120:121] /*v[632:633]*/
	s_set_vgpr_msb 0x8a89
	v_pk_add_f32 v[122:123] /*v[634:635]*/, v[246:247] /*v[502:503]*/, v[122:123] /*v[634:635]*/
	s_set_vgpr_msb 0x898a
	v_pk_add_f32 v[124:125] /*v[636:637]*/, v[4:5] /*v[516:517]*/, v[8:9] /*v[520:521]*/
	v_pk_add_f32 v[126:127] /*v[638:639]*/, v[24:25] /*v[536:537]*/, v[34:35] /*v[546:547]*/
	v_pk_add_f32 v[128:129] /*v[640:641]*/, v[46:47] /*v[558:559]*/, v[48:49] /*v[560:561]*/
	v_pk_add_f32 v[66:67] /*v[578:579]*/, v[94:95] /*v[606:607]*/, v[66:67] /*v[578:579]*/
	v_pk_add_f32 v[94:95] /*v[606:607]*/, v[96:97] /*v[608:609]*/, v[98:99] /*v[610:611]*/
	v_pk_add_f32 v[96:97] /*v[608:609]*/, v[108:109] /*v[620:621]*/, v[110:111] /*v[622:623]*/
	v_pk_add_f32 v[98:99] /*v[610:611]*/, v[114:115] /*v[626:627]*/, v[116:117] /*v[628:629]*/
	v_exp_f32_e32 v62 /*v574*/, v64 /*v576*/
	v_exp_f32_e32 v63 /*v575*/, v90 /*v602*/
	v_pk_add_f32 v[106:107] /*v[618:619]*/, v[6:7] /*v[518:519]*/, v[106:107] /*v[618:619]*/
	v_pk_add_f32 v[130:131] /*v[642:643]*/, v[56:57] /*v[568:569]*/, v[60:61] /*v[572:573]*/
	v_pk_add_f32 v[92:93] /*v[604:605]*/, v[20:21] /*v[532:533]*/, v[124:125] /*v[636:637]*/
	v_pk_add_f32 v[124:125] /*v[636:637]*/, v[38:39] /*v[550:551]*/, v[126:127] /*v[638:639]*/
	v_pk_add_f32 v[126:127] /*v[638:639]*/, v[54:55] /*v[566:567]*/, v[128:129] /*v[640:641]*/
	v_pk_add_f32 v[102:103] /*v[614:615]*/, v[102:103] /*v[614:615]*/, v[104:105] /*v[616:617]*/
	v_pk_add_f32 v[104:105] /*v[616:617]*/, v[120:121] /*v[632:633]*/, v[122:123] /*v[634:635]*/
	v_pk_add_f32 v[94:95] /*v[606:607]*/, v[100:101] /*v[612:613]*/, v[94:95] /*v[606:607]*/
	v_pk_add_f32 v[96:97] /*v[608:609]*/, v[112:113] /*v[624:625]*/, v[96:97] /*v[608:609]*/
	v_pk_add_f32 v[98:99] /*v[610:611]*/, v[118:119] /*v[630:631]*/, v[98:99] /*v[610:611]*/
	v_pk_add_f32 v[128:129] /*v[640:641]*/, v[62:63] /*v[574:575]*/, v[130:131] /*v[642:643]*/
	v_pk_add_f32 v[100:101] /*v[612:613]*/, v[106:107] /*v[618:619]*/, v[102:103] /*v[614:615]*/
	v_pk_add_f32 v[92:93] /*v[604:605]*/, v[92:93] /*v[604:605]*/, v[104:105] /*v[616:617]*/
	v_pk_add_f32 v[102:103] /*v[614:615]*/, v[124:125] /*v[636:637]*/, v[126:127] /*v[638:639]*/
	v_pk_add_f32 v[66:67] /*v[578:579]*/, v[66:67] /*v[578:579]*/, v[94:95] /*v[606:607]*/
	v_pk_add_f32 v[94:95] /*v[606:607]*/, v[96:97] /*v[608:609]*/, v[98:99] /*v[610:611]*/
	v_exp_f32_e32 v64 /*v576*/, v65 /*v577*/
	v_exp_f32_e32 v65 /*v577*/, v91 /*v603*/
	v_nop
	v_pk_add_f32 v[90:91] /*v[602:603]*/, v[128:129] /*v[640:641]*/, v[102:103] /*v[614:615]*/
	v_pk_add_f32 v[66:67] /*v[578:579]*/, v[100:101] /*v[612:613]*/, v[66:67] /*v[578:579]*/
	v_pk_add_f32 v[92:93] /*v[604:605]*/, v[92:93] /*v[604:605]*/, v[94:95] /*v[606:607]*/
	v_sub_f32_e32 v68 /*v580*/, v88 /*v600*/, v69 /*v581*/
	v_pk_add_f32 v[90:91] /*v[602:603]*/, v[64:65] /*v[576:577]*/, v[90:91] /*v[602:603]*/
	v_pk_add_f32 v[66:67] /*v[578:579]*/, v[66:67] /*v[578:579]*/, v[92:93] /*v[604:605]*/
	v_mul_f32_e32 v68 /*v580*/, 0x3fb8aa3b, v68 /*v580*/
	v_pk_add_f32 v[66:67] /*v[578:579]*/, v[90:91] /*v[602:603]*/, v[66:67] /*v[578:579]*/
	v_exp_f32_e32 v68 /*v580*/, v68 /*v580*/
	v_dual_mov_b32 v88 /*v600*/, v66 /*v578*/ :: v_dual_mov_b32 v90 /*v602*/, v67 /*v579*/
	v_permlanex16_b32 v88 /*v600*/, v88 /*v600*/, s17, 0xfedcba98
	v_permlanex16_b32 v90 /*v602*/, v90 /*v602*/, s17, 0xfedcba98
	s_set_vgpr_msb 0x8a00
	s_cbranch_vccz .LBB0_62
	s_set_vgpr_msb 8
	v_pk_mul_f32 v[126:127], v[126:127], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[124:125], v[124:125], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[122:123], v[122:123], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[120:121], v[120:121], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[118:119], v[118:119], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[116:117], v[116:117], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[114:115], v[114:115], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[112:113], v[112:113], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[110:111], v[110:111], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[108:109], v[108:109], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[106:107], v[106:107], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[104:105], v[104:105], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[102:103], v[102:103], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[100:101], v[100:101], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[98:99], v[98:99], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[96:97], v[96:97], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[94:95], v[94:95], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[92:93], v[92:93], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[90:91], v[90:91], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[88:89], v[88:89], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[86:87], v[86:87], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[84:85], v[84:85], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[82:83], v[82:83], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[80:81], v[80:81], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[78:79], v[78:79], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[76:77], v[76:77], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[74:75], v[74:75], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[72:73], v[72:73], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[70:71], v[70:71], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[68:69], v[68:69], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[66:67], v[66:67], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[64:65], v[64:65], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x800
.LBB0_62:
	s_set_vgpr_msb 0x8a
	v_sub_f32_e32 v70 /*v582*/, v87 /*v599*/, v71 /*v583*/
	s_and_b32 s2, s3, exec_lo
	s_cselect_b32 s2, 1, 0
	s_cmp_lg_u32 s2, 1
	v_mul_f32_e32 v70 /*v582*/, 0x3fb8aa3b, v70 /*v582*/
	v_exp_f32_e32 v70 /*v582*/, v70 /*v582*/
	s_set_vgpr_msb 0x8a00
	s_cbranch_scc1 .LBB0_57
	v_nop
	s_set_vgpr_msb 8
	v_pk_mul_f32 v[62:63], v[62:63], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[60:61], v[60:61], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59], v[58:59], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[56:57], v[56:57], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[54:55], v[54:55], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[52:53], v[52:53], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[50:51], v[50:51], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[48:49], v[48:49], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[46:47], v[46:47], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[44:45], v[44:45], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[42:43], v[42:43], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[40:41], v[40:41], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39], v[38:39], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[36:37], v[36:37], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35], v[34:35], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[32:33], v[32:33], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[30:31], v[30:31], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[28:29], v[28:29], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[26:27], v[26:27], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25], v[24:25], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[22:23], v[22:23], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21], v[20:21], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[18:19], v[18:19], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[16:17], v[16:17], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[14:15], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13], v[12:13], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[10:11], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[8:9], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[6:7], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[4:5], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[2:3], v[2:3], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[0:1], v[0:1], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x800
	s_branch .LBB0_57
.LBB0_64:
	s_set_vgpr_msb 0x82
	v_dual_mov_b32 v71 /*v583*/, v87 /*v599*/ :: v_dual_mov_b32 v69 /*v581*/, v88 /*v600*/
	s_set_vgpr_msb 0x8200
.LBB0_65:
	s_set_vgpr_msb 5
	v_div_scale_f32 v129, null, v248 /*v504*/, v248 /*v504*/, 1.0
	v_div_scale_f32 v132, vcc_lo, 1.0, v248 /*v504*/, 1.0
	v_div_scale_f32 v150, null, v249 /*v505*/, v249 /*v505*/, 1.0
	s_sub_co_i32 s2, s10, s60
	s_set_vgpr_msb 0x508
	v_mul_u32_u24_e32 v128, 0x110, v76 /*v588*/
	v_rcp_f32_e32 v130, v129
	s_lshr_b32 s3, s2, 31
	v_rcp_f32_e32 v151, v150
	s_add_co_i32 s3, s2, s3
	s_mov_b32 s4, 0
	s_and_b32 s3, s3, -2
	s_wait_dscnt 0x0
	s_sub_co_i32 s2, s2, s3
	s_set_vgpr_msb 0x800
	v_fma_f32 v131, -v129, v130, 1.0
	s_lshl_b32 s2, s2, 17
	s_add_co_i32 s2, s2, s96
	s_set_vgpr_msb 8
	v_add_nc_u32_e32 v152, s2, v78 /*v590*/
	s_set_vgpr_msb 0x800
	v_fmac_f32_e32 v130, v131, v130
	v_mul_f32_e32 v131, v132, v130
	v_fma_f32 v133, -v129, v131, v132
	v_fmac_f32_e32 v131, v133, v130
	v_fma_f32 v129, -v129, v131, v132
	v_div_fmas_f32 v129, v129, v130, v131
	s_set_vgpr_msb 4
	v_cmp_lt_f32_e32 vcc_lo, 0, v248 /*v504*/
	s_set_vgpr_msb 0x400
	v_fma_f32 v131, -v150, v151, 1.0
	s_set_vgpr_msb 4
	v_div_fixup_f32 v129, v129, v248 /*v504*/, 1.0
	s_set_vgpr_msb 0x400
	v_cndmask_b32_e32 v130, 0, v129, vcc_lo
	s_set_vgpr_msb 4
	v_div_scale_f32 v129, vcc_lo, 1.0, v249 /*v505*/, 1.0
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
	v_cmp_lt_f32_e32 vcc_lo, 0, v249 /*v505*/
	s_set_vgpr_msb 0x400
	v_pk_mul_f32 v[120:121], v[120:121], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[122:123], v[122:123], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[124:125], v[124:125], v[130:131] op_sel_hi:[1,0]
	s_set_vgpr_msb 4
	v_div_fixup_f32 v92, v88, v249 /*v505*/, 1.0
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
	s_set_vgpr_msb 8
	v_dual_lshrrev_b32 v26, 4, v77 /*v589*/ :: v_dual_bitop2_b32 v0, 28, v72 /*v584*/ bitop3:0x54
	s_add_co_i32 s3, s3, -1
	s_set_vgpr_msb 0x800
	v_add_nc_u32_e32 v2, s2, v128
	s_min_u32 s3, s3, 31
	v_or_b32_e32 v1, 30, v26
	v_min_u32_e32 v3, s3, v0
	s_set_vgpr_msb 8
	v_dual_lshlrev_b32 v4, 8, v76 /*v588*/ :: v_dual_lshlrev_b32 v0, 3, v76 /*v588*/
	s_set_vgpr_msb 0x800
	v_or_b32_e32 v5, 26, v26
	v_min_u32_e32 v6, s3, v1
	v_dual_sub_nc_u32 v32, v2, v4 :: v_dual_bitop2_b32 v7, s95, v3 bitop3:0x54
	v_mov_b32_e32 v1, 0
	v_min_u32_e32 v9, s3, v5
	v_dual_ashrrev_i32 v10, 31, v7 :: v_dual_bitop2_b32 v8, s95, v6 bitop3:0x54
	s_set_vgpr_msb 8
	v_or_b32_e32 v5, 24, v72 /*v584*/
	s_set_vgpr_msb 0x800
	v_mad_u32_u24 v33, 0x110, v3, v32
	v_mad_u32_u24 v34, 0x110, v6, v32
	v_dual_ashrrev_i32 v2, 31, v8 :: v_dual_bitop2_b32 v11, s95, v9 bitop3:0x54
	v_lshrrev_b32_e32 v4, 30, v10
	v_min_u32_e32 v14, s3, v5
	v_mad_u32_u24 v35, 0x110, v9, v32
	s_set_vgpr_msb 8
	v_or_b32_e32 v21, 16, v72 /*v584*/
	s_set_vgpr_msb 0x800
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
	s_set_vgpr_msb 8
	v_or_b32_e32 v13, 20, v72 /*v584*/
	v_cndmask_b32_e64 v11, 0, 1, s2
	s_set_vgpr_msb 0x800
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
	s_set_vgpr_msb 8
	v_or_b32_e32 v30, 8, v72 /*v584*/
	v_mad_nc_i64_i32 v[8:9], v12, s8, v[8:9]
	s_set_vgpr_msb 0x800
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
	s_set_vgpr_msb 8
	v_or_b32_e32 v27, 12, v72 /*v584*/
	v_cndmask_b32_e64 v28, 0, 1, s2
	v_mad_nc_i64_i32 v[14:15], v14, s12, v[0:1]
	v_mad_nc_i64_i32 v[16:17], v16, s12, v[0:1]
	s_set_vgpr_msb 0x800
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
	s_set_vgpr_msb 8
	v_or_b32_e32 v31, 4, v72 /*v584*/
	s_set_vgpr_msb 0x800
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
	s_set_vgpr_msb 8
	v_min_i32_e32 v47, s3, v72 /*v584*/
	s_set_vgpr_msb 0x800
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
	s_set_vgpr_msb 6
	v_cmp_gt_f32_e32 vcc_lo, 0x800000, v248 /*v504*/
	v_cmp_gt_f32_e64 s2, 0x800000, v249 /*v505*/
	s_wait_alu depctr_vm_vsrc(0)
	v_mul_lo_u32 v4, v75 /*v587*/, s14
	s_load_b64 s[6:7], s[0:1], 0x40 nv
	s_mul_i32 s8, s33, s15
	v_cndmask_b32_e64 v2, 0, 32, vcc_lo
	v_cndmask_b32_e64 v3, 0, 32, s2
	v_cndmask_b32_e64 v0, 0, 0x42000000, vcc_lo
	v_cndmask_b32_e64 v1, 0, 0x42000000, s2
	v_mad_u32 v5, v74 /*v586*/, s13, s8
	s_set_vgpr_msb 0x601
	v_ldexp_f32 v2, v248 /*v504*/, v2
	v_ldexp_f32 v3, v249 /*v505*/, v3
	s_set_vgpr_msb 0x10a
	v_mul_lo_u32 v6, v73 /*v585*/, s13
	v_cmp_eq_u32_e32 vcc_lo, 0, v72 /*v584*/
	v_cmp_gt_i32_e64 s0, s18, v74 /*v586*/
	s_set_vgpr_msb 0xa00
	v_log_f32_e32 v2, v2
	v_log_f32_e32 v3, v3
	v_add_nc_u32_e32 v7, s8, v4
	s_set_vgpr_msb 8
	v_cmp_gt_i32_e64 s1, s18, v73 /*v585*/
	s_add_co_i32 s2, s8, s15
	s_and_b32 s0, vcc_lo, s0
	s_ashr_i32 s3, s2, 31
	s_lshl_b32 s5, s2, 27
	s_and_b32 vcc_lo, vcc_lo, s1
	s_set_vgpr_msb 0x800
	v_dual_sub_f32 v1, v3, v1 :: v_dual_sub_f32 v0, v2, v0
	v_add_lshl_u32 v2, v5, v4, 2
	v_add_lshl_u32 v3, v7, v6, 2
	s_lshr_b64 s[2:3], s[2:3], 5
	v_dual_mul_f32 v1, 0x3f317218, v1 :: v_dual_mul_f32 v0, 0x3f317218, v0
	v_cndmask_b32_e64 v2, 0x7fffffff, v2, s0
	v_cndmask_b32_e32 v3, 0x7fffffff, v3, vcc_lo
	s_and_b64 s[2:3], s[2:3], 0x1ffffffffffffff
	s_set_vgpr_msb 8
	v_dual_add_f32 v1, v1, v71 /*v583*/ :: v_dual_add_f32 v0, v0, v69 /*v581*/
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
		.amdhsa_next_free_vgpr 684
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

	.set .Lkn_fmha_fwd_prefill_a16w16_m32x8_bshd_0.num_vgpr, 684
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
    .vgpr_count:     684
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa-unknown-gfx1250
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
