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
	s_clause 0x1
	s_load_b96 s[12:14], s[0:1], 0xa0 nv
	s_load_b96 s[76:78], s[0:1], 0x90 nv
	s_bfe_u32 s2, ttmp6, 0x40010
	s_and_b32 s3, ttmp7, 0xffff
	s_add_co_i32 s2, s2, 1
	s_bfe_u32 s5, ttmp6, 0x4000c
	s_mul_i32 s2, s3, s2
	s_bfe_u32 s4, ttmp6, 0x40004
	s_add_co_i32 s5, s5, 1
	s_bfe_u32 s6, ttmp6, 0x40014
	s_add_co_i32 s4, s4, s2
	s_and_b32 s2, ttmp6, 15
	s_mul_i32 s5, ttmp9, s5
	s_lshr_b32 s7, ttmp7, 16
	s_add_co_i32 s6, s6, 1
	s_add_co_i32 s2, s2, s5
	s_mul_i32 s5, s7, s6
	s_bfe_u32 s6, ttmp6, 0x40008
	s_getreg_b32 s8, hwreg(HW_REG_IB_STS2, 6, 4)
	s_add_co_i32 s6, s6, s5
	s_cmp_eq_u32 s8, 0
	v_and_b32_e32 v229, 15, v0
	s_cselect_b32 s5, s7, s6
	s_wait_kmcnt 0x0
	s_mul_i32 s6, s14, s13
	s_cselect_b32 s2, ttmp9, s2
	s_cselect_b32 s3, s3, s4
	s_abs_i32 s4, s6
	s_mul_i32 s5, s13, s5
	s_cvt_f32_u32 s7, s4
	s_add_co_i32 s3, s5, s3
	s_sub_co_i32 s5, 0, s4
	s_mul_i32 s3, s3, s12
	v_s_rcp_f32 s7, s7
	s_add_co_i32 s3, s3, s2
	v_and_b32_e32 v1, 16, v0
	v_bfe_u32 v225, v0, 4, 1
	s_clause 0x2
	s_load_b64 s[16:17], s[0:1], 0x10 nv
	s_load_b64 s[74:75], s[0:1], 0x20 nv
	s_load_b64 s[90:91], s[0:1], 0x30 nv
	s_mov_b32 s19, 0
	s_mov_b32 s28, 1
	s_mov_b32 s37, 0xffff0000
	s_mul_f32 s7, s7, 0x4f7ffffe
	s_mov_b32 s36, 0x7510000
	s_mov_b32 s48, 8
	s_mov_b32 s47, 0x800000
	s_cvt_u32_f32 s7, s7
	s_mov_b32 s40, 0x40004
	s_mov_b32 s39, 0x807fff
	s_mov_b32 s38, 0xffff7fff
	s_mul_i32 s5, s5, s7
	s_mov_b32 s45, s37
	s_mul_hi_u32 s2, s7, s5
	s_abs_i32 s5, s3
	s_add_co_i32 s7, s7, s2
	s_mov_b32 s51, s19
	s_mul_hi_u32 s2, s5, s7
	s_xor_b32 s7, s3, s6
	s_mul_i32 s8, s2, s4
	s_ashr_i32 s9, s7, 31
	s_sub_co_i32 s5, s5, s8
	s_add_co_i32 s8, s2, 1
	s_sub_co_i32 s10, s5, s4
	s_cmp_ge_u32 s5, s4
	s_mov_b32 s52, 0xf510000
	s_cselect_b32 s2, s8, s2
	s_cselect_b32 s5, s10, s5
	s_add_co_i32 s8, s2, 1
	s_cmp_ge_u32 s5, s4
	s_mov_b32 s53, s37
	s_cselect_b32 s2, s8, s2
	s_mov_b32 s55, s47
	s_xor_b32 s2, s2, s9
	s_mov_b32 s56, s48
	s_sub_co_i32 s4, s2, s9
	s_mov_b32 s59, s19
	s_mul_i32 s4, s4, s6
	v_lshlrev_b32_e32 v233, 3, v225
	s_cmp_lg_u32 s3, s4
	s_cselect_b32 s4, -1, 0
	s_cmp_lt_i32 s7, 0
	s_cselect_b32 s5, -1, 0
	s_and_b32 s4, s4, s5
	s_sub_co_ci_u32 s2, s2, s9
	s_abs_i32 s4, s13
	s_mul_i32 s6, s2, s6
	s_cvt_f32_u32 s5, s4
	s_sub_co_i32 s7, 0, s4
	s_sub_co_i32 s14, s3, s6
	v_s_rcp_f32 s5, s5
	s_abs_i32 s6, s14
	s_xor_b32 s15, s14, s13
	s_ashr_i32 s18, s15, 31
	s_mul_f32 s5, s5, 0x4f7ffffe
	s_cvt_u32_f32 s5, s5
	s_mul_i32 s7, s7, s5
	s_mul_hi_u32 s3, s5, s7
	s_add_co_i32 s5, s5, s3
	s_mul_hi_u32 s3, s6, s5
	s_mul_i32 s5, s3, s4
	s_sub_co_i32 s5, s6, s5
	s_add_co_i32 s6, s3, 1
	s_sub_co_i32 s7, s5, s4
	s_cmp_ge_u32 s5, s4
	s_cselect_b32 s3, s6, s3
	s_cselect_b32 s5, s7, s5
	s_add_co_i32 s6, s3, 1
	s_cmp_ge_u32 s5, s4
	s_cselect_b32 s3, s6, s3
	s_load_b256 s[4:11], s[0:1], 0x5c nv
	s_xor_b32 s20, s3, s18
	s_sub_co_i32 s3, s20, s18
	s_mul_i32 s13, s3, s13
	s_mov_b32 s3, s19
	s_cmp_lg_u32 s14, s13
	s_cselect_b32 s21, -1, 0
	s_cmp_lt_i32 s15, 0
	s_cselect_b32 s15, -1, 0
	s_and_b32 s15, s21, s15
	s_sub_co_ci_u32 s79, s20, s18
	s_bfe_u32 s18, ttmp8, 0x50019
	s_mul_i32 s72, s79, s77
	s_and_b32 s15, s18, 28
	s_lshr_b32 s33, s18, 2
	s_cmp_lg_u32 s18, s15
	s_wait_kmcnt 0x0
	s_mov_b32 s22, s9
	s_cselect_b32 s15, -1, 0
	s_cmp_lt_i32 s18, 0
	s_mul_i32 s84, s79, s78
	s_cselect_b32 s20, -1, 0
	s_mov_b32 s80, s6
	s_and_b32 s15, s20, s15
	s_mov_b32 s20, s5
	s_and_b32 s15, s15, exec_lo
	s_cselect_b32 s44, 1, 0
	s_not_b32 s2, s2
	s_sub_co_i32 s14, s14, s13
	s_add_co_i32 s2, s12, s2
	s_lshl_b32 s34, s14, 2
	s_lshl_b32 s93, s2, 7
	s_mul_i32 s15, s18, 0x1100
	s_lshl4_add_u32 vcc_hi, s18, s93
	s_set_vgpr_msb 64
	v_writelane_b32 v0 /*v256*/, s15, 0
	s_set_vgpr_msb 0x4000
	v_or_b32_e32 v226, vcc_hi, v229
	s_cmp_lt_i32 vcc_hi, 0
	s_mov_b32 s12, s10
	s_cselect_b32 s104, -1, 0
	s_add_co_i32 s29, s15, 0x20000
	v_dual_ashrrev_i32 v0, 31, v226 :: v_dual_bitop2_b32 v230, 31, v0 bitop3:0x40
	s_ashr_i32 s13, vcc_hi, 2
	s_ashr_i32 s21, s5, 31
	s_add_co_i32 s26, s72, s13
	s_ashr_i32 s35, s34, 31
	v_lshrrev_b32_e32 v0, 30, v0
	s_ashr_i32 s27, s26, 31
	s_ashr_i32 s23, s9, 31
	s_mul_u64 s[26:27], s[26:27], s[20:21]
	s_sub_co_i32 s15, s77, s13
	v_add_nc_u32_e32 v0, v226, v0
	s_mul_u64 s[30:31], s[34:35], s[22:23]
	s_lshl_b64 s[26:27], s[26:27], 1
	s_max_i32 s60, s15, 0
	s_lshl_b64 s[30:31], s[30:31], 1
	v_ashrrev_i32_e32 v228, 2, v0
	v_mad_u32_u24 v231, 0x110, v229, v1
	s_add_nc_u64 s[16:17], s[16:17], s[26:27]
	s_mov_b32 s82, s7
	s_add_nc_u64 s[30:31], s[16:17], s[30:31]
	s_mov_b32 s24, s11
	v_dual_add_nc_u32 v236, s29, v231 :: v_dual_bitop2_b32 v1, -4, v0 bitop3:0x40
	v_lshlrev_b32_e32 v237, 1, v230
	v_cvt_pk_bf16_f32 v235, s4, s4
	s_bitset1_b32 s31, 31
	v_or_b32_e32 v234, 0x20000, v231
	v_cmp_ne_u32_e32 vcc_lo, v226, v1
	s_mov_b64 s[66:67], s[30:31]
	s_mov_b64 s[64:65], s[28:29]
	s_mov_b64 s[70:71], s[30:31]
	s_mov_b64 s[68:69], s[28:29]
	s_and_b32 vcc_lo, s104, vcc_lo
	s_cmp_lg_u32 s5, 0x80000000
	v_subrev_co_ci_u32_e64 v227, null, 0, v228, vcc_lo
	s_cselect_b32 s17, s21, 0
	s_cselect_b32 s16, s5, 0x200
	s_cmp_lg_u32 s9, 0x80000000
	s_cselect_b32 s41, s9, 0x80
	s_cselect_b32 s5, s23, 0
	s_bfe_i32 s2, s2, 0x10018
	s_lshl_b32 s9, s16, 16
	s_and_b32 s5, s5, 0xffff
	s_lshr_b32 s23, s2, 30
	s_or_b32 s42, s5, s9
	s_or_b32 s5, s23, s93
	s_lshr_b64 s[20:21], s[16:17], 16
	s_addk_co_i32 s5, 0x7f
	s_lshr_b32 s21, s16, 16
	s_sub_co_i32 s16, s78, s77
	s_ashr_i32 s5, s5, 2
	s_add_co_i32 s22, s77, -1
	s_add_co_i32 s92, s76, s16
	s_add_co_i32 s5, s5, s2
	s_add_co_i32 s92, s92, 1
	s_min_i32 s2, s5, s22
	s_ashr_i32 s85, s84, 31
	s_add_co_i32 s2, s92, s2
	s_ashr_i32 s81, s6, 31
	s_min_i32 s2, s2, s78
	s_ashr_i32 s15, s14, 31
	s_ashr_i32 s13, s10, 31
	s_ashr_i32 s83, s7, 31
	s_ashr_i32 s25, s11, 31
	s_mul_u64 s[10:11], s[84:85], s[80:81]
	s_max_i32 s9, s2, 1
	s_and_b32 s20, s20, 0xffff0000
	s_mul_u64 s[12:13], s[14:15], s[12:13]
	s_mul_u64 s[16:17], s[84:85], s[82:83]
	s_mul_u64 s[14:15], s[14:15], s[24:25]
	s_lshl_b64 s[10:11], s[10:11], 1
	s_add_co_i32 s2, s9, 63
	s_or_b32 s43, s20, s21
	s_lshl_b64 s[94:95], s[12:13], 1
	s_lshl_b64 s[12:13], s[16:17], 1
	s_lshl_b64 s[96:97], s[14:15], 1
	s_add_nc_u64 s[14:15], s[74:75], s[10:11]
	s_min_i32 s5, s9, 64
	s_lshr_b32 s10, s2, 6
	s_cmp_lg_u32 s6, 0x80000000
	s_add_nc_u64 s[14:15], s[14:15], s[94:95]
	s_cselect_b32 s21, s81, 0
	s_cselect_b32 s17, s6, 0x80
	s_and_b32 s85, s18, 7
	s_mov_b32 s20, s17
	s_lshl_b32 s35, s85, 3
	s_lshl_b32 s2, s85, 4
	s_sub_co_i32 s5, s5, s35
	s_mul_u64 s[86:87], s[20:21], s[2:3]
	s_max_i32 s5, s5, 0
	s_add_nc_u64 s[14:15], s[86:87], s[14:15]
	s_lshl_b32 s5, s5, 16
	s_and_b32 s18, s21, 0xffff
	s_or_b32 s11, s15, 0x80000000
	s_or_b32 s46, s5, 0x7fff
	s_cmp_lg_u32 s7, 0x80000000
	s_mul_i32 s5, s85, 0x900
	s_cselect_b32 s25, s7, 0x80
	s_cselect_b32 s7, s83, 0
	s_mov_b32 s6, s25
	s_and_b32 s26, s7, 0xffff
	s_mul_u64 s[88:89], s[6:7], s[2:3]
	s_or_b32 s3, s5, 0x10000
	s_load_b128 s[4:7], s[0:1], 0x7c nv
	s_add_nc_u64 s[12:13], s[90:91], s[12:13]
	s_mulk_i32 s85, 0x880
	s_add_nc_u64 s[12:13], s[12:13], s[96:97]
	s_mov_b32 s65, s85
	s_add_nc_u64 s[12:13], s[88:89], s[12:13]
	s_mov_b32 s66, s14
	s_or_b32 s2, s13, 0x80000000
	s_cmp_lg_u32 s33, s44
	s_mov_b32 s67, s11
	s_mov_b32 s44, s36
	s_mov_b32 s49, s17
	s_mov_b32 s50, s18
	s_mov_b32 s69, s3
	s_mov_b32 s70, s12
	s_mov_b32 s71, s2
	s_mov_b32 s54, s46
	s_mov_b32 s57, s25
	s_mov_b32 s58, s26
	s_mov_b32 s2, -1
	s_cbranch_scc0 .LBB0_8
	s_mov_b32 s61, s19
	s_mov_b32 s62, s19
	s_mov_b32 s63, s19
	s_mov_b32 s12, s19
	s_mov_b32 s13, s19
	s_mov_b32 s14, s19
	s_mov_b32 s15, s19
	s_ashr_i32 s2, s93, 2
	tensor_load_to_lds s[28:31], s[36:43], s[60:63], s[12:15]
	s_add_co_i32 s11, s92, -1
	v_and_or_b32 v32, v230, 7, v233
	s_add_co_i32 s2, s11, s2
	s_add_co_i32 s12, s10, -1
	s_max_i32 s2, s2, 0
	s_set_vgpr_msb 64
	v_writelane_b32 v0 /*v256*/, s72, 1
	s_add_co_i32 s2, s2, 1
	s_set_vgpr_msb 0x4000
	v_mul_u32_u24_e32 v32, 0x120, v32
	s_ashr_i32 s13, s2, 31
	v_or_b32_e32 v242, 0x20000, v231
	s_lshr_b32 s13, s13, 26
	s_add_nc_u64 s[98:99], s[90:91], s[96:97]
	s_add_co_i32 s13, s2, s13
	v_and_or_b32 v32, v237, 16, v32
	s_and_b32 s14, s13, 0xffffffc0
	s_ashr_i32 s13, s13, 6
	s_cmp_lg_u32 s2, s14
	s_add_nc_u64 s[100:101], s[74:75], s[94:95]
	s_cselect_b32 s14, -1, 0
	s_cmp_lt_i32 s2, 0
	v_or_b32_e32 v238, 0x10000, v32
	s_cselect_b32 s2, -1, 0
	v_or_b32_e32 v243, 0x30000, v32
	s_and_b32 s2, s2, s14
	s_sub_co_ci_u32 s2, s13, 0
	s_min_i32 s2, s2, s12
	s_max_i32 s62, s2, 0
	s_cmp_lt_i32 s2, 1
	s_wait_tensorcnt 0x0
	s_wait_alu depctr_va_vdst(0)
	ds_load_b128 v[0:3], v236
	ds_load_b128 v[4:7], v236 offset:32
	ds_load_b128 v[8:11], v236 offset:64
	ds_load_b128 v[12:15], v236 offset:96
	ds_load_b128 v[16:19], v236 offset:128
	ds_load_b128 v[20:23], v236 offset:160
	ds_load_b128 v[24:27], v236 offset:192
	ds_load_b128 v[28:31], v236 offset:224
	tensor_load_to_lds s[64:67], s[44:51]
	tensor_load_to_lds s[68:71], s[52:59]
	s_wait_dscnt 0x7
	v_pk_mul_bf16 v67, v235, v3
	s_wait_dscnt 0x6
	v_pk_mul_bf16 v71, v235, v7
	v_pk_mul_bf16 v70, v235, v6
	v_pk_mul_bf16 v69, v235, v5
	v_pk_mul_bf16 v68, v235, v4
	v_pk_mul_bf16 v66, v235, v2
	v_pk_mul_bf16 v65, v235, v1
	v_pk_mul_bf16 v64, v235, v0
	s_wait_dscnt 0x4
	v_pk_mul_bf16 v79, v235, v15
	v_pk_mul_bf16 v78, v235, v14
	v_pk_mul_bf16 v77, v235, v13
	v_pk_mul_bf16 v76, v235, v12
	v_pk_mul_bf16 v75, v235, v11
	v_pk_mul_bf16 v74, v235, v10
	v_pk_mul_bf16 v73, v235, v9
	v_pk_mul_bf16 v72, v235, v8
	s_wait_dscnt 0x2
	v_pk_mul_bf16 v87, v235, v23
	v_pk_mul_bf16 v86, v235, v22
	v_pk_mul_bf16 v85, v235, v21
	v_pk_mul_bf16 v84, v235, v20
	v_pk_mul_bf16 v83, v235, v19
	v_pk_mul_bf16 v82, v235, v18
	v_pk_mul_bf16 v81, v235, v17
	v_pk_mul_bf16 v80, v235, v16
	s_wait_dscnt 0x0
	v_pk_mul_bf16 v95, v235, v31
	v_pk_mul_bf16 v94, v235, v30
	v_pk_mul_bf16 v93, v235, v29
	v_pk_mul_bf16 v92, v235, v28
	v_pk_mul_bf16 v91, v235, v27
	v_pk_mul_bf16 v90, v235, v26
	v_pk_mul_bf16 v89, v235, v25
	v_pk_mul_bf16 v88, v235, v24
	s_wait_tensorcnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	s_cbranch_scc1 .LBB0_9
	v_dual_mov_b32 v0, 0 :: v_dual_mov_b32 v104, v243
	s_set_vgpr_msb 64
	v_writelane_b32 v0 /*v256*/, s79, 2
	s_set_vgpr_msb 0x4000
	v_dual_mov_b32 v105, v242 :: v_dual_mov_b32 v239, v231
	v_dual_mov_b32 v1, v0 :: v_dual_mov_b32 v2, v0
	v_dual_mov_b32 v3, v0 :: v_dual_mov_b32 v4, v0
	v_dual_mov_b32 v5, v0 :: v_dual_mov_b32 v6, v0
	v_dual_mov_b32 v7, v0 :: v_dual_mov_b32 v224, 0xf149f2ca
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
	v_mov_b32_e32 v232, v0
	s_lshl_b64 s[102:103], s[62:63], 6
	s_mov_b32 s72, 1
	s_sub_co_i32 s61, s9, 64
	s_add_co_i32 s90, s84, 64
	s_mov_b32 s16, 8
	s_mov_b32 s15, 0x800000
	s_mov_b32 s13, 0xffff0000
	s_mov_b32 s12, 0x7510000
	s_mov_b32 s20, 0xf510000
	s_mov_b32 s79, 0x76543210
	s_mov_b32 s76, 0x3fb8aa3b
	s_mov_b32 s33, 1
	s_branch .LBB0_4
.LBB0_3:
	s_set_vgpr_msb 0x45
	v_cvt_pk_bf16_f32 v29 /*v285*/, v18 /*v274*/, v19 /*v275*/
	v_cvt_pk_bf16_f32 v28 /*v284*/, v14 /*v270*/, v16 /*v272*/
	v_cvt_pk_bf16_f32 v27 /*v283*/, v11 /*v267*/, v12 /*v268*/
	v_cvt_pk_bf16_f32 v26 /*v282*/, v9 /*v265*/, v10 /*v266*/
	v_cvt_pk_bf16_f32 v25 /*v281*/, v6 /*v262*/, v7 /*v263*/
	v_cvt_pk_bf16_f32 v24 /*v280*/, v3 /*v259*/, v4 /*v260*/
	v_cvt_pk_bf16_f32 v23 /*v279*/, v1 /*v257*/, v2 /*v258*/
	s_set_vgpr_msb 0x4540
	v_cvt_pk_bf16_f32 v22 /*v278*/, v252, v254
	s_set_vgpr_msb 0x4004
	s_wait_dscnt 0x1d
	v_wmma_f32_16x16x32_bf16 v[56:63], v[216:223], v[22:29] /*v[278:285]*/, v[56:63]
	s_set_vgpr_msb 0x400
	v_fmac_f32_e32 v244, v224, v232
	s_mov_b64 s[22:23], 0xffffffffffffffc0
	v_mov_b32_e32 v224, v241
	s_add_nc_u64 s[102:103], s[102:103], s[22:23]
	s_sub_co_i32 s61, s61, 64
	s_add_co_i32 s90, s90, 64
	s_add_co_i32 s33, s33, 1
	s_set_vgpr_msb 4
	s_wait_dscnt 0x1c
	v_wmma_f32_16x16x32_bf16 v[48:55], v[200:207], v[22:29] /*v[278:285]*/, v[48:55]
	s_cmp_lg_u64 s[102:103], 0
	s_wait_dscnt 0x0
	v_nop
	s_set_vgpr_msb 0x405
	v_cvt_pk_bf16_f32 v223, v20 /*v276*/, v21 /*v277*/
	v_cvt_pk_bf16_f32 v222, v15 /*v271*/, v17 /*v273*/
	v_cvt_pk_bf16_f32 v221, v8 /*v264*/, v13 /*v269*/
	s_set_vgpr_msb 0x504
	v_cvt_pk_bf16_f32 v220, v255, v5 /*v261*/
	v_wmma_f32_16x16x32_bf16 v[40:47], v[184:191], v[22:29] /*v[278:285]*/, v[40:47]
	s_set_vgpr_msb 0x400
	v_cvt_pk_bf16_f32 v219, v251, v253
	v_cvt_pk_bf16_f32 v218, v249, v250
	v_cvt_pk_bf16_f32 v217, v247, v248
	v_cvt_pk_bf16_f32 v216, v245, v246
	s_set_vgpr_msb 4
	v_wmma_f32_16x16x32_bf16 v[32:39], v[168:175], v[22:29] /*v[278:285]*/, v[32:39]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[152:159], v[22:29] /*v[278:285]*/, v[24:31]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[136:143], v[22:29] /*v[278:285]*/, v[16:23]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[120:127], v[22:29] /*v[278:285]*/, v[8:15]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[104:111], v[22:29] /*v[278:285]*/, v[0:7]
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x400
	v_dual_mov_b32 v104, v243 :: v_dual_add_f32 v232, v244, v240
	v_mov_b32_e32 v105, v242
	v_wmma_f32_16x16x32_bf16 v[56:63], v[208:215], v[216:223], v[56:63]
	v_wmma_f32_16x16x32_bf16 v[48:55], v[192:199], v[216:223], v[48:55]
	v_wmma_f32_16x16x32_bf16 v[40:47], v[176:183], v[216:223], v[40:47]
	v_wmma_f32_16x16x32_bf16 v[32:39], v[160:167], v[216:223], v[32:39]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[144:151], v[216:223], v[24:31]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[128:135], v[216:223], v[16:23]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[112:119], v[216:223], v[8:15]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[96:103], v[216:223], v[0:7]
	s_cbranch_scc0 .LBB0_19
.LBB0_4:
	s_wait_alu depctr_vm_vsrc(6)
	v_dual_mov_b32 v242, v239 :: v_dual_mov_b32 v239, v105
	s_wait_alu depctr_vm_vsrc(0)
	v_dual_mov_b32 v243, v238 :: v_dual_mov_b32 v238, v104
	s_wait_tensorcnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	s_cmp_ge_i32 s33, s10
	s_cbranch_scc1 .LBB0_6
	v_nop
	v_nop
	v_med3_i32 v96, s61, 0, 64
	s_ashr_i32 s91, s90, 31
	s_lshr_b32 s2, s33, 31
	s_mul_u64 s[74:75], s[90:91], s[80:81]
	s_add_co_i32 s2, s33, s2
	v_readfirstlane_b32 s14, v96
	s_and_b32 s2, s2, 0x7ffe
	s_lshl_b64 s[74:75], s[74:75], 1
	s_mul_u64 s[22:23], s[90:91], s[82:83]
	s_sub_co_i32 s2, s33, s2
	s_sub_co_i32 s14, s14, s35
	s_add_nc_u64 s[74:75], s[100:101], s[74:75]
	s_max_i32 s14, s14, 0
	s_lshl_b64 s[22:23], s[22:23], 1
	s_lshl_b32 s2, s2, 17
	s_add_nc_u64 s[74:75], s[86:87], s[74:75]
	s_lshl_b32 s14, s14, 16
	s_add_nc_u64 s[22:23], s[98:99], s[22:23]
	s_or_b32 s73, s85, s2
	s_bitset1_b32 s75, 31
	s_addk_co_i32 s14, 0x7fff
	s_mov_b32 s21, s13
	tensor_load_to_lds s[72:75], s[12:19]
	s_add_nc_u64 s[74:75], s[88:89], s[22:23]
	s_or_b32 s73, s3, s2
	s_bitset1_b32 s75, 31
	s_mov_b32 s22, s14
	s_mov_b32 s23, s15
	s_mov_b32 s24, s16
	s_mov_b32 s27, s19
	tensor_load_to_lds s[72:75], s[20:27]
.LBB0_6:
	s_wait_alu depctr_va_vdst(0)
	ds_load_b128 v[96:99], v242
	ds_load_b128 v[100:103], v242 offset:32
	ds_load_b128 v[104:107], v242 offset:64
	ds_load_b128 v[108:111], v242 offset:96
	ds_load_b128 v[112:115], v242 offset:128
	ds_load_b128 v[116:119], v242 offset:160
	ds_load_b128 v[120:123], v242 offset:192
	ds_load_b128 v[124:127], v242 offset:224
	ds_load_b128 v[128:131], v242 offset:4352
	ds_load_b128 v[132:135], v242 offset:4384
	ds_load_b128 v[136:139], v242 offset:4416
	ds_load_b128 v[140:143], v242 offset:4448
	ds_load_b128 v[144:147], v242 offset:4480
	ds_load_b128 v[148:151], v242 offset:4512
	ds_load_b128 v[152:155], v242 offset:4544
	ds_load_b128 v[156:159], v242 offset:4576
	ds_load_b128 v[160:163], v242 offset:8704
	ds_load_b128 v[164:167], v242 offset:8736
	ds_load_b128 v[168:171], v242 offset:8768
	ds_load_b128 v[172:175], v242 offset:8800
	ds_load_b128 v[176:179], v242 offset:8832
	ds_load_b128 v[180:183], v242 offset:8864
	ds_load_b128 v[184:187], v242 offset:8896
	ds_load_b128 v[188:191], v242 offset:8928
	ds_load_b128 v[192:195], v242 offset:13056
	ds_load_b128 v[196:199], v242 offset:13088
	ds_load_b128 v[200:203], v242 offset:13120
	ds_load_b128 v[204:207], v242 offset:13152
	ds_load_b128 v[208:211], v242 offset:13184
	ds_load_b128 v[212:215], v242 offset:13216
	ds_load_b128 v[216:219], v242 offset:13248
	ds_load_b128 v[220:223], v242 offset:13280
	s_wait_dscnt 0x1e
	v_wmma_f32_16x16x32_bf16 v[244:251], v[96:103], v[64:71], 0
	s_set_vgpr_msb 64
	s_wait_dscnt 0x16
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[258:265]*/, v[128:135], v[64:71], 0
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[266:273]*/, v[160:167], v[64:71], 0
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[274:281]*/, v[192:199], v[64:71], 0
	s_set_vgpr_msb 0x4000
	v_wmma_f32_16x16x32_bf16 v[244:251], v[104:111], v[72:79], v[244:251]
	s_set_vgpr_msb 0x50
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[258:265]*/, v[136:143], v[72:79], v[2:9] /*v[258:265]*/
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[266:273]*/, v[168:175], v[72:79], v[10:17] /*v[266:273]*/
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[274:281]*/, v[200:207], v[72:79], v[18:25] /*v[274:281]*/
	s_set_vgpr_msb 0x5000
	v_wmma_f32_16x16x32_bf16 v[244:251], v[112:119], v[80:87], v[244:251]
	s_set_vgpr_msb 0x50
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[258:265]*/, v[144:151], v[80:87], v[2:9] /*v[258:265]*/
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[266:273]*/, v[176:183], v[80:87], v[10:17] /*v[266:273]*/
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[274:281]*/, v[208:215], v[80:87], v[18:25] /*v[274:281]*/
	s_set_vgpr_msb 0x5000
	v_wmma_f32_16x16x32_bf16 v[244:251], v[120:127], v[88:95], v[244:251]
	s_set_vgpr_msb 0x50
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[258:265]*/, v[152:159], v[88:95], v[2:9] /*v[258:265]*/
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[266:273]*/, v[184:191], v[88:95], v[10:17] /*v[266:273]*/
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[274:281]*/, v[216:223], v[88:95], v[18:25] /*v[274:281]*/
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x5000
	ds_load_tr16_b128 v[216:219], v243
	ds_load_tr16_b128 v[200:203], v243 offset:32
	ds_load_tr16_b128 v[220:223], v243 offset:4608
	ds_load_tr16_b128 v[204:207], v243 offset:4640
	ds_load_tr16_b128 v[208:211], v243 offset:9216
	ds_load_tr16_b128 v[192:195], v243 offset:9248
	ds_load_tr16_b128 v[212:215], v243 offset:13824
	ds_load_tr16_b128 v[196:199], v243 offset:13856
	ds_load_tr16_b128 v[184:187], v243 offset:64
	ds_load_tr16_b128 v[168:171], v243 offset:96
	ds_load_tr16_b128 v[188:191], v243 offset:4672
	ds_load_tr16_b128 v[172:175], v243 offset:4704
	ds_load_tr16_b128 v[176:179], v243 offset:9280
	ds_load_tr16_b128 v[160:163], v243 offset:9312
	ds_load_tr16_b128 v[180:183], v243 offset:13888
	ds_load_tr16_b128 v[164:167], v243 offset:13920
	ds_load_tr16_b128 v[152:155], v243 offset:128
	ds_load_tr16_b128 v[136:139], v243 offset:160
	ds_load_tr16_b128 v[156:159], v243 offset:4736
	ds_load_tr16_b128 v[140:143], v243 offset:4768
	ds_load_tr16_b128 v[144:147], v243 offset:9344
	ds_load_tr16_b128 v[128:131], v243 offset:9376
	ds_load_tr16_b128 v[148:151], v243 offset:13952
	ds_load_tr16_b128 v[132:135], v243 offset:13984
	ds_load_tr16_b128 v[120:123], v243 offset:192
	ds_load_tr16_b128 v[104:107], v243 offset:224
	ds_load_tr16_b128 v[124:127], v243 offset:4800
	ds_load_tr16_b128 v[108:111], v243 offset:4832
	ds_load_tr16_b128 v[112:115], v243 offset:9408
	ds_load_tr16_b128 v[96:99], v243 offset:9440
	ds_load_tr16_b128 v[116:119], v243 offset:14016
	ds_load_tr16_b128 v[100:103], v243 offset:14048
	v_nop
	v_max_num_f32_e32 v240, v244, v245
	v_max3_num_f32 v241, v247, v248, v249
	s_set_vgpr_msb 21
	v_max3_num_f32 v253, v3 /*v259*/, v4 /*v260*/, v5 /*v261*/
	v_max3_num_f32 v254, v6 /*v262*/, v7 /*v263*/, v8 /*v264*/
	v_max3_num_f32 v255, v9 /*v265*/, v10 /*v266*/, v11 /*v267*/
	s_set_vgpr_msb 0x1510
	v_max3_num_f32 v252, v250, v251, v2 /*v258*/
	s_set_vgpr_msb 0x1055
	v_max3_num_f32 v1 /*v257*/, v12 /*v268*/, v13 /*v269*/, v14 /*v270*/
	v_max3_num_f32 v26 /*v282*/, v15 /*v271*/, v16 /*v272*/, v17 /*v273*/
	v_max3_num_f32 v27 /*v283*/, v18 /*v274*/, v19 /*v275*/, v20 /*v276*/
	v_max3_num_f32 v28 /*v284*/, v21 /*v277*/, v22 /*v278*/, v23 /*v279*/
	s_set_vgpr_msb 0x5500
	v_max3_num_f32 v240, v240, v246, v241
	v_max3_num_f32 v241, v253, v254, v255
	s_set_vgpr_msb 21
	v_max3_num_f32 v253, v1 /*v257*/, v26 /*v282*/, v27 /*v283*/
	v_max3_num_f32 v254, v28 /*v284*/, v24 /*v280*/, v25 /*v281*/
	s_set_vgpr_msb 0x1500
	v_max3_num_f32 v240, v240, v252, v241
	v_max3_num_f32 v240, v240, v253, v254
	v_mov_b32_e32 v241, v240
	v_permlanex16_b32 v241, v241, s79, 0xfedcba98
	v_max_num_f32_e32 v240, v240, v241
	v_sub_f32_e32 v241, v240, v224
	v_max_num_f32_e32 v240, v224, v240
	v_cmp_lt_f32_e32 vcc_lo, 0x41000000, v241
	s_cmp_eq_u32 vcc_lo, 0
	s_cselect_b32 s2, -1, 0
	v_cndmask_b32_e64 v241, v240, v224, s2
	v_mul_f32_e32 v240, 0xbfb8aa3b, v241
	v_pk_fma_f32 v[244:245], v[244:245], s[76:77], v[240:241] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[246:247], v[246:247], s[76:77], v[240:241] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[248:249], v[248:249], s[76:77], v[240:241] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[250:251], v[250:251], s[76:77], v[240:241] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x41
	v_pk_fma_f32 v[26:27] /*v[282:283]*/, v[2:3] /*v[258:259]*/, s[76:77], v[240:241] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[28:29] /*v[284:285]*/, v[4:5] /*v[260:261]*/, s[76:77], v[240:241] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[30:31] /*v[286:287]*/, v[6:7] /*v[262:263]*/, s[76:77], v[240:241] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[36:37] /*v[292:293]*/, v[12:13] /*v[268:269]*/, s[76:77], v[240:241] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[38:39] /*v[294:295]*/, v[14:15] /*v[270:271]*/, s[76:77], v[240:241] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[40:41] /*v[296:297]*/, v[16:17] /*v[272:273]*/, s[76:77], v[240:241] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x4100
	v_exp_f32_e32 v252, v244
	v_exp_f32_e32 v254, v245
	s_set_vgpr_msb 64
	v_exp_f32_e32 v2 /*v258*/, v247
	v_exp_f32_e32 v3 /*v259*/, v248
	s_set_vgpr_msb 0x4041
	v_pk_fma_f32 v[32:33] /*v[288:289]*/, v[8:9] /*v[264:265]*/, s[76:77], v[240:241] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[34:35] /*v[290:291]*/, v[10:11] /*v[266:267]*/, s[76:77], v[240:241] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[20:21] /*v[276:277]*/, v[20:21] /*v[276:277]*/, s[76:77], v[240:241] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x4140
	v_exp_f32_e32 v1 /*v257*/, v246
	v_exp_f32_e32 v4 /*v260*/, v249
	v_exp_f32_e32 v6 /*v262*/, v250
	v_exp_f32_e32 v7 /*v263*/, v251
	s_set_vgpr_msb 0x4041
	v_exp_f32_e32 v10 /*v266*/, v27 /*v283*/
	v_exp_f32_e32 v11 /*v267*/, v28 /*v284*/
	v_exp_f32_e32 v14 /*v270*/, v30 /*v286*/
	v_exp_f32_e32 v16 /*v272*/, v31 /*v287*/
	s_set_vgpr_msb 0x4101
	v_exp_f32_e32 v247, v36 /*v292*/
	v_exp_f32_e32 v248, v37 /*v293*/
	v_exp_f32_e32 v250, v39 /*v295*/
	v_exp_f32_e32 v251, v40 /*v296*/
	s_set_vgpr_msb 0x141
	v_pk_fma_f32 v[22:23] /*v[278:279]*/, v[22:23] /*v[278:279]*/, s[76:77], v[240:241] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[42:43] /*v[298:299]*/, v[18:19] /*v[274:275]*/, s[76:77], v[240:241] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[24:25] /*v[280:281]*/, v[24:25] /*v[280:281]*/, s[76:77], v[240:241] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x4100
	v_add_f32_e32 v240, v252, v254
	s_set_vgpr_msb 5
	v_add_f32_e32 v244, v2 /*v258*/, v3 /*v259*/
	s_set_vgpr_msb 0x541
	v_exp_f32_e32 v12 /*v268*/, v29 /*v285*/
	v_exp_f32_e32 v18 /*v274*/, v32 /*v288*/
	v_exp_f32_e32 v19 /*v275*/, v33 /*v289*/
	s_set_vgpr_msb 0x4101
	v_exp_f32_e32 v245, v34 /*v290*/
	v_exp_f32_e32 v253, v41 /*v297*/
	s_set_vgpr_msb 0x141
	v_exp_f32_e32 v13 /*v269*/, v21 /*v277*/
	v_exp_f32_e32 v15 /*v271*/, v22 /*v278*/
	v_exp_f32_e32 v9 /*v265*/, v26 /*v282*/
	v_exp_f32_e32 v8 /*v264*/, v20 /*v276*/
	v_exp_f32_e32 v17 /*v273*/, v23 /*v279*/
	v_exp_f32_e32 v20 /*v276*/, v24 /*v280*/
	s_set_vgpr_msb 0x4101
	v_add_f32_e32 v240, v1 /*v257*/, v240
	s_set_vgpr_msb 0x145
	v_add_f32_e32 v23 /*v279*/, v10 /*v266*/, v11 /*v267*/
	s_set_vgpr_msb 0x4501
	v_add_f32_e32 v244, v4 /*v260*/, v244
	s_set_vgpr_msb 0x145
	v_add_f32_e32 v24 /*v280*/, v14 /*v270*/, v16 /*v272*/
	s_set_vgpr_msb 0x4540
	v_dual_add_f32 v26 /*v282*/, v247, v248 :: v_dual_add_f32 v27 /*v283*/, v250, v251
	s_set_vgpr_msb 0x4001
	v_exp_f32_e32 v246, v35 /*v291*/
	v_exp_f32_e32 v255, v42 /*v298*/
	s_set_vgpr_msb 0x141
	v_exp_f32_e32 v5 /*v261*/, v43 /*v299*/
	v_exp_f32_e32 v21 /*v277*/, v25 /*v281*/
	v_nop
	v_add_f32_e32 v25 /*v281*/, v19 /*v275*/, v245
	s_set_vgpr_msb 0x4145
	v_dual_add_f32 v23 /*v279*/, v12 /*v268*/, v23 /*v279*/ :: v_dual_add_f32 v24 /*v280*/, v18 /*v274*/, v24 /*v280*/
	s_set_vgpr_msb 0x4500
	v_add_f32_e32 v240, v240, v244
	s_set_vgpr_msb 4
	v_add_f32_e32 v244, v253, v27 /*v283*/
	s_set_vgpr_msb 0x445
	v_add_f32_e32 v27 /*v283*/, v13 /*v269*/, v15 /*v271*/
	v_dual_add_f32 v22 /*v278*/, v6 /*v262*/, v7 /*v263*/ :: v_dual_add_f32 v23 /*v279*/, v23 /*v279*/, v24 /*v280*/
	s_set_vgpr_msb 0x4501
	v_exp_f32_e32 v249, v38 /*v294*/
	s_set_vgpr_msb 0x144
	v_add_f32_e32 v25 /*v281*/, v246, v25 /*v281*/
	v_add_f32_e32 v28 /*v284*/, v255, v5 /*v261*/
	s_set_vgpr_msb 0x4445
	v_add_f32_e32 v22 /*v278*/, v9 /*v265*/, v22 /*v278*/
	s_set_vgpr_msb 0x4544
	v_add_f32_e32 v26 /*v282*/, v249, v26 /*v282*/
	s_set_vgpr_msb 0x4445
	v_add_f32_e32 v24 /*v280*/, v8 /*v264*/, v28 /*v284*/
	s_set_vgpr_msb 0x4501
	v_add_f32_e32 v240, v22 /*v278*/, v240
	s_set_vgpr_msb 0x145
	v_add_f32_e32 v22 /*v278*/, v25 /*v281*/, v23 /*v279*/
	v_dual_add_f32 v23 /*v279*/, v17 /*v273*/, v27 /*v283*/ :: v_dual_add_f32 v25 /*v281*/, v20 /*v276*/, v21 /*v277*/
	s_set_vgpr_msb 0x4504
	v_add_f32_e32 v240, v240, v22 /*v278*/
	v_add_f32_e32 v244, v244, v26 /*v282*/
	s_set_vgpr_msb 0x445
	v_add_f32_e32 v22 /*v278*/, v25 /*v281*/, v23 /*v279*/
	s_set_vgpr_msb 0x4501
	v_add_f32_e32 v244, v24 /*v280*/, v244
	s_set_vgpr_msb 0x100
	v_add_f32_e32 v240, v244, v240
	s_set_vgpr_msb 1
	v_add_f32_e32 v240, v22 /*v278*/, v240
	s_set_vgpr_msb 0x100
	v_mov_b32_e32 v244, v240
	v_sub_f32_e32 v224, v224, v241
	v_permlanex16_b32 v244, v244, s79, 0xfedcba98
	v_mul_f32_e32 v224, 0x3fb8aa3b, v224
	v_exp_f32_e32 v224, v224
	s_cbranch_vccz .LBB0_3
	v_nop
	v_pk_mul_f32 v[62:63], v[62:63], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[60:61], v[60:61], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59], v[58:59], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[56:57], v[56:57], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[54:55], v[54:55], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[52:53], v[52:53], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[50:51], v[50:51], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[48:49], v[48:49], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[46:47], v[46:47], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[44:45], v[44:45], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[42:43], v[42:43], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[40:41], v[40:41], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39], v[38:39], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[36:37], v[36:37], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35], v[34:35], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[32:33], v[32:33], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[30:31], v[30:31], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[28:29], v[28:29], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[26:27], v[26:27], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25], v[24:25], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[22:23], v[22:23], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21], v[20:21], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[18:19], v[18:19], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[16:17], v[16:17], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[14:15], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13], v[12:13], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[10:11], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[8:9], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[6:7], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[4:5], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[2:3], v[2:3], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[0:1], v[0:1], v[224:225] op_sel_hi:[1,0]
	s_branch .LBB0_3
.LBB0_8:
	s_and_b32 vcc_lo, exec_lo, s2
	s_cbranch_vccnz .LBB0_12
	s_branch .LBB0_35
.LBB0_9:
	v_dual_mov_b32 v0, 0 :: v_dual_mov_b32 v241, 0xf149f2ca
	v_dual_mov_b32 v239, v231 :: v_dual_mov_b32 v7, v0
	v_dual_mov_b32 v1, v0 :: v_dual_mov_b32 v2, v0
	v_dual_mov_b32 v3, v0 :: v_dual_mov_b32 v4, v0
	v_dual_mov_b32 v5, v0 :: v_dual_mov_b32 v6, v0
	v_mov_b64_e32 v[8:9], v[0:1]
	v_mov_b64_e32 v[10:11], v[2:3]
	v_mov_b64_e32 v[18:19], v[2:3]
	v_mov_b64_e32 v[12:13], v[4:5]
	v_mov_b64_e32 v[14:15], v[6:7]
	v_mov_b64_e32 v[22:23], v[6:7]
	v_mov_b64_e32 v[20:21], v[4:5]
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
	v_mov_b32_e32 v232, v0
	s_cmp_ge_u32 s62, s10
	s_cbranch_scc0 .LBB0_20
.LBB0_10:
	v_mov_b32_e32 v240, v241
.LBB0_11:
	s_clause 0x1
	s_load_b64 s[74:75], s[0:1], 0x20 nv
	s_load_b64 s[90:91], s[0:1], 0x30 nv
	s_set_vgpr_msb 1
	v_readlane_b32 s72, v0 /*v256*/, 1
	s_set_vgpr_msb 0x100
	s_branch .LBB0_35
.LBB0_12:
	s_mov_b32 s19, 0
	s_ashr_i32 s2, s93, 2
	s_mov_b32 s61, s19
	s_mov_b32 s62, s19
	s_mov_b32 s63, s19
	s_mov_b32 s12, s19
	s_mov_b32 s13, s19
	s_mov_b32 s14, s19
	s_mov_b32 s15, s19
	s_add_co_i32 s11, s92, -1
	tensor_load_to_lds s[28:31], s[36:43], s[60:63], s[12:15]
	s_add_co_i32 s2, s11, s2
	v_and_or_b32 v32, v230, 7, v233
	s_max_i32 s2, s2, 0
	s_add_co_i32 s12, s10, -1
	s_add_co_i32 s2, s2, 1
	s_mov_b32 s28, 1
	s_ashr_i32 s13, s2, 31
	v_mul_u32_u24_e32 v32, 0x120, v32
	s_lshr_b32 s13, s13, 26
	s_wait_kmcnt 0x0
	s_add_nc_u64 s[36:37], s[90:91], s[96:97]
	s_add_co_i32 s13, s2, s13
	s_add_nc_u64 s[38:39], s[74:75], s[94:95]
	s_and_b32 s14, s13, 0xffffffc0
	s_ashr_i32 s13, s13, 6
	s_cmp_lg_u32 s2, s14
	v_and_or_b32 v32, v237, 16, v32
	s_cselect_b32 s14, -1, 0
	s_cmp_lt_i32 s2, 0
	s_mov_b32 s41, s19
	s_cselect_b32 s2, -1, 0
	v_or_b32_e32 v238, 0x30000, v32
	s_and_b32 s2, s2, s14
	s_sub_co_ci_u32 s2, s13, 0
	s_min_i32 s2, s2, s12
	s_max_i32 s40, s2, 0
	s_cmp_lt_i32 s2, 1
	s_wait_tensorcnt 0x0
	s_wait_alu depctr_va_vdst(0)
	ds_load_b128 v[0:3], v236
	ds_load_b128 v[4:7], v236 offset:32
	ds_load_b128 v[8:11], v236 offset:64
	ds_load_b128 v[12:15], v236 offset:96
	ds_load_b128 v[16:19], v236 offset:128
	ds_load_b128 v[20:23], v236 offset:160
	ds_load_b128 v[24:27], v236 offset:192
	ds_load_b128 v[28:31], v236 offset:224
	tensor_load_to_lds s[64:67], s[44:51]
	tensor_load_to_lds s[68:71], s[52:59]
	s_wait_alu depctr_vm_vsrc(0)
	v_or_b32_e32 v236, 0x10000, v32
	s_wait_dscnt 0x7
	v_pk_mul_bf16 v67, v235, v3
	s_wait_dscnt 0x6
	v_pk_mul_bf16 v71, v235, v7
	v_pk_mul_bf16 v70, v235, v6
	v_pk_mul_bf16 v69, v235, v5
	v_pk_mul_bf16 v68, v235, v4
	v_pk_mul_bf16 v66, v235, v2
	v_pk_mul_bf16 v65, v235, v1
	v_pk_mul_bf16 v64, v235, v0
	s_wait_dscnt 0x4
	v_pk_mul_bf16 v79, v235, v15
	v_pk_mul_bf16 v78, v235, v14
	v_pk_mul_bf16 v77, v235, v13
	v_pk_mul_bf16 v76, v235, v12
	v_pk_mul_bf16 v75, v235, v11
	v_pk_mul_bf16 v74, v235, v10
	v_pk_mul_bf16 v73, v235, v9
	v_pk_mul_bf16 v72, v235, v8
	s_wait_dscnt 0x2
	v_pk_mul_bf16 v87, v235, v23
	v_pk_mul_bf16 v86, v235, v22
	v_pk_mul_bf16 v85, v235, v21
	v_pk_mul_bf16 v84, v235, v20
	v_pk_mul_bf16 v83, v235, v19
	v_pk_mul_bf16 v82, v235, v18
	v_pk_mul_bf16 v81, v235, v17
	v_pk_mul_bf16 v80, v235, v16
	s_wait_dscnt 0x0
	v_pk_mul_bf16 v95, v235, v31
	v_pk_mul_bf16 v94, v235, v30
	v_pk_mul_bf16 v93, v235, v29
	v_pk_mul_bf16 v92, v235, v28
	v_pk_mul_bf16 v91, v235, v27
	v_pk_mul_bf16 v90, v235, v26
	v_pk_mul_bf16 v89, v235, v25
	v_pk_mul_bf16 v88, v235, v24
	v_mov_b32_e32 v0, 0
	s_wait_tensorcnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	s_cbranch_scc1 .LBB0_26
	v_dual_mov_b32 v1, v0 :: v_dual_mov_b32 v2, v0
	v_dual_mov_b32 v3, v0 :: v_dual_mov_b32 v4, v0
	v_dual_mov_b32 v5, v0 :: v_dual_mov_b32 v6, v0
	v_dual_mov_b32 v7, v0 :: v_dual_mov_b32 v224, 0xf149f2ca
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
	v_dual_mov_b32 v104, v238 :: v_dual_mov_b32 v105, v234
	v_dual_mov_b32 v235, v231 :: v_dual_mov_b32 v232, v0
	s_lshl_b64 s[42:43], s[40:41], 6
	s_sub_co_i32 s49, s9, 64
	s_add_co_i32 s44, s84, 64
	s_mov_b32 s16, 8
	s_mov_b32 s15, 0x800000
	s_mov_b32 s13, 0xffff0000
	s_mov_b32 s12, 0x7510000
	s_mov_b32 s20, 0xf510000
	s_mov_b64 s[46:47], 0xffffffffffffffc0
	s_mov_b32 s50, 0x76543210
	s_mov_b32 s48, 0x3fb8aa3b
	s_mov_b32 s33, 1
	s_branch .LBB0_15
.LBB0_14:
	s_set_vgpr_msb 0x45
	v_cvt_pk_bf16_f32 v25 /*v281*/, v14 /*v270*/, v15 /*v271*/
	v_cvt_pk_bf16_f32 v24 /*v280*/, v10 /*v266*/, v12 /*v268*/
	v_cvt_pk_bf16_f32 v23 /*v279*/, v7 /*v263*/, v8 /*v264*/
	v_cvt_pk_bf16_f32 v22 /*v278*/, v5 /*v261*/, v6 /*v262*/
	v_cvt_pk_bf16_f32 v21 /*v277*/, v2 /*v258*/, v3 /*v259*/
	s_set_vgpr_msb 0x4540
	v_cvt_pk_bf16_f32 v20 /*v276*/, v254, v255
	v_cvt_pk_bf16_f32 v19 /*v275*/, v252, v253
	v_cvt_pk_bf16_f32 v18 /*v274*/, v248, v250
	s_set_vgpr_msb 0x4004
	s_wait_dscnt 0x1d
	v_wmma_f32_16x16x32_bf16 v[56:63], v[216:223], v[18:25] /*v[274:281]*/, v[56:63]
	s_set_vgpr_msb 0x400
	v_dual_fmac_f32 v240, v224, v232 :: v_dual_mov_b32 v224, v237
	s_add_nc_u64 s[42:43], s[42:43], s[46:47]
	s_sub_co_i32 s49, s49, 64
	s_add_co_i32 s44, s44, 64
	s_add_co_i32 s33, s33, 1
	s_cmp_lg_u64 s[42:43], 0
	s_set_vgpr_msb 4
	s_wait_dscnt 0x1c
	v_wmma_f32_16x16x32_bf16 v[48:55], v[200:207], v[18:25] /*v[274:281]*/, v[48:55]
	s_wait_dscnt 0x0
	v_nop
	v_nop
	s_set_vgpr_msb 0x405
	v_cvt_pk_bf16_f32 v223, v16 /*v272*/, v17 /*v273*/
	v_cvt_pk_bf16_f32 v222, v11 /*v267*/, v13 /*v269*/
	v_cvt_pk_bf16_f32 v221, v4 /*v260*/, v9 /*v265*/
	s_set_vgpr_msb 0x504
	v_cvt_pk_bf16_f32 v220, v251, v1 /*v257*/
	s_set_vgpr_msb 0x400
	v_cvt_pk_bf16_f32 v219, v247, v249
	s_set_vgpr_msb 4
	v_wmma_f32_16x16x32_bf16 v[40:47], v[184:191], v[18:25] /*v[274:281]*/, v[40:47]
	s_set_vgpr_msb 0x400
	v_cvt_pk_bf16_f32 v218, v245, v246
	v_cvt_pk_bf16_f32 v217, v243, v244
	v_cvt_pk_bf16_f32 v216, v241, v242
	s_set_vgpr_msb 4
	v_wmma_f32_16x16x32_bf16 v[32:39], v[168:175], v[18:25] /*v[274:281]*/, v[32:39]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[152:159], v[18:25] /*v[274:281]*/, v[24:31]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[136:143], v[18:25] /*v[274:281]*/, v[16:23]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[120:127], v[18:25] /*v[274:281]*/, v[8:15]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[104:111], v[18:25] /*v[274:281]*/, v[0:7]
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x400
	v_dual_mov_b32 v104, v238 :: v_dual_add_f32 v232, v240, v239
	v_mov_b32_e32 v105, v234
	v_wmma_f32_16x16x32_bf16 v[56:63], v[208:215], v[216:223], v[56:63]
	v_wmma_f32_16x16x32_bf16 v[48:55], v[192:199], v[216:223], v[48:55]
	v_wmma_f32_16x16x32_bf16 v[40:47], v[176:183], v[216:223], v[40:47]
	v_wmma_f32_16x16x32_bf16 v[32:39], v[160:167], v[216:223], v[32:39]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[144:151], v[216:223], v[24:31]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[128:135], v[216:223], v[16:23]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[112:119], v[216:223], v[8:15]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[96:103], v[216:223], v[0:7]
	s_cbranch_scc0 .LBB0_27
.LBB0_15:
	s_wait_alu depctr_vm_vsrc(6)
	v_dual_mov_b32 v234, v235 :: v_dual_mov_b32 v235, v105
	s_wait_alu depctr_vm_vsrc(0)
	v_dual_mov_b32 v238, v236 :: v_dual_mov_b32 v236, v104
	s_wait_tensorcnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	s_wait_alu depctr_va_vdst(0)
	ds_load_b128 v[216:219], v234
	ds_load_b128 v[220:223], v234 offset:32
	ds_load_b128 v[208:211], v234 offset:64
	ds_load_b128 v[212:215], v234 offset:96
	ds_load_b128 v[200:203], v234 offset:128
	ds_load_b128 v[204:207], v234 offset:160
	ds_load_b128 v[192:195], v234 offset:192
	ds_load_b128 v[196:199], v234 offset:224
	ds_load_b128 v[184:187], v234 offset:4352
	ds_load_b128 v[188:191], v234 offset:4384
	ds_load_b128 v[176:179], v234 offset:4416
	ds_load_b128 v[180:183], v234 offset:4448
	ds_load_b128 v[168:171], v234 offset:4480
	ds_load_b128 v[172:175], v234 offset:4512
	ds_load_b128 v[160:163], v234 offset:4544
	ds_load_b128 v[164:167], v234 offset:4576
	ds_load_b128 v[152:155], v234 offset:8704
	ds_load_b128 v[156:159], v234 offset:8736
	ds_load_b128 v[144:147], v234 offset:8768
	ds_load_b128 v[148:151], v234 offset:8800
	ds_load_b128 v[136:139], v234 offset:8832
	ds_load_b128 v[140:143], v234 offset:8864
	ds_load_b128 v[128:131], v234 offset:8896
	ds_load_b128 v[132:135], v234 offset:8928
	ds_load_b128 v[120:123], v234 offset:13056
	ds_load_b128 v[124:127], v234 offset:13088
	ds_load_b128 v[112:115], v234 offset:13120
	ds_load_b128 v[116:119], v234 offset:13152
	ds_load_b128 v[104:107], v234 offset:13184
	ds_load_b128 v[108:111], v234 offset:13216
	ds_load_b128 v[96:99], v234 offset:13248
	ds_load_b128 v[100:103], v234 offset:13280
	s_cmp_ge_i32 s33, s10
	s_cbranch_scc1 .LBB0_17
	v_med3_i32 v237, s49, 0, 64
	s_ashr_i32 s45, s44, 31
	s_lshr_b32 s2, s33, 31
	s_mul_u64 s[30:31], s[44:45], s[80:81]
	s_add_co_i32 s2, s33, s2
	v_readfirstlane_b32 s14, v237
	s_and_b32 s2, s2, 0x7ffe
	s_lshl_b64 s[30:31], s[30:31], 1
	s_mul_u64 s[22:23], s[44:45], s[82:83]
	s_sub_co_i32 s2, s33, s2
	s_sub_co_i32 s14, s14, s35
	s_add_nc_u64 s[30:31], s[38:39], s[30:31]
	s_max_i32 s14, s14, 0
	s_lshl_b64 s[22:23], s[22:23], 1
	s_lshl_b32 s2, s2, 17
	s_add_nc_u64 s[30:31], s[86:87], s[30:31]
	s_lshl_b32 s14, s14, 16
	s_add_nc_u64 s[22:23], s[36:37], s[22:23]
	s_or_b32 s29, s85, s2
	s_bitset1_b32 s31, 31
	s_addk_co_i32 s14, 0x7fff
	s_mov_b32 s21, s13
	tensor_load_to_lds s[28:31], s[12:19]
	s_add_nc_u64 s[30:31], s[88:89], s[22:23]
	s_or_b32 s29, s3, s2
	s_bitset1_b32 s31, 31
	s_mov_b32 s22, s14
	s_mov_b32 s23, s15
	s_mov_b32 s24, s16
	s_mov_b32 s27, s19
	tensor_load_to_lds s[28:31], s[20:27]
.LBB0_17:
	s_wait_dscnt 0x1e
	v_wmma_f32_16x16x32_bf16 v[240:247], v[216:223], v[64:71], 0
	s_wait_dscnt 0x16
	v_wmma_f32_16x16x32_bf16 v[248:255], v[184:191], v[64:71], 0
	s_set_vgpr_msb 64
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[258:265]*/, v[152:159], v[64:71], 0
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x32_bf16 v[12:19] /*v[268:275]*/, v[120:127], v[64:71], 0
	s_set_vgpr_msb 0x4000
	v_wmma_f32_16x16x32_bf16 v[240:247], v[208:215], v[72:79], v[240:247]
	v_wmma_f32_16x16x32_bf16 v[248:255], v[176:183], v[72:79], v[248:255]
	s_set_vgpr_msb 0x50
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[258:265]*/, v[144:151], v[72:79], v[2:9] /*v[258:265]*/
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x32_bf16 v[12:19] /*v[268:275]*/, v[112:119], v[72:79], v[12:19] /*v[268:275]*/
	s_set_vgpr_msb 0x5000
	v_wmma_f32_16x16x32_bf16 v[240:247], v[200:207], v[80:87], v[240:247]
	v_wmma_f32_16x16x32_bf16 v[248:255], v[168:175], v[80:87], v[248:255]
	s_set_vgpr_msb 0x50
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[258:265]*/, v[136:143], v[80:87], v[2:9] /*v[258:265]*/
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[12:19] /*v[268:275]*/, v[104:111], v[80:87], v[12:19] /*v[268:275]*/
	s_set_vgpr_msb 0x5000
	v_wmma_f32_16x16x32_bf16 v[240:247], v[192:199], v[88:95], v[240:247]
	v_wmma_f32_16x16x32_bf16 v[248:255], v[160:167], v[88:95], v[248:255]
	s_set_vgpr_msb 0x50
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[258:265]*/, v[128:135], v[88:95], v[2:9] /*v[258:265]*/
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[12:19] /*v[268:275]*/, v[96:103], v[88:95], v[12:19] /*v[268:275]*/
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x5000
	ds_load_tr16_b128 v[216:219], v238
	ds_load_tr16_b128 v[200:203], v238 offset:32
	ds_load_tr16_b128 v[220:223], v238 offset:4608
	ds_load_tr16_b128 v[204:207], v238 offset:4640
	ds_load_tr16_b128 v[208:211], v238 offset:9216
	ds_load_tr16_b128 v[192:195], v238 offset:9248
	ds_load_tr16_b128 v[212:215], v238 offset:13824
	ds_load_tr16_b128 v[196:199], v238 offset:13856
	ds_load_tr16_b128 v[184:187], v238 offset:64
	ds_load_tr16_b128 v[168:171], v238 offset:96
	ds_load_tr16_b128 v[188:191], v238 offset:4672
	ds_load_tr16_b128 v[172:175], v238 offset:4704
	ds_load_tr16_b128 v[176:179], v238 offset:9280
	ds_load_tr16_b128 v[160:163], v238 offset:9312
	ds_load_tr16_b128 v[180:183], v238 offset:13888
	ds_load_tr16_b128 v[164:167], v238 offset:13920
	ds_load_tr16_b128 v[152:155], v238 offset:128
	ds_load_tr16_b128 v[136:139], v238 offset:160
	ds_load_tr16_b128 v[156:159], v238 offset:4736
	ds_load_tr16_b128 v[140:143], v238 offset:4768
	ds_load_tr16_b128 v[144:147], v238 offset:9344
	ds_load_tr16_b128 v[128:131], v238 offset:9376
	ds_load_tr16_b128 v[148:151], v238 offset:13952
	ds_load_tr16_b128 v[132:135], v238 offset:13984
	ds_load_tr16_b128 v[120:123], v238 offset:192
	ds_load_tr16_b128 v[104:107], v238 offset:224
	ds_load_tr16_b128 v[124:127], v238 offset:4800
	ds_load_tr16_b128 v[108:111], v238 offset:4832
	ds_load_tr16_b128 v[112:115], v238 offset:9408
	ds_load_tr16_b128 v[96:99], v238 offset:9440
	ds_load_tr16_b128 v[116:119], v238 offset:14016
	ds_load_tr16_b128 v[100:103], v238 offset:14048
	v_nop
	v_max_num_f32_e32 v237, v240, v241
	v_max3_num_f32 v239, v243, v244, v245
	s_set_vgpr_msb 64
	v_max3_num_f32 v10 /*v266*/, v249, v250, v251
	v_max3_num_f32 v11 /*v267*/, v252, v253, v254
	s_set_vgpr_msb 0x4054
	v_max3_num_f32 v20 /*v276*/, v255, v2 /*v258*/, v3 /*v259*/
	s_set_vgpr_msb 0x5440
	v_max3_num_f32 v1 /*v257*/, v246, v247, v248
	s_set_vgpr_msb 0x4055
	v_max3_num_f32 v21 /*v277*/, v4 /*v260*/, v5 /*v261*/, v6 /*v262*/
	v_max3_num_f32 v22 /*v278*/, v7 /*v263*/, v8 /*v264*/, v9 /*v265*/
	v_max3_num_f32 v23 /*v279*/, v12 /*v268*/, v13 /*v269*/, v14 /*v270*/
	v_max3_num_f32 v24 /*v280*/, v15 /*v271*/, v16 /*v272*/, v17 /*v273*/
	s_set_vgpr_msb 0x5500
	v_max3_num_f32 v237, v237, v242, v239
	s_set_vgpr_msb 21
	v_max3_num_f32 v239, v10 /*v266*/, v11 /*v267*/, v20 /*v276*/
	s_set_vgpr_msb 0x1555
	v_max3_num_f32 v10 /*v266*/, v21 /*v277*/, v22 /*v278*/, v23 /*v279*/
	v_max3_num_f32 v11 /*v267*/, v24 /*v280*/, v18 /*v274*/, v19 /*v275*/
	s_set_vgpr_msb 0x5504
	v_max3_num_f32 v237, v237, v1 /*v257*/, v239
	s_set_vgpr_msb 0x414
	v_max3_num_f32 v237, v237, v10 /*v266*/, v11 /*v267*/
	v_mov_b32_e32 v239, v237
	v_permlanex16_b32 v239, v239, s50, 0xfedcba98
	s_set_vgpr_msb 0x1400
	v_max_num_f32_e32 v237, v237, v239
	v_dual_sub_f32 v239, v237, v224 :: v_dual_max_num_f32 v237, v224, v237
	v_cmp_lt_f32_e32 vcc_lo, 0x41000000, v239
	s_cmp_eq_u32 vcc_lo, 0
	s_cselect_b32 s2, -1, 0
	v_cndmask_b32_e64 v237, v237, v224, s2
	s_set_vgpr_msb 64
	v_mul_f32_e32 v20 /*v276*/, 0xbfb8aa3b, v237
	s_set_vgpr_msb 0x4010
	v_sub_f32_e32 v224, v224, v237
	v_pk_fma_f32 v[246:247], v[246:247], s[48:49], v[20:21] /*v[276:277]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1051
	v_pk_fma_f32 v[32:33] /*v[288:289]*/, v[6:7] /*v[262:263]*/, s[48:49], v[20:21] /*v[276:277]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[34:35] /*v[290:291]*/, v[8:9] /*v[264:265]*/, s[48:49], v[20:21] /*v[276:277]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[28:29] /*v[284:285]*/, v[2:3] /*v[258:259]*/, s[48:49], v[20:21] /*v[276:277]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5150
	v_pk_fma_f32 v[22:23] /*v[278:279]*/, v[250:251], s[48:49], v[20:21] /*v[276:277]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v2 /*v258*/, v246
	v_exp_f32_e32 v3 /*v259*/, v247
	s_set_vgpr_msb 0x5001
	v_exp_f32_e32 v246, v33 /*v289*/
	v_exp_f32_e32 v247, v34 /*v290*/
	s_set_vgpr_msb 0x141
	v_exp_f32_e32 v8 /*v264*/, v23 /*v279*/
	v_exp_f32_e32 v7 /*v263*/, v22 /*v278*/
	s_set_vgpr_msb 0x4100
	v_mul_f32_e32 v224, 0x3fb8aa3b, v224
	s_set_vgpr_msb 64
	v_add_f32_e32 v23 /*v279*/, v246, v247
	s_set_vgpr_msb 0x4010
	v_pk_fma_f32 v[240:241], v[240:241], s[48:49], v[20:21] /*v[276:277]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[242:243], v[242:243], s[48:49], v[20:21] /*v[276:277]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[244:245], v[244:245], s[48:49], v[20:21] /*v[276:277]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1050
	v_pk_fma_f32 v[10:11] /*v[266:267]*/, v[248:249], s[48:49], v[20:21] /*v[276:277]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[24:25] /*v[280:281]*/, v[252:253], s[48:49], v[20:21] /*v[276:277]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[26:27] /*v[282:283]*/, v[254:255], s[48:49], v[20:21] /*v[276:277]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5000
	v_exp_f32_e32 v248, v240
	v_exp_f32_e32 v250, v241
	v_exp_f32_e32 v253, v243
	v_exp_f32_e32 v254, v244
	s_set_vgpr_msb 0x51
	v_pk_fma_f32 v[30:31] /*v[286:287]*/, v[4:5] /*v[260:261]*/, s[48:49], v[20:21] /*v[276:277]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[36:37] /*v[292:293]*/, v[12:13] /*v[268:269]*/, s[48:49], v[20:21] /*v[276:277]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[38:39] /*v[294:295]*/, v[14:15] /*v[270:271]*/, s[48:49], v[20:21] /*v[276:277]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5100
	v_exp_f32_e32 v252, v242
	v_exp_f32_e32 v255, v245
	s_set_vgpr_msb 0x51
	v_exp_f32_e32 v5 /*v261*/, v10 /*v266*/
	v_exp_f32_e32 v6 /*v262*/, v11 /*v267*/
	v_exp_f32_e32 v10 /*v266*/, v24 /*v280*/
	v_exp_f32_e32 v12 /*v268*/, v25 /*v281*/
	v_pk_fma_f32 v[16:17] /*v[272:273]*/, v[16:17] /*v[272:273]*/, s[48:49], v[20:21] /*v[276:277]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[18:19] /*v[274:275]*/, v[18:19] /*v[274:275]*/, s[48:49], v[20:21] /*v[276:277]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5100
	v_add_f32_e32 v239, v248, v250
	v_add_f32_e32 v240, v253, v254
	s_set_vgpr_msb 0x41
	v_exp_f32_e32 v14 /*v270*/, v26 /*v282*/
	v_exp_f32_e32 v15 /*v271*/, v27 /*v283*/
	s_set_vgpr_msb 0x4101
	v_exp_f32_e32 v241, v28 /*v284*/
	v_exp_f32_e32 v249, v35 /*v291*/
	s_set_vgpr_msb 0x141
	v_exp_f32_e32 v9 /*v265*/, v39 /*v295*/
	v_exp_f32_e32 v11 /*v267*/, v16 /*v272*/
	v_exp_f32_e32 v13 /*v269*/, v17 /*v273*/
	v_exp_f32_e32 v17 /*v273*/, v19 /*v275*/
	s_set_vgpr_msb 0x4100
	v_dual_add_f32 v239, v252, v239 :: v_dual_add_f32 v240, v255, v240
	s_set_vgpr_msb 0x45
	v_add_f32_e32 v19 /*v275*/, v6 /*v262*/, v7 /*v263*/
	v_add_f32_e32 v20 /*v276*/, v10 /*v266*/, v12 /*v268*/
	s_set_vgpr_msb 0x4501
	v_exp_f32_e32 v242, v29 /*v285*/
	v_exp_f32_e32 v243, v30 /*v286*/
	v_exp_f32_e32 v244, v31 /*v287*/
	v_exp_f32_e32 v251, v36 /*v292*/
	s_set_vgpr_msb 0x141
	v_exp_f32_e32 v1 /*v257*/, v37 /*v293*/
	v_exp_f32_e32 v16 /*v272*/, v18 /*v274*/
	v_add_f32_e32 v21 /*v277*/, v15 /*v271*/, v241
	s_set_vgpr_msb 0x4145
	v_dual_add_f32 v19 /*v275*/, v8 /*v264*/, v19 /*v275*/ :: v_dual_add_f32 v20 /*v276*/, v14 /*v270*/, v20 /*v276*/
	s_set_vgpr_msb 0x4500
	v_add_f32_e32 v239, v239, v240
	s_set_vgpr_msb 4
	v_add_f32_e32 v240, v249, v23 /*v279*/
	s_set_vgpr_msb 0x445
	v_add_f32_e32 v23 /*v279*/, v9 /*v265*/, v11 /*v267*/
	v_dual_add_f32 v18 /*v274*/, v2 /*v258*/, v3 /*v259*/ :: v_dual_add_f32 v19 /*v275*/, v19 /*v275*/, v20 /*v276*/
	s_set_vgpr_msb 0x4501
	v_exp_f32_e32 v245, v32 /*v288*/
	s_set_vgpr_msb 0x141
	v_exp_f32_e32 v4 /*v260*/, v38 /*v294*/
	s_set_vgpr_msb 0x4140
	v_add_f32_e32 v22 /*v278*/, v243, v244
	s_set_vgpr_msb 0x4045
	v_add_f32_e32 v18 /*v274*/, v5 /*v261*/, v18 /*v274*/
	s_set_vgpr_msb 0x4544
	v_add_f32_e32 v21 /*v277*/, v242, v21 /*v277*/
	v_add_f32_e32 v24 /*v280*/, v251, v1 /*v257*/
	s_set_vgpr_msb 0x4400
	v_exp_f32_e32 v224, v224
	s_set_vgpr_msb 0x44
	v_add_f32_e32 v22 /*v278*/, v245, v22 /*v278*/
	s_set_vgpr_msb 0x4445
	v_add_f32_e32 v20 /*v276*/, v4 /*v260*/, v24 /*v280*/
	s_set_vgpr_msb 0x4501
	v_add_f32_e32 v239, v18 /*v274*/, v239
	s_set_vgpr_msb 0x145
	v_add_f32_e32 v18 /*v274*/, v21 /*v277*/, v19 /*v275*/
	v_dual_add_f32 v19 /*v275*/, v13 /*v269*/, v23 /*v279*/ :: v_dual_add_f32 v21 /*v277*/, v16 /*v272*/, v17 /*v273*/
	s_set_vgpr_msb 0x4504
	v_add_f32_e32 v239, v239, v18 /*v274*/
	v_add_f32_e32 v240, v240, v22 /*v278*/
	s_set_vgpr_msb 0x445
	v_add_f32_e32 v18 /*v274*/, v21 /*v277*/, v19 /*v275*/
	s_set_vgpr_msb 0x4501
	v_add_f32_e32 v240, v20 /*v276*/, v240
	s_set_vgpr_msb 0x100
	v_add_f32_e32 v239, v240, v239
	s_set_vgpr_msb 1
	v_add_f32_e32 v239, v18 /*v274*/, v239
	s_set_vgpr_msb 0x100
	v_mov_b32_e32 v240, v239
	v_permlanex16_b32 v240, v240, s50, 0xfedcba98
	s_cbranch_vccz .LBB0_14
	v_pk_mul_f32 v[62:63], v[62:63], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[60:61], v[60:61], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59], v[58:59], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[56:57], v[56:57], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[54:55], v[54:55], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[52:53], v[52:53], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[50:51], v[50:51], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[48:49], v[48:49], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[46:47], v[46:47], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[44:45], v[44:45], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[42:43], v[42:43], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[40:41], v[40:41], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39], v[38:39], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[36:37], v[36:37], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35], v[34:35], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[32:33], v[32:33], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[30:31], v[30:31], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[28:29], v[28:29], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[26:27], v[26:27], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25], v[24:25], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[22:23], v[22:23], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21], v[20:21], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[18:19], v[18:19], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[16:17], v[16:17], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[14:15], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13], v[12:13], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[10:11], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[8:9], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[6:7], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[4:5], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[2:3], v[2:3], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[0:1], v[0:1], v[224:225] op_sel_hi:[1,0]
	s_branch .LBB0_14
.LBB0_19:
	s_set_vgpr_msb 1
	v_readlane_b32 s79, v0 /*v256*/, 2
	s_cmp_ge_u32 s62, s10
	s_set_vgpr_msb 0x100
	s_cbranch_scc1 .LBB0_10
.LBB0_20:
	v_nop
	v_nop
	v_nop
	v_add_nc_u32_e32 v96, s11, v227
	s_add_co_i32 s2, s78, -1
	s_mov_b32 s19, 0
	s_mov_b32 s72, 1
	s_mov_b32 s11, s19
	v_min_i32_e32 v244, s2, v96
	s_mov_b32 s16, 8
	s_mov_b32 s15, 0x800000
	s_mov_b32 s13, 0xffff0000
	s_mov_b32 s12, 0x7510000
	s_mov_b32 s20, 0xf510000
	s_mov_b32 s61, 0x76543210
	s_mov_b32 s76, 0x3fb8aa3b
	s_branch .LBB0_22
.LBB0_21:
	s_set_vgpr_msb 0x45
	v_cvt_pk_bf16_f32 v31 /*v287*/, v12 /*v268*/, v14 /*v270*/
	v_cvt_pk_bf16_f32 v30 /*v286*/, v8 /*v264*/, v10 /*v266*/
	v_cvt_pk_bf16_f32 v29 /*v285*/, v4 /*v260*/, v6 /*v262*/
	s_set_vgpr_msb 0x4544
	v_cvt_pk_bf16_f32 v28 /*v284*/, v255, v2 /*v258*/
	s_set_vgpr_msb 0x4440
	v_cvt_pk_bf16_f32 v27 /*v283*/, v252, v253
	v_cvt_pk_bf16_f32 v26 /*v282*/, v250, v251
	v_cvt_pk_bf16_f32 v25 /*v281*/, v248, v249
	v_cvt_pk_bf16_f32 v24 /*v280*/, v246, v247
	s_set_vgpr_msb 0x4004
	s_wait_dscnt 0x1d
	v_wmma_f32_16x16x32_bf16 v[56:63], v[216:223], v[24:31] /*v[280:287]*/, v[56:63]
	s_set_vgpr_msb 0x400
	v_fmac_f32_e32 v243, v224, v232
	s_add_nc_u64 s[62:63], s[62:63], 1
	s_wait_dscnt 0x0
	v_cmp_ge_u64_e64 s2, s[62:63], s[10:11]
	v_dual_add_f32 v232, v243, v241 :: v_dual_mov_b32 v243, v238
	s_set_vgpr_msb 4
	v_wmma_f32_16x16x32_bf16 v[48:55], v[200:207], v[24:31] /*v[280:287]*/, v[48:55]
	s_set_vgpr_msb 0x405
	v_cvt_pk_bf16_f32 v223, v21 /*v277*/, v22 /*v278*/
	v_cvt_pk_bf16_f32 v222, v19 /*v275*/, v20 /*v276*/
	v_cvt_pk_bf16_f32 v221, v17 /*v273*/, v18 /*v274*/
	v_cvt_pk_bf16_f32 v220, v15 /*v271*/, v16 /*v272*/
	v_cvt_pk_bf16_f32 v219, v11 /*v267*/, v13 /*v269*/
	v_cvt_pk_bf16_f32 v218, v7 /*v263*/, v9 /*v265*/
	v_cvt_pk_bf16_f32 v217, v3 /*v259*/, v5 /*v261*/
	s_set_vgpr_msb 0x504
	v_wmma_f32_16x16x32_bf16 v[40:47], v[184:191], v[24:31] /*v[280:287]*/, v[40:47]
	v_cvt_pk_bf16_f32 v216, v254, v1 /*v257*/
	s_wait_alu depctr_vm_vsrc(0)
	v_dual_mov_b32 v238, v242 :: v_dual_mov_b32 v242, v239
	v_dual_mov_b32 v239, v245 :: v_dual_mov_b32 v241, v240
	s_and_b32 vcc_lo, exec_lo, s2
	v_wmma_f32_16x16x32_bf16 v[32:39], v[168:175], v[24:31] /*v[280:287]*/, v[32:39]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[152:159], v[24:31] /*v[280:287]*/, v[24:31]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[136:143], v[24:31] /*v[280:287]*/, v[16:23]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[120:127], v[24:31] /*v[280:287]*/, v[8:15]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[104:111], v[24:31] /*v[280:287]*/, v[0:7]
	s_set_vgpr_msb 0x400
	v_wmma_f32_16x16x32_bf16 v[56:63], v[208:215], v[216:223], v[56:63]
	v_wmma_f32_16x16x32_bf16 v[48:55], v[192:199], v[216:223], v[48:55]
	v_wmma_f32_16x16x32_bf16 v[40:47], v[176:183], v[216:223], v[40:47]
	v_wmma_f32_16x16x32_bf16 v[32:39], v[160:167], v[216:223], v[32:39]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[144:151], v[216:223], v[24:31]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[128:135], v[216:223], v[16:23]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[112:119], v[216:223], v[8:15]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[96:103], v[216:223], v[0:7]
	s_cbranch_vccnz .LBB0_11
.LBB0_22:
	s_wait_alu depctr_vm_vsrc(6)
	v_dual_mov_b32 v245, v242 :: v_dual_mov_b32 v242, v243
	s_add_co_i32 s2, s62, 1
	s_wait_tensorcnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	s_cmp_ge_i32 s2, s10
	s_cbranch_scc1 .LBB0_24
	s_lshl_b32 s14, s2, 6
	s_lshl_b32 s2, s2, 17
	s_sub_co_i32 s21, s9, s14
	s_add_co_i32 s22, s14, s84
	v_nop
	v_nop
	v_nop
	v_med3_i32 v96, s21, 0, 64
	s_ashr_i32 s23, s22, 31
	s_and_b32 s2, s2, 0x20000
	s_mul_u64 s[74:75], s[22:23], s[82:83]
	s_mul_u64 s[22:23], s[22:23], s[80:81]
	v_readfirstlane_b32 s14, v96
	s_lshl_b64 s[22:23], s[22:23], 1
	s_lshl_b64 s[74:75], s[74:75], 1
	s_add_nc_u64 s[22:23], s[100:101], s[22:23]
	s_add_nc_u64 s[90:91], s[98:99], s[74:75]
	s_sub_co_i32 s14, s14, s35
	s_add_nc_u64 s[74:75], s[86:87], s[22:23]
	s_max_i32 s14, s14, 0
	s_or_b32 s73, s85, s2
	s_lshl_b32 s14, s14, 16
	s_bitset1_b32 s75, 31
	s_addk_co_i32 s14, 0x7fff
	s_mov_b32 s21, s13
	tensor_load_to_lds s[72:75], s[12:19]
	s_add_nc_u64 s[74:75], s[88:89], s[90:91]
	s_or_b32 s73, s3, s2
	s_bitset1_b32 s75, 31
	s_mov_b32 s22, s14
	s_mov_b32 s23, s15
	s_mov_b32 s24, s16
	s_mov_b32 s27, s19
	tensor_load_to_lds s[72:75], s[20:27]
.LBB0_24:
	s_wait_alu depctr_va_vdst(0)
	ds_load_b128 v[96:99], v239
	ds_load_b128 v[100:103], v239 offset:32
	ds_load_b128 v[104:107], v239 offset:64
	ds_load_b128 v[108:111], v239 offset:96
	ds_load_b128 v[112:115], v239 offset:128
	ds_load_b128 v[116:119], v239 offset:160
	ds_load_b128 v[120:123], v239 offset:192
	ds_load_b128 v[124:127], v239 offset:224
	ds_load_b128 v[128:131], v239 offset:4352
	ds_load_b128 v[132:135], v239 offset:4384
	ds_load_b128 v[136:139], v239 offset:4416
	ds_load_b128 v[140:143], v239 offset:4448
	ds_load_b128 v[144:147], v239 offset:4480
	ds_load_b128 v[148:151], v239 offset:4512
	ds_load_b128 v[152:155], v239 offset:4544
	ds_load_b128 v[156:159], v239 offset:4576
	ds_load_b128 v[160:163], v239 offset:8704
	ds_load_b128 v[164:167], v239 offset:8736
	ds_load_b128 v[168:171], v239 offset:8768
	ds_load_b128 v[172:175], v239 offset:8800
	ds_load_b128 v[176:179], v239 offset:8832
	ds_load_b128 v[180:183], v239 offset:8864
	ds_load_b128 v[184:187], v239 offset:8896
	ds_load_b128 v[188:191], v239 offset:8928
	ds_load_b128 v[192:195], v239 offset:13056
	ds_load_b128 v[196:199], v239 offset:13088
	ds_load_b128 v[200:203], v239 offset:13120
	ds_load_b128 v[204:207], v239 offset:13152
	ds_load_b128 v[208:211], v239 offset:13184
	ds_load_b128 v[212:215], v239 offset:13216
	ds_load_b128 v[216:219], v239 offset:13248
	ds_load_b128 v[220:223], v239 offset:13280
	s_wait_dscnt 0x1e
	v_wmma_f32_16x16x32_bf16 v[246:253], v[96:103], v[64:71], 0
	s_set_vgpr_msb 64
	s_wait_dscnt 0x16
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[258:265]*/, v[128:135], v[64:71], 0
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[266:273]*/, v[160:167], v[64:71], 0
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[274:281]*/, v[192:199], v[64:71], 0
	s_set_vgpr_msb 0x4000
	v_wmma_f32_16x16x32_bf16 v[246:253], v[104:111], v[72:79], v[246:253]
	s_set_vgpr_msb 0x50
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[258:265]*/, v[136:143], v[72:79], v[2:9] /*v[258:265]*/
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[266:273]*/, v[168:175], v[72:79], v[10:17] /*v[266:273]*/
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[274:281]*/, v[200:207], v[72:79], v[18:25] /*v[274:281]*/
	s_set_vgpr_msb 0x5000
	v_wmma_f32_16x16x32_bf16 v[246:253], v[112:119], v[80:87], v[246:253]
	s_set_vgpr_msb 0x50
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[258:265]*/, v[144:151], v[80:87], v[2:9] /*v[258:265]*/
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[266:273]*/, v[176:183], v[80:87], v[10:17] /*v[266:273]*/
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[274:281]*/, v[208:215], v[80:87], v[18:25] /*v[274:281]*/
	s_set_vgpr_msb 0x5000
	v_wmma_f32_16x16x32_bf16 v[246:253], v[120:127], v[88:95], v[246:253]
	s_set_vgpr_msb 0x50
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[258:265]*/, v[152:159], v[88:95], v[2:9] /*v[258:265]*/
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[266:273]*/, v[184:191], v[88:95], v[10:17] /*v[266:273]*/
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[274:281]*/, v[216:223], v[88:95], v[18:25] /*v[274:281]*/
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x5000
	ds_load_tr16_b128 v[216:219], v238
	ds_load_tr16_b128 v[200:203], v238 offset:32
	ds_load_tr16_b128 v[220:223], v238 offset:4608
	ds_load_tr16_b128 v[204:207], v238 offset:4640
	ds_load_tr16_b128 v[208:211], v238 offset:9216
	ds_load_tr16_b128 v[192:195], v238 offset:9248
	ds_load_tr16_b128 v[212:215], v238 offset:13824
	ds_load_tr16_b128 v[196:199], v238 offset:13856
	ds_load_tr16_b128 v[184:187], v238 offset:64
	ds_load_tr16_b128 v[168:171], v238 offset:96
	ds_load_tr16_b128 v[188:191], v238 offset:4672
	ds_load_tr16_b128 v[172:175], v238 offset:4704
	ds_load_tr16_b128 v[176:179], v238 offset:9280
	ds_load_tr16_b128 v[160:163], v238 offset:9312
	ds_load_tr16_b128 v[180:183], v238 offset:13888
	ds_load_tr16_b128 v[164:167], v238 offset:13920
	ds_load_tr16_b128 v[152:155], v238 offset:128
	ds_load_tr16_b128 v[136:139], v238 offset:160
	ds_load_tr16_b128 v[156:159], v238 offset:4736
	ds_load_tr16_b128 v[140:143], v238 offset:4768
	ds_load_tr16_b128 v[144:147], v238 offset:9344
	ds_load_tr16_b128 v[128:131], v238 offset:9376
	ds_load_tr16_b128 v[148:151], v238 offset:13952
	ds_load_tr16_b128 v[132:135], v238 offset:13984
	ds_load_tr16_b128 v[120:123], v238 offset:192
	ds_load_tr16_b128 v[104:107], v238 offset:224
	ds_load_tr16_b128 v[124:127], v238 offset:4800
	ds_load_tr16_b128 v[108:111], v238 offset:4832
	ds_load_tr16_b128 v[112:115], v238 offset:9408
	ds_load_tr16_b128 v[96:99], v238 offset:9440
	ds_load_tr16_b128 v[116:119], v238 offset:14016
	ds_load_tr16_b128 v[100:103], v238 offset:14048
	v_lshl_or_b32 v224, s62, 6, v233
	v_cmp_le_i32_e32 vcc_lo, v224, v244
	v_or_b32_e32 v240, 2, v224
	s_wait_alu depctr_vm_vsrc(6)
	v_or_b32_e32 v243, 3, v224
	v_or_b32_e32 v254, 4, v224
	s_set_vgpr_msb 64
	v_add_nc_u32_e32 v1 /*v257*/, 18, v224
	s_set_vgpr_msb 0x4000
	v_cndmask_b32_e32 v246, 0xff800000, v246, vcc_lo
	v_cmp_lt_i32_e32 vcc_lo, v224, v244
	v_cndmask_b32_e32 v247, 0xff800000, v247, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v240, v244
	v_or_b32_e32 v240, 5, v224
	v_cndmask_b32_e32 v248, 0xff800000, v248, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v243, v244
	v_or_b32_e32 v243, 6, v224
	v_cndmask_b32_e32 v249, 0xff800000, v249, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v254, v244
	v_or_b32_e32 v254, 7, v224
	v_cndmask_b32_e32 v250, 0xff800000, v250, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v240, v244
	v_or_b32_e32 v240, 16, v224
	v_cndmask_b32_e32 v251, 0xff800000, v251, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v243, v244
	v_add_nc_u32_e32 v243, 17, v224
	v_cndmask_b32_e32 v252, 0xff800000, v252, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v254, v244
	v_cndmask_b32_e32 v253, 0xff800000, v253, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v240, v244
	v_add_nc_u32_e32 v240, 19, v224
	s_set_vgpr_msb 4
	v_cndmask_b32_e32 v254, 0xff800000, v2 /*v258*/, vcc_lo
	s_set_vgpr_msb 0x400
	v_cmp_le_i32_e32 vcc_lo, v243, v244
	v_add_nc_u32_e32 v243, 20, v224
	s_set_vgpr_msb 4
	v_cndmask_b32_e32 v255, 0xff800000, v3 /*v259*/, vcc_lo
	v_cmp_ge_i32_e32 vcc_lo, v244, v1 /*v257*/
	s_set_vgpr_msb 0x440
	v_add_nc_u32_e32 v1 /*v257*/, 22, v224
	s_set_vgpr_msb 0x4044
	v_cndmask_b32_e32 v2 /*v258*/, 0xff800000, v4 /*v260*/, vcc_lo
	s_set_vgpr_msb 0x4400
	v_cmp_le_i32_e32 vcc_lo, v240, v244
	v_add_nc_u32_e32 v240, 21, v224
	s_set_vgpr_msb 0x44
	v_cndmask_b32_e32 v3 /*v259*/, 0xff800000, v5 /*v261*/, vcc_lo
	s_set_vgpr_msb 0x4400
	v_cmp_le_i32_e32 vcc_lo, v243, v244
	v_add_nc_u32_e32 v243, 23, v224
	s_set_vgpr_msb 0x44
	v_cndmask_b32_e32 v4 /*v260*/, 0xff800000, v6 /*v262*/, vcc_lo
	s_set_vgpr_msb 0x4400
	v_cmp_le_i32_e32 vcc_lo, v240, v244
	v_or_b32_e32 v240, 32, v224
	s_set_vgpr_msb 0x44
	v_cndmask_b32_e32 v5 /*v261*/, 0xff800000, v7 /*v263*/, vcc_lo
	v_cmp_ge_i32_e32 vcc_lo, v244, v1 /*v257*/
	s_set_vgpr_msb 0x4440
	v_or_b32_e32 v1 /*v257*/, 34, v224
	s_set_vgpr_msb 0x4044
	v_cndmask_b32_e32 v6 /*v262*/, 0xff800000, v8 /*v264*/, vcc_lo
	s_set_vgpr_msb 0x4400
	v_cmp_le_i32_e32 vcc_lo, v243, v244
	v_or_b32_e32 v243, 33, v224
	s_set_vgpr_msb 0x44
	v_cndmask_b32_e32 v7 /*v263*/, 0xff800000, v9 /*v265*/, vcc_lo
	s_set_vgpr_msb 0x4400
	v_cmp_le_i32_e32 vcc_lo, v240, v244
	v_or_b32_e32 v240, 35, v224
	s_set_vgpr_msb 0x55
	v_max3_num_f32 v8 /*v264*/, v4 /*v260*/, v5 /*v261*/, v6 /*v262*/
	v_cndmask_b32_e32 v10 /*v266*/, 0xff800000, v10 /*v266*/, vcc_lo
	s_set_vgpr_msb 0x5500
	v_cmp_le_i32_e32 vcc_lo, v243, v244
	v_or_b32_e32 v243, 36, v224
	s_set_vgpr_msb 0x44
	v_cndmask_b32_e32 v11 /*v267*/, 0xff800000, v11 /*v267*/, vcc_lo
	v_cmp_ge_i32_e32 vcc_lo, v244, v1 /*v257*/
	s_set_vgpr_msb 0x4440
	v_or_b32_e32 v1 /*v257*/, 38, v224
	s_set_vgpr_msb 0x4055
	v_max3_num_f32 v9 /*v265*/, v7 /*v263*/, v10 /*v266*/, v11 /*v267*/
	v_cndmask_b32_e32 v26 /*v282*/, 0xff800000, v12 /*v268*/, vcc_lo
	s_set_vgpr_msb 0x5500
	v_cmp_le_i32_e32 vcc_lo, v240, v244
	v_or_b32_e32 v240, 37, v224
	s_set_vgpr_msb 0x44
	v_cndmask_b32_e32 v27 /*v283*/, 0xff800000, v13 /*v269*/, vcc_lo
	s_set_vgpr_msb 0x4400
	v_cmp_le_i32_e32 vcc_lo, v243, v244
	v_or_b32_e32 v243, 39, v224
	s_set_vgpr_msb 0x44
	v_cndmask_b32_e32 v28 /*v284*/, 0xff800000, v14 /*v270*/, vcc_lo
	s_set_vgpr_msb 0x4400
	v_cmp_le_i32_e32 vcc_lo, v240, v244
	v_or_b32_e32 v240, 48, v224
	s_set_vgpr_msb 0x44
	v_cndmask_b32_e32 v29 /*v285*/, 0xff800000, v15 /*v271*/, vcc_lo
	v_cmp_ge_i32_e32 vcc_lo, v244, v1 /*v257*/
	s_set_vgpr_msb 0x4440
	v_add_nc_u32_e32 v1 /*v257*/, 50, v224
	s_set_vgpr_msb 0x4055
	v_max3_num_f32 v12 /*v268*/, v26 /*v282*/, v27 /*v283*/, v28 /*v284*/
	v_cndmask_b32_e32 v16 /*v272*/, 0xff800000, v16 /*v272*/, vcc_lo
	s_set_vgpr_msb 0x5500
	v_cmp_le_i32_e32 vcc_lo, v243, v244
	v_add_nc_u32_e32 v243, 49, v224
	s_set_vgpr_msb 0x44
	v_cndmask_b32_e32 v17 /*v273*/, 0xff800000, v17 /*v273*/, vcc_lo
	s_set_vgpr_msb 0x4400
	v_cmp_le_i32_e32 vcc_lo, v240, v244
	v_add_nc_u32_e32 v240, 51, v224
	s_set_vgpr_msb 0x55
	v_max3_num_f32 v13 /*v269*/, v29 /*v285*/, v16 /*v272*/, v17 /*v273*/
	v_cndmask_b32_e32 v18 /*v274*/, 0xff800000, v18 /*v274*/, vcc_lo
	s_set_vgpr_msb 0x5500
	v_cmp_le_i32_e32 vcc_lo, v243, v244
	v_add_nc_u32_e32 v243, 52, v224
	s_set_vgpr_msb 0x44
	v_cndmask_b32_e32 v19 /*v275*/, 0xff800000, v19 /*v275*/, vcc_lo
	v_cmp_ge_i32_e32 vcc_lo, v244, v1 /*v257*/
	s_set_vgpr_msb 0x4440
	v_add_nc_u32_e32 v1 /*v257*/, 54, v224
	s_set_vgpr_msb 0x4044
	v_cndmask_b32_e32 v20 /*v276*/, 0xff800000, v20 /*v276*/, vcc_lo
	s_set_vgpr_msb 0x4400
	v_cmp_le_i32_e32 vcc_lo, v240, v244
	v_dual_add_nc_u32 v240, 53, v224 :: v_dual_add_nc_u32 v224, 55, v224
	s_set_vgpr_msb 0x44
	v_cndmask_b32_e32 v21 /*v277*/, 0xff800000, v21 /*v277*/, vcc_lo
	s_set_vgpr_msb 0x4400
	v_cmp_le_i32_e32 vcc_lo, v243, v244
	v_max3_num_f32 v243, v252, v253, v254
	s_set_vgpr_msb 0x55
	v_max3_num_f32 v14 /*v270*/, v18 /*v274*/, v19 /*v275*/, v20 /*v276*/
	v_cndmask_b32_e32 v22 /*v278*/, 0xff800000, v22 /*v278*/, vcc_lo
	s_set_vgpr_msb 0x5500
	v_cmp_le_i32_e32 vcc_lo, v240, v244
	v_max3_num_f32 v240, v249, v250, v251
	s_set_vgpr_msb 0x54
	v_cndmask_b32_e32 v23 /*v279*/, 0xff800000, v23 /*v279*/, vcc_lo
	v_cmp_ge_i32_e32 vcc_lo, v244, v1 /*v257*/
	v_max3_num_f32 v1 /*v257*/, v255, v2 /*v258*/, v3 /*v259*/
	s_set_vgpr_msb 0x5455
	v_max3_num_f32 v15 /*v271*/, v21 /*v277*/, v22 /*v278*/, v23 /*v279*/
	v_cndmask_b32_e32 v24 /*v280*/, 0xff800000, v24 /*v280*/, vcc_lo
	s_set_vgpr_msb 0x5500
	v_cmp_le_i32_e32 vcc_lo, v224, v244
	v_max_num_f32_e32 v224, v246, v247
	s_set_vgpr_msb 0x44
	v_cndmask_b32_e32 v25 /*v281*/, 0xff800000, v25 /*v281*/, vcc_lo
	s_set_vgpr_msb 0x4400
	v_max3_num_f32 v224, v224, v248, v240
	s_set_vgpr_msb 21
	v_max3_num_f32 v240, v1 /*v257*/, v8 /*v264*/, v9 /*v265*/
	s_set_vgpr_msb 0x1555
	v_max3_num_f32 v1 /*v257*/, v12 /*v268*/, v13 /*v269*/, v14 /*v270*/
	v_max3_num_f32 v8 /*v264*/, v15 /*v271*/, v24 /*v280*/, v25 /*v281*/
	s_set_vgpr_msb 0x5500
	v_max3_num_f32 v224, v224, v243, v240
	s_set_vgpr_msb 20
	v_max3_num_f32 v224, v224, v1 /*v257*/, v8 /*v264*/
	v_mov_b32_e32 v240, v224
	v_permlanex16_b32 v240, v240, s61, 0xfedcba98
	s_set_vgpr_msb 0x1400
	v_max_num_f32_e32 v224, v224, v240
	v_dual_sub_f32 v240, v224, v241 :: v_dual_max_num_f32 v224, v241, v224
	v_cmp_lt_f32_e32 vcc_lo, 0x41000000, v240
	s_cmp_eq_u32 vcc_lo, 0
	s_cselect_b32 s2, -1, 0
	v_cndmask_b32_e64 v240, v224, v241, s2
	v_mul_f32_e32 v224, 0xbfb8aa3b, v240
	v_pk_fma_f32 v[246:247], v[246:247], s[76:77], v[224:225] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[248:249], v[248:249], s[76:77], v[224:225] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[250:251], v[250:251], s[76:77], v[224:225] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 64
	v_pk_fma_f32 v[8:9] /*v[264:265]*/, v[254:255], s[76:77], v[224:225] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x4041
	v_pk_fma_f32 v[12:13] /*v[268:269]*/, v[2:3] /*v[258:259]*/, s[76:77], v[224:225] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[14:15] /*v[270:271]*/, v[4:5] /*v[260:261]*/, s[76:77], v[224:225] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x4100
	v_exp_f32_e32 v246, v246
	v_exp_f32_e32 v247, v247
	v_pk_fma_f32 v[252:253], v[252:253], s[76:77], v[224:225] op_sel_hi:[1,0,0]
	v_exp_f32_e32 v249, v249
	v_exp_f32_e32 v250, v250
	s_set_vgpr_msb 1
	v_exp_f32_e32 v255, v8 /*v264*/
	s_set_vgpr_msb 0x141
	v_exp_f32_e32 v2 /*v258*/, v9 /*v265*/
	v_exp_f32_e32 v4 /*v260*/, v12 /*v268*/
	v_pk_fma_f32 v[30:31] /*v[286:287]*/, v[6:7] /*v[262:263]*/, s[76:77], v[224:225] op_sel_hi:[1,0,0]
	v_exp_f32_e32 v8 /*v264*/, v14 /*v270*/
	v_pk_fma_f32 v[32:33] /*v[288:289]*/, v[10:11] /*v[266:267]*/, s[76:77], v[224:225] op_sel_hi:[1,0,0]
	v_exp_f32_e32 v10 /*v266*/, v15 /*v271*/
	v_pk_fma_f32 v[26:27] /*v[282:283]*/, v[26:27] /*v[282:283]*/, s[76:77], v[224:225] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[28:29] /*v[284:285]*/, v[28:29] /*v[284:285]*/, s[76:77], v[224:225] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[16:17] /*v[272:273]*/, v[16:17] /*v[272:273]*/, s[76:77], v[224:225] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[18:19] /*v[274:275]*/, v[18:19] /*v[274:275]*/, s[76:77], v[224:225] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[20:21] /*v[276:277]*/, v[20:21] /*v[276:277]*/, s[76:77], v[224:225] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[22:23] /*v[278:279]*/, v[22:23] /*v[278:279]*/, s[76:77], v[224:225] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[24:25] /*v[280:281]*/, v[24:25] /*v[280:281]*/, s[76:77], v[224:225] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x4100
	v_exp_f32_e32 v248, v248
	v_exp_f32_e32 v251, v251
	v_exp_f32_e32 v252, v252
	v_exp_f32_e32 v253, v253
	s_set_vgpr_msb 0x41
	v_exp_f32_e32 v6 /*v262*/, v13 /*v269*/
	v_exp_f32_e32 v12 /*v268*/, v30 /*v286*/
	v_exp_f32_e32 v14 /*v270*/, v31 /*v287*/
	s_set_vgpr_msb 0x4101
	v_exp_f32_e32 v254, v32 /*v288*/
	s_set_vgpr_msb 0x141
	v_exp_f32_e32 v3 /*v259*/, v26 /*v282*/
	v_exp_f32_e32 v5 /*v261*/, v27 /*v283*/
	v_exp_f32_e32 v9 /*v265*/, v29 /*v285*/
	v_exp_f32_e32 v11 /*v267*/, v16 /*v272*/
	v_exp_f32_e32 v15 /*v271*/, v18 /*v274*/
	v_exp_f32_e32 v16 /*v272*/, v19 /*v275*/
	v_exp_f32_e32 v18 /*v274*/, v21 /*v277*/
	v_exp_f32_e32 v19 /*v275*/, v22 /*v278*/
	s_set_vgpr_msb 0x4100
	v_add_f32_e32 v224, v246, v247
	s_set_vgpr_msb 0x41
	v_exp_f32_e32 v21 /*v277*/, v24 /*v280*/
	s_set_vgpr_msb 0x4100
	v_add_f32_e32 v243, v249, v250
	s_set_vgpr_msb 0x45
	v_exp_f32_e32 v22 /*v278*/, v25 /*v281*/
	v_dual_add_f32 v24 /*v280*/, v2 /*v258*/, v4 /*v260*/ :: v_dual_add_f32 v25 /*v281*/, v8 /*v264*/, v10 /*v266*/
	v_exp_f32_e32 v1 /*v257*/, v33 /*v289*/
	v_exp_f32_e32 v7 /*v263*/, v28 /*v284*/
	v_exp_f32_e32 v13 /*v269*/, v17 /*v273*/
	v_exp_f32_e32 v17 /*v273*/, v20 /*v276*/
	v_exp_f32_e32 v20 /*v276*/, v23 /*v279*/
	v_nop
	s_set_vgpr_msb 0x4540
	v_add_f32_e32 v23 /*v279*/, v252, v253
	s_set_vgpr_msb 0x4000
	v_dual_add_f32 v224, v248, v224 :: v_dual_add_f32 v243, v251, v243
	s_set_vgpr_msb 0x41
	v_add_f32_e32 v26 /*v282*/, v14 /*v270*/, v254
	s_set_vgpr_msb 0x4145
	v_dual_add_f32 v24 /*v280*/, v6 /*v262*/, v24 /*v280*/ :: v_dual_add_f32 v27 /*v283*/, v3 /*v259*/, v5 /*v261*/
	v_dual_add_f32 v25 /*v281*/, v12 /*v268*/, v25 /*v281*/ :: v_dual_add_f32 v28 /*v284*/, v9 /*v265*/, v11 /*v267*/
	s_set_vgpr_msb 0x4544
	v_add_f32_e32 v23 /*v279*/, v255, v23 /*v279*/
	s_set_vgpr_msb 0x4445
	v_dual_add_f32 v26 /*v282*/, v1 /*v257*/, v26 /*v282*/ :: v_dual_add_f32 v29 /*v285*/, v15 /*v271*/, v16 /*v272*/
	v_dual_add_f32 v27 /*v283*/, v7 /*v263*/, v27 /*v283*/ :: v_dual_add_f32 v24 /*v280*/, v24 /*v280*/, v25 /*v281*/
	s_set_vgpr_msb 0x4500
	v_add_f32_e32 v224, v224, v243
	s_set_vgpr_msb 5
	v_add_f32_e32 v243, v13 /*v269*/, v28 /*v284*/
	s_set_vgpr_msb 0x545
	v_dual_add_f32 v25 /*v281*/, v17 /*v273*/, v29 /*v285*/ :: v_dual_add_f32 v28 /*v284*/, v18 /*v274*/, v19 /*v275*/
	s_set_vgpr_msb 0x4501
	v_add_f32_e32 v224, v23 /*v279*/, v224
	v_add_f32_e32 v243, v27 /*v283*/, v243
	s_set_vgpr_msb 0x145
	v_dual_add_f32 v23 /*v279*/, v26 /*v282*/, v24 /*v280*/ :: v_dual_add_f32 v26 /*v282*/, v21 /*v277*/, v22 /*v278*/
	v_add_f32_e32 v24 /*v280*/, v20 /*v276*/, v28 /*v284*/
	s_set_vgpr_msb 0x4501
	v_add_f32_e32 v243, v25 /*v281*/, v243
	v_add_f32_e32 v224, v23 /*v279*/, v224
	s_set_vgpr_msb 0x145
	v_add_f32_e32 v23 /*v279*/, v26 /*v282*/, v24 /*v280*/
	s_set_vgpr_msb 0x4500
	v_add_f32_e32 v224, v243, v224
	v_sub_f32_e32 v243, v241, v240
	s_set_vgpr_msb 1
	v_dual_add_f32 v241, v23 /*v279*/, v224 :: v_dual_mul_f32 v224, 0x3fb8aa3b, v243
	s_set_vgpr_msb 0x100
	v_mov_b32_e32 v243, v241
	v_exp_f32_e32 v224, v224
	v_permlanex16_b32 v243, v243, s61, 0xfedcba98
	s_cbranch_vccz .LBB0_21
	v_pk_mul_f32 v[62:63], v[62:63], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[60:61], v[60:61], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59], v[58:59], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[56:57], v[56:57], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[54:55], v[54:55], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[52:53], v[52:53], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[50:51], v[50:51], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[48:49], v[48:49], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[46:47], v[46:47], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[44:45], v[44:45], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[42:43], v[42:43], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[40:41], v[40:41], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39], v[38:39], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[36:37], v[36:37], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35], v[34:35], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[32:33], v[32:33], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[30:31], v[30:31], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[28:29], v[28:29], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[26:27], v[26:27], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25], v[24:25], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[22:23], v[22:23], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21], v[20:21], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[18:19], v[18:19], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[16:17], v[16:17], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[14:15], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13], v[12:13], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[10:11], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[8:9], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[6:7], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[4:5], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[2:3], v[2:3], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[0:1], v[0:1], v[224:225] op_sel_hi:[1,0]
	s_branch .LBB0_21
.LBB0_26:
	v_dual_mov_b32 v1, v0 :: v_dual_mov_b32 v2, v0
	v_dual_mov_b32 v3, v0 :: v_dual_mov_b32 v4, v0
	v_dual_mov_b32 v5, v0 :: v_dual_mov_b32 v6, v0
	v_dual_mov_b32 v7, v0 :: v_dual_mov_b32 v235, v231
	v_dual_mov_b32 v237, 0xf149f2ca :: v_dual_mov_b32 v232, v0
	v_mov_b64_e32 v[12:13], v[4:5]
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
.LBB0_27:
	s_cmp_ge_u32 s40, s10
	s_cbranch_scc1 .LBB0_34
	v_nop
	v_nop
	v_nop
	v_nop
	v_add_nc_u32_e32 v96, s11, v227
	s_add_co_i32 s2, s78, -1
	s_mov_b32 s19, 0
	s_mov_b32 s28, 1
	s_mov_b32 s11, s19
	v_min_i32_e32 v239, s2, v96
	s_mov_b32 s16, 8
	s_mov_b32 s15, 0x800000
	s_mov_b32 s13, 0xffff0000
	s_mov_b32 s12, 0x7510000
	s_mov_b32 s20, 0xf510000
	s_mov_b32 s43, 0x76543210
	s_mov_b32 s42, 0x3fb8aa3b
	s_branch .LBB0_30
.LBB0_29:
	s_set_vgpr_msb 0x45
	v_cvt_pk_bf16_f32 v27 /*v283*/, v8 /*v264*/, v10 /*v266*/
	v_cvt_pk_bf16_f32 v26 /*v282*/, v4 /*v260*/, v6 /*v262*/
	s_set_vgpr_msb 0x4544
	v_cvt_pk_bf16_f32 v25 /*v281*/, v255, v2 /*v258*/
	s_set_vgpr_msb 0x4440
	v_cvt_pk_bf16_f32 v24 /*v280*/, v251, v253
	v_cvt_pk_bf16_f32 v23 /*v279*/, v248, v249
	v_cvt_pk_bf16_f32 v22 /*v278*/, v246, v247
	v_cvt_pk_bf16_f32 v21 /*v277*/, v244, v245
	v_cvt_pk_bf16_f32 v20 /*v276*/, v242, v243
	s_set_vgpr_msb 0x4004
	s_wait_dscnt 0x1d
	v_wmma_f32_16x16x32_bf16 v[56:63], v[216:223], v[20:27] /*v[276:283]*/, v[56:63]
	s_set_vgpr_msb 0x400
	v_fmac_f32_e32 v238, v224, v232
	s_add_nc_u64 s[40:41], s[40:41], 1
	s_wait_dscnt 0x0
	v_cmp_ge_u64_e64 s2, s[40:41], s[10:11]
	v_dual_add_f32 v232, v238, v237 :: v_dual_mov_b32 v238, v236
	s_set_vgpr_msb 4
	v_wmma_f32_16x16x32_bf16 v[48:55], v[200:207], v[20:27] /*v[276:283]*/, v[48:55]
	s_set_vgpr_msb 0x405
	v_cvt_pk_bf16_f32 v223, v17 /*v273*/, v18 /*v274*/
	v_cvt_pk_bf16_f32 v222, v15 /*v271*/, v16 /*v272*/
	v_cvt_pk_bf16_f32 v221, v13 /*v269*/, v14 /*v270*/
	v_cvt_pk_bf16_f32 v220, v11 /*v267*/, v12 /*v268*/
	v_cvt_pk_bf16_f32 v219, v7 /*v263*/, v9 /*v265*/
	v_cvt_pk_bf16_f32 v218, v3 /*v259*/, v5 /*v261*/
	s_set_vgpr_msb 0x504
	v_cvt_pk_bf16_f32 v217, v254, v1 /*v257*/
	v_wmma_f32_16x16x32_bf16 v[40:47], v[184:191], v[20:27] /*v[276:283]*/, v[40:47]
	s_set_vgpr_msb 0x400
	v_cvt_pk_bf16_f32 v216, v250, v252
	s_wait_alu depctr_vm_vsrc(0)
	v_dual_mov_b32 v236, v234 :: v_dual_mov_b32 v234, v235
	v_dual_mov_b32 v235, v241 :: v_dual_mov_b32 v237, v240
	s_and_b32 vcc_lo, exec_lo, s2
	s_set_vgpr_msb 4
	v_wmma_f32_16x16x32_bf16 v[32:39], v[168:175], v[20:27] /*v[276:283]*/, v[32:39]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[152:159], v[20:27] /*v[276:283]*/, v[24:31]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[136:143], v[20:27] /*v[276:283]*/, v[16:23]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[120:127], v[20:27] /*v[276:283]*/, v[8:15]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[104:111], v[20:27] /*v[276:283]*/, v[0:7]
	s_set_vgpr_msb 0x400
	v_wmma_f32_16x16x32_bf16 v[56:63], v[208:215], v[216:223], v[56:63]
	v_wmma_f32_16x16x32_bf16 v[48:55], v[192:199], v[216:223], v[48:55]
	v_wmma_f32_16x16x32_bf16 v[40:47], v[176:183], v[216:223], v[40:47]
	v_wmma_f32_16x16x32_bf16 v[32:39], v[160:167], v[216:223], v[32:39]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[144:151], v[216:223], v[24:31]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[128:135], v[216:223], v[16:23]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[112:119], v[216:223], v[8:15]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[96:103], v[216:223], v[0:7]
	s_cbranch_vccnz .LBB0_35
.LBB0_30:
	s_wait_alu depctr_vm_vsrc(6)
	v_dual_mov_b32 v241, v234 :: v_dual_mov_b32 v234, v238
	s_add_co_i32 s2, s40, 1
	s_wait_tensorcnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	s_wait_alu depctr_va_vdst(0)
	ds_load_b128 v[216:219], v235
	ds_load_b128 v[220:223], v235 offset:32
	ds_load_b128 v[208:211], v235 offset:64
	ds_load_b128 v[212:215], v235 offset:96
	ds_load_b128 v[200:203], v235 offset:128
	ds_load_b128 v[204:207], v235 offset:160
	ds_load_b128 v[192:195], v235 offset:192
	ds_load_b128 v[196:199], v235 offset:224
	ds_load_b128 v[184:187], v235 offset:4352
	ds_load_b128 v[188:191], v235 offset:4384
	ds_load_b128 v[176:179], v235 offset:4416
	ds_load_b128 v[180:183], v235 offset:4448
	ds_load_b128 v[168:171], v235 offset:4480
	ds_load_b128 v[172:175], v235 offset:4512
	ds_load_b128 v[160:163], v235 offset:4544
	ds_load_b128 v[164:167], v235 offset:4576
	ds_load_b128 v[152:155], v235 offset:8704
	ds_load_b128 v[156:159], v235 offset:8736
	ds_load_b128 v[144:147], v235 offset:8768
	ds_load_b128 v[148:151], v235 offset:8800
	ds_load_b128 v[136:139], v235 offset:8832
	ds_load_b128 v[140:143], v235 offset:8864
	ds_load_b128 v[128:131], v235 offset:8896
	ds_load_b128 v[132:135], v235 offset:8928
	ds_load_b128 v[120:123], v235 offset:13056
	ds_load_b128 v[124:127], v235 offset:13088
	ds_load_b128 v[112:115], v235 offset:13120
	ds_load_b128 v[116:119], v235 offset:13152
	ds_load_b128 v[104:107], v235 offset:13184
	ds_load_b128 v[108:111], v235 offset:13216
	ds_load_b128 v[96:99], v235 offset:13248
	ds_load_b128 v[100:103], v235 offset:13280
	s_cmp_ge_i32 s2, s10
	s_cbranch_scc1 .LBB0_32
	s_lshl_b32 s14, s2, 6
	s_lshl_b32 s2, s2, 17
	s_sub_co_i32 s21, s9, s14
	s_add_co_i32 s22, s14, s84
	v_med3_i32 v224, s21, 0, 64
	s_ashr_i32 s23, s22, 31
	s_and_b32 s2, s2, 0x20000
	s_mul_u64 s[30:31], s[22:23], s[82:83]
	s_mul_u64 s[22:23], s[22:23], s[80:81]
	v_readfirstlane_b32 s14, v224
	s_lshl_b64 s[22:23], s[22:23], 1
	s_lshl_b64 s[30:31], s[30:31], 1
	s_add_nc_u64 s[22:23], s[38:39], s[22:23]
	s_add_nc_u64 s[44:45], s[36:37], s[30:31]
	s_sub_co_i32 s14, s14, s35
	s_add_nc_u64 s[30:31], s[86:87], s[22:23]
	s_max_i32 s14, s14, 0
	s_or_b32 s29, s85, s2
	s_lshl_b32 s14, s14, 16
	s_bitset1_b32 s31, 31
	s_addk_co_i32 s14, 0x7fff
	s_mov_b32 s21, s13
	tensor_load_to_lds s[28:31], s[12:19]
	s_add_nc_u64 s[30:31], s[88:89], s[44:45]
	s_or_b32 s29, s3, s2
	s_bitset1_b32 s31, 31
	s_mov_b32 s22, s14
	s_mov_b32 s23, s15
	s_mov_b32 s24, s16
	s_mov_b32 s27, s19
	tensor_load_to_lds s[28:31], s[20:27]
.LBB0_32:
	s_wait_dscnt 0x1e
	v_wmma_f32_16x16x32_bf16 v[242:249], v[216:223], v[64:71], 0
	s_set_vgpr_msb 64
	s_wait_dscnt 0x16
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[258:265]*/, v[184:191], v[64:71], 0
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[266:273]*/, v[152:159], v[64:71], 0
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[274:281]*/, v[120:127], v[64:71], 0
	s_set_vgpr_msb 0x4000
	v_wmma_f32_16x16x32_bf16 v[242:249], v[208:215], v[72:79], v[242:249]
	s_set_vgpr_msb 0x50
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[258:265]*/, v[176:183], v[72:79], v[2:9] /*v[258:265]*/
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[266:273]*/, v[144:151], v[72:79], v[10:17] /*v[266:273]*/
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[274:281]*/, v[112:119], v[72:79], v[18:25] /*v[274:281]*/
	s_set_vgpr_msb 0x5000
	v_wmma_f32_16x16x32_bf16 v[242:249], v[200:207], v[80:87], v[242:249]
	s_set_vgpr_msb 0x50
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[258:265]*/, v[168:175], v[80:87], v[2:9] /*v[258:265]*/
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[266:273]*/, v[136:143], v[80:87], v[10:17] /*v[266:273]*/
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[274:281]*/, v[104:111], v[80:87], v[18:25] /*v[274:281]*/
	s_set_vgpr_msb 0x5000
	v_wmma_f32_16x16x32_bf16 v[242:249], v[192:199], v[88:95], v[242:249]
	s_set_vgpr_msb 0x50
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[258:265]*/, v[160:167], v[88:95], v[2:9] /*v[258:265]*/
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[266:273]*/, v[128:135], v[88:95], v[10:17] /*v[266:273]*/
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[274:281]*/, v[96:103], v[88:95], v[18:25] /*v[274:281]*/
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x5000
	ds_load_tr16_b128 v[216:219], v236
	ds_load_tr16_b128 v[200:203], v236 offset:32
	ds_load_tr16_b128 v[220:223], v236 offset:4608
	ds_load_tr16_b128 v[204:207], v236 offset:4640
	ds_load_tr16_b128 v[208:211], v236 offset:9216
	ds_load_tr16_b128 v[192:195], v236 offset:9248
	ds_load_tr16_b128 v[212:215], v236 offset:13824
	ds_load_tr16_b128 v[196:199], v236 offset:13856
	ds_load_tr16_b128 v[184:187], v236 offset:64
	ds_load_tr16_b128 v[168:171], v236 offset:96
	ds_load_tr16_b128 v[188:191], v236 offset:4672
	ds_load_tr16_b128 v[172:175], v236 offset:4704
	ds_load_tr16_b128 v[176:179], v236 offset:9280
	ds_load_tr16_b128 v[160:163], v236 offset:9312
	ds_load_tr16_b128 v[180:183], v236 offset:13888
	ds_load_tr16_b128 v[164:167], v236 offset:13920
	ds_load_tr16_b128 v[152:155], v236 offset:128
	ds_load_tr16_b128 v[136:139], v236 offset:160
	ds_load_tr16_b128 v[156:159], v236 offset:4736
	ds_load_tr16_b128 v[140:143], v236 offset:4768
	ds_load_tr16_b128 v[144:147], v236 offset:9344
	ds_load_tr16_b128 v[128:131], v236 offset:9376
	ds_load_tr16_b128 v[148:151], v236 offset:13952
	ds_load_tr16_b128 v[132:135], v236 offset:13984
	ds_load_tr16_b128 v[120:123], v236 offset:192
	ds_load_tr16_b128 v[104:107], v236 offset:224
	ds_load_tr16_b128 v[124:127], v236 offset:4800
	ds_load_tr16_b128 v[108:111], v236 offset:4832
	ds_load_tr16_b128 v[112:115], v236 offset:9408
	ds_load_tr16_b128 v[96:99], v236 offset:9440
	ds_load_tr16_b128 v[116:119], v236 offset:14016
	ds_load_tr16_b128 v[100:103], v236 offset:14048
	v_lshl_or_b32 v224, s40, 6, v233
	v_cmp_le_i32_e32 vcc_lo, v224, v239
	s_wait_alu depctr_vm_vsrc(6)
	v_or_b32_e32 v238, 2, v224
	v_dual_add_nc_u32 v252, 18, v224 :: v_dual_bitop2_b32 v240, 3, v224 bitop3:0x54
	v_or_b32_e32 v250, 4, v224
	v_cndmask_b32_e32 v242, 0xff800000, v242, vcc_lo
	v_cmp_lt_i32_e32 vcc_lo, v224, v239
	s_set_vgpr_msb 64
	v_add_nc_u32_e32 v1 /*v257*/, 22, v224
	s_set_vgpr_msb 0x4000
	v_cndmask_b32_e32 v243, 0xff800000, v243, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v238, v239
	v_or_b32_e32 v238, 5, v224
	v_cndmask_b32_e32 v244, 0xff800000, v244, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v240, v239
	v_or_b32_e32 v240, 6, v224
	v_cndmask_b32_e32 v245, 0xff800000, v245, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v250, v239
	v_or_b32_e32 v250, 7, v224
	v_cndmask_b32_e32 v246, 0xff800000, v246, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v238, v239
	v_or_b32_e32 v238, 16, v224
	v_cndmask_b32_e32 v247, 0xff800000, v247, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v240, v239
	v_add_nc_u32_e32 v240, 17, v224
	v_cndmask_b32_e32 v248, 0xff800000, v248, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v250, v239
	v_cndmask_b32_e32 v249, 0xff800000, v249, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v238, v239
	v_add_nc_u32_e32 v238, 19, v224
	s_set_vgpr_msb 4
	v_cndmask_b32_e32 v250, 0xff800000, v2 /*v258*/, vcc_lo
	s_set_vgpr_msb 0x400
	v_cmp_le_i32_e32 vcc_lo, v240, v239
	v_add_nc_u32_e32 v240, 20, v224
	s_set_vgpr_msb 4
	v_cndmask_b32_e32 v251, 0xff800000, v3 /*v259*/, vcc_lo
	s_set_vgpr_msb 0x400
	v_cmp_le_i32_e32 vcc_lo, v252, v239
	s_set_vgpr_msb 4
	v_cndmask_b32_e32 v252, 0xff800000, v4 /*v260*/, vcc_lo
	s_set_vgpr_msb 0x400
	v_cmp_le_i32_e32 vcc_lo, v238, v239
	v_add_nc_u32_e32 v238, 21, v224
	s_set_vgpr_msb 4
	v_cndmask_b32_e32 v253, 0xff800000, v5 /*v261*/, vcc_lo
	s_set_vgpr_msb 0x400
	v_cmp_le_i32_e32 vcc_lo, v240, v239
	v_add_nc_u32_e32 v240, 23, v224
	s_set_vgpr_msb 4
	v_cndmask_b32_e32 v254, 0xff800000, v6 /*v262*/, vcc_lo
	s_set_vgpr_msb 0x400
	v_cmp_le_i32_e32 vcc_lo, v238, v239
	v_or_b32_e32 v238, 32, v224
	s_set_vgpr_msb 4
	v_cndmask_b32_e32 v255, 0xff800000, v7 /*v263*/, vcc_lo
	v_cmp_ge_i32_e32 vcc_lo, v239, v1 /*v257*/
	s_set_vgpr_msb 0x440
	v_or_b32_e32 v1 /*v257*/, 34, v224
	s_set_vgpr_msb 0x4044
	v_cndmask_b32_e32 v2 /*v258*/, 0xff800000, v8 /*v264*/, vcc_lo
	s_set_vgpr_msb 0x4400
	v_cmp_le_i32_e32 vcc_lo, v240, v239
	v_or_b32_e32 v240, 33, v224
	s_set_vgpr_msb 0x44
	v_cndmask_b32_e32 v3 /*v259*/, 0xff800000, v9 /*v265*/, vcc_lo
	s_set_vgpr_msb 0x4400
	v_cmp_le_i32_e32 vcc_lo, v238, v239
	v_or_b32_e32 v238, 35, v224
	s_set_vgpr_msb 0x50
	v_max3_num_f32 v4 /*v260*/, v254, v255, v2 /*v258*/
	s_set_vgpr_msb 0x5044
	v_cndmask_b32_e32 v6 /*v262*/, 0xff800000, v10 /*v266*/, vcc_lo
	s_set_vgpr_msb 0x4400
	v_cmp_le_i32_e32 vcc_lo, v240, v239
	v_or_b32_e32 v240, 36, v224
	s_set_vgpr_msb 0x44
	v_cndmask_b32_e32 v7 /*v263*/, 0xff800000, v11 /*v267*/, vcc_lo
	v_cmp_ge_i32_e32 vcc_lo, v239, v1 /*v257*/
	s_set_vgpr_msb 0x4440
	v_or_b32_e32 v1 /*v257*/, 38, v224
	s_set_vgpr_msb 0x4055
	v_max3_num_f32 v5 /*v261*/, v3 /*v259*/, v6 /*v262*/, v7 /*v263*/
	v_cndmask_b32_e32 v10 /*v266*/, 0xff800000, v12 /*v268*/, vcc_lo
	s_set_vgpr_msb 0x5500
	v_cmp_le_i32_e32 vcc_lo, v238, v239
	v_or_b32_e32 v238, 37, v224
	s_set_vgpr_msb 0x44
	v_cndmask_b32_e32 v11 /*v267*/, 0xff800000, v13 /*v269*/, vcc_lo
	s_set_vgpr_msb 0x4400
	v_cmp_le_i32_e32 vcc_lo, v240, v239
	v_or_b32_e32 v240, 39, v224
	s_set_vgpr_msb 0x44
	v_cndmask_b32_e32 v12 /*v268*/, 0xff800000, v14 /*v270*/, vcc_lo
	s_set_vgpr_msb 0x4400
	v_cmp_le_i32_e32 vcc_lo, v238, v239
	v_or_b32_e32 v238, 48, v224
	s_set_vgpr_msb 0x44
	v_cndmask_b32_e32 v13 /*v269*/, 0xff800000, v15 /*v271*/, vcc_lo
	v_cmp_ge_i32_e32 vcc_lo, v239, v1 /*v257*/
	s_set_vgpr_msb 0x4440
	v_add_nc_u32_e32 v1 /*v257*/, 50, v224
	s_set_vgpr_msb 0x4055
	v_max3_num_f32 v8 /*v264*/, v10 /*v266*/, v11 /*v267*/, v12 /*v268*/
	v_cndmask_b32_e32 v14 /*v270*/, 0xff800000, v16 /*v272*/, vcc_lo
	s_set_vgpr_msb 0x5500
	v_cmp_le_i32_e32 vcc_lo, v240, v239
	v_add_nc_u32_e32 v240, 49, v224
	s_set_vgpr_msb 0x44
	v_cndmask_b32_e32 v15 /*v271*/, 0xff800000, v17 /*v273*/, vcc_lo
	s_set_vgpr_msb 0x4400
	v_cmp_le_i32_e32 vcc_lo, v238, v239
	v_add_nc_u32_e32 v238, 51, v224
	s_set_vgpr_msb 0x55
	v_max3_num_f32 v9 /*v265*/, v13 /*v269*/, v14 /*v270*/, v15 /*v271*/
	v_cndmask_b32_e32 v16 /*v272*/, 0xff800000, v18 /*v274*/, vcc_lo
	s_set_vgpr_msb 0x5500
	v_cmp_le_i32_e32 vcc_lo, v240, v239
	v_add_nc_u32_e32 v240, 52, v224
	s_set_vgpr_msb 0x44
	v_cndmask_b32_e32 v17 /*v273*/, 0xff800000, v19 /*v275*/, vcc_lo
	v_cmp_ge_i32_e32 vcc_lo, v239, v1 /*v257*/
	s_set_vgpr_msb 0x4440
	v_add_nc_u32_e32 v1 /*v257*/, 54, v224
	s_set_vgpr_msb 0x4044
	v_cndmask_b32_e32 v18 /*v274*/, 0xff800000, v20 /*v276*/, vcc_lo
	s_set_vgpr_msb 0x4400
	v_cmp_le_i32_e32 vcc_lo, v238, v239
	v_dual_add_nc_u32 v238, 53, v224 :: v_dual_add_nc_u32 v224, 55, v224
	s_set_vgpr_msb 0x44
	v_cndmask_b32_e32 v19 /*v275*/, 0xff800000, v21 /*v277*/, vcc_lo
	s_set_vgpr_msb 0x4400
	v_cmp_le_i32_e32 vcc_lo, v240, v239
	v_max3_num_f32 v240, v248, v249, v250
	s_set_vgpr_msb 0x44
	v_cndmask_b32_e32 v20 /*v276*/, 0xff800000, v22 /*v278*/, vcc_lo
	s_set_vgpr_msb 0x4400
	v_cmp_le_i32_e32 vcc_lo, v238, v239
	v_max3_num_f32 v238, v245, v246, v247
	s_set_vgpr_msb 0x44
	v_cndmask_b32_e32 v21 /*v277*/, 0xff800000, v23 /*v279*/, vcc_lo
	v_cmp_ge_i32_e32 vcc_lo, v239, v1 /*v257*/
	s_set_vgpr_msb 0x4440
	v_max3_num_f32 v1 /*v257*/, v251, v252, v253
	s_set_vgpr_msb 0x4044
	v_cndmask_b32_e32 v22 /*v278*/, 0xff800000, v24 /*v280*/, vcc_lo
	s_set_vgpr_msb 0x4400
	v_cmp_le_i32_e32 vcc_lo, v224, v239
	v_max_num_f32_e32 v224, v242, v243
	s_set_vgpr_msb 0x55
	v_max3_num_f32 v24 /*v280*/, v16 /*v272*/, v17 /*v273*/, v18 /*v274*/
	v_cndmask_b32_e32 v23 /*v279*/, 0xff800000, v25 /*v281*/, vcc_lo
	v_max3_num_f32 v25 /*v281*/, v19 /*v275*/, v20 /*v276*/, v21 /*v277*/
	s_set_vgpr_msb 0x5500
	v_max3_num_f32 v224, v224, v244, v238
	s_set_vgpr_msb 21
	v_max3_num_f32 v238, v1 /*v257*/, v4 /*v260*/, v5 /*v261*/
	s_set_vgpr_msb 0x1555
	v_max3_num_f32 v1 /*v257*/, v8 /*v264*/, v9 /*v265*/, v24 /*v280*/
	v_max3_num_f32 v4 /*v260*/, v25 /*v281*/, v22 /*v278*/, v23 /*v279*/
	s_set_vgpr_msb 0x5500
	v_max3_num_f32 v224, v224, v240, v238
	s_set_vgpr_msb 20
	v_max3_num_f32 v224, v224, v1 /*v257*/, v4 /*v260*/
	v_mov_b32_e32 v238, v224
	v_permlanex16_b32 v238, v238, s43, 0xfedcba98
	s_set_vgpr_msb 0x1400
	v_max_num_f32_e32 v224, v224, v238
	v_dual_sub_f32 v238, v224, v237 :: v_dual_max_num_f32 v224, v237, v224
	v_cmp_lt_f32_e32 vcc_lo, 0x41000000, v238
	s_cmp_eq_u32 vcc_lo, 0
	s_cselect_b32 s2, -1, 0
	v_cndmask_b32_e64 v240, v224, v237, s2
	v_mul_f32_e32 v224, 0xbfb8aa3b, v240
	v_pk_fma_f32 v[242:243], v[242:243], s[42:43], v[224:225] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[244:245], v[244:245], s[42:43], v[224:225] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[246:247], v[246:247], s[42:43], v[224:225] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[248:249], v[248:249], s[42:43], v[224:225] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 64
	v_pk_fma_f32 v[4:5] /*v[260:261]*/, v[250:251], s[42:43], v[224:225] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[8:9] /*v[264:265]*/, v[252:253], s[42:43], v[224:225] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[24:25] /*v[280:281]*/, v[254:255], s[42:43], v[224:225] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x4000
	v_exp_f32_e32 v242, v242
	v_exp_f32_e32 v243, v243
	v_exp_f32_e32 v245, v245
	v_exp_f32_e32 v246, v246
	v_exp_f32_e32 v248, v248
	v_exp_f32_e32 v249, v249
	s_set_vgpr_msb 1
	v_exp_f32_e32 v251, v4 /*v260*/
	v_exp_f32_e32 v253, v5 /*v261*/
	v_exp_f32_e32 v255, v8 /*v264*/
	s_set_vgpr_msb 0x141
	v_pk_fma_f32 v[26:27] /*v[282:283]*/, v[2:3] /*v[258:259]*/, s[42:43], v[224:225] op_sel_hi:[1,0,0]
	v_exp_f32_e32 v4 /*v260*/, v24 /*v280*/
	v_pk_fma_f32 v[28:29] /*v[284:285]*/, v[6:7] /*v[262:263]*/, s[42:43], v[224:225] op_sel_hi:[1,0,0]
	v_exp_f32_e32 v6 /*v262*/, v25 /*v281*/
	v_nop
	v_pk_fma_f32 v[24:25] /*v[280:281]*/, v[10:11] /*v[266:267]*/, s[42:43], v[224:225] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[12:13] /*v[268:269]*/, v[12:13] /*v[268:269]*/, s[42:43], v[224:225] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[14:15] /*v[270:271]*/, v[14:15] /*v[270:271]*/, s[42:43], v[224:225] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[16:17] /*v[272:273]*/, v[16:17] /*v[272:273]*/, s[42:43], v[224:225] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[18:19] /*v[274:275]*/, v[18:19] /*v[274:275]*/, s[42:43], v[224:225] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[20:21] /*v[276:277]*/, v[20:21] /*v[276:277]*/, s[42:43], v[224:225] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x4100
	v_exp_f32_e32 v244, v244
	v_exp_f32_e32 v247, v247
	s_set_vgpr_msb 0x41
	v_exp_f32_e32 v2 /*v258*/, v9 /*v265*/
	v_exp_f32_e32 v8 /*v264*/, v26 /*v282*/
	v_exp_f32_e32 v10 /*v266*/, v27 /*v283*/
	s_set_vgpr_msb 0x4101
	v_exp_f32_e32 v250, v28 /*v284*/
	v_exp_f32_e32 v254, v24 /*v280*/
	s_set_vgpr_msb 0x141
	v_exp_f32_e32 v1 /*v257*/, v25 /*v281*/
	v_exp_f32_e32 v5 /*v261*/, v13 /*v269*/
	v_exp_f32_e32 v7 /*v263*/, v14 /*v270*/
	v_exp_f32_e32 v9 /*v265*/, v15 /*v271*/
	v_exp_f32_e32 v11 /*v267*/, v16 /*v272*/
	v_pk_fma_f32 v[22:23] /*v[278:279]*/, v[22:23] /*v[278:279]*/, s[42:43], v[224:225] op_sel_hi:[1,0,0]
	v_exp_f32_e32 v14 /*v270*/, v19 /*v275*/
	v_exp_f32_e32 v15 /*v271*/, v20 /*v276*/
	v_exp_f32_e32 v16 /*v272*/, v21 /*v277*/
	s_set_vgpr_msb 0x4100
	v_dual_add_f32 v224, v242, v243 :: v_dual_add_f32 v238, v245, v246
	s_set_vgpr_msb 64
	v_dual_add_f32 v19 /*v275*/, v248, v249 :: v_dual_add_f32 v20 /*v276*/, v253, v255
	s_set_vgpr_msb 0x4045
	v_add_f32_e32 v21 /*v277*/, v4 /*v260*/, v6 /*v262*/
	s_set_vgpr_msb 0x4501
	v_exp_f32_e32 v252, v29 /*v285*/
	s_set_vgpr_msb 0x141
	v_exp_f32_e32 v3 /*v259*/, v12 /*v268*/
	v_exp_f32_e32 v12 /*v268*/, v17 /*v273*/
	v_exp_f32_e32 v13 /*v269*/, v18 /*v274*/
	v_exp_f32_e32 v17 /*v273*/, v22 /*v278*/
	v_exp_f32_e32 v18 /*v274*/, v23 /*v279*/
	s_set_vgpr_msb 0x4100
	v_dual_add_f32 v224, v244, v224 :: v_dual_add_f32 v238, v247, v238
	s_set_vgpr_msb 0x44
	v_dual_add_f32 v19 /*v275*/, v251, v19 /*v275*/ :: v_dual_add_f32 v23 /*v279*/, v254, v1 /*v257*/
	v_add_f32_e32 v22 /*v278*/, v250, v10 /*v266*/
	s_set_vgpr_msb 0x4445
	v_dual_add_f32 v20 /*v276*/, v2 /*v258*/, v20 /*v276*/ :: v_dual_add_f32 v21 /*v277*/, v8 /*v264*/, v21 /*v277*/
	v_dual_add_f32 v24 /*v280*/, v5 /*v261*/, v7 /*v263*/ :: v_dual_add_f32 v25 /*v281*/, v11 /*v267*/, v12 /*v268*/
	s_set_vgpr_msb 0x4544
	v_add_f32_e32 v22 /*v278*/, v252, v22 /*v278*/
	s_set_vgpr_msb 0x4445
	v_add_f32_e32 v23 /*v279*/, v3 /*v259*/, v23 /*v279*/
	s_set_vgpr_msb 0x4500
	v_add_f32_e32 v224, v224, v238
	s_set_vgpr_msb 5
	v_add_f32_e32 v238, v9 /*v265*/, v24 /*v280*/
	s_set_vgpr_msb 0x545
	v_add_f32_e32 v20 /*v276*/, v20 /*v276*/, v21 /*v277*/
	v_dual_add_f32 v21 /*v277*/, v13 /*v269*/, v25 /*v281*/ :: v_dual_add_f32 v24 /*v280*/, v14 /*v270*/, v15 /*v271*/
	s_set_vgpr_msb 0x4501
	v_add_f32_e32 v238, v23 /*v279*/, v238
	v_add_f32_e32 v224, v19 /*v275*/, v224
	s_set_vgpr_msb 0x145
	v_dual_add_f32 v19 /*v275*/, v22 /*v278*/, v20 /*v276*/ :: v_dual_add_f32 v22 /*v278*/, v17 /*v273*/, v18 /*v274*/
	v_add_f32_e32 v20 /*v276*/, v16 /*v272*/, v24 /*v280*/
	s_set_vgpr_msb 0x4501
	v_add_f32_e32 v238, v21 /*v277*/, v238
	v_add_f32_e32 v224, v19 /*v275*/, v224
	s_set_vgpr_msb 0x145
	v_add_f32_e32 v19 /*v275*/, v22 /*v278*/, v20 /*v276*/
	s_set_vgpr_msb 0x4500
	v_add_f32_e32 v224, v238, v224
	v_sub_f32_e32 v238, v237, v240
	s_set_vgpr_msb 1
	v_dual_add_f32 v237, v19 /*v275*/, v224 :: v_dual_mul_f32 v224, 0x3fb8aa3b, v238
	s_set_vgpr_msb 0x100
	v_mov_b32_e32 v238, v237
	v_exp_f32_e32 v224, v224
	v_permlanex16_b32 v238, v238, s43, 0xfedcba98
	s_cbranch_vccz .LBB0_29
	v_pk_mul_f32 v[62:63], v[62:63], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[60:61], v[60:61], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59], v[58:59], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[56:57], v[56:57], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[54:55], v[54:55], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[52:53], v[52:53], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[50:51], v[50:51], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[48:49], v[48:49], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[46:47], v[46:47], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[44:45], v[44:45], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[42:43], v[42:43], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[40:41], v[40:41], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39], v[38:39], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[36:37], v[36:37], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35], v[34:35], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[32:33], v[32:33], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[30:31], v[30:31], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[28:29], v[28:29], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[26:27], v[26:27], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25], v[24:25], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[22:23], v[22:23], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21], v[20:21], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[18:19], v[18:19], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[16:17], v[16:17], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[14:15], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13], v[12:13], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[10:11], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[8:9], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[6:7], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[4:5], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[2:3], v[2:3], v[224:225] op_sel_hi:[1,0]
	v_pk_mul_f32 v[0:1], v[0:1], v[224:225] op_sel_hi:[1,0]
	s_branch .LBB0_29
.LBB0_34:
	v_mov_b32_e32 v240, v237
.LBB0_35:
	v_div_scale_f32 v65, null, v232, v232, 1.0
	v_div_scale_f32 v68, vcc_lo, 1.0, v232, 1.0
	s_set_vgpr_msb 1
	v_readlane_b32 s3, v0 /*v256*/, 0
	s_lshl_b32 s2, s10, 17
	v_mul_u32_u24_e32 v64, 0x110, v229
	s_set_vgpr_msb 0x100
	v_rcp_f32_e32 v66, v65
	s_and_b32 s2, s2, 0x20000
	s_mov_b32 s10, 0
	s_add_co_i32 s2, s2, s3
	v_nop
	v_fma_f32 v67, -v65, v66, 1.0
	v_fmac_f32_e32 v66, v67, v66
	v_mul_f32_e32 v67, v68, v66
	v_fma_f32 v69, -v65, v67, v68
	v_fmac_f32_e32 v67, v69, v66
	v_fma_f32 v65, -v65, v67, v68
	v_div_fmas_f32 v65, v65, v66, v67
	v_cmp_lt_f32_e32 vcc_lo, 0, v232
	v_div_fixup_f32 v65, v65, v232, 1.0
	v_dual_cndmask_b32 v66, 0, v65 :: v_dual_add_nc_u32 v65, s2, v231
	v_pk_mul_f32 v[56:57], v[56:57], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59], v[58:59], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[60:61], v[60:61], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[62:63], v[62:63], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[48:49], v[48:49], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[50:51], v[50:51], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[52:53], v[52:53], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[54:55], v[54:55], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[40:41], v[40:41], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[42:43], v[42:43], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[44:45], v[44:45], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[46:47], v[46:47], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[32:33], v[32:33], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35], v[34:35], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[36:37], v[36:37], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39], v[38:39], v[66:67] op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25], v[66:67], v[24:25] op_sel_hi:[0,1]
	v_pk_mul_f32 v[26:27], v[66:67], v[26:27] op_sel_hi:[0,1]
	v_pk_mul_f32 v[28:29], v[66:67], v[28:29] op_sel_hi:[0,1]
	v_pk_mul_f32 v[30:31], v[66:67], v[30:31] op_sel_hi:[0,1]
	v_pk_mul_f32 v[68:69], v[66:67], v[16:17] op_sel_hi:[0,1]
	v_pk_mul_f32 v[70:71], v[66:67], v[18:19] op_sel_hi:[0,1]
	v_pk_mul_f32 v[20:21], v[66:67], v[20:21] op_sel_hi:[0,1]
	v_pk_mul_f32 v[22:23], v[66:67], v[22:23] op_sel_hi:[0,1]
	v_pk_mul_f32 v[72:73], v[66:67], v[8:9] op_sel_hi:[0,1]
	v_pk_mul_f32 v[74:75], v[66:67], v[10:11] op_sel_hi:[0,1]
	v_pk_mul_f32 v[76:77], v[66:67], v[12:13] op_sel_hi:[0,1]
	v_pk_mul_f32 v[78:79], v[66:67], v[14:15] op_sel_hi:[0,1]
	v_pk_mul_f32 v[80:81], v[66:67], v[0:1] op_sel_hi:[0,1]
	v_pk_mul_f32 v[82:83], v[66:67], v[2:3] op_sel_hi:[0,1]
	v_pk_mul_f32 v[84:85], v[66:67], v[4:5] op_sel_hi:[0,1]
	v_pk_mul_f32 v[66:67], v[66:67], v[6:7] op_sel_hi:[0,1]
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
	v_cvt_pk_bf16_f32 v21, v70, v71
	v_cvt_pk_bf16_f32 v20, v68, v69
	v_cvt_pk_bf16_f32 v27, v78, v79
	v_cvt_pk_bf16_f32 v26, v76, v77
	v_cvt_pk_bf16_f32 v25, v74, v75
	v_cvt_pk_bf16_f32 v24, v72, v73
	v_cvt_pk_bf16_f32 v31, v66, v67
	v_cvt_pk_bf16_f32 v30, v84, v85
	v_cvt_pk_bf16_f32 v29, v82, v83
	v_cvt_pk_bf16_f32 v28, v80, v81
	s_lshl_b32 s3, s77, 2
	s_wait_alu depctr_va_vdst(0)
	ds_store_b128 v65, v[0:3]
	ds_store_b128 v65, v[4:7] offset:32
	ds_store_b128 v65, v[8:11] offset:64
	ds_store_b128 v65, v[12:15] offset:96
	s_sub_co_i32 s3, s3, vcc_hi
	ds_store_b128 v65, v[16:19] offset:128
	ds_store_b128 v65, v[20:23] offset:160
	ds_store_b128 v65, v[24:27] offset:192
	ds_store_b128 v65, v[28:31] offset:224
	s_cmp_lt_i32 s3, 1
	s_wait_dscnt 0x0
	s_cbranch_scc1 .LBB0_37
	s_wait_alu depctr_vm_vsrc(5)
	v_dual_lshrrev_b32 v10, 4, v230 :: v_dual_add_nc_u32 v2, s2, v64
	v_or_b32_e32 v0, 12, v225
	s_add_co_i32 s3, s3, -1
	s_load_b64 s[12:13], s[0:1], 0x0 nv
	v_dual_lshlrev_b32 v3, 8, v229 :: v_dual_bitop2_b32 v1, 14, v10 bitop3:0x54
	s_min_u32 s3, s3, 15
	v_or_b32_e32 v5, 10, v10
	v_min_u32_e32 v4, s3, v1
	v_mov_b32_e32 v1, 0
	v_min_u32_e32 v8, s3, v0
	v_lshlrev_b32_e32 v0, 3, v229
	v_min_u32_e32 v11, s3, v5
	s_wait_alu depctr_vm_vsrc(3)
	v_dual_sub_nc_u32 v17, v2, v3 :: v_dual_bitop2_b32 v7, vcc_hi, v8 bitop3:0x54
	v_or_b32_e32 v6, 8, v225
	v_mad_u32_u24 v18, 0x110, v4, v17
	v_dual_ashrrev_i32 v5, 31, v7 :: v_dual_bitop2_b32 v9, vcc_hi, v4 bitop3:0x54
	v_min_u32_e32 v16, s3, v6
	v_dual_lshrrev_b32 v3, 30, v5 :: v_dual_bitop2_b32 v6, vcc_hi, v11 bitop3:0x54
	v_dual_ashrrev_i32 v2, 31, v9 :: v_dual_bitop2_b32 v12, vcc_hi, v16 bitop3:0x54
	v_ashrrev_i32_e32 v5, 31, v6
	v_mad_u32_u24 v16, 0x110, v16, v17
	v_dual_add_nc_u32 v3, v7, v3 :: v_dual_lshrrev_b32 v2, 30, v2
	v_dual_ashrrev_i32 v14, 31, v12 :: v_dual_bitop2_b32 v13, 6, v10 bitop3:0x54
	v_dual_lshrrev_b32 v4, 30, v5 :: v_dual_bitop2_b32 v5, -4, v3 bitop3:0x40
	v_dual_add_nc_u32 v2, v9, v2 :: v_dual_ashrrev_i32 v3, 2, v3
	v_dual_lshrrev_b32 v14, 30, v14 :: v_dual_add_nc_u32 v4, v6, v4
	v_cmp_ne_u32_e32 vcc_lo, v7, v5
	v_and_b32_e32 v15, -4, v2
	v_dual_ashrrev_i32 v2, 2, v2 :: v_dual_add_nc_u32 v19, s72, v3
	v_dual_sub_nc_u32 v5, v7, v5 :: v_dual_ashrrev_i32 v7, 2, v4
	v_sub_nc_u32_e32 v3, v9, v15
	v_cmp_ne_u32_e64 s2, v9, v15
	v_dual_add_nc_u32 v4, s72, v2 :: v_dual_bitop2_b32 v9, -4, v4 bitop3:0x40
	v_dual_add_nc_u32 v5, s34, v5 :: v_dual_add_nc_u32 v3, s34, v3
	s_and_b32 s2, s104, s2
	s_wait_alu depctr_vm_vsrc(2)
	v_sub_nc_u32_e32 v21, v6, v9
	v_cndmask_b32_e64 v15, 0, 1, s2
	s_and_b32 s2, s104, vcc_lo
	s_wait_kmcnt 0x0
	v_mad_nc_i64_i32 v[2:3], v3, s4, v[0:1]
	v_cndmask_b32_e64 v20, 0, 1, s2
	v_cmp_ne_u32_e32 vcc_lo, v6, v9
	v_sub_nc_u32_e32 v15, v4, v15
	v_mad_nc_i64_i32 v[4:5], v5, s4, v[0:1]
	v_dual_add_nc_u32 v9, v12, v14 :: v_dual_sub_nc_u32 v6, v19, v20
	v_add_nc_u32_e32 v14, s34, v21
	s_and_b32 s2, s104, vcc_lo
	v_mad_nc_i64_i32 v[2:3], v15, s8, v[2:3]
	v_add_nc_u32_e32 v15, s72, v7
	v_cndmask_b32_e64 v19, 0, 1, s2
	v_min_u32_e32 v20, s3, v13
	v_mad_nc_i64_i32 v[4:5], v6, s8, v[4:5]
	v_mad_nc_i64_i32 v[6:7], v14, s4, v[0:1]
	v_mad_u32_u24 v21, 0x110, v8, v17
	v_dual_sub_nc_u32 v8, v15, v19 :: v_dual_bitop2_b32 v13, -4, v9 bitop3:0x40
	v_or_b32_e32 v14, vcc_hi, v20
	v_mad_u32_u24 v19, 0x110, v11, v17
	v_mad_u32_u24 v20, 0x110, v20, v17
	v_lshl_add_u64 v[2:3], v[2:3], 1, s[12:13]
	v_mad_nc_i64_i32 v[6:7], v8, s8, v[6:7]
	v_ashrrev_i32_e32 v8, 2, v9
	v_dual_sub_nc_u32 v9, v12, v13 :: v_dual_ashrrev_i32 v11, 31, v14
	v_cmp_ne_u32_e32 vcc_lo, v12, v13
	v_dual_add_nc_u32 v12, s72, v8 :: v_dual_bitop2_b32 v15, 4, v225 bitop3:0x54
	v_dual_add_nc_u32 v8, s34, v9 :: v_dual_lshrrev_b32 v11, 30, v11
	s_and_b32 s2, s104, vcc_lo
	v_min_u32_e32 v22, s3, v15
	v_cndmask_b32_e64 v13, 0, 1, s2
	v_dual_add_nc_u32 v11, v14, v11 :: v_dual_bitop2_b32 v10, 2, v10 bitop3:0x54
	v_mad_nc_i64_i32 v[8:9], v8, s4, v[0:1]
	v_dual_sub_nc_u32 v12, v12, v13 :: v_dual_bitop2_b32 v15, vcc_hi, v22 bitop3:0x54
	s_wait_alu depctr_vm_vsrc(1)
	v_min_u32_e32 v24, s3, v10
	v_and_b32_e32 v13, -4, v11
	v_ashrrev_i32_e32 v11, 2, v11
	v_ashrrev_i32_e32 v23, 31, v15
	v_mad_u32_u24 v22, 0x110, v22, v17
	v_mad_nc_i64_i32 v[8:9], v12, s8, v[8:9]
	v_dual_add_nc_u32 v10, s72, v11 :: v_dual_bitop2_b32 v12, vcc_hi, v24 bitop3:0x54
	v_cmp_ne_u32_e32 vcc_lo, v14, v13
	v_dual_lshrrev_b32 v11, 30, v23 :: v_dual_sub_nc_u32 v13, v14, v13
	v_dual_ashrrev_i32 v26, 31, v12 :: v_dual_min_i32 v23, s3, v225
	s_and_b32 s2, s104, vcc_lo
	v_lshl_add_u64 v[8:9], v[8:9], 1, s[12:13]
	v_cndmask_b32_e64 v25, 0, 1, s2
	v_or_b32_e32 v14, vcc_hi, v23
	v_add_nc_u32_e32 v11, v15, v11
	v_mad_u32_u24 v23, 0x110, v23, v17
	v_mad_u32_u24 v17, 0x110, v24, v17
	v_sub_nc_u32_e32 v25, v10, v25
	v_dual_add_nc_u32 v10, s34, v13 :: v_dual_ashrrev_i32 v13, 31, v14
	v_dual_lshrrev_b32 v26, 30, v26 :: v_dual_bitop2_b32 v27, -4, v11 bitop3:0x40
	s_wait_alu depctr_vm_vsrc(0)
	v_ashrrev_i32_e32 v28, 2, v11
	v_mad_nc_i64_i32 v[10:11], v10, s4, v[0:1]
	v_dual_lshrrev_b32 v13, 30, v13 :: v_dual_add_nc_u32 v26, v12, v26
	v_cmp_ne_u32_e32 vcc_lo, v15, v27
	v_dual_sub_nc_u32 v15, v15, v27 :: v_dual_add_nc_u32 v28, s72, v28
	v_dual_add_nc_u32 v13, v14, v13 :: v_dual_bitop2_b32 v27, -4, v26 bitop3:0x40
	s_and_b32 s2, s104, vcc_lo
	v_ashrrev_i32_e32 v26, 2, v26
	v_cndmask_b32_e64 v29, 0, 1, s2
	v_dual_add_nc_u32 v15, s34, v15 :: v_dual_bitop2_b32 v30, -4, v13 bitop3:0x40
	v_cmp_ne_u32_e32 vcc_lo, v12, v27
	v_dual_sub_nc_u32 v12, v12, v27 :: v_dual_ashrrev_i32 v13, 2, v13
	v_cmp_ne_u32_e64 s2, v14, v30
	v_sub_nc_u32_e32 v27, v14, v30
	v_dual_add_nc_u32 v26, s72, v26 :: v_dual_add_nc_u32 v30, s34, v12
	v_add_nc_u32_e32 v31, s72, v13
	s_and_b32 s2, s104, s2
	v_add_nc_u32_e32 v12, s34, v27
	v_cndmask_b32_e64 v27, 0, 1, s2
	s_and_b32 s2, s104, vcc_lo
	v_mad_nc_i64_i32 v[14:15], v15, s4, v[0:1]
	v_cndmask_b32_e64 v32, 0, 1, s2
	v_mad_nc_i64_i32 v[10:11], v25, s8, v[10:11]
	v_sub_nc_u32_e32 v27, v31, v27
	v_mad_nc_i64_i32 v[12:13], v12, s4, v[0:1]
	v_mad_nc_i64_i32 v[0:1], v30, s4, v[0:1]
	v_dual_sub_nc_u32 v25, v26, v32 :: v_dual_sub_nc_u32 v26, v28, v29
	v_lshl_add_u64 v[4:5], v[4:5], 1, s[12:13]
	v_lshl_add_u64 v[6:7], v[6:7], 1, s[12:13]
	v_lshl_add_u64 v[10:11], v[10:11], 1, s[12:13]
	v_mad_nc_i64_i32 v[12:13], v27, s8, v[12:13]
	v_mad_nc_i64_i32 v[0:1], v25, s8, v[0:1]
	v_mad_nc_i64_i32 v[14:15], v26, s8, v[14:15]
	v_lshl_add_u64 v[12:13], v[12:13], 1, s[12:13]
	v_lshl_add_u64 v[0:1], v[0:1], 1, s[12:13]
	v_lshl_add_u64 v[14:15], v[14:15], 1, s[12:13]
	s_wait_alu depctr_va_vdst(2)
	global_store_async_from_lds_b128 v[12:13], v23, off
	s_wait_alu depctr_va_vdst(1)
	global_store_async_from_lds_b128 v[0:1], v17, off
	s_wait_alu depctr_va_vdst(0)
	global_store_async_from_lds_b128 v[14:15], v22, off
	global_store_async_from_lds_b128 v[10:11], v20, off
	global_store_async_from_lds_b128 v[8:9], v16, off
	global_store_async_from_lds_b128 v[6:7], v19, off
	global_store_async_from_lds_b128 v[4:5], v21, off
	global_store_async_from_lds_b128 v[2:3], v18, off
.LBB0_37:
	v_cmp_gt_f32_e32 vcc_lo, 0x800000, v232
	s_load_b64 s[8:9], s[0:1], 0x40 nv
	s_wait_xcnt 0x0
	v_cmp_gt_i32_e64 s0, s77, v227
	s_wait_kmcnt 0x0
	s_mul_i32 s1, s79, s7
	s_wait_alu depctr_vm_vsrc(0)
	v_mad_u32 v3, v227, s5, s1
	v_cndmask_b32_e64 v1, 0, 32, vcc_lo
	v_cndmask_b32_e64 v0, 0, 0x42000000, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, 0, v225
	s_add_co_i32 s2, s1, s7
	v_ldexp_f32 v1, v232, v1
	s_ashr_i32 s3, s2, 31
	s_and_b32 vcc_lo, vcc_lo, s0
	v_lshlrev_b32_e32 v2, 2, v228
	s_lshr_b64 s[4:5], s[2:3], 5
	v_log_f32_e32 v1, v1
	s_lshl_b32 s11, s2, 27
	s_and_b64 s[2:3], s[4:5], 0x1ffffffffffffff
	v_sub_nc_u32_e32 v2, v226, v2
	s_or_b64 s[0:1], s[8:9], s[10:11]
	v_sub_f32_e32 v0, v1, v0
	v_mul_f32_e32 v0, 0x3f317218, v0
	v_dual_add_nc_u32 v2, s34, v2 :: v_dual_add_f32 v0, v0, v240
	v_mul_lo_u32 v2, v2, s6
	v_add_lshl_u32 v1, v3, v2, 2
	v_cndmask_b32_e32 v1, 0x7fffffff, v1, vcc_lo
	s_wait_alu depctr_va_vdst(0)
	buffer_store_b32 v0, v1, s[0:3], null offen
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
		.amdhsa_next_free_vgpr 337
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

	.set .Lkn_fmha_fwd_prefill_a16w16_m32x8_bshd_0.num_vgpr, 300
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
    .max_flat_workgroup_size: 256
    .name:           kn_fmha_fwd_prefill_a16w16_m32x8_bshd_0
    .private_segment_fixed_size: 0
    .reqd_workgroup_size:
      - 256
      - 1
      - 1
    .sgpr_count:     107
    .sgpr_spill_count: 3
    .symbol:         kn_fmha_fwd_prefill_a16w16_m32x8_bshd_0.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     300
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa-unknown-gfx1250
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
