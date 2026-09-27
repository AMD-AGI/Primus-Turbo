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
	s_load_b96 s[16:18], s[0:1], 0xa0 nv
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
	s_set_vgpr_msb 0x80
	v_and_b32_e32 v77 /*v589*/, 15, v0
	s_cselect_b32 s5, s7, s6
	s_wait_kmcnt 0x0
	s_mul_i32 s6, s18, s17
	s_cselect_b32 s2, ttmp9, s2
	s_cselect_b32 s3, s3, s4
	s_abs_i32 s4, s6
	s_mul_i32 s5, s17, s5
	s_cvt_f32_u32 s7, s4
	s_add_co_i32 s3, s5, s3
	s_sub_co_i32 s5, 0, s4
	s_mul_i32 s3, s3, s16
	v_s_rcp_f32 s7, s7
	s_add_co_i32 s3, s3, s2
	v_and_b32_e32 v79 /*v591*/, 16, v0
	s_clause 0x1
	s_load_b128 s[28:31], s[0:1], 0x7c nv
	s_load_b96 s[56:58], s[0:1], 0x90 nv
	v_and_b32_e32 v78 /*v590*/, 31, v0
	v_bfe_u32 v76 /*v588*/, v0, 4, 1
	s_mov_b32 s12, 1
	s_mov_b32 s43, 0
	s_mul_f32 s7, s7, 0x4f7ffffe
	s_set_vgpr_msb 0x8088
	v_dual_lshlrev_b32 v84 /*v596*/, 1, v78 /*v590*/ :: v_dual_lshlrev_b32 v81 /*v593*/, 3, v76 /*v588*/
	s_mov_b32 s21, 0xffff0000
	s_cvt_u32_f32 s7, s7
	s_mul_i32 s5, s5, s7
	s_mul_hi_u32 s2, s7, s5
	s_abs_i32 s5, s3
	s_add_co_i32 s7, s7, s2
	s_mul_hi_u32 s2, s5, s7
	s_xor_b32 s7, s3, s6
	s_mul_i32 s8, s2, s4
	s_ashr_i32 s9, s7, 31
	s_sub_co_i32 s5, s5, s8
	s_add_co_i32 s8, s2, 1
	s_sub_co_i32 s10, s5, s4
	s_cmp_ge_u32 s5, s4
	s_cselect_b32 s2, s8, s2
	s_cselect_b32 s5, s10, s5
	s_add_co_i32 s8, s2, 1
	s_cmp_ge_u32 s5, s4
	s_cselect_b32 s2, s8, s2
	s_xor_b32 s2, s2, s9
	s_sub_co_i32 s4, s2, s9
	s_mul_i32 s4, s4, s6
	s_cmp_lg_u32 s3, s4
	s_cselect_b32 s4, -1, 0
	s_cmp_lt_i32 s7, 0
	s_cselect_b32 s5, -1, 0
	s_and_b32 s4, s4, s5
	s_sub_co_ci_u32 s2, s2, s9
	s_abs_i32 s14, s17
	s_mul_i32 s6, s2, s6
	s_cvt_f32_u32 s4, s14
	s_sub_co_i32 s5, 0, s14
	s_sub_co_i32 s13, s3, s6
	v_s_rcp_f32 s4, s4
	s_xor_b32 s19, s13, s17
	s_ashr_i32 s20, s19, 31
	s_mul_f32 s4, s4, 0x4f7ffffe
	s_cvt_u32_f32 s4, s4
	s_mul_i32 s5, s5, s4
	s_mul_hi_u32 s3, s4, s5
	s_abs_i32 s5, s13
	s_add_co_i32 s4, s4, s3
	s_mul_hi_u32 s3, s5, s4
	s_mul_i32 s4, s3, s14
	s_add_co_i32 s18, s3, 1
	s_sub_co_i32 s15, s5, s4
	s_load_b256 s[4:11], s[0:1], 0x5c nv
	s_sub_co_i32 s22, s15, s14
	s_cmp_ge_u32 s15, s14
	s_cselect_b32 s3, s18, s3
	s_cselect_b32 s15, s22, s15
	s_add_co_i32 s22, s3, 1
	s_cmp_ge_u32 s15, s14
	s_clause 0x2
	s_load_b64 s[14:15], s[0:1], 0x10 nv
	s_load_b64 s[66:67], s[0:1], 0x20 nv
	s_load_b64 s[68:69], s[0:1], 0x30 nv
	s_cselect_b32 s3, s22, s3
	s_mov_b32 s18, -1
	s_xor_b32 s3, s3, s20
	s_sub_co_i32 s22, s3, s20
	s_mul_i32 s17, s22, s17
	s_cmp_lg_u32 s13, s17
	s_cselect_b32 s23, -1, 0
	s_cmp_lt_i32 s19, 0
	s_wait_kmcnt 0x0
	s_mov_b32 s22, s5
	s_cselect_b32 s19, -1, 0
	s_mov_b32 s26, s9
	s_and_b32 s19, s23, s19
	s_sub_co_ci_u32 s3, s3, s20
	s_bfe_u32 s91, ttmp8, 0x50019
	s_mul_i32 s33, s3, s57
	s_and_b32 s19, s91, 30
	s_lshr_b32 s20, s91, 1
	s_cmp_lg_u32 s91, s19
	s_mul_i32 s89, s91, 0x2200
	s_cselect_b32 s19, -1, 0
	s_cmp_lt_i32 s91, 0
	s_mul_i32 s64, s3, s58
	s_cselect_b32 s23, -1, 0
	s_mov_b32 s36, s10
	s_and_b32 s19, s23, s19
	s_mov_b32 s62, s7
	s_and_b32 s19, s19, exec_lo
	s_cselect_b32 s19, 1, 0
	s_not_b32 s2, s2
	s_sub_co_i32 s40, s13, s17
	s_add_co_i32 s37, s16, s2
	s_lshl_b32 s2, s91, 5
	s_lshl_b32 s90, s37, 7
	s_lshl_b32 s34, s40, 2
	s_add_co_i32 s88, s90, s2
	s_mov_b32 s60, s6
	s_cmp_lt_i32 s88, 0
	s_mov_b32 s38, s11
	s_cselect_b32 s59, -1, 0
	s_add_co_i32 s13, s89, 0x20000
	v_or_b32_e32 v72 /*v584*/, s88, v77 /*v589*/
	s_ashr_i32 s2, s88, 31
	s_ashr_i32 s16, s88, 2
	s_lshr_b32 s2, s2, 30
	s_add_co_i32 s24, s33, s16
	s_set_vgpr_msb 0x8808
	v_ashrrev_i32_e32 v1, 31, v72 /*v584*/
	s_ashr_i32 s23, s5, 31
	s_ashr_i32 s35, s34, 31
	s_ashr_i32 s27, s9, 31
	s_ashr_i32 s25, s24, 31
	s_set_vgpr_msb 0x802
	v_lshrrev_b32_e32 v1, 30, v1
	s_sub_co_i32 s17, s57, s16
	s_mul_u64 s[44:45], s[34:35], s[26:27]
	s_mul_u64 s[24:25], s[24:25], s[22:23]
	s_max_i32 s16, s17, 0
	v_add_nc_u32_e32 v1, v72 /*v584*/, v1
	s_lshl_b64 s[44:45], s[44:45], 1
	s_lshl_b64 s[24:25], s[24:25], 1
	s_set_vgpr_msb 0x280
	v_cvt_pk_bf16_f32 v83 /*v595*/, s4, s4
	s_add_nc_u64 s[14:15], s[14:15], s[24:25]
	v_ashrrev_i32_e32 v73 /*v585*/, 2, v1
	s_set_vgpr_msb 0x80a8
	v_mad_u32_u24 v80 /*v592*/, 0x110, v77 /*v589*/, v79 /*v591*/
	s_add_nc_u64 s[14:15], s[14:15], s[44:45]
	s_bitset1_b32 s15, 31
	v_add_nc_u32_e32 v85 /*v597*/, s13, v80 /*v592*/
	s_set_vgpr_msb 0xa808
	v_or_b32_e32 v2, 16, v72 /*v584*/
	s_set_vgpr_msb 0x888
	v_or_b32_e32 v82 /*v594*/, 0x20000, v80 /*v592*/
	s_set_vgpr_msb 0x8800
	v_add_nc_u32_e32 v3, s2, v2
	v_and_b32_e32 v4, -4, v3
	v_and_b32_e32 v5, -4, v1
	v_cmp_ne_u32_e32 vcc_lo, v2, v4
	v_ashrrev_i32_e32 v2, 2, v3
	s_set_vgpr_msb 2
	v_cmp_ne_u32_e64 s2, v72 /*v584*/, v5
	s_and_b32 vcc_lo, s59, vcc_lo
	s_and_b32 s2, s59, s2
	s_cmp_lg_u32 s5, 0x80000000
	s_set_vgpr_msb 0x288
	v_subrev_co_ci_u32_e64 v74 /*v586*/, null, 0, v73 /*v585*/, s2
	s_cselect_b32 s23, s23, 0
	s_cselect_b32 s22, s5, 0x200
	s_cmp_lg_u32 s9, 0x80000000
	v_sub_co_ci_u32_e64 v75 /*v587*/, null, v2, 0, vcc_lo
	s_cselect_b32 s25, s9, 0x80
	s_cselect_b32 s5, s27, 0
	s_bfe_i32 s24, s37, 0x10018
	s_lshl_b32 s9, s22, 16
	s_lshr_b64 s[26:27], s[22:23], 16
	s_and_b32 s5, s5, 0xffff
	s_lshr_b32 s35, s24, 30
	s_and_b32 s27, s26, 0xffff0000
	s_or_b32 s26, s5, s9
	s_or_b32 s5, s35, s90
	s_lshr_b32 s17, s22, 16
	s_addk_co_i32 s5, 0x7f
	s_sub_co_i32 s22, s58, s57
	s_ashr_i32 s5, s5, 2
	s_add_co_i32 s42, s57, -1
	s_add_co_i32 s44, s56, s22
	s_add_co_i32 s5, s5, s24
	s_add_co_i32 s35, s44, 1
	s_min_i32 s5, s5, s42
	s_ashr_i32 s65, s64, 31
	s_ashr_i32 s41, s40, 31
	s_ashr_i32 s37, s10, 31
	s_ashr_i32 s63, s7, 31
	s_add_co_i32 s5, s35, s5
	s_ashr_i32 s61, s6, 31
	s_ashr_i32 s39, s11, 31
	s_mul_u64 s[10:11], s[40:41], s[36:37]
	s_mul_u64 s[22:23], s[64:65], s[62:63]
	s_min_i32 s5, s5, s58
	s_mul_u64 s[6:7], s[64:65], s[60:61]
	s_mul_u64 s[36:37], s[40:41], s[38:39]
	s_lshl_b64 s[70:71], s[10:11], 1
	s_lshl_b64 s[10:11], s[22:23], 1
	s_max_i32 s9, s5, 1
	s_lshl_b64 s[6:7], s[6:7], 1
	s_lshl_b64 s[72:73], s[36:37], 1
	s_add_nc_u64 s[10:11], s[68:69], s[10:11]
	s_add_co_i32 s5, s9, 0x7f
	s_add_nc_u64 s[6:7], s[66:67], s[6:7]
	s_or_b32 s27, s27, s17
	s_add_nc_u64 s[74:75], s[10:11], s[72:73]
	s_min_i32 s65, s9, 0x80
	s_lshr_b32 s10, s5, 7
	s_add_nc_u64 s[76:77], s[6:7], s[70:71]
	s_cmp_lg_u32 s20, s19
	s_mov_b32 s24, 0x80004
	s_mov_b32 s23, 0x807fff
	s_mov_b32 s22, 0xffff7fff
	s_mov_b32 s20, 0x7510000
	s_set_vgpr_msb 0x8800
	s_cbranch_scc0 .LBB0_10
	s_mov_b32 s17, s43
	s_mov_b32 s18, s43
	s_mov_b32 s19, s43
	s_mov_b32 s40, s43
	s_mov_b32 s41, s43
	s_mov_b32 s42, s43
	s_cmp_lg_u32 s60, 0x80000000
	tensor_load_to_lds s[12:15], s[20:27], s[16:19], s[40:43]
	s_mov_b64 s[6:7], s[14:15]
	s_cselect_b32 s7, s61, 0
	s_cselect_b32 s41, s60, 0x80
	s_and_b32 s2, s91, 3
	s_and_b32 s42, s7, 0xffff
	s_lshl_b32 s17, s2, 5
	s_lshl_b32 s78, s2, 6
	s_mul_i32 s92, s2, 0x2200
	s_mul_i32 s93, s2, 0x2400
	s_sub_co_i32 s2, s65, s17
	s_mov_b32 s79, s43
	s_max_i32 s2, s2, 0
	s_mov_b64 s[54:55], s[14:15]
	s_lshl_b32 s2, s2, 16
	s_mov_b32 s6, s41
	s_or_b32 s38, s2, 0x7fff
	s_cmp_lg_u32 s62, 0x80000000
	s_mul_u64 s[18:19], s[6:7], s[78:79]
	s_cselect_b32 s49, s62, 0x80
	s_cselect_b32 s55, s63, 0
	s_mov_b32 s54, s49
	s_add_nc_u64 s[6:7], s[18:19], s[76:77]
	s_mul_u64 s[78:79], s[54:55], s[78:79]
	s_mov_b64 s[4:5], s[12:13]
	s_mov_b32 s39, 0x800000
	s_mov_b32 s40, 32
	s_bitset1_b32 s93, 16
	s_add_nc_u64 s[80:81], s[78:79], s[74:75]
	s_mov_b32 s36, s20
	s_mov_b32 s37, s21
	s_mov_b64 s[52:53], s[12:13]
	s_mov_b32 s5, s92
	s_bitset1_b32 s7, 31
	s_mov_b32 s44, 0xf510000
	s_mov_b32 s45, s21
	s_mov_b32 s51, s43
	s_mov_b32 s47, s39
	s_mov_b32 s48, s40
	s_mov_b32 s46, s38
	s_mov_b32 s53, s93
	s_and_b32 s50, s55, 0xffff
	s_or_b32 s55, s81, 0x80000000
	s_mov_b32 s54, s80
	s_ashr_i32 s2, s90, 2
	s_add_co_i32 s11, s35, -1
	s_set_vgpr_msb 34
	v_and_or_b32 v64, v78 /*v590*/, 7, v81 /*v593*/
	s_add_co_i32 s2, s11, s2
	s_set_vgpr_msb 0x2288
	v_or_b32_e32 v93 /*v605*/, 0x20000, v80 /*v592*/
	s_max_i32 s2, s2, 0
	s_add_nc_u64 s[80:81], s[66:67], s[70:71]
	s_add_co_i32 s2, s2, 1
	s_set_vgpr_msb 0x8802
	v_mul_u32_u24_e32 v64, 0x120, v64
	v_and_or_b32 v64, v84 /*v596*/, 16, v64
	s_set_vgpr_msb 0x280
	v_or_b32_e32 v86 /*v598*/, 0x10000, v64
	v_or_b32_e32 v94 /*v606*/, 0x30000, v64
	s_wait_tensorcnt 0x0
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x8002
	ds_load_b128 v[0:3], v85 /*v597*/
	ds_load_b128 v[4:7], v85 /*v597*/ offset:32
	ds_load_b128 v[8:11], v85 /*v597*/ offset:64
	ds_load_b128 v[12:15], v85 /*v597*/ offset:96
	ds_load_b128 v[16:19], v85 /*v597*/ offset:128
	ds_load_b128 v[20:23], v85 /*v597*/ offset:160
	ds_load_b128 v[24:27], v85 /*v597*/ offset:192
	ds_load_b128 v[28:31], v85 /*v597*/ offset:224
	ds_load_b128 v[32:35], v85 /*v597*/ offset:4352
	ds_load_b128 v[36:39], v85 /*v597*/ offset:4384
	ds_load_b128 v[40:43], v85 /*v597*/ offset:4416
	ds_load_b128 v[44:47], v85 /*v597*/ offset:4448
	ds_load_b128 v[48:51], v85 /*v597*/ offset:4480
	ds_load_b128 v[52:55], v85 /*v597*/ offset:4512
	ds_load_b128 v[56:59], v85 /*v597*/ offset:4544
	ds_load_b128 v[60:63], v85 /*v597*/ offset:4576
	tensor_load_to_lds s[4:7], s[36:43]
	tensor_load_to_lds s[52:55], s[44:51]
	s_ashr_i32 s4, s2, 31
	s_add_co_i32 s5, s10, -1
	s_lshr_b32 s4, s4, 25
	s_wait_dscnt 0xe
	v_pk_mul_bf16 v135, v83 /*v595*/, v7
	s_add_co_i32 s4, s2, s4
	v_pk_mul_bf16 v134, v83 /*v595*/, v6
	s_and_b32 s6, s4, 0xffffff80
	s_ashr_i32 s4, s4, 7
	s_cmp_lg_u32 s2, s6
	v_pk_mul_bf16 v133, v83 /*v595*/, v5
	s_cselect_b32 s6, -1, 0
	s_cmp_lt_i32 s2, 0
	v_pk_mul_bf16 v132, v83 /*v595*/, v4
	s_cselect_b32 s2, -1, 0
	v_pk_mul_bf16 v131, v83 /*v595*/, v3
	s_and_b32 s2, s2, s6
	s_sub_co_ci_u32 s2, s4, 0
	v_pk_mul_bf16 v130, v83 /*v595*/, v2
	s_min_i32 s2, s2, s5
	v_pk_mul_bf16 v129, v83 /*v595*/, v1
	v_pk_mul_bf16 v128, v83 /*v595*/, v0
	s_wait_dscnt 0xc
	v_pk_mul_bf16 v143, v83 /*v595*/, v15
	v_pk_mul_bf16 v142, v83 /*v595*/, v14
	v_pk_mul_bf16 v141, v83 /*v595*/, v13
	v_pk_mul_bf16 v140, v83 /*v595*/, v12
	v_pk_mul_bf16 v139, v83 /*v595*/, v11
	v_pk_mul_bf16 v138, v83 /*v595*/, v10
	v_pk_mul_bf16 v137, v83 /*v595*/, v9
	v_pk_mul_bf16 v136, v83 /*v595*/, v8
	s_wait_dscnt 0xa
	v_pk_mul_bf16 v151, v83 /*v595*/, v23
	v_pk_mul_bf16 v150, v83 /*v595*/, v22
	v_pk_mul_bf16 v149, v83 /*v595*/, v21
	v_pk_mul_bf16 v148, v83 /*v595*/, v20
	v_pk_mul_bf16 v147, v83 /*v595*/, v19
	v_pk_mul_bf16 v146, v83 /*v595*/, v18
	v_pk_mul_bf16 v145, v83 /*v595*/, v17
	v_pk_mul_bf16 v144, v83 /*v595*/, v16
	s_wait_dscnt 0x8
	v_pk_mul_bf16 v159, v83 /*v595*/, v31
	v_pk_mul_bf16 v158, v83 /*v595*/, v30
	v_pk_mul_bf16 v157, v83 /*v595*/, v29
	v_pk_mul_bf16 v156, v83 /*v595*/, v28
	v_pk_mul_bf16 v155, v83 /*v595*/, v27
	v_pk_mul_bf16 v154, v83 /*v595*/, v26
	v_pk_mul_bf16 v153, v83 /*v595*/, v25
	v_pk_mul_bf16 v152, v83 /*v595*/, v24
	s_wait_dscnt 0x6
	v_pk_mul_bf16 v167, v83 /*v595*/, v39
	v_pk_mul_bf16 v166, v83 /*v595*/, v38
	v_pk_mul_bf16 v165, v83 /*v595*/, v37
	v_pk_mul_bf16 v164, v83 /*v595*/, v36
	v_pk_mul_bf16 v163, v83 /*v595*/, v35
	v_pk_mul_bf16 v162, v83 /*v595*/, v34
	v_pk_mul_bf16 v161, v83 /*v595*/, v33
	v_pk_mul_bf16 v160, v83 /*v595*/, v32
	s_wait_dscnt 0x4
	v_pk_mul_bf16 v175, v83 /*v595*/, v47
	v_pk_mul_bf16 v174, v83 /*v595*/, v46
	v_pk_mul_bf16 v173, v83 /*v595*/, v45
	v_pk_mul_bf16 v172, v83 /*v595*/, v44
	v_pk_mul_bf16 v171, v83 /*v595*/, v43
	v_pk_mul_bf16 v170, v83 /*v595*/, v42
	v_pk_mul_bf16 v169, v83 /*v595*/, v41
	v_pk_mul_bf16 v168, v83 /*v595*/, v40
	s_wait_dscnt 0x2
	v_pk_mul_bf16 v183, v83 /*v595*/, v55
	v_pk_mul_bf16 v182, v83 /*v595*/, v54
	v_pk_mul_bf16 v181, v83 /*v595*/, v53
	v_pk_mul_bf16 v180, v83 /*v595*/, v52
	v_pk_mul_bf16 v179, v83 /*v595*/, v51
	v_pk_mul_bf16 v178, v83 /*v595*/, v50
	v_pk_mul_bf16 v177, v83 /*v595*/, v49
	v_pk_mul_bf16 v176, v83 /*v595*/, v48
	s_wait_dscnt 0x0
	v_pk_mul_bf16 v191, v83 /*v595*/, v63
	v_pk_mul_bf16 v190, v83 /*v595*/, v62
	v_pk_mul_bf16 v189, v83 /*v595*/, v61
	v_pk_mul_bf16 v188, v83 /*v595*/, v60
	v_pk_mul_bf16 v187, v83 /*v595*/, v59
	v_pk_mul_bf16 v186, v83 /*v595*/, v58
	v_pk_mul_bf16 v185, v83 /*v595*/, v57
	v_pk_mul_bf16 v184, v83 /*v595*/, v56
	v_mov_b32_e32 v0, 0
	s_max_i32 s52, s2, 0
	s_mov_b32 s53, s43
	s_add_nc_u64 s[54:55], s[68:69], s[72:73]
	s_cmp_lt_i32 s2, 1
	s_wait_tensorcnt 0x0
	s_set_vgpr_msb 0x200
	s_barrier_signal -1
	s_barrier_wait -1
	s_cbranch_scc1 .LBB0_11
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
	s_set_vgpr_msb 0x80
	v_dual_mov_b32 v71 /*v583*/, 0xf149f2ca :: v_dual_mov_b32 v66 /*v578*/, 0xf149f2ca
	s_set_vgpr_msb 0x8002
	v_dual_mov_b32 v200, v94 /*v606*/ :: v_dual_mov_b32 v201, v93 /*v605*/
	s_set_vgpr_msb 0x282
	v_mov_b32_e32 v87 /*v599*/, v80 /*v592*/
	s_set_vgpr_msb 0x8240
	v_dual_mov_b32 v192 /*v448*/, v0 :: v_dual_mov_b32 v193 /*v449*/, v0
	s_lshl_b64 s[82:83], s[52:53], 7
	s_mov_b32 s4, 1
	s_add_co_i32 s94, s9, 0xffffff80
	s_add_co_i32 s84, s64, 0x80
	s_mov_b32 s37, 0xffff0000
	s_mov_b32 s36, 0x7510000
	s_mov_b64 s[86:87], 0xffffffffffffff80
	s_mov_b32 s95, 0x76543210
	s_mov_b32 s56, 0x3fb8aa3b
	s_mov_b32 s96, 1
	s_set_vgpr_msb 0x4000
	s_branch .LBB0_4
.LBB0_3:
	s_set_vgpr_msb 0x8a
	v_cvt_pk_bf16_f32 v103 /*v615*/, v20 /*v532*/, v34 /*v546*/
	v_cvt_pk_bf16_f32 v102 /*v614*/, v10 /*v522*/, v18 /*v530*/
	s_set_vgpr_msb 0x8a89
	v_cvt_pk_bf16_f32 v101 /*v613*/, v248 /*v504*/, v2 /*v514*/
	s_set_vgpr_msb 0x8985
	v_cvt_pk_bf16_f32 v100 /*v612*/, v230 /*v486*/, v242 /*v498*/
	v_cvt_pk_bf16_f32 v99 /*v611*/, v214 /*v470*/, v226 /*v482*/
	v_cvt_pk_bf16_f32 v98 /*v610*/, v206 /*v462*/, v212 /*v468*/
	v_cvt_pk_bf16_f32 v97 /*v609*/, v198 /*v454*/, v202 /*v458*/
	v_cvt_pk_bf16_f32 v96 /*v608*/, v194 /*v450*/, v196 /*v452*/
	s_set_vgpr_msb 0x858a
	v_cvt_pk_bf16_f32 v111 /*v623*/, v21 /*v533*/, v35 /*v547*/
	v_cvt_pk_bf16_f32 v110 /*v622*/, v11 /*v523*/, v19 /*v531*/
	s_set_vgpr_msb 0x8a89
	v_cvt_pk_bf16_f32 v109 /*v621*/, v249 /*v505*/, v3 /*v515*/
	s_set_vgpr_msb 0x8985
	v_cvt_pk_bf16_f32 v108 /*v620*/, v231 /*v487*/, v243 /*v499*/
	v_cvt_pk_bf16_f32 v107 /*v619*/, v215 /*v471*/, v227 /*v483*/
	v_cvt_pk_bf16_f32 v106 /*v618*/, v207 /*v463*/, v213 /*v469*/
	v_cvt_pk_bf16_f32 v105 /*v617*/, v199 /*v455*/, v203 /*v459*/
	v_cvt_pk_bf16_f32 v104 /*v616*/, v195 /*v451*/, v197 /*v453*/
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
	v_cvt_pk_bf16_f32 v115 /*v627*/, v233 /*v489*/, v245 /*v501*/
	v_cvt_pk_bf16_f32 v114 /*v626*/, v221 /*v477*/, v229 /*v485*/
	v_cvt_pk_bf16_f32 v113 /*v625*/, v209 /*v465*/, v217 /*v473*/
	s_set_vgpr_msb 0x8509
	v_wmma_f32_16x16x32_bf16 v[56:63], v[184:191] /*v[440:447]*/, v[104:111] /*v[616:623]*/, v[56:63]
	s_set_vgpr_msb 0x985
	v_cvt_pk_bf16_f32 v112 /*v624*/, v201 /*v457*/, v205 /*v461*/
	s_set_vgpr_msb 0x854a
	v_cvt_pk_bf16_f32 v201 /*v457*/, v53 /*v565*/, v59 /*v571*/
	v_cvt_pk_bf16_f32 v199 /*v455*/, v31 /*v543*/, v41 /*v553*/
	v_cvt_pk_bf16_f32 v198 /*v454*/, v15 /*v527*/, v25 /*v537*/
	v_cvt_pk_bf16_f32 v191 /*v447*/, v38 /*v550*/, v48 /*v560*/
	v_cvt_pk_bf16_f32 v190 /*v446*/, v28 /*v540*/, v36 /*v548*/
	v_cvt_pk_bf16_f32 v189 /*v445*/, v12 /*v524*/, v22 /*v534*/
	s_set_vgpr_msb 0x4a09
	s_wait_dscnt 0x3c
	v_wmma_f32_16x16x32_bf16 v[112:119], v[152:159] /*v[408:415]*/, v[96:103] /*v[608:615]*/, v[112:119]
	s_set_vgpr_msb 0x949
	v_cvt_pk_bf16_f32 v188 /*v444*/, v250 /*v506*/, v4 /*v516*/
	s_set_vgpr_msb 0x4945
	v_cvt_pk_bf16_f32 v187 /*v443*/, v232 /*v488*/, v244 /*v500*/
	v_cvt_pk_bf16_f32 v186 /*v442*/, v220 /*v476*/, v228 /*v484*/
	v_cvt_pk_bf16_f32 v185 /*v441*/, v208 /*v464*/, v216 /*v472*/
	v_cvt_pk_bf16_f32 v184 /*v440*/, v200 /*v456*/, v204 /*v460*/
	s_set_vgpr_msb 0x454a
	v_cvt_pk_bf16_f32 v200 /*v456*/, v45 /*v557*/, v51 /*v563*/
	s_set_vgpr_msb 0x4a49
	v_cvt_pk_bf16_f32 v197 /*v453*/, v253 /*v509*/, v7 /*v519*/
	s_set_vgpr_msb 0x4909
	v_wmma_f32_16x16x32_bf16 v[48:55], v[152:159] /*v[408:415]*/, v[104:111] /*v[616:623]*/, v[48:55]
	s_set_vgpr_msb 0x945
	v_cvt_pk_bf16_f32 v196 /*v452*/, v239 /*v495*/, v247 /*v503*/
	v_cvt_pk_bf16_f32 v195 /*v451*/, v223 /*v479*/, v235 /*v491*/
	v_cvt_pk_bf16_f32 v194 /*v450*/, v211 /*v467*/, v219 /*v475*/
	s_set_vgpr_msb 0x454a
	v_cvt_pk_bf16_f32 v209 /*v465*/, v63 /*v575*/, v65 /*v577*/
	v_cvt_pk_bf16_f32 v208 /*v464*/, v57 /*v569*/, v61 /*v573*/
	v_cvt_pk_bf16_f32 v207 /*v463*/, v47 /*v559*/, v55 /*v567*/
	v_cvt_pk_bf16_f32 v206 /*v462*/, v33 /*v545*/, v43 /*v555*/
	s_set_vgpr_msb 0x4a09
	s_wait_dscnt 0x2d
	v_wmma_f32_16x16x32_bf16 v[104:111], v[120:127] /*v[376:383]*/, v[96:103] /*v[608:615]*/, v[104:111]
	s_set_vgpr_msb 0x94a
	v_cvt_pk_bf16_f32 v205 /*v461*/, v17 /*v529*/, v27 /*v539*/
	v_cvt_pk_bf16_f32 v204 /*v460*/, v1 /*v513*/, v9 /*v521*/
	s_set_vgpr_msb 0x4a45
	v_cvt_pk_bf16_f32 v203 /*v459*/, v241 /*v497*/, v255 /*v511*/
	v_cvt_pk_bf16_f32 v202 /*v458*/, v225 /*v481*/, v237 /*v493*/
	s_add_nc_u64 s[82:83], s[82:83], s[86:87]
	s_addk_co_i32 s94, 0xff80
	s_addk_co_i32 s84, 0x80
	s_set_vgpr_msb 0x4509
	v_wmma_f32_16x16x32_bf16 v[40:47], v[120:127] /*v[376:383]*/, v[104:111] /*v[616:623]*/, v[40:47]
	s_add_co_i32 s96, s96, 1
	s_cmp_lg_u64 s[82:83], 0
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[96:103], v[88:95] /*v[344:351]*/, v[96:103] /*v[608:615]*/, v[96:103]
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
	v_cvt_pk_bf16_f32 v178 /*v434*/, v238 /*v494*/, v246 /*v502*/
	v_cvt_pk_bf16_f32 v177 /*v433*/, v222 /*v478*/, v234 /*v490*/
	v_cvt_pk_bf16_f32 v176 /*v432*/, v210 /*v466*/, v218 /*v474*/
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
	v_wmma_f32_16x16x32_bf16 v[56:63], v[168:175] /*v[424:431]*/, v[194:201] /*v[450:457]*/, v[56:63]
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
	v_cvt_pk_bf16_f32 v169 /*v425*/, v240 /*v496*/, v254 /*v510*/
	v_cvt_pk_bf16_f32 v168 /*v424*/, v224 /*v480*/, v236 /*v492*/
	s_set_vgpr_msb 0x4505
	v_wmma_f32_16x16x32_bf16 v[48:55], v[136:143] /*v[392:399]*/, v[194:201] /*v[450:457]*/, v[48:55]
	v_wmma_f32_16x16x32_bf16 v[104:111], v[104:111] /*v[360:367]*/, v[176:183] /*v[432:439]*/, v[104:111]
	v_wmma_f32_16x16x32_bf16 v[40:47], v[104:111] /*v[360:367]*/, v[194:201] /*v[450:457]*/, v[40:47]
	v_wmma_f32_16x16x32_bf16 v[96:103], v[72:79] /*v[328:335]*/, v[176:183] /*v[432:439]*/, v[96:103]
	v_wmma_f32_16x16x32_bf16 v[32:39], v[72:79] /*v[328:335]*/, v[194:201] /*v[450:457]*/, v[32:39]
	v_wmma_f32_16x16x32_bf16 v[88:95], v[40:47] /*v[296:303]*/, v[176:183] /*v[432:439]*/, v[88:95]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[40:47] /*v[296:303]*/, v[194:201] /*v[450:457]*/, v[24:31]
	v_wmma_f32_16x16x32_bf16 v[80:87], v[8:15] /*v[264:271]*/, v[176:183] /*v[432:439]*/, v[80:87]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[8:15] /*v[264:271]*/, v[194:201] /*v[450:457]*/, v[16:23]
	s_set_vgpr_msb 0x504
	v_wmma_f32_16x16x32_bf16 v[72:79], v[232:239], v[176:183] /*v[432:439]*/, v[72:79]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[232:239], v[194:201] /*v[450:457]*/, v[8:15]
	v_wmma_f32_16x16x32_bf16 v[64:71], v[200:207], v[176:183] /*v[432:439]*/, v[64:71]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[200:207], v[194:201] /*v[450:457]*/, v[0:7]
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x426
	v_pk_fma_f32 v[200:201], v[70:71] /*v[582:583]*/, v[192:193] /*v[448:449]*/, v[68:69] /*v[580:581]*/
	s_set_vgpr_msb 0x2682
	v_mov_b32_e32 v71 /*v583*/, v88 /*v600*/
	s_set_vgpr_msb 0x8248
	v_pk_add_f32 v[192:193] /*v[448:449]*/, v[200:201], v[66:67] /*v[578:579]*/
	s_set_vgpr_msb 0x4805
	v_wmma_f32_16x16x32_bf16 v[120:127], v[160:167] /*v[416:423]*/, v[168:175] /*v[424:431]*/, v[120:127]
	s_set_vgpr_msb 0x502
	v_dual_mov_b32 v200, v94 /*v606*/ :: v_dual_mov_b32 v201, v93 /*v605*/
	s_set_vgpr_msb 0x282
	v_mov_b32_e32 v66 /*v578*/, v89 /*v601*/
	s_set_vgpr_msb 0x8205
	v_wmma_f32_16x16x32_bf16 v[56:63], v[160:167] /*v[416:423]*/, v[202:209] /*v[458:465]*/, v[56:63]
	v_wmma_f32_16x16x32_bf16 v[112:119], v[128:135] /*v[384:391]*/, v[168:175] /*v[424:431]*/, v[112:119]
	v_wmma_f32_16x16x32_bf16 v[48:55], v[128:135] /*v[384:391]*/, v[202:209] /*v[458:465]*/, v[48:55]
	v_wmma_f32_16x16x32_bf16 v[104:111], v[96:103] /*v[352:359]*/, v[168:175] /*v[424:431]*/, v[104:111]
	v_wmma_f32_16x16x32_bf16 v[40:47], v[96:103] /*v[352:359]*/, v[202:209] /*v[458:465]*/, v[40:47]
	v_wmma_f32_16x16x32_bf16 v[96:103], v[64:71] /*v[320:327]*/, v[168:175] /*v[424:431]*/, v[96:103]
	v_wmma_f32_16x16x32_bf16 v[32:39], v[64:71] /*v[320:327]*/, v[202:209] /*v[458:465]*/, v[32:39]
	v_wmma_f32_16x16x32_bf16 v[88:95], v[32:39] /*v[288:295]*/, v[168:175] /*v[424:431]*/, v[88:95]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[32:39] /*v[288:295]*/, v[202:209] /*v[458:465]*/, v[24:31]
	v_wmma_f32_16x16x32_bf16 v[80:87], v[0:7] /*v[256:263]*/, v[168:175] /*v[424:431]*/, v[80:87]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[0:7] /*v[256:263]*/, v[202:209] /*v[458:465]*/, v[16:23]
	s_set_vgpr_msb 0x504
	v_wmma_f32_16x16x32_bf16 v[72:79], v[224:231], v[168:175] /*v[424:431]*/, v[72:79]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[224:231], v[202:209] /*v[458:465]*/, v[8:15]
	v_wmma_f32_16x16x32_bf16 v[64:71], v[192:199], v[168:175] /*v[424:431]*/, v[64:71]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[192:199], v[202:209] /*v[458:465]*/, v[0:7]
	s_set_vgpr_msb 0x400
	s_cbranch_scc0 .LBB0_12
.LBB0_4:
	s_wait_alu depctr_vm_vsrc(0)
	s_set_vgpr_msb 0x82
	v_dual_mov_b32 v93 /*v605*/, v87 /*v599*/ :: v_dual_mov_b32 v94 /*v606*/, v86 /*v598*/
	s_set_vgpr_msb 0x8280
	v_dual_mov_b32 v87 /*v599*/, v201 :: v_dual_mov_b32 v86 /*v598*/, v200
	s_wait_tensorcnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	s_cmp_ge_i32 s96, s10
	s_set_vgpr_msb 0x8000
	s_cbranch_scc1 .LBB0_6
	v_nop
	v_nop
	v_med3_i32 v192, s94, 0, 0x80
	s_ashr_i32 s85, s84, 31
	s_lshr_b32 s2, s96, 31
	s_mul_u64 s[6:7], s[84:85], s[62:63]
	s_add_co_i32 s2, s96, s2
	v_readfirstlane_b32 s5, v192
	s_lshl_b64 s[6:7], s[6:7], 1
	s_and_b32 s2, s2, 0x7ffe
	s_add_nc_u64 s[46:47], s[54:55], s[6:7]
	s_mul_u64 s[6:7], s[84:85], s[60:61]
	s_sub_co_i32 s5, s5, s17
	s_lshl_b64 s[6:7], s[6:7], 1
	s_sub_co_i32 s2, s96, s2
	s_add_nc_u64 s[6:7], s[80:81], s[6:7]
	s_max_i32 s38, s5, 0
	s_lshl_b32 s2, s2, 17
	s_add_nc_u64 s[6:7], s[18:19], s[6:7]
	s_lshl_b32 s38, s38, 16
	s_or_b32 s5, s92, s2
	s_bitset1_b32 s7, 31
	s_addk_co_i32 s38, 0x7fff
	s_mov_b32 s45, s37
	tensor_load_to_lds s[4:7], s[36:43]
	s_add_nc_u64 s[6:7], s[78:79], s[46:47]
	s_or_b32 s5, s93, s2
	s_bitset1_b32 s7, 31
	s_mov_b32 s46, s38
	s_mov_b32 s47, s39
	s_mov_b32 s48, s40
	s_mov_b32 s51, s43
	tensor_load_to_lds s[4:7], s[44:51]
.LBB0_6:
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
	v_wmma_f32_16x16x32_bf16 v[194:201] /*v[450:457]*/, v[192:199], v[128:135], 0
	s_wait_dscnt 0x36
	v_wmma_f32_16x16x32_bf16 v[214:221] /*v[470:477]*/, v[224:231], v[128:135], 0
	s_set_vgpr_msb 0x4041
	s_wait_dscnt 0x2e
	v_wmma_f32_16x16x32_bf16 v[232:239] /*v[488:495]*/, v[0:7] /*v[256:263]*/, v[128:135], 0
	s_wait_dscnt 0x26
	v_wmma_f32_16x16x32_bf16 v[250:257] /*v[506:513]*/, v[32:39] /*v[288:295]*/, v[128:135], 0
	s_set_vgpr_msb 0x4181
	s_wait_dscnt 0x1e
	v_wmma_f32_16x16x32_bf16 v[38:45] /*v[550:557]*/, v[64:71] /*v[320:327]*/, v[128:135], 0
	s_wait_dscnt 0x16
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[562:569]*/, v[96:103] /*v[352:359]*/, v[128:135], 0
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[570:577]*/, v[128:135] /*v[384:391]*/, v[128:135], 0
	s_set_vgpr_msb 0x8180
	v_wmma_f32_16x16x32_bf16 v[96:103] /*v[608:615]*/, v[192:199], v[160:167], 0
	s_set_vgpr_msb 0x8050
	v_wmma_f32_16x16x32_bf16 v[194:201] /*v[450:457]*/, v[200:207], v[136:143], v[194:201] /*v[450:457]*/
	s_set_vgpr_msb 0x5080
	v_wmma_f32_16x16x32_bf16 v[104:111] /*v[616:623]*/, v[224:231], v[160:167], 0
	s_set_vgpr_msb 0x8050
	v_wmma_f32_16x16x32_bf16 v[214:221] /*v[470:477]*/, v[232:239], v[136:143], v[214:221] /*v[470:477]*/
	s_set_vgpr_msb 0x5081
	v_wmma_f32_16x16x32_bf16 v[112:119] /*v[624:631]*/, v[0:7] /*v[256:263]*/, v[160:167], 0
	s_set_vgpr_msb 0x8151
	v_wmma_f32_16x16x32_bf16 v[232:239] /*v[488:495]*/, v[8:15] /*v[264:271]*/, v[136:143], v[232:239] /*v[488:495]*/
	s_set_vgpr_msb 0x5181
	v_wmma_f32_16x16x32_bf16 v[120:127] /*v[632:639]*/, v[32:39] /*v[288:295]*/, v[160:167], 0
	s_set_vgpr_msb 0x8151
	v_wmma_f32_16x16x32_bf16 v[250:257] /*v[506:513]*/, v[40:47] /*v[296:303]*/, v[136:143], v[250:257] /*v[506:513]*/
	s_set_vgpr_msb 0x51a1
	v_wmma_f32_16x16x32_bf16 v[128:135] /*v[640:647]*/, v[64:71] /*v[320:327]*/, v[160:167], 0
	v_wmma_f32_16x16x32_bf16 v[38:45] /*v[550:557]*/, v[72:79] /*v[328:335]*/, v[136:143], v[38:45] /*v[550:557]*/
	v_wmma_f32_16x16x32_bf16 v[136:143] /*v[648:655]*/, v[96:103] /*v[352:359]*/, v[160:167], 0
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[562:569]*/, v[104:111] /*v[360:367]*/, v[136:143], v[50:57] /*v[562:569]*/
	v_wmma_f32_16x16x32_bf16 v[144:151] /*v[656:663]*/, v[128:135] /*v[384:391]*/, v[160:167], 0
	s_wait_dscnt 0xc
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[570:577]*/, v[136:143] /*v[392:399]*/, v[136:143], v[58:65] /*v[570:577]*/
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x32_bf16 v[152:159] /*v[664:671]*/, v[160:167] /*v[416:423]*/, v[128:135], 0
	v_wmma_f32_16x16x32_bf16 v[160:167] /*v[672:679]*/, v[160:167] /*v[416:423]*/, v[160:167], 0
	s_set_vgpr_msb 0xa1a0
	v_wmma_f32_16x16x32_bf16 v[96:103] /*v[608:615]*/, v[200:207], v[168:175], v[96:103] /*v[608:615]*/
	s_set_vgpr_msb 0xa050
	v_wmma_f32_16x16x32_bf16 v[194:201] /*v[450:457]*/, v[208:215], v[144:151], v[194:201] /*v[450:457]*/
	s_set_vgpr_msb 0x50a0
	v_wmma_f32_16x16x32_bf16 v[104:111] /*v[616:623]*/, v[232:239], v[168:175], v[104:111] /*v[616:623]*/
	s_set_vgpr_msb 0xa050
	v_wmma_f32_16x16x32_bf16 v[214:221] /*v[470:477]*/, v[240:247], v[144:151], v[214:221] /*v[470:477]*/
	s_set_vgpr_msb 0x50a1
	v_wmma_f32_16x16x32_bf16 v[112:119] /*v[624:631]*/, v[8:15] /*v[264:271]*/, v[168:175], v[112:119] /*v[624:631]*/
	s_set_vgpr_msb 0xa151
	v_wmma_f32_16x16x32_bf16 v[232:239] /*v[488:495]*/, v[16:23] /*v[272:279]*/, v[144:151], v[232:239] /*v[488:495]*/
	s_set_vgpr_msb 0x51a1
	v_wmma_f32_16x16x32_bf16 v[120:127] /*v[632:639]*/, v[40:47] /*v[296:303]*/, v[168:175], v[120:127] /*v[632:639]*/
	s_set_vgpr_msb 0xa151
	v_wmma_f32_16x16x32_bf16 v[250:257] /*v[506:513]*/, v[48:55] /*v[304:311]*/, v[144:151], v[250:257] /*v[506:513]*/
	s_set_vgpr_msb 0x51a1
	v_wmma_f32_16x16x32_bf16 v[128:135] /*v[640:647]*/, v[72:79] /*v[328:335]*/, v[168:175], v[128:135] /*v[640:647]*/
	v_wmma_f32_16x16x32_bf16 v[38:45] /*v[550:557]*/, v[80:87] /*v[336:343]*/, v[144:151], v[38:45] /*v[550:557]*/
	v_wmma_f32_16x16x32_bf16 v[136:143] /*v[648:655]*/, v[104:111] /*v[360:367]*/, v[168:175], v[136:143] /*v[648:655]*/
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[562:569]*/, v[112:119] /*v[368:375]*/, v[144:151], v[50:57] /*v[562:569]*/
	v_wmma_f32_16x16x32_bf16 v[144:151] /*v[656:663]*/, v[136:143] /*v[392:399]*/, v[168:175], v[144:151] /*v[656:663]*/
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[570:577]*/, v[144:151] /*v[400:407]*/, v[144:151], v[58:65] /*v[570:577]*/
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x32_bf16 v[152:159] /*v[664:671]*/, v[168:175] /*v[424:431]*/, v[136:143], v[152:159] /*v[664:671]*/
	v_wmma_f32_16x16x32_bf16 v[160:167] /*v[672:679]*/, v[168:175] /*v[424:431]*/, v[168:175], v[160:167] /*v[672:679]*/
	s_set_vgpr_msb 0xa1a0
	v_wmma_f32_16x16x32_bf16 v[96:103] /*v[608:615]*/, v[208:215], v[176:183], v[96:103] /*v[608:615]*/
	s_set_vgpr_msb 0xa050
	v_wmma_f32_16x16x32_bf16 v[194:201] /*v[450:457]*/, v[216:223], v[152:159], v[194:201] /*v[450:457]*/
	s_set_vgpr_msb 0x50a0
	v_wmma_f32_16x16x32_bf16 v[104:111] /*v[616:623]*/, v[240:247], v[176:183], v[104:111] /*v[616:623]*/
	s_set_vgpr_msb 0xa050
	v_wmma_f32_16x16x32_bf16 v[214:221] /*v[470:477]*/, v[248:255], v[152:159], v[214:221] /*v[470:477]*/
	s_set_vgpr_msb 0x50a1
	v_wmma_f32_16x16x32_bf16 v[112:119] /*v[624:631]*/, v[16:23] /*v[272:279]*/, v[176:183], v[112:119] /*v[624:631]*/
	s_set_vgpr_msb 0xa151
	v_wmma_f32_16x16x32_bf16 v[232:239] /*v[488:495]*/, v[24:31] /*v[280:287]*/, v[152:159], v[232:239] /*v[488:495]*/
	s_set_vgpr_msb 0x51a1
	v_wmma_f32_16x16x32_bf16 v[120:127] /*v[632:639]*/, v[48:55] /*v[304:311]*/, v[176:183], v[120:127] /*v[632:639]*/
	s_set_vgpr_msb 0xa151
	v_wmma_f32_16x16x32_bf16 v[250:257] /*v[506:513]*/, v[56:63] /*v[312:319]*/, v[152:159], v[250:257] /*v[506:513]*/
	s_set_vgpr_msb 0x51a1
	v_wmma_f32_16x16x32_bf16 v[128:135] /*v[640:647]*/, v[80:87] /*v[336:343]*/, v[176:183], v[128:135] /*v[640:647]*/
	v_wmma_f32_16x16x32_bf16 v[38:45] /*v[550:557]*/, v[88:95] /*v[344:351]*/, v[152:159], v[38:45] /*v[550:557]*/
	v_wmma_f32_16x16x32_bf16 v[136:143] /*v[648:655]*/, v[112:119] /*v[368:375]*/, v[176:183], v[136:143] /*v[648:655]*/
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[562:569]*/, v[120:127] /*v[376:383]*/, v[152:159], v[50:57] /*v[562:569]*/
	v_wmma_f32_16x16x32_bf16 v[144:151] /*v[656:663]*/, v[144:151] /*v[400:407]*/, v[176:183], v[144:151] /*v[656:663]*/
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[570:577]*/, v[152:159] /*v[408:415]*/, v[152:159], v[58:65] /*v[570:577]*/
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[152:159] /*v[664:671]*/, v[176:183] /*v[432:439]*/, v[144:151], v[152:159] /*v[664:671]*/
	v_wmma_f32_16x16x32_bf16 v[160:167] /*v[672:679]*/, v[176:183] /*v[432:439]*/, v[176:183], v[160:167] /*v[672:679]*/
	s_set_vgpr_msb 0xa1a0
	v_wmma_f32_16x16x32_bf16 v[96:103] /*v[608:615]*/, v[216:223], v[184:191], v[96:103] /*v[608:615]*/
	v_wmma_f32_16x16x32_bf16 v[104:111] /*v[616:623]*/, v[248:255], v[184:191], v[104:111] /*v[616:623]*/
	s_set_vgpr_msb 0xa0a1
	v_wmma_f32_16x16x32_bf16 v[112:119] /*v[624:631]*/, v[24:31] /*v[280:287]*/, v[184:191], v[112:119] /*v[624:631]*/
	v_wmma_f32_16x16x32_bf16 v[120:127] /*v[632:639]*/, v[56:63] /*v[312:319]*/, v[184:191], v[120:127] /*v[632:639]*/
	v_wmma_f32_16x16x32_bf16 v[128:135] /*v[640:647]*/, v[88:95] /*v[344:351]*/, v[184:191], v[128:135] /*v[640:647]*/
	v_wmma_f32_16x16x32_bf16 v[136:143] /*v[648:655]*/, v[120:127] /*v[376:383]*/, v[184:191], v[136:143] /*v[648:655]*/
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
	v_max3_num_f32 v202 /*v458*/, v194 /*v450*/, v195 /*v451*/, v196 /*v452*/
	s_set_vgpr_msb 0x556a
	v_max3_num_f32 v203 /*v459*/, v96 /*v608*/, v97 /*v609*/, v98 /*v610*/
	s_set_vgpr_msb 0x6a55
	v_max3_num_f32 v204 /*v460*/, v197 /*v453*/, v198 /*v454*/, v199 /*v455*/
	s_set_vgpr_msb 0x556a
	v_max3_num_f32 v205 /*v461*/, v99 /*v611*/, v100 /*v612*/, v101 /*v613*/
	s_set_vgpr_msb 0x6a55
	v_max3_num_f32 v206 /*v462*/, v200 /*v456*/, v201 /*v457*/, v214 /*v470*/
	s_set_vgpr_msb 0x556a
	v_max3_num_f32 v207 /*v463*/, v102 /*v614*/, v103 /*v615*/, v104 /*v616*/
	s_set_vgpr_msb 0x6a55
	v_max3_num_f32 v208 /*v464*/, v215 /*v471*/, v216 /*v472*/, v217 /*v473*/
	v_max3_num_f32 v210 /*v466*/, v218 /*v474*/, v219 /*v475*/, v220 /*v476*/
	v_max3_num_f32 v212 /*v468*/, v221 /*v477*/, v232 /*v488*/, v233 /*v489*/
	v_max3_num_f32 v222 /*v478*/, v234 /*v490*/, v235 /*v491*/, v236 /*v492*/
	v_max3_num_f32 v224 /*v480*/, v237 /*v493*/, v238 /*v494*/, v239 /*v495*/
	v_max3_num_f32 v226 /*v482*/, v250 /*v506*/, v251 /*v507*/, v252 /*v508*/
	v_max3_num_f32 v228 /*v484*/, v253 /*v509*/, v254 /*v510*/, v255 /*v511*/
	s_set_vgpr_msb 0x556a
	v_max3_num_f32 v230 /*v486*/, v0 /*v512*/, v1 /*v513*/, v38 /*v550*/
	v_max3_num_f32 v240 /*v496*/, v39 /*v551*/, v40 /*v552*/, v41 /*v553*/
	v_max3_num_f32 v242 /*v498*/, v42 /*v554*/, v43 /*v555*/, v44 /*v556*/
	v_max3_num_f32 v244 /*v500*/, v45 /*v557*/, v50 /*v562*/, v51 /*v563*/
	v_max3_num_f32 v246 /*v502*/, v52 /*v564*/, v53 /*v565*/, v54 /*v566*/
	v_max3_num_f32 v248 /*v504*/, v55 /*v567*/, v56 /*v568*/, v57 /*v569*/
	s_set_vgpr_msb 0x6aaa
	v_max3_num_f32 v2 /*v514*/, v58 /*v570*/, v59 /*v571*/, v60 /*v572*/
	v_max3_num_f32 v4 /*v516*/, v61 /*v573*/, v62 /*v574*/, v63 /*v575*/
	v_dual_max_num_f32 v6 /*v518*/, v64 /*v576*/, v65 /*v577*/ :: v_dual_max_num_f32 v7 /*v519*/, v150 /*v662*/, v151 /*v663*/
	v_max3_num_f32 v8 /*v520*/, v153 /*v665*/, v154 /*v666*/, v155 /*v667*/
	s_set_vgpr_msb 0xaa6a
	v_max3_num_f32 v209 /*v465*/, v105 /*v617*/, v106 /*v618*/, v107 /*v619*/
	v_max3_num_f32 v211 /*v467*/, v108 /*v620*/, v109 /*v621*/, v110 /*v622*/
	v_max3_num_f32 v213 /*v469*/, v111 /*v623*/, v112 /*v624*/, v113 /*v625*/
	v_max3_num_f32 v223 /*v479*/, v114 /*v626*/, v115 /*v627*/, v116 /*v628*/
	v_max3_num_f32 v225 /*v481*/, v117 /*v629*/, v118 /*v630*/, v119 /*v631*/
	v_max3_num_f32 v227 /*v483*/, v120 /*v632*/, v121 /*v633*/, v122 /*v634*/
	v_max3_num_f32 v229 /*v485*/, v123 /*v635*/, v124 /*v636*/, v125 /*v637*/
	v_max3_num_f32 v231 /*v487*/, v126 /*v638*/, v127 /*v639*/, v128 /*v640*/
	v_max3_num_f32 v241 /*v497*/, v129 /*v641*/, v130 /*v642*/, v131 /*v643*/
	v_max3_num_f32 v243 /*v499*/, v132 /*v644*/, v133 /*v645*/, v134 /*v646*/
	v_max3_num_f32 v245 /*v501*/, v135 /*v647*/, v136 /*v648*/, v137 /*v649*/
	v_max3_num_f32 v247 /*v503*/, v138 /*v650*/, v139 /*v651*/, v140 /*v652*/
	v_max3_num_f32 v249 /*v505*/, v141 /*v653*/, v142 /*v654*/, v143 /*v655*/
	s_set_vgpr_msb 0x6aaa
	v_max3_num_f32 v3 /*v515*/, v144 /*v656*/, v145 /*v657*/, v146 /*v658*/
	v_max3_num_f32 v5 /*v517*/, v147 /*v659*/, v148 /*v660*/, v149 /*v661*/
	v_max3_num_f32 v9 /*v521*/, v161 /*v673*/, v162 /*v674*/, v163 /*v675*/
	v_max3_num_f32 v10 /*v522*/, v156 /*v668*/, v157 /*v669*/, v158 /*v670*/
	s_set_vgpr_msb 0xaa55
	v_max3_num_f32 v202 /*v458*/, v202 /*v458*/, v204 /*v460*/, v206 /*v462*/
	v_max3_num_f32 v203 /*v459*/, v203 /*v459*/, v205 /*v461*/, v207 /*v463*/
	v_max3_num_f32 v204 /*v460*/, v208 /*v464*/, v210 /*v466*/, v212 /*v468*/
	v_max3_num_f32 v205 /*v461*/, v222 /*v478*/, v224 /*v480*/, v226 /*v482*/
	v_max3_num_f32 v206 /*v462*/, v228 /*v484*/, v230 /*v486*/, v240 /*v496*/
	v_max3_num_f32 v207 /*v463*/, v242 /*v498*/, v244 /*v500*/, v246 /*v502*/
	s_set_vgpr_msb 0x5569
	v_max3_num_f32 v208 /*v464*/, v248 /*v504*/, v2 /*v514*/, v4 /*v516*/
	s_set_vgpr_msb 0x696a
	v_max3_num_f32 v210 /*v466*/, v6 /*v518*/, v152 /*v664*/, v8 /*v520*/
	s_set_vgpr_msb 0x6aaa
	v_max3_num_f32 v11 /*v523*/, v164 /*v676*/, v165 /*v677*/, v166 /*v678*/
	s_set_vgpr_msb 0xaa55
	v_max3_num_f32 v209 /*v465*/, v209 /*v465*/, v211 /*v467*/, v213 /*v469*/
	v_max3_num_f32 v211 /*v467*/, v223 /*v479*/, v225 /*v481*/, v227 /*v483*/
	v_max3_num_f32 v202 /*v458*/, v202 /*v458*/, v204 /*v460*/, v205 /*v461*/
	v_max3_num_f32 v204 /*v460*/, v206 /*v462*/, v207 /*v463*/, v208 /*v464*/
	s_set_vgpr_msb 0x5569
	v_max3_num_f32 v205 /*v461*/, v210 /*v466*/, v10 /*v522*/, v159 /*v671*/
	s_set_vgpr_msb 0x6955
	v_max3_num_f32 v206 /*v462*/, v229 /*v485*/, v231 /*v487*/, v241 /*v497*/
	v_max3_num_f32 v207 /*v463*/, v243 /*v499*/, v245 /*v501*/, v247 /*v503*/
	s_set_vgpr_msb 0x5569
	v_max3_num_f32 v208 /*v464*/, v249 /*v505*/, v3 /*v515*/, v5 /*v517*/
	s_set_vgpr_msb 0x696a
	v_max3_num_f32 v210 /*v466*/, v7 /*v519*/, v160 /*v672*/, v9 /*v521*/
	s_set_vgpr_msb 0x6a55
	v_max3_num_f32 v202 /*v458*/, v202 /*v458*/, v204 /*v460*/, v205 /*v461*/
	v_max3_num_f32 v203 /*v459*/, v203 /*v459*/, v209 /*v465*/, v211 /*v467*/
	v_max3_num_f32 v204 /*v460*/, v206 /*v462*/, v207 /*v463*/, v208 /*v464*/
	s_set_vgpr_msb 0x5569
	v_max3_num_f32 v205 /*v461*/, v210 /*v466*/, v11 /*v523*/, v167 /*v679*/
	s_set_vgpr_msb 0x6955
	v_max3_num_f32 v203 /*v459*/, v203 /*v459*/, v204 /*v460*/, v205 /*v461*/
	v_dual_mov_b32 v206 /*v462*/, v202 /*v458*/ :: v_dual_mov_b32 v204 /*v460*/, v203 /*v459*/
	v_permlanex16_b32 v206 /*v462*/, v206 /*v462*/, s95, 0xfedcba98
	v_permlanex16_b32 v204 /*v460*/, v204 /*v460*/, s95, 0xfedcba98
	v_dual_max_num_f32 v202 /*v458*/, v202 /*v458*/, v206 /*v462*/ :: v_dual_max_num_f32 v203 /*v459*/, v203 /*v459*/, v204 /*v460*/
	s_set_vgpr_msb 0x5549
	v_sub_f32_e32 v205 /*v461*/, v202 /*v458*/, v66 /*v578*/
	v_max_num_f32_e32 v202 /*v458*/, v202 /*v458*/, v66 /*v578*/
	v_sub_f32_e32 v204 /*v460*/, v203 /*v459*/, v71 /*v583*/
	s_set_vgpr_msb 0x4904
	v_cmp_lt_f32_e32 vcc_lo, 0x41000000, v205 /*v461*/
	s_cmp_eq_u32 vcc_lo, 0
	s_cselect_b32 s2, -1, 0
	s_set_vgpr_msb 0x489
	v_cndmask_b32_e64 v89 /*v601*/, v202 /*v458*/, v66 /*v578*/, s2
	s_set_vgpr_msb 0x8946
	v_cmp_lt_f32_e64 s2, 0x41000000, v204 /*v460*/
	v_max_num_f32_e32 v202 /*v458*/, v71 /*v583*/, v203 /*v459*/
	s_cmp_lg_u32 s2, 0
	s_cselect_b32 s5, -1, 0
	s_cmp_eq_u32 s2, 0
	s_cselect_b32 s2, -1, 0
	s_set_vgpr_msb 0x4689
	v_cndmask_b32_e64 v88 /*v600*/, v202 /*v458*/, v71 /*v583*/, s2
	v_mul_f32_e32 v68 /*v580*/, 0xbfb8aa3b, v89 /*v601*/
	v_mul_f32_e32 v70 /*v582*/, 0xbfb8aa3b, v88 /*v600*/
	s_set_vgpr_msb 0x8961
	v_pk_fma_f32 v[204:205] /*v[460:461]*/, v[198:199] /*v[454:455]*/, s[56:57], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[200:201] /*v[456:457]*/, v[200:201] /*v[456:457]*/, s[56:57], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[208:209] /*v[464:465]*/, v[214:215] /*v[470:471]*/, s[56:57], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[210:211] /*v[466:467]*/, v[234:235] /*v[490:491]*/, s[56:57], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[222:223] /*v[478:479]*/, v[238:239] /*v[494:495]*/, s[56:57], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v206 /*v462*/, v204 /*v460*/
	v_exp_f32_e32 v212 /*v468*/, v205 /*v461*/
	v_exp_f32_e32 v214 /*v470*/, v200 /*v456*/
	v_pk_fma_f32 v[204:205] /*v[460:461]*/, v[216:217] /*v[472:473]*/, s[56:57], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v226 /*v482*/, v201 /*v457*/
	v_exp_f32_e32 v230 /*v486*/, v208 /*v464*/
	v_pk_fma_f32 v[200:201] /*v[456:457]*/, v[218:219] /*v[474:475]*/, s[56:57], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v242 /*v498*/, v209 /*v465*/
	v_nop
	v_pk_fma_f32 v[208:209] /*v[464:465]*/, v[220:221] /*v[476:477]*/, s[56:57], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[218:219] /*v[474:475]*/, v[236:237] /*v[492:493]*/, s[56:57], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6162
	v_pk_fma_f32 v[224:225] /*v[480:481]*/, v[42:43] /*v[554:555]*/, s[56:57], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6241
	v_exp_f32_e32 v248 /*v504*/, v204 /*v460*/
	s_set_vgpr_msb 0x4181
	v_exp_f32_e32 v2 /*v514*/, v205 /*v461*/
	v_nop
	s_set_vgpr_msb 0x8161
	v_pk_fma_f32 v[204:205] /*v[460:461]*/, v[232:233] /*v[488:489]*/, s[56:57], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6181
	v_exp_f32_e32 v20 /*v532*/, v208 /*v464*/
	s_set_vgpr_msb 0x8161
	v_exp_f32_e32 v208 /*v464*/, v210 /*v466*/
	v_exp_f32_e32 v216 /*v472*/, v211 /*v467*/
	v_exp_f32_e32 v220 /*v476*/, v218 /*v474*/
	v_pk_fma_f32 v[210:211] /*v[466:467]*/, v[250:251] /*v[506:507]*/, s[56:57], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v228 /*v484*/, v219 /*v475*/
	v_exp_f32_e32 v232 /*v488*/, v222 /*v478*/
	v_pk_fma_f32 v[218:219] /*v[474:475]*/, v[252:253] /*v[508:509]*/, s[56:57], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v244 /*v500*/, v223 /*v479*/
	v_nop
	v_pk_fma_f32 v[222:223] /*v[478:479]*/, v[254:255] /*v[510:511]*/, s[56:57], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6162
	v_pk_fma_f32 v[236:237] /*v[492:493]*/, v[44:45] /*v[556:557]*/, s[56:57], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6241
	v_exp_f32_e32 v238 /*v494*/, v224 /*v480*/
	s_set_vgpr_msb 0x4162
	v_pk_fma_f32 v[240:241] /*v[496:497]*/, v[50:51] /*v[562:563]*/, s[56:57], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6241
	v_exp_f32_e32 v246 /*v502*/, v225 /*v481*/
	v_nop
	s_set_vgpr_msb 0x4162
	v_pk_fma_f32 v[224:225] /*v[480:481]*/, v[52:53] /*v[564:565]*/, s[56:57], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6261
	v_pk_fma_f32 v[194:195] /*v[450:451]*/, v[194:195] /*v[450:451]*/, s[56:57], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[202:203] /*v[458:459]*/, v[196:197] /*v[452:453]*/, s[56:57], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v250 /*v506*/, v210 /*v466*/
	s_set_vgpr_msb 0x6181
	v_exp_f32_e32 v4 /*v516*/, v211 /*v467*/
	v_exp_f32_e32 v12 /*v524*/, v218 /*v474*/
	s_set_vgpr_msb 0x8162
	v_pk_fma_f32 v[210:211] /*v[466:467]*/, v[0:1] /*v[512:513]*/, s[56:57], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6281
	v_exp_f32_e32 v22 /*v534*/, v219 /*v475*/
	v_exp_f32_e32 v28 /*v540*/, v222 /*v478*/
	s_set_vgpr_msb 0x8162
	v_pk_fma_f32 v[218:219] /*v[474:475]*/, v[38:39] /*v[550:551]*/, s[56:57], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6281
	v_exp_f32_e32 v36 /*v548*/, v223 /*v479*/
	v_nop
	s_set_vgpr_msb 0x8162
	v_pk_fma_f32 v[222:223] /*v[478:479]*/, v[40:41] /*v[552:553]*/, s[56:57], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6241
	v_exp_f32_e32 v252 /*v508*/, v236 /*v492*/
	s_set_vgpr_msb 0x4181
	v_exp_f32_e32 v6 /*v518*/, v237 /*v493*/
	v_exp_f32_e32 v14 /*v526*/, v240 /*v496*/
	s_set_vgpr_msb 0x8162
	v_pk_fma_f32 v[236:237] /*v[492:493]*/, v[54:55] /*v[566:567]*/, s[56:57], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6281
	v_exp_f32_e32 v24 /*v536*/, v241 /*v497*/
	v_exp_f32_e32 v30 /*v542*/, v224 /*v480*/
	s_set_vgpr_msb 0x8162
	v_pk_fma_f32 v[240:241] /*v[496:497]*/, v[56:57] /*v[568:569]*/, s[56:57], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6281
	v_exp_f32_e32 v40 /*v552*/, v225 /*v481*/
	v_nop
	s_set_vgpr_msb 0x8162
	v_pk_fma_f32 v[224:225] /*v[480:481]*/, v[58:59] /*v[570:571]*/, s[56:57], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[254:255] /*v[510:511]*/, v[60:61] /*v[572:573]*/, s[56:57], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x62a2
	v_pk_fma_f32 v[0:1] /*v[512:513]*/, v[62:63] /*v[574:575]*/, s[56:57], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[16:17] /*v[528:529]*/, v[64:65] /*v[576:577]*/, s[56:57], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[32:33] /*v[544:545]*/, v[152:153] /*v[664:665]*/, s[56:57], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[46:47] /*v[558:559]*/, v[154:155] /*v[666:667]*/, s[56:57], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[56:57] /*v[568:569]*/, v[156:157] /*v[668:669]*/, s[56:57], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[62:63] /*v[574:575]*/, v[158:159] /*v[670:671]*/, s[56:57], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[68:69] /*v[580:581]*/, v[96:97] /*v[608:609]*/, s[56:57], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[96:97] /*v[608:609]*/, v[100:101] /*v[612:613]*/, s[56:57], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa241
	v_exp_f32_e32 v196 /*v452*/, v195 /*v451*/
	s_set_vgpr_msb 0x41a2
	v_pk_fma_f32 v[90:91] /*v[602:603]*/, v[98:99] /*v[610:611]*/, s[56:57], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa241
	v_exp_f32_e32 v198 /*v454*/, v202 /*v458*/
	s_set_vgpr_msb 0x4142
	v_exp_f32_e32 v195 /*v451*/, v68 /*v580*/
	v_exp_f32_e32 v197 /*v453*/, v69 /*v581*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[68:69] /*v[580:581]*/, v[102:103] /*v[614:615]*/, s[56:57], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa242
	v_exp_f32_e32 v207 /*v463*/, v96 /*v608*/
	v_exp_f32_e32 v213 /*v469*/, v97 /*v609*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[96:97] /*v[608:609]*/, v[106:107] /*v[618:619]*/, s[56:57], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa241
	v_exp_f32_e32 v202 /*v458*/, v203 /*v459*/
	s_set_vgpr_msb 0x4142
	v_exp_f32_e32 v215 /*v471*/, v68 /*v580*/
	v_exp_f32_e32 v227 /*v483*/, v69 /*v581*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[68:69] /*v[580:581]*/, v[108:109] /*v[620:621]*/, s[56:57], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa242
	v_exp_f32_e32 v249 /*v505*/, v96 /*v608*/
	s_set_vgpr_msb 0x42a2
	v_exp_f32_e32 v3 /*v515*/, v97 /*v609*/
	v_nop
	v_pk_fma_f32 v[96:97] /*v[608:609]*/, v[112:113] /*v[624:625]*/, s[56:57], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa242
	v_exp_f32_e32 v199 /*v455*/, v90 /*v602*/
	v_exp_f32_e32 v203 /*v459*/, v91 /*v603*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[90:91] /*v[602:603]*/, v[104:105] /*v[616:617]*/, s[56:57], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa281
	v_exp_f32_e32 v10 /*v522*/, v200 /*v456*/
	v_exp_f32_e32 v18 /*v530*/, v201 /*v457*/
	s_set_vgpr_msb 0x8141
	v_exp_f32_e32 v200 /*v456*/, v204 /*v460*/
	v_exp_f32_e32 v204 /*v460*/, v205 /*v461*/
	s_set_vgpr_msb 0x41a2
	v_exp_f32_e32 v11 /*v523*/, v68 /*v580*/
	v_exp_f32_e32 v19 /*v531*/, v69 /*v581*/
	v_nop
	v_pk_fma_f32 v[68:69] /*v[580:581]*/, v[114:115] /*v[626:627]*/, s[56:57], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa242
	v_exp_f32_e32 v201 /*v457*/, v96 /*v608*/
	v_exp_f32_e32 v205 /*v461*/, v97 /*v609*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[96:97] /*v[608:609]*/, v[118:119] /*v[630:631]*/, s[56:57], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa242
	v_exp_f32_e32 v231 /*v487*/, v90 /*v602*/
	v_exp_f32_e32 v243 /*v499*/, v91 /*v603*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[90:91] /*v[602:603]*/, v[110:111] /*v[622:623]*/, s[56:57], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa281
	v_exp_f32_e32 v34 /*v546*/, v209 /*v465*/
	s_set_vgpr_msb 0x8142
	v_exp_f32_e32 v209 /*v465*/, v68 /*v580*/
	v_exp_f32_e32 v217 /*v473*/, v69 /*v581*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[68:69] /*v[580:581]*/, v[120:121] /*v[632:633]*/, s[56:57], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa242
	v_exp_f32_e32 v233 /*v489*/, v96 /*v608*/
	v_exp_f32_e32 v245 /*v501*/, v97 /*v609*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[96:97] /*v[608:609]*/, v[124:125] /*v[636:637]*/, s[56:57], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v21 /*v533*/, v90 /*v602*/
	v_exp_f32_e32 v35 /*v547*/, v91 /*v603*/
	v_nop
	v_pk_fma_f32 v[90:91] /*v[602:603]*/, v[116:117] /*v[628:629]*/, s[56:57], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa242
	v_exp_f32_e32 v251 /*v507*/, v68 /*v580*/
	s_set_vgpr_msb 0x42a2
	v_exp_f32_e32 v5 /*v517*/, v69 /*v581*/
	v_nop
	v_pk_fma_f32 v[68:69] /*v[580:581]*/, v[126:127] /*v[638:639]*/, s[56:57], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v29 /*v541*/, v96 /*v608*/
	v_exp_f32_e32 v37 /*v549*/, v97 /*v609*/
	v_nop
	v_pk_fma_f32 v[96:97] /*v[608:609]*/, v[130:131] /*v[642:643]*/, s[56:57], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa242
	v_exp_f32_e32 v221 /*v477*/, v90 /*v602*/
	v_exp_f32_e32 v229 /*v485*/, v91 /*v603*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[90:91] /*v[602:603]*/, v[122:123] /*v[634:635]*/, s[56:57], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa241
	v_exp_f32_e32 v234 /*v490*/, v223 /*v479*/
	s_set_vgpr_msb 0x41a2
	v_exp_f32_e32 v39 /*v551*/, v68 /*v580*/
	v_exp_f32_e32 v49 /*v561*/, v69 /*v581*/
	v_nop
	v_pk_fma_f32 v[68:69] /*v[580:581]*/, v[132:133] /*v[644:645]*/, s[56:57], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa242
	v_exp_f32_e32 v223 /*v479*/, v96 /*v608*/
	v_exp_f32_e32 v235 /*v491*/, v97 /*v609*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[96:97] /*v[608:609]*/, v[136:137] /*v[648:649]*/, s[56:57], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v13 /*v525*/, v90 /*v602*/
	v_exp_f32_e32 v23 /*v535*/, v91 /*v603*/
	v_nop
	v_pk_fma_f32 v[90:91] /*v[602:603]*/, v[128:129] /*v[640:641]*/, s[56:57], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa242
	v_exp_f32_e32 v239 /*v495*/, v68 /*v580*/
	v_exp_f32_e32 v247 /*v503*/, v69 /*v581*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[68:69] /*v[580:581]*/, v[138:139] /*v[650:651]*/, s[56:57], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v15 /*v527*/, v96 /*v608*/
	v_exp_f32_e32 v25 /*v537*/, v97 /*v609*/
	v_nop
	v_pk_fma_f32 v[96:97] /*v[608:609]*/, v[142:143] /*v[654:655]*/, s[56:57], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa281
	v_exp_f32_e32 v38 /*v550*/, v210 /*v466*/
	v_exp_f32_e32 v48 /*v560*/, v211 /*v467*/
	s_set_vgpr_msb 0x8141
	v_exp_f32_e32 v210 /*v466*/, v218 /*v474*/
	v_exp_f32_e32 v218 /*v474*/, v219 /*v475*/
	s_set_vgpr_msb 0x4142
	v_exp_f32_e32 v211 /*v467*/, v90 /*v602*/
	v_exp_f32_e32 v219 /*v475*/, v91 /*v603*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[90:91] /*v[602:603]*/, v[134:135] /*v[646:647]*/, s[56:57], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v31 /*v543*/, v68 /*v580*/
	v_exp_f32_e32 v41 /*v553*/, v69 /*v581*/
	v_nop
	v_pk_fma_f32 v[68:69] /*v[580:581]*/, v[144:145] /*v[656:657]*/, s[56:57], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v53 /*v565*/, v96 /*v608*/
	v_exp_f32_e32 v59 /*v571*/, v97 /*v609*/
	v_nop
	v_pk_fma_f32 v[96:97] /*v[608:609]*/, v[148:149] /*v[660:661]*/, s[56:57], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa241
	v_exp_f32_e32 v194 /*v450*/, v194 /*v450*/
	s_set_vgpr_msb 0x4142
	v_exp_f32_e32 v253 /*v509*/, v90 /*v602*/
	s_set_vgpr_msb 0x42a2
	v_exp_f32_e32 v7 /*v519*/, v91 /*v603*/
	v_nop
	v_pk_fma_f32 v[90:91] /*v[602:603]*/, v[140:141] /*v[652:653]*/, s[56:57], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa281
	v_exp_f32_e32 v44 /*v556*/, v236 /*v492*/
	v_exp_f32_e32 v50 /*v562*/, v237 /*v493*/
	s_set_vgpr_msb 0x8141
	v_exp_f32_e32 v236 /*v492*/, v225 /*v481*/
	s_set_vgpr_msb 0x4182
	v_exp_f32_e32 v8 /*v520*/, v1 /*v513*/
	s_set_vgpr_msb 0x8242
	v_exp_f32_e32 v225 /*v481*/, v68 /*v580*/
	v_exp_f32_e32 v237 /*v493*/, v69 /*v581*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[68:69] /*v[580:581]*/, v[150:151] /*v[662:663]*/, s[56:57], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v1 /*v513*/, v96 /*v608*/
	v_exp_f32_e32 v9 /*v521*/, v97 /*v609*/
	v_nop
	v_pk_fma_f32 v[96:97] /*v[608:609]*/, v[162:163] /*v[674:675]*/, s[56:57], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v45 /*v557*/, v90 /*v602*/
	v_exp_f32_e32 v51 /*v563*/, v91 /*v603*/
	v_nop
	v_pk_fma_f32 v[90:91] /*v[602:603]*/, v[146:147] /*v[658:659]*/, s[56:57], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v26 /*v538*/, v17 /*v529*/
	v_exp_f32_e32 v54 /*v566*/, v47 /*v559*/
	v_exp_f32_e32 v17 /*v529*/, v68 /*v580*/
	v_exp_f32_e32 v27 /*v539*/, v69 /*v581*/
	v_exp_f32_e32 v47 /*v559*/, v96 /*v608*/
	v_pk_fma_f32 v[68:69] /*v[580:581]*/, v[164:165] /*v[676:677]*/, s[56:57], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v55 /*v567*/, v97 /*v609*/
	v_nop
	s_set_vgpr_msb 0xa285
	v_pk_add_f32 v[96:97] /*v[608:609]*/, v[194:195] /*v[450:451]*/, v[196:197] /*v[452:453]*/
	v_pk_add_f32 v[98:99] /*v[610:611]*/, v[202:203] /*v[458:459]*/, v[206:207] /*v[462:463]*/
	s_set_vgpr_msb 0x8541
	v_exp_f32_e32 v222 /*v478*/, v222 /*v478*/
	s_set_vgpr_msb 0x4181
	v_exp_f32_e32 v52 /*v564*/, v240 /*v496*/
	v_exp_f32_e32 v58 /*v570*/, v241 /*v497*/
	s_set_vgpr_msb 0x8141
	v_exp_f32_e32 v224 /*v480*/, v224 /*v480*/
	v_exp_f32_e32 v240 /*v496*/, v254 /*v510*/
	v_exp_f32_e32 v254 /*v510*/, v255 /*v511*/
	s_set_vgpr_msb 0x4142
	v_exp_f32_e32 v241 /*v497*/, v90 /*v602*/
	v_exp_f32_e32 v255 /*v511*/, v91 /*v603*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[90:91] /*v[602:603]*/, v[160:161] /*v[672:673]*/, s[56:57], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v60 /*v572*/, v57 /*v569*/
	v_exp_f32_e32 v57 /*v569*/, v68 /*v580*/
	v_exp_f32_e32 v61 /*v573*/, v69 /*v581*/
	v_nop
	s_set_vgpr_msb 0xa289
	v_pk_add_f32 v[68:69] /*v[580:581]*/, v[198:199] /*v[454:455]*/, v[96:97] /*v[608:609]*/
	v_pk_add_f32 v[96:97] /*v[608:609]*/, v[212:213] /*v[468:469]*/, v[98:99] /*v[610:611]*/
	s_set_vgpr_msb 0x8985
	v_pk_add_f32 v[98:99] /*v[610:611]*/, v[214:215] /*v[470:471]*/, v[226:227] /*v[482:483]*/
	v_pk_add_f32 v[100:101] /*v[612:613]*/, v[242:243] /*v[498:499]*/, v[248:249] /*v[504:505]*/
	s_set_vgpr_msb 0x858a
	v_pk_add_f32 v[102:103] /*v[614:615]*/, v[10:11] /*v[522:523]*/, v[18:19] /*v[530:531]*/
	v_pk_add_f32 v[112:113] /*v[624:625]*/, v[22:23] /*v[534:535]*/, v[28:29] /*v[540:541]*/
	v_pk_add_f32 v[114:115] /*v[626:627]*/, v[38:39] /*v[550:551]*/, v[48:49] /*v[560:561]*/
	s_set_vgpr_msb 0x8a85
	v_pk_add_f32 v[118:119] /*v[630:631]*/, v[238:239] /*v[494:495]*/, v[246:247] /*v[502:503]*/
	s_set_vgpr_msb 0x858a
	v_pk_add_f32 v[120:121] /*v[632:633]*/, v[6:7] /*v[518:519]*/, v[14:15] /*v[526:527]*/
	v_exp_f32_e32 v0 /*v512*/, v0 /*v512*/
	v_exp_f32_e32 v16 /*v528*/, v16 /*v528*/
	v_exp_f32_e32 v42 /*v554*/, v33 /*v545*/
	v_exp_f32_e32 v46 /*v558*/, v46 /*v558*/
	v_exp_f32_e32 v43 /*v555*/, v91 /*v603*/
	s_set_vgpr_msb 0x8a86
	v_pk_add_f32 v[104:105] /*v[616:617]*/, v[34:35] /*v[546:547]*/, v[200:201] /*v[456:457]*/
	s_set_vgpr_msb 0x8685
	v_pk_add_f32 v[106:107] /*v[618:619]*/, v[208:209] /*v[464:465]*/, v[216:217] /*v[472:473]*/
	s_set_vgpr_msb 0x8589
	v_pk_add_f32 v[98:99] /*v[610:611]*/, v[230:231] /*v[486:487]*/, v[98:99] /*v[610:611]*/
	s_set_vgpr_msb 0x898a
	v_pk_add_f32 v[100:101] /*v[612:613]*/, v[2:3] /*v[514:515]*/, v[100:101] /*v[612:613]*/
	v_pk_add_f32 v[102:103] /*v[614:615]*/, v[20:21] /*v[532:533]*/, v[102:103] /*v[614:615]*/
	s_set_vgpr_msb 0x8a85
	v_pk_add_f32 v[108:109] /*v[620:621]*/, v[228:229] /*v[484:485]*/, v[232:233] /*v[488:489]*/
	v_pk_add_f32 v[116:117] /*v[628:629]*/, v[218:219] /*v[474:475]*/, v[222:223] /*v[478:479]*/
	s_set_vgpr_msb 0x858a
	v_pk_add_f32 v[112:113] /*v[624:625]*/, v[36:37] /*v[548:549]*/, v[112:113] /*v[624:625]*/
	s_set_vgpr_msb 0x8a89
	v_pk_add_f32 v[114:115] /*v[626:627]*/, v[210:211] /*v[466:467]*/, v[114:115] /*v[626:627]*/
	s_set_vgpr_msb 0x898a
	v_pk_add_f32 v[122:123] /*v[634:635]*/, v[30:31] /*v[542:543]*/, v[40:41] /*v[552:553]*/
	v_pk_add_f32 v[124:125] /*v[636:637]*/, v[50:51] /*v[562:563]*/, v[52:53] /*v[564:565]*/
	s_set_vgpr_msb 0x8a85
	v_pk_add_f32 v[126:127] /*v[638:639]*/, v[224:225] /*v[480:481]*/, v[236:237] /*v[492:493]*/
	s_set_vgpr_msb 0x8589
	v_pk_add_f32 v[118:119] /*v[630:631]*/, v[252:253] /*v[508:509]*/, v[118:119] /*v[630:631]*/
	s_set_vgpr_msb 0x89aa
	v_pk_add_f32 v[120:121] /*v[632:633]*/, v[24:25] /*v[536:537]*/, v[120:121] /*v[632:633]*/
	v_pk_add_f32 v[68:69] /*v[580:581]*/, v[68:69] /*v[580:581]*/, v[96:97] /*v[608:609]*/
	v_exp_f32_e32 v32 /*v544*/, v32 /*v544*/
	v_exp_f32_e32 v56 /*v568*/, v56 /*v568*/
	v_exp_f32_e32 v33 /*v545*/, v90 /*v602*/
	v_nop
	v_pk_fma_f32 v[90:91] /*v[602:603]*/, v[166:167] /*v[678:679]*/, s[56:57], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xaa89
	v_pk_add_f32 v[104:105] /*v[616:617]*/, v[204:205] /*v[460:461]*/, v[104:105] /*v[616:617]*/
	v_pk_add_f32 v[106:107] /*v[618:619]*/, v[220:221] /*v[476:477]*/, v[106:107] /*v[618:619]*/
	v_pk_add_f32 v[110:111] /*v[622:623]*/, v[250:251] /*v[506:507]*/, v[4:5] /*v[516:517]*/
	v_pk_add_f32 v[108:109] /*v[620:621]*/, v[244:245] /*v[500:501]*/, v[108:109] /*v[620:621]*/
	v_pk_add_f32 v[116:117] /*v[628:629]*/, v[234:235] /*v[490:491]*/, v[116:117] /*v[628:629]*/
	s_set_vgpr_msb 0x898a
	v_pk_add_f32 v[122:123] /*v[634:635]*/, v[44:45] /*v[556:557]*/, v[122:123] /*v[634:635]*/
	v_pk_add_f32 v[124:125] /*v[636:637]*/, v[58:59] /*v[570:571]*/, v[124:125] /*v[636:637]*/
	s_set_vgpr_msb 0x8a89
	v_pk_add_f32 v[126:127] /*v[638:639]*/, v[240:241] /*v[496:497]*/, v[126:127] /*v[638:639]*/
	v_pk_add_f32 v[128:129] /*v[640:641]*/, v[254:255] /*v[510:511]*/, v[0:1] /*v[512:513]*/
	s_set_vgpr_msb 0x898a
	v_pk_add_f32 v[130:131] /*v[642:643]*/, v[16:17] /*v[528:529]*/, v[26:27] /*v[538:539]*/
	v_pk_add_f32 v[132:133] /*v[644:645]*/, v[42:43] /*v[554:555]*/, v[46:47] /*v[558:559]*/
	v_pk_add_f32 v[68:69] /*v[580:581]*/, v[98:99] /*v[610:611]*/, v[68:69] /*v[580:581]*/
	v_pk_add_f32 v[98:99] /*v[610:611]*/, v[100:101] /*v[612:613]*/, v[102:103] /*v[614:615]*/
	v_pk_add_f32 v[100:101] /*v[612:613]*/, v[112:113] /*v[624:625]*/, v[114:115] /*v[626:627]*/
	v_pk_add_f32 v[102:103] /*v[614:615]*/, v[118:119] /*v[630:631]*/, v[120:121] /*v[632:633]*/
	v_exp_f32_e32 v62 /*v574*/, v62 /*v574*/
	v_exp_f32_e32 v64 /*v576*/, v63 /*v575*/
	v_exp_f32_e32 v63 /*v575*/, v90 /*v602*/
	v_pk_add_f32 v[110:111] /*v[622:623]*/, v[12:13] /*v[524:525]*/, v[110:111] /*v[622:623]*/
	v_pk_add_f32 v[134:135] /*v[646:647]*/, v[56:57] /*v[568:569]*/, v[60:61] /*v[572:573]*/
	v_pk_add_f32 v[96:97] /*v[608:609]*/, v[8:9] /*v[520:521]*/, v[128:129] /*v[640:641]*/
	v_pk_add_f32 v[128:129] /*v[640:641]*/, v[32:33] /*v[544:545]*/, v[130:131] /*v[642:643]*/
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
	v_exp_f32_e32 v65 /*v577*/, v91 /*v603*/
	v_sub_f32_e32 v70 /*v582*/, v66 /*v578*/, v89 /*v601*/
	v_pk_add_f32 v[90:91] /*v[602:603]*/, v[132:133] /*v[644:645]*/, v[106:107] /*v[618:619]*/
	v_pk_add_f32 v[68:69] /*v[580:581]*/, v[104:105] /*v[616:617]*/, v[68:69] /*v[580:581]*/
	v_pk_add_f32 v[96:97] /*v[608:609]*/, v[96:97] /*v[608:609]*/, v[98:99] /*v[610:611]*/
	v_pk_add_f32 v[90:91] /*v[602:603]*/, v[64:65] /*v[576:577]*/, v[90:91] /*v[602:603]*/
	v_pk_add_f32 v[68:69] /*v[580:581]*/, v[68:69] /*v[580:581]*/, v[96:97] /*v[608:609]*/
	v_pk_add_f32 v[66:67] /*v[578:579]*/, v[90:91] /*v[602:603]*/, v[68:69] /*v[580:581]*/
	v_mov_b32_e32 v68 /*v580*/, v66 /*v578*/
	v_dual_mul_f32 v70 /*v582*/, 0x3fb8aa3b, v70 /*v582*/ :: v_dual_mov_b32 v69 /*v581*/, v67 /*v579*/
	v_permlanex16_b32 v68 /*v580*/, v68 /*v580*/, s95, 0xfedcba98
	v_exp_f32_e32 v70 /*v582*/, v70 /*v582*/
	v_permlanex16_b32 v69 /*v581*/, v69 /*v581*/, s95, 0xfedcba98
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
	v_sub_f32_e32 v71 /*v583*/, v71 /*v583*/, v88 /*v600*/
	s_and_b32 s2, s5, exec_lo
	s_cselect_b32 s2, 1, 0
	s_cmp_lg_u32 s2, 1
	v_mul_f32_e32 v71 /*v583*/, 0x3fb8aa3b, v71 /*v583*/
	v_exp_f32_e32 v71 /*v583*/, v71 /*v583*/
	s_set_vgpr_msb 0x8a00
	s_cbranch_scc1 .LBB0_3
	v_nop
	s_set_vgpr_msb 0x82
	v_mov_b32_e32 v90 /*v602*/, v71 /*v583*/
	s_set_vgpr_msb 0x8208
	v_pk_mul_f32 v[62:63], v[62:63], v[90:91] /*v[602:603]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[60:61], v[60:61], v[90:91] /*v[602:603]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59], v[58:59], v[90:91] /*v[602:603]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[56:57], v[56:57], v[90:91] /*v[602:603]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[54:55], v[54:55], v[90:91] /*v[602:603]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[52:53], v[52:53], v[90:91] /*v[602:603]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[50:51], v[50:51], v[90:91] /*v[602:603]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[48:49], v[48:49], v[90:91] /*v[602:603]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[46:47], v[46:47], v[90:91] /*v[602:603]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[44:45], v[44:45], v[90:91] /*v[602:603]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[42:43], v[42:43], v[90:91] /*v[602:603]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[40:41], v[40:41], v[90:91] /*v[602:603]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39], v[38:39], v[90:91] /*v[602:603]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[36:37], v[36:37], v[90:91] /*v[602:603]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35], v[34:35], v[90:91] /*v[602:603]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[32:33], v[32:33], v[90:91] /*v[602:603]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[30:31], v[30:31], v[90:91] /*v[602:603]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[28:29], v[28:29], v[90:91] /*v[602:603]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[26:27], v[26:27], v[90:91] /*v[602:603]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25], v[24:25], v[90:91] /*v[602:603]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[22:23], v[22:23], v[90:91] /*v[602:603]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21], v[20:21], v[90:91] /*v[602:603]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[18:19], v[18:19], v[90:91] /*v[602:603]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[16:17], v[16:17], v[90:91] /*v[602:603]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[14:15], v[90:91] /*v[602:603]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13], v[12:13], v[90:91] /*v[602:603]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[10:11], v[90:91] /*v[602:603]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[8:9], v[90:91] /*v[602:603]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[6:7], v[90:91] /*v[602:603]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[4:5], v[90:91] /*v[602:603]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[2:3], v[2:3], v[90:91] /*v[602:603]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[0:1], v[0:1], v[90:91] /*v[602:603]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x800
	s_branch .LBB0_3
.LBB0_10:
	s_and_b32 vcc_lo, exec_lo, s18
	s_cbranch_vccnz .LBB0_23
	s_branch .LBB0_43
.LBB0_11:
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
	s_set_vgpr_msb 0x82
	v_dual_mov_b32 v88 /*v600*/, 0xf149f2ca :: v_dual_mov_b32 v87 /*v599*/, v80 /*v592*/
	s_set_vgpr_msb 0x8240
	v_dual_mov_b32 v193 /*v449*/, v0 :: v_dual_mov_b32 v192 /*v448*/, v0
	s_set_vgpr_msb 0x4080
	v_mov_b32_e32 v89 /*v601*/, 0xf149f2ca
	s_set_vgpr_msb 0x8000
.LBB0_12:
	s_cmp_ge_u32 s52, s10
	s_cbranch_scc1 .LBB0_21
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 8
	v_dual_add_nc_u32 v192, s11, v74 /*v586*/ :: v_dual_add_nc_u32 v193, s11, v75 /*v587*/
	s_add_co_i32 s2, s58, -1
	s_mov_b32 s43, 0
	s_mov_b32 s4, 1
	s_set_vgpr_msb 0x880
	v_min_i32_e32 v90 /*v602*/, s2, v192
	v_min_i32_e32 v91 /*v603*/, s2, v193
	s_mov_b32 s11, s43
	s_mov_b32 s40, 32
	s_mov_b32 s39, 0x800000
	s_mov_b32 s37, 0xffff0000
	s_mov_b32 s36, 0x7510000
	s_mov_b32 s44, 0xf510000
	s_mov_b32 s82, 0x76543210
	s_mov_b32 s56, 0x3fb8aa3b
	s_set_vgpr_msb 0x8000
	s_branch .LBB0_15
.LBB0_14:
	s_set_vgpr_msb 0x8a
	v_cvt_pk_bf16_f32 v103 /*v615*/, v14 /*v526*/, v26 /*v538*/
	s_set_vgpr_msb 0x8a89
	v_cvt_pk_bf16_f32 v102 /*v614*/, v254 /*v510*/, v10 /*v522*/
	s_set_vgpr_msb 0x8985
	v_cvt_pk_bf16_f32 v101 /*v613*/, v238 /*v494*/, v250 /*v506*/
	v_cvt_pk_bf16_f32 v100 /*v612*/, v224 /*v480*/, v234 /*v490*/
	v_cvt_pk_bf16_f32 v99 /*v611*/, v212 /*v468*/, v220 /*v476*/
	v_cvt_pk_bf16_f32 v98 /*v610*/, v204 /*v460*/, v210 /*v466*/
	v_cvt_pk_bf16_f32 v97 /*v609*/, v198 /*v454*/, v202 /*v458*/
	v_cvt_pk_bf16_f32 v96 /*v608*/, v194 /*v450*/, v196 /*v452*/
	s_set_vgpr_msb 0x858a
	v_cvt_pk_bf16_f32 v111 /*v623*/, v15 /*v527*/, v27 /*v539*/
	s_set_vgpr_msb 0x8a89
	v_cvt_pk_bf16_f32 v110 /*v622*/, v255 /*v511*/, v11 /*v523*/
	s_set_vgpr_msb 0x8985
	v_cvt_pk_bf16_f32 v109 /*v621*/, v239 /*v495*/, v251 /*v507*/
	v_cvt_pk_bf16_f32 v108 /*v620*/, v225 /*v481*/, v235 /*v491*/
	v_cvt_pk_bf16_f32 v107 /*v619*/, v213 /*v469*/, v221 /*v477*/
	v_cvt_pk_bf16_f32 v106 /*v618*/, v205 /*v461*/, v211 /*v467*/
	v_cvt_pk_bf16_f32 v105 /*v617*/, v199 /*v455*/, v203 /*v459*/
	v_cvt_pk_bf16_f32 v104 /*v616*/, v195 /*v451*/, v197 /*v453*/
	s_set_vgpr_msb 0x8509
	s_wait_dscnt 0x3d
	v_wmma_f32_16x16x32_bf16 v[120:127], v[184:191] /*v[440:447]*/, v[96:103] /*v[608:615]*/, v[120:127]
	s_set_vgpr_msb 0x98a
	v_cvt_pk_bf16_f32 v119 /*v631*/, v37 /*v549*/, v45 /*v557*/
	v_cvt_pk_bf16_f32 v118 /*v630*/, v23 /*v535*/, v33 /*v545*/
	v_cvt_pk_bf16_f32 v117 /*v629*/, v7 /*v519*/, v19 /*v531*/
	s_set_vgpr_msb 0x8a89
	v_cvt_pk_bf16_f32 v116 /*v628*/, v247 /*v503*/, v3 /*v515*/
	s_set_vgpr_msb 0x8985
	v_cvt_pk_bf16_f32 v115 /*v627*/, v231 /*v487*/, v243 /*v499*/
	v_cvt_pk_bf16_f32 v114 /*v626*/, v219 /*v475*/, v229 /*v485*/
	v_cvt_pk_bf16_f32 v113 /*v625*/, v209 /*v465*/, v217 /*v473*/
	s_set_vgpr_msb 0x8509
	v_wmma_f32_16x16x32_bf16 v[56:63], v[184:191] /*v[440:447]*/, v[104:111] /*v[616:623]*/, v[56:63]
	s_set_vgpr_msb 0x985
	v_cvt_pk_bf16_f32 v112 /*v624*/, v201 /*v457*/, v207 /*v463*/
	s_set_vgpr_msb 0x854a
	v_cvt_pk_bf16_f32 v201 /*v457*/, v53 /*v565*/, v59 /*v571*/
	v_cvt_pk_bf16_f32 v199 /*v455*/, v31 /*v543*/, v41 /*v553*/
	v_cvt_pk_bf16_f32 v198 /*v454*/, v17 /*v529*/, v29 /*v541*/
	v_cvt_pk_bf16_f32 v191 /*v447*/, v36 /*v548*/, v44 /*v556*/
	v_cvt_pk_bf16_f32 v190 /*v446*/, v22 /*v534*/, v32 /*v544*/
	v_cvt_pk_bf16_f32 v189 /*v445*/, v6 /*v518*/, v18 /*v530*/
	s_set_vgpr_msb 0x4a09
	s_wait_dscnt 0x3c
	v_wmma_f32_16x16x32_bf16 v[112:119], v[152:159] /*v[408:415]*/, v[96:103] /*v[608:615]*/, v[112:119]
	s_set_vgpr_msb 0x949
	v_cvt_pk_bf16_f32 v188 /*v444*/, v246 /*v502*/, v2 /*v514*/
	s_set_vgpr_msb 0x4945
	v_cvt_pk_bf16_f32 v187 /*v443*/, v230 /*v486*/, v242 /*v498*/
	v_cvt_pk_bf16_f32 v186 /*v442*/, v218 /*v474*/, v228 /*v484*/
	v_cvt_pk_bf16_f32 v185 /*v441*/, v208 /*v464*/, v216 /*v472*/
	v_cvt_pk_bf16_f32 v184 /*v440*/, v200 /*v456*/, v206 /*v462*/
	s_set_vgpr_msb 0x454a
	v_cvt_pk_bf16_f32 v200 /*v456*/, v43 /*v555*/, v51 /*v563*/
	v_cvt_pk_bf16_f32 v197 /*v453*/, v1 /*v513*/, v13 /*v525*/
	s_set_vgpr_msb 0x4a09
	v_wmma_f32_16x16x32_bf16 v[48:55], v[152:159] /*v[408:415]*/, v[104:111] /*v[616:623]*/, v[48:55]
	s_set_vgpr_msb 0x945
	v_cvt_pk_bf16_f32 v196 /*v452*/, v241 /*v497*/, v253 /*v509*/
	v_cvt_pk_bf16_f32 v195 /*v451*/, v227 /*v483*/, v237 /*v493*/
	v_cvt_pk_bf16_f32 v194 /*v450*/, v215 /*v471*/, v223 /*v479*/
	s_set_vgpr_msb 0x454a
	v_cvt_pk_bf16_f32 v209 /*v465*/, v63 /*v575*/, v65 /*v577*/
	v_cvt_pk_bf16_f32 v208 /*v464*/, v57 /*v569*/, v61 /*v573*/
	v_cvt_pk_bf16_f32 v207 /*v463*/, v49 /*v561*/, v55 /*v567*/
	v_cvt_pk_bf16_f32 v206 /*v462*/, v39 /*v551*/, v47 /*v559*/
	s_set_vgpr_msb 0x4a09
	s_wait_dscnt 0x2d
	v_wmma_f32_16x16x32_bf16 v[104:111], v[120:127] /*v[376:383]*/, v[96:103] /*v[608:615]*/, v[104:111]
	s_set_vgpr_msb 0x94a
	v_cvt_pk_bf16_f32 v205 /*v461*/, v25 /*v537*/, v35 /*v547*/
	v_cvt_pk_bf16_f32 v204 /*v460*/, v9 /*v521*/, v21 /*v533*/
	s_set_vgpr_msb 0x4a49
	v_cvt_pk_bf16_f32 v203 /*v459*/, v249 /*v505*/, v5 /*v517*/
	s_set_vgpr_msb 0x4945
	v_cvt_pk_bf16_f32 v202 /*v458*/, v233 /*v489*/, v245 /*v501*/
	s_set_vgpr_msb 0x4586
	v_dual_fmac_f32 v89 /*v601*/, v68 /*v580*/, v192 /*v448*/ :: v_dual_fmac_f32 v94 /*v606*/, v70 /*v582*/, v193 /*v449*/
	s_add_nc_u64 s[52:53], s[52:53], 1
	s_set_vgpr_msb 0x8609
	v_wmma_f32_16x16x32_bf16 v[40:47], v[120:127] /*v[376:383]*/, v[104:111] /*v[616:623]*/, v[40:47]
	v_cmp_ge_u64_e64 s2, s[52:53], s[10:11]
	s_set_vgpr_msb 0x94a
	v_dual_add_f32 v192 /*v448*/, v89 /*v601*/, v66 /*v578*/ :: v_dual_add_f32 v193 /*v449*/, v94 /*v606*/, v67 /*v579*/
	s_wait_alu depctr_vm_vsrc(0)
	s_set_vgpr_msb 0x4a82
	v_dual_mov_b32 v94 /*v606*/, v86 /*v598*/ :: v_dual_mov_b32 v86 /*v598*/, v93 /*v605*/
	v_dual_mov_b32 v93 /*v605*/, v87 /*v599*/ :: v_dual_mov_b32 v87 /*v599*/, v92 /*v604*/
	s_set_vgpr_msb 0x8209
	s_wait_dscnt 0x2c
	v_wmma_f32_16x16x32_bf16 v[96:103], v[88:95] /*v[344:351]*/, v[96:103] /*v[608:615]*/, v[96:103]
	s_set_vgpr_msb 0x982
	v_dual_mov_b32 v88 /*v600*/, v71 /*v583*/ :: v_dual_mov_b32 v89 /*v601*/, v69 /*v581*/
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
	v_cvt_pk_bf16_f32 v178 /*v434*/, v240 /*v496*/, v252 /*v508*/
	v_cvt_pk_bf16_f32 v177 /*v433*/, v226 /*v482*/, v236 /*v492*/
	v_cvt_pk_bf16_f32 v176 /*v432*/, v214 /*v470*/, v222 /*v478*/
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
	v_wmma_f32_16x16x32_bf16 v[56:63], v[168:175] /*v[424:431]*/, v[194:201] /*v[450:457]*/, v[56:63]
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
	v_cvt_pk_bf16_f32 v169 /*v425*/, v248 /*v504*/, v4 /*v516*/
	s_set_vgpr_msb 0x4945
	v_cvt_pk_bf16_f32 v168 /*v424*/, v232 /*v488*/, v244 /*v500*/
	s_set_vgpr_msb 0x4505
	v_wmma_f32_16x16x32_bf16 v[48:55], v[136:143] /*v[392:399]*/, v[194:201] /*v[450:457]*/, v[48:55]
	v_wmma_f32_16x16x32_bf16 v[104:111], v[104:111] /*v[360:367]*/, v[176:183] /*v[432:439]*/, v[104:111]
	v_wmma_f32_16x16x32_bf16 v[40:47], v[104:111] /*v[360:367]*/, v[194:201] /*v[450:457]*/, v[40:47]
	v_wmma_f32_16x16x32_bf16 v[96:103], v[72:79] /*v[328:335]*/, v[176:183] /*v[432:439]*/, v[96:103]
	v_wmma_f32_16x16x32_bf16 v[32:39], v[72:79] /*v[328:335]*/, v[194:201] /*v[450:457]*/, v[32:39]
	v_wmma_f32_16x16x32_bf16 v[88:95], v[40:47] /*v[296:303]*/, v[176:183] /*v[432:439]*/, v[88:95]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[40:47] /*v[296:303]*/, v[194:201] /*v[450:457]*/, v[24:31]
	v_wmma_f32_16x16x32_bf16 v[80:87], v[8:15] /*v[264:271]*/, v[176:183] /*v[432:439]*/, v[80:87]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[8:15] /*v[264:271]*/, v[194:201] /*v[450:457]*/, v[16:23]
	s_set_vgpr_msb 0x504
	v_wmma_f32_16x16x32_bf16 v[72:79], v[232:239], v[176:183] /*v[432:439]*/, v[72:79]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[232:239], v[194:201] /*v[450:457]*/, v[8:15]
	v_wmma_f32_16x16x32_bf16 v[64:71], v[200:207], v[176:183] /*v[432:439]*/, v[64:71]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[200:207], v[194:201] /*v[450:457]*/, v[0:7]
	s_set_vgpr_msb 0x405
	v_wmma_f32_16x16x32_bf16 v[120:127], v[160:167] /*v[416:423]*/, v[168:175] /*v[424:431]*/, v[120:127]
	v_wmma_f32_16x16x32_bf16 v[56:63], v[160:167] /*v[416:423]*/, v[202:209] /*v[458:465]*/, v[56:63]
	v_wmma_f32_16x16x32_bf16 v[112:119], v[128:135] /*v[384:391]*/, v[168:175] /*v[424:431]*/, v[112:119]
	v_wmma_f32_16x16x32_bf16 v[48:55], v[128:135] /*v[384:391]*/, v[202:209] /*v[458:465]*/, v[48:55]
	v_wmma_f32_16x16x32_bf16 v[104:111], v[96:103] /*v[352:359]*/, v[168:175] /*v[424:431]*/, v[104:111]
	v_wmma_f32_16x16x32_bf16 v[40:47], v[96:103] /*v[352:359]*/, v[202:209] /*v[458:465]*/, v[40:47]
	v_wmma_f32_16x16x32_bf16 v[96:103], v[64:71] /*v[320:327]*/, v[168:175] /*v[424:431]*/, v[96:103]
	v_wmma_f32_16x16x32_bf16 v[32:39], v[64:71] /*v[320:327]*/, v[202:209] /*v[458:465]*/, v[32:39]
	v_wmma_f32_16x16x32_bf16 v[88:95], v[32:39] /*v[288:295]*/, v[168:175] /*v[424:431]*/, v[88:95]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[32:39] /*v[288:295]*/, v[202:209] /*v[458:465]*/, v[24:31]
	v_wmma_f32_16x16x32_bf16 v[80:87], v[0:7] /*v[256:263]*/, v[168:175] /*v[424:431]*/, v[80:87]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[0:7] /*v[256:263]*/, v[202:209] /*v[458:465]*/, v[16:23]
	s_set_vgpr_msb 0x504
	v_wmma_f32_16x16x32_bf16 v[72:79], v[224:231], v[168:175] /*v[424:431]*/, v[72:79]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[224:231], v[202:209] /*v[458:465]*/, v[8:15]
	v_wmma_f32_16x16x32_bf16 v[64:71], v[192:199], v[168:175] /*v[424:431]*/, v[64:71]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[192:199], v[202:209] /*v[458:465]*/, v[0:7]
	s_set_vgpr_msb 0x400
	s_cbranch_vccnz .LBB0_22
.LBB0_15:
	s_wait_alu depctr_vm_vsrc(6)
	s_set_vgpr_msb 0x82
	v_dual_mov_b32 v92 /*v604*/, v93 /*v605*/ :: v_dual_mov_b32 v93 /*v605*/, v94 /*v606*/
	s_add_co_i32 s2, s52, 1
	s_wait_tensorcnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	s_cmp_ge_i32 s2, s10
	s_set_vgpr_msb 0x8200
	s_cbranch_scc1 .LBB0_17
	s_lshl_b32 s5, s2, 7
	s_lshl_b32 s2, s2, 17
	s_sub_co_i32 s7, s9, s5
	s_add_co_i32 s6, s5, s64
	v_nop
	v_nop
	v_nop
	v_med3_i32 v192, s7, 0, 0x80
	s_ashr_i32 s7, s6, 31
	s_and_b32 s2, s2, 0x20000
	s_mul_u64 s[46:47], s[6:7], s[62:63]
	s_mul_u64 s[6:7], s[6:7], s[60:61]
	v_readfirstlane_b32 s38, v192
	s_lshl_b64 s[6:7], s[6:7], 1
	s_lshl_b64 s[46:47], s[46:47], 1
	s_add_nc_u64 s[6:7], s[80:81], s[6:7]
	s_or_b32 s5, s92, s2
	s_sub_co_i32 s38, s38, s17
	s_add_nc_u64 s[6:7], s[18:19], s[6:7]
	s_max_i32 s38, s38, 0
	s_add_nc_u64 s[46:47], s[54:55], s[46:47]
	s_lshl_b32 s38, s38, 16
	s_bitset1_b32 s7, 31
	s_addk_co_i32 s38, 0x7fff
	s_mov_b32 s45, s37
	tensor_load_to_lds s[4:7], s[36:43]
	s_add_nc_u64 s[6:7], s[78:79], s[46:47]
	s_or_b32 s5, s93, s2
	s_bitset1_b32 s7, 31
	s_mov_b32 s46, s38
	s_mov_b32 s47, s39
	s_mov_b32 s48, s40
	s_mov_b32 s51, s43
	tensor_load_to_lds s[4:7], s[44:51]
.LBB0_17:
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 2
	ds_load_b128 v[192:195], v87 /*v599*/
	ds_load_b128 v[196:199], v87 /*v599*/ offset:32
	ds_load_b128 v[200:203], v87 /*v599*/ offset:64
	ds_load_b128 v[204:207], v87 /*v599*/ offset:96
	ds_load_b128 v[208:211], v87 /*v599*/ offset:128
	ds_load_b128 v[212:215], v87 /*v599*/ offset:160
	ds_load_b128 v[216:219], v87 /*v599*/ offset:192
	ds_load_b128 v[220:223], v87 /*v599*/ offset:224
	ds_load_b128 v[224:227], v87 /*v599*/ offset:4352
	ds_load_b128 v[228:231], v87 /*v599*/ offset:4384
	ds_load_b128 v[232:235], v87 /*v599*/ offset:4416
	ds_load_b128 v[236:239], v87 /*v599*/ offset:4448
	ds_load_b128 v[240:243], v87 /*v599*/ offset:4480
	ds_load_b128 v[244:247], v87 /*v599*/ offset:4512
	ds_load_b128 v[248:251], v87 /*v599*/ offset:4544
	ds_load_b128 v[252:255], v87 /*v599*/ offset:4576
	s_set_vgpr_msb 0x242
	ds_load_b128 v[0:3] /*v[256:259]*/, v87 /*v599*/ offset:8704
	ds_load_b128 v[4:7] /*v[260:263]*/, v87 /*v599*/ offset:8736
	ds_load_b128 v[8:11] /*v[264:267]*/, v87 /*v599*/ offset:8768
	ds_load_b128 v[12:15] /*v[268:271]*/, v87 /*v599*/ offset:8800
	ds_load_b128 v[16:19] /*v[272:275]*/, v87 /*v599*/ offset:8832
	ds_load_b128 v[20:23] /*v[276:279]*/, v87 /*v599*/ offset:8864
	ds_load_b128 v[24:27] /*v[280:283]*/, v87 /*v599*/ offset:8896
	ds_load_b128 v[28:31] /*v[284:287]*/, v87 /*v599*/ offset:8928
	ds_load_b128 v[32:35] /*v[288:291]*/, v87 /*v599*/ offset:13056
	ds_load_b128 v[36:39] /*v[292:295]*/, v87 /*v599*/ offset:13088
	ds_load_b128 v[40:43] /*v[296:299]*/, v87 /*v599*/ offset:13120
	ds_load_b128 v[44:47] /*v[300:303]*/, v87 /*v599*/ offset:13152
	ds_load_b128 v[48:51] /*v[304:307]*/, v87 /*v599*/ offset:13184
	ds_load_b128 v[52:55] /*v[308:311]*/, v87 /*v599*/ offset:13216
	ds_load_b128 v[56:59] /*v[312:315]*/, v87 /*v599*/ offset:13248
	ds_load_b128 v[60:63] /*v[316:319]*/, v87 /*v599*/ offset:13280
	ds_load_b128 v[64:67] /*v[320:323]*/, v87 /*v599*/ offset:17408
	ds_load_b128 v[68:71] /*v[324:327]*/, v87 /*v599*/ offset:17440
	ds_load_b128 v[72:75] /*v[328:331]*/, v87 /*v599*/ offset:17472
	ds_load_b128 v[76:79] /*v[332:335]*/, v87 /*v599*/ offset:17504
	ds_load_b128 v[80:83] /*v[336:339]*/, v87 /*v599*/ offset:17536
	ds_load_b128 v[84:87] /*v[340:343]*/, v87 /*v599*/ offset:17568
	ds_load_b128 v[88:91] /*v[344:347]*/, v87 /*v599*/ offset:17600
	ds_load_b128 v[92:95] /*v[348:351]*/, v87 /*v599*/ offset:17632
	ds_load_b128 v[96:99] /*v[352:355]*/, v87 /*v599*/ offset:21760
	ds_load_b128 v[100:103] /*v[356:359]*/, v87 /*v599*/ offset:21792
	ds_load_b128 v[104:107] /*v[360:363]*/, v87 /*v599*/ offset:21824
	ds_load_b128 v[108:111] /*v[364:367]*/, v87 /*v599*/ offset:21856
	ds_load_b128 v[112:115] /*v[368:371]*/, v87 /*v599*/ offset:21888
	ds_load_b128 v[116:119] /*v[372:375]*/, v87 /*v599*/ offset:21920
	ds_load_b128 v[120:123] /*v[376:379]*/, v87 /*v599*/ offset:21952
	ds_load_b128 v[124:127] /*v[380:383]*/, v87 /*v599*/ offset:21984
	ds_load_b128 v[128:131] /*v[384:387]*/, v87 /*v599*/ offset:26112
	ds_load_b128 v[132:135] /*v[388:391]*/, v87 /*v599*/ offset:26144
	ds_load_b128 v[136:139] /*v[392:395]*/, v87 /*v599*/ offset:26176
	ds_load_b128 v[140:143] /*v[396:399]*/, v87 /*v599*/ offset:26208
	ds_load_b128 v[144:147] /*v[400:403]*/, v87 /*v599*/ offset:26240
	ds_load_b128 v[148:151] /*v[404:407]*/, v87 /*v599*/ offset:26272
	ds_load_b128 v[152:155] /*v[408:411]*/, v87 /*v599*/ offset:26304
	ds_load_b128 v[156:159] /*v[412:415]*/, v87 /*v599*/ offset:26336
	ds_load_b128 v[160:163] /*v[416:419]*/, v87 /*v599*/ offset:30464
	ds_load_b128 v[164:167] /*v[420:423]*/, v87 /*v599*/ offset:30496
	ds_load_b128 v[168:171] /*v[424:427]*/, v87 /*v599*/ offset:30528
	ds_load_b128 v[172:175] /*v[428:431]*/, v87 /*v599*/ offset:30560
	ds_load_b128 v[176:179] /*v[432:435]*/, v87 /*v599*/ offset:30592
	ds_load_b128 v[180:183] /*v[436:439]*/, v87 /*v599*/ offset:30624
	ds_load_b128 v[184:187] /*v[440:443]*/, v87 /*v599*/ offset:30656
	ds_load_b128 v[188:191] /*v[444:447]*/, v87 /*v599*/ offset:30688
	s_set_vgpr_msb 0x4240
	s_wait_dscnt 0x3e
	v_wmma_f32_16x16x32_bf16 v[194:201] /*v[450:457]*/, v[192:199], v[128:135], 0
	v_wmma_f32_16x16x32_bf16 v[202:209] /*v[458:465]*/, v[192:199], v[160:167], 0
	s_wait_dscnt 0x36
	v_wmma_f32_16x16x32_bf16 v[210:217] /*v[466:473]*/, v[224:231], v[128:135], 0
	v_wmma_f32_16x16x32_bf16 v[218:225] /*v[474:481]*/, v[224:231], v[160:167], 0
	s_set_vgpr_msb 0x4041
	s_wait_dscnt 0x2e
	v_wmma_f32_16x16x32_bf16 v[226:233] /*v[482:489]*/, v[0:7] /*v[256:263]*/, v[128:135], 0
	v_wmma_f32_16x16x32_bf16 v[234:241] /*v[490:497]*/, v[0:7] /*v[256:263]*/, v[160:167], 0
	s_wait_dscnt 0x26
	v_wmma_f32_16x16x32_bf16 v[242:249] /*v[498:505]*/, v[32:39] /*v[288:295]*/, v[128:135], 0
	v_wmma_f32_16x16x32_bf16 v[250:257] /*v[506:513]*/, v[32:39] /*v[288:295]*/, v[160:167], 0
	s_set_vgpr_msb 0x4181
	s_wait_dscnt 0x1e
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[514:521]*/, v[64:71] /*v[320:327]*/, v[128:135], 0
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[522:529]*/, v[64:71] /*v[320:327]*/, v[160:167], 0
	s_wait_dscnt 0x16
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[530:537]*/, v[96:103] /*v[352:359]*/, v[128:135], 0
	v_wmma_f32_16x16x32_bf16 v[26:33] /*v[538:545]*/, v[96:103] /*v[352:359]*/, v[160:167], 0
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x32_bf16 v[34:41] /*v[546:553]*/, v[128:135] /*v[384:391]*/, v[128:135], 0
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[554:561]*/, v[128:135] /*v[384:391]*/, v[160:167], 0
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[562:569]*/, v[160:167] /*v[416:423]*/, v[128:135], 0
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[570:577]*/, v[160:167] /*v[416:423]*/, v[160:167], 0
	s_set_vgpr_msb 0x8150
	v_wmma_f32_16x16x32_bf16 v[194:201] /*v[450:457]*/, v[200:207], v[136:143], v[194:201] /*v[450:457]*/
	v_wmma_f32_16x16x32_bf16 v[202:209] /*v[458:465]*/, v[200:207], v[168:175], v[202:209] /*v[458:465]*/
	v_wmma_f32_16x16x32_bf16 v[210:217] /*v[466:473]*/, v[232:239], v[136:143], v[210:217] /*v[466:473]*/
	v_wmma_f32_16x16x32_bf16 v[218:225] /*v[474:481]*/, v[232:239], v[168:175], v[218:225] /*v[474:481]*/
	s_set_vgpr_msb 0x5051
	v_wmma_f32_16x16x32_bf16 v[226:233] /*v[482:489]*/, v[8:15] /*v[264:271]*/, v[136:143], v[226:233] /*v[482:489]*/
	v_wmma_f32_16x16x32_bf16 v[234:241] /*v[490:497]*/, v[8:15] /*v[264:271]*/, v[168:175], v[234:241] /*v[490:497]*/
	v_wmma_f32_16x16x32_bf16 v[242:249] /*v[498:505]*/, v[40:47] /*v[296:303]*/, v[136:143], v[242:249] /*v[498:505]*/
	v_wmma_f32_16x16x32_bf16 v[250:257] /*v[506:513]*/, v[40:47] /*v[296:303]*/, v[168:175], v[250:257] /*v[506:513]*/
	s_set_vgpr_msb 0x51a1
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[514:521]*/, v[72:79] /*v[328:335]*/, v[136:143], v[2:9] /*v[514:521]*/
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[522:529]*/, v[72:79] /*v[328:335]*/, v[168:175], v[10:17] /*v[522:529]*/
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[530:537]*/, v[104:111] /*v[360:367]*/, v[136:143], v[18:25] /*v[530:537]*/
	v_wmma_f32_16x16x32_bf16 v[26:33] /*v[538:545]*/, v[104:111] /*v[360:367]*/, v[168:175], v[26:33] /*v[538:545]*/
	v_wmma_f32_16x16x32_bf16 v[34:41] /*v[546:553]*/, v[136:143] /*v[392:399]*/, v[136:143], v[34:41] /*v[546:553]*/
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[554:561]*/, v[136:143] /*v[392:399]*/, v[168:175], v[42:49] /*v[554:561]*/
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[562:569]*/, v[168:175] /*v[424:431]*/, v[136:143], v[50:57] /*v[562:569]*/
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[570:577]*/, v[168:175] /*v[424:431]*/, v[168:175], v[58:65] /*v[570:577]*/
	s_set_vgpr_msb 0xa150
	v_wmma_f32_16x16x32_bf16 v[194:201] /*v[450:457]*/, v[208:215], v[144:151], v[194:201] /*v[450:457]*/
	v_wmma_f32_16x16x32_bf16 v[202:209] /*v[458:465]*/, v[208:215], v[176:183], v[202:209] /*v[458:465]*/
	v_wmma_f32_16x16x32_bf16 v[210:217] /*v[466:473]*/, v[240:247], v[144:151], v[210:217] /*v[466:473]*/
	v_wmma_f32_16x16x32_bf16 v[218:225] /*v[474:481]*/, v[240:247], v[176:183], v[218:225] /*v[474:481]*/
	s_set_vgpr_msb 0x5051
	v_wmma_f32_16x16x32_bf16 v[226:233] /*v[482:489]*/, v[16:23] /*v[272:279]*/, v[144:151], v[226:233] /*v[482:489]*/
	v_wmma_f32_16x16x32_bf16 v[234:241] /*v[490:497]*/, v[16:23] /*v[272:279]*/, v[176:183], v[234:241] /*v[490:497]*/
	v_wmma_f32_16x16x32_bf16 v[242:249] /*v[498:505]*/, v[48:55] /*v[304:311]*/, v[144:151], v[242:249] /*v[498:505]*/
	v_wmma_f32_16x16x32_bf16 v[250:257] /*v[506:513]*/, v[48:55] /*v[304:311]*/, v[176:183], v[250:257] /*v[506:513]*/
	s_set_vgpr_msb 0x51a1
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[514:521]*/, v[80:87] /*v[336:343]*/, v[144:151], v[2:9] /*v[514:521]*/
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[522:529]*/, v[80:87] /*v[336:343]*/, v[176:183], v[10:17] /*v[522:529]*/
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[530:537]*/, v[112:119] /*v[368:375]*/, v[144:151], v[18:25] /*v[530:537]*/
	v_wmma_f32_16x16x32_bf16 v[26:33] /*v[538:545]*/, v[112:119] /*v[368:375]*/, v[176:183], v[26:33] /*v[538:545]*/
	v_wmma_f32_16x16x32_bf16 v[34:41] /*v[546:553]*/, v[144:151] /*v[400:407]*/, v[144:151], v[34:41] /*v[546:553]*/
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[554:561]*/, v[144:151] /*v[400:407]*/, v[176:183], v[42:49] /*v[554:561]*/
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[562:569]*/, v[176:183] /*v[432:439]*/, v[144:151], v[50:57] /*v[562:569]*/
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[570:577]*/, v[176:183] /*v[432:439]*/, v[176:183], v[58:65] /*v[570:577]*/
	s_set_vgpr_msb 0xa150
	v_wmma_f32_16x16x32_bf16 v[194:201] /*v[450:457]*/, v[216:223], v[152:159], v[194:201] /*v[450:457]*/
	v_wmma_f32_16x16x32_bf16 v[202:209] /*v[458:465]*/, v[216:223], v[184:191], v[202:209] /*v[458:465]*/
	v_wmma_f32_16x16x32_bf16 v[210:217] /*v[466:473]*/, v[248:255], v[152:159], v[210:217] /*v[466:473]*/
	v_wmma_f32_16x16x32_bf16 v[218:225] /*v[474:481]*/, v[248:255], v[184:191], v[218:225] /*v[474:481]*/
	s_set_vgpr_msb 0x5051
	v_wmma_f32_16x16x32_bf16 v[226:233] /*v[482:489]*/, v[24:31] /*v[280:287]*/, v[152:159], v[226:233] /*v[482:489]*/
	v_wmma_f32_16x16x32_bf16 v[234:241] /*v[490:497]*/, v[24:31] /*v[280:287]*/, v[184:191], v[234:241] /*v[490:497]*/
	v_wmma_f32_16x16x32_bf16 v[242:249] /*v[498:505]*/, v[56:63] /*v[312:319]*/, v[152:159], v[242:249] /*v[498:505]*/
	v_wmma_f32_16x16x32_bf16 v[250:257] /*v[506:513]*/, v[56:63] /*v[312:319]*/, v[184:191], v[250:257] /*v[506:513]*/
	s_set_vgpr_msb 0x51a1
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[514:521]*/, v[88:95] /*v[344:351]*/, v[152:159], v[2:9] /*v[514:521]*/
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[522:529]*/, v[88:95] /*v[344:351]*/, v[184:191], v[10:17] /*v[522:529]*/
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[530:537]*/, v[120:127] /*v[376:383]*/, v[152:159], v[18:25] /*v[530:537]*/
	v_wmma_f32_16x16x32_bf16 v[26:33] /*v[538:545]*/, v[120:127] /*v[376:383]*/, v[184:191], v[26:33] /*v[538:545]*/
	v_wmma_f32_16x16x32_bf16 v[34:41] /*v[546:553]*/, v[152:159] /*v[408:415]*/, v[152:159], v[34:41] /*v[546:553]*/
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[554:561]*/, v[152:159] /*v[408:415]*/, v[184:191], v[42:49] /*v[554:561]*/
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[562:569]*/, v[184:191] /*v[440:447]*/, v[152:159], v[50:57] /*v[562:569]*/
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[570:577]*/, v[184:191] /*v[440:447]*/, v[184:191], v[58:65] /*v[570:577]*/
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0xa142
	ds_load_tr16_b128 v[184:187] /*v[440:443]*/, v86 /*v598*/
	ds_load_tr16_b128 v[152:155] /*v[408:411]*/, v86 /*v598*/ offset:32
	ds_load_tr16_b128 v[188:191] /*v[444:447]*/, v86 /*v598*/ offset:4608
	ds_load_tr16_b128 v[156:159] /*v[412:415]*/, v86 /*v598*/ offset:4640
	ds_load_tr16_b128 v[176:179] /*v[432:435]*/, v86 /*v598*/ offset:9216
	ds_load_tr16_b128 v[144:147] /*v[400:403]*/, v86 /*v598*/ offset:9248
	ds_load_tr16_b128 v[180:183] /*v[436:439]*/, v86 /*v598*/ offset:13824
	ds_load_tr16_b128 v[148:151] /*v[404:407]*/, v86 /*v598*/ offset:13856
	ds_load_tr16_b128 v[168:171] /*v[424:427]*/, v86 /*v598*/ offset:18432
	ds_load_tr16_b128 v[136:139] /*v[392:395]*/, v86 /*v598*/ offset:18464
	ds_load_tr16_b128 v[172:175] /*v[428:431]*/, v86 /*v598*/ offset:23040
	ds_load_tr16_b128 v[140:143] /*v[396:399]*/, v86 /*v598*/ offset:23072
	ds_load_tr16_b128 v[160:163] /*v[416:419]*/, v86 /*v598*/ offset:27648
	ds_load_tr16_b128 v[128:131] /*v[384:387]*/, v86 /*v598*/ offset:27680
	ds_load_tr16_b128 v[164:167] /*v[420:423]*/, v86 /*v598*/ offset:32256
	ds_load_tr16_b128 v[132:135] /*v[388:391]*/, v86 /*v598*/ offset:32288
	ds_load_tr16_b128 v[120:123] /*v[376:379]*/, v86 /*v598*/ offset:64
	ds_load_tr16_b128 v[88:91] /*v[344:347]*/, v86 /*v598*/ offset:96
	ds_load_tr16_b128 v[124:127] /*v[380:383]*/, v86 /*v598*/ offset:4672
	ds_load_tr16_b128 v[92:95] /*v[348:351]*/, v86 /*v598*/ offset:4704
	ds_load_tr16_b128 v[112:115] /*v[368:371]*/, v86 /*v598*/ offset:9280
	ds_load_tr16_b128 v[80:83] /*v[336:339]*/, v86 /*v598*/ offset:9312
	ds_load_tr16_b128 v[116:119] /*v[372:375]*/, v86 /*v598*/ offset:13888
	ds_load_tr16_b128 v[84:87] /*v[340:343]*/, v86 /*v598*/ offset:13920
	ds_load_tr16_b128 v[104:107] /*v[360:363]*/, v86 /*v598*/ offset:18496
	ds_load_tr16_b128 v[72:75] /*v[328:331]*/, v86 /*v598*/ offset:18528
	ds_load_tr16_b128 v[108:111] /*v[364:367]*/, v86 /*v598*/ offset:23104
	ds_load_tr16_b128 v[76:79] /*v[332:335]*/, v86 /*v598*/ offset:23136
	ds_load_tr16_b128 v[96:99] /*v[352:355]*/, v86 /*v598*/ offset:27712
	ds_load_tr16_b128 v[64:67] /*v[320:323]*/, v86 /*v598*/ offset:27744
	ds_load_tr16_b128 v[100:103] /*v[356:359]*/, v86 /*v598*/ offset:32320
	ds_load_tr16_b128 v[68:71] /*v[324:327]*/, v86 /*v598*/ offset:32352
	ds_load_tr16_b128 v[56:59] /*v[312:315]*/, v86 /*v598*/ offset:128
	ds_load_tr16_b128 v[24:27] /*v[280:283]*/, v86 /*v598*/ offset:160
	ds_load_tr16_b128 v[60:63] /*v[316:319]*/, v86 /*v598*/ offset:4736
	ds_load_tr16_b128 v[28:31] /*v[284:287]*/, v86 /*v598*/ offset:4768
	ds_load_tr16_b128 v[48:51] /*v[304:307]*/, v86 /*v598*/ offset:9344
	ds_load_tr16_b128 v[16:19] /*v[272:275]*/, v86 /*v598*/ offset:9376
	ds_load_tr16_b128 v[52:55] /*v[308:311]*/, v86 /*v598*/ offset:13952
	ds_load_tr16_b128 v[20:23] /*v[276:279]*/, v86 /*v598*/ offset:13984
	ds_load_tr16_b128 v[40:43] /*v[296:299]*/, v86 /*v598*/ offset:18560
	ds_load_tr16_b128 v[8:11] /*v[264:267]*/, v86 /*v598*/ offset:18592
	ds_load_tr16_b128 v[44:47] /*v[300:303]*/, v86 /*v598*/ offset:23168
	ds_load_tr16_b128 v[12:15] /*v[268:271]*/, v86 /*v598*/ offset:23200
	ds_load_tr16_b128 v[32:35] /*v[288:291]*/, v86 /*v598*/ offset:27776
	ds_load_tr16_b128 v[0:3] /*v[256:259]*/, v86 /*v598*/ offset:27808
	ds_load_tr16_b128 v[36:39] /*v[292:295]*/, v86 /*v598*/ offset:32384
	ds_load_tr16_b128 v[4:7] /*v[260:263]*/, v86 /*v598*/ offset:32416
	s_set_vgpr_msb 0x4202
	ds_load_tr16_b128 v[248:251], v86 /*v598*/ offset:192
	ds_load_tr16_b128 v[216:219], v86 /*v598*/ offset:224
	ds_load_tr16_b128 v[252:255], v86 /*v598*/ offset:4800
	ds_load_tr16_b128 v[220:223], v86 /*v598*/ offset:4832
	ds_load_tr16_b128 v[240:243], v86 /*v598*/ offset:9408
	ds_load_tr16_b128 v[208:211], v86 /*v598*/ offset:9440
	ds_load_tr16_b128 v[244:247], v86 /*v598*/ offset:14016
	ds_load_tr16_b128 v[212:215], v86 /*v598*/ offset:14048
	ds_load_tr16_b128 v[232:235], v86 /*v598*/ offset:18624
	ds_load_tr16_b128 v[200:203], v86 /*v598*/ offset:18656
	ds_load_tr16_b128 v[236:239], v86 /*v598*/ offset:23232
	ds_load_tr16_b128 v[204:207], v86 /*v598*/ offset:23264
	ds_load_tr16_b128 v[224:227], v86 /*v598*/ offset:27840
	ds_load_tr16_b128 v[192:195], v86 /*v598*/ offset:27872
	ds_load_tr16_b128 v[228:231], v86 /*v598*/ offset:32448
	ds_load_tr16_b128 v[196:199], v86 /*v598*/ offset:32480
	s_set_vgpr_msb 0x2aa
	v_lshl_or_b32 v68 /*v580*/, s52, 7, v81 /*v593*/
	v_cmp_le_i32_e32 vcc_lo, v68 /*v580*/, v90 /*v602*/
	v_dual_add_nc_u32 v121 /*v633*/, 17, v68 /*v580*/ :: v_dual_bitop2_b32 v69 /*v581*/, 2, v68 /*v580*/ bitop3:0x54
	v_dual_add_nc_u32 v122 /*v634*/, 18, v68 /*v580*/ :: v_dual_bitop2_b32 v70 /*v582*/, 3, v68 /*v580*/ bitop3:0x54
	s_set_vgpr_msb 0xaa44
	v_cndmask_b32_e32 v194 /*v450*/, 0xff800000, v194 /*v450*/, vcc_lo
	s_set_vgpr_msb 0x448a
	v_cmp_lt_i32_e32 vcc_lo, v68 /*v580*/, v90 /*v602*/
	v_dual_add_nc_u32 v123 /*v635*/, 19, v68 /*v580*/ :: v_dual_bitop2_b32 v71 /*v583*/, 4, v68 /*v580*/ bitop3:0x54
	v_dual_add_nc_u32 v126 /*v638*/, 22, v68 /*v580*/ :: v_dual_bitop2_b32 v117 /*v629*/, 5, v68 /*v580*/ bitop3:0x54
	s_set_vgpr_msb 0x8a44
	v_cndmask_b32_e32 v195 /*v451*/, 0xff800000, v195 /*v451*/, vcc_lo
	s_set_vgpr_msb 0x448a
	v_cmp_le_i32_e32 vcc_lo, v69 /*v581*/, v90 /*v602*/
	v_dual_add_nc_u32 v127 /*v639*/, 23, v68 /*v580*/ :: v_dual_bitop2_b32 v118 /*v630*/, 6, v68 /*v580*/ bitop3:0x54
	v_dual_add_nc_u32 v137 /*v649*/, 49, v68 /*v580*/ :: v_dual_bitop2_b32 v119 /*v631*/, 7, v68 /*v580*/ bitop3:0x54
	s_set_vgpr_msb 0x8a44
	v_cndmask_b32_e32 v196 /*v452*/, 0xff800000, v196 /*v452*/, vcc_lo
	s_set_vgpr_msb 0x448a
	v_cmp_le_i32_e32 vcc_lo, v70 /*v582*/, v90 /*v602*/
	v_dual_add_nc_u32 v138 /*v650*/, 50, v68 /*v580*/ :: v_dual_bitop2_b32 v120 /*v632*/, 16, v68 /*v580*/ bitop3:0x54
	v_dual_add_nc_u32 v139 /*v651*/, 51, v68 /*v580*/ :: v_dual_bitop2_b32 v128 /*v640*/, 32, v68 /*v580*/ bitop3:0x54
	s_set_vgpr_msb 0x8a44
	v_cndmask_b32_e32 v197 /*v453*/, 0xff800000, v197 /*v453*/, vcc_lo
	s_set_vgpr_msb 0x448a
	v_cmp_le_i32_e32 vcc_lo, v71 /*v583*/, v90 /*v602*/
	v_dual_add_nc_u32 v140 /*v652*/, 52, v68 /*v580*/ :: v_dual_bitop2_b32 v129 /*v641*/, 33, v68 /*v580*/ bitop3:0x54
	v_dual_add_nc_u32 v141 /*v653*/, 53, v68 /*v580*/ :: v_dual_bitop2_b32 v130 /*v642*/, 34, v68 /*v580*/ bitop3:0x54
	s_set_vgpr_msb 0x8a44
	v_cndmask_b32_e32 v198 /*v454*/, 0xff800000, v198 /*v454*/, vcc_lo
	s_set_vgpr_msb 0x448a
	v_cmp_le_i32_e32 vcc_lo, v117 /*v629*/, v90 /*v602*/
	v_dual_add_nc_u32 v142 /*v654*/, 54, v68 /*v580*/ :: v_dual_bitop2_b32 v131 /*v643*/, 35, v68 /*v580*/ bitop3:0x54
	v_or_b32_e32 v132 /*v644*/, 36, v68 /*v580*/
	v_or_b32_e32 v133 /*v645*/, 37, v68 /*v580*/
	s_set_vgpr_msb 0x8a44
	v_cndmask_b32_e32 v199 /*v455*/, 0xff800000, v199 /*v455*/, vcc_lo
	s_set_vgpr_msb 0x448a
	v_cmp_le_i32_e32 vcc_lo, v118 /*v630*/, v90 /*v602*/
	v_or_b32_e32 v134 /*v646*/, 38, v68 /*v580*/
	v_or_b32_e32 v135 /*v647*/, 39, v68 /*v580*/
	v_or_b32_e32 v136 /*v648*/, 48, v68 /*v580*/
	v_or_b32_e32 v145 /*v657*/, 0x41, v68 /*v580*/
	s_set_vgpr_msb 0x8a44
	v_cndmask_b32_e32 v200 /*v456*/, 0xff800000, v200 /*v456*/, vcc_lo
	s_set_vgpr_msb 0x448a
	v_cmp_le_i32_e32 vcc_lo, v119 /*v631*/, v90 /*v602*/
	v_or_b32_e32 v146 /*v658*/, 0x42, v68 /*v580*/
	v_or_b32_e32 v149 /*v661*/, 0x45, v68 /*v580*/
	v_or_b32_e32 v150 /*v662*/, 0x46, v68 /*v580*/
	v_add_nc_u32_e32 v153 /*v665*/, 0x51, v68 /*v580*/
	s_set_vgpr_msb 0x8a44
	v_cndmask_b32_e32 v201 /*v457*/, 0xff800000, v201 /*v457*/, vcc_lo
	s_set_vgpr_msb 0x448a
	v_cmp_le_i32_e32 vcc_lo, v120 /*v632*/, v90 /*v602*/
	v_add_nc_u32_e32 v154 /*v666*/, 0x52, v68 /*v580*/
	v_add_nc_u32_e32 v157 /*v669*/, 0x55, v68 /*v580*/
	v_add_nc_u32_e32 v158 /*v670*/, 0x56, v68 /*v580*/
	v_or_b32_e32 v161 /*v673*/, 0x61, v68 /*v580*/
	s_set_vgpr_msb 0x8a44
	v_cndmask_b32_e32 v210 /*v466*/, 0xff800000, v210 /*v466*/, vcc_lo
	s_set_vgpr_msb 0x448a
	v_cmp_le_i32_e32 vcc_lo, v121 /*v633*/, v90 /*v602*/
	v_or_b32_e32 v162 /*v674*/, 0x62, v68 /*v580*/
	v_or_b32_e32 v163 /*v675*/, 0x63, v68 /*v580*/
	v_or_b32_e32 v166 /*v678*/, 0x66, v68 /*v580*/
	v_or_b32_e32 v167 /*v679*/, 0x67, v68 /*v580*/
	s_set_vgpr_msb 0x8a44
	v_cndmask_b32_e32 v211 /*v467*/, 0xff800000, v211 /*v467*/, vcc_lo
	s_set_vgpr_msb 0x448a
	v_cmp_le_i32_e32 vcc_lo, v122 /*v634*/, v90 /*v602*/
	v_add_nc_u32_e32 v170 /*v682*/, 0x72, v68 /*v580*/
	v_add_nc_u32_e32 v175 /*v687*/, 0x77, v68 /*v580*/
	s_set_vgpr_msb 0x8a84
	v_cndmask_b32_e32 v66 /*v578*/, 0xff800000, v212 /*v468*/, vcc_lo
	s_set_vgpr_msb 0x844a
	v_add_nc_u32_e32 v212 /*v468*/, 20, v68 /*v580*/
	v_cmp_le_i32_e32 vcc_lo, v123 /*v635*/, v90 /*v602*/
	s_set_vgpr_msb 0x4a84
	v_cndmask_b32_e32 v67 /*v579*/, 0xff800000, v213 /*v469*/, vcc_lo
	s_set_vgpr_msb 0x8449
	v_add_nc_u32_e32 v213 /*v469*/, 21, v68 /*v580*/
	v_cmp_le_i32_e32 vcc_lo, v212 /*v468*/, v90 /*v602*/
	s_set_vgpr_msb 0x4944
	v_cndmask_b32_e32 v214 /*v470*/, 0xff800000, v214 /*v470*/, vcc_lo
	s_set_vgpr_msb 0x4409
	v_cmp_le_i32_e32 vcc_lo, v213 /*v469*/, v90 /*v602*/
	s_set_vgpr_msb 0x944
	v_cndmask_b32_e32 v215 /*v471*/, 0xff800000, v215 /*v471*/, vcc_lo
	s_set_vgpr_msb 0x440a
	v_cmp_le_i32_e32 vcc_lo, v126 /*v638*/, v90 /*v602*/
	s_set_vgpr_msb 0xa44
	v_cndmask_b32_e32 v216 /*v472*/, 0xff800000, v216 /*v472*/, vcc_lo
	s_set_vgpr_msb 0x440a
	v_cmp_le_i32_e32 vcc_lo, v127 /*v639*/, v90 /*v602*/
	s_set_vgpr_msb 0xa44
	v_cndmask_b32_e32 v217 /*v473*/, 0xff800000, v217 /*v473*/, vcc_lo
	s_set_vgpr_msb 0x440a
	v_cmp_le_i32_e32 vcc_lo, v128 /*v640*/, v90 /*v602*/
	s_set_vgpr_msb 0xa44
	v_cndmask_b32_e32 v226 /*v482*/, 0xff800000, v226 /*v482*/, vcc_lo
	s_set_vgpr_msb 0x440a
	v_cmp_le_i32_e32 vcc_lo, v129 /*v641*/, v90 /*v602*/
	s_set_vgpr_msb 0xa44
	v_cndmask_b32_e32 v227 /*v483*/, 0xff800000, v227 /*v483*/, vcc_lo
	s_set_vgpr_msb 0x440a
	v_cmp_le_i32_e32 vcc_lo, v130 /*v642*/, v90 /*v602*/
	s_set_vgpr_msb 0xa44
	v_cndmask_b32_e32 v228 /*v484*/, 0xff800000, v228 /*v484*/, vcc_lo
	s_set_vgpr_msb 0x440a
	v_cmp_le_i32_e32 vcc_lo, v131 /*v643*/, v90 /*v602*/
	s_set_vgpr_msb 0xa44
	v_cndmask_b32_e32 v229 /*v485*/, 0xff800000, v229 /*v485*/, vcc_lo
	s_set_vgpr_msb 0x440a
	v_cmp_le_i32_e32 vcc_lo, v132 /*v644*/, v90 /*v602*/
	s_set_vgpr_msb 0xa44
	v_cndmask_b32_e32 v230 /*v486*/, 0xff800000, v230 /*v486*/, vcc_lo
	s_set_vgpr_msb 0x440a
	v_cmp_le_i32_e32 vcc_lo, v133 /*v645*/, v90 /*v602*/
	s_set_vgpr_msb 0xa44
	v_cndmask_b32_e32 v231 /*v487*/, 0xff800000, v231 /*v487*/, vcc_lo
	s_set_vgpr_msb 0x440a
	v_cmp_le_i32_e32 vcc_lo, v134 /*v646*/, v90 /*v602*/
	s_set_vgpr_msb 0xa44
	v_cndmask_b32_e32 v232 /*v488*/, 0xff800000, v232 /*v488*/, vcc_lo
	s_set_vgpr_msb 0x440a
	v_cmp_le_i32_e32 vcc_lo, v135 /*v647*/, v90 /*v602*/
	s_set_vgpr_msb 0xa44
	v_cndmask_b32_e32 v233 /*v489*/, 0xff800000, v233 /*v489*/, vcc_lo
	s_set_vgpr_msb 0x440a
	v_cmp_le_i32_e32 vcc_lo, v136 /*v648*/, v90 /*v602*/
	s_set_vgpr_msb 0xa44
	v_cndmask_b32_e32 v242 /*v498*/, 0xff800000, v242 /*v498*/, vcc_lo
	s_set_vgpr_msb 0x440a
	v_cmp_le_i32_e32 vcc_lo, v137 /*v649*/, v90 /*v602*/
	s_set_vgpr_msb 0xa44
	v_cndmask_b32_e32 v243 /*v499*/, 0xff800000, v243 /*v499*/, vcc_lo
	s_set_vgpr_msb 0x440a
	v_cmp_le_i32_e32 vcc_lo, v138 /*v650*/, v90 /*v602*/
	s_set_vgpr_msb 0xa44
	v_cndmask_b32_e32 v244 /*v500*/, 0xff800000, v244 /*v500*/, vcc_lo
	s_set_vgpr_msb 0x440a
	v_cmp_le_i32_e32 vcc_lo, v139 /*v651*/, v90 /*v602*/
	s_set_vgpr_msb 0xa44
	v_cndmask_b32_e32 v245 /*v501*/, 0xff800000, v245 /*v501*/, vcc_lo
	s_set_vgpr_msb 0x440a
	v_cmp_le_i32_e32 vcc_lo, v140 /*v652*/, v90 /*v602*/
	s_wait_alu depctr_vm_vsrc(6)
	s_set_vgpr_msb 0xa84
	v_cndmask_b32_e32 v94 /*v606*/, 0xff800000, v246 /*v502*/, vcc_lo
	s_set_vgpr_msb 0x844a
	v_cmp_le_i32_e32 vcc_lo, v141 /*v653*/, v90 /*v602*/
	v_add_nc_u32_e32 v246 /*v502*/, 55, v68 /*v580*/
	s_set_vgpr_msb 0x4a84
	v_cndmask_b32_e32 v95 /*v607*/, 0xff800000, v247 /*v503*/, vcc_lo
	s_set_vgpr_msb 0x844a
	v_cmp_le_i32_e32 vcc_lo, v142 /*v654*/, v90 /*v602*/
	v_or_b32_e32 v247 /*v503*/, 64, v68 /*v580*/
	s_set_vgpr_msb 0x4a44
	v_cndmask_b32_e32 v248 /*v504*/, 0xff800000, v248 /*v504*/, vcc_lo
	s_set_vgpr_msb 0x4409
	v_cmp_le_i32_e32 vcc_lo, v246 /*v502*/, v90 /*v602*/
	s_set_vgpr_msb 0x944
	v_cndmask_b32_e32 v249 /*v505*/, 0xff800000, v249 /*v505*/, vcc_lo
	s_set_vgpr_msb 0x4489
	v_cmp_le_i32_e32 vcc_lo, v247 /*v503*/, v90 /*v602*/
	v_cndmask_b32_e32 v96 /*v608*/, 0xff800000, v2 /*v514*/, vcc_lo
	s_set_vgpr_msb 0x898a
	v_cmp_le_i32_e32 vcc_lo, v145 /*v657*/, v90 /*v602*/
	v_or_b32_e32 v2 /*v514*/, 0x43, v68 /*v580*/
	v_cndmask_b32_e32 v97 /*v609*/, 0xff800000, v3 /*v515*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v146 /*v658*/, v90 /*v602*/
	v_or_b32_e32 v3 /*v515*/, 0x44, v68 /*v580*/
	v_cndmask_b32_e32 v4 /*v516*/, 0xff800000, v4 /*v516*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v2 /*v514*/, v90 /*v602*/
	v_cndmask_b32_e32 v5 /*v517*/, 0xff800000, v5 /*v517*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v3 /*v515*/, v90 /*v602*/
	v_cndmask_b32_e32 v98 /*v610*/, 0xff800000, v6 /*v518*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v149 /*v661*/, v90 /*v602*/
	v_or_b32_e32 v6 /*v518*/, 0x47, v68 /*v580*/
	v_cndmask_b32_e32 v99 /*v611*/, 0xff800000, v7 /*v519*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v150 /*v662*/, v90 /*v602*/
	v_or_b32_e32 v7 /*v519*/, 0x50, v68 /*v580*/
	v_cndmask_b32_e32 v8 /*v520*/, 0xff800000, v8 /*v520*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v6 /*v518*/, v90 /*v602*/
	v_cndmask_b32_e32 v9 /*v521*/, 0xff800000, v9 /*v521*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v7 /*v519*/, v90 /*v602*/
	v_cndmask_b32_e32 v100 /*v612*/, 0xff800000, v18 /*v530*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v153 /*v665*/, v90 /*v602*/
	v_add_nc_u32_e32 v18 /*v530*/, 0x53, v68 /*v580*/
	v_cndmask_b32_e32 v101 /*v613*/, 0xff800000, v19 /*v531*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v154 /*v666*/, v90 /*v602*/
	v_add_nc_u32_e32 v19 /*v531*/, 0x54, v68 /*v580*/
	v_cndmask_b32_e32 v20 /*v532*/, 0xff800000, v20 /*v532*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v18 /*v530*/, v90 /*v602*/
	v_cndmask_b32_e32 v21 /*v533*/, 0xff800000, v21 /*v533*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v19 /*v531*/, v90 /*v602*/
	v_cndmask_b32_e32 v102 /*v614*/, 0xff800000, v22 /*v534*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v157 /*v669*/, v90 /*v602*/
	v_add_nc_u32_e32 v22 /*v534*/, 0x57, v68 /*v580*/
	v_cndmask_b32_e32 v103 /*v615*/, 0xff800000, v23 /*v535*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v158 /*v670*/, v90 /*v602*/
	v_or_b32_e32 v23 /*v535*/, 0x60, v68 /*v580*/
	v_cndmask_b32_e32 v24 /*v536*/, 0xff800000, v24 /*v536*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v22 /*v534*/, v90 /*v602*/
	v_cndmask_b32_e32 v25 /*v537*/, 0xff800000, v25 /*v537*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v23 /*v535*/, v90 /*v602*/
	v_cndmask_b32_e32 v34 /*v546*/, 0xff800000, v34 /*v546*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v161 /*v673*/, v90 /*v602*/
	v_cndmask_b32_e32 v35 /*v547*/, 0xff800000, v35 /*v547*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v162 /*v674*/, v90 /*v602*/
	v_cndmask_b32_e32 v104 /*v616*/, 0xff800000, v36 /*v548*/, vcc_lo
	v_or_b32_e32 v36 /*v548*/, 0x64, v68 /*v580*/
	v_cmp_le_i32_e32 vcc_lo, v163 /*v675*/, v90 /*v602*/
	v_cndmask_b32_e32 v105 /*v617*/, 0xff800000, v37 /*v549*/, vcc_lo
	v_or_b32_e32 v37 /*v549*/, 0x65, v68 /*v580*/
	v_cmp_le_i32_e32 vcc_lo, v36 /*v548*/, v90 /*v602*/
	v_cndmask_b32_e32 v38 /*v550*/, 0xff800000, v38 /*v550*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v37 /*v549*/, v90 /*v602*/
	v_cndmask_b32_e32 v39 /*v551*/, 0xff800000, v39 /*v551*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v166 /*v678*/, v90 /*v602*/
	v_cndmask_b32_e32 v106 /*v618*/, 0xff800000, v40 /*v552*/, vcc_lo
	v_or_b32_e32 v40 /*v552*/, 0x70, v68 /*v580*/
	v_cmp_le_i32_e32 vcc_lo, v167 /*v679*/, v90 /*v602*/
	v_cndmask_b32_e32 v107 /*v619*/, 0xff800000, v41 /*v553*/, vcc_lo
	v_add_nc_u32_e32 v41 /*v553*/, 0x71, v68 /*v580*/
	v_cmp_le_i32_e32 vcc_lo, v40 /*v552*/, v90 /*v602*/
	v_cndmask_b32_e32 v108 /*v620*/, 0xff800000, v50 /*v562*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v41 /*v553*/, v90 /*v602*/
	v_add_nc_u32_e32 v50 /*v562*/, 0x73, v68 /*v580*/
	v_cndmask_b32_e32 v109 /*v621*/, 0xff800000, v51 /*v563*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v170 /*v682*/, v90 /*v602*/
	v_add_nc_u32_e32 v51 /*v563*/, 0x74, v68 /*v580*/
	v_cndmask_b32_e32 v110 /*v622*/, 0xff800000, v52 /*v564*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v50 /*v562*/, v90 /*v602*/
	v_add_nc_u32_e32 v52 /*v564*/, 0x75, v68 /*v580*/
	v_cndmask_b32_e32 v111 /*v623*/, 0xff800000, v53 /*v565*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v51 /*v563*/, v90 /*v602*/
	v_add_nc_u32_e32 v53 /*v565*/, 0x76, v68 /*v580*/
	v_cndmask_b32_e32 v54 /*v566*/, 0xff800000, v54 /*v566*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v52 /*v564*/, v90 /*v602*/
	v_cndmask_b32_e32 v55 /*v567*/, 0xff800000, v55 /*v567*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v53 /*v565*/, v90 /*v602*/
	v_cndmask_b32_e32 v56 /*v568*/, 0xff800000, v56 /*v568*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v175 /*v687*/, v90 /*v602*/
	v_cndmask_b32_e32 v57 /*v569*/, 0xff800000, v57 /*v569*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v68 /*v580*/, v91 /*v603*/
	s_set_vgpr_msb 0x8a84
	v_cndmask_b32_e32 v112 /*v624*/, 0xff800000, v202 /*v458*/, vcc_lo
	s_set_vgpr_msb 0x840a
	v_cmp_lt_i32_e32 vcc_lo, v68 /*v580*/, v91 /*v603*/
	s_set_vgpr_msb 0xa55
	v_max3_num_f32 v202 /*v458*/, v194 /*v450*/, v195 /*v451*/, v196 /*v452*/
	s_set_vgpr_msb 0x5584
	v_cndmask_b32_e32 v113 /*v625*/, 0xff800000, v203 /*v459*/, vcc_lo
	s_set_vgpr_msb 0x840a
	v_cmp_le_i32_e32 vcc_lo, v69 /*v581*/, v91 /*v603*/
	s_set_vgpr_msb 0xa84
	v_cndmask_b32_e32 v114 /*v626*/, 0xff800000, v204 /*v460*/, vcc_lo
	s_set_vgpr_msb 0x840a
	v_cmp_le_i32_e32 vcc_lo, v70 /*v582*/, v91 /*v603*/
	s_set_vgpr_msb 0xa55
	v_max3_num_f32 v204 /*v460*/, v197 /*v453*/, v198 /*v454*/, v199 /*v455*/
	s_set_vgpr_msb 0x5584
	v_cndmask_b32_e32 v115 /*v627*/, 0xff800000, v205 /*v461*/, vcc_lo
	s_set_vgpr_msb 0x846a
	v_cmp_le_i32_e32 vcc_lo, v71 /*v583*/, v91 /*v603*/
	v_max3_num_f32 v203 /*v459*/, v112 /*v624*/, v113 /*v625*/, v114 /*v626*/
	s_set_vgpr_msb 0x6a84
	v_cndmask_b32_e32 v116 /*v628*/, 0xff800000, v206 /*v462*/, vcc_lo
	s_set_vgpr_msb 0x840a
	v_cmp_le_i32_e32 vcc_lo, v117 /*v629*/, v91 /*v603*/
	s_set_vgpr_msb 0xa55
	v_max3_num_f32 v206 /*v462*/, v200 /*v456*/, v201 /*v457*/, v210 /*v466*/
	s_set_vgpr_msb 0x5584
	v_cndmask_b32_e32 v117 /*v629*/, 0xff800000, v207 /*v463*/, vcc_lo
	s_set_vgpr_msb 0x840a
	v_cmp_le_i32_e32 vcc_lo, v118 /*v630*/, v91 /*v603*/
	s_set_vgpr_msb 0xa55
	v_max3_num_f32 v202 /*v458*/, v202 /*v458*/, v204 /*v460*/, v206 /*v462*/
	s_set_vgpr_msb 0x556a
	v_max3_num_f32 v205 /*v461*/, v115 /*v627*/, v116 /*v628*/, v117 /*v629*/
	s_set_vgpr_msb 0x6a84
	v_cndmask_b32_e32 v118 /*v630*/, 0xff800000, v208 /*v464*/, vcc_lo
	s_set_vgpr_msb 0x840a
	v_cmp_le_i32_e32 vcc_lo, v119 /*v631*/, v91 /*v603*/
	s_set_vgpr_msb 0xa69
	v_max3_num_f32 v208 /*v464*/, v211 /*v467*/, v66 /*v578*/, v67 /*v579*/
	s_set_vgpr_msb 0x6984
	v_cndmask_b32_e32 v119 /*v631*/, 0xff800000, v209 /*v465*/, vcc_lo
	s_set_vgpr_msb 0x840a
	v_cmp_le_i32_e32 vcc_lo, v120 /*v632*/, v91 /*v603*/
	s_set_vgpr_msb 0xa84
	v_cndmask_b32_e32 v120 /*v632*/, 0xff800000, v218 /*v474*/, vcc_lo
	s_set_vgpr_msb 0x840a
	v_cmp_le_i32_e32 vcc_lo, v121 /*v633*/, v91 /*v603*/
	s_set_vgpr_msb 0xa55
	v_max3_num_f32 v218 /*v474*/, v217 /*v473*/, v226 /*v482*/, v227 /*v483*/
	s_set_vgpr_msb 0x5584
	v_cndmask_b32_e32 v121 /*v633*/, 0xff800000, v219 /*v475*/, vcc_lo
	s_set_vgpr_msb 0x846a
	v_cmp_le_i32_e32 vcc_lo, v122 /*v634*/, v91 /*v603*/
	v_max3_num_f32 v207 /*v463*/, v118 /*v630*/, v119 /*v631*/, v120 /*v632*/
	s_set_vgpr_msb 0x6a84
	v_cndmask_b32_e32 v122 /*v634*/, 0xff800000, v220 /*v476*/, vcc_lo
	s_set_vgpr_msb 0x840a
	v_cmp_le_i32_e32 vcc_lo, v123 /*v635*/, v91 /*v603*/
	s_set_vgpr_msb 0xa55
	v_max3_num_f32 v220 /*v476*/, v228 /*v484*/, v229 /*v485*/, v230 /*v486*/
	v_max3_num_f32 v203 /*v459*/, v203 /*v459*/, v205 /*v461*/, v207 /*v463*/
	s_set_vgpr_msb 0x5584
	v_cndmask_b32_e32 v123 /*v635*/, 0xff800000, v221 /*v477*/, vcc_lo
	s_set_vgpr_msb 0x8409
	v_cmp_le_i32_e32 vcc_lo, v212 /*v468*/, v91 /*v603*/
	s_set_vgpr_msb 0x955
	v_max3_num_f32 v212 /*v468*/, v214 /*v470*/, v215 /*v471*/, v216 /*v472*/
	s_set_vgpr_msb 0x556a
	v_max3_num_f32 v209 /*v465*/, v121 /*v633*/, v122 /*v634*/, v123 /*v635*/
	s_set_vgpr_msb 0x6a84
	v_cndmask_b32_e32 v124 /*v636*/, 0xff800000, v222 /*v478*/, vcc_lo
	s_set_vgpr_msb 0x8409
	v_cmp_le_i32_e32 vcc_lo, v213 /*v469*/, v91 /*v603*/
	s_set_vgpr_msb 0x955
	v_max3_num_f32 v222 /*v478*/, v231 /*v487*/, v232 /*v488*/, v233 /*v489*/
	v_max3_num_f32 v204 /*v460*/, v208 /*v464*/, v212 /*v468*/, v218 /*v474*/
	s_set_vgpr_msb 0x5584
	v_cndmask_b32_e32 v125 /*v637*/, 0xff800000, v223 /*v479*/, vcc_lo
	s_set_vgpr_msb 0x840a
	v_cmp_le_i32_e32 vcc_lo, v126 /*v638*/, v91 /*v603*/
	s_set_vgpr_msb 0xa84
	v_cndmask_b32_e32 v126 /*v638*/, 0xff800000, v224 /*v480*/, vcc_lo
	s_set_vgpr_msb 0x840a
	v_cmp_le_i32_e32 vcc_lo, v127 /*v639*/, v91 /*v603*/
	s_set_vgpr_msb 0xa55
	v_max3_num_f32 v224 /*v480*/, v242 /*v498*/, v243 /*v499*/, v244 /*v500*/
	s_set_vgpr_msb 0x5584
	v_cndmask_b32_e32 v127 /*v639*/, 0xff800000, v225 /*v481*/, vcc_lo
	s_set_vgpr_msb 0x846a
	v_cmp_le_i32_e32 vcc_lo, v128 /*v640*/, v91 /*v603*/
	v_max3_num_f32 v213 /*v469*/, v124 /*v636*/, v125 /*v637*/, v126 /*v638*/
	s_set_vgpr_msb 0x6a55
	v_max3_num_f32 v205 /*v461*/, v220 /*v476*/, v222 /*v478*/, v224 /*v480*/
	s_set_vgpr_msb 0x5584
	v_cndmask_b32_e32 v128 /*v640*/, 0xff800000, v234 /*v490*/, vcc_lo
	s_set_vgpr_msb 0x840a
	v_cmp_le_i32_e32 vcc_lo, v129 /*v641*/, v91 /*v603*/
	s_set_vgpr_msb 0xa69
	v_max3_num_f32 v234 /*v490*/, v245 /*v501*/, v94 /*v606*/, v95 /*v607*/
	s_set_vgpr_msb 0x6955
	v_max3_num_f32 v202 /*v458*/, v202 /*v458*/, v204 /*v460*/, v205 /*v461*/
	s_set_vgpr_msb 0x5584
	v_cndmask_b32_e32 v129 /*v641*/, 0xff800000, v235 /*v491*/, vcc_lo
	s_set_vgpr_msb 0x846a
	v_cmp_le_i32_e32 vcc_lo, v130 /*v642*/, v91 /*v603*/
	v_max3_num_f32 v219 /*v475*/, v127 /*v639*/, v128 /*v640*/, v129 /*v641*/
	s_set_vgpr_msb 0x6a84
	v_cndmask_b32_e32 v130 /*v642*/, 0xff800000, v236 /*v492*/, vcc_lo
	s_set_vgpr_msb 0x840a
	v_cmp_le_i32_e32 vcc_lo, v131 /*v643*/, v91 /*v603*/
	s_set_vgpr_msb 0xa65
	v_max3_num_f32 v236 /*v492*/, v248 /*v504*/, v249 /*v505*/, v96 /*v608*/
	s_set_vgpr_msb 0x6555
	v_max3_num_f32 v209 /*v465*/, v209 /*v465*/, v213 /*v469*/, v219 /*v475*/
	s_set_vgpr_msb 0x5584
	v_cndmask_b32_e32 v131 /*v643*/, 0xff800000, v237 /*v493*/, vcc_lo
	s_set_vgpr_msb 0x840a
	v_cmp_le_i32_e32 vcc_lo, v132 /*v644*/, v91 /*v603*/
	s_set_vgpr_msb 0xa84
	v_cndmask_b32_e32 v132 /*v644*/, 0xff800000, v238 /*v494*/, vcc_lo
	s_set_vgpr_msb 0x846a
	v_cmp_le_i32_e32 vcc_lo, v133 /*v645*/, v91 /*v603*/
	v_max3_num_f32 v238 /*v494*/, v97 /*v609*/, v4 /*v516*/, v5 /*v517*/
	s_set_vgpr_msb 0x6a84
	v_cndmask_b32_e32 v133 /*v645*/, 0xff800000, v239 /*v495*/, vcc_lo
	s_set_vgpr_msb 0x846a
	v_cmp_le_i32_e32 vcc_lo, v134 /*v646*/, v91 /*v603*/
	v_max3_num_f32 v221 /*v477*/, v130 /*v642*/, v131 /*v643*/, v132 /*v644*/
	s_set_vgpr_msb 0x6a55
	v_max3_num_f32 v206 /*v462*/, v234 /*v490*/, v236 /*v492*/, v238 /*v494*/
	s_set_vgpr_msb 0x5584
	v_cndmask_b32_e32 v134 /*v646*/, 0xff800000, v240 /*v496*/, vcc_lo
	s_set_vgpr_msb 0x846a
	v_cmp_le_i32_e32 vcc_lo, v135 /*v647*/, v91 /*v603*/
	v_max3_num_f32 v240 /*v496*/, v98 /*v610*/, v99 /*v611*/, v8 /*v520*/
	s_set_vgpr_msb 0x6a84
	v_cndmask_b32_e32 v135 /*v647*/, 0xff800000, v241 /*v497*/, vcc_lo
	s_set_vgpr_msb 0x846a
	v_cmp_le_i32_e32 vcc_lo, v136 /*v648*/, v91 /*v603*/
	v_max3_num_f32 v223 /*v479*/, v133 /*v645*/, v134 /*v646*/, v135 /*v647*/
	s_set_vgpr_msb 0x6a84
	v_cndmask_b32_e32 v136 /*v648*/, 0xff800000, v250 /*v506*/, vcc_lo
	s_set_vgpr_msb 0x846a
	v_cmp_le_i32_e32 vcc_lo, v137 /*v649*/, v91 /*v603*/
	v_max3_num_f32 v250 /*v506*/, v20 /*v532*/, v21 /*v533*/, v102 /*v614*/
	s_set_vgpr_msb 0x6a84
	v_cndmask_b32_e32 v137 /*v649*/, 0xff800000, v251 /*v507*/, vcc_lo
	s_set_vgpr_msb 0x840a
	v_cmp_le_i32_e32 vcc_lo, v138 /*v650*/, v91 /*v603*/
	s_set_vgpr_msb 0xa84
	v_cndmask_b32_e32 v138 /*v650*/, 0xff800000, v252 /*v508*/, vcc_lo
	s_set_vgpr_msb 0x846a
	v_cmp_le_i32_e32 vcc_lo, v139 /*v651*/, v91 /*v603*/
	v_max3_num_f32 v252 /*v508*/, v103 /*v615*/, v24 /*v536*/, v25 /*v537*/
	s_set_vgpr_msb 0x6a84
	v_cndmask_b32_e32 v139 /*v651*/, 0xff800000, v253 /*v509*/, vcc_lo
	s_set_vgpr_msb 0x846a
	v_cmp_le_i32_e32 vcc_lo, v140 /*v652*/, v91 /*v603*/
	v_max3_num_f32 v225 /*v481*/, v136 /*v648*/, v137 /*v649*/, v138 /*v650*/
	s_set_vgpr_msb 0x6a84
	v_cndmask_b32_e32 v140 /*v652*/, 0xff800000, v254 /*v510*/, vcc_lo
	s_set_vgpr_msb 0x846a
	v_cmp_le_i32_e32 vcc_lo, v141 /*v653*/, v91 /*v603*/
	v_max3_num_f32 v254 /*v510*/, v34 /*v546*/, v35 /*v547*/, v104 /*v616*/
	s_set_vgpr_msb 0x6a55
	v_max3_num_f32 v213 /*v469*/, v221 /*v477*/, v223 /*v479*/, v225 /*v481*/
	s_set_vgpr_msb 0x5584
	v_cndmask_b32_e32 v141 /*v653*/, 0xff800000, v255 /*v511*/, vcc_lo
	s_set_vgpr_msb 0x840a
	v_cmp_le_i32_e32 vcc_lo, v142 /*v654*/, v91 /*v603*/
	s_set_vgpr_msb 0xa55
	v_max3_num_f32 v203 /*v459*/, v203 /*v459*/, v209 /*v465*/, v213 /*v469*/
	s_set_vgpr_msb 0x556a
	v_max3_num_f32 v235 /*v491*/, v139 /*v651*/, v140 /*v652*/, v141 /*v653*/
	s_set_vgpr_msb 0x6a89
	v_cndmask_b32_e32 v142 /*v654*/, 0xff800000, v0 /*v512*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v246 /*v502*/, v91 /*v603*/
	s_set_vgpr_msb 0x896a
	v_max3_num_f32 v246 /*v502*/, v9 /*v521*/, v100 /*v612*/, v101 /*v613*/
	s_set_vgpr_msb 0x6aaa
	v_max3_num_f32 v0 /*v512*/, v105 /*v617*/, v38 /*v550*/, v39 /*v551*/
	v_cndmask_b32_e32 v143 /*v655*/, 0xff800000, v1 /*v513*/, vcc_lo
	s_set_vgpr_msb 0xaa09
	v_cmp_le_i32_e32 vcc_lo, v247 /*v503*/, v91 /*v603*/
	s_set_vgpr_msb 0x955
	v_max3_num_f32 v207 /*v463*/, v240 /*v496*/, v246 /*v502*/, v250 /*v506*/
	s_set_vgpr_msb 0x5565
	v_max3_num_f32 v208 /*v464*/, v252 /*v508*/, v254 /*v510*/, v0 /*v512*/
	s_set_vgpr_msb 0x65aa
	v_cndmask_b32_e32 v144 /*v656*/, 0xff800000, v10 /*v522*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v145 /*v657*/, v91 /*v603*/
	v_max3_num_f32 v10 /*v522*/, v54 /*v566*/, v55 /*v567*/, v56 /*v568*/
	s_set_vgpr_msb 0xaa55
	v_max3_num_f32 v204 /*v460*/, v206 /*v462*/, v207 /*v463*/, v208 /*v464*/
	s_set_vgpr_msb 0x558a
	v_cndmask_b32_e32 v145 /*v657*/, 0xff800000, v11 /*v523*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v146 /*v658*/, v91 /*v603*/
	s_set_vgpr_msb 0x8a6a
	v_max3_num_f32 v237 /*v493*/, v142 /*v654*/, v143 /*v655*/, v144 /*v656*/
	s_set_vgpr_msb 0x6a8a
	v_cndmask_b32_e32 v146 /*v658*/, 0xff800000, v12 /*v524*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v2 /*v514*/, v91 /*v603*/
	v_dual_max_num_f32 v2 /*v514*/, v106 /*v618*/, v107 /*v619*/ :: v_dual_cndmask_b32 v147 /*v659*/, 0xff800000, v13 /*v525*/
	v_cmp_le_i32_e32 vcc_lo, v3 /*v515*/, v91 /*v603*/
	s_set_vgpr_msb 0x8a6a
	v_max3_num_f32 v239 /*v495*/, v145 /*v657*/, v146 /*v658*/, v147 /*v659*/
	s_set_vgpr_msb 0x6a8a
	v_cndmask_b32_e32 v148 /*v660*/, 0xff800000, v14 /*v526*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v149 /*v661*/, v91 /*v603*/
	s_set_vgpr_msb 0x8a55
	v_max3_num_f32 v206 /*v462*/, v235 /*v491*/, v237 /*v493*/, v239 /*v495*/
	s_set_vgpr_msb 0x55aa
	v_cndmask_b32_e32 v149 /*v661*/, 0xff800000, v15 /*v527*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v150 /*v662*/, v91 /*v603*/
	v_cndmask_b32_e32 v150 /*v662*/, 0xff800000, v16 /*v528*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v6 /*v518*/, v91 /*v603*/
	v_max3_num_f32 v6 /*v518*/, v109 /*v621*/, v110 /*v622*/, v111 /*v623*/
	v_cndmask_b32_e32 v151 /*v663*/, 0xff800000, v17 /*v529*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v7 /*v519*/, v91 /*v603*/
	s_set_vgpr_msb 0xaa6a
	v_max3_num_f32 v241 /*v497*/, v148 /*v660*/, v149 /*v661*/, v150 /*v662*/
	v_max3_num_f32 v212 /*v468*/, v2 /*v514*/, v108 /*v620*/, v6 /*v518*/
	s_set_vgpr_msb 0x6a8a
	v_cndmask_b32_e32 v152 /*v664*/, 0xff800000, v26 /*v538*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v153 /*v665*/, v91 /*v603*/
	s_set_vgpr_msb 0x8a69
	v_max3_num_f32 v205 /*v461*/, v212 /*v468*/, v10 /*v522*/, v57 /*v569*/
	s_set_vgpr_msb 0x698a
	v_cndmask_b32_e32 v153 /*v665*/, 0xff800000, v27 /*v539*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v154 /*v666*/, v91 /*v603*/
	s_set_vgpr_msb 0x8a55
	v_max3_num_f32 v202 /*v458*/, v202 /*v458*/, v204 /*v460*/, v205 /*v461*/
	s_set_vgpr_msb 0x556a
	v_max3_num_f32 v247 /*v503*/, v151 /*v663*/, v152 /*v664*/, v153 /*v665*/
	s_set_vgpr_msb 0x6a8a
	v_cndmask_b32_e32 v154 /*v666*/, 0xff800000, v28 /*v540*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v18 /*v530*/, v91 /*v603*/
	v_cndmask_b32_e32 v155 /*v667*/, 0xff800000, v29 /*v541*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v19 /*v531*/, v91 /*v603*/
	v_cndmask_b32_e32 v156 /*v668*/, 0xff800000, v30 /*v542*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v157 /*v669*/, v91 /*v603*/
	v_cndmask_b32_e32 v157 /*v669*/, 0xff800000, v31 /*v543*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v158 /*v670*/, v91 /*v603*/
	s_set_vgpr_msb 0x8a6a
	v_max3_num_f32 v251 /*v507*/, v154 /*v666*/, v155 /*v667*/, v156 /*v668*/
	s_set_vgpr_msb 0x6a8a
	v_cndmask_b32_e32 v158 /*v670*/, 0xff800000, v32 /*v544*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v22 /*v534*/, v91 /*v603*/
	s_set_vgpr_msb 0x8a55
	v_max3_num_f32 v207 /*v463*/, v241 /*v497*/, v247 /*v503*/, v251 /*v507*/
	s_set_vgpr_msb 0x558a
	v_cndmask_b32_e32 v159 /*v671*/, 0xff800000, v33 /*v545*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v23 /*v535*/, v91 /*v603*/
	s_set_vgpr_msb 0x8a6a
	v_max3_num_f32 v253 /*v509*/, v157 /*v669*/, v158 /*v670*/, v159 /*v671*/
	s_set_vgpr_msb 0x6a8a
	v_cndmask_b32_e32 v160 /*v672*/, 0xff800000, v42 /*v554*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v161 /*v673*/, v91 /*v603*/
	v_cndmask_b32_e32 v161 /*v673*/, 0xff800000, v43 /*v555*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v162 /*v674*/, v91 /*v603*/
	v_cndmask_b32_e32 v162 /*v674*/, 0xff800000, v44 /*v556*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v163 /*v675*/, v91 /*v603*/
	v_cndmask_b32_e32 v163 /*v675*/, 0xff800000, v45 /*v557*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v36 /*v548*/, v91 /*v603*/
	s_set_vgpr_msb 0x8a6a
	v_max3_num_f32 v255 /*v511*/, v160 /*v672*/, v161 /*v673*/, v162 /*v674*/
	s_set_vgpr_msb 0x6aaa
	v_cndmask_b32_e32 v164 /*v676*/, 0xff800000, v46 /*v558*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v37 /*v549*/, v91 /*v603*/
	v_cndmask_b32_e32 v165 /*v677*/, 0xff800000, v47 /*v559*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v166 /*v678*/, v91 /*v603*/
	v_max3_num_f32 v1 /*v513*/, v163 /*v675*/, v164 /*v676*/, v165 /*v677*/
	v_cndmask_b32_e32 v166 /*v678*/, 0xff800000, v48 /*v560*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v167 /*v679*/, v91 /*v603*/
	s_set_vgpr_msb 0xaa65
	v_max3_num_f32 v208 /*v464*/, v253 /*v509*/, v255 /*v511*/, v1 /*v513*/
	s_set_vgpr_msb 0x658a
	v_cndmask_b32_e32 v167 /*v679*/, 0xff800000, v49 /*v561*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v40 /*v552*/, v91 /*v603*/
	s_set_vgpr_msb 0x8a55
	v_max3_num_f32 v204 /*v460*/, v206 /*v462*/, v207 /*v463*/, v208 /*v464*/
	s_set_vgpr_msb 0x55aa
	v_dual_max_num_f32 v3 /*v515*/, v166 /*v678*/, v167 /*v679*/ :: v_dual_cndmask_b32 v168 /*v680*/, 0xff800000, v58 /*v570*/
	v_cmp_le_i32_e32 vcc_lo, v41 /*v553*/, v91 /*v603*/
	v_cndmask_b32_e32 v169 /*v681*/, 0xff800000, v59 /*v571*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v170 /*v682*/, v91 /*v603*/
	v_cndmask_b32_e32 v170 /*v682*/, 0xff800000, v60 /*v572*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v50 /*v562*/, v91 /*v603*/
	v_cndmask_b32_e32 v171 /*v683*/, 0xff800000, v61 /*v573*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v51 /*v563*/, v91 /*v603*/
	v_max3_num_f32 v7 /*v519*/, v169 /*v681*/, v170 /*v682*/, v171 /*v683*/
	v_cndmask_b32_e32 v172 /*v684*/, 0xff800000, v62 /*v574*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v52 /*v564*/, v91 /*v603*/
	s_set_vgpr_msb 0xaa6a
	v_max3_num_f32 v212 /*v468*/, v3 /*v515*/, v168 /*v680*/, v7 /*v519*/
	s_set_vgpr_msb 0x6aaa
	v_cndmask_b32_e32 v173 /*v685*/, 0xff800000, v63 /*v575*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v53 /*v565*/, v91 /*v603*/
	v_cndmask_b32_e32 v174 /*v686*/, 0xff800000, v64 /*v576*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v175 /*v687*/, v91 /*v603*/
	v_cndmask_b32_e32 v175 /*v687*/, 0xff800000, v65 /*v577*/, vcc_lo
	v_max3_num_f32 v11 /*v523*/, v172 /*v684*/, v173 /*v685*/, v174 /*v686*/
	s_set_vgpr_msb 0xaa69
	v_max3_num_f32 v205 /*v461*/, v212 /*v468*/, v11 /*v523*/, v175 /*v687*/
	s_set_vgpr_msb 0x6955
	v_max3_num_f32 v203 /*v459*/, v203 /*v459*/, v204 /*v460*/, v205 /*v461*/
	v_dual_mov_b32 v206 /*v462*/, v202 /*v458*/ :: v_dual_mov_b32 v204 /*v460*/, v203 /*v459*/
	v_permlanex16_b32 v206 /*v462*/, v206 /*v462*/, s82, 0xfedcba98
	v_permlanex16_b32 v204 /*v460*/, v204 /*v460*/, s82, 0xfedcba98
	v_dual_max_num_f32 v202 /*v458*/, v202 /*v458*/, v206 /*v462*/ :: v_dual_max_num_f32 v203 /*v459*/, v203 /*v459*/, v204 /*v460*/
	s_set_vgpr_msb 0x5549
	v_sub_f32_e32 v205 /*v461*/, v202 /*v458*/, v89 /*v601*/
	v_max_num_f32_e32 v202 /*v458*/, v202 /*v458*/, v89 /*v601*/
	v_sub_f32_e32 v204 /*v460*/, v203 /*v459*/, v88 /*v600*/
	s_set_vgpr_msb 0x4904
	v_cmp_lt_f32_e32 vcc_lo, 0x41000000, v205 /*v461*/
	s_cmp_eq_u32 vcc_lo, 0
	s_cselect_b32 s2, -1, 0
	s_set_vgpr_msb 0x489
	v_cndmask_b32_e64 v69 /*v581*/, v202 /*v458*/, v89 /*v601*/, s2
	s_set_vgpr_msb 0x8946
	v_cmp_lt_f32_e64 s2, 0x41000000, v204 /*v460*/
	v_max_num_f32_e32 v202 /*v458*/, v88 /*v600*/, v203 /*v459*/
	s_set_vgpr_msb 0x4688
	v_mul_f32_e32 v60 /*v572*/, 0xbfb8aa3b, v69 /*v581*/
	s_cmp_lg_u32 s2, 0
	s_cselect_b32 s5, -1, 0
	s_cmp_eq_u32 s2, 0
	s_set_vgpr_msb 0x8862
	v_pk_fma_f32 v[208:209] /*v[464:465]*/, v[66:67] /*v[578:579]*/, s[56:57], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_cselect_b32 s2, -1, 0
	s_set_vgpr_msb 0x6261
	v_pk_fma_f32 v[200:201] /*v[456:457]*/, v[200:201] /*v[456:457]*/, s[56:57], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6189
	v_cndmask_b32_e64 v71 /*v583*/, v202 /*v458*/, v88 /*v600*/, s2
	s_set_vgpr_msb 0x8961
	v_pk_fma_f32 v[194:195] /*v[450:451]*/, v[194:195] /*v[450:451]*/, s[56:57], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v238 /*v494*/, v208 /*v464*/
	v_exp_f32_e32 v250 /*v506*/, v209 /*v465*/
	v_nop
	v_pk_fma_f32 v[208:209] /*v[464:465]*/, v[226:227] /*v[482:483]*/, s[56:57], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6188
	v_mul_f32_e32 v68 /*v580*/, 0xbfb8aa3b, v71 /*v583*/
	s_set_vgpr_msb 0x8861
	v_pk_fma_f32 v[226:227] /*v[482:483]*/, v[244:245] /*v[500:501]*/, s[56:57], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[204:205] /*v[460:461]*/, v[198:199] /*v[454:455]*/, s[56:57], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[206:207] /*v[462:463]*/, v[210:211] /*v[466:467]*/, s[56:57], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v212 /*v468*/, v200 /*v456*/
	s_set_vgpr_msb 0x61a2
	v_pk_fma_f32 v[66:67] /*v[578:579]*/, v[112:113] /*v[624:625]*/, s[56:57], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa281
	v_exp_f32_e32 v6 /*v518*/, v226 /*v482*/
	v_exp_f32_e32 v18 /*v530*/, v227 /*v483*/
	v_nop
	s_set_vgpr_msb 0x8162
	v_pk_fma_f32 v[226:227] /*v[482:483]*/, v[96:97] /*v[608:609]*/, s[56:57], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x62a2
	v_pk_fma_f32 v[96:97] /*v[608:609]*/, v[116:117] /*v[628:629]*/, s[56:57], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa261
	v_exp_f32_e32 v220 /*v476*/, v201 /*v457*/
	v_nop
	v_pk_fma_f32 v[200:201] /*v[456:457]*/, v[214:215] /*v[470:471]*/, s[56:57], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[214:215] /*v[470:471]*/, v[228:229] /*v[484:485]*/, s[56:57], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[222:223] /*v[478:479]*/, v[232:233] /*v[488:489]*/, s[56:57], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[202:203] /*v[458:459]*/, v[196:197] /*v[452:453]*/, s[56:57], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v196 /*v452*/, v195 /*v451*/
	v_exp_f32_e32 v210 /*v466*/, v205 /*v461*/
	s_set_vgpr_msb 0x6142
	v_exp_f32_e32 v195 /*v451*/, v66 /*v578*/
	v_exp_f32_e32 v197 /*v453*/, v67 /*v579*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[66:67] /*v[578:579]*/, v[118:119] /*v[630:631]*/, s[56:57], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa242
	v_exp_f32_e32 v205 /*v461*/, v96 /*v608*/
	v_exp_f32_e32 v211 /*v467*/, v97 /*v609*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[96:97] /*v[608:609]*/, v[122:123] /*v[634:635]*/, s[56:57], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa261
	v_exp_f32_e32 v224 /*v480*/, v206 /*v462*/
	v_exp_f32_e32 v234 /*v490*/, v207 /*v463*/
	v_nop
	v_pk_fma_f32 v[206:207] /*v[462:463]*/, v[216:217] /*v[472:473]*/, s[56:57], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v254 /*v510*/, v200 /*v456*/
	v_exp_f32_e32 v200 /*v456*/, v208 /*v464*/
	v_pk_fma_f32 v[218:219] /*v[474:475]*/, v[230:231] /*v[486:487]*/, s[56:57], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v208 /*v464*/, v214 /*v470*/
	v_exp_f32_e32 v216 /*v472*/, v215 /*v471*/
	v_nop
	v_pk_fma_f32 v[214:215] /*v[470:471]*/, v[242:243] /*v[498:499]*/, s[56:57], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v230 /*v486*/, v222 /*v478*/
	v_exp_f32_e32 v242 /*v498*/, v223 /*v479*/
	v_nop
	s_set_vgpr_msb 0x6162
	v_pk_fma_f32 v[222:223] /*v[478:479]*/, v[94:95] /*v[606:607]*/, s[56:57], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x62a2
	v_pk_fma_f32 v[94:95] /*v[606:607]*/, v[114:115] /*v[626:627]*/, s[56:57], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa242
	v_exp_f32_e32 v213 /*v469*/, v66 /*v578*/
	v_exp_f32_e32 v221 /*v477*/, v67 /*v579*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[66:67] /*v[578:579]*/, v[124:125] /*v[636:637]*/, s[56:57], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa242
	v_exp_f32_e32 v239 /*v495*/, v96 /*v608*/
	v_exp_f32_e32 v251 /*v507*/, v97 /*v609*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[96:97] /*v[608:609]*/, v[128:129] /*v[640:641]*/, s[56:57], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa241
	v_exp_f32_e32 v198 /*v454*/, v202 /*v458*/
	v_exp_f32_e32 v202 /*v458*/, v203 /*v459*/
	s_set_vgpr_msb 0x4142
	v_exp_f32_e32 v199 /*v455*/, v94 /*v606*/
	v_exp_f32_e32 v203 /*v459*/, v95 /*v607*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[94:95] /*v[606:607]*/, v[120:121] /*v[632:633]*/, s[56:57], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa281
	v_exp_f32_e32 v10 /*v522*/, v201 /*v457*/
	v_exp_f32_e32 v26 /*v538*/, v207 /*v463*/
	s_set_vgpr_msb 0x8142
	v_exp_f32_e32 v255 /*v511*/, v66 /*v578*/
	s_set_vgpr_msb 0x42a2
	v_exp_f32_e32 v11 /*v523*/, v67 /*v579*/
	v_nop
	v_pk_fma_f32 v[66:67] /*v[578:579]*/, v[130:131] /*v[642:643]*/, s[56:57], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa242
	v_exp_f32_e32 v201 /*v457*/, v96 /*v608*/
	v_exp_f32_e32 v207 /*v463*/, v97 /*v609*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[96:97] /*v[608:609]*/, v[134:135] /*v[646:647]*/, s[56:57], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa242
	v_exp_f32_e32 v225 /*v481*/, v94 /*v606*/
	v_exp_f32_e32 v235 /*v491*/, v95 /*v607*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[94:95] /*v[606:607]*/, v[126:127] /*v[638:639]*/, s[56:57], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa281
	v_exp_f32_e32 v14 /*v526*/, v206 /*v462*/
	s_set_vgpr_msb 0x8141
	v_exp_f32_e32 v206 /*v462*/, v209 /*v465*/
	s_set_vgpr_msb 0x4142
	v_exp_f32_e32 v209 /*v465*/, v66 /*v578*/
	v_exp_f32_e32 v217 /*v473*/, v67 /*v579*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[66:67] /*v[578:579]*/, v[136:137] /*v[648:649]*/, s[56:57], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa242
	v_exp_f32_e32 v231 /*v487*/, v96 /*v608*/
	v_exp_f32_e32 v243 /*v499*/, v97 /*v609*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[96:97] /*v[608:609]*/, v[140:141] /*v[652:653]*/, s[56:57], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v15 /*v527*/, v94 /*v606*/
	v_exp_f32_e32 v27 /*v539*/, v95 /*v607*/
	v_nop
	v_pk_fma_f32 v[94:95] /*v[606:607]*/, v[132:133] /*v[644:645]*/, s[56:57], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa242
	v_exp_f32_e32 v247 /*v503*/, v66 /*v578*/
	s_set_vgpr_msb 0x42a2
	v_exp_f32_e32 v3 /*v515*/, v67 /*v579*/
	v_nop
	v_pk_fma_f32 v[66:67] /*v[578:579]*/, v[142:143] /*v[654:655]*/, s[56:57], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v23 /*v535*/, v96 /*v608*/
	v_exp_f32_e32 v33 /*v545*/, v97 /*v609*/
	v_nop
	v_pk_fma_f32 v[96:97] /*v[608:609]*/, v[146:147] /*v[658:659]*/, s[56:57], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa241
	v_exp_f32_e32 v228 /*v484*/, v219 /*v475*/
	s_set_vgpr_msb 0x4142
	v_exp_f32_e32 v219 /*v475*/, v94 /*v606*/
	v_exp_f32_e32 v229 /*v485*/, v95 /*v607*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[94:95] /*v[606:607]*/, v[138:139] /*v[650:651]*/, s[56:57], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa241
	v_exp_f32_e32 v246 /*v502*/, v214 /*v470*/
	s_set_vgpr_msb 0x4181
	v_exp_f32_e32 v2 /*v514*/, v215 /*v471*/
	v_nop
	s_set_vgpr_msb 0x8161
	v_pk_fma_f32 v[214:215] /*v[470:471]*/, v[248:249] /*v[504:505]*/, s[56:57], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6181
	v_exp_f32_e32 v22 /*v534*/, v222 /*v478*/
	s_set_vgpr_msb 0x8162
	v_pk_fma_f32 v[232:233] /*v[488:489]*/, v[4:5] /*v[516:517]*/, s[56:57], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[240:241] /*v[496:497]*/, v[98:99] /*v[610:611]*/, s[56:57], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6241
	v_exp_f32_e32 v222 /*v478*/, v227 /*v483*/
	s_set_vgpr_msb 0x41a2
	v_exp_f32_e32 v37 /*v549*/, v66 /*v578*/
	v_exp_f32_e32 v45 /*v557*/, v67 /*v579*/
	v_nop
	v_pk_fma_f32 v[66:67] /*v[578:579]*/, v[148:149] /*v[660:661]*/, s[56:57], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa242
	v_exp_f32_e32 v227 /*v483*/, v96 /*v608*/
	v_exp_f32_e32 v237 /*v493*/, v97 /*v609*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[96:97] /*v[608:609]*/, v[152:153] /*v[664:665]*/, s[56:57], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v7 /*v519*/, v94 /*v606*/
	v_exp_f32_e32 v19 /*v531*/, v95 /*v607*/
	v_nop
	v_pk_fma_f32 v[94:95] /*v[606:607]*/, v[144:145] /*v[656:657]*/, s[56:57], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa281
	v_exp_f32_e32 v36 /*v548*/, v214 /*v470*/
	s_set_vgpr_msb 0x8141
	v_exp_f32_e32 v214 /*v470*/, v226 /*v482*/
	v_exp_f32_e32 v226 /*v482*/, v232 /*v488*/
	s_set_vgpr_msb 0x4162
	v_pk_fma_f32 v[244:245] /*v[500:501]*/, v[8:9] /*v[520:521]*/, s[56:57], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6241
	v_exp_f32_e32 v236 /*v492*/, v233 /*v489*/
	v_nop
	s_set_vgpr_msb 0x4162
	v_pk_fma_f32 v[232:233] /*v[488:489]*/, v[100:101] /*v[612:613]*/, s[56:57], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6241
	v_exp_f32_e32 v252 /*v508*/, v241 /*v497*/
	s_set_vgpr_msb 0x4142
	v_exp_f32_e32 v241 /*v497*/, v66 /*v578*/
	v_exp_f32_e32 v253 /*v509*/, v67 /*v579*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[66:67] /*v[578:579]*/, v[154:155] /*v[666:667]*/, s[56:57], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v17 /*v529*/, v96 /*v608*/
	v_exp_f32_e32 v29 /*v541*/, v97 /*v609*/
	v_nop
	v_pk_fma_f32 v[96:97] /*v[608:609]*/, v[158:159] /*v[670:671]*/, s[56:57], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa281
	v_exp_f32_e32 v32 /*v544*/, v223 /*v479*/
	v_exp_f32_e32 v44 /*v556*/, v215 /*v471*/
	s_set_vgpr_msb 0x8142
	v_exp_f32_e32 v215 /*v471*/, v94 /*v606*/
	v_exp_f32_e32 v223 /*v479*/, v95 /*v607*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[94:95] /*v[606:607]*/, v[150:151] /*v[662:663]*/, s[56:57], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa281
	v_exp_f32_e32 v0 /*v512*/, v244 /*v500*/
	v_exp_f32_e32 v12 /*v524*/, v245 /*v501*/
	v_exp_f32_e32 v16 /*v528*/, v232 /*v488*/
	s_set_vgpr_msb 0x8162
	v_pk_fma_f32 v[244:245] /*v[500:501]*/, v[102:103] /*v[614:615]*/, s[56:57], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6281
	v_exp_f32_e32 v28 /*v540*/, v233 /*v489*/
	v_nop
	s_set_vgpr_msb 0x8162
	v_pk_fma_f32 v[232:233] /*v[488:489]*/, v[24:25] /*v[536:537]*/, s[56:57], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x62a2
	v_pk_fma_f32 v[8:9] /*v[520:521]*/, v[38:39] /*v[550:551]*/, s[56:57], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v31 /*v543*/, v66 /*v578*/
	v_exp_f32_e32 v41 /*v553*/, v67 /*v579*/
	v_nop
	v_pk_fma_f32 v[66:67] /*v[578:579]*/, v[160:161] /*v[672:673]*/, s[56:57], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v53 /*v565*/, v96 /*v608*/
	v_exp_f32_e32 v59 /*v571*/, v97 /*v609*/
	v_nop
	v_pk_fma_f32 v[96:97] /*v[608:609]*/, v[164:165] /*v[676:677]*/, s[56:57], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa241
	v_exp_f32_e32 v194 /*v450*/, v194 /*v450*/
	v_exp_f32_e32 v204 /*v460*/, v204 /*v460*/
	s_set_vgpr_msb 0x4162
	v_pk_fma_f32 v[248:249] /*v[504:505]*/, v[20:21] /*v[532:533]*/, s[56:57], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x62a2
	v_exp_f32_e32 v1 /*v513*/, v94 /*v606*/
	v_exp_f32_e32 v13 /*v525*/, v95 /*v607*/
	v_nop
	v_pk_fma_f32 v[94:95] /*v[606:607]*/, v[156:157] /*v[668:669]*/, s[56:57], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa281
	v_exp_f32_e32 v50 /*v562*/, v245 /*v501*/
	v_exp_f32_e32 v58 /*v570*/, v233 /*v489*/
	s_set_vgpr_msb 0x81a2
	v_pk_fma_f32 v[24:25] /*v[536:537]*/, v[106:107] /*v[618:619]*/, s[56:57], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v20 /*v532*/, v9 /*v521*/
	v_pk_fma_f32 v[48:49] /*v[560:561]*/, v[110:111] /*v[622:623]*/, s[56:57], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa242
	v_exp_f32_e32 v233 /*v489*/, v66 /*v578*/
	v_exp_f32_e32 v245 /*v501*/, v67 /*v579*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[66:67] /*v[578:579]*/, v[166:167] /*v[678:679]*/, s[56:57], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v9 /*v521*/, v96 /*v608*/
	v_exp_f32_e32 v21 /*v533*/, v97 /*v609*/
	v_nop
	v_pk_fma_f32 v[96:97] /*v[608:609]*/, v[170:171] /*v[682:683]*/, s[56:57], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa241
	v_exp_f32_e32 v240 /*v496*/, v240 /*v496*/
	s_set_vgpr_msb 0x4181
	v_exp_f32_e32 v30 /*v542*/, v248 /*v504*/
	v_exp_f32_e32 v40 /*v552*/, v249 /*v505*/
	v_nop
	s_set_vgpr_msb 0x8162
	v_pk_fma_f32 v[248:249] /*v[504:505]*/, v[34:35] /*v[546:547]*/, s[56:57], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x62a2
	v_pk_fma_f32 v[4:5] /*v[516:517]*/, v[104:105] /*v[616:617]*/, s[56:57], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v43 /*v555*/, v94 /*v606*/
	v_exp_f32_e32 v51 /*v563*/, v95 /*v607*/
	v_nop
	v_pk_fma_f32 v[94:95] /*v[606:607]*/, v[162:163] /*v[674:675]*/, s[56:57], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v34 /*v546*/, v25 /*v537*/
	v_pk_fma_f32 v[62:63] /*v[574:575]*/, v[54:55] /*v[566:567]*/, s[56:57], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v54 /*v566*/, v49 /*v561*/
	v_exp_f32_e32 v25 /*v537*/, v66 /*v578*/
	v_exp_f32_e32 v35 /*v547*/, v67 /*v579*/
	v_exp_f32_e32 v49 /*v561*/, v96 /*v608*/
	v_pk_fma_f32 v[66:67] /*v[578:579]*/, v[172:173] /*v[684:685]*/, s[56:57], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v55 /*v567*/, v97 /*v609*/
	v_nop
	s_set_vgpr_msb 0xa285
	v_pk_add_f32 v[96:97] /*v[608:609]*/, v[194:195] /*v[450:451]*/, v[196:197] /*v[452:453]*/
	v_pk_add_f32 v[98:99] /*v[610:611]*/, v[202:203] /*v[458:459]*/, v[204:205] /*v[460:461]*/
	v_exp_f32_e32 v42 /*v554*/, v244 /*v500*/
	v_exp_f32_e32 v52 /*v564*/, v232 /*v488*/
	s_set_vgpr_msb 0x8541
	v_exp_f32_e32 v232 /*v488*/, v248 /*v504*/
	v_exp_f32_e32 v244 /*v500*/, v249 /*v505*/
	s_set_vgpr_msb 0x4142
	v_exp_f32_e32 v248 /*v504*/, v4 /*v516*/
	s_set_vgpr_msb 0x42a2
	v_exp_f32_e32 v4 /*v516*/, v5 /*v517*/
	v_pk_fma_f32 v[38:39] /*v[550:551]*/, v[108:109] /*v[620:621]*/, s[56:57], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa242
	v_exp_f32_e32 v249 /*v505*/, v94 /*v606*/
	s_set_vgpr_msb 0x42a2
	v_exp_f32_e32 v5 /*v517*/, v95 /*v607*/
	v_nop
	v_pk_fma_f32 v[94:95] /*v[606:607]*/, v[168:169] /*v[680:681]*/, s[56:57], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[64:65] /*v[576:577]*/, v[56:57] /*v[568:569]*/, s[56:57], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v57 /*v569*/, v66 /*v578*/
	v_exp_f32_e32 v61 /*v573*/, v67 /*v579*/
	v_nop
	s_set_vgpr_msb 0xa289
	v_pk_add_f32 v[66:67] /*v[578:579]*/, v[198:199] /*v[454:455]*/, v[96:97] /*v[608:609]*/
	v_pk_add_f32 v[96:97] /*v[608:609]*/, v[210:211] /*v[466:467]*/, v[98:99] /*v[610:611]*/
	s_set_vgpr_msb 0x8985
	v_pk_add_f32 v[98:99] /*v[610:611]*/, v[212:213] /*v[468:469]*/, v[220:221] /*v[476:477]*/
	v_pk_add_f32 v[100:101] /*v[612:613]*/, v[234:235] /*v[490:491]*/, v[238:239] /*v[494:495]*/
	s_set_vgpr_msb 0x8589
	v_pk_add_f32 v[102:103] /*v[614:615]*/, v[254:255] /*v[510:511]*/, v[10:11] /*v[522:523]*/
	s_set_vgpr_msb 0x898a
	v_pk_add_f32 v[112:113] /*v[624:625]*/, v[18:19] /*v[530:531]*/, v[22:23] /*v[534:535]*/
	v_pk_add_f32 v[114:115] /*v[626:627]*/, v[36:37] /*v[548:549]*/, v[44:45] /*v[556:557]*/
	s_set_vgpr_msb 0x8a85
	v_pk_add_f32 v[118:119] /*v[630:631]*/, v[240:241] /*v[496:497]*/, v[252:253] /*v[508:509]*/
	s_set_vgpr_msb 0x858a
	v_pk_add_f32 v[120:121] /*v[632:633]*/, v[12:13] /*v[524:525]*/, v[16:17] /*v[528:529]*/
	s_set_vgpr_msb 0x8a41
	v_exp_f32_e32 v218 /*v474*/, v218 /*v474*/
	s_set_vgpr_msb 0x4186
	v_exp_f32_e32 v8 /*v520*/, v8 /*v520*/
	v_exp_f32_e32 v24 /*v536*/, v24 /*v536*/
	v_exp_f32_e32 v46 /*v558*/, v39 /*v551*/
	v_exp_f32_e32 v48 /*v560*/, v48 /*v560*/
	v_exp_f32_e32 v47 /*v559*/, v95 /*v607*/
	v_pk_add_f32 v[104:105] /*v[616:617]*/, v[26:27] /*v[538:539]*/, v[200:201] /*v[456:457]*/
	s_set_vgpr_msb 0x8685
	v_pk_add_f32 v[106:107] /*v[618:619]*/, v[208:209] /*v[464:465]*/, v[216:217] /*v[472:473]*/
	s_set_vgpr_msb 0x8589
	v_pk_add_f32 v[98:99] /*v[610:611]*/, v[224:225] /*v[480:481]*/, v[98:99] /*v[610:611]*/
	v_pk_add_f32 v[100:101] /*v[612:613]*/, v[250:251] /*v[506:507]*/, v[100:101] /*v[612:613]*/
	s_set_vgpr_msb 0x898a
	v_pk_add_f32 v[102:103] /*v[614:615]*/, v[14:15] /*v[526:527]*/, v[102:103] /*v[614:615]*/
	s_set_vgpr_msb 0x8a85
	v_pk_add_f32 v[108:109] /*v[620:621]*/, v[228:229] /*v[484:485]*/, v[230:231] /*v[486:487]*/
	v_pk_add_f32 v[116:117] /*v[628:629]*/, v[222:223] /*v[478:479]*/, v[226:227] /*v[482:483]*/
	s_set_vgpr_msb 0x858a
	v_pk_add_f32 v[112:113] /*v[624:625]*/, v[32:33] /*v[544:545]*/, v[112:113] /*v[624:625]*/
	s_set_vgpr_msb 0x8a89
	v_pk_add_f32 v[114:115] /*v[626:627]*/, v[214:215] /*v[470:471]*/, v[114:115] /*v[626:627]*/
	s_set_vgpr_msb 0x898a
	v_pk_add_f32 v[122:123] /*v[634:635]*/, v[30:31] /*v[542:543]*/, v[40:41] /*v[552:553]*/
	v_pk_add_f32 v[124:125] /*v[636:637]*/, v[50:51] /*v[562:563]*/, v[52:53] /*v[564:565]*/
	s_set_vgpr_msb 0x8a85
	v_pk_add_f32 v[126:127] /*v[638:639]*/, v[232:233] /*v[488:489]*/, v[244:245] /*v[500:501]*/
	s_set_vgpr_msb 0x85aa
	v_pk_add_f32 v[118:119] /*v[630:631]*/, v[0:1] /*v[512:513]*/, v[118:119] /*v[630:631]*/
	v_pk_add_f32 v[120:121] /*v[632:633]*/, v[28:29] /*v[540:541]*/, v[120:121] /*v[632:633]*/
	v_pk_add_f32 v[66:67] /*v[578:579]*/, v[66:67] /*v[578:579]*/, v[96:97] /*v[608:609]*/
	v_exp_f32_e32 v38 /*v550*/, v38 /*v550*/
	v_exp_f32_e32 v56 /*v568*/, v62 /*v574*/
	v_exp_f32_e32 v60 /*v572*/, v63 /*v575*/
	v_exp_f32_e32 v39 /*v551*/, v94 /*v606*/
	v_nop
	v_pk_fma_f32 v[94:95] /*v[606:607]*/, v[174:175] /*v[686:687]*/, s[56:57], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xaa89
	v_pk_add_f32 v[104:105] /*v[616:617]*/, v[206:207] /*v[462:463]*/, v[104:105] /*v[616:617]*/
	v_pk_add_f32 v[106:107] /*v[618:619]*/, v[218:219] /*v[474:475]*/, v[106:107] /*v[618:619]*/
	v_pk_add_f32 v[110:111] /*v[622:623]*/, v[246:247] /*v[502:503]*/, v[2:3] /*v[514:515]*/
	v_pk_add_f32 v[108:109] /*v[620:621]*/, v[242:243] /*v[498:499]*/, v[108:109] /*v[620:621]*/
	v_pk_add_f32 v[116:117] /*v[628:629]*/, v[236:237] /*v[492:493]*/, v[116:117] /*v[628:629]*/
	s_set_vgpr_msb 0x898a
	v_pk_add_f32 v[122:123] /*v[634:635]*/, v[42:43] /*v[554:555]*/, v[122:123] /*v[634:635]*/
	v_pk_add_f32 v[124:125] /*v[636:637]*/, v[58:59] /*v[570:571]*/, v[124:125] /*v[636:637]*/
	s_set_vgpr_msb 0x8a89
	v_pk_add_f32 v[126:127] /*v[638:639]*/, v[248:249] /*v[504:505]*/, v[126:127] /*v[638:639]*/
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
	v_sub_f32_e32 v68 /*v580*/, v89 /*v601*/, v69 /*v581*/
	v_pk_add_f32 v[94:95] /*v[606:607]*/, v[64:65] /*v[576:577]*/, v[94:95] /*v[606:607]*/
	v_pk_add_f32 v[66:67] /*v[578:579]*/, v[66:67] /*v[578:579]*/, v[96:97] /*v[608:609]*/
	v_mul_f32_e32 v68 /*v580*/, 0x3fb8aa3b, v68 /*v580*/
	v_pk_add_f32 v[66:67] /*v[578:579]*/, v[94:95] /*v[606:607]*/, v[66:67] /*v[578:579]*/
	v_exp_f32_e32 v68 /*v580*/, v68 /*v580*/
	v_dual_mov_b32 v89 /*v601*/, v66 /*v578*/ :: v_dual_mov_b32 v94 /*v606*/, v67 /*v579*/
	v_permlanex16_b32 v89 /*v601*/, v89 /*v601*/, s82, 0xfedcba98
	v_permlanex16_b32 v94 /*v606*/, v94 /*v606*/, s82, 0xfedcba98
	s_set_vgpr_msb 0x8a00
	s_cbranch_vccz .LBB0_19
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
.LBB0_19:
	s_set_vgpr_msb 0x8a
	v_sub_f32_e32 v70 /*v582*/, v88 /*v600*/, v71 /*v583*/
	s_and_b32 s2, s5, exec_lo
	s_cselect_b32 s2, 1, 0
	s_cmp_lg_u32 s2, 1
	v_mul_f32_e32 v70 /*v582*/, 0x3fb8aa3b, v70 /*v582*/
	v_exp_f32_e32 v70 /*v582*/, v70 /*v582*/
	s_set_vgpr_msb 0x8a00
	s_cbranch_scc1 .LBB0_14
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
	s_branch .LBB0_14
.LBB0_21:
	s_set_vgpr_msb 0x82
	v_dual_mov_b32 v71 /*v583*/, v88 /*v600*/ :: v_dual_mov_b32 v69 /*v581*/, v89 /*v601*/
	s_set_vgpr_msb 0x8200
.LBB0_22:
	s_branch .LBB0_43
.LBB0_23:
	s_mov_b32 s19, 0
	s_cmp_lg_u32 s60, 0x80000000
	s_mov_b32 s17, s19
	s_mov_b32 s18, s19
	s_mov_b32 s4, s19
	s_mov_b32 s5, s19
	s_mov_b32 s6, s19
	s_mov_b32 s7, s19
	s_mov_b32 s39, s19
	tensor_load_to_lds s[12:15], s[20:27], s[16:19], s[4:7]
	s_cselect_b32 s7, s61, 0
	s_cselect_b32 s17, s60, 0x80
	s_and_b32 s2, s91, 3
	s_and_b32 s18, s7, 0xffff
	s_lshl_b32 s53, s2, 5
	s_lshl_b32 s38, s2, 6
	s_mul_i32 s54, s2, 0x2200
	s_mul_i32 s55, s2, 0x2400
	s_sub_co_i32 s2, s65, s53
	s_mov_b32 s6, s17
	s_max_i32 s2, s2, 0
	s_mul_u64 s[36:37], s[6:7], s[38:39]
	s_lshl_b32 s2, s2, 16
	s_add_nc_u64 s[6:7], s[36:37], s[76:77]
	s_or_b32 s14, s2, 0x7fff
	s_cmp_lg_u32 s62, 0x80000000
	s_mov_b32 s4, 1
	s_cselect_b32 s25, s62, 0x80
	s_cselect_b32 s41, s63, 0
	s_mov_b32 s40, s25
	s_mov_b32 s15, 0x800000
	s_mov_b32 s13, 0xffff0000
	s_mov_b32 s12, 0x7510000
	s_mov_b32 s16, 32
	s_mov_b32 s5, s54
	s_bitset1_b32 s7, 31
	s_mul_u64 s[38:39], s[40:41], s[38:39]
	s_bitset1_b32 s55, 16
	s_mov_b32 s20, 0xf510000
	s_mov_b32 s21, s13
	s_mov_b32 s23, s15
	s_mov_b32 s27, s19
	s_mov_b32 s24, s16
	s_mov_b32 s22, s14
	s_and_b32 s26, s41, 0xffff
	s_ashr_i32 s2, s90, 2
	s_add_co_i32 s11, s35, -1
	s_set_vgpr_msb 34
	v_and_or_b32 v64, v78 /*v590*/, 7, v81 /*v593*/
	s_add_co_i32 s2, s11, s2
	s_add_nc_u64 s[40:41], s[68:69], s[72:73]
	s_max_i32 s2, s2, 0
	s_add_nc_u64 s[42:43], s[66:67], s[70:71]
	s_add_co_i32 s2, s2, 1
	v_mul_u32_u24_e32 v64, 0x120, v64
	s_mov_b32 s45, s19
	s_set_vgpr_msb 0x2202
	v_and_or_b32 v64, v84 /*v596*/, 16, v64
	s_set_vgpr_msb 0x280
	v_or_b32_e32 v84 /*v596*/, 0x10000, v64
	v_or_b32_e32 v90 /*v602*/, 0x30000, v64
	s_wait_tensorcnt 0x0
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x8002
	ds_load_b128 v[0:3], v85 /*v597*/
	ds_load_b128 v[4:7], v85 /*v597*/ offset:32
	ds_load_b128 v[8:11], v85 /*v597*/ offset:64
	ds_load_b128 v[12:15], v85 /*v597*/ offset:96
	ds_load_b128 v[16:19], v85 /*v597*/ offset:128
	ds_load_b128 v[20:23], v85 /*v597*/ offset:160
	ds_load_b128 v[24:27], v85 /*v597*/ offset:192
	ds_load_b128 v[28:31], v85 /*v597*/ offset:224
	ds_load_b128 v[32:35], v85 /*v597*/ offset:4352
	ds_load_b128 v[36:39], v85 /*v597*/ offset:4384
	ds_load_b128 v[40:43], v85 /*v597*/ offset:4416
	ds_load_b128 v[44:47], v85 /*v597*/ offset:4448
	ds_load_b128 v[48:51], v85 /*v597*/ offset:4480
	ds_load_b128 v[52:55], v85 /*v597*/ offset:4512
	ds_load_b128 v[56:59], v85 /*v597*/ offset:4544
	ds_load_b128 v[60:63], v85 /*v597*/ offset:4576
	tensor_load_to_lds s[4:7], s[12:19]
	s_add_nc_u64 s[6:7], s[38:39], s[74:75]
	s_mov_b32 s5, s55
	s_bitset1_b32 s7, 31
	s_wait_dscnt 0xe
	v_pk_mul_bf16 v135, v83 /*v595*/, v7
	tensor_load_to_lds s[4:7], s[20:27]
	s_ashr_i32 s5, s2, 31
	s_add_co_i32 s6, s10, -1
	s_lshr_b32 s5, s5, 25
	v_pk_mul_bf16 v134, v83 /*v595*/, v6
	s_add_co_i32 s5, s2, s5
	v_pk_mul_bf16 v133, v83 /*v595*/, v5
	s_and_b32 s7, s5, 0xffffff80
	s_ashr_i32 s5, s5, 7
	s_cmp_lg_u32 s2, s7
	v_pk_mul_bf16 v132, v83 /*v595*/, v4
	s_cselect_b32 s7, -1, 0
	s_cmp_lt_i32 s2, 0
	v_pk_mul_bf16 v131, v83 /*v595*/, v3
	s_cselect_b32 s2, -1, 0
	v_pk_mul_bf16 v130, v83 /*v595*/, v2
	s_and_b32 s2, s2, s7
	s_sub_co_ci_u32 s2, s5, 0
	v_pk_mul_bf16 v129, v83 /*v595*/, v1
	s_min_i32 s2, s2, s6
	v_pk_mul_bf16 v128, v83 /*v595*/, v0
	s_wait_dscnt 0xc
	v_pk_mul_bf16 v143, v83 /*v595*/, v15
	v_pk_mul_bf16 v142, v83 /*v595*/, v14
	v_pk_mul_bf16 v141, v83 /*v595*/, v13
	v_pk_mul_bf16 v140, v83 /*v595*/, v12
	v_pk_mul_bf16 v139, v83 /*v595*/, v11
	v_pk_mul_bf16 v138, v83 /*v595*/, v10
	v_pk_mul_bf16 v137, v83 /*v595*/, v9
	v_pk_mul_bf16 v136, v83 /*v595*/, v8
	s_wait_dscnt 0xa
	v_pk_mul_bf16 v151, v83 /*v595*/, v23
	v_pk_mul_bf16 v150, v83 /*v595*/, v22
	v_pk_mul_bf16 v149, v83 /*v595*/, v21
	v_pk_mul_bf16 v148, v83 /*v595*/, v20
	v_pk_mul_bf16 v147, v83 /*v595*/, v19
	v_pk_mul_bf16 v146, v83 /*v595*/, v18
	v_pk_mul_bf16 v145, v83 /*v595*/, v17
	v_pk_mul_bf16 v144, v83 /*v595*/, v16
	s_wait_dscnt 0x8
	v_pk_mul_bf16 v159, v83 /*v595*/, v31
	v_pk_mul_bf16 v158, v83 /*v595*/, v30
	v_pk_mul_bf16 v157, v83 /*v595*/, v29
	v_pk_mul_bf16 v156, v83 /*v595*/, v28
	v_pk_mul_bf16 v155, v83 /*v595*/, v27
	v_pk_mul_bf16 v154, v83 /*v595*/, v26
	v_pk_mul_bf16 v153, v83 /*v595*/, v25
	v_pk_mul_bf16 v152, v83 /*v595*/, v24
	s_wait_dscnt 0x6
	v_pk_mul_bf16 v167, v83 /*v595*/, v39
	v_pk_mul_bf16 v166, v83 /*v595*/, v38
	v_pk_mul_bf16 v165, v83 /*v595*/, v37
	v_pk_mul_bf16 v164, v83 /*v595*/, v36
	v_pk_mul_bf16 v163, v83 /*v595*/, v35
	v_pk_mul_bf16 v162, v83 /*v595*/, v34
	v_pk_mul_bf16 v161, v83 /*v595*/, v33
	v_pk_mul_bf16 v160, v83 /*v595*/, v32
	s_wait_dscnt 0x4
	v_pk_mul_bf16 v175, v83 /*v595*/, v47
	v_pk_mul_bf16 v174, v83 /*v595*/, v46
	v_pk_mul_bf16 v173, v83 /*v595*/, v45
	v_pk_mul_bf16 v172, v83 /*v595*/, v44
	v_pk_mul_bf16 v171, v83 /*v595*/, v43
	v_pk_mul_bf16 v170, v83 /*v595*/, v42
	v_pk_mul_bf16 v169, v83 /*v595*/, v41
	v_pk_mul_bf16 v168, v83 /*v595*/, v40
	s_wait_dscnt 0x2
	v_pk_mul_bf16 v183, v83 /*v595*/, v55
	v_pk_mul_bf16 v182, v83 /*v595*/, v54
	v_pk_mul_bf16 v181, v83 /*v595*/, v53
	v_pk_mul_bf16 v180, v83 /*v595*/, v52
	v_pk_mul_bf16 v179, v83 /*v595*/, v51
	v_pk_mul_bf16 v178, v83 /*v595*/, v50
	v_pk_mul_bf16 v177, v83 /*v595*/, v49
	v_pk_mul_bf16 v176, v83 /*v595*/, v48
	s_wait_dscnt 0x0
	v_pk_mul_bf16 v191, v83 /*v595*/, v63
	v_pk_mul_bf16 v190, v83 /*v595*/, v62
	v_pk_mul_bf16 v189, v83 /*v595*/, v61
	v_pk_mul_bf16 v188, v83 /*v595*/, v60
	v_pk_mul_bf16 v187, v83 /*v595*/, v59
	v_pk_mul_bf16 v186, v83 /*v595*/, v58
	v_pk_mul_bf16 v185, v83 /*v595*/, v57
	v_pk_mul_bf16 v184, v83 /*v595*/, v56
	v_mov_b32_e32 v0, 0
	s_max_i32 s44, s2, 0
	s_cmp_lt_i32 s2, 1
	s_wait_tensorcnt 0x0
	s_set_vgpr_msb 0x200
	s_barrier_signal -1
	s_barrier_wait -1
	s_cbranch_scc1 .LBB0_32
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
	s_set_vgpr_msb 0x80
	v_dual_mov_b32 v71 /*v583*/, 0xf149f2ca :: v_dual_mov_b32 v66 /*v578*/, 0xf149f2ca
	s_set_vgpr_msb 0x8002
	v_dual_mov_b32 v200, v90 /*v602*/ :: v_dual_mov_b32 v201, v82 /*v594*/
	s_set_vgpr_msb 0x282
	v_mov_b32_e32 v83 /*v595*/, v80 /*v592*/
	s_set_vgpr_msb 0x8240
	v_dual_mov_b32 v192 /*v448*/, v0 :: v_dual_mov_b32 v193 /*v449*/, v0
	s_lshl_b64 s[46:47], s[44:45], 7
	s_add_co_i32 s35, s9, 0xffffff80
	s_add_co_i32 s48, s64, 0x80
	s_mov_b64 s[50:51], 0xffffffffffffff80
	s_mov_b32 s56, 0x76543210
	s_mov_b32 s52, 0x3fb8aa3b
	s_mov_b32 s65, 1
	s_set_vgpr_msb 0x4000
	s_branch .LBB0_26
.LBB0_25:
	s_set_vgpr_msb 0x8a
	v_cvt_pk_bf16_f32 v99 /*v611*/, v20 /*v532*/, v34 /*v546*/
	v_cvt_pk_bf16_f32 v98 /*v610*/, v10 /*v522*/, v18 /*v530*/
	s_set_vgpr_msb 0x8a89
	v_cvt_pk_bf16_f32 v97 /*v609*/, v248 /*v504*/, v2 /*v514*/
	s_set_vgpr_msb 0x8985
	v_cvt_pk_bf16_f32 v96 /*v608*/, v230 /*v486*/, v242 /*v498*/
	v_cvt_pk_bf16_f32 v95 /*v607*/, v214 /*v470*/, v226 /*v482*/
	v_cvt_pk_bf16_f32 v94 /*v606*/, v206 /*v462*/, v212 /*v468*/
	v_cvt_pk_bf16_f32 v93 /*v605*/, v198 /*v454*/, v202 /*v458*/
	v_cvt_pk_bf16_f32 v92 /*v604*/, v194 /*v450*/, v196 /*v452*/
	s_set_vgpr_msb 0x858a
	v_cvt_pk_bf16_f32 v107 /*v619*/, v21 /*v533*/, v35 /*v547*/
	v_cvt_pk_bf16_f32 v106 /*v618*/, v11 /*v523*/, v19 /*v531*/
	s_set_vgpr_msb 0x8a89
	v_cvt_pk_bf16_f32 v105 /*v617*/, v249 /*v505*/, v3 /*v515*/
	s_set_vgpr_msb 0x8985
	v_cvt_pk_bf16_f32 v104 /*v616*/, v231 /*v487*/, v243 /*v499*/
	v_cvt_pk_bf16_f32 v103 /*v615*/, v215 /*v471*/, v227 /*v483*/
	v_cvt_pk_bf16_f32 v102 /*v614*/, v207 /*v463*/, v213 /*v469*/
	v_cvt_pk_bf16_f32 v101 /*v613*/, v199 /*v455*/, v203 /*v459*/
	v_cvt_pk_bf16_f32 v100 /*v612*/, v195 /*v451*/, v197 /*v453*/
	s_set_vgpr_msb 0x8509
	s_wait_dscnt 0x3d
	v_wmma_f32_16x16x32_bf16 v[120:127], v[184:191] /*v[440:447]*/, v[92:99] /*v[604:611]*/, v[120:127]
	s_set_vgpr_msb 0x98a
	v_cvt_pk_bf16_f32 v115 /*v627*/, v39 /*v551*/, v49 /*v561*/
	v_cvt_pk_bf16_f32 v114 /*v626*/, v29 /*v541*/, v37 /*v549*/
	v_cvt_pk_bf16_f32 v113 /*v625*/, v13 /*v525*/, v23 /*v535*/
	s_set_vgpr_msb 0x8a89
	v_cvt_pk_bf16_f32 v112 /*v624*/, v251 /*v507*/, v5 /*v517*/
	s_set_vgpr_msb 0x8985
	v_cvt_pk_bf16_f32 v111 /*v623*/, v233 /*v489*/, v245 /*v501*/
	v_cvt_pk_bf16_f32 v110 /*v622*/, v221 /*v477*/, v229 /*v485*/
	v_cvt_pk_bf16_f32 v109 /*v621*/, v209 /*v465*/, v217 /*v473*/
	s_set_vgpr_msb 0x8509
	v_wmma_f32_16x16x32_bf16 v[56:63], v[184:191] /*v[440:447]*/, v[100:107] /*v[612:619]*/, v[56:63]
	s_set_vgpr_msb 0x985
	v_cvt_pk_bf16_f32 v108 /*v620*/, v201 /*v457*/, v205 /*v461*/
	s_set_vgpr_msb 0x854a
	v_cvt_pk_bf16_f32 v201 /*v457*/, v53 /*v565*/, v59 /*v571*/
	v_cvt_pk_bf16_f32 v199 /*v455*/, v31 /*v543*/, v41 /*v553*/
	v_cvt_pk_bf16_f32 v198 /*v454*/, v15 /*v527*/, v25 /*v537*/
	v_cvt_pk_bf16_f32 v191 /*v447*/, v38 /*v550*/, v48 /*v560*/
	v_cvt_pk_bf16_f32 v190 /*v446*/, v28 /*v540*/, v36 /*v548*/
	v_cvt_pk_bf16_f32 v189 /*v445*/, v12 /*v524*/, v22 /*v534*/
	s_set_vgpr_msb 0x4a09
	s_wait_dscnt 0x3c
	v_wmma_f32_16x16x32_bf16 v[112:119], v[152:159] /*v[408:415]*/, v[92:99] /*v[604:611]*/, v[112:119]
	s_set_vgpr_msb 0x949
	v_cvt_pk_bf16_f32 v188 /*v444*/, v250 /*v506*/, v4 /*v516*/
	s_set_vgpr_msb 0x4945
	v_cvt_pk_bf16_f32 v187 /*v443*/, v232 /*v488*/, v244 /*v500*/
	v_cvt_pk_bf16_f32 v186 /*v442*/, v220 /*v476*/, v228 /*v484*/
	v_cvt_pk_bf16_f32 v185 /*v441*/, v208 /*v464*/, v216 /*v472*/
	v_cvt_pk_bf16_f32 v184 /*v440*/, v200 /*v456*/, v204 /*v460*/
	s_set_vgpr_msb 0x454a
	v_cvt_pk_bf16_f32 v200 /*v456*/, v45 /*v557*/, v51 /*v563*/
	s_set_vgpr_msb 0x4a49
	v_cvt_pk_bf16_f32 v197 /*v453*/, v253 /*v509*/, v7 /*v519*/
	s_set_vgpr_msb 0x4909
	v_wmma_f32_16x16x32_bf16 v[48:55], v[152:159] /*v[408:415]*/, v[100:107] /*v[612:619]*/, v[48:55]
	s_set_vgpr_msb 0x945
	v_cvt_pk_bf16_f32 v196 /*v452*/, v239 /*v495*/, v247 /*v503*/
	v_cvt_pk_bf16_f32 v195 /*v451*/, v223 /*v479*/, v235 /*v491*/
	v_cvt_pk_bf16_f32 v194 /*v450*/, v211 /*v467*/, v219 /*v475*/
	s_set_vgpr_msb 0x454a
	v_cvt_pk_bf16_f32 v209 /*v465*/, v63 /*v575*/, v65 /*v577*/
	v_cvt_pk_bf16_f32 v208 /*v464*/, v57 /*v569*/, v61 /*v573*/
	v_cvt_pk_bf16_f32 v207 /*v463*/, v47 /*v559*/, v55 /*v567*/
	v_cvt_pk_bf16_f32 v206 /*v462*/, v33 /*v545*/, v43 /*v555*/
	s_set_vgpr_msb 0x4a09
	s_wait_dscnt 0x2d
	v_wmma_f32_16x16x32_bf16 v[104:111], v[120:127] /*v[376:383]*/, v[92:99] /*v[604:611]*/, v[104:111]
	s_set_vgpr_msb 0x94a
	v_cvt_pk_bf16_f32 v205 /*v461*/, v17 /*v529*/, v27 /*v539*/
	v_cvt_pk_bf16_f32 v204 /*v460*/, v1 /*v513*/, v9 /*v521*/
	s_set_vgpr_msb 0x4a45
	v_cvt_pk_bf16_f32 v203 /*v459*/, v241 /*v497*/, v255 /*v511*/
	v_cvt_pk_bf16_f32 v202 /*v458*/, v225 /*v481*/, v237 /*v493*/
	s_add_nc_u64 s[46:47], s[46:47], s[50:51]
	s_addk_co_i32 s35, 0xff80
	s_addk_co_i32 s48, 0x80
	s_set_vgpr_msb 0x4509
	v_wmma_f32_16x16x32_bf16 v[40:47], v[120:127] /*v[376:383]*/, v[100:107] /*v[612:619]*/, v[40:47]
	s_add_co_i32 s65, s65, 1
	s_cmp_lg_u64 s[46:47], 0
	s_wait_dscnt 0x0
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
	v_cvt_pk_bf16_f32 v182 /*v438*/, v44 /*v556*/, v50 /*v562*/
	v_cvt_pk_bf16_f32 v181 /*v437*/, v30 /*v542*/, v40 /*v552*/
	s_set_vgpr_msb 0x4a05
	v_wmma_f32_16x16x32_bf16 v[112:119], v[144:151] /*v[400:407]*/, v[184:191] /*v[440:447]*/, v[112:119]
	s_set_vgpr_msb 0x54a
	v_cvt_pk_bf16_f32 v180 /*v436*/, v14 /*v526*/, v24 /*v536*/
	s_set_vgpr_msb 0x4a49
	v_cvt_pk_bf16_f32 v179 /*v435*/, v252 /*v508*/, v6 /*v518*/
	s_set_vgpr_msb 0x4945
	v_cvt_pk_bf16_f32 v178 /*v434*/, v238 /*v494*/, v246 /*v502*/
	v_cvt_pk_bf16_f32 v177 /*v433*/, v222 /*v478*/, v234 /*v490*/
	v_cvt_pk_bf16_f32 v176 /*v432*/, v210 /*v466*/, v218 /*v474*/
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
	v_wmma_f32_16x16x32_bf16 v[56:63], v[168:175] /*v[424:431]*/, v[194:201] /*v[450:457]*/, v[56:63]
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
	v_cvt_pk_bf16_f32 v169 /*v425*/, v240 /*v496*/, v254 /*v510*/
	v_cvt_pk_bf16_f32 v168 /*v424*/, v224 /*v480*/, v236 /*v492*/
	s_set_vgpr_msb 0x4505
	v_wmma_f32_16x16x32_bf16 v[48:55], v[136:143] /*v[392:399]*/, v[194:201] /*v[450:457]*/, v[48:55]
	v_wmma_f32_16x16x32_bf16 v[104:111], v[104:111] /*v[360:367]*/, v[176:183] /*v[432:439]*/, v[104:111]
	v_wmma_f32_16x16x32_bf16 v[40:47], v[104:111] /*v[360:367]*/, v[194:201] /*v[450:457]*/, v[40:47]
	v_wmma_f32_16x16x32_bf16 v[96:103], v[72:79] /*v[328:335]*/, v[176:183] /*v[432:439]*/, v[96:103]
	v_wmma_f32_16x16x32_bf16 v[32:39], v[72:79] /*v[328:335]*/, v[194:201] /*v[450:457]*/, v[32:39]
	v_wmma_f32_16x16x32_bf16 v[88:95], v[40:47] /*v[296:303]*/, v[176:183] /*v[432:439]*/, v[88:95]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[40:47] /*v[296:303]*/, v[194:201] /*v[450:457]*/, v[24:31]
	v_wmma_f32_16x16x32_bf16 v[80:87], v[8:15] /*v[264:271]*/, v[176:183] /*v[432:439]*/, v[80:87]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[8:15] /*v[264:271]*/, v[194:201] /*v[450:457]*/, v[16:23]
	s_set_vgpr_msb 0x504
	v_wmma_f32_16x16x32_bf16 v[72:79], v[232:239], v[176:183] /*v[432:439]*/, v[72:79]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[232:239], v[194:201] /*v[450:457]*/, v[8:15]
	v_wmma_f32_16x16x32_bf16 v[64:71], v[200:207], v[176:183] /*v[432:439]*/, v[64:71]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[200:207], v[194:201] /*v[450:457]*/, v[0:7]
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x426
	v_pk_fma_f32 v[200:201], v[70:71] /*v[582:583]*/, v[192:193] /*v[448:449]*/, v[68:69] /*v[580:581]*/
	s_set_vgpr_msb 0x2682
	v_mov_b32_e32 v71 /*v583*/, v85 /*v597*/
	s_set_vgpr_msb 0x8248
	v_pk_add_f32 v[192:193] /*v[448:449]*/, v[200:201], v[66:67] /*v[578:579]*/
	s_set_vgpr_msb 0x4805
	v_wmma_f32_16x16x32_bf16 v[120:127], v[160:167] /*v[416:423]*/, v[168:175] /*v[424:431]*/, v[120:127]
	s_set_vgpr_msb 0x502
	v_dual_mov_b32 v200, v90 /*v602*/ :: v_dual_mov_b32 v201, v82 /*v594*/
	s_set_vgpr_msb 0x282
	v_mov_b32_e32 v66 /*v578*/, v86 /*v598*/
	s_set_vgpr_msb 0x8205
	v_wmma_f32_16x16x32_bf16 v[56:63], v[160:167] /*v[416:423]*/, v[202:209] /*v[458:465]*/, v[56:63]
	v_wmma_f32_16x16x32_bf16 v[112:119], v[128:135] /*v[384:391]*/, v[168:175] /*v[424:431]*/, v[112:119]
	v_wmma_f32_16x16x32_bf16 v[48:55], v[128:135] /*v[384:391]*/, v[202:209] /*v[458:465]*/, v[48:55]
	v_wmma_f32_16x16x32_bf16 v[104:111], v[96:103] /*v[352:359]*/, v[168:175] /*v[424:431]*/, v[104:111]
	v_wmma_f32_16x16x32_bf16 v[40:47], v[96:103] /*v[352:359]*/, v[202:209] /*v[458:465]*/, v[40:47]
	v_wmma_f32_16x16x32_bf16 v[96:103], v[64:71] /*v[320:327]*/, v[168:175] /*v[424:431]*/, v[96:103]
	v_wmma_f32_16x16x32_bf16 v[32:39], v[64:71] /*v[320:327]*/, v[202:209] /*v[458:465]*/, v[32:39]
	v_wmma_f32_16x16x32_bf16 v[88:95], v[32:39] /*v[288:295]*/, v[168:175] /*v[424:431]*/, v[88:95]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[32:39] /*v[288:295]*/, v[202:209] /*v[458:465]*/, v[24:31]
	v_wmma_f32_16x16x32_bf16 v[80:87], v[0:7] /*v[256:263]*/, v[168:175] /*v[424:431]*/, v[80:87]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[0:7] /*v[256:263]*/, v[202:209] /*v[458:465]*/, v[16:23]
	s_set_vgpr_msb 0x504
	v_wmma_f32_16x16x32_bf16 v[72:79], v[224:231], v[168:175] /*v[424:431]*/, v[72:79]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[224:231], v[202:209] /*v[458:465]*/, v[8:15]
	v_wmma_f32_16x16x32_bf16 v[64:71], v[192:199], v[168:175] /*v[424:431]*/, v[64:71]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[192:199], v[202:209] /*v[458:465]*/, v[0:7]
	s_set_vgpr_msb 0x400
	s_cbranch_scc0 .LBB0_33
.LBB0_26:
	s_wait_alu depctr_vm_vsrc(0)
	s_set_vgpr_msb 0x82
	v_dual_mov_b32 v82 /*v594*/, v83 /*v595*/ :: v_dual_mov_b32 v90 /*v602*/, v84 /*v596*/
	s_set_vgpr_msb 0x8280
	v_dual_mov_b32 v83 /*v595*/, v201 :: v_dual_mov_b32 v84 /*v596*/, v200
	s_wait_tensorcnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x8042
	ds_load_b128 v[184:187] /*v[440:443]*/, v82 /*v594*/
	ds_load_b128 v[188:191] /*v[444:447]*/, v82 /*v594*/ offset:32
	ds_load_b128 v[176:179] /*v[432:435]*/, v82 /*v594*/ offset:64
	ds_load_b128 v[180:183] /*v[436:439]*/, v82 /*v594*/ offset:96
	ds_load_b128 v[168:171] /*v[424:427]*/, v82 /*v594*/ offset:128
	ds_load_b128 v[172:175] /*v[428:431]*/, v82 /*v594*/ offset:160
	ds_load_b128 v[160:163] /*v[416:419]*/, v82 /*v594*/ offset:192
	ds_load_b128 v[164:167] /*v[420:423]*/, v82 /*v594*/ offset:224
	ds_load_b128 v[152:155] /*v[408:411]*/, v82 /*v594*/ offset:4352
	ds_load_b128 v[156:159] /*v[412:415]*/, v82 /*v594*/ offset:4384
	ds_load_b128 v[144:147] /*v[400:403]*/, v82 /*v594*/ offset:4416
	ds_load_b128 v[148:151] /*v[404:407]*/, v82 /*v594*/ offset:4448
	ds_load_b128 v[136:139] /*v[392:395]*/, v82 /*v594*/ offset:4480
	ds_load_b128 v[140:143] /*v[396:399]*/, v82 /*v594*/ offset:4512
	ds_load_b128 v[128:131] /*v[384:387]*/, v82 /*v594*/ offset:4544
	ds_load_b128 v[132:135] /*v[388:391]*/, v82 /*v594*/ offset:4576
	ds_load_b128 v[120:123] /*v[376:379]*/, v82 /*v594*/ offset:8704
	ds_load_b128 v[124:127] /*v[380:383]*/, v82 /*v594*/ offset:8736
	ds_load_b128 v[112:115] /*v[368:371]*/, v82 /*v594*/ offset:8768
	ds_load_b128 v[116:119] /*v[372:375]*/, v82 /*v594*/ offset:8800
	ds_load_b128 v[104:107] /*v[360:363]*/, v82 /*v594*/ offset:8832
	ds_load_b128 v[108:111] /*v[364:367]*/, v82 /*v594*/ offset:8864
	ds_load_b128 v[96:99] /*v[352:355]*/, v82 /*v594*/ offset:8896
	ds_load_b128 v[100:103] /*v[356:359]*/, v82 /*v594*/ offset:8928
	ds_load_b128 v[88:91] /*v[344:347]*/, v82 /*v594*/ offset:13056
	ds_load_b128 v[92:95] /*v[348:351]*/, v82 /*v594*/ offset:13088
	ds_load_b128 v[80:83] /*v[336:339]*/, v82 /*v594*/ offset:13120
	ds_load_b128 v[84:87] /*v[340:343]*/, v82 /*v594*/ offset:13152
	ds_load_b128 v[72:75] /*v[328:331]*/, v82 /*v594*/ offset:13184
	ds_load_b128 v[76:79] /*v[332:335]*/, v82 /*v594*/ offset:13216
	ds_load_b128 v[64:67] /*v[320:323]*/, v82 /*v594*/ offset:13248
	ds_load_b128 v[68:71] /*v[324:327]*/, v82 /*v594*/ offset:13280
	ds_load_b128 v[56:59] /*v[312:315]*/, v82 /*v594*/ offset:17408
	ds_load_b128 v[60:63] /*v[316:319]*/, v82 /*v594*/ offset:17440
	ds_load_b128 v[48:51] /*v[304:307]*/, v82 /*v594*/ offset:17472
	ds_load_b128 v[52:55] /*v[308:311]*/, v82 /*v594*/ offset:17504
	ds_load_b128 v[40:43] /*v[296:299]*/, v82 /*v594*/ offset:17536
	ds_load_b128 v[44:47] /*v[300:303]*/, v82 /*v594*/ offset:17568
	ds_load_b128 v[32:35] /*v[288:291]*/, v82 /*v594*/ offset:17600
	ds_load_b128 v[36:39] /*v[292:295]*/, v82 /*v594*/ offset:17632
	ds_load_b128 v[24:27] /*v[280:283]*/, v82 /*v594*/ offset:21760
	ds_load_b128 v[28:31] /*v[284:287]*/, v82 /*v594*/ offset:21792
	ds_load_b128 v[16:19] /*v[272:275]*/, v82 /*v594*/ offset:21824
	ds_load_b128 v[20:23] /*v[276:279]*/, v82 /*v594*/ offset:21856
	ds_load_b128 v[8:11] /*v[264:267]*/, v82 /*v594*/ offset:21888
	ds_load_b128 v[12:15] /*v[268:271]*/, v82 /*v594*/ offset:21920
	ds_load_b128 v[0:3] /*v[256:259]*/, v82 /*v594*/ offset:21952
	ds_load_b128 v[4:7] /*v[260:263]*/, v82 /*v594*/ offset:21984
	s_set_vgpr_msb 0x4202
	ds_load_b128 v[248:251], v82 /*v594*/ offset:26112
	ds_load_b128 v[252:255], v82 /*v594*/ offset:26144
	ds_load_b128 v[240:243], v82 /*v594*/ offset:26176
	ds_load_b128 v[244:247], v82 /*v594*/ offset:26208
	ds_load_b128 v[232:235], v82 /*v594*/ offset:26240
	ds_load_b128 v[236:239], v82 /*v594*/ offset:26272
	ds_load_b128 v[224:227], v82 /*v594*/ offset:26304
	ds_load_b128 v[228:231], v82 /*v594*/ offset:26336
	ds_load_b128 v[216:219], v82 /*v594*/ offset:30464
	ds_load_b128 v[220:223], v82 /*v594*/ offset:30496
	ds_load_b128 v[208:211], v82 /*v594*/ offset:30528
	ds_load_b128 v[212:215], v82 /*v594*/ offset:30560
	ds_load_b128 v[200:203], v82 /*v594*/ offset:30592
	ds_load_b128 v[204:207], v82 /*v594*/ offset:30624
	ds_load_b128 v[192:195], v82 /*v594*/ offset:30656
	ds_load_b128 v[196:199], v82 /*v594*/ offset:30688
	s_cmp_ge_i32 s65, s10
	s_set_vgpr_msb 0x200
	s_cbranch_scc1 .LBB0_28
	s_set_vgpr_msb 0x41
	v_med3_i32 v194 /*v450*/, s35, 0, 0x80
	s_ashr_i32 s49, s48, 31
	s_lshr_b32 s2, s65, 31
	s_mul_u64 s[6:7], s[48:49], s[62:63]
	s_add_co_i32 s2, s65, s2
	v_readfirstlane_b32 s5, v194 /*v450*/
	s_lshl_b64 s[6:7], s[6:7], 1
	s_and_b32 s2, s2, 0x7ffe
	s_add_nc_u64 s[22:23], s[40:41], s[6:7]
	s_mul_u64 s[6:7], s[48:49], s[60:61]
	s_sub_co_i32 s5, s5, s53
	s_lshl_b64 s[6:7], s[6:7], 1
	s_sub_co_i32 s2, s65, s2
	s_add_nc_u64 s[6:7], s[42:43], s[6:7]
	s_max_i32 s14, s5, 0
	s_lshl_b32 s2, s2, 17
	s_add_nc_u64 s[6:7], s[36:37], s[6:7]
	s_lshl_b32 s14, s14, 16
	s_or_b32 s5, s54, s2
	s_bitset1_b32 s7, 31
	s_addk_co_i32 s14, 0x7fff
	s_mov_b32 s21, s13
	tensor_load_to_lds s[4:7], s[12:19]
	s_add_nc_u64 s[6:7], s[38:39], s[22:23]
	s_or_b32 s5, s55, s2
	s_bitset1_b32 s7, 31
	s_mov_b32 s22, s14
	s_mov_b32 s23, s15
	s_mov_b32 s24, s16
	s_mov_b32 s27, s19
	tensor_load_to_lds s[4:7], s[20:27]
	s_set_vgpr_msb 0x4100
.LBB0_28:
	s_set_vgpr_msb 0x41
	s_wait_dscnt 0x3e
	v_wmma_f32_16x16x32_bf16 v[194:201] /*v[450:457]*/, v[184:191] /*v[440:447]*/, v[128:135], 0
	s_wait_dscnt 0x36
	v_wmma_f32_16x16x32_bf16 v[214:221] /*v[470:477]*/, v[152:159] /*v[408:415]*/, v[128:135], 0
	s_wait_dscnt 0x2e
	v_wmma_f32_16x16x32_bf16 v[232:239] /*v[488:495]*/, v[120:127] /*v[376:383]*/, v[128:135], 0
	s_wait_dscnt 0x26
	v_wmma_f32_16x16x32_bf16 v[250:257] /*v[506:513]*/, v[88:95] /*v[344:351]*/, v[128:135], 0
	s_set_vgpr_msb 0x4181
	s_wait_dscnt 0x1e
	v_wmma_f32_16x16x32_bf16 v[38:45] /*v[550:557]*/, v[56:63] /*v[312:319]*/, v[128:135], 0
	s_wait_dscnt 0x16
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[562:569]*/, v[24:31] /*v[280:287]*/, v[128:135], 0
	s_set_vgpr_msb 0x8180
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[570:577]*/, v[248:255], v[128:135], 0
	s_wait_alu depctr_vm_vsrc(6)
	s_set_vgpr_msb 0x8081
	v_wmma_f32_16x16x32_bf16 v[92:99] /*v[604:611]*/, v[184:191] /*v[440:447]*/, v[160:167], 0
	s_set_vgpr_msb 0x8151
	v_wmma_f32_16x16x32_bf16 v[194:201] /*v[450:457]*/, v[176:183] /*v[432:439]*/, v[136:143], v[194:201] /*v[450:457]*/
	s_set_vgpr_msb 0x5181
	v_wmma_f32_16x16x32_bf16 v[100:107] /*v[612:619]*/, v[152:159] /*v[408:415]*/, v[160:167], 0
	s_set_vgpr_msb 0x8151
	v_wmma_f32_16x16x32_bf16 v[214:221] /*v[470:477]*/, v[144:151] /*v[400:407]*/, v[136:143], v[214:221] /*v[470:477]*/
	s_set_vgpr_msb 0x5181
	v_wmma_f32_16x16x32_bf16 v[108:115] /*v[620:627]*/, v[120:127] /*v[376:383]*/, v[160:167], 0
	s_set_vgpr_msb 0x8151
	v_wmma_f32_16x16x32_bf16 v[232:239] /*v[488:495]*/, v[112:119] /*v[368:375]*/, v[136:143], v[232:239] /*v[488:495]*/
	s_set_vgpr_msb 0x5181
	v_wmma_f32_16x16x32_bf16 v[116:123] /*v[628:635]*/, v[88:95] /*v[344:351]*/, v[160:167], 0
	s_set_vgpr_msb 0x8151
	v_wmma_f32_16x16x32_bf16 v[250:257] /*v[506:513]*/, v[80:87] /*v[336:343]*/, v[136:143], v[250:257] /*v[506:513]*/
	s_set_vgpr_msb 0x51a1
	v_wmma_f32_16x16x32_bf16 v[124:131] /*v[636:643]*/, v[56:63] /*v[312:319]*/, v[160:167], 0
	v_wmma_f32_16x16x32_bf16 v[38:45] /*v[550:557]*/, v[48:55] /*v[304:311]*/, v[136:143], v[38:45] /*v[550:557]*/
	v_wmma_f32_16x16x32_bf16 v[132:139] /*v[644:651]*/, v[24:31] /*v[280:287]*/, v[160:167], 0
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[562:569]*/, v[16:23] /*v[272:279]*/, v[136:143], v[50:57] /*v[562:569]*/
	s_set_vgpr_msb 0xa1a0
	v_wmma_f32_16x16x32_bf16 v[140:147] /*v[652:659]*/, v[248:255], v[160:167], 0
	s_wait_dscnt 0xc
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[570:577]*/, v[240:247], v[136:143], v[58:65] /*v[570:577]*/
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x32_bf16 v[148:155] /*v[660:667]*/, v[216:223], v[128:135], 0
	v_wmma_f32_16x16x32_bf16 v[156:163] /*v[668:675]*/, v[216:223], v[160:167], 0
	s_set_vgpr_msb 0xa0a1
	v_wmma_f32_16x16x32_bf16 v[92:99] /*v[604:611]*/, v[176:183] /*v[432:439]*/, v[168:175], v[92:99] /*v[604:611]*/
	s_set_vgpr_msb 0xa151
	v_wmma_f32_16x16x32_bf16 v[194:201] /*v[450:457]*/, v[168:175] /*v[424:431]*/, v[144:151], v[194:201] /*v[450:457]*/
	s_set_vgpr_msb 0x51a1
	v_wmma_f32_16x16x32_bf16 v[100:107] /*v[612:619]*/, v[144:151] /*v[400:407]*/, v[168:175], v[100:107] /*v[612:619]*/
	s_set_vgpr_msb 0xa151
	v_wmma_f32_16x16x32_bf16 v[214:221] /*v[470:477]*/, v[136:143] /*v[392:399]*/, v[144:151], v[214:221] /*v[470:477]*/
	s_set_vgpr_msb 0x51a1
	v_wmma_f32_16x16x32_bf16 v[108:115] /*v[620:627]*/, v[112:119] /*v[368:375]*/, v[168:175], v[108:115] /*v[620:627]*/
	s_set_vgpr_msb 0xa151
	v_wmma_f32_16x16x32_bf16 v[232:239] /*v[488:495]*/, v[104:111] /*v[360:367]*/, v[144:151], v[232:239] /*v[488:495]*/
	s_set_vgpr_msb 0x51a1
	v_wmma_f32_16x16x32_bf16 v[116:123] /*v[628:635]*/, v[80:87] /*v[336:343]*/, v[168:175], v[116:123] /*v[628:635]*/
	s_set_vgpr_msb 0xa151
	v_wmma_f32_16x16x32_bf16 v[250:257] /*v[506:513]*/, v[72:79] /*v[328:335]*/, v[144:151], v[250:257] /*v[506:513]*/
	s_set_vgpr_msb 0x51a1
	v_wmma_f32_16x16x32_bf16 v[124:131] /*v[636:643]*/, v[48:55] /*v[304:311]*/, v[168:175], v[124:131] /*v[636:643]*/
	v_wmma_f32_16x16x32_bf16 v[38:45] /*v[550:557]*/, v[40:47] /*v[296:303]*/, v[144:151], v[38:45] /*v[550:557]*/
	v_wmma_f32_16x16x32_bf16 v[132:139] /*v[644:651]*/, v[16:23] /*v[272:279]*/, v[168:175], v[132:139] /*v[644:651]*/
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[562:569]*/, v[8:15] /*v[264:271]*/, v[144:151], v[50:57] /*v[562:569]*/
	s_set_vgpr_msb 0xa1a0
	v_wmma_f32_16x16x32_bf16 v[140:147] /*v[652:659]*/, v[240:247], v[168:175], v[140:147] /*v[652:659]*/
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[570:577]*/, v[232:239], v[144:151], v[58:65] /*v[570:577]*/
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x32_bf16 v[148:155] /*v[660:667]*/, v[208:215], v[136:143], v[148:155] /*v[660:667]*/
	v_wmma_f32_16x16x32_bf16 v[156:163] /*v[668:675]*/, v[208:215], v[168:175], v[156:163] /*v[668:675]*/
	s_set_vgpr_msb 0xa0a1
	v_wmma_f32_16x16x32_bf16 v[92:99] /*v[604:611]*/, v[168:175] /*v[424:431]*/, v[176:183], v[92:99] /*v[604:611]*/
	s_set_vgpr_msb 0xa151
	v_wmma_f32_16x16x32_bf16 v[194:201] /*v[450:457]*/, v[160:167] /*v[416:423]*/, v[152:159], v[194:201] /*v[450:457]*/
	s_set_vgpr_msb 0x51a1
	v_wmma_f32_16x16x32_bf16 v[100:107] /*v[612:619]*/, v[136:143] /*v[392:399]*/, v[176:183], v[100:107] /*v[612:619]*/
	s_set_vgpr_msb 0xa151
	v_wmma_f32_16x16x32_bf16 v[214:221] /*v[470:477]*/, v[128:135] /*v[384:391]*/, v[152:159], v[214:221] /*v[470:477]*/
	s_set_vgpr_msb 0x51a1
	v_wmma_f32_16x16x32_bf16 v[108:115] /*v[620:627]*/, v[104:111] /*v[360:367]*/, v[176:183], v[108:115] /*v[620:627]*/
	s_set_vgpr_msb 0xa151
	v_wmma_f32_16x16x32_bf16 v[232:239] /*v[488:495]*/, v[96:103] /*v[352:359]*/, v[152:159], v[232:239] /*v[488:495]*/
	s_set_vgpr_msb 0x51a1
	v_wmma_f32_16x16x32_bf16 v[116:123] /*v[628:635]*/, v[72:79] /*v[328:335]*/, v[176:183], v[116:123] /*v[628:635]*/
	s_set_vgpr_msb 0xa151
	v_wmma_f32_16x16x32_bf16 v[250:257] /*v[506:513]*/, v[64:71] /*v[320:327]*/, v[152:159], v[250:257] /*v[506:513]*/
	s_set_vgpr_msb 0x51a1
	v_wmma_f32_16x16x32_bf16 v[124:131] /*v[636:643]*/, v[40:47] /*v[296:303]*/, v[176:183], v[124:131] /*v[636:643]*/
	v_wmma_f32_16x16x32_bf16 v[38:45] /*v[550:557]*/, v[32:39] /*v[288:295]*/, v[152:159], v[38:45] /*v[550:557]*/
	v_wmma_f32_16x16x32_bf16 v[132:139] /*v[644:651]*/, v[8:15] /*v[264:271]*/, v[176:183], v[132:139] /*v[644:651]*/
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[562:569]*/, v[0:7] /*v[256:263]*/, v[152:159], v[50:57] /*v[562:569]*/
	s_set_vgpr_msb 0xa1a0
	v_wmma_f32_16x16x32_bf16 v[140:147] /*v[652:659]*/, v[232:239], v[176:183], v[140:147] /*v[652:659]*/
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[570:577]*/, v[224:231], v[152:159], v[58:65] /*v[570:577]*/
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[148:155] /*v[660:667]*/, v[200:207], v[144:151], v[148:155] /*v[660:667]*/
	v_wmma_f32_16x16x32_bf16 v[156:163] /*v[668:675]*/, v[200:207], v[176:183], v[156:163] /*v[668:675]*/
	s_set_vgpr_msb 0xa0a1
	v_wmma_f32_16x16x32_bf16 v[92:99] /*v[604:611]*/, v[160:167] /*v[416:423]*/, v[184:191], v[92:99] /*v[604:611]*/
	v_wmma_f32_16x16x32_bf16 v[100:107] /*v[612:619]*/, v[128:135] /*v[384:391]*/, v[184:191], v[100:107] /*v[612:619]*/
	v_wmma_f32_16x16x32_bf16 v[108:115] /*v[620:627]*/, v[96:103] /*v[352:359]*/, v[184:191], v[108:115] /*v[620:627]*/
	v_wmma_f32_16x16x32_bf16 v[116:123] /*v[628:635]*/, v[64:71] /*v[320:327]*/, v[184:191], v[116:123] /*v[628:635]*/
	v_wmma_f32_16x16x32_bf16 v[124:131] /*v[636:643]*/, v[32:39] /*v[288:295]*/, v[184:191], v[124:131] /*v[636:643]*/
	v_wmma_f32_16x16x32_bf16 v[132:139] /*v[644:651]*/, v[0:7] /*v[256:263]*/, v[184:191], v[132:139] /*v[644:651]*/
	s_set_vgpr_msb 0xa1a0
	v_wmma_f32_16x16x32_bf16 v[140:147] /*v[652:659]*/, v[224:231], v[184:191], v[140:147] /*v[652:659]*/
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[148:155] /*v[660:667]*/, v[192:199], v[152:159], v[148:155] /*v[660:667]*/
	v_wmma_f32_16x16x32_bf16 v[156:163] /*v[668:675]*/, v[192:199], v[184:191], v[156:163] /*v[668:675]*/
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0xa042
	ds_load_tr16_b128 v[184:187] /*v[440:443]*/, v90 /*v602*/
	ds_load_tr16_b128 v[152:155] /*v[408:411]*/, v90 /*v602*/ offset:32
	ds_load_tr16_b128 v[188:191] /*v[444:447]*/, v90 /*v602*/ offset:4608
	ds_load_tr16_b128 v[156:159] /*v[412:415]*/, v90 /*v602*/ offset:4640
	ds_load_tr16_b128 v[176:179] /*v[432:435]*/, v90 /*v602*/ offset:9216
	ds_load_tr16_b128 v[144:147] /*v[400:403]*/, v90 /*v602*/ offset:9248
	ds_load_tr16_b128 v[180:183] /*v[436:439]*/, v90 /*v602*/ offset:13824
	ds_load_tr16_b128 v[148:151] /*v[404:407]*/, v90 /*v602*/ offset:13856
	ds_load_tr16_b128 v[168:171] /*v[424:427]*/, v90 /*v602*/ offset:18432
	ds_load_tr16_b128 v[136:139] /*v[392:395]*/, v90 /*v602*/ offset:18464
	ds_load_tr16_b128 v[172:175] /*v[428:431]*/, v90 /*v602*/ offset:23040
	ds_load_tr16_b128 v[140:143] /*v[396:399]*/, v90 /*v602*/ offset:23072
	ds_load_tr16_b128 v[160:163] /*v[416:419]*/, v90 /*v602*/ offset:27648
	ds_load_tr16_b128 v[128:131] /*v[384:387]*/, v90 /*v602*/ offset:27680
	ds_load_tr16_b128 v[164:167] /*v[420:423]*/, v90 /*v602*/ offset:32256
	ds_load_tr16_b128 v[132:135] /*v[388:391]*/, v90 /*v602*/ offset:32288
	ds_load_tr16_b128 v[120:123] /*v[376:379]*/, v90 /*v602*/ offset:64
	ds_load_tr16_b128 v[88:91] /*v[344:347]*/, v90 /*v602*/ offset:96
	ds_load_tr16_b128 v[124:127] /*v[380:383]*/, v90 /*v602*/ offset:4672
	ds_load_tr16_b128 v[92:95] /*v[348:351]*/, v90 /*v602*/ offset:4704
	ds_load_tr16_b128 v[112:115] /*v[368:371]*/, v90 /*v602*/ offset:9280
	ds_load_tr16_b128 v[80:83] /*v[336:339]*/, v90 /*v602*/ offset:9312
	ds_load_tr16_b128 v[116:119] /*v[372:375]*/, v90 /*v602*/ offset:13888
	ds_load_tr16_b128 v[84:87] /*v[340:343]*/, v90 /*v602*/ offset:13920
	ds_load_tr16_b128 v[104:107] /*v[360:363]*/, v90 /*v602*/ offset:18496
	ds_load_tr16_b128 v[72:75] /*v[328:331]*/, v90 /*v602*/ offset:18528
	ds_load_tr16_b128 v[108:111] /*v[364:367]*/, v90 /*v602*/ offset:23104
	ds_load_tr16_b128 v[76:79] /*v[332:335]*/, v90 /*v602*/ offset:23136
	ds_load_tr16_b128 v[96:99] /*v[352:355]*/, v90 /*v602*/ offset:27712
	ds_load_tr16_b128 v[64:67] /*v[320:323]*/, v90 /*v602*/ offset:27744
	ds_load_tr16_b128 v[100:103] /*v[356:359]*/, v90 /*v602*/ offset:32320
	ds_load_tr16_b128 v[68:71] /*v[324:327]*/, v90 /*v602*/ offset:32352
	ds_load_tr16_b128 v[56:59] /*v[312:315]*/, v90 /*v602*/ offset:128
	ds_load_tr16_b128 v[24:27] /*v[280:283]*/, v90 /*v602*/ offset:160
	ds_load_tr16_b128 v[60:63] /*v[316:319]*/, v90 /*v602*/ offset:4736
	ds_load_tr16_b128 v[28:31] /*v[284:287]*/, v90 /*v602*/ offset:4768
	ds_load_tr16_b128 v[48:51] /*v[304:307]*/, v90 /*v602*/ offset:9344
	ds_load_tr16_b128 v[16:19] /*v[272:275]*/, v90 /*v602*/ offset:9376
	ds_load_tr16_b128 v[52:55] /*v[308:311]*/, v90 /*v602*/ offset:13952
	ds_load_tr16_b128 v[20:23] /*v[276:279]*/, v90 /*v602*/ offset:13984
	ds_load_tr16_b128 v[40:43] /*v[296:299]*/, v90 /*v602*/ offset:18560
	ds_load_tr16_b128 v[8:11] /*v[264:267]*/, v90 /*v602*/ offset:18592
	ds_load_tr16_b128 v[44:47] /*v[300:303]*/, v90 /*v602*/ offset:23168
	ds_load_tr16_b128 v[12:15] /*v[268:271]*/, v90 /*v602*/ offset:23200
	ds_load_tr16_b128 v[32:35] /*v[288:291]*/, v90 /*v602*/ offset:27776
	ds_load_tr16_b128 v[0:3] /*v[256:259]*/, v90 /*v602*/ offset:27808
	ds_load_tr16_b128 v[36:39] /*v[292:295]*/, v90 /*v602*/ offset:32384
	ds_load_tr16_b128 v[4:7] /*v[260:263]*/, v90 /*v602*/ offset:32416
	s_set_vgpr_msb 0x4202
	ds_load_tr16_b128 v[248:251], v90 /*v602*/ offset:192
	ds_load_tr16_b128 v[216:219], v90 /*v602*/ offset:224
	ds_load_tr16_b128 v[252:255], v90 /*v602*/ offset:4800
	ds_load_tr16_b128 v[220:223], v90 /*v602*/ offset:4832
	ds_load_tr16_b128 v[240:243], v90 /*v602*/ offset:9408
	ds_load_tr16_b128 v[208:211], v90 /*v602*/ offset:9440
	ds_load_tr16_b128 v[244:247], v90 /*v602*/ offset:14016
	ds_load_tr16_b128 v[212:215], v90 /*v602*/ offset:14048
	ds_load_tr16_b128 v[232:235], v90 /*v602*/ offset:18624
	ds_load_tr16_b128 v[200:203], v90 /*v602*/ offset:18656
	ds_load_tr16_b128 v[236:239], v90 /*v602*/ offset:23232
	ds_load_tr16_b128 v[204:207], v90 /*v602*/ offset:23264
	ds_load_tr16_b128 v[224:227], v90 /*v602*/ offset:27840
	ds_load_tr16_b128 v[192:195], v90 /*v602*/ offset:27872
	ds_load_tr16_b128 v[228:231], v90 /*v602*/ offset:32448
	ds_load_tr16_b128 v[196:199], v90 /*v602*/ offset:32480
	s_set_vgpr_msb 0x255
	v_max3_num_f32 v202 /*v458*/, v194 /*v450*/, v195 /*v451*/, v196 /*v452*/
	s_set_vgpr_msb 0x556a
	v_max3_num_f32 v203 /*v459*/, v92 /*v604*/, v93 /*v605*/, v94 /*v606*/
	s_set_vgpr_msb 0x6a55
	v_max3_num_f32 v204 /*v460*/, v197 /*v453*/, v198 /*v454*/, v199 /*v455*/
	s_set_vgpr_msb 0x556a
	v_max3_num_f32 v205 /*v461*/, v95 /*v607*/, v96 /*v608*/, v97 /*v609*/
	s_set_vgpr_msb 0x6a55
	v_max3_num_f32 v206 /*v462*/, v200 /*v456*/, v201 /*v457*/, v214 /*v470*/
	s_set_vgpr_msb 0x556a
	v_max3_num_f32 v207 /*v463*/, v98 /*v610*/, v99 /*v611*/, v100 /*v612*/
	s_set_vgpr_msb 0x6a55
	v_max3_num_f32 v208 /*v464*/, v215 /*v471*/, v216 /*v472*/, v217 /*v473*/
	v_max3_num_f32 v210 /*v466*/, v218 /*v474*/, v219 /*v475*/, v220 /*v476*/
	v_max3_num_f32 v212 /*v468*/, v221 /*v477*/, v232 /*v488*/, v233 /*v489*/
	v_max3_num_f32 v222 /*v478*/, v234 /*v490*/, v235 /*v491*/, v236 /*v492*/
	v_max3_num_f32 v224 /*v480*/, v237 /*v493*/, v238 /*v494*/, v239 /*v495*/
	v_max3_num_f32 v226 /*v482*/, v250 /*v506*/, v251 /*v507*/, v252 /*v508*/
	v_max3_num_f32 v228 /*v484*/, v253 /*v509*/, v254 /*v510*/, v255 /*v511*/
	s_set_vgpr_msb 0x556a
	v_max3_num_f32 v230 /*v486*/, v0 /*v512*/, v1 /*v513*/, v38 /*v550*/
	v_max3_num_f32 v240 /*v496*/, v39 /*v551*/, v40 /*v552*/, v41 /*v553*/
	v_max3_num_f32 v242 /*v498*/, v42 /*v554*/, v43 /*v555*/, v44 /*v556*/
	v_max3_num_f32 v244 /*v500*/, v45 /*v557*/, v50 /*v562*/, v51 /*v563*/
	v_max3_num_f32 v246 /*v502*/, v52 /*v564*/, v53 /*v565*/, v54 /*v566*/
	v_max3_num_f32 v248 /*v504*/, v55 /*v567*/, v56 /*v568*/, v57 /*v569*/
	s_set_vgpr_msb 0x6aaa
	v_max3_num_f32 v2 /*v514*/, v58 /*v570*/, v59 /*v571*/, v60 /*v572*/
	v_max3_num_f32 v4 /*v516*/, v61 /*v573*/, v62 /*v574*/, v63 /*v575*/
	v_dual_max_num_f32 v6 /*v518*/, v64 /*v576*/, v65 /*v577*/ :: v_dual_max_num_f32 v7 /*v519*/, v146 /*v658*/, v147 /*v659*/
	v_max3_num_f32 v8 /*v520*/, v149 /*v661*/, v150 /*v662*/, v151 /*v663*/
	s_set_vgpr_msb 0xaa6a
	v_max3_num_f32 v209 /*v465*/, v101 /*v613*/, v102 /*v614*/, v103 /*v615*/
	v_max3_num_f32 v211 /*v467*/, v104 /*v616*/, v105 /*v617*/, v106 /*v618*/
	v_max3_num_f32 v213 /*v469*/, v107 /*v619*/, v108 /*v620*/, v109 /*v621*/
	v_max3_num_f32 v223 /*v479*/, v110 /*v622*/, v111 /*v623*/, v112 /*v624*/
	v_max3_num_f32 v225 /*v481*/, v113 /*v625*/, v114 /*v626*/, v115 /*v627*/
	v_max3_num_f32 v227 /*v483*/, v116 /*v628*/, v117 /*v629*/, v118 /*v630*/
	v_max3_num_f32 v229 /*v485*/, v119 /*v631*/, v120 /*v632*/, v121 /*v633*/
	v_max3_num_f32 v231 /*v487*/, v122 /*v634*/, v123 /*v635*/, v124 /*v636*/
	v_max3_num_f32 v241 /*v497*/, v125 /*v637*/, v126 /*v638*/, v127 /*v639*/
	v_max3_num_f32 v243 /*v499*/, v128 /*v640*/, v129 /*v641*/, v130 /*v642*/
	v_max3_num_f32 v245 /*v501*/, v131 /*v643*/, v132 /*v644*/, v133 /*v645*/
	v_max3_num_f32 v247 /*v503*/, v134 /*v646*/, v135 /*v647*/, v136 /*v648*/
	v_max3_num_f32 v249 /*v505*/, v137 /*v649*/, v138 /*v650*/, v139 /*v651*/
	s_set_vgpr_msb 0x6aaa
	v_max3_num_f32 v3 /*v515*/, v140 /*v652*/, v141 /*v653*/, v142 /*v654*/
	v_max3_num_f32 v5 /*v517*/, v143 /*v655*/, v144 /*v656*/, v145 /*v657*/
	v_max3_num_f32 v9 /*v521*/, v157 /*v669*/, v158 /*v670*/, v159 /*v671*/
	v_max3_num_f32 v10 /*v522*/, v152 /*v664*/, v153 /*v665*/, v154 /*v666*/
	s_set_vgpr_msb 0xaa55
	v_max3_num_f32 v202 /*v458*/, v202 /*v458*/, v204 /*v460*/, v206 /*v462*/
	v_max3_num_f32 v203 /*v459*/, v203 /*v459*/, v205 /*v461*/, v207 /*v463*/
	v_max3_num_f32 v204 /*v460*/, v208 /*v464*/, v210 /*v466*/, v212 /*v468*/
	v_max3_num_f32 v205 /*v461*/, v222 /*v478*/, v224 /*v480*/, v226 /*v482*/
	v_max3_num_f32 v206 /*v462*/, v228 /*v484*/, v230 /*v486*/, v240 /*v496*/
	v_max3_num_f32 v207 /*v463*/, v242 /*v498*/, v244 /*v500*/, v246 /*v502*/
	s_set_vgpr_msb 0x5569
	v_max3_num_f32 v208 /*v464*/, v248 /*v504*/, v2 /*v514*/, v4 /*v516*/
	s_set_vgpr_msb 0x696a
	v_max3_num_f32 v210 /*v466*/, v6 /*v518*/, v148 /*v660*/, v8 /*v520*/
	s_set_vgpr_msb 0x6aaa
	v_max3_num_f32 v11 /*v523*/, v160 /*v672*/, v161 /*v673*/, v162 /*v674*/
	s_set_vgpr_msb 0xaa55
	v_max3_num_f32 v209 /*v465*/, v209 /*v465*/, v211 /*v467*/, v213 /*v469*/
	v_max3_num_f32 v211 /*v467*/, v223 /*v479*/, v225 /*v481*/, v227 /*v483*/
	v_max3_num_f32 v202 /*v458*/, v202 /*v458*/, v204 /*v460*/, v205 /*v461*/
	v_max3_num_f32 v204 /*v460*/, v206 /*v462*/, v207 /*v463*/, v208 /*v464*/
	s_set_vgpr_msb 0x5569
	v_max3_num_f32 v205 /*v461*/, v210 /*v466*/, v10 /*v522*/, v155 /*v667*/
	s_set_vgpr_msb 0x6955
	v_max3_num_f32 v206 /*v462*/, v229 /*v485*/, v231 /*v487*/, v241 /*v497*/
	v_max3_num_f32 v207 /*v463*/, v243 /*v499*/, v245 /*v501*/, v247 /*v503*/
	s_set_vgpr_msb 0x5569
	v_max3_num_f32 v208 /*v464*/, v249 /*v505*/, v3 /*v515*/, v5 /*v517*/
	s_set_vgpr_msb 0x696a
	v_max3_num_f32 v210 /*v466*/, v7 /*v519*/, v156 /*v668*/, v9 /*v521*/
	s_set_vgpr_msb 0x6a55
	v_max3_num_f32 v202 /*v458*/, v202 /*v458*/, v204 /*v460*/, v205 /*v461*/
	v_max3_num_f32 v203 /*v459*/, v203 /*v459*/, v209 /*v465*/, v211 /*v467*/
	v_max3_num_f32 v204 /*v460*/, v206 /*v462*/, v207 /*v463*/, v208 /*v464*/
	s_set_vgpr_msb 0x5569
	v_max3_num_f32 v205 /*v461*/, v210 /*v466*/, v11 /*v523*/, v163 /*v675*/
	s_set_vgpr_msb 0x6955
	v_max3_num_f32 v203 /*v459*/, v203 /*v459*/, v204 /*v460*/, v205 /*v461*/
	v_dual_mov_b32 v206 /*v462*/, v202 /*v458*/ :: v_dual_mov_b32 v204 /*v460*/, v203 /*v459*/
	v_permlanex16_b32 v206 /*v462*/, v206 /*v462*/, s56, 0xfedcba98
	v_permlanex16_b32 v204 /*v460*/, v204 /*v460*/, s56, 0xfedcba98
	v_dual_max_num_f32 v202 /*v458*/, v202 /*v458*/, v206 /*v462*/ :: v_dual_max_num_f32 v203 /*v459*/, v203 /*v459*/, v204 /*v460*/
	s_set_vgpr_msb 0x5549
	v_sub_f32_e32 v205 /*v461*/, v202 /*v458*/, v66 /*v578*/
	v_max_num_f32_e32 v202 /*v458*/, v202 /*v458*/, v66 /*v578*/
	v_sub_f32_e32 v204 /*v460*/, v203 /*v459*/, v71 /*v583*/
	s_set_vgpr_msb 0x4904
	v_cmp_lt_f32_e32 vcc_lo, 0x41000000, v205 /*v461*/
	s_cmp_eq_u32 vcc_lo, 0
	s_cselect_b32 s2, -1, 0
	s_set_vgpr_msb 0x489
	v_cndmask_b32_e64 v86 /*v598*/, v202 /*v458*/, v66 /*v578*/, s2
	s_set_vgpr_msb 0x8946
	v_cmp_lt_f32_e64 s2, 0x41000000, v204 /*v460*/
	v_max_num_f32_e32 v202 /*v458*/, v71 /*v583*/, v203 /*v459*/
	s_cmp_lg_u32 s2, 0
	s_cselect_b32 s5, -1, 0
	s_cmp_eq_u32 s2, 0
	s_cselect_b32 s2, -1, 0
	s_set_vgpr_msb 0x4689
	v_cndmask_b32_e64 v85 /*v597*/, v202 /*v458*/, v71 /*v583*/, s2
	v_mul_f32_e32 v68 /*v580*/, 0xbfb8aa3b, v86 /*v598*/
	v_mul_f32_e32 v70 /*v582*/, 0xbfb8aa3b, v85 /*v597*/
	s_set_vgpr_msb 0x8961
	v_pk_fma_f32 v[204:205] /*v[460:461]*/, v[198:199] /*v[454:455]*/, s[52:53], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[200:201] /*v[456:457]*/, v[200:201] /*v[456:457]*/, s[52:53], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[208:209] /*v[464:465]*/, v[214:215] /*v[470:471]*/, s[52:53], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[210:211] /*v[466:467]*/, v[234:235] /*v[490:491]*/, s[52:53], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[222:223] /*v[478:479]*/, v[238:239] /*v[494:495]*/, s[52:53], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v206 /*v462*/, v204 /*v460*/
	v_exp_f32_e32 v212 /*v468*/, v205 /*v461*/
	v_exp_f32_e32 v214 /*v470*/, v200 /*v456*/
	v_pk_fma_f32 v[204:205] /*v[460:461]*/, v[216:217] /*v[472:473]*/, s[52:53], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v226 /*v482*/, v201 /*v457*/
	v_exp_f32_e32 v230 /*v486*/, v208 /*v464*/
	v_pk_fma_f32 v[200:201] /*v[456:457]*/, v[218:219] /*v[474:475]*/, s[52:53], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v242 /*v498*/, v209 /*v465*/
	v_nop
	v_pk_fma_f32 v[208:209] /*v[464:465]*/, v[220:221] /*v[476:477]*/, s[52:53], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[218:219] /*v[474:475]*/, v[236:237] /*v[492:493]*/, s[52:53], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6162
	v_pk_fma_f32 v[224:225] /*v[480:481]*/, v[42:43] /*v[554:555]*/, s[52:53], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6241
	v_exp_f32_e32 v248 /*v504*/, v204 /*v460*/
	s_set_vgpr_msb 0x4181
	v_exp_f32_e32 v2 /*v514*/, v205 /*v461*/
	v_nop
	s_set_vgpr_msb 0x8161
	v_pk_fma_f32 v[204:205] /*v[460:461]*/, v[232:233] /*v[488:489]*/, s[52:53], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6181
	v_exp_f32_e32 v20 /*v532*/, v208 /*v464*/
	s_set_vgpr_msb 0x8161
	v_exp_f32_e32 v208 /*v464*/, v210 /*v466*/
	v_exp_f32_e32 v216 /*v472*/, v211 /*v467*/
	v_exp_f32_e32 v220 /*v476*/, v218 /*v474*/
	v_pk_fma_f32 v[210:211] /*v[466:467]*/, v[250:251] /*v[506:507]*/, s[52:53], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v228 /*v484*/, v219 /*v475*/
	v_exp_f32_e32 v232 /*v488*/, v222 /*v478*/
	v_pk_fma_f32 v[218:219] /*v[474:475]*/, v[252:253] /*v[508:509]*/, s[52:53], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v244 /*v500*/, v223 /*v479*/
	v_nop
	v_pk_fma_f32 v[222:223] /*v[478:479]*/, v[254:255] /*v[510:511]*/, s[52:53], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6162
	v_pk_fma_f32 v[236:237] /*v[492:493]*/, v[44:45] /*v[556:557]*/, s[52:53], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6241
	v_exp_f32_e32 v238 /*v494*/, v224 /*v480*/
	s_set_vgpr_msb 0x4162
	v_pk_fma_f32 v[240:241] /*v[496:497]*/, v[50:51] /*v[562:563]*/, s[52:53], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6241
	v_exp_f32_e32 v246 /*v502*/, v225 /*v481*/
	v_nop
	s_set_vgpr_msb 0x4162
	v_pk_fma_f32 v[224:225] /*v[480:481]*/, v[52:53] /*v[564:565]*/, s[52:53], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6261
	v_pk_fma_f32 v[194:195] /*v[450:451]*/, v[194:195] /*v[450:451]*/, s[52:53], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[202:203] /*v[458:459]*/, v[196:197] /*v[452:453]*/, s[52:53], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v250 /*v506*/, v210 /*v466*/
	s_set_vgpr_msb 0x6181
	v_exp_f32_e32 v4 /*v516*/, v211 /*v467*/
	v_exp_f32_e32 v12 /*v524*/, v218 /*v474*/
	s_set_vgpr_msb 0x8162
	v_pk_fma_f32 v[210:211] /*v[466:467]*/, v[0:1] /*v[512:513]*/, s[52:53], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6281
	v_exp_f32_e32 v22 /*v534*/, v219 /*v475*/
	v_exp_f32_e32 v28 /*v540*/, v222 /*v478*/
	s_set_vgpr_msb 0x8162
	v_pk_fma_f32 v[218:219] /*v[474:475]*/, v[38:39] /*v[550:551]*/, s[52:53], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6281
	v_exp_f32_e32 v36 /*v548*/, v223 /*v479*/
	v_nop
	s_set_vgpr_msb 0x8162
	v_pk_fma_f32 v[222:223] /*v[478:479]*/, v[40:41] /*v[552:553]*/, s[52:53], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6241
	v_exp_f32_e32 v252 /*v508*/, v236 /*v492*/
	s_set_vgpr_msb 0x4181
	v_exp_f32_e32 v6 /*v518*/, v237 /*v493*/
	v_exp_f32_e32 v14 /*v526*/, v240 /*v496*/
	s_set_vgpr_msb 0x8162
	v_pk_fma_f32 v[236:237] /*v[492:493]*/, v[54:55] /*v[566:567]*/, s[52:53], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6281
	v_exp_f32_e32 v24 /*v536*/, v241 /*v497*/
	v_exp_f32_e32 v30 /*v542*/, v224 /*v480*/
	s_set_vgpr_msb 0x8162
	v_pk_fma_f32 v[240:241] /*v[496:497]*/, v[56:57] /*v[568:569]*/, s[52:53], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6281
	v_exp_f32_e32 v40 /*v552*/, v225 /*v481*/
	v_nop
	s_set_vgpr_msb 0x8162
	v_pk_fma_f32 v[224:225] /*v[480:481]*/, v[58:59] /*v[570:571]*/, s[52:53], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[254:255] /*v[510:511]*/, v[60:61] /*v[572:573]*/, s[52:53], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x62a2
	v_pk_fma_f32 v[0:1] /*v[512:513]*/, v[62:63] /*v[574:575]*/, s[52:53], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[16:17] /*v[528:529]*/, v[64:65] /*v[576:577]*/, s[52:53], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[32:33] /*v[544:545]*/, v[148:149] /*v[660:661]*/, s[52:53], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[46:47] /*v[558:559]*/, v[150:151] /*v[662:663]*/, s[52:53], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[56:57] /*v[568:569]*/, v[152:153] /*v[664:665]*/, s[52:53], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[62:63] /*v[574:575]*/, v[154:155] /*v[666:667]*/, s[52:53], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[68:69] /*v[580:581]*/, v[92:93] /*v[604:605]*/, s[52:53], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[92:93] /*v[604:605]*/, v[96:97] /*v[608:609]*/, s[52:53], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa241
	v_exp_f32_e32 v196 /*v452*/, v195 /*v451*/
	s_set_vgpr_msb 0x41a2
	v_pk_fma_f32 v[88:89] /*v[600:601]*/, v[94:95] /*v[606:607]*/, s[52:53], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa241
	v_exp_f32_e32 v198 /*v454*/, v202 /*v458*/
	s_set_vgpr_msb 0x4142
	v_exp_f32_e32 v195 /*v451*/, v68 /*v580*/
	v_exp_f32_e32 v197 /*v453*/, v69 /*v581*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[68:69] /*v[580:581]*/, v[98:99] /*v[610:611]*/, s[52:53], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa242
	v_exp_f32_e32 v207 /*v463*/, v92 /*v604*/
	v_exp_f32_e32 v213 /*v469*/, v93 /*v605*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[92:93] /*v[604:605]*/, v[102:103] /*v[614:615]*/, s[52:53], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa241
	v_exp_f32_e32 v202 /*v458*/, v203 /*v459*/
	s_set_vgpr_msb 0x4142
	v_exp_f32_e32 v215 /*v471*/, v68 /*v580*/
	v_exp_f32_e32 v227 /*v483*/, v69 /*v581*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[68:69] /*v[580:581]*/, v[104:105] /*v[616:617]*/, s[52:53], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa242
	v_exp_f32_e32 v249 /*v505*/, v92 /*v604*/
	s_set_vgpr_msb 0x42a2
	v_exp_f32_e32 v3 /*v515*/, v93 /*v605*/
	v_nop
	v_pk_fma_f32 v[92:93] /*v[604:605]*/, v[108:109] /*v[620:621]*/, s[52:53], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa242
	v_exp_f32_e32 v199 /*v455*/, v88 /*v600*/
	v_exp_f32_e32 v203 /*v459*/, v89 /*v601*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[88:89] /*v[600:601]*/, v[100:101] /*v[612:613]*/, s[52:53], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa281
	v_exp_f32_e32 v10 /*v522*/, v200 /*v456*/
	v_exp_f32_e32 v18 /*v530*/, v201 /*v457*/
	s_set_vgpr_msb 0x8141
	v_exp_f32_e32 v200 /*v456*/, v204 /*v460*/
	v_exp_f32_e32 v204 /*v460*/, v205 /*v461*/
	s_set_vgpr_msb 0x41a2
	v_exp_f32_e32 v11 /*v523*/, v68 /*v580*/
	v_exp_f32_e32 v19 /*v531*/, v69 /*v581*/
	v_nop
	v_pk_fma_f32 v[68:69] /*v[580:581]*/, v[110:111] /*v[622:623]*/, s[52:53], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa242
	v_exp_f32_e32 v201 /*v457*/, v92 /*v604*/
	v_exp_f32_e32 v205 /*v461*/, v93 /*v605*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[92:93] /*v[604:605]*/, v[114:115] /*v[626:627]*/, s[52:53], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa242
	v_exp_f32_e32 v231 /*v487*/, v88 /*v600*/
	v_exp_f32_e32 v243 /*v499*/, v89 /*v601*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[88:89] /*v[600:601]*/, v[106:107] /*v[618:619]*/, s[52:53], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa281
	v_exp_f32_e32 v34 /*v546*/, v209 /*v465*/
	s_set_vgpr_msb 0x8142
	v_exp_f32_e32 v209 /*v465*/, v68 /*v580*/
	v_exp_f32_e32 v217 /*v473*/, v69 /*v581*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[68:69] /*v[580:581]*/, v[116:117] /*v[628:629]*/, s[52:53], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa242
	v_exp_f32_e32 v233 /*v489*/, v92 /*v604*/
	v_exp_f32_e32 v245 /*v501*/, v93 /*v605*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[92:93] /*v[604:605]*/, v[120:121] /*v[632:633]*/, s[52:53], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v21 /*v533*/, v88 /*v600*/
	v_exp_f32_e32 v35 /*v547*/, v89 /*v601*/
	v_nop
	v_pk_fma_f32 v[88:89] /*v[600:601]*/, v[112:113] /*v[624:625]*/, s[52:53], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa242
	v_exp_f32_e32 v251 /*v507*/, v68 /*v580*/
	s_set_vgpr_msb 0x42a2
	v_exp_f32_e32 v5 /*v517*/, v69 /*v581*/
	v_nop
	v_pk_fma_f32 v[68:69] /*v[580:581]*/, v[122:123] /*v[634:635]*/, s[52:53], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v29 /*v541*/, v92 /*v604*/
	v_exp_f32_e32 v37 /*v549*/, v93 /*v605*/
	v_nop
	v_pk_fma_f32 v[92:93] /*v[604:605]*/, v[126:127] /*v[638:639]*/, s[52:53], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa242
	v_exp_f32_e32 v221 /*v477*/, v88 /*v600*/
	v_exp_f32_e32 v229 /*v485*/, v89 /*v601*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[88:89] /*v[600:601]*/, v[118:119] /*v[630:631]*/, s[52:53], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa241
	v_exp_f32_e32 v234 /*v490*/, v223 /*v479*/
	s_set_vgpr_msb 0x41a2
	v_exp_f32_e32 v39 /*v551*/, v68 /*v580*/
	v_exp_f32_e32 v49 /*v561*/, v69 /*v581*/
	v_nop
	v_pk_fma_f32 v[68:69] /*v[580:581]*/, v[128:129] /*v[640:641]*/, s[52:53], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa242
	v_exp_f32_e32 v223 /*v479*/, v92 /*v604*/
	v_exp_f32_e32 v235 /*v491*/, v93 /*v605*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[92:93] /*v[604:605]*/, v[132:133] /*v[644:645]*/, s[52:53], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v13 /*v525*/, v88 /*v600*/
	v_exp_f32_e32 v23 /*v535*/, v89 /*v601*/
	v_nop
	v_pk_fma_f32 v[88:89] /*v[600:601]*/, v[124:125] /*v[636:637]*/, s[52:53], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa242
	v_exp_f32_e32 v239 /*v495*/, v68 /*v580*/
	v_exp_f32_e32 v247 /*v503*/, v69 /*v581*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[68:69] /*v[580:581]*/, v[134:135] /*v[646:647]*/, s[52:53], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v15 /*v527*/, v92 /*v604*/
	v_exp_f32_e32 v25 /*v537*/, v93 /*v605*/
	v_nop
	v_pk_fma_f32 v[92:93] /*v[604:605]*/, v[138:139] /*v[650:651]*/, s[52:53], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa281
	v_exp_f32_e32 v38 /*v550*/, v210 /*v466*/
	v_exp_f32_e32 v48 /*v560*/, v211 /*v467*/
	s_set_vgpr_msb 0x8141
	v_exp_f32_e32 v210 /*v466*/, v218 /*v474*/
	v_exp_f32_e32 v218 /*v474*/, v219 /*v475*/
	s_set_vgpr_msb 0x4142
	v_exp_f32_e32 v211 /*v467*/, v88 /*v600*/
	v_exp_f32_e32 v219 /*v475*/, v89 /*v601*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[88:89] /*v[600:601]*/, v[130:131] /*v[642:643]*/, s[52:53], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v31 /*v543*/, v68 /*v580*/
	v_exp_f32_e32 v41 /*v553*/, v69 /*v581*/
	v_nop
	v_pk_fma_f32 v[68:69] /*v[580:581]*/, v[140:141] /*v[652:653]*/, s[52:53], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v53 /*v565*/, v92 /*v604*/
	v_exp_f32_e32 v59 /*v571*/, v93 /*v605*/
	v_nop
	v_pk_fma_f32 v[92:93] /*v[604:605]*/, v[144:145] /*v[656:657]*/, s[52:53], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa241
	v_exp_f32_e32 v194 /*v450*/, v194 /*v450*/
	s_set_vgpr_msb 0x4142
	v_exp_f32_e32 v253 /*v509*/, v88 /*v600*/
	s_set_vgpr_msb 0x42a2
	v_exp_f32_e32 v7 /*v519*/, v89 /*v601*/
	v_nop
	v_pk_fma_f32 v[88:89] /*v[600:601]*/, v[136:137] /*v[648:649]*/, s[52:53], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa281
	v_exp_f32_e32 v44 /*v556*/, v236 /*v492*/
	v_exp_f32_e32 v50 /*v562*/, v237 /*v493*/
	s_set_vgpr_msb 0x8141
	v_exp_f32_e32 v236 /*v492*/, v225 /*v481*/
	s_set_vgpr_msb 0x4182
	v_exp_f32_e32 v8 /*v520*/, v1 /*v513*/
	s_set_vgpr_msb 0x8242
	v_exp_f32_e32 v225 /*v481*/, v68 /*v580*/
	v_exp_f32_e32 v237 /*v493*/, v69 /*v581*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[68:69] /*v[580:581]*/, v[146:147] /*v[658:659]*/, s[52:53], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v1 /*v513*/, v92 /*v604*/
	v_exp_f32_e32 v9 /*v521*/, v93 /*v605*/
	v_nop
	v_pk_fma_f32 v[92:93] /*v[604:605]*/, v[158:159] /*v[670:671]*/, s[52:53], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v45 /*v557*/, v88 /*v600*/
	v_exp_f32_e32 v51 /*v563*/, v89 /*v601*/
	v_nop
	v_pk_fma_f32 v[88:89] /*v[600:601]*/, v[142:143] /*v[654:655]*/, s[52:53], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v26 /*v538*/, v17 /*v529*/
	v_exp_f32_e32 v54 /*v566*/, v47 /*v559*/
	v_exp_f32_e32 v17 /*v529*/, v68 /*v580*/
	v_exp_f32_e32 v27 /*v539*/, v69 /*v581*/
	v_exp_f32_e32 v47 /*v559*/, v92 /*v604*/
	v_pk_fma_f32 v[68:69] /*v[580:581]*/, v[160:161] /*v[672:673]*/, s[52:53], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v55 /*v567*/, v93 /*v605*/
	v_nop
	s_set_vgpr_msb 0xa285
	v_pk_add_f32 v[92:93] /*v[604:605]*/, v[194:195] /*v[450:451]*/, v[196:197] /*v[452:453]*/
	v_pk_add_f32 v[94:95] /*v[606:607]*/, v[202:203] /*v[458:459]*/, v[206:207] /*v[462:463]*/
	s_set_vgpr_msb 0x8541
	v_exp_f32_e32 v222 /*v478*/, v222 /*v478*/
	s_set_vgpr_msb 0x4181
	v_exp_f32_e32 v52 /*v564*/, v240 /*v496*/
	v_exp_f32_e32 v58 /*v570*/, v241 /*v497*/
	s_set_vgpr_msb 0x8141
	v_exp_f32_e32 v224 /*v480*/, v224 /*v480*/
	v_exp_f32_e32 v240 /*v496*/, v254 /*v510*/
	v_exp_f32_e32 v254 /*v510*/, v255 /*v511*/
	s_set_vgpr_msb 0x4142
	v_exp_f32_e32 v241 /*v497*/, v88 /*v600*/
	v_exp_f32_e32 v255 /*v511*/, v89 /*v601*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[88:89] /*v[600:601]*/, v[156:157] /*v[668:669]*/, s[52:53], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v60 /*v572*/, v57 /*v569*/
	v_exp_f32_e32 v57 /*v569*/, v68 /*v580*/
	v_exp_f32_e32 v61 /*v573*/, v69 /*v581*/
	v_nop
	s_set_vgpr_msb 0xa289
	v_pk_add_f32 v[68:69] /*v[580:581]*/, v[198:199] /*v[454:455]*/, v[92:93] /*v[604:605]*/
	v_pk_add_f32 v[92:93] /*v[604:605]*/, v[212:213] /*v[468:469]*/, v[94:95] /*v[606:607]*/
	s_set_vgpr_msb 0x8985
	v_pk_add_f32 v[94:95] /*v[606:607]*/, v[214:215] /*v[470:471]*/, v[226:227] /*v[482:483]*/
	v_pk_add_f32 v[96:97] /*v[608:609]*/, v[242:243] /*v[498:499]*/, v[248:249] /*v[504:505]*/
	s_set_vgpr_msb 0x858a
	v_pk_add_f32 v[98:99] /*v[610:611]*/, v[10:11] /*v[522:523]*/, v[18:19] /*v[530:531]*/
	v_pk_add_f32 v[108:109] /*v[620:621]*/, v[22:23] /*v[534:535]*/, v[28:29] /*v[540:541]*/
	v_pk_add_f32 v[110:111] /*v[622:623]*/, v[38:39] /*v[550:551]*/, v[48:49] /*v[560:561]*/
	s_set_vgpr_msb 0x8a85
	v_pk_add_f32 v[114:115] /*v[626:627]*/, v[238:239] /*v[494:495]*/, v[246:247] /*v[502:503]*/
	s_set_vgpr_msb 0x858a
	v_pk_add_f32 v[116:117] /*v[628:629]*/, v[6:7] /*v[518:519]*/, v[14:15] /*v[526:527]*/
	v_exp_f32_e32 v0 /*v512*/, v0 /*v512*/
	v_exp_f32_e32 v16 /*v528*/, v16 /*v528*/
	v_exp_f32_e32 v42 /*v554*/, v33 /*v545*/
	v_exp_f32_e32 v46 /*v558*/, v46 /*v558*/
	v_exp_f32_e32 v43 /*v555*/, v89 /*v601*/
	s_set_vgpr_msb 0x8a86
	v_pk_add_f32 v[100:101] /*v[612:613]*/, v[34:35] /*v[546:547]*/, v[200:201] /*v[456:457]*/
	s_set_vgpr_msb 0x8685
	v_pk_add_f32 v[102:103] /*v[614:615]*/, v[208:209] /*v[464:465]*/, v[216:217] /*v[472:473]*/
	s_set_vgpr_msb 0x8589
	v_pk_add_f32 v[94:95] /*v[606:607]*/, v[230:231] /*v[486:487]*/, v[94:95] /*v[606:607]*/
	s_set_vgpr_msb 0x898a
	v_pk_add_f32 v[96:97] /*v[608:609]*/, v[2:3] /*v[514:515]*/, v[96:97] /*v[608:609]*/
	v_pk_add_f32 v[98:99] /*v[610:611]*/, v[20:21] /*v[532:533]*/, v[98:99] /*v[610:611]*/
	s_set_vgpr_msb 0x8a85
	v_pk_add_f32 v[104:105] /*v[616:617]*/, v[228:229] /*v[484:485]*/, v[232:233] /*v[488:489]*/
	v_pk_add_f32 v[112:113] /*v[624:625]*/, v[218:219] /*v[474:475]*/, v[222:223] /*v[478:479]*/
	s_set_vgpr_msb 0x858a
	v_pk_add_f32 v[108:109] /*v[620:621]*/, v[36:37] /*v[548:549]*/, v[108:109] /*v[620:621]*/
	s_set_vgpr_msb 0x8a89
	v_pk_add_f32 v[110:111] /*v[622:623]*/, v[210:211] /*v[466:467]*/, v[110:111] /*v[622:623]*/
	s_set_vgpr_msb 0x898a
	v_pk_add_f32 v[118:119] /*v[630:631]*/, v[30:31] /*v[542:543]*/, v[40:41] /*v[552:553]*/
	v_pk_add_f32 v[120:121] /*v[632:633]*/, v[50:51] /*v[562:563]*/, v[52:53] /*v[564:565]*/
	s_set_vgpr_msb 0x8a85
	v_pk_add_f32 v[122:123] /*v[634:635]*/, v[224:225] /*v[480:481]*/, v[236:237] /*v[492:493]*/
	s_set_vgpr_msb 0x8589
	v_pk_add_f32 v[114:115] /*v[626:627]*/, v[252:253] /*v[508:509]*/, v[114:115] /*v[626:627]*/
	s_set_vgpr_msb 0x89aa
	v_pk_add_f32 v[116:117] /*v[628:629]*/, v[24:25] /*v[536:537]*/, v[116:117] /*v[628:629]*/
	v_pk_add_f32 v[68:69] /*v[580:581]*/, v[68:69] /*v[580:581]*/, v[92:93] /*v[604:605]*/
	v_exp_f32_e32 v32 /*v544*/, v32 /*v544*/
	v_exp_f32_e32 v56 /*v568*/, v56 /*v568*/
	v_exp_f32_e32 v33 /*v545*/, v88 /*v600*/
	v_nop
	v_pk_fma_f32 v[88:89] /*v[600:601]*/, v[162:163] /*v[674:675]*/, s[52:53], v[70:71] /*v[582:583]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xaa89
	v_pk_add_f32 v[100:101] /*v[612:613]*/, v[204:205] /*v[460:461]*/, v[100:101] /*v[612:613]*/
	v_pk_add_f32 v[102:103] /*v[614:615]*/, v[220:221] /*v[476:477]*/, v[102:103] /*v[614:615]*/
	v_pk_add_f32 v[106:107] /*v[618:619]*/, v[250:251] /*v[506:507]*/, v[4:5] /*v[516:517]*/
	v_pk_add_f32 v[104:105] /*v[616:617]*/, v[244:245] /*v[500:501]*/, v[104:105] /*v[616:617]*/
	v_pk_add_f32 v[112:113] /*v[624:625]*/, v[234:235] /*v[490:491]*/, v[112:113] /*v[624:625]*/
	s_set_vgpr_msb 0x898a
	v_pk_add_f32 v[118:119] /*v[630:631]*/, v[44:45] /*v[556:557]*/, v[118:119] /*v[630:631]*/
	v_pk_add_f32 v[120:121] /*v[632:633]*/, v[58:59] /*v[570:571]*/, v[120:121] /*v[632:633]*/
	s_set_vgpr_msb 0x8a89
	v_pk_add_f32 v[122:123] /*v[634:635]*/, v[240:241] /*v[496:497]*/, v[122:123] /*v[634:635]*/
	v_pk_add_f32 v[124:125] /*v[636:637]*/, v[254:255] /*v[510:511]*/, v[0:1] /*v[512:513]*/
	s_set_vgpr_msb 0x898a
	v_pk_add_f32 v[126:127] /*v[638:639]*/, v[16:17] /*v[528:529]*/, v[26:27] /*v[538:539]*/
	v_pk_add_f32 v[128:129] /*v[640:641]*/, v[42:43] /*v[554:555]*/, v[46:47] /*v[558:559]*/
	v_pk_add_f32 v[68:69] /*v[580:581]*/, v[94:95] /*v[606:607]*/, v[68:69] /*v[580:581]*/
	v_pk_add_f32 v[94:95] /*v[606:607]*/, v[96:97] /*v[608:609]*/, v[98:99] /*v[610:611]*/
	v_pk_add_f32 v[96:97] /*v[608:609]*/, v[108:109] /*v[620:621]*/, v[110:111] /*v[622:623]*/
	v_pk_add_f32 v[98:99] /*v[610:611]*/, v[114:115] /*v[626:627]*/, v[116:117] /*v[628:629]*/
	v_exp_f32_e32 v62 /*v574*/, v62 /*v574*/
	v_exp_f32_e32 v64 /*v576*/, v63 /*v575*/
	v_exp_f32_e32 v63 /*v575*/, v88 /*v600*/
	v_pk_add_f32 v[106:107] /*v[618:619]*/, v[12:13] /*v[524:525]*/, v[106:107] /*v[618:619]*/
	v_pk_add_f32 v[130:131] /*v[642:643]*/, v[56:57] /*v[568:569]*/, v[60:61] /*v[572:573]*/
	v_pk_add_f32 v[92:93] /*v[604:605]*/, v[8:9] /*v[520:521]*/, v[124:125] /*v[636:637]*/
	v_pk_add_f32 v[124:125] /*v[636:637]*/, v[32:33] /*v[544:545]*/, v[126:127] /*v[638:639]*/
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
	v_pk_add_f32 v[68:69] /*v[580:581]*/, v[68:69] /*v[580:581]*/, v[94:95] /*v[606:607]*/
	v_pk_add_f32 v[94:95] /*v[606:607]*/, v[96:97] /*v[608:609]*/, v[98:99] /*v[610:611]*/
	v_exp_f32_e32 v65 /*v577*/, v89 /*v601*/
	v_sub_f32_e32 v70 /*v582*/, v66 /*v578*/, v86 /*v598*/
	v_pk_add_f32 v[88:89] /*v[600:601]*/, v[128:129] /*v[640:641]*/, v[102:103] /*v[614:615]*/
	v_pk_add_f32 v[68:69] /*v[580:581]*/, v[100:101] /*v[612:613]*/, v[68:69] /*v[580:581]*/
	v_pk_add_f32 v[92:93] /*v[604:605]*/, v[92:93] /*v[604:605]*/, v[94:95] /*v[606:607]*/
	v_pk_add_f32 v[88:89] /*v[600:601]*/, v[64:65] /*v[576:577]*/, v[88:89] /*v[600:601]*/
	v_pk_add_f32 v[68:69] /*v[580:581]*/, v[68:69] /*v[580:581]*/, v[92:93] /*v[604:605]*/
	v_pk_add_f32 v[66:67] /*v[578:579]*/, v[88:89] /*v[600:601]*/, v[68:69] /*v[580:581]*/
	v_mov_b32_e32 v68 /*v580*/, v66 /*v578*/
	v_dual_mul_f32 v70 /*v582*/, 0x3fb8aa3b, v70 /*v582*/ :: v_dual_mov_b32 v69 /*v581*/, v67 /*v579*/
	v_permlanex16_b32 v68 /*v580*/, v68 /*v580*/, s56, 0xfedcba98
	v_exp_f32_e32 v70 /*v582*/, v70 /*v582*/
	v_permlanex16_b32 v69 /*v581*/, v69 /*v581*/, s56, 0xfedcba98
	s_set_vgpr_msb 0x8a00
	s_cbranch_vccz .LBB0_30
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
.LBB0_30:
	s_set_vgpr_msb 0x8a
	v_sub_f32_e32 v71 /*v583*/, v71 /*v583*/, v85 /*v597*/
	s_and_b32 s2, s5, exec_lo
	s_cselect_b32 s2, 1, 0
	s_cmp_lg_u32 s2, 1
	v_mul_f32_e32 v71 /*v583*/, 0x3fb8aa3b, v71 /*v583*/
	v_exp_f32_e32 v71 /*v583*/, v71 /*v583*/
	s_set_vgpr_msb 0x8a00
	s_cbranch_scc1 .LBB0_25
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
	s_branch .LBB0_25
.LBB0_32:
	v_dual_mov_b32 v1, v0 :: v_dual_mov_b32 v2, v0
	v_dual_mov_b32 v3, v0 :: v_dual_mov_b32 v4, v0
	v_dual_mov_b32 v5, v0 :: v_dual_mov_b32 v6, v0
	v_mov_b32_e32 v7, v0
	s_wait_alu depctr_vm_vsrc(0)
	s_set_vgpr_msb 0x80
	v_dual_mov_b32 v85 /*v597*/, 0xf149f2ca :: v_dual_mov_b32 v86 /*v598*/, 0xf149f2ca
	s_set_vgpr_msb 0x8040
	v_dual_mov_b32 v193 /*v449*/, v0 :: v_dual_mov_b32 v192 /*v448*/, v0
	s_set_vgpr_msb 0x4082
	v_mov_b32_e32 v83 /*v595*/, v80 /*v592*/
	s_set_vgpr_msb 0x8200
	v_mov_b64_e32 v[14:15], v[6:7]
	v_mov_b64_e32 v[12:13], v[4:5]
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
.LBB0_33:
	s_cmp_ge_u32 s44, s10
	s_cbranch_scc1 .LBB0_42
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 8
	v_dual_add_nc_u32 v192, s11, v74 /*v586*/ :: v_dual_add_nc_u32 v193, s11, v75 /*v587*/
	s_add_co_i32 s2, s58, -1
	s_mov_b32 s19, 0
	s_mov_b32 s4, 1
	s_set_vgpr_msb 0x880
	v_min_i32_e32 v87 /*v599*/, s2, v192
	v_min_i32_e32 v88 /*v600*/, s2, v193
	s_mov_b32 s11, s19
	s_mov_b32 s16, 32
	s_mov_b32 s15, 0x800000
	s_mov_b32 s13, 0xffff0000
	s_mov_b32 s12, 0x7510000
	s_mov_b32 s20, 0xf510000
	s_mov_b32 s35, 0x76543210
	s_mov_b32 s46, 0x3fb8aa3b
	s_set_vgpr_msb 0x8000
	s_branch .LBB0_36
.LBB0_35:
	s_set_vgpr_msb 0x8a
	v_cvt_pk_bf16_f32 v99 /*v611*/, v14 /*v526*/, v26 /*v538*/
	s_set_vgpr_msb 0x8a89
	v_cvt_pk_bf16_f32 v98 /*v610*/, v254 /*v510*/, v10 /*v522*/
	s_set_vgpr_msb 0x8985
	v_cvt_pk_bf16_f32 v97 /*v609*/, v238 /*v494*/, v250 /*v506*/
	v_cvt_pk_bf16_f32 v96 /*v608*/, v224 /*v480*/, v234 /*v490*/
	v_cvt_pk_bf16_f32 v95 /*v607*/, v212 /*v468*/, v220 /*v476*/
	v_cvt_pk_bf16_f32 v94 /*v606*/, v204 /*v460*/, v210 /*v466*/
	v_cvt_pk_bf16_f32 v93 /*v605*/, v198 /*v454*/, v202 /*v458*/
	v_cvt_pk_bf16_f32 v92 /*v604*/, v194 /*v450*/, v196 /*v452*/
	s_set_vgpr_msb 0x858a
	v_cvt_pk_bf16_f32 v107 /*v619*/, v15 /*v527*/, v27 /*v539*/
	s_set_vgpr_msb 0x8a89
	v_cvt_pk_bf16_f32 v106 /*v618*/, v255 /*v511*/, v11 /*v523*/
	s_set_vgpr_msb 0x8985
	v_cvt_pk_bf16_f32 v105 /*v617*/, v239 /*v495*/, v251 /*v507*/
	v_cvt_pk_bf16_f32 v104 /*v616*/, v225 /*v481*/, v235 /*v491*/
	v_cvt_pk_bf16_f32 v103 /*v615*/, v213 /*v469*/, v221 /*v477*/
	v_cvt_pk_bf16_f32 v102 /*v614*/, v205 /*v461*/, v211 /*v467*/
	v_cvt_pk_bf16_f32 v101 /*v613*/, v199 /*v455*/, v203 /*v459*/
	v_cvt_pk_bf16_f32 v100 /*v612*/, v195 /*v451*/, v197 /*v453*/
	s_set_vgpr_msb 0x8509
	s_wait_dscnt 0x3d
	v_wmma_f32_16x16x32_bf16 v[120:127], v[184:191] /*v[440:447]*/, v[92:99] /*v[604:611]*/, v[120:127]
	s_set_vgpr_msb 0x98a
	v_cvt_pk_bf16_f32 v115 /*v627*/, v37 /*v549*/, v45 /*v557*/
	v_cvt_pk_bf16_f32 v114 /*v626*/, v23 /*v535*/, v33 /*v545*/
	v_cvt_pk_bf16_f32 v113 /*v625*/, v7 /*v519*/, v19 /*v531*/
	s_set_vgpr_msb 0x8a89
	v_cvt_pk_bf16_f32 v112 /*v624*/, v247 /*v503*/, v3 /*v515*/
	s_set_vgpr_msb 0x8985
	v_cvt_pk_bf16_f32 v111 /*v623*/, v231 /*v487*/, v243 /*v499*/
	v_cvt_pk_bf16_f32 v110 /*v622*/, v219 /*v475*/, v229 /*v485*/
	v_cvt_pk_bf16_f32 v109 /*v621*/, v209 /*v465*/, v217 /*v473*/
	s_set_vgpr_msb 0x8509
	v_wmma_f32_16x16x32_bf16 v[56:63], v[184:191] /*v[440:447]*/, v[100:107] /*v[612:619]*/, v[56:63]
	s_set_vgpr_msb 0x985
	v_cvt_pk_bf16_f32 v108 /*v620*/, v201 /*v457*/, v207 /*v463*/
	s_set_vgpr_msb 0x854a
	v_cvt_pk_bf16_f32 v201 /*v457*/, v53 /*v565*/, v59 /*v571*/
	v_cvt_pk_bf16_f32 v199 /*v455*/, v31 /*v543*/, v41 /*v553*/
	v_cvt_pk_bf16_f32 v198 /*v454*/, v17 /*v529*/, v29 /*v541*/
	v_cvt_pk_bf16_f32 v191 /*v447*/, v36 /*v548*/, v44 /*v556*/
	v_cvt_pk_bf16_f32 v190 /*v446*/, v22 /*v534*/, v32 /*v544*/
	v_cvt_pk_bf16_f32 v189 /*v445*/, v6 /*v518*/, v18 /*v530*/
	s_set_vgpr_msb 0x4a09
	s_wait_dscnt 0x3c
	v_wmma_f32_16x16x32_bf16 v[112:119], v[152:159] /*v[408:415]*/, v[92:99] /*v[604:611]*/, v[112:119]
	s_set_vgpr_msb 0x949
	v_cvt_pk_bf16_f32 v188 /*v444*/, v246 /*v502*/, v2 /*v514*/
	s_set_vgpr_msb 0x4945
	v_cvt_pk_bf16_f32 v187 /*v443*/, v230 /*v486*/, v242 /*v498*/
	v_cvt_pk_bf16_f32 v186 /*v442*/, v218 /*v474*/, v228 /*v484*/
	v_cvt_pk_bf16_f32 v185 /*v441*/, v208 /*v464*/, v216 /*v472*/
	v_cvt_pk_bf16_f32 v184 /*v440*/, v200 /*v456*/, v206 /*v462*/
	s_set_vgpr_msb 0x454a
	v_cvt_pk_bf16_f32 v200 /*v456*/, v43 /*v555*/, v51 /*v563*/
	v_cvt_pk_bf16_f32 v197 /*v453*/, v1 /*v513*/, v13 /*v525*/
	s_set_vgpr_msb 0x4a09
	v_wmma_f32_16x16x32_bf16 v[48:55], v[152:159] /*v[408:415]*/, v[100:107] /*v[612:619]*/, v[48:55]
	s_set_vgpr_msb 0x945
	v_cvt_pk_bf16_f32 v196 /*v452*/, v241 /*v497*/, v253 /*v509*/
	v_cvt_pk_bf16_f32 v195 /*v451*/, v227 /*v483*/, v237 /*v493*/
	v_cvt_pk_bf16_f32 v194 /*v450*/, v215 /*v471*/, v223 /*v479*/
	s_set_vgpr_msb 0x454a
	v_cvt_pk_bf16_f32 v209 /*v465*/, v63 /*v575*/, v65 /*v577*/
	v_cvt_pk_bf16_f32 v208 /*v464*/, v57 /*v569*/, v61 /*v573*/
	v_cvt_pk_bf16_f32 v207 /*v463*/, v49 /*v561*/, v55 /*v567*/
	v_cvt_pk_bf16_f32 v206 /*v462*/, v39 /*v551*/, v47 /*v559*/
	s_set_vgpr_msb 0x4a09
	s_wait_dscnt 0x2d
	v_wmma_f32_16x16x32_bf16 v[104:111], v[120:127] /*v[376:383]*/, v[92:99] /*v[604:611]*/, v[104:111]
	s_set_vgpr_msb 0x94a
	v_cvt_pk_bf16_f32 v205 /*v461*/, v25 /*v537*/, v35 /*v547*/
	v_cvt_pk_bf16_f32 v204 /*v460*/, v9 /*v521*/, v21 /*v533*/
	s_set_vgpr_msb 0x4a49
	v_cvt_pk_bf16_f32 v203 /*v459*/, v249 /*v505*/, v5 /*v517*/
	s_set_vgpr_msb 0x4945
	v_cvt_pk_bf16_f32 v202 /*v458*/, v233 /*v489*/, v245 /*v501*/
	s_set_vgpr_msb 0x4586
	v_fmac_f32_e32 v86 /*v598*/, v68 /*v580*/, v192 /*v448*/
	v_fmac_f32_e32 v90 /*v602*/, v70 /*v582*/, v193 /*v449*/
	s_add_nc_u64 s[44:45], s[44:45], 1
	s_set_vgpr_msb 0x8609
	v_wmma_f32_16x16x32_bf16 v[40:47], v[120:127] /*v[376:383]*/, v[100:107] /*v[612:619]*/, v[40:47]
	v_cmp_ge_u64_e64 s2, s[44:45], s[10:11]
	s_set_vgpr_msb 0x982
	v_mov_b32_e32 v85 /*v597*/, v71 /*v583*/
	s_set_vgpr_msb 0x824a
	v_add_f32_e32 v193 /*v449*/, v90 /*v602*/, v67 /*v579*/
	s_wait_alu depctr_vm_vsrc(0)
	s_set_vgpr_msb 0x4a82
	v_dual_mov_b32 v90 /*v602*/, v84 /*v596*/ :: v_dual_mov_b32 v84 /*v596*/, v82 /*v594*/
	s_set_vgpr_msb 0x824a
	v_add_f32_e32 v192 /*v448*/, v86 /*v598*/, v66 /*v578*/
	s_set_vgpr_msb 0x4a82
	v_mov_b32_e32 v82 /*v594*/, v83 /*v595*/
	s_set_vgpr_msb 0x8209
	s_wait_dscnt 0x2c
	v_wmma_f32_16x16x32_bf16 v[96:103], v[88:95] /*v[344:351]*/, v[92:99] /*v[604:611]*/, v[96:103]
	s_set_vgpr_msb 0x982
	v_dual_mov_b32 v83 /*v595*/, v89 /*v601*/ :: v_dual_mov_b32 v86 /*v598*/, v69 /*v581*/
	s_and_b32 vcc_lo, exec_lo, s2
	s_set_vgpr_msb 0x8209
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
	v_cvt_pk_bf16_f32 v178 /*v434*/, v240 /*v496*/, v252 /*v508*/
	v_cvt_pk_bf16_f32 v177 /*v433*/, v226 /*v482*/, v236 /*v492*/
	v_cvt_pk_bf16_f32 v176 /*v432*/, v214 /*v470*/, v222 /*v478*/
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
	v_wmma_f32_16x16x32_bf16 v[56:63], v[168:175] /*v[424:431]*/, v[194:201] /*v[450:457]*/, v[56:63]
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
	v_cvt_pk_bf16_f32 v169 /*v425*/, v248 /*v504*/, v4 /*v516*/
	s_set_vgpr_msb 0x4945
	v_cvt_pk_bf16_f32 v168 /*v424*/, v232 /*v488*/, v244 /*v500*/
	s_set_vgpr_msb 0x4505
	v_wmma_f32_16x16x32_bf16 v[48:55], v[136:143] /*v[392:399]*/, v[194:201] /*v[450:457]*/, v[48:55]
	v_wmma_f32_16x16x32_bf16 v[104:111], v[104:111] /*v[360:367]*/, v[176:183] /*v[432:439]*/, v[104:111]
	v_wmma_f32_16x16x32_bf16 v[40:47], v[104:111] /*v[360:367]*/, v[194:201] /*v[450:457]*/, v[40:47]
	v_wmma_f32_16x16x32_bf16 v[96:103], v[72:79] /*v[328:335]*/, v[176:183] /*v[432:439]*/, v[96:103]
	v_wmma_f32_16x16x32_bf16 v[32:39], v[72:79] /*v[328:335]*/, v[194:201] /*v[450:457]*/, v[32:39]
	v_wmma_f32_16x16x32_bf16 v[88:95], v[40:47] /*v[296:303]*/, v[176:183] /*v[432:439]*/, v[88:95]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[40:47] /*v[296:303]*/, v[194:201] /*v[450:457]*/, v[24:31]
	v_wmma_f32_16x16x32_bf16 v[80:87], v[8:15] /*v[264:271]*/, v[176:183] /*v[432:439]*/, v[80:87]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[8:15] /*v[264:271]*/, v[194:201] /*v[450:457]*/, v[16:23]
	s_set_vgpr_msb 0x504
	v_wmma_f32_16x16x32_bf16 v[72:79], v[232:239], v[176:183] /*v[432:439]*/, v[72:79]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[232:239], v[194:201] /*v[450:457]*/, v[8:15]
	v_wmma_f32_16x16x32_bf16 v[64:71], v[200:207], v[176:183] /*v[432:439]*/, v[64:71]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[200:207], v[194:201] /*v[450:457]*/, v[0:7]
	s_set_vgpr_msb 0x405
	v_wmma_f32_16x16x32_bf16 v[120:127], v[160:167] /*v[416:423]*/, v[168:175] /*v[424:431]*/, v[120:127]
	v_wmma_f32_16x16x32_bf16 v[56:63], v[160:167] /*v[416:423]*/, v[202:209] /*v[458:465]*/, v[56:63]
	v_wmma_f32_16x16x32_bf16 v[112:119], v[128:135] /*v[384:391]*/, v[168:175] /*v[424:431]*/, v[112:119]
	v_wmma_f32_16x16x32_bf16 v[48:55], v[128:135] /*v[384:391]*/, v[202:209] /*v[458:465]*/, v[48:55]
	v_wmma_f32_16x16x32_bf16 v[104:111], v[96:103] /*v[352:359]*/, v[168:175] /*v[424:431]*/, v[104:111]
	v_wmma_f32_16x16x32_bf16 v[40:47], v[96:103] /*v[352:359]*/, v[202:209] /*v[458:465]*/, v[40:47]
	v_wmma_f32_16x16x32_bf16 v[96:103], v[64:71] /*v[320:327]*/, v[168:175] /*v[424:431]*/, v[96:103]
	v_wmma_f32_16x16x32_bf16 v[32:39], v[64:71] /*v[320:327]*/, v[202:209] /*v[458:465]*/, v[32:39]
	v_wmma_f32_16x16x32_bf16 v[88:95], v[32:39] /*v[288:295]*/, v[168:175] /*v[424:431]*/, v[88:95]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[32:39] /*v[288:295]*/, v[202:209] /*v[458:465]*/, v[24:31]
	v_wmma_f32_16x16x32_bf16 v[80:87], v[0:7] /*v[256:263]*/, v[168:175] /*v[424:431]*/, v[80:87]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[0:7] /*v[256:263]*/, v[202:209] /*v[458:465]*/, v[16:23]
	s_set_vgpr_msb 0x504
	v_wmma_f32_16x16x32_bf16 v[72:79], v[224:231], v[168:175] /*v[424:431]*/, v[72:79]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[224:231], v[202:209] /*v[458:465]*/, v[8:15]
	v_wmma_f32_16x16x32_bf16 v[64:71], v[192:199], v[168:175] /*v[424:431]*/, v[64:71]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[192:199], v[202:209] /*v[458:465]*/, v[0:7]
	s_set_vgpr_msb 0x400
	s_cbranch_vccnz .LBB0_43
.LBB0_36:
	s_wait_alu depctr_vm_vsrc(6)
	s_set_vgpr_msb 0x82
	v_dual_mov_b32 v89 /*v601*/, v82 /*v594*/ :: v_dual_mov_b32 v82 /*v594*/, v90 /*v602*/
	s_add_co_i32 s2, s44, 1
	s_wait_tensorcnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x8242
	ds_load_b128 v[184:187] /*v[440:443]*/, v83 /*v595*/
	ds_load_b128 v[188:191] /*v[444:447]*/, v83 /*v595*/ offset:32
	ds_load_b128 v[176:179] /*v[432:435]*/, v83 /*v595*/ offset:64
	ds_load_b128 v[180:183] /*v[436:439]*/, v83 /*v595*/ offset:96
	ds_load_b128 v[168:171] /*v[424:427]*/, v83 /*v595*/ offset:128
	ds_load_b128 v[172:175] /*v[428:431]*/, v83 /*v595*/ offset:160
	ds_load_b128 v[160:163] /*v[416:419]*/, v83 /*v595*/ offset:192
	ds_load_b128 v[164:167] /*v[420:423]*/, v83 /*v595*/ offset:224
	ds_load_b128 v[152:155] /*v[408:411]*/, v83 /*v595*/ offset:4352
	ds_load_b128 v[156:159] /*v[412:415]*/, v83 /*v595*/ offset:4384
	ds_load_b128 v[144:147] /*v[400:403]*/, v83 /*v595*/ offset:4416
	ds_load_b128 v[148:151] /*v[404:407]*/, v83 /*v595*/ offset:4448
	ds_load_b128 v[136:139] /*v[392:395]*/, v83 /*v595*/ offset:4480
	ds_load_b128 v[140:143] /*v[396:399]*/, v83 /*v595*/ offset:4512
	ds_load_b128 v[128:131] /*v[384:387]*/, v83 /*v595*/ offset:4544
	ds_load_b128 v[132:135] /*v[388:391]*/, v83 /*v595*/ offset:4576
	ds_load_b128 v[120:123] /*v[376:379]*/, v83 /*v595*/ offset:8704
	ds_load_b128 v[124:127] /*v[380:383]*/, v83 /*v595*/ offset:8736
	ds_load_b128 v[112:115] /*v[368:371]*/, v83 /*v595*/ offset:8768
	ds_load_b128 v[116:119] /*v[372:375]*/, v83 /*v595*/ offset:8800
	ds_load_b128 v[104:107] /*v[360:363]*/, v83 /*v595*/ offset:8832
	ds_load_b128 v[108:111] /*v[364:367]*/, v83 /*v595*/ offset:8864
	ds_load_b128 v[96:99] /*v[352:355]*/, v83 /*v595*/ offset:8896
	ds_load_b128 v[100:103] /*v[356:359]*/, v83 /*v595*/ offset:8928
	ds_load_b128 v[88:91] /*v[344:347]*/, v83 /*v595*/ offset:13056
	ds_load_b128 v[92:95] /*v[348:351]*/, v83 /*v595*/ offset:13088
	ds_load_b128 v[80:83] /*v[336:339]*/, v83 /*v595*/ offset:13120
	ds_load_b128 v[84:87] /*v[340:343]*/, v83 /*v595*/ offset:13152
	ds_load_b128 v[72:75] /*v[328:331]*/, v83 /*v595*/ offset:13184
	ds_load_b128 v[76:79] /*v[332:335]*/, v83 /*v595*/ offset:13216
	ds_load_b128 v[64:67] /*v[320:323]*/, v83 /*v595*/ offset:13248
	ds_load_b128 v[68:71] /*v[324:327]*/, v83 /*v595*/ offset:13280
	ds_load_b128 v[56:59] /*v[312:315]*/, v83 /*v595*/ offset:17408
	ds_load_b128 v[60:63] /*v[316:319]*/, v83 /*v595*/ offset:17440
	ds_load_b128 v[48:51] /*v[304:307]*/, v83 /*v595*/ offset:17472
	ds_load_b128 v[52:55] /*v[308:311]*/, v83 /*v595*/ offset:17504
	ds_load_b128 v[40:43] /*v[296:299]*/, v83 /*v595*/ offset:17536
	ds_load_b128 v[44:47] /*v[300:303]*/, v83 /*v595*/ offset:17568
	ds_load_b128 v[32:35] /*v[288:291]*/, v83 /*v595*/ offset:17600
	ds_load_b128 v[36:39] /*v[292:295]*/, v83 /*v595*/ offset:17632
	ds_load_b128 v[24:27] /*v[280:283]*/, v83 /*v595*/ offset:21760
	ds_load_b128 v[28:31] /*v[284:287]*/, v83 /*v595*/ offset:21792
	ds_load_b128 v[16:19] /*v[272:275]*/, v83 /*v595*/ offset:21824
	ds_load_b128 v[20:23] /*v[276:279]*/, v83 /*v595*/ offset:21856
	ds_load_b128 v[8:11] /*v[264:267]*/, v83 /*v595*/ offset:21888
	ds_load_b128 v[12:15] /*v[268:271]*/, v83 /*v595*/ offset:21920
	ds_load_b128 v[0:3] /*v[256:259]*/, v83 /*v595*/ offset:21952
	ds_load_b128 v[4:7] /*v[260:263]*/, v83 /*v595*/ offset:21984
	s_set_vgpr_msb 0x4202
	ds_load_b128 v[248:251], v83 /*v595*/ offset:26112
	ds_load_b128 v[252:255], v83 /*v595*/ offset:26144
	ds_load_b128 v[240:243], v83 /*v595*/ offset:26176
	ds_load_b128 v[244:247], v83 /*v595*/ offset:26208
	ds_load_b128 v[232:235], v83 /*v595*/ offset:26240
	ds_load_b128 v[236:239], v83 /*v595*/ offset:26272
	ds_load_b128 v[224:227], v83 /*v595*/ offset:26304
	ds_load_b128 v[228:231], v83 /*v595*/ offset:26336
	ds_load_b128 v[216:219], v83 /*v595*/ offset:30464
	ds_load_b128 v[220:223], v83 /*v595*/ offset:30496
	ds_load_b128 v[208:211], v83 /*v595*/ offset:30528
	ds_load_b128 v[212:215], v83 /*v595*/ offset:30560
	ds_load_b128 v[200:203], v83 /*v595*/ offset:30592
	ds_load_b128 v[204:207], v83 /*v595*/ offset:30624
	ds_load_b128 v[192:195], v83 /*v595*/ offset:30656
	ds_load_b128 v[196:199], v83 /*v595*/ offset:30688
	s_cmp_ge_i32 s2, s10
	s_set_vgpr_msb 0x200
	s_cbranch_scc1 .LBB0_38
	s_lshl_b32 s5, s2, 7
	s_lshl_b32 s2, s2, 17
	s_sub_co_i32 s7, s9, s5
	s_add_co_i32 s6, s5, s64
	s_set_vgpr_msb 0x41
	v_med3_i32 v194 /*v450*/, s7, 0, 0x80
	s_ashr_i32 s7, s6, 31
	s_and_b32 s2, s2, 0x20000
	s_mul_u64 s[22:23], s[6:7], s[62:63]
	s_mul_u64 s[6:7], s[6:7], s[60:61]
	v_readfirstlane_b32 s14, v194 /*v450*/
	s_lshl_b64 s[6:7], s[6:7], 1
	s_lshl_b64 s[22:23], s[22:23], 1
	s_add_nc_u64 s[6:7], s[42:43], s[6:7]
	s_or_b32 s5, s54, s2
	s_sub_co_i32 s14, s14, s53
	s_add_nc_u64 s[6:7], s[36:37], s[6:7]
	s_max_i32 s14, s14, 0
	s_add_nc_u64 s[22:23], s[40:41], s[22:23]
	s_lshl_b32 s14, s14, 16
	s_bitset1_b32 s7, 31
	s_addk_co_i32 s14, 0x7fff
	s_mov_b32 s21, s13
	tensor_load_to_lds s[4:7], s[12:19]
	s_add_nc_u64 s[6:7], s[38:39], s[22:23]
	s_or_b32 s5, s55, s2
	s_bitset1_b32 s7, 31
	s_mov_b32 s22, s14
	s_mov_b32 s23, s15
	s_mov_b32 s24, s16
	s_mov_b32 s27, s19
	tensor_load_to_lds s[4:7], s[20:27]
	s_set_vgpr_msb 0x4100
.LBB0_38:
	s_set_vgpr_msb 0x41
	s_wait_dscnt 0x3e
	v_wmma_f32_16x16x32_bf16 v[194:201] /*v[450:457]*/, v[184:191] /*v[440:447]*/, v[128:135], 0
	v_wmma_f32_16x16x32_bf16 v[202:209] /*v[458:465]*/, v[184:191] /*v[440:447]*/, v[160:167], 0
	s_wait_dscnt 0x36
	v_wmma_f32_16x16x32_bf16 v[210:217] /*v[466:473]*/, v[152:159] /*v[408:415]*/, v[128:135], 0
	v_wmma_f32_16x16x32_bf16 v[218:225] /*v[474:481]*/, v[152:159] /*v[408:415]*/, v[160:167], 0
	s_wait_dscnt 0x2e
	v_wmma_f32_16x16x32_bf16 v[226:233] /*v[482:489]*/, v[120:127] /*v[376:383]*/, v[128:135], 0
	v_wmma_f32_16x16x32_bf16 v[234:241] /*v[490:497]*/, v[120:127] /*v[376:383]*/, v[160:167], 0
	s_wait_dscnt 0x26
	v_wmma_f32_16x16x32_bf16 v[242:249] /*v[498:505]*/, v[88:95] /*v[344:351]*/, v[128:135], 0
	v_wmma_f32_16x16x32_bf16 v[250:257] /*v[506:513]*/, v[88:95] /*v[344:351]*/, v[160:167], 0
	s_set_vgpr_msb 0x4181
	s_wait_dscnt 0x1e
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[514:521]*/, v[56:63] /*v[312:319]*/, v[128:135], 0
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[522:529]*/, v[56:63] /*v[312:319]*/, v[160:167], 0
	s_wait_dscnt 0x16
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[530:537]*/, v[24:31] /*v[280:287]*/, v[128:135], 0
	v_wmma_f32_16x16x32_bf16 v[26:33] /*v[538:545]*/, v[24:31] /*v[280:287]*/, v[160:167], 0
	s_set_vgpr_msb 0x8180
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x32_bf16 v[34:41] /*v[546:553]*/, v[248:255], v[128:135], 0
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[554:561]*/, v[248:255], v[160:167], 0
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[562:569]*/, v[216:223], v[128:135], 0
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[570:577]*/, v[216:223], v[160:167], 0
	s_set_vgpr_msb 0x8051
	v_wmma_f32_16x16x32_bf16 v[194:201] /*v[450:457]*/, v[176:183] /*v[432:439]*/, v[136:143], v[194:201] /*v[450:457]*/
	v_wmma_f32_16x16x32_bf16 v[202:209] /*v[458:465]*/, v[176:183] /*v[432:439]*/, v[168:175], v[202:209] /*v[458:465]*/
	v_wmma_f32_16x16x32_bf16 v[210:217] /*v[466:473]*/, v[144:151] /*v[400:407]*/, v[136:143], v[210:217] /*v[466:473]*/
	v_wmma_f32_16x16x32_bf16 v[218:225] /*v[474:481]*/, v[144:151] /*v[400:407]*/, v[168:175], v[218:225] /*v[474:481]*/
	v_wmma_f32_16x16x32_bf16 v[226:233] /*v[482:489]*/, v[112:119] /*v[368:375]*/, v[136:143], v[226:233] /*v[482:489]*/
	v_wmma_f32_16x16x32_bf16 v[234:241] /*v[490:497]*/, v[112:119] /*v[368:375]*/, v[168:175], v[234:241] /*v[490:497]*/
	v_wmma_f32_16x16x32_bf16 v[242:249] /*v[498:505]*/, v[80:87] /*v[336:343]*/, v[136:143], v[242:249] /*v[498:505]*/
	v_wmma_f32_16x16x32_bf16 v[250:257] /*v[506:513]*/, v[80:87] /*v[336:343]*/, v[168:175], v[250:257] /*v[506:513]*/
	s_set_vgpr_msb 0x51a1
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[514:521]*/, v[48:55] /*v[304:311]*/, v[136:143], v[2:9] /*v[514:521]*/
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[522:529]*/, v[48:55] /*v[304:311]*/, v[168:175], v[10:17] /*v[522:529]*/
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[530:537]*/, v[16:23] /*v[272:279]*/, v[136:143], v[18:25] /*v[530:537]*/
	v_wmma_f32_16x16x32_bf16 v[26:33] /*v[538:545]*/, v[16:23] /*v[272:279]*/, v[168:175], v[26:33] /*v[538:545]*/
	s_set_vgpr_msb 0xa1a0
	v_wmma_f32_16x16x32_bf16 v[34:41] /*v[546:553]*/, v[240:247], v[136:143], v[34:41] /*v[546:553]*/
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[554:561]*/, v[240:247], v[168:175], v[42:49] /*v[554:561]*/
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[562:569]*/, v[208:215], v[136:143], v[50:57] /*v[562:569]*/
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[570:577]*/, v[208:215], v[168:175], v[58:65] /*v[570:577]*/
	s_set_vgpr_msb 0xa051
	v_wmma_f32_16x16x32_bf16 v[194:201] /*v[450:457]*/, v[168:175] /*v[424:431]*/, v[144:151], v[194:201] /*v[450:457]*/
	v_wmma_f32_16x16x32_bf16 v[202:209] /*v[458:465]*/, v[168:175] /*v[424:431]*/, v[176:183], v[202:209] /*v[458:465]*/
	v_wmma_f32_16x16x32_bf16 v[210:217] /*v[466:473]*/, v[136:143] /*v[392:399]*/, v[144:151], v[210:217] /*v[466:473]*/
	v_wmma_f32_16x16x32_bf16 v[218:225] /*v[474:481]*/, v[136:143] /*v[392:399]*/, v[176:183], v[218:225] /*v[474:481]*/
	v_wmma_f32_16x16x32_bf16 v[226:233] /*v[482:489]*/, v[104:111] /*v[360:367]*/, v[144:151], v[226:233] /*v[482:489]*/
	v_wmma_f32_16x16x32_bf16 v[234:241] /*v[490:497]*/, v[104:111] /*v[360:367]*/, v[176:183], v[234:241] /*v[490:497]*/
	v_wmma_f32_16x16x32_bf16 v[242:249] /*v[498:505]*/, v[72:79] /*v[328:335]*/, v[144:151], v[242:249] /*v[498:505]*/
	v_wmma_f32_16x16x32_bf16 v[250:257] /*v[506:513]*/, v[72:79] /*v[328:335]*/, v[176:183], v[250:257] /*v[506:513]*/
	s_set_vgpr_msb 0x51a1
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[514:521]*/, v[40:47] /*v[296:303]*/, v[144:151], v[2:9] /*v[514:521]*/
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[522:529]*/, v[40:47] /*v[296:303]*/, v[176:183], v[10:17] /*v[522:529]*/
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[530:537]*/, v[8:15] /*v[264:271]*/, v[144:151], v[18:25] /*v[530:537]*/
	v_wmma_f32_16x16x32_bf16 v[26:33] /*v[538:545]*/, v[8:15] /*v[264:271]*/, v[176:183], v[26:33] /*v[538:545]*/
	s_set_vgpr_msb 0xa1a0
	v_wmma_f32_16x16x32_bf16 v[34:41] /*v[546:553]*/, v[232:239], v[144:151], v[34:41] /*v[546:553]*/
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[554:561]*/, v[232:239], v[176:183], v[42:49] /*v[554:561]*/
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[562:569]*/, v[200:207], v[144:151], v[50:57] /*v[562:569]*/
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[570:577]*/, v[200:207], v[176:183], v[58:65] /*v[570:577]*/
	s_set_vgpr_msb 0xa051
	v_wmma_f32_16x16x32_bf16 v[194:201] /*v[450:457]*/, v[160:167] /*v[416:423]*/, v[152:159], v[194:201] /*v[450:457]*/
	v_wmma_f32_16x16x32_bf16 v[202:209] /*v[458:465]*/, v[160:167] /*v[416:423]*/, v[184:191], v[202:209] /*v[458:465]*/
	v_wmma_f32_16x16x32_bf16 v[210:217] /*v[466:473]*/, v[128:135] /*v[384:391]*/, v[152:159], v[210:217] /*v[466:473]*/
	v_wmma_f32_16x16x32_bf16 v[218:225] /*v[474:481]*/, v[128:135] /*v[384:391]*/, v[184:191], v[218:225] /*v[474:481]*/
	v_wmma_f32_16x16x32_bf16 v[226:233] /*v[482:489]*/, v[96:103] /*v[352:359]*/, v[152:159], v[226:233] /*v[482:489]*/
	v_wmma_f32_16x16x32_bf16 v[234:241] /*v[490:497]*/, v[96:103] /*v[352:359]*/, v[184:191], v[234:241] /*v[490:497]*/
	v_wmma_f32_16x16x32_bf16 v[242:249] /*v[498:505]*/, v[64:71] /*v[320:327]*/, v[152:159], v[242:249] /*v[498:505]*/
	v_wmma_f32_16x16x32_bf16 v[250:257] /*v[506:513]*/, v[64:71] /*v[320:327]*/, v[184:191], v[250:257] /*v[506:513]*/
	s_set_vgpr_msb 0x51a1
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[514:521]*/, v[32:39] /*v[288:295]*/, v[152:159], v[2:9] /*v[514:521]*/
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[522:529]*/, v[32:39] /*v[288:295]*/, v[184:191], v[10:17] /*v[522:529]*/
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[530:537]*/, v[0:7] /*v[256:263]*/, v[152:159], v[18:25] /*v[530:537]*/
	v_wmma_f32_16x16x32_bf16 v[26:33] /*v[538:545]*/, v[0:7] /*v[256:263]*/, v[184:191], v[26:33] /*v[538:545]*/
	s_set_vgpr_msb 0xa1a0
	v_wmma_f32_16x16x32_bf16 v[34:41] /*v[546:553]*/, v[224:231], v[152:159], v[34:41] /*v[546:553]*/
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[554:561]*/, v[224:231], v[184:191], v[42:49] /*v[554:561]*/
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[562:569]*/, v[192:199], v[152:159], v[50:57] /*v[562:569]*/
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[570:577]*/, v[192:199], v[184:191], v[58:65] /*v[570:577]*/
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0xa042
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
	v_lshl_or_b32 v68 /*v580*/, s44, 7, v81 /*v593*/
	v_cmp_le_i32_e32 vcc_lo, v68 /*v580*/, v87 /*v599*/
	v_dual_add_nc_u32 v117 /*v629*/, 17, v68 /*v580*/ :: v_dual_bitop2_b32 v69 /*v581*/, 2, v68 /*v580*/ bitop3:0x54
	v_dual_add_nc_u32 v118 /*v630*/, 18, v68 /*v580*/ :: v_dual_bitop2_b32 v70 /*v582*/, 3, v68 /*v580*/ bitop3:0x54
	s_set_vgpr_msb 0xaa44
	v_cndmask_b32_e32 v194 /*v450*/, 0xff800000, v194 /*v450*/, vcc_lo
	s_set_vgpr_msb 0x448a
	v_cmp_lt_i32_e32 vcc_lo, v68 /*v580*/, v87 /*v599*/
	v_dual_add_nc_u32 v119 /*v631*/, 19, v68 /*v580*/ :: v_dual_bitop2_b32 v71 /*v583*/, 4, v68 /*v580*/ bitop3:0x54
	v_dual_add_nc_u32 v122 /*v634*/, 22, v68 /*v580*/ :: v_dual_bitop2_b32 v113 /*v625*/, 5, v68 /*v580*/ bitop3:0x54
	s_set_vgpr_msb 0x8a44
	v_cndmask_b32_e32 v195 /*v451*/, 0xff800000, v195 /*v451*/, vcc_lo
	s_set_vgpr_msb 0x448a
	v_cmp_le_i32_e32 vcc_lo, v69 /*v581*/, v87 /*v599*/
	v_dual_add_nc_u32 v123 /*v635*/, 23, v68 /*v580*/ :: v_dual_bitop2_b32 v114 /*v626*/, 6, v68 /*v580*/ bitop3:0x54
	v_dual_add_nc_u32 v133 /*v645*/, 49, v68 /*v580*/ :: v_dual_bitop2_b32 v115 /*v627*/, 7, v68 /*v580*/ bitop3:0x54
	s_set_vgpr_msb 0x8a44
	v_cndmask_b32_e32 v196 /*v452*/, 0xff800000, v196 /*v452*/, vcc_lo
	s_set_vgpr_msb 0x448a
	v_cmp_le_i32_e32 vcc_lo, v70 /*v582*/, v87 /*v599*/
	v_dual_add_nc_u32 v134 /*v646*/, 50, v68 /*v580*/ :: v_dual_bitop2_b32 v116 /*v628*/, 16, v68 /*v580*/ bitop3:0x54
	v_dual_add_nc_u32 v135 /*v647*/, 51, v68 /*v580*/ :: v_dual_bitop2_b32 v124 /*v636*/, 32, v68 /*v580*/ bitop3:0x54
	s_set_vgpr_msb 0x8a44
	v_cndmask_b32_e32 v197 /*v453*/, 0xff800000, v197 /*v453*/, vcc_lo
	s_set_vgpr_msb 0x448a
	v_cmp_le_i32_e32 vcc_lo, v71 /*v583*/, v87 /*v599*/
	v_dual_add_nc_u32 v136 /*v648*/, 52, v68 /*v580*/ :: v_dual_bitop2_b32 v125 /*v637*/, 33, v68 /*v580*/ bitop3:0x54
	v_dual_add_nc_u32 v137 /*v649*/, 53, v68 /*v580*/ :: v_dual_bitop2_b32 v126 /*v638*/, 34, v68 /*v580*/ bitop3:0x54
	s_set_vgpr_msb 0x8a44
	v_cndmask_b32_e32 v198 /*v454*/, 0xff800000, v198 /*v454*/, vcc_lo
	s_set_vgpr_msb 0x448a
	v_cmp_le_i32_e32 vcc_lo, v113 /*v625*/, v87 /*v599*/
	v_dual_add_nc_u32 v138 /*v650*/, 54, v68 /*v580*/ :: v_dual_bitop2_b32 v127 /*v639*/, 35, v68 /*v580*/ bitop3:0x54
	v_or_b32_e32 v128 /*v640*/, 36, v68 /*v580*/
	v_or_b32_e32 v129 /*v641*/, 37, v68 /*v580*/
	s_set_vgpr_msb 0x8a44
	v_cndmask_b32_e32 v199 /*v455*/, 0xff800000, v199 /*v455*/, vcc_lo
	s_set_vgpr_msb 0x448a
	v_cmp_le_i32_e32 vcc_lo, v114 /*v626*/, v87 /*v599*/
	v_or_b32_e32 v130 /*v642*/, 38, v68 /*v580*/
	v_or_b32_e32 v131 /*v643*/, 39, v68 /*v580*/
	v_or_b32_e32 v132 /*v644*/, 48, v68 /*v580*/
	v_or_b32_e32 v141 /*v653*/, 0x41, v68 /*v580*/
	s_set_vgpr_msb 0x8a44
	v_cndmask_b32_e32 v200 /*v456*/, 0xff800000, v200 /*v456*/, vcc_lo
	s_set_vgpr_msb 0x448a
	v_cmp_le_i32_e32 vcc_lo, v115 /*v627*/, v87 /*v599*/
	v_or_b32_e32 v142 /*v654*/, 0x42, v68 /*v580*/
	v_or_b32_e32 v145 /*v657*/, 0x45, v68 /*v580*/
	v_or_b32_e32 v146 /*v658*/, 0x46, v68 /*v580*/
	v_add_nc_u32_e32 v149 /*v661*/, 0x51, v68 /*v580*/
	s_set_vgpr_msb 0x8a44
	v_cndmask_b32_e32 v201 /*v457*/, 0xff800000, v201 /*v457*/, vcc_lo
	s_set_vgpr_msb 0x448a
	v_cmp_le_i32_e32 vcc_lo, v116 /*v628*/, v87 /*v599*/
	v_add_nc_u32_e32 v150 /*v662*/, 0x52, v68 /*v580*/
	v_add_nc_u32_e32 v153 /*v665*/, 0x55, v68 /*v580*/
	v_add_nc_u32_e32 v154 /*v666*/, 0x56, v68 /*v580*/
	v_or_b32_e32 v157 /*v669*/, 0x61, v68 /*v580*/
	s_set_vgpr_msb 0x8a44
	v_cndmask_b32_e32 v210 /*v466*/, 0xff800000, v210 /*v466*/, vcc_lo
	s_set_vgpr_msb 0x448a
	v_cmp_le_i32_e32 vcc_lo, v117 /*v629*/, v87 /*v599*/
	v_or_b32_e32 v158 /*v670*/, 0x62, v68 /*v580*/
	v_or_b32_e32 v159 /*v671*/, 0x63, v68 /*v580*/
	v_or_b32_e32 v162 /*v674*/, 0x66, v68 /*v580*/
	v_or_b32_e32 v163 /*v675*/, 0x67, v68 /*v580*/
	s_set_vgpr_msb 0x8a44
	v_cndmask_b32_e32 v211 /*v467*/, 0xff800000, v211 /*v467*/, vcc_lo
	s_set_vgpr_msb 0x448a
	v_cmp_le_i32_e32 vcc_lo, v118 /*v630*/, v87 /*v599*/
	v_add_nc_u32_e32 v166 /*v678*/, 0x72, v68 /*v580*/
	v_add_nc_u32_e32 v171 /*v683*/, 0x77, v68 /*v580*/
	s_set_vgpr_msb 0x8a84
	v_cndmask_b32_e32 v66 /*v578*/, 0xff800000, v212 /*v468*/, vcc_lo
	s_set_vgpr_msb 0x844a
	v_add_nc_u32_e32 v212 /*v468*/, 20, v68 /*v580*/
	v_cmp_le_i32_e32 vcc_lo, v119 /*v631*/, v87 /*v599*/
	s_set_vgpr_msb 0x4a84
	v_cndmask_b32_e32 v67 /*v579*/, 0xff800000, v213 /*v469*/, vcc_lo
	s_set_vgpr_msb 0x8449
	v_add_nc_u32_e32 v213 /*v469*/, 21, v68 /*v580*/
	v_cmp_le_i32_e32 vcc_lo, v212 /*v468*/, v87 /*v599*/
	s_set_vgpr_msb 0x4944
	v_cndmask_b32_e32 v214 /*v470*/, 0xff800000, v214 /*v470*/, vcc_lo
	s_set_vgpr_msb 0x4409
	v_cmp_le_i32_e32 vcc_lo, v213 /*v469*/, v87 /*v599*/
	s_set_vgpr_msb 0x944
	v_cndmask_b32_e32 v215 /*v471*/, 0xff800000, v215 /*v471*/, vcc_lo
	s_set_vgpr_msb 0x440a
	v_cmp_le_i32_e32 vcc_lo, v122 /*v634*/, v87 /*v599*/
	s_set_vgpr_msb 0xa44
	v_cndmask_b32_e32 v216 /*v472*/, 0xff800000, v216 /*v472*/, vcc_lo
	s_set_vgpr_msb 0x440a
	v_cmp_le_i32_e32 vcc_lo, v123 /*v635*/, v87 /*v599*/
	s_set_vgpr_msb 0xa44
	v_cndmask_b32_e32 v217 /*v473*/, 0xff800000, v217 /*v473*/, vcc_lo
	s_set_vgpr_msb 0x440a
	v_cmp_le_i32_e32 vcc_lo, v124 /*v636*/, v87 /*v599*/
	s_set_vgpr_msb 0xa44
	v_cndmask_b32_e32 v226 /*v482*/, 0xff800000, v226 /*v482*/, vcc_lo
	s_set_vgpr_msb 0x440a
	v_cmp_le_i32_e32 vcc_lo, v125 /*v637*/, v87 /*v599*/
	s_set_vgpr_msb 0xa44
	v_cndmask_b32_e32 v227 /*v483*/, 0xff800000, v227 /*v483*/, vcc_lo
	s_set_vgpr_msb 0x440a
	v_cmp_le_i32_e32 vcc_lo, v126 /*v638*/, v87 /*v599*/
	s_set_vgpr_msb 0xa44
	v_cndmask_b32_e32 v228 /*v484*/, 0xff800000, v228 /*v484*/, vcc_lo
	s_set_vgpr_msb 0x440a
	v_cmp_le_i32_e32 vcc_lo, v127 /*v639*/, v87 /*v599*/
	s_set_vgpr_msb 0xa44
	v_cndmask_b32_e32 v229 /*v485*/, 0xff800000, v229 /*v485*/, vcc_lo
	s_set_vgpr_msb 0x440a
	v_cmp_le_i32_e32 vcc_lo, v128 /*v640*/, v87 /*v599*/
	s_set_vgpr_msb 0xa44
	v_cndmask_b32_e32 v230 /*v486*/, 0xff800000, v230 /*v486*/, vcc_lo
	s_set_vgpr_msb 0x440a
	v_cmp_le_i32_e32 vcc_lo, v129 /*v641*/, v87 /*v599*/
	s_set_vgpr_msb 0xa44
	v_cndmask_b32_e32 v231 /*v487*/, 0xff800000, v231 /*v487*/, vcc_lo
	s_set_vgpr_msb 0x440a
	v_cmp_le_i32_e32 vcc_lo, v130 /*v642*/, v87 /*v599*/
	s_set_vgpr_msb 0xa44
	v_cndmask_b32_e32 v232 /*v488*/, 0xff800000, v232 /*v488*/, vcc_lo
	s_set_vgpr_msb 0x440a
	v_cmp_le_i32_e32 vcc_lo, v131 /*v643*/, v87 /*v599*/
	s_set_vgpr_msb 0xa44
	v_cndmask_b32_e32 v233 /*v489*/, 0xff800000, v233 /*v489*/, vcc_lo
	s_set_vgpr_msb 0x440a
	v_cmp_le_i32_e32 vcc_lo, v132 /*v644*/, v87 /*v599*/
	s_set_vgpr_msb 0xa44
	v_cndmask_b32_e32 v242 /*v498*/, 0xff800000, v242 /*v498*/, vcc_lo
	s_set_vgpr_msb 0x440a
	v_cmp_le_i32_e32 vcc_lo, v133 /*v645*/, v87 /*v599*/
	s_set_vgpr_msb 0xa44
	v_cndmask_b32_e32 v243 /*v499*/, 0xff800000, v243 /*v499*/, vcc_lo
	s_set_vgpr_msb 0x440a
	v_cmp_le_i32_e32 vcc_lo, v134 /*v646*/, v87 /*v599*/
	s_set_vgpr_msb 0xa44
	v_cndmask_b32_e32 v244 /*v500*/, 0xff800000, v244 /*v500*/, vcc_lo
	s_set_vgpr_msb 0x440a
	v_cmp_le_i32_e32 vcc_lo, v135 /*v647*/, v87 /*v599*/
	s_set_vgpr_msb 0xa44
	v_cndmask_b32_e32 v245 /*v501*/, 0xff800000, v245 /*v501*/, vcc_lo
	s_set_vgpr_msb 0x440a
	v_cmp_le_i32_e32 vcc_lo, v136 /*v648*/, v87 /*v599*/
	s_wait_alu depctr_vm_vsrc(6)
	s_set_vgpr_msb 0xa84
	v_cndmask_b32_e32 v90 /*v602*/, 0xff800000, v246 /*v502*/, vcc_lo
	s_set_vgpr_msb 0x844a
	v_cmp_le_i32_e32 vcc_lo, v137 /*v649*/, v87 /*v599*/
	v_add_nc_u32_e32 v246 /*v502*/, 55, v68 /*v580*/
	s_set_vgpr_msb 0x4a84
	v_cndmask_b32_e32 v91 /*v603*/, 0xff800000, v247 /*v503*/, vcc_lo
	s_set_vgpr_msb 0x844a
	v_cmp_le_i32_e32 vcc_lo, v138 /*v650*/, v87 /*v599*/
	v_or_b32_e32 v247 /*v503*/, 64, v68 /*v580*/
	s_set_vgpr_msb 0x4a44
	v_cndmask_b32_e32 v248 /*v504*/, 0xff800000, v248 /*v504*/, vcc_lo
	s_set_vgpr_msb 0x4409
	v_cmp_le_i32_e32 vcc_lo, v246 /*v502*/, v87 /*v599*/
	s_set_vgpr_msb 0x944
	v_cndmask_b32_e32 v249 /*v505*/, 0xff800000, v249 /*v505*/, vcc_lo
	s_set_vgpr_msb 0x4489
	v_cmp_le_i32_e32 vcc_lo, v247 /*v503*/, v87 /*v599*/
	v_cndmask_b32_e32 v92 /*v604*/, 0xff800000, v2 /*v514*/, vcc_lo
	s_set_vgpr_msb 0x898a
	v_cmp_le_i32_e32 vcc_lo, v141 /*v653*/, v87 /*v599*/
	v_or_b32_e32 v2 /*v514*/, 0x43, v68 /*v580*/
	v_cndmask_b32_e32 v93 /*v605*/, 0xff800000, v3 /*v515*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v142 /*v654*/, v87 /*v599*/
	v_or_b32_e32 v3 /*v515*/, 0x44, v68 /*v580*/
	v_cndmask_b32_e32 v4 /*v516*/, 0xff800000, v4 /*v516*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v2 /*v514*/, v87 /*v599*/
	v_cndmask_b32_e32 v5 /*v517*/, 0xff800000, v5 /*v517*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v3 /*v515*/, v87 /*v599*/
	v_cndmask_b32_e32 v94 /*v606*/, 0xff800000, v6 /*v518*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v145 /*v657*/, v87 /*v599*/
	v_or_b32_e32 v6 /*v518*/, 0x47, v68 /*v580*/
	v_cndmask_b32_e32 v95 /*v607*/, 0xff800000, v7 /*v519*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v146 /*v658*/, v87 /*v599*/
	v_or_b32_e32 v7 /*v519*/, 0x50, v68 /*v580*/
	v_cndmask_b32_e32 v8 /*v520*/, 0xff800000, v8 /*v520*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v6 /*v518*/, v87 /*v599*/
	v_cndmask_b32_e32 v9 /*v521*/, 0xff800000, v9 /*v521*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v7 /*v519*/, v87 /*v599*/
	v_cndmask_b32_e32 v96 /*v608*/, 0xff800000, v18 /*v530*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v149 /*v661*/, v87 /*v599*/
	v_add_nc_u32_e32 v18 /*v530*/, 0x53, v68 /*v580*/
	v_cndmask_b32_e32 v97 /*v609*/, 0xff800000, v19 /*v531*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v150 /*v662*/, v87 /*v599*/
	v_add_nc_u32_e32 v19 /*v531*/, 0x54, v68 /*v580*/
	v_cndmask_b32_e32 v20 /*v532*/, 0xff800000, v20 /*v532*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v18 /*v530*/, v87 /*v599*/
	v_cndmask_b32_e32 v21 /*v533*/, 0xff800000, v21 /*v533*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v19 /*v531*/, v87 /*v599*/
	v_cndmask_b32_e32 v98 /*v610*/, 0xff800000, v22 /*v534*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v153 /*v665*/, v87 /*v599*/
	v_add_nc_u32_e32 v22 /*v534*/, 0x57, v68 /*v580*/
	v_cndmask_b32_e32 v99 /*v611*/, 0xff800000, v23 /*v535*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v154 /*v666*/, v87 /*v599*/
	v_or_b32_e32 v23 /*v535*/, 0x60, v68 /*v580*/
	v_cndmask_b32_e32 v24 /*v536*/, 0xff800000, v24 /*v536*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v22 /*v534*/, v87 /*v599*/
	v_cndmask_b32_e32 v25 /*v537*/, 0xff800000, v25 /*v537*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v23 /*v535*/, v87 /*v599*/
	v_cndmask_b32_e32 v34 /*v546*/, 0xff800000, v34 /*v546*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v157 /*v669*/, v87 /*v599*/
	v_cndmask_b32_e32 v35 /*v547*/, 0xff800000, v35 /*v547*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v158 /*v670*/, v87 /*v599*/
	v_cndmask_b32_e32 v100 /*v612*/, 0xff800000, v36 /*v548*/, vcc_lo
	v_or_b32_e32 v36 /*v548*/, 0x64, v68 /*v580*/
	v_cmp_le_i32_e32 vcc_lo, v159 /*v671*/, v87 /*v599*/
	v_cndmask_b32_e32 v101 /*v613*/, 0xff800000, v37 /*v549*/, vcc_lo
	v_or_b32_e32 v37 /*v549*/, 0x65, v68 /*v580*/
	v_cmp_le_i32_e32 vcc_lo, v36 /*v548*/, v87 /*v599*/
	v_cndmask_b32_e32 v38 /*v550*/, 0xff800000, v38 /*v550*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v37 /*v549*/, v87 /*v599*/
	v_cndmask_b32_e32 v39 /*v551*/, 0xff800000, v39 /*v551*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v162 /*v674*/, v87 /*v599*/
	v_cndmask_b32_e32 v102 /*v614*/, 0xff800000, v40 /*v552*/, vcc_lo
	v_or_b32_e32 v40 /*v552*/, 0x70, v68 /*v580*/
	v_cmp_le_i32_e32 vcc_lo, v163 /*v675*/, v87 /*v599*/
	v_cndmask_b32_e32 v103 /*v615*/, 0xff800000, v41 /*v553*/, vcc_lo
	v_add_nc_u32_e32 v41 /*v553*/, 0x71, v68 /*v580*/
	v_cmp_le_i32_e32 vcc_lo, v40 /*v552*/, v87 /*v599*/
	v_cndmask_b32_e32 v104 /*v616*/, 0xff800000, v50 /*v562*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v41 /*v553*/, v87 /*v599*/
	v_add_nc_u32_e32 v50 /*v562*/, 0x73, v68 /*v580*/
	v_cndmask_b32_e32 v105 /*v617*/, 0xff800000, v51 /*v563*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v166 /*v678*/, v87 /*v599*/
	v_add_nc_u32_e32 v51 /*v563*/, 0x74, v68 /*v580*/
	v_cndmask_b32_e32 v106 /*v618*/, 0xff800000, v52 /*v564*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v50 /*v562*/, v87 /*v599*/
	v_add_nc_u32_e32 v52 /*v564*/, 0x75, v68 /*v580*/
	v_cndmask_b32_e32 v107 /*v619*/, 0xff800000, v53 /*v565*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v51 /*v563*/, v87 /*v599*/
	v_add_nc_u32_e32 v53 /*v565*/, 0x76, v68 /*v580*/
	v_cndmask_b32_e32 v54 /*v566*/, 0xff800000, v54 /*v566*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v52 /*v564*/, v87 /*v599*/
	v_cndmask_b32_e32 v55 /*v567*/, 0xff800000, v55 /*v567*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v53 /*v565*/, v87 /*v599*/
	v_cndmask_b32_e32 v56 /*v568*/, 0xff800000, v56 /*v568*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v171 /*v683*/, v87 /*v599*/
	v_cndmask_b32_e32 v57 /*v569*/, 0xff800000, v57 /*v569*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v68 /*v580*/, v88 /*v600*/
	s_set_vgpr_msb 0x8a84
	v_cndmask_b32_e32 v108 /*v620*/, 0xff800000, v202 /*v458*/, vcc_lo
	s_set_vgpr_msb 0x840a
	v_cmp_lt_i32_e32 vcc_lo, v68 /*v580*/, v88 /*v600*/
	s_set_vgpr_msb 0xa55
	v_max3_num_f32 v202 /*v458*/, v194 /*v450*/, v195 /*v451*/, v196 /*v452*/
	s_set_vgpr_msb 0x5584
	v_cndmask_b32_e32 v109 /*v621*/, 0xff800000, v203 /*v459*/, vcc_lo
	s_set_vgpr_msb 0x840a
	v_cmp_le_i32_e32 vcc_lo, v69 /*v581*/, v88 /*v600*/
	s_set_vgpr_msb 0xa84
	v_cndmask_b32_e32 v110 /*v622*/, 0xff800000, v204 /*v460*/, vcc_lo
	s_set_vgpr_msb 0x840a
	v_cmp_le_i32_e32 vcc_lo, v70 /*v582*/, v88 /*v600*/
	s_set_vgpr_msb 0xa55
	v_max3_num_f32 v204 /*v460*/, v197 /*v453*/, v198 /*v454*/, v199 /*v455*/
	s_set_vgpr_msb 0x5584
	v_cndmask_b32_e32 v111 /*v623*/, 0xff800000, v205 /*v461*/, vcc_lo
	s_set_vgpr_msb 0x846a
	v_cmp_le_i32_e32 vcc_lo, v71 /*v583*/, v88 /*v600*/
	v_max3_num_f32 v203 /*v459*/, v108 /*v620*/, v109 /*v621*/, v110 /*v622*/
	s_set_vgpr_msb 0x6a84
	v_cndmask_b32_e32 v112 /*v624*/, 0xff800000, v206 /*v462*/, vcc_lo
	s_set_vgpr_msb 0x840a
	v_cmp_le_i32_e32 vcc_lo, v113 /*v625*/, v88 /*v600*/
	s_set_vgpr_msb 0xa55
	v_max3_num_f32 v206 /*v462*/, v200 /*v456*/, v201 /*v457*/, v210 /*v466*/
	s_set_vgpr_msb 0x5584
	v_cndmask_b32_e32 v113 /*v625*/, 0xff800000, v207 /*v463*/, vcc_lo
	s_set_vgpr_msb 0x840a
	v_cmp_le_i32_e32 vcc_lo, v114 /*v626*/, v88 /*v600*/
	s_set_vgpr_msb 0xa55
	v_max3_num_f32 v202 /*v458*/, v202 /*v458*/, v204 /*v460*/, v206 /*v462*/
	s_set_vgpr_msb 0x556a
	v_max3_num_f32 v205 /*v461*/, v111 /*v623*/, v112 /*v624*/, v113 /*v625*/
	s_set_vgpr_msb 0x6a84
	v_cndmask_b32_e32 v114 /*v626*/, 0xff800000, v208 /*v464*/, vcc_lo
	s_set_vgpr_msb 0x840a
	v_cmp_le_i32_e32 vcc_lo, v115 /*v627*/, v88 /*v600*/
	s_set_vgpr_msb 0xa69
	v_max3_num_f32 v208 /*v464*/, v211 /*v467*/, v66 /*v578*/, v67 /*v579*/
	s_set_vgpr_msb 0x6984
	v_cndmask_b32_e32 v115 /*v627*/, 0xff800000, v209 /*v465*/, vcc_lo
	s_set_vgpr_msb 0x840a
	v_cmp_le_i32_e32 vcc_lo, v116 /*v628*/, v88 /*v600*/
	s_set_vgpr_msb 0xa84
	v_cndmask_b32_e32 v116 /*v628*/, 0xff800000, v218 /*v474*/, vcc_lo
	s_set_vgpr_msb 0x840a
	v_cmp_le_i32_e32 vcc_lo, v117 /*v629*/, v88 /*v600*/
	s_set_vgpr_msb 0xa55
	v_max3_num_f32 v218 /*v474*/, v217 /*v473*/, v226 /*v482*/, v227 /*v483*/
	s_set_vgpr_msb 0x5584
	v_cndmask_b32_e32 v117 /*v629*/, 0xff800000, v219 /*v475*/, vcc_lo
	s_set_vgpr_msb 0x846a
	v_cmp_le_i32_e32 vcc_lo, v118 /*v630*/, v88 /*v600*/
	v_max3_num_f32 v207 /*v463*/, v114 /*v626*/, v115 /*v627*/, v116 /*v628*/
	s_set_vgpr_msb 0x6a84
	v_cndmask_b32_e32 v118 /*v630*/, 0xff800000, v220 /*v476*/, vcc_lo
	s_set_vgpr_msb 0x840a
	v_cmp_le_i32_e32 vcc_lo, v119 /*v631*/, v88 /*v600*/
	s_set_vgpr_msb 0xa55
	v_max3_num_f32 v220 /*v476*/, v228 /*v484*/, v229 /*v485*/, v230 /*v486*/
	v_max3_num_f32 v203 /*v459*/, v203 /*v459*/, v205 /*v461*/, v207 /*v463*/
	s_set_vgpr_msb 0x5584
	v_cndmask_b32_e32 v119 /*v631*/, 0xff800000, v221 /*v477*/, vcc_lo
	s_set_vgpr_msb 0x8409
	v_cmp_le_i32_e32 vcc_lo, v212 /*v468*/, v88 /*v600*/
	s_set_vgpr_msb 0x955
	v_max3_num_f32 v212 /*v468*/, v214 /*v470*/, v215 /*v471*/, v216 /*v472*/
	s_set_vgpr_msb 0x556a
	v_max3_num_f32 v209 /*v465*/, v117 /*v629*/, v118 /*v630*/, v119 /*v631*/
	s_set_vgpr_msb 0x6a84
	v_cndmask_b32_e32 v120 /*v632*/, 0xff800000, v222 /*v478*/, vcc_lo
	s_set_vgpr_msb 0x8409
	v_cmp_le_i32_e32 vcc_lo, v213 /*v469*/, v88 /*v600*/
	s_set_vgpr_msb 0x955
	v_max3_num_f32 v222 /*v478*/, v231 /*v487*/, v232 /*v488*/, v233 /*v489*/
	v_max3_num_f32 v204 /*v460*/, v208 /*v464*/, v212 /*v468*/, v218 /*v474*/
	s_set_vgpr_msb 0x5584
	v_cndmask_b32_e32 v121 /*v633*/, 0xff800000, v223 /*v479*/, vcc_lo
	s_set_vgpr_msb 0x840a
	v_cmp_le_i32_e32 vcc_lo, v122 /*v634*/, v88 /*v600*/
	s_set_vgpr_msb 0xa84
	v_cndmask_b32_e32 v122 /*v634*/, 0xff800000, v224 /*v480*/, vcc_lo
	s_set_vgpr_msb 0x840a
	v_cmp_le_i32_e32 vcc_lo, v123 /*v635*/, v88 /*v600*/
	s_set_vgpr_msb 0xa55
	v_max3_num_f32 v224 /*v480*/, v242 /*v498*/, v243 /*v499*/, v244 /*v500*/
	s_set_vgpr_msb 0x5584
	v_cndmask_b32_e32 v123 /*v635*/, 0xff800000, v225 /*v481*/, vcc_lo
	s_set_vgpr_msb 0x846a
	v_cmp_le_i32_e32 vcc_lo, v124 /*v636*/, v88 /*v600*/
	v_max3_num_f32 v213 /*v469*/, v120 /*v632*/, v121 /*v633*/, v122 /*v634*/
	s_set_vgpr_msb 0x6a55
	v_max3_num_f32 v205 /*v461*/, v220 /*v476*/, v222 /*v478*/, v224 /*v480*/
	s_set_vgpr_msb 0x5584
	v_cndmask_b32_e32 v124 /*v636*/, 0xff800000, v234 /*v490*/, vcc_lo
	s_set_vgpr_msb 0x840a
	v_cmp_le_i32_e32 vcc_lo, v125 /*v637*/, v88 /*v600*/
	s_set_vgpr_msb 0xa69
	v_max3_num_f32 v234 /*v490*/, v245 /*v501*/, v90 /*v602*/, v91 /*v603*/
	s_set_vgpr_msb 0x6955
	v_max3_num_f32 v202 /*v458*/, v202 /*v458*/, v204 /*v460*/, v205 /*v461*/
	s_set_vgpr_msb 0x5584
	v_cndmask_b32_e32 v125 /*v637*/, 0xff800000, v235 /*v491*/, vcc_lo
	s_set_vgpr_msb 0x846a
	v_cmp_le_i32_e32 vcc_lo, v126 /*v638*/, v88 /*v600*/
	v_max3_num_f32 v219 /*v475*/, v123 /*v635*/, v124 /*v636*/, v125 /*v637*/
	s_set_vgpr_msb 0x6a84
	v_cndmask_b32_e32 v126 /*v638*/, 0xff800000, v236 /*v492*/, vcc_lo
	s_set_vgpr_msb 0x840a
	v_cmp_le_i32_e32 vcc_lo, v127 /*v639*/, v88 /*v600*/
	s_set_vgpr_msb 0xa65
	v_max3_num_f32 v236 /*v492*/, v248 /*v504*/, v249 /*v505*/, v92 /*v604*/
	s_set_vgpr_msb 0x6555
	v_max3_num_f32 v209 /*v465*/, v209 /*v465*/, v213 /*v469*/, v219 /*v475*/
	s_set_vgpr_msb 0x5584
	v_cndmask_b32_e32 v127 /*v639*/, 0xff800000, v237 /*v493*/, vcc_lo
	s_set_vgpr_msb 0x840a
	v_cmp_le_i32_e32 vcc_lo, v128 /*v640*/, v88 /*v600*/
	s_set_vgpr_msb 0xa84
	v_cndmask_b32_e32 v128 /*v640*/, 0xff800000, v238 /*v494*/, vcc_lo
	s_set_vgpr_msb 0x846a
	v_cmp_le_i32_e32 vcc_lo, v129 /*v641*/, v88 /*v600*/
	v_max3_num_f32 v238 /*v494*/, v93 /*v605*/, v4 /*v516*/, v5 /*v517*/
	s_set_vgpr_msb 0x6a84
	v_cndmask_b32_e32 v129 /*v641*/, 0xff800000, v239 /*v495*/, vcc_lo
	s_set_vgpr_msb 0x846a
	v_cmp_le_i32_e32 vcc_lo, v130 /*v642*/, v88 /*v600*/
	v_max3_num_f32 v221 /*v477*/, v126 /*v638*/, v127 /*v639*/, v128 /*v640*/
	s_set_vgpr_msb 0x6a55
	v_max3_num_f32 v206 /*v462*/, v234 /*v490*/, v236 /*v492*/, v238 /*v494*/
	s_set_vgpr_msb 0x5584
	v_cndmask_b32_e32 v130 /*v642*/, 0xff800000, v240 /*v496*/, vcc_lo
	s_set_vgpr_msb 0x846a
	v_cmp_le_i32_e32 vcc_lo, v131 /*v643*/, v88 /*v600*/
	v_max3_num_f32 v240 /*v496*/, v94 /*v606*/, v95 /*v607*/, v8 /*v520*/
	s_set_vgpr_msb 0x6a84
	v_cndmask_b32_e32 v131 /*v643*/, 0xff800000, v241 /*v497*/, vcc_lo
	s_set_vgpr_msb 0x846a
	v_cmp_le_i32_e32 vcc_lo, v132 /*v644*/, v88 /*v600*/
	v_max3_num_f32 v223 /*v479*/, v129 /*v641*/, v130 /*v642*/, v131 /*v643*/
	s_set_vgpr_msb 0x6a84
	v_cndmask_b32_e32 v132 /*v644*/, 0xff800000, v250 /*v506*/, vcc_lo
	s_set_vgpr_msb 0x846a
	v_cmp_le_i32_e32 vcc_lo, v133 /*v645*/, v88 /*v600*/
	v_max3_num_f32 v250 /*v506*/, v20 /*v532*/, v21 /*v533*/, v98 /*v610*/
	s_set_vgpr_msb 0x6a84
	v_cndmask_b32_e32 v133 /*v645*/, 0xff800000, v251 /*v507*/, vcc_lo
	s_set_vgpr_msb 0x840a
	v_cmp_le_i32_e32 vcc_lo, v134 /*v646*/, v88 /*v600*/
	s_set_vgpr_msb 0xa84
	v_cndmask_b32_e32 v134 /*v646*/, 0xff800000, v252 /*v508*/, vcc_lo
	s_set_vgpr_msb 0x846a
	v_cmp_le_i32_e32 vcc_lo, v135 /*v647*/, v88 /*v600*/
	v_max3_num_f32 v252 /*v508*/, v99 /*v611*/, v24 /*v536*/, v25 /*v537*/
	s_set_vgpr_msb 0x6a84
	v_cndmask_b32_e32 v135 /*v647*/, 0xff800000, v253 /*v509*/, vcc_lo
	s_set_vgpr_msb 0x846a
	v_cmp_le_i32_e32 vcc_lo, v136 /*v648*/, v88 /*v600*/
	v_max3_num_f32 v225 /*v481*/, v132 /*v644*/, v133 /*v645*/, v134 /*v646*/
	s_set_vgpr_msb 0x6a84
	v_cndmask_b32_e32 v136 /*v648*/, 0xff800000, v254 /*v510*/, vcc_lo
	s_set_vgpr_msb 0x846a
	v_cmp_le_i32_e32 vcc_lo, v137 /*v649*/, v88 /*v600*/
	v_max3_num_f32 v254 /*v510*/, v34 /*v546*/, v35 /*v547*/, v100 /*v612*/
	s_set_vgpr_msb 0x6a55
	v_max3_num_f32 v213 /*v469*/, v221 /*v477*/, v223 /*v479*/, v225 /*v481*/
	s_set_vgpr_msb 0x5584
	v_cndmask_b32_e32 v137 /*v649*/, 0xff800000, v255 /*v511*/, vcc_lo
	s_set_vgpr_msb 0x840a
	v_cmp_le_i32_e32 vcc_lo, v138 /*v650*/, v88 /*v600*/
	s_set_vgpr_msb 0xa55
	v_max3_num_f32 v203 /*v459*/, v203 /*v459*/, v209 /*v465*/, v213 /*v469*/
	s_set_vgpr_msb 0x556a
	v_max3_num_f32 v235 /*v491*/, v135 /*v647*/, v136 /*v648*/, v137 /*v649*/
	s_set_vgpr_msb 0x6a89
	v_cndmask_b32_e32 v138 /*v650*/, 0xff800000, v0 /*v512*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v246 /*v502*/, v88 /*v600*/
	s_set_vgpr_msb 0x896a
	v_max3_num_f32 v246 /*v502*/, v9 /*v521*/, v96 /*v608*/, v97 /*v609*/
	s_set_vgpr_msb 0x6aaa
	v_max3_num_f32 v0 /*v512*/, v101 /*v613*/, v38 /*v550*/, v39 /*v551*/
	v_cndmask_b32_e32 v139 /*v651*/, 0xff800000, v1 /*v513*/, vcc_lo
	s_set_vgpr_msb 0xaa09
	v_cmp_le_i32_e32 vcc_lo, v247 /*v503*/, v88 /*v600*/
	s_set_vgpr_msb 0x955
	v_max3_num_f32 v207 /*v463*/, v240 /*v496*/, v246 /*v502*/, v250 /*v506*/
	s_set_vgpr_msb 0x5565
	v_max3_num_f32 v208 /*v464*/, v252 /*v508*/, v254 /*v510*/, v0 /*v512*/
	s_set_vgpr_msb 0x65aa
	v_cndmask_b32_e32 v140 /*v652*/, 0xff800000, v10 /*v522*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v141 /*v653*/, v88 /*v600*/
	v_max3_num_f32 v10 /*v522*/, v54 /*v566*/, v55 /*v567*/, v56 /*v568*/
	s_set_vgpr_msb 0xaa55
	v_max3_num_f32 v204 /*v460*/, v206 /*v462*/, v207 /*v463*/, v208 /*v464*/
	s_set_vgpr_msb 0x558a
	v_cndmask_b32_e32 v141 /*v653*/, 0xff800000, v11 /*v523*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v142 /*v654*/, v88 /*v600*/
	s_set_vgpr_msb 0x8a6a
	v_max3_num_f32 v237 /*v493*/, v138 /*v650*/, v139 /*v651*/, v140 /*v652*/
	s_set_vgpr_msb 0x6a8a
	v_cndmask_b32_e32 v142 /*v654*/, 0xff800000, v12 /*v524*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v2 /*v514*/, v88 /*v600*/
	v_dual_max_num_f32 v2 /*v514*/, v102 /*v614*/, v103 /*v615*/ :: v_dual_cndmask_b32 v143 /*v655*/, 0xff800000, v13 /*v525*/
	v_cmp_le_i32_e32 vcc_lo, v3 /*v515*/, v88 /*v600*/
	s_set_vgpr_msb 0x8a6a
	v_max3_num_f32 v239 /*v495*/, v141 /*v653*/, v142 /*v654*/, v143 /*v655*/
	s_set_vgpr_msb 0x6a8a
	v_cndmask_b32_e32 v144 /*v656*/, 0xff800000, v14 /*v526*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v145 /*v657*/, v88 /*v600*/
	s_set_vgpr_msb 0x8a55
	v_max3_num_f32 v206 /*v462*/, v235 /*v491*/, v237 /*v493*/, v239 /*v495*/
	s_set_vgpr_msb 0x55aa
	v_cndmask_b32_e32 v145 /*v657*/, 0xff800000, v15 /*v527*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v146 /*v658*/, v88 /*v600*/
	v_cndmask_b32_e32 v146 /*v658*/, 0xff800000, v16 /*v528*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v6 /*v518*/, v88 /*v600*/
	v_max3_num_f32 v6 /*v518*/, v105 /*v617*/, v106 /*v618*/, v107 /*v619*/
	v_cndmask_b32_e32 v147 /*v659*/, 0xff800000, v17 /*v529*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v7 /*v519*/, v88 /*v600*/
	s_set_vgpr_msb 0xaa6a
	v_max3_num_f32 v241 /*v497*/, v144 /*v656*/, v145 /*v657*/, v146 /*v658*/
	v_max3_num_f32 v212 /*v468*/, v2 /*v514*/, v104 /*v616*/, v6 /*v518*/
	s_set_vgpr_msb 0x6a8a
	v_cndmask_b32_e32 v148 /*v660*/, 0xff800000, v26 /*v538*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v149 /*v661*/, v88 /*v600*/
	s_set_vgpr_msb 0x8a69
	v_max3_num_f32 v205 /*v461*/, v212 /*v468*/, v10 /*v522*/, v57 /*v569*/
	s_set_vgpr_msb 0x698a
	v_cndmask_b32_e32 v149 /*v661*/, 0xff800000, v27 /*v539*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v150 /*v662*/, v88 /*v600*/
	s_set_vgpr_msb 0x8a55
	v_max3_num_f32 v202 /*v458*/, v202 /*v458*/, v204 /*v460*/, v205 /*v461*/
	s_set_vgpr_msb 0x556a
	v_max3_num_f32 v247 /*v503*/, v147 /*v659*/, v148 /*v660*/, v149 /*v661*/
	s_set_vgpr_msb 0x6a8a
	v_cndmask_b32_e32 v150 /*v662*/, 0xff800000, v28 /*v540*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v18 /*v530*/, v88 /*v600*/
	v_cndmask_b32_e32 v151 /*v663*/, 0xff800000, v29 /*v541*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v19 /*v531*/, v88 /*v600*/
	v_cndmask_b32_e32 v152 /*v664*/, 0xff800000, v30 /*v542*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v153 /*v665*/, v88 /*v600*/
	v_cndmask_b32_e32 v153 /*v665*/, 0xff800000, v31 /*v543*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v154 /*v666*/, v88 /*v600*/
	s_set_vgpr_msb 0x8a6a
	v_max3_num_f32 v251 /*v507*/, v150 /*v662*/, v151 /*v663*/, v152 /*v664*/
	s_set_vgpr_msb 0x6a8a
	v_cndmask_b32_e32 v154 /*v666*/, 0xff800000, v32 /*v544*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v22 /*v534*/, v88 /*v600*/
	s_set_vgpr_msb 0x8a55
	v_max3_num_f32 v207 /*v463*/, v241 /*v497*/, v247 /*v503*/, v251 /*v507*/
	s_set_vgpr_msb 0x558a
	v_cndmask_b32_e32 v155 /*v667*/, 0xff800000, v33 /*v545*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v23 /*v535*/, v88 /*v600*/
	s_set_vgpr_msb 0x8a6a
	v_max3_num_f32 v253 /*v509*/, v153 /*v665*/, v154 /*v666*/, v155 /*v667*/
	s_set_vgpr_msb 0x6a8a
	v_cndmask_b32_e32 v156 /*v668*/, 0xff800000, v42 /*v554*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v157 /*v669*/, v88 /*v600*/
	v_cndmask_b32_e32 v157 /*v669*/, 0xff800000, v43 /*v555*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v158 /*v670*/, v88 /*v600*/
	v_cndmask_b32_e32 v158 /*v670*/, 0xff800000, v44 /*v556*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v159 /*v671*/, v88 /*v600*/
	v_cndmask_b32_e32 v159 /*v671*/, 0xff800000, v45 /*v557*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v36 /*v548*/, v88 /*v600*/
	s_set_vgpr_msb 0x8a6a
	v_max3_num_f32 v255 /*v511*/, v156 /*v668*/, v157 /*v669*/, v158 /*v670*/
	s_set_vgpr_msb 0x6aaa
	v_cndmask_b32_e32 v160 /*v672*/, 0xff800000, v46 /*v558*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v37 /*v549*/, v88 /*v600*/
	v_cndmask_b32_e32 v161 /*v673*/, 0xff800000, v47 /*v559*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v162 /*v674*/, v88 /*v600*/
	v_max3_num_f32 v1 /*v513*/, v159 /*v671*/, v160 /*v672*/, v161 /*v673*/
	v_cndmask_b32_e32 v162 /*v674*/, 0xff800000, v48 /*v560*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v163 /*v675*/, v88 /*v600*/
	s_set_vgpr_msb 0xaa65
	v_max3_num_f32 v208 /*v464*/, v253 /*v509*/, v255 /*v511*/, v1 /*v513*/
	s_set_vgpr_msb 0x658a
	v_cndmask_b32_e32 v163 /*v675*/, 0xff800000, v49 /*v561*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v40 /*v552*/, v88 /*v600*/
	s_set_vgpr_msb 0x8a55
	v_max3_num_f32 v204 /*v460*/, v206 /*v462*/, v207 /*v463*/, v208 /*v464*/
	s_set_vgpr_msb 0x55aa
	v_dual_max_num_f32 v3 /*v515*/, v162 /*v674*/, v163 /*v675*/ :: v_dual_cndmask_b32 v164 /*v676*/, 0xff800000, v58 /*v570*/
	v_cmp_le_i32_e32 vcc_lo, v41 /*v553*/, v88 /*v600*/
	v_cndmask_b32_e32 v165 /*v677*/, 0xff800000, v59 /*v571*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v166 /*v678*/, v88 /*v600*/
	v_cndmask_b32_e32 v166 /*v678*/, 0xff800000, v60 /*v572*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v50 /*v562*/, v88 /*v600*/
	v_cndmask_b32_e32 v167 /*v679*/, 0xff800000, v61 /*v573*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v51 /*v563*/, v88 /*v600*/
	v_max3_num_f32 v7 /*v519*/, v165 /*v677*/, v166 /*v678*/, v167 /*v679*/
	v_cndmask_b32_e32 v168 /*v680*/, 0xff800000, v62 /*v574*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v52 /*v564*/, v88 /*v600*/
	s_set_vgpr_msb 0xaa6a
	v_max3_num_f32 v212 /*v468*/, v3 /*v515*/, v164 /*v676*/, v7 /*v519*/
	s_set_vgpr_msb 0x6aaa
	v_cndmask_b32_e32 v169 /*v681*/, 0xff800000, v63 /*v575*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v53 /*v565*/, v88 /*v600*/
	v_cndmask_b32_e32 v170 /*v682*/, 0xff800000, v64 /*v576*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v171 /*v683*/, v88 /*v600*/
	v_cndmask_b32_e32 v171 /*v683*/, 0xff800000, v65 /*v577*/, vcc_lo
	v_max3_num_f32 v11 /*v523*/, v168 /*v680*/, v169 /*v681*/, v170 /*v682*/
	s_set_vgpr_msb 0xaa69
	v_max3_num_f32 v205 /*v461*/, v212 /*v468*/, v11 /*v523*/, v171 /*v683*/
	s_set_vgpr_msb 0x6955
	v_max3_num_f32 v203 /*v459*/, v203 /*v459*/, v204 /*v460*/, v205 /*v461*/
	v_dual_mov_b32 v206 /*v462*/, v202 /*v458*/ :: v_dual_mov_b32 v204 /*v460*/, v203 /*v459*/
	v_permlanex16_b32 v206 /*v462*/, v206 /*v462*/, s35, 0xfedcba98
	v_permlanex16_b32 v204 /*v460*/, v204 /*v460*/, s35, 0xfedcba98
	v_dual_max_num_f32 v202 /*v458*/, v202 /*v458*/, v206 /*v462*/ :: v_dual_max_num_f32 v203 /*v459*/, v203 /*v459*/, v204 /*v460*/
	s_set_vgpr_msb 0x5549
	v_sub_f32_e32 v205 /*v461*/, v202 /*v458*/, v86 /*v598*/
	v_max_num_f32_e32 v202 /*v458*/, v202 /*v458*/, v86 /*v598*/
	v_sub_f32_e32 v204 /*v460*/, v203 /*v459*/, v85 /*v597*/
	s_set_vgpr_msb 0x4904
	v_cmp_lt_f32_e32 vcc_lo, 0x41000000, v205 /*v461*/
	s_cmp_eq_u32 vcc_lo, 0
	s_cselect_b32 s2, -1, 0
	s_set_vgpr_msb 0x489
	v_cndmask_b32_e64 v69 /*v581*/, v202 /*v458*/, v86 /*v598*/, s2
	s_set_vgpr_msb 0x8946
	v_cmp_lt_f32_e64 s2, 0x41000000, v204 /*v460*/
	v_max_num_f32_e32 v202 /*v458*/, v85 /*v597*/, v203 /*v459*/
	s_set_vgpr_msb 0x4688
	v_mul_f32_e32 v60 /*v572*/, 0xbfb8aa3b, v69 /*v581*/
	s_cmp_lg_u32 s2, 0
	s_cselect_b32 s5, -1, 0
	s_cmp_eq_u32 s2, 0
	s_set_vgpr_msb 0x8862
	v_pk_fma_f32 v[208:209] /*v[464:465]*/, v[66:67] /*v[578:579]*/, s[46:47], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_cselect_b32 s2, -1, 0
	s_set_vgpr_msb 0x6261
	v_pk_fma_f32 v[200:201] /*v[456:457]*/, v[200:201] /*v[456:457]*/, s[46:47], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6189
	v_cndmask_b32_e64 v71 /*v583*/, v202 /*v458*/, v85 /*v597*/, s2
	s_set_vgpr_msb 0x8961
	v_pk_fma_f32 v[194:195] /*v[450:451]*/, v[194:195] /*v[450:451]*/, s[46:47], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v238 /*v494*/, v208 /*v464*/
	v_exp_f32_e32 v250 /*v506*/, v209 /*v465*/
	v_nop
	v_pk_fma_f32 v[208:209] /*v[464:465]*/, v[226:227] /*v[482:483]*/, s[46:47], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6188
	v_mul_f32_e32 v68 /*v580*/, 0xbfb8aa3b, v71 /*v583*/
	s_set_vgpr_msb 0x8861
	v_pk_fma_f32 v[226:227] /*v[482:483]*/, v[244:245] /*v[500:501]*/, s[46:47], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[204:205] /*v[460:461]*/, v[198:199] /*v[454:455]*/, s[46:47], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[206:207] /*v[462:463]*/, v[210:211] /*v[466:467]*/, s[46:47], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v212 /*v468*/, v200 /*v456*/
	s_set_vgpr_msb 0x61a2
	v_pk_fma_f32 v[66:67] /*v[578:579]*/, v[108:109] /*v[620:621]*/, s[46:47], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa281
	v_exp_f32_e32 v6 /*v518*/, v226 /*v482*/
	v_exp_f32_e32 v18 /*v530*/, v227 /*v483*/
	v_nop
	s_set_vgpr_msb 0x8162
	v_pk_fma_f32 v[226:227] /*v[482:483]*/, v[92:93] /*v[604:605]*/, s[46:47], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x62a2
	v_pk_fma_f32 v[92:93] /*v[604:605]*/, v[112:113] /*v[624:625]*/, s[46:47], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa261
	v_exp_f32_e32 v220 /*v476*/, v201 /*v457*/
	v_nop
	v_pk_fma_f32 v[200:201] /*v[456:457]*/, v[214:215] /*v[470:471]*/, s[46:47], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[214:215] /*v[470:471]*/, v[228:229] /*v[484:485]*/, s[46:47], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[222:223] /*v[478:479]*/, v[232:233] /*v[488:489]*/, s[46:47], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[202:203] /*v[458:459]*/, v[196:197] /*v[452:453]*/, s[46:47], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v196 /*v452*/, v195 /*v451*/
	v_exp_f32_e32 v210 /*v466*/, v205 /*v461*/
	s_set_vgpr_msb 0x6142
	v_exp_f32_e32 v195 /*v451*/, v66 /*v578*/
	v_exp_f32_e32 v197 /*v453*/, v67 /*v579*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[66:67] /*v[578:579]*/, v[114:115] /*v[626:627]*/, s[46:47], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa242
	v_exp_f32_e32 v205 /*v461*/, v92 /*v604*/
	v_exp_f32_e32 v211 /*v467*/, v93 /*v605*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[92:93] /*v[604:605]*/, v[118:119] /*v[630:631]*/, s[46:47], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa261
	v_exp_f32_e32 v224 /*v480*/, v206 /*v462*/
	v_exp_f32_e32 v234 /*v490*/, v207 /*v463*/
	v_nop
	v_pk_fma_f32 v[206:207] /*v[462:463]*/, v[216:217] /*v[472:473]*/, s[46:47], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v254 /*v510*/, v200 /*v456*/
	v_exp_f32_e32 v200 /*v456*/, v208 /*v464*/
	v_pk_fma_f32 v[218:219] /*v[474:475]*/, v[230:231] /*v[486:487]*/, s[46:47], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v208 /*v464*/, v214 /*v470*/
	v_exp_f32_e32 v216 /*v472*/, v215 /*v471*/
	v_nop
	v_pk_fma_f32 v[214:215] /*v[470:471]*/, v[242:243] /*v[498:499]*/, s[46:47], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v230 /*v486*/, v222 /*v478*/
	v_exp_f32_e32 v242 /*v498*/, v223 /*v479*/
	v_nop
	s_set_vgpr_msb 0x6162
	v_pk_fma_f32 v[222:223] /*v[478:479]*/, v[90:91] /*v[602:603]*/, s[46:47], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x62a2
	v_pk_fma_f32 v[90:91] /*v[602:603]*/, v[110:111] /*v[622:623]*/, s[46:47], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa242
	v_exp_f32_e32 v213 /*v469*/, v66 /*v578*/
	v_exp_f32_e32 v221 /*v477*/, v67 /*v579*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[66:67] /*v[578:579]*/, v[120:121] /*v[632:633]*/, s[46:47], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa242
	v_exp_f32_e32 v239 /*v495*/, v92 /*v604*/
	v_exp_f32_e32 v251 /*v507*/, v93 /*v605*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[92:93] /*v[604:605]*/, v[124:125] /*v[636:637]*/, s[46:47], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa241
	v_exp_f32_e32 v198 /*v454*/, v202 /*v458*/
	v_exp_f32_e32 v202 /*v458*/, v203 /*v459*/
	s_set_vgpr_msb 0x4142
	v_exp_f32_e32 v199 /*v455*/, v90 /*v602*/
	v_exp_f32_e32 v203 /*v459*/, v91 /*v603*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[90:91] /*v[602:603]*/, v[116:117] /*v[628:629]*/, s[46:47], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa281
	v_exp_f32_e32 v10 /*v522*/, v201 /*v457*/
	v_exp_f32_e32 v26 /*v538*/, v207 /*v463*/
	s_set_vgpr_msb 0x8142
	v_exp_f32_e32 v255 /*v511*/, v66 /*v578*/
	s_set_vgpr_msb 0x42a2
	v_exp_f32_e32 v11 /*v523*/, v67 /*v579*/
	v_nop
	v_pk_fma_f32 v[66:67] /*v[578:579]*/, v[126:127] /*v[638:639]*/, s[46:47], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa242
	v_exp_f32_e32 v201 /*v457*/, v92 /*v604*/
	v_exp_f32_e32 v207 /*v463*/, v93 /*v605*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[92:93] /*v[604:605]*/, v[130:131] /*v[642:643]*/, s[46:47], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa242
	v_exp_f32_e32 v225 /*v481*/, v90 /*v602*/
	v_exp_f32_e32 v235 /*v491*/, v91 /*v603*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[90:91] /*v[602:603]*/, v[122:123] /*v[634:635]*/, s[46:47], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa281
	v_exp_f32_e32 v14 /*v526*/, v206 /*v462*/
	s_set_vgpr_msb 0x8141
	v_exp_f32_e32 v206 /*v462*/, v209 /*v465*/
	s_set_vgpr_msb 0x4142
	v_exp_f32_e32 v209 /*v465*/, v66 /*v578*/
	v_exp_f32_e32 v217 /*v473*/, v67 /*v579*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[66:67] /*v[578:579]*/, v[132:133] /*v[644:645]*/, s[46:47], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa242
	v_exp_f32_e32 v231 /*v487*/, v92 /*v604*/
	v_exp_f32_e32 v243 /*v499*/, v93 /*v605*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[92:93] /*v[604:605]*/, v[136:137] /*v[648:649]*/, s[46:47], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v15 /*v527*/, v90 /*v602*/
	v_exp_f32_e32 v27 /*v539*/, v91 /*v603*/
	v_nop
	v_pk_fma_f32 v[90:91] /*v[602:603]*/, v[128:129] /*v[640:641]*/, s[46:47], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa242
	v_exp_f32_e32 v247 /*v503*/, v66 /*v578*/
	s_set_vgpr_msb 0x42a2
	v_exp_f32_e32 v3 /*v515*/, v67 /*v579*/
	v_nop
	v_pk_fma_f32 v[66:67] /*v[578:579]*/, v[138:139] /*v[650:651]*/, s[46:47], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v23 /*v535*/, v92 /*v604*/
	v_exp_f32_e32 v33 /*v545*/, v93 /*v605*/
	v_nop
	v_pk_fma_f32 v[92:93] /*v[604:605]*/, v[142:143] /*v[654:655]*/, s[46:47], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa241
	v_exp_f32_e32 v228 /*v484*/, v219 /*v475*/
	s_set_vgpr_msb 0x4142
	v_exp_f32_e32 v219 /*v475*/, v90 /*v602*/
	v_exp_f32_e32 v229 /*v485*/, v91 /*v603*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[90:91] /*v[602:603]*/, v[134:135] /*v[646:647]*/, s[46:47], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa241
	v_exp_f32_e32 v246 /*v502*/, v214 /*v470*/
	s_set_vgpr_msb 0x4181
	v_exp_f32_e32 v2 /*v514*/, v215 /*v471*/
	v_nop
	s_set_vgpr_msb 0x8161
	v_pk_fma_f32 v[214:215] /*v[470:471]*/, v[248:249] /*v[504:505]*/, s[46:47], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6181
	v_exp_f32_e32 v22 /*v534*/, v222 /*v478*/
	s_set_vgpr_msb 0x8162
	v_pk_fma_f32 v[232:233] /*v[488:489]*/, v[4:5] /*v[516:517]*/, s[46:47], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[240:241] /*v[496:497]*/, v[94:95] /*v[606:607]*/, s[46:47], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6241
	v_exp_f32_e32 v222 /*v478*/, v227 /*v483*/
	s_set_vgpr_msb 0x41a2
	v_exp_f32_e32 v37 /*v549*/, v66 /*v578*/
	v_exp_f32_e32 v45 /*v557*/, v67 /*v579*/
	v_nop
	v_pk_fma_f32 v[66:67] /*v[578:579]*/, v[144:145] /*v[656:657]*/, s[46:47], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa242
	v_exp_f32_e32 v227 /*v483*/, v92 /*v604*/
	v_exp_f32_e32 v237 /*v493*/, v93 /*v605*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[92:93] /*v[604:605]*/, v[148:149] /*v[660:661]*/, s[46:47], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v7 /*v519*/, v90 /*v602*/
	v_exp_f32_e32 v19 /*v531*/, v91 /*v603*/
	v_nop
	v_pk_fma_f32 v[90:91] /*v[602:603]*/, v[140:141] /*v[652:653]*/, s[46:47], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa281
	v_exp_f32_e32 v36 /*v548*/, v214 /*v470*/
	s_set_vgpr_msb 0x8141
	v_exp_f32_e32 v214 /*v470*/, v226 /*v482*/
	v_exp_f32_e32 v226 /*v482*/, v232 /*v488*/
	s_set_vgpr_msb 0x4162
	v_pk_fma_f32 v[244:245] /*v[500:501]*/, v[8:9] /*v[520:521]*/, s[46:47], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6241
	v_exp_f32_e32 v236 /*v492*/, v233 /*v489*/
	v_nop
	s_set_vgpr_msb 0x4162
	v_pk_fma_f32 v[232:233] /*v[488:489]*/, v[96:97] /*v[608:609]*/, s[46:47], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6241
	v_exp_f32_e32 v252 /*v508*/, v241 /*v497*/
	s_set_vgpr_msb 0x4142
	v_exp_f32_e32 v241 /*v497*/, v66 /*v578*/
	v_exp_f32_e32 v253 /*v509*/, v67 /*v579*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[66:67] /*v[578:579]*/, v[150:151] /*v[662:663]*/, s[46:47], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v17 /*v529*/, v92 /*v604*/
	v_exp_f32_e32 v29 /*v541*/, v93 /*v605*/
	v_nop
	v_pk_fma_f32 v[92:93] /*v[604:605]*/, v[154:155] /*v[666:667]*/, s[46:47], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa281
	v_exp_f32_e32 v32 /*v544*/, v223 /*v479*/
	v_exp_f32_e32 v44 /*v556*/, v215 /*v471*/
	s_set_vgpr_msb 0x8142
	v_exp_f32_e32 v215 /*v471*/, v90 /*v602*/
	v_exp_f32_e32 v223 /*v479*/, v91 /*v603*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[90:91] /*v[602:603]*/, v[146:147] /*v[658:659]*/, s[46:47], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa281
	v_exp_f32_e32 v0 /*v512*/, v244 /*v500*/
	v_exp_f32_e32 v12 /*v524*/, v245 /*v501*/
	v_exp_f32_e32 v16 /*v528*/, v232 /*v488*/
	s_set_vgpr_msb 0x8162
	v_pk_fma_f32 v[244:245] /*v[500:501]*/, v[98:99] /*v[610:611]*/, s[46:47], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6281
	v_exp_f32_e32 v28 /*v540*/, v233 /*v489*/
	v_nop
	s_set_vgpr_msb 0x8162
	v_pk_fma_f32 v[232:233] /*v[488:489]*/, v[24:25] /*v[536:537]*/, s[46:47], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x62a2
	v_pk_fma_f32 v[8:9] /*v[520:521]*/, v[38:39] /*v[550:551]*/, s[46:47], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v31 /*v543*/, v66 /*v578*/
	v_exp_f32_e32 v41 /*v553*/, v67 /*v579*/
	v_nop
	v_pk_fma_f32 v[66:67] /*v[578:579]*/, v[156:157] /*v[668:669]*/, s[46:47], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v53 /*v565*/, v92 /*v604*/
	v_exp_f32_e32 v59 /*v571*/, v93 /*v605*/
	v_nop
	v_pk_fma_f32 v[92:93] /*v[604:605]*/, v[160:161] /*v[672:673]*/, s[46:47], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa241
	v_exp_f32_e32 v194 /*v450*/, v194 /*v450*/
	v_exp_f32_e32 v204 /*v460*/, v204 /*v460*/
	s_set_vgpr_msb 0x4162
	v_pk_fma_f32 v[248:249] /*v[504:505]*/, v[20:21] /*v[532:533]*/, s[46:47], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x62a2
	v_exp_f32_e32 v1 /*v513*/, v90 /*v602*/
	v_exp_f32_e32 v13 /*v525*/, v91 /*v603*/
	v_nop
	v_pk_fma_f32 v[90:91] /*v[602:603]*/, v[152:153] /*v[664:665]*/, s[46:47], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa281
	v_exp_f32_e32 v50 /*v562*/, v245 /*v501*/
	v_exp_f32_e32 v58 /*v570*/, v233 /*v489*/
	s_set_vgpr_msb 0x81a2
	v_pk_fma_f32 v[24:25] /*v[536:537]*/, v[102:103] /*v[614:615]*/, s[46:47], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v20 /*v532*/, v9 /*v521*/
	v_pk_fma_f32 v[48:49] /*v[560:561]*/, v[106:107] /*v[618:619]*/, s[46:47], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa242
	v_exp_f32_e32 v233 /*v489*/, v66 /*v578*/
	v_exp_f32_e32 v245 /*v501*/, v67 /*v579*/
	v_nop
	s_set_vgpr_msb 0x42a2
	v_pk_fma_f32 v[66:67] /*v[578:579]*/, v[162:163] /*v[674:675]*/, s[46:47], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v9 /*v521*/, v92 /*v604*/
	v_exp_f32_e32 v21 /*v533*/, v93 /*v605*/
	v_nop
	v_pk_fma_f32 v[92:93] /*v[604:605]*/, v[166:167] /*v[678:679]*/, s[46:47], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa241
	v_exp_f32_e32 v240 /*v496*/, v240 /*v496*/
	s_set_vgpr_msb 0x4181
	v_exp_f32_e32 v30 /*v542*/, v248 /*v504*/
	v_exp_f32_e32 v40 /*v552*/, v249 /*v505*/
	v_nop
	s_set_vgpr_msb 0x8162
	v_pk_fma_f32 v[248:249] /*v[504:505]*/, v[34:35] /*v[546:547]*/, s[46:47], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x62a2
	v_pk_fma_f32 v[4:5] /*v[516:517]*/, v[100:101] /*v[612:613]*/, s[46:47], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v43 /*v555*/, v90 /*v602*/
	v_exp_f32_e32 v51 /*v563*/, v91 /*v603*/
	v_nop
	v_pk_fma_f32 v[90:91] /*v[602:603]*/, v[158:159] /*v[670:671]*/, s[46:47], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v34 /*v546*/, v25 /*v537*/
	v_pk_fma_f32 v[62:63] /*v[574:575]*/, v[54:55] /*v[566:567]*/, s[46:47], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v54 /*v566*/, v49 /*v561*/
	v_exp_f32_e32 v25 /*v537*/, v66 /*v578*/
	v_exp_f32_e32 v35 /*v547*/, v67 /*v579*/
	v_exp_f32_e32 v49 /*v561*/, v92 /*v604*/
	v_pk_fma_f32 v[66:67] /*v[578:579]*/, v[168:169] /*v[680:681]*/, s[46:47], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v55 /*v567*/, v93 /*v605*/
	v_nop
	s_set_vgpr_msb 0xa285
	v_pk_add_f32 v[92:93] /*v[604:605]*/, v[194:195] /*v[450:451]*/, v[196:197] /*v[452:453]*/
	v_pk_add_f32 v[94:95] /*v[606:607]*/, v[202:203] /*v[458:459]*/, v[204:205] /*v[460:461]*/
	v_exp_f32_e32 v42 /*v554*/, v244 /*v500*/
	v_exp_f32_e32 v52 /*v564*/, v232 /*v488*/
	s_set_vgpr_msb 0x8541
	v_exp_f32_e32 v232 /*v488*/, v248 /*v504*/
	v_exp_f32_e32 v244 /*v500*/, v249 /*v505*/
	s_set_vgpr_msb 0x4142
	v_exp_f32_e32 v248 /*v504*/, v4 /*v516*/
	s_set_vgpr_msb 0x42a2
	v_exp_f32_e32 v4 /*v516*/, v5 /*v517*/
	v_pk_fma_f32 v[38:39] /*v[550:551]*/, v[104:105] /*v[616:617]*/, s[46:47], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xa242
	v_exp_f32_e32 v249 /*v505*/, v90 /*v602*/
	s_set_vgpr_msb 0x42a2
	v_exp_f32_e32 v5 /*v517*/, v91 /*v603*/
	v_nop
	v_pk_fma_f32 v[90:91] /*v[602:603]*/, v[164:165] /*v[676:677]*/, s[46:47], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[64:65] /*v[576:577]*/, v[56:57] /*v[568:569]*/, s[46:47], v[60:61] /*v[572:573]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v57 /*v569*/, v66 /*v578*/
	v_exp_f32_e32 v61 /*v573*/, v67 /*v579*/
	v_nop
	s_set_vgpr_msb 0xa289
	v_pk_add_f32 v[66:67] /*v[578:579]*/, v[198:199] /*v[454:455]*/, v[92:93] /*v[604:605]*/
	v_pk_add_f32 v[92:93] /*v[604:605]*/, v[210:211] /*v[466:467]*/, v[94:95] /*v[606:607]*/
	s_set_vgpr_msb 0x8985
	v_pk_add_f32 v[94:95] /*v[606:607]*/, v[212:213] /*v[468:469]*/, v[220:221] /*v[476:477]*/
	v_pk_add_f32 v[96:97] /*v[608:609]*/, v[234:235] /*v[490:491]*/, v[238:239] /*v[494:495]*/
	s_set_vgpr_msb 0x8589
	v_pk_add_f32 v[98:99] /*v[610:611]*/, v[254:255] /*v[510:511]*/, v[10:11] /*v[522:523]*/
	s_set_vgpr_msb 0x898a
	v_pk_add_f32 v[108:109] /*v[620:621]*/, v[18:19] /*v[530:531]*/, v[22:23] /*v[534:535]*/
	v_pk_add_f32 v[110:111] /*v[622:623]*/, v[36:37] /*v[548:549]*/, v[44:45] /*v[556:557]*/
	s_set_vgpr_msb 0x8a85
	v_pk_add_f32 v[114:115] /*v[626:627]*/, v[240:241] /*v[496:497]*/, v[252:253] /*v[508:509]*/
	s_set_vgpr_msb 0x858a
	v_pk_add_f32 v[116:117] /*v[628:629]*/, v[12:13] /*v[524:525]*/, v[16:17] /*v[528:529]*/
	s_set_vgpr_msb 0x8a41
	v_exp_f32_e32 v218 /*v474*/, v218 /*v474*/
	s_set_vgpr_msb 0x4186
	v_exp_f32_e32 v8 /*v520*/, v8 /*v520*/
	v_exp_f32_e32 v24 /*v536*/, v24 /*v536*/
	v_exp_f32_e32 v46 /*v558*/, v39 /*v551*/
	v_exp_f32_e32 v48 /*v560*/, v48 /*v560*/
	v_exp_f32_e32 v47 /*v559*/, v91 /*v603*/
	v_pk_add_f32 v[100:101] /*v[612:613]*/, v[26:27] /*v[538:539]*/, v[200:201] /*v[456:457]*/
	s_set_vgpr_msb 0x8685
	v_pk_add_f32 v[102:103] /*v[614:615]*/, v[208:209] /*v[464:465]*/, v[216:217] /*v[472:473]*/
	s_set_vgpr_msb 0x8589
	v_pk_add_f32 v[94:95] /*v[606:607]*/, v[224:225] /*v[480:481]*/, v[94:95] /*v[606:607]*/
	v_pk_add_f32 v[96:97] /*v[608:609]*/, v[250:251] /*v[506:507]*/, v[96:97] /*v[608:609]*/
	s_set_vgpr_msb 0x898a
	v_pk_add_f32 v[98:99] /*v[610:611]*/, v[14:15] /*v[526:527]*/, v[98:99] /*v[610:611]*/
	s_set_vgpr_msb 0x8a85
	v_pk_add_f32 v[104:105] /*v[616:617]*/, v[228:229] /*v[484:485]*/, v[230:231] /*v[486:487]*/
	v_pk_add_f32 v[112:113] /*v[624:625]*/, v[222:223] /*v[478:479]*/, v[226:227] /*v[482:483]*/
	s_set_vgpr_msb 0x858a
	v_pk_add_f32 v[108:109] /*v[620:621]*/, v[32:33] /*v[544:545]*/, v[108:109] /*v[620:621]*/
	s_set_vgpr_msb 0x8a89
	v_pk_add_f32 v[110:111] /*v[622:623]*/, v[214:215] /*v[470:471]*/, v[110:111] /*v[622:623]*/
	s_set_vgpr_msb 0x898a
	v_pk_add_f32 v[118:119] /*v[630:631]*/, v[30:31] /*v[542:543]*/, v[40:41] /*v[552:553]*/
	v_pk_add_f32 v[120:121] /*v[632:633]*/, v[50:51] /*v[562:563]*/, v[52:53] /*v[564:565]*/
	s_set_vgpr_msb 0x8a85
	v_pk_add_f32 v[122:123] /*v[634:635]*/, v[232:233] /*v[488:489]*/, v[244:245] /*v[500:501]*/
	s_set_vgpr_msb 0x85aa
	v_pk_add_f32 v[114:115] /*v[626:627]*/, v[0:1] /*v[512:513]*/, v[114:115] /*v[626:627]*/
	v_pk_add_f32 v[116:117] /*v[628:629]*/, v[28:29] /*v[540:541]*/, v[116:117] /*v[628:629]*/
	v_pk_add_f32 v[66:67] /*v[578:579]*/, v[66:67] /*v[578:579]*/, v[92:93] /*v[604:605]*/
	v_exp_f32_e32 v38 /*v550*/, v38 /*v550*/
	v_exp_f32_e32 v56 /*v568*/, v62 /*v574*/
	v_exp_f32_e32 v60 /*v572*/, v63 /*v575*/
	v_exp_f32_e32 v39 /*v551*/, v90 /*v602*/
	v_nop
	v_pk_fma_f32 v[90:91] /*v[602:603]*/, v[170:171] /*v[682:683]*/, s[46:47], v[68:69] /*v[580:581]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xaa89
	v_pk_add_f32 v[100:101] /*v[612:613]*/, v[206:207] /*v[462:463]*/, v[100:101] /*v[612:613]*/
	v_pk_add_f32 v[102:103] /*v[614:615]*/, v[218:219] /*v[474:475]*/, v[102:103] /*v[614:615]*/
	v_pk_add_f32 v[106:107] /*v[618:619]*/, v[246:247] /*v[502:503]*/, v[2:3] /*v[514:515]*/
	v_pk_add_f32 v[104:105] /*v[616:617]*/, v[242:243] /*v[498:499]*/, v[104:105] /*v[616:617]*/
	v_pk_add_f32 v[112:113] /*v[624:625]*/, v[236:237] /*v[492:493]*/, v[112:113] /*v[624:625]*/
	s_set_vgpr_msb 0x898a
	v_pk_add_f32 v[118:119] /*v[630:631]*/, v[42:43] /*v[554:555]*/, v[118:119] /*v[630:631]*/
	v_pk_add_f32 v[120:121] /*v[632:633]*/, v[58:59] /*v[570:571]*/, v[120:121] /*v[632:633]*/
	s_set_vgpr_msb 0x8a89
	v_pk_add_f32 v[122:123] /*v[634:635]*/, v[248:249] /*v[504:505]*/, v[122:123] /*v[634:635]*/
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
	v_sub_f32_e32 v68 /*v580*/, v86 /*v598*/, v69 /*v581*/
	v_pk_add_f32 v[90:91] /*v[602:603]*/, v[64:65] /*v[576:577]*/, v[90:91] /*v[602:603]*/
	v_pk_add_f32 v[66:67] /*v[578:579]*/, v[66:67] /*v[578:579]*/, v[92:93] /*v[604:605]*/
	v_mul_f32_e32 v68 /*v580*/, 0x3fb8aa3b, v68 /*v580*/
	v_pk_add_f32 v[66:67] /*v[578:579]*/, v[90:91] /*v[602:603]*/, v[66:67] /*v[578:579]*/
	v_exp_f32_e32 v68 /*v580*/, v68 /*v580*/
	v_dual_mov_b32 v86 /*v598*/, v66 /*v578*/ :: v_dual_mov_b32 v90 /*v602*/, v67 /*v579*/
	v_permlanex16_b32 v86 /*v598*/, v86 /*v598*/, s35, 0xfedcba98
	v_permlanex16_b32 v90 /*v602*/, v90 /*v602*/, s35, 0xfedcba98
	s_set_vgpr_msb 0x8a00
	s_cbranch_vccz .LBB0_40
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
.LBB0_40:
	s_set_vgpr_msb 0x8a
	v_sub_f32_e32 v70 /*v582*/, v85 /*v597*/, v71 /*v583*/
	s_and_b32 s2, s5, exec_lo
	s_cselect_b32 s2, 1, 0
	s_cmp_lg_u32 s2, 1
	v_mul_f32_e32 v70 /*v582*/, 0x3fb8aa3b, v70 /*v582*/
	v_exp_f32_e32 v70 /*v582*/, v70 /*v582*/
	s_set_vgpr_msb 0x8a00
	s_cbranch_scc1 .LBB0_35
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
	s_branch .LBB0_35
.LBB0_42:
	s_set_vgpr_msb 0x82
	v_dual_mov_b32 v71 /*v583*/, v85 /*v597*/ :: v_dual_mov_b32 v69 /*v581*/, v86 /*v598*/
	s_set_vgpr_msb 0x8200
.LBB0_43:
	s_set_vgpr_msb 5
	v_div_scale_f32 v129, null, v192 /*v448*/, v192 /*v448*/, 1.0
	v_div_scale_f32 v132, vcc_lo, 1.0, v192 /*v448*/, 1.0
	v_div_scale_f32 v150, null, v193 /*v449*/, v193 /*v449*/, 1.0
	s_lshl_b32 s2, s10, 17
	s_set_vgpr_msb 0x508
	v_mul_u32_u24_e32 v128, 0x110, v77 /*v589*/
	v_rcp_f32_e32 v130, v129
	s_and_b32 s4, s2, 0x20000
	v_rcp_f32_e32 v151, v150
	s_set_vgpr_msb 0x804
	v_cmp_lt_f32_e64 s2, 0, v192 /*v448*/
	s_set_vgpr_msb 0x400
	s_wait_dscnt 0x0
	v_fma_f32 v131, -v129, v130, 1.0
	v_fmac_f32_e32 v130, v131, v130
	v_mul_f32_e32 v131, v132, v130
	v_fma_f32 v133, -v129, v131, v132
	v_fmac_f32_e32 v131, v133, v130
	v_fma_f32 v129, -v129, v131, v132
	v_div_fmas_f32 v129, v129, v130, v131
	v_fma_f32 v130, -v150, v151, 1.0
	s_set_vgpr_msb 4
	v_div_scale_f32 v152, vcc_lo, 1.0, v193 /*v449*/, 1.0
	v_div_fixup_f32 v129, v129, v192 /*v448*/, 1.0
	s_set_vgpr_msb 0x400
	v_dual_fmac_f32 v151, v130, v151 :: v_dual_cndmask_b32 v130, 0, v129, s2
	v_mul_f32_e32 v153, v152, v151
	s_add_co_i32 s2, s4, s89
	s_mov_b32 s4, 0
	s_set_vgpr_msb 8
	v_add_nc_u32_e32 v129, s2, v80 /*v592*/
	s_set_vgpr_msb 0x800
	v_pk_mul_f32 v[140:141], v[130:131], v[76:77] op_sel_hi:[0,1]
	v_fma_f32 v76, -v150, v153, v152
	v_pk_mul_f32 v[90:91], v[130:131], v[90:91] op_sel_hi:[0,1]
	v_pk_mul_f32 v[88:89], v[130:131], v[88:89] op_sel_hi:[0,1]
	v_pk_mul_f32 v[132:133], v[130:131], v[80:81] op_sel_hi:[0,1]
	v_pk_mul_f32 v[92:93], v[130:131], v[92:93] op_sel_hi:[0,1]
	v_fmac_f32_e32 v153, v76, v151
	v_cvt_pk_bf16_f32 v81, v90, v91
	v_cvt_pk_bf16_f32 v80, v88, v89
	v_pk_mul_f32 v[96:97], v[96:97], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[134:135], v[130:131], v[82:83] op_sel_hi:[0,1]
	v_fma_f32 v90, -v150, v153, v152
	v_cvt_pk_bf16_f32 v82, v92, v93
	v_pk_mul_f32 v[112:113], v[112:113], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[114:115], v[114:115], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[104:105], v[104:105], v[130:131] op_sel_hi:[1,0]
	v_div_fmas_f32 v88, v90, v151, v153
	s_set_vgpr_msb 4
	v_cmp_lt_f32_e32 vcc_lo, 0, v193 /*v449*/
	s_set_vgpr_msb 0x400
	v_pk_mul_f32 v[106:107], v[106:107], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[108:109], v[108:109], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[110:111], v[110:111], v[130:131] op_sel_hi:[1,0]
	s_set_vgpr_msb 4
	v_div_fixup_f32 v92, v88, v193 /*v449*/, 1.0
	s_set_vgpr_msb 0x420
	v_pk_mul_f32 v[98:99], v[98:99], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[100:101], v[100:101], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[102:103], v[102:103], v[130:131] op_sel_hi:[1,0]
	v_cvt_pk_bf16_f32 v76, v96, v97
	v_cndmask_b32_e32 v96, 0, v92, vcc_lo
	v_pk_mul_f32 v[120:121], v[120:121], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[122:123], v[122:123], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[124:125], v[124:125], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[126:127], v[126:127], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[116:117], v[116:117], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[118:119], v[118:119], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[94:95], v[130:131], v[94:95] op_sel_hi:[0,1]
	v_pk_mul_f32 v[84:85], v[130:131], v[84:85] op_sel_hi:[0,1]
	v_pk_mul_f32 v[86:87], v[130:131], v[86:87] op_sel_hi:[0,1]
	v_pk_mul_f32 v[136:137], v[130:131], v[72:73] op_sel_hi:[0,1]
	v_pk_mul_f32 v[138:139], v[130:131], v[74:75] op_sel_hi:[0,1]
	v_pk_mul_f32 v[142:143], v[130:131], v[78:79] op_sel_hi:[0,1]
	v_pk_mul_f32 v[144:145], v[130:131], v[64:65] op_sel_hi:[0,1]
	v_pk_mul_f32 v[146:147], v[130:131], v[66:67] op_sel_hi:[0,1]
	v_pk_mul_f32 v[148:149], v[130:131], v[68:69] op_sel_hi:[0,1]
	v_pk_mul_f32 v[130:131], v[130:131], v[70:71] op_sel_hi:[0,1]
	v_cvt_pk_bf16_f32 v69, v114, v115
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
	v_add3_u32 v116, v128, s2, v79 /*v591*/
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
	s_lshl_b32 s5, s57, 2
	s_wait_alu depctr_va_vdst(0)
	ds_store_b128 v129, v[64:67]
	ds_store_b128 v129, v[68:71] offset:32
	ds_store_b128 v129, v[72:75] offset:64
	ds_store_b128 v129, v[76:79] offset:96
	ds_store_b128 v129, v[80:83] offset:128
	ds_store_b128 v129, v[84:87] offset:160
	ds_store_b128 v129, v[88:91] offset:192
	ds_store_b128 v129, v[92:95] offset:224
	ds_store_b128 v116, v[0:3] offset:4352
	ds_store_b128 v116, v[4:7] offset:4384
	ds_store_b128 v116, v[8:11] offset:4416
	ds_store_b128 v116, v[12:15] offset:4448
	s_sub_co_i32 s5, s5, s88
	ds_store_b128 v116, v[16:19] offset:4480
	ds_store_b128 v116, v[20:23] offset:4512
	ds_store_b128 v116, v[24:27] offset:4544
	ds_store_b128 v116, v[28:31] offset:4576
	s_cmp_lt_i32 s5, 1
	s_set_vgpr_msb 0x2000
	s_wait_dscnt 0x0
	s_cbranch_scc1 .LBB0_45
	s_wait_alu depctr_vm_vsrc(1)
	s_set_vgpr_msb 10
	v_dual_lshrrev_b32 v26, 4, v78 /*v590*/ :: v_dual_bitop2_b32 v0, 28, v76 /*v588*/ bitop3:0x54
	s_add_co_i32 s5, s5, -1
	v_lshl_add_u32 v32, v77 /*v589*/, 4, s2
	s_min_u32 s5, s5, 31
	s_set_vgpr_msb 0xa00
	v_or_b32_e32 v3, 26, v26
	v_min_u32_e32 v2, s5, v0
	s_set_vgpr_msb 8
	v_dual_lshlrev_b32 v0, 3, v77 /*v589*/ :: v_dual_bitop2_b32 v6, 24, v76 /*v588*/ bitop3:0x54
	v_or_b32_e32 v21, 16, v76 /*v588*/
	s_wait_alu depctr_vm_vsrc(0)
	v_or_b32_e32 v30, 8, v76 /*v588*/
	s_set_vgpr_msb 0x800
	v_or_b32_e32 v5, s88, v2
	v_or_b32_e32 v1, 30, v26
	v_min_u32_e32 v14, s5, v6
	v_mad_u32_u24 v33, 0x110, v2, v32
	v_min_u32_e32 v21, s5, v21
	v_ashrrev_i32_e32 v7, 31, v5
	v_min_u32_e32 v4, s5, v1
	v_mov_b32_e32 v1, 0
	v_min_u32_e32 v8, s5, v3
	v_mad_u32_u24 v36, 0x110, v14, v32
	v_dual_lshrrev_b32 v7, 30, v7 :: v_dual_bitop2_b32 v3, s88, v4 bitop3:0x54
	v_mad_u32_u24 v34, 0x110, v4, v32
	v_mad_u32_u24 v35, 0x110, v8, v32
	v_mad_u32_u24 v40, 0x110, v21, v32
	v_min_u32_e32 v41, s5, v30
	v_dual_ashrrev_i32 v9, 31, v3 :: v_dual_bitop2_b32 v6, s88, v8 bitop3:0x54
	v_dual_add_nc_u32 v7, v5, v7 :: v_dual_bitop2_b32 v10, 22, v26 bitop3:0x54
	v_dual_lshrrev_b32 v9, 30, v9 :: v_dual_bitop2_b32 v11, s88, v14 bitop3:0x54
	v_ashrrev_i32_e32 v2, 31, v6
	v_min_u32_e32 v15, s5, v10
	v_and_b32_e32 v4, -4, v7
	v_dual_ashrrev_i32 v7, 2, v7 :: v_dual_add_nc_u32 v9, v3, v9
	v_lshrrev_b32_e32 v2, 30, v2
	v_mad_u32_u24 v37, 0x110, v15, v32
	v_cmp_ne_u32_e32 vcc_lo, v5, v4
	v_add_nc_u32_e32 v7, s33, v7
	s_load_b64 s[6:7], s[0:1], 0x0 nv
	v_dual_add_nc_u32 v10, v6, v2 :: v_dual_bitop2_b32 v2, -4, v9 bitop3:0x40
	v_dual_sub_nc_u32 v4, v5, v4 :: v_dual_ashrrev_i32 v9, 2, v9
	s_and_b32 s9, s59, vcc_lo
	v_and_b32_e32 v12, -4, v10
	v_sub_nc_u32_e32 v5, v3, v2
	v_cmp_ne_u32_e64 s2, v3, v2
	v_dual_add_nc_u32 v2, s34, v4 :: v_dual_add_nc_u32 v9, s33, v9
	v_cndmask_b32_e64 v13, 0, 1, s9
	v_add_nc_u32_e32 v4, s34, v5
	s_and_b32 s2, s59, s2
	v_ashrrev_i32_e32 v17, 31, v11
	v_cndmask_b32_e64 v16, 0, 1, s2
	v_ashrrev_i32_e32 v10, 2, v10
	v_mad_nc_i64_i32 v[4:5], v4, s28, v[0:1]
	v_cmp_ne_u32_e32 vcc_lo, v6, v12
	v_dual_sub_nc_u32 v6, v6, v12 :: v_dual_bitop2_b32 v12, s88, v15 bitop3:0x54
	v_dual_sub_nc_u32 v9, v9, v16 :: v_dual_sub_nc_u32 v7, v7, v13
	v_mad_nc_i64_i32 v[2:3], v2, s28, v[0:1]
	v_add_nc_u32_e32 v10, s33, v10
	s_and_b32 s2, s59, vcc_lo
	v_mad_nc_i64_i32 v[4:5], v9, s8, v[4:5]
	v_dual_lshrrev_b32 v9, 30, v17 :: v_dual_add_nc_u32 v6, s34, v6
	v_cndmask_b32_e64 v13, 0, 1, s2
	s_set_vgpr_msb 8
	v_or_b32_e32 v16, 20, v76 /*v588*/
	v_mad_nc_i64_i32 v[2:3], v7, s8, v[2:3]
	s_set_vgpr_msb 0x800
	v_add_nc_u32_e32 v9, v11, v9
	v_mad_nc_i64_i32 v[6:7], v6, s28, v[0:1]
	v_dual_ashrrev_i32 v17, 31, v12 :: v_dual_sub_nc_u32 v10, v10, v13
	v_min_u32_e32 v16, s5, v16
	v_and_b32_e32 v8, -4, v9
	v_ashrrev_i32_e32 v9, 2, v9
	v_lshrrev_b32_e32 v13, 30, v17
	s_wait_kmcnt 0x0
	v_lshl_add_u64 v[2:3], v[2:3], 1, s[6:7]
	v_mad_nc_i64_i32 v[6:7], v10, s8, v[6:7]
	v_cmp_ne_u32_e32 vcc_lo, v11, v8
	v_sub_nc_u32_e32 v10, v11, v8
	v_dual_add_nc_u32 v13, v12, v13 :: v_dual_bitop2_b32 v17, s88, v16 bitop3:0x54
	v_add_nc_u32_e32 v11, s33, v9
	s_and_b32 s2, s59, vcc_lo
	v_add_nc_u32_e32 v8, s34, v10
	v_cndmask_b32_e64 v18, 0, 1, s2
	v_ashrrev_i32_e32 v9, 31, v17
	v_mad_u32_u24 v38, 0x110, v16, v32
	v_lshl_add_u64 v[4:5], v[4:5], 1, s[6:7]
	v_lshl_add_u64 v[6:7], v[6:7], 1, s[6:7]
	v_dual_sub_nc_u32 v11, v11, v18 :: v_dual_lshrrev_b32 v19, 30, v9
	v_mad_nc_i64_i32 v[8:9], v8, s28, v[0:1]
	v_or_b32_e32 v30, s88, v41
	v_mad_u32_u24 v41, 0x110, v41, v32
	v_ashrrev_i32_e32 v31, 31, v30
	v_mad_nc_i64_i32 v[8:9], v11, s8, v[8:9]
	v_or_b32_e32 v11, 18, v26
	v_dual_add_nc_u32 v18, v17, v19 :: v_dual_bitop2_b32 v10, -4, v13 bitop3:0x40
	v_ashrrev_i32_e32 v13, 2, v13
	v_cmp_ne_u32_e32 vcc_lo, v12, v10
	v_dual_sub_nc_u32 v10, v12, v10 :: v_dual_add_nc_u32 v13, s33, v13
	v_and_b32_e32 v19, -4, v18
	v_lshl_add_u64 v[8:9], v[8:9], 1, s[6:7]
	s_and_b32 s2, s59, vcc_lo
	v_cndmask_b32_e64 v12, 0, 1, s2
	v_add_nc_u32_e32 v10, s34, v10
	v_cmp_ne_u32_e32 vcc_lo, v17, v19
	v_dual_sub_nc_u32 v20, v13, v12 :: v_dual_ashrrev_i32 v12, 2, v18
	v_sub_nc_u32_e32 v13, v17, v19
	v_min_u32_e32 v18, s5, v11
	v_mad_nc_i64_i32 v[10:11], v10, s28, v[0:1]
	s_and_b32 s2, s59, vcc_lo
	v_dual_add_nc_u32 v17, s33, v12 :: v_dual_add_nc_u32 v12, s34, v13
	v_or_b32_e32 v19, s88, v18
	v_cndmask_b32_e64 v22, 0, 1, s2
	v_mad_u32_u24 v39, 0x110, v18, v32
	v_mad_nc_i64_i32 v[10:11], v20, s8, v[10:11]
	v_dual_ashrrev_i32 v23, 31, v19 :: v_dual_sub_nc_u32 v17, v17, v22
	v_mad_nc_i64_i32 v[12:13], v12, s28, v[0:1]
	v_dual_lshrrev_b32 v20, 30, v23 :: v_dual_bitop2_b32 v22, s88, v21 bitop3:0x54
	v_lshl_add_u64 v[10:11], v[10:11], 1, s[6:7]
	v_add_nc_u32_e32 v14, v19, v20
	v_mad_nc_i64_i32 v[12:13], v17, s8, v[12:13]
	v_and_b32_e32 v15, -4, v14
	v_ashrrev_i32_e32 v17, 31, v22
	v_lshl_add_u64 v[12:13], v[12:13], 1, s[6:7]
	v_dual_sub_nc_u32 v20, v19, v15 :: v_dual_lshrrev_b32 v16, 30, v17
	v_or_b32_e32 v17, 14, v26
	v_cmp_ne_u32_e32 vcc_lo, v19, v15
	v_dual_add_nc_u32 v16, v22, v16 :: v_dual_ashrrev_i32 v14, 2, v14
	v_min_u32_e32 v23, s5, v17
	s_and_b32 s2, s59, vcc_lo
	v_dual_add_nc_u32 v19, s33, v14 :: v_dual_bitop2_b32 v17, -4, v16 bitop3:0x40
	v_dual_add_nc_u32 v14, s34, v20 :: v_dual_bitop2_b32 v20, s88, v23 bitop3:0x54
	v_dual_ashrrev_i32 v16, 2, v16 :: v_dual_sub_nc_u32 v25, v22, v17
	v_cmp_ne_u32_e32 vcc_lo, v22, v17
	v_cndmask_b32_e64 v24, 0, 1, s2
	v_ashrrev_i32_e32 v27, 31, v20
	v_dual_add_nc_u32 v22, s33, v16 :: v_dual_add_nc_u32 v16, s34, v25
	v_mad_nc_i64_i32 v[14:15], v14, s28, v[0:1]
	s_and_b32 s2, s59, vcc_lo
	v_dual_lshrrev_b32 v25, 30, v27 :: v_dual_sub_nc_u32 v19, v19, v24
	s_set_vgpr_msb 8
	v_or_b32_e32 v27, 12, v76 /*v588*/
	v_cndmask_b32_e64 v28, 0, 1, s2
	v_mad_nc_i64_i32 v[16:17], v16, s28, v[0:1]
	s_set_vgpr_msb 0x800
	v_add_nc_u32_e32 v25, v20, v25
	v_mad_nc_i64_i32 v[14:15], v19, s8, v[14:15]
	v_min_u32_e32 v27, s5, v27
	v_dual_sub_nc_u32 v22, v22, v28 :: v_dual_bitop2_b32 v28, 10, v26 bitop3:0x54
	v_dual_ashrrev_i32 v19, 2, v25 :: v_dual_bitop2_b32 v18, -4, v25 bitop3:0x40
	v_or_b32_e32 v24, s88, v27
	v_mad_nc_i64_i32 v[16:17], v22, s8, v[16:17]
	v_mad_u32_u24 v42, 0x110, v23, v32
	v_sub_nc_u32_e32 v22, v20, v18
	v_cmp_ne_u32_e32 vcc_lo, v20, v18
	v_dual_ashrrev_i32 v25, 31, v24 :: v_dual_add_nc_u32 v20, s33, v19
	v_mad_u32_u24 v44, 0x110, v27, v32
	v_add_nc_u32_e32 v18, s34, v22
	s_and_b32 s2, s59, vcc_lo
	v_lshrrev_b32_e32 v22, 30, v25
	v_min_u32_e32 v25, s5, v28
	v_cndmask_b32_e64 v28, 0, 1, s2
	v_mad_nc_i64_i32 v[18:19], v18, s28, v[0:1]
	v_lshl_add_u64 v[14:15], v[14:15], 1, s[6:7]
	v_lshl_add_u64 v[16:17], v[16:17], 1, s[6:7]
	v_dual_add_nc_u32 v22, v24, v22 :: v_dual_bitop2_b32 v29, s88, v25 bitop3:0x54
	v_sub_nc_u32_e32 v20, v20, v28
	v_mad_u32_u24 v45, 0x110, v25, v32
	v_dual_ashrrev_i32 v28, 31, v29 :: v_dual_bitop2_b32 v21, -4, v22 bitop3:0x40
	v_mad_nc_i64_i32 v[18:19], v20, s8, v[18:19]
	v_dual_ashrrev_i32 v20, 2, v22 :: v_dual_sub_nc_u32 v22, v24, v21
	v_lshrrev_b32_e32 v28, 30, v28
	v_cmp_ne_u32_e32 vcc_lo, v24, v21
	v_add_nc_u32_e32 v24, s33, v20
	v_lshl_add_u64 v[18:19], v[18:19], 1, s[6:7]
	v_dual_add_nc_u32 v20, s34, v22 :: v_dual_add_nc_u32 v22, v29, v28
	s_and_b32 s2, s59, vcc_lo
	v_cndmask_b32_e64 v28, 0, 1, s2
	v_mad_nc_i64_i32 v[20:21], v20, s28, v[0:1]
	v_and_b32_e32 v23, -4, v22
	v_dual_ashrrev_i32 v22, 2, v22 :: v_dual_sub_nc_u32 v24, v24, v28
	v_sub_nc_u32_e32 v28, v29, v23
	v_cmp_ne_u32_e32 vcc_lo, v29, v23
	v_or_b32_e32 v29, 6, v26
	v_mad_nc_i64_i32 v[20:21], v24, s8, v[20:21]
	v_dual_add_nc_u32 v24, s33, v22 :: v_dual_add_nc_u32 v22, s34, v28
	s_and_b32 s2, s59, vcc_lo
	v_lshrrev_b32_e32 v28, 30, v31
	v_cndmask_b32_e64 v31, 0, 1, s2
	v_min_u32_e32 v43, s5, v29
	v_mad_nc_i64_i32 v[22:23], v22, s28, v[0:1]
	v_lshl_add_u64 v[20:21], v[20:21], 1, s[6:7]
	v_dual_add_nc_u32 v28, v30, v28 :: v_dual_sub_nc_u32 v24, v24, v31
	s_set_vgpr_msb 8
	v_or_b32_e32 v31, 4, v76 /*v588*/
	s_set_vgpr_msb 0x800
	v_and_b32_e32 v27, -4, v28
	v_mad_nc_i64_i32 v[22:23], v24, s8, v[22:23]
	v_dual_ashrrev_i32 v24, 2, v28 :: v_dual_bitop2_b32 v29, s88, v43 bitop3:0x54
	v_min_u32_e32 v46, s5, v31
	v_sub_nc_u32_e32 v25, v30, v27
	v_cmp_ne_u32_e32 vcc_lo, v30, v27
	v_dual_add_nc_u32 v27, s33, v24 :: v_dual_ashrrev_i32 v28, 31, v29
	v_dual_add_nc_u32 v24, s34, v25 :: v_dual_bitop2_b32 v31, s88, v46 bitop3:0x54
	s_and_b32 s2, s59, vcc_lo
	v_mad_u32_u24 v43, 0x110, v43, v32
	v_lshrrev_b32_e32 v28, 30, v28
	v_cndmask_b32_e64 v30, 0, 1, s2
	v_mad_nc_i64_i32 v[24:25], v24, s28, v[0:1]
	v_mad_u32_u24 v46, 0x110, v46, v32
	v_lshl_add_u64 v[22:23], v[22:23], 1, s[6:7]
	v_dual_add_nc_u32 v28, v29, v28 :: v_dual_sub_nc_u32 v27, v27, v30
	v_or_b32_e32 v26, 2, v26
	v_and_b32_e32 v30, -4, v28
	v_dual_ashrrev_i32 v28, 2, v28 :: v_dual_ashrrev_i32 v47, 31, v31
	v_mad_nc_i64_i32 v[24:25], v27, s8, v[24:25]
	v_min_u32_e32 v48, s5, v26
	v_cmp_ne_u32_e32 vcc_lo, v29, v30
	v_dual_add_nc_u32 v26, s33, v28 :: v_dual_lshrrev_b32 v27, 30, v47
	s_set_vgpr_msb 8
	v_min_i32_e32 v47, s5, v76 /*v588*/
	s_set_vgpr_msb 0x800
	v_or_b32_e32 v28, s88, v48
	s_and_b32 s2, s59, vcc_lo
	v_sub_nc_u32_e32 v29, v29, v30
	v_cndmask_b32_e64 v49, 0, 1, s2
	v_dual_ashrrev_i32 v50, 31, v28 :: v_dual_bitop2_b32 v30, s88, v47 bitop3:0x54
	v_add_nc_u32_e32 v27, v31, v27
	v_mad_u32_u24 v47, 0x110, v47, v32
	v_sub_nc_u32_e32 v49, v26, v49
	v_dual_add_nc_u32 v26, s34, v29 :: v_dual_lshrrev_b32 v50, 30, v50
	v_dual_ashrrev_i32 v29, 31, v30 :: v_dual_bitop2_b32 v51, -4, v27 bitop3:0x40
	v_ashrrev_i32_e32 v52, 2, v27
	v_mad_nc_i64_i32 v[26:27], v26, s28, v[0:1]
	v_dual_add_nc_u32 v50, v28, v50 :: v_dual_lshrrev_b32 v29, 30, v29
	v_cmp_ne_u32_e32 vcc_lo, v31, v51
	v_dual_add_nc_u32 v52, s33, v52 :: v_dual_sub_nc_u32 v31, v31, v51
	v_dual_add_nc_u32 v29, v30, v29 :: v_dual_bitop2_b32 v51, -4, v50 bitop3:0x40
	s_and_b32 s2, s59, vcc_lo
	v_dual_ashrrev_i32 v50, 2, v50 :: v_dual_add_nc_u32 v31, s34, v31
	v_cndmask_b32_e64 v53, 0, 1, s2
	v_and_b32_e32 v54, -4, v29
	v_cmp_ne_u32_e32 vcc_lo, v28, v51
	v_dual_sub_nc_u32 v28, v28, v51 :: v_dual_ashrrev_i32 v29, 2, v29
	v_add_nc_u32_e32 v50, s33, v50
	v_sub_nc_u32_e32 v51, v30, v54
	v_cmp_ne_u32_e64 s2, v30, v54
	v_dual_add_nc_u32 v54, s34, v28 :: v_dual_add_nc_u32 v55, s33, v29
	v_mad_nc_i64_i32 v[30:31], v31, s28, v[0:1]
	v_add_nc_u32_e32 v28, s34, v51
	s_and_b32 s2, s59, s2
	v_mad_nc_i64_i32 v[26:27], v49, s8, v[26:27]
	v_cndmask_b32_e64 v51, 0, 1, s2
	s_and_b32 s2, s59, vcc_lo
	v_mad_nc_i64_i32 v[28:29], v28, s28, v[0:1]
	v_cndmask_b32_e64 v56, 0, 1, s2
	v_mad_nc_i64_i32 v[0:1], v54, s28, v[0:1]
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
.LBB0_45:
	s_set_vgpr_msb 6
	v_cmp_gt_f32_e32 vcc_lo, 0x800000, v192 /*v448*/
	v_cmp_gt_f32_e64 s2, 0x800000, v193 /*v449*/
	s_load_b64 s[6:7], s[0:1], 0x40 nv
	s_mul_i32 s8, s3, s31
	s_wait_alu depctr_vm_vsrc(0)
	v_mul_lo_u32 v5, v75 /*v587*/, s29
	v_cndmask_b32_e64 v2, 0, 32, vcc_lo
	v_cndmask_b32_e64 v4, 0, 32, s2
	v_cndmask_b32_e64 v1, 0, 0x42000000, vcc_lo
	v_cndmask_b32_e64 v3, 0, 0x42000000, s2
	v_mad_u32 v6, v74 /*v586*/, s29, s8
	s_set_vgpr_msb 0x601
	v_ldexp_f32 v2, v192 /*v448*/, v2
	v_ldexp_f32 v4, v193 /*v449*/, v4
	s_set_vgpr_msb 0x108
	v_cmp_eq_u32_e32 vcc_lo, 0, v76 /*v588*/
	v_cmp_gt_i32_e64 s0, s57, v74 /*v586*/
	v_cmp_gt_i32_e64 s1, s57, v75 /*v587*/
	v_log_f32_e32 v2, v2
	v_log_f32_e32 v4, v4
	s_add_co_i32 s2, s8, s31
	s_and_b32 s0, vcc_lo, s0
	s_ashr_i32 s3, s2, 31
	s_and_b32 vcc_lo, vcc_lo, s1
	s_lshl_b32 s5, s2, 27
	s_lshr_b64 s[2:3], s[2:3], 5
	v_nop
	s_set_vgpr_msb 0x800
	v_dual_sub_f32 v1, v2, v1 :: v_dual_sub_f32 v2, v4, v3
	s_and_b64 s[2:3], s[2:3], 0x1ffffffffffffff
	v_dual_mul_f32 v1, 0x3f317218, v1 :: v_dual_mul_f32 v2, 0x3f317218, v2
	s_set_vgpr_msb 8
	v_lshlrev_b32_e32 v0, 2, v73 /*v585*/
	v_dual_add_f32 v1, v1, v69 /*v581*/ :: v_dual_add_f32 v2, v2, v71 /*v583*/
	v_subrev_nc_u32_e32 v0, v0, v72 /*v584*/
	s_set_vgpr_msb 0x800
	v_add_nc_u32_e32 v0, s34, v0
	v_mul_lo_u32 v0, v0, s30
	v_add_nc_u32_e32 v3, s8, v0
	v_add_lshl_u32 v0, v6, v0, 2
	v_add_lshl_u32 v3, v3, v5, 2
	v_cndmask_b32_e64 v0, 0x7fffffff, v0, s0
	s_wait_kmcnt 0x0
	s_or_b64 s[0:1], s[6:7], s[4:5]
	v_cndmask_b32_e32 v3, 0x7fffffff, v3, vcc_lo
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	buffer_store_b32 v1, v0, s[0:3], null offen
	buffer_store_b32 v2, v3, s[0:3], null offen
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
		.amdhsa_next_free_vgpr 688
		.amdhsa_next_free_sgpr 97
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

	.set .Lkn_fmha_fwd_prefill_a16w16_m32x8_bshd_0.num_vgpr, 688
	.set .Lkn_fmha_fwd_prefill_a16w16_m32x8_bshd_0.num_agpr, 0
	.set .Lkn_fmha_fwd_prefill_a16w16_m32x8_bshd_0.numbered_sgpr, 97
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
    .sgpr_count:     99
    .sgpr_spill_count: 0
    .symbol:         kn_fmha_fwd_prefill_a16w16_m32x8_bshd_0.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     688
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa-unknown-gfx1250
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
