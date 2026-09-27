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
	v_and_b32_e32 v154, 15, v0
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
	s_clause 0x1
	s_load_b128 s[28:31], s[0:1], 0x7c nv
	s_load_b96 s[56:58], s[0:1], 0x90 nv
	s_mov_b32 s12, 1
	s_mov_b32 s43, 0
	s_mov_b32 s21, 0xffff0000
	s_mul_f32 s7, s7, 0x4f7ffffe
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
	s_mov_b32 s60, s6
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
	s_mov_b32 s36, s10
	v_or_b32_e32 v7, s88, v154
	s_cmp_lt_i32 s88, 0
	s_mov_b32 s38, s11
	s_cselect_b32 s59, -1, 0
	s_ashr_i32 s2, s88, 31
	v_or_b32_e32 v3, 16, v7
	s_lshr_b32 s2, s2, 30
	v_and_b32_e32 v1, 16, v0
	s_ashr_i32 s16, s88, 2
	s_ashr_i32 s23, s5, 31
	v_add_nc_u32_e32 v4, s2, v3
	s_add_co_i32 s24, s33, s16
	s_ashr_i32 s35, s34, 31
	s_ashr_i32 s27, s9, 31
	s_ashr_i32 s25, s24, 31
	v_and_b32_e32 v5, -4, v4
	s_sub_co_i32 s17, s57, s16
	s_mul_u64 s[44:45], s[34:35], s[26:27]
	s_mul_u64 s[24:25], s[24:25], s[22:23]
	s_add_co_i32 s13, s89, 0x23000
	v_cmp_ne_u32_e32 vcc_lo, v3, v5
	v_dual_ashrrev_i32 v3, 2, v4 :: v_dual_ashrrev_i32 v2, 31, v7
	v_and_b32_e32 v155, 31, v0
	s_max_i32 s16, s17, 0
	s_lshl_b64 s[44:45], s[44:45], 1
	s_lshl_b64 s[24:25], s[24:25], 1
	v_dual_lshrrev_b32 v2, 30, v2 :: v_dual_lshlrev_b32 v128, 1, v155
	s_and_b32 vcc_lo, s59, vcc_lo
	s_add_nc_u64 s[14:15], s[14:15], s[24:25]
	v_mad_u32_u24 v194, 0x110, v154, v1
	v_add_nc_u32_e32 v2, v7, v2
	s_add_nc_u64 s[14:15], s[14:15], s[44:45]
	v_bfe_u32 v0, v0, 4, 1
	v_subrev_co_ci_u32_e64 v193, null, 0, v3, vcc_lo
	v_and_b32_e32 v6, -4, v2
	v_ashrrev_i32_e32 v2, 2, v2
	v_add_nc_u32_e32 v130, s13, v194
	v_cvt_pk_bf16_f32 v129, s4, s4
	s_set_vgpr_msb 0xc0
	v_lshlrev_b32_e32 v200 /*v968*/, 3, v0
	v_cmp_ne_u32_e64 s2, v7, v6
	s_set_vgpr_msb 0xc000
	v_add_nc_u32_e32 v135, 0x23000, v194
	s_bitset1_b32 s15, 31
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x2
	scratch_store_b32 off, v7, off offset:556 nv
	scratch_store_b32 off, v2, off offset:560 nv
	scratch_store_b32 off, v0, off offset:572 nv
	s_and_b32 s2, s59, s2
	s_cmp_lg_u32 s5, 0x80000000
	v_subrev_co_ci_u32_e64 v192, null, 0, v2, s2
	s_cselect_b32 s23, s23, 0
	s_cselect_b32 s22, s5, 0x200
	s_cmp_lg_u32 s9, 0x80000000
	s_set_vgpr_msb 12
	scratch_store_b32 off, v200 /*v968*/, off offset:8 nv
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0xc00
	s_clause 0x4
	scratch_store_b32 off, v192, off offset:564 nv
	scratch_store_b32 off, v193, off offset:568 nv
	scratch_store_b32 off, v154, off offset:576 nv
	scratch_store_b32 off, v155, off offset:580 nv
	scratch_store_b32 off, v194, off offset:584 nv
	s_cselect_b32 s25, s9, 0x80
	s_cselect_b32 s5, s27, 0
	s_bfe_i32 s24, s37, 0x10018
	s_lshl_b32 s9, s22, 16
	s_lshr_b64 s[26:27], s[22:23], 16
	s_and_b32 s5, s5, 0xffff
	s_lshr_b32 s42, s24, 30
	s_and_b32 s27, s26, 0xffff0000
	s_or_b32 s26, s5, s9
	s_or_b32 s5, s42, s90
	s_lshr_b32 s17, s22, 16
	s_addk_co_i32 s5, 0x7f
	s_sub_co_i32 s22, s58, s57
	s_ashr_i32 s5, s5, 2
	s_add_co_i32 s35, s57, -1
	s_ashr_i32 s65, s64, 31
	s_ashr_i32 s61, s6, 31
	s_ashr_i32 s63, s7, 31
	s_add_co_i32 s44, s56, s22
	s_add_co_i32 s5, s5, s24
	s_mul_u64 s[6:7], s[64:65], s[60:61]
	s_mul_u64 s[22:23], s[64:65], s[62:63]
	s_add_co_i32 s65, s44, 1
	s_min_i32 s5, s5, s35
	s_ashr_i32 s41, s40, 31
	s_ashr_i32 s37, s10, 31
	s_add_co_i32 s5, s65, s5
	s_ashr_i32 s39, s11, 31
	s_mul_u64 s[10:11], s[40:41], s[36:37]
	s_min_i32 s5, s5, s58
	s_mul_u64 s[36:37], s[40:41], s[38:39]
	s_lshl_b64 s[70:71], s[10:11], 1
	s_lshl_b64 s[10:11], s[22:23], 1
	s_max_i32 s35, s5, 1
	s_lshl_b64 s[6:7], s[6:7], 1
	s_lshl_b64 s[72:73], s[36:37], 1
	s_add_nc_u64 s[10:11], s[68:69], s[10:11]
	s_add_co_i32 s9, s35, 0xff
	s_add_nc_u64 s[6:7], s[66:67], s[6:7]
	s_or_b32 s27, s27, s17
	s_add_nc_u64 s[74:75], s[10:11], s[72:73]
	s_min_i32 s92, s35, 0x100
	s_lshr_b32 s10, s9, 8
	s_add_nc_u64 s[76:77], s[6:7], s[70:71]
	s_cmp_lg_u32 s20, s19
	s_mov_b32 s24, 0x80004
	s_mov_b32 s23, 0x807fff
	s_mov_b32 s22, 0xffff7fff
	s_mov_b32 s20, 0x7510000
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
	s_lshl_b32 s17, s2, 6
	s_lshl_b32 s78, s2, 7
	s_mul_i32 s93, s2, 0x4400
	s_mul_i32 s94, s2, 0x4800
	s_sub_co_i32 s2, s92, s17
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
	s_mov_b32 s40, 64
	s_add_co_i32 s94, s94, 0x11000
	s_add_nc_u64 s[80:81], s[78:79], s[74:75]
	s_mov_b32 s36, s20
	s_mov_b32 s37, s21
	s_mov_b64 s[52:53], s[12:13]
	s_mov_b32 s5, s93
	s_bitset1_b32 s7, 31
	s_mov_b32 s44, 0xf510000
	s_mov_b32 s45, s21
	s_mov_b32 s51, s43
	s_mov_b32 s47, s39
	s_mov_b32 s48, s40
	s_mov_b32 s46, s38
	s_mov_b32 s53, s94
	s_and_b32 s50, s55, 0xffff
	s_or_b32 s55, s81, 0x80000000
	s_mov_b32 s54, s80
	s_ashr_i32 s2, s90, 2
	s_add_co_i32 s11, s65, -1
	s_set_vgpr_msb 48
	v_and_or_b32 v64, v155, 7, v200 /*v968*/
	s_add_co_i32 s2, s11, s2
	s_add_nc_u64 s[80:81], s[66:67], s[70:71]
	s_max_i32 s2, s2, 0
	s_add_co_i32 s2, s2, 1
	v_mul_u32_u24_e32 v64, 0x120, v64
	s_set_vgpr_msb 0x3000
	v_and_or_b32 v64, v128, 16, v64
	v_add_nc_u32_e32 v132, 0x11000, v64
	s_set_vgpr_msb 64
	v_or_b32_e32 v40 /*v296*/, 0x34000, v64
	s_wait_tensorcnt 0x0
	s_wait_alu depctr_vm_vsrc(6)
	s_set_vgpr_msb 0x4000
	ds_load_b128 v[0:3], v130
	ds_load_b128 v[4:7], v130 offset:32
	ds_load_b128 v[8:11], v130 offset:64
	ds_load_b128 v[12:15], v130 offset:96
	ds_load_b128 v[16:19], v130 offset:128
	ds_load_b128 v[20:23], v130 offset:160
	ds_load_b128 v[24:27], v130 offset:192
	ds_load_b128 v[28:31], v130 offset:224
	ds_load_b128 v[32:35], v130 offset:4352
	ds_load_b128 v[36:39], v130 offset:4384
	ds_load_b128 v[40:43], v130 offset:4416
	ds_load_b128 v[44:47], v130 offset:4448
	ds_load_b128 v[48:51], v130 offset:4480
	ds_load_b128 v[52:55], v130 offset:4512
	ds_load_b128 v[56:59], v130 offset:4544
	ds_load_b128 v[60:63], v130 offset:4576
	tensor_load_to_lds s[4:7], s[36:43]
	tensor_load_to_lds s[52:55], s[44:51]
	s_ashr_i32 s4, s2, 31
	scratch_store_b32 off, v128, off offset:492 nv
	s_lshr_b32 s4, s4, 24
	s_add_co_i32 s5, s10, -1
	s_add_co_i32 s4, s2, s4
	s_set_vgpr_msb 64
	s_wait_dscnt 0xe
	v_pk_mul_bf16 v49 /*v305*/, v129, v7
	s_and_b32 s6, s4, 0xffffff00
	s_ashr_i32 s4, s4, 8
	s_cmp_lg_u32 s2, s6
	v_pk_mul_bf16 v48 /*v304*/, v129, v6
	s_cselect_b32 s6, -1, 0
	s_cmp_lt_i32 s2, 0
	v_pk_mul_bf16 v47 /*v303*/, v129, v5
	s_cselect_b32 s2, -1, 0
	v_pk_mul_bf16 v46 /*v302*/, v129, v4
	s_and_b32 s2, s2, s6
	s_sub_co_ci_u32 s2, s4, 0
	v_pk_mul_bf16 v45 /*v301*/, v129, v3
	s_min_i32 s2, s2, s5
	v_pk_mul_bf16 v44 /*v300*/, v129, v2
	v_pk_mul_bf16 v43 /*v299*/, v129, v1
	v_pk_mul_bf16 v42 /*v298*/, v129, v0
	s_set_vgpr_msb 0x4000
	s_wait_dscnt 0xc
	v_pk_mul_bf16 v143, v129, v15
	v_pk_mul_bf16 v142, v129, v14
	v_pk_mul_bf16 v141, v129, v13
	v_pk_mul_bf16 v140, v129, v12
	v_pk_mul_bf16 v139, v129, v11
	v_pk_mul_bf16 v138, v129, v10
	v_pk_mul_bf16 v137, v129, v9
	v_pk_mul_bf16 v136, v129, v8
	s_wait_dscnt 0xa
	v_pk_mul_bf16 v151, v129, v23
	v_pk_mul_bf16 v150, v129, v22
	v_pk_mul_bf16 v149, v129, v21
	v_pk_mul_bf16 v148, v129, v20
	v_pk_mul_bf16 v147, v129, v19
	v_pk_mul_bf16 v146, v129, v18
	v_pk_mul_bf16 v145, v129, v17
	v_pk_mul_bf16 v144, v129, v16
	s_wait_dscnt 0x8
	v_pk_mul_bf16 v159, v129, v31
	v_pk_mul_bf16 v158, v129, v30
	v_pk_mul_bf16 v157, v129, v29
	v_pk_mul_bf16 v156, v129, v28
	s_wait_alu depctr_vm_vsrc(0)
	v_pk_mul_bf16 v155, v129, v27
	v_pk_mul_bf16 v154, v129, v26
	v_pk_mul_bf16 v153, v129, v25
	v_pk_mul_bf16 v152, v129, v24
	s_wait_dscnt 0x6
	v_pk_mul_bf16 v167, v129, v39
	v_pk_mul_bf16 v166, v129, v38
	v_pk_mul_bf16 v165, v129, v37
	v_pk_mul_bf16 v164, v129, v36
	v_pk_mul_bf16 v163, v129, v35
	v_pk_mul_bf16 v162, v129, v34
	v_pk_mul_bf16 v161, v129, v33
	v_pk_mul_bf16 v160, v129, v32
	s_wait_dscnt 0x4
	v_pk_mul_bf16 v175, v129, v47
	v_pk_mul_bf16 v174, v129, v46
	v_pk_mul_bf16 v173, v129, v45
	v_pk_mul_bf16 v172, v129, v44
	v_pk_mul_bf16 v171, v129, v43
	v_pk_mul_bf16 v170, v129, v42
	v_pk_mul_bf16 v169, v129, v41
	v_pk_mul_bf16 v168, v129, v40
	s_wait_dscnt 0x2
	v_pk_mul_bf16 v183, v129, v55
	v_pk_mul_bf16 v182, v129, v54
	v_pk_mul_bf16 v181, v129, v53
	v_pk_mul_bf16 v180, v129, v52
	v_pk_mul_bf16 v179, v129, v51
	v_pk_mul_bf16 v178, v129, v50
	v_pk_mul_bf16 v177, v129, v49
	v_pk_mul_bf16 v176, v129, v48
	s_wait_dscnt 0x0
	v_pk_mul_bf16 v191, v129, v63
	v_pk_mul_bf16 v190, v129, v62
	v_pk_mul_bf16 v189, v129, v61
	v_pk_mul_bf16 v188, v129, v60
	v_pk_mul_bf16 v187, v129, v59
	v_pk_mul_bf16 v186, v129, v58
	v_pk_mul_bf16 v185, v129, v57
	v_pk_mul_bf16 v184, v129, v56
	v_mov_b32_e32 v0, 0
	s_max_i32 s52, s2, 0
	s_mov_b32 s53, s43
	s_add_nc_u64 s[54:55], s[68:69], s[72:73]
	s_cmp_lt_i32 s2, 1
	s_wait_tensorcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	scratch_store_b32 off, v130, off offset:524 nv
	s_cbranch_scc1 .LBB0_11
	v_dual_mov_b32 v1, v0 :: v_dual_mov_b32 v2, v0
	v_dual_mov_b32 v3, v0 :: v_dual_mov_b32 v4, v0
	v_dual_mov_b32 v5, v0 :: v_dual_mov_b32 v6, v0
	v_dual_mov_b32 v7, v0 :: v_dual_mov_b32 v193, v135
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
	s_set_vgpr_msb 64
	v_dual_mov_b32 v50 /*v306*/, 0xf149f2ca :: v_dual_mov_b32 v51 /*v307*/, 0xf149f2ca
	s_set_vgpr_msb 0x4001
	v_mov_b32_e32 v192, v40 /*v296*/
	s_set_vgpr_msb 0x100
	v_dual_mov_b32 v133, v194 :: v_dual_mov_b32 v204, v0
	v_mov_b32_e32 v205, v0
	s_lshl_b64 s[82:83], s[52:53], 8
	s_mov_b32 s4, 1
	s_add_co_i32 s95, s35, 0xffffff00
	s_add_co_i32 s84, s64, 0x100
	s_mov_b32 s37, 0xffff0000
	s_mov_b32 s36, 0x7510000
	s_mov_b64 s[86:87], 0xffffffffffffff00
	s_mov_b32 s96, 0x76543210
	s_mov_b32 s56, 0x3fb8aa3b
	s_mov_b32 s97, 1
	s_branch .LBB0_4
.LBB0_3:
	s_set_vgpr_msb 0x4f
	v_cvt_pk_bf16_f32 v23 /*v279*/, v164 /*v932*/, v194 /*v962*/
	v_cvt_pk_bf16_f32 v22 /*v278*/, v148 /*v916*/, v162 /*v930*/
	v_cvt_pk_bf16_f32 v21 /*v277*/, v138 /*v906*/, v142 /*v910*/
	v_cvt_pk_bf16_f32 v20 /*v276*/, v130 /*v898*/, v134 /*v902*/
	v_cvt_pk_bf16_f32 v19 /*v275*/, v126 /*v894*/, v128 /*v896*/
	v_cvt_pk_bf16_f32 v18 /*v274*/, v122 /*v890*/, v124 /*v892*/
	v_cvt_pk_bf16_f32 v17 /*v273*/, v68 /*v836*/, v120 /*v888*/
	v_cvt_pk_bf16_f32 v16 /*v272*/, v64 /*v832*/, v66 /*v834*/
	v_cvt_pk_bf16_f32 v31 /*v287*/, v165 /*v933*/, v195 /*v963*/
	v_cvt_pk_bf16_f32 v30 /*v286*/, v149 /*v917*/, v163 /*v931*/
	v_cvt_pk_bf16_f32 v29 /*v285*/, v139 /*v907*/, v143 /*v911*/
	v_cvt_pk_bf16_f32 v28 /*v284*/, v131 /*v899*/, v135 /*v903*/
	v_cvt_pk_bf16_f32 v27 /*v283*/, v127 /*v895*/, v129 /*v897*/
	v_cvt_pk_bf16_f32 v26 /*v282*/, v123 /*v891*/, v125 /*v893*/
	v_cvt_pk_bf16_f32 v25 /*v281*/, v69 /*v837*/, v121 /*v889*/
	v_cvt_pk_bf16_f32 v24 /*v280*/, v65 /*v833*/, v67 /*v835*/
	s_set_vgpr_msb 0x4f06
	v_wmma_f32_16x16x32_bf16 v[120:127], v[184:191] /*v[696:703]*/, v[16:23] /*v[272:279]*/, v[120:127]
	s_set_vgpr_msb 0x68f
	v_cvt_pk_bf16_f32 v205 /*v717*/, v199 /*v967*/, v225 /*v993*/
	v_cvt_pk_bf16_f32 v204 /*v716*/, v175 /*v943*/, v197 /*v965*/
	v_cvt_pk_bf16_f32 v203 /*v715*/, v151 /*v919*/, v167 /*v935*/
	v_cvt_pk_bf16_f32 v202 /*v714*/, v141 /*v909*/, v145 /*v913*/
	v_cvt_pk_bf16_f32 v201 /*v713*/, v133 /*v901*/, v137 /*v905*/
	v_cvt_pk_bf16_f32 v200 /*v712*/, v83 /*v851*/, v89 /*v857*/
	v_cvt_pk_bf16_f32 v199 /*v711*/, v75 /*v843*/, v79 /*v847*/
	s_set_vgpr_msb 0x8f06
	v_wmma_f32_16x16x32_bf16 v[56:63], v[184:191] /*v[696:703]*/, v[24:31] /*v[280:287]*/, v[56:63]
	s_set_vgpr_msb 0x68f
	v_cvt_pk_bf16_f32 v198 /*v710*/, v71 /*v839*/, v73 /*v841*/
	v_cvt_pk_bf16_f32 v213 /*v725*/, v229 /*v997*/, v251 /*v1019*/
	v_cvt_pk_bf16_f32 v212 /*v724*/, v209 /*v977*/, v227 /*v995*/
	v_cvt_pk_bf16_f32 v211 /*v723*/, v179 /*v947*/, v201 /*v969*/
	v_cvt_pk_bf16_f32 v191 /*v703*/, v198 /*v966*/, v224 /*v992*/
	v_cvt_pk_bf16_f32 v190 /*v702*/, v174 /*v942*/, v196 /*v964*/
	v_cvt_pk_bf16_f32 v189 /*v701*/, v150 /*v918*/, v166 /*v934*/
	v_cvt_pk_bf16_f32 v188 /*v700*/, v140 /*v908*/, v144 /*v912*/
	v_cvt_pk_bf16_f32 v187 /*v699*/, v132 /*v900*/, v136 /*v904*/
	v_cvt_pk_bf16_f32 v186 /*v698*/, v82 /*v850*/, v88 /*v856*/
	v_cvt_pk_bf16_f32 v185 /*v697*/, v74 /*v842*/, v78 /*v846*/
	v_cvt_pk_bf16_f32 v184 /*v696*/, v70 /*v838*/, v72 /*v840*/
	s_set_vgpr_msb 0x8f0a
	v_wmma_f32_16x16x32_bf16 v[120:127], v[176:183] /*v[688:695]*/, v[184:191] /*v[696:703]*/, v[120:127]
	s_set_vgpr_msb 0xa8f
	v_cvt_pk_bf16_f32 v210 /*v722*/, v153 /*v921*/, v169 /*v937*/
	v_cvt_pk_bf16_f32 v209 /*v721*/, v103 /*v871*/, v147 /*v915*/
	v_cvt_pk_bf16_f32 v208 /*v720*/, v95 /*v863*/, v101 /*v869*/
	v_cvt_pk_bf16_f32 v207 /*v719*/, v85 /*v853*/, v91 /*v859*/
	v_cvt_pk_bf16_f32 v206 /*v718*/, v77 /*v845*/, v81 /*v849*/
	s_set_vgpr_msb 0x8f83
	v_cvt_pk_bf16_f32 v221 /*v733*/, v255 /*v1023*/, v207
	s_set_vgpr_msb 0x838f
	v_cvt_pk_bf16_f32 v220 /*v732*/, v239 /*v1007*/, v253 /*v1021*/
	s_set_vgpr_msb 0x8f0a
	v_wmma_f32_16x16x32_bf16 v[56:63], v[176:183] /*v[688:695]*/, v[198:205] /*v[710:717]*/, v[56:63]
	s_set_vgpr_msb 0xa8f
	v_cvt_pk_bf16_f32 v219 /*v731*/, v211 /*v979*/, v231 /*v999*/
	v_cvt_pk_bf16_f32 v218 /*v730*/, v181 /*v949*/, v203 /*v971*/
	v_cvt_pk_bf16_f32 v217 /*v729*/, v117 /*v885*/, v171 /*v939*/
	v_cvt_pk_bf16_f32 v216 /*v728*/, v109 /*v877*/, v115 /*v883*/
	v_cvt_pk_bf16_f32 v183 /*v695*/, v228 /*v996*/, v250 /*v1018*/
	v_cvt_pk_bf16_f32 v182 /*v694*/, v208 /*v976*/, v226 /*v994*/
	v_cvt_pk_bf16_f32 v181 /*v693*/, v178 /*v946*/, v200 /*v968*/
	v_cvt_pk_bf16_f32 v180 /*v692*/, v152 /*v920*/, v168 /*v936*/
	v_cvt_pk_bf16_f32 v179 /*v691*/, v102 /*v870*/, v146 /*v914*/
	v_cvt_pk_bf16_f32 v178 /*v690*/, v94 /*v862*/, v100 /*v868*/
	v_cvt_pk_bf16_f32 v177 /*v689*/, v84 /*v852*/, v90 /*v858*/
	v_cvt_pk_bf16_f32 v176 /*v688*/, v76 /*v844*/, v80 /*v848*/
	s_set_vgpr_msb 0x8f0a
	v_wmma_f32_16x16x32_bf16 v[120:127], v[168:175] /*v[680:687]*/, v[176:183] /*v[688:695]*/, v[120:127]
	s_set_vgpr_msb 0xa8f
	v_cvt_pk_bf16_f32 v215 /*v727*/, v97 /*v865*/, v105 /*v873*/
	v_cvt_pk_bf16_f32 v214 /*v726*/, v87 /*v855*/, v93 /*v861*/
	s_set_vgpr_msb 0x8f80
	v_cvt_pk_bf16_f32 v229 /*v741*/, v211, v225
	v_cvt_pk_bf16_f32 v228 /*v740*/, v199, v209
	s_set_vgpr_msb 0x808f
	v_cvt_pk_bf16_f32 v227 /*v739*/, v241 /*v1009*/, v177 /*v945*/
	v_cvt_pk_bf16_f32 v226 /*v738*/, v213 /*v981*/, v233 /*v1001*/
	v_cvt_pk_bf16_f32 v225 /*v737*/, v183 /*v951*/, v205 /*v973*/
	s_set_vgpr_msb 0x8f0a
	v_wmma_f32_16x16x32_bf16 v[56:63], v[168:175] /*v[680:687]*/, v[206:213] /*v[718:725]*/, v[56:63]
	s_set_vgpr_msb 0xa8f
	v_cvt_pk_bf16_f32 v224 /*v736*/, v157 /*v925*/, v173 /*v941*/
	v_cvt_pk_bf16_f32 v223 /*v735*/, v111 /*v879*/, v119 /*v887*/
	v_cvt_pk_bf16_f32 v222 /*v734*/, v99 /*v867*/, v107 /*v875*/
	s_set_vgpr_msb 0x8f80
	v_cvt_pk_bf16_f32 v237 /*v749*/, v229, v239
	s_set_vgpr_msb 0x8083
	v_cvt_pk_bf16_f32 v175 /*v687*/, v254 /*v1022*/, v206
	s_set_vgpr_msb 0x838f
	v_cvt_pk_bf16_f32 v174 /*v686*/, v238 /*v1006*/, v252 /*v1020*/
	v_cvt_pk_bf16_f32 v173 /*v685*/, v210 /*v978*/, v230 /*v998*/
	v_cvt_pk_bf16_f32 v172 /*v684*/, v180 /*v948*/, v202 /*v970*/
	v_cvt_pk_bf16_f32 v171 /*v683*/, v116 /*v884*/, v170 /*v938*/
	v_cvt_pk_bf16_f32 v170 /*v682*/, v108 /*v876*/, v114 /*v882*/
	v_cvt_pk_bf16_f32 v169 /*v681*/, v96 /*v864*/, v104 /*v872*/
	v_cvt_pk_bf16_f32 v168 /*v680*/, v86 /*v854*/, v92 /*v860*/
	s_set_vgpr_msb 0x8f0a
	v_wmma_f32_16x16x32_bf16 v[120:127], v[160:167] /*v[672:679]*/, v[168:175] /*v[680:687]*/, v[120:127]
	s_set_vgpr_msb 0xa80
	v_cvt_pk_bf16_f32 v236 /*v748*/, v219, v227
	v_cvt_pk_bf16_f32 v235 /*v747*/, v201, v213
	s_set_vgpr_msb 0x8083
	v_cvt_pk_bf16_f32 v234 /*v746*/, v243 /*v1011*/, v193
	s_set_vgpr_msb 0x838f
	v_cvt_pk_bf16_f32 v233 /*v745*/, v215 /*v983*/, v235 /*v1003*/
	v_cvt_pk_bf16_f32 v232 /*v744*/, v189 /*v957*/, v207 /*v975*/
	v_cvt_pk_bf16_f32 v231 /*v743*/, v159 /*v927*/, v185 /*v953*/
	v_cvt_pk_bf16_f32 v230 /*v742*/, v113 /*v881*/, v155 /*v923*/
	s_set_vgpr_msb 0x8f0a
	v_wmma_f32_16x16x32_bf16 v[56:63], v[160:167] /*v[672:679]*/, v[214:221] /*v[726:733]*/, v[56:63]
	s_set_vgpr_msb 0xa00
	v_cvt_pk_bf16_f32 v211, v220, v230
	v_cvt_pk_bf16_f32 v227, v204, v216
	v_cvt_pk_bf16_f32 v199, v205, v217
	v_cvt_pk_bf16_f32 v213, v242, v248
	s_set_vgpr_msb 0x80
	v_cvt_pk_bf16_f32 v167 /*v679*/, v210, v224
	v_cvt_pk_bf16_f32 v166 /*v678*/, v198, v208
	s_set_vgpr_msb 0x808f
	v_cvt_pk_bf16_f32 v165 /*v677*/, v240 /*v1008*/, v176 /*v944*/
	v_cvt_pk_bf16_f32 v164 /*v676*/, v212 /*v980*/, v232 /*v1000*/
	v_cvt_pk_bf16_f32 v163 /*v675*/, v182 /*v950*/, v204 /*v972*/
	v_cvt_pk_bf16_f32 v162 /*v674*/, v156 /*v924*/, v172 /*v940*/
	v_cvt_pk_bf16_f32 v161 /*v673*/, v110 /*v878*/, v118 /*v886*/
	v_cvt_pk_bf16_f32 v160 /*v672*/, v98 /*v866*/, v106 /*v874*/
	s_set_vgpr_msb 0x8f0a
	v_wmma_f32_16x16x32_bf16 v[120:127], v[152:159] /*v[664:671]*/, v[160:167] /*v[672:679]*/, v[120:127]
	s_set_vgpr_msb 0xa00
	v_cvt_pk_bf16_f32 v210, v202, v214
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v209, v244 /*v1012*/, v194
	s_set_vgpr_msb 0x30f
	v_cvt_pk_bf16_f32 v208, v220 /*v988*/, v236 /*v1004*/
	v_cvt_pk_bf16_f32 v207, v190 /*v958*/, v216 /*v984*/
	v_cvt_pk_bf16_f32 v206, v160 /*v928*/, v186 /*v954*/
	s_set_vgpr_msb 0xf00
	v_cvt_pk_bf16_f32 v230, v246, v250
	v_cvt_pk_bf16_f32 v229, v236, v244
	s_set_vgpr_msb 10
	v_wmma_f32_16x16x32_bf16 v[56:63], v[152:159] /*v[664:671]*/, v[222:229] /*v[734:741]*/, v[56:63]
	s_set_vgpr_msb 0xa0f
	v_cvt_pk_bf16_f32 v225, v222 /*v990*/, v246 /*v1014*/
	v_cvt_pk_bf16_f32 v224, v192 /*v960*/, v218 /*v986*/
	s_set_vgpr_msb 0xf00
	v_cvt_pk_bf16_f32 v202, v247, v251
	v_cvt_pk_bf16_f32 v201, v237, v245
	s_set_vgpr_msb 0x80
	v_cvt_pk_bf16_f32 v159 /*v671*/, v228, v238
	v_cvt_pk_bf16_f32 v158 /*v670*/, v218, v226
	v_cvt_pk_bf16_f32 v157 /*v669*/, v200, v212
	s_set_vgpr_msb 0x8083
	v_cvt_pk_bf16_f32 v156 /*v668*/, v242 /*v1010*/, v192
	s_set_vgpr_msb 0x838f
	v_cvt_pk_bf16_f32 v155 /*v667*/, v214 /*v982*/, v234 /*v1002*/
	v_cvt_pk_bf16_f32 v154 /*v666*/, v188 /*v956*/, v206 /*v974*/
	v_cvt_pk_bf16_f32 v153 /*v665*/, v158 /*v926*/, v184 /*v952*/
	v_cvt_pk_bf16_f32 v152 /*v664*/, v112 /*v880*/, v154 /*v922*/
	s_set_vgpr_msb 0x8f0a
	v_wmma_f32_16x16x32_bf16 v[120:127], v[144:151] /*v[656:663]*/, v[152:159] /*v[664:671]*/, v[120:127]
	s_set_vgpr_msb 0xa00
	v_cvt_pk_bf16_f32 v212, v234, v240
	v_cvt_pk_bf16_f32 v228, v222, v232
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v226, v248 /*v1016*/, v196
	s_set_vgpr_msb 0x300
	v_cvt_pk_bf16_f32 v200, v223, v233
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v198, v249 /*v1017*/, v197
	s_set_vgpr_msb 0x30f
	v_cvt_pk_bf16_f32 v197, v223 /*v991*/, v247 /*v1015*/
	v_cvt_pk_bf16_f32 v196, v193 /*v961*/, v219 /*v987*/
	s_set_vgpr_msb 0xf0a
	v_wmma_f32_16x16x32_bf16 v[56:63], v[144:151] /*v[656:663]*/, v[230:237] /*v[742:749]*/, v[56:63]
	s_wait_alu depctr_va_vdst(0)
	scratch_load_b64 v[130:131], off, off th:TH_LOAD_LU nv
	s_set_vgpr_msb 0xa40
	v_dual_mov_b32 v50 /*v306*/, v134 :: v_dual_mov_b32 v51 /*v307*/, v128
	s_add_nc_u64 s[82:83], s[82:83], s[86:87]
	s_addk_co_i32 s95, 0xff00
	s_addk_co_i32 s84, 0x100
	s_add_co_i32 s97, s97, 1
	s_set_vgpr_msb 0x4005
	v_wmma_f32_16x16x32_bf16 v[72:79], v[72:79] /*v[328:335]*/, v[16:23] /*v[272:279]*/, v[72:79]
	s_cmp_lg_u64 s[82:83], 0
	s_set_vgpr_msb 0x522
	s_wait_loadcnt 0x0
	v_pk_fma_f32 v[192:193], v[196:197] /*v[708:709]*/, v[130:131], v[194:195] /*v[706:707]*/
	v_nop
	s_set_vgpr_msb 0x2280
	v_cvt_pk_bf16_f32 v149 /*v661*/, v221, v231
	v_cvt_pk_bf16_f32 v148 /*v660*/, v203, v215
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x8005
	s_clause 0x1
	scratch_load_b128 v[214:217], off, off offset:364 th:TH_LOAD_LU nv
	scratch_load_b128 v[218:221], off, off offset:380 th:TH_LOAD_LU nv
	v_wmma_f32_16x16x32_bf16 v[8:15], v[72:79] /*v[328:335]*/, v[24:31] /*v[280:287]*/, v[8:15]
	s_set_vgpr_msb 0x580
	v_cvt_pk_bf16_f32 v151 /*v663*/, v243, v249
	v_cvt_pk_bf16_f32 v150 /*v662*/, v235, v241
	s_set_vgpr_msb 0x8083
	v_cvt_pk_bf16_f32 v147 /*v659*/, v245 /*v1013*/, v195
	s_set_vgpr_msb 0x838f
	v_cvt_pk_bf16_f32 v146 /*v658*/, v221 /*v989*/, v237 /*v1005*/
	v_cvt_pk_bf16_f32 v145 /*v657*/, v191 /*v959*/, v217 /*v985*/
	v_cvt_pk_bf16_f32 v144 /*v656*/, v161 /*v929*/, v187 /*v955*/
	s_set_vgpr_msb 0x8f00
	v_cvt_pk_bf16_f32 v231, v252, v254
	s_set_vgpr_msb 9
	v_wmma_f32_16x16x32_bf16 v[72:79], v[56:63] /*v[312:319]*/, v[184:191] /*v[696:703]*/, v[72:79]
	s_set_vgpr_msb 0x900
	v_cvt_pk_bf16_f32 v203, v253, v255
	s_set_vgpr_msb 8
	v_pk_add_f32 v[204:205], v[192:193], v[192:193] /*v[704:705]*/
	s_set_vgpr_msb 0x809
	v_dual_mov_b32 v192, v40 /*v296*/ :: v_dual_mov_b32 v193, v41 /*v297*/
	v_wmma_f32_16x16x32_bf16 v[8:15], v[56:63] /*v[312:319]*/, v[198:205] /*v[710:717]*/, v[8:15]
	v_wmma_f32_16x16x32_bf16 v[72:79], v[0:7] /*v[256:263]*/, v[176:183] /*v[688:695]*/, v[72:79]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[0:7] /*v[256:263]*/, v[206:213] /*v[718:725]*/, v[8:15]
	v_wmma_f32_16x16x32_bf16 v[72:79], v[32:39] /*v[288:295]*/, v[168:175] /*v[680:687]*/, v[72:79]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[32:39] /*v[288:295]*/, v[214:221] /*v[726:733]*/, v[8:15]
	s_set_vgpr_msb 0x908
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[72:79], v[214:221], v[160:167] /*v[672:679]*/, v[72:79]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[214:221], v[222:229] /*v[734:741]*/, v[8:15]
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b128 v[214:217], off, off offset:332 th:TH_LOAD_LU nv
	scratch_load_b128 v[218:221], off, off offset:348 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[72:79], v[214:221], v[152:159] /*v[664:671]*/, v[72:79]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[214:221], v[230:237] /*v[742:749]*/, v[8:15]
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b128 v[214:217], off, off offset:300 th:TH_LOAD_LU nv
	scratch_load_b128 v[218:221], off, off offset:316 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0x800
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[72:79], v[214:221], v[206:213], v[72:79]
	s_set_vgpr_msb 8
	v_wmma_f32_16x16x32_bf16 v[8:15], v[214:221], v[144:151] /*v[656:663]*/, v[8:15]
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b128 v[214:217], off, off offset:268 th:TH_LOAD_LU nv
	scratch_load_b128 v[218:221], off, off offset:284 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0x800
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[72:79], v[214:221], v[224:231], v[72:79]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[214:221], v[196:203], v[8:15]
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b128 v[214:217], off, off offset:236 th:TH_LOAD_LU nv
	scratch_load_b128 v[218:221], off, off offset:252 th:TH_LOAD_LU nv
	s_set_vgpr_msb 4
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[64:71], v[214:221], v[16:23] /*v[272:279]*/, v[64:71]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[214:221], v[24:31] /*v[280:287]*/, v[0:7]
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b128 v[214:217], off, off offset:204 th:TH_LOAD_LU nv
	scratch_load_b128 v[218:221], off, off offset:220 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0x408
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[64:71], v[214:221], v[184:191] /*v[696:703]*/, v[64:71]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[214:221], v[198:205] /*v[710:717]*/, v[0:7]
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b128 v[214:217], off, off offset:172 th:TH_LOAD_LU nv
	scratch_load_b128 v[218:221], off, off offset:188 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[64:71], v[214:221], v[176:183] /*v[688:695]*/, v[64:71]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[214:221], v[206:213] /*v[718:725]*/, v[0:7]
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b128 v[214:217], off, off offset:140 th:TH_LOAD_LU nv
	scratch_load_b128 v[218:221], off, off offset:156 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[64:71], v[214:221], v[168:175] /*v[680:687]*/, v[64:71]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[214:221], v[214:221] /*v[726:733]*/, v[0:7]
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b128 v[214:217], off, off offset:108 th:TH_LOAD_LU nv
	scratch_load_b128 v[218:221], off, off offset:124 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[64:71], v[214:221], v[160:167] /*v[672:679]*/, v[64:71]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[214:221], v[222:229] /*v[734:741]*/, v[0:7]
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b128 v[214:217], off, off offset:76 th:TH_LOAD_LU nv
	scratch_load_b128 v[218:221], off, off offset:92 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[64:71], v[214:221], v[152:159] /*v[664:671]*/, v[64:71]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[214:221], v[230:237] /*v[742:749]*/, v[0:7]
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b128 v[214:217], off, off offset:44 th:TH_LOAD_LU nv
	scratch_load_b128 v[218:221], off, off offset:60 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0x806
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[112:119], v[104:111] /*v[616:623]*/, v[16:23] /*v[272:279]*/, v[112:119]
	v_wmma_f32_16x16x32_bf16 v[104:111], v[72:79] /*v[584:591]*/, v[16:23] /*v[272:279]*/, v[104:111]
	s_set_vgpr_msb 0x605
	v_wmma_f32_16x16x32_bf16 v[96:103], v[248:255] /*v[504:511]*/, v[16:23] /*v[272:279]*/, v[96:103]
	v_wmma_f32_16x16x32_bf16 v[88:95], v[200:207] /*v[456:463]*/, v[16:23] /*v[272:279]*/, v[88:95]
	v_wmma_f32_16x16x32_bf16 v[80:87], v[120:127] /*v[376:383]*/, v[16:23] /*v[272:279]*/, v[80:87]
	s_set_vgpr_msb 0x50a
	v_wmma_f32_16x16x32_bf16 v[112:119], v[120:127] /*v[632:639]*/, v[184:191] /*v[696:703]*/, v[112:119]
	v_wmma_f32_16x16x32_bf16 v[104:111], v[56:63] /*v[568:575]*/, v[184:191] /*v[696:703]*/, v[104:111]
	s_set_vgpr_msb 0xa09
	v_wmma_f32_16x16x32_bf16 v[96:103], v[240:247] /*v[496:503]*/, v[184:191] /*v[696:703]*/, v[96:103]
	v_wmma_f32_16x16x32_bf16 v[88:95], v[184:191] /*v[440:447]*/, v[184:191] /*v[696:703]*/, v[88:95]
	v_wmma_f32_16x16x32_bf16 v[80:87], v[112:119] /*v[368:375]*/, v[184:191] /*v[696:703]*/, v[80:87]
	s_set_vgpr_msb 0x90a
	v_wmma_f32_16x16x32_bf16 v[112:119], v[112:119] /*v[624:631]*/, v[176:183] /*v[688:695]*/, v[112:119]
	v_wmma_f32_16x16x32_bf16 v[104:111], v[40:47] /*v[552:559]*/, v[176:183] /*v[688:695]*/, v[104:111]
	s_set_vgpr_msb 0xa09
	v_wmma_f32_16x16x32_bf16 v[96:103], v[232:239] /*v[488:495]*/, v[176:183] /*v[688:695]*/, v[96:103]
	v_wmma_f32_16x16x32_bf16 v[88:95], v[168:175] /*v[424:431]*/, v[176:183] /*v[688:695]*/, v[88:95]
	v_wmma_f32_16x16x32_bf16 v[80:87], v[104:111] /*v[360:367]*/, v[176:183] /*v[688:695]*/, v[80:87]
	s_set_vgpr_msb 0x90a
	v_wmma_f32_16x16x32_bf16 v[112:119], v[96:103] /*v[608:615]*/, v[168:175] /*v[680:687]*/, v[112:119]
	v_wmma_f32_16x16x32_bf16 v[104:111], v[32:39] /*v[544:551]*/, v[168:175] /*v[680:687]*/, v[104:111]
	s_set_vgpr_msb 0xa09
	v_wmma_f32_16x16x32_bf16 v[96:103], v[224:231] /*v[480:487]*/, v[168:175] /*v[680:687]*/, v[96:103]
	v_wmma_f32_16x16x32_bf16 v[88:95], v[160:167] /*v[416:423]*/, v[168:175] /*v[680:687]*/, v[88:95]
	v_wmma_f32_16x16x32_bf16 v[80:87], v[96:103] /*v[352:359]*/, v[168:175] /*v[680:687]*/, v[80:87]
	s_set_vgpr_msb 0x90a
	v_wmma_f32_16x16x32_bf16 v[112:119], v[88:95] /*v[600:607]*/, v[160:167] /*v[672:679]*/, v[112:119]
	v_wmma_f32_16x16x32_bf16 v[104:111], v[24:31] /*v[536:543]*/, v[160:167] /*v[672:679]*/, v[104:111]
	s_set_vgpr_msb 0xa09
	v_wmma_f32_16x16x32_bf16 v[96:103], v[216:223] /*v[472:479]*/, v[160:167] /*v[672:679]*/, v[96:103]
	v_wmma_f32_16x16x32_bf16 v[88:95], v[152:159] /*v[408:415]*/, v[160:167] /*v[672:679]*/, v[88:95]
	v_wmma_f32_16x16x32_bf16 v[80:87], v[88:95] /*v[344:351]*/, v[160:167] /*v[672:679]*/, v[80:87]
	s_set_vgpr_msb 0x90a
	v_wmma_f32_16x16x32_bf16 v[112:119], v[80:87] /*v[592:599]*/, v[152:159] /*v[664:671]*/, v[112:119]
	v_wmma_f32_16x16x32_bf16 v[104:111], v[16:23] /*v[528:535]*/, v[152:159] /*v[664:671]*/, v[104:111]
	s_set_vgpr_msb 0xa09
	v_wmma_f32_16x16x32_bf16 v[96:103], v[208:215] /*v[464:471]*/, v[152:159] /*v[664:671]*/, v[96:103]
	v_wmma_f32_16x16x32_bf16 v[88:95], v[144:151] /*v[400:407]*/, v[152:159] /*v[664:671]*/, v[88:95]
	v_wmma_f32_16x16x32_bf16 v[80:87], v[80:87] /*v[336:343]*/, v[152:159] /*v[664:671]*/, v[80:87]
	s_set_vgpr_msb 0x902
	v_wmma_f32_16x16x32_bf16 v[120:127], v[136:143] /*v[648:655]*/, v[206:213], v[120:127]
	v_wmma_f32_16x16x32_bf16 v[112:119], v[64:71] /*v[576:583]*/, v[206:213], v[112:119]
	v_wmma_f32_16x16x32_bf16 v[104:111], v[8:15] /*v[520:527]*/, v[206:213], v[104:111]
	s_set_vgpr_msb 0x201
	v_wmma_f32_16x16x32_bf16 v[96:103], v[192:199] /*v[448:455]*/, v[206:213], v[96:103]
	v_wmma_f32_16x16x32_bf16 v[88:95], v[136:143] /*v[392:399]*/, v[206:213], v[88:95]
	v_wmma_f32_16x16x32_bf16 v[80:87], v[64:71] /*v[320:327]*/, v[206:213], v[80:87]
	s_set_vgpr_msb 0x100
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[64:71], v[214:221], v[206:213], v[64:71]
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b128 v[206:209], off, off offset:12 th:TH_LOAD_LU nv
	scratch_load_b128 v[210:213], off, off offset:28 th:TH_LOAD_LU nv
	s_set_vgpr_msb 6
	v_wmma_f32_16x16x32_bf16 v[48:55], v[104:111] /*v[616:623]*/, v[24:31] /*v[280:287]*/, v[48:55]
	v_wmma_f32_16x16x32_bf16 v[40:47], v[72:79] /*v[584:591]*/, v[24:31] /*v[280:287]*/, v[40:47]
	s_set_vgpr_msb 0x605
	v_wmma_f32_16x16x32_bf16 v[32:39], v[248:255] /*v[504:511]*/, v[24:31] /*v[280:287]*/, v[32:39]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[200:207] /*v[456:463]*/, v[24:31] /*v[280:287]*/, v[24:31]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[120:127] /*v[376:383]*/, v[24:31] /*v[280:287]*/, v[16:23]
	s_set_vgpr_msb 0x50a
	v_wmma_f32_16x16x32_bf16 v[48:55], v[120:127] /*v[632:639]*/, v[198:205] /*v[710:717]*/, v[48:55]
	v_wmma_f32_16x16x32_bf16 v[40:47], v[56:63] /*v[568:575]*/, v[198:205] /*v[710:717]*/, v[40:47]
	s_set_vgpr_msb 0xa09
	v_wmma_f32_16x16x32_bf16 v[32:39], v[240:247] /*v[496:503]*/, v[198:205] /*v[710:717]*/, v[32:39]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[184:191] /*v[440:447]*/, v[198:205] /*v[710:717]*/, v[24:31]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[112:119] /*v[368:375]*/, v[198:205] /*v[710:717]*/, v[16:23]
	s_set_vgpr_msb 0x90a
	v_wmma_f32_16x16x32_bf16 v[48:55], v[112:119] /*v[624:631]*/, v[206:213] /*v[718:725]*/, v[48:55]
	v_wmma_f32_16x16x32_bf16 v[40:47], v[40:47] /*v[552:559]*/, v[206:213] /*v[718:725]*/, v[40:47]
	s_set_vgpr_msb 0xa09
	v_wmma_f32_16x16x32_bf16 v[32:39], v[232:239] /*v[488:495]*/, v[206:213] /*v[718:725]*/, v[32:39]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[168:175] /*v[424:431]*/, v[206:213] /*v[718:725]*/, v[24:31]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[104:111] /*v[360:367]*/, v[206:213] /*v[718:725]*/, v[16:23]
	s_set_vgpr_msb 0x90a
	v_wmma_f32_16x16x32_bf16 v[48:55], v[96:103] /*v[608:615]*/, v[214:221] /*v[726:733]*/, v[48:55]
	v_wmma_f32_16x16x32_bf16 v[40:47], v[32:39] /*v[544:551]*/, v[214:221] /*v[726:733]*/, v[40:47]
	s_set_vgpr_msb 0xa09
	v_wmma_f32_16x16x32_bf16 v[32:39], v[224:231] /*v[480:487]*/, v[214:221] /*v[726:733]*/, v[32:39]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[160:167] /*v[416:423]*/, v[214:221] /*v[726:733]*/, v[24:31]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[96:103] /*v[352:359]*/, v[214:221] /*v[726:733]*/, v[16:23]
	s_set_vgpr_msb 0x90a
	v_wmma_f32_16x16x32_bf16 v[48:55], v[88:95] /*v[600:607]*/, v[222:229] /*v[734:741]*/, v[48:55]
	v_wmma_f32_16x16x32_bf16 v[40:47], v[24:31] /*v[536:543]*/, v[222:229] /*v[734:741]*/, v[40:47]
	s_set_vgpr_msb 0xa09
	v_wmma_f32_16x16x32_bf16 v[32:39], v[216:223] /*v[472:479]*/, v[222:229] /*v[734:741]*/, v[32:39]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[152:159] /*v[408:415]*/, v[222:229] /*v[734:741]*/, v[24:31]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[88:95] /*v[344:351]*/, v[222:229] /*v[734:741]*/, v[16:23]
	s_set_vgpr_msb 0x90a
	v_wmma_f32_16x16x32_bf16 v[48:55], v[80:87] /*v[592:599]*/, v[230:237] /*v[742:749]*/, v[48:55]
	v_wmma_f32_16x16x32_bf16 v[40:47], v[16:23] /*v[528:535]*/, v[230:237] /*v[742:749]*/, v[40:47]
	s_set_vgpr_msb 0xa09
	v_wmma_f32_16x16x32_bf16 v[32:39], v[208:215] /*v[464:471]*/, v[230:237] /*v[742:749]*/, v[32:39]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[144:151] /*v[400:407]*/, v[230:237] /*v[742:749]*/, v[24:31]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[80:87] /*v[336:343]*/, v[230:237] /*v[742:749]*/, v[16:23]
	s_set_vgpr_msb 0x90a
	v_wmma_f32_16x16x32_bf16 v[56:63], v[136:143] /*v[648:655]*/, v[144:151] /*v[656:663]*/, v[56:63]
	v_wmma_f32_16x16x32_bf16 v[48:55], v[64:71] /*v[576:583]*/, v[144:151] /*v[656:663]*/, v[48:55]
	v_wmma_f32_16x16x32_bf16 v[40:47], v[8:15] /*v[520:527]*/, v[144:151] /*v[656:663]*/, v[40:47]
	s_set_vgpr_msb 0xa09
	v_wmma_f32_16x16x32_bf16 v[32:39], v[192:199] /*v[448:455]*/, v[144:151] /*v[656:663]*/, v[32:39]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[136:143] /*v[392:399]*/, v[144:151] /*v[656:663]*/, v[24:31]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[64:71] /*v[320:327]*/, v[144:151] /*v[656:663]*/, v[16:23]
	s_set_vgpr_msb 0x908
	v_wmma_f32_16x16x32_bf16 v[0:7], v[214:221], v[144:151] /*v[656:663]*/, v[0:7]
	s_set_vgpr_msb 0x802
	v_wmma_f32_16x16x32_bf16 v[120:127], v[128:135] /*v[640:647]*/, v[224:231], v[120:127]
	v_wmma_f32_16x16x32_bf16 v[56:63], v[128:135] /*v[640:647]*/, v[196:203], v[56:63]
	v_wmma_f32_16x16x32_bf16 v[112:119], v[48:55] /*v[560:567]*/, v[224:231], v[112:119]
	v_wmma_f32_16x16x32_bf16 v[48:55], v[48:55] /*v[560:567]*/, v[196:203], v[48:55]
	v_wmma_f32_16x16x32_bf16 v[104:111], v[0:7] /*v[512:519]*/, v[224:231], v[104:111]
	v_wmma_f32_16x16x32_bf16 v[40:47], v[0:7] /*v[512:519]*/, v[196:203], v[40:47]
	s_set_vgpr_msb 0x201
	v_wmma_f32_16x16x32_bf16 v[96:103], v[176:183] /*v[432:439]*/, v[224:231], v[96:103]
	v_wmma_f32_16x16x32_bf16 v[32:39], v[176:183] /*v[432:439]*/, v[196:203], v[32:39]
	v_wmma_f32_16x16x32_bf16 v[88:95], v[128:135] /*v[384:391]*/, v[224:231], v[88:95]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[128:135] /*v[384:391]*/, v[196:203], v[24:31]
	v_wmma_f32_16x16x32_bf16 v[80:87], v[8:15] /*v[264:271]*/, v[224:231], v[80:87]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[8:15] /*v[264:271]*/, v[196:203], v[16:23]
	s_set_vgpr_msb 0x100
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[64:71], v[206:213], v[224:231], v[64:71]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[206:213], v[196:203], v[0:7]
	s_cbranch_scc0 .LBB0_13
.LBB0_4:
	s_set_vgpr_msb 64
	v_dual_mov_b32 v41 /*v297*/, v133 :: v_dual_mov_b32 v40 /*v296*/, v132
	s_set_vgpr_msb 0x4000
	v_dual_mov_b32 v133, v193 :: v_dual_mov_b32 v132, v192
	s_wait_alu depctr_va_vdst(0)
	scratch_store_b64 off, v[204:205], off nv
	s_wait_tensorcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	s_cmp_ge_i32 s97, s10
	s_cbranch_scc1 .LBB0_6
	v_med3_i32 v128, s95, 0, 0x100
	s_ashr_i32 s85, s84, 31
	s_lshr_b32 s2, s97, 31
	s_mul_u64 s[6:7], s[84:85], s[62:63]
	s_add_co_i32 s2, s97, s2
	v_readfirstlane_b32 s5, v128
	s_lshl_b64 s[6:7], s[6:7], 1
	s_and_b32 s2, s2, 0xffffe
	s_add_nc_u64 s[46:47], s[54:55], s[6:7]
	s_mul_u64 s[6:7], s[84:85], s[60:61]
	s_sub_co_i32 s5, s5, s17
	s_lshl_b64 s[6:7], s[6:7], 1
	s_sub_co_i32 s2, s97, s2
	s_add_nc_u64 s[6:7], s[80:81], s[6:7]
	s_max_i32 s38, s5, 0
	s_mul_i32 s2, s2, 0x23000
	s_add_nc_u64 s[6:7], s[18:19], s[6:7]
	s_lshl_b32 s38, s38, 16
	s_add_co_i32 s5, s93, s2
	s_bitset1_b32 s7, 31
	s_addk_co_i32 s38, 0x7fff
	s_mov_b32 s45, s37
	tensor_load_to_lds s[4:7], s[36:43]
	s_add_nc_u64 s[6:7], s[78:79], s[46:47]
	s_add_co_i32 s5, s94, s2
	s_bitset1_b32 s7, 31
	s_mov_b32 s46, s38
	s_mov_b32 s47, s39
	s_mov_b32 s48, s40
	s_mov_b32 s51, s43
	tensor_load_to_lds s[4:7], s[44:51]
.LBB0_6:
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 1
	ds_load_b128 v[192:195], v41 /*v297*/
	ds_load_b128 v[196:199], v41 /*v297*/ offset:32
	ds_load_b128 v[200:203], v41 /*v297*/ offset:64
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_b128 v[204:207], v41 /*v297*/ offset:96
	ds_load_b128 v[208:211], v41 /*v297*/ offset:128
	ds_load_b128 v[212:215], v41 /*v297*/ offset:160
	ds_load_b128 v[216:219], v41 /*v297*/ offset:192
	ds_load_b128 v[220:223], v41 /*v297*/ offset:224
	ds_load_b128 v[224:227], v41 /*v297*/ offset:4352
	ds_load_b128 v[228:231], v41 /*v297*/ offset:4384
	ds_load_b128 v[232:235], v41 /*v297*/ offset:4416
	ds_load_b128 v[236:239], v41 /*v297*/ offset:4448
	ds_load_b128 v[240:243], v41 /*v297*/ offset:4480
	ds_load_b128 v[244:247], v41 /*v297*/ offset:4512
	ds_load_b128 v[248:251], v41 /*v297*/ offset:4544
	ds_load_b128 v[252:255], v41 /*v297*/ offset:4576
	s_set_vgpr_msb 0x141
	ds_load_b128 v[0:3] /*v[256:259]*/, v41 /*v297*/ offset:8704
	ds_load_b128 v[4:7] /*v[260:263]*/, v41 /*v297*/ offset:8736
	ds_load_b128 v[8:11] /*v[264:267]*/, v41 /*v297*/ offset:8768
	ds_load_b128 v[12:15] /*v[268:271]*/, v41 /*v297*/ offset:8800
	ds_load_b128 v[16:19] /*v[272:275]*/, v41 /*v297*/ offset:8832
	ds_load_b128 v[20:23] /*v[276:279]*/, v41 /*v297*/ offset:8864
	ds_load_b128 v[24:27] /*v[280:283]*/, v41 /*v297*/ offset:8896
	ds_load_b128 v[28:31] /*v[284:287]*/, v41 /*v297*/ offset:8928
	ds_load_b128 v[32:35] /*v[288:291]*/, v41 /*v297*/ offset:13056
	ds_load_b128 v[36:39] /*v[292:295]*/, v41 /*v297*/ offset:13088
	ds_load_b128 v[52:55] /*v[308:311]*/, v41 /*v297*/ offset:13120
	ds_load_b128 v[56:59] /*v[312:315]*/, v41 /*v297*/ offset:13152
	ds_load_b128 v[60:63] /*v[316:319]*/, v41 /*v297*/ offset:13184
	ds_load_b128 v[64:67] /*v[320:323]*/, v41 /*v297*/ offset:13216
	ds_load_b128 v[68:71] /*v[324:327]*/, v41 /*v297*/ offset:13248
	ds_load_b128 v[72:75] /*v[328:331]*/, v41 /*v297*/ offset:13280
	ds_load_b128 v[76:79] /*v[332:335]*/, v41 /*v297*/ offset:17408
	ds_load_b128 v[80:83] /*v[336:339]*/, v41 /*v297*/ offset:17440
	ds_load_b128 v[84:87] /*v[340:343]*/, v41 /*v297*/ offset:17472
	ds_load_b128 v[88:91] /*v[344:347]*/, v41 /*v297*/ offset:17504
	ds_load_b128 v[92:95] /*v[348:351]*/, v41 /*v297*/ offset:17536
	ds_load_b128 v[96:99] /*v[352:355]*/, v41 /*v297*/ offset:17568
	ds_load_b128 v[100:103] /*v[356:359]*/, v41 /*v297*/ offset:17600
	ds_load_b128 v[104:107] /*v[360:363]*/, v41 /*v297*/ offset:17632
	ds_load_b128 v[108:111] /*v[364:367]*/, v41 /*v297*/ offset:21760
	ds_load_b128 v[112:115] /*v[368:371]*/, v41 /*v297*/ offset:21792
	ds_load_b128 v[116:119] /*v[372:375]*/, v41 /*v297*/ offset:21824
	ds_load_b128 v[120:123] /*v[376:379]*/, v41 /*v297*/ offset:21856
	ds_load_b128 v[124:127] /*v[380:383]*/, v41 /*v297*/ offset:21888
	ds_load_b128 v[128:131] /*v[384:387]*/, v41 /*v297*/ offset:21920
	ds_load_b128 v[132:135] /*v[388:391]*/, v41 /*v297*/ offset:21952
	ds_load_b128 v[136:139] /*v[392:395]*/, v41 /*v297*/ offset:21984
	ds_load_b128 v[140:143] /*v[396:399]*/, v41 /*v297*/ offset:26112
	ds_load_b128 v[144:147] /*v[400:403]*/, v41 /*v297*/ offset:26144
	ds_load_b128 v[148:151] /*v[404:407]*/, v41 /*v297*/ offset:26176
	ds_load_b128 v[152:155] /*v[408:411]*/, v41 /*v297*/ offset:26208
	ds_load_b128 v[156:159] /*v[412:415]*/, v41 /*v297*/ offset:26240
	ds_load_b128 v[160:163] /*v[416:419]*/, v41 /*v297*/ offset:26272
	ds_load_b128 v[164:167] /*v[420:423]*/, v41 /*v297*/ offset:26304
	ds_load_b128 v[168:171] /*v[424:427]*/, v41 /*v297*/ offset:26336
	ds_load_b128 v[172:175] /*v[428:431]*/, v41 /*v297*/ offset:30464
	ds_load_b128 v[176:179] /*v[432:435]*/, v41 /*v297*/ offset:30496
	ds_load_b128 v[180:183] /*v[436:439]*/, v41 /*v297*/ offset:30528
	ds_load_b128 v[184:187] /*v[440:443]*/, v41 /*v297*/ offset:30560
	ds_load_b128 v[188:191] /*v[444:447]*/, v41 /*v297*/ offset:30592
	ds_load_b128 v[192:195] /*v[448:451]*/, v41 /*v297*/ offset:30624
	ds_load_b128 v[196:199] /*v[452:455]*/, v41 /*v297*/ offset:30656
	ds_load_b128 v[200:203] /*v[456:459]*/, v41 /*v297*/ offset:30688
	ds_load_b128 v[204:207] /*v[460:463]*/, v41 /*v297*/ offset:34816
	ds_load_b128 v[208:211] /*v[464:467]*/, v41 /*v297*/ offset:34848
	ds_load_b128 v[212:215] /*v[468:471]*/, v41 /*v297*/ offset:34880
	ds_load_b128 v[216:219] /*v[472:475]*/, v41 /*v297*/ offset:34912
	ds_load_b128 v[220:223] /*v[476:479]*/, v41 /*v297*/ offset:34944
	ds_load_b128 v[224:227] /*v[480:483]*/, v41 /*v297*/ offset:34976
	ds_load_b128 v[228:231] /*v[484:487]*/, v41 /*v297*/ offset:35008
	ds_load_b128 v[232:235] /*v[488:491]*/, v41 /*v297*/ offset:35040
	ds_load_b128 v[236:239] /*v[492:495]*/, v41 /*v297*/ offset:39168
	ds_load_b128 v[240:243] /*v[496:499]*/, v41 /*v297*/ offset:39200
	ds_load_b128 v[244:247] /*v[500:503]*/, v41 /*v297*/ offset:39232
	ds_load_b128 v[248:251] /*v[504:507]*/, v41 /*v297*/ offset:39264
	ds_load_b128 v[252:255] /*v[508:511]*/, v41 /*v297*/ offset:39296
	s_set_vgpr_msb 0x4181
	ds_load_b128 v[0:3] /*v[512:515]*/, v41 /*v297*/ offset:39328
	ds_load_b128 v[4:7] /*v[516:519]*/, v41 /*v297*/ offset:39360
	ds_load_b128 v[8:11] /*v[520:523]*/, v41 /*v297*/ offset:39392
	ds_load_b128 v[12:15] /*v[524:527]*/, v41 /*v297*/ offset:43520
	ds_load_b128 v[16:19] /*v[528:531]*/, v41 /*v297*/ offset:43552
	ds_load_b128 v[20:23] /*v[532:535]*/, v41 /*v297*/ offset:43584
	ds_load_b128 v[24:27] /*v[536:539]*/, v41 /*v297*/ offset:43616
	ds_load_b128 v[28:31] /*v[540:543]*/, v41 /*v297*/ offset:43648
	ds_load_b128 v[32:35] /*v[544:547]*/, v41 /*v297*/ offset:43680
	ds_load_b128 v[36:39] /*v[548:551]*/, v41 /*v297*/ offset:43712
	ds_load_b128 v[40:43] /*v[552:555]*/, v41 /*v297*/ offset:43744
	ds_load_b128 v[44:47] /*v[556:559]*/, v41 /*v297*/ offset:47872
	ds_load_b128 v[48:51] /*v[560:563]*/, v41 /*v297*/ offset:47904
	ds_load_b128 v[52:55] /*v[564:567]*/, v41 /*v297*/ offset:47936
	ds_load_b128 v[56:59] /*v[568:571]*/, v41 /*v297*/ offset:47968
	ds_load_b128 v[60:63] /*v[572:575]*/, v41 /*v297*/ offset:48000
	ds_load_b128 v[64:67] /*v[576:579]*/, v41 /*v297*/ offset:48032
	ds_load_b128 v[68:71] /*v[580:583]*/, v41 /*v297*/ offset:48064
	ds_load_b128 v[72:75] /*v[584:587]*/, v41 /*v297*/ offset:48096
	ds_load_b128 v[76:79] /*v[588:591]*/, v41 /*v297*/ offset:52224
	ds_load_b128 v[80:83] /*v[592:595]*/, v41 /*v297*/ offset:52256
	ds_load_b128 v[84:87] /*v[596:599]*/, v41 /*v297*/ offset:52288
	ds_load_b128 v[88:91] /*v[600:603]*/, v41 /*v297*/ offset:52320
	ds_load_b128 v[92:95] /*v[604:607]*/, v41 /*v297*/ offset:52352
	ds_load_b128 v[96:99] /*v[608:611]*/, v41 /*v297*/ offset:52384
	ds_load_b128 v[100:103] /*v[612:615]*/, v41 /*v297*/ offset:52416
	ds_load_b128 v[104:107] /*v[616:619]*/, v41 /*v297*/ offset:52448
	ds_load_b128 v[108:111] /*v[620:623]*/, v41 /*v297*/ offset:56576
	ds_load_b128 v[112:115] /*v[624:627]*/, v41 /*v297*/ offset:56608
	ds_load_b128 v[116:119] /*v[628:631]*/, v41 /*v297*/ offset:56640
	ds_load_b128 v[120:123] /*v[632:635]*/, v41 /*v297*/ offset:56672
	ds_load_b128 v[124:127] /*v[636:639]*/, v41 /*v297*/ offset:56704
	ds_load_b128 v[128:131] /*v[640:643]*/, v41 /*v297*/ offset:56736
	ds_load_b128 v[132:135] /*v[644:647]*/, v41 /*v297*/ offset:56768
	ds_load_b128 v[136:139] /*v[648:651]*/, v41 /*v297*/ offset:56800
	ds_load_b128 v[140:143] /*v[652:655]*/, v41 /*v297*/ offset:60928
	ds_load_b128 v[144:147] /*v[656:659]*/, v41 /*v297*/ offset:60960
	ds_load_b128 v[148:151] /*v[660:663]*/, v41 /*v297*/ offset:60992
	ds_load_b128 v[152:155] /*v[664:667]*/, v41 /*v297*/ offset:61024
	ds_load_b128 v[156:159] /*v[668:671]*/, v41 /*v297*/ offset:61056
	ds_load_b128 v[160:163] /*v[672:675]*/, v41 /*v297*/ offset:61088
	ds_load_b128 v[164:167] /*v[676:679]*/, v41 /*v297*/ offset:61120
	ds_load_b128 v[168:171] /*v[680:683]*/, v41 /*v297*/ offset:61152
	ds_load_b128 v[172:175] /*v[684:687]*/, v41 /*v297*/ offset:65280
	ds_load_b128 v[176:179] /*v[688:691]*/, v41 /*v297*/ offset:65312
	ds_load_b128 v[180:183] /*v[692:695]*/, v41 /*v297*/ offset:65344
	ds_load_b128 v[184:187] /*v[696:699]*/, v41 /*v297*/ offset:65376
	s_set_vgpr_msb 0x81c1
	ds_load_b128 v[120:123] /*v[888:891]*/, v41 /*v297*/ offset:65408
	ds_load_b128 v[124:127] /*v[892:895]*/, v41 /*v297*/ offset:65440
	ds_load_b128 v[128:131] /*v[896:899]*/, v41 /*v297*/ offset:65472
	ds_load_b128 v[132:135] /*v[900:903]*/, v41 /*v297*/ offset:65504
	s_set_vgpr_msb 0xc1c4
	s_wait_dscnt 0x3e
	v_wmma_f32_16x16x32_bf16 v[64:71] /*v[832:839]*/, v[192:199], v[42:49] /*v[298:305]*/, 0
	s_set_vgpr_msb 0xc480
	v_wmma_f32_16x16x32_bf16 v[192:199] /*v[704:711]*/, v[192:199], v[160:167], 0
	s_set_vgpr_msb 0x80c4
	v_wmma_f32_16x16x32_bf16 v[72:79] /*v[840:847]*/, v[224:231], v[42:49] /*v[298:305]*/, 0
	s_set_vgpr_msb 0xc480
	v_wmma_f32_16x16x32_bf16 v[200:207] /*v[712:719]*/, v[224:231], v[160:167], 0
	s_set_vgpr_msb 0x80f0
	v_wmma_f32_16x16x32_bf16 v[64:71] /*v[832:839]*/, v[200:207], v[136:143], v[64:71] /*v[832:839]*/
	s_set_vgpr_msb 0xf0a0
	v_wmma_f32_16x16x32_bf16 v[192:199] /*v[704:711]*/, v[200:207], v[168:175], v[192:199] /*v[704:711]*/
	s_set_vgpr_msb 0xa0f0
	v_wmma_f32_16x16x32_bf16 v[72:79] /*v[840:847]*/, v[232:239], v[136:143], v[72:79] /*v[840:847]*/
	s_set_vgpr_msb 0xf0a0
	v_wmma_f32_16x16x32_bf16 v[200:207] /*v[712:719]*/, v[232:239], v[168:175], v[200:207] /*v[712:719]*/
	s_set_vgpr_msb 0xa0c5
	v_wmma_f32_16x16x32_bf16 v[80:87] /*v[848:855]*/, v[0:7] /*v[256:263]*/, v[42:49] /*v[298:305]*/, 0
	s_set_vgpr_msb 0xc581
	v_wmma_f32_16x16x32_bf16 v[208:215] /*v[720:727]*/, v[0:7] /*v[256:263]*/, v[160:167], 0
	s_set_vgpr_msb 0x81f0
	v_wmma_f32_16x16x32_bf16 v[64:71] /*v[832:839]*/, v[208:215], v[144:151], v[64:71] /*v[832:839]*/
	s_set_vgpr_msb 0xf0a0
	v_wmma_f32_16x16x32_bf16 v[192:199] /*v[704:711]*/, v[208:215], v[176:183], v[192:199] /*v[704:711]*/
	s_set_vgpr_msb 0xa0f0
	v_wmma_f32_16x16x32_bf16 v[72:79] /*v[840:847]*/, v[240:247], v[144:151], v[72:79] /*v[840:847]*/
	s_set_vgpr_msb 0xf0a0
	v_wmma_f32_16x16x32_bf16 v[200:207] /*v[712:719]*/, v[240:247], v[176:183], v[200:207] /*v[712:719]*/
	s_set_vgpr_msb 0xa0f1
	v_wmma_f32_16x16x32_bf16 v[80:87] /*v[848:855]*/, v[8:15] /*v[264:271]*/, v[136:143], v[80:87] /*v[848:855]*/
	s_set_vgpr_msb 0xf1a1
	v_wmma_f32_16x16x32_bf16 v[208:215] /*v[720:727]*/, v[8:15] /*v[264:271]*/, v[168:175], v[208:215] /*v[720:727]*/
	s_set_vgpr_msb 0xa1f0
	v_wmma_f32_16x16x32_bf16 v[64:71] /*v[832:839]*/, v[216:223], v[152:159], v[64:71] /*v[832:839]*/
	s_set_vgpr_msb 0xf0a0
	v_wmma_f32_16x16x32_bf16 v[192:199] /*v[704:711]*/, v[216:223], v[184:191], v[192:199] /*v[704:711]*/
	s_set_vgpr_msb 0xa0f0
	v_wmma_f32_16x16x32_bf16 v[72:79] /*v[840:847]*/, v[248:255], v[152:159], v[72:79] /*v[840:847]*/
	s_set_vgpr_msb 0xf0a0
	v_wmma_f32_16x16x32_bf16 v[200:207] /*v[712:719]*/, v[248:255], v[184:191], v[200:207] /*v[712:719]*/
	s_set_vgpr_msb 0xa0f1
	v_wmma_f32_16x16x32_bf16 v[80:87] /*v[848:855]*/, v[16:23] /*v[272:279]*/, v[144:151], v[80:87] /*v[848:855]*/
	s_set_vgpr_msb 0xf1a1
	v_wmma_f32_16x16x32_bf16 v[208:215] /*v[720:727]*/, v[16:23] /*v[272:279]*/, v[176:183], v[208:215] /*v[720:727]*/
	s_set_vgpr_msb 0xa1c5
	v_wmma_f32_16x16x32_bf16 v[88:95] /*v[856:863]*/, v[32:39] /*v[288:295]*/, v[42:49] /*v[298:305]*/, 0
	v_wmma_f32_16x16x32_bf16 v[96:103] /*v[864:871]*/, v[76:83] /*v[332:339]*/, v[42:49] /*v[298:305]*/, 0
	v_wmma_f32_16x16x32_bf16 v[104:111] /*v[872:879]*/, v[108:115] /*v[364:371]*/, v[42:49] /*v[298:305]*/, 0
	v_wmma_f32_16x16x32_bf16 v[112:119] /*v[880:887]*/, v[140:147] /*v[396:403]*/, v[42:49] /*v[298:305]*/, 0
	s_set_vgpr_msb 0xc505
	v_wmma_f32_16x16x32_bf16 v[192:199], v[172:179] /*v[428:435]*/, v[42:49] /*v[298:305]*/, 0
	v_wmma_f32_16x16x32_bf16 v[204:211], v[204:211] /*v[460:467]*/, v[42:49] /*v[298:305]*/, 0
	s_wait_dscnt 0x36
	v_wmma_f32_16x16x32_bf16 v[212:219], v[236:243] /*v[492:499]*/, v[42:49] /*v[298:305]*/, 0
	s_set_vgpr_msb 0x506
	s_wait_dscnt 0x2e
	v_wmma_f32_16x16x32_bf16 v[222:229], v[12:19] /*v[524:531]*/, v[42:49] /*v[298:305]*/, 0
	s_wait_dscnt 0x26
	v_wmma_f32_16x16x32_bf16 v[230:237], v[44:51] /*v[556:563]*/, v[42:49] /*v[298:305]*/, 0
	s_wait_dscnt 0x1e
	v_wmma_f32_16x16x32_bf16 v[238:245], v[76:83] /*v[588:595]*/, v[42:49] /*v[298:305]*/, 0
	s_wait_dscnt 0x16
	v_wmma_f32_16x16x32_bf16 v[246:253], v[108:115] /*v[620:627]*/, v[42:49] /*v[298:305]*/, 0
	s_set_vgpr_msb 0x6f1
	v_wmma_f32_16x16x32_bf16 v[80:87] /*v[848:855]*/, v[24:31] /*v[280:287]*/, v[152:159], v[80:87] /*v[848:855]*/
	s_set_vgpr_msb 0xf1a1
	v_wmma_f32_16x16x32_bf16 v[208:215] /*v[720:727]*/, v[24:31] /*v[280:287]*/, v[184:191], v[208:215] /*v[720:727]*/
	v_wmma_f32_16x16x32_bf16 v[216:223] /*v[728:735]*/, v[32:39] /*v[288:295]*/, v[160:167], 0
	s_set_vgpr_msb 0xa1f1
	v_wmma_f32_16x16x32_bf16 v[88:95] /*v[856:863]*/, v[52:59] /*v[308:315]*/, v[136:143], v[88:95] /*v[856:863]*/
	s_set_vgpr_msb 0xf181
	v_wmma_f32_16x16x32_bf16 v[224:231] /*v[736:743]*/, v[76:83] /*v[332:339]*/, v[160:167], 0
	s_set_vgpr_msb 0x81f1
	v_wmma_f32_16x16x32_bf16 v[96:103] /*v[864:871]*/, v[84:91] /*v[340:347]*/, v[136:143], v[96:103] /*v[864:871]*/
	s_set_vgpr_msb 0xf181
	v_wmma_f32_16x16x32_bf16 v[232:239] /*v[744:751]*/, v[108:115] /*v[364:371]*/, v[160:167], 0
	s_set_vgpr_msb 0x81f1
	v_wmma_f32_16x16x32_bf16 v[104:111] /*v[872:879]*/, v[116:123] /*v[372:379]*/, v[136:143], v[104:111] /*v[872:879]*/
	s_set_vgpr_msb 0xf181
	v_wmma_f32_16x16x32_bf16 v[240:247] /*v[752:759]*/, v[140:147] /*v[396:403]*/, v[160:167], 0
	s_set_vgpr_msb 0x81f1
	v_wmma_f32_16x16x32_bf16 v[112:119] /*v[880:887]*/, v[148:155] /*v[404:411]*/, v[136:143], v[112:119] /*v[880:887]*/
	s_set_vgpr_msb 0xf181
	v_wmma_f32_16x16x32_bf16 v[248:255] /*v[760:767]*/, v[172:179] /*v[428:435]*/, v[160:167], 0
	s_set_vgpr_msb 0x8101
	v_wmma_f32_16x16x32_bf16 v[192:199], v[180:187] /*v[436:443]*/, v[136:143], v[192:199]
	s_set_vgpr_msb 0x1c1
	v_wmma_f32_16x16x32_bf16 v[0:7] /*v[768:775]*/, v[204:211] /*v[460:467]*/, v[160:167], 0
	s_set_vgpr_msb 0xc101
	v_wmma_f32_16x16x32_bf16 v[204:211], v[212:219] /*v[468:475]*/, v[136:143], v[204:211]
	s_set_vgpr_msb 0x1c1
	v_wmma_f32_16x16x32_bf16 v[8:15] /*v[776:783]*/, v[236:243] /*v[492:499]*/, v[160:167], 0
	s_set_vgpr_msb 0xc101
	v_wmma_f32_16x16x32_bf16 v[212:219], v[244:251] /*v[500:507]*/, v[136:143], v[212:219]
	s_set_vgpr_msb 0x1c2
	v_wmma_f32_16x16x32_bf16 v[16:23] /*v[784:791]*/, v[12:19] /*v[524:531]*/, v[160:167], 0
	s_set_vgpr_msb 0xc202
	v_wmma_f32_16x16x32_bf16 v[222:229], v[20:27] /*v[532:539]*/, v[136:143], v[222:229]
	s_set_vgpr_msb 0x2c2
	v_wmma_f32_16x16x32_bf16 v[24:31] /*v[792:799]*/, v[44:51] /*v[556:563]*/, v[160:167], 0
	s_set_vgpr_msb 0xc202
	v_wmma_f32_16x16x32_bf16 v[230:237], v[52:59] /*v[564:571]*/, v[136:143], v[230:237]
	s_set_vgpr_msb 0x2c2
	v_wmma_f32_16x16x32_bf16 v[32:39] /*v[800:807]*/, v[76:83] /*v[588:595]*/, v[160:167], 0
	s_set_vgpr_msb 0xc202
	v_wmma_f32_16x16x32_bf16 v[238:245], v[84:91] /*v[596:603]*/, v[136:143], v[238:245]
	s_set_vgpr_msb 0x2c2
	v_wmma_f32_16x16x32_bf16 v[40:47] /*v[808:815]*/, v[108:115] /*v[620:627]*/, v[160:167], 0
	s_set_vgpr_msb 0xc202
	s_wait_dscnt 0x14
	v_wmma_f32_16x16x32_bf16 v[246:253], v[116:123] /*v[628:635]*/, v[136:143], v[246:253]
	s_set_vgpr_msb 0x246
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x32_bf16 v[16:23] /*v[272:279]*/, v[140:147] /*v[652:659]*/, v[42:49] /*v[298:305]*/, 0
	s_set_vgpr_msb 0x46c2
	v_wmma_f32_16x16x32_bf16 v[48:55] /*v[816:823]*/, v[140:147] /*v[652:659]*/, v[160:167], 0
	s_set_vgpr_msb 0xc246
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x32_bf16 v[24:31] /*v[280:287]*/, v[172:179] /*v[684:691]*/, v[42:49] /*v[298:305]*/, 0
	s_set_vgpr_msb 0x46c2
	v_wmma_f32_16x16x32_bf16 v[56:63] /*v[824:831]*/, v[172:179] /*v[684:691]*/, v[160:167], 0
	s_set_vgpr_msb 0xc2a1
	v_wmma_f32_16x16x32_bf16 v[216:223] /*v[728:735]*/, v[52:59] /*v[308:315]*/, v[168:175], v[216:223] /*v[728:735]*/
	s_set_vgpr_msb 0xa1f1
	v_wmma_f32_16x16x32_bf16 v[88:95] /*v[856:863]*/, v[60:67] /*v[316:323]*/, v[144:151], v[88:95] /*v[856:863]*/
	s_set_vgpr_msb 0xf1a1
	v_wmma_f32_16x16x32_bf16 v[224:231] /*v[736:743]*/, v[84:91] /*v[340:347]*/, v[168:175], v[224:231] /*v[736:743]*/
	s_set_vgpr_msb 0xa1f1
	v_wmma_f32_16x16x32_bf16 v[96:103] /*v[864:871]*/, v[92:99] /*v[348:355]*/, v[144:151], v[96:103] /*v[864:871]*/
	s_set_vgpr_msb 0xf1a1
	v_wmma_f32_16x16x32_bf16 v[232:239] /*v[744:751]*/, v[116:123] /*v[372:379]*/, v[168:175], v[232:239] /*v[744:751]*/
	s_set_vgpr_msb 0xa1f1
	v_wmma_f32_16x16x32_bf16 v[104:111] /*v[872:879]*/, v[124:131] /*v[380:387]*/, v[144:151], v[104:111] /*v[872:879]*/
	s_set_vgpr_msb 0xf1a1
	v_wmma_f32_16x16x32_bf16 v[240:247] /*v[752:759]*/, v[148:155] /*v[404:411]*/, v[168:175], v[240:247] /*v[752:759]*/
	s_set_vgpr_msb 0xa1f1
	v_wmma_f32_16x16x32_bf16 v[112:119] /*v[880:887]*/, v[156:163] /*v[412:419]*/, v[144:151], v[112:119] /*v[880:887]*/
	s_set_vgpr_msb 0xf1a1
	v_wmma_f32_16x16x32_bf16 v[248:255] /*v[760:767]*/, v[180:187] /*v[436:443]*/, v[168:175], v[248:255] /*v[760:767]*/
	s_set_vgpr_msb 0xa101
	v_wmma_f32_16x16x32_bf16 v[192:199], v[188:195] /*v[444:451]*/, v[144:151], v[192:199]
	s_set_vgpr_msb 0x1f1
	v_wmma_f32_16x16x32_bf16 v[0:7] /*v[768:775]*/, v[212:219] /*v[468:475]*/, v[168:175], v[0:7] /*v[768:775]*/
	s_set_vgpr_msb 0xf101
	v_wmma_f32_16x16x32_bf16 v[204:211], v[220:227] /*v[476:483]*/, v[144:151], v[204:211]
	s_set_vgpr_msb 0x1f1
	v_wmma_f32_16x16x32_bf16 v[8:15] /*v[776:783]*/, v[244:251] /*v[500:507]*/, v[168:175], v[8:15] /*v[776:783]*/
	s_set_vgpr_msb 0xf101
	v_wmma_f32_16x16x32_bf16 v[212:219], v[252:259] /*v[508:515]*/, v[144:151], v[212:219]
	s_set_vgpr_msb 0x1f2
	v_wmma_f32_16x16x32_bf16 v[16:23] /*v[784:791]*/, v[20:27] /*v[532:539]*/, v[168:175], v[16:23] /*v[784:791]*/
	s_set_vgpr_msb 0xf202
	v_wmma_f32_16x16x32_bf16 v[222:229], v[28:35] /*v[540:547]*/, v[144:151], v[222:229]
	s_set_vgpr_msb 0x2f2
	v_wmma_f32_16x16x32_bf16 v[24:31] /*v[792:799]*/, v[52:59] /*v[564:571]*/, v[168:175], v[24:31] /*v[792:799]*/
	s_set_vgpr_msb 0xf202
	v_wmma_f32_16x16x32_bf16 v[230:237], v[60:67] /*v[572:579]*/, v[144:151], v[230:237]
	s_set_vgpr_msb 0x2f2
	v_wmma_f32_16x16x32_bf16 v[32:39] /*v[800:807]*/, v[84:91] /*v[596:603]*/, v[168:175], v[32:39] /*v[800:807]*/
	s_set_vgpr_msb 0xf202
	v_wmma_f32_16x16x32_bf16 v[238:245], v[92:99] /*v[604:611]*/, v[144:151], v[238:245]
	s_set_vgpr_msb 0x2f2
	v_wmma_f32_16x16x32_bf16 v[40:47] /*v[808:815]*/, v[116:123] /*v[628:635]*/, v[168:175], v[40:47] /*v[808:815]*/
	s_set_vgpr_msb 0xf202
	v_wmma_f32_16x16x32_bf16 v[246:253], v[124:131] /*v[636:643]*/, v[144:151], v[246:253]
	s_set_vgpr_msb 0x252
	v_wmma_f32_16x16x32_bf16 v[16:23] /*v[272:279]*/, v[148:155] /*v[660:667]*/, v[136:143], v[16:23] /*v[272:279]*/
	s_set_vgpr_msb 0x52f2
	v_wmma_f32_16x16x32_bf16 v[48:55] /*v[816:823]*/, v[148:155] /*v[660:667]*/, v[168:175], v[48:55] /*v[816:823]*/
	s_set_vgpr_msb 0xf252
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x32_bf16 v[24:31] /*v[280:287]*/, v[180:187] /*v[692:699]*/, v[136:143], v[24:31] /*v[280:287]*/
	s_set_vgpr_msb 0x52f2
	v_wmma_f32_16x16x32_bf16 v[56:63] /*v[824:831]*/, v[180:187] /*v[692:699]*/, v[168:175], v[56:63] /*v[824:831]*/
	s_set_vgpr_msb 0xf2a1
	v_wmma_f32_16x16x32_bf16 v[216:223] /*v[728:735]*/, v[60:67] /*v[316:323]*/, v[176:183], v[216:223] /*v[728:735]*/
	s_set_vgpr_msb 0xa1f1
	v_wmma_f32_16x16x32_bf16 v[88:95] /*v[856:863]*/, v[68:75] /*v[324:331]*/, v[152:159], v[88:95] /*v[856:863]*/
	s_set_vgpr_msb 0xf1a1
	v_wmma_f32_16x16x32_bf16 v[224:231] /*v[736:743]*/, v[92:99] /*v[348:355]*/, v[176:183], v[224:231] /*v[736:743]*/
	s_set_vgpr_msb 0xa1f1
	v_wmma_f32_16x16x32_bf16 v[96:103] /*v[864:871]*/, v[100:107] /*v[356:363]*/, v[152:159], v[96:103] /*v[864:871]*/
	s_set_vgpr_msb 0xf1a1
	v_wmma_f32_16x16x32_bf16 v[232:239] /*v[744:751]*/, v[124:131] /*v[380:387]*/, v[176:183], v[232:239] /*v[744:751]*/
	s_set_vgpr_msb 0xa1f1
	v_wmma_f32_16x16x32_bf16 v[104:111] /*v[872:879]*/, v[132:139] /*v[388:395]*/, v[152:159], v[104:111] /*v[872:879]*/
	s_set_vgpr_msb 0xf1a1
	v_wmma_f32_16x16x32_bf16 v[240:247] /*v[752:759]*/, v[156:163] /*v[412:419]*/, v[176:183], v[240:247] /*v[752:759]*/
	s_set_vgpr_msb 0xa1f1
	v_wmma_f32_16x16x32_bf16 v[112:119] /*v[880:887]*/, v[164:171] /*v[420:427]*/, v[152:159], v[112:119] /*v[880:887]*/
	s_set_vgpr_msb 0xf1a1
	v_wmma_f32_16x16x32_bf16 v[248:255] /*v[760:767]*/, v[188:195] /*v[444:451]*/, v[176:183], v[248:255] /*v[760:767]*/
	s_set_vgpr_msb 0xa101
	v_wmma_f32_16x16x32_bf16 v[192:199], v[196:203] /*v[452:459]*/, v[152:159], v[192:199]
	s_set_vgpr_msb 0x1f1
	v_wmma_f32_16x16x32_bf16 v[0:7] /*v[768:775]*/, v[220:227] /*v[476:483]*/, v[176:183], v[0:7] /*v[768:775]*/
	s_set_vgpr_msb 0xf101
	v_wmma_f32_16x16x32_bf16 v[204:211], v[228:235] /*v[484:491]*/, v[152:159], v[204:211]
	s_set_vgpr_msb 0x1f1
	v_wmma_f32_16x16x32_bf16 v[8:15] /*v[776:783]*/, v[252:259] /*v[508:515]*/, v[176:183], v[8:15] /*v[776:783]*/
	s_set_vgpr_msb 0xf102
	v_wmma_f32_16x16x32_bf16 v[212:219], v[4:11] /*v[516:523]*/, v[152:159], v[212:219]
	s_set_vgpr_msb 0x2f2
	v_wmma_f32_16x16x32_bf16 v[16:23] /*v[784:791]*/, v[28:35] /*v[540:547]*/, v[176:183], v[16:23] /*v[784:791]*/
	s_set_vgpr_msb 0xf202
	v_wmma_f32_16x16x32_bf16 v[222:229], v[36:43] /*v[548:555]*/, v[152:159], v[222:229]
	s_set_vgpr_msb 0x2f2
	v_wmma_f32_16x16x32_bf16 v[24:31] /*v[792:799]*/, v[60:67] /*v[572:579]*/, v[176:183], v[24:31] /*v[792:799]*/
	s_set_vgpr_msb 0xf202
	v_wmma_f32_16x16x32_bf16 v[230:237], v[68:75] /*v[580:587]*/, v[152:159], v[230:237]
	s_set_vgpr_msb 0x2f2
	v_wmma_f32_16x16x32_bf16 v[32:39] /*v[800:807]*/, v[92:99] /*v[604:611]*/, v[176:183], v[32:39] /*v[800:807]*/
	s_set_vgpr_msb 0xf202
	v_wmma_f32_16x16x32_bf16 v[238:245], v[100:107] /*v[612:619]*/, v[152:159], v[238:245]
	s_set_vgpr_msb 0x2f2
	v_wmma_f32_16x16x32_bf16 v[40:47] /*v[808:815]*/, v[124:131] /*v[636:643]*/, v[176:183], v[40:47] /*v[808:815]*/
	s_set_vgpr_msb 0xf202
	v_wmma_f32_16x16x32_bf16 v[246:253], v[132:139] /*v[644:651]*/, v[152:159], v[246:253]
	s_set_vgpr_msb 0x252
	v_wmma_f32_16x16x32_bf16 v[16:23] /*v[272:279]*/, v[156:163] /*v[668:675]*/, v[144:151], v[16:23] /*v[272:279]*/
	s_set_vgpr_msb 0x52f2
	v_wmma_f32_16x16x32_bf16 v[48:55] /*v[816:823]*/, v[156:163] /*v[668:675]*/, v[176:183], v[48:55] /*v[816:823]*/
	s_set_vgpr_msb 0xf253
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[24:31] /*v[280:287]*/, v[120:127] /*v[888:895]*/, v[144:151], v[24:31] /*v[280:287]*/
	s_set_vgpr_msb 0x53f3
	v_wmma_f32_16x16x32_bf16 v[56:63] /*v[824:831]*/, v[120:127] /*v[888:895]*/, v[176:183], v[56:63] /*v[824:831]*/
	s_set_vgpr_msb 0xf3a1
	v_wmma_f32_16x16x32_bf16 v[216:223] /*v[728:735]*/, v[68:75] /*v[324:331]*/, v[184:191], v[216:223] /*v[728:735]*/
	v_wmma_f32_16x16x32_bf16 v[224:231] /*v[736:743]*/, v[100:107] /*v[356:363]*/, v[184:191], v[224:231] /*v[736:743]*/
	v_wmma_f32_16x16x32_bf16 v[232:239] /*v[744:751]*/, v[132:139] /*v[388:395]*/, v[184:191], v[232:239] /*v[744:751]*/
	v_wmma_f32_16x16x32_bf16 v[240:247] /*v[752:759]*/, v[164:171] /*v[420:427]*/, v[184:191], v[240:247] /*v[752:759]*/
	v_wmma_f32_16x16x32_bf16 v[248:255] /*v[760:767]*/, v[196:203] /*v[452:459]*/, v[184:191], v[248:255] /*v[760:767]*/
	s_set_vgpr_msb 0xa1f1
	v_wmma_f32_16x16x32_bf16 v[0:7] /*v[768:775]*/, v[228:235] /*v[484:491]*/, v[184:191], v[0:7] /*v[768:775]*/
	s_set_vgpr_msb 0xf1f2
	v_wmma_f32_16x16x32_bf16 v[8:15] /*v[776:783]*/, v[4:11] /*v[516:523]*/, v[184:191], v[8:15] /*v[776:783]*/
	v_wmma_f32_16x16x32_bf16 v[16:23] /*v[784:791]*/, v[36:43] /*v[548:555]*/, v[184:191], v[16:23] /*v[784:791]*/
	v_wmma_f32_16x16x32_bf16 v[24:31] /*v[792:799]*/, v[68:75] /*v[580:587]*/, v[184:191], v[24:31] /*v[792:799]*/
	v_wmma_f32_16x16x32_bf16 v[32:39] /*v[800:807]*/, v[100:107] /*v[612:619]*/, v[184:191], v[32:39] /*v[800:807]*/
	v_wmma_f32_16x16x32_bf16 v[40:47] /*v[808:815]*/, v[132:139] /*v[644:651]*/, v[184:191], v[40:47] /*v[808:815]*/
	s_set_vgpr_msb 0xf252
	v_wmma_f32_16x16x32_bf16 v[16:23] /*v[272:279]*/, v[164:171] /*v[676:683]*/, v[152:159], v[16:23] /*v[272:279]*/
	s_set_vgpr_msb 0x52f2
	v_wmma_f32_16x16x32_bf16 v[48:55] /*v[816:823]*/, v[164:171] /*v[676:683]*/, v[184:191], v[48:55] /*v[816:823]*/
	s_set_vgpr_msb 0xf253
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[24:31] /*v[280:287]*/, v[128:135] /*v[896:903]*/, v[152:159], v[24:31] /*v[280:287]*/
	s_set_vgpr_msb 0x53f3
	v_wmma_f32_16x16x32_bf16 v[56:63] /*v[824:831]*/, v[128:135] /*v[896:903]*/, v[184:191], v[56:63] /*v[824:831]*/
	s_wait_alu depctr_va_vdst(14)
	s_set_vgpr_msb 0xf341
	ds_load_tr16_b128 v[72:75] /*v[328:331]*/, v40 /*v296*/ offset:192
	s_set_vgpr_msb 0x4101
	ds_load_tr16_b128 v[254:257], v40 /*v296*/ offset:224
	s_set_vgpr_msb 0x141
	ds_load_tr16_b128 v[76:79] /*v[332:335]*/, v40 /*v296*/ offset:4800
	ds_load_tr16_b128 v[2:5] /*v[258:261]*/, v40 /*v296*/ offset:4832
	s_set_vgpr_msb 0x4181
	ds_load_tr16_b128 v[188:191] /*v[700:703]*/, v40 /*v296*/ offset:4608
	ds_load_tr16_b128 v[108:111] /*v[620:623]*/, v40 /*v296*/ offset:4640
	ds_load_tr16_b128 v[176:179] /*v[688:691]*/, v40 /*v296*/ offset:9216
	ds_load_tr16_b128 v[120:123] /*v[632:635]*/, v40 /*v296*/ offset:9248
	ds_load_tr16_b128 v[180:183] /*v[692:695]*/, v40 /*v296*/ offset:13824
	ds_load_tr16_b128 v[124:127] /*v[636:639]*/, v40 /*v296*/ offset:13856
	s_wait_alu depctr_va_vdst(2)
	ds_load_tr16_b128 v[168:171] /*v[680:683]*/, v40 /*v296*/ offset:18432
	ds_load_tr16_b128 v[112:115] /*v[624:627]*/, v40 /*v296*/ offset:18464
	ds_load_tr16_b128 v[172:175] /*v[684:687]*/, v40 /*v296*/ offset:23040
	ds_load_tr16_b128 v[116:119] /*v[628:631]*/, v40 /*v296*/ offset:23072
	ds_load_tr16_b128 v[160:163] /*v[672:675]*/, v40 /*v296*/ offset:27648
	ds_load_tr16_b128 v[96:99] /*v[608:611]*/, v40 /*v296*/ offset:27680
	ds_load_tr16_b128 v[164:167] /*v[676:679]*/, v40 /*v296*/ offset:32256
	ds_load_tr16_b128 v[100:103] /*v[612:615]*/, v40 /*v296*/ offset:32288
	ds_load_tr16_b128 v[152:155] /*v[664:667]*/, v40 /*v296*/ offset:36864
	ds_load_tr16_b128 v[88:91] /*v[600:603]*/, v40 /*v296*/ offset:36896
	ds_load_tr16_b128 v[156:159] /*v[668:671]*/, v40 /*v296*/ offset:41472
	ds_load_tr16_b128 v[92:95] /*v[604:607]*/, v40 /*v296*/ offset:41504
	ds_load_tr16_b128 v[144:147] /*v[656:659]*/, v40 /*v296*/ offset:46080
	ds_load_tr16_b128 v[80:83] /*v[592:595]*/, v40 /*v296*/ offset:46112
	ds_load_tr16_b128 v[148:151] /*v[660:663]*/, v40 /*v296*/ offset:50688
	ds_load_tr16_b128 v[84:87] /*v[596:599]*/, v40 /*v296*/ offset:50720
	ds_load_tr16_b128 v[136:139] /*v[648:651]*/, v40 /*v296*/ offset:55296
	ds_load_tr16_b128 v[64:67] /*v[576:579]*/, v40 /*v296*/ offset:55328
	ds_load_tr16_b128 v[72:75] /*v[584:587]*/, v40 /*v296*/ offset:64
	s_set_vgpr_msb 0x8141
	ds_load_tr16_b128 v[248:251] /*v[504:507]*/, v40 /*v296*/ offset:96
	s_set_vgpr_msb 0x4181
	ds_load_tr16_b128 v[76:79] /*v[588:591]*/, v40 /*v296*/ offset:4672
	s_set_vgpr_msb 0x8141
	ds_load_tr16_b128 v[252:255] /*v[508:511]*/, v40 /*v296*/ offset:4704
	s_set_vgpr_msb 0x4181
	ds_load_tr16_b128 v[56:59] /*v[568:571]*/, v40 /*v296*/ offset:9280
	s_set_vgpr_msb 0x8141
	ds_load_tr16_b128 v[240:243] /*v[496:499]*/, v40 /*v296*/ offset:9312
	s_set_vgpr_msb 0x4181
	ds_load_tr16_b128 v[60:63] /*v[572:575]*/, v40 /*v296*/ offset:13888
	s_set_vgpr_msb 0x8141
	ds_load_tr16_b128 v[244:247] /*v[500:503]*/, v40 /*v296*/ offset:13920
	s_set_vgpr_msb 0x4181
	ds_load_tr16_b128 v[40:43] /*v[552:555]*/, v40 /*v296*/ offset:18496
	s_set_vgpr_msb 0x8141
	ds_load_tr16_b128 v[232:235] /*v[488:491]*/, v40 /*v296*/ offset:18528
	s_set_vgpr_msb 0x4181
	ds_load_tr16_b128 v[44:47] /*v[556:559]*/, v40 /*v296*/ offset:23104
	s_set_vgpr_msb 0x8141
	ds_load_tr16_b128 v[236:239] /*v[492:495]*/, v40 /*v296*/ offset:23136
	s_set_vgpr_msb 0x4181
	ds_load_tr16_b128 v[32:35] /*v[544:547]*/, v40 /*v296*/ offset:27712
	s_set_vgpr_msb 0x8141
	ds_load_tr16_b128 v[224:227] /*v[480:483]*/, v40 /*v296*/ offset:27744
	s_set_vgpr_msb 0x4181
	ds_load_tr16_b128 v[36:39] /*v[548:551]*/, v40 /*v296*/ offset:32320
	s_set_vgpr_msb 0x8141
	ds_load_tr16_b128 v[228:231] /*v[484:487]*/, v40 /*v296*/ offset:32352
	s_set_vgpr_msb 0x4181
	ds_load_tr16_b128 v[24:27] /*v[536:539]*/, v40 /*v296*/ offset:36928
	s_set_vgpr_msb 0x8141
	ds_load_tr16_b128 v[216:219] /*v[472:475]*/, v40 /*v296*/ offset:36960
	s_set_vgpr_msb 0x4181
	ds_load_tr16_b128 v[28:31] /*v[540:543]*/, v40 /*v296*/ offset:41536
	s_set_vgpr_msb 0x8141
	ds_load_tr16_b128 v[220:223] /*v[476:479]*/, v40 /*v296*/ offset:41568
	s_set_vgpr_msb 0x4181
	ds_load_tr16_b128 v[16:19] /*v[528:531]*/, v40 /*v296*/ offset:46144
	s_set_vgpr_msb 0x8141
	ds_load_tr16_b128 v[208:211] /*v[464:467]*/, v40 /*v296*/ offset:46176
	s_set_vgpr_msb 0x4181
	ds_load_tr16_b128 v[20:23] /*v[532:535]*/, v40 /*v296*/ offset:50752
	s_set_vgpr_msb 0x8141
	ds_load_tr16_b128 v[212:215] /*v[468:471]*/, v40 /*v296*/ offset:50784
	s_set_vgpr_msb 0x4181
	ds_load_tr16_b128 v[8:11] /*v[520:523]*/, v40 /*v296*/ offset:55360
	s_set_vgpr_msb 0x8141
	ds_load_tr16_b128 v[192:195] /*v[448:451]*/, v40 /*v296*/ offset:55392
	ds_load_tr16_b128 v[200:203] /*v[456:459]*/, v40 /*v296*/ offset:128
	ds_load_tr16_b128 v[120:123] /*v[376:379]*/, v40 /*v296*/ offset:160
	ds_load_tr16_b128 v[204:207] /*v[460:463]*/, v40 /*v296*/ offset:4736
	ds_load_tr16_b128 v[124:127] /*v[380:383]*/, v40 /*v296*/ offset:4768
	ds_load_tr16_b128 v[184:187] /*v[440:443]*/, v40 /*v296*/ offset:9344
	ds_load_tr16_b128 v[112:115] /*v[368:371]*/, v40 /*v296*/ offset:9376
	ds_load_tr16_b128 v[188:191] /*v[444:447]*/, v40 /*v296*/ offset:13952
	ds_load_tr16_b128 v[116:119] /*v[372:375]*/, v40 /*v296*/ offset:13984
	ds_load_tr16_b128 v[168:171] /*v[424:427]*/, v40 /*v296*/ offset:18560
	ds_load_tr16_b128 v[104:107] /*v[360:363]*/, v40 /*v296*/ offset:18592
	ds_load_tr16_b128 v[172:175] /*v[428:431]*/, v40 /*v296*/ offset:23168
	ds_load_tr16_b128 v[108:111] /*v[364:367]*/, v40 /*v296*/ offset:23200
	ds_load_tr16_b128 v[160:163] /*v[416:419]*/, v40 /*v296*/ offset:27776
	ds_load_tr16_b128 v[96:99] /*v[352:355]*/, v40 /*v296*/ offset:27808
	ds_load_tr16_b128 v[164:167] /*v[420:423]*/, v40 /*v296*/ offset:32384
	ds_load_tr16_b128 v[100:103] /*v[356:359]*/, v40 /*v296*/ offset:32416
	ds_load_tr16_b128 v[152:155] /*v[408:411]*/, v40 /*v296*/ offset:36992
	ds_load_tr16_b128 v[88:91] /*v[344:347]*/, v40 /*v296*/ offset:37024
	ds_load_tr16_b128 v[156:159] /*v[412:415]*/, v40 /*v296*/ offset:41600
	ds_load_tr16_b128 v[92:95] /*v[348:351]*/, v40 /*v296*/ offset:41632
	ds_load_tr16_b128 v[144:147] /*v[400:403]*/, v40 /*v296*/ offset:46208
	ds_load_tr16_b128 v[80:83] /*v[336:339]*/, v40 /*v296*/ offset:46240
	ds_load_tr16_b128 v[148:151] /*v[404:407]*/, v40 /*v296*/ offset:50816
	ds_load_tr16_b128 v[84:87] /*v[340:343]*/, v40 /*v296*/ offset:50848
	ds_load_tr16_b128 v[136:139] /*v[392:395]*/, v40 /*v296*/ offset:55424
	ds_load_tr16_b128 v[64:67] /*v[320:323]*/, v40 /*v296*/ offset:55456
	s_set_vgpr_msb 0x4104
	v_add_nc_u32_e32 v128, 0x10e00, v40 /*v296*/
	v_add_nc_u32_e32 v130, 0x10e20, v40 /*v296*/
	s_set_vgpr_msb 0x481
	ds_load_tr16_b128 v[140:143] /*v[652:655]*/, v40 /*v296*/ offset:59904
	ds_load_tr16_b128 v[68:71] /*v[580:583]*/, v40 /*v296*/ offset:59936
	ds_load_tr16_b128 v[128:131] /*v[640:643]*/, v40 /*v296*/ offset:64512
	ds_load_tr16_b128 v[48:51] /*v[560:563]*/, v40 /*v296*/ offset:64544
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x8180
	ds_load_tr16_b128 v[132:135] /*v[644:647]*/, v128
	ds_load_tr16_b128 v[52:55] /*v[564:567]*/, v130
	s_wait_alu depctr_vm_vsrc(1)
	s_set_vgpr_msb 0x8004
	v_add_nc_u32_e32 v128, 0x10e40, v40 /*v296*/
	s_wait_alu depctr_vm_vsrc(0)
	v_add_nc_u32_e32 v130, 0x10e60, v40 /*v296*/
	s_set_vgpr_msb 0x481
	ds_load_tr16_b128 v[12:15] /*v[524:527]*/, v40 /*v296*/ offset:59968
	s_set_vgpr_msb 0x8141
	ds_load_tr16_b128 v[196:199] /*v[452:455]*/, v40 /*v296*/ offset:60000
	s_set_vgpr_msb 0x4181
	ds_load_tr16_b128 v[0:3] /*v[512:515]*/, v40 /*v296*/ offset:64576
	s_set_vgpr_msb 0x8141
	ds_load_tr16_b128 v[176:179] /*v[432:435]*/, v40 /*v296*/ offset:64608
	s_wait_alu depctr_va_vdst(1)
	s_set_vgpr_msb 0x4180
	ds_load_tr16_b128 v[4:7] /*v[516:519]*/, v128
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x8040
	ds_load_tr16_b128 v[180:183] /*v[436:439]*/, v130
	s_wait_alu depctr_vm_vsrc(1)
	s_set_vgpr_msb 0x4004
	v_add_nc_u32_e32 v128, 0x10e80, v40 /*v296*/
	s_wait_alu depctr_vm_vsrc(0)
	v_add_nc_u32_e32 v130, 0x10ea0, v40 /*v296*/
	s_set_vgpr_msb 0x441
	ds_load_tr16_b128 v[140:143] /*v[396:399]*/, v40 /*v296*/ offset:60032
	ds_load_tr16_b128 v[68:71] /*v[324:327]*/, v40 /*v296*/ offset:60064
	ds_load_tr16_b128 v[128:131] /*v[384:387]*/, v40 /*v296*/ offset:64640
	ds_load_tr16_b128 v[8:11] /*v[264:267]*/, v40 /*v296*/ offset:64672
	s_wait_alu depctr_va_vdst(1)
	s_set_vgpr_msb 0x4140
	ds_load_tr16_b128 v[132:135] /*v[388:391]*/, v128
	s_wait_alu depctr_va_vdst(0)
	ds_load_tr16_b128 v[12:15] /*v[268:271]*/, v130
	s_wait_alu depctr_vm_vsrc(1)
	s_set_vgpr_msb 0x4004
	v_add_nc_u32_e32 v128, 0x10ec0, v40 /*v296*/
	s_wait_alu depctr_vm_vsrc(0)
	v_add_nc_u32_e32 v130, 0x10ee0, v40 /*v296*/
	s_set_vgpr_msb 0x481
	ds_load_tr16_b128 v[184:187] /*v[696:699]*/, v40 /*v296*/
	ds_load_tr16_b128 v[104:107] /*v[616:619]*/, v40 /*v296*/ offset:32
	s_wait_dscnt 0x3e
	s_clause 0x2
	scratch_store_b128 off, v[254:257], off offset:236 nv
	s_set_vgpr_msb 0x8145
	scratch_store_b128 off, v[2:5] /*v[258:261]*/, off offset:252 nv
	ds_load_tr16_b128 v[56:59] /*v[312:315]*/, v40 /*v296*/ offset:9408
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	s_set_vgpr_msb 0x4501
	ds_load_tr16_b128 v[254:257], v40 /*v296*/ offset:9440
	s_set_vgpr_msb 0x141
	ds_load_tr16_b128 v[60:63] /*v[316:319]*/, v40 /*v296*/ offset:14016
	ds_load_tr16_b128 v[2:5] /*v[258:261]*/, v40 /*v296*/ offset:14048
	s_wait_dscnt 0x2
	scratch_store_b128 off, v[254:257], off offset:204 nv
	s_set_vgpr_msb 0x4145
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[2:5] /*v[258:261]*/, off offset:220 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_tr16_b128 v[0:3] /*v[256:259]*/, v40 /*v296*/ offset:18624
	ds_load_tr16_b128 v[32:35] /*v[288:291]*/, v40 /*v296*/ offset:18656
	ds_load_tr16_b128 v[4:7] /*v[260:263]*/, v40 /*v296*/ offset:23232
	ds_load_tr16_b128 v[36:39] /*v[292:295]*/, v40 /*v296*/ offset:23264
	s_wait_dscnt 0x2
	scratch_store_b128 off, v[32:35] /*v[288:291]*/, off offset:172 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[36:39] /*v[292:295]*/, off offset:188 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_tr16_b128 v[32:35] /*v[288:291]*/, v40 /*v296*/ offset:27840
	s_set_vgpr_msb 0x45c1
	ds_load_tr16_b128 v[120:123] /*v[888:891]*/, v40 /*v296*/ offset:27872
	s_set_vgpr_msb 0xc141
	ds_load_tr16_b128 v[36:39] /*v[292:295]*/, v40 /*v296*/ offset:32448
	s_set_vgpr_msb 0x41cd
	ds_load_tr16_b128 v[124:127] /*v[892:895]*/, v40 /*v296*/ offset:32480
	s_wait_dscnt 0x2
	scratch_store_b128 off, v[120:123] /*v[888:891]*/, off offset:140 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[124:127] /*v[892:895]*/, off offset:156 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_tr16_b128 v[124:127] /*v[892:895]*/, v40 /*v296*/ offset:37056
	ds_load_tr16_b128 v[120:123] /*v[888:891]*/, v40 /*v296*/ offset:37088
	ds_load_tr16_b128 v[128:131] /*v[896:899]*/, v40 /*v296*/ offset:41664
	s_wait_dscnt 0x2
	scratch_store_b128 off, v[124:127] /*v[892:895]*/, off offset:364 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[128:131] /*v[896:899]*/, off offset:380 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_tr16_b128 v[124:127] /*v[892:895]*/, v40 /*v296*/ offset:41696
	scratch_store_b128 off, v[120:123] /*v[888:891]*/, off offset:108 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[124:127] /*v[892:895]*/, off offset:124 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_tr16_b128 v[124:127] /*v[892:895]*/, v40 /*v296*/ offset:46272
	ds_load_tr16_b128 v[120:123] /*v[888:891]*/, v40 /*v296*/ offset:46304
	ds_load_tr16_b128 v[128:131] /*v[896:899]*/, v40 /*v296*/ offset:50880
	s_wait_dscnt 0x2
	scratch_store_b128 off, v[124:127] /*v[892:895]*/, off offset:332 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[128:131] /*v[896:899]*/, off offset:348 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_tr16_b128 v[124:127] /*v[892:895]*/, v40 /*v296*/ offset:50912
	ds_load_tr16_b128 v[128:131] /*v[896:899]*/, v40 /*v296*/ offset:60096
	scratch_store_b128 off, v[120:123] /*v[888:891]*/, off offset:76 nv
	s_wait_dscnt 0x1
	scratch_store_b128 off, v[124:127] /*v[892:895]*/, off offset:92 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_tr16_b128 v[124:127] /*v[892:895]*/, v40 /*v296*/ offset:55488
	ds_load_tr16_b128 v[120:123] /*v[888:891]*/, v40 /*v296*/ offset:55520
	s_wait_dscnt 0x1
	s_clause 0x1
	scratch_store_b128 off, v[124:127] /*v[892:895]*/, off offset:300 nv
	scratch_store_b128 off, v[128:131] /*v[896:899]*/, off offset:316 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_tr16_b128 v[124:127] /*v[892:895]*/, v40 /*v296*/ offset:60128
	s_wait_dscnt 0x1
	scratch_store_b128 off, v[120:123] /*v[888:891]*/, off offset:44 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[124:127] /*v[892:895]*/, off offset:60 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_tr16_b128 v[124:127] /*v[892:895]*/, v40 /*v296*/ offset:64704
	ds_load_tr16_b128 v[120:123] /*v[888:891]*/, v40 /*v296*/ offset:64736
	s_wait_alu depctr_va_vdst(1)
	s_set_vgpr_msb 0xcdcc
	ds_load_tr16_b128 v[128:131] /*v[896:899]*/, v128
	s_wait_dscnt 0x2
	scratch_store_b128 off, v[124:127] /*v[892:895]*/, off offset:268 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[128:131] /*v[896:899]*/, off offset:284 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_va_vdst(0) depctr_vm_vsrc(0)
	ds_load_tr16_b128 v[124:127] /*v[892:895]*/, v130
	scratch_store_b128 off, v[120:123] /*v[888:891]*/, off offset:12 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[124:127] /*v[892:895]*/, off offset:28 nv
	s_set_vgpr_msb 0xcc0f
	v_max_num_f32_e32 v128, v64 /*v832*/, v65 /*v833*/
	s_wait_alu depctr_vm_vsrc(0)
	s_set_vgpr_msb 0xf0a
	v_max_num_f32_e32 v130, v192 /*v704*/, v193 /*v705*/
	s_set_vgpr_msb 0xa3f
	v_max3_num_f32 v131, v67 /*v835*/, v68 /*v836*/, v69 /*v837*/
	s_set_vgpr_msb 0x3f2a
	v_max3_num_f32 v134, v195 /*v707*/, v196 /*v708*/, v197 /*v709*/
	s_set_vgpr_msb 0x2a3f
	v_max3_num_f32 v202, v73 /*v841*/, v74 /*v842*/, v75 /*v843*/
	s_set_vgpr_msb 0x3f2a
	v_max3_num_f32 v203, v201 /*v713*/, v202 /*v714*/, v203 /*v715*/
	s_set_vgpr_msb 0x2a3f
	v_max3_num_f32 v220, v76 /*v844*/, v77 /*v845*/, v78 /*v846*/
	s_set_vgpr_msb 0x3f2a
	v_max3_num_f32 v221, v204 /*v716*/, v205 /*v717*/, v206 /*v718*/
	s_set_vgpr_msb 0x2a3f
	v_max3_num_f32 v254, v79 /*v847*/, v80 /*v848*/, v81 /*v849*/
	s_set_vgpr_msb 0x3f2a
	v_max3_num_f32 v255, v207 /*v719*/, v208 /*v720*/, v209 /*v721*/
	s_set_vgpr_msb 0x2aff
	v_max3_num_f32 v122 /*v890*/, v91 /*v859*/, v92 /*v860*/, v93 /*v861*/
	s_set_vgpr_msb 0xffea
	v_max3_num_f32 v123 /*v891*/, v219 /*v731*/, v220 /*v732*/, v221 /*v733*/
	s_set_vgpr_msb 0xeaff
	v_max3_num_f32 v124 /*v892*/, v94 /*v862*/, v95 /*v863*/, v96 /*v864*/
	s_set_vgpr_msb 0xffea
	v_max3_num_f32 v125 /*v893*/, v222 /*v734*/, v223 /*v735*/, v224 /*v736*/
	s_set_vgpr_msb 0xeaff
	v_max3_num_f32 v126 /*v894*/, v97 /*v865*/, v98 /*v866*/, v99 /*v867*/
	s_set_vgpr_msb 0xffea
	v_max3_num_f32 v127 /*v895*/, v225 /*v737*/, v226 /*v738*/, v227 /*v739*/
	s_set_vgpr_msb 0xeac0
	v_dual_max_num_f32 v158 /*v926*/, v223, v224 :: v_dual_max_num_f32 v176 /*v944*/, v250, v251
	s_set_vgpr_msb 0xc0d4
	v_max3_num_f32 v178 /*v946*/, v253, v16 /*v272*/, v17 /*v273*/
	s_set_vgpr_msb 0xd4d5
	v_max3_num_f32 v182 /*v950*/, v21 /*v277*/, v22 /*v278*/, v23 /*v279*/
	v_max3_num_f32 v184 /*v952*/, v24 /*v280*/, v25 /*v281*/, v26 /*v282*/
	v_max3_num_f32 v186 /*v954*/, v27 /*v283*/, v28 /*v284*/, v29 /*v285*/
	s_set_vgpr_msb 0xd57f
	v_max3_num_f32 v52 /*v308*/, v82 /*v850*/, v83 /*v851*/, v84 /*v852*/
	v_max3_num_f32 v54 /*v310*/, v85 /*v853*/, v86 /*v854*/, v87 /*v855*/
	s_set_vgpr_msb 0x7fff
	v_max3_num_f32 v120 /*v888*/, v88 /*v856*/, v89 /*v857*/, v90 /*v858*/
	v_max3_num_f32 v128 /*v896*/, v100 /*v868*/, v101 /*v869*/, v102 /*v870*/
	s_set_vgpr_msb 0xffea
	v_max3_num_f32 v129 /*v897*/, v228 /*v740*/, v229 /*v741*/, v230 /*v742*/
	s_set_vgpr_msb 0xeaff
	v_max3_num_f32 v130 /*v898*/, v103 /*v871*/, v104 /*v872*/, v105 /*v873*/
	s_set_vgpr_msb 0xffea
	v_max3_num_f32 v131 /*v899*/, v231 /*v743*/, v232 /*v744*/, v233 /*v745*/
	s_set_vgpr_msb 0xeaff
	v_max3_num_f32 v132 /*v900*/, v106 /*v874*/, v107 /*v875*/, v108 /*v876*/
	s_set_vgpr_msb 0xffea
	v_max3_num_f32 v133 /*v901*/, v234 /*v746*/, v235 /*v747*/, v236 /*v748*/
	s_set_vgpr_msb 0xeaff
	v_max3_num_f32 v134 /*v902*/, v109 /*v877*/, v110 /*v878*/, v111 /*v879*/
	v_max3_num_f32 v136 /*v904*/, v112 /*v880*/, v113 /*v881*/, v114 /*v882*/
	v_max3_num_f32 v138 /*v906*/, v115 /*v883*/, v116 /*v884*/, v117 /*v885*/
	s_set_vgpr_msb 0xffcf
	v_max3_num_f32 v140 /*v908*/, v118 /*v886*/, v119 /*v887*/, v192
	s_set_vgpr_msb 0xcfc0
	v_max3_num_f32 v142 /*v910*/, v193, v194, v195
	v_max3_num_f32 v144 /*v912*/, v196, v197, v198
	v_max3_num_f32 v146 /*v914*/, v199, v204, v205
	v_max3_num_f32 v148 /*v916*/, v206, v207, v208
	v_max3_num_f32 v150 /*v918*/, v209, v210, v211
	v_max3_num_f32 v152 /*v920*/, v212, v213, v214
	v_max3_num_f32 v154 /*v922*/, v215, v216, v217
	v_max3_num_f32 v156 /*v924*/, v218, v219, v222
	s_set_vgpr_msb 0xc0cf
	v_dual_max_num_f32 v159 /*v927*/, v17 /*v785*/, v18 /*v786*/ :: v_dual_max_num_f32 v177 /*v945*/, v44 /*v812*/, v45 /*v813*/
	s_set_vgpr_msb 0xcfc0
	v_max3_num_f32 v160 /*v928*/, v226, v227, v228
	v_max3_num_f32 v164 /*v932*/, v232, v233, v234
	v_max3_num_f32 v166 /*v934*/, v235, v236, v237
	v_max3_num_f32 v168 /*v936*/, v238, v239, v240
	s_set_vgpr_msb 0xc0ff
	v_max3_num_f32 v179 /*v947*/, v47 /*v815*/, v48 /*v816*/, v49 /*v817*/
	s_set_vgpr_msb 0xffd5
	v_max3_num_f32 v180 /*v948*/, v18 /*v274*/, v19 /*v275*/, v20 /*v276*/
	s_set_vgpr_msb 0xd5ff
	v_max3_num_f32 v183 /*v951*/, v53 /*v821*/, v54 /*v822*/, v55 /*v823*/
	v_max3_num_f32 v185 /*v953*/, v56 /*v824*/, v57 /*v825*/, v58 /*v826*/
	v_max3_num_f32 v187 /*v955*/, v59 /*v827*/, v60 /*v828*/, v61 /*v829*/
	s_set_vgpr_msb 0xff0c
	v_max3_num_f32 v128, v128, v66 /*v834*/, v131
	s_set_vgpr_msb 0xc08
	v_max3_num_f32 v130, v130, v194 /*v706*/, v134
	s_set_vgpr_msb 0x800
	v_max3_num_f32 v131, v202, v220, v254
	v_max3_num_f32 v134, v203, v221, v255
	s_set_vgpr_msb 63
	v_max3_num_f32 v220, v122 /*v890*/, v124 /*v892*/, v126 /*v894*/
	v_max3_num_f32 v221, v123 /*v891*/, v125 /*v893*/, v127 /*v895*/
	s_set_vgpr_msb 0x3ff3
	v_max3_num_f32 v126 /*v894*/, v176 /*v944*/, v252, v178 /*v946*/
	s_set_vgpr_msb 0xf3ff
	v_max3_num_f32 v127 /*v895*/, v182 /*v950*/, v184 /*v952*/, v186 /*v954*/
	s_set_vgpr_msb 0xff3f
	v_max3_num_f32 v200, v70 /*v838*/, v71 /*v839*/, v72 /*v840*/
	s_set_vgpr_msb 0x3f6a
	v_max3_num_f32 v53 /*v309*/, v210 /*v722*/, v211 /*v723*/, v212 /*v724*/
	v_max3_num_f32 v55 /*v311*/, v213 /*v725*/, v214 /*v726*/, v215 /*v727*/
	s_set_vgpr_msb 0x6aea
	v_max3_num_f32 v121 /*v889*/, v216 /*v728*/, v217 /*v729*/, v218 /*v730*/
	v_max3_num_f32 v135 /*v903*/, v237 /*v749*/, v238 /*v750*/, v239 /*v751*/
	v_max3_num_f32 v137 /*v905*/, v240 /*v752*/, v241 /*v753*/, v242 /*v754*/
	v_max3_num_f32 v139 /*v907*/, v243 /*v755*/, v244 /*v756*/, v245 /*v757*/
	s_set_vgpr_msb 0xeaff
	v_max3_num_f32 v161 /*v929*/, v20 /*v788*/, v21 /*v789*/, v22 /*v790*/
	s_set_vgpr_msb 0xffc0
	v_max3_num_f32 v162 /*v930*/, v229, v230, v231
	s_set_vgpr_msb 0xc0ff
	v_max3_num_f32 v165 /*v933*/, v26 /*v794*/, v27 /*v795*/, v28 /*v796*/
	v_max3_num_f32 v167 /*v935*/, v29 /*v797*/, v30 /*v798*/, v31 /*v799*/
	v_max3_num_f32 v169 /*v937*/, v32 /*v800*/, v33 /*v801*/, v34 /*v802*/
	s_set_vgpr_msb 0xffc0
	v_max3_num_f32 v170 /*v938*/, v241, v242, v243
	v_max3_num_f32 v172 /*v940*/, v244, v245, v246
	v_max3_num_f32 v174 /*v942*/, v247, v248, v249
	s_set_vgpr_msb 0xc0ff
	v_max3_num_f32 v181 /*v949*/, v50 /*v818*/, v51 /*v819*/, v52 /*v820*/
	s_set_vgpr_msb 0xff35
	v_max3_num_f32 v202, v52 /*v308*/, v54 /*v310*/, v120 /*v888*/
	s_set_vgpr_msb 0x353f
	v_max3_num_f32 v254, v128 /*v896*/, v130 /*v898*/, v132 /*v900*/
	v_max3_num_f32 v255, v129 /*v897*/, v131 /*v899*/, v133 /*v901*/
	s_set_vgpr_msb 0x3f7f
	v_max3_num_f32 v52 /*v308*/, v134 /*v902*/, v136 /*v904*/, v138 /*v906*/
	v_max3_num_f32 v54 /*v310*/, v140 /*v908*/, v142 /*v910*/, v144 /*v912*/
	s_set_vgpr_msb 0x7fff
	v_max3_num_f32 v120 /*v888*/, v146 /*v914*/, v148 /*v916*/, v150 /*v918*/
	v_max3_num_f32 v122 /*v890*/, v152 /*v920*/, v154 /*v922*/, v156 /*v924*/
	s_set_vgpr_msb 0xfff3
	v_max3_num_f32 v124 /*v892*/, v158 /*v926*/, v225, v160 /*v928*/
	s_set_vgpr_msb 0xf3ff
	v_max3_num_f32 v128 /*v896*/, v164 /*v932*/, v166 /*v934*/, v168 /*v936*/
	v_max3_num_f32 v131 /*v899*/, v177 /*v945*/, v46 /*v814*/, v179 /*v947*/
	v_max3_num_f32 v126 /*v894*/, v126 /*v894*/, v180 /*v948*/, v127 /*v895*/
	v_max3_num_f32 v127 /*v895*/, v183 /*v951*/, v185 /*v953*/, v187 /*v955*/
	s_set_vgpr_msb 0xff2a
	v_max3_num_f32 v201, v198 /*v710*/, v199 /*v711*/, v200 /*v712*/
	s_set_vgpr_msb 0x2aea
	v_max3_num_f32 v141 /*v909*/, v246 /*v758*/, v247 /*v759*/, v248 /*v760*/
	v_max3_num_f32 v143 /*v911*/, v249 /*v761*/, v250 /*v762*/, v251 /*v763*/
	v_max3_num_f32 v145 /*v913*/, v252 /*v764*/, v253 /*v765*/, v254 /*v766*/
	s_set_vgpr_msb 0xeafe
	v_max3_num_f32 v147 /*v915*/, v255 /*v767*/, v0 /*v768*/, v1 /*v769*/
	s_set_vgpr_msb 0xfeff
	v_max3_num_f32 v149 /*v917*/, v2 /*v770*/, v3 /*v771*/, v4 /*v772*/
	v_max3_num_f32 v151 /*v919*/, v5 /*v773*/, v6 /*v774*/, v7 /*v775*/
	v_max3_num_f32 v153 /*v921*/, v8 /*v776*/, v9 /*v777*/, v10 /*v778*/
	v_max3_num_f32 v155 /*v923*/, v11 /*v779*/, v12 /*v780*/, v13 /*v781*/
	v_max3_num_f32 v157 /*v925*/, v14 /*v782*/, v15 /*v783*/, v16 /*v784*/
	v_max3_num_f32 v163 /*v931*/, v23 /*v791*/, v24 /*v792*/, v25 /*v793*/
	v_max3_num_f32 v171 /*v939*/, v35 /*v803*/, v36 /*v804*/, v37 /*v805*/
	v_max3_num_f32 v173 /*v941*/, v38 /*v806*/, v39 /*v807*/, v40 /*v808*/
	v_max3_num_f32 v175 /*v943*/, v41 /*v809*/, v42 /*v810*/, v43 /*v811*/
	s_set_vgpr_msb 0xff35
	v_max3_num_f32 v203, v53 /*v309*/, v55 /*v311*/, v121 /*v889*/
	s_set_vgpr_msb 0x357f
	v_max3_num_f32 v53 /*v309*/, v135 /*v903*/, v137 /*v905*/, v139 /*v907*/
	s_set_vgpr_msb 0x7fff
	v_max3_num_f32 v125 /*v893*/, v159 /*v927*/, v19 /*v787*/, v161 /*v929*/
	v_max3_num_f32 v129 /*v897*/, v165 /*v933*/, v167 /*v935*/, v169 /*v937*/
	v_max3_num_f32 v130 /*v898*/, v170 /*v938*/, v172 /*v940*/, v174 /*v942*/
	s_set_vgpr_msb 0xff00
	v_max3_num_f32 v128, v128, v200, v131
	s_set_vgpr_msb 16
	v_max3_num_f32 v131, v220, v254, v52 /*v308*/
	s_set_vgpr_msb 0x103f
	v_max3_num_f32 v200, v124 /*v892*/, v162 /*v930*/, v128 /*v896*/
	s_set_vgpr_msb 0x3f17
	v_max3_num_f32 v220, v126 /*v894*/, v30 /*v286*/, v31 /*v287*/
	s_set_vgpr_msb 0x177d
	v_max3_num_f32 v52 /*v308*/, v54 /*v310*/, v120 /*v888*/, v122 /*v890*/
	s_set_vgpr_msb 0x7d7f
	v_max3_num_f32 v54 /*v310*/, v131 /*v899*/, v181 /*v949*/, v127 /*v895*/
	v_max3_num_f32 v55 /*v311*/, v141 /*v909*/, v143 /*v911*/, v145 /*v913*/
	s_set_vgpr_msb 0x7fff
	v_max3_num_f32 v121 /*v889*/, v147 /*v915*/, v149 /*v917*/, v151 /*v919*/
	v_max3_num_f32 v123 /*v891*/, v153 /*v921*/, v155 /*v923*/, v157 /*v925*/
	s_set_vgpr_msb 0xff3f
	v_max3_num_f32 v254, v171 /*v939*/, v173 /*v941*/, v175 /*v943*/
	s_set_vgpr_msb 0x3f00
	v_max3_num_f32 v128, v128, v202, v131
	s_set_vgpr_msb 12
	v_max3_num_f32 v131, v200, v130 /*v898*/, v220
	s_set_vgpr_msb 0xc00
	v_max3_num_f32 v130, v130, v201, v134
	s_set_vgpr_msb 16
	v_max3_num_f32 v134, v221, v255, v53 /*v309*/
	s_set_vgpr_msb 0x103f
	v_max3_num_f32 v200, v125 /*v893*/, v163 /*v931*/, v129 /*v897*/
	s_set_vgpr_msb 0x3f3d
	v_max3_num_f32 v201, v54 /*v310*/, v62 /*v830*/, v63 /*v831*/
	s_set_vgpr_msb 0x3d04
	v_max3_num_f32 v128, v128, v52 /*v308*/, v131
	s_set_vgpr_msb 0x43d
	v_max3_num_f32 v131, v55 /*v311*/, v121 /*v889*/, v123 /*v891*/
	s_set_vgpr_msb 0x3d00
	v_max3_num_f32 v130, v130, v203, v134
	v_max3_num_f32 v134, v200, v254, v201
	v_max3_num_f32 v130, v130, v131, v134
	v_dual_mov_b32 v200, v128 :: v_dual_mov_b32 v131, v130
	v_permlanex16_b32 v200, v200, s96, 0xfedcba98
	v_permlanex16_b32 v131, v131, s96, 0xfedcba98
	v_dual_max_num_f32 v128, v128, v200 :: v_dual_max_num_f32 v130, v130, v131
	s_set_vgpr_msb 4
	v_sub_f32_e32 v134, v128, v51 /*v307*/
	v_max_num_f32_e32 v128, v128, v51 /*v307*/
	v_sub_f32_e32 v131, v130, v50 /*v306*/
	s_set_vgpr_msb 0x401
	v_cmp_lt_f32_e32 vcc_lo, 0x41000000, v134
	v_max_num_f32_e32 v130, v50 /*v306*/, v130
	s_cmp_eq_u32 vcc_lo, 0
	s_cselect_b32 s2, -1, 0
	s_set_vgpr_msb 0x104
	v_cndmask_b32_e64 v128, v128, v51 /*v307*/, s2
	s_set_vgpr_msb 0x400
	v_cmp_lt_f32_e64 s2, 0x41000000, v131
	s_cmp_lg_u32 s2, 0
	s_cselect_b32 s5, -1, 0
	s_cmp_eq_u32 s2, 0
	s_cselect_b32 s2, -1, 0
	s_set_vgpr_msb 4
	v_cndmask_b32_e64 v134, v130, v50 /*v306*/, s2
	s_set_vgpr_msb 0x401
	v_mul_f32_e32 v254, 0xbfb8aa3b, v128
	v_sub_f32_e32 v130, v51 /*v307*/, v128
	s_set_vgpr_msb 0x140
	v_mul_f32_e32 v52 /*v308*/, 0xbfb8aa3b, v134
	s_set_vgpr_msb 0x4003
	v_pk_fma_f32 v[200:201], v[64:65] /*v[832:833]*/, s[56:57], v[254:255] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[202:203], v[66:67] /*v[834:835]*/, s[56:57], v[254:255] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[220:221], v[68:69] /*v[836:837]*/, s[56:57], v[254:255] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x300
	v_pk_fma_f32 v[192:193], v[192:193], s[56:57], v[254:255] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[194:195], v[194:195], s[56:57], v[254:255] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xc0
	v_exp_f32_e32 v64 /*v832*/, v200
	v_exp_f32_e32 v66 /*v834*/, v201
	v_nop
	s_set_vgpr_msb 0xc003
	v_pk_fma_f32 v[200:201], v[70:71] /*v[838:839]*/, s[56:57], v[254:255] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x3c0
	v_exp_f32_e32 v68 /*v836*/, v202
	v_exp_f32_e32 v120 /*v888*/, v203
	v_exp_f32_e32 v122 /*v890*/, v220
	s_set_vgpr_msb 0xc003
	v_pk_fma_f32 v[202:203], v[72:73] /*v[840:841]*/, s[56:57], v[254:255] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x3c0
	v_exp_f32_e32 v126 /*v894*/, v200
	v_exp_f32_e32 v128 /*v896*/, v201
	v_nop
	s_set_vgpr_msb 0xc003
	v_pk_fma_f32 v[200:201], v[76:77] /*v[844:845]*/, s[56:57], v[254:255] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x3c0
	v_exp_f32_e32 v124 /*v892*/, v221
	v_nop
	s_set_vgpr_msb 0xc003
	v_pk_fma_f32 v[220:221], v[74:75] /*v[842:843]*/, s[56:57], v[254:255] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x3c0
	v_exp_f32_e32 v130 /*v898*/, v202
	v_exp_f32_e32 v134 /*v902*/, v203
	v_exp_f32_e32 v148 /*v916*/, v200
	v_exp_f32_e32 v162 /*v930*/, v201
	v_nop
	s_set_vgpr_msb 0xc003
	v_pk_fma_f32 v[200:201], v[82:83] /*v[850:851]*/, s[56:57], v[254:255] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x3c0
	v_exp_f32_e32 v138 /*v906*/, v220
	s_set_vgpr_msb 0xc003
	v_pk_fma_f32 v[202:203], v[78:79] /*v[846:847]*/, s[56:57], v[254:255] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x3c0
	v_exp_f32_e32 v142 /*v910*/, v221
	v_nop
	s_set_vgpr_msb 0xc003
	v_pk_fma_f32 v[220:221], v[80:81] /*v[848:849]*/, s[56:57], v[254:255] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x3c0
	v_exp_f32_e32 v74 /*v842*/, v200
	v_exp_f32_e32 v78 /*v846*/, v201
	v_nop
	s_set_vgpr_msb 0xc003
	v_pk_fma_f32 v[200:201], v[88:89] /*v[856:857]*/, s[56:57], v[254:255] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x3c0
	v_exp_f32_e32 v164 /*v932*/, v202
	v_exp_f32_e32 v70 /*v838*/, v220
	v_exp_f32_e32 v72 /*v840*/, v221
	v_nop
	s_set_vgpr_msb 0xc003
	v_pk_fma_f32 v[220:221], v[86:87] /*v[854:855]*/, s[56:57], v[254:255] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x3c0
	v_exp_f32_e32 v140 /*v908*/, v200
	v_exp_f32_e32 v144 /*v912*/, v201
	v_nop
	s_set_vgpr_msb 0xc003
	v_pk_fma_f32 v[200:201], v[94:95] /*v[862:863]*/, s[56:57], v[254:255] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x3c0
	v_exp_f32_e32 v194 /*v962*/, v203
	v_exp_f32_e32 v132 /*v900*/, v220
	v_exp_f32_e32 v136 /*v904*/, v221
	v_nop
	s_set_vgpr_msb 0xc003
	v_pk_fma_f32 v[220:221], v[92:93] /*v[860:861]*/, s[56:57], v[254:255] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x3c0
	v_exp_f32_e32 v198 /*v966*/, v200
	v_exp_f32_e32 v224 /*v992*/, v201
	v_nop
	s_set_vgpr_msb 0xc003
	v_pk_fma_f32 v[200:201], v[100:101] /*v[868:869]*/, s[56:57], v[254:255] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[202:203], v[84:85] /*v[852:853]*/, s[56:57], v[254:255] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x3c0
	v_exp_f32_e32 v174 /*v942*/, v220
	v_exp_f32_e32 v196 /*v964*/, v221
	v_nop
	s_set_vgpr_msb 0xc003
	v_pk_fma_f32 v[220:221], v[98:99] /*v[866:867]*/, s[56:57], v[254:255] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x3c0
	v_exp_f32_e32 v94 /*v862*/, v200
	v_exp_f32_e32 v100 /*v868*/, v201
	v_nop
	s_set_vgpr_msb 0xc003
	v_pk_fma_f32 v[200:201], v[106:107] /*v[874:875]*/, s[56:57], v[254:255] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x300
	v_pk_fma_f32 v[196:197], v[196:197], s[56:57], v[254:255] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xc0
	v_exp_f32_e32 v82 /*v850*/, v202
	v_exp_f32_e32 v88 /*v856*/, v203
	v_nop
	s_set_vgpr_msb 0xc003
	v_pk_fma_f32 v[202:203], v[90:91] /*v[858:859]*/, s[56:57], v[254:255] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x3c0
	v_exp_f32_e32 v84 /*v852*/, v220
	v_exp_f32_e32 v90 /*v858*/, v221
	v_nop
	s_set_vgpr_msb 0xc003
	v_pk_fma_f32 v[220:221], v[104:105] /*v[872:873]*/, s[56:57], v[254:255] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x3c0
	v_exp_f32_e32 v178 /*v946*/, v200
	v_exp_f32_e32 v200 /*v968*/, v201
	v_nop
	s_set_vgpr_msb 0xc003
	v_pk_fma_f32 v[200:201], v[112:113] /*v[880:881]*/, s[56:57], v[254:255] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x3c0
	v_exp_f32_e32 v180 /*v948*/, v192
	v_exp_f32_e32 v202 /*v970*/, v193
	v_exp_f32_e32 v210 /*v978*/, v194
	s_set_vgpr_msb 0xc000
	v_pk_fma_f32 v[192:193], v[198:199], s[56:57], v[254:255] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xc0
	v_exp_f32_e32 v230 /*v998*/, v195
	v_exp_f32_e32 v238 /*v1006*/, v196
	s_set_vgpr_msb 0xc000
	v_pk_fma_f32 v[194:195], v[204:205], s[56:57], v[254:255] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xc0
	v_exp_f32_e32 v252 /*v1020*/, v197
	v_nop
	s_set_vgpr_msb 0xc000
	v_pk_fma_f32 v[196:197], v[206:207], s[56:57], v[254:255] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xc0
	v_exp_f32_e32 v152 /*v920*/, v220
	v_exp_f32_e32 v168 /*v936*/, v221
	v_nop
	s_set_vgpr_msb 0xc003
	v_pk_fma_f32 v[220:221], v[110:111] /*v[878:879]*/, s[56:57], v[254:255] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x3c0
	v_exp_f32_e32 v86 /*v854*/, v200
	v_exp_f32_e32 v92 /*v860*/, v201
	v_nop
	s_set_vgpr_msb 0xc003
	v_pk_fma_f32 v[200:201], v[118:119] /*v[886:887]*/, s[56:57], v[254:255] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x3c0
	v_exp_f32_e32 v254 /*v1022*/, v192
	s_set_vgpr_msb 0xc000
	v_exp_f32_e32 v206, v193
	s_set_vgpr_msb 0xc0
	v_exp_f32_e32 v98 /*v866*/, v194
	s_set_vgpr_msb 0xc000
	v_pk_fma_f32 v[192:193], v[208:209], s[56:57], v[254:255] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xc0
	v_exp_f32_e32 v106 /*v874*/, v195
	v_exp_f32_e32 v110 /*v878*/, v196
	s_set_vgpr_msb 0xc000
	v_pk_fma_f32 v[194:195], v[210:211], s[56:57], v[254:255] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xc0
	v_exp_f32_e32 v118 /*v886*/, v197
	v_nop
	s_set_vgpr_msb 0xc000
	v_pk_fma_f32 v[196:197], v[212:213], s[56:57], v[254:255] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xc0
	v_exp_f32_e32 v156 /*v924*/, v192
	v_exp_f32_e32 v172 /*v940*/, v193
	v_exp_f32_e32 v182 /*v950*/, v194
	s_set_vgpr_msb 0xc000
	v_pk_fma_f32 v[192:193], v[214:215], s[56:57], v[254:255] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xc0
	v_exp_f32_e32 v204 /*v972*/, v195
	v_exp_f32_e32 v212 /*v980*/, v196
	s_set_vgpr_msb 0xc000
	v_pk_fma_f32 v[194:195], v[216:217], s[56:57], v[254:255] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xc0
	v_exp_f32_e32 v232 /*v1000*/, v197
	v_nop
	s_set_vgpr_msb 0xc000
	v_pk_fma_f32 v[196:197], v[218:219], s[56:57], v[254:255] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xc0
	v_exp_f32_e32 v150 /*v918*/, v202
	v_exp_f32_e32 v166 /*v934*/, v203
	v_nop
	s_set_vgpr_msb 0xc003
	v_pk_fma_f32 v[202:203], v[96:97] /*v[864:865]*/, s[56:57], v[254:255] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x3c0
	v_exp_f32_e32 v240 /*v1008*/, v192
	v_exp_f32_e32 v176 /*v944*/, v193
	s_set_vgpr_msb 0xc000
	v_exp_f32_e32 v198, v194
	v_pk_fma_f32 v[192:193], v[222:223], s[56:57], v[254:255] op_sel_hi:[1,0,0]
	v_exp_f32_e32 v208, v195
	v_exp_f32_e32 v210, v196
	v_pk_fma_f32 v[194:195], v[224:225], s[56:57], v[254:255] op_sel_hi:[1,0,0]
	v_exp_f32_e32 v224, v197
	v_nop
	v_pk_fma_f32 v[196:197], v[226:227], s[56:57], v[254:255] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xc0
	v_exp_f32_e32 v76 /*v844*/, v202
	v_exp_f32_e32 v80 /*v848*/, v203
	v_nop
	s_set_vgpr_msb 0xc003
	v_pk_fma_f32 v[202:203], v[102:103] /*v[870:871]*/, s[56:57], v[254:255] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x3c0
	v_exp_f32_e32 v112 /*v880*/, v192
	v_exp_f32_e32 v154 /*v922*/, v193
	v_exp_f32_e32 v158 /*v926*/, v194
	s_set_vgpr_msb 0xc000
	v_pk_fma_f32 v[192:193], v[228:229], s[56:57], v[254:255] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xc0
	v_exp_f32_e32 v184 /*v952*/, v195
	v_exp_f32_e32 v188 /*v956*/, v196
	s_set_vgpr_msb 0xc000
	v_pk_fma_f32 v[194:195], v[230:231], s[56:57], v[254:255] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xc0
	v_exp_f32_e32 v206 /*v974*/, v197
	v_nop
	s_set_vgpr_msb 0xc000
	v_pk_fma_f32 v[196:197], v[232:233], s[56:57], v[254:255] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xc0
	v_exp_f32_e32 v102 /*v870*/, v202
	v_exp_f32_e32 v146 /*v914*/, v203
	v_nop
	s_set_vgpr_msb 0xc003
	v_pk_fma_f32 v[202:203], v[108:109] /*v[876:877]*/, s[56:57], v[254:255] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x3c0
	v_exp_f32_e32 v228 /*v996*/, v220
	v_exp_f32_e32 v250 /*v1018*/, v221
	v_nop
	s_set_vgpr_msb 0xc003
	v_pk_fma_f32 v[220:221], v[116:117] /*v[884:885]*/, s[56:57], v[254:255] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x3c0
	v_exp_f32_e32 v116 /*v884*/, v200
	v_exp_f32_e32 v214 /*v982*/, v192
	v_exp_f32_e32 v242 /*v1010*/, v194
	s_set_vgpr_msb 0xc000
	v_exp_f32_e32 v192, v195
	v_exp_f32_e32 v200, v196
	v_pk_fma_f32 v[194:195], v[236:237], s[56:57], v[254:255] op_sel_hi:[1,0,0]
	v_exp_f32_e32 v212, v197
	v_nop
	v_pk_fma_f32 v[196:197], v[238:239], s[56:57], v[254:255] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xc0
	v_exp_f32_e32 v208 /*v976*/, v202
	v_exp_f32_e32 v226 /*v994*/, v203
	v_nop
	s_set_vgpr_msb 0xc003
	v_pk_fma_f32 v[202:203], v[114:115] /*v[882:883]*/, s[56:57], v[254:255] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x300
	v_exp_f32_e32 v228, v194
	v_exp_f32_e32 v238, v195
	s_set_vgpr_msb 0xc0
	v_exp_f32_e32 v160 /*v928*/, v196
	s_set_vgpr_msb 0xc000
	v_pk_fma_f32 v[194:195], v[242:243], s[56:57], v[254:255] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xc0
	v_exp_f32_e32 v186 /*v954*/, v197
	v_nop
	s_set_vgpr_msb 0xc000
	v_pk_fma_f32 v[196:197], v[244:245], s[56:57], v[254:255] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xc0
	v_exp_f32_e32 v96 /*v864*/, v202
	v_exp_f32_e32 v104 /*v872*/, v203
	v_nop
	s_set_vgpr_msb 0xc000
	v_pk_fma_f32 v[202:203], v[234:235], s[56:57], v[254:255] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xc0
	v_exp_f32_e32 v220 /*v988*/, v194
	v_exp_f32_e32 v244 /*v1012*/, v196
	s_set_vgpr_msb 0xc000
	v_pk_fma_f32 v[204:205], v[248:249], s[56:57], v[254:255] op_sel_hi:[1,0,0]
	v_exp_f32_e32 v194, v197
	v_nop
	v_pk_fma_f32 v[196:197], v[250:251], s[56:57], v[254:255] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[216:217], v[252:253], s[56:57], v[254:255] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xc0
	v_exp_f32_e32 v108 /*v876*/, v220
	s_set_vgpr_msb 0xc000
	v_exp_f32_e32 v218, v202
	v_exp_f32_e32 v226, v203
	v_nop
	v_pk_fma_f32 v[202:203], v[240:241], s[56:57], v[254:255] op_sel_hi:[1,0,0]
	v_exp_f32_e32 v220, v204
	v_exp_f32_e32 v230, v205
	v_exp_f32_e32 v234, v196
	s_set_vgpr_msb 1
	v_pk_fma_f32 v[204:205], v[16:17] /*v[272:273]*/, s[56:57], v[254:255] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x100
	v_exp_f32_e32 v240, v197
	v_exp_f32_e32 v242, v216
	s_set_vgpr_msb 1
	v_pk_fma_f32 v[196:197], v[18:19] /*v[274:275]*/, s[56:57], v[254:255] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x100
	v_exp_f32_e32 v248, v217
	v_nop
	s_set_vgpr_msb 1
	v_pk_fma_f32 v[216:217], v[20:21] /*v[276:277]*/, s[56:57], v[254:255] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x152
	v_pk_fma_f32 v[16:17] /*v[272:273]*/, v[192:193] /*v[704:705]*/, s[56:57], v[52:53] /*v[308:309]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[18:19] /*v[274:275]*/, v[194:195] /*v[706:707]*/, s[56:57], v[52:53] /*v[308:309]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[20:21] /*v[276:277]*/, v[196:197] /*v[708:709]*/, s[56:57], v[52:53] /*v[308:309]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x52c0
	v_exp_f32_e32 v234 /*v1002*/, v193
	v_exp_f32_e32 v170 /*v938*/, v201
	s_set_vgpr_msb 0xc0c1
	v_exp_f32_e32 v65 /*v833*/, v16 /*v272*/
	v_exp_f32_e32 v67 /*v835*/, v17 /*v273*/
	v_exp_f32_e32 v69 /*v837*/, v18 /*v274*/
	s_set_vgpr_msb 0xc152
	v_pk_fma_f32 v[16:17] /*v[272:273]*/, v[198:199] /*v[710:711]*/, s[56:57], v[52:53] /*v[308:309]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x52c1
	v_exp_f32_e32 v121 /*v889*/, v19 /*v275*/
	v_exp_f32_e32 v123 /*v891*/, v20 /*v276*/
	s_set_vgpr_msb 0xc152
	v_pk_fma_f32 v[18:19] /*v[274:275]*/, v[200:201] /*v[712:713]*/, s[56:57], v[52:53] /*v[308:309]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x52c1
	v_exp_f32_e32 v125 /*v893*/, v21 /*v277*/
	v_nop
	s_set_vgpr_msb 0xc152
	v_pk_fma_f32 v[20:21] /*v[276:277]*/, v[202:203] /*v[714:715]*/, s[56:57], v[52:53] /*v[308:309]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x52c1
	v_exp_f32_e32 v127 /*v895*/, v16 /*v272*/
	v_exp_f32_e32 v129 /*v897*/, v17 /*v273*/
	v_exp_f32_e32 v131 /*v899*/, v18 /*v274*/
	s_set_vgpr_msb 0xc152
	v_pk_fma_f32 v[16:17] /*v[272:273]*/, v[204:205] /*v[716:717]*/, s[56:57], v[52:53] /*v[308:309]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x52c1
	v_exp_f32_e32 v135 /*v903*/, v19 /*v275*/
	v_exp_f32_e32 v139 /*v907*/, v20 /*v276*/
	s_set_vgpr_msb 0xc152
	v_pk_fma_f32 v[18:19] /*v[274:275]*/, v[206:207] /*v[718:719]*/, s[56:57], v[52:53] /*v[308:309]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x52c1
	v_exp_f32_e32 v143 /*v911*/, v21 /*v277*/
	v_nop
	s_set_vgpr_msb 0xc152
	v_pk_fma_f32 v[20:21] /*v[276:277]*/, v[208:209] /*v[720:721]*/, s[56:57], v[52:53] /*v[308:309]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x52c1
	v_exp_f32_e32 v149 /*v917*/, v16 /*v272*/
	v_exp_f32_e32 v163 /*v931*/, v17 /*v273*/
	v_exp_f32_e32 v165 /*v933*/, v18 /*v274*/
	s_set_vgpr_msb 0xc152
	v_pk_fma_f32 v[16:17] /*v[272:273]*/, v[210:211] /*v[722:723]*/, s[56:57], v[52:53] /*v[308:309]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x52c1
	v_exp_f32_e32 v195 /*v963*/, v19 /*v275*/
	v_exp_f32_e32 v71 /*v839*/, v20 /*v276*/
	s_set_vgpr_msb 0xc152
	v_pk_fma_f32 v[18:19] /*v[274:275]*/, v[212:213] /*v[724:725]*/, s[56:57], v[52:53] /*v[308:309]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x52c1
	v_exp_f32_e32 v73 /*v841*/, v21 /*v277*/
	v_nop
	s_set_vgpr_msb 0xc152
	v_pk_fma_f32 v[20:21] /*v[276:277]*/, v[214:215] /*v[726:727]*/, s[56:57], v[52:53] /*v[308:309]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x52c1
	v_exp_f32_e32 v75 /*v843*/, v16 /*v272*/
	v_exp_f32_e32 v79 /*v847*/, v17 /*v273*/
	v_exp_f32_e32 v83 /*v851*/, v18 /*v274*/
	s_set_vgpr_msb 0xc152
	v_pk_fma_f32 v[16:17] /*v[272:273]*/, v[216:217] /*v[728:729]*/, s[56:57], v[52:53] /*v[308:309]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x52c1
	v_exp_f32_e32 v89 /*v857*/, v19 /*v275*/
	v_exp_f32_e32 v133 /*v901*/, v20 /*v276*/
	s_set_vgpr_msb 0xc152
	v_pk_fma_f32 v[18:19] /*v[274:275]*/, v[218:219] /*v[730:731]*/, s[56:57], v[52:53] /*v[308:309]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x52c1
	v_exp_f32_e32 v137 /*v905*/, v21 /*v277*/
	v_nop
	s_set_vgpr_msb 0xc152
	v_pk_fma_f32 v[20:21] /*v[276:277]*/, v[220:221] /*v[732:733]*/, s[56:57], v[52:53] /*v[308:309]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x52c1
	v_exp_f32_e32 v141 /*v909*/, v16 /*v272*/
	v_exp_f32_e32 v145 /*v913*/, v17 /*v273*/
	v_exp_f32_e32 v151 /*v919*/, v18 /*v274*/
	s_set_vgpr_msb 0xc152
	v_pk_fma_f32 v[16:17] /*v[272:273]*/, v[222:223] /*v[734:735]*/, s[56:57], v[52:53] /*v[308:309]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x52c1
	v_exp_f32_e32 v167 /*v935*/, v19 /*v275*/
	v_exp_f32_e32 v175 /*v943*/, v20 /*v276*/
	s_set_vgpr_msb 0xc152
	v_pk_fma_f32 v[18:19] /*v[274:275]*/, v[224:225] /*v[736:737]*/, s[56:57], v[52:53] /*v[308:309]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x52c1
	v_exp_f32_e32 v197 /*v965*/, v21 /*v277*/
	v_nop
	s_set_vgpr_msb 0xc152
	v_pk_fma_f32 v[20:21] /*v[276:277]*/, v[226:227] /*v[738:739]*/, s[56:57], v[52:53] /*v[308:309]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x52c1
	v_exp_f32_e32 v199 /*v967*/, v16 /*v272*/
	v_exp_f32_e32 v225 /*v993*/, v17 /*v273*/
	v_exp_f32_e32 v77 /*v845*/, v18 /*v274*/
	s_set_vgpr_msb 0xc152
	v_pk_fma_f32 v[16:17] /*v[272:273]*/, v[228:229] /*v[740:741]*/, s[56:57], v[52:53] /*v[308:309]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x52c1
	v_exp_f32_e32 v81 /*v849*/, v19 /*v275*/
	v_exp_f32_e32 v85 /*v853*/, v20 /*v276*/
	s_set_vgpr_msb 0xc152
	v_pk_fma_f32 v[18:19] /*v[274:275]*/, v[230:231] /*v[742:743]*/, s[56:57], v[52:53] /*v[308:309]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x52c1
	v_exp_f32_e32 v91 /*v859*/, v21 /*v277*/
	v_nop
	s_set_vgpr_msb 0xc152
	v_pk_fma_f32 v[20:21] /*v[276:277]*/, v[232:233] /*v[744:745]*/, s[56:57], v[52:53] /*v[308:309]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x52c1
	v_exp_f32_e32 v95 /*v863*/, v16 /*v272*/
	v_exp_f32_e32 v101 /*v869*/, v17 /*v273*/
	v_exp_f32_e32 v103 /*v871*/, v18 /*v274*/
	s_set_vgpr_msb 0xc152
	v_pk_fma_f32 v[16:17] /*v[272:273]*/, v[234:235] /*v[746:747]*/, s[56:57], v[52:53] /*v[308:309]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x52c1
	v_exp_f32_e32 v147 /*v915*/, v19 /*v275*/
	v_exp_f32_e32 v153 /*v921*/, v20 /*v276*/
	s_set_vgpr_msb 0xc152
	v_pk_fma_f32 v[18:19] /*v[274:275]*/, v[236:237] /*v[748:749]*/, s[56:57], v[52:53] /*v[308:309]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x52c1
	v_exp_f32_e32 v169 /*v937*/, v21 /*v277*/
	v_nop
	s_set_vgpr_msb 0xc152
	v_pk_fma_f32 v[20:21] /*v[276:277]*/, v[238:239] /*v[750:751]*/, s[56:57], v[52:53] /*v[308:309]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x52c1
	v_exp_f32_e32 v179 /*v947*/, v16 /*v272*/
	v_exp_f32_e32 v201 /*v969*/, v17 /*v273*/
	v_exp_f32_e32 v209 /*v977*/, v18 /*v274*/
	s_set_vgpr_msb 0xc152
	v_pk_fma_f32 v[16:17] /*v[272:273]*/, v[240:241] /*v[752:753]*/, s[56:57], v[52:53] /*v[308:309]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x52c1
	v_exp_f32_e32 v227 /*v995*/, v19 /*v275*/
	v_exp_f32_e32 v229 /*v997*/, v20 /*v276*/
	s_set_vgpr_msb 0xc152
	v_pk_fma_f32 v[18:19] /*v[274:275]*/, v[242:243] /*v[754:755]*/, s[56:57], v[52:53] /*v[308:309]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x52c1
	v_exp_f32_e32 v251 /*v1019*/, v21 /*v277*/
	v_nop
	s_set_vgpr_msb 0xc152
	v_pk_fma_f32 v[20:21] /*v[276:277]*/, v[244:245] /*v[756:757]*/, s[56:57], v[52:53] /*v[308:309]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x52c1
	v_exp_f32_e32 v87 /*v855*/, v16 /*v272*/
	v_exp_f32_e32 v93 /*v861*/, v17 /*v273*/
	v_exp_f32_e32 v97 /*v865*/, v18 /*v274*/
	s_set_vgpr_msb 0xc152
	v_pk_fma_f32 v[16:17] /*v[272:273]*/, v[246:247] /*v[758:759]*/, s[56:57], v[52:53] /*v[308:309]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x52c1
	v_exp_f32_e32 v105 /*v873*/, v19 /*v275*/
	v_exp_f32_e32 v109 /*v877*/, v20 /*v276*/
	s_set_vgpr_msb 0xc152
	v_pk_fma_f32 v[18:19] /*v[274:275]*/, v[248:249] /*v[760:761]*/, s[56:57], v[52:53] /*v[308:309]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x52c1
	v_exp_f32_e32 v115 /*v883*/, v21 /*v277*/
	v_nop
	s_set_vgpr_msb 0xc152
	v_pk_fma_f32 v[20:21] /*v[276:277]*/, v[250:251] /*v[762:763]*/, s[56:57], v[52:53] /*v[308:309]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x52c1
	v_exp_f32_e32 v117 /*v885*/, v16 /*v272*/
	v_exp_f32_e32 v171 /*v939*/, v17 /*v273*/
	v_exp_f32_e32 v181 /*v949*/, v18 /*v274*/
	s_set_vgpr_msb 0xc152
	v_pk_fma_f32 v[16:17] /*v[272:273]*/, v[252:253] /*v[764:765]*/, s[56:57], v[52:53] /*v[308:309]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x52c1
	v_exp_f32_e32 v203 /*v971*/, v19 /*v275*/
	v_exp_f32_e32 v211 /*v979*/, v20 /*v276*/
	s_set_vgpr_msb 0xc152
	v_pk_fma_f32 v[18:19] /*v[274:275]*/, v[254:255] /*v[766:767]*/, s[56:57], v[52:53] /*v[308:309]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x52c1
	v_exp_f32_e32 v231 /*v999*/, v21 /*v277*/
	v_nop
	s_set_vgpr_msb 0xc153
	v_pk_fma_f32 v[20:21] /*v[276:277]*/, v[0:1] /*v[768:769]*/, s[56:57], v[52:53] /*v[308:309]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x53c1
	v_exp_f32_e32 v239 /*v1007*/, v16 /*v272*/
	v_exp_f32_e32 v253 /*v1021*/, v17 /*v273*/
	v_exp_f32_e32 v255 /*v1023*/, v18 /*v274*/
	s_set_vgpr_msb 0xc153
	v_pk_fma_f32 v[16:17] /*v[272:273]*/, v[2:3] /*v[770:771]*/, s[56:57], v[52:53] /*v[308:309]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5301
	v_exp_f32_e32 v207, v19 /*v275*/
	s_set_vgpr_msb 0x1c1
	v_exp_f32_e32 v99 /*v867*/, v20 /*v276*/
	s_set_vgpr_msb 0xc153
	v_pk_fma_f32 v[18:19] /*v[274:275]*/, v[4:5] /*v[772:773]*/, s[56:57], v[52:53] /*v[308:309]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x53c1
	v_exp_f32_e32 v107 /*v875*/, v21 /*v277*/
	v_nop
	s_set_vgpr_msb 0xc153
	v_pk_fma_f32 v[20:21] /*v[276:277]*/, v[6:7] /*v[774:775]*/, s[56:57], v[52:53] /*v[308:309]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x53c1
	v_exp_f32_e32 v111 /*v879*/, v16 /*v272*/
	v_exp_f32_e32 v119 /*v887*/, v17 /*v273*/
	v_exp_f32_e32 v157 /*v925*/, v18 /*v274*/
	s_set_vgpr_msb 0xc153
	v_pk_fma_f32 v[16:17] /*v[272:273]*/, v[8:9] /*v[776:777]*/, s[56:57], v[52:53] /*v[308:309]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x53c1
	v_exp_f32_e32 v173 /*v941*/, v19 /*v275*/
	v_exp_f32_e32 v183 /*v951*/, v20 /*v276*/
	s_set_vgpr_msb 0xc153
	v_pk_fma_f32 v[18:19] /*v[274:275]*/, v[10:11] /*v[778:779]*/, s[56:57], v[52:53] /*v[308:309]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x53c1
	v_exp_f32_e32 v205 /*v973*/, v21 /*v277*/
	v_nop
	s_set_vgpr_msb 0xc153
	v_pk_fma_f32 v[20:21] /*v[276:277]*/, v[12:13] /*v[780:781]*/, s[56:57], v[52:53] /*v[308:309]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x53c1
	v_exp_f32_e32 v213 /*v981*/, v16 /*v272*/
	v_exp_f32_e32 v233 /*v1001*/, v17 /*v273*/
	v_exp_f32_e32 v241 /*v1009*/, v18 /*v274*/
	s_set_vgpr_msb 0xc153
	v_pk_fma_f32 v[16:17] /*v[272:273]*/, v[14:15] /*v[782:783]*/, s[56:57], v[52:53] /*v[308:309]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x53c1
	v_exp_f32_e32 v177 /*v945*/, v19 /*v275*/
	s_set_vgpr_msb 0xc101
	v_exp_f32_e32 v199, v20 /*v276*/
	s_set_vgpr_msb 0x153
	v_pk_fma_f32 v[18:19] /*v[274:275]*/, v[16:17] /*v[784:785]*/, s[56:57], v[52:53] /*v[308:309]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5301
	v_exp_f32_e32 v209, v21 /*v277*/
	v_nop
	s_set_vgpr_msb 0x153
	v_pk_fma_f32 v[20:21] /*v[276:277]*/, v[18:19] /*v[786:787]*/, s[56:57], v[52:53] /*v[308:309]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5301
	v_exp_f32_e32 v211, v16 /*v272*/
	v_exp_f32_e32 v225, v17 /*v273*/
	s_set_vgpr_msb 0x1c1
	v_exp_f32_e32 v113 /*v881*/, v18 /*v274*/
	s_set_vgpr_msb 0xc153
	v_pk_fma_f32 v[16:17] /*v[272:273]*/, v[20:21] /*v[788:789]*/, s[56:57], v[52:53] /*v[308:309]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x53c1
	v_exp_f32_e32 v155 /*v923*/, v19 /*v275*/
	v_exp_f32_e32 v159 /*v927*/, v20 /*v276*/
	s_set_vgpr_msb 0xc153
	v_pk_fma_f32 v[18:19] /*v[274:275]*/, v[22:23] /*v[790:791]*/, s[56:57], v[52:53] /*v[308:309]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x53c1
	v_exp_f32_e32 v185 /*v953*/, v21 /*v277*/
	v_nop
	s_set_vgpr_msb 0xc153
	v_pk_fma_f32 v[20:21] /*v[276:277]*/, v[24:25] /*v[792:793]*/, s[56:57], v[52:53] /*v[308:309]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x53c1
	v_exp_f32_e32 v189 /*v957*/, v16 /*v272*/
	v_exp_f32_e32 v207 /*v975*/, v17 /*v273*/
	v_exp_f32_e32 v215 /*v983*/, v18 /*v274*/
	s_set_vgpr_msb 0xc153
	v_pk_fma_f32 v[16:17] /*v[272:273]*/, v[26:27] /*v[794:795]*/, s[56:57], v[52:53] /*v[308:309]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x53c1
	v_exp_f32_e32 v235 /*v1003*/, v19 /*v275*/
	v_exp_f32_e32 v243 /*v1011*/, v20 /*v276*/
	s_set_vgpr_msb 0xc153
	v_pk_fma_f32 v[18:19] /*v[274:275]*/, v[28:29] /*v[796:797]*/, s[56:57], v[52:53] /*v[308:309]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5301
	v_exp_f32_e32 v193, v21 /*v277*/
	v_nop
	s_set_vgpr_msb 0x153
	v_pk_fma_f32 v[20:21] /*v[276:277]*/, v[30:31] /*v[798:799]*/, s[56:57], v[52:53] /*v[308:309]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5301
	v_exp_f32_e32 v201, v16 /*v272*/
	v_exp_f32_e32 v213, v17 /*v273*/
	v_exp_f32_e32 v219, v18 /*v274*/
	s_set_vgpr_msb 0x153
	v_pk_fma_f32 v[16:17] /*v[272:273]*/, v[32:33] /*v[800:801]*/, s[56:57], v[52:53] /*v[308:309]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5301
	v_exp_f32_e32 v227, v19 /*v275*/
	v_exp_f32_e32 v229, v20 /*v276*/
	s_set_vgpr_msb 0x153
	v_pk_fma_f32 v[18:19] /*v[274:275]*/, v[34:35] /*v[802:803]*/, s[56:57], v[52:53] /*v[308:309]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5301
	v_exp_f32_e32 v239, v21 /*v277*/
	v_nop
	s_set_vgpr_msb 0x153
	v_pk_fma_f32 v[20:21] /*v[276:277]*/, v[36:37] /*v[804:805]*/, s[56:57], v[52:53] /*v[308:309]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x53c0
	v_exp_f32_e32 v190 /*v958*/, v202
	v_exp_f32_e32 v216 /*v984*/, v203
	v_nop
	s_set_vgpr_msb 0xc000
	v_pk_fma_f32 v[202:203], v[246:247], s[56:57], v[254:255] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xc1
	v_exp_f32_e32 v161 /*v929*/, v16 /*v272*/
	v_exp_f32_e32 v187 /*v955*/, v17 /*v273*/
	v_exp_f32_e32 v191 /*v959*/, v18 /*v274*/
	s_set_vgpr_msb 0xc153
	v_pk_fma_f32 v[16:17] /*v[272:273]*/, v[38:39] /*v[806:807]*/, s[56:57], v[52:53] /*v[308:309]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x53c1
	v_exp_f32_e32 v217 /*v985*/, v19 /*v275*/
	v_exp_f32_e32 v221 /*v989*/, v20 /*v276*/
	s_set_vgpr_msb 0xc153
	v_pk_fma_f32 v[18:19] /*v[274:275]*/, v[40:41] /*v[808:809]*/, s[56:57], v[52:53] /*v[308:309]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x53c1
	v_exp_f32_e32 v237 /*v1005*/, v21 /*v277*/
	v_nop
	s_set_vgpr_msb 0xc153
	v_pk_fma_f32 v[20:21] /*v[276:277]*/, v[42:43] /*v[810:811]*/, s[56:57], v[52:53] /*v[308:309]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x53c0
	v_exp_f32_e32 v114 /*v882*/, v221
	v_exp_f32_e32 v236 /*v1004*/, v195
	s_set_vgpr_msb 0xc000
	v_exp_f32_e32 v214, v203
	s_set_vgpr_msb 0xc1
	v_exp_f32_e32 v245 /*v1013*/, v16 /*v272*/
	s_set_vgpr_msb 0xc101
	v_exp_f32_e32 v195, v17 /*v273*/
	v_exp_f32_e32 v203, v18 /*v274*/
	s_set_vgpr_msb 0x153
	v_pk_fma_f32 v[16:17] /*v[272:273]*/, v[44:45] /*v[812:813]*/, s[56:57], v[52:53] /*v[308:309]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5301
	v_exp_f32_e32 v215, v19 /*v275*/
	v_exp_f32_e32 v221, v20 /*v276*/
	s_set_vgpr_msb 0x153
	v_pk_fma_f32 v[18:19] /*v[274:275]*/, v[46:47] /*v[814:815]*/, s[56:57], v[52:53] /*v[308:309]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5301
	v_exp_f32_e32 v231, v21 /*v277*/
	v_nop
	s_set_vgpr_msb 0x153
	v_pk_fma_f32 v[20:21] /*v[276:277]*/, v[48:49] /*v[816:817]*/, s[56:57], v[52:53] /*v[308:309]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x53c0
	v_exp_f32_e32 v192 /*v960*/, v204
	v_exp_f32_e32 v218 /*v986*/, v205
	v_nop
	s_set_vgpr_msb 0xc001
	v_pk_fma_f32 v[204:205], v[22:23] /*v[278:279]*/, s[56:57], v[254:255] op_sel_hi:[1,0,0]
	v_exp_f32_e32 v235, v16 /*v272*/
	v_exp_f32_e32 v241, v17 /*v273*/
	v_exp_f32_e32 v243, v18 /*v274*/
	s_set_vgpr_msb 0x153
	v_pk_fma_f32 v[16:17] /*v[272:273]*/, v[50:51] /*v[818:819]*/, s[56:57], v[52:53] /*v[308:309]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5301
	v_exp_f32_e32 v249, v19 /*v275*/
	s_set_vgpr_msb 0x1c1
	v_exp_f32_e32 v193 /*v961*/, v20 /*v276*/
	s_set_vgpr_msb 0xc153
	v_pk_fma_f32 v[18:19] /*v[274:275]*/, v[52:53] /*v[820:821]*/, s[56:57], v[52:53] /*v[308:309]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x53c1
	v_exp_f32_e32 v219 /*v987*/, v21 /*v277*/
	v_nop
	s_set_vgpr_msb 0xc153
	v_pk_fma_f32 v[20:21] /*v[276:277]*/, v[54:55] /*v[822:823]*/, s[56:57], v[52:53] /*v[308:309]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x53c0
	v_exp_f32_e32 v222 /*v990*/, v196
	v_exp_f32_e32 v246 /*v1014*/, v197
	v_exp_f32_e32 v248 /*v1016*/, v216
	s_set_vgpr_msb 0xc001
	v_pk_fma_f32 v[222:223], v[24:25] /*v[280:281]*/, s[56:57], v[254:255] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x100
	v_exp_f32_e32 v196, v217
	s_set_vgpr_msb 1
	v_pk_fma_f32 v[236:237], v[26:27] /*v[282:283]*/, s[56:57], v[254:255] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x100
	v_exp_f32_e32 v216, v205
	s_set_vgpr_msb 1
	v_pk_fma_f32 v[246:247], v[28:29] /*v[284:285]*/, s[56:57], v[254:255] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1c1
	v_exp_f32_e32 v223 /*v991*/, v16 /*v272*/
	v_exp_f32_e32 v247 /*v1015*/, v17 /*v273*/
	v_exp_f32_e32 v249 /*v1017*/, v18 /*v274*/
	s_set_vgpr_msb 0xc153
	v_pk_fma_f32 v[16:17] /*v[272:273]*/, v[56:57] /*v[824:825]*/, s[56:57], v[52:53] /*v[308:309]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5301
	v_exp_f32_e32 v197, v19 /*v275*/
	v_exp_f32_e32 v205, v20 /*v276*/
	s_set_vgpr_msb 0x153
	v_pk_fma_f32 v[18:19] /*v[274:275]*/, v[58:59] /*v[826:827]*/, s[56:57], v[52:53] /*v[308:309]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5301
	v_exp_f32_e32 v217, v21 /*v277*/
	v_nop
	s_set_vgpr_msb 0x153
	v_pk_fma_f32 v[20:21] /*v[276:277]*/, v[60:61] /*v[828:829]*/, s[56:57], v[52:53] /*v[308:309]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5300
	v_exp_f32_e32 v232, v223
	s_set_vgpr_msb 1
	v_pk_fma_f32 v[252:253], v[30:31] /*v[286:287]*/, s[56:57], v[254:255] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x100
	v_exp_f32_e32 v244, v237
	v_exp_f32_e32 v250, v247
	s_set_vgpr_msb 1
	v_exp_f32_e32 v223, v16 /*v272*/
	v_exp_f32_e32 v233, v17 /*v273*/
	v_exp_f32_e32 v237, v18 /*v274*/
	v_exp_f32_e32 v245, v19 /*v275*/
	s_set_vgpr_msb 0x153
	v_pk_fma_f32 v[16:17] /*v[272:273]*/, v[62:63] /*v[830:831]*/, s[56:57], v[52:53] /*v[308:309]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5301
	v_exp_f32_e32 v247, v20 /*v276*/
	s_set_vgpr_msb 0x14f
	v_pk_add_f32 v[18:19] /*v[274:275]*/, v[64:65] /*v[832:833]*/, v[66:67] /*v[834:835]*/
	s_set_vgpr_msb 0x4f01
	v_exp_f32_e32 v251, v21 /*v277*/
	v_nop
	s_set_vgpr_msb 0x14f
	v_pk_add_f32 v[20:21] /*v[276:277]*/, v[120:121] /*v[888:889]*/, v[122:123] /*v[890:891]*/
	s_set_vgpr_msb 0x4f00
	v_exp_f32_e32 v204, v204
	v_exp_f32_e32 v222, v222
	v_exp_f32_e32 v254, v253
	s_set_vgpr_msb 1
	v_exp_f32_e32 v253, v16 /*v272*/
	v_exp_f32_e32 v255, v17 /*v273*/
	v_nop
	s_set_vgpr_msb 0x147
	v_pk_add_f32 v[16:17] /*v[272:273]*/, v[68:69] /*v[836:837]*/, v[18:19] /*v[274:275]*/
	s_set_vgpr_msb 0x474f
	v_pk_add_f32 v[18:19] /*v[274:275]*/, v[126:127] /*v[894:895]*/, v[128:129] /*v[896:897]*/
	s_set_vgpr_msb 0x4f47
	v_pk_add_f32 v[20:21] /*v[276:277]*/, v[124:125] /*v[892:893]*/, v[20:21] /*v[276:277]*/
	s_set_vgpr_msb 0x474f
	v_pk_add_f32 v[22:23] /*v[278:279]*/, v[134:135] /*v[902:903]*/, v[138:139] /*v[906:907]*/
	v_pk_add_f32 v[24:25] /*v[280:281]*/, v[148:149] /*v[916:917]*/, v[162:163] /*v[930:931]*/
	v_pk_add_f32 v[28:29] /*v[284:285]*/, v[74:75] /*v[842:843]*/, v[78:79] /*v[846:847]*/
	v_pk_add_f32 v[30:31] /*v[286:287]*/, v[88:89] /*v[856:857]*/, v[132:133] /*v[900:901]*/
	v_pk_add_f32 v[54:55] /*v[310:311]*/, v[166:167] /*v[934:935]*/, v[174:175] /*v[942:943]*/
	s_set_vgpr_msb 0x4f8f
	v_pk_add_f32 v[192:193] /*v[704:705]*/, v[198:199] /*v[966:967]*/, v[224:225] /*v[992:993]*/
	v_pk_add_f32 v[196:197] /*v[708:709]*/, v[94:95] /*v[862:863]*/, v[100:101] /*v[868:869]*/
	v_pk_add_f32 v[198:199] /*v[710:711]*/, v[146:147] /*v[914:915]*/, v[152:153] /*v[920:921]*/
	s_set_vgpr_msb 0x8f00
	v_exp_f32_e32 v202, v202
	v_exp_f32_e32 v236, v236
	v_exp_f32_e32 v246, v246
	s_set_vgpr_msb 0x4f
	v_pk_add_f32 v[26:27] /*v[282:283]*/, v[194:195] /*v[962:963]*/, v[70:71] /*v[838:839]*/
	s_set_vgpr_msb 0x4f47
	v_pk_add_f32 v[18:19] /*v[274:275]*/, v[130:131] /*v[898:899]*/, v[18:19] /*v[274:275]*/
	v_pk_add_f32 v[22:23] /*v[278:279]*/, v[142:143] /*v[910:911]*/, v[22:23] /*v[278:279]*/
	v_pk_add_f32 v[24:25] /*v[280:281]*/, v[164:165] /*v[932:933]*/, v[24:25] /*v[280:281]*/
	v_pk_add_f32 v[28:29] /*v[284:285]*/, v[82:83] /*v[850:851]*/, v[28:29] /*v[284:285]*/
	s_set_vgpr_msb 0x474f
	v_pk_add_f32 v[52:53] /*v[308:309]*/, v[140:141] /*v[908:909]*/, v[144:145] /*v[912:913]*/
	s_set_vgpr_msb 0x4f47
	v_pk_add_f32 v[30:31] /*v[286:287]*/, v[136:137] /*v[904:905]*/, v[30:31] /*v[286:287]*/
	s_set_vgpr_msb 0x478f
	v_pk_add_f32 v[194:195] /*v[706:707]*/, v[80:81] /*v[848:849]*/, v[84:85] /*v[852:853]*/
	s_set_vgpr_msb 0x8f47
	v_pk_add_f32 v[54:55] /*v[310:311]*/, v[196:197] /*v[964:965]*/, v[54:55] /*v[310:311]*/
	s_set_vgpr_msb 0x478b
	v_pk_add_f32 v[192:193] /*v[704:705]*/, v[76:77] /*v[844:845]*/, v[192:193] /*v[704:705]*/
	s_set_vgpr_msb 0x8b8f
	v_pk_add_f32 v[200:201] /*v[712:713]*/, v[178:179] /*v[946:947]*/, v[200:201] /*v[968:969]*/
	v_pk_add_f32 v[202:203] /*v[714:715]*/, v[226:227] /*v[994:995]*/, v[228:229] /*v[996:997]*/
	s_set_vgpr_msb 0x8f8b
	v_pk_add_f32 v[196:197] /*v[708:709]*/, v[102:103] /*v[870:871]*/, v[196:197] /*v[708:709]*/
	s_set_vgpr_msb 0x8b8f
	v_pk_add_f32 v[204:205] /*v[716:717]*/, v[86:87] /*v[854:855]*/, v[92:93] /*v[860:861]*/
	s_set_vgpr_msb 0x8f8b
	v_pk_add_f32 v[198:199] /*v[710:711]*/, v[168:169] /*v[936:937]*/, v[198:199] /*v[710:711]*/
	s_set_vgpr_msb 0x8b8f
	v_pk_add_f32 v[208:209] /*v[720:721]*/, v[116:117] /*v[884:885]*/, v[170:171] /*v[938:939]*/
	v_pk_add_f32 v[210:211] /*v[722:723]*/, v[202:203] /*v[970:971]*/, v[210:211] /*v[978:979]*/
	s_set_vgpr_msb 0x8f8c
	v_pk_add_f32 v[214:215] /*v[726:727]*/, v[206:207], v[98:99] /*v[866:867]*/
	s_set_vgpr_msb 0x8c8f
	v_pk_add_f32 v[216:217] /*v[728:729]*/, v[110:111] /*v[878:879]*/, v[118:119] /*v[886:887]*/
	v_pk_add_f32 v[226:227] /*v[738:739]*/, v[154:155] /*v[922:923]*/, v[158:159] /*v[926:927]*/
	v_pk_add_f32 v[228:229] /*v[740:741]*/, v[188:189] /*v[956:957]*/, v[206:207] /*v[974:975]*/
	s_set_vgpr_msb 0x8f80
	v_pk_add_f32 v[232:233] /*v[744:745]*/, v[200:201], v[212:213]
	v_pk_add_f32 v[234:235] /*v[746:747]*/, v[226:227], v[228:229]
	s_set_vgpr_msb 0x808f
	v_pk_add_f32 v[238:239] /*v[750:751]*/, v[216:217] /*v[984:985]*/, v[220:221] /*v[988:989]*/
	s_set_vgpr_msb 0x8f83
	v_pk_add_f32 v[240:241] /*v[752:753]*/, v[244:245] /*v[1012:1013]*/, v[194:195]
	s_set_vgpr_msb 0x8380
	v_pk_add_f32 v[244:245] /*v[756:757]*/, v[234:235], v[240:241]
	s_set_vgpr_msb 0x808c
	v_pk_add_f32 v[246:247] /*v[758:759]*/, v[248:249], v[192:193] /*v[960:961]*/
	s_set_vgpr_msb 0x8c80
	v_pk_add_f32 v[250:251] /*v[762:763]*/, v[196:197], v[204:205]
	v_pk_add_f32 v[252:253] /*v[764:765]*/, v[222:223], v[232:233]
	s_set_vgpr_msb 0x8045
	v_pk_add_f32 v[16:17] /*v[272:273]*/, v[16:17] /*v[272:273]*/, v[20:21] /*v[276:277]*/
	s_set_vgpr_msb 0x4547
	v_pk_add_f32 v[26:27] /*v[282:283]*/, v[72:73] /*v[840:841]*/, v[26:27] /*v[282:283]*/
	v_pk_add_f32 v[52:53] /*v[308:309]*/, v[150:151] /*v[918:919]*/, v[52:53] /*v[308:309]*/
	s_set_vgpr_msb 0x478b
	v_pk_add_f32 v[194:195] /*v[706:707]*/, v[90:91] /*v[858:859]*/, v[194:195] /*v[706:707]*/
	v_pk_add_f32 v[200:201] /*v[712:713]*/, v[208:209] /*v[976:977]*/, v[200:201] /*v[712:713]*/
	v_pk_add_f32 v[202:203] /*v[714:715]*/, v[250:251] /*v[1018:1019]*/, v[202:203] /*v[714:715]*/
	s_set_vgpr_msb 0x8b8f
	v_pk_add_f32 v[206:207] /*v[718:719]*/, v[104:105] /*v[872:873]*/, v[108:109] /*v[876:877]*/
	s_set_vgpr_msb 0x8f8b
	v_pk_add_f32 v[204:205] /*v[716:717]*/, v[96:97] /*v[864:865]*/, v[204:205] /*v[716:717]*/
	s_set_vgpr_msb 0x8b8f
	v_pk_add_f32 v[212:213] /*v[724:725]*/, v[238:239] /*v[1006:1007]*/, v[252:253] /*v[1020:1021]*/
	s_set_vgpr_msb 0x8f8b
	v_pk_add_f32 v[208:209] /*v[720:721]*/, v[180:181] /*v[948:949]*/, v[208:209] /*v[720:721]*/
	v_pk_add_f32 v[210:211] /*v[722:723]*/, v[230:231] /*v[998:999]*/, v[210:211] /*v[722:723]*/
	v_pk_add_f32 v[214:215] /*v[726:727]*/, v[106:107] /*v[874:875]*/, v[214:215] /*v[726:727]*/
	s_set_vgpr_msb 0x8b8f
	v_pk_add_f32 v[218:219] /*v[730:731]*/, v[172:173] /*v[940:941]*/, v[182:183] /*v[950:951]*/
	v_pk_add_f32 v[220:221] /*v[732:733]*/, v[212:213] /*v[980:981]*/, v[232:233] /*v[1000:1001]*/
	s_set_vgpr_msb 0x8f83
	v_pk_add_f32 v[222:223] /*v[734:735]*/, v[176:177] /*v[944:945]*/, v[198:199]
	s_set_vgpr_msb 0x838b
	v_pk_add_f32 v[216:217] /*v[728:729]*/, v[156:157] /*v[924:925]*/, v[216:217] /*v[728:729]*/
	s_set_vgpr_msb 0x8b8f
	v_pk_add_f32 v[230:231] /*v[742:743]*/, v[234:235] /*v[1002:1003]*/, v[242:243] /*v[1010:1011]*/
	s_set_vgpr_msb 0x8f8b
	v_pk_add_f32 v[226:227] /*v[738:739]*/, v[184:185] /*v[952:953]*/, v[226:227] /*v[738:739]*/
	v_pk_add_f32 v[228:229] /*v[740:741]*/, v[214:215] /*v[982:983]*/, v[228:229] /*v[740:741]*/
	s_set_vgpr_msb 0x8b88
	v_pk_add_f32 v[232:233] /*v[744:745]*/, v[218:219], v[232:233] /*v[744:745]*/
	s_set_vgpr_msb 0x888f
	v_pk_add_f32 v[236:237] /*v[748:749]*/, v[160:161] /*v[928:929]*/, v[186:187] /*v[954:955]*/
	s_set_vgpr_msb 0x8f88
	v_pk_add_f32 v[234:235] /*v[746:747]*/, v[238:239], v[234:235] /*v[746:747]*/
	s_set_vgpr_msb 0x8880
	v_pk_add_f32 v[242:243] /*v[754:755]*/, v[214:215], v[220:221]
	s_set_vgpr_msb 0x808b
	v_pk_add_f32 v[238:239] /*v[750:751]*/, v[236:237] /*v[1004:1005]*/, v[238:239] /*v[750:751]*/
	s_set_vgpr_msb 0x8b88
	v_pk_add_f32 v[240:241] /*v[752:753]*/, v[202:203], v[240:241] /*v[752:753]*/
	v_pk_add_f32 v[244:245] /*v[756:757]*/, v[242:243], v[244:245] /*v[756:757]*/
	s_set_vgpr_msb 0x888f
	v_pk_add_f32 v[248:249] /*v[760:761]*/, v[222:223] /*v[990:991]*/, v[246:247] /*v[1014:1015]*/
	s_set_vgpr_msb 0x8f8b
	v_pk_add_f32 v[246:247] /*v[758:759]*/, v[218:219] /*v[986:987]*/, v[246:247] /*v[758:759]*/
	s_set_vgpr_msb 0x8b80
	v_pk_add_f32 v[254:255] /*v[766:767]*/, v[244:245], v[246:247]
	s_set_vgpr_msb 0x8088
	v_pk_add_f32 v[250:251] /*v[762:763]*/, v[216:217], v[250:251] /*v[762:763]*/
	v_pk_add_f32 v[252:253] /*v[764:765]*/, v[236:237], v[252:253] /*v[764:765]*/
	s_set_vgpr_msb 0x8845
	v_pk_add_f32 v[22:23] /*v[278:279]*/, v[22:23] /*v[278:279]*/, v[24:25] /*v[280:281]*/
	v_pk_add_f32 v[24:25] /*v[280:281]*/, v[28:29] /*v[284:285]*/, v[30:31] /*v[286:287]*/
	v_pk_add_f32 v[16:17] /*v[272:273]*/, v[18:19] /*v[274:275]*/, v[16:17] /*v[272:273]*/
	s_set_vgpr_msb 0x4549
	v_pk_add_f32 v[18:19] /*v[274:275]*/, v[54:55] /*v[310:311]*/, v[192:193] /*v[704:705]*/
	s_set_vgpr_msb 0x494a
	v_pk_add_f32 v[28:29] /*v[284:285]*/, v[196:197] /*v[708:709]*/, v[198:199] /*v[710:711]*/
	s_set_vgpr_msb 0x4a8b
	v_pk_add_f32 v[206:207] /*v[718:719]*/, v[114:115] /*v[882:883]*/, v[206:207] /*v[718:719]*/
	v_pk_add_f32 v[212:213] /*v[724:725]*/, v[254:255] /*v[1022:1023]*/, v[212:213] /*v[724:725]*/
	s_set_vgpr_msb 0x8b80
	v_pk_add_f32 v[224:225] /*v[736:737]*/, v[210:211], v[224:225]
	s_set_vgpr_msb 0x808b
	v_pk_add_f32 v[218:219] /*v[730:731]*/, v[204:205] /*v[972:973]*/, v[218:219] /*v[730:731]*/
	v_pk_add_f32 v[220:221] /*v[732:733]*/, v[240:241] /*v[1008:1009]*/, v[220:221] /*v[732:733]*/
	s_set_vgpr_msb 0x8b88
	v_pk_add_f32 v[222:223] /*v[734:735]*/, v[208:209], v[222:223] /*v[734:735]*/
	v_pk_add_f32 v[230:231] /*v[742:743]*/, v[192:193], v[230:231] /*v[742:743]*/
	s_set_vgpr_msb 0x888b
	v_pk_add_f32 v[236:237] /*v[748:749]*/, v[190:191] /*v[958:959]*/, v[236:237] /*v[748:749]*/
	s_set_vgpr_msb 0x8b88
	v_pk_add_f32 v[242:243] /*v[754:755]*/, v[230:231], v[242:243] /*v[754:755]*/
	s_set_vgpr_msb 0x888b
	v_pk_add_f32 v[248:249] /*v[760:761]*/, v[248:249] /*v[1016:1017]*/, v[248:249] /*v[760:761]*/
	s_set_vgpr_msb 0x8b48
	v_pk_add_f32 v[20:21] /*v[276:277]*/, v[250:251], v[254:255] /*v[766:767]*/
	s_set_vgpr_msb 0x4845
	v_pk_add_f32 v[22:23] /*v[278:279]*/, v[26:27] /*v[282:283]*/, v[22:23] /*v[278:279]*/
	v_pk_add_f32 v[24:25] /*v[280:281]*/, v[52:53] /*v[308:309]*/, v[24:25] /*v[280:281]*/
	s_set_vgpr_msb 0x454a
	v_pk_add_f32 v[26:27] /*v[282:283]*/, v[202:203] /*v[714:715]*/, v[204:205] /*v[716:717]*/
	s_set_vgpr_msb 0x4a46
	v_pk_add_f32 v[18:19] /*v[274:275]*/, v[194:195] /*v[706:707]*/, v[18:19] /*v[274:275]*/
	v_pk_add_f32 v[28:29] /*v[284:285]*/, v[200:201] /*v[712:713]*/, v[28:29] /*v[284:285]*/
	s_set_vgpr_msb 0x464a
	v_pk_add_f32 v[30:31] /*v[286:287]*/, v[208:209] /*v[720:721]*/, v[210:211] /*v[722:723]*/
	v_pk_add_f32 v[52:53] /*v[308:309]*/, v[214:215] /*v[726:727]*/, v[216:217] /*v[728:729]*/
	s_set_vgpr_msb 0x4a8a
	v_pk_add_f32 v[192:193] /*v[704:705]*/, v[226:227] /*v[738:739]*/, v[228:229] /*v[740:741]*/
	v_pk_add_f32 v[194:195] /*v[706:707]*/, v[232:233] /*v[744:745]*/, v[234:235] /*v[746:747]*/
	v_pk_add_f32 v[196:197] /*v[708:709]*/, v[238:239] /*v[750:751]*/, v[240:241] /*v[752:753]*/
	v_pk_add_f32 v[198:199] /*v[710:711]*/, v[244:245] /*v[756:757]*/, v[246:247] /*v[758:759]*/
	v_pk_add_f32 v[200:201] /*v[712:713]*/, v[250:251] /*v[762:763]*/, v[252:253] /*v[764:765]*/
	s_set_vgpr_msb 0x8a00
	v_exp_f32_e32 v252, v252
	s_set_vgpr_msb 0x8b
	v_pk_add_f32 v[224:225] /*v[736:737]*/, v[112:113] /*v[880:881]*/, v[224:225] /*v[736:737]*/
	s_set_vgpr_msb 0x8b46
	v_pk_add_f32 v[26:27] /*v[282:283]*/, v[206:207] /*v[718:719]*/, v[26:27] /*v[282:283]*/
	s_set_vgpr_msb 0x464a
	v_pk_add_f32 v[54:55] /*v[310:311]*/, v[220:221] /*v[732:733]*/, v[222:223] /*v[734:735]*/
	s_set_vgpr_msb 0x4a46
	v_pk_add_f32 v[30:31] /*v[286:287]*/, v[212:213] /*v[724:725]*/, v[30:31] /*v[286:287]*/
	v_pk_add_f32 v[52:53] /*v[308:309]*/, v[218:219] /*v[730:731]*/, v[52:53] /*v[308:309]*/
	s_set_vgpr_msb 0x468a
	v_pk_add_f32 v[192:193] /*v[704:705]*/, v[230:231] /*v[742:743]*/, v[192:193] /*v[704:705]*/
	v_pk_add_f32 v[194:195] /*v[706:707]*/, v[236:237] /*v[748:749]*/, v[194:195] /*v[706:707]*/
	s_set_vgpr_msb 0x8a45
	v_pk_add_f32 v[16:17] /*v[272:273]*/, v[16:17] /*v[272:273]*/, v[22:23] /*v[278:279]*/
	s_set_vgpr_msb 0x454a
	v_pk_add_f32 v[22:23] /*v[278:279]*/, v[242:243] /*v[754:755]*/, v[196:197] /*v[708:709]*/
	s_set_vgpr_msb 0x4a8a
	v_pk_add_f32 v[196:197] /*v[708:709]*/, v[248:249] /*v[760:761]*/, v[198:199] /*v[710:711]*/
	s_set_vgpr_msb 0x8a45
	v_pk_add_f32 v[18:19] /*v[274:275]*/, v[18:19] /*v[274:275]*/, v[28:29] /*v[284:285]*/
	s_set_vgpr_msb 0x4549
	v_pk_add_f32 v[20:21] /*v[276:277]*/, v[20:21] /*v[276:277]*/, v[200:201] /*v[712:713]*/
	s_set_vgpr_msb 0x4980
	v_pk_add_f32 v[254:255] /*v[766:767]*/, v[252:253], v[254:255]
	s_set_vgpr_msb 0x8046
	v_pk_add_f32 v[54:55] /*v[310:311]*/, v[224:225] /*v[736:737]*/, v[54:55] /*v[310:311]*/
	s_set_vgpr_msb 0x4645
	v_pk_add_f32 v[16:17] /*v[272:273]*/, v[24:25] /*v[280:281]*/, v[16:17] /*v[272:273]*/
	v_pk_add_f32 v[24:25] /*v[280:281]*/, v[30:31] /*v[286:287]*/, v[52:53] /*v[308:309]*/
	s_set_vgpr_msb 0x454a
	v_pk_add_f32 v[28:29] /*v[284:285]*/, v[192:193] /*v[704:705]*/, v[194:195] /*v[706:707]*/
	s_set_vgpr_msb 0x4a45
	v_pk_add_f32 v[18:19] /*v[274:275]*/, v[26:27] /*v[282:283]*/, v[18:19] /*v[274:275]*/
	s_set_vgpr_msb 0x4546
	v_pk_add_f32 v[20:21] /*v[276:277]*/, v[196:197] /*v[708:709]*/, v[20:21] /*v[276:277]*/
	s_set_vgpr_msb 0x4600
	v_mul_f32_e32 v130, 0x3fb8aa3b, v130
	s_set_vgpr_msb 0x45
	v_pk_add_f32 v[24:25] /*v[280:281]*/, v[54:55] /*v[310:311]*/, v[24:25] /*v[280:281]*/
	v_pk_add_f32 v[22:23] /*v[278:279]*/, v[22:23] /*v[278:279]*/, v[28:29] /*v[284:285]*/
	v_pk_add_f32 v[16:17] /*v[272:273]*/, v[16:17] /*v[272:273]*/, v[18:19] /*v[274:275]*/
	s_set_vgpr_msb 0x4546
	v_pk_add_f32 v[18:19] /*v[274:275]*/, v[254:255] /*v[766:767]*/, v[20:21] /*v[276:277]*/
	s_set_vgpr_msb 0x4680
	v_exp_f32_e32 v196 /*v708*/, v130
	s_set_vgpr_msb 0x8045
	v_pk_add_f32 v[16:17] /*v[272:273]*/, v[24:25] /*v[280:281]*/, v[16:17] /*v[272:273]*/
	v_pk_add_f32 v[18:19] /*v[274:275]*/, v[22:23] /*v[278:279]*/, v[18:19] /*v[274:275]*/
	s_set_vgpr_msb 0x4585
	v_pk_add_f32 v[192:193] /*v[704:705]*/, v[18:19] /*v[274:275]*/, v[16:17] /*v[272:273]*/
	s_set_vgpr_msb 0x8582
	v_dual_mov_b32 v194 /*v706*/, v192 /*v704*/ :: v_dual_mov_b32 v195 /*v707*/, v193 /*v705*/
	v_permlanex16_b32 v194 /*v706*/, v194 /*v706*/, s96, 0xfedcba98
	v_permlanex16_b32 v195 /*v707*/, v195 /*v707*/, s96, 0xfedcba98
	s_set_vgpr_msb 0x8200
	s_cbranch_vccz .LBB0_8
	s_set_vgpr_msb 8
	v_pk_mul_f32 v[126:127], v[126:127], v[196:197] /*v[708:709]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[124:125], v[124:125], v[196:197] /*v[708:709]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[122:123], v[122:123], v[196:197] /*v[708:709]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[120:121], v[120:121], v[196:197] /*v[708:709]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[118:119], v[118:119], v[196:197] /*v[708:709]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[116:117], v[116:117], v[196:197] /*v[708:709]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[114:115], v[114:115], v[196:197] /*v[708:709]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[112:113], v[112:113], v[196:197] /*v[708:709]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[110:111], v[110:111], v[196:197] /*v[708:709]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[108:109], v[108:109], v[196:197] /*v[708:709]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[106:107], v[106:107], v[196:197] /*v[708:709]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[104:105], v[104:105], v[196:197] /*v[708:709]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[102:103], v[102:103], v[196:197] /*v[708:709]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[100:101], v[100:101], v[196:197] /*v[708:709]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[98:99], v[98:99], v[196:197] /*v[708:709]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[96:97], v[96:97], v[196:197] /*v[708:709]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[94:95], v[94:95], v[196:197] /*v[708:709]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[92:93], v[92:93], v[196:197] /*v[708:709]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[90:91], v[90:91], v[196:197] /*v[708:709]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[88:89], v[88:89], v[196:197] /*v[708:709]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[86:87], v[86:87], v[196:197] /*v[708:709]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[84:85], v[84:85], v[196:197] /*v[708:709]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[82:83], v[82:83], v[196:197] /*v[708:709]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[80:81], v[80:81], v[196:197] /*v[708:709]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[78:79], v[78:79], v[196:197] /*v[708:709]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[76:77], v[76:77], v[196:197] /*v[708:709]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[74:75], v[74:75], v[196:197] /*v[708:709]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[72:73], v[72:73], v[196:197] /*v[708:709]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[70:71], v[70:71], v[196:197] /*v[708:709]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[68:69], v[68:69], v[196:197] /*v[708:709]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[66:67], v[66:67], v[196:197] /*v[708:709]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[64:65], v[64:65], v[196:197] /*v[708:709]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x800
.LBB0_8:
	s_set_vgpr_msb 1
	v_sub_f32_e32 v130, v50 /*v306*/, v134
	s_and_b32 s2, s5, exec_lo
	s_cselect_b32 s2, 1, 0
	s_cmp_lg_u32 s2, 1
	v_mul_f32_e32 v130, 0x3fb8aa3b, v130
	s_set_vgpr_msb 0x180
	v_exp_f32_e32 v197 /*v709*/, v130
	s_set_vgpr_msb 0x8000
	s_cbranch_scc1 .LBB0_3
	v_nop
	s_set_vgpr_msb 0x42
	v_mov_b32_e32 v16 /*v272*/, v197 /*v709*/
	s_set_vgpr_msb 0x4204
	v_pk_mul_f32 v[62:63], v[62:63], v[16:17] /*v[272:273]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[60:61], v[60:61], v[16:17] /*v[272:273]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59], v[58:59], v[16:17] /*v[272:273]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[56:57], v[56:57], v[16:17] /*v[272:273]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[54:55], v[54:55], v[16:17] /*v[272:273]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[52:53], v[52:53], v[16:17] /*v[272:273]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[50:51], v[50:51], v[16:17] /*v[272:273]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[48:49], v[48:49], v[16:17] /*v[272:273]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[46:47], v[46:47], v[16:17] /*v[272:273]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[44:45], v[44:45], v[16:17] /*v[272:273]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[42:43], v[42:43], v[16:17] /*v[272:273]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[40:41], v[40:41], v[16:17] /*v[272:273]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39], v[38:39], v[16:17] /*v[272:273]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[36:37], v[36:37], v[16:17] /*v[272:273]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35], v[34:35], v[16:17] /*v[272:273]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[32:33], v[32:33], v[16:17] /*v[272:273]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[30:31], v[30:31], v[16:17] /*v[272:273]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[28:29], v[28:29], v[16:17] /*v[272:273]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[26:27], v[26:27], v[16:17] /*v[272:273]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25], v[24:25], v[16:17] /*v[272:273]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[22:23], v[22:23], v[16:17] /*v[272:273]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21], v[20:21], v[16:17] /*v[272:273]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[18:19], v[18:19], v[16:17] /*v[272:273]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[16:17], v[16:17], v[16:17] /*v[272:273]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[14:15], v[16:17] /*v[272:273]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13], v[12:13], v[16:17] /*v[272:273]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[10:11], v[16:17] /*v[272:273]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[8:9], v[16:17] /*v[272:273]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[6:7], v[16:17] /*v[272:273]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[4:5], v[16:17] /*v[272:273]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[2:3], v[2:3], v[16:17] /*v[272:273]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[0:1], v[0:1], v[16:17] /*v[272:273]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x400
	s_branch .LBB0_3
.LBB0_10:
	s_and_b32 vcc_lo, exec_lo, s18
	s_cbranch_vccnz .LBB0_24
	s_branch .LBB0_46
.LBB0_11:
	v_dual_mov_b32 v1, v0 :: v_dual_mov_b32 v2, v0
	v_dual_mov_b32 v3, v0 :: v_dual_mov_b32 v4, v0
	v_dual_mov_b32 v5, v0 :: v_dual_mov_b32 v6, v0
	v_dual_mov_b32 v7, v0 :: v_dual_mov_b32 v134, 0xf149f2ca
	v_dual_mov_b32 v205, v0 :: v_dual_mov_b32 v204, v0
	s_set_vgpr_msb 64
	v_mov_b32_e32 v41 /*v297*/, v135
	s_set_vgpr_msb 0x4000
	v_dual_mov_b32 v133, v194 :: v_dual_mov_b32 v128, 0xf149f2ca
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
	s_cmp_ge_u32 s52, s10
	s_cbranch_scc0 .LBB0_14
.LBB0_12:
	s_set_vgpr_msb 64
	v_dual_mov_b32 v51 /*v307*/, v134 :: v_dual_mov_b32 v50 /*v306*/, v128
	s_set_vgpr_msb 0x4000
	s_branch .LBB0_23
.LBB0_13:
	s_clause 0x6
	scratch_load_b32 v192, off, off offset:564 nv
	scratch_load_b32 v193, off, off offset:568 nv
	scratch_load_b32 v194, off, off offset:584 nv
	s_set_vgpr_msb 0xc0
	scratch_load_b32 v200 /*v968*/, off, off offset:8 nv
	s_set_vgpr_msb 0xc000
	scratch_load_b32 v130, off, off offset:524 nv
	s_cmp_ge_u32 s52, s10
	s_cbranch_scc1 .LBB0_12
.LBB0_14:
	s_wait_loadcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	v_dual_add_nc_u32 v130, s11, v192 :: v_dual_add_nc_u32 v131, s11, v193
	s_add_co_i32 s2, s58, -1
	s_mov_b32 s43, 0
	s_mov_b32 s4, 1
	s_set_vgpr_msb 64
	v_min_i32_e32 v52 /*v308*/, s2, v130
	v_min_i32_e32 v53 /*v309*/, s2, v131
	s_mov_b32 s11, s43
	s_mov_b32 s40, 64
	s_mov_b32 s39, 0x800000
	s_mov_b32 s37, 0xffff0000
	s_mov_b32 s36, 0x7510000
	s_mov_b32 s44, 0xf510000
	s_mov_b32 s82, 0x76543210
	s_mov_b32 s56, 0x3fb8aa3b
	s_set_vgpr_msb 0x4000
	s_branch .LBB0_16
.LBB0_15:
	s_set_vgpr_msb 15
	v_cvt_pk_bf16_f32 v201, v16 /*v784*/, v32 /*v800*/
	s_set_vgpr_msb 0xf0e
	v_cvt_pk_bf16_f32 v200, v252 /*v764*/, v10 /*v778*/
	s_set_vgpr_msb 0xe0a
	v_cvt_pk_bf16_f32 v199, v236 /*v748*/, v248 /*v760*/
	v_cvt_pk_bf16_f32 v198, v222 /*v734*/, v232 /*v744*/
	v_cvt_pk_bf16_f32 v197, v210 /*v722*/, v218 /*v730*/
	v_cvt_pk_bf16_f32 v196, v202 /*v714*/, v208 /*v720*/
	v_cvt_pk_bf16_f32 v195, v196 /*v708*/, v200 /*v712*/
	v_cvt_pk_bf16_f32 v194, v192 /*v704*/, v194 /*v706*/
	s_set_vgpr_msb 0xa0f
	v_cvt_pk_bf16_f32 v209, v17 /*v785*/, v33 /*v801*/
	s_set_vgpr_msb 0xf0e
	v_cvt_pk_bf16_f32 v208, v253 /*v765*/, v11 /*v779*/
	s_set_vgpr_msb 0xe0a
	v_cvt_pk_bf16_f32 v207, v237 /*v749*/, v249 /*v761*/
	v_cvt_pk_bf16_f32 v206, v223 /*v735*/, v233 /*v745*/
	v_cvt_pk_bf16_f32 v205, v211 /*v723*/, v219 /*v731*/
	v_cvt_pk_bf16_f32 v204, v203 /*v715*/, v209 /*v721*/
	v_cvt_pk_bf16_f32 v203, v197 /*v709*/, v201 /*v713*/
	v_cvt_pk_bf16_f32 v202, v193 /*v705*/, v195 /*v707*/
	s_set_vgpr_msb 0xa01
	v_wmma_f32_16x16x32_bf16 v[72:79], v[72:79] /*v[328:335]*/, v[194:201], v[72:79]
	s_set_vgpr_msb 0x10f
	v_cvt_pk_bf16_f32 v217, v48 /*v816*/, v64 /*v832*/
	v_cvt_pk_bf16_f32 v216, v26 /*v794*/, v42 /*v810*/
	v_cvt_pk_bf16_f32 v215, v6 /*v774*/, v22 /*v790*/
	s_set_vgpr_msb 0xf0e
	v_cvt_pk_bf16_f32 v214, v244 /*v756*/, v2 /*v770*/
	s_set_vgpr_msb 0xe0a
	v_cvt_pk_bf16_f32 v213, v228 /*v740*/, v240 /*v752*/
	v_cvt_pk_bf16_f32 v212, v216 /*v728*/, v226 /*v738*/
	v_cvt_pk_bf16_f32 v211, v206 /*v718*/, v214 /*v726*/
	s_set_vgpr_msb 0xa01
	v_wmma_f32_16x16x32_bf16 v[8:15], v[72:79] /*v[328:335]*/, v[202:209], v[8:15]
	s_set_vgpr_msb 0x10a
	v_cvt_pk_bf16_f32 v210, v198 /*v710*/, v204 /*v716*/
	s_set_vgpr_msb 0xa0f
	v_cvt_pk_bf16_f32 v225, v49 /*v817*/, v65 /*v833*/
	v_cvt_pk_bf16_f32 v224, v27 /*v795*/, v43 /*v811*/
	v_cvt_pk_bf16_f32 v223, v7 /*v775*/, v23 /*v791*/
	s_set_vgpr_msb 0xf0e
	v_cvt_pk_bf16_f32 v222, v245 /*v757*/, v3 /*v771*/
	s_set_vgpr_msb 0xe0a
	v_cvt_pk_bf16_f32 v221, v229 /*v741*/, v241 /*v753*/
	v_cvt_pk_bf16_f32 v220, v217 /*v729*/, v227 /*v739*/
	v_cvt_pk_bf16_f32 v219, v207 /*v719*/, v215 /*v727*/
	v_cvt_pk_bf16_f32 v218, v199 /*v711*/, v205 /*v717*/
	s_set_vgpr_msb 0xa01
	v_wmma_f32_16x16x32_bf16 v[72:79], v[56:63] /*v[312:319]*/, v[210:217], v[72:79]
	s_set_vgpr_msb 0x10f
	v_cvt_pk_bf16_f32 v233, v80 /*v848*/, v96 /*v864*/
	v_cvt_pk_bf16_f32 v232, v58 /*v826*/, v74 /*v842*/
	v_cvt_pk_bf16_f32 v231, v38 /*v806*/, v54 /*v822*/
	v_cvt_pk_bf16_f32 v230, v18 /*v786*/, v34 /*v802*/
	s_set_vgpr_msb 0xf0e
	v_cvt_pk_bf16_f32 v229, v254 /*v766*/, v12 /*v780*/
	s_set_vgpr_msb 0xe0a
	v_cvt_pk_bf16_f32 v228, v238 /*v750*/, v250 /*v762*/
	v_cvt_pk_bf16_f32 v227, v224 /*v736*/, v234 /*v746*/
	s_set_vgpr_msb 0xa01
	v_wmma_f32_16x16x32_bf16 v[8:15], v[56:63] /*v[312:319]*/, v[218:225], v[8:15]
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x140
	s_clause 0x1
	scratch_load_b128 v[56:59] /*v[312:315]*/, off, off offset:428 th:TH_LOAD_LU nv
	scratch_load_b128 v[60:63] /*v[316:319]*/, off, off offset:444 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0x400a
	v_cvt_pk_bf16_f32 v226, v212 /*v724*/, v220 /*v732*/
	s_set_vgpr_msb 0xa0f
	v_cvt_pk_bf16_f32 v241, v81 /*v849*/, v97 /*v865*/
	v_cvt_pk_bf16_f32 v240, v59 /*v827*/, v75 /*v843*/
	v_cvt_pk_bf16_f32 v239, v39 /*v807*/, v55 /*v823*/
	v_cvt_pk_bf16_f32 v238, v19 /*v787*/, v35 /*v803*/
	s_set_vgpr_msb 0xf0e
	v_cvt_pk_bf16_f32 v237, v255 /*v767*/, v13 /*v781*/
	s_set_vgpr_msb 0xe0a
	v_cvt_pk_bf16_f32 v236, v239 /*v751*/, v251 /*v763*/
	v_cvt_pk_bf16_f32 v235, v225 /*v737*/, v235 /*v747*/
	v_cvt_pk_bf16_f32 v234, v213 /*v725*/, v221 /*v733*/
	s_set_vgpr_msb 0xa0f
	v_cvt_pk_bf16_f32 v249, v112 /*v880*/, v126 /*v894*/
	v_cvt_pk_bf16_f32 v248, v90 /*v858*/, v106 /*v874*/
	v_cvt_pk_bf16_f32 v247, v70 /*v838*/, v86 /*v854*/
	v_cvt_pk_bf16_f32 v246, v50 /*v818*/, v66 /*v834*/
	v_cvt_pk_bf16_f32 v245, v28 /*v796*/, v44 /*v812*/
	v_cvt_pk_bf16_f32 v244, v8 /*v776*/, v24 /*v792*/
	s_set_vgpr_msb 0xf0e
	v_cvt_pk_bf16_f32 v243, v246 /*v758*/, v4 /*v772*/
	s_set_vgpr_msb 0xe0a
	v_cvt_pk_bf16_f32 v242, v230 /*v742*/, v242 /*v754*/
	s_set_vgpr_msb 0xa4f
	v_cvt_pk_bf16_f32 v1 /*v257*/, v113 /*v881*/, v127 /*v895*/
	v_cvt_pk_bf16_f32 v0 /*v256*/, v91 /*v859*/, v107 /*v875*/
	s_set_vgpr_msb 0x4f0f
	v_cvt_pk_bf16_f32 v255, v71 /*v839*/, v87 /*v855*/
	v_cvt_pk_bf16_f32 v254, v51 /*v819*/, v67 /*v835*/
	v_cvt_pk_bf16_f32 v253, v29 /*v797*/, v45 /*v813*/
	v_cvt_pk_bf16_f32 v252, v9 /*v777*/, v25 /*v793*/
	s_set_vgpr_msb 0xf0e
	v_cvt_pk_bf16_f32 v251, v247 /*v759*/, v5 /*v773*/
	s_set_vgpr_msb 0xe0a
	v_cvt_pk_bf16_f32 v250, v231 /*v743*/, v243 /*v755*/
	s_set_vgpr_msb 0xa4f
	v_cvt_pk_bf16_f32 v9 /*v265*/, v140 /*v908*/, v152 /*v920*/
	v_cvt_pk_bf16_f32 v8 /*v264*/, v122 /*v890*/, v136 /*v904*/
	v_cvt_pk_bf16_f32 v7 /*v263*/, v102 /*v870*/, v118 /*v886*/
	v_cvt_pk_bf16_f32 v6 /*v262*/, v82 /*v850*/, v98 /*v866*/
	v_cvt_pk_bf16_f32 v5 /*v261*/, v60 /*v828*/, v76 /*v844*/
	v_cvt_pk_bf16_f32 v4 /*v260*/, v40 /*v808*/, v56 /*v824*/
	v_cvt_pk_bf16_f32 v3 /*v259*/, v20 /*v788*/, v36 /*v804*/
	v_cvt_pk_bf16_f32 v2 /*v258*/, v0 /*v768*/, v14 /*v782*/
	v_cvt_pk_bf16_f32 v17 /*v273*/, v141 /*v909*/, v153 /*v921*/
	v_cvt_pk_bf16_f32 v16 /*v272*/, v123 /*v891*/, v137 /*v905*/
	v_cvt_pk_bf16_f32 v15 /*v271*/, v103 /*v871*/, v119 /*v887*/
	v_cvt_pk_bf16_f32 v14 /*v270*/, v83 /*v851*/, v99 /*v867*/
	v_cvt_pk_bf16_f32 v13 /*v269*/, v61 /*v829*/, v77 /*v845*/
	v_cvt_pk_bf16_f32 v12 /*v268*/, v41 /*v809*/, v57 /*v825*/
	v_cvt_pk_bf16_f32 v11 /*v267*/, v21 /*v789*/, v37 /*v805*/
	v_cvt_pk_bf16_f32 v10 /*v266*/, v1 /*v769*/, v15 /*v783*/
	v_cvt_pk_bf16_f32 v25 /*v281*/, v162 /*v930*/, v170 /*v938*/
	v_cvt_pk_bf16_f32 v24 /*v280*/, v148 /*v916*/, v158 /*v926*/
	v_cvt_pk_bf16_f32 v23 /*v279*/, v132 /*v900*/, v144 /*v912*/
	v_cvt_pk_bf16_f32 v22 /*v278*/, v114 /*v882*/, v128 /*v896*/
	v_cvt_pk_bf16_f32 v21 /*v277*/, v92 /*v860*/, v108 /*v876*/
	v_cvt_pk_bf16_f32 v20 /*v276*/, v72 /*v840*/, v88 /*v856*/
	v_cvt_pk_bf16_f32 v19 /*v275*/, v52 /*v820*/, v68 /*v836*/
	v_cvt_pk_bf16_f32 v18 /*v274*/, v30 /*v798*/, v46 /*v814*/
	v_cvt_pk_bf16_f32 v33 /*v289*/, v163 /*v931*/, v171 /*v939*/
	v_cvt_pk_bf16_f32 v32 /*v288*/, v149 /*v917*/, v159 /*v927*/
	v_cvt_pk_bf16_f32 v31 /*v287*/, v133 /*v901*/, v145 /*v913*/
	v_cvt_pk_bf16_f32 v30 /*v286*/, v115 /*v883*/, v129 /*v897*/
	v_cvt_pk_bf16_f32 v29 /*v285*/, v93 /*v861*/, v109 /*v877*/
	v_cvt_pk_bf16_f32 v28 /*v284*/, v73 /*v841*/, v89 /*v857*/
	v_cvt_pk_bf16_f32 v27 /*v283*/, v53 /*v821*/, v69 /*v837*/
	v_cvt_pk_bf16_f32 v26 /*v282*/, v31 /*v799*/, v47 /*v815*/
	s_set_vgpr_msb 0x4f02
	v_wmma_f32_16x16x32_bf16 v[120:127], v[184:191] /*v[696:703]*/, v[194:201], v[120:127]
	s_set_vgpr_msb 0x24f
	v_cvt_pk_bf16_f32 v41 /*v297*/, v180 /*v948*/, v186 /*v954*/
	v_cvt_pk_bf16_f32 v40 /*v296*/, v168 /*v936*/, v178 /*v946*/
	v_cvt_pk_bf16_f32 v39 /*v295*/, v156 /*v924*/, v166 /*v934*/
	v_cvt_pk_bf16_f32 v38 /*v294*/, v142 /*v910*/, v154 /*v922*/
	v_cvt_pk_bf16_f32 v37 /*v293*/, v124 /*v892*/, v138 /*v906*/
	v_cvt_pk_bf16_f32 v36 /*v292*/, v104 /*v872*/, v120 /*v888*/
	v_cvt_pk_bf16_f32 v35 /*v291*/, v84 /*v852*/, v100 /*v868*/
	s_set_vgpr_msb 0x4f02
	v_wmma_f32_16x16x32_bf16 v[56:63], v[184:191] /*v[696:703]*/, v[202:209], v[56:63]
	s_set_vgpr_msb 0x24f
	v_cvt_pk_bf16_f32 v34 /*v290*/, v62 /*v830*/, v78 /*v846*/
	s_add_nc_u64 s[52:53], s[52:53], 1
	s_set_vgpr_msb 0x4f01
	v_mov_b32_e32 v134, v51 /*v307*/
	v_cmp_ge_u64_e64 s2, s[52:53], s[10:11]
	s_and_b32 vcc_lo, exec_lo, s2
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[72:79], v[56:63] /*v[312:319]*/, v[226:233], v[72:79]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[56:63] /*v[312:319]*/, v[234:241], v[8:15]
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x140
	s_clause 0x1
	scratch_load_b128 v[56:59] /*v[312:315]*/, off, off offset:396 th:TH_LOAD_LU nv
	scratch_load_b128 v[60:63] /*v[316:319]*/, off, off offset:412 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0x4001
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[72:79], v[56:63] /*v[312:319]*/, v[242:249], v[72:79]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[56:63] /*v[312:319]*/, v[250:257], v[8:15]
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x140
	s_clause 0x1
	scratch_load_b128 v[56:59] /*v[312:315]*/, off, off offset:364 th:TH_LOAD_LU nv
	scratch_load_b128 v[60:63] /*v[316:319]*/, off, off offset:380 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0x4005
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[72:79], v[56:63] /*v[312:319]*/, v[2:9] /*v[258:265]*/, v[72:79]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[56:63] /*v[312:319]*/, v[10:17] /*v[266:273]*/, v[8:15]
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x540
	s_clause 0x1
	scratch_load_b128 v[56:59] /*v[312:315]*/, off, off offset:332 th:TH_LOAD_LU nv
	scratch_load_b128 v[60:63] /*v[316:319]*/, off, off offset:348 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0x4005
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[72:79], v[56:63] /*v[312:319]*/, v[18:25] /*v[274:281]*/, v[72:79]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[56:63] /*v[312:319]*/, v[26:33] /*v[282:289]*/, v[8:15]
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x540
	s_clause 0x1
	scratch_load_b128 v[56:59] /*v[312:315]*/, off, off offset:300 th:TH_LOAD_LU nv
	scratch_load_b128 v[60:63] /*v[316:319]*/, off, off offset:316 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0x4002
	v_wmma_f32_16x16x32_bf16 v[120:127], v[176:183] /*v[688:695]*/, v[210:217], v[120:127]
	v_wmma_f32_16x16x32_bf16 v[56:63], v[176:183] /*v[688:695]*/, v[218:225], v[56:63]
	v_wmma_f32_16x16x32_bf16 v[120:127], v[168:175] /*v[680:687]*/, v[226:233], v[120:127]
	v_wmma_f32_16x16x32_bf16 v[56:63], v[168:175] /*v[680:687]*/, v[234:241], v[56:63]
	v_wmma_f32_16x16x32_bf16 v[120:127], v[160:167] /*v[672:679]*/, v[242:249], v[120:127]
	v_wmma_f32_16x16x32_bf16 v[56:63], v[160:167] /*v[672:679]*/, v[250:257], v[56:63]
	s_set_vgpr_msb 0x206
	v_wmma_f32_16x16x32_bf16 v[120:127], v[152:159] /*v[664:671]*/, v[2:9] /*v[258:265]*/, v[120:127]
	v_wmma_f32_16x16x32_bf16 v[56:63], v[152:159] /*v[664:671]*/, v[10:17] /*v[266:273]*/, v[56:63]
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x68f
	v_cvt_pk_bf16_f32 v159 /*v671*/, v189 /*v957*/, v191 /*v959*/
	v_cvt_pk_bf16_f32 v158 /*v670*/, v185 /*v953*/, v177 /*v945*/
	v_cvt_pk_bf16_f32 v157 /*v669*/, v175 /*v943*/, v183 /*v951*/
	s_set_vgpr_msb 0x8f06
	v_wmma_f32_16x16x32_bf16 v[120:127], v[144:151] /*v[656:663]*/, v[18:25] /*v[274:281]*/, v[120:127]
	s_set_vgpr_msb 0x68f
	v_cvt_pk_bf16_f32 v156 /*v668*/, v165 /*v933*/, v173 /*v941*/
	v_cvt_pk_bf16_f32 v155 /*v667*/, v151 /*v919*/, v161 /*v929*/
	v_cvt_pk_bf16_f32 v154 /*v666*/, v135 /*v903*/, v147 /*v915*/
	v_cvt_pk_bf16_f32 v153 /*v665*/, v117 /*v885*/, v131 /*v899*/
	v_cvt_pk_bf16_f32 v152 /*v664*/, v95 /*v863*/, v111 /*v879*/
	s_set_vgpr_msb 0x8f06
	v_wmma_f32_16x16x32_bf16 v[56:63], v[144:151] /*v[656:663]*/, v[26:33] /*v[282:289]*/, v[56:63]
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x68f
	v_cvt_pk_bf16_f32 v151 /*v663*/, v181 /*v949*/, v187 /*v955*/
	v_cvt_pk_bf16_f32 v150 /*v662*/, v169 /*v937*/, v179 /*v947*/
	v_cvt_pk_bf16_f32 v149 /*v661*/, v157 /*v925*/, v167 /*v935*/
	v_cvt_pk_bf16_f32 v148 /*v660*/, v143 /*v911*/, v155 /*v923*/
	v_cvt_pk_bf16_f32 v147 /*v659*/, v125 /*v893*/, v139 /*v907*/
	v_cvt_pk_bf16_f32 v146 /*v658*/, v105 /*v873*/, v121 /*v889*/
	v_cvt_pk_bf16_f32 v145 /*v657*/, v85 /*v853*/, v101 /*v869*/
	v_cvt_pk_bf16_f32 v144 /*v656*/, v63 /*v831*/, v79 /*v847*/
	s_set_vgpr_msb 0x8f05
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[72:79], v[56:63] /*v[312:319]*/, v[34:41] /*v[290:297]*/, v[72:79]
	s_set_vgpr_msb 0x509
	v_wmma_f32_16x16x32_bf16 v[8:15], v[56:63] /*v[312:319]*/, v[144:151] /*v[656:663]*/, v[8:15]
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x940
	s_clause 0x1
	scratch_load_b128 v[56:59] /*v[312:315]*/, off, off offset:268 th:TH_LOAD_LU nv
	scratch_load_b128 v[60:63] /*v[316:319]*/, off, off offset:284 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0x4006
	v_wmma_f32_16x16x32_bf16 v[120:127], v[136:143] /*v[648:655]*/, v[34:41] /*v[290:297]*/, v[120:127]
	s_set_vgpr_msb 0x60a
	v_wmma_f32_16x16x32_bf16 v[56:63], v[136:143] /*v[648:655]*/, v[144:151] /*v[656:663]*/, v[56:63]
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0xa8f
	v_cvt_pk_bf16_f32 v143 /*v655*/, v188 /*v956*/, v190 /*v958*/
	v_cvt_pk_bf16_f32 v142 /*v654*/, v184 /*v952*/, v176 /*v944*/
	v_cvt_pk_bf16_f32 v141 /*v653*/, v174 /*v942*/, v182 /*v950*/
	v_cvt_pk_bf16_f32 v140 /*v652*/, v164 /*v932*/, v172 /*v940*/
	v_cvt_pk_bf16_f32 v139 /*v651*/, v150 /*v918*/, v160 /*v928*/
	v_cvt_pk_bf16_f32 v138 /*v650*/, v134 /*v902*/, v146 /*v914*/
	v_cvt_pk_bf16_f32 v137 /*v649*/, v116 /*v884*/, v130 /*v898*/
	v_cvt_pk_bf16_f32 v136 /*v648*/, v94 /*v862*/, v110 /*v878*/
	s_set_vgpr_msb 0x8f09
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[72:79], v[56:63] /*v[312:319]*/, v[136:143] /*v[648:655]*/, v[72:79]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[56:63] /*v[312:319]*/, v[152:159] /*v[664:671]*/, v[8:15]
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x940
	s_clause 0x1
	scratch_load_b128 v[56:59] /*v[312:315]*/, off, off offset:236 th:TH_LOAD_LU nv
	scratch_load_b128 v[60:63] /*v[316:319]*/, off, off offset:252 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0x4002
	v_wmma_f32_16x16x32_bf16 v[112:119], v[104:111] /*v[616:623]*/, v[194:201], v[112:119]
	v_wmma_f32_16x16x32_bf16 v[104:111], v[72:79] /*v[584:591]*/, v[194:201], v[104:111]
	s_set_vgpr_msb 0x201
	v_wmma_f32_16x16x32_bf16 v[96:103], v[248:255] /*v[504:511]*/, v[194:201], v[96:103]
	v_wmma_f32_16x16x32_bf16 v[88:95], v[200:207] /*v[456:463]*/, v[194:201], v[88:95]
	v_wmma_f32_16x16x32_bf16 v[80:87], v[120:127] /*v[376:383]*/, v[194:201], v[80:87]
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[64:71], v[56:63] /*v[312:319]*/, v[194:201], v[64:71]
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b128 v[194:197], off, off offset:204 th:TH_LOAD_LU nv
	scratch_load_b128 v[198:201], off, off offset:220 th:TH_LOAD_LU nv
	v_wmma_f32_16x16x32_bf16 v[0:7], v[56:63] /*v[312:319]*/, v[202:209], v[0:7]
	s_set_vgpr_msb 0x100
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[64:71], v[194:201], v[210:217], v[64:71]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[194:201], v[218:225], v[0:7]
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b128 v[194:197], off, off offset:172 th:TH_LOAD_LU nv
	scratch_load_b128 v[198:201], off, off offset:188 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[64:71], v[194:201], v[226:233], v[64:71]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[194:201], v[234:241], v[0:7]
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b128 v[194:197], off, off offset:140 th:TH_LOAD_LU nv
	scratch_load_b128 v[198:201], off, off offset:156 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[64:71], v[194:201], v[242:249], v[64:71]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[194:201], v[250:257], v[0:7]
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b128 v[194:197], off, off offset:108 th:TH_LOAD_LU nv
	scratch_load_b128 v[198:201], off, off offset:124 th:TH_LOAD_LU nv
	s_set_vgpr_msb 4
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[64:71], v[194:201], v[2:9] /*v[258:265]*/, v[64:71]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[194:201], v[10:17] /*v[266:273]*/, v[0:7]
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b128 v[194:197], off, off offset:76 th:TH_LOAD_LU nv
	scratch_load_b128 v[198:201], off, off offset:92 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[64:71], v[194:201], v[18:25] /*v[274:281]*/, v[64:71]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[194:201], v[26:33] /*v[282:289]*/, v[0:7]
	s_wait_alu depctr_va_vdst(0)
	scratch_load_b64 v[194:195], off, off th:TH_LOAD_LU nv
	s_set_vgpr_msb 0x403
	s_wait_loadcnt 0x0
	v_fmac_f32_e32 v192, v196 /*v964*/, v195
	s_set_vgpr_msb 0x302
	v_wmma_f32_16x16x32_bf16 v[48:55], v[104:111] /*v[616:623]*/, v[202:209], v[48:55]
	s_set_vgpr_msb 0x203
	v_fmac_f32_e32 v128, v194 /*v962*/, v194
	s_set_vgpr_msb 0x302
	v_wmma_f32_16x16x32_bf16 v[40:47], v[72:79] /*v[584:591]*/, v[202:209], v[40:47]
	s_set_vgpr_msb 0x201
	v_wmma_f32_16x16x32_bf16 v[32:39], v[248:255] /*v[504:511]*/, v[202:209], v[32:39]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[200:207] /*v[456:463]*/, v[202:209], v[24:31]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[120:127] /*v[376:383]*/, v[202:209], v[16:23]
	scratch_load_b128 v[196:199], off, off offset:44 th:TH_LOAD_LU nv
	s_wait_alu depctr_va_vdst(0)
	scratch_load_b128 v[200:203], off, off offset:60 th:TH_LOAD_LU nv
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x10c
	v_add_f32_e32 v205, v192, v193 /*v961*/
	v_add_f32_e32 v204, v128, v192 /*v960*/
	s_set_vgpr_msb 0xc01
	v_wmma_f32_16x16x32_bf16 v[80:87], v[112:119] /*v[368:375]*/, v[210:217], v[80:87]
	v_mov_b32_e32 v128, v50 /*v306*/
	v_wmma_f32_16x16x32_bf16 v[16:23], v[112:119] /*v[368:375]*/, v[218:225], v[16:23]
	v_wmma_f32_16x16x32_bf16 v[80:87], v[104:111] /*v[360:367]*/, v[226:233], v[80:87]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[104:111] /*v[360:367]*/, v[234:241], v[16:23]
	v_wmma_f32_16x16x32_bf16 v[80:87], v[96:103] /*v[352:359]*/, v[242:249], v[80:87]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[96:103] /*v[352:359]*/, v[250:257], v[16:23]
	s_set_vgpr_msb 0x105
	v_wmma_f32_16x16x32_bf16 v[80:87], v[88:95] /*v[344:351]*/, v[2:9] /*v[258:265]*/, v[80:87]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[88:95] /*v[344:351]*/, v[10:17] /*v[266:273]*/, v[16:23]
	v_wmma_f32_16x16x32_bf16 v[80:87], v[80:87] /*v[336:343]*/, v[18:25] /*v[274:281]*/, v[80:87]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[80:87] /*v[336:343]*/, v[26:33] /*v[282:289]*/, v[16:23]
	v_wmma_f32_16x16x32_bf16 v[80:87], v[64:71] /*v[320:327]*/, v[34:41] /*v[290:297]*/, v[80:87]
	s_set_vgpr_msb 0x509
	v_wmma_f32_16x16x32_bf16 v[16:23], v[64:71] /*v[320:327]*/, v[144:151] /*v[656:663]*/, v[16:23]
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x940
	s_clause 0x1
	scratch_load_b128 v[64:67] /*v[320:323]*/, off, off offset:460 th:TH_LOAD_LU nv
	scratch_load_b128 v[68:71] /*v[324:327]*/, off, off offset:476 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0x4004
	s_wait_loadcnt_dscnt 0x200
	v_wmma_f32_16x16x32_bf16 v[64:71], v[196:203], v[34:41] /*v[290:297]*/, v[64:71]
	s_set_vgpr_msb 0x408
	v_wmma_f32_16x16x32_bf16 v[0:7], v[196:203], v[144:151] /*v[656:663]*/, v[0:7]
	scratch_load_b128 v[192:195], off, off offset:12 th:TH_LOAD_LU nv
	s_wait_alu depctr_va_vdst(0)
	scratch_load_b128 v[196:199], off, off offset:28 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0x802
	v_wmma_f32_16x16x32_bf16 v[112:119], v[120:127] /*v[632:639]*/, v[210:217], v[112:119]
	v_wmma_f32_16x16x32_bf16 v[48:55], v[120:127] /*v[632:639]*/, v[218:225], v[48:55]
	v_wmma_f32_16x16x32_bf16 v[104:111], v[56:63] /*v[568:575]*/, v[210:217], v[104:111]
	v_wmma_f32_16x16x32_bf16 v[40:47], v[56:63] /*v[568:575]*/, v[218:225], v[40:47]
	s_set_vgpr_msb 0x201
	v_wmma_f32_16x16x32_bf16 v[96:103], v[240:247] /*v[496:503]*/, v[210:217], v[96:103]
	v_wmma_f32_16x16x32_bf16 v[32:39], v[240:247] /*v[496:503]*/, v[218:225], v[32:39]
	v_wmma_f32_16x16x32_bf16 v[88:95], v[184:191] /*v[440:447]*/, v[210:217], v[88:95]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[184:191] /*v[440:447]*/, v[218:225], v[24:31]
	s_set_vgpr_msb 0x102
	v_wmma_f32_16x16x32_bf16 v[112:119], v[112:119] /*v[624:631]*/, v[226:233], v[112:119]
	v_wmma_f32_16x16x32_bf16 v[48:55], v[112:119] /*v[624:631]*/, v[234:241], v[48:55]
	v_wmma_f32_16x16x32_bf16 v[104:111], v[40:47] /*v[552:559]*/, v[226:233], v[104:111]
	v_wmma_f32_16x16x32_bf16 v[40:47], v[40:47] /*v[552:559]*/, v[234:241], v[40:47]
	s_set_vgpr_msb 0x201
	v_wmma_f32_16x16x32_bf16 v[96:103], v[232:239] /*v[488:495]*/, v[226:233], v[96:103]
	v_wmma_f32_16x16x32_bf16 v[32:39], v[232:239] /*v[488:495]*/, v[234:241], v[32:39]
	v_wmma_f32_16x16x32_bf16 v[88:95], v[168:175] /*v[424:431]*/, v[226:233], v[88:95]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[168:175] /*v[424:431]*/, v[234:241], v[24:31]
	s_set_vgpr_msb 0x102
	v_wmma_f32_16x16x32_bf16 v[112:119], v[96:103] /*v[608:615]*/, v[242:249], v[112:119]
	v_wmma_f32_16x16x32_bf16 v[48:55], v[96:103] /*v[608:615]*/, v[250:257], v[48:55]
	v_wmma_f32_16x16x32_bf16 v[104:111], v[32:39] /*v[544:551]*/, v[242:249], v[104:111]
	v_wmma_f32_16x16x32_bf16 v[40:47], v[32:39] /*v[544:551]*/, v[250:257], v[40:47]
	s_set_vgpr_msb 0x201
	v_wmma_f32_16x16x32_bf16 v[96:103], v[224:231] /*v[480:487]*/, v[242:249], v[96:103]
	v_wmma_f32_16x16x32_bf16 v[32:39], v[224:231] /*v[480:487]*/, v[250:257], v[32:39]
	v_wmma_f32_16x16x32_bf16 v[88:95], v[160:167] /*v[416:423]*/, v[242:249], v[88:95]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[160:167] /*v[416:423]*/, v[250:257], v[24:31]
	s_set_vgpr_msb 0x106
	v_wmma_f32_16x16x32_bf16 v[112:119], v[88:95] /*v[600:607]*/, v[2:9] /*v[258:265]*/, v[112:119]
	v_wmma_f32_16x16x32_bf16 v[48:55], v[88:95] /*v[600:607]*/, v[10:17] /*v[266:273]*/, v[48:55]
	v_wmma_f32_16x16x32_bf16 v[104:111], v[24:31] /*v[536:543]*/, v[2:9] /*v[258:265]*/, v[104:111]
	v_wmma_f32_16x16x32_bf16 v[40:47], v[24:31] /*v[536:543]*/, v[10:17] /*v[266:273]*/, v[40:47]
	s_set_vgpr_msb 0x605
	v_wmma_f32_16x16x32_bf16 v[96:103], v[216:223] /*v[472:479]*/, v[2:9] /*v[258:265]*/, v[96:103]
	v_wmma_f32_16x16x32_bf16 v[32:39], v[216:223] /*v[472:479]*/, v[10:17] /*v[266:273]*/, v[32:39]
	v_wmma_f32_16x16x32_bf16 v[88:95], v[152:159] /*v[408:415]*/, v[2:9] /*v[258:265]*/, v[88:95]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[152:159] /*v[408:415]*/, v[10:17] /*v[266:273]*/, v[24:31]
	s_set_vgpr_msb 0x506
	v_wmma_f32_16x16x32_bf16 v[112:119], v[80:87] /*v[592:599]*/, v[18:25] /*v[274:281]*/, v[112:119]
	v_wmma_f32_16x16x32_bf16 v[48:55], v[80:87] /*v[592:599]*/, v[26:33] /*v[282:289]*/, v[48:55]
	v_wmma_f32_16x16x32_bf16 v[104:111], v[16:23] /*v[528:535]*/, v[18:25] /*v[274:281]*/, v[104:111]
	v_wmma_f32_16x16x32_bf16 v[40:47], v[16:23] /*v[528:535]*/, v[26:33] /*v[282:289]*/, v[40:47]
	s_set_vgpr_msb 0x605
	v_wmma_f32_16x16x32_bf16 v[96:103], v[208:215] /*v[464:471]*/, v[18:25] /*v[274:281]*/, v[96:103]
	v_wmma_f32_16x16x32_bf16 v[32:39], v[208:215] /*v[464:471]*/, v[26:33] /*v[282:289]*/, v[32:39]
	v_wmma_f32_16x16x32_bf16 v[88:95], v[144:151] /*v[400:407]*/, v[18:25] /*v[274:281]*/, v[88:95]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[144:151] /*v[400:407]*/, v[26:33] /*v[282:289]*/, v[24:31]
	s_set_vgpr_msb 0x506
	v_wmma_f32_16x16x32_bf16 v[112:119], v[64:71] /*v[576:583]*/, v[34:41] /*v[290:297]*/, v[112:119]
	s_set_vgpr_msb 0x60a
	v_wmma_f32_16x16x32_bf16 v[48:55], v[64:71] /*v[576:583]*/, v[144:151] /*v[656:663]*/, v[48:55]
	s_set_vgpr_msb 0xa06
	v_wmma_f32_16x16x32_bf16 v[104:111], v[8:15] /*v[520:527]*/, v[34:41] /*v[290:297]*/, v[104:111]
	s_set_vgpr_msb 0x60a
	v_wmma_f32_16x16x32_bf16 v[40:47], v[8:15] /*v[520:527]*/, v[144:151] /*v[656:663]*/, v[40:47]
	s_set_vgpr_msb 0xa05
	v_wmma_f32_16x16x32_bf16 v[96:103], v[192:199] /*v[448:455]*/, v[34:41] /*v[290:297]*/, v[96:103]
	s_set_vgpr_msb 0x509
	v_wmma_f32_16x16x32_bf16 v[32:39], v[192:199] /*v[448:455]*/, v[144:151] /*v[656:663]*/, v[32:39]
	s_set_vgpr_msb 0x905
	v_wmma_f32_16x16x32_bf16 v[88:95], v[136:143] /*v[392:399]*/, v[34:41] /*v[290:297]*/, v[88:95]
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x540
	v_dual_mov_b32 v40 /*v296*/, v132 :: v_dual_mov_b32 v41 /*v297*/, v133
	s_set_vgpr_msb 0x4000
	v_mov_b32_e32 v132, v131
	s_set_vgpr_msb 9
	v_wmma_f32_16x16x32_bf16 v[24:31], v[136:143] /*v[392:399]*/, v[144:151] /*v[656:663]*/, v[24:31]
	v_mov_b32_e32 v133, v55 /*v311*/
	s_set_vgpr_msb 0x90a
	v_wmma_f32_16x16x32_bf16 v[120:127], v[128:135] /*v[640:647]*/, v[136:143] /*v[648:655]*/, v[120:127]
	v_wmma_f32_16x16x32_bf16 v[56:63], v[128:135] /*v[640:647]*/, v[152:159] /*v[664:671]*/, v[56:63]
	v_wmma_f32_16x16x32_bf16 v[112:119], v[48:55] /*v[560:567]*/, v[136:143] /*v[648:655]*/, v[112:119]
	v_wmma_f32_16x16x32_bf16 v[48:55], v[48:55] /*v[560:567]*/, v[152:159] /*v[664:671]*/, v[48:55]
	v_wmma_f32_16x16x32_bf16 v[104:111], v[0:7] /*v[512:519]*/, v[136:143] /*v[648:655]*/, v[104:111]
	v_wmma_f32_16x16x32_bf16 v[40:47], v[0:7] /*v[512:519]*/, v[152:159] /*v[664:671]*/, v[40:47]
	s_set_vgpr_msb 0xa09
	v_wmma_f32_16x16x32_bf16 v[96:103], v[176:183] /*v[432:439]*/, v[136:143] /*v[648:655]*/, v[96:103]
	v_wmma_f32_16x16x32_bf16 v[32:39], v[176:183] /*v[432:439]*/, v[152:159] /*v[664:671]*/, v[32:39]
	v_wmma_f32_16x16x32_bf16 v[88:95], v[128:135] /*v[384:391]*/, v[136:143] /*v[648:655]*/, v[88:95]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[128:135] /*v[384:391]*/, v[152:159] /*v[664:671]*/, v[24:31]
	s_wait_loadcnt 0x2
	v_wmma_f32_16x16x32_bf16 v[80:87], v[64:71] /*v[320:327]*/, v[136:143] /*v[648:655]*/, v[80:87]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[64:71] /*v[320:327]*/, v[152:159] /*v[664:671]*/, v[16:23]
	s_set_vgpr_msb 0x908
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[64:71], v[192:199], v[136:143] /*v[648:655]*/, v[64:71]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[192:199], v[152:159] /*v[664:671]*/, v[0:7]
	s_set_vgpr_msb 0x800
	s_cbranch_vccnz .LBB0_22
.LBB0_16:
	s_set_vgpr_msb 0x41
	v_mov_b32_e32 v55 /*v311*/, v41 /*v297*/
	s_set_vgpr_msb 0x4101
	v_mov_b32_e32 v131, v40 /*v296*/
	s_wait_alu depctr_va_vdst(0)
	scratch_store_b64 off, v[204:205], off nv
	s_add_co_i32 s2, s52, 1
	s_wait_tensorcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	s_cmp_ge_i32 s2, s10
	s_set_vgpr_msb 0x100
	s_cbranch_scc1 .LBB0_18
	s_lshl_b32 s5, s2, 8
	s_mov_b32 s45, s37
	s_add_co_i32 s6, s5, s64
	s_mov_b32 s48, s40
	s_ashr_i32 s7, s6, 31
	s_mov_b32 s51, s43
	s_mul_u64 s[46:47], s[6:7], s[62:63]
	s_mul_u64 s[6:7], s[6:7], s[60:61]
	s_lshl_b64 s[84:85], s[46:47], 1
	s_bitcmp1_b32 s2, 0
	s_add_nc_u64 s[84:85], s[54:55], s[84:85]
	s_cselect_b32 s2, 0x23000, 0
	s_sub_co_i32 s5, s35, s5
	s_lshl_b64 s[6:7], s[6:7], 1
	v_med3_i32 v130, s5, 0, 0x100
	s_add_nc_u64 s[6:7], s[80:81], s[6:7]
	s_mov_b32 s47, s39
	s_add_nc_u64 s[6:7], s[18:19], s[6:7]
	v_readfirstlane_b32 s5, v130
	s_bitset1_b32 s7, 31
	s_sub_co_i32 s38, s5, s17
	s_add_co_i32 s5, s93, s2
	s_max_i32 s38, s38, 0
	s_lshl_b32 s38, s38, 16
	s_addk_co_i32 s38, 0x7fff
	tensor_load_to_lds s[4:7], s[36:43]
	s_add_nc_u64 s[6:7], s[78:79], s[84:85]
	s_add_co_i32 s5, s94, s2
	s_mov_b32 s46, s38
	s_bitset1_b32 s7, 31
	tensor_load_to_lds s[4:7], s[44:51]
.LBB0_18:
	ds_load_b128 v[192:195], v133
	ds_load_b128 v[196:199], v133 offset:32
	ds_load_b128 v[200:203], v133 offset:64
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_b128 v[204:207], v133 offset:96
	ds_load_b128 v[208:211], v133 offset:128
	ds_load_b128 v[212:215], v133 offset:160
	ds_load_b128 v[216:219], v133 offset:192
	ds_load_b128 v[220:223], v133 offset:224
	ds_load_b128 v[224:227], v133 offset:4352
	ds_load_b128 v[228:231], v133 offset:4384
	ds_load_b128 v[232:235], v133 offset:4416
	ds_load_b128 v[236:239], v133 offset:4448
	ds_load_b128 v[240:243], v133 offset:4480
	ds_load_b128 v[244:247], v133 offset:4512
	ds_load_b128 v[248:251], v133 offset:4544
	ds_load_b128 v[252:255], v133 offset:4576
	s_set_vgpr_msb 64
	ds_load_b128 v[0:3] /*v[256:259]*/, v133 offset:8704
	ds_load_b128 v[4:7] /*v[260:263]*/, v133 offset:8736
	ds_load_b128 v[8:11] /*v[264:267]*/, v133 offset:8768
	ds_load_b128 v[12:15] /*v[268:271]*/, v133 offset:8800
	ds_load_b128 v[16:19] /*v[272:275]*/, v133 offset:8832
	ds_load_b128 v[20:23] /*v[276:279]*/, v133 offset:8864
	ds_load_b128 v[24:27] /*v[280:283]*/, v133 offset:8896
	ds_load_b128 v[28:31] /*v[284:287]*/, v133 offset:8928
	ds_load_b128 v[32:35] /*v[288:291]*/, v133 offset:13056
	ds_load_b128 v[36:39] /*v[292:295]*/, v133 offset:13088
	ds_load_b128 v[56:59] /*v[312:315]*/, v133 offset:13120
	ds_load_b128 v[60:63] /*v[316:319]*/, v133 offset:13152
	ds_load_b128 v[64:67] /*v[320:323]*/, v133 offset:13184
	ds_load_b128 v[68:71] /*v[324:327]*/, v133 offset:13216
	ds_load_b128 v[72:75] /*v[328:331]*/, v133 offset:13248
	ds_load_b128 v[76:79] /*v[332:335]*/, v133 offset:13280
	ds_load_b128 v[80:83] /*v[336:339]*/, v133 offset:17408
	ds_load_b128 v[84:87] /*v[340:343]*/, v133 offset:17440
	ds_load_b128 v[88:91] /*v[344:347]*/, v133 offset:17472
	ds_load_b128 v[92:95] /*v[348:351]*/, v133 offset:17504
	ds_load_b128 v[96:99] /*v[352:355]*/, v133 offset:17536
	ds_load_b128 v[100:103] /*v[356:359]*/, v133 offset:17568
	ds_load_b128 v[104:107] /*v[360:363]*/, v133 offset:17600
	ds_load_b128 v[108:111] /*v[364:367]*/, v133 offset:17632
	ds_load_b128 v[112:115] /*v[368:371]*/, v133 offset:21760
	ds_load_b128 v[116:119] /*v[372:375]*/, v133 offset:21792
	ds_load_b128 v[120:123] /*v[376:379]*/, v133 offset:21824
	ds_load_b128 v[124:127] /*v[380:383]*/, v133 offset:21856
	ds_load_b128 v[128:131] /*v[384:387]*/, v133 offset:21888
	ds_load_b128 v[132:135] /*v[388:391]*/, v133 offset:21920
	ds_load_b128 v[136:139] /*v[392:395]*/, v133 offset:21952
	ds_load_b128 v[140:143] /*v[396:399]*/, v133 offset:21984
	ds_load_b128 v[144:147] /*v[400:403]*/, v133 offset:26112
	ds_load_b128 v[148:151] /*v[404:407]*/, v133 offset:26144
	ds_load_b128 v[152:155] /*v[408:411]*/, v133 offset:26176
	ds_load_b128 v[156:159] /*v[412:415]*/, v133 offset:26208
	ds_load_b128 v[160:163] /*v[416:419]*/, v133 offset:26240
	ds_load_b128 v[164:167] /*v[420:423]*/, v133 offset:26272
	ds_load_b128 v[168:171] /*v[424:427]*/, v133 offset:26304
	ds_load_b128 v[172:175] /*v[428:431]*/, v133 offset:26336
	ds_load_b128 v[176:179] /*v[432:435]*/, v133 offset:30464
	ds_load_b128 v[180:183] /*v[436:439]*/, v133 offset:30496
	ds_load_b128 v[184:187] /*v[440:443]*/, v133 offset:30528
	ds_load_b128 v[188:191] /*v[444:447]*/, v133 offset:30560
	ds_load_b128 v[192:195] /*v[448:451]*/, v133 offset:30592
	ds_load_b128 v[196:199] /*v[452:455]*/, v133 offset:30624
	ds_load_b128 v[200:203] /*v[456:459]*/, v133 offset:30656
	ds_load_b128 v[204:207] /*v[460:463]*/, v133 offset:30688
	ds_load_b128 v[208:211] /*v[464:467]*/, v133 offset:34816
	ds_load_b128 v[212:215] /*v[468:471]*/, v133 offset:34848
	ds_load_b128 v[216:219] /*v[472:475]*/, v133 offset:34880
	ds_load_b128 v[220:223] /*v[476:479]*/, v133 offset:34912
	ds_load_b128 v[224:227] /*v[480:483]*/, v133 offset:34944
	ds_load_b128 v[228:231] /*v[484:487]*/, v133 offset:34976
	ds_load_b128 v[232:235] /*v[488:491]*/, v133 offset:35008
	ds_load_b128 v[236:239] /*v[492:495]*/, v133 offset:35040
	ds_load_b128 v[240:243] /*v[496:499]*/, v133 offset:39168
	ds_load_b128 v[244:247] /*v[500:503]*/, v133 offset:39200
	ds_load_b128 v[248:251] /*v[504:507]*/, v133 offset:39232
	ds_load_b128 v[252:255] /*v[508:511]*/, v133 offset:39264
	s_set_vgpr_msb 0x4080
	ds_load_b128 v[0:3] /*v[512:515]*/, v133 offset:39296
	ds_load_b128 v[4:7] /*v[516:519]*/, v133 offset:39328
	ds_load_b128 v[8:11] /*v[520:523]*/, v133 offset:39360
	ds_load_b128 v[12:15] /*v[524:527]*/, v133 offset:39392
	ds_load_b128 v[16:19] /*v[528:531]*/, v133 offset:43520
	ds_load_b128 v[20:23] /*v[532:535]*/, v133 offset:43552
	ds_load_b128 v[24:27] /*v[536:539]*/, v133 offset:43584
	ds_load_b128 v[28:31] /*v[540:543]*/, v133 offset:43616
	ds_load_b128 v[32:35] /*v[544:547]*/, v133 offset:43648
	ds_load_b128 v[36:39] /*v[548:551]*/, v133 offset:43680
	ds_load_b128 v[40:43] /*v[552:555]*/, v133 offset:43712
	ds_load_b128 v[44:47] /*v[556:559]*/, v133 offset:43744
	ds_load_b128 v[48:51] /*v[560:563]*/, v133 offset:47872
	ds_load_b128 v[52:55] /*v[564:567]*/, v133 offset:47904
	ds_load_b128 v[56:59] /*v[568:571]*/, v133 offset:47936
	ds_load_b128 v[60:63] /*v[572:575]*/, v133 offset:47968
	ds_load_b128 v[64:67] /*v[576:579]*/, v133 offset:48000
	ds_load_b128 v[68:71] /*v[580:583]*/, v133 offset:48032
	ds_load_b128 v[72:75] /*v[584:587]*/, v133 offset:48064
	ds_load_b128 v[76:79] /*v[588:591]*/, v133 offset:48096
	ds_load_b128 v[80:83] /*v[592:595]*/, v133 offset:52224
	ds_load_b128 v[84:87] /*v[596:599]*/, v133 offset:52256
	ds_load_b128 v[88:91] /*v[600:603]*/, v133 offset:52288
	ds_load_b128 v[92:95] /*v[604:607]*/, v133 offset:52320
	ds_load_b128 v[96:99] /*v[608:611]*/, v133 offset:52352
	ds_load_b128 v[100:103] /*v[612:615]*/, v133 offset:52384
	ds_load_b128 v[104:107] /*v[616:619]*/, v133 offset:52416
	ds_load_b128 v[108:111] /*v[620:623]*/, v133 offset:52448
	ds_load_b128 v[112:115] /*v[624:627]*/, v133 offset:56576
	ds_load_b128 v[116:119] /*v[628:631]*/, v133 offset:56608
	ds_load_b128 v[120:123] /*v[632:635]*/, v133 offset:56640
	ds_load_b128 v[124:127] /*v[636:639]*/, v133 offset:56672
	ds_load_b128 v[128:131] /*v[640:643]*/, v133 offset:56704
	ds_load_b128 v[132:135] /*v[644:647]*/, v133 offset:56736
	ds_load_b128 v[136:139] /*v[648:651]*/, v133 offset:56768
	ds_load_b128 v[140:143] /*v[652:655]*/, v133 offset:56800
	ds_load_b128 v[144:147] /*v[656:659]*/, v133 offset:60928
	ds_load_b128 v[148:151] /*v[660:663]*/, v133 offset:60960
	ds_load_b128 v[152:155] /*v[664:667]*/, v133 offset:60992
	ds_load_b128 v[156:159] /*v[668:671]*/, v133 offset:61024
	ds_load_b128 v[160:163] /*v[672:675]*/, v133 offset:61056
	ds_load_b128 v[164:167] /*v[676:679]*/, v133 offset:61088
	ds_load_b128 v[168:171] /*v[680:683]*/, v133 offset:61120
	ds_load_b128 v[172:175] /*v[684:687]*/, v133 offset:61152
	ds_load_b128 v[176:179] /*v[688:691]*/, v133 offset:65280
	ds_load_b128 v[180:183] /*v[692:695]*/, v133 offset:65312
	ds_load_b128 v[184:187] /*v[696:699]*/, v133 offset:65344
	ds_load_b128 v[188:191] /*v[700:703]*/, v133 offset:65376
	s_set_vgpr_msb 0x80c4
	ds_load_b128 v[176:179] /*v[944:947]*/, v133 offset:65408
	ds_load_b128 v[180:183] /*v[948:951]*/, v133 offset:65440
	ds_load_b128 v[184:187] /*v[952:955]*/, v133 offset:65472
	ds_load_b128 v[188:191] /*v[956:959]*/, v133 offset:65504
	s_wait_dscnt 0x3e
	v_wmma_f32_16x16x32_bf16 v[192:199] /*v[960:967]*/, v[192:199], v[42:49] /*v[298:305]*/, 0
	s_set_vgpr_msb 0xc480
	v_wmma_f32_16x16x32_bf16 v[192:199] /*v[704:711]*/, v[192:199], v[160:167], 0
	s_set_vgpr_msb 0x8004
	v_wmma_f32_16x16x32_bf16 v[192:199], v[224:231], v[42:49] /*v[298:305]*/, 0
	s_set_vgpr_msb 0x480
	v_wmma_f32_16x16x32_bf16 v[200:207] /*v[712:719]*/, v[224:231], v[160:167], 0
	s_set_vgpr_msb 0x80c5
	v_wmma_f32_16x16x32_bf16 v[168:175] /*v[936:943]*/, v[0:7] /*v[256:263]*/, v[42:49] /*v[298:305]*/, 0
	s_set_vgpr_msb 0xc581
	v_wmma_f32_16x16x32_bf16 v[208:215] /*v[720:727]*/, v[0:7] /*v[256:263]*/, v[160:167], 0
	s_set_vgpr_msb 0x81c5
	v_wmma_f32_16x16x32_bf16 v[160:167] /*v[928:935]*/, v[32:39] /*v[288:295]*/, v[42:49] /*v[298:305]*/, 0
	s_set_vgpr_msb 0xc581
	v_wmma_f32_16x16x32_bf16 v[216:223] /*v[728:735]*/, v[32:39] /*v[288:295]*/, v[160:167], 0
	s_set_vgpr_msb 0x81c5
	v_wmma_f32_16x16x32_bf16 v[152:159] /*v[920:927]*/, v[80:87] /*v[336:343]*/, v[42:49] /*v[298:305]*/, 0
	s_set_vgpr_msb 0xc581
	v_wmma_f32_16x16x32_bf16 v[224:231] /*v[736:743]*/, v[80:87] /*v[336:343]*/, v[160:167], 0
	s_set_vgpr_msb 0x81c5
	v_wmma_f32_16x16x32_bf16 v[144:151] /*v[912:919]*/, v[112:119] /*v[368:375]*/, v[42:49] /*v[298:305]*/, 0
	s_set_vgpr_msb 0xc581
	v_wmma_f32_16x16x32_bf16 v[232:239] /*v[744:751]*/, v[112:119] /*v[368:375]*/, v[160:167], 0
	s_set_vgpr_msb 0x81c5
	v_wmma_f32_16x16x32_bf16 v[136:143] /*v[904:911]*/, v[144:151] /*v[400:407]*/, v[42:49] /*v[298:305]*/, 0
	s_set_vgpr_msb 0xc581
	v_wmma_f32_16x16x32_bf16 v[240:247] /*v[752:759]*/, v[144:151] /*v[400:407]*/, v[160:167], 0
	s_set_vgpr_msb 0x81c5
	v_wmma_f32_16x16x32_bf16 v[128:135] /*v[896:903]*/, v[176:183] /*v[432:439]*/, v[42:49] /*v[298:305]*/, 0
	s_set_vgpr_msb 0xc581
	v_wmma_f32_16x16x32_bf16 v[248:255] /*v[760:767]*/, v[176:183] /*v[432:439]*/, v[160:167], 0
	s_set_vgpr_msb 0x81c5
	v_wmma_f32_16x16x32_bf16 v[120:127] /*v[888:895]*/, v[208:215] /*v[464:471]*/, v[42:49] /*v[298:305]*/, 0
	s_set_vgpr_msb 0xc5c1
	v_wmma_f32_16x16x32_bf16 v[0:7] /*v[768:775]*/, v[208:215] /*v[464:471]*/, v[160:167], 0
	s_set_vgpr_msb 0xc1c5
	s_wait_dscnt 0x36
	v_wmma_f32_16x16x32_bf16 v[112:119] /*v[880:887]*/, v[240:247] /*v[496:503]*/, v[42:49] /*v[298:305]*/, 0
	s_set_vgpr_msb 0xc5c1
	v_wmma_f32_16x16x32_bf16 v[8:15] /*v[776:783]*/, v[240:247] /*v[496:503]*/, v[160:167], 0
	s_set_vgpr_msb 0xc1c6
	s_wait_dscnt 0x2e
	v_wmma_f32_16x16x32_bf16 v[104:111] /*v[872:879]*/, v[16:23] /*v[528:535]*/, v[42:49] /*v[298:305]*/, 0
	s_set_vgpr_msb 0xc6c2
	v_wmma_f32_16x16x32_bf16 v[16:23] /*v[784:791]*/, v[16:23] /*v[528:535]*/, v[160:167], 0
	s_set_vgpr_msb 0xc2c6
	s_wait_dscnt 0x26
	v_wmma_f32_16x16x32_bf16 v[96:103] /*v[864:871]*/, v[48:55] /*v[560:567]*/, v[42:49] /*v[298:305]*/, 0
	s_set_vgpr_msb 0xc6c2
	v_wmma_f32_16x16x32_bf16 v[24:31] /*v[792:799]*/, v[48:55] /*v[560:567]*/, v[160:167], 0
	s_set_vgpr_msb 0xc2c6
	s_wait_dscnt 0x1e
	v_wmma_f32_16x16x32_bf16 v[88:95] /*v[856:863]*/, v[80:87] /*v[592:599]*/, v[42:49] /*v[298:305]*/, 0
	s_set_vgpr_msb 0xc6c2
	v_wmma_f32_16x16x32_bf16 v[32:39] /*v[800:807]*/, v[80:87] /*v[592:599]*/, v[160:167], 0
	s_set_vgpr_msb 0xc2c6
	s_wait_dscnt 0x16
	v_wmma_f32_16x16x32_bf16 v[80:87] /*v[848:855]*/, v[112:119] /*v[624:631]*/, v[42:49] /*v[298:305]*/, 0
	s_set_vgpr_msb 0xc6c2
	v_wmma_f32_16x16x32_bf16 v[40:47] /*v[808:815]*/, v[112:119] /*v[624:631]*/, v[160:167], 0
	s_set_vgpr_msb 0xc2c6
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x32_bf16 v[72:79] /*v[840:847]*/, v[144:151] /*v[656:663]*/, v[42:49] /*v[298:305]*/, 0
	s_set_vgpr_msb 0xc6c2
	v_wmma_f32_16x16x32_bf16 v[48:55] /*v[816:823]*/, v[144:151] /*v[656:663]*/, v[160:167], 0
	s_set_vgpr_msb 0xc2c6
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x32_bf16 v[64:71] /*v[832:839]*/, v[176:183] /*v[688:695]*/, v[42:49] /*v[298:305]*/, 0
	s_set_vgpr_msb 0xc6c2
	v_wmma_f32_16x16x32_bf16 v[56:63] /*v[824:831]*/, v[176:183] /*v[688:695]*/, v[160:167], 0
	s_set_vgpr_msb 0xc2f0
	v_wmma_f32_16x16x32_bf16 v[192:199] /*v[960:967]*/, v[200:207], v[136:143], v[192:199] /*v[960:967]*/
	s_set_vgpr_msb 0xf0a0
	v_wmma_f32_16x16x32_bf16 v[192:199] /*v[704:711]*/, v[200:207], v[168:175], v[192:199] /*v[704:711]*/
	s_set_vgpr_msb 0xa000
	v_wmma_f32_16x16x32_bf16 v[192:199], v[232:239], v[136:143], v[192:199]
	s_set_vgpr_msb 0xa0
	v_wmma_f32_16x16x32_bf16 v[200:207] /*v[712:719]*/, v[232:239], v[168:175], v[200:207] /*v[712:719]*/
	s_set_vgpr_msb 0xa0f1
	v_wmma_f32_16x16x32_bf16 v[168:175] /*v[936:943]*/, v[8:15] /*v[264:271]*/, v[136:143], v[168:175] /*v[936:943]*/
	s_set_vgpr_msb 0xf1a1
	v_wmma_f32_16x16x32_bf16 v[208:215] /*v[720:727]*/, v[8:15] /*v[264:271]*/, v[168:175], v[208:215] /*v[720:727]*/
	s_set_vgpr_msb 0xa1f1
	v_wmma_f32_16x16x32_bf16 v[160:167] /*v[928:935]*/, v[56:63] /*v[312:319]*/, v[136:143], v[160:167] /*v[928:935]*/
	s_set_vgpr_msb 0xf1a1
	v_wmma_f32_16x16x32_bf16 v[216:223] /*v[728:735]*/, v[56:63] /*v[312:319]*/, v[168:175], v[216:223] /*v[728:735]*/
	s_set_vgpr_msb 0xa1f1
	v_wmma_f32_16x16x32_bf16 v[152:159] /*v[920:927]*/, v[88:95] /*v[344:351]*/, v[136:143], v[152:159] /*v[920:927]*/
	s_set_vgpr_msb 0xf1a1
	v_wmma_f32_16x16x32_bf16 v[224:231] /*v[736:743]*/, v[88:95] /*v[344:351]*/, v[168:175], v[224:231] /*v[736:743]*/
	s_set_vgpr_msb 0xa1f1
	v_wmma_f32_16x16x32_bf16 v[144:151] /*v[912:919]*/, v[120:127] /*v[376:383]*/, v[136:143], v[144:151] /*v[912:919]*/
	s_set_vgpr_msb 0xf1a1
	v_wmma_f32_16x16x32_bf16 v[232:239] /*v[744:751]*/, v[120:127] /*v[376:383]*/, v[168:175], v[232:239] /*v[744:751]*/
	s_set_vgpr_msb 0xa1f1
	v_wmma_f32_16x16x32_bf16 v[136:143] /*v[904:911]*/, v[152:159] /*v[408:415]*/, v[136:143], v[136:143] /*v[904:911]*/
	s_set_vgpr_msb 0xf1a1
	v_wmma_f32_16x16x32_bf16 v[240:247] /*v[752:759]*/, v[152:159] /*v[408:415]*/, v[168:175], v[240:247] /*v[752:759]*/
	s_set_vgpr_msb 0xa1f1
	v_wmma_f32_16x16x32_bf16 v[128:135] /*v[896:903]*/, v[184:191] /*v[440:447]*/, v[136:143], v[128:135] /*v[896:903]*/
	s_set_vgpr_msb 0xf1a1
	v_wmma_f32_16x16x32_bf16 v[248:255] /*v[760:767]*/, v[184:191] /*v[440:447]*/, v[168:175], v[248:255] /*v[760:767]*/
	s_set_vgpr_msb 0xa1f1
	v_wmma_f32_16x16x32_bf16 v[120:127] /*v[888:895]*/, v[216:223] /*v[472:479]*/, v[136:143], v[120:127] /*v[888:895]*/
	v_wmma_f32_16x16x32_bf16 v[0:7] /*v[768:775]*/, v[216:223] /*v[472:479]*/, v[168:175], v[0:7] /*v[768:775]*/
	v_wmma_f32_16x16x32_bf16 v[112:119] /*v[880:887]*/, v[248:255] /*v[504:511]*/, v[136:143], v[112:119] /*v[880:887]*/
	v_wmma_f32_16x16x32_bf16 v[8:15] /*v[776:783]*/, v[248:255] /*v[504:511]*/, v[168:175], v[8:15] /*v[776:783]*/
	s_set_vgpr_msb 0xf1f2
	v_wmma_f32_16x16x32_bf16 v[104:111] /*v[872:879]*/, v[24:31] /*v[536:543]*/, v[136:143], v[104:111] /*v[872:879]*/
	v_wmma_f32_16x16x32_bf16 v[16:23] /*v[784:791]*/, v[24:31] /*v[536:543]*/, v[168:175], v[16:23] /*v[784:791]*/
	v_wmma_f32_16x16x32_bf16 v[96:103] /*v[864:871]*/, v[56:63] /*v[568:575]*/, v[136:143], v[96:103] /*v[864:871]*/
	v_wmma_f32_16x16x32_bf16 v[24:31] /*v[792:799]*/, v[56:63] /*v[568:575]*/, v[168:175], v[24:31] /*v[792:799]*/
	v_wmma_f32_16x16x32_bf16 v[88:95] /*v[856:863]*/, v[88:95] /*v[600:607]*/, v[136:143], v[88:95] /*v[856:863]*/
	v_wmma_f32_16x16x32_bf16 v[32:39] /*v[800:807]*/, v[88:95] /*v[600:607]*/, v[168:175], v[32:39] /*v[800:807]*/
	v_wmma_f32_16x16x32_bf16 v[80:87] /*v[848:855]*/, v[120:127] /*v[632:639]*/, v[136:143], v[80:87] /*v[848:855]*/
	v_wmma_f32_16x16x32_bf16 v[40:47] /*v[808:815]*/, v[120:127] /*v[632:639]*/, v[168:175], v[40:47] /*v[808:815]*/
	v_wmma_f32_16x16x32_bf16 v[72:79] /*v[840:847]*/, v[152:159] /*v[664:671]*/, v[136:143], v[72:79] /*v[840:847]*/
	v_wmma_f32_16x16x32_bf16 v[48:55] /*v[816:823]*/, v[152:159] /*v[664:671]*/, v[168:175], v[48:55] /*v[816:823]*/
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x32_bf16 v[64:71] /*v[832:839]*/, v[184:191] /*v[696:703]*/, v[136:143], v[64:71] /*v[832:839]*/
	v_wmma_f32_16x16x32_bf16 v[56:63] /*v[824:831]*/, v[184:191] /*v[696:703]*/, v[168:175], v[56:63] /*v[824:831]*/
	s_set_vgpr_msb 0xf2f0
	v_wmma_f32_16x16x32_bf16 v[192:199] /*v[960:967]*/, v[208:215], v[144:151], v[192:199] /*v[960:967]*/
	s_set_vgpr_msb 0xf0a0
	v_wmma_f32_16x16x32_bf16 v[192:199] /*v[704:711]*/, v[208:215], v[176:183], v[192:199] /*v[704:711]*/
	s_set_vgpr_msb 0xa000
	v_wmma_f32_16x16x32_bf16 v[192:199], v[240:247], v[144:151], v[192:199]
	s_set_vgpr_msb 0xa0
	v_wmma_f32_16x16x32_bf16 v[200:207] /*v[712:719]*/, v[240:247], v[176:183], v[200:207] /*v[712:719]*/
	s_set_vgpr_msb 0xa0f1
	v_wmma_f32_16x16x32_bf16 v[168:175] /*v[936:943]*/, v[16:23] /*v[272:279]*/, v[144:151], v[168:175] /*v[936:943]*/
	s_set_vgpr_msb 0xf1a1
	v_wmma_f32_16x16x32_bf16 v[208:215] /*v[720:727]*/, v[16:23] /*v[272:279]*/, v[176:183], v[208:215] /*v[720:727]*/
	s_set_vgpr_msb 0xa1f1
	v_wmma_f32_16x16x32_bf16 v[160:167] /*v[928:935]*/, v[64:71] /*v[320:327]*/, v[144:151], v[160:167] /*v[928:935]*/
	s_set_vgpr_msb 0xf1a1
	v_wmma_f32_16x16x32_bf16 v[216:223] /*v[728:735]*/, v[64:71] /*v[320:327]*/, v[176:183], v[216:223] /*v[728:735]*/
	s_set_vgpr_msb 0xa1f1
	v_wmma_f32_16x16x32_bf16 v[152:159] /*v[920:927]*/, v[96:103] /*v[352:359]*/, v[144:151], v[152:159] /*v[920:927]*/
	s_set_vgpr_msb 0xf1a1
	v_wmma_f32_16x16x32_bf16 v[224:231] /*v[736:743]*/, v[96:103] /*v[352:359]*/, v[176:183], v[224:231] /*v[736:743]*/
	s_set_vgpr_msb 0xa1f1
	v_wmma_f32_16x16x32_bf16 v[144:151] /*v[912:919]*/, v[128:135] /*v[384:391]*/, v[144:151], v[144:151] /*v[912:919]*/
	s_set_vgpr_msb 0xf1a1
	v_wmma_f32_16x16x32_bf16 v[232:239] /*v[744:751]*/, v[128:135] /*v[384:391]*/, v[176:183], v[232:239] /*v[744:751]*/
	s_set_vgpr_msb 0xa1f1
	v_wmma_f32_16x16x32_bf16 v[136:143] /*v[904:911]*/, v[160:167] /*v[416:423]*/, v[144:151], v[136:143] /*v[904:911]*/
	s_set_vgpr_msb 0xf1a1
	v_wmma_f32_16x16x32_bf16 v[240:247] /*v[752:759]*/, v[160:167] /*v[416:423]*/, v[176:183], v[240:247] /*v[752:759]*/
	s_set_vgpr_msb 0xa1f1
	v_wmma_f32_16x16x32_bf16 v[128:135] /*v[896:903]*/, v[192:199] /*v[448:455]*/, v[144:151], v[128:135] /*v[896:903]*/
	s_set_vgpr_msb 0xf1a1
	v_wmma_f32_16x16x32_bf16 v[248:255] /*v[760:767]*/, v[192:199] /*v[448:455]*/, v[176:183], v[248:255] /*v[760:767]*/
	s_set_vgpr_msb 0xa1f1
	v_wmma_f32_16x16x32_bf16 v[120:127] /*v[888:895]*/, v[224:231] /*v[480:487]*/, v[144:151], v[120:127] /*v[888:895]*/
	v_wmma_f32_16x16x32_bf16 v[0:7] /*v[768:775]*/, v[224:231] /*v[480:487]*/, v[176:183], v[0:7] /*v[768:775]*/
	s_set_vgpr_msb 0xf1f2
	v_wmma_f32_16x16x32_bf16 v[112:119] /*v[880:887]*/, v[0:7] /*v[512:519]*/, v[144:151], v[112:119] /*v[880:887]*/
	v_wmma_f32_16x16x32_bf16 v[8:15] /*v[776:783]*/, v[0:7] /*v[512:519]*/, v[176:183], v[8:15] /*v[776:783]*/
	v_wmma_f32_16x16x32_bf16 v[104:111] /*v[872:879]*/, v[32:39] /*v[544:551]*/, v[144:151], v[104:111] /*v[872:879]*/
	v_wmma_f32_16x16x32_bf16 v[16:23] /*v[784:791]*/, v[32:39] /*v[544:551]*/, v[176:183], v[16:23] /*v[784:791]*/
	v_wmma_f32_16x16x32_bf16 v[96:103] /*v[864:871]*/, v[64:71] /*v[576:583]*/, v[144:151], v[96:103] /*v[864:871]*/
	v_wmma_f32_16x16x32_bf16 v[24:31] /*v[792:799]*/, v[64:71] /*v[576:583]*/, v[176:183], v[24:31] /*v[792:799]*/
	v_wmma_f32_16x16x32_bf16 v[88:95] /*v[856:863]*/, v[96:103] /*v[608:615]*/, v[144:151], v[88:95] /*v[856:863]*/
	v_wmma_f32_16x16x32_bf16 v[32:39] /*v[800:807]*/, v[96:103] /*v[608:615]*/, v[176:183], v[32:39] /*v[800:807]*/
	v_wmma_f32_16x16x32_bf16 v[80:87] /*v[848:855]*/, v[128:135] /*v[640:647]*/, v[144:151], v[80:87] /*v[848:855]*/
	v_wmma_f32_16x16x32_bf16 v[40:47] /*v[808:815]*/, v[128:135] /*v[640:647]*/, v[176:183], v[40:47] /*v[808:815]*/
	v_wmma_f32_16x16x32_bf16 v[72:79] /*v[840:847]*/, v[160:167] /*v[672:679]*/, v[144:151], v[72:79] /*v[840:847]*/
	v_wmma_f32_16x16x32_bf16 v[48:55] /*v[816:823]*/, v[160:167] /*v[672:679]*/, v[176:183], v[48:55] /*v[816:823]*/
	s_set_vgpr_msb 0xf2f3
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[64:71] /*v[832:839]*/, v[176:183] /*v[944:951]*/, v[144:151], v[64:71] /*v[832:839]*/
	v_wmma_f32_16x16x32_bf16 v[56:63] /*v[824:831]*/, v[176:183] /*v[944:951]*/, v[176:183], v[56:63] /*v[824:831]*/
	s_set_vgpr_msb 0xf3f0
	v_wmma_f32_16x16x32_bf16 v[192:199] /*v[960:967]*/, v[216:223], v[152:159], v[192:199] /*v[960:967]*/
	s_set_vgpr_msb 0xf0a0
	v_wmma_f32_16x16x32_bf16 v[192:199] /*v[704:711]*/, v[216:223], v[184:191], v[192:199] /*v[704:711]*/
	s_set_vgpr_msb 0xa000
	v_wmma_f32_16x16x32_bf16 v[192:199], v[248:255], v[152:159], v[192:199]
	s_set_vgpr_msb 0xa0
	v_wmma_f32_16x16x32_bf16 v[200:207] /*v[712:719]*/, v[248:255], v[184:191], v[200:207] /*v[712:719]*/
	s_set_vgpr_msb 0xa0f1
	v_wmma_f32_16x16x32_bf16 v[168:175] /*v[936:943]*/, v[24:31] /*v[280:287]*/, v[152:159], v[168:175] /*v[936:943]*/
	s_set_vgpr_msb 0xf1a1
	v_wmma_f32_16x16x32_bf16 v[208:215] /*v[720:727]*/, v[24:31] /*v[280:287]*/, v[184:191], v[208:215] /*v[720:727]*/
	s_set_vgpr_msb 0xa1f1
	v_wmma_f32_16x16x32_bf16 v[160:167] /*v[928:935]*/, v[72:79] /*v[328:335]*/, v[152:159], v[160:167] /*v[928:935]*/
	s_set_vgpr_msb 0xf1a1
	v_wmma_f32_16x16x32_bf16 v[216:223] /*v[728:735]*/, v[72:79] /*v[328:335]*/, v[184:191], v[216:223] /*v[728:735]*/
	s_set_vgpr_msb 0xa1f1
	v_wmma_f32_16x16x32_bf16 v[152:159] /*v[920:927]*/, v[104:111] /*v[360:367]*/, v[152:159], v[152:159] /*v[920:927]*/
	s_set_vgpr_msb 0xf1a1
	v_wmma_f32_16x16x32_bf16 v[224:231] /*v[736:743]*/, v[104:111] /*v[360:367]*/, v[184:191], v[224:231] /*v[736:743]*/
	s_set_vgpr_msb 0xa1f1
	v_wmma_f32_16x16x32_bf16 v[144:151] /*v[912:919]*/, v[136:143] /*v[392:399]*/, v[152:159], v[144:151] /*v[912:919]*/
	s_set_vgpr_msb 0xf1a1
	v_wmma_f32_16x16x32_bf16 v[232:239] /*v[744:751]*/, v[136:143] /*v[392:399]*/, v[184:191], v[232:239] /*v[744:751]*/
	s_set_vgpr_msb 0xa1f1
	v_wmma_f32_16x16x32_bf16 v[136:143] /*v[904:911]*/, v[168:175] /*v[424:431]*/, v[152:159], v[136:143] /*v[904:911]*/
	s_set_vgpr_msb 0xf1a1
	v_wmma_f32_16x16x32_bf16 v[240:247] /*v[752:759]*/, v[168:175] /*v[424:431]*/, v[184:191], v[240:247] /*v[752:759]*/
	s_set_vgpr_msb 0xa1f1
	v_wmma_f32_16x16x32_bf16 v[128:135] /*v[896:903]*/, v[200:207] /*v[456:463]*/, v[152:159], v[128:135] /*v[896:903]*/
	s_set_vgpr_msb 0xf1a1
	v_wmma_f32_16x16x32_bf16 v[248:255] /*v[760:767]*/, v[200:207] /*v[456:463]*/, v[184:191], v[248:255] /*v[760:767]*/
	s_set_vgpr_msb 0xa1f1
	v_wmma_f32_16x16x32_bf16 v[120:127] /*v[888:895]*/, v[232:239] /*v[488:495]*/, v[152:159], v[120:127] /*v[888:895]*/
	v_wmma_f32_16x16x32_bf16 v[0:7] /*v[768:775]*/, v[232:239] /*v[488:495]*/, v[184:191], v[0:7] /*v[768:775]*/
	s_set_vgpr_msb 0xf1f2
	v_wmma_f32_16x16x32_bf16 v[112:119] /*v[880:887]*/, v[8:15] /*v[520:527]*/, v[152:159], v[112:119] /*v[880:887]*/
	v_wmma_f32_16x16x32_bf16 v[8:15] /*v[776:783]*/, v[8:15] /*v[520:527]*/, v[184:191], v[8:15] /*v[776:783]*/
	v_wmma_f32_16x16x32_bf16 v[104:111] /*v[872:879]*/, v[40:47] /*v[552:559]*/, v[152:159], v[104:111] /*v[872:879]*/
	v_wmma_f32_16x16x32_bf16 v[16:23] /*v[784:791]*/, v[40:47] /*v[552:559]*/, v[184:191], v[16:23] /*v[784:791]*/
	v_wmma_f32_16x16x32_bf16 v[96:103] /*v[864:871]*/, v[72:79] /*v[584:591]*/, v[152:159], v[96:103] /*v[864:871]*/
	v_wmma_f32_16x16x32_bf16 v[24:31] /*v[792:799]*/, v[72:79] /*v[584:591]*/, v[184:191], v[24:31] /*v[792:799]*/
	v_wmma_f32_16x16x32_bf16 v[88:95] /*v[856:863]*/, v[104:111] /*v[616:623]*/, v[152:159], v[88:95] /*v[856:863]*/
	v_wmma_f32_16x16x32_bf16 v[32:39] /*v[800:807]*/, v[104:111] /*v[616:623]*/, v[184:191], v[32:39] /*v[800:807]*/
	v_wmma_f32_16x16x32_bf16 v[80:87] /*v[848:855]*/, v[136:143] /*v[648:655]*/, v[152:159], v[80:87] /*v[848:855]*/
	v_wmma_f32_16x16x32_bf16 v[40:47] /*v[808:815]*/, v[136:143] /*v[648:655]*/, v[184:191], v[40:47] /*v[808:815]*/
	v_wmma_f32_16x16x32_bf16 v[72:79] /*v[840:847]*/, v[168:175] /*v[680:687]*/, v[152:159], v[72:79] /*v[840:847]*/
	v_wmma_f32_16x16x32_bf16 v[48:55] /*v[816:823]*/, v[168:175] /*v[680:687]*/, v[184:191], v[48:55] /*v[816:823]*/
	s_set_vgpr_msb 0xf2f3
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[64:71] /*v[832:839]*/, v[184:191] /*v[952:959]*/, v[152:159], v[64:71] /*v[832:839]*/
	v_wmma_f32_16x16x32_bf16 v[56:63] /*v[824:831]*/, v[184:191] /*v[952:959]*/, v[184:191], v[56:63] /*v[824:831]*/
	s_set_vgpr_msb 0xf300
	v_add_nc_u32_e32 v130, 0x10e00, v132
	v_add_nc_u32_e32 v200, 0x10e20, v132
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x80
	ds_load_tr16_b128 v[140:143] /*v[652:655]*/, v132 offset:59904
	ds_load_tr16_b128 v[68:71] /*v[580:583]*/, v132 offset:59936
	ds_load_tr16_b128 v[128:131] /*v[640:643]*/, v132 offset:64512
	ds_load_tr16_b128 v[48:51] /*v[560:563]*/, v132 offset:64544
	ds_load_tr16_b128 v[132:135] /*v[644:647]*/, v130
	ds_load_tr16_b128 v[52:55] /*v[564:567]*/, v200
	s_wait_alu depctr_vm_vsrc(1)
	s_set_vgpr_msb 0x8000
	v_add_nc_u32_e32 v130, 0x10e40, v132
	s_wait_alu depctr_vm_vsrc(0)
	v_add_nc_u32_e32 v200, 0x10e60, v132
	s_set_vgpr_msb 0x80
	ds_load_tr16_b128 v[12:15] /*v[524:527]*/, v132 offset:59968
	s_set_vgpr_msb 0x8040
	ds_load_tr16_b128 v[196:199] /*v[452:455]*/, v132 offset:60000
	s_set_vgpr_msb 0x4080
	ds_load_tr16_b128 v[0:3] /*v[512:515]*/, v132 offset:64576
	s_set_vgpr_msb 0x8040
	ds_load_tr16_b128 v[176:179] /*v[432:435]*/, v132 offset:64608
	s_wait_alu depctr_va_vdst(1)
	s_set_vgpr_msb 0x4080
	ds_load_tr16_b128 v[4:7] /*v[516:519]*/, v130
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x8040
	ds_load_tr16_b128 v[180:183] /*v[436:439]*/, v200
	s_wait_alu depctr_vm_vsrc(1)
	s_set_vgpr_msb 0x4000
	v_add_nc_u32_e32 v130, 0x10e80, v132
	s_wait_alu depctr_vm_vsrc(0)
	v_add_nc_u32_e32 v200, 0x10ea0, v132
	s_set_vgpr_msb 64
	ds_load_tr16_b128 v[140:143] /*v[396:399]*/, v132 offset:60032
	ds_load_tr16_b128 v[68:71] /*v[324:327]*/, v132 offset:60064
	ds_load_tr16_b128 v[128:131] /*v[384:387]*/, v132 offset:64640
	s_set_vgpr_msb 0x4000
	ds_load_tr16_b128 v[202:205], v132 offset:64672
	s_wait_alu depctr_va_vdst(1)
	s_set_vgpr_msb 64
	ds_load_tr16_b128 v[132:135] /*v[388:391]*/, v130
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x4000
	ds_load_tr16_b128 v[206:209], v200
	s_set_vgpr_msb 0x80
	ds_load_tr16_b128 v[188:191] /*v[700:703]*/, v132 offset:4608
	ds_load_tr16_b128 v[108:111] /*v[620:623]*/, v132 offset:4640
	ds_load_tr16_b128 v[176:179] /*v[688:691]*/, v132 offset:9216
	ds_load_tr16_b128 v[120:123] /*v[632:635]*/, v132 offset:9248
	ds_load_tr16_b128 v[180:183] /*v[692:695]*/, v132 offset:13824
	ds_load_tr16_b128 v[124:127] /*v[636:639]*/, v132 offset:13856
	ds_load_tr16_b128 v[168:171] /*v[680:683]*/, v132 offset:18432
	ds_load_tr16_b128 v[112:115] /*v[624:627]*/, v132 offset:18464
	ds_load_tr16_b128 v[172:175] /*v[684:687]*/, v132 offset:23040
	ds_load_tr16_b128 v[116:119] /*v[628:631]*/, v132 offset:23072
	ds_load_tr16_b128 v[160:163] /*v[672:675]*/, v132 offset:27648
	ds_load_tr16_b128 v[96:99] /*v[608:611]*/, v132 offset:27680
	ds_load_tr16_b128 v[164:167] /*v[676:679]*/, v132 offset:32256
	ds_load_tr16_b128 v[100:103] /*v[612:615]*/, v132 offset:32288
	ds_load_tr16_b128 v[152:155] /*v[664:667]*/, v132 offset:36864
	ds_load_tr16_b128 v[88:91] /*v[600:603]*/, v132 offset:36896
	ds_load_tr16_b128 v[156:159] /*v[668:671]*/, v132 offset:41472
	ds_load_tr16_b128 v[92:95] /*v[604:607]*/, v132 offset:41504
	ds_load_tr16_b128 v[144:147] /*v[656:659]*/, v132 offset:46080
	ds_load_tr16_b128 v[80:83] /*v[592:595]*/, v132 offset:46112
	ds_load_tr16_b128 v[148:151] /*v[660:663]*/, v132 offset:50688
	ds_load_tr16_b128 v[84:87] /*v[596:599]*/, v132 offset:50720
	ds_load_tr16_b128 v[136:139] /*v[648:651]*/, v132 offset:55296
	ds_load_tr16_b128 v[64:67] /*v[576:579]*/, v132 offset:55328
	ds_load_tr16_b128 v[72:75] /*v[584:587]*/, v132 offset:64
	s_set_vgpr_msb 0x8040
	ds_load_tr16_b128 v[248:251] /*v[504:507]*/, v132 offset:96
	s_set_vgpr_msb 0x4080
	ds_load_tr16_b128 v[76:79] /*v[588:591]*/, v132 offset:4672
	s_set_vgpr_msb 0x8040
	ds_load_tr16_b128 v[252:255] /*v[508:511]*/, v132 offset:4704
	s_set_vgpr_msb 0x4080
	ds_load_tr16_b128 v[56:59] /*v[568:571]*/, v132 offset:9280
	s_set_vgpr_msb 0x8040
	ds_load_tr16_b128 v[240:243] /*v[496:499]*/, v132 offset:9312
	s_set_vgpr_msb 0x4080
	ds_load_tr16_b128 v[60:63] /*v[572:575]*/, v132 offset:13888
	s_set_vgpr_msb 0x8040
	ds_load_tr16_b128 v[244:247] /*v[500:503]*/, v132 offset:13920
	s_set_vgpr_msb 0x4080
	ds_load_tr16_b128 v[40:43] /*v[552:555]*/, v132 offset:18496
	s_set_vgpr_msb 0x8040
	ds_load_tr16_b128 v[232:235] /*v[488:491]*/, v132 offset:18528
	s_set_vgpr_msb 0x4080
	ds_load_tr16_b128 v[44:47] /*v[556:559]*/, v132 offset:23104
	s_set_vgpr_msb 0x8040
	ds_load_tr16_b128 v[236:239] /*v[492:495]*/, v132 offset:23136
	s_set_vgpr_msb 0x4080
	ds_load_tr16_b128 v[32:35] /*v[544:547]*/, v132 offset:27712
	s_set_vgpr_msb 0x8040
	ds_load_tr16_b128 v[224:227] /*v[480:483]*/, v132 offset:27744
	s_set_vgpr_msb 0x4080
	ds_load_tr16_b128 v[36:39] /*v[548:551]*/, v132 offset:32320
	s_set_vgpr_msb 0x8040
	ds_load_tr16_b128 v[228:231] /*v[484:487]*/, v132 offset:32352
	s_set_vgpr_msb 0x4080
	ds_load_tr16_b128 v[24:27] /*v[536:539]*/, v132 offset:36928
	s_set_vgpr_msb 0x8040
	ds_load_tr16_b128 v[216:219] /*v[472:475]*/, v132 offset:36960
	s_set_vgpr_msb 0x4080
	ds_load_tr16_b128 v[28:31] /*v[540:543]*/, v132 offset:41536
	s_set_vgpr_msb 0x8040
	ds_load_tr16_b128 v[220:223] /*v[476:479]*/, v132 offset:41568
	s_set_vgpr_msb 0x4080
	ds_load_tr16_b128 v[16:19] /*v[528:531]*/, v132 offset:46144
	s_set_vgpr_msb 0x8040
	ds_load_tr16_b128 v[208:211] /*v[464:467]*/, v132 offset:46176
	s_set_vgpr_msb 0x4080
	ds_load_tr16_b128 v[20:23] /*v[532:535]*/, v132 offset:50752
	s_set_vgpr_msb 0x8040
	ds_load_tr16_b128 v[212:215] /*v[468:471]*/, v132 offset:50784
	s_set_vgpr_msb 0x4080
	ds_load_tr16_b128 v[8:11] /*v[520:523]*/, v132 offset:55360
	s_set_vgpr_msb 0x8040
	ds_load_tr16_b128 v[192:195] /*v[448:451]*/, v132 offset:55392
	ds_load_tr16_b128 v[200:203] /*v[456:459]*/, v132 offset:128
	ds_load_tr16_b128 v[120:123] /*v[376:379]*/, v132 offset:160
	ds_load_tr16_b128 v[204:207] /*v[460:463]*/, v132 offset:4736
	ds_load_tr16_b128 v[124:127] /*v[380:383]*/, v132 offset:4768
	ds_load_tr16_b128 v[184:187] /*v[440:443]*/, v132 offset:9344
	ds_load_tr16_b128 v[112:115] /*v[368:371]*/, v132 offset:9376
	ds_load_tr16_b128 v[188:191] /*v[444:447]*/, v132 offset:13952
	ds_load_tr16_b128 v[116:119] /*v[372:375]*/, v132 offset:13984
	ds_load_tr16_b128 v[168:171] /*v[424:427]*/, v132 offset:18560
	ds_load_tr16_b128 v[104:107] /*v[360:363]*/, v132 offset:18592
	ds_load_tr16_b128 v[172:175] /*v[428:431]*/, v132 offset:23168
	ds_load_tr16_b128 v[108:111] /*v[364:367]*/, v132 offset:23200
	ds_load_tr16_b128 v[160:163] /*v[416:419]*/, v132 offset:27776
	ds_load_tr16_b128 v[96:99] /*v[352:355]*/, v132 offset:27808
	ds_load_tr16_b128 v[164:167] /*v[420:423]*/, v132 offset:32384
	ds_load_tr16_b128 v[100:103] /*v[356:359]*/, v132 offset:32416
	ds_load_tr16_b128 v[152:155] /*v[408:411]*/, v132 offset:36992
	ds_load_tr16_b128 v[88:91] /*v[344:347]*/, v132 offset:37024
	ds_load_tr16_b128 v[156:159] /*v[412:415]*/, v132 offset:41600
	ds_load_tr16_b128 v[92:95] /*v[348:351]*/, v132 offset:41632
	ds_load_tr16_b128 v[144:147] /*v[400:403]*/, v132 offset:46208
	ds_load_tr16_b128 v[80:83] /*v[336:339]*/, v132 offset:46240
	ds_load_tr16_b128 v[148:151] /*v[404:407]*/, v132 offset:50816
	ds_load_tr16_b128 v[84:87] /*v[340:343]*/, v132 offset:50848
	ds_load_tr16_b128 v[136:139] /*v[392:395]*/, v132 offset:55424
	ds_load_tr16_b128 v[64:67] /*v[320:323]*/, v132 offset:55456
	s_wait_alu depctr_vm_vsrc(6)
	s_set_vgpr_msb 0x4000
	v_add_nc_u32_e32 v130, 0x10ec0, v132
	s_set_vgpr_msb 0x80
	ds_load_tr16_b128 v[184:187] /*v[696:699]*/, v132
	ds_load_tr16_b128 v[104:107] /*v[616:619]*/, v132 offset:32
	s_wait_dscnt 0x3e
	s_clause 0x1
	scratch_store_b128 off, v[202:205], off offset:460 nv
	scratch_store_b128 off, v[206:209], off offset:476 nv
	s_set_vgpr_msb 0x8040
	ds_load_tr16_b128 v[72:75] /*v[328:331]*/, v132 offset:192
	s_wait_alu depctr_vm_vsrc(0)
	s_set_vgpr_msb 0x4000
	ds_load_tr16_b128 v[200:203], v132 offset:224
	s_set_vgpr_msb 64
	ds_load_tr16_b128 v[76:79] /*v[332:335]*/, v132 offset:4800
	s_set_vgpr_msb 0x4000
	ds_load_tr16_b128 v[204:207], v132 offset:4832
	s_wait_dscnt 0x2
	scratch_store_b128 off, v[200:203], off offset:236 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[204:207], off offset:252 nv
	s_set_vgpr_msb 64
	ds_load_tr16_b128 v[56:59] /*v[312:315]*/, v132 offset:9408
	s_wait_alu depctr_vm_vsrc(0)
	s_set_vgpr_msb 0x4000
	ds_load_tr16_b128 v[200:203], v132 offset:9440
	s_set_vgpr_msb 64
	ds_load_tr16_b128 v[60:63] /*v[316:319]*/, v132 offset:14016
	s_set_vgpr_msb 0x4030
	ds_load_tr16_b128 v[204:207], v132 offset:14048
	s_wait_dscnt 0x2
	scratch_store_b128 off, v[200:203], off offset:204 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[204:207], off offset:220 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_tr16_b128 v[204:207], v132 offset:18624
	ds_load_tr16_b128 v[200:203], v132 offset:18656
	ds_load_tr16_b128 v[208:211], v132 offset:23232
	s_wait_dscnt 0x2
	scratch_store_b128 off, v[204:207], off offset:428 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[208:211], off offset:444 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_tr16_b128 v[204:207], v132 offset:23264
	scratch_store_b128 off, v[200:203], off offset:172 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[204:207], off offset:188 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_tr16_b128 v[204:207], v132 offset:27840
	ds_load_tr16_b128 v[200:203], v132 offset:27872
	ds_load_tr16_b128 v[208:211], v132 offset:32448
	s_wait_dscnt 0x2
	scratch_store_b128 off, v[204:207], off offset:396 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[208:211], off offset:412 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_tr16_b128 v[204:207], v132 offset:32480
	scratch_store_b128 off, v[200:203], off offset:140 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[204:207], off offset:156 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_tr16_b128 v[204:207], v132 offset:37056
	ds_load_tr16_b128 v[200:203], v132 offset:37088
	ds_load_tr16_b128 v[208:211], v132 offset:41664
	s_wait_dscnt 0x2
	scratch_store_b128 off, v[204:207], off offset:364 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[208:211], off offset:380 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_tr16_b128 v[204:207], v132 offset:41696
	scratch_store_b128 off, v[200:203], off offset:108 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[204:207], off offset:124 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_tr16_b128 v[204:207], v132 offset:46272
	ds_load_tr16_b128 v[200:203], v132 offset:46304
	ds_load_tr16_b128 v[208:211], v132 offset:50880
	s_wait_dscnt 0x2
	scratch_store_b128 off, v[204:207], off offset:332 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[208:211], off offset:348 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_tr16_b128 v[204:207], v132 offset:50912
	ds_load_tr16_b128 v[210:213], v132 offset:60096
	scratch_store_b128 off, v[200:203], off offset:76 nv
	s_wait_dscnt 0x1
	scratch_store_b128 off, v[204:207], off offset:92 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_tr16_b128 v[206:209], v132 offset:55488
	ds_load_tr16_b128 v[202:205], v132 offset:55520
	v_add_nc_u32_e32 v200, 0x10ee0, v132
	s_wait_dscnt 0x1
	s_clause 0x1
	scratch_store_b128 off, v[206:209], off offset:300 nv
	scratch_store_b128 off, v[210:213], off offset:316 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_tr16_b128 v[206:209], v132 offset:60128
	s_wait_dscnt 0x1
	scratch_store_b128 off, v[202:205], off offset:44 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[206:209], off offset:60 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_tr16_b128 v[206:209], v132 offset:64704
	ds_load_tr16_b128 v[202:205], v132 offset:64736
	s_wait_alu depctr_va_vdst(1)
	ds_load_tr16_b128 v[210:213], v130
	s_wait_dscnt 0x2
	scratch_store_b128 off, v[206:209], off offset:268 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[210:213], off offset:284 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_va_vdst(0) depctr_vm_vsrc(0)
	ds_load_tr16_b128 v[206:209], v200
	scratch_store_b128 off, v[202:205], off offset:12 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[206:209], off offset:28 nv
	v_lshl_or_b32 v130, s52, 8, v200 /*v968*/
	s_set_vgpr_msb 0x3004
	v_cmp_le_i32_e32 vcc_lo, v130, v52 /*v308*/
	s_wait_alu depctr_vm_vsrc(0)
	s_set_vgpr_msb 0x400
	v_dual_add_nc_u32 v207, 17, v130 :: v_dual_bitop2_b32 v200, 2, v130 bitop3:0x54
	v_dual_add_nc_u32 v208, 18, v130 :: v_dual_bitop2_b32 v201, 3, v130 bitop3:0x54
	s_set_vgpr_msb 0xcc
	v_cndmask_b32_e32 v178 /*v946*/, 0xff800000, v192 /*v960*/, vcc_lo
	s_set_vgpr_msb 0xcc04
	v_cmp_lt_i32_e32 vcc_lo, v130, v52 /*v308*/
	s_set_vgpr_msb 0x400
	v_or_b32_e32 v202, 4, v130
	v_or_b32_e32 v203, 5, v130
	v_or_b32_e32 v204, 6, v130
	v_or_b32_e32 v205, 7, v130
	s_set_vgpr_msb 0xcc
	v_cndmask_b32_e32 v179 /*v947*/, 0xff800000, v193 /*v961*/, vcc_lo
	s_set_vgpr_msb 0xcc04
	v_cmp_le_i32_e32 vcc_lo, v200, v52 /*v308*/
	s_set_vgpr_msb 0x400
	v_or_b32_e32 v206, 16, v130
	v_dual_add_nc_u32 v218, 50, v130 :: v_dual_bitop2_b32 v209, 33, v130 bitop3:0x54
	v_dual_add_nc_u32 v219, 51, v130 :: v_dual_bitop2_b32 v210, 34, v130 bitop3:0x54
	s_set_vgpr_msb 0xcc
	v_cndmask_b32_e32 v180 /*v948*/, 0xff800000, v194 /*v962*/, vcc_lo
	s_set_vgpr_msb 0xcc04
	v_cmp_le_i32_e32 vcc_lo, v201, v52 /*v308*/
	s_set_vgpr_msb 0x400
	v_dual_add_nc_u32 v220, 52, v130 :: v_dual_bitop2_b32 v211, 35, v130 bitop3:0x54
	v_dual_add_nc_u32 v221, 53, v130 :: v_dual_bitop2_b32 v212, 36, v130 bitop3:0x54
	s_set_vgpr_msb 0xcc
	v_cndmask_b32_e32 v181 /*v949*/, 0xff800000, v195 /*v963*/, vcc_lo
	s_set_vgpr_msb 0xcc04
	v_cmp_le_i32_e32 vcc_lo, v202, v52 /*v308*/
	s_set_vgpr_msb 0x400
	v_dual_add_nc_u32 v222, 54, v130 :: v_dual_bitop2_b32 v213, 37, v130 bitop3:0x54
	v_dual_add_nc_u32 v223, 55, v130 :: v_dual_bitop2_b32 v214, 38, v130 bitop3:0x54
	s_set_vgpr_msb 0xcc
	v_cndmask_b32_e32 v182 /*v950*/, 0xff800000, v196 /*v964*/, vcc_lo
	s_set_vgpr_msb 0xcc04
	v_cmp_le_i32_e32 vcc_lo, v203, v52 /*v308*/
	s_set_vgpr_msb 0x400
	v_or_b32_e32 v215, 39, v130
	v_or_b32_e32 v216, 48, v130
	v_or_b32_e32 v224, 64, v130
	v_or_b32_e32 v225, 0x41, v130
	s_set_vgpr_msb 0xcc
	v_cndmask_b32_e32 v183 /*v951*/, 0xff800000, v197 /*v965*/, vcc_lo
	s_set_vgpr_msb 0xcc04
	v_cmp_le_i32_e32 vcc_lo, v204, v52 /*v308*/
	s_set_vgpr_msb 0x400
	v_or_b32_e32 v226, 0x42, v130
	v_or_b32_e32 v227, 0x43, v130
	v_or_b32_e32 v228, 0x44, v130
	v_or_b32_e32 v229, 0x45, v130
	s_set_vgpr_msb 0xcc
	v_cndmask_b32_e32 v184 /*v952*/, 0xff800000, v198 /*v966*/, vcc_lo
	s_set_vgpr_msb 0xcc04
	v_cmp_le_i32_e32 vcc_lo, v205, v52 /*v308*/
	s_set_vgpr_msb 0x400
	v_or_b32_e32 v230, 0x46, v130
	v_or_b32_e32 v231, 0x47, v130
	v_or_b32_e32 v232, 0x50, v130
	v_add_nc_u32_e32 v233, 0x51, v130
	s_set_vgpr_msb 0xcc
	v_cndmask_b32_e32 v185 /*v953*/, 0xff800000, v199 /*v967*/, vcc_lo
	s_set_vgpr_msb 0xcc04
	v_cmp_le_i32_e32 vcc_lo, v206, v52 /*v308*/
	s_set_vgpr_msb 0x400
	v_add_nc_u32_e32 v234, 0x52, v130
	v_add_nc_u32_e32 v235, 0x53, v130
	v_add_nc_u32_e32 v236, 0x54, v130
	v_add_nc_u32_e32 v237, 0x55, v130
	s_set_vgpr_msb 0xc0
	v_cndmask_b32_e32 v186 /*v954*/, 0xff800000, v192, vcc_lo
	s_set_vgpr_msb 0xc004
	v_cmp_le_i32_e32 vcc_lo, v207, v52 /*v308*/
	s_set_vgpr_msb 0x400
	v_add_nc_u32_e32 v192, 19, v130
	v_add_nc_u32_e32 v238, 0x56, v130
	v_add_nc_u32_e32 v239, 0x57, v130
	v_or_b32_e32 v240, 0x60, v130
	s_set_vgpr_msb 0xc0
	v_cndmask_b32_e32 v187 /*v955*/, 0xff800000, v193, vcc_lo
	s_set_vgpr_msb 0xc004
	v_cmp_le_i32_e32 vcc_lo, v208, v52 /*v308*/
	s_set_vgpr_msb 0x400
	v_add_nc_u32_e32 v193, 20, v130
	v_or_b32_e32 v241, 0x61, v130
	v_or_b32_e32 v242, 0x62, v130
	v_or_b32_e32 v243, 0x63, v130
	s_set_vgpr_msb 0xc0
	v_cndmask_b32_e32 v188 /*v956*/, 0xff800000, v194, vcc_lo
	s_set_vgpr_msb 0xc004
	v_cmp_le_i32_e32 vcc_lo, v192, v52 /*v308*/
	s_set_vgpr_msb 0x400
	v_add_nc_u32_e32 v194, 21, v130
	v_or_b32_e32 v244, 0x64, v130
	v_add_nc_u32_e32 v217, 49, v130
	v_or_b32_e32 v245, 0x65, v130
	s_set_vgpr_msb 0xc0
	v_cndmask_b32_e32 v189 /*v957*/, 0xff800000, v195, vcc_lo
	s_set_vgpr_msb 0xc004
	v_cmp_le_i32_e32 vcc_lo, v193, v52 /*v308*/
	s_set_vgpr_msb 0x400
	v_add_nc_u32_e32 v195, 22, v130
	v_or_b32_e32 v246, 0x66, v130
	v_or_b32_e32 v247, 0x67, v130
	v_or_b32_e32 v248, 0x70, v130
	s_set_vgpr_msb 0xc0
	v_cndmask_b32_e32 v190 /*v958*/, 0xff800000, v196, vcc_lo
	s_set_vgpr_msb 0xc004
	v_cmp_le_i32_e32 vcc_lo, v194, v52 /*v308*/
	s_set_vgpr_msb 0x400
	v_add_nc_u32_e32 v196, 23, v130
	v_add_nc_u32_e32 v249, 0x71, v130
	v_add_nc_u32_e32 v250, 0x72, v130
	v_add_nc_u32_e32 v251, 0x73, v130
	s_set_vgpr_msb 0xc0
	v_cndmask_b32_e32 v191 /*v959*/, 0xff800000, v197, vcc_lo
	s_set_vgpr_msb 0xc004
	v_cmp_le_i32_e32 vcc_lo, v195, v52 /*v308*/
	s_set_vgpr_msb 0x400
	v_or_b32_e32 v197, 32, v130
	v_add_nc_u32_e32 v252, 0x74, v130
	v_add_nc_u32_e32 v253, 0x75, v130
	v_add_nc_u32_e32 v254, 0x76, v130
	s_set_vgpr_msb 0xc0
	v_cndmask_b32_e32 v192 /*v960*/, 0xff800000, v198, vcc_lo
	s_set_vgpr_msb 0xc004
	v_cmp_le_i32_e32 vcc_lo, v196, v52 /*v308*/
	s_set_vgpr_msb 0x400
	v_add_nc_u32_e32 v255, 0x77, v130
	s_set_vgpr_msb 64
	v_or_b32_e32 v0 /*v256*/, 0x80, v130
	v_or_b32_e32 v1 /*v257*/, 0x81, v130
	v_or_b32_e32 v2 /*v258*/, 0x82, v130
	s_set_vgpr_msb 0x40c0
	v_cndmask_b32_e32 v193 /*v961*/, 0xff800000, v199, vcc_lo
	s_set_vgpr_msb 0xc004
	v_cmp_le_i32_e32 vcc_lo, v197, v52 /*v308*/
	s_set_vgpr_msb 0x440
	v_or_b32_e32 v3 /*v259*/, 0x83, v130
	v_or_b32_e32 v4 /*v260*/, 0x84, v130
	v_or_b32_e32 v5 /*v261*/, 0x85, v130
	v_or_b32_e32 v6 /*v262*/, 0x86, v130
	s_set_vgpr_msb 0x40cc
	v_cndmask_b32_e32 v168 /*v936*/, 0xff800000, v168 /*v936*/, vcc_lo
	s_set_vgpr_msb 0xcc04
	v_cmp_le_i32_e32 vcc_lo, v209, v52 /*v308*/
	s_set_vgpr_msb 0x440
	v_or_b32_e32 v7 /*v263*/, 0x87, v130
	v_or_b32_e32 v8 /*v264*/, 0x90, v130
	v_add_nc_u32_e32 v9 /*v265*/, 0x91, v130
	v_add_nc_u32_e32 v10 /*v266*/, 0x92, v130
	s_set_vgpr_msb 0x40cc
	v_cndmask_b32_e32 v169 /*v937*/, 0xff800000, v169 /*v937*/, vcc_lo
	s_set_vgpr_msb 0xcc04
	v_cmp_le_i32_e32 vcc_lo, v210, v52 /*v308*/
	s_set_vgpr_msb 0x440
	v_add_nc_u32_e32 v11 /*v267*/, 0x93, v130
	v_add_nc_u32_e32 v12 /*v268*/, 0x94, v130
	v_add_nc_u32_e32 v13 /*v269*/, 0x95, v130
	v_add_nc_u32_e32 v14 /*v270*/, 0x96, v130
	s_set_vgpr_msb 0x40cc
	v_cndmask_b32_e32 v170 /*v938*/, 0xff800000, v170 /*v938*/, vcc_lo
	s_set_vgpr_msb 0xcc04
	v_cmp_le_i32_e32 vcc_lo, v211, v52 /*v308*/
	s_set_vgpr_msb 0x440
	v_add_nc_u32_e32 v15 /*v271*/, 0x97, v130
	v_or_b32_e32 v16 /*v272*/, 0xa0, v130
	v_or_b32_e32 v17 /*v273*/, 0xa1, v130
	v_or_b32_e32 v18 /*v274*/, 0xa2, v130
	s_set_vgpr_msb 0x40cc
	v_cndmask_b32_e32 v171 /*v939*/, 0xff800000, v171 /*v939*/, vcc_lo
	s_set_vgpr_msb 0xcc04
	v_cmp_le_i32_e32 vcc_lo, v212, v52 /*v308*/
	s_set_vgpr_msb 0x440
	v_or_b32_e32 v19 /*v275*/, 0xa3, v130
	v_or_b32_e32 v20 /*v276*/, 0xa4, v130
	v_or_b32_e32 v21 /*v277*/, 0xa5, v130
	v_or_b32_e32 v22 /*v278*/, 0xa6, v130
	s_set_vgpr_msb 0x40cc
	v_cndmask_b32_e32 v172 /*v940*/, 0xff800000, v172 /*v940*/, vcc_lo
	s_set_vgpr_msb 0xcc04
	v_cmp_le_i32_e32 vcc_lo, v213, v52 /*v308*/
	s_set_vgpr_msb 0x440
	v_or_b32_e32 v23 /*v279*/, 0xa7, v130
	v_or_b32_e32 v24 /*v280*/, 0xb0, v130
	v_add_nc_u32_e32 v25 /*v281*/, 0xb1, v130
	v_add_nc_u32_e32 v26 /*v282*/, 0xb2, v130
	s_set_vgpr_msb 0x40cc
	v_cndmask_b32_e32 v173 /*v941*/, 0xff800000, v173 /*v941*/, vcc_lo
	s_set_vgpr_msb 0xcc04
	v_cmp_le_i32_e32 vcc_lo, v214, v52 /*v308*/
	s_set_vgpr_msb 0x440
	v_add_nc_u32_e32 v27 /*v283*/, 0xb3, v130
	v_add_nc_u32_e32 v28 /*v284*/, 0xb4, v130
	v_add_nc_u32_e32 v29 /*v285*/, 0xb5, v130
	v_add_nc_u32_e32 v30 /*v286*/, 0xb6, v130
	s_set_vgpr_msb 0x40cc
	v_cndmask_b32_e32 v174 /*v942*/, 0xff800000, v174 /*v942*/, vcc_lo
	s_set_vgpr_msb 0xcc04
	v_cmp_le_i32_e32 vcc_lo, v215, v52 /*v308*/
	s_set_vgpr_msb 0x440
	v_add_nc_u32_e32 v31 /*v287*/, 0xb7, v130
	v_or_b32_e32 v32 /*v288*/, 0xc0, v130
	v_or_b32_e32 v33 /*v289*/, 0xc1, v130
	v_or_b32_e32 v34 /*v290*/, 0xc2, v130
	s_set_vgpr_msb 0x40cc
	v_cndmask_b32_e32 v175 /*v943*/, 0xff800000, v175 /*v943*/, vcc_lo
	s_set_vgpr_msb 0xcc04
	v_cmp_le_i32_e32 vcc_lo, v216, v52 /*v308*/
	s_set_vgpr_msb 0x440
	v_or_b32_e32 v35 /*v291*/, 0xc3, v130
	v_or_b32_e32 v36 /*v292*/, 0xc4, v130
	v_or_b32_e32 v37 /*v293*/, 0xc5, v130
	v_or_b32_e32 v38 /*v294*/, 0xc6, v130
	s_set_vgpr_msb 0x40cc
	v_cndmask_b32_e32 v160 /*v928*/, 0xff800000, v160 /*v928*/, vcc_lo
	s_set_vgpr_msb 0xcc04
	v_cmp_le_i32_e32 vcc_lo, v217, v52 /*v308*/
	s_set_vgpr_msb 0x440
	v_or_b32_e32 v39 /*v295*/, 0xc7, v130
	v_or_b32_e32 v40 /*v296*/, 0xd0, v130
	v_add_nc_u32_e32 v41 /*v297*/, 0xd1, v130
	v_add_nc_u32_e32 v50 /*v306*/, 0xd2, v130
	s_set_vgpr_msb 0x40cc
	v_cndmask_b32_e32 v161 /*v929*/, 0xff800000, v161 /*v929*/, vcc_lo
	s_set_vgpr_msb 0xcc04
	v_cmp_le_i32_e32 vcc_lo, v218, v52 /*v308*/
	s_set_vgpr_msb 0x440
	v_add_nc_u32_e32 v51 /*v307*/, 0xd3, v130
	v_add_nc_u32_e32 v54 /*v310*/, 0xd4, v130
	s_set_vgpr_msb 0x40cc
	v_cndmask_b32_e32 v162 /*v930*/, 0xff800000, v162 /*v930*/, vcc_lo
	s_set_vgpr_msb 0xcc04
	v_cmp_le_i32_e32 vcc_lo, v219, v52 /*v308*/
	s_set_vgpr_msb 0x4cc
	v_cndmask_b32_e32 v163 /*v931*/, 0xff800000, v163 /*v931*/, vcc_lo
	s_set_vgpr_msb 0xcc04
	v_cmp_le_i32_e32 vcc_lo, v220, v52 /*v308*/
	s_set_vgpr_msb 0x4cc
	v_cndmask_b32_e32 v164 /*v932*/, 0xff800000, v164 /*v932*/, vcc_lo
	s_set_vgpr_msb 0xcc04
	v_cmp_le_i32_e32 vcc_lo, v221, v52 /*v308*/
	s_set_vgpr_msb 0x4cc
	v_cndmask_b32_e32 v165 /*v933*/, 0xff800000, v165 /*v933*/, vcc_lo
	s_set_vgpr_msb 0xcc04
	v_cmp_le_i32_e32 vcc_lo, v222, v52 /*v308*/
	s_set_vgpr_msb 0x4cc
	v_cndmask_b32_e32 v166 /*v934*/, 0xff800000, v166 /*v934*/, vcc_lo
	s_set_vgpr_msb 0xcc04
	v_cmp_le_i32_e32 vcc_lo, v223, v52 /*v308*/
	s_set_vgpr_msb 0x4cc
	v_cndmask_b32_e32 v167 /*v935*/, 0xff800000, v167 /*v935*/, vcc_lo
	s_set_vgpr_msb 0xcc04
	v_cmp_le_i32_e32 vcc_lo, v224, v52 /*v308*/
	s_set_vgpr_msb 0x4cc
	v_cndmask_b32_e32 v152 /*v920*/, 0xff800000, v152 /*v920*/, vcc_lo
	s_set_vgpr_msb 0xcc04
	v_cmp_le_i32_e32 vcc_lo, v225, v52 /*v308*/
	s_set_vgpr_msb 0x4cc
	v_cndmask_b32_e32 v153 /*v921*/, 0xff800000, v153 /*v921*/, vcc_lo
	s_set_vgpr_msb 0xcc04
	v_cmp_le_i32_e32 vcc_lo, v226, v52 /*v308*/
	s_set_vgpr_msb 0x4cc
	v_cndmask_b32_e32 v154 /*v922*/, 0xff800000, v154 /*v922*/, vcc_lo
	s_set_vgpr_msb 0xcc04
	v_cmp_le_i32_e32 vcc_lo, v227, v52 /*v308*/
	s_set_vgpr_msb 0x4cc
	v_cndmask_b32_e32 v155 /*v923*/, 0xff800000, v155 /*v923*/, vcc_lo
	s_set_vgpr_msb 0xcc04
	v_cmp_le_i32_e32 vcc_lo, v228, v52 /*v308*/
	s_set_vgpr_msb 0x4cc
	v_cndmask_b32_e32 v156 /*v924*/, 0xff800000, v156 /*v924*/, vcc_lo
	s_set_vgpr_msb 0xcc04
	v_cmp_le_i32_e32 vcc_lo, v229, v52 /*v308*/
	s_set_vgpr_msb 0x4cc
	v_cndmask_b32_e32 v157 /*v925*/, 0xff800000, v157 /*v925*/, vcc_lo
	s_set_vgpr_msb 0xcc04
	v_cmp_le_i32_e32 vcc_lo, v230, v52 /*v308*/
	s_set_vgpr_msb 0x4cc
	v_cndmask_b32_e32 v158 /*v926*/, 0xff800000, v158 /*v926*/, vcc_lo
	s_set_vgpr_msb 0xcc04
	v_cmp_le_i32_e32 vcc_lo, v231, v52 /*v308*/
	s_set_vgpr_msb 0x4cc
	v_cndmask_b32_e32 v159 /*v927*/, 0xff800000, v159 /*v927*/, vcc_lo
	s_set_vgpr_msb 0xcc04
	v_cmp_le_i32_e32 vcc_lo, v232, v52 /*v308*/
	s_set_vgpr_msb 0x4cc
	v_cndmask_b32_e32 v144 /*v912*/, 0xff800000, v144 /*v912*/, vcc_lo
	s_set_vgpr_msb 0xcc04
	v_cmp_le_i32_e32 vcc_lo, v233, v52 /*v308*/
	s_set_vgpr_msb 0x4cc
	v_cndmask_b32_e32 v145 /*v913*/, 0xff800000, v145 /*v913*/, vcc_lo
	s_set_vgpr_msb 0xcc04
	v_cmp_le_i32_e32 vcc_lo, v234, v52 /*v308*/
	s_set_vgpr_msb 0x4cc
	v_cndmask_b32_e32 v146 /*v914*/, 0xff800000, v146 /*v914*/, vcc_lo
	s_set_vgpr_msb 0xcc04
	v_cmp_le_i32_e32 vcc_lo, v235, v52 /*v308*/
	s_set_vgpr_msb 0x4cc
	v_cndmask_b32_e32 v147 /*v915*/, 0xff800000, v147 /*v915*/, vcc_lo
	s_set_vgpr_msb 0xcc04
	v_cmp_le_i32_e32 vcc_lo, v236, v52 /*v308*/
	s_set_vgpr_msb 0x4cc
	v_cndmask_b32_e32 v148 /*v916*/, 0xff800000, v148 /*v916*/, vcc_lo
	s_set_vgpr_msb 0xcc04
	v_cmp_le_i32_e32 vcc_lo, v237, v52 /*v308*/
	s_set_vgpr_msb 0x4cc
	v_cndmask_b32_e32 v149 /*v917*/, 0xff800000, v149 /*v917*/, vcc_lo
	s_set_vgpr_msb 0xcc04
	v_cmp_le_i32_e32 vcc_lo, v238, v52 /*v308*/
	s_set_vgpr_msb 0x4cc
	v_cndmask_b32_e32 v150 /*v918*/, 0xff800000, v150 /*v918*/, vcc_lo
	s_set_vgpr_msb 0xcc04
	v_cmp_le_i32_e32 vcc_lo, v239, v52 /*v308*/
	s_set_vgpr_msb 0x4cc
	v_cndmask_b32_e32 v151 /*v919*/, 0xff800000, v151 /*v919*/, vcc_lo
	s_set_vgpr_msb 0xcc04
	v_cmp_le_i32_e32 vcc_lo, v240, v52 /*v308*/
	s_set_vgpr_msb 0x4cc
	v_cndmask_b32_e32 v136 /*v904*/, 0xff800000, v136 /*v904*/, vcc_lo
	s_set_vgpr_msb 0xcc04
	v_cmp_le_i32_e32 vcc_lo, v241, v52 /*v308*/
	s_set_vgpr_msb 0x4cc
	v_cndmask_b32_e32 v137 /*v905*/, 0xff800000, v137 /*v905*/, vcc_lo
	s_set_vgpr_msb 0xcc04
	v_cmp_le_i32_e32 vcc_lo, v242, v52 /*v308*/
	s_set_vgpr_msb 0x4cc
	v_cndmask_b32_e32 v138 /*v906*/, 0xff800000, v138 /*v906*/, vcc_lo
	s_set_vgpr_msb 0xcc04
	v_cmp_le_i32_e32 vcc_lo, v243, v52 /*v308*/
	s_set_vgpr_msb 0x4cc
	v_cndmask_b32_e32 v139 /*v907*/, 0xff800000, v139 /*v907*/, vcc_lo
	s_set_vgpr_msb 0xcc04
	v_cmp_le_i32_e32 vcc_lo, v244, v52 /*v308*/
	s_set_vgpr_msb 0x4cc
	v_cndmask_b32_e32 v140 /*v908*/, 0xff800000, v140 /*v908*/, vcc_lo
	s_set_vgpr_msb 0xcc04
	v_cmp_le_i32_e32 vcc_lo, v245, v52 /*v308*/
	s_set_vgpr_msb 0x4cc
	v_cndmask_b32_e32 v141 /*v909*/, 0xff800000, v141 /*v909*/, vcc_lo
	s_set_vgpr_msb 0xcc04
	v_cmp_le_i32_e32 vcc_lo, v246, v52 /*v308*/
	s_set_vgpr_msb 0x4cc
	v_cndmask_b32_e32 v142 /*v910*/, 0xff800000, v142 /*v910*/, vcc_lo
	s_set_vgpr_msb 0xcc04
	v_cmp_le_i32_e32 vcc_lo, v247, v52 /*v308*/
	s_set_vgpr_msb 0x4cc
	v_cndmask_b32_e32 v143 /*v911*/, 0xff800000, v143 /*v911*/, vcc_lo
	s_set_vgpr_msb 0xcc04
	v_cmp_le_i32_e32 vcc_lo, v248, v52 /*v308*/
	s_set_vgpr_msb 0x4cc
	v_cndmask_b32_e32 v128 /*v896*/, 0xff800000, v128 /*v896*/, vcc_lo
	s_set_vgpr_msb 0xcc04
	v_cmp_le_i32_e32 vcc_lo, v249, v52 /*v308*/
	s_set_vgpr_msb 0x4cc
	v_cndmask_b32_e32 v129 /*v897*/, 0xff800000, v129 /*v897*/, vcc_lo
	s_set_vgpr_msb 0xcc04
	v_cmp_le_i32_e32 vcc_lo, v250, v52 /*v308*/
	s_set_vgpr_msb 0x4cc
	v_cndmask_b32_e32 v130 /*v898*/, 0xff800000, v130 /*v898*/, vcc_lo
	s_set_vgpr_msb 0xcc04
	v_cmp_le_i32_e32 vcc_lo, v251, v52 /*v308*/
	s_set_vgpr_msb 0x4cc
	v_cndmask_b32_e32 v131 /*v899*/, 0xff800000, v131 /*v899*/, vcc_lo
	s_set_vgpr_msb 0xcc04
	v_cmp_le_i32_e32 vcc_lo, v252, v52 /*v308*/
	s_set_vgpr_msb 0x4cc
	v_cndmask_b32_e32 v132 /*v900*/, 0xff800000, v132 /*v900*/, vcc_lo
	s_set_vgpr_msb 0xcc04
	v_cmp_le_i32_e32 vcc_lo, v253, v52 /*v308*/
	s_set_vgpr_msb 0x4cc
	v_cndmask_b32_e32 v133 /*v901*/, 0xff800000, v133 /*v901*/, vcc_lo
	s_set_vgpr_msb 0xcc04
	v_cmp_le_i32_e32 vcc_lo, v254, v52 /*v308*/
	s_set_vgpr_msb 0x4cc
	v_cndmask_b32_e32 v134 /*v902*/, 0xff800000, v134 /*v902*/, vcc_lo
	s_set_vgpr_msb 0xcc04
	v_cmp_le_i32_e32 vcc_lo, v255, v52 /*v308*/
	s_set_vgpr_msb 0x4cc
	v_cndmask_b32_e32 v135 /*v903*/, 0xff800000, v135 /*v903*/, vcc_lo
	s_set_vgpr_msb 0xcc05
	v_cmp_le_i32_e32 vcc_lo, v0 /*v256*/, v52 /*v308*/
	s_set_vgpr_msb 0x5cc
	v_cndmask_b32_e32 v120 /*v888*/, 0xff800000, v120 /*v888*/, vcc_lo
	s_set_vgpr_msb 0xcc05
	v_cmp_le_i32_e32 vcc_lo, v1 /*v257*/, v52 /*v308*/
	s_set_vgpr_msb 0x5cc
	v_cndmask_b32_e32 v121 /*v889*/, 0xff800000, v121 /*v889*/, vcc_lo
	s_set_vgpr_msb 0xcc05
	v_cmp_le_i32_e32 vcc_lo, v2 /*v258*/, v52 /*v308*/
	s_set_vgpr_msb 0x5cc
	v_cndmask_b32_e32 v122 /*v890*/, 0xff800000, v122 /*v890*/, vcc_lo
	s_set_vgpr_msb 0xcc05
	v_cmp_le_i32_e32 vcc_lo, v3 /*v259*/, v52 /*v308*/
	s_set_vgpr_msb 0x5cc
	v_cndmask_b32_e32 v123 /*v891*/, 0xff800000, v123 /*v891*/, vcc_lo
	s_set_vgpr_msb 0xcc05
	v_cmp_le_i32_e32 vcc_lo, v4 /*v260*/, v52 /*v308*/
	s_set_vgpr_msb 0x5cc
	v_cndmask_b32_e32 v124 /*v892*/, 0xff800000, v124 /*v892*/, vcc_lo
	s_set_vgpr_msb 0xcc05
	v_cmp_le_i32_e32 vcc_lo, v5 /*v261*/, v52 /*v308*/
	s_set_vgpr_msb 0x5cc
	v_cndmask_b32_e32 v125 /*v893*/, 0xff800000, v125 /*v893*/, vcc_lo
	s_set_vgpr_msb 0xcc05
	v_cmp_le_i32_e32 vcc_lo, v6 /*v262*/, v52 /*v308*/
	s_set_vgpr_msb 0x5cc
	v_cndmask_b32_e32 v176 /*v944*/, 0xff800000, v126 /*v894*/, vcc_lo
	s_set_vgpr_msb 0xcc05
	v_cmp_le_i32_e32 vcc_lo, v7 /*v263*/, v52 /*v308*/
	s_set_vgpr_msb 0x5cc
	v_cndmask_b32_e32 v177 /*v945*/, 0xff800000, v127 /*v895*/, vcc_lo
	s_set_vgpr_msb 0xcc05
	v_cmp_le_i32_e32 vcc_lo, v8 /*v264*/, v52 /*v308*/
	s_set_vgpr_msb 0x5cc
	v_cndmask_b32_e32 v194 /*v962*/, 0xff800000, v112 /*v880*/, vcc_lo
	s_set_vgpr_msb 0xcc05
	v_cmp_le_i32_e32 vcc_lo, v9 /*v265*/, v52 /*v308*/
	s_set_vgpr_msb 0x5cc
	v_cndmask_b32_e32 v195 /*v963*/, 0xff800000, v113 /*v881*/, vcc_lo
	s_set_vgpr_msb 0xcc05
	v_cmp_le_i32_e32 vcc_lo, v10 /*v266*/, v52 /*v308*/
	s_set_vgpr_msb 0x5cc
	v_cndmask_b32_e32 v114 /*v882*/, 0xff800000, v114 /*v882*/, vcc_lo
	s_set_vgpr_msb 0xcc05
	v_cmp_le_i32_e32 vcc_lo, v11 /*v267*/, v52 /*v308*/
	s_set_vgpr_msb 0x5cc
	v_cndmask_b32_e32 v115 /*v883*/, 0xff800000, v115 /*v883*/, vcc_lo
	s_set_vgpr_msb 0xcc05
	v_cmp_le_i32_e32 vcc_lo, v12 /*v268*/, v52 /*v308*/
	s_set_vgpr_msb 0x5cc
	v_cndmask_b32_e32 v116 /*v884*/, 0xff800000, v116 /*v884*/, vcc_lo
	s_set_vgpr_msb 0xcc05
	v_cmp_le_i32_e32 vcc_lo, v13 /*v269*/, v52 /*v308*/
	s_set_vgpr_msb 0x5cc
	v_cndmask_b32_e32 v117 /*v885*/, 0xff800000, v117 /*v885*/, vcc_lo
	s_set_vgpr_msb 0xcc05
	v_cmp_le_i32_e32 vcc_lo, v14 /*v270*/, v52 /*v308*/
	s_set_vgpr_msb 0x5cc
	v_cndmask_b32_e32 v118 /*v886*/, 0xff800000, v118 /*v886*/, vcc_lo
	s_set_vgpr_msb 0xcc05
	v_cmp_le_i32_e32 vcc_lo, v15 /*v271*/, v52 /*v308*/
	s_set_vgpr_msb 0x5cc
	v_cndmask_b32_e32 v119 /*v887*/, 0xff800000, v119 /*v887*/, vcc_lo
	s_set_vgpr_msb 0xcc05
	v_cmp_le_i32_e32 vcc_lo, v16 /*v272*/, v52 /*v308*/
	s_set_vgpr_msb 0x5cc
	v_cndmask_b32_e32 v104 /*v872*/, 0xff800000, v104 /*v872*/, vcc_lo
	s_set_vgpr_msb 0xcc05
	v_cmp_le_i32_e32 vcc_lo, v17 /*v273*/, v52 /*v308*/
	s_set_vgpr_msb 0x5cc
	v_cndmask_b32_e32 v105 /*v873*/, 0xff800000, v105 /*v873*/, vcc_lo
	s_set_vgpr_msb 0xcc05
	v_cmp_le_i32_e32 vcc_lo, v18 /*v274*/, v52 /*v308*/
	s_set_vgpr_msb 0x5cc
	v_cndmask_b32_e32 v196 /*v964*/, 0xff800000, v106 /*v874*/, vcc_lo
	s_set_vgpr_msb 0xcc05
	v_cmp_le_i32_e32 vcc_lo, v19 /*v275*/, v52 /*v308*/
	s_set_vgpr_msb 0x5cc
	v_cndmask_b32_e32 v197 /*v965*/, 0xff800000, v107 /*v875*/, vcc_lo
	s_set_vgpr_msb 0xcc05
	v_cmp_le_i32_e32 vcc_lo, v20 /*v276*/, v52 /*v308*/
	s_set_vgpr_msb 0x5cc
	v_cndmask_b32_e32 v108 /*v876*/, 0xff800000, v108 /*v876*/, vcc_lo
	s_set_vgpr_msb 0xcc05
	v_cmp_le_i32_e32 vcc_lo, v21 /*v277*/, v52 /*v308*/
	s_set_vgpr_msb 0x5cc
	v_cndmask_b32_e32 v109 /*v877*/, 0xff800000, v109 /*v877*/, vcc_lo
	s_set_vgpr_msb 0xcc05
	v_cmp_le_i32_e32 vcc_lo, v22 /*v278*/, v52 /*v308*/
	s_set_vgpr_msb 0x5cc
	v_cndmask_b32_e32 v110 /*v878*/, 0xff800000, v110 /*v878*/, vcc_lo
	s_set_vgpr_msb 0xcc05
	v_cmp_le_i32_e32 vcc_lo, v23 /*v279*/, v52 /*v308*/
	s_set_vgpr_msb 0x5cc
	v_cndmask_b32_e32 v111 /*v879*/, 0xff800000, v111 /*v879*/, vcc_lo
	s_set_vgpr_msb 0xcc05
	v_cmp_le_i32_e32 vcc_lo, v24 /*v280*/, v52 /*v308*/
	s_set_vgpr_msb 0x5cc
	v_cndmask_b32_e32 v198 /*v966*/, 0xff800000, v96 /*v864*/, vcc_lo
	s_set_vgpr_msb 0xcc05
	v_cmp_le_i32_e32 vcc_lo, v25 /*v281*/, v52 /*v308*/
	s_set_vgpr_msb 0x5cc
	v_cndmask_b32_e32 v199 /*v967*/, 0xff800000, v97 /*v865*/, vcc_lo
	s_set_vgpr_msb 0xcc05
	v_cmp_le_i32_e32 vcc_lo, v26 /*v282*/, v52 /*v308*/
	s_set_vgpr_msb 0x5cc
	v_cndmask_b32_e32 v200 /*v968*/, 0xff800000, v98 /*v866*/, vcc_lo
	s_set_vgpr_msb 0xcc05
	v_cmp_le_i32_e32 vcc_lo, v27 /*v283*/, v52 /*v308*/
	s_set_vgpr_msb 0x5cc
	v_cndmask_b32_e32 v201 /*v969*/, 0xff800000, v99 /*v867*/, vcc_lo
	s_set_vgpr_msb 0xcc05
	v_cmp_le_i32_e32 vcc_lo, v28 /*v284*/, v52 /*v308*/
	s_set_vgpr_msb 0x5cc
	v_cndmask_b32_e32 v100 /*v868*/, 0xff800000, v100 /*v868*/, vcc_lo
	s_set_vgpr_msb 0xcc05
	v_cmp_le_i32_e32 vcc_lo, v29 /*v285*/, v52 /*v308*/
	s_set_vgpr_msb 0x5cc
	v_cndmask_b32_e32 v101 /*v869*/, 0xff800000, v101 /*v869*/, vcc_lo
	s_set_vgpr_msb 0xcc05
	v_cmp_le_i32_e32 vcc_lo, v30 /*v286*/, v52 /*v308*/
	s_set_vgpr_msb 0x5cc
	v_cndmask_b32_e32 v202 /*v970*/, 0xff800000, v102 /*v870*/, vcc_lo
	s_set_vgpr_msb 0xcc05
	v_cmp_le_i32_e32 vcc_lo, v31 /*v287*/, v52 /*v308*/
	s_set_vgpr_msb 0x5cc
	v_cndmask_b32_e32 v203 /*v971*/, 0xff800000, v103 /*v871*/, vcc_lo
	s_set_vgpr_msb 0xcc05
	v_cmp_le_i32_e32 vcc_lo, v32 /*v288*/, v52 /*v308*/
	s_set_vgpr_msb 0x5cc
	v_cndmask_b32_e32 v204 /*v972*/, 0xff800000, v88 /*v856*/, vcc_lo
	s_set_vgpr_msb 0xcc05
	v_cmp_le_i32_e32 vcc_lo, v33 /*v289*/, v52 /*v308*/
	s_set_vgpr_msb 0x5cc
	v_cndmask_b32_e32 v205 /*v973*/, 0xff800000, v89 /*v857*/, vcc_lo
	s_set_vgpr_msb 0xcc05
	v_cmp_le_i32_e32 vcc_lo, v34 /*v290*/, v52 /*v308*/
	s_set_vgpr_msb 0x5cc
	v_cndmask_b32_e32 v206 /*v974*/, 0xff800000, v90 /*v858*/, vcc_lo
	s_set_vgpr_msb 0xcc05
	v_cmp_le_i32_e32 vcc_lo, v35 /*v291*/, v52 /*v308*/
	s_set_vgpr_msb 0x5cc
	v_cndmask_b32_e32 v207 /*v975*/, 0xff800000, v91 /*v859*/, vcc_lo
	s_set_vgpr_msb 0xcc05
	v_cmp_le_i32_e32 vcc_lo, v36 /*v292*/, v52 /*v308*/
	s_set_vgpr_msb 0x5cc
	v_cndmask_b32_e32 v208 /*v976*/, 0xff800000, v92 /*v860*/, vcc_lo
	s_set_vgpr_msb 0xcc05
	v_cmp_le_i32_e32 vcc_lo, v37 /*v293*/, v52 /*v308*/
	s_set_vgpr_msb 0x5cc
	v_cndmask_b32_e32 v209 /*v977*/, 0xff800000, v93 /*v861*/, vcc_lo
	s_set_vgpr_msb 0xcc05
	v_cmp_le_i32_e32 vcc_lo, v38 /*v294*/, v52 /*v308*/
	s_set_vgpr_msb 0x5cc
	v_cndmask_b32_e32 v94 /*v862*/, 0xff800000, v94 /*v862*/, vcc_lo
	s_set_vgpr_msb 0xcc05
	v_cmp_le_i32_e32 vcc_lo, v39 /*v295*/, v52 /*v308*/
	s_set_vgpr_msb 0x5cc
	v_cndmask_b32_e32 v95 /*v863*/, 0xff800000, v95 /*v863*/, vcc_lo
	s_set_vgpr_msb 0xcc05
	v_cmp_le_i32_e32 vcc_lo, v40 /*v296*/, v52 /*v308*/
	s_set_vgpr_msb 0x5cc
	v_cndmask_b32_e32 v210 /*v978*/, 0xff800000, v80 /*v848*/, vcc_lo
	s_set_vgpr_msb 0xcc05
	v_cmp_le_i32_e32 vcc_lo, v41 /*v297*/, v52 /*v308*/
	s_set_vgpr_msb 0x5c0
	v_add_nc_u32_e32 v80 /*v848*/, 0xd5, v130
	s_set_vgpr_msb 0xc0cc
	v_cndmask_b32_e32 v211 /*v979*/, 0xff800000, v81 /*v849*/, vcc_lo
	s_set_vgpr_msb 0xcc05
	v_cmp_le_i32_e32 vcc_lo, v50 /*v306*/, v52 /*v308*/
	s_set_vgpr_msb 0x5c0
	v_add_nc_u32_e32 v81 /*v849*/, 0xd6, v130
	s_set_vgpr_msb 0xc0cc
	v_cndmask_b32_e32 v212 /*v980*/, 0xff800000, v82 /*v850*/, vcc_lo
	s_set_vgpr_msb 0xcc05
	v_cmp_le_i32_e32 vcc_lo, v51 /*v307*/, v52 /*v308*/
	s_set_vgpr_msb 0x5c0
	v_add_nc_u32_e32 v82 /*v850*/, 0xd7, v130
	s_set_vgpr_msb 0xc0cc
	v_cndmask_b32_e32 v213 /*v981*/, 0xff800000, v83 /*v851*/, vcc_lo
	s_set_vgpr_msb 0xcc05
	v_cmp_le_i32_e32 vcc_lo, v54 /*v310*/, v52 /*v308*/
	s_set_vgpr_msb 0x5c0
	v_or_b32_e32 v83 /*v851*/, 0xe0, v130
	s_set_vgpr_msb 0xc0cc
	v_cndmask_b32_e32 v214 /*v982*/, 0xff800000, v84 /*v852*/, vcc_lo
	s_set_vgpr_msb 0xcc07
	v_cmp_le_i32_e32 vcc_lo, v80 /*v848*/, v52 /*v308*/
	s_set_vgpr_msb 0x7c0
	v_or_b32_e32 v84 /*v852*/, 0xe1, v130
	s_set_vgpr_msb 0xc0cc
	v_cndmask_b32_e32 v215 /*v983*/, 0xff800000, v85 /*v853*/, vcc_lo
	s_set_vgpr_msb 0xcc07
	v_cmp_le_i32_e32 vcc_lo, v81 /*v849*/, v52 /*v308*/
	s_set_vgpr_msb 0x7c0
	v_or_b32_e32 v85 /*v853*/, 0xe2, v130
	s_set_vgpr_msb 0xc0cc
	v_cndmask_b32_e32 v216 /*v984*/, 0xff800000, v86 /*v854*/, vcc_lo
	s_set_vgpr_msb 0xcc07
	v_cmp_le_i32_e32 vcc_lo, v82 /*v850*/, v52 /*v308*/
	s_set_vgpr_msb 0x7cc
	v_cndmask_b32_e32 v217 /*v985*/, 0xff800000, v87 /*v855*/, vcc_lo
	s_set_vgpr_msb 0xcc07
	v_cmp_le_i32_e32 vcc_lo, v83 /*v851*/, v52 /*v308*/
	s_set_vgpr_msb 0x7cc
	v_cndmask_b32_e32 v218 /*v986*/, 0xff800000, v72 /*v840*/, vcc_lo
	s_set_vgpr_msb 0xcc07
	v_cmp_le_i32_e32 vcc_lo, v84 /*v852*/, v52 /*v308*/
	s_set_vgpr_msb 0x7c0
	v_or_b32_e32 v72 /*v840*/, 0xe3, v130
	s_set_vgpr_msb 0xc0cc
	v_cndmask_b32_e32 v219 /*v987*/, 0xff800000, v73 /*v841*/, vcc_lo
	s_set_vgpr_msb 0xcc07
	v_cmp_le_i32_e32 vcc_lo, v85 /*v853*/, v52 /*v308*/
	s_set_vgpr_msb 0x7c0
	v_or_b32_e32 v73 /*v841*/, 0xe4, v130
	s_set_vgpr_msb 0xc0cc
	v_cndmask_b32_e32 v220 /*v988*/, 0xff800000, v74 /*v842*/, vcc_lo
	s_set_vgpr_msb 0xcc07
	v_cmp_le_i32_e32 vcc_lo, v72 /*v840*/, v52 /*v308*/
	s_set_vgpr_msb 0x7c0
	v_or_b32_e32 v74 /*v842*/, 0xe5, v130
	s_set_vgpr_msb 0xc0cc
	v_cndmask_b32_e32 v221 /*v989*/, 0xff800000, v75 /*v843*/, vcc_lo
	s_set_vgpr_msb 0xcc07
	v_cmp_le_i32_e32 vcc_lo, v73 /*v841*/, v52 /*v308*/
	s_set_vgpr_msb 0x7c0
	v_or_b32_e32 v75 /*v843*/, 0xe6, v130
	s_set_vgpr_msb 0xc0cc
	v_cndmask_b32_e32 v222 /*v990*/, 0xff800000, v76 /*v844*/, vcc_lo
	s_set_vgpr_msb 0xcc07
	v_cmp_le_i32_e32 vcc_lo, v74 /*v842*/, v52 /*v308*/
	s_set_vgpr_msb 0x7c0
	v_or_b32_e32 v76 /*v844*/, 0xe7, v130
	s_set_vgpr_msb 0xc0cc
	v_cndmask_b32_e32 v223 /*v991*/, 0xff800000, v77 /*v845*/, vcc_lo
	s_set_vgpr_msb 0xcc07
	v_cmp_le_i32_e32 vcc_lo, v75 /*v843*/, v52 /*v308*/
	s_set_vgpr_msb 0x7c0
	v_or_b32_e32 v77 /*v845*/, 0xf0, v130
	s_set_vgpr_msb 0xc0cc
	v_cndmask_b32_e32 v224 /*v992*/, 0xff800000, v78 /*v846*/, vcc_lo
	s_set_vgpr_msb 0xcc07
	v_cmp_le_i32_e32 vcc_lo, v76 /*v844*/, v52 /*v308*/
	s_set_vgpr_msb 0x7c0
	v_add_nc_u32_e32 v78 /*v846*/, 0xf1, v130
	s_set_vgpr_msb 0xc0cc
	v_cndmask_b32_e32 v225 /*v993*/, 0xff800000, v79 /*v847*/, vcc_lo
	s_set_vgpr_msb 0xcc07
	v_cmp_le_i32_e32 vcc_lo, v77 /*v845*/, v52 /*v308*/
	s_set_vgpr_msb 0x7c0
	v_add_nc_u32_e32 v79 /*v847*/, 0xf2, v130
	s_set_vgpr_msb 0xc0cc
	v_cndmask_b32_e32 v226 /*v994*/, 0xff800000, v64 /*v832*/, vcc_lo
	s_set_vgpr_msb 0xcc07
	v_cmp_le_i32_e32 vcc_lo, v78 /*v846*/, v52 /*v308*/
	s_set_vgpr_msb 0x7c0
	v_add_nc_u32_e32 v64 /*v832*/, 0xf3, v130
	s_set_vgpr_msb 0xc0cc
	v_cndmask_b32_e32 v227 /*v995*/, 0xff800000, v65 /*v833*/, vcc_lo
	s_set_vgpr_msb 0xcc07
	v_cmp_le_i32_e32 vcc_lo, v79 /*v847*/, v52 /*v308*/
	s_set_vgpr_msb 0x7c0
	v_add_nc_u32_e32 v65 /*v833*/, 0xf4, v130
	s_set_vgpr_msb 0xc0cc
	v_cndmask_b32_e32 v228 /*v996*/, 0xff800000, v66 /*v834*/, vcc_lo
	s_set_vgpr_msb 0xcc07
	v_cmp_le_i32_e32 vcc_lo, v64 /*v832*/, v52 /*v308*/
	s_set_vgpr_msb 0x7c0
	v_add_nc_u32_e32 v66 /*v834*/, 0xf5, v130
	s_set_vgpr_msb 0xc0cc
	v_cndmask_b32_e32 v229 /*v997*/, 0xff800000, v67 /*v835*/, vcc_lo
	s_set_vgpr_msb 0xcc07
	v_cmp_le_i32_e32 vcc_lo, v65 /*v833*/, v52 /*v308*/
	s_set_vgpr_msb 0x7c0
	v_add_nc_u32_e32 v67 /*v835*/, 0xf6, v130
	s_set_vgpr_msb 0xc0cc
	v_cndmask_b32_e32 v230 /*v998*/, 0xff800000, v68 /*v836*/, vcc_lo
	s_set_vgpr_msb 0xcc07
	v_cmp_le_i32_e32 vcc_lo, v66 /*v834*/, v52 /*v308*/
	s_set_vgpr_msb 0x7c0
	v_add_nc_u32_e32 v68 /*v836*/, 0xf7, v130
	s_set_vgpr_msb 0xc0cc
	v_cndmask_b32_e32 v231 /*v999*/, 0xff800000, v69 /*v837*/, vcc_lo
	s_set_vgpr_msb 0xcc07
	v_cmp_le_i32_e32 vcc_lo, v67 /*v835*/, v52 /*v308*/
	s_set_vgpr_msb 0x7cc
	v_cndmask_b32_e32 v242 /*v1010*/, 0xff800000, v70 /*v838*/, vcc_lo
	s_set_vgpr_msb 0xcc07
	v_cmp_le_i32_e32 vcc_lo, v68 /*v836*/, v52 /*v308*/
	s_set_vgpr_msb 0x7cc
	v_cndmask_b32_e32 v243 /*v1011*/, 0xff800000, v71 /*v839*/, vcc_lo
	s_set_vgpr_msb 0xcc04
	v_cmp_le_i32_e32 vcc_lo, v130, v53 /*v309*/
	s_set_vgpr_msb 0x4c8
	v_cndmask_b32_e32 v232 /*v1000*/, 0xff800000, v192 /*v704*/, vcc_lo
	s_set_vgpr_msb 0xc804
	v_cmp_lt_i32_e32 vcc_lo, v130, v53 /*v309*/
	s_set_vgpr_msb 0x40f
	v_max_num_f32_e32 v130, v178 /*v946*/, v179 /*v947*/
	s_set_vgpr_msb 0xfbf
	v_max3_num_f32 v192 /*v704*/, v184 /*v952*/, v185 /*v953*/, v186 /*v954*/
	s_set_vgpr_msb 0xbfc8
	v_cndmask_b32_e32 v233 /*v1001*/, 0xff800000, v193 /*v705*/, vcc_lo
	s_set_vgpr_msb 0xc804
	v_cmp_le_i32_e32 vcc_lo, v200, v53 /*v309*/
	s_set_vgpr_msb 0x4c8
	v_cndmask_b32_e32 v252 /*v1020*/, 0xff800000, v194 /*v706*/, vcc_lo
	s_set_vgpr_msb 0xc804
	v_cmp_le_i32_e32 vcc_lo, v201, v53 /*v309*/
	s_set_vgpr_msb 0x4bf
	v_max3_num_f32 v194 /*v706*/, v187 /*v955*/, v188 /*v956*/, v189 /*v957*/
	s_set_vgpr_msb 0xbfc8
	v_cndmask_b32_e32 v253 /*v1021*/, 0xff800000, v195 /*v707*/, vcc_lo
	s_set_vgpr_msb 0xc804
	v_cmp_le_i32_e32 vcc_lo, v202, v53 /*v309*/
	s_set_vgpr_msb 0x4c8
	v_cndmask_b32_e32 v236 /*v1004*/, 0xff800000, v196 /*v708*/, vcc_lo
	s_set_vgpr_msb 0xc804
	v_cmp_le_i32_e32 vcc_lo, v203, v53 /*v309*/
	s_set_vgpr_msb 0x4bf
	v_max3_num_f32 v196 /*v708*/, v190 /*v958*/, v191 /*v959*/, v192 /*v960*/
	s_set_vgpr_msb 0xbfc8
	v_cndmask_b32_e32 v237 /*v1005*/, 0xff800000, v197 /*v709*/, vcc_lo
	s_set_vgpr_msb 0xc804
	v_cmp_le_i32_e32 vcc_lo, v204, v53 /*v309*/
	s_set_vgpr_msb 0x4c8
	v_cndmask_b32_e32 v234 /*v1002*/, 0xff800000, v198 /*v710*/, vcc_lo
	s_set_vgpr_msb 0xc804
	v_cmp_le_i32_e32 vcc_lo, v205, v53 /*v309*/
	s_set_vgpr_msb 0x4bf
	v_max3_num_f32 v198 /*v710*/, v193 /*v961*/, v168 /*v936*/, v169 /*v937*/
	s_set_vgpr_msb 0xbfc8
	v_cndmask_b32_e32 v235 /*v1003*/, 0xff800000, v199 /*v711*/, vcc_lo
	s_set_vgpr_msb 0xc804
	v_cmp_le_i32_e32 vcc_lo, v206, v53 /*v309*/
	s_set_vgpr_msb 0x4c8
	v_cndmask_b32_e32 v244 /*v1012*/, 0xff800000, v200 /*v712*/, vcc_lo
	s_set_vgpr_msb 0xc804
	v_cmp_le_i32_e32 vcc_lo, v207, v53 /*v309*/
	s_set_vgpr_msb 0x4bf
	v_max3_num_f32 v200 /*v712*/, v170 /*v938*/, v171 /*v939*/, v172 /*v940*/
	s_set_vgpr_msb 0xbfc8
	v_cndmask_b32_e32 v245 /*v1013*/, 0xff800000, v201 /*v713*/, vcc_lo
	s_set_vgpr_msb 0xc804
	v_cmp_le_i32_e32 vcc_lo, v208, v53 /*v309*/
	s_set_vgpr_msb 0x4bf
	v_max3_num_f32 v193 /*v705*/, v234 /*v1002*/, v235 /*v1003*/, v244 /*v1012*/
	s_set_vgpr_msb 0xbfc8
	v_cndmask_b32_e32 v238 /*v1006*/, 0xff800000, v202 /*v714*/, vcc_lo
	s_set_vgpr_msb 0xc804
	v_cmp_le_i32_e32 vcc_lo, v192, v53 /*v309*/
	s_set_vgpr_msb 0x4bf
	v_max3_num_f32 v202 /*v714*/, v173 /*v941*/, v174 /*v942*/, v175 /*v943*/
	s_set_vgpr_msb 0xbfc8
	v_cndmask_b32_e32 v239 /*v1007*/, 0xff800000, v203 /*v715*/, vcc_lo
	s_set_vgpr_msb 0xc804
	v_cmp_le_i32_e32 vcc_lo, v193, v53 /*v309*/
	s_set_vgpr_msb 0x4bf
	v_max3_num_f32 v195 /*v707*/, v245 /*v1013*/, v238 /*v1006*/, v239 /*v1007*/
	s_set_vgpr_msb 0xbf08
	v_cndmask_b32_e32 v198, 0xff800000, v204 /*v716*/, vcc_lo
	s_set_vgpr_msb 0x804
	v_cmp_le_i32_e32 vcc_lo, v194, v53 /*v309*/
	s_set_vgpr_msb 0x4bf
	v_max3_num_f32 v204 /*v716*/, v160 /*v928*/, v161 /*v929*/, v162 /*v930*/
	s_set_vgpr_msb 0xbf08
	v_cndmask_b32_e32 v199, 0xff800000, v205 /*v717*/, vcc_lo
	s_set_vgpr_msb 0x804
	v_cmp_le_i32_e32 vcc_lo, v195, v53 /*v309*/
	s_set_vgpr_msb 0x4c8
	v_cndmask_b32_e32 v246 /*v1014*/, 0xff800000, v206 /*v718*/, vcc_lo
	s_set_vgpr_msb 0xc804
	v_cmp_le_i32_e32 vcc_lo, v196, v53 /*v309*/
	s_set_vgpr_msb 0x4bf
	v_max3_num_f32 v206 /*v718*/, v163 /*v931*/, v164 /*v932*/, v165 /*v933*/
	s_set_vgpr_msb 0xbfc8
	v_cndmask_b32_e32 v247 /*v1015*/, 0xff800000, v207 /*v719*/, vcc_lo
	s_set_vgpr_msb 0xc804
	v_cmp_le_i32_e32 vcc_lo, v197, v53 /*v309*/
	s_set_vgpr_msb 0x4b0
	v_max3_num_f32 v197 /*v709*/, v198, v199, v246 /*v1014*/
	s_set_vgpr_msb 0xb0c8
	v_cndmask_b32_e32 v240 /*v1008*/, 0xff800000, v208 /*v720*/, vcc_lo
	s_set_vgpr_msb 0xc804
	v_cmp_le_i32_e32 vcc_lo, v209, v53 /*v309*/
	s_set_vgpr_msb 0x4bf
	v_max3_num_f32 v208 /*v720*/, v166 /*v934*/, v167 /*v935*/, v152 /*v920*/
	s_set_vgpr_msb 0xbfc8
	v_cndmask_b32_e32 v241 /*v1009*/, 0xff800000, v209 /*v721*/, vcc_lo
	s_set_vgpr_msb 0xc804
	v_cmp_le_i32_e32 vcc_lo, v210, v53 /*v309*/
	s_set_vgpr_msb 0x4bf
	v_max3_num_f32 v199 /*v711*/, v247 /*v1015*/, v240 /*v1008*/, v241 /*v1009*/
	s_set_vgpr_msb 0xbfc8
	v_cndmask_b32_e32 v254 /*v1022*/, 0xff800000, v210 /*v722*/, vcc_lo
	s_set_vgpr_msb 0xc804
	v_cmp_le_i32_e32 vcc_lo, v211, v53 /*v309*/
	s_set_vgpr_msb 0x4bf
	v_max3_num_f32 v210 /*v722*/, v153 /*v921*/, v154 /*v922*/, v155 /*v923*/
	s_set_vgpr_msb 0xbfc8
	v_cndmask_b32_e32 v255 /*v1023*/, 0xff800000, v211 /*v723*/, vcc_lo
	s_set_vgpr_msb 0xc804
	v_cmp_le_i32_e32 vcc_lo, v212, v53 /*v309*/
	s_set_vgpr_msb 0x4c8
	v_cndmask_b32_e32 v248 /*v1016*/, 0xff800000, v212 /*v724*/, vcc_lo
	s_set_vgpr_msb 0xc804
	v_cmp_le_i32_e32 vcc_lo, v213, v53 /*v309*/
	s_set_vgpr_msb 0x4bf
	v_max3_num_f32 v212 /*v724*/, v156 /*v924*/, v157 /*v925*/, v158 /*v926*/
	s_set_vgpr_msb 0xbfc8
	v_cndmask_b32_e32 v249 /*v1017*/, 0xff800000, v213 /*v725*/, vcc_lo
	s_set_vgpr_msb 0xc804
	v_cmp_le_i32_e32 vcc_lo, v214, v53 /*v309*/
	s_set_vgpr_msb 0x4bf
	v_max3_num_f32 v201 /*v713*/, v254 /*v1022*/, v255 /*v1023*/, v248 /*v1016*/
	s_set_vgpr_msb 0xbf08
	v_cndmask_b32_e32 v208, 0xff800000, v214 /*v726*/, vcc_lo
	s_set_vgpr_msb 0x804
	v_cmp_le_i32_e32 vcc_lo, v215, v53 /*v309*/
	s_set_vgpr_msb 0x4bf
	v_max3_num_f32 v214 /*v726*/, v159 /*v927*/, v144 /*v912*/, v145 /*v913*/
	s_set_vgpr_msb 0xbf08
	v_cndmask_b32_e32 v209, 0xff800000, v215 /*v727*/, vcc_lo
	s_set_vgpr_msb 0x804
	v_cmp_le_i32_e32 vcc_lo, v216, v53 /*v309*/
	s_set_vgpr_msb 0x483
	v_max3_num_f32 v203 /*v715*/, v249 /*v1017*/, v208, v209
	s_set_vgpr_msb 0x8308
	v_cndmask_b32_e32 v192, 0xff800000, v216 /*v728*/, vcc_lo
	s_set_vgpr_msb 0x804
	v_cmp_le_i32_e32 vcc_lo, v217, v53 /*v309*/
	s_set_vgpr_msb 0x4bf
	v_max3_num_f32 v216 /*v728*/, v146 /*v914*/, v147 /*v915*/, v148 /*v916*/
	s_set_vgpr_msb 0xbf08
	v_cndmask_b32_e32 v193, 0xff800000, v217 /*v729*/, vcc_lo
	s_set_vgpr_msb 0x804
	v_cmp_le_i32_e32 vcc_lo, v218, v53 /*v309*/
	s_set_vgpr_msb 0x4c8
	v_cndmask_b32_e32 v250 /*v1018*/, 0xff800000, v218 /*v730*/, vcc_lo
	s_set_vgpr_msb 0xc804
	v_cmp_le_i32_e32 vcc_lo, v219, v53 /*v309*/
	s_set_vgpr_msb 0x4bf
	v_max3_num_f32 v218 /*v730*/, v149 /*v917*/, v150 /*v918*/, v151 /*v919*/
	s_set_vgpr_msb 0xbfc8
	v_cndmask_b32_e32 v251 /*v1019*/, 0xff800000, v219 /*v731*/, vcc_lo
	s_set_vgpr_msb 0xc804
	v_cmp_le_i32_e32 vcc_lo, v220, v53 /*v309*/
	s_set_vgpr_msb 0x4b0
	v_max3_num_f32 v205 /*v717*/, v192, v193, v250 /*v1018*/
	s_set_vgpr_msb 0xb008
	v_cndmask_b32_e32 v200, 0xff800000, v220 /*v732*/, vcc_lo
	s_set_vgpr_msb 0x804
	v_cmp_le_i32_e32 vcc_lo, v221, v53 /*v309*/
	s_set_vgpr_msb 0x4bf
	v_max3_num_f32 v220 /*v732*/, v136 /*v904*/, v137 /*v905*/, v138 /*v906*/
	s_set_vgpr_msb 0xbf08
	v_cndmask_b32_e32 v201, 0xff800000, v221 /*v733*/, vcc_lo
	s_set_vgpr_msb 0x804
	v_cmp_le_i32_e32 vcc_lo, v222, v53 /*v309*/
	s_set_vgpr_msb 0x483
	v_max3_num_f32 v207 /*v719*/, v251 /*v1019*/, v200, v201
	s_set_vgpr_msb 0x8308
	v_cndmask_b32_e32 v194, 0xff800000, v222 /*v734*/, vcc_lo
	s_set_vgpr_msb 0x804
	v_cmp_le_i32_e32 vcc_lo, v223, v53 /*v309*/
	s_set_vgpr_msb 0x4bf
	v_max3_num_f32 v222 /*v734*/, v139 /*v907*/, v140 /*v908*/, v141 /*v909*/
	s_set_vgpr_msb 0xbf08
	v_cndmask_b32_e32 v195, 0xff800000, v223 /*v735*/, vcc_lo
	s_set_vgpr_msb 0x804
	v_cmp_le_i32_e32 vcc_lo, v224, v53 /*v309*/
	s_set_vgpr_msb 0x408
	v_cndmask_b32_e32 v218, 0xff800000, v224 /*v736*/, vcc_lo
	s_set_vgpr_msb 0x804
	v_cmp_le_i32_e32 vcc_lo, v225, v53 /*v309*/
	s_set_vgpr_msb 0x4bf
	v_max3_num_f32 v224 /*v736*/, v142 /*v910*/, v143 /*v911*/, v128 /*v896*/
	s_set_vgpr_msb 0xbf08
	v_cndmask_b32_e32 v219, 0xff800000, v225 /*v737*/, vcc_lo
	s_set_vgpr_msb 0x804
	v_cmp_le_i32_e32 vcc_lo, v226, v53 /*v309*/
	s_set_vgpr_msb 0x480
	v_max3_num_f32 v209 /*v721*/, v194, v195, v218
	s_set_vgpr_msb 0x8008
	v_cndmask_b32_e32 v202, 0xff800000, v226 /*v738*/, vcc_lo
	s_set_vgpr_msb 0x804
	v_cmp_le_i32_e32 vcc_lo, v227, v53 /*v309*/
	s_set_vgpr_msb 0x4bf
	v_max3_num_f32 v226 /*v738*/, v129 /*v897*/, v130 /*v898*/, v131 /*v899*/
	s_set_vgpr_msb 0xbf08
	v_cndmask_b32_e32 v203, 0xff800000, v227 /*v739*/, vcc_lo
	s_set_vgpr_msb 0x804
	v_cmp_le_i32_e32 vcc_lo, v228, v53 /*v309*/
	s_set_vgpr_msb 0x480
	v_max3_num_f32 v211 /*v723*/, v219, v202, v203
	s_set_vgpr_msb 0x8008
	v_cndmask_b32_e32 v196, 0xff800000, v228 /*v740*/, vcc_lo
	s_set_vgpr_msb 0x804
	v_cmp_le_i32_e32 vcc_lo, v229, v53 /*v309*/
	s_set_vgpr_msb 0x4bf
	v_max3_num_f32 v228 /*v740*/, v132 /*v900*/, v133 /*v901*/, v134 /*v902*/
	s_set_vgpr_msb 0xbf08
	v_cndmask_b32_e32 v197, 0xff800000, v229 /*v741*/, vcc_lo
	s_set_vgpr_msb 0x804
	v_cmp_le_i32_e32 vcc_lo, v230, v53 /*v309*/
	s_set_vgpr_msb 0x408
	v_cndmask_b32_e32 v210, 0xff800000, v230 /*v742*/, vcc_lo
	s_set_vgpr_msb 0x804
	v_cmp_le_i32_e32 vcc_lo, v231, v53 /*v309*/
	s_set_vgpr_msb 0x4bf
	v_max3_num_f32 v230 /*v742*/, v135 /*v903*/, v120 /*v888*/, v121 /*v889*/
	s_set_vgpr_msb 0xbf08
	v_cndmask_b32_e32 v211, 0xff800000, v231 /*v743*/, vcc_lo
	s_set_vgpr_msb 0x804
	v_cmp_le_i32_e32 vcc_lo, v232, v53 /*v309*/
	s_set_vgpr_msb 0x480
	v_max3_num_f32 v213 /*v725*/, v196, v197, v210
	s_set_vgpr_msb 0x8008
	v_cndmask_b32_e32 v204, 0xff800000, v232 /*v744*/, vcc_lo
	s_set_vgpr_msb 0x804
	v_cmp_le_i32_e32 vcc_lo, v233, v53 /*v309*/
	s_set_vgpr_msb 0x4bf
	v_max3_num_f32 v232 /*v744*/, v122 /*v890*/, v123 /*v891*/, v124 /*v892*/
	s_set_vgpr_msb 0xbf08
	v_cndmask_b32_e32 v205, 0xff800000, v233 /*v745*/, vcc_lo
	s_set_vgpr_msb 0x804
	v_cmp_le_i32_e32 vcc_lo, v234, v53 /*v309*/
	s_set_vgpr_msb 0x480
	v_max3_num_f32 v215 /*v727*/, v211, v204, v205
	s_set_vgpr_msb 0x8008
	v_cndmask_b32_e32 v228, 0xff800000, v234 /*v746*/, vcc_lo
	s_set_vgpr_msb 0x804
	v_cmp_le_i32_e32 vcc_lo, v235, v53 /*v309*/
	s_set_vgpr_msb 0x4bf
	v_max3_num_f32 v234 /*v746*/, v125 /*v893*/, v176 /*v944*/, v177 /*v945*/
	s_set_vgpr_msb 0xbf08
	v_cndmask_b32_e32 v229, 0xff800000, v235 /*v747*/, vcc_lo
	s_set_vgpr_msb 0x804
	v_cmp_le_i32_e32 vcc_lo, v236, v53 /*v309*/
	s_set_vgpr_msb 0x408
	v_cndmask_b32_e32 v212, 0xff800000, v236 /*v748*/, vcc_lo
	s_set_vgpr_msb 0x804
	v_cmp_le_i32_e32 vcc_lo, v237, v53 /*v309*/
	s_set_vgpr_msb 0x4bf
	v_max3_num_f32 v236 /*v748*/, v194 /*v962*/, v195 /*v963*/, v114 /*v882*/
	s_set_vgpr_msb 0xbf08
	v_cndmask_b32_e32 v213, 0xff800000, v237 /*v749*/, vcc_lo
	s_set_vgpr_msb 0x804
	v_cmp_le_i32_e32 vcc_lo, v238, v53 /*v309*/
	s_set_vgpr_msb 0x480
	v_max3_num_f32 v217 /*v729*/, v228, v229, v212
	s_set_vgpr_msb 0x8008
	v_cndmask_b32_e32 v206, 0xff800000, v238 /*v750*/, vcc_lo
	s_set_vgpr_msb 0x804
	v_cmp_le_i32_e32 vcc_lo, v239, v53 /*v309*/
	s_set_vgpr_msb 0x4bf
	v_max3_num_f32 v238 /*v750*/, v115 /*v883*/, v116 /*v884*/, v117 /*v885*/
	s_set_vgpr_msb 0xbf08
	v_cndmask_b32_e32 v207, 0xff800000, v239 /*v751*/, vcc_lo
	s_set_vgpr_msb 0x804
	v_cmp_le_i32_e32 vcc_lo, v240, v53 /*v309*/
	s_set_vgpr_msb 0x480
	v_max3_num_f32 v219 /*v731*/, v213, v206, v207
	s_set_vgpr_msb 0x8008
	v_cndmask_b32_e32 v220, 0xff800000, v240 /*v752*/, vcc_lo
	s_set_vgpr_msb 0x804
	v_cmp_le_i32_e32 vcc_lo, v241, v53 /*v309*/
	s_set_vgpr_msb 0x4bf
	v_max3_num_f32 v240 /*v752*/, v118 /*v886*/, v119 /*v887*/, v104 /*v872*/
	s_set_vgpr_msb 0xbf08
	v_cndmask_b32_e32 v221, 0xff800000, v241 /*v753*/, vcc_lo
	s_set_vgpr_msb 0x804
	v_cmp_le_i32_e32 vcc_lo, v242, v53 /*v309*/
	s_set_vgpr_msb 0x408
	v_cndmask_b32_e32 v214, 0xff800000, v242 /*v754*/, vcc_lo
	s_set_vgpr_msb 0x804
	v_cmp_le_i32_e32 vcc_lo, v243, v53 /*v309*/
	s_set_vgpr_msb 0x48f
	v_max_num_f32_e32 v242 /*v754*/, v105 /*v873*/, v196 /*v964*/
	s_set_vgpr_msb 0x8f08
	v_cndmask_b32_e32 v215, 0xff800000, v243 /*v755*/, vcc_lo
	s_set_vgpr_msb 0x804
	v_cmp_le_i32_e32 vcc_lo, v244, v53 /*v309*/
	s_set_vgpr_msb 0x480
	v_max3_num_f32 v221 /*v733*/, v220, v221, v214
	s_set_vgpr_msb 0x8008
	v_cndmask_b32_e32 v238, 0xff800000, v244 /*v756*/, vcc_lo
	s_set_vgpr_msb 0x804
	v_cmp_le_i32_e32 vcc_lo, v245, v53 /*v309*/
	s_set_vgpr_msb 0x4bf
	v_max3_num_f32 v244 /*v756*/, v108 /*v876*/, v109 /*v877*/, v110 /*v878*/
	s_set_vgpr_msb 0xbf08
	v_cndmask_b32_e32 v239, 0xff800000, v245 /*v757*/, vcc_lo
	s_set_vgpr_msb 0x804
	v_cmp_le_i32_e32 vcc_lo, v246, v53 /*v309*/
	s_set_vgpr_msb 0x480
	v_max3_num_f32 v223 /*v735*/, v215, v238, v239
	s_set_vgpr_msb 0x8008
	v_cndmask_b32_e32 v222, 0xff800000, v246 /*v758*/, vcc_lo
	s_set_vgpr_msb 0x804
	v_cmp_le_i32_e32 vcc_lo, v247, v53 /*v309*/
	s_set_vgpr_msb 0x4bf
	v_max3_num_f32 v246 /*v758*/, v111 /*v879*/, v198 /*v966*/, v199 /*v967*/
	s_set_vgpr_msb 0xbf08
	v_cndmask_b32_e32 v223, 0xff800000, v247 /*v759*/, vcc_lo
	s_set_vgpr_msb 0x804
	v_cmp_le_i32_e32 vcc_lo, v248, v53 /*v309*/
	s_set_vgpr_msb 0x408
	v_cndmask_b32_e32 v216, 0xff800000, v248 /*v760*/, vcc_lo
	s_set_vgpr_msb 0x804
	v_cmp_le_i32_e32 vcc_lo, v249, v53 /*v309*/
	s_set_vgpr_msb 0x4bf
	v_max3_num_f32 v248 /*v760*/, v200 /*v968*/, v201 /*v969*/, v100 /*v868*/
	s_set_vgpr_msb 0xbf08
	v_cndmask_b32_e32 v217, 0xff800000, v249 /*v761*/, vcc_lo
	s_set_vgpr_msb 0x804
	v_cmp_le_i32_e32 vcc_lo, v250, v53 /*v309*/
	s_set_vgpr_msb 0x480
	v_max3_num_f32 v225 /*v737*/, v222, v223, v216
	s_set_vgpr_msb 0x8008
	v_cndmask_b32_e32 v230, 0xff800000, v250 /*v762*/, vcc_lo
	s_set_vgpr_msb 0x804
	v_cmp_le_i32_e32 vcc_lo, v251, v53 /*v309*/
	s_set_vgpr_msb 0x4bf
	v_max3_num_f32 v250 /*v762*/, v101 /*v869*/, v202 /*v970*/, v203 /*v971*/
	s_set_vgpr_msb 0xbf08
	v_cndmask_b32_e32 v231, 0xff800000, v251 /*v763*/, vcc_lo
	s_set_vgpr_msb 0x804
	v_cmp_le_i32_e32 vcc_lo, v252, v53 /*v309*/
	s_set_vgpr_msb 0x480
	v_max3_num_f32 v227 /*v739*/, v217, v230, v231
	s_set_vgpr_msb 0x8008
	v_cndmask_b32_e32 v224, 0xff800000, v252 /*v764*/, vcc_lo
	s_set_vgpr_msb 0x804
	v_cmp_le_i32_e32 vcc_lo, v253, v53 /*v309*/
	s_set_vgpr_msb 0x4bf
	v_max3_num_f32 v252 /*v764*/, v204 /*v972*/, v205 /*v973*/, v206 /*v974*/
	s_set_vgpr_msb 0xbf08
	v_cndmask_b32_e32 v225, 0xff800000, v253 /*v765*/, vcc_lo
	s_set_vgpr_msb 0x804
	v_cmp_le_i32_e32 vcc_lo, v254, v53 /*v309*/
	s_set_vgpr_msb 0x408
	v_cndmask_b32_e32 v248, 0xff800000, v254 /*v766*/, vcc_lo
	s_set_vgpr_msb 0x804
	v_cmp_le_i32_e32 vcc_lo, v255, v53 /*v309*/
	s_set_vgpr_msb 0x4bf
	v_max3_num_f32 v254 /*v766*/, v207 /*v975*/, v208 /*v976*/, v209 /*v977*/
	s_set_vgpr_msb 0xbf08
	v_cndmask_b32_e32 v249, 0xff800000, v255 /*v767*/, vcc_lo
	s_set_vgpr_msb 0x805
	v_cmp_le_i32_e32 vcc_lo, v0 /*v256*/, v53 /*v309*/
	s_set_vgpr_msb 0x580
	v_max3_num_f32 v229 /*v741*/, v224, v225, v248
	s_set_vgpr_msb 0x800c
	v_cndmask_b32_e32 v232, 0xff800000, v0 /*v768*/, vcc_lo
	s_set_vgpr_msb 0xc05
	v_cmp_le_i32_e32 vcc_lo, v1 /*v257*/, v53 /*v309*/
	s_set_vgpr_msb 0x5ff
	v_max3_num_f32 v0 /*v768*/, v94 /*v862*/, v95 /*v863*/, v210 /*v978*/
	s_set_vgpr_msb 0xff0c
	v_cndmask_b32_e32 v233, 0xff800000, v1 /*v769*/, vcc_lo
	s_set_vgpr_msb 0xc05
	v_cmp_le_i32_e32 vcc_lo, v2 /*v258*/, v53 /*v309*/
	s_set_vgpr_msb 0x580
	v_max3_num_f32 v231 /*v743*/, v249, v232, v233
	s_set_vgpr_msb 0x800c
	v_cndmask_b32_e32 v226, 0xff800000, v2 /*v770*/, vcc_lo
	s_set_vgpr_msb 0xc05
	v_cmp_le_i32_e32 vcc_lo, v3 /*v259*/, v53 /*v309*/
	s_set_vgpr_msb 0x5ff
	v_max3_num_f32 v2 /*v770*/, v211 /*v979*/, v212 /*v980*/, v213 /*v981*/
	s_set_vgpr_msb 0xff0c
	v_cndmask_b32_e32 v227, 0xff800000, v3 /*v771*/, vcc_lo
	s_set_vgpr_msb 0xc05
	v_cmp_le_i32_e32 vcc_lo, v4 /*v260*/, v53 /*v309*/
	s_set_vgpr_msb 0x50c
	v_cndmask_b32_e32 v240, 0xff800000, v4 /*v772*/, vcc_lo
	s_set_vgpr_msb 0xc05
	v_cmp_le_i32_e32 vcc_lo, v5 /*v261*/, v53 /*v309*/
	s_set_vgpr_msb 0x5cf
	v_max_num_f32_e32 v4 /*v772*/, v214 /*v982*/, v215 /*v983*/
	s_set_vgpr_msb 0xcf0c
	v_cndmask_b32_e32 v241, 0xff800000, v5 /*v773*/, vcc_lo
	s_set_vgpr_msb 0xc05
	v_cmp_le_i32_e32 vcc_lo, v6 /*v262*/, v53 /*v309*/
	s_set_vgpr_msb 0x580
	v_max3_num_f32 v233 /*v745*/, v226, v227, v240
	s_set_vgpr_msb 0x800c
	v_cndmask_b32_e32 v234, 0xff800000, v6 /*v774*/, vcc_lo
	s_set_vgpr_msb 0xc05
	v_cmp_le_i32_e32 vcc_lo, v7 /*v263*/, v53 /*v309*/
	s_set_vgpr_msb 0x5ff
	v_max3_num_f32 v6 /*v774*/, v217 /*v985*/, v218 /*v986*/, v219 /*v987*/
	s_set_vgpr_msb 0xff0c
	v_cndmask_b32_e32 v235, 0xff800000, v7 /*v775*/, vcc_lo
	s_set_vgpr_msb 0xc05
	v_cmp_le_i32_e32 vcc_lo, v8 /*v264*/, v53 /*v309*/
	s_set_vgpr_msb 0x580
	v_max3_num_f32 v235 /*v747*/, v241, v234, v235
	s_set_vgpr_msb 0x804c
	v_cndmask_b32_e32 v2 /*v258*/, 0xff800000, v8 /*v776*/, vcc_lo
	s_set_vgpr_msb 0x4c05
	v_cmp_le_i32_e32 vcc_lo, v9 /*v265*/, v53 /*v309*/
	s_set_vgpr_msb 0x5ff
	v_max3_num_f32 v8 /*v776*/, v220 /*v988*/, v221 /*v989*/, v222 /*v990*/
	s_set_vgpr_msb 0xff4c
	v_cndmask_b32_e32 v3 /*v259*/, 0xff800000, v9 /*v777*/, vcc_lo
	s_set_vgpr_msb 0x4c05
	v_cmp_le_i32_e32 vcc_lo, v10 /*v266*/, v53 /*v309*/
	s_set_vgpr_msb 0x50c
	v_cndmask_b32_e32 v242, 0xff800000, v10 /*v778*/, vcc_lo
	s_set_vgpr_msb 0xc05
	v_cmp_le_i32_e32 vcc_lo, v11 /*v267*/, v53 /*v309*/
	s_set_vgpr_msb 0x5ff
	v_max3_num_f32 v10 /*v778*/, v223 /*v991*/, v224 /*v992*/, v225 /*v993*/
	s_set_vgpr_msb 0xff0c
	v_cndmask_b32_e32 v243, 0xff800000, v11 /*v779*/, vcc_lo
	s_set_vgpr_msb 0xc85
	v_cmp_le_i32_e32 vcc_lo, v12 /*v268*/, v53 /*v309*/
	v_max3_num_f32 v237 /*v749*/, v2 /*v258*/, v3 /*v259*/, v242
	s_set_vgpr_msb 0x850c
	v_cndmask_b32_e32 v236, 0xff800000, v12 /*v780*/, vcc_lo
	s_set_vgpr_msb 0xc05
	v_cmp_le_i32_e32 vcc_lo, v13 /*v269*/, v53 /*v309*/
	s_set_vgpr_msb 0x5ff
	v_max3_num_f32 v12 /*v780*/, v226 /*v994*/, v227 /*v995*/, v228 /*v996*/
	s_set_vgpr_msb 0xff0c
	v_cndmask_b32_e32 v237, 0xff800000, v13 /*v781*/, vcc_lo
	s_set_vgpr_msb 0xc05
	v_cmp_le_i32_e32 vcc_lo, v14 /*v270*/, v53 /*v309*/
	s_set_vgpr_msb 0x580
	v_max3_num_f32 v239 /*v751*/, v243, v236, v237
	s_set_vgpr_msb 0x800c
	v_cndmask_b32_e32 v250, 0xff800000, v14 /*v782*/, vcc_lo
	s_set_vgpr_msb 0xc05
	v_cmp_le_i32_e32 vcc_lo, v15 /*v271*/, v53 /*v309*/
	s_set_vgpr_msb 0x5ff
	v_max3_num_f32 v14 /*v782*/, v229 /*v997*/, v230 /*v998*/, v231 /*v999*/
	s_set_vgpr_msb 0xff0c
	v_cndmask_b32_e32 v251, 0xff800000, v15 /*v783*/, vcc_lo
	s_set_vgpr_msb 0xc05
	v_cmp_le_i32_e32 vcc_lo, v16 /*v272*/, v53 /*v309*/
	s_set_vgpr_msb 0x50c
	v_cndmask_b32_e32 v244, 0xff800000, v16 /*v784*/, vcc_lo
	s_set_vgpr_msb 0xc05
	v_cmp_le_i32_e32 vcc_lo, v17 /*v273*/, v53 /*v309*/
	s_set_vgpr_msb 0x50c
	v_cndmask_b32_e32 v245, 0xff800000, v17 /*v785*/, vcc_lo
	s_set_vgpr_msb 0xc05
	v_cmp_le_i32_e32 vcc_lo, v18 /*v274*/, v53 /*v309*/
	s_set_vgpr_msb 0x580
	v_max3_num_f32 v241 /*v753*/, v250, v251, v244
	s_set_vgpr_msb 0x804c
	v_cndmask_b32_e32 v12 /*v268*/, 0xff800000, v18 /*v786*/, vcc_lo
	s_set_vgpr_msb 0x4c05
	v_cmp_le_i32_e32 vcc_lo, v19 /*v275*/, v53 /*v309*/
	s_set_vgpr_msb 0x54c
	v_cndmask_b32_e32 v13 /*v269*/, 0xff800000, v19 /*v787*/, vcc_lo
	s_set_vgpr_msb 0x4c05
	v_cmp_le_i32_e32 vcc_lo, v20 /*v276*/, v53 /*v309*/
	s_set_vgpr_msb 0x50c
	v_cndmask_b32_e32 v252, 0xff800000, v20 /*v788*/, vcc_lo
	s_set_vgpr_msb 0xc05
	v_cmp_le_i32_e32 vcc_lo, v21 /*v277*/, v53 /*v309*/
	s_set_vgpr_msb 0x50c
	v_cndmask_b32_e32 v253, 0xff800000, v21 /*v789*/, vcc_lo
	s_set_vgpr_msb 0xc05
	v_cmp_le_i32_e32 vcc_lo, v22 /*v278*/, v53 /*v309*/
	s_set_vgpr_msb 0x50c
	v_cndmask_b32_e32 v246, 0xff800000, v22 /*v790*/, vcc_lo
	s_set_vgpr_msb 0xc05
	v_cmp_le_i32_e32 vcc_lo, v23 /*v279*/, v53 /*v309*/
	s_set_vgpr_msb 0x50c
	v_cndmask_b32_e32 v247, 0xff800000, v23 /*v791*/, vcc_lo
	s_set_vgpr_msb 0xc05
	v_cmp_le_i32_e32 vcc_lo, v24 /*v280*/, v53 /*v309*/
	s_set_vgpr_msb 0x580
	v_max3_num_f32 v245 /*v757*/, v252, v253, v246
	s_set_vgpr_msb 0x804c
	v_cndmask_b32_e32 v4 /*v260*/, 0xff800000, v24 /*v792*/, vcc_lo
	s_set_vgpr_msb 0x4c05
	v_cmp_le_i32_e32 vcc_lo, v25 /*v281*/, v53 /*v309*/
	s_set_vgpr_msb 0x54c
	v_cndmask_b32_e32 v5 /*v261*/, 0xff800000, v25 /*v793*/, vcc_lo
	s_set_vgpr_msb 0x4c05
	v_cmp_le_i32_e32 vcc_lo, v26 /*v282*/, v53 /*v309*/
	s_set_vgpr_msb 0x594
	v_max3_num_f32 v247 /*v759*/, v247, v4 /*v260*/, v5 /*v261*/
	s_set_vgpr_msb 0x940c
	v_cndmask_b32_e32 v254, 0xff800000, v26 /*v794*/, vcc_lo
	s_set_vgpr_msb 0xc05
	v_cmp_le_i32_e32 vcc_lo, v27 /*v283*/, v53 /*v309*/
	s_set_vgpr_msb 0x50c
	v_cndmask_b32_e32 v255, 0xff800000, v27 /*v795*/, vcc_lo
	s_set_vgpr_msb 0xc05
	v_cmp_le_i32_e32 vcc_lo, v28 /*v284*/, v53 /*v309*/
	s_set_vgpr_msb 0x54c
	v_cndmask_b32_e32 v24 /*v280*/, 0xff800000, v28 /*v796*/, vcc_lo
	s_set_vgpr_msb 0x4c05
	v_cmp_le_i32_e32 vcc_lo, v29 /*v285*/, v53 /*v309*/
	s_set_vgpr_msb 0x54c
	v_cndmask_b32_e32 v25 /*v281*/, 0xff800000, v29 /*v797*/, vcc_lo
	s_set_vgpr_msb 0x4c05
	v_cmp_le_i32_e32 vcc_lo, v30 /*v286*/, v53 /*v309*/
	s_set_vgpr_msb 0x590
	v_max3_num_f32 v249 /*v761*/, v254, v255, v24 /*v280*/
	s_set_vgpr_msb 0x904c
	v_cndmask_b32_e32 v6 /*v262*/, 0xff800000, v30 /*v798*/, vcc_lo
	s_set_vgpr_msb 0x4c05
	v_cmp_le_i32_e32 vcc_lo, v31 /*v287*/, v53 /*v309*/
	s_set_vgpr_msb 0x54c
	v_cndmask_b32_e32 v7 /*v263*/, 0xff800000, v31 /*v799*/, vcc_lo
	s_set_vgpr_msb 0x4c95
	v_cmp_le_i32_e32 vcc_lo, v32 /*v288*/, v53 /*v309*/
	v_max3_num_f32 v251 /*v763*/, v25 /*v281*/, v6 /*v262*/, v7 /*v263*/
	s_set_vgpr_msb 0x954c
	v_cndmask_b32_e32 v0 /*v256*/, 0xff800000, v32 /*v800*/, vcc_lo
	s_set_vgpr_msb 0x4c05
	v_cmp_le_i32_e32 vcc_lo, v33 /*v289*/, v53 /*v309*/
	s_set_vgpr_msb 0x54c
	v_cndmask_b32_e32 v1 /*v257*/, 0xff800000, v33 /*v801*/, vcc_lo
	s_set_vgpr_msb 0x4c05
	v_cmp_le_i32_e32 vcc_lo, v34 /*v290*/, v53 /*v309*/
	s_set_vgpr_msb 0x54c
	v_cndmask_b32_e32 v14 /*v270*/, 0xff800000, v34 /*v802*/, vcc_lo
	s_set_vgpr_msb 0x4c05
	v_cmp_le_i32_e32 vcc_lo, v35 /*v291*/, v53 /*v309*/
	s_set_vgpr_msb 0x54c
	v_cndmask_b32_e32 v15 /*v271*/, 0xff800000, v35 /*v803*/, vcc_lo
	s_set_vgpr_msb 0x4c95
	v_cmp_le_i32_e32 vcc_lo, v36 /*v292*/, v53 /*v309*/
	v_max3_num_f32 v253 /*v765*/, v0 /*v256*/, v1 /*v257*/, v14 /*v270*/
	s_set_vgpr_msb 0x954c
	v_cndmask_b32_e32 v8 /*v264*/, 0xff800000, v36 /*v804*/, vcc_lo
	s_set_vgpr_msb 0x4c05
	v_cmp_le_i32_e32 vcc_lo, v37 /*v293*/, v53 /*v309*/
	s_set_vgpr_msb 0x54c
	v_cndmask_b32_e32 v9 /*v265*/, 0xff800000, v37 /*v805*/, vcc_lo
	s_set_vgpr_msb 0x4c95
	v_cmp_le_i32_e32 vcc_lo, v38 /*v294*/, v53 /*v309*/
	v_max3_num_f32 v255 /*v767*/, v15 /*v271*/, v8 /*v264*/, v9 /*v265*/
	s_set_vgpr_msb 0x954c
	v_cndmask_b32_e32 v32 /*v288*/, 0xff800000, v38 /*v806*/, vcc_lo
	s_set_vgpr_msb 0x4c05
	v_cmp_le_i32_e32 vcc_lo, v39 /*v295*/, v53 /*v309*/
	s_set_vgpr_msb 0x54c
	v_cndmask_b32_e32 v33 /*v289*/, 0xff800000, v39 /*v807*/, vcc_lo
	s_set_vgpr_msb 0x4c05
	v_cmp_le_i32_e32 vcc_lo, v40 /*v296*/, v53 /*v309*/
	s_set_vgpr_msb 0x54c
	v_cndmask_b32_e32 v16 /*v272*/, 0xff800000, v40 /*v808*/, vcc_lo
	s_set_vgpr_msb 0x4c05
	v_cmp_le_i32_e32 vcc_lo, v41 /*v297*/, v53 /*v309*/
	s_set_vgpr_msb 0x54c
	v_cndmask_b32_e32 v17 /*v273*/, 0xff800000, v41 /*v809*/, vcc_lo
	s_set_vgpr_msb 0x4c05
	v_cmp_le_i32_e32 vcc_lo, v50 /*v306*/, v53 /*v309*/
	s_set_vgpr_msb 0x54f
	v_max_num_f32_e32 v50 /*v306*/, v232 /*v1000*/, v233 /*v1001*/
	s_set_vgpr_msb 0x4fd5
	v_max3_num_f32 v1 /*v769*/, v32 /*v288*/, v33 /*v289*/, v16 /*v272*/
	s_set_vgpr_msb 0xd54c
	v_cndmask_b32_e32 v10 /*v266*/, 0xff800000, v42 /*v810*/, vcc_lo
	s_set_vgpr_msb 0x4c05
	v_cmp_le_i32_e32 vcc_lo, v51 /*v307*/, v53 /*v309*/
	s_set_vgpr_msb 0x57f
	v_max3_num_f32 v51 /*v307*/, v181 /*v949*/, v182 /*v950*/, v183 /*v951*/
	v_cndmask_b32_e32 v11 /*v267*/, 0xff800000, v43 /*v811*/, vcc_lo
	s_set_vgpr_msb 0x7f05
	v_cmp_le_i32_e32 vcc_lo, v54 /*v310*/, v53 /*v309*/
	s_set_vgpr_msb 0x57f
	v_max3_num_f32 v54 /*v310*/, v253 /*v1021*/, v236 /*v1004*/, v237 /*v1005*/
	s_set_vgpr_msb 0x7f1c
	v_max3_num_f32 v130, v130, v180 /*v948*/, v51 /*v307*/
	s_set_vgpr_msb 0x1c6a
	v_max3_num_f32 v51 /*v307*/, v194 /*v706*/, v196 /*v708*/, v198 /*v710*/
	s_set_vgpr_msb 0x6aaa
	v_max3_num_f32 v196 /*v708*/, v206 /*v718*/, v208 /*v720*/, v210 /*v722*/
	s_set_vgpr_msb 0xaa4c
	v_cndmask_b32_e32 v26 /*v282*/, 0xff800000, v44 /*v812*/, vcc_lo
	s_set_vgpr_msb 0x4c07
	v_cmp_le_i32_e32 vcc_lo, v80 /*v848*/, v53 /*v309*/
	s_set_vgpr_msb 0x75d
	v_max3_num_f32 v50 /*v306*/, v50 /*v306*/, v252 /*v1020*/, v54 /*v310*/
	s_set_vgpr_msb 0x5d6a
	v_max3_num_f32 v54 /*v310*/, v195 /*v707*/, v197 /*v709*/, v199 /*v711*/
	s_set_vgpr_msb 0x6aaa
	v_max3_num_f32 v197 /*v709*/, v207 /*v719*/, v209 /*v721*/, v211 /*v723*/
	s_set_vgpr_msb 0xaabf
	v_max3_num_f32 v210 /*v722*/, v4 /*v772*/, v216 /*v984*/, v6 /*v774*/
	s_set_vgpr_msb 0xbf4c
	v_cndmask_b32_e32 v27 /*v283*/, 0xff800000, v45 /*v813*/, vcc_lo
	s_set_vgpr_msb 0x4c07
	v_cmp_le_i32_e32 vcc_lo, v81 /*v849*/, v53 /*v309*/
	s_set_vgpr_msb 0x7bf
	v_max3_num_f32 v211 /*v723*/, v10 /*v778*/, v12 /*v780*/, v14 /*v782*/
	s_set_vgpr_msb 0xbfaa
	v_max3_num_f32 v194 /*v706*/, v200 /*v712*/, v202 /*v714*/, v204 /*v716*/
	v_max3_num_f32 v198 /*v710*/, v212 /*v724*/, v214 /*v726*/, v216 /*v728*/
	v_max3_num_f32 v200 /*v712*/, v218 /*v730*/, v220 /*v732*/, v222 /*v734*/
	s_set_vgpr_msb 0xaa4c
	v_cndmask_b32_e32 v18 /*v274*/, 0xff800000, v46 /*v814*/, vcc_lo
	s_set_vgpr_msb 0x4c07
	v_cmp_le_i32_e32 vcc_lo, v82 /*v850*/, v53 /*v309*/
	s_set_vgpr_msb 0x7ae
	v_max3_num_f32 v208 /*v720*/, v242 /*v754*/, v197 /*v965*/, v244 /*v756*/
	s_set_vgpr_msb 0xaeaa
	v_max3_num_f32 v212 /*v724*/, v248 /*v760*/, v250 /*v762*/, v252 /*v764*/
	s_set_vgpr_msb 0xaaae
	v_max3_num_f32 v210 /*v722*/, v210 /*v722*/, v8 /*v776*/, v211 /*v723*/
	s_set_vgpr_msb 0xaeaa
	v_max3_num_f32 v202 /*v714*/, v224 /*v736*/, v226 /*v738*/, v228 /*v740*/
	s_set_vgpr_msb 0xaa4c
	v_cndmask_b32_e32 v19 /*v275*/, 0xff800000, v47 /*v815*/, vcc_lo
	s_set_vgpr_msb 0x4c07
	v_cmp_le_i32_e32 vcc_lo, v83 /*v851*/, v53 /*v309*/
	s_set_vgpr_msb 0x7aa
	v_max3_num_f32 v204 /*v716*/, v230 /*v742*/, v232 /*v744*/, v234 /*v746*/
	v_max3_num_f32 v206 /*v718*/, v236 /*v748*/, v238 /*v750*/, v240 /*v752*/
	s_set_vgpr_msb 0xaabe
	v_max3_num_f32 v214 /*v726*/, v254 /*v766*/, v0 /*v768*/, v2 /*v770*/
	s_set_vgpr_msb 0xbe18
	v_max3_num_f32 v130, v130, v192 /*v704*/, v51 /*v307*/
	s_set_vgpr_msb 0x184c
	v_cndmask_b32_e32 v38 /*v294*/, 0xff800000, v48 /*v816*/, vcc_lo
	s_set_vgpr_msb 0x4c07
	v_cmp_le_i32_e32 vcc_lo, v84 /*v852*/, v53 /*v309*/
	s_set_vgpr_msb 0x76a
	v_max3_num_f32 v51 /*v307*/, v196 /*v708*/, v198 /*v710*/, v200 /*v712*/
	s_set_vgpr_msb 0x6aaa
	v_max3_num_f32 v192 /*v704*/, v208 /*v720*/, v246 /*v758*/, v212 /*v724*/
	s_set_vgpr_msb 0xaabe
	v_max3_num_f32 v196 /*v708*/, v210 /*v722*/, v242 /*v1010*/, v243 /*v1011*/
	s_set_vgpr_msb 0xbec5
	v_max_num_f32_e32 v5 /*v773*/, v26 /*v282*/, v27 /*v283*/
	s_set_vgpr_msb 0xc54c
	v_cndmask_b32_e32 v39 /*v295*/, 0xff800000, v49 /*v817*/, vcc_lo
	s_set_vgpr_msb 0x4c07
	v_cmp_le_i32_e32 vcc_lo, v85 /*v853*/, v53 /*v309*/
	s_set_vgpr_msb 0x7aa
	v_max3_num_f32 v200 /*v712*/, v202 /*v714*/, v204 /*v716*/, v206 /*v718*/
	s_set_vgpr_msb 0xaa18
	v_max3_num_f32 v130, v130, v194 /*v706*/, v51 /*v307*/
	s_set_vgpr_msb 0x186a
	v_max3_num_f32 v51 /*v307*/, v192 /*v704*/, v214 /*v726*/, v196 /*v708*/
	s_set_vgpr_msb 0x6ad5
	v_max3_num_f32 v7 /*v775*/, v19 /*v275*/, v38 /*v294*/, v39 /*v295*/
	s_set_vgpr_msb 0xd54c
	v_cndmask_b32_e32 v28 /*v284*/, 0xff800000, v50 /*v818*/, vcc_lo
	s_set_vgpr_msb 0x4c07
	v_cmp_le_i32_e32 vcc_lo, v72 /*v840*/, v53 /*v309*/
	s_set_vgpr_msb 0x7aa
	v_max3_num_f32 v199 /*v711*/, v213 /*v725*/, v215 /*v727*/, v217 /*v729*/
	s_set_vgpr_msb 0xaa18
	v_max3_num_f32 v130, v130, v200 /*v712*/, v51 /*v307*/
	s_set_vgpr_msb 0x1884
	v_max_num_f32_e32 v243 /*v755*/, v245, v12 /*v268*/
	s_set_vgpr_msb 0x84b7
	v_max3_num_f32 v215 /*v727*/, v5 /*v773*/, v18 /*v274*/, v7 /*v775*/
	s_set_vgpr_msb 0xb74c
	v_cndmask_b32_e32 v29 /*v285*/, 0xff800000, v51 /*v819*/, vcc_lo
	s_set_vgpr_msb 0x4c07
	v_cmp_le_i32_e32 vcc_lo, v73 /*v841*/, v53 /*v309*/
	s_set_vgpr_msb 0x7d5
	v_max3_num_f32 v3 /*v771*/, v17 /*v273*/, v10 /*v266*/, v11 /*v267*/
	s_set_vgpr_msb 0xd5aa
	v_max3_num_f32 v195 /*v707*/, v201 /*v713*/, v203 /*v715*/, v205 /*v717*/
	v_max3_num_f32 v201 /*v713*/, v219 /*v731*/, v221 /*v733*/, v223 /*v735*/
	s_set_vgpr_msb 0xaaa6
	v_max3_num_f32 v209 /*v721*/, v243 /*v755*/, v13 /*v269*/, v245 /*v757*/
	s_set_vgpr_msb 0xa64c
	v_cndmask_b32_e32 v20 /*v276*/, 0xff800000, v52 /*v820*/, vcc_lo
	s_set_vgpr_msb 0x4c07
	v_cmp_le_i32_e32 vcc_lo, v74 /*v842*/, v53 /*v309*/
	s_set_vgpr_msb 0x7aa
	v_max3_num_f32 v213 /*v725*/, v249 /*v761*/, v251 /*v763*/, v253 /*v765*/
	v_max3_num_f32 v203 /*v715*/, v225 /*v737*/, v227 /*v739*/, v229 /*v741*/
	v_max3_num_f32 v205 /*v717*/, v231 /*v743*/, v233 /*v745*/, v235 /*v747*/
	v_max3_num_f32 v207 /*v719*/, v237 /*v749*/, v239 /*v751*/, v241 /*v753*/
	s_set_vgpr_msb 0xaa4c
	v_cndmask_b32_e32 v21 /*v277*/, 0xff800000, v53 /*v821*/, vcc_lo
	s_set_vgpr_msb 0x4c07
	v_cmp_le_i32_e32 vcc_lo, v75 /*v843*/, v53 /*v309*/
	s_set_vgpr_msb 0x7d5
	v_max3_num_f32 v9 /*v777*/, v28 /*v284*/, v29 /*v285*/, v20 /*v276*/
	s_set_vgpr_msb 0xd5be
	v_max3_num_f32 v198 /*v710*/, v255 /*v767*/, v1 /*v769*/, v3 /*v771*/
	s_set_vgpr_msb 0xbe59
	v_max3_num_f32 v50 /*v306*/, v50 /*v306*/, v193 /*v705*/, v54 /*v310*/
	s_set_vgpr_msb 0x596a
	v_max3_num_f32 v54 /*v310*/, v197 /*v709*/, v199 /*v711*/, v201 /*v713*/
	s_set_vgpr_msb 0x6a4c
	v_cndmask_b32_e32 v34 /*v290*/, 0xff800000, v54 /*v822*/, vcc_lo
	s_set_vgpr_msb 0x4c07
	v_cmp_le_i32_e32 vcc_lo, v76 /*v844*/, v53 /*v309*/
	s_set_vgpr_msb 0x7aa
	v_max3_num_f32 v192 /*v704*/, v209 /*v721*/, v247 /*v759*/, v213 /*v725*/
	s_set_vgpr_msb 0xaa6a
	v_max3_num_f32 v51 /*v307*/, v203 /*v715*/, v205 /*v717*/, v207 /*v719*/
	s_set_vgpr_msb 0x6a59
	v_max3_num_f32 v50 /*v306*/, v50 /*v306*/, v195 /*v707*/, v54 /*v310*/
	s_set_vgpr_msb 0x594c
	v_cndmask_b32_e32 v35 /*v291*/, 0xff800000, v55 /*v823*/, vcc_lo
	s_set_vgpr_msb 0x4c07
	v_cmp_le_i32_e32 vcc_lo, v77 /*v845*/, v53 /*v309*/
	s_set_vgpr_msb 0x7d5
	v_max3_num_f32 v11 /*v779*/, v21 /*v277*/, v34 /*v290*/, v35 /*v291*/
	s_set_vgpr_msb 0xd54c
	v_cndmask_b32_e32 v30 /*v286*/, 0xff800000, v56 /*v824*/, vcc_lo
	s_set_vgpr_msb 0x4c07
	v_cmp_le_i32_e32 vcc_lo, v78 /*v846*/, v53 /*v309*/
	s_set_vgpr_msb 0x74c
	v_cndmask_b32_e32 v31 /*v287*/, 0xff800000, v57 /*v825*/, vcc_lo
	s_set_vgpr_msb 0x4c07
	v_cmp_le_i32_e32 vcc_lo, v79 /*v847*/, v53 /*v309*/
	s_set_vgpr_msb 0x74c
	v_cndmask_b32_e32 v40 /*v296*/, 0xff800000, v58 /*v826*/, vcc_lo
	s_set_vgpr_msb 0x4c07
	v_cmp_le_i32_e32 vcc_lo, v64 /*v832*/, v53 /*v309*/
	s_set_vgpr_msb 0x74c
	v_cndmask_b32_e32 v41 /*v297*/, 0xff800000, v59 /*v827*/, vcc_lo
	s_set_vgpr_msb 0x4c07
	v_cmp_le_i32_e32 vcc_lo, v65 /*v833*/, v53 /*v309*/
	s_set_vgpr_msb 0x7d5
	v_max3_num_f32 v13 /*v781*/, v30 /*v286*/, v31 /*v287*/, v40 /*v296*/
	s_set_vgpr_msb 0xd54c
	v_cndmask_b32_e32 v36 /*v292*/, 0xff800000, v60 /*v828*/, vcc_lo
	s_set_vgpr_msb 0x4c07
	v_cmp_le_i32_e32 vcc_lo, v66 /*v834*/, v53 /*v309*/
	s_set_vgpr_msb 0x74c
	v_cndmask_b32_e32 v37 /*v293*/, 0xff800000, v61 /*v829*/, vcc_lo
	s_set_vgpr_msb 0x4c07
	v_cmp_le_i32_e32 vcc_lo, v67 /*v835*/, v53 /*v309*/
	s_set_vgpr_msb 0x7d5
	v_max3_num_f32 v15 /*v783*/, v41 /*v297*/, v36 /*v292*/, v37 /*v293*/
	s_set_vgpr_msb 0xd54c
	v_cndmask_b32_e32 v22 /*v278*/, 0xff800000, v62 /*v830*/, vcc_lo
	s_set_vgpr_msb 0x4c07
	v_cmp_le_i32_e32 vcc_lo, v68 /*v836*/, v53 /*v309*/
	s_set_vgpr_msb 0x7bf
	v_max3_num_f32 v211 /*v723*/, v11 /*v779*/, v13 /*v781*/, v15 /*v783*/
	s_set_vgpr_msb 0xbf4c
	v_cndmask_b32_e32 v23 /*v279*/, 0xff800000, v63 /*v831*/, vcc_lo
	s_set_vgpr_msb 0x4cae
	v_max3_num_f32 v202 /*v714*/, v215 /*v727*/, v9 /*v777*/, v211 /*v723*/
	s_set_vgpr_msb 0xae96
	v_max3_num_f32 v193 /*v705*/, v202 /*v714*/, v22 /*v278*/, v23 /*v279*/
	s_set_vgpr_msb 0x966a
	v_max3_num_f32 v54 /*v310*/, v192 /*v704*/, v198 /*v710*/, v193 /*v705*/
	s_set_vgpr_msb 0x6a80
	v_mov_b32_e32 v192 /*v704*/, v130
	s_set_vgpr_msb 0x8055
	v_max3_num_f32 v50 /*v306*/, v50 /*v306*/, v51 /*v307*/, v54 /*v310*/
	s_set_vgpr_msb 0x5582
	v_permlanex16_b32 v192 /*v704*/, v192 /*v704*/, s82, 0xfedcba98
	s_set_vgpr_msb 0x8241
	v_mov_b32_e32 v51 /*v307*/, v50 /*v306*/
	s_set_vgpr_msb 0x4108
	v_max_num_f32_e32 v130, v130, v192 /*v704*/
	s_set_vgpr_msb 0x841
	v_permlanex16_b32 v51 /*v307*/, v51 /*v307*/, s82, 0xfedcba98
	s_set_vgpr_msb 0x4140
	v_sub_f32_e32 v54 /*v310*/, v130, v128
	s_set_vgpr_msb 0x4000
	v_max_num_f32_e32 v130, v128, v130
	s_set_vgpr_msb 0x45
	v_max_num_f32_e32 v51 /*v307*/, v50 /*v306*/, v51 /*v307*/
	v_cmp_lt_f32_e32 vcc_lo, 0x41000000, v54 /*v310*/
	s_set_vgpr_msb 0x4541
	v_sub_f32_e32 v54 /*v310*/, v51 /*v307*/, v134
	s_cmp_eq_u32 vcc_lo, 0
	v_max_num_f32_e32 v51 /*v307*/, v51 /*v307*/, v134
	s_cselect_b32 s2, -1, 0
	s_set_vgpr_msb 0x4140
	v_cndmask_b32_e64 v50 /*v306*/, v130, v128, s2
	s_set_vgpr_msb 0x4004
	v_cmp_lt_f32_e64 s2, 0x41000000, v54 /*v310*/
	v_mul_f32_e32 v130, 0xbfb8aa3b, v50 /*v306*/
	s_cmp_lg_u32 s2, 0
	v_sub_f32_e32 v128, v128, v50 /*v306*/
	s_cselect_b32 s5, -1, 0
	s_cmp_eq_u32 s2, 0
	s_set_vgpr_msb 0x483
	v_pk_fma_f32 v[206:207] /*v[718:719]*/, v[186:187] /*v[954:955]*/, s[56:57], v[130:131] op_sel_hi:[1,0,0]
	s_cselect_b32 s2, -1, 0
	v_pk_fma_f32 v[212:213] /*v[724:725]*/, v[170:171] /*v[938:939]*/, s[56:57], v[130:131] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x8341
	v_cndmask_b32_e64 v51 /*v307*/, v51 /*v307*/, v134, s2
	s_set_vgpr_msb 0x4183
	v_pk_fma_f32 v[220:221] /*v[732:733]*/, v[174:175] /*v[942:943]*/, s[56:57], v[130:131] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x8382
	v_exp_f32_e32 v222 /*v734*/, v206 /*v718*/
	v_exp_f32_e32 v232 /*v744*/, v207 /*v719*/
	v_nop
	s_set_vgpr_msb 0x8283
	v_pk_fma_f32 v[206:207] /*v[718:719]*/, v[192:193] /*v[960:961]*/, s[56:57], v[130:131] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x8344
	v_mul_f32_e32 v54 /*v310*/, 0xbfb8aa3b, v51 /*v307*/
	s_set_vgpr_msb 0x4482
	v_exp_f32_e32 v214 /*v726*/, v213 /*v725*/
	v_exp_f32_e32 v228 /*v740*/, v220 /*v732*/
	s_set_vgpr_msb 0x8283
	v_pk_fma_f32 v[224:225] /*v[736:737]*/, v[162:163] /*v[930:931]*/, s[56:57], v[130:131] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x83c2
	v_exp_f32_e32 v16 /*v784*/, v206 /*v718*/
	s_set_vgpr_msb 0xc282
	v_exp_f32_e32 v206 /*v718*/, v212 /*v724*/
	v_nop
	s_set_vgpr_msb 0x8283
	v_pk_fma_f32 v[212:213] /*v[724:725]*/, v[160:161] /*v[928:929]*/, s[56:57], v[130:131] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x8310
	v_pk_fma_f32 v[192:193], v[192:193], s[56:57], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1082
	v_exp_f32_e32 v240 /*v752*/, v221 /*v733*/
	v_nop
	s_set_vgpr_msb 0x8283
	v_pk_fma_f32 v[220:221] /*v[732:733]*/, v[164:165] /*v[932:933]*/, s[56:57], v[130:131] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x83c2
	v_exp_f32_e32 v6 /*v774*/, v224 /*v736*/
	s_set_vgpr_msb 0xc282
	v_exp_f32_e32 v244 /*v756*/, v212 /*v724*/
	s_set_vgpr_msb 0x82c2
	v_exp_f32_e32 v2 /*v770*/, v213 /*v725*/
	v_nop
	s_set_vgpr_msb 0xc283
	v_pk_fma_f32 v[212:213] /*v[724:725]*/, v[166:167] /*v[934:935]*/, s[56:57], v[130:131] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x8380
	v_exp_f32_e32 v245 /*v757*/, v192
	s_set_vgpr_msb 0x80c0
	v_exp_f32_e32 v3 /*v771*/, v193
	v_nop
	s_set_vgpr_msb 0xc010
	v_pk_fma_f32 v[192:193], v[194:195], s[56:57], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[194:195], v[218:219], s[56:57], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x10c2
	v_exp_f32_e32 v22 /*v790*/, v225 /*v737*/
	v_nop
	s_set_vgpr_msb 0xc283
	v_pk_fma_f32 v[224:225] /*v[736:737]*/, v[152:153] /*v[920:921]*/, s[56:57], v[130:131] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x83c2
	v_exp_f32_e32 v42 /*v810*/, v221 /*v733*/
	s_set_vgpr_msb 0xc283
	v_pk_fma_f32 v[230:231] /*v[742:743]*/, v[154:155] /*v[922:923]*/, s[56:57], v[130:131] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x83c2
	v_exp_f32_e32 v64 /*v832*/, v213 /*v725*/
	s_set_vgpr_msb 0xc283
	v_pk_fma_f32 v[238:239] /*v[750:751]*/, v[156:157] /*v[924:925]*/, s[56:57], v[130:131] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x83c0
	v_exp_f32_e32 v49 /*v817*/, v192
	v_exp_f32_e32 v65 /*v833*/, v193
	s_set_vgpr_msb 0xc080
	v_exp_f32_e32 v213 /*v725*/, v194
	s_set_vgpr_msb 0x8010
	v_pk_fma_f32 v[192:193], v[196:197], s[56:57], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1080
	v_exp_f32_e32 v221 /*v733*/, v195
	v_nop
	s_set_vgpr_msb 0x8010
	v_pk_fma_f32 v[194:195], v[210:211], s[56:57], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[196:197], v[204:205], s[56:57], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x10c2
	v_exp_f32_e32 v48 /*v816*/, v212 /*v724*/
	s_set_vgpr_msb 0xc282
	v_exp_f32_e32 v212 /*v724*/, v224 /*v736*/
	v_exp_f32_e32 v224 /*v736*/, v230 /*v742*/
	s_set_vgpr_msb 0x8283
	v_pk_fma_f32 v[242:243] /*v[754:755]*/, v[158:159] /*v[926:927]*/, s[56:57], v[130:131] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x8382
	v_exp_f32_e32 v234 /*v746*/, v231 /*v743*/
	v_nop
	s_set_vgpr_msb 0x8283
	v_pk_fma_f32 v[230:231] /*v[742:743]*/, v[144:145] /*v[912:913]*/, s[56:57], v[130:131] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x8382
	v_exp_f32_e32 v250 /*v762*/, v239 /*v751*/
	s_set_vgpr_msb 0x8283
	v_pk_fma_f32 v[246:247] /*v[758:759]*/, v[146:147] /*v[914:915]*/, s[56:57], v[130:131] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x8380
	v_exp_f32_e32 v239 /*v751*/, v192
	v_exp_f32_e32 v251 /*v763*/, v193
	v_exp_f32_e32 v255 /*v767*/, v194
	s_set_vgpr_msb 0x8010
	v_pk_fma_f32 v[192:193], v[228:229], s[56:57], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x10c0
	v_exp_f32_e32 v13 /*v781*/, v195
	v_exp_f32_e32 v19 /*v787*/, v196
	s_set_vgpr_msb 0xc010
	v_pk_fma_f32 v[194:195], v[212:213], s[56:57], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x10c0
	v_exp_f32_e32 v35 /*v803*/, v197
	v_nop
	s_set_vgpr_msb 0xc010
	v_pk_fma_f32 v[196:197], v[206:207], s[56:57], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1082
	v_exp_f32_e32 v254 /*v766*/, v242 /*v754*/
	s_set_vgpr_msb 0x82c2
	v_exp_f32_e32 v12 /*v780*/, v243 /*v755*/
	v_exp_f32_e32 v18 /*v786*/, v230 /*v742*/
	s_set_vgpr_msb 0xc283
	v_pk_fma_f32 v[242:243] /*v[754:755]*/, v[148:149] /*v[916:917]*/, s[56:57], v[130:131] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x83c2
	v_exp_f32_e32 v34 /*v802*/, v231 /*v743*/
	v_exp_f32_e32 v38 /*v806*/, v246 /*v758*/
	s_set_vgpr_msb 0xc283
	v_pk_fma_f32 v[230:231] /*v[742:743]*/, v[150:151] /*v[918:919]*/, s[56:57], v[130:131] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x83c2
	v_exp_f32_e32 v54 /*v822*/, v247 /*v759*/
	v_nop
	s_set_vgpr_msb 0xc283
	v_pk_fma_f32 v[246:247] /*v[758:759]*/, v[136:137] /*v[904:905]*/, s[56:57], v[130:131] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x83c3
	v_pk_fma_f32 v[0:1] /*v[768:769]*/, v[138:139] /*v[906:907]*/, s[56:57], v[130:131] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[8:9] /*v[776:777]*/, v[140:141] /*v[908:909]*/, s[56:57], v[130:131] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xc3c0
	v_exp_f32_e32 v39 /*v807*/, v192
	v_exp_f32_e32 v55 /*v823*/, v193
	v_exp_f32_e32 v59 /*v827*/, v194
	s_set_vgpr_msb 0xc010
	v_pk_fma_f32 v[192:193], v[220:221], s[56:57], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x10c0
	v_exp_f32_e32 v75 /*v843*/, v195
	v_exp_f32_e32 v81 /*v849*/, v196
	s_set_vgpr_msb 0xc010
	v_pk_fma_f32 v[194:195], v[214:215], s[56:57], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x10c0
	v_exp_f32_e32 v97 /*v865*/, v197
	v_nop
	s_set_vgpr_msb 0xc010
	v_pk_fma_f32 v[196:197], v[238:239], s[56:57], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x10c2
	v_exp_f32_e32 v58 /*v826*/, v242 /*v754*/
	v_exp_f32_e32 v74 /*v842*/, v243 /*v755*/
	v_exp_f32_e32 v80 /*v848*/, v230 /*v742*/
	v_exp_f32_e32 v96 /*v864*/, v231 /*v743*/
	s_set_vgpr_msb 0xc282
	v_exp_f32_e32 v230 /*v742*/, v246 /*v758*/
	v_exp_f32_e32 v242 /*v754*/, v247 /*v759*/
	s_set_vgpr_msb 0x8283
	v_exp_f32_e32 v246 /*v758*/, v0 /*v768*/
	s_set_vgpr_msb 0x83c3
	v_pk_fma_f32 v[14:15] /*v[782:783]*/, v[142:143] /*v[910:911]*/, s[56:57], v[130:131] op_sel_hi:[1,0,0]
	v_exp_f32_e32 v4 /*v772*/, v1 /*v769*/
	v_nop
	v_pk_fma_f32 v[0:1] /*v[768:769]*/, v[128:129] /*v[896:897]*/, s[56:57], v[130:131] op_sel_hi:[1,0,0]
	v_exp_f32_e32 v24 /*v792*/, v9 /*v777*/
	s_set_vgpr_msb 0xc380
	v_exp_f32_e32 v231 /*v743*/, v192
	v_exp_f32_e32 v243 /*v755*/, v193
	v_exp_f32_e32 v247 /*v759*/, v194
	s_set_vgpr_msb 0x8010
	v_pk_fma_f32 v[192:193], v[222:223], s[56:57], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x10c0
	v_exp_f32_e32 v5 /*v773*/, v195
	v_exp_f32_e32 v9 /*v777*/, v196
	s_set_vgpr_msb 0xc010
	v_pk_fma_f32 v[194:195], v[216:217], s[56:57], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x10c0
	v_exp_f32_e32 v25 /*v793*/, v197
	v_nop
	s_set_vgpr_msb 0xc010
	v_pk_fma_f32 v[196:197], v[230:231], s[56:57], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x10c3
	v_exp_f32_e32 v28 /*v796*/, v14 /*v782*/
	v_pk_fma_f32 v[20:21] /*v[788:789]*/, v[130:131] /*v[898:899]*/, s[56:57], v[130:131] op_sel_hi:[1,0,0]
	v_exp_f32_e32 v44 /*v812*/, v15 /*v783*/
	v_exp_f32_e32 v50 /*v818*/, v0 /*v768*/
	v_pk_fma_f32 v[14:15] /*v[782:783]*/, v[132:133] /*v[900:901]*/, s[56:57], v[130:131] op_sel_hi:[1,0,0]
	v_exp_f32_e32 v66 /*v834*/, v1 /*v769*/
	v_nop
	v_pk_fma_f32 v[0:1] /*v[768:769]*/, v[134:135] /*v[902:903]*/, s[56:57], v[130:131] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xc3c0
	v_exp_f32_e32 v29 /*v797*/, v192
	v_exp_f32_e32 v45 /*v813*/, v193
	v_exp_f32_e32 v51 /*v819*/, v194
	s_set_vgpr_msb 0xc010
	v_pk_fma_f32 v[192:193], v[224:225], s[56:57], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x10c0
	v_exp_f32_e32 v67 /*v835*/, v195
	v_exp_f32_e32 v71 /*v839*/, v196
	s_set_vgpr_msb 0xc010
	v_pk_fma_f32 v[194:195], v[248:249], s[56:57], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x10c0
	v_exp_f32_e32 v87 /*v855*/, v197
	v_nop
	s_set_vgpr_msb 0xc010
	v_pk_fma_f32 v[196:197], v[232:233], s[56:57], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x10c3
	v_exp_f32_e32 v70 /*v838*/, v20 /*v788*/
	v_exp_f32_e32 v86 /*v854*/, v21 /*v789*/
	v_nop
	v_pk_fma_f32 v[20:21] /*v[788:789]*/, v[120:121] /*v[888:889]*/, s[56:57], v[130:131] op_sel_hi:[1,0,0]
	v_exp_f32_e32 v106 /*v874*/, v15 /*v783*/
	v_pk_fma_f32 v[30:31] /*v[798:799]*/, v[122:123] /*v[890:891]*/, s[56:57], v[130:131] op_sel_hi:[1,0,0]
	v_exp_f32_e32 v126 /*v894*/, v1 /*v769*/
	v_pk_fma_f32 v[40:41] /*v[808:809]*/, v[124:125] /*v[892:893]*/, s[56:57], v[130:131] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xc3c0
	v_exp_f32_e32 v91 /*v859*/, v192
	v_exp_f32_e32 v107 /*v875*/, v193
	v_exp_f32_e32 v113 /*v881*/, v194
	s_set_vgpr_msb 0xc010
	v_pk_fma_f32 v[192:193], v[226:227], s[56:57], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x10c0
	v_exp_f32_e32 v127 /*v895*/, v195
	v_exp_f32_e32 v1 /*v769*/, v196
	s_set_vgpr_msb 0xc010
	v_pk_fma_f32 v[194:195], v[240:241], s[56:57], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x10c0
	v_exp_f32_e32 v15 /*v783*/, v197
	v_nop
	s_set_vgpr_msb 0xc010
	v_pk_fma_f32 v[196:197], v[234:235], s[56:57], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x10c3
	v_exp_f32_e32 v90 /*v858*/, v14 /*v782*/
	v_exp_f32_e32 v112 /*v880*/, v0 /*v768*/
	v_exp_f32_e32 v0 /*v768*/, v20 /*v788*/
	v_exp_f32_e32 v14 /*v782*/, v21 /*v789*/
	v_exp_f32_e32 v20 /*v788*/, v30 /*v798*/
	v_pk_fma_f32 v[46:47] /*v[814:815]*/, v[176:177] /*v[944:945]*/, s[56:57], v[130:131] op_sel_hi:[1,0,0]
	v_exp_f32_e32 v36 /*v804*/, v31 /*v799*/
	v_nop
	v_pk_fma_f32 v[30:31] /*v[798:799]*/, v[194:195] /*v[962:963]*/, s[56:57], v[130:131] op_sel_hi:[1,0,0]
	v_exp_f32_e32 v56 /*v824*/, v41 /*v809*/
	v_pk_fma_f32 v[52:53] /*v[820:821]*/, v[114:115] /*v[882:883]*/, s[56:57], v[130:131] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xc3c0
	v_exp_f32_e32 v21 /*v789*/, v192
	v_exp_f32_e32 v37 /*v805*/, v193
	v_exp_f32_e32 v41 /*v809*/, v194
	s_set_vgpr_msb 0xc011
	v_pk_fma_f32 v[192:193], v[2:3] /*v[258:259]*/, s[56:57], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x11c0
	v_exp_f32_e32 v57 /*v825*/, v195
	v_exp_f32_e32 v61 /*v829*/, v196
	s_set_vgpr_msb 0xc010
	v_pk_fma_f32 v[194:195], v[242:243], s[56:57], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x10c0
	v_exp_f32_e32 v77 /*v845*/, v197
	v_nop
	s_set_vgpr_msb 0xc010
	v_pk_fma_f32 v[196:197], v[236:237], s[56:57], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x10c3
	v_exp_f32_e32 v60 /*v828*/, v46 /*v814*/
	v_exp_f32_e32 v76 /*v844*/, v47 /*v815*/
	v_exp_f32_e32 v82 /*v850*/, v30 /*v798*/
	v_pk_fma_f32 v[46:47] /*v[814:815]*/, v[116:117] /*v[884:885]*/, s[56:57], v[130:131] op_sel_hi:[1,0,0]
	v_exp_f32_e32 v98 /*v866*/, v31 /*v799*/
	v_exp_f32_e32 v102 /*v870*/, v52 /*v820*/
	v_pk_fma_f32 v[30:31] /*v[798:799]*/, v[118:119] /*v[886:887]*/, s[56:57], v[130:131] op_sel_hi:[1,0,0]
	v_exp_f32_e32 v118 /*v886*/, v53 /*v821*/
	v_nop
	v_pk_fma_f32 v[52:53] /*v[820:821]*/, v[104:105] /*v[872:873]*/, s[56:57], v[130:131] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xc3c0
	v_exp_f32_e32 v83 /*v851*/, v192
	v_exp_f32_e32 v99 /*v867*/, v193
	v_exp_f32_e32 v103 /*v871*/, v194
	s_set_vgpr_msb 0xc010
	v_pk_fma_f32 v[192:193], v[250:251], s[56:57], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x10c0
	v_exp_f32_e32 v119 /*v887*/, v195
	v_exp_f32_e32 v123 /*v891*/, v196
	s_set_vgpr_msb 0xc010
	v_pk_fma_f32 v[194:195], v[244:245], s[56:57], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x10c0
	v_exp_f32_e32 v137 /*v905*/, v197
	v_nop
	s_set_vgpr_msb 0xc011
	v_pk_fma_f32 v[196:197], v[12:13] /*v[268:269]*/, s[56:57], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x11c3
	v_exp_f32_e32 v122 /*v890*/, v46 /*v814*/
	v_exp_f32_e32 v136 /*v904*/, v47 /*v815*/
	v_pk_fma_f32 v[62:63] /*v[830:831]*/, v[196:197] /*v[964:965]*/, s[56:57], v[130:131] op_sel_hi:[1,0,0]
	v_exp_f32_e32 v152 /*v920*/, v31 /*v799*/
	v_pk_fma_f32 v[72:73] /*v[840:841]*/, v[108:109] /*v[876:877]*/, s[56:57], v[130:131] op_sel_hi:[1,0,0]
	v_exp_f32_e32 v46 /*v814*/, v53 /*v821*/
	s_set_vgpr_msb 0xc3c0
	v_exp_f32_e32 v141 /*v909*/, v192
	v_exp_f32_e32 v153 /*v921*/, v193
	v_exp_f32_e32 v31 /*v799*/, v194
	s_set_vgpr_msb 0xc010
	v_pk_fma_f32 v[192:193], v[252:253], s[56:57], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x10c0
	v_exp_f32_e32 v47 /*v815*/, v195
	v_exp_f32_e32 v53 /*v821*/, v196
	s_set_vgpr_msb 0xc010
	v_pk_fma_f32 v[194:195], v[246:247], s[56:57], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x10c0
	v_exp_f32_e32 v69 /*v837*/, v197
	v_nop
	s_set_vgpr_msb 0xc011
	v_pk_fma_f32 v[196:197], v[4:5] /*v[260:261]*/, s[56:57], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x11c3
	v_exp_f32_e32 v140 /*v908*/, v30 /*v798*/
	v_exp_f32_e32 v30 /*v798*/, v52 /*v820*/
	v_exp_f32_e32 v52 /*v820*/, v62 /*v830*/
	v_pk_fma_f32 v[78:79] /*v[846:847]*/, v[110:111] /*v[878:879]*/, s[56:57], v[130:131] op_sel_hi:[1,0,0]
	v_exp_f32_e32 v68 /*v836*/, v63 /*v831*/
	v_nop
	v_pk_fma_f32 v[62:63] /*v[830:831]*/, v[198:199] /*v[966:967]*/, s[56:57], v[130:131] op_sel_hi:[1,0,0]
	v_exp_f32_e32 v88 /*v856*/, v73 /*v841*/
	v_pk_fma_f32 v[84:85] /*v[852:853]*/, v[200:201] /*v[968:969]*/, s[56:57], v[130:131] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xc3c0
	v_exp_f32_e32 v73 /*v841*/, v192
	v_exp_f32_e32 v89 /*v857*/, v193
	v_exp_f32_e32 v93 /*v861*/, v194
	s_set_vgpr_msb 0xc010
	v_pk_fma_f32 v[192:193], v[254:255], s[56:57], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x10c0
	v_exp_f32_e32 v109 /*v877*/, v195
	v_exp_f32_e32 v115 /*v883*/, v196
	s_set_vgpr_msb 0xc011
	v_pk_fma_f32 v[194:195], v[24:25] /*v[280:281]*/, s[56:57], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x11c0
	v_exp_f32_e32 v129 /*v897*/, v197
	v_nop
	s_set_vgpr_msb 0xc011
	v_pk_fma_f32 v[196:197], v[6:7] /*v[262:263]*/, s[56:57], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x11c3
	v_exp_f32_e32 v92 /*v860*/, v78 /*v846*/
	v_exp_f32_e32 v108 /*v876*/, v79 /*v847*/
	v_exp_f32_e32 v114 /*v882*/, v62 /*v830*/
	v_pk_fma_f32 v[78:79] /*v[846:847]*/, v[100:101] /*v[868:869]*/, s[56:57], v[130:131] op_sel_hi:[1,0,0]
	v_exp_f32_e32 v128 /*v896*/, v63 /*v831*/
	v_exp_f32_e32 v132 /*v900*/, v84 /*v852*/
	v_pk_fma_f32 v[62:63] /*v[830:831]*/, v[202:203] /*v[970:971]*/, s[56:57], v[130:131] op_sel_hi:[1,0,0]
	v_exp_f32_e32 v144 /*v912*/, v85 /*v853*/
	v_nop
	v_pk_fma_f32 v[84:85] /*v[852:853]*/, v[204:205] /*v[972:973]*/, s[56:57], v[130:131] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[100:101] /*v[868:869]*/, v[206:207] /*v[974:975]*/, s[56:57], v[130:131] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[104:105] /*v[872:873]*/, v[208:209] /*v[976:977]*/, s[56:57], v[130:131] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xc3c0
	v_exp_f32_e32 v133 /*v901*/, v192
	v_exp_f32_e32 v145 /*v913*/, v193
	v_exp_f32_e32 v149 /*v917*/, v194
	s_set_vgpr_msb 0xc011
	v_pk_fma_f32 v[192:193], v[0:1] /*v[256:257]*/, s[56:57], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x11c0
	v_exp_f32_e32 v159 /*v927*/, v195
	v_exp_f32_e32 v163 /*v931*/, v196
	s_set_vgpr_msb 0xc011
	v_pk_fma_f32 v[194:195], v[14:15] /*v[270:271]*/, s[56:57], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x11c0
	v_exp_f32_e32 v171 /*v939*/, v197
	v_nop
	s_set_vgpr_msb 0xc011
	v_pk_fma_f32 v[196:197], v[8:9] /*v[264:265]*/, s[56:57], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1183
	v_pk_fma_f32 v[192:193] /*v[704:705]*/, v[178:179] /*v[946:947]*/, s[56:57], v[130:131] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x83d3
	v_pk_fma_f32 v[192:193] /*v[960:961]*/, v[232:233] /*v[1000:1001]*/, s[56:57], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xd383
	v_pk_fma_f32 v[198:199] /*v[710:711]*/, v[182:183] /*v[950:951]*/, s[56:57], v[130:131] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x83c3
	v_exp_f32_e32 v148 /*v916*/, v78 /*v846*/
	v_exp_f32_e32 v158 /*v926*/, v79 /*v847*/
	v_exp_f32_e32 v162 /*v930*/, v62 /*v830*/
	v_exp_f32_e32 v170 /*v938*/, v63 /*v831*/
	v_exp_f32_e32 v62 /*v830*/, v84 /*v852*/
	v_exp_f32_e32 v78 /*v846*/, v85 /*v853*/
	v_exp_f32_e32 v84 /*v852*/, v100 /*v868*/
	v_pk_fma_f32 v[94:95] /*v[862:863]*/, v[94:95] /*v[862:863]*/, s[56:57], v[130:131] op_sel_hi:[1,0,0]
	v_exp_f32_e32 v100 /*v868*/, v101 /*v869*/
	v_pk_fma_f32 v[110:111] /*v[878:879]*/, v[210:211] /*v[978:979]*/, s[56:57], v[130:131] op_sel_hi:[1,0,0]
	v_exp_f32_e32 v120 /*v888*/, v105 /*v873*/
	s_set_vgpr_msb 0xc3c0
	v_exp_f32_e32 v63 /*v831*/, v192
	v_exp_f32_e32 v79 /*v847*/, v193
	v_exp_f32_e32 v85 /*v853*/, v194
	s_set_vgpr_msb 0xc011
	v_pk_fma_f32 v[192:193], v[32:33] /*v[288:289]*/, s[56:57], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x11c0
	v_exp_f32_e32 v101 /*v869*/, v195
	v_exp_f32_e32 v105 /*v873*/, v196
	s_set_vgpr_msb 0xc011
	v_pk_fma_f32 v[194:195], v[16:17] /*v[272:273]*/, s[56:57], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x11c0
	v_exp_f32_e32 v121 /*v889*/, v197
	v_nop
	s_set_vgpr_msb 0xc011
	v_pk_fma_f32 v[196:197], v[10:11] /*v[266:267]*/, s[56:57], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1182
	v_exp_f32_e32 v194 /*v706*/, v193 /*v705*/
	s_set_vgpr_msb 0x8283
	v_exp_f32_e32 v193 /*v705*/, v192 /*v960*/
	v_exp_f32_e32 v195 /*v707*/, v193 /*v961*/
	v_nop
	s_set_vgpr_msb 0x83d3
	v_pk_fma_f32 v[192:193] /*v[960:961]*/, v[234:235] /*v[1002:1003]*/, s[56:57], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xd310
	v_pk_fma_f32 v[198:199], v[198:199], s[56:57], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1082
	v_exp_f32_e32 v202 /*v714*/, v198 /*v710*/
	v_exp_f32_e32 v208 /*v720*/, v199 /*v711*/
	v_nop
	s_set_vgpr_msb 0x8283
	v_pk_fma_f32 v[198:199] /*v[710:711]*/, v[188:189] /*v[956:957]*/, s[56:57], v[130:131] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x83c3
	v_exp_f32_e32 v124 /*v892*/, v94 /*v862*/
	v_pk_fma_f32 v[116:117] /*v[884:885]*/, v[212:213] /*v[980:981]*/, s[56:57], v[130:131] op_sel_hi:[1,0,0]
	v_exp_f32_e32 v138 /*v906*/, v95 /*v863*/
	v_exp_f32_e32 v142 /*v910*/, v110 /*v878*/
	v_pk_fma_f32 v[94:95] /*v[862:863]*/, v[214:215] /*v[982:983]*/, s[56:57], v[130:131] op_sel_hi:[1,0,0]
	v_exp_f32_e32 v154 /*v922*/, v111 /*v879*/
	v_nop
	v_pk_fma_f32 v[110:111] /*v[878:879]*/, v[216:217] /*v[984:985]*/, s[56:57], v[130:131] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xc3c0
	v_exp_f32_e32 v125 /*v893*/, v192
	v_exp_f32_e32 v139 /*v907*/, v193
	v_exp_f32_e32 v143 /*v911*/, v194
	s_set_vgpr_msb 0xc011
	v_pk_fma_f32 v[192:193], v[26:27] /*v[282:283]*/, s[56:57], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x11c0
	v_exp_f32_e32 v155 /*v923*/, v195
	v_exp_f32_e32 v157 /*v925*/, v196
	s_set_vgpr_msb 0xc011
	v_pk_fma_f32 v[194:195], v[18:19] /*v[274:275]*/, s[56:57], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x11c0
	v_exp_f32_e32 v167 /*v935*/, v197
	v_nop
	s_set_vgpr_msb 0xc011
	v_pk_fma_f32 v[196:197], v[38:39] /*v[294:295]*/, s[56:57], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1183
	v_pk_fma_f32 v[196:197] /*v[708:709]*/, v[180:181] /*v[948:949]*/, s[56:57], v[130:131] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[204:205] /*v[716:717]*/, v[184:185] /*v[952:953]*/, s[56:57], v[130:131] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x83d3
	v_pk_fma_f32 v[194:195] /*v[962:963]*/, v[252:253] /*v[1020:1021]*/, s[56:57], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[196:197] /*v[964:965]*/, v[236:237] /*v[1004:1005]*/, s[56:57], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xd383
	v_exp_f32_e32 v211 /*v723*/, v192 /*v960*/
	v_exp_f32_e32 v219 /*v731*/, v193 /*v961*/
	v_nop
	s_set_vgpr_msb 0x83d3
	v_pk_fma_f32 v[192:193] /*v[960:961]*/, v[246:247] /*v[1014:1015]*/, s[56:57], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xd380
	v_exp_f32_e32 v253 /*v765*/, v198
	s_set_vgpr_msb 0x80c0
	v_exp_f32_e32 v11 /*v779*/, v199
	v_nop
	s_set_vgpr_msb 0xc013
	v_pk_fma_f32 v[198:199], v[254:255] /*v[1022:1023]*/, s[56:57], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1382
	v_exp_f32_e32 v236 /*v748*/, v198 /*v710*/
	v_exp_f32_e32 v248 /*v760*/, v199 /*v711*/
	v_nop
	s_set_vgpr_msb 0x8283
	v_pk_fma_f32 v[198:199] /*v[710:711]*/, v[168:169] /*v[936:937]*/, s[56:57], v[130:131] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x83c3
	v_exp_f32_e32 v156 /*v924*/, v116 /*v884*/
	v_exp_f32_e32 v166 /*v934*/, v117 /*v885*/
	v_nop
	v_pk_fma_f32 v[116:117] /*v[884:885]*/, v[218:219] /*v[986:987]*/, s[56:57], v[130:131] op_sel_hi:[1,0,0]
	v_exp_f32_e32 v178 /*v946*/, v95 /*v863*/
	v_pk_fma_f32 v[130:131] /*v[898:899]*/, v[220:221] /*v[988:989]*/, s[56:57], v[130:131] op_sel_hi:[1,0,0]
	v_exp_f32_e32 v186 /*v954*/, v111 /*v879*/
	v_pk_fma_f32 v[134:135] /*v[902:903]*/, v[222:223] /*v[990:991]*/, s[56:57], v[130:131] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[150:151] /*v[918:919]*/, v[224:225] /*v[992:993]*/, s[56:57], v[130:131] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xc3c0
	v_exp_f32_e32 v169 /*v937*/, v192
	v_exp_f32_e32 v179 /*v947*/, v193
	v_exp_f32_e32 v181 /*v949*/, v194
	s_set_vgpr_msb 0xc011
	v_pk_fma_f32 v[192:193], v[28:29] /*v[284:285]*/, s[56:57], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x11c0
	v_exp_f32_e32 v187 /*v955*/, v195
	v_exp_f32_e32 v95 /*v863*/, v196
	s_set_vgpr_msb 0xc011
	v_pk_fma_f32 v[194:195], v[20:21] /*v[276:277]*/, s[56:57], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x11c0
	v_exp_f32_e32 v111 /*v879*/, v197
	v_nop
	s_set_vgpr_msb 0xc011
	v_pk_fma_f32 v[196:197], v[34:35] /*v[290:291]*/, s[56:57], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1182
	v_exp_f32_e32 v192 /*v704*/, v192 /*v704*/
	v_exp_f32_e32 v200 /*v712*/, v197 /*v709*/
	v_exp_f32_e32 v210 /*v722*/, v204 /*v716*/
	v_exp_f32_e32 v218 /*v730*/, v205 /*v717*/
	v_nop
	s_set_vgpr_msb 0x8283
	v_pk_fma_f32 v[204:205] /*v[716:717]*/, v[190:191] /*v[958:959]*/, s[56:57], v[130:131] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x83c2
	v_exp_f32_e32 v32 /*v800*/, v207 /*v719*/
	s_set_vgpr_msb 0xc283
	v_pk_fma_f32 v[216:217] /*v[728:729]*/, v[172:173] /*v[940:941]*/, s[56:57], v[130:131] op_sel_hi:[1,0,0]
	v_exp_f32_e32 v197 /*v709*/, v194 /*v962*/
	v_exp_f32_e32 v201 /*v713*/, v195 /*v963*/
	v_exp_f32_e32 v203 /*v715*/, v196 /*v964*/
	s_set_vgpr_msb 0x83d3
	v_pk_fma_f32 v[194:195] /*v[962:963]*/, v[244:245] /*v[1012:1013]*/, s[56:57], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xd383
	v_exp_f32_e32 v209 /*v721*/, v197 /*v965*/
	v_nop
	s_set_vgpr_msb 0x83d3
	v_pk_fma_f32 v[196:197] /*v[964:965]*/, v[238:239] /*v[1006:1007]*/, s[56:57], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v17 /*v785*/, v192 /*v960*/
	v_exp_f32_e32 v33 /*v801*/, v193 /*v961*/
	v_nop
	v_pk_fma_f32 v[192:193] /*v[960:961]*/, v[248:249] /*v[1016:1017]*/, s[56:57], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xd380
	v_exp_f32_e32 v207 /*v719*/, v198
	s_set_vgpr_msb 0x8010
	v_pk_fma_f32 v[208:209], v[208:209], s[56:57], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1080
	v_exp_f32_e32 v215 /*v727*/, v199
	v_nop
	s_set_vgpr_msb 0x8013
	v_pk_fma_f32 v[198:199], v[250:251] /*v[1018:1019]*/, s[56:57], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1310
	v_pk_fma_f32 v[200:201], v[200:201], s[56:57], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x10c3
	v_exp_f32_e32 v168 /*v936*/, v94 /*v862*/
	v_exp_f32_e32 v180 /*v948*/, v110 /*v878*/
	v_exp_f32_e32 v94 /*v862*/, v116 /*v884*/
	v_exp_f32_e32 v110 /*v878*/, v117 /*v885*/
	v_exp_f32_e32 v116 /*v884*/, v130 /*v898*/
	v_exp_f32_e32 v130 /*v898*/, v131 /*v899*/
	v_pk_fma_f32 v[164:165] /*v[932:933]*/, v[226:227] /*v[994:995]*/, s[56:57], v[130:131] op_sel_hi:[1,0,0]
	v_exp_f32_e32 v146 /*v914*/, v135 /*v903*/
	v_pk_fma_f32 v[174:175] /*v[942:943]*/, v[228:229] /*v[996:997]*/, s[56:57], v[130:131] op_sel_hi:[1,0,0]
	v_exp_f32_e32 v160 /*v928*/, v151 /*v919*/
	v_pk_fma_f32 v[176:177] /*v[944:945]*/, v[230:231] /*v[998:999]*/, s[56:57], v[130:131] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xc3c0
	v_exp_f32_e32 v117 /*v885*/, v192
	v_exp_f32_e32 v131 /*v899*/, v193
	v_exp_f32_e32 v135 /*v903*/, v194
	s_set_vgpr_msb 0xc011
	v_pk_fma_f32 v[192:193], v[30:31] /*v[286:287]*/, s[56:57], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x11c0
	v_exp_f32_e32 v147 /*v915*/, v195
	v_exp_f32_e32 v151 /*v919*/, v196
	s_set_vgpr_msb 0xc011
	v_pk_fma_f32 v[194:195], v[40:41] /*v[296:297]*/, s[56:57], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x11c0
	v_exp_f32_e32 v161 /*v929*/, v197
	v_nop
	s_set_vgpr_msb 0xc011
	v_pk_fma_f32 v[196:197], v[36:37] /*v[292:293]*/, s[56:57], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1182
	v_exp_f32_e32 v196 /*v708*/, v196 /*v708*/
	v_exp_f32_e32 v252 /*v764*/, v204 /*v716*/
	s_set_vgpr_msb 0x82c2
	v_exp_f32_e32 v10 /*v778*/, v205 /*v717*/
	s_set_vgpr_msb 0xc282
	v_exp_f32_e32 v226 /*v738*/, v217 /*v729*/
	s_set_vgpr_msb 0x82c2
	v_exp_f32_e32 v26 /*v794*/, v220 /*v732*/
	s_set_vgpr_msb 0xc282
	v_exp_f32_e32 v238 /*v750*/, v238 /*v750*/
	s_set_vgpr_msb 0x8283
	v_exp_f32_e32 v223 /*v735*/, v194 /*v962*/
	v_exp_f32_e32 v233 /*v745*/, v195 /*v963*/
	v_exp_f32_e32 v237 /*v749*/, v196 /*v964*/
	s_set_vgpr_msb 0x83d3
	v_pk_fma_f32 v[194:195] /*v[962:963]*/, v[240:241] /*v[1008:1009]*/, s[56:57], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0xd383
	v_exp_f32_e32 v227 /*v739*/, v193 /*v961*/
	s_set_vgpr_msb 0x8380
	v_exp_f32_e32 v229 /*v741*/, v208
	s_set_vgpr_msb 0x80c0
	v_exp_f32_e32 v7 /*v775*/, v198
	v_exp_f32_e32 v23 /*v791*/, v199
	v_exp_f32_e32 v27 /*v795*/, v200
	s_set_vgpr_msb 0xc010
	v_pk_fma_f32 v[198:199], v[202:203], s[56:57], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x10c3
	v_exp_f32_e32 v172 /*v940*/, v165 /*v933*/
	v_pk_fma_f32 v[188:189] /*v[956:957]*/, v[242:243] /*v[1010:1011]*/, s[56:57], v[130:131] op_sel_hi:[1,0,0]
	v_exp_f32_e32 v182 /*v950*/, v175 /*v943*/
	v_exp_f32_e32 v184 /*v952*/, v176 /*v944*/
	v_exp_f32_e32 v176 /*v944*/, v177 /*v945*/
	s_set_vgpr_msb 0xc3c0
	v_exp_f32_e32 v165 /*v933*/, v192
	v_exp_f32_e32 v173 /*v941*/, v193
	v_exp_f32_e32 v175 /*v943*/, v194
	v_exp_f32_e32 v183 /*v951*/, v195
	s_set_vgpr_msb 0xc011
	v_pk_fma_f32 v[192:193], v[22:23] /*v[278:279]*/, s[56:57], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x11c0
	v_exp_f32_e32 v185 /*v953*/, v196
	s_set_vgpr_msb 0xc00a
	v_pk_add_f32 v[194:195], v[192:193] /*v[704:705]*/, v[194:195] /*v[706:707]*/
	s_set_vgpr_msb 0xac0
	v_exp_f32_e32 v177 /*v945*/, v197
	v_nop
	s_set_vgpr_msb 0xc00a
	v_pk_add_f32 v[196:197], v[200:201] /*v[712:713]*/, v[202:203] /*v[714:715]*/
	s_set_vgpr_msb 0xa82
	v_exp_f32_e32 v198 /*v710*/, v198 /*v710*/
	v_exp_f32_e32 v204 /*v716*/, v199 /*v711*/
	v_exp_f32_e32 v216 /*v728*/, v216 /*v728*/
	v_exp_f32_e32 v220 /*v732*/, v225 /*v737*/
	s_set_vgpr_msb 0x82c3
	v_exp_f32_e32 v72 /*v840*/, v72 /*v840*/
	v_exp_f32_e32 v104 /*v872*/, v104 /*v872*/
	v_exp_f32_e32 v150 /*v918*/, v150 /*v918*/
	v_exp_f32_e32 v164 /*v932*/, v164 /*v932*/
	s_set_vgpr_msb 0xc383
	v_exp_f32_e32 v249 /*v761*/, v197 /*v965*/
	v_exp_f32_e32 v199 /*v711*/, v194 /*v962*/
	v_exp_f32_e32 v217 /*v729*/, v192 /*v960*/
	s_set_vgpr_msb 0x8380
	v_exp_f32_e32 v241 /*v753*/, v209
	s_set_vgpr_msb 0x80c0
	v_exp_f32_e32 v43 /*v811*/, v201
	s_set_vgpr_msb 0xc080
	v_exp_f32_e32 v225 /*v737*/, v198
	s_set_vgpr_msb 0x80c3
	v_exp_f32_e32 v190 /*v958*/, v189 /*v957*/
	s_set_vgpr_msb 0xc380
	v_exp_f32_e32 v235 /*v747*/, v199
	s_set_vgpr_msb 0x80c0
	v_exp_f32_e32 v189 /*v957*/, v192
	v_exp_f32_e32 v191 /*v959*/, v193
	v_nop
	s_set_vgpr_msb 0xc002
	v_pk_add_f32 v[192:193], v[196:197] /*v[708:709]*/, v[194:195]
	s_set_vgpr_msb 0x20a
	v_pk_add_f32 v[194:195], v[210:211] /*v[722:723]*/, v[218:219] /*v[730:731]*/
	s_set_vgpr_msb 0xa02
	v_pk_add_f32 v[196:197], v[208:209] /*v[720:721]*/, v[196:197]
	s_set_vgpr_msb 0x20a
	v_pk_add_f32 v[198:199], v[232:233] /*v[744:745]*/, v[236:237] /*v[748:749]*/
	s_set_vgpr_msb 0xa0e
	v_pk_add_f32 v[200:201], v[252:253] /*v[764:765]*/, v[10:11] /*v[778:779]*/
	s_set_vgpr_msb 0xe0a
	v_pk_add_f32 v[204:205], v[206:207] /*v[718:719]*/, v[214:215] /*v[726:727]*/
	v_pk_add_f32 v[206:207], v[226:227] /*v[738:739]*/, v[228:229] /*v[740:741]*/
	s_set_vgpr_msb 0xa0f
	v_pk_add_f32 v[210:211], v[22:23] /*v[790:791]*/, v[26:27] /*v[794:795]*/
	v_pk_add_f32 v[212:213], v[48:49] /*v[816:817]*/, v[64:65] /*v[832:833]*/
	s_set_vgpr_msb 0xf0a
	v_pk_add_f32 v[216:217], v[238:239] /*v[750:751]*/, v[250:251] /*v[762:763]*/
	s_set_vgpr_msb 0xa0f
	v_pk_add_f32 v[218:219], v[12:13] /*v[780:781]*/, v[18:19] /*v[786:787]*/
	s_set_vgpr_msb 0xfc3
	v_exp_f32_e32 v8 /*v776*/, v8 /*v776*/
	v_exp_f32_e32 v40 /*v808*/, v40 /*v808*/
	v_exp_f32_e32 v174 /*v942*/, v174 /*v942*/
	s_set_vgpr_msb 0xc383
	v_exp_f32_e32 v205 /*v717*/, v195 /*v963*/
	s_set_vgpr_msb 0x830b
	v_pk_add_f32 v[202:203], v[32:33] /*v[800:801]*/, v[198:199] /*v[710:711]*/
	s_set_vgpr_msb 0xb02
	v_pk_add_f32 v[194:195], v[222:223] /*v[734:735]*/, v[194:195]
	v_pk_add_f32 v[198:199], v[248:249] /*v[760:761]*/, v[198:199]
	s_set_vgpr_msb 0x203
	v_pk_add_f32 v[200:201], v[16:17] /*v[784:785]*/, v[200:201]
	s_set_vgpr_msb 0x302
	v_pk_add_f32 v[204:205], v[216:217] /*v[728:729]*/, v[204:205]
	s_set_vgpr_msb 0x20e
	v_pk_add_f32 v[208:209], v[244:245] /*v[756:757]*/, v[2:3] /*v[770:771]*/
	s_set_vgpr_msb 0xe02
	v_pk_add_f32 v[206:207], v[240:241] /*v[752:753]*/, v[206:207]
	s_set_vgpr_msb 0x20a
	v_pk_add_f32 v[214:215], v[220:221] /*v[732:733]*/, v[224:225] /*v[736:737]*/
	s_set_vgpr_msb 0xa03
	v_pk_add_f32 v[210:211], v[42:43] /*v[810:811]*/, v[210:211]
	s_set_vgpr_msb 0x302
	v_pk_add_f32 v[212:213], v[212:213] /*v[724:725]*/, v[212:213]
	s_set_vgpr_msb 0x20f
	v_pk_add_f32 v[220:221], v[38:39] /*v[806:807]*/, v[54:55] /*v[822:823]*/
	v_pk_add_f32 v[222:223], v[74:75] /*v[842:843]*/, v[80:81] /*v[848:849]*/
	s_set_vgpr_msb 0xf02
	v_pk_add_f32 v[216:217], v[254:255] /*v[766:767]*/, v[216:217]
	s_set_vgpr_msb 0x20a
	v_pk_add_f32 v[224:225], v[230:231] /*v[742:743]*/, v[242:243] /*v[754:755]*/
	s_set_vgpr_msb 0xa03
	v_pk_add_f32 v[218:219], v[34:35] /*v[802:803]*/, v[218:219]
	s_set_vgpr_msb 0x30f
	v_pk_add_f32 v[228:229], v[28:29] /*v[796:797]*/, v[44:45] /*v[812:813]*/
	v_pk_add_f32 v[230:231], v[66:67] /*v[834:835]*/, v[70:71] /*v[838:839]*/
	v_pk_add_f32 v[234:235], v[126:127] /*v[894:895]*/, v[0:1] /*v[768:769]*/
	v_pk_add_f32 v[236:237], v[20:21] /*v[788:789]*/, v[36:37] /*v[804:805]*/
	v_pk_add_f32 v[246:247], v[46:47] /*v[814:815]*/, v[52:53] /*v[820:821]*/
	v_pk_add_f32 v[248:249], v[72:73] /*v[840:841]*/, v[88:89] /*v[856:857]*/
	v_pk_add_f32 v[252:253], v[132:133] /*v[900:901]*/, v[144:145] /*v[912:913]*/
	v_pk_add_f32 v[254:255], v[158:159] /*v[926:927]*/, v[162:163] /*v[930:931]*/
	s_set_vgpr_msb 0xf4f
	v_pk_add_f32 v[2:3] /*v[258:259]*/, v[100:101] /*v[868:869]*/, v[104:105] /*v[872:873]*/
	v_pk_add_f32 v[4:5] /*v[260:261]*/, v[124:125] /*v[892:893]*/, v[138:139] /*v[906:907]*/
	v_pk_add_f32 v[8:9] /*v[264:265]*/, v[168:169] /*v[936:937]*/, v[178:179] /*v[946:947]*/
	v_pk_add_f32 v[10:11] /*v[266:267]*/, v[186:187] /*v[954:955]*/, v[94:95] /*v[862:863]*/
	v_pk_add_f32 v[14:15] /*v[270:271]*/, v[146:147] /*v[914:915]*/, v[150:151] /*v[918:919]*/
	v_pk_add_f32 v[16:17] /*v[272:273]*/, v[164:165] /*v[932:933]*/, v[172:173] /*v[940:941]*/
	s_set_vgpr_msb 0x4f00
	v_pk_add_f32 v[192:193], v[192:193], v[196:197]
	s_set_vgpr_msb 0xc3
	v_exp_f32_e32 v134 /*v902*/, v134 /*v902*/
	s_set_vgpr_msb 0xc302
	v_pk_add_f32 v[202:203], v[204:205] /*v[716:717]*/, v[202:203]
	s_set_vgpr_msb 0x203
	v_pk_add_f32 v[208:209], v[6:7] /*v[774:775]*/, v[208:209]
	s_set_vgpr_msb 0x302
	v_pk_add_f32 v[214:215], v[234:235] /*v[746:747]*/, v[214:215]
	s_set_vgpr_msb 0x203
	v_pk_add_f32 v[220:221], v[58:59] /*v[826:827]*/, v[220:221]
	v_pk_add_f32 v[222:223], v[96:97] /*v[864:865]*/, v[222:223]
	s_set_vgpr_msb 0x30f
	v_pk_add_f32 v[226:227], v[4:5] /*v[772:773]*/, v[8:9] /*v[776:777]*/
	s_set_vgpr_msb 0xf02
	v_pk_add_f32 v[224:225], v[246:247] /*v[758:759]*/, v[224:225]
	s_set_vgpr_msb 0x20f
	v_pk_add_f32 v[232:233], v[90:91] /*v[858:859]*/, v[106:107] /*v[874:875]*/
	s_set_vgpr_msb 0xf03
	v_pk_add_f32 v[228:229], v[50:51] /*v[818:819]*/, v[228:229]
	v_pk_add_f32 v[230:231], v[86:87] /*v[854:855]*/, v[230:231]
	v_pk_add_f32 v[234:235], v[14:15] /*v[782:783]*/, v[234:235]
	s_set_vgpr_msb 0x30f
	v_pk_add_f32 v[238:239], v[56:57] /*v[824:825]*/, v[60:61] /*v[828:829]*/
	v_pk_add_f32 v[240:241], v[82:83] /*v[850:851]*/, v[98:99] /*v[866:867]*/
	v_pk_add_f32 v[242:243], v[118:119] /*v[886:887]*/, v[122:123] /*v[890:891]*/
	s_set_vgpr_msb 0xf03
	v_pk_add_f32 v[236:237], v[40:41] /*v[808:809]*/, v[236:237]
	s_set_vgpr_msb 0x30f
	v_pk_add_f32 v[250:251], v[108:109] /*v[876:877]*/, v[114:115] /*v[882:883]*/
	s_set_vgpr_msb 0xf03
	v_pk_add_f32 v[246:247], v[68:69] /*v[836:837]*/, v[246:247]
	v_pk_add_f32 v[248:249], v[92:93] /*v[860:861]*/, v[248:249]
	v_pk_add_f32 v[252:253], v[148:149] /*v[916:917]*/, v[252:253]
	s_set_vgpr_msb 0x34f
	v_pk_add_f32 v[0:1] /*v[256:257]*/, v[62:63] /*v[830:831]*/, v[78:79] /*v[846:847]*/
	s_set_vgpr_msb 0x4f03
	v_pk_add_f32 v[254:255], v[170:171] /*v[938:939]*/, v[254:255]
	s_set_vgpr_msb 0x34f
	v_pk_add_f32 v[6:7] /*v[262:263]*/, v[154:155] /*v[922:923]*/, v[156:157] /*v[924:925]*/
	s_set_vgpr_msb 0x4f47
	v_pk_add_f32 v[2:3] /*v[258:259]*/, v[120:121] /*v[888:889]*/, v[2:3] /*v[258:259]*/
	v_pk_add_f32 v[4:5] /*v[260:261]*/, v[142:143] /*v[910:911]*/, v[4:5] /*v[260:261]*/
	v_pk_add_f32 v[8:9] /*v[264:265]*/, v[180:181] /*v[948:949]*/, v[8:9] /*v[264:265]*/
	s_set_vgpr_msb 0x474f
	v_pk_add_f32 v[12:13] /*v[268:269]*/, v[116:117] /*v[884:885]*/, v[130:131] /*v[898:899]*/
	s_set_vgpr_msb 0x4f47
	v_pk_add_f32 v[10:11] /*v[266:267]*/, v[110:111] /*v[878:879]*/, v[10:11] /*v[266:267]*/
	s_set_vgpr_msb 0x474f
	v_pk_add_f32 v[18:19] /*v[274:275]*/, v[182:183] /*v[950:951]*/, v[184:185] /*v[952:953]*/
	s_set_vgpr_msb 0x4f47
	v_pk_add_f32 v[14:15] /*v[270:271]*/, v[160:161] /*v[928:929]*/, v[14:15] /*v[270:271]*/
	v_pk_add_f32 v[16:17] /*v[272:273]*/, v[174:175] /*v[942:943]*/, v[16:17] /*v[272:273]*/
	s_set_vgpr_msb 0x4700
	v_pk_add_f32 v[198:199], v[198:199], v[200:201]
	v_pk_add_f32 v[200:201], v[204:205], v[206:207]
	v_pk_add_f32 v[192:193], v[194:195], v[192:193]
	v_pk_add_f32 v[194:195], v[210:211], v[212:213]
	v_pk_add_f32 v[204:205], v[216:217], v[218:219]
	s_set_vgpr_msb 3
	v_pk_add_f32 v[226:227], v[24:25] /*v[792:793]*/, v[226:227]
	v_pk_add_f32 v[232:233], v[112:113] /*v[880:881]*/, v[232:233]
	s_set_vgpr_msb 0x30f
	v_pk_add_f32 v[244:245], v[140:141] /*v[908:909]*/, v[152:153] /*v[920:921]*/
	s_set_vgpr_msb 0xf03
	v_pk_add_f32 v[238:239], v[76:77] /*v[844:845]*/, v[238:239]
	v_pk_add_f32 v[240:241], v[102:103] /*v[870:871]*/, v[240:241]
	v_pk_add_f32 v[242:243], v[136:137] /*v[904:905]*/, v[242:243]
	v_pk_add_f32 v[250:251], v[128:129] /*v[896:897]*/, v[250:251]
	s_set_vgpr_msb 0x347
	v_pk_add_f32 v[0:1] /*v[256:257]*/, v[84:85] /*v[852:853]*/, v[0:1] /*v[256:257]*/
	v_pk_add_f32 v[6:7] /*v[262:263]*/, v[166:167] /*v[934:935]*/, v[6:7] /*v[262:263]*/
	v_pk_add_f32 v[12:13] /*v[268:269]*/, v[134:135] /*v[902:903]*/, v[12:13] /*v[268:269]*/
	s_set_vgpr_msb 0x4707
	v_pk_add_f32 v[196:197], v[176:177] /*v[944:945]*/, v[18:19] /*v[274:275]*/
	s_set_vgpr_msb 0x700
	v_pk_add_f32 v[198:199], v[202:203], v[198:199]
	v_pk_add_f32 v[200:201], v[208:209], v[200:201]
	v_pk_add_f32 v[202:203], v[222:223], v[224:225]
	v_pk_add_f32 v[194:195], v[214:215], v[194:195]
	v_pk_add_f32 v[204:205], v[220:221], v[204:205]
	v_pk_add_f32 v[206:207], v[228:229], v[230:231]
	v_pk_add_f32 v[208:209], v[234:235], v[236:237]
	v_pk_add_f32 v[212:213], v[246:247], v[248:249]
	v_pk_add_f32 v[214:215], v[252:253], v[254:255]
	s_set_vgpr_msb 5
	v_pk_add_f32 v[216:217], v[2:3] /*v[258:259]*/, v[4:5] /*v[260:261]*/
	v_pk_add_f32 v[218:219], v[8:9] /*v[264:265]*/, v[10:11] /*v[266:267]*/
	v_pk_add_f32 v[220:221], v[14:15] /*v[270:271]*/, v[16:17] /*v[272:273]*/
	s_set_vgpr_msb 0x5c3
	v_exp_f32_e32 v188 /*v956*/, v188 /*v956*/
	s_set_vgpr_msb 0xc303
	v_pk_add_f32 v[244:245], v[30:31] /*v[798:799]*/, v[244:245]
	s_set_vgpr_msb 0x300
	v_pk_add_f32 v[202:203], v[226:227], v[202:203]
	v_pk_add_f32 v[210:211], v[240:241], v[242:243]
	v_pk_add_f32 v[206:207], v[232:233], v[206:207]
	v_pk_add_f32 v[208:209], v[238:239], v[208:209]
	v_pk_add_f32 v[212:213], v[250:251], v[212:213]
	s_set_vgpr_msb 1
	v_pk_add_f32 v[214:215], v[0:1] /*v[256:257]*/, v[214:215]
	s_set_vgpr_msb 0x100
	v_pk_add_f32 v[192:193], v[192:193], v[198:199]
	s_set_vgpr_msb 1
	v_pk_add_f32 v[198:199], v[6:7] /*v[262:263]*/, v[216:217]
	v_pk_add_f32 v[216:217], v[12:13] /*v[268:269]*/, v[218:219]
	s_set_vgpr_msb 0x100
	v_pk_add_f32 v[194:195], v[194:195], v[204:205]
	v_pk_add_f32 v[196:197], v[196:197], v[220:221]
	s_set_vgpr_msb 0x4f
	v_pk_add_f32 v[18:19] /*v[274:275]*/, v[188:189] /*v[956:957]*/, v[190:191] /*v[958:959]*/
	s_set_vgpr_msb 0x4f00
	v_pk_add_f32 v[210:211], v[244:245], v[210:211]
	v_pk_add_f32 v[192:193], v[200:201], v[192:193]
	v_pk_add_f32 v[200:201], v[206:207], v[208:209]
	v_pk_add_f32 v[204:205], v[212:213], v[214:215]
	v_pk_add_f32 v[194:195], v[202:203], v[194:195]
	v_pk_add_f32 v[196:197], v[216:217], v[196:197]
	v_mul_f32_e32 v130, 0x3fb8aa3b, v128
	v_pk_add_f32 v[200:201], v[210:211], v[200:201]
	v_pk_add_f32 v[198:199], v[198:199], v[204:205]
	v_pk_add_f32 v[192:193], v[192:193], v[194:195]
	s_set_vgpr_msb 1
	v_pk_add_f32 v[194:195], v[18:19] /*v[274:275]*/, v[196:197]
	s_set_vgpr_msb 0x1c0
	v_exp_f32_e32 v194 /*v962*/, v130
	s_set_vgpr_msb 0xc000
	v_pk_add_f32 v[192:193], v[200:201], v[192:193]
	v_pk_add_f32 v[194:195], v[198:199], v[194:195]
	s_set_vgpr_msb 0xc0
	v_pk_add_f32 v[192:193] /*v[960:961]*/, v[194:195], v[192:193]
	s_set_vgpr_msb 0xc003
	v_dual_mov_b32 v128, v192 /*v960*/ :: v_dual_mov_b32 v192, v193 /*v961*/
	s_set_vgpr_msb 0x300
	v_permlanex16_b32 v128, v128, s82, 0xfedcba98
	v_permlanex16_b32 v192, v192, s82, 0xfedcba98
	s_cbranch_vccz .LBB0_20
	s_set_vgpr_msb 12
	v_pk_mul_f32 v[126:127], v[126:127], v[194:195] /*v[962:963]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[124:125], v[124:125], v[194:195] /*v[962:963]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[122:123], v[122:123], v[194:195] /*v[962:963]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[120:121], v[120:121], v[194:195] /*v[962:963]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[118:119], v[118:119], v[194:195] /*v[962:963]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[116:117], v[116:117], v[194:195] /*v[962:963]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[114:115], v[114:115], v[194:195] /*v[962:963]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[112:113], v[112:113], v[194:195] /*v[962:963]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[110:111], v[110:111], v[194:195] /*v[962:963]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[108:109], v[108:109], v[194:195] /*v[962:963]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[106:107], v[106:107], v[194:195] /*v[962:963]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[104:105], v[104:105], v[194:195] /*v[962:963]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[102:103], v[102:103], v[194:195] /*v[962:963]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[100:101], v[100:101], v[194:195] /*v[962:963]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[98:99], v[98:99], v[194:195] /*v[962:963]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[96:97], v[96:97], v[194:195] /*v[962:963]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[94:95], v[94:95], v[194:195] /*v[962:963]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[92:93], v[92:93], v[194:195] /*v[962:963]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[90:91], v[90:91], v[194:195] /*v[962:963]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[88:89], v[88:89], v[194:195] /*v[962:963]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[86:87], v[86:87], v[194:195] /*v[962:963]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[84:85], v[84:85], v[194:195] /*v[962:963]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[82:83], v[82:83], v[194:195] /*v[962:963]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[80:81], v[80:81], v[194:195] /*v[962:963]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[78:79], v[78:79], v[194:195] /*v[962:963]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[76:77], v[76:77], v[194:195] /*v[962:963]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[74:75], v[74:75], v[194:195] /*v[962:963]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[72:73], v[72:73], v[194:195] /*v[962:963]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[70:71], v[70:71], v[194:195] /*v[962:963]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[68:69], v[68:69], v[194:195] /*v[962:963]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[66:67], v[66:67], v[194:195] /*v[962:963]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[64:65], v[64:65], v[194:195] /*v[962:963]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0xc00
.LBB0_20:
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0xc0
	scratch_load_b32 v200 /*v968*/, off, off offset:8 nv
	s_set_vgpr_msb 0xc004
	v_sub_f32_e32 v130, v134, v51 /*v307*/
	s_and_b32 s2, s5, exec_lo
	s_cselect_b32 s2, 1, 0
	s_cmp_lg_u32 s2, 1
	s_set_vgpr_msb 0x400
	v_mul_f32_e32 v130, 0x3fb8aa3b, v130
	s_set_vgpr_msb 0xc0
	v_exp_f32_e32 v196 /*v964*/, v130
	s_set_vgpr_msb 0xc000
	s_cbranch_scc1 .LBB0_15
	v_nop
	s_set_vgpr_msb 12
	v_pk_mul_f32 v[62:63], v[62:63], v[196:197] /*v[964:965]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[60:61], v[60:61], v[196:197] /*v[964:965]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59], v[58:59], v[196:197] /*v[964:965]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[56:57], v[56:57], v[196:197] /*v[964:965]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[54:55], v[54:55], v[196:197] /*v[964:965]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[52:53], v[52:53], v[196:197] /*v[964:965]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[50:51], v[50:51], v[196:197] /*v[964:965]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[48:49], v[48:49], v[196:197] /*v[964:965]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[46:47], v[46:47], v[196:197] /*v[964:965]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[44:45], v[44:45], v[196:197] /*v[964:965]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[42:43], v[42:43], v[196:197] /*v[964:965]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[40:41], v[40:41], v[196:197] /*v[964:965]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39], v[38:39], v[196:197] /*v[964:965]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[36:37], v[36:37], v[196:197] /*v[964:965]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35], v[34:35], v[196:197] /*v[964:965]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[32:33], v[32:33], v[196:197] /*v[964:965]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[30:31], v[30:31], v[196:197] /*v[964:965]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[28:29], v[28:29], v[196:197] /*v[964:965]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[26:27], v[26:27], v[196:197] /*v[964:965]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25], v[24:25], v[196:197] /*v[964:965]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[22:23], v[22:23], v[196:197] /*v[964:965]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21], v[20:21], v[196:197] /*v[964:965]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[18:19], v[18:19], v[196:197] /*v[964:965]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[16:17], v[16:17], v[196:197] /*v[964:965]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[14:15], v[196:197] /*v[964:965]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13], v[12:13], v[196:197] /*v[964:965]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[10:11], v[196:197] /*v[964:965]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[8:9], v[196:197] /*v[964:965]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[6:7], v[196:197] /*v[964:965]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[4:5], v[196:197] /*v[964:965]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[2:3], v[2:3], v[196:197] /*v[964:965]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[0:1], v[0:1], v[196:197] /*v[964:965]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0xc00
	s_branch .LBB0_15
.LBB0_22:
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x3
	scratch_load_b32 v192, off, off offset:564 nv
	scratch_load_b32 v193, off, off offset:568 nv
	scratch_load_b32 v194, off, off offset:584 nv
	scratch_load_b32 v130, off, off offset:524 nv
.LBB0_23:
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x2
	scratch_load_b32 v154, off, off offset:576 nv
	scratch_load_b32 v155, off, off offset:580 nv
	scratch_load_b32 v128, off, off offset:492 nv
	s_branch .LBB0_46
.LBB0_24:
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
	s_lshl_b32 s53, s2, 6
	s_lshl_b32 s38, s2, 7
	s_mul_i32 s54, s2, 0x4400
	s_mul_i32 s55, s2, 0x4800
	s_sub_co_i32 s2, s92, s53
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
	s_mov_b32 s16, 64
	s_mov_b32 s5, s54
	s_bitset1_b32 s7, 31
	s_mul_u64 s[38:39], s[40:41], s[38:39]
	s_add_co_i32 s55, s55, 0x11000
	s_mov_b32 s20, 0xf510000
	s_mov_b32 s21, s13
	s_mov_b32 s23, s15
	s_mov_b32 s27, s19
	s_mov_b32 s24, s16
	s_mov_b32 s22, s14
	s_and_b32 s26, s41, 0xffff
	s_ashr_i32 s2, s90, 2
	s_add_co_i32 s11, s65, -1
	s_set_vgpr_msb 48
	s_wait_loadcnt 0x1
	v_and_or_b32 v64, v155, 7, v200 /*v968*/
	s_add_co_i32 s2, s11, s2
	s_add_nc_u64 s[40:41], s[68:69], s[72:73]
	s_max_i32 s2, s2, 0
	s_add_nc_u64 s[42:43], s[66:67], s[70:71]
	s_add_co_i32 s2, s2, 1
	v_mul_u32_u24_e32 v64, 0x120, v64
	s_mov_b32 s45, s19
	s_set_vgpr_msb 0x3000
	s_wait_loadcnt 0x0
	v_and_or_b32 v64, v128, 16, v64
	v_add_nc_u32_e32 v128, 0x11000, v64
	s_wait_tensorcnt 0x0
	s_wait_alu depctr_vm_vsrc(6)
	ds_load_b128 v[0:3], v130
	ds_load_b128 v[4:7], v130 offset:32
	ds_load_b128 v[8:11], v130 offset:64
	ds_load_b128 v[12:15], v130 offset:96
	ds_load_b128 v[16:19], v130 offset:128
	ds_load_b128 v[20:23], v130 offset:160
	ds_load_b128 v[24:27], v130 offset:192
	ds_load_b128 v[28:31], v130 offset:224
	ds_load_b128 v[32:35], v130 offset:4352
	ds_load_b128 v[36:39], v130 offset:4384
	ds_load_b128 v[40:43], v130 offset:4416
	ds_load_b128 v[44:47], v130 offset:4448
	ds_load_b128 v[48:51], v130 offset:4480
	ds_load_b128 v[52:55], v130 offset:4512
	ds_load_b128 v[56:59], v130 offset:4544
	ds_load_b128 v[60:63], v130 offset:4576
	tensor_load_to_lds s[4:7], s[12:19]
	s_add_nc_u64 s[6:7], s[38:39], s[74:75]
	s_mov_b32 s5, s55
	s_bitset1_b32 s7, 31
	s_wait_dscnt 0xe
	v_pk_mul_bf16 v7, v129, v7
	tensor_load_to_lds s[4:7], s[20:27]
	s_ashr_i32 s5, s2, 31
	v_pk_mul_bf16 v6, v129, v6
	v_pk_mul_bf16 v5, v129, v5
	v_pk_mul_bf16 v4, v129, v4
	v_pk_mul_bf16 v3, v129, v3
	v_pk_mul_bf16 v2, v129, v2
	v_pk_mul_bf16 v1, v129, v1
	v_pk_mul_bf16 v0, v129, v0
	s_lshr_b32 s5, s5, 24
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_store_b128 off, v[0:3], off offset:12 nv
	scratch_store_b128 off, v[4:7], off offset:28 nv
	s_add_co_i32 s5, s2, s5
	s_add_co_i32 s6, s10, -1
	s_and_b32 s7, s5, 0xffffff00
	s_ashr_i32 s5, s5, 8
	s_cmp_lg_u32 s2, s7
	s_wait_alu depctr_vm_vsrc(0)
	v_or_b32_e32 v130, 0x34000, v64
	s_cselect_b32 s7, -1, 0
	s_cmp_lt_i32 s2, 0
	s_wait_dscnt 0xc
	v_pk_mul_bf16 v143, v129, v15
	s_cselect_b32 s2, -1, 0
	v_pk_mul_bf16 v142, v129, v14
	s_and_b32 s2, s2, s7
	s_sub_co_ci_u32 s2, s5, 0
	v_pk_mul_bf16 v141, v129, v13
	s_min_i32 s2, s2, s6
	v_pk_mul_bf16 v140, v129, v12
	v_pk_mul_bf16 v139, v129, v11
	v_pk_mul_bf16 v138, v129, v10
	v_pk_mul_bf16 v137, v129, v9
	v_pk_mul_bf16 v136, v129, v8
	s_wait_dscnt 0xa
	v_pk_mul_bf16 v151, v129, v23
	v_pk_mul_bf16 v150, v129, v22
	v_pk_mul_bf16 v149, v129, v21
	v_pk_mul_bf16 v148, v129, v20
	v_pk_mul_bf16 v147, v129, v19
	v_pk_mul_bf16 v146, v129, v18
	v_pk_mul_bf16 v145, v129, v17
	v_pk_mul_bf16 v144, v129, v16
	s_wait_dscnt 0x8
	v_pk_mul_bf16 v159, v129, v31
	v_pk_mul_bf16 v158, v129, v30
	v_pk_mul_bf16 v157, v129, v29
	v_pk_mul_bf16 v156, v129, v28
	v_pk_mul_bf16 v155, v129, v27
	v_pk_mul_bf16 v154, v129, v26
	v_pk_mul_bf16 v153, v129, v25
	v_pk_mul_bf16 v152, v129, v24
	s_wait_dscnt 0x6
	v_pk_mul_bf16 v167, v129, v39
	v_pk_mul_bf16 v166, v129, v38
	v_pk_mul_bf16 v165, v129, v37
	v_pk_mul_bf16 v164, v129, v36
	v_pk_mul_bf16 v163, v129, v35
	v_pk_mul_bf16 v162, v129, v34
	v_pk_mul_bf16 v161, v129, v33
	v_pk_mul_bf16 v160, v129, v32
	s_wait_dscnt 0x4
	v_pk_mul_bf16 v175, v129, v47
	v_pk_mul_bf16 v174, v129, v46
	v_pk_mul_bf16 v173, v129, v45
	v_pk_mul_bf16 v172, v129, v44
	v_pk_mul_bf16 v171, v129, v43
	v_pk_mul_bf16 v170, v129, v42
	v_pk_mul_bf16 v169, v129, v41
	v_pk_mul_bf16 v168, v129, v40
	s_wait_dscnt 0x2
	v_pk_mul_bf16 v183, v129, v55
	v_pk_mul_bf16 v182, v129, v54
	v_pk_mul_bf16 v181, v129, v53
	v_pk_mul_bf16 v180, v129, v52
	v_pk_mul_bf16 v179, v129, v51
	v_pk_mul_bf16 v178, v129, v50
	v_pk_mul_bf16 v177, v129, v49
	v_pk_mul_bf16 v176, v129, v48
	s_wait_dscnt 0x0
	v_pk_mul_bf16 v191, v129, v63
	v_pk_mul_bf16 v190, v129, v62
	v_pk_mul_bf16 v189, v129, v61
	v_pk_mul_bf16 v188, v129, v60
	v_pk_mul_bf16 v187, v129, v59
	v_pk_mul_bf16 v186, v129, v58
	v_pk_mul_bf16 v185, v129, v57
	v_pk_mul_bf16 v184, v129, v56
	s_wait_xcnt 0x0
	v_mov_b32_e32 v0, 0
	s_max_i32 s44, s2, 0
	s_cmp_lt_i32 s2, 1
	s_wait_tensorcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	s_cbranch_scc1 .LBB0_33
	v_dual_mov_b32 v1, v0 :: v_dual_mov_b32 v2, v0
	v_dual_mov_b32 v3, v0 :: v_dual_mov_b32 v4, v0
	v_dual_mov_b32 v5, v0 :: v_dual_mov_b32 v6, v0
	v_dual_mov_b32 v7, v0 :: v_dual_mov_b32 v192, v130
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
	v_dual_mov_b32 v133, 0xf149f2ca :: v_dual_mov_b32 v134, 0xf149f2ca
	v_dual_mov_b32 v193, v135 :: v_dual_mov_b32 v129, v194
	v_dual_mov_b32 v204, v0 :: v_dual_mov_b32 v205, v0
	s_lshl_b64 s[46:47], s[44:45], 8
	s_add_co_i32 s56, s35, 0xffffff00
	s_add_co_i32 s48, s64, 0x100
	s_mov_b64 s[50:51], 0xffffffffffffff00
	s_mov_b32 s65, 0x76543210
	s_mov_b32 s52, 0x3fb8aa3b
	s_mov_b32 s66, 1
	s_branch .LBB0_27
.LBB0_26:
	s_set_vgpr_msb 0x4f
	v_cvt_pk_bf16_f32 v23 /*v279*/, v164 /*v932*/, v194 /*v962*/
	v_cvt_pk_bf16_f32 v22 /*v278*/, v148 /*v916*/, v162 /*v930*/
	v_cvt_pk_bf16_f32 v21 /*v277*/, v138 /*v906*/, v142 /*v910*/
	v_cvt_pk_bf16_f32 v20 /*v276*/, v130 /*v898*/, v134 /*v902*/
	v_cvt_pk_bf16_f32 v19 /*v275*/, v126 /*v894*/, v128 /*v896*/
	v_cvt_pk_bf16_f32 v18 /*v274*/, v122 /*v890*/, v124 /*v892*/
	v_cvt_pk_bf16_f32 v17 /*v273*/, v68 /*v836*/, v120 /*v888*/
	v_cvt_pk_bf16_f32 v16 /*v272*/, v64 /*v832*/, v66 /*v834*/
	v_cvt_pk_bf16_f32 v31 /*v287*/, v165 /*v933*/, v195 /*v963*/
	v_cvt_pk_bf16_f32 v30 /*v286*/, v149 /*v917*/, v163 /*v931*/
	v_cvt_pk_bf16_f32 v29 /*v285*/, v139 /*v907*/, v143 /*v911*/
	v_cvt_pk_bf16_f32 v28 /*v284*/, v131 /*v899*/, v135 /*v903*/
	v_cvt_pk_bf16_f32 v27 /*v283*/, v127 /*v895*/, v129 /*v897*/
	v_cvt_pk_bf16_f32 v26 /*v282*/, v123 /*v891*/, v125 /*v893*/
	v_cvt_pk_bf16_f32 v25 /*v281*/, v69 /*v837*/, v121 /*v889*/
	v_cvt_pk_bf16_f32 v24 /*v280*/, v65 /*v833*/, v67 /*v835*/
	s_set_vgpr_msb 0x4f06
	v_wmma_f32_16x16x32_bf16 v[120:127], v[184:191] /*v[696:703]*/, v[16:23] /*v[272:279]*/, v[120:127]
	s_set_vgpr_msb 0x64f
	v_cvt_pk_bf16_f32 v39 /*v295*/, v198 /*v966*/, v224 /*v992*/
	v_cvt_pk_bf16_f32 v38 /*v294*/, v174 /*v942*/, v196 /*v964*/
	v_cvt_pk_bf16_f32 v37 /*v293*/, v150 /*v918*/, v166 /*v934*/
	v_cvt_pk_bf16_f32 v36 /*v292*/, v140 /*v908*/, v144 /*v912*/
	v_cvt_pk_bf16_f32 v35 /*v291*/, v132 /*v900*/, v136 /*v904*/
	v_cvt_pk_bf16_f32 v34 /*v290*/, v82 /*v850*/, v88 /*v856*/
	v_cvt_pk_bf16_f32 v33 /*v289*/, v74 /*v842*/, v78 /*v846*/
	s_set_vgpr_msb 0x4f06
	v_wmma_f32_16x16x32_bf16 v[56:63], v[184:191] /*v[696:703]*/, v[24:31] /*v[280:287]*/, v[56:63]
	s_set_vgpr_msb 0x64f
	v_cvt_pk_bf16_f32 v32 /*v288*/, v70 /*v838*/, v72 /*v840*/
	v_cvt_pk_bf16_f32 v71 /*v327*/, v199 /*v967*/, v225 /*v993*/
	v_cvt_pk_bf16_f32 v70 /*v326*/, v175 /*v943*/, v197 /*v965*/
	v_cvt_pk_bf16_f32 v69 /*v325*/, v151 /*v919*/, v167 /*v935*/
	v_cvt_pk_bf16_f32 v68 /*v324*/, v141 /*v909*/, v145 /*v913*/
	v_cvt_pk_bf16_f32 v67 /*v323*/, v133 /*v901*/, v137 /*v905*/
	v_cvt_pk_bf16_f32 v66 /*v322*/, v83 /*v851*/, v89 /*v857*/
	v_cvt_pk_bf16_f32 v65 /*v321*/, v75 /*v843*/, v79 /*v847*/
	v_cvt_pk_bf16_f32 v64 /*v320*/, v71 /*v839*/, v73 /*v841*/
	s_set_vgpr_msb 0x4f06
	v_wmma_f32_16x16x32_bf16 v[120:127], v[176:183] /*v[688:695]*/, v[32:39] /*v[288:295]*/, v[120:127]
	s_set_vgpr_msb 0x64f
	v_cvt_pk_bf16_f32 v103 /*v359*/, v228 /*v996*/, v250 /*v1018*/
	v_cvt_pk_bf16_f32 v102 /*v358*/, v208 /*v976*/, v226 /*v994*/
	v_cvt_pk_bf16_f32 v101 /*v357*/, v178 /*v946*/, v200 /*v968*/
	v_cvt_pk_bf16_f32 v100 /*v356*/, v152 /*v920*/, v168 /*v936*/
	v_cvt_pk_bf16_f32 v99 /*v355*/, v102 /*v870*/, v146 /*v914*/
	v_cvt_pk_bf16_f32 v98 /*v354*/, v94 /*v862*/, v100 /*v868*/
	v_cvt_pk_bf16_f32 v97 /*v353*/, v84 /*v852*/, v90 /*v858*/
	s_set_vgpr_msb 0x4f06
	v_wmma_f32_16x16x32_bf16 v[56:63], v[176:183] /*v[688:695]*/, v[64:71] /*v[320:327]*/, v[56:63]
	s_set_vgpr_msb 0x64f
	v_cvt_pk_bf16_f32 v96 /*v352*/, v76 /*v844*/, v80 /*v848*/
	s_set_vgpr_msb 0x4f83
	v_cvt_pk_bf16_f32 v191 /*v703*/, v177 /*v945*/, v207
	s_set_vgpr_msb 0x838f
	v_cvt_pk_bf16_f32 v190 /*v702*/, v239 /*v1007*/, v253 /*v1021*/
	v_cvt_pk_bf16_f32 v189 /*v701*/, v211 /*v979*/, v231 /*v999*/
	v_cvt_pk_bf16_f32 v183 /*v695*/, v229 /*v997*/, v251 /*v1019*/
	v_cvt_pk_bf16_f32 v182 /*v694*/, v209 /*v977*/, v227 /*v995*/
	v_cvt_pk_bf16_f32 v181 /*v693*/, v179 /*v947*/, v201 /*v969*/
	v_cvt_pk_bf16_f32 v180 /*v692*/, v153 /*v921*/, v169 /*v937*/
	v_cvt_pk_bf16_f32 v179 /*v691*/, v103 /*v871*/, v147 /*v915*/
	v_cvt_pk_bf16_f32 v178 /*v690*/, v95 /*v863*/, v101 /*v869*/
	v_cvt_pk_bf16_f32 v177 /*v689*/, v85 /*v853*/, v91 /*v859*/
	v_cvt_pk_bf16_f32 v176 /*v688*/, v77 /*v845*/, v81 /*v849*/
	s_set_vgpr_msb 0x8f06
	v_wmma_f32_16x16x32_bf16 v[120:127], v[168:175] /*v[680:687]*/, v[96:103] /*v[352:359]*/, v[120:127]
	s_set_vgpr_msb 0x68f
	v_cvt_pk_bf16_f32 v188 /*v700*/, v181 /*v949*/, v203 /*v971*/
	v_cvt_pk_bf16_f32 v187 /*v699*/, v117 /*v885*/, v171 /*v939*/
	v_cvt_pk_bf16_f32 v186 /*v698*/, v109 /*v877*/, v115 /*v883*/
	v_cvt_pk_bf16_f32 v185 /*v697*/, v97 /*v865*/, v105 /*v873*/
	v_cvt_pk_bf16_f32 v184 /*v696*/, v87 /*v855*/, v93 /*v861*/
	s_set_vgpr_msb 0x8f80
	v_cvt_pk_bf16_f32 v205 /*v717*/, v211, v225
	v_cvt_pk_bf16_f32 v204 /*v716*/, v199, v209
	s_set_vgpr_msb 0x800a
	v_wmma_f32_16x16x32_bf16 v[56:63], v[168:175] /*v[680:687]*/, v[176:183] /*v[688:695]*/, v[56:63]
	s_set_vgpr_msb 0xa8f
	v_cvt_pk_bf16_f32 v203 /*v715*/, v241 /*v1009*/, v255 /*v1023*/
	v_cvt_pk_bf16_f32 v202 /*v714*/, v213 /*v981*/, v233 /*v1001*/
	v_cvt_pk_bf16_f32 v201 /*v713*/, v183 /*v951*/, v205 /*v973*/
	v_cvt_pk_bf16_f32 v200 /*v712*/, v157 /*v925*/, v173 /*v941*/
	s_set_vgpr_msb 0x8f83
	v_cvt_pk_bf16_f32 v175 /*v687*/, v176 /*v944*/, v206
	s_set_vgpr_msb 0x838f
	v_cvt_pk_bf16_f32 v174 /*v686*/, v238 /*v1006*/, v252 /*v1020*/
	v_cvt_pk_bf16_f32 v173 /*v685*/, v210 /*v978*/, v230 /*v998*/
	v_cvt_pk_bf16_f32 v172 /*v684*/, v180 /*v948*/, v202 /*v970*/
	v_cvt_pk_bf16_f32 v171 /*v683*/, v116 /*v884*/, v170 /*v938*/
	v_cvt_pk_bf16_f32 v170 /*v682*/, v108 /*v876*/, v114 /*v882*/
	v_cvt_pk_bf16_f32 v169 /*v681*/, v96 /*v864*/, v104 /*v872*/
	v_cvt_pk_bf16_f32 v168 /*v680*/, v86 /*v854*/, v92 /*v860*/
	s_set_vgpr_msb 0x8f0a
	v_wmma_f32_16x16x32_bf16 v[120:127], v[160:167] /*v[672:679]*/, v[168:175] /*v[680:687]*/, v[120:127]
	s_set_vgpr_msb 0xa8f
	v_cvt_pk_bf16_f32 v199 /*v711*/, v111 /*v879*/, v119 /*v887*/
	v_cvt_pk_bf16_f32 v198 /*v710*/, v99 /*v867*/, v107 /*v875*/
	s_set_vgpr_msb 0x8f80
	v_cvt_pk_bf16_f32 v213 /*v725*/, v229, v239
	v_cvt_pk_bf16_f32 v212 /*v724*/, v219, v227
	v_cvt_pk_bf16_f32 v211 /*v723*/, v201, v213
	s_set_vgpr_msb 0x8083
	v_cvt_pk_bf16_f32 v210 /*v722*/, v243 /*v1011*/, v193
	s_set_vgpr_msb 0x838f
	v_cvt_pk_bf16_f32 v209 /*v721*/, v215 /*v983*/, v235 /*v1003*/
	s_set_vgpr_msb 0x8f0a
	v_wmma_f32_16x16x32_bf16 v[56:63], v[160:167] /*v[672:679]*/, v[184:191] /*v[696:703]*/, v[56:63]
	s_set_vgpr_msb 0xa8f
	v_cvt_pk_bf16_f32 v208 /*v720*/, v189 /*v957*/, v207 /*v975*/
	v_cvt_pk_bf16_f32 v207 /*v719*/, v159 /*v927*/, v185 /*v953*/
	v_cvt_pk_bf16_f32 v206 /*v718*/, v113 /*v881*/, v155 /*v923*/
	s_set_vgpr_msb 0x8f00
	v_cvt_pk_bf16_f32 v211, v220, v230
	s_set_vgpr_msb 0x80
	v_cvt_pk_bf16_f32 v167 /*v679*/, v210, v224
	v_cvt_pk_bf16_f32 v166 /*v678*/, v198, v208
	s_set_vgpr_msb 0x808f
	v_cvt_pk_bf16_f32 v165 /*v677*/, v240 /*v1008*/, v254 /*v1022*/
	v_cvt_pk_bf16_f32 v164 /*v676*/, v212 /*v980*/, v232 /*v1000*/
	v_cvt_pk_bf16_f32 v163 /*v675*/, v182 /*v950*/, v204 /*v972*/
	v_cvt_pk_bf16_f32 v162 /*v674*/, v156 /*v924*/, v172 /*v940*/
	v_cvt_pk_bf16_f32 v161 /*v673*/, v110 /*v878*/, v118 /*v886*/
	v_cvt_pk_bf16_f32 v160 /*v672*/, v98 /*v866*/, v106 /*v874*/
	s_set_vgpr_msb 0x8f0a
	v_wmma_f32_16x16x32_bf16 v[120:127], v[152:159] /*v[664:671]*/, v[160:167] /*v[672:679]*/, v[120:127]
	s_set_vgpr_msb 0xa00
	v_cvt_pk_bf16_f32 v210, v202, v214
	v_cvt_pk_bf16_f32 v227, v204, v216
	v_cvt_pk_bf16_f32 v199, v205, v217
	v_cvt_pk_bf16_f32 v213, v242, v248
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v209, v244 /*v1012*/, v194
	s_set_vgpr_msb 0x30f
	v_cvt_pk_bf16_f32 v208, v220 /*v988*/, v236 /*v1004*/
	v_cvt_pk_bf16_f32 v207, v190 /*v958*/, v216 /*v984*/
	s_set_vgpr_msb 0xf0a
	v_wmma_f32_16x16x32_bf16 v[56:63], v[152:159] /*v[664:671]*/, v[198:205] /*v[710:717]*/, v[56:63]
	s_set_vgpr_msb 0xa0f
	v_cvt_pk_bf16_f32 v206, v160 /*v928*/, v186 /*v954*/
	s_set_vgpr_msb 0xf00
	v_cvt_pk_bf16_f32 v230, v246, v250
	v_cvt_pk_bf16_f32 v229, v236, v244
	s_set_vgpr_msb 15
	v_cvt_pk_bf16_f32 v225, v222 /*v990*/, v246 /*v1014*/
	s_set_vgpr_msb 0xf80
	v_cvt_pk_bf16_f32 v159 /*v671*/, v228, v238
	v_cvt_pk_bf16_f32 v158 /*v670*/, v218, v226
	v_cvt_pk_bf16_f32 v157 /*v669*/, v200, v212
	s_set_vgpr_msb 0x8083
	v_cvt_pk_bf16_f32 v156 /*v668*/, v242 /*v1010*/, v192
	s_set_vgpr_msb 0x838f
	v_cvt_pk_bf16_f32 v155 /*v667*/, v214 /*v982*/, v234 /*v1002*/
	v_cvt_pk_bf16_f32 v154 /*v666*/, v188 /*v956*/, v206 /*v974*/
	v_cvt_pk_bf16_f32 v153 /*v665*/, v158 /*v926*/, v184 /*v952*/
	v_cvt_pk_bf16_f32 v152 /*v664*/, v112 /*v880*/, v154 /*v922*/
	s_set_vgpr_msb 0x8f0a
	v_wmma_f32_16x16x32_bf16 v[120:127], v[144:151] /*v[656:663]*/, v[152:159] /*v[664:671]*/, v[120:127]
	s_set_vgpr_msb 0xa00
	v_cvt_pk_bf16_f32 v212, v234, v240
	v_cvt_pk_bf16_f32 v228, v222, v232
	s_set_vgpr_msb 3
	v_cvt_pk_bf16_f32 v226, v248 /*v1016*/, v196
	s_set_vgpr_msb 0x30f
	v_cvt_pk_bf16_f32 v224, v192 /*v960*/, v218 /*v986*/
	s_set_vgpr_msb 0xf00
	v_cvt_pk_bf16_f32 v202, v247, v251
	v_cvt_pk_bf16_f32 v201, v237, v245
	v_cvt_pk_bf16_f32 v200, v223, v233
	s_set_vgpr_msb 10
	v_wmma_f32_16x16x32_bf16 v[56:63], v[144:151] /*v[656:663]*/, v[206:213] /*v[718:725]*/, v[56:63]
	s_set_vgpr_msb 0xa03
	v_cvt_pk_bf16_f32 v198, v249 /*v1017*/, v197
	s_set_vgpr_msb 0x30f
	v_cvt_pk_bf16_f32 v197, v223 /*v991*/, v247 /*v1015*/
	v_cvt_pk_bf16_f32 v196, v193 /*v961*/, v219 /*v987*/
	s_wait_alu depctr_va_vdst(0)
	scratch_load_b64 v[192:193], off, off th:TH_LOAD_LU nv
	s_set_vgpr_msb 0xf00
	v_dual_mov_b32 v133, v131 :: v_dual_mov_b32 v134, v132
	s_set_vgpr_msb 0x80
	v_cvt_pk_bf16_f32 v149 /*v661*/, v221, v231
	v_cvt_pk_bf16_f32 v148 /*v660*/, v203, v215
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x8005
	s_clause 0x1
	scratch_load_b128 v[214:217], off, off offset:460 th:TH_LOAD_LU nv
	scratch_load_b128 v[218:221], off, off offset:476 th:TH_LOAD_LU nv
	v_wmma_f32_16x16x32_bf16 v[72:79], v[72:79] /*v[328:335]*/, v[16:23] /*v[272:279]*/, v[72:79]
	s_set_vgpr_msb 0x580
	v_cvt_pk_bf16_f32 v151 /*v663*/, v243, v249
	v_cvt_pk_bf16_f32 v150 /*v662*/, v235, v241
	s_set_vgpr_msb 0x8083
	v_cvt_pk_bf16_f32 v147 /*v659*/, v245 /*v1013*/, v195
	s_set_vgpr_msb 0x838f
	v_cvt_pk_bf16_f32 v146 /*v658*/, v221 /*v989*/, v237 /*v1005*/
	v_cvt_pk_bf16_f32 v145 /*v657*/, v191 /*v959*/, v217 /*v985*/
	v_cvt_pk_bf16_f32 v144 /*v656*/, v161 /*v929*/, v187 /*v955*/
	s_set_vgpr_msb 0x8f00
	v_cvt_pk_bf16_f32 v231, v252, v254
	s_set_vgpr_msb 5
	v_wmma_f32_16x16x32_bf16 v[8:15], v[72:79] /*v[328:335]*/, v[24:31] /*v[280:287]*/, v[8:15]
	s_set_vgpr_msb 0x500
	v_cvt_pk_bf16_f32 v203, v253, v255
	s_add_nc_u64 s[46:47], s[46:47], s[50:51]
	s_addk_co_i32 s56, 0xff00
	s_addk_co_i32 s48, 0x100
	s_add_co_i32 s66, s66, 1
	s_cmp_lg_u64 s[46:47], 0
	s_set_vgpr_msb 34
	s_wait_loadcnt 0x2
	v_pk_fma_f32 v[192:193], v[196:197] /*v[708:709]*/, v[192:193], v[194:195] /*v[706:707]*/
	s_set_vgpr_msb 0x2205
	v_wmma_f32_16x16x32_bf16 v[72:79], v[56:63] /*v[312:319]*/, v[32:39] /*v[288:295]*/, v[72:79]
	s_set_vgpr_msb 0x508
	v_pk_add_f32 v[204:205], v[192:193], v[192:193] /*v[704:705]*/
	v_dual_mov_b32 v192, v130 :: v_dual_mov_b32 v193, v135
	s_set_vgpr_msb 0x805
	v_wmma_f32_16x16x32_bf16 v[8:15], v[56:63] /*v[312:319]*/, v[64:71] /*v[320:327]*/, v[8:15]
	s_set_vgpr_msb 0x504
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[72:79], v[214:221], v[96:103] /*v[352:359]*/, v[72:79]
	s_set_vgpr_msb 0x408
	v_wmma_f32_16x16x32_bf16 v[8:15], v[214:221], v[176:183] /*v[688:695]*/, v[8:15]
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b128 v[214:217], off, off offset:428 th:TH_LOAD_LU nv
	scratch_load_b128 v[218:221], off, off offset:444 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0x806
	v_wmma_f32_16x16x32_bf16 v[112:119], v[104:111] /*v[616:623]*/, v[16:23] /*v[272:279]*/, v[112:119]
	v_wmma_f32_16x16x32_bf16 v[104:111], v[72:79] /*v[584:591]*/, v[16:23] /*v[272:279]*/, v[104:111]
	s_set_vgpr_msb 0x605
	v_wmma_f32_16x16x32_bf16 v[96:103], v[248:255] /*v[504:511]*/, v[16:23] /*v[272:279]*/, v[96:103]
	v_wmma_f32_16x16x32_bf16 v[88:95], v[200:207] /*v[456:463]*/, v[16:23] /*v[272:279]*/, v[88:95]
	v_wmma_f32_16x16x32_bf16 v[80:87], v[120:127] /*v[376:383]*/, v[16:23] /*v[272:279]*/, v[80:87]
	s_set_vgpr_msb 0x506
	v_wmma_f32_16x16x32_bf16 v[112:119], v[120:127] /*v[632:639]*/, v[32:39] /*v[288:295]*/, v[112:119]
	v_wmma_f32_16x16x32_bf16 v[104:111], v[56:63] /*v[568:575]*/, v[32:39] /*v[288:295]*/, v[104:111]
	s_set_vgpr_msb 0x605
	v_wmma_f32_16x16x32_bf16 v[96:103], v[240:247] /*v[496:503]*/, v[32:39] /*v[288:295]*/, v[96:103]
	v_wmma_f32_16x16x32_bf16 v[88:95], v[184:191] /*v[440:447]*/, v[32:39] /*v[288:295]*/, v[88:95]
	v_wmma_f32_16x16x32_bf16 v[80:87], v[112:119] /*v[368:375]*/, v[32:39] /*v[288:295]*/, v[80:87]
	s_set_vgpr_msb 0x506
	v_wmma_f32_16x16x32_bf16 v[112:119], v[112:119] /*v[624:631]*/, v[96:103] /*v[352:359]*/, v[112:119]
	v_wmma_f32_16x16x32_bf16 v[104:111], v[40:47] /*v[552:559]*/, v[96:103] /*v[352:359]*/, v[104:111]
	s_set_vgpr_msb 0x605
	v_wmma_f32_16x16x32_bf16 v[96:103], v[232:239] /*v[488:495]*/, v[96:103] /*v[352:359]*/, v[96:103]
	v_wmma_f32_16x16x32_bf16 v[88:95], v[168:175] /*v[424:431]*/, v[96:103] /*v[352:359]*/, v[88:95]
	v_wmma_f32_16x16x32_bf16 v[80:87], v[104:111] /*v[360:367]*/, v[96:103] /*v[352:359]*/, v[80:87]
	s_set_vgpr_msb 0x50a
	v_wmma_f32_16x16x32_bf16 v[112:119], v[96:103] /*v[608:615]*/, v[168:175] /*v[680:687]*/, v[112:119]
	v_wmma_f32_16x16x32_bf16 v[104:111], v[32:39] /*v[544:551]*/, v[168:175] /*v[680:687]*/, v[104:111]
	s_set_vgpr_msb 0xa09
	v_wmma_f32_16x16x32_bf16 v[96:103], v[224:231] /*v[480:487]*/, v[168:175] /*v[680:687]*/, v[96:103]
	v_wmma_f32_16x16x32_bf16 v[88:95], v[160:167] /*v[416:423]*/, v[168:175] /*v[680:687]*/, v[88:95]
	v_wmma_f32_16x16x32_bf16 v[80:87], v[0:7] /*v[256:263]*/, v[168:175] /*v[680:687]*/, v[80:87]
	s_set_vgpr_msb 0x90a
	v_wmma_f32_16x16x32_bf16 v[112:119], v[88:95] /*v[600:607]*/, v[160:167] /*v[672:679]*/, v[112:119]
	v_wmma_f32_16x16x32_bf16 v[104:111], v[24:31] /*v[536:543]*/, v[160:167] /*v[672:679]*/, v[104:111]
	s_set_vgpr_msb 0xa09
	v_wmma_f32_16x16x32_bf16 v[96:103], v[216:223] /*v[472:479]*/, v[160:167] /*v[672:679]*/, v[96:103]
	v_wmma_f32_16x16x32_bf16 v[88:95], v[152:159] /*v[408:415]*/, v[160:167] /*v[672:679]*/, v[88:95]
	s_set_vgpr_msb 0x908
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[72:79], v[214:221], v[168:175] /*v[680:687]*/, v[72:79]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[214:221], v[184:191] /*v[696:703]*/, v[8:15]
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b128 v[214:217], off, off offset:396 th:TH_LOAD_LU nv
	scratch_load_b128 v[218:221], off, off offset:412 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0x809
	v_wmma_f32_16x16x32_bf16 v[80:87], v[88:95] /*v[344:351]*/, v[160:167] /*v[672:679]*/, v[80:87]
	s_set_vgpr_msb 0x90a
	v_wmma_f32_16x16x32_bf16 v[112:119], v[80:87] /*v[592:599]*/, v[152:159] /*v[664:671]*/, v[112:119]
	v_wmma_f32_16x16x32_bf16 v[104:111], v[16:23] /*v[528:535]*/, v[152:159] /*v[664:671]*/, v[104:111]
	s_set_vgpr_msb 0xa09
	v_wmma_f32_16x16x32_bf16 v[96:103], v[208:215] /*v[464:471]*/, v[152:159] /*v[664:671]*/, v[96:103]
	v_wmma_f32_16x16x32_bf16 v[88:95], v[144:151] /*v[400:407]*/, v[152:159] /*v[664:671]*/, v[88:95]
	v_wmma_f32_16x16x32_bf16 v[80:87], v[80:87] /*v[336:343]*/, v[152:159] /*v[664:671]*/, v[80:87]
	s_set_vgpr_msb 0x902
	v_wmma_f32_16x16x32_bf16 v[120:127], v[136:143] /*v[648:655]*/, v[206:213], v[120:127]
	v_wmma_f32_16x16x32_bf16 v[112:119], v[64:71] /*v[576:583]*/, v[206:213], v[112:119]
	v_wmma_f32_16x16x32_bf16 v[104:111], v[8:15] /*v[520:527]*/, v[206:213], v[104:111]
	s_set_vgpr_msb 0x201
	v_wmma_f32_16x16x32_bf16 v[96:103], v[192:199] /*v[448:455]*/, v[206:213], v[96:103]
	v_wmma_f32_16x16x32_bf16 v[88:95], v[136:143] /*v[392:399]*/, v[206:213], v[88:95]
	v_wmma_f32_16x16x32_bf16 v[80:87], v[8:15] /*v[264:271]*/, v[206:213], v[80:87]
	s_set_vgpr_msb 0x106
	v_wmma_f32_16x16x32_bf16 v[48:55], v[104:111] /*v[616:623]*/, v[24:31] /*v[280:287]*/, v[48:55]
	v_wmma_f32_16x16x32_bf16 v[40:47], v[72:79] /*v[584:591]*/, v[24:31] /*v[280:287]*/, v[40:47]
	s_set_vgpr_msb 0x605
	v_wmma_f32_16x16x32_bf16 v[32:39], v[248:255] /*v[504:511]*/, v[24:31] /*v[280:287]*/, v[32:39]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[200:207] /*v[456:463]*/, v[24:31] /*v[280:287]*/, v[24:31]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[120:127] /*v[376:383]*/, v[24:31] /*v[280:287]*/, v[16:23]
	s_set_vgpr_msb 0x506
	v_wmma_f32_16x16x32_bf16 v[48:55], v[120:127] /*v[632:639]*/, v[64:71] /*v[320:327]*/, v[48:55]
	v_wmma_f32_16x16x32_bf16 v[40:47], v[56:63] /*v[568:575]*/, v[64:71] /*v[320:327]*/, v[40:47]
	s_set_vgpr_msb 0x605
	v_wmma_f32_16x16x32_bf16 v[32:39], v[240:247] /*v[496:503]*/, v[64:71] /*v[320:327]*/, v[32:39]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[184:191] /*v[440:447]*/, v[64:71] /*v[320:327]*/, v[24:31]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[112:119] /*v[368:375]*/, v[64:71] /*v[320:327]*/, v[16:23]
	s_set_vgpr_msb 0x50a
	v_wmma_f32_16x16x32_bf16 v[48:55], v[112:119] /*v[624:631]*/, v[176:183] /*v[688:695]*/, v[48:55]
	s_set_vgpr_msb 0xa08
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[72:79], v[214:221], v[160:167] /*v[672:679]*/, v[72:79]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[214:221], v[198:205] /*v[710:717]*/, v[8:15]
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b128 v[214:217], off, off offset:364 th:TH_LOAD_LU nv
	scratch_load_b128 v[218:221], off, off offset:380 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0x80a
	v_wmma_f32_16x16x32_bf16 v[40:47], v[40:47] /*v[552:559]*/, v[176:183] /*v[688:695]*/, v[40:47]
	s_set_vgpr_msb 0xa09
	v_wmma_f32_16x16x32_bf16 v[32:39], v[232:239] /*v[488:495]*/, v[176:183] /*v[688:695]*/, v[32:39]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[168:175] /*v[424:431]*/, v[176:183] /*v[688:695]*/, v[24:31]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[104:111] /*v[360:367]*/, v[176:183] /*v[688:695]*/, v[16:23]
	s_set_vgpr_msb 0x90a
	v_wmma_f32_16x16x32_bf16 v[48:55], v[96:103] /*v[608:615]*/, v[184:191] /*v[696:703]*/, v[48:55]
	v_wmma_f32_16x16x32_bf16 v[40:47], v[32:39] /*v[544:551]*/, v[184:191] /*v[696:703]*/, v[40:47]
	s_set_vgpr_msb 0xa09
	v_wmma_f32_16x16x32_bf16 v[32:39], v[224:231] /*v[480:487]*/, v[184:191] /*v[696:703]*/, v[32:39]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[160:167] /*v[416:423]*/, v[184:191] /*v[696:703]*/, v[24:31]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[0:7] /*v[256:263]*/, v[184:191] /*v[696:703]*/, v[16:23]
	s_set_vgpr_msb 0x90a
	v_wmma_f32_16x16x32_bf16 v[48:55], v[88:95] /*v[600:607]*/, v[198:205] /*v[710:717]*/, v[48:55]
	v_wmma_f32_16x16x32_bf16 v[40:47], v[24:31] /*v[536:543]*/, v[198:205] /*v[710:717]*/, v[40:47]
	s_set_vgpr_msb 0xa09
	v_wmma_f32_16x16x32_bf16 v[32:39], v[216:223] /*v[472:479]*/, v[198:205] /*v[710:717]*/, v[32:39]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[152:159] /*v[408:415]*/, v[198:205] /*v[710:717]*/, v[24:31]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[88:95] /*v[344:351]*/, v[198:205] /*v[710:717]*/, v[16:23]
	s_set_vgpr_msb 0x90a
	v_wmma_f32_16x16x32_bf16 v[48:55], v[80:87] /*v[592:599]*/, v[206:213] /*v[718:725]*/, v[48:55]
	v_wmma_f32_16x16x32_bf16 v[40:47], v[16:23] /*v[528:535]*/, v[206:213] /*v[718:725]*/, v[40:47]
	s_set_vgpr_msb 0xa09
	v_wmma_f32_16x16x32_bf16 v[32:39], v[208:215] /*v[464:471]*/, v[206:213] /*v[718:725]*/, v[32:39]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[144:151] /*v[400:407]*/, v[206:213] /*v[718:725]*/, v[24:31]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[80:87] /*v[336:343]*/, v[206:213] /*v[718:725]*/, v[16:23]
	s_set_vgpr_msb 0x90a
	v_wmma_f32_16x16x32_bf16 v[56:63], v[136:143] /*v[648:655]*/, v[144:151] /*v[656:663]*/, v[56:63]
	v_wmma_f32_16x16x32_bf16 v[48:55], v[64:71] /*v[576:583]*/, v[144:151] /*v[656:663]*/, v[48:55]
	v_wmma_f32_16x16x32_bf16 v[40:47], v[8:15] /*v[520:527]*/, v[144:151] /*v[656:663]*/, v[40:47]
	s_set_vgpr_msb 0xa08
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[72:79], v[214:221], v[152:159] /*v[664:671]*/, v[72:79]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[214:221], v[206:213] /*v[718:725]*/, v[8:15]
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b128 v[214:217], off, off offset:332 th:TH_LOAD_LU nv
	scratch_load_b128 v[218:221], off, off offset:348 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0x809
	v_wmma_f32_16x16x32_bf16 v[32:39], v[192:199] /*v[448:455]*/, v[144:151] /*v[656:663]*/, v[32:39]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[136:143] /*v[392:399]*/, v[144:151] /*v[656:663]*/, v[24:31]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[8:15] /*v[264:271]*/, v[144:151] /*v[656:663]*/, v[16:23]
	s_set_vgpr_msb 0x902
	v_wmma_f32_16x16x32_bf16 v[120:127], v[128:135] /*v[640:647]*/, v[224:231], v[120:127]
	v_wmma_f32_16x16x32_bf16 v[56:63], v[128:135] /*v[640:647]*/, v[196:203], v[56:63]
	v_wmma_f32_16x16x32_bf16 v[112:119], v[48:55] /*v[560:567]*/, v[224:231], v[112:119]
	v_wmma_f32_16x16x32_bf16 v[48:55], v[48:55] /*v[560:567]*/, v[196:203], v[48:55]
	v_wmma_f32_16x16x32_bf16 v[104:111], v[0:7] /*v[512:519]*/, v[224:231], v[104:111]
	v_wmma_f32_16x16x32_bf16 v[40:47], v[0:7] /*v[512:519]*/, v[196:203], v[40:47]
	s_set_vgpr_msb 0x201
	v_wmma_f32_16x16x32_bf16 v[96:103], v[176:183] /*v[432:439]*/, v[224:231], v[96:103]
	v_wmma_f32_16x16x32_bf16 v[32:39], v[176:183] /*v[432:439]*/, v[196:203], v[32:39]
	v_wmma_f32_16x16x32_bf16 v[88:95], v[128:135] /*v[384:391]*/, v[224:231], v[88:95]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[128:135] /*v[384:391]*/, v[196:203], v[24:31]
	v_wmma_f32_16x16x32_bf16 v[80:87], v[48:55] /*v[304:311]*/, v[224:231], v[80:87]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[48:55] /*v[304:311]*/, v[196:203], v[16:23]
	s_set_vgpr_msb 0x100
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[72:79], v[214:221], v[206:213], v[72:79]
	s_set_vgpr_msb 8
	v_wmma_f32_16x16x32_bf16 v[8:15], v[214:221], v[144:151] /*v[656:663]*/, v[8:15]
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b128 v[214:217], off, off offset:300 th:TH_LOAD_LU nv
	scratch_load_b128 v[218:221], off, off offset:316 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0x800
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[72:79], v[214:221], v[224:231], v[72:79]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[214:221], v[196:203], v[8:15]
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b128 v[214:217], off, off offset:268 th:TH_LOAD_LU nv
	scratch_load_b128 v[218:221], off, off offset:284 th:TH_LOAD_LU nv
	s_set_vgpr_msb 4
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[64:71], v[214:221], v[16:23] /*v[272:279]*/, v[64:71]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[214:221], v[24:31] /*v[280:287]*/, v[0:7]
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b128 v[214:217], off, off offset:236 th:TH_LOAD_LU nv
	scratch_load_b128 v[218:221], off, off offset:252 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[64:71], v[214:221], v[32:39] /*v[288:295]*/, v[64:71]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[214:221], v[64:71] /*v[320:327]*/, v[0:7]
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b128 v[214:217], off, off offset:204 th:TH_LOAD_LU nv
	scratch_load_b128 v[218:221], off, off offset:220 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[64:71], v[214:221], v[96:103] /*v[352:359]*/, v[64:71]
	s_set_vgpr_msb 0x408
	v_wmma_f32_16x16x32_bf16 v[0:7], v[214:221], v[176:183] /*v[688:695]*/, v[0:7]
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b128 v[214:217], off, off offset:172 th:TH_LOAD_LU nv
	scratch_load_b128 v[218:221], off, off offset:188 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[64:71], v[214:221], v[168:175] /*v[680:687]*/, v[64:71]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[214:221], v[184:191] /*v[696:703]*/, v[0:7]
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b128 v[214:217], off, off offset:140 th:TH_LOAD_LU nv
	scratch_load_b128 v[218:221], off, off offset:156 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[64:71], v[214:221], v[160:167] /*v[672:679]*/, v[64:71]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[214:221], v[198:205] /*v[710:717]*/, v[0:7]
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b128 v[214:217], off, off offset:108 th:TH_LOAD_LU nv
	scratch_load_b128 v[218:221], off, off offset:124 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[64:71], v[214:221], v[152:159] /*v[664:671]*/, v[64:71]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[214:221], v[206:213] /*v[718:725]*/, v[0:7]
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b128 v[214:217], off, off offset:76 th:TH_LOAD_LU nv
	scratch_load_b128 v[218:221], off, off offset:92 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0x800
	s_wait_loadcnt_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[64:71], v[214:221], v[206:213], v[64:71]
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b128 v[206:209], off, off offset:44 th:TH_LOAD_LU nv
	scratch_load_b128 v[210:213], off, off offset:60 th:TH_LOAD_LU nv
	s_set_vgpr_msb 8
	v_wmma_f32_16x16x32_bf16 v[0:7], v[214:221], v[144:151] /*v[656:663]*/, v[0:7]
	s_set_vgpr_msb 0x800
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[64:71], v[206:213], v[224:231], v[64:71]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[206:213], v[196:203], v[0:7]
	s_cbranch_scc0 .LBB0_35
.LBB0_27:
	v_dual_mov_b32 v135, v129 :: v_dual_mov_b32 v129, v193
	v_dual_mov_b32 v130, v128 :: v_dual_mov_b32 v128, v192
	s_wait_alu depctr_va_vdst(0)
	scratch_store_b64 off, v[204:205], off nv
	s_wait_tensorcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	s_set_vgpr_msb 0x80
	ds_load_b128 v[184:187] /*v[696:699]*/, v135
	ds_load_b128 v[188:191] /*v[700:703]*/, v135 offset:32
	ds_load_b128 v[176:179] /*v[688:691]*/, v135 offset:64
	ds_load_b128 v[180:183] /*v[692:695]*/, v135 offset:96
	ds_load_b128 v[168:171] /*v[680:683]*/, v135 offset:128
	ds_load_b128 v[172:175] /*v[684:687]*/, v135 offset:160
	ds_load_b128 v[160:163] /*v[672:675]*/, v135 offset:192
	ds_load_b128 v[164:167] /*v[676:679]*/, v135 offset:224
	ds_load_b128 v[152:155] /*v[664:667]*/, v135 offset:4352
	ds_load_b128 v[156:159] /*v[668:671]*/, v135 offset:4384
	ds_load_b128 v[144:147] /*v[656:659]*/, v135 offset:4416
	ds_load_b128 v[148:151] /*v[660:663]*/, v135 offset:4448
	ds_load_b128 v[136:139] /*v[648:651]*/, v135 offset:4480
	ds_load_b128 v[140:143] /*v[652:655]*/, v135 offset:4512
	ds_load_b128 v[128:131] /*v[640:643]*/, v135 offset:4544
	ds_load_b128 v[132:135] /*v[644:647]*/, v135 offset:4576
	ds_load_b128 v[120:123] /*v[632:635]*/, v135 offset:8704
	ds_load_b128 v[124:127] /*v[636:639]*/, v135 offset:8736
	ds_load_b128 v[112:115] /*v[624:627]*/, v135 offset:8768
	ds_load_b128 v[116:119] /*v[628:631]*/, v135 offset:8800
	ds_load_b128 v[104:107] /*v[616:619]*/, v135 offset:8832
	ds_load_b128 v[108:111] /*v[620:623]*/, v135 offset:8864
	ds_load_b128 v[96:99] /*v[608:611]*/, v135 offset:8896
	ds_load_b128 v[100:103] /*v[612:615]*/, v135 offset:8928
	ds_load_b128 v[88:91] /*v[600:603]*/, v135 offset:13056
	ds_load_b128 v[92:95] /*v[604:607]*/, v135 offset:13088
	ds_load_b128 v[80:83] /*v[592:595]*/, v135 offset:13120
	ds_load_b128 v[84:87] /*v[596:599]*/, v135 offset:13152
	ds_load_b128 v[72:75] /*v[584:587]*/, v135 offset:13184
	ds_load_b128 v[76:79] /*v[588:591]*/, v135 offset:13216
	ds_load_b128 v[64:67] /*v[576:579]*/, v135 offset:13248
	ds_load_b128 v[68:71] /*v[580:583]*/, v135 offset:13280
	ds_load_b128 v[56:59] /*v[568:571]*/, v135 offset:17408
	ds_load_b128 v[60:63] /*v[572:575]*/, v135 offset:17440
	ds_load_b128 v[48:51] /*v[560:563]*/, v135 offset:17472
	ds_load_b128 v[52:55] /*v[564:567]*/, v135 offset:17504
	ds_load_b128 v[40:43] /*v[552:555]*/, v135 offset:17536
	ds_load_b128 v[44:47] /*v[556:559]*/, v135 offset:17568
	ds_load_b128 v[32:35] /*v[544:547]*/, v135 offset:17600
	ds_load_b128 v[36:39] /*v[548:551]*/, v135 offset:17632
	ds_load_b128 v[24:27] /*v[536:539]*/, v135 offset:21760
	ds_load_b128 v[28:31] /*v[540:543]*/, v135 offset:21792
	ds_load_b128 v[16:19] /*v[528:531]*/, v135 offset:21824
	ds_load_b128 v[20:23] /*v[532:535]*/, v135 offset:21856
	ds_load_b128 v[8:11] /*v[520:523]*/, v135 offset:21888
	ds_load_b128 v[12:15] /*v[524:527]*/, v135 offset:21920
	ds_load_b128 v[0:3] /*v[512:515]*/, v135 offset:21952
	ds_load_b128 v[4:7] /*v[516:519]*/, v135 offset:21984
	s_set_vgpr_msb 0x8040
	ds_load_b128 v[248:251] /*v[504:507]*/, v135 offset:26112
	ds_load_b128 v[252:255] /*v[508:511]*/, v135 offset:26144
	ds_load_b128 v[240:243] /*v[496:499]*/, v135 offset:26176
	ds_load_b128 v[244:247] /*v[500:503]*/, v135 offset:26208
	ds_load_b128 v[232:235] /*v[488:491]*/, v135 offset:26240
	ds_load_b128 v[236:239] /*v[492:495]*/, v135 offset:26272
	ds_load_b128 v[224:227] /*v[480:483]*/, v135 offset:26304
	ds_load_b128 v[228:231] /*v[484:487]*/, v135 offset:26336
	ds_load_b128 v[216:219] /*v[472:475]*/, v135 offset:30464
	ds_load_b128 v[220:223] /*v[476:479]*/, v135 offset:30496
	ds_load_b128 v[208:211] /*v[464:467]*/, v135 offset:30528
	ds_load_b128 v[212:215] /*v[468:471]*/, v135 offset:30560
	ds_load_b128 v[200:203] /*v[456:459]*/, v135 offset:30592
	ds_load_b128 v[204:207] /*v[460:463]*/, v135 offset:30624
	ds_load_b128 v[192:195] /*v[448:451]*/, v135 offset:30656
	ds_load_b128 v[196:199] /*v[452:455]*/, v135 offset:30688
	ds_load_b128 v[184:187] /*v[440:443]*/, v135 offset:34816
	ds_load_b128 v[188:191] /*v[444:447]*/, v135 offset:34848
	ds_load_b128 v[176:179] /*v[432:435]*/, v135 offset:34880
	ds_load_b128 v[180:183] /*v[436:439]*/, v135 offset:34912
	ds_load_b128 v[168:171] /*v[424:427]*/, v135 offset:34944
	ds_load_b128 v[172:175] /*v[428:431]*/, v135 offset:34976
	ds_load_b128 v[160:163] /*v[416:419]*/, v135 offset:35008
	ds_load_b128 v[164:167] /*v[420:423]*/, v135 offset:35040
	ds_load_b128 v[152:155] /*v[408:411]*/, v135 offset:39168
	ds_load_b128 v[156:159] /*v[412:415]*/, v135 offset:39200
	ds_load_b128 v[144:147] /*v[400:403]*/, v135 offset:39232
	ds_load_b128 v[148:151] /*v[404:407]*/, v135 offset:39264
	ds_load_b128 v[136:139] /*v[392:395]*/, v135 offset:39296
	ds_load_b128 v[140:143] /*v[396:399]*/, v135 offset:39328
	ds_load_b128 v[128:131] /*v[384:387]*/, v135 offset:39360
	ds_load_b128 v[132:135] /*v[388:391]*/, v135 offset:39392
	ds_load_b128 v[120:123] /*v[376:379]*/, v135 offset:43520
	ds_load_b128 v[124:127] /*v[380:383]*/, v135 offset:43552
	ds_load_b128 v[112:115] /*v[368:371]*/, v135 offset:43584
	ds_load_b128 v[116:119] /*v[372:375]*/, v135 offset:43616
	ds_load_b128 v[104:107] /*v[360:363]*/, v135 offset:43648
	ds_load_b128 v[108:111] /*v[364:367]*/, v135 offset:43680
	ds_load_b128 v[96:99] /*v[352:355]*/, v135 offset:43712
	ds_load_b128 v[100:103] /*v[356:359]*/, v135 offset:43744
	ds_load_b128 v[88:91] /*v[344:347]*/, v135 offset:47872
	ds_load_b128 v[92:95] /*v[348:351]*/, v135 offset:47904
	ds_load_b128 v[80:83] /*v[336:339]*/, v135 offset:47936
	ds_load_b128 v[84:87] /*v[340:343]*/, v135 offset:47968
	ds_load_b128 v[72:75] /*v[328:331]*/, v135 offset:48000
	ds_load_b128 v[76:79] /*v[332:335]*/, v135 offset:48032
	ds_load_b128 v[64:67] /*v[320:323]*/, v135 offset:48064
	ds_load_b128 v[68:71] /*v[324:327]*/, v135 offset:48096
	ds_load_b128 v[56:59] /*v[312:315]*/, v135 offset:52224
	ds_load_b128 v[60:63] /*v[316:319]*/, v135 offset:52256
	ds_load_b128 v[48:51] /*v[304:307]*/, v135 offset:52288
	ds_load_b128 v[52:55] /*v[308:311]*/, v135 offset:52320
	ds_load_b128 v[40:43] /*v[296:299]*/, v135 offset:52352
	ds_load_b128 v[44:47] /*v[300:303]*/, v135 offset:52384
	ds_load_b128 v[32:35] /*v[288:291]*/, v135 offset:52416
	ds_load_b128 v[36:39] /*v[292:295]*/, v135 offset:52448
	ds_load_b128 v[24:27] /*v[280:283]*/, v135 offset:56576
	ds_load_b128 v[28:31] /*v[284:287]*/, v135 offset:56608
	ds_load_b128 v[16:19] /*v[272:275]*/, v135 offset:56640
	ds_load_b128 v[20:23] /*v[276:279]*/, v135 offset:56672
	ds_load_b128 v[8:11] /*v[264:267]*/, v135 offset:56704
	ds_load_b128 v[12:15] /*v[268:271]*/, v135 offset:56736
	ds_load_b128 v[0:3] /*v[256:259]*/, v135 offset:56768
	ds_load_b128 v[4:7] /*v[260:263]*/, v135 offset:56800
	s_set_vgpr_msb 0x4000
	ds_load_b128 v[248:251], v135 offset:60928
	ds_load_b128 v[252:255], v135 offset:60960
	ds_load_b128 v[240:243], v135 offset:60992
	ds_load_b128 v[244:247], v135 offset:61024
	ds_load_b128 v[232:235], v135 offset:61056
	ds_load_b128 v[236:239], v135 offset:61088
	ds_load_b128 v[224:227], v135 offset:61120
	ds_load_b128 v[228:231], v135 offset:61152
	ds_load_b128 v[216:219], v135 offset:65280
	ds_load_b128 v[220:223], v135 offset:65312
	ds_load_b128 v[208:211], v135 offset:65344
	ds_load_b128 v[212:215], v135 offset:65376
	ds_load_b128 v[200:203], v135 offset:65408
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_b128 v[204:207], v135 offset:65440
	ds_load_b128 v[192:195], v135 offset:65472
	ds_load_b128 v[196:199], v135 offset:65504
	s_cmp_ge_i32 s66, s10
	s_cbranch_scc1 .LBB0_29
	v_med3_i32 v131, s56, 0, 0x100
	s_ashr_i32 s49, s48, 31
	s_lshr_b32 s2, s66, 31
	s_mul_u64 s[6:7], s[48:49], s[62:63]
	s_add_co_i32 s2, s66, s2
	v_readfirstlane_b32 s5, v131
	s_lshl_b64 s[6:7], s[6:7], 1
	s_and_b32 s2, s2, 0xffffe
	s_add_nc_u64 s[22:23], s[40:41], s[6:7]
	s_mul_u64 s[6:7], s[48:49], s[60:61]
	s_sub_co_i32 s5, s5, s53
	s_lshl_b64 s[6:7], s[6:7], 1
	s_sub_co_i32 s2, s66, s2
	s_add_nc_u64 s[6:7], s[42:43], s[6:7]
	s_max_i32 s14, s5, 0
	s_mul_i32 s2, s2, 0x23000
	s_add_nc_u64 s[6:7], s[36:37], s[6:7]
	s_lshl_b32 s14, s14, 16
	s_add_co_i32 s5, s54, s2
	s_bitset1_b32 s7, 31
	s_addk_co_i32 s14, 0x7fff
	s_mov_b32 s21, s13
	tensor_load_to_lds s[4:7], s[12:19]
	s_add_nc_u64 s[6:7], s[38:39], s[22:23]
	s_add_co_i32 s5, s55, s2
	s_bitset1_b32 s7, 31
	s_mov_b32 s22, s14
	s_mov_b32 s23, s15
	s_mov_b32 s24, s16
	s_mov_b32 s27, s19
	tensor_load_to_lds s[4:7], s[20:27]
.LBB0_29:
	s_set_vgpr_msb 0xf1
	s_clause 0x1
	scratch_load_b128 v[56:59] /*v[824:827]*/, off, off offset:12 nv
	scratch_load_b128 v[60:63] /*v[828:831]*/, off, off offset:28 nv
	s_wait_dscnt 0x2e
	v_wmma_f32_16x16x32_bf16 v[16:23] /*v[784:791]*/, v[120:127] /*v[376:383]*/, v[160:167], 0
	s_wait_dscnt 0x2c
	v_wmma_f32_16x16x32_bf16 v[16:23] /*v[784:791]*/, v[112:119] /*v[368:375]*/, v[168:175], v[16:23] /*v[784:791]*/
	s_wait_dscnt 0x2a
	v_wmma_f32_16x16x32_bf16 v[16:23] /*v[784:791]*/, v[104:111] /*v[360:367]*/, v[176:183], v[16:23] /*v[784:791]*/
	s_wait_dscnt 0x28
	v_wmma_f32_16x16x32_bf16 v[16:23] /*v[784:791]*/, v[96:103] /*v[352:359]*/, v[184:191], v[16:23] /*v[784:791]*/
	s_wait_dscnt 0x26
	v_wmma_f32_16x16x32_bf16 v[24:31] /*v[792:799]*/, v[88:95] /*v[344:351]*/, v[160:167], 0
	s_wait_dscnt 0x24
	v_wmma_f32_16x16x32_bf16 v[24:31] /*v[792:799]*/, v[80:87] /*v[336:343]*/, v[168:175], v[24:31] /*v[792:799]*/
	s_wait_dscnt 0x22
	v_wmma_f32_16x16x32_bf16 v[24:31] /*v[792:799]*/, v[72:79] /*v[328:335]*/, v[176:183], v[24:31] /*v[792:799]*/
	s_wait_dscnt 0x20
	v_wmma_f32_16x16x32_bf16 v[24:31] /*v[792:799]*/, v[64:71] /*v[320:327]*/, v[184:191], v[24:31] /*v[792:799]*/
	s_wait_dscnt 0x1e
	v_wmma_f32_16x16x32_bf16 v[32:39] /*v[800:807]*/, v[56:63] /*v[312:319]*/, v[160:167], 0
	s_wait_dscnt 0x1c
	v_wmma_f32_16x16x32_bf16 v[32:39] /*v[800:807]*/, v[48:55] /*v[304:311]*/, v[168:175], v[32:39] /*v[800:807]*/
	s_wait_dscnt 0x1a
	v_wmma_f32_16x16x32_bf16 v[32:39] /*v[800:807]*/, v[40:47] /*v[296:303]*/, v[176:183], v[32:39] /*v[800:807]*/
	s_set_vgpr_msb 0xf182
	v_wmma_f32_16x16x32_bf16 v[192:199] /*v[704:711]*/, v[184:191] /*v[696:703]*/, v[160:167], 0
	s_set_vgpr_msb 0x82f1
	s_wait_dscnt 0x18
	v_wmma_f32_16x16x32_bf16 v[32:39] /*v[800:807]*/, v[32:39] /*v[288:295]*/, v[184:191], v[32:39] /*v[800:807]*/
	s_wait_dscnt 0x16
	v_wmma_f32_16x16x32_bf16 v[40:47] /*v[808:815]*/, v[24:31] /*v[280:287]*/, v[160:167], 0
	s_set_vgpr_msb 0xf1a2
	v_wmma_f32_16x16x32_bf16 v[192:199] /*v[704:711]*/, v[176:183] /*v[688:695]*/, v[168:175], v[192:199] /*v[704:711]*/
	v_wmma_f32_16x16x32_bf16 v[200:207] /*v[712:719]*/, v[152:159] /*v[664:671]*/, v[160:167], 0
	v_wmma_f32_16x16x32_bf16 v[208:215] /*v[720:727]*/, v[120:127] /*v[632:639]*/, v[160:167], 0
	v_wmma_f32_16x16x32_bf16 v[216:223] /*v[728:735]*/, v[88:95] /*v[600:607]*/, v[160:167], 0
	v_wmma_f32_16x16x32_bf16 v[224:231] /*v[736:743]*/, v[56:63] /*v[568:575]*/, v[160:167], 0
	v_wmma_f32_16x16x32_bf16 v[232:239] /*v[744:751]*/, v[24:31] /*v[536:543]*/, v[160:167], 0
	s_set_vgpr_msb 0xa281
	v_wmma_f32_16x16x32_bf16 v[240:247] /*v[752:759]*/, v[248:255] /*v[504:511]*/, v[160:167], 0
	v_wmma_f32_16x16x32_bf16 v[248:255] /*v[760:767]*/, v[216:223] /*v[472:479]*/, v[160:167], 0
	s_set_vgpr_msb 0x81f1
	v_wmma_f32_16x16x32_bf16 v[0:7] /*v[768:775]*/, v[184:191] /*v[440:447]*/, v[160:167], 0
	v_wmma_f32_16x16x32_bf16 v[8:15] /*v[776:783]*/, v[152:159] /*v[408:415]*/, v[160:167], 0
	s_wait_dscnt 0x14
	v_wmma_f32_16x16x32_bf16 v[40:47] /*v[808:815]*/, v[16:23] /*v[272:279]*/, v[168:175], v[40:47] /*v[808:815]*/
	s_set_vgpr_msb 0xf1c0
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x32_bf16 v[48:55] /*v[816:823]*/, v[248:255], v[160:167], 0
	s_set_vgpr_msb 0xc0a2
	v_wmma_f32_16x16x32_bf16 v[192:199] /*v[704:711]*/, v[168:175] /*v[680:687]*/, v[176:183], v[192:199] /*v[704:711]*/
	v_wmma_f32_16x16x32_bf16 v[200:207] /*v[712:719]*/, v[144:151] /*v[656:663]*/, v[168:175], v[200:207] /*v[712:719]*/
	v_wmma_f32_16x16x32_bf16 v[208:215] /*v[720:727]*/, v[112:119] /*v[624:631]*/, v[168:175], v[208:215] /*v[720:727]*/
	v_wmma_f32_16x16x32_bf16 v[216:223] /*v[728:735]*/, v[80:87] /*v[592:599]*/, v[168:175], v[216:223] /*v[728:735]*/
	v_wmma_f32_16x16x32_bf16 v[224:231] /*v[736:743]*/, v[48:55] /*v[560:567]*/, v[168:175], v[224:231] /*v[736:743]*/
	v_wmma_f32_16x16x32_bf16 v[232:239] /*v[744:751]*/, v[16:23] /*v[528:535]*/, v[168:175], v[232:239] /*v[744:751]*/
	s_set_vgpr_msb 0xa2a1
	v_wmma_f32_16x16x32_bf16 v[240:247] /*v[752:759]*/, v[240:247] /*v[496:503]*/, v[168:175], v[240:247] /*v[752:759]*/
	v_wmma_f32_16x16x32_bf16 v[248:255] /*v[760:767]*/, v[208:215] /*v[464:471]*/, v[168:175], v[248:255] /*v[760:767]*/
	s_set_vgpr_msb 0xa1f1
	v_wmma_f32_16x16x32_bf16 v[0:7] /*v[768:775]*/, v[176:183] /*v[432:439]*/, v[168:175], v[0:7] /*v[768:775]*/
	v_wmma_f32_16x16x32_bf16 v[8:15] /*v[776:783]*/, v[144:151] /*v[400:407]*/, v[168:175], v[8:15] /*v[776:783]*/
	s_set_vgpr_msb 0xf1f0
	s_wait_dscnt 0xc
	v_wmma_f32_16x16x32_bf16 v[48:55] /*v[816:823]*/, v[240:247], v[168:175], v[48:55] /*v[816:823]*/
	s_set_vgpr_msb 0xf0a2
	v_wmma_f32_16x16x32_bf16 v[192:199] /*v[704:711]*/, v[160:167] /*v[672:679]*/, v[184:191], v[192:199] /*v[704:711]*/
	v_wmma_f32_16x16x32_bf16 v[200:207] /*v[712:719]*/, v[136:143] /*v[648:655]*/, v[176:183], v[200:207] /*v[712:719]*/
	v_wmma_f32_16x16x32_bf16 v[208:215] /*v[720:727]*/, v[104:111] /*v[616:623]*/, v[176:183], v[208:215] /*v[720:727]*/
	s_set_vgpr_msb 0xa2cd
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[242:249] /*v[1010:1017]*/, v[120:127] /*v[376:383]*/, v[56:63] /*v[824:831]*/, 0
	s_set_vgpr_msb 0xcdf1
	v_wmma_f32_16x16x32_bf16 v[242:249] /*v[1010:1017]*/, v[112:119] /*v[368:375]*/, v[136:143], v[242:249] /*v[1010:1017]*/
	v_wmma_f32_16x16x32_bf16 v[242:249] /*v[1010:1017]*/, v[104:111] /*v[360:367]*/, v[144:151], v[242:249] /*v[1010:1017]*/
	v_wmma_f32_16x16x32_bf16 v[242:249] /*v[1010:1017]*/, v[96:103] /*v[352:359]*/, v[152:159], v[242:249] /*v[1010:1017]*/
	s_set_vgpr_msb 0xf14d
	v_wmma_f32_16x16x32_bf16 v[96:103] /*v[352:359]*/, v[88:95] /*v[344:351]*/, v[56:63] /*v[824:831]*/, 0
	s_set_vgpr_msb 0x4d51
	v_wmma_f32_16x16x32_bf16 v[96:103] /*v[352:359]*/, v[80:87] /*v[336:343]*/, v[136:143], v[96:103] /*v[352:359]*/
	v_wmma_f32_16x16x32_bf16 v[96:103] /*v[352:359]*/, v[72:79] /*v[328:335]*/, v[144:151], v[96:103] /*v[352:359]*/
	v_wmma_f32_16x16x32_bf16 v[96:103] /*v[352:359]*/, v[64:71] /*v[320:327]*/, v[152:159], v[96:103] /*v[352:359]*/
	s_set_vgpr_msb 0x514d
	v_wmma_f32_16x16x32_bf16 v[64:71] /*v[320:327]*/, v[56:63] /*v[312:319]*/, v[56:63] /*v[824:831]*/, 0
	s_set_vgpr_msb 0x4d51
	v_wmma_f32_16x16x32_bf16 v[64:71] /*v[320:327]*/, v[48:55] /*v[304:311]*/, v[136:143], v[64:71] /*v[320:327]*/
	v_wmma_f32_16x16x32_bf16 v[64:71] /*v[320:327]*/, v[40:47] /*v[296:303]*/, v[144:151], v[64:71] /*v[320:327]*/
	s_set_vgpr_msb 0x51ce
	v_wmma_f32_16x16x32_bf16 v[64:71] /*v[832:839]*/, v[184:191] /*v[696:703]*/, v[56:63] /*v[824:831]*/, 0
	v_wmma_f32_16x16x32_bf16 v[72:79] /*v[840:847]*/, v[152:159] /*v[664:671]*/, v[56:63] /*v[824:831]*/, 0
	v_wmma_f32_16x16x32_bf16 v[80:87] /*v[848:855]*/, v[120:127] /*v[632:639]*/, v[56:63] /*v[824:831]*/, 0
	v_wmma_f32_16x16x32_bf16 v[88:95] /*v[856:863]*/, v[88:95] /*v[600:607]*/, v[56:63] /*v[824:831]*/, 0
	v_wmma_f32_16x16x32_bf16 v[96:103] /*v[864:871]*/, v[56:63] /*v[568:575]*/, v[56:63] /*v[824:831]*/, 0
	v_wmma_f32_16x16x32_bf16 v[104:111] /*v[872:879]*/, v[24:31] /*v[536:543]*/, v[56:63] /*v[824:831]*/, 0
	s_set_vgpr_msb 0xcecd
	v_wmma_f32_16x16x32_bf16 v[112:119] /*v[880:887]*/, v[248:255] /*v[504:511]*/, v[56:63] /*v[824:831]*/, 0
	v_wmma_f32_16x16x32_bf16 v[154:161] /*v[922:929]*/, v[216:223] /*v[472:479]*/, v[56:63] /*v[824:831]*/, 0
	v_wmma_f32_16x16x32_bf16 v[182:189] /*v[950:957]*/, v[184:191] /*v[440:447]*/, v[56:63] /*v[824:831]*/, 0
	v_wmma_f32_16x16x32_bf16 v[212:219] /*v[980:987]*/, v[152:159] /*v[408:415]*/, v[56:63] /*v[824:831]*/, 0
	s_set_vgpr_msb 0xcd51
	v_wmma_f32_16x16x32_bf16 v[64:71] /*v[320:327]*/, v[32:39] /*v[288:295]*/, v[152:159], v[64:71] /*v[320:327]*/
	s_set_vgpr_msb 0x514d
	v_wmma_f32_16x16x32_bf16 v[32:39] /*v[288:295]*/, v[24:31] /*v[280:287]*/, v[56:63] /*v[824:831]*/, 0
	s_set_vgpr_msb 0x4df2
	v_wmma_f32_16x16x32_bf16 v[64:71] /*v[832:839]*/, v[176:183] /*v[688:695]*/, v[136:143], v[64:71] /*v[832:839]*/
	v_wmma_f32_16x16x32_bf16 v[72:79] /*v[840:847]*/, v[144:151] /*v[656:663]*/, v[136:143], v[72:79] /*v[840:847]*/
	v_wmma_f32_16x16x32_bf16 v[80:87] /*v[848:855]*/, v[112:119] /*v[624:631]*/, v[136:143], v[80:87] /*v[848:855]*/
	v_wmma_f32_16x16x32_bf16 v[88:95] /*v[856:863]*/, v[80:87] /*v[592:599]*/, v[136:143], v[88:95] /*v[856:863]*/
	v_wmma_f32_16x16x32_bf16 v[96:103] /*v[864:871]*/, v[48:55] /*v[560:567]*/, v[136:143], v[96:103] /*v[864:871]*/
	v_wmma_f32_16x16x32_bf16 v[104:111] /*v[872:879]*/, v[16:23] /*v[528:535]*/, v[136:143], v[104:111] /*v[872:879]*/
	s_set_vgpr_msb 0xf2f1
	v_wmma_f32_16x16x32_bf16 v[112:119] /*v[880:887]*/, v[240:247] /*v[496:503]*/, v[136:143], v[112:119] /*v[880:887]*/
	v_wmma_f32_16x16x32_bf16 v[154:161] /*v[922:929]*/, v[208:215] /*v[464:471]*/, v[136:143], v[154:161] /*v[922:929]*/
	v_wmma_f32_16x16x32_bf16 v[182:189] /*v[950:957]*/, v[176:183] /*v[432:439]*/, v[136:143], v[182:189] /*v[950:957]*/
	v_wmma_f32_16x16x32_bf16 v[212:219] /*v[980:987]*/, v[144:151] /*v[400:407]*/, v[136:143], v[212:219] /*v[980:987]*/
	s_set_vgpr_msb 0xf151
	v_wmma_f32_16x16x32_bf16 v[32:39] /*v[288:295]*/, v[16:23] /*v[272:279]*/, v[136:143], v[32:39] /*v[288:295]*/
	s_set_vgpr_msb 0x514c
	v_wmma_f32_16x16x32_bf16 v[24:31] /*v[280:287]*/, v[248:255], v[56:63] /*v[824:831]*/, 0
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x32_bf16 v[16:23] /*v[272:279]*/, v[216:223], v[56:63] /*v[824:831]*/, 0
	s_set_vgpr_msb 0x4cc0
	v_wmma_f32_16x16x32_bf16 v[56:63] /*v[824:831]*/, v[216:223], v[160:167], 0
	s_set_vgpr_msb 0xc0f2
	v_wmma_f32_16x16x32_bf16 v[64:71] /*v[832:839]*/, v[168:175] /*v[680:687]*/, v[144:151], v[64:71] /*v[832:839]*/
	v_wmma_f32_16x16x32_bf16 v[72:79] /*v[840:847]*/, v[136:143] /*v[648:655]*/, v[144:151], v[72:79] /*v[840:847]*/
	v_wmma_f32_16x16x32_bf16 v[80:87] /*v[848:855]*/, v[104:111] /*v[616:623]*/, v[144:151], v[80:87] /*v[848:855]*/
	v_wmma_f32_16x16x32_bf16 v[88:95] /*v[856:863]*/, v[72:79] /*v[584:591]*/, v[144:151], v[88:95] /*v[856:863]*/
	v_wmma_f32_16x16x32_bf16 v[96:103] /*v[864:871]*/, v[40:47] /*v[552:559]*/, v[144:151], v[96:103] /*v[864:871]*/
	v_wmma_f32_16x16x32_bf16 v[104:111] /*v[872:879]*/, v[8:15] /*v[520:527]*/, v[144:151], v[104:111] /*v[872:879]*/
	s_set_vgpr_msb 0xf2f1
	v_wmma_f32_16x16x32_bf16 v[112:119] /*v[880:887]*/, v[232:239] /*v[488:495]*/, v[144:151], v[112:119] /*v[880:887]*/
	v_wmma_f32_16x16x32_bf16 v[154:161] /*v[922:929]*/, v[200:207] /*v[456:463]*/, v[144:151], v[154:161] /*v[922:929]*/
	v_wmma_f32_16x16x32_bf16 v[182:189] /*v[950:957]*/, v[168:175] /*v[424:431]*/, v[144:151], v[182:189] /*v[950:957]*/
	v_wmma_f32_16x16x32_bf16 v[212:219] /*v[980:987]*/, v[136:143] /*v[392:399]*/, v[144:151], v[212:219] /*v[980:987]*/
	s_set_vgpr_msb 0xf150
	v_wmma_f32_16x16x32_bf16 v[24:31] /*v[280:287]*/, v[240:247], v[136:143], v[24:31] /*v[280:287]*/
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x32_bf16 v[16:23] /*v[272:279]*/, v[208:215], v[136:143], v[16:23] /*v[272:279]*/
	s_set_vgpr_msb 0x50f0
	v_wmma_f32_16x16x32_bf16 v[56:63] /*v[824:831]*/, v[208:215], v[168:175], v[56:63] /*v[824:831]*/
	s_set_vgpr_msb 0xf0f2
	v_wmma_f32_16x16x32_bf16 v[64:71] /*v[832:839]*/, v[160:167] /*v[672:679]*/, v[152:159], v[64:71] /*v[832:839]*/
	v_wmma_f32_16x16x32_bf16 v[72:79] /*v[840:847]*/, v[128:135] /*v[640:647]*/, v[152:159], v[72:79] /*v[840:847]*/
	v_wmma_f32_16x16x32_bf16 v[80:87] /*v[848:855]*/, v[96:103] /*v[608:615]*/, v[152:159], v[80:87] /*v[848:855]*/
	s_set_vgpr_msb 0xf2a2
	v_wmma_f32_16x16x32_bf16 v[216:223] /*v[728:735]*/, v[72:79] /*v[584:591]*/, v[176:183], v[216:223] /*v[728:735]*/
	s_set_vgpr_msb 0xa2f2
	v_wmma_f32_16x16x32_bf16 v[88:95] /*v[856:863]*/, v[64:71] /*v[576:583]*/, v[152:159], v[88:95] /*v[856:863]*/
	s_set_vgpr_msb 0xf2a2
	v_wmma_f32_16x16x32_bf16 v[224:231] /*v[736:743]*/, v[40:47] /*v[552:559]*/, v[176:183], v[224:231] /*v[736:743]*/
	s_set_vgpr_msb 0xa2f2
	v_wmma_f32_16x16x32_bf16 v[96:103] /*v[864:871]*/, v[32:39] /*v[544:551]*/, v[152:159], v[96:103] /*v[864:871]*/
	s_set_vgpr_msb 0xf2a2
	v_wmma_f32_16x16x32_bf16 v[232:239] /*v[744:751]*/, v[8:15] /*v[520:527]*/, v[176:183], v[232:239] /*v[744:751]*/
	s_set_vgpr_msb 0xa2f2
	v_wmma_f32_16x16x32_bf16 v[104:111] /*v[872:879]*/, v[0:7] /*v[512:519]*/, v[152:159], v[104:111] /*v[872:879]*/
	s_set_vgpr_msb 0xf2a1
	v_wmma_f32_16x16x32_bf16 v[240:247] /*v[752:759]*/, v[232:239] /*v[488:495]*/, v[176:183], v[240:247] /*v[752:759]*/
	s_set_vgpr_msb 0xa1f1
	v_wmma_f32_16x16x32_bf16 v[112:119] /*v[880:887]*/, v[224:231] /*v[480:487]*/, v[152:159], v[112:119] /*v[880:887]*/
	s_set_vgpr_msb 0xf1a1
	v_wmma_f32_16x16x32_bf16 v[248:255] /*v[760:767]*/, v[200:207] /*v[456:463]*/, v[176:183], v[248:255] /*v[760:767]*/
	s_set_vgpr_msb 0xa1f1
	v_wmma_f32_16x16x32_bf16 v[154:161] /*v[922:929]*/, v[192:199] /*v[448:455]*/, v[152:159], v[154:161] /*v[922:929]*/
	v_wmma_f32_16x16x32_bf16 v[0:7] /*v[768:775]*/, v[168:175] /*v[424:431]*/, v[176:183], v[0:7] /*v[768:775]*/
	v_wmma_f32_16x16x32_bf16 v[182:189] /*v[950:957]*/, v[160:167] /*v[416:423]*/, v[152:159], v[182:189] /*v[950:957]*/
	v_wmma_f32_16x16x32_bf16 v[8:15] /*v[776:783]*/, v[136:143] /*v[392:399]*/, v[176:183], v[8:15] /*v[776:783]*/
	v_wmma_f32_16x16x32_bf16 v[212:219] /*v[980:987]*/, v[128:135] /*v[384:391]*/, v[152:159], v[212:219] /*v[980:987]*/
	s_set_vgpr_msb 0xf151
	v_wmma_f32_16x16x32_bf16 v[32:39] /*v[288:295]*/, v[8:15] /*v[264:271]*/, v[144:151], v[32:39] /*v[288:295]*/
	s_set_vgpr_msb 0x51f1
	v_wmma_f32_16x16x32_bf16 v[40:47] /*v[808:815]*/, v[8:15] /*v[264:271]*/, v[176:183], v[40:47] /*v[808:815]*/
	s_set_vgpr_msb 0xf150
	v_wmma_f32_16x16x32_bf16 v[24:31] /*v[280:287]*/, v[232:239], v[144:151], v[24:31] /*v[280:287]*/
	s_set_vgpr_msb 0x50f0
	v_wmma_f32_16x16x32_bf16 v[48:55] /*v[816:823]*/, v[232:239], v[176:183], v[48:55] /*v[816:823]*/
	s_set_vgpr_msb 0xf050
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[16:23] /*v[272:279]*/, v[200:207], v[144:151], v[16:23] /*v[272:279]*/
	s_set_vgpr_msb 0x50f0
	v_wmma_f32_16x16x32_bf16 v[56:63] /*v[824:831]*/, v[200:207], v[176:183], v[56:63] /*v[824:831]*/
	s_set_vgpr_msb 0xf0a2
	v_wmma_f32_16x16x32_bf16 v[200:207] /*v[712:719]*/, v[128:135] /*v[640:647]*/, v[184:191], v[200:207] /*v[712:719]*/
	v_wmma_f32_16x16x32_bf16 v[208:215] /*v[720:727]*/, v[96:103] /*v[608:615]*/, v[184:191], v[208:215] /*v[720:727]*/
	v_wmma_f32_16x16x32_bf16 v[216:223] /*v[728:735]*/, v[64:71] /*v[576:583]*/, v[184:191], v[216:223] /*v[728:735]*/
	v_wmma_f32_16x16x32_bf16 v[224:231] /*v[736:743]*/, v[32:39] /*v[544:551]*/, v[184:191], v[224:231] /*v[736:743]*/
	v_wmma_f32_16x16x32_bf16 v[232:239] /*v[744:751]*/, v[0:7] /*v[512:519]*/, v[184:191], v[232:239] /*v[744:751]*/
	s_set_vgpr_msb 0xa2a1
	v_wmma_f32_16x16x32_bf16 v[240:247] /*v[752:759]*/, v[224:231] /*v[480:487]*/, v[184:191], v[240:247] /*v[752:759]*/
	v_wmma_f32_16x16x32_bf16 v[248:255] /*v[760:767]*/, v[192:199] /*v[448:455]*/, v[184:191], v[248:255] /*v[760:767]*/
	s_set_vgpr_msb 0xa1f1
	v_wmma_f32_16x16x32_bf16 v[0:7] /*v[768:775]*/, v[160:167] /*v[416:423]*/, v[184:191], v[0:7] /*v[768:775]*/
	v_wmma_f32_16x16x32_bf16 v[8:15] /*v[776:783]*/, v[128:135] /*v[384:391]*/, v[184:191], v[8:15] /*v[776:783]*/
	s_set_vgpr_msb 0xf151
	v_wmma_f32_16x16x32_bf16 v[32:39] /*v[288:295]*/, v[0:7] /*v[256:263]*/, v[152:159], v[32:39] /*v[288:295]*/
	s_set_vgpr_msb 0x51f1
	v_wmma_f32_16x16x32_bf16 v[40:47] /*v[808:815]*/, v[0:7] /*v[256:263]*/, v[184:191], v[40:47] /*v[808:815]*/
	s_set_vgpr_msb 0xf150
	v_wmma_f32_16x16x32_bf16 v[24:31] /*v[280:287]*/, v[224:231], v[152:159], v[24:31] /*v[280:287]*/
	s_set_vgpr_msb 0x50f0
	v_wmma_f32_16x16x32_bf16 v[48:55] /*v[816:823]*/, v[224:231], v[184:191], v[48:55] /*v[816:823]*/
	s_set_vgpr_msb 0xf050
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[16:23] /*v[272:279]*/, v[192:199], v[152:159], v[16:23] /*v[272:279]*/
	s_set_vgpr_msb 0x50f0
	v_wmma_f32_16x16x32_bf16 v[56:63] /*v[824:831]*/, v[192:199], v[184:191], v[56:63] /*v[824:831]*/
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0xf040
	ds_load_tr16_b128 v[72:75] /*v[328:331]*/, v130 offset:192
	s_set_vgpr_msb 0x4000
	ds_load_tr16_b128 v[192:195], v130 offset:224
	s_set_vgpr_msb 64
	ds_load_tr16_b128 v[76:79] /*v[332:335]*/, v130 offset:4800
	s_set_vgpr_msb 0x4000
	ds_load_tr16_b128 v[196:199], v130 offset:4832
	s_set_vgpr_msb 0x80
	ds_load_tr16_b128 v[188:191] /*v[700:703]*/, v130 offset:4608
	ds_load_tr16_b128 v[108:111] /*v[620:623]*/, v130 offset:4640
	ds_load_tr16_b128 v[176:179] /*v[688:691]*/, v130 offset:9216
	ds_load_tr16_b128 v[120:123] /*v[632:635]*/, v130 offset:9248
	ds_load_tr16_b128 v[180:183] /*v[692:695]*/, v130 offset:13824
	ds_load_tr16_b128 v[124:127] /*v[636:639]*/, v130 offset:13856
	ds_load_tr16_b128 v[168:171] /*v[680:683]*/, v130 offset:18432
	ds_load_tr16_b128 v[112:115] /*v[624:627]*/, v130 offset:18464
	ds_load_tr16_b128 v[172:175] /*v[684:687]*/, v130 offset:23040
	ds_load_tr16_b128 v[116:119] /*v[628:631]*/, v130 offset:23072
	ds_load_tr16_b128 v[160:163] /*v[672:675]*/, v130 offset:27648
	ds_load_tr16_b128 v[96:99] /*v[608:611]*/, v130 offset:27680
	ds_load_tr16_b128 v[164:167] /*v[676:679]*/, v130 offset:32256
	ds_load_tr16_b128 v[100:103] /*v[612:615]*/, v130 offset:32288
	ds_load_tr16_b128 v[152:155] /*v[664:667]*/, v130 offset:36864
	ds_load_tr16_b128 v[88:91] /*v[600:603]*/, v130 offset:36896
	ds_load_tr16_b128 v[156:159] /*v[668:671]*/, v130 offset:41472
	ds_load_tr16_b128 v[92:95] /*v[604:607]*/, v130 offset:41504
	ds_load_tr16_b128 v[144:147] /*v[656:659]*/, v130 offset:46080
	ds_load_tr16_b128 v[80:83] /*v[592:595]*/, v130 offset:46112
	ds_load_tr16_b128 v[148:151] /*v[660:663]*/, v130 offset:50688
	ds_load_tr16_b128 v[84:87] /*v[596:599]*/, v130 offset:50720
	ds_load_tr16_b128 v[136:139] /*v[648:651]*/, v130 offset:55296
	ds_load_tr16_b128 v[64:67] /*v[576:579]*/, v130 offset:55328
	ds_load_tr16_b128 v[72:75] /*v[584:587]*/, v130 offset:64
	s_set_vgpr_msb 0x8040
	ds_load_tr16_b128 v[248:251] /*v[504:507]*/, v130 offset:96
	s_set_vgpr_msb 0x4080
	ds_load_tr16_b128 v[76:79] /*v[588:591]*/, v130 offset:4672
	s_set_vgpr_msb 0x8040
	ds_load_tr16_b128 v[252:255] /*v[508:511]*/, v130 offset:4704
	s_set_vgpr_msb 0x4080
	ds_load_tr16_b128 v[56:59] /*v[568:571]*/, v130 offset:9280
	s_set_vgpr_msb 0x8040
	ds_load_tr16_b128 v[240:243] /*v[496:499]*/, v130 offset:9312
	s_set_vgpr_msb 0x4080
	ds_load_tr16_b128 v[60:63] /*v[572:575]*/, v130 offset:13888
	s_set_vgpr_msb 0x8040
	ds_load_tr16_b128 v[244:247] /*v[500:503]*/, v130 offset:13920
	s_set_vgpr_msb 0x4080
	ds_load_tr16_b128 v[40:43] /*v[552:555]*/, v130 offset:18496
	s_set_vgpr_msb 0x8040
	ds_load_tr16_b128 v[232:235] /*v[488:491]*/, v130 offset:18528
	s_set_vgpr_msb 0x4080
	ds_load_tr16_b128 v[44:47] /*v[556:559]*/, v130 offset:23104
	s_set_vgpr_msb 0x8040
	ds_load_tr16_b128 v[236:239] /*v[492:495]*/, v130 offset:23136
	s_set_vgpr_msb 0x4080
	ds_load_tr16_b128 v[32:35] /*v[544:547]*/, v130 offset:27712
	s_set_vgpr_msb 0x8040
	ds_load_tr16_b128 v[224:227] /*v[480:483]*/, v130 offset:27744
	s_set_vgpr_msb 0x4080
	ds_load_tr16_b128 v[36:39] /*v[548:551]*/, v130 offset:32320
	s_set_vgpr_msb 0x8040
	ds_load_tr16_b128 v[228:231] /*v[484:487]*/, v130 offset:32352
	s_set_vgpr_msb 0x4080
	ds_load_tr16_b128 v[24:27] /*v[536:539]*/, v130 offset:36928
	s_set_vgpr_msb 0x8040
	ds_load_tr16_b128 v[216:219] /*v[472:475]*/, v130 offset:36960
	s_set_vgpr_msb 0x4080
	ds_load_tr16_b128 v[28:31] /*v[540:543]*/, v130 offset:41536
	s_set_vgpr_msb 0x8040
	ds_load_tr16_b128 v[220:223] /*v[476:479]*/, v130 offset:41568
	s_set_vgpr_msb 0x4080
	ds_load_tr16_b128 v[16:19] /*v[528:531]*/, v130 offset:46144
	s_set_vgpr_msb 0x8040
	ds_load_tr16_b128 v[208:211] /*v[464:467]*/, v130 offset:46176
	s_set_vgpr_msb 0x4080
	ds_load_tr16_b128 v[20:23] /*v[532:535]*/, v130 offset:50752
	s_set_vgpr_msb 0x8040
	ds_load_tr16_b128 v[212:215] /*v[468:471]*/, v130 offset:50784
	s_set_vgpr_msb 0x4080
	ds_load_tr16_b128 v[8:11] /*v[520:523]*/, v130 offset:55360
	s_set_vgpr_msb 0x8040
	ds_load_tr16_b128 v[192:195] /*v[448:451]*/, v130 offset:55392
	ds_load_tr16_b128 v[200:203] /*v[456:459]*/, v130 offset:128
	ds_load_tr16_b128 v[120:123] /*v[376:379]*/, v130 offset:160
	ds_load_tr16_b128 v[204:207] /*v[460:463]*/, v130 offset:4736
	ds_load_tr16_b128 v[124:127] /*v[380:383]*/, v130 offset:4768
	ds_load_tr16_b128 v[184:187] /*v[440:443]*/, v130 offset:9344
	ds_load_tr16_b128 v[112:115] /*v[368:371]*/, v130 offset:9376
	ds_load_tr16_b128 v[188:191] /*v[444:447]*/, v130 offset:13952
	ds_load_tr16_b128 v[116:119] /*v[372:375]*/, v130 offset:13984
	ds_load_tr16_b128 v[168:171] /*v[424:427]*/, v130 offset:18560
	ds_load_tr16_b128 v[104:107] /*v[360:363]*/, v130 offset:18592
	ds_load_tr16_b128 v[172:175] /*v[428:431]*/, v130 offset:23168
	ds_load_tr16_b128 v[108:111] /*v[364:367]*/, v130 offset:23200
	ds_load_tr16_b128 v[160:163] /*v[416:419]*/, v130 offset:27776
	ds_load_tr16_b128 v[0:3] /*v[256:259]*/, v130 offset:27808
	ds_load_tr16_b128 v[164:167] /*v[420:423]*/, v130 offset:32384
	ds_load_tr16_b128 v[4:7] /*v[260:263]*/, v130 offset:32416
	ds_load_tr16_b128 v[152:155] /*v[408:411]*/, v130 offset:36992
	ds_load_tr16_b128 v[88:91] /*v[344:347]*/, v130 offset:37024
	ds_load_tr16_b128 v[156:159] /*v[412:415]*/, v130 offset:41600
	ds_load_tr16_b128 v[92:95] /*v[348:351]*/, v130 offset:41632
	ds_load_tr16_b128 v[144:147] /*v[400:403]*/, v130 offset:46208
	ds_load_tr16_b128 v[80:83] /*v[336:339]*/, v130 offset:46240
	ds_load_tr16_b128 v[148:151] /*v[404:407]*/, v130 offset:50816
	ds_load_tr16_b128 v[84:87] /*v[340:343]*/, v130 offset:50848
	ds_load_tr16_b128 v[136:139] /*v[392:395]*/, v130 offset:55424
	ds_load_tr16_b128 v[8:11] /*v[264:267]*/, v130 offset:55456
	s_set_vgpr_msb 0x4000
	v_add_nc_u32_e32 v131, 0x10e00, v130
	v_add_nc_u32_e32 v132, 0x10e20, v130
	s_set_vgpr_msb 0x80
	ds_load_tr16_b128 v[140:143] /*v[652:655]*/, v130 offset:59904
	ds_load_tr16_b128 v[68:71] /*v[580:583]*/, v130 offset:59936
	ds_load_tr16_b128 v[128:131] /*v[640:643]*/, v130 offset:64512
	ds_load_tr16_b128 v[48:51] /*v[560:563]*/, v130 offset:64544
	s_wait_alu depctr_va_vdst(1)
	ds_load_tr16_b128 v[132:135] /*v[644:647]*/, v131
	s_wait_alu depctr_va_vdst(0)
	ds_load_tr16_b128 v[52:55] /*v[564:567]*/, v132
	s_wait_alu depctr_vm_vsrc(0)
	s_set_vgpr_msb 0x8000
	v_add_nc_u32_e32 v131, 0x10e40, v130
	v_add_nc_u32_e32 v132, 0x10e60, v130
	s_set_vgpr_msb 0x80
	ds_load_tr16_b128 v[12:15] /*v[524:527]*/, v130 offset:59968
	s_set_vgpr_msb 0x8040
	ds_load_tr16_b128 v[196:199] /*v[452:455]*/, v130 offset:60000
	s_set_vgpr_msb 0x4080
	ds_load_tr16_b128 v[0:3] /*v[512:515]*/, v130 offset:64576
	s_set_vgpr_msb 0x8040
	ds_load_tr16_b128 v[176:179] /*v[432:435]*/, v130 offset:64608
	s_wait_alu depctr_va_vdst(1)
	s_set_vgpr_msb 0x4080
	ds_load_tr16_b128 v[4:7] /*v[516:519]*/, v131
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x8040
	ds_load_tr16_b128 v[180:183] /*v[436:439]*/, v132
	s_wait_alu depctr_vm_vsrc(1)
	s_set_vgpr_msb 0x4000
	v_add_nc_u32_e32 v131, 0x10e80, v130
	s_wait_alu depctr_vm_vsrc(0)
	v_add_nc_u32_e32 v132, 0x10ea0, v130
	s_set_vgpr_msb 64
	ds_load_tr16_b128 v[140:143] /*v[396:399]*/, v130 offset:60032
	ds_load_tr16_b128 v[12:15] /*v[268:271]*/, v130 offset:60064
	ds_load_tr16_b128 v[128:131] /*v[384:387]*/, v130 offset:64640
	ds_load_tr16_b128 v[48:51] /*v[304:307]*/, v130 offset:64672
	s_wait_alu depctr_va_vdst(1)
	ds_load_tr16_b128 v[132:135] /*v[388:391]*/, v131
	s_wait_alu depctr_va_vdst(0)
	ds_load_tr16_b128 v[52:55] /*v[308:311]*/, v132
	s_wait_alu depctr_vm_vsrc(1)
	s_set_vgpr_msb 0x4000
	v_add_nc_u32_e32 v131, 0x10ec0, v130
	s_wait_alu depctr_vm_vsrc(0)
	v_add_nc_u32_e32 v132, 0x10ee0, v130
	s_set_vgpr_msb 0x80
	ds_load_tr16_b128 v[184:187] /*v[696:699]*/, v130
	ds_load_tr16_b128 v[104:107] /*v[616:619]*/, v130 offset:32
	s_wait_dscnt 0x3e
	s_clause 0x1
	scratch_store_b128 off, v[192:195], off offset:268 nv
	scratch_store_b128 off, v[196:199], off offset:284 nv
	s_set_vgpr_msb 0x8040
	ds_load_tr16_b128 v[56:59] /*v[312:315]*/, v130 offset:9408
	s_wait_alu depctr_vm_vsrc(0)
	s_set_vgpr_msb 0x4000
	ds_load_tr16_b128 v[192:195], v130 offset:9440
	s_set_vgpr_msb 64
	ds_load_tr16_b128 v[60:63] /*v[316:319]*/, v130 offset:14016
	s_set_vgpr_msb 0x4000
	ds_load_tr16_b128 v[196:199], v130 offset:14048
	s_wait_dscnt 0x2
	scratch_store_b128 off, v[192:195], off offset:236 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[196:199], off offset:252 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_tr16_b128 v[196:199], v130 offset:18624
	ds_load_tr16_b128 v[192:195], v130 offset:18656
	ds_load_tr16_b128 v[200:203], v130 offset:23232
	s_wait_dscnt 0x2
	scratch_store_b128 off, v[196:199], off offset:460 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[200:203], off offset:476 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_tr16_b128 v[196:199], v130 offset:23264
	scratch_store_b128 off, v[192:195], off offset:204 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[196:199], off offset:220 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_tr16_b128 v[196:199], v130 offset:27840
	ds_load_tr16_b128 v[192:195], v130 offset:27872
	ds_load_tr16_b128 v[200:203], v130 offset:32448
	s_wait_dscnt 0x2
	scratch_store_b128 off, v[196:199], off offset:428 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[200:203], off offset:444 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_tr16_b128 v[196:199], v130 offset:32480
	scratch_store_b128 off, v[192:195], off offset:172 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[196:199], off offset:188 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_tr16_b128 v[196:199], v130 offset:37056
	ds_load_tr16_b128 v[192:195], v130 offset:37088
	ds_load_tr16_b128 v[200:203], v130 offset:41664
	s_wait_dscnt 0x2
	scratch_store_b128 off, v[196:199], off offset:396 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[200:203], off offset:412 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_tr16_b128 v[196:199], v130 offset:41696
	scratch_store_b128 off, v[192:195], off offset:140 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[196:199], off offset:156 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_tr16_b128 v[196:199], v130 offset:46272
	ds_load_tr16_b128 v[192:195], v130 offset:46304
	ds_load_tr16_b128 v[200:203], v130 offset:50880
	s_wait_dscnt 0x2
	scratch_store_b128 off, v[196:199], off offset:364 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[200:203], off offset:380 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_tr16_b128 v[196:199], v130 offset:50912
	ds_load_tr16_b128 v[200:203], v130 offset:60096
	scratch_store_b128 off, v[192:195], off offset:108 nv
	s_wait_dscnt 0x1
	scratch_store_b128 off, v[196:199], off offset:124 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_tr16_b128 v[196:199], v130 offset:55488
	ds_load_tr16_b128 v[192:195], v130 offset:55520
	s_wait_dscnt 0x1
	s_clause 0x1
	scratch_store_b128 off, v[196:199], off offset:332 nv
	scratch_store_b128 off, v[200:203], off offset:348 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_tr16_b128 v[196:199], v130 offset:60128
	s_wait_dscnt 0x1
	scratch_store_b128 off, v[192:195], off offset:76 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[196:199], off offset:92 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_tr16_b128 v[196:199], v130 offset:64704
	ds_load_tr16_b128 v[192:195], v130 offset:64736
	s_wait_alu depctr_va_vdst(1)
	ds_load_tr16_b128 v[200:203], v131
	s_wait_dscnt 0x2
	scratch_store_b128 off, v[196:199], off offset:300 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[200:203], off offset:316 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_va_vdst(0) depctr_vm_vsrc(0)
	ds_load_tr16_b128 v[196:199], v132
	scratch_store_b128 off, v[192:195], off offset:44 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[196:199], off offset:60 nv
	s_set_vgpr_msb 15
	v_dual_max_num_f32 v131, v64 /*v832*/, v65 /*v833*/ :: v_dual_max_num_f32 v244, v243 /*v1011*/, v244 /*v1012*/
	s_wait_alu depctr_vm_vsrc(0)
	s_set_vgpr_msb 0xf0a
	v_max_num_f32_e32 v132, v192 /*v704*/, v193 /*v705*/
	s_set_vgpr_msb 0xa3f
	v_max3_num_f32 v192, v67 /*v835*/, v68 /*v836*/, v69 /*v837*/
	s_set_vgpr_msb 0x3f2a
	v_max3_num_f32 v193, v195 /*v707*/, v196 /*v708*/, v197 /*v709*/
	s_set_vgpr_msb 0x2a3f
	v_max3_num_f32 v196, v73 /*v841*/, v74 /*v842*/, v75 /*v843*/
	s_set_vgpr_msb 0x3f2a
	v_max3_num_f32 v197, v201 /*v713*/, v202 /*v714*/, v203 /*v715*/
	s_set_vgpr_msb 0x2a3f
	v_max3_num_f32 v198, v76 /*v844*/, v77 /*v845*/, v78 /*v846*/
	s_set_vgpr_msb 0x3f2a
	v_max3_num_f32 v199, v204 /*v716*/, v205 /*v717*/, v206 /*v718*/
	s_set_vgpr_msb 0x2a3f
	v_max3_num_f32 v200, v79 /*v847*/, v80 /*v848*/, v81 /*v849*/
	s_set_vgpr_msb 0x3f2a
	v_max3_num_f32 v201, v207 /*v719*/, v208 /*v720*/, v209 /*v721*/
	s_set_vgpr_msb 0x2a3f
	v_max3_num_f32 v208, v91 /*v859*/, v92 /*v860*/, v93 /*v861*/
	s_set_vgpr_msb 0x3f2a
	v_max3_num_f32 v209, v219 /*v731*/, v220 /*v732*/, v221 /*v733*/
	s_set_vgpr_msb 0x2a3f
	v_max3_num_f32 v210, v94 /*v862*/, v95 /*v863*/, v96 /*v864*/
	s_set_vgpr_msb 0x3f2a
	v_max3_num_f32 v211, v222 /*v734*/, v223 /*v735*/, v224 /*v736*/
	s_set_vgpr_msb 0x2a3f
	v_max3_num_f32 v212, v97 /*v865*/, v98 /*v866*/, v99 /*v867*/
	s_set_vgpr_msb 0x3f2a
	v_max3_num_f32 v213, v225 /*v737*/, v226 /*v738*/, v227 /*v739*/
	s_set_vgpr_msb 0x2a45
	v_max_num_f32_e32 v46 /*v302*/, v36 /*v292*/, v37 /*v293*/
	s_set_vgpr_msb 0x45d5
	v_max3_num_f32 v120 /*v888*/, v39 /*v295*/, v24 /*v280*/, v25 /*v281*/
	v_max3_num_f32 v124 /*v892*/, v29 /*v285*/, v30 /*v286*/, v31 /*v287*/
	v_max3_num_f32 v126 /*v894*/, v16 /*v272*/, v17 /*v273*/, v18 /*v274*/
	v_max3_num_f32 v128 /*v896*/, v19 /*v275*/, v20 /*v276*/, v21 /*v277*/
	s_set_vgpr_msb 0xd53f
	v_max3_num_f32 v202, v82 /*v850*/, v83 /*v851*/, v84 /*v852*/
	v_max3_num_f32 v204, v85 /*v853*/, v86 /*v854*/, v87 /*v855*/
	v_max3_num_f32 v206, v88 /*v856*/, v89 /*v857*/, v90 /*v858*/
	v_max3_num_f32 v214, v100 /*v868*/, v101 /*v869*/, v102 /*v870*/
	s_set_vgpr_msb 0x3f2a
	v_max3_num_f32 v215, v228 /*v740*/, v229 /*v741*/, v230 /*v742*/
	s_set_vgpr_msb 0x2a3f
	v_max3_num_f32 v216, v103 /*v871*/, v104 /*v872*/, v105 /*v873*/
	s_set_vgpr_msb 0x3f2a
	v_max3_num_f32 v217, v231 /*v743*/, v232 /*v744*/, v233 /*v745*/
	s_set_vgpr_msb 0x2a3f
	v_max3_num_f32 v218, v106 /*v874*/, v107 /*v875*/, v108 /*v876*/
	s_set_vgpr_msb 0x3f2a
	v_max3_num_f32 v219, v234 /*v746*/, v235 /*v747*/, v236 /*v748*/
	s_set_vgpr_msb 0x2a3f
	v_max3_num_f32 v220, v109 /*v877*/, v110 /*v878*/, v111 /*v879*/
	v_max3_num_f32 v222, v112 /*v880*/, v113 /*v881*/, v114 /*v882*/
	v_max3_num_f32 v224, v115 /*v883*/, v116 /*v884*/, v117 /*v885*/
	v_max3_num_f32 v226, v118 /*v886*/, v119 /*v887*/, v154 /*v922*/
	v_max3_num_f32 v228, v155 /*v923*/, v156 /*v924*/, v157 /*v925*/
	v_max3_num_f32 v230, v158 /*v926*/, v159 /*v927*/, v160 /*v928*/
	v_max3_num_f32 v232, v161 /*v929*/, v182 /*v950*/, v183 /*v951*/
	v_max3_num_f32 v234, v184 /*v952*/, v185 /*v953*/, v186 /*v954*/
	v_max3_num_f32 v236, v187 /*v955*/, v188 /*v956*/, v189 /*v957*/
	v_max3_num_f32 v238, v212 /*v980*/, v213 /*v981*/, v214 /*v982*/
	v_max3_num_f32 v240, v215 /*v983*/, v216 /*v984*/, v217 /*v985*/
	v_max3_num_f32 v242, v218 /*v986*/, v219 /*v987*/, v242 /*v1010*/
	v_max3_num_f32 v246, v246 /*v1014*/, v247 /*v1015*/, v248 /*v1016*/
	s_set_vgpr_msb 0x3f15
	v_max3_num_f32 v250, v98 /*v354*/, v99 /*v355*/, v100 /*v356*/
	v_max3_num_f32 v252, v101 /*v357*/, v102 /*v358*/, v103 /*v359*/
	v_max3_num_f32 v254, v64 /*v320*/, v65 /*v321*/, v66 /*v322*/
	s_set_vgpr_msb 0x154f
	v_max_num_f32_e32 v47 /*v303*/, v44 /*v812*/, v45 /*v813*/
	s_set_vgpr_msb 0x4fff
	v_max3_num_f32 v121 /*v889*/, v47 /*v815*/, v48 /*v816*/, v49 /*v817*/
	s_set_vgpr_msb 0xffd5
	v_max3_num_f32 v122 /*v890*/, v26 /*v282*/, v27 /*v283*/, v28 /*v284*/
	s_set_vgpr_msb 0xd5ff
	v_max3_num_f32 v125 /*v893*/, v53 /*v821*/, v54 /*v822*/, v55 /*v823*/
	v_max3_num_f32 v127 /*v895*/, v56 /*v824*/, v57 /*v825*/, v58 /*v826*/
	v_max3_num_f32 v129 /*v897*/, v59 /*v827*/, v60 /*v828*/, v61 /*v829*/
	s_set_vgpr_msb 0xff0c
	v_max3_num_f32 v131, v131, v66 /*v834*/, v192
	s_set_vgpr_msb 0xc08
	v_max3_num_f32 v132, v132, v194 /*v706*/, v193
	s_set_vgpr_msb 0x800
	v_max3_num_f32 v192, v196, v198, v200
	v_max3_num_f32 v193, v197, v199, v201
	v_max3_num_f32 v198, v208, v210, v212
	v_max3_num_f32 v199, v209, v211, v213
	s_set_vgpr_msb 53
	v_max3_num_f32 v212, v46 /*v302*/, v38 /*v294*/, v120 /*v888*/
	s_set_vgpr_msb 0x353f
	v_max3_num_f32 v213, v124 /*v892*/, v126 /*v894*/, v128 /*v896*/
	v_max3_num_f32 v194, v70 /*v838*/, v71 /*v839*/, v72 /*v840*/
	s_set_vgpr_msb 0x3f2a
	v_max3_num_f32 v203, v210 /*v722*/, v211 /*v723*/, v212 /*v724*/
	v_max3_num_f32 v205, v213 /*v725*/, v214 /*v726*/, v215 /*v727*/
	v_max3_num_f32 v207, v216 /*v728*/, v217 /*v729*/, v218 /*v730*/
	v_max3_num_f32 v221, v237 /*v749*/, v238 /*v750*/, v239 /*v751*/
	v_max3_num_f32 v223, v240 /*v752*/, v241 /*v753*/, v242 /*v754*/
	v_max3_num_f32 v225, v243 /*v755*/, v244 /*v756*/, v245 /*v757*/
	s_set_vgpr_msb 0x2a3f
	v_max_num_f32_e32 v245, v17 /*v785*/, v18 /*v786*/
	v_max3_num_f32 v247, v20 /*v788*/, v21 /*v789*/, v22 /*v790*/
	s_set_vgpr_msb 0x3f17
	v_max3_num_f32 v248, v249 /*v1017*/, v96 /*v352*/, v97 /*v353*/
	s_set_vgpr_msb 0x173f
	v_max3_num_f32 v251, v26 /*v794*/, v27 /*v795*/, v28 /*v796*/
	v_max3_num_f32 v253, v29 /*v797*/, v30 /*v798*/, v31 /*v799*/
	v_max3_num_f32 v255, v32 /*v800*/, v33 /*v801*/, v34 /*v802*/
	s_set_vgpr_msb 0x3f55
	v_max3_num_f32 v40 /*v296*/, v67 /*v323*/, v68 /*v324*/, v69 /*v325*/
	v_max3_num_f32 v42 /*v298*/, v70 /*v326*/, v71 /*v327*/, v32 /*v288*/
	v_max3_num_f32 v44 /*v300*/, v33 /*v289*/, v34 /*v290*/, v35 /*v291*/
	s_set_vgpr_msb 0x55ff
	v_max3_num_f32 v123 /*v891*/, v50 /*v818*/, v51 /*v819*/, v52 /*v820*/
	s_set_vgpr_msb 0xff00
	v_max3_num_f32 v196, v202, v204, v206
	v_max3_num_f32 v200, v214, v216, v218
	v_max3_num_f32 v201, v215, v217, v219
	v_max3_num_f32 v202, v220, v222, v224
	v_max3_num_f32 v204, v226, v228, v230
	v_max3_num_f32 v206, v232, v234, v236
	v_max3_num_f32 v208, v238, v240, v242
	s_set_vgpr_msb 12
	v_max3_num_f32 v210, v244, v245 /*v1013*/, v246
	s_set_vgpr_msb 0xc00
	v_max3_num_f32 v214, v250, v252, v254
	s_set_vgpr_msb 61
	v_max3_num_f32 v217, v47 /*v303*/, v46 /*v814*/, v121 /*v889*/
	s_set_vgpr_msb 0x3d0c
	v_max3_num_f32 v212, v212, v122 /*v890*/, v213
	s_set_vgpr_msb 0xc3f
	v_max3_num_f32 v213, v125 /*v893*/, v127 /*v895*/, v129 /*v897*/
	s_set_vgpr_msb 0x3f2a
	v_max3_num_f32 v195, v198 /*v710*/, v199 /*v711*/, v200 /*v712*/
	v_max3_num_f32 v227, v246 /*v758*/, v247 /*v759*/, v248 /*v760*/
	v_max3_num_f32 v229, v249 /*v761*/, v250 /*v762*/, v251 /*v763*/
	v_max3_num_f32 v231, v252 /*v764*/, v253 /*v765*/, v254 /*v766*/
	s_set_vgpr_msb 0x2a3e
	v_max3_num_f32 v233, v255 /*v767*/, v0 /*v768*/, v1 /*v769*/
	s_set_vgpr_msb 0x3e3f
	v_max3_num_f32 v235, v2 /*v770*/, v3 /*v771*/, v4 /*v772*/
	v_max3_num_f32 v237, v5 /*v773*/, v6 /*v774*/, v7 /*v775*/
	v_max3_num_f32 v239, v8 /*v776*/, v9 /*v777*/, v10 /*v778*/
	v_max3_num_f32 v241, v11 /*v779*/, v12 /*v780*/, v13 /*v781*/
	v_max3_num_f32 v243, v14 /*v782*/, v15 /*v783*/, v16 /*v784*/
	v_max3_num_f32 v249, v23 /*v791*/, v24 /*v792*/, v25 /*v793*/
	s_set_vgpr_msb 0x3f7f
	v_max3_num_f32 v41 /*v297*/, v35 /*v803*/, v36 /*v804*/, v37 /*v805*/
	v_max3_num_f32 v43 /*v299*/, v38 /*v806*/, v39 /*v807*/, v40 /*v808*/
	v_max3_num_f32 v45 /*v301*/, v41 /*v809*/, v42 /*v810*/, v43 /*v811*/
	s_set_vgpr_msb 0x7f00
	v_max3_num_f32 v197, v203, v205, v207
	v_max3_num_f32 v203, v221, v223, v225
	s_set_vgpr_msb 12
	v_max3_num_f32 v211, v245, v19 /*v787*/, v247
	s_set_vgpr_msb 0xc00
	v_max3_num_f32 v215, v251, v253, v255
	s_set_vgpr_msb 21
	v_max3_num_f32 v216, v40 /*v296*/, v42 /*v298*/, v44 /*v300*/
	s_set_vgpr_msb 0x1500
	v_max3_num_f32 v131, v131, v194, v192
	v_max3_num_f32 v192, v198, v200, v202
	v_max3_num_f32 v194, v210, v248, v214
	s_set_vgpr_msb 20
	v_max3_num_f32 v198, v212, v22 /*v278*/, v23 /*v279*/
	s_set_vgpr_msb 0x1400
	v_max3_num_f32 v202, v204, v206, v208
	s_set_vgpr_msb 12
	v_max3_num_f32 v204, v217, v123 /*v891*/, v213
	s_set_vgpr_msb 0xc00
	v_max3_num_f32 v205, v227, v229, v231
	v_max3_num_f32 v207, v233, v235, v237
	v_max3_num_f32 v209, v239, v241, v243
	s_set_vgpr_msb 21
	v_max3_num_f32 v200, v41 /*v297*/, v43 /*v299*/, v45 /*v301*/
	s_set_vgpr_msb 0x1500
	v_max3_num_f32 v131, v131, v196, v192
	v_max3_num_f32 v192, v194, v216, v198
	v_max3_num_f32 v132, v132, v195, v193
	v_max3_num_f32 v193, v199, v201, v203
	v_max3_num_f32 v194, v211, v249, v215
	s_set_vgpr_msb 60
	v_max3_num_f32 v195, v204, v62 /*v830*/, v63 /*v831*/
	s_set_vgpr_msb 0x3c00
	v_max3_num_f32 v131, v131, v202, v192
	v_max3_num_f32 v192, v205, v207, v209
	v_max3_num_f32 v132, v132, v197, v193
	v_max3_num_f32 v193, v194, v200, v195
	v_max3_num_f32 v132, v132, v192, v193
	v_dual_mov_b32 v194, v131 :: v_dual_mov_b32 v192, v132
	v_permlanex16_b32 v194, v194, s65, 0xfedcba98
	v_permlanex16_b32 v192, v192, s65, 0xfedcba98
	v_dual_max_num_f32 v131, v131, v194 :: v_dual_max_num_f32 v192, v132, v192
	v_dual_sub_f32 v193, v131, v134 :: v_dual_max_num_f32 v131, v134, v131
	v_cmp_lt_f32_e32 vcc_lo, 0x41000000, v193
	s_cmp_eq_u32 vcc_lo, 0
	s_cselect_b32 s2, -1, 0
	v_dual_sub_f32 v193, v192, v133 :: v_dual_cndmask_b32 v132, v131, v134, s2
	v_cmp_lt_f32_e64 s2, 0x41000000, v193
	v_sub_f32_e32 v134, v134, v132
	v_max_num_f32_e32 v131, v133, v192
	v_mul_f32_e32 v244, 0xbfb8aa3b, v132
	s_cmp_lg_u32 s2, 0
	v_mul_f32_e32 v134, 0x3fb8aa3b, v134
	s_cselect_b32 s5, -1, 0
	s_cmp_eq_u32 s2, 0
	s_set_vgpr_msb 3
	v_pk_fma_f32 v[192:193], v[64:65] /*v[832:833]*/, s[52:53], v[244:245] op_sel_hi:[1,0,0]
	s_cselect_b32 s2, -1, 0
	s_set_vgpr_msb 0x301
	v_pk_fma_f32 v[232:233], v[16:17] /*v[272:273]*/, s[52:53], v[244:245] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x100
	v_cndmask_b32_e64 v131, v131, v133, s2
	s_set_vgpr_msb 1
	v_pk_fma_f32 v[236:237], v[18:19] /*v[274:275]*/, s[52:53], v[244:245] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[246:247], v[20:21] /*v[276:277]*/, s[52:53], v[244:245] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x103
	v_pk_fma_f32 v[194:195], v[66:67] /*v[834:835]*/, s[52:53], v[244:245] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x3c0
	v_exp_f32_e32 v64 /*v832*/, v192
	s_set_vgpr_msb 0xc040
	v_mul_f32_e32 v40 /*v296*/, 0xbfb8aa3b, v131
	s_set_vgpr_msb 0x4003
	v_pk_fma_f32 v[196:197], v[68:69] /*v[836:837]*/, s[52:53], v[244:245] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x3c0
	v_exp_f32_e32 v66 /*v834*/, v193
	v_nop
	s_set_vgpr_msb 0xc003
	v_pk_fma_f32 v[192:193], v[70:71] /*v[838:839]*/, s[52:53], v[244:245] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x3c0
	v_exp_f32_e32 v68 /*v836*/, v194
	s_set_vgpr_msb 0xc052
	v_pk_fma_f32 v[16:17] /*v[272:273]*/, v[192:193] /*v[704:705]*/, s[52:53], v[40:41] /*v[296:297]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[18:19] /*v[274:275]*/, v[194:195] /*v[706:707]*/, s[52:53], v[40:41] /*v[296:297]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[20:21] /*v[276:277]*/, v[196:197] /*v[708:709]*/, s[52:53], v[40:41] /*v[296:297]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x52c0
	v_exp_f32_e32 v120 /*v888*/, v195
	v_exp_f32_e32 v122 /*v890*/, v196
	s_set_vgpr_msb 0xc0c1
	v_exp_f32_e32 v65 /*v833*/, v16 /*v272*/
	v_exp_f32_e32 v67 /*v835*/, v17 /*v273*/
	v_exp_f32_e32 v69 /*v837*/, v18 /*v274*/
	s_set_vgpr_msb 0xc152
	v_pk_fma_f32 v[16:17] /*v[272:273]*/, v[198:199] /*v[710:711]*/, s[52:53], v[40:41] /*v[296:297]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x52c1
	v_exp_f32_e32 v121 /*v889*/, v19 /*v275*/
	v_exp_f32_e32 v123 /*v891*/, v20 /*v276*/
	s_set_vgpr_msb 0xc152
	v_pk_fma_f32 v[18:19] /*v[274:275]*/, v[200:201] /*v[712:713]*/, s[52:53], v[40:41] /*v[296:297]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x52c1
	v_exp_f32_e32 v125 /*v893*/, v21 /*v277*/
	v_nop
	s_set_vgpr_msb 0xc152
	v_pk_fma_f32 v[20:21] /*v[276:277]*/, v[202:203] /*v[714:715]*/, s[52:53], v[40:41] /*v[296:297]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5203
	v_pk_fma_f32 v[194:195], v[72:73] /*v[840:841]*/, s[52:53], v[244:245] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x3c0
	v_exp_f32_e32 v124 /*v892*/, v197
	v_exp_f32_e32 v126 /*v894*/, v192
	s_set_vgpr_msb 0xc003
	v_pk_fma_f32 v[196:197], v[74:75] /*v[842:843]*/, s[52:53], v[244:245] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x3c0
	v_exp_f32_e32 v128 /*v896*/, v193
	v_nop
	s_set_vgpr_msb 0xc003
	v_pk_fma_f32 v[192:193], v[76:77] /*v[844:845]*/, s[52:53], v[244:245] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x3c1
	v_exp_f32_e32 v127 /*v895*/, v16 /*v272*/
	v_exp_f32_e32 v129 /*v897*/, v17 /*v273*/
	v_exp_f32_e32 v131 /*v899*/, v18 /*v274*/
	s_set_vgpr_msb 0xc152
	v_pk_fma_f32 v[16:17] /*v[272:273]*/, v[204:205] /*v[716:717]*/, s[52:53], v[40:41] /*v[296:297]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x52c1
	v_exp_f32_e32 v135 /*v903*/, v19 /*v275*/
	v_exp_f32_e32 v139 /*v907*/, v20 /*v276*/
	s_set_vgpr_msb 0xc152
	v_pk_fma_f32 v[18:19] /*v[274:275]*/, v[206:207] /*v[718:719]*/, s[52:53], v[40:41] /*v[296:297]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x52c1
	v_exp_f32_e32 v143 /*v911*/, v21 /*v277*/
	v_nop
	s_set_vgpr_msb 0xc152
	v_pk_fma_f32 v[20:21] /*v[276:277]*/, v[208:209] /*v[720:721]*/, s[52:53], v[40:41] /*v[296:297]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x52c0
	v_exp_f32_e32 v130 /*v898*/, v194
	v_exp_f32_e32 v134 /*v902*/, v195
	v_exp_f32_e32 v138 /*v906*/, v196
	s_set_vgpr_msb 0xc003
	v_pk_fma_f32 v[194:195], v[78:79] /*v[846:847]*/, s[52:53], v[244:245] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x3c0
	v_exp_f32_e32 v142 /*v910*/, v197
	v_exp_f32_e32 v148 /*v916*/, v192
	s_set_vgpr_msb 0xc003
	v_pk_fma_f32 v[196:197], v[80:81] /*v[848:849]*/, s[52:53], v[244:245] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x3c0
	v_exp_f32_e32 v162 /*v930*/, v193
	v_nop
	s_set_vgpr_msb 0xc003
	v_pk_fma_f32 v[192:193], v[82:83] /*v[850:851]*/, s[52:53], v[244:245] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x3c1
	v_exp_f32_e32 v149 /*v917*/, v16 /*v272*/
	v_exp_f32_e32 v163 /*v931*/, v17 /*v273*/
	v_exp_f32_e32 v165 /*v933*/, v18 /*v274*/
	s_set_vgpr_msb 0xc152
	v_pk_fma_f32 v[16:17] /*v[272:273]*/, v[210:211] /*v[722:723]*/, s[52:53], v[40:41] /*v[296:297]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x52c1
	v_exp_f32_e32 v195 /*v963*/, v19 /*v275*/
	v_exp_f32_e32 v71 /*v839*/, v20 /*v276*/
	s_set_vgpr_msb 0xc152
	v_pk_fma_f32 v[18:19] /*v[274:275]*/, v[212:213] /*v[724:725]*/, s[52:53], v[40:41] /*v[296:297]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x52c1
	v_exp_f32_e32 v73 /*v841*/, v21 /*v277*/
	v_nop
	s_set_vgpr_msb 0xc152
	v_pk_fma_f32 v[20:21] /*v[276:277]*/, v[214:215] /*v[726:727]*/, s[52:53], v[40:41] /*v[296:297]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x52c0
	v_exp_f32_e32 v164 /*v932*/, v194
	v_exp_f32_e32 v194 /*v962*/, v195
	v_exp_f32_e32 v70 /*v838*/, v196
	s_set_vgpr_msb 0xc003
	v_pk_fma_f32 v[194:195], v[84:85] /*v[852:853]*/, s[52:53], v[244:245] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x3c0
	v_exp_f32_e32 v72 /*v840*/, v197
	v_exp_f32_e32 v74 /*v842*/, v192
	s_set_vgpr_msb 0xc003
	v_pk_fma_f32 v[196:197], v[86:87] /*v[854:855]*/, s[52:53], v[244:245] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x3c0
	v_exp_f32_e32 v78 /*v846*/, v193
	v_nop
	s_set_vgpr_msb 0xc003
	v_pk_fma_f32 v[192:193], v[88:89] /*v[856:857]*/, s[52:53], v[244:245] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x3c1
	v_exp_f32_e32 v75 /*v843*/, v16 /*v272*/
	v_exp_f32_e32 v79 /*v847*/, v17 /*v273*/
	v_exp_f32_e32 v83 /*v851*/, v18 /*v274*/
	s_set_vgpr_msb 0xc152
	v_pk_fma_f32 v[16:17] /*v[272:273]*/, v[216:217] /*v[728:729]*/, s[52:53], v[40:41] /*v[296:297]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x52c1
	v_exp_f32_e32 v89 /*v857*/, v19 /*v275*/
	v_exp_f32_e32 v133 /*v901*/, v20 /*v276*/
	s_set_vgpr_msb 0xc152
	v_pk_fma_f32 v[18:19] /*v[274:275]*/, v[218:219] /*v[730:731]*/, s[52:53], v[40:41] /*v[296:297]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x52c1
	v_exp_f32_e32 v137 /*v905*/, v21 /*v277*/
	v_nop
	s_set_vgpr_msb 0xc152
	v_pk_fma_f32 v[20:21] /*v[276:277]*/, v[220:221] /*v[732:733]*/, s[52:53], v[40:41] /*v[296:297]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x52c0
	v_exp_f32_e32 v82 /*v850*/, v194
	v_exp_f32_e32 v88 /*v856*/, v195
	v_exp_f32_e32 v132 /*v900*/, v196
	s_set_vgpr_msb 0xc003
	v_pk_fma_f32 v[194:195], v[90:91] /*v[858:859]*/, s[52:53], v[244:245] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x3c0
	v_exp_f32_e32 v136 /*v904*/, v197
	v_exp_f32_e32 v140 /*v908*/, v192
	s_set_vgpr_msb 0xc003
	v_pk_fma_f32 v[196:197], v[92:93] /*v[860:861]*/, s[52:53], v[244:245] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x3c0
	v_exp_f32_e32 v144 /*v912*/, v193
	v_nop
	s_set_vgpr_msb 0xc003
	v_pk_fma_f32 v[192:193], v[94:95] /*v[862:863]*/, s[52:53], v[244:245] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x3c1
	v_exp_f32_e32 v141 /*v909*/, v16 /*v272*/
	v_exp_f32_e32 v145 /*v913*/, v17 /*v273*/
	v_exp_f32_e32 v151 /*v919*/, v18 /*v274*/
	s_set_vgpr_msb 0xc152
	v_pk_fma_f32 v[16:17] /*v[272:273]*/, v[222:223] /*v[734:735]*/, s[52:53], v[40:41] /*v[296:297]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x52c1
	v_exp_f32_e32 v167 /*v935*/, v19 /*v275*/
	v_exp_f32_e32 v175 /*v943*/, v20 /*v276*/
	s_set_vgpr_msb 0xc152
	v_pk_fma_f32 v[18:19] /*v[274:275]*/, v[224:225] /*v[736:737]*/, s[52:53], v[40:41] /*v[296:297]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x52c1
	v_exp_f32_e32 v197 /*v965*/, v21 /*v277*/
	v_nop
	s_set_vgpr_msb 0xc152
	v_pk_fma_f32 v[20:21] /*v[276:277]*/, v[226:227] /*v[738:739]*/, s[52:53], v[40:41] /*v[296:297]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x52c0
	v_exp_f32_e32 v150 /*v918*/, v194
	v_exp_f32_e32 v166 /*v934*/, v195
	v_exp_f32_e32 v174 /*v942*/, v196
	s_set_vgpr_msb 0xc003
	v_pk_fma_f32 v[194:195], v[96:97] /*v[864:865]*/, s[52:53], v[244:245] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x3c0
	v_exp_f32_e32 v196 /*v964*/, v197
	v_exp_f32_e32 v198 /*v966*/, v192
	s_set_vgpr_msb 0xc003
	v_pk_fma_f32 v[196:197], v[98:99] /*v[866:867]*/, s[52:53], v[244:245] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x3c0
	v_exp_f32_e32 v224 /*v992*/, v193
	v_nop
	s_set_vgpr_msb 0xc003
	v_pk_fma_f32 v[192:193], v[100:101] /*v[868:869]*/, s[52:53], v[244:245] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x3c1
	v_exp_f32_e32 v199 /*v967*/, v16 /*v272*/
	v_exp_f32_e32 v225 /*v993*/, v17 /*v273*/
	v_exp_f32_e32 v77 /*v845*/, v18 /*v274*/
	s_set_vgpr_msb 0xc152
	v_pk_fma_f32 v[16:17] /*v[272:273]*/, v[228:229] /*v[740:741]*/, s[52:53], v[40:41] /*v[296:297]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x52c1
	v_exp_f32_e32 v81 /*v849*/, v19 /*v275*/
	v_exp_f32_e32 v85 /*v853*/, v20 /*v276*/
	s_set_vgpr_msb 0xc152
	v_pk_fma_f32 v[18:19] /*v[274:275]*/, v[230:231] /*v[742:743]*/, s[52:53], v[40:41] /*v[296:297]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x52c1
	v_exp_f32_e32 v91 /*v859*/, v21 /*v277*/
	v_nop
	s_set_vgpr_msb 0xc152
	v_pk_fma_f32 v[20:21] /*v[276:277]*/, v[232:233] /*v[744:745]*/, s[52:53], v[40:41] /*v[296:297]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x52c0
	v_exp_f32_e32 v76 /*v844*/, v194
	v_exp_f32_e32 v80 /*v848*/, v195
	v_exp_f32_e32 v84 /*v852*/, v196
	s_set_vgpr_msb 0xc003
	v_pk_fma_f32 v[194:195], v[102:103] /*v[870:871]*/, s[52:53], v[244:245] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x3c0
	v_exp_f32_e32 v90 /*v858*/, v197
	v_exp_f32_e32 v94 /*v862*/, v192
	s_set_vgpr_msb 0xc003
	v_pk_fma_f32 v[196:197], v[104:105] /*v[872:873]*/, s[52:53], v[244:245] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x3c0
	v_exp_f32_e32 v100 /*v868*/, v193
	v_nop
	s_set_vgpr_msb 0xc003
	v_pk_fma_f32 v[192:193], v[106:107] /*v[874:875]*/, s[52:53], v[244:245] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x3c1
	v_exp_f32_e32 v95 /*v863*/, v16 /*v272*/
	v_exp_f32_e32 v101 /*v869*/, v17 /*v273*/
	v_exp_f32_e32 v103 /*v871*/, v18 /*v274*/
	s_set_vgpr_msb 0xc152
	v_pk_fma_f32 v[16:17] /*v[272:273]*/, v[234:235] /*v[746:747]*/, s[52:53], v[40:41] /*v[296:297]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x52c1
	v_exp_f32_e32 v147 /*v915*/, v19 /*v275*/
	v_exp_f32_e32 v153 /*v921*/, v20 /*v276*/
	s_set_vgpr_msb 0xc152
	v_pk_fma_f32 v[18:19] /*v[274:275]*/, v[236:237] /*v[748:749]*/, s[52:53], v[40:41] /*v[296:297]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x52c1
	v_exp_f32_e32 v169 /*v937*/, v21 /*v277*/
	v_nop
	s_set_vgpr_msb 0xc152
	v_pk_fma_f32 v[20:21] /*v[276:277]*/, v[238:239] /*v[750:751]*/, s[52:53], v[40:41] /*v[296:297]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x52c0
	v_exp_f32_e32 v102 /*v870*/, v194
	v_exp_f32_e32 v146 /*v914*/, v195
	v_exp_f32_e32 v152 /*v920*/, v196
	s_set_vgpr_msb 0xc003
	v_pk_fma_f32 v[194:195], v[108:109] /*v[876:877]*/, s[52:53], v[244:245] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x3c0
	v_exp_f32_e32 v168 /*v936*/, v197
	v_exp_f32_e32 v178 /*v946*/, v192
	s_set_vgpr_msb 0xc003
	v_pk_fma_f32 v[196:197], v[110:111] /*v[878:879]*/, s[52:53], v[244:245] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x3c0
	v_exp_f32_e32 v200 /*v968*/, v193
	v_nop
	s_set_vgpr_msb 0xc003
	v_pk_fma_f32 v[192:193], v[112:113] /*v[880:881]*/, s[52:53], v[244:245] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x3c1
	v_exp_f32_e32 v179 /*v947*/, v16 /*v272*/
	v_exp_f32_e32 v201 /*v969*/, v17 /*v273*/
	v_exp_f32_e32 v209 /*v977*/, v18 /*v274*/
	s_set_vgpr_msb 0xc152
	v_pk_fma_f32 v[16:17] /*v[272:273]*/, v[240:241] /*v[752:753]*/, s[52:53], v[40:41] /*v[296:297]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x52c1
	v_exp_f32_e32 v227 /*v995*/, v19 /*v275*/
	v_exp_f32_e32 v229 /*v997*/, v20 /*v276*/
	s_set_vgpr_msb 0xc152
	v_pk_fma_f32 v[18:19] /*v[274:275]*/, v[242:243] /*v[754:755]*/, s[52:53], v[40:41] /*v[296:297]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x52c1
	v_exp_f32_e32 v251 /*v1019*/, v21 /*v277*/
	v_nop
	s_set_vgpr_msb 0xc152
	v_pk_fma_f32 v[20:21] /*v[276:277]*/, v[244:245] /*v[756:757]*/, s[52:53], v[40:41] /*v[296:297]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x52c0
	v_exp_f32_e32 v208 /*v976*/, v194
	v_exp_f32_e32 v226 /*v994*/, v195
	v_exp_f32_e32 v228 /*v996*/, v196
	s_set_vgpr_msb 0xc003
	v_pk_fma_f32 v[194:195], v[114:115] /*v[882:883]*/, s[52:53], v[244:245] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x3c0
	v_exp_f32_e32 v250 /*v1018*/, v197
	v_exp_f32_e32 v86 /*v854*/, v192
	s_set_vgpr_msb 0xc003
	v_pk_fma_f32 v[196:197], v[116:117] /*v[884:885]*/, s[52:53], v[244:245] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x3c0
	v_exp_f32_e32 v92 /*v860*/, v193
	v_nop
	s_set_vgpr_msb 0xc003
	v_pk_fma_f32 v[192:193], v[118:119] /*v[886:887]*/, s[52:53], v[244:245] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x3c1
	v_exp_f32_e32 v87 /*v855*/, v16 /*v272*/
	v_exp_f32_e32 v93 /*v861*/, v17 /*v273*/
	v_exp_f32_e32 v97 /*v865*/, v18 /*v274*/
	s_set_vgpr_msb 0xc152
	v_pk_fma_f32 v[16:17] /*v[272:273]*/, v[246:247] /*v[758:759]*/, s[52:53], v[40:41] /*v[296:297]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x52c1
	v_exp_f32_e32 v105 /*v873*/, v19 /*v275*/
	v_exp_f32_e32 v109 /*v877*/, v20 /*v276*/
	s_set_vgpr_msb 0xc152
	v_pk_fma_f32 v[18:19] /*v[274:275]*/, v[248:249] /*v[760:761]*/, s[52:53], v[40:41] /*v[296:297]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x52c1
	v_exp_f32_e32 v115 /*v883*/, v21 /*v277*/
	v_nop
	s_set_vgpr_msb 0xc152
	v_pk_fma_f32 v[20:21] /*v[276:277]*/, v[250:251] /*v[762:763]*/, s[52:53], v[40:41] /*v[296:297]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x52c0
	v_exp_f32_e32 v96 /*v864*/, v194
	v_exp_f32_e32 v104 /*v872*/, v195
	v_exp_f32_e32 v108 /*v876*/, v196
	s_set_vgpr_msb 0xc003
	v_pk_fma_f32 v[194:195], v[154:155] /*v[922:923]*/, s[52:53], v[244:245] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x3c0
	v_exp_f32_e32 v114 /*v882*/, v197
	v_exp_f32_e32 v116 /*v884*/, v192
	s_set_vgpr_msb 0xc003
	v_pk_fma_f32 v[196:197], v[156:157] /*v[924:925]*/, s[52:53], v[244:245] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x3c0
	v_exp_f32_e32 v170 /*v938*/, v193
	v_nop
	s_set_vgpr_msb 0xc003
	v_pk_fma_f32 v[192:193], v[158:159] /*v[926:927]*/, s[52:53], v[244:245] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x3c1
	v_exp_f32_e32 v117 /*v885*/, v16 /*v272*/
	v_exp_f32_e32 v171 /*v939*/, v17 /*v273*/
	v_exp_f32_e32 v181 /*v949*/, v18 /*v274*/
	s_set_vgpr_msb 0xc152
	v_pk_fma_f32 v[16:17] /*v[272:273]*/, v[252:253] /*v[764:765]*/, s[52:53], v[40:41] /*v[296:297]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x52c1
	v_exp_f32_e32 v203 /*v971*/, v19 /*v275*/
	v_exp_f32_e32 v211 /*v979*/, v20 /*v276*/
	s_set_vgpr_msb 0xc152
	v_pk_fma_f32 v[18:19] /*v[274:275]*/, v[254:255] /*v[766:767]*/, s[52:53], v[40:41] /*v[296:297]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x52c1
	v_exp_f32_e32 v231 /*v999*/, v21 /*v277*/
	v_nop
	s_set_vgpr_msb 0xc153
	v_pk_fma_f32 v[20:21] /*v[276:277]*/, v[0:1] /*v[768:769]*/, s[52:53], v[40:41] /*v[296:297]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x53c0
	v_exp_f32_e32 v180 /*v948*/, v194
	v_exp_f32_e32 v202 /*v970*/, v195
	v_exp_f32_e32 v210 /*v978*/, v196
	s_set_vgpr_msb 0xc003
	v_pk_fma_f32 v[194:195], v[160:161] /*v[928:929]*/, s[52:53], v[244:245] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x3c0
	v_exp_f32_e32 v230 /*v998*/, v197
	v_exp_f32_e32 v238 /*v1006*/, v192
	s_set_vgpr_msb 0xc003
	v_pk_fma_f32 v[196:197], v[182:183] /*v[950:951]*/, s[52:53], v[244:245] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x3c0
	v_exp_f32_e32 v252 /*v1020*/, v193
	v_nop
	s_set_vgpr_msb 0xc003
	v_pk_fma_f32 v[192:193], v[184:185] /*v[952:953]*/, s[52:53], v[244:245] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x3c1
	v_exp_f32_e32 v239 /*v1007*/, v16 /*v272*/
	v_exp_f32_e32 v253 /*v1021*/, v17 /*v273*/
	v_exp_f32_e32 v177 /*v945*/, v18 /*v274*/
	s_set_vgpr_msb 0xc153
	v_pk_fma_f32 v[16:17] /*v[272:273]*/, v[2:3] /*v[770:771]*/, s[52:53], v[40:41] /*v[296:297]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5301
	v_exp_f32_e32 v207, v19 /*v275*/
	s_set_vgpr_msb 0x1c1
	v_exp_f32_e32 v99 /*v867*/, v20 /*v276*/
	s_set_vgpr_msb 0xc153
	v_pk_fma_f32 v[18:19] /*v[274:275]*/, v[4:5] /*v[772:773]*/, s[52:53], v[40:41] /*v[296:297]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x53c1
	v_exp_f32_e32 v107 /*v875*/, v21 /*v277*/
	v_nop
	s_set_vgpr_msb 0xc153
	v_pk_fma_f32 v[20:21] /*v[276:277]*/, v[6:7] /*v[774:775]*/, s[52:53], v[40:41] /*v[296:297]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x53c0
	v_exp_f32_e32 v176 /*v944*/, v194
	s_set_vgpr_msb 0xc000
	v_exp_f32_e32 v206, v195
	s_set_vgpr_msb 0xc0
	v_exp_f32_e32 v98 /*v866*/, v196
	s_set_vgpr_msb 0xc003
	v_pk_fma_f32 v[194:195], v[186:187] /*v[954:955]*/, s[52:53], v[244:245] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x3c0
	v_exp_f32_e32 v106 /*v874*/, v197
	v_exp_f32_e32 v110 /*v878*/, v192
	s_set_vgpr_msb 0xc003
	v_pk_fma_f32 v[196:197], v[188:189] /*v[956:957]*/, s[52:53], v[244:245] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x3c0
	v_exp_f32_e32 v118 /*v886*/, v193
	v_nop
	s_set_vgpr_msb 0xc003
	v_pk_fma_f32 v[192:193], v[212:213] /*v[980:981]*/, s[52:53], v[244:245] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x3c1
	v_exp_f32_e32 v111 /*v879*/, v16 /*v272*/
	v_exp_f32_e32 v119 /*v887*/, v17 /*v273*/
	v_exp_f32_e32 v157 /*v925*/, v18 /*v274*/
	s_set_vgpr_msb 0xc153
	v_pk_fma_f32 v[16:17] /*v[272:273]*/, v[8:9] /*v[776:777]*/, s[52:53], v[40:41] /*v[296:297]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x53c1
	v_exp_f32_e32 v173 /*v941*/, v19 /*v275*/
	v_exp_f32_e32 v183 /*v951*/, v20 /*v276*/
	s_set_vgpr_msb 0xc153
	v_pk_fma_f32 v[18:19] /*v[274:275]*/, v[10:11] /*v[778:779]*/, s[52:53], v[40:41] /*v[296:297]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x53c1
	v_exp_f32_e32 v205 /*v973*/, v21 /*v277*/
	v_nop
	s_set_vgpr_msb 0xc153
	v_pk_fma_f32 v[20:21] /*v[276:277]*/, v[12:13] /*v[780:781]*/, s[52:53], v[40:41] /*v[296:297]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x53c0
	v_exp_f32_e32 v156 /*v924*/, v194
	v_exp_f32_e32 v172 /*v940*/, v195
	v_exp_f32_e32 v182 /*v950*/, v196
	s_set_vgpr_msb 0xc003
	v_pk_fma_f32 v[194:195], v[214:215] /*v[982:983]*/, s[52:53], v[244:245] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x3c0
	v_exp_f32_e32 v204 /*v972*/, v197
	v_exp_f32_e32 v212 /*v980*/, v192
	s_set_vgpr_msb 0xc003
	v_pk_fma_f32 v[196:197], v[216:217] /*v[984:985]*/, s[52:53], v[244:245] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x3c0
	v_exp_f32_e32 v232 /*v1000*/, v193
	v_nop
	s_set_vgpr_msb 0xc003
	v_pk_fma_f32 v[192:193], v[218:219] /*v[986:987]*/, s[52:53], v[244:245] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x3c1
	v_exp_f32_e32 v213 /*v981*/, v16 /*v272*/
	v_exp_f32_e32 v233 /*v1001*/, v17 /*v273*/
	v_exp_f32_e32 v241 /*v1009*/, v18 /*v274*/
	s_set_vgpr_msb 0xc153
	v_pk_fma_f32 v[16:17] /*v[272:273]*/, v[14:15] /*v[782:783]*/, s[52:53], v[40:41] /*v[296:297]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x53c1
	v_exp_f32_e32 v255 /*v1023*/, v19 /*v275*/
	s_set_vgpr_msb 0xc101
	v_exp_f32_e32 v199, v20 /*v276*/
	s_set_vgpr_msb 0x153
	v_pk_fma_f32 v[18:19] /*v[274:275]*/, v[16:17] /*v[784:785]*/, s[52:53], v[40:41] /*v[296:297]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5301
	v_exp_f32_e32 v209, v21 /*v277*/
	v_nop
	s_set_vgpr_msb 0x153
	v_pk_fma_f32 v[20:21] /*v[276:277]*/, v[18:19] /*v[786:787]*/, s[52:53], v[40:41] /*v[296:297]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x53c0
	v_exp_f32_e32 v240 /*v1008*/, v194
	v_exp_f32_e32 v254 /*v1022*/, v195
	s_set_vgpr_msb 0xc000
	v_exp_f32_e32 v198, v196
	s_set_vgpr_msb 3
	v_pk_fma_f32 v[194:195], v[242:243] /*v[1010:1011]*/, s[52:53], v[244:245] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x300
	v_exp_f32_e32 v208, v197
	v_exp_f32_e32 v210, v192
	s_set_vgpr_msb 3
	v_pk_fma_f32 v[196:197], v[244:245] /*v[1012:1013]*/, s[52:53], v[244:245] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x300
	v_exp_f32_e32 v224, v193
	v_nop
	s_set_vgpr_msb 3
	v_pk_fma_f32 v[192:193], v[246:247] /*v[1014:1015]*/, s[52:53], v[244:245] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x301
	v_exp_f32_e32 v211, v16 /*v272*/
	v_exp_f32_e32 v225, v17 /*v273*/
	s_set_vgpr_msb 0x1c1
	v_exp_f32_e32 v113 /*v881*/, v18 /*v274*/
	s_set_vgpr_msb 0xc153
	v_pk_fma_f32 v[16:17] /*v[272:273]*/, v[20:21] /*v[788:789]*/, s[52:53], v[40:41] /*v[296:297]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x53c1
	v_exp_f32_e32 v155 /*v923*/, v19 /*v275*/
	v_exp_f32_e32 v159 /*v927*/, v20 /*v276*/
	s_set_vgpr_msb 0xc153
	v_pk_fma_f32 v[18:19] /*v[274:275]*/, v[22:23] /*v[790:791]*/, s[52:53], v[40:41] /*v[296:297]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x53c1
	v_exp_f32_e32 v185 /*v953*/, v21 /*v277*/
	v_nop
	s_set_vgpr_msb 0xc153
	v_pk_fma_f32 v[20:21] /*v[276:277]*/, v[24:25] /*v[792:793]*/, s[52:53], v[40:41] /*v[296:297]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x53c0
	v_exp_f32_e32 v112 /*v880*/, v194
	v_exp_f32_e32 v154 /*v922*/, v195
	v_exp_f32_e32 v158 /*v926*/, v196
	s_set_vgpr_msb 0xc003
	v_pk_fma_f32 v[194:195], v[248:249] /*v[1016:1017]*/, s[52:53], v[244:245] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x3c0
	v_exp_f32_e32 v184 /*v952*/, v197
	v_nop
	s_set_vgpr_msb 0xc001
	v_pk_fma_f32 v[196:197], v[96:97] /*v[352:353]*/, s[52:53], v[244:245] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1c0
	v_exp_f32_e32 v206 /*v974*/, v193
	s_set_vgpr_msb 0xc001
	v_pk_fma_f32 v[200:201], v[98:99] /*v[354:355]*/, s[52:53], v[244:245] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1c1
	v_exp_f32_e32 v189 /*v957*/, v16 /*v272*/
	v_exp_f32_e32 v207 /*v975*/, v17 /*v273*/
	v_exp_f32_e32 v215 /*v983*/, v18 /*v274*/
	s_set_vgpr_msb 0xc153
	v_pk_fma_f32 v[16:17] /*v[272:273]*/, v[26:27] /*v[794:795]*/, s[52:53], v[40:41] /*v[296:297]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x53c1
	v_exp_f32_e32 v235 /*v1003*/, v19 /*v275*/
	v_exp_f32_e32 v243 /*v1011*/, v20 /*v276*/
	s_set_vgpr_msb 0xc153
	v_pk_fma_f32 v[18:19] /*v[274:275]*/, v[28:29] /*v[796:797]*/, s[52:53], v[40:41] /*v[296:297]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5301
	v_exp_f32_e32 v193, v21 /*v277*/
	v_nop
	s_set_vgpr_msb 0x153
	v_pk_fma_f32 v[20:21] /*v[276:277]*/, v[30:31] /*v[798:799]*/, s[52:53], v[40:41] /*v[296:297]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x53c0
	v_exp_f32_e32 v188 /*v956*/, v192
	v_exp_f32_e32 v214 /*v982*/, v194
	v_exp_f32_e32 v234 /*v1002*/, v195
	v_exp_f32_e32 v242 /*v1010*/, v196
	s_set_vgpr_msb 0xc001
	v_pk_fma_f32 v[194:195], v[100:101] /*v[356:357]*/, s[52:53], v[244:245] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x100
	v_exp_f32_e32 v192, v197
	v_nop
	s_set_vgpr_msb 1
	v_pk_fma_f32 v[196:197], v[102:103] /*v[358:359]*/, s[52:53], v[244:245] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x100
	v_exp_f32_e32 v212, v201
	s_set_vgpr_msb 1
	v_pk_fma_f32 v[202:203], v[64:65] /*v[320:321]*/, s[52:53], v[244:245] op_sel_hi:[1,0,0]
	v_exp_f32_e32 v201, v16 /*v272*/
	v_exp_f32_e32 v213, v17 /*v273*/
	v_exp_f32_e32 v219, v18 /*v274*/
	s_set_vgpr_msb 0x153
	v_pk_fma_f32 v[16:17] /*v[272:273]*/, v[32:33] /*v[800:801]*/, s[52:53], v[40:41] /*v[296:297]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5301
	v_exp_f32_e32 v227, v19 /*v275*/
	v_exp_f32_e32 v229, v20 /*v276*/
	s_set_vgpr_msb 0x153
	v_pk_fma_f32 v[18:19] /*v[274:275]*/, v[34:35] /*v[802:803]*/, s[52:53], v[40:41] /*v[296:297]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5301
	v_exp_f32_e32 v239, v21 /*v277*/
	v_nop
	s_set_vgpr_msb 0x153
	v_pk_fma_f32 v[20:21] /*v[276:277]*/, v[36:37] /*v[804:805]*/, s[52:53], v[40:41] /*v[296:297]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5300
	v_exp_f32_e32 v218, v194
	v_exp_f32_e32 v226, v195
	v_exp_f32_e32 v228, v196
	s_set_vgpr_msb 1
	v_pk_fma_f32 v[194:195], v[66:67] /*v[322:323]*/, s[52:53], v[244:245] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x100
	v_exp_f32_e32 v238, v197
	s_set_vgpr_msb 0xc0
	v_exp_f32_e32 v160 /*v928*/, v202
	s_set_vgpr_msb 0xc001
	v_pk_fma_f32 v[196:197], v[68:69] /*v[324:325]*/, s[52:53], v[244:245] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1c0
	v_exp_f32_e32 v186 /*v954*/, v203
	v_nop
	s_set_vgpr_msb 0xc001
	v_pk_fma_f32 v[202:203], v[70:71] /*v[326:327]*/, s[52:53], v[244:245] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[204:205], v[32:33] /*v[288:289]*/, s[52:53], v[244:245] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1c1
	v_exp_f32_e32 v161 /*v929*/, v16 /*v272*/
	v_exp_f32_e32 v187 /*v955*/, v17 /*v273*/
	v_exp_f32_e32 v191 /*v959*/, v18 /*v274*/
	s_set_vgpr_msb 0xc153
	v_pk_fma_f32 v[16:17] /*v[272:273]*/, v[38:39] /*v[806:807]*/, s[52:53], v[40:41] /*v[296:297]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x53c1
	v_exp_f32_e32 v217 /*v985*/, v19 /*v275*/
	v_exp_f32_e32 v221 /*v989*/, v20 /*v276*/
	s_set_vgpr_msb 0xc153
	v_pk_fma_f32 v[18:19] /*v[274:275]*/, v[40:41] /*v[808:809]*/, s[52:53], v[40:41] /*v[296:297]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x53c1
	v_exp_f32_e32 v237 /*v1005*/, v21 /*v277*/
	v_nop
	s_set_vgpr_msb 0xc153
	v_pk_fma_f32 v[20:21] /*v[276:277]*/, v[42:43] /*v[810:811]*/, s[52:53], v[40:41] /*v[296:297]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x53c0
	v_exp_f32_e32 v190 /*v958*/, v194
	v_exp_f32_e32 v216 /*v984*/, v195
	v_exp_f32_e32 v220 /*v988*/, v196
	v_exp_f32_e32 v236 /*v1004*/, v197
	v_exp_f32_e32 v244 /*v1012*/, v202
	s_set_vgpr_msb 0xc001
	v_pk_fma_f32 v[196:197], v[34:35] /*v[290:291]*/, s[52:53], v[244:245] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x100
	v_exp_f32_e32 v194, v203
	v_exp_f32_e32 v202, v204
	s_set_vgpr_msb 1
	v_pk_fma_f32 v[216:217], v[36:37] /*v[292:293]*/, s[52:53], v[244:245] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x100
	v_exp_f32_e32 v214, v205
	v_nop
	s_set_vgpr_msb 1
	v_pk_fma_f32 v[204:205], v[38:39] /*v[294:295]*/, s[52:53], v[244:245] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1c1
	v_exp_f32_e32 v245 /*v1013*/, v16 /*v272*/
	s_set_vgpr_msb 0xc101
	v_exp_f32_e32 v195, v17 /*v273*/
	v_exp_f32_e32 v203, v18 /*v274*/
	s_set_vgpr_msb 0x153
	v_pk_fma_f32 v[16:17] /*v[272:273]*/, v[44:45] /*v[812:813]*/, s[52:53], v[40:41] /*v[296:297]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5301
	v_exp_f32_e32 v215, v19 /*v275*/
	v_exp_f32_e32 v221, v20 /*v276*/
	s_set_vgpr_msb 0x153
	v_pk_fma_f32 v[18:19] /*v[274:275]*/, v[46:47] /*v[814:815]*/, s[52:53], v[40:41] /*v[296:297]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5301
	v_exp_f32_e32 v231, v21 /*v277*/
	v_nop
	s_set_vgpr_msb 0x153
	v_pk_fma_f32 v[20:21] /*v[276:277]*/, v[48:49] /*v[816:817]*/, s[52:53], v[40:41] /*v[296:297]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5300
	v_exp_f32_e32 v220, v196
	v_exp_f32_e32 v230, v197
	v_exp_f32_e32 v234, v216
	s_set_vgpr_msb 1
	v_pk_fma_f32 v[196:197], v[24:25] /*v[280:281]*/, s[52:53], v[244:245] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x100
	v_exp_f32_e32 v240, v217
	v_exp_f32_e32 v242, v204
	s_set_vgpr_msb 1
	v_pk_fma_f32 v[216:217], v[26:27] /*v[282:283]*/, s[52:53], v[244:245] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x100
	v_exp_f32_e32 v248, v205
	v_nop
	s_set_vgpr_msb 1
	v_pk_fma_f32 v[204:205], v[28:29] /*v[284:285]*/, s[52:53], v[244:245] op_sel_hi:[1,0,0]
	v_exp_f32_e32 v235, v16 /*v272*/
	v_exp_f32_e32 v241, v17 /*v273*/
	v_exp_f32_e32 v243, v18 /*v274*/
	s_set_vgpr_msb 0x153
	v_pk_fma_f32 v[16:17] /*v[272:273]*/, v[50:51] /*v[818:819]*/, s[52:53], v[40:41] /*v[296:297]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5301
	v_exp_f32_e32 v249, v19 /*v275*/
	s_set_vgpr_msb 0x1c1
	v_exp_f32_e32 v193 /*v961*/, v20 /*v276*/
	s_set_vgpr_msb 0xc153
	v_pk_fma_f32 v[18:19] /*v[274:275]*/, v[52:53] /*v[820:821]*/, s[52:53], v[40:41] /*v[296:297]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x53c1
	v_exp_f32_e32 v219 /*v987*/, v21 /*v277*/
	v_nop
	s_set_vgpr_msb 0xc153
	v_pk_fma_f32 v[20:21] /*v[276:277]*/, v[54:55] /*v[822:823]*/, s[52:53], v[40:41] /*v[296:297]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x53c0
	v_exp_f32_e32 v192 /*v960*/, v196
	v_exp_f32_e32 v218 /*v986*/, v197
	s_set_vgpr_msb 0xc001
	v_pk_fma_f32 v[222:223], v[30:31] /*v[286:287]*/, s[52:53], v[244:245] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1c0
	v_exp_f32_e32 v246 /*v1014*/, v217
	s_set_vgpr_msb 0xc000
	v_exp_f32_e32 v196, v205
	s_set_vgpr_msb 0xc1
	v_exp_f32_e32 v223 /*v991*/, v16 /*v272*/
	v_exp_f32_e32 v247 /*v1015*/, v17 /*v273*/
	v_exp_f32_e32 v249 /*v1017*/, v18 /*v274*/
	s_set_vgpr_msb 0xc153
	v_pk_fma_f32 v[16:17] /*v[272:273]*/, v[56:57] /*v[824:825]*/, s[52:53], v[40:41] /*v[296:297]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5301
	v_exp_f32_e32 v197, v19 /*v275*/
	v_exp_f32_e32 v205, v20 /*v276*/
	s_set_vgpr_msb 0x153
	v_pk_fma_f32 v[18:19] /*v[274:275]*/, v[58:59] /*v[826:827]*/, s[52:53], v[40:41] /*v[296:297]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5301
	v_exp_f32_e32 v217, v21 /*v277*/
	v_nop
	s_set_vgpr_msb 0x153
	v_pk_fma_f32 v[20:21] /*v[276:277]*/, v[60:61] /*v[828:829]*/, s[52:53], v[40:41] /*v[296:297]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x53c0
	v_exp_f32_e32 v222 /*v990*/, v216
	v_exp_f32_e32 v248 /*v1016*/, v204
	s_set_vgpr_msb 0xc000
	v_exp_f32_e32 v204, v222
	v_exp_f32_e32 v216, v223
	v_exp_f32_e32 v222, v232
	v_exp_f32_e32 v232, v233
	s_set_vgpr_msb 1
	v_pk_fma_f32 v[252:253], v[22:23] /*v[278:279]*/, s[52:53], v[244:245] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x100
	v_exp_f32_e32 v244, v237
	v_exp_f32_e32 v250, v247
	s_set_vgpr_msb 1
	v_exp_f32_e32 v223, v16 /*v272*/
	v_exp_f32_e32 v233, v17 /*v273*/
	v_exp_f32_e32 v237, v18 /*v274*/
	v_exp_f32_e32 v245, v19 /*v275*/
	s_set_vgpr_msb 0x153
	v_pk_fma_f32 v[16:17] /*v[272:273]*/, v[62:63] /*v[830:831]*/, s[52:53], v[40:41] /*v[296:297]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5301
	v_exp_f32_e32 v247, v20 /*v276*/
	s_set_vgpr_msb 0x14f
	v_pk_add_f32 v[18:19] /*v[274:275]*/, v[64:65] /*v[832:833]*/, v[66:67] /*v[834:835]*/
	s_set_vgpr_msb 0x4f01
	v_exp_f32_e32 v251, v21 /*v277*/
	v_nop
	s_set_vgpr_msb 0x14f
	v_pk_add_f32 v[20:21] /*v[276:277]*/, v[120:121] /*v[888:889]*/, v[122:123] /*v[890:891]*/
	s_set_vgpr_msb 0x4f00
	v_exp_f32_e32 v200, v200
	v_exp_f32_e32 v254, v253
	s_set_vgpr_msb 1
	v_exp_f32_e32 v253, v16 /*v272*/
	v_exp_f32_e32 v255, v17 /*v273*/
	v_nop
	s_set_vgpr_msb 0x147
	v_pk_add_f32 v[16:17] /*v[272:273]*/, v[68:69] /*v[836:837]*/, v[18:19] /*v[274:275]*/
	s_set_vgpr_msb 0x474f
	v_pk_add_f32 v[18:19] /*v[274:275]*/, v[126:127] /*v[894:895]*/, v[128:129] /*v[896:897]*/
	s_set_vgpr_msb 0x4f47
	v_pk_add_f32 v[20:21] /*v[276:277]*/, v[124:125] /*v[892:893]*/, v[20:21] /*v[276:277]*/
	s_set_vgpr_msb 0x474f
	v_pk_add_f32 v[22:23] /*v[278:279]*/, v[134:135] /*v[902:903]*/, v[138:139] /*v[906:907]*/
	v_pk_add_f32 v[24:25] /*v[280:281]*/, v[148:149] /*v[916:917]*/, v[162:163] /*v[930:931]*/
	v_pk_add_f32 v[28:29] /*v[284:285]*/, v[74:75] /*v[842:843]*/, v[78:79] /*v[846:847]*/
	v_pk_add_f32 v[30:31] /*v[286:287]*/, v[88:89] /*v[856:857]*/, v[132:133] /*v[900:901]*/
	v_pk_add_f32 v[34:35] /*v[290:291]*/, v[166:167] /*v[934:935]*/, v[174:175] /*v[942:943]*/
	v_pk_add_f32 v[36:37] /*v[292:293]*/, v[198:199] /*v[966:967]*/, v[224:225] /*v[992:993]*/
	v_pk_add_f32 v[40:41] /*v[296:297]*/, v[94:95] /*v[862:863]*/, v[100:101] /*v[868:869]*/
	v_pk_add_f32 v[42:43] /*v[298:299]*/, v[146:147] /*v[914:915]*/, v[152:153] /*v[920:921]*/
	s_set_vgpr_msb 0x4f00
	v_exp_f32_e32 v236, v236
	v_exp_f32_e32 v246, v246
	s_set_vgpr_msb 0x4f
	v_pk_add_f32 v[26:27] /*v[282:283]*/, v[194:195] /*v[962:963]*/, v[70:71] /*v[838:839]*/
	s_set_vgpr_msb 0x4f47
	v_pk_add_f32 v[18:19] /*v[274:275]*/, v[130:131] /*v[898:899]*/, v[18:19] /*v[274:275]*/
	v_pk_add_f32 v[22:23] /*v[278:279]*/, v[142:143] /*v[910:911]*/, v[22:23] /*v[278:279]*/
	v_pk_add_f32 v[24:25] /*v[280:281]*/, v[164:165] /*v[932:933]*/, v[24:25] /*v[280:281]*/
	v_pk_add_f32 v[28:29] /*v[284:285]*/, v[82:83] /*v[850:851]*/, v[28:29] /*v[284:285]*/
	s_set_vgpr_msb 0x474f
	v_pk_add_f32 v[32:33] /*v[288:289]*/, v[140:141] /*v[908:909]*/, v[144:145] /*v[912:913]*/
	s_set_vgpr_msb 0x4f47
	v_pk_add_f32 v[30:31] /*v[286:287]*/, v[136:137] /*v[904:905]*/, v[30:31] /*v[286:287]*/
	s_set_vgpr_msb 0x474f
	v_pk_add_f32 v[38:39] /*v[294:295]*/, v[80:81] /*v[848:849]*/, v[84:85] /*v[852:853]*/
	s_set_vgpr_msb 0x4f47
	v_pk_add_f32 v[34:35] /*v[290:291]*/, v[196:197] /*v[964:965]*/, v[34:35] /*v[290:291]*/
	v_pk_add_f32 v[36:37] /*v[292:293]*/, v[76:77] /*v[844:845]*/, v[36:37] /*v[292:293]*/
	s_set_vgpr_msb 0x474f
	v_pk_add_f32 v[44:45] /*v[300:301]*/, v[178:179] /*v[946:947]*/, v[200:201] /*v[968:969]*/
	v_pk_add_f32 v[46:47] /*v[302:303]*/, v[226:227] /*v[994:995]*/, v[228:229] /*v[996:997]*/
	s_set_vgpr_msb 0x4f47
	v_pk_add_f32 v[40:41] /*v[296:297]*/, v[102:103] /*v[870:871]*/, v[40:41] /*v[296:297]*/
	s_set_vgpr_msb 0x474f
	v_pk_add_f32 v[64:65] /*v[320:321]*/, v[86:87] /*v[854:855]*/, v[92:93] /*v[860:861]*/
	s_set_vgpr_msb 0x4f47
	v_pk_add_f32 v[42:43] /*v[298:299]*/, v[168:169] /*v[936:937]*/, v[42:43] /*v[298:299]*/
	s_set_vgpr_msb 0x474f
	v_pk_add_f32 v[68:69] /*v[324:325]*/, v[116:117] /*v[884:885]*/, v[170:171] /*v[938:939]*/
	v_pk_add_f32 v[70:71] /*v[326:327]*/, v[202:203] /*v[970:971]*/, v[210:211] /*v[978:979]*/
	s_set_vgpr_msb 0x4f4c
	v_pk_add_f32 v[98:99] /*v[354:355]*/, v[206:207], v[98:99] /*v[866:867]*/
	s_set_vgpr_msb 0x4c4f
	v_pk_add_f32 v[100:101] /*v[356:357]*/, v[110:111] /*v[878:879]*/, v[118:119] /*v[886:887]*/
	s_set_vgpr_msb 0x4f8f
	v_pk_add_f32 v[198:199] /*v[710:711]*/, v[154:155] /*v[922:923]*/, v[158:159] /*v[926:927]*/
	v_pk_add_f32 v[200:201] /*v[712:713]*/, v[188:189] /*v[956:957]*/, v[206:207] /*v[974:975]*/
	s_set_vgpr_msb 0x8f80
	v_pk_add_f32 v[204:205] /*v[716:717]*/, v[200:201], v[212:213]
	v_pk_add_f32 v[206:207] /*v[718:719]*/, v[226:227], v[228:229]
	s_set_vgpr_msb 0x808f
	v_pk_add_f32 v[210:211] /*v[722:723]*/, v[216:217] /*v[984:985]*/, v[220:221] /*v[988:989]*/
	s_set_vgpr_msb 0x8f83
	v_pk_add_f32 v[212:213] /*v[724:725]*/, v[244:245] /*v[1012:1013]*/, v[194:195]
	s_set_vgpr_msb 0x8380
	v_pk_add_f32 v[216:217] /*v[728:729]*/, v[234:235], v[240:241]
	s_set_vgpr_msb 0x808c
	v_pk_add_f32 v[218:219] /*v[730:731]*/, v[248:249], v[192:193] /*v[960:961]*/
	s_set_vgpr_msb 0x8c80
	v_pk_add_f32 v[222:223] /*v[734:735]*/, v[196:197], v[204:205]
	v_pk_add_f32 v[224:225] /*v[736:737]*/, v[222:223], v[232:233]
	s_set_vgpr_msb 0x8045
	v_pk_add_f32 v[16:17] /*v[272:273]*/, v[16:17] /*v[272:273]*/, v[20:21] /*v[276:277]*/
	s_set_vgpr_msb 0x4547
	v_pk_add_f32 v[26:27] /*v[282:283]*/, v[72:73] /*v[840:841]*/, v[26:27] /*v[282:283]*/
	v_pk_add_f32 v[32:33] /*v[288:289]*/, v[150:151] /*v[918:919]*/, v[32:33] /*v[288:289]*/
	v_pk_add_f32 v[38:39] /*v[294:295]*/, v[90:91] /*v[858:859]*/, v[38:39] /*v[294:295]*/
	v_pk_add_f32 v[44:45] /*v[300:301]*/, v[208:209] /*v[976:977]*/, v[44:45] /*v[300:301]*/
	v_pk_add_f32 v[46:47] /*v[302:303]*/, v[250:251] /*v[1018:1019]*/, v[46:47] /*v[302:303]*/
	s_set_vgpr_msb 0x474f
	v_pk_add_f32 v[66:67] /*v[322:323]*/, v[104:105] /*v[872:873]*/, v[108:109] /*v[876:877]*/
	s_set_vgpr_msb 0x4f47
	v_pk_add_f32 v[64:65] /*v[320:321]*/, v[96:97] /*v[864:865]*/, v[64:65] /*v[320:321]*/
	s_set_vgpr_msb 0x474f
	v_pk_add_f32 v[96:97] /*v[352:353]*/, v[238:239] /*v[1006:1007]*/, v[252:253] /*v[1020:1021]*/
	s_set_vgpr_msb 0x4f47
	v_pk_add_f32 v[68:69] /*v[324:325]*/, v[180:181] /*v[948:949]*/, v[68:69] /*v[324:325]*/
	v_pk_add_f32 v[70:71] /*v[326:327]*/, v[230:231] /*v[998:999]*/, v[70:71] /*v[326:327]*/
	v_pk_add_f32 v[98:99] /*v[354:355]*/, v[106:107] /*v[874:875]*/, v[98:99] /*v[354:355]*/
	s_set_vgpr_msb 0x474f
	v_pk_add_f32 v[102:103] /*v[358:359]*/, v[172:173] /*v[940:941]*/, v[182:183] /*v[950:951]*/
	s_set_vgpr_msb 0x4f8f
	v_pk_add_f32 v[192:193] /*v[704:705]*/, v[212:213] /*v[980:981]*/, v[232:233] /*v[1000:1001]*/
	s_set_vgpr_msb 0x8f83
	v_pk_add_f32 v[194:195] /*v[706:707]*/, v[254:255] /*v[1022:1023]*/, v[198:199]
	s_set_vgpr_msb 0x8347
	v_pk_add_f32 v[100:101] /*v[356:357]*/, v[156:157] /*v[924:925]*/, v[100:101] /*v[356:357]*/
	s_set_vgpr_msb 0x478f
	v_pk_add_f32 v[202:203] /*v[714:715]*/, v[234:235] /*v[1002:1003]*/, v[242:243] /*v[1010:1011]*/
	s_set_vgpr_msb 0x8f8b
	v_pk_add_f32 v[198:199] /*v[710:711]*/, v[184:185] /*v[952:953]*/, v[198:199] /*v[710:711]*/
	v_pk_add_f32 v[200:201] /*v[712:713]*/, v[214:215] /*v[982:983]*/, v[200:201] /*v[712:713]*/
	s_set_vgpr_msb 0x8b88
	v_pk_add_f32 v[204:205] /*v[716:717]*/, v[218:219], v[204:205] /*v[716:717]*/
	s_set_vgpr_msb 0x888f
	v_pk_add_f32 v[208:209] /*v[720:721]*/, v[160:161] /*v[928:929]*/, v[186:187] /*v[954:955]*/
	s_set_vgpr_msb 0x8f88
	v_pk_add_f32 v[206:207] /*v[718:719]*/, v[238:239], v[206:207] /*v[718:719]*/
	s_set_vgpr_msb 0x8880
	v_pk_add_f32 v[214:215] /*v[726:727]*/, v[214:215], v[220:221]
	s_set_vgpr_msb 0x808b
	v_pk_add_f32 v[210:211] /*v[722:723]*/, v[236:237] /*v[1004:1005]*/, v[210:211] /*v[722:723]*/
	s_set_vgpr_msb 0x8b88
	v_pk_add_f32 v[212:213] /*v[724:725]*/, v[202:203], v[212:213] /*v[724:725]*/
	v_pk_add_f32 v[216:217] /*v[728:729]*/, v[242:243], v[216:217] /*v[728:729]*/
	s_set_vgpr_msb 0x888f
	v_pk_add_f32 v[220:221] /*v[732:733]*/, v[222:223] /*v[990:991]*/, v[246:247] /*v[1014:1015]*/
	s_set_vgpr_msb 0x8f8b
	v_pk_add_f32 v[218:219] /*v[730:731]*/, v[218:219] /*v[986:987]*/, v[218:219] /*v[730:731]*/
	s_set_vgpr_msb 0x8b80
	v_pk_add_f32 v[226:227] /*v[738:739]*/, v[244:245], v[246:247]
	s_set_vgpr_msb 0x8088
	v_pk_add_f32 v[222:223] /*v[734:735]*/, v[216:217], v[222:223] /*v[734:735]*/
	v_pk_add_f32 v[224:225] /*v[736:737]*/, v[236:237], v[224:225] /*v[736:737]*/
	s_set_vgpr_msb 0x8845
	v_pk_add_f32 v[22:23] /*v[278:279]*/, v[22:23] /*v[278:279]*/, v[24:25] /*v[280:281]*/
	v_pk_add_f32 v[24:25] /*v[280:281]*/, v[28:29] /*v[284:285]*/, v[30:31] /*v[286:287]*/
	v_pk_add_f32 v[16:17] /*v[272:273]*/, v[18:19] /*v[274:275]*/, v[16:17] /*v[272:273]*/
	v_pk_add_f32 v[18:19] /*v[274:275]*/, v[34:35] /*v[290:291]*/, v[36:37] /*v[292:293]*/
	v_pk_add_f32 v[28:29] /*v[284:285]*/, v[40:41] /*v[296:297]*/, v[42:43] /*v[298:299]*/
	s_set_vgpr_msb 0x4547
	v_pk_add_f32 v[66:67] /*v[322:323]*/, v[114:115] /*v[882:883]*/, v[66:67] /*v[322:323]*/
	v_pk_add_f32 v[96:97] /*v[352:353]*/, v[176:177] /*v[944:945]*/, v[96:97] /*v[352:353]*/
	s_set_vgpr_msb 0x4780
	v_pk_add_f32 v[196:197] /*v[708:709]*/, v[210:211], v[224:225]
	s_set_vgpr_msb 0x8047
	v_pk_add_f32 v[102:103] /*v[358:359]*/, v[204:205] /*v[972:973]*/, v[102:103] /*v[358:359]*/
	s_set_vgpr_msb 0x478b
	v_pk_add_f32 v[192:193] /*v[704:705]*/, v[240:241] /*v[1008:1009]*/, v[192:193] /*v[704:705]*/
	s_set_vgpr_msb 0x8b88
	v_pk_add_f32 v[194:195] /*v[706:707]*/, v[208:209], v[194:195] /*v[706:707]*/
	v_pk_add_f32 v[202:203] /*v[714:715]*/, v[192:193], v[202:203] /*v[714:715]*/
	s_set_vgpr_msb 0x888b
	v_pk_add_f32 v[208:209] /*v[720:721]*/, v[190:191] /*v[958:959]*/, v[208:209] /*v[720:721]*/
	s_set_vgpr_msb 0x8b88
	v_pk_add_f32 v[214:215] /*v[726:727]*/, v[230:231], v[214:215] /*v[726:727]*/
	s_set_vgpr_msb 0x888b
	v_pk_add_f32 v[220:221] /*v[732:733]*/, v[248:249] /*v[1016:1017]*/, v[220:221] /*v[732:733]*/
	s_set_vgpr_msb 0x8b48
	v_pk_add_f32 v[20:21] /*v[276:277]*/, v[250:251], v[226:227] /*v[738:739]*/
	s_set_vgpr_msb 0x4845
	v_pk_add_f32 v[22:23] /*v[278:279]*/, v[26:27] /*v[282:283]*/, v[22:23] /*v[278:279]*/
	v_pk_add_f32 v[24:25] /*v[280:281]*/, v[32:33] /*v[288:289]*/, v[24:25] /*v[280:281]*/
	v_pk_add_f32 v[26:27] /*v[282:283]*/, v[46:47] /*v[302:303]*/, v[64:65] /*v[320:321]*/
	v_pk_add_f32 v[18:19] /*v[274:275]*/, v[38:39] /*v[294:295]*/, v[18:19] /*v[274:275]*/
	v_pk_add_f32 v[28:29] /*v[284:285]*/, v[44:45] /*v[300:301]*/, v[28:29] /*v[284:285]*/
	v_pk_add_f32 v[30:31] /*v[286:287]*/, v[68:69] /*v[324:325]*/, v[70:71] /*v[326:327]*/
	v_pk_add_f32 v[32:33] /*v[288:289]*/, v[98:99] /*v[354:355]*/, v[100:101] /*v[356:357]*/
	s_set_vgpr_msb 0x454a
	v_pk_add_f32 v[36:37] /*v[292:293]*/, v[198:199] /*v[710:711]*/, v[200:201] /*v[712:713]*/
	v_pk_add_f32 v[38:39] /*v[294:295]*/, v[204:205] /*v[716:717]*/, v[206:207] /*v[718:719]*/
	v_pk_add_f32 v[40:41] /*v[296:297]*/, v[210:211] /*v[722:723]*/, v[212:213] /*v[724:725]*/
	v_pk_add_f32 v[42:43] /*v[298:299]*/, v[216:217] /*v[728:729]*/, v[218:219] /*v[730:731]*/
	v_pk_add_f32 v[44:45] /*v[300:301]*/, v[222:223] /*v[734:735]*/, v[224:225] /*v[736:737]*/
	s_set_vgpr_msb 0x4a00
	v_exp_f32_e32 v252, v252
	s_set_vgpr_msb 0x8b
	v_pk_add_f32 v[196:197] /*v[708:709]*/, v[112:113] /*v[880:881]*/, v[196:197] /*v[708:709]*/
	s_set_vgpr_msb 0x8b45
	v_pk_add_f32 v[26:27] /*v[282:283]*/, v[66:67] /*v[322:323]*/, v[26:27] /*v[282:283]*/
	s_set_vgpr_msb 0x454a
	v_pk_add_f32 v[34:35] /*v[290:291]*/, v[192:193] /*v[704:705]*/, v[194:195] /*v[706:707]*/
	s_set_vgpr_msb 0x4a45
	v_pk_add_f32 v[30:31] /*v[286:287]*/, v[96:97] /*v[352:353]*/, v[30:31] /*v[286:287]*/
	v_pk_add_f32 v[32:33] /*v[288:289]*/, v[102:103] /*v[358:359]*/, v[32:33] /*v[288:289]*/
	s_set_vgpr_msb 0x4546
	v_pk_add_f32 v[36:37] /*v[292:293]*/, v[202:203] /*v[714:715]*/, v[36:37] /*v[292:293]*/
	v_pk_add_f32 v[38:39] /*v[294:295]*/, v[208:209] /*v[720:721]*/, v[38:39] /*v[294:295]*/
	s_set_vgpr_msb 0x4645
	v_pk_add_f32 v[16:17] /*v[272:273]*/, v[16:17] /*v[272:273]*/, v[22:23] /*v[278:279]*/
	s_set_vgpr_msb 0x4546
	v_pk_add_f32 v[22:23] /*v[278:279]*/, v[214:215] /*v[726:727]*/, v[40:41] /*v[296:297]*/
	v_pk_add_f32 v[40:41] /*v[296:297]*/, v[220:221] /*v[732:733]*/, v[42:43] /*v[298:299]*/
	s_set_vgpr_msb 0x4645
	v_pk_add_f32 v[18:19] /*v[274:275]*/, v[18:19] /*v[274:275]*/, v[28:29] /*v[284:285]*/
	v_pk_add_f32 v[20:21] /*v[276:277]*/, v[20:21] /*v[276:277]*/, v[44:45] /*v[300:301]*/
	s_set_vgpr_msb 0x4580
	v_pk_add_f32 v[226:227] /*v[738:739]*/, v[252:253], v[254:255]
	s_set_vgpr_msb 0x8046
	v_pk_add_f32 v[34:35] /*v[290:291]*/, v[196:197] /*v[708:709]*/, v[34:35] /*v[290:291]*/
	s_set_vgpr_msb 0x4645
	v_pk_add_f32 v[16:17] /*v[272:273]*/, v[24:25] /*v[280:281]*/, v[16:17] /*v[272:273]*/
	v_pk_add_f32 v[24:25] /*v[280:281]*/, v[30:31] /*v[286:287]*/, v[32:33] /*v[288:289]*/
	v_pk_add_f32 v[28:29] /*v[284:285]*/, v[36:37] /*v[292:293]*/, v[38:39] /*v[294:295]*/
	v_pk_add_f32 v[18:19] /*v[274:275]*/, v[26:27] /*v[282:283]*/, v[18:19] /*v[274:275]*/
	v_pk_add_f32 v[20:21] /*v[276:277]*/, v[40:41] /*v[296:297]*/, v[20:21] /*v[276:277]*/
	s_set_vgpr_msb 0x4580
	v_exp_f32_e32 v196 /*v708*/, v134
	s_set_vgpr_msb 0x8045
	v_pk_add_f32 v[24:25] /*v[280:281]*/, v[34:35] /*v[290:291]*/, v[24:25] /*v[280:281]*/
	v_pk_add_f32 v[22:23] /*v[278:279]*/, v[22:23] /*v[278:279]*/, v[28:29] /*v[284:285]*/
	v_pk_add_f32 v[16:17] /*v[272:273]*/, v[16:17] /*v[272:273]*/, v[18:19] /*v[274:275]*/
	s_set_vgpr_msb 0x4546
	v_pk_add_f32 v[18:19] /*v[274:275]*/, v[226:227] /*v[738:739]*/, v[20:21] /*v[276:277]*/
	s_set_vgpr_msb 0x4645
	v_pk_add_f32 v[16:17] /*v[272:273]*/, v[24:25] /*v[280:281]*/, v[16:17] /*v[272:273]*/
	v_pk_add_f32 v[18:19] /*v[274:275]*/, v[22:23] /*v[278:279]*/, v[18:19] /*v[274:275]*/
	s_set_vgpr_msb 0x4585
	v_pk_add_f32 v[192:193] /*v[704:705]*/, v[18:19] /*v[274:275]*/, v[16:17] /*v[272:273]*/
	s_set_vgpr_msb 0x8582
	v_dual_mov_b32 v194 /*v706*/, v192 /*v704*/ :: v_dual_mov_b32 v195 /*v707*/, v193 /*v705*/
	v_permlanex16_b32 v194 /*v706*/, v194 /*v706*/, s65, 0xfedcba98
	v_permlanex16_b32 v195 /*v707*/, v195 /*v707*/, s65, 0xfedcba98
	s_set_vgpr_msb 0x8200
	s_cbranch_vccz .LBB0_31
	s_set_vgpr_msb 8
	v_pk_mul_f32 v[126:127], v[126:127], v[196:197] /*v[708:709]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[124:125], v[124:125], v[196:197] /*v[708:709]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[122:123], v[122:123], v[196:197] /*v[708:709]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[120:121], v[120:121], v[196:197] /*v[708:709]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[118:119], v[118:119], v[196:197] /*v[708:709]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[116:117], v[116:117], v[196:197] /*v[708:709]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[114:115], v[114:115], v[196:197] /*v[708:709]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[112:113], v[112:113], v[196:197] /*v[708:709]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[110:111], v[110:111], v[196:197] /*v[708:709]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[108:109], v[108:109], v[196:197] /*v[708:709]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[106:107], v[106:107], v[196:197] /*v[708:709]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[104:105], v[104:105], v[196:197] /*v[708:709]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[102:103], v[102:103], v[196:197] /*v[708:709]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[100:101], v[100:101], v[196:197] /*v[708:709]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[98:99], v[98:99], v[196:197] /*v[708:709]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[96:97], v[96:97], v[196:197] /*v[708:709]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[94:95], v[94:95], v[196:197] /*v[708:709]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[92:93], v[92:93], v[196:197] /*v[708:709]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[90:91], v[90:91], v[196:197] /*v[708:709]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[88:89], v[88:89], v[196:197] /*v[708:709]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[86:87], v[86:87], v[196:197] /*v[708:709]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[84:85], v[84:85], v[196:197] /*v[708:709]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[82:83], v[82:83], v[196:197] /*v[708:709]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[80:81], v[80:81], v[196:197] /*v[708:709]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[78:79], v[78:79], v[196:197] /*v[708:709]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[76:77], v[76:77], v[196:197] /*v[708:709]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[74:75], v[74:75], v[196:197] /*v[708:709]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[72:73], v[72:73], v[196:197] /*v[708:709]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[70:71], v[70:71], v[196:197] /*v[708:709]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[68:69], v[68:69], v[196:197] /*v[708:709]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[66:67], v[66:67], v[196:197] /*v[708:709]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[64:65], v[64:65], v[196:197] /*v[708:709]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x800
.LBB0_31:
	v_sub_f32_e32 v133, v133, v131
	s_and_b32 s2, s5, exec_lo
	s_cselect_b32 s2, 1, 0
	s_cmp_lg_u32 s2, 1
	v_mul_f32_e32 v133, 0x3fb8aa3b, v133
	s_set_vgpr_msb 0x80
	v_exp_f32_e32 v197 /*v709*/, v133
	s_set_vgpr_msb 0x8000
	s_cbranch_scc1 .LBB0_26
	v_nop
	s_set_vgpr_msb 2
	v_mov_b32_e32 v134, v197 /*v709*/
	s_set_vgpr_msb 0x200
	v_pk_mul_f32 v[62:63], v[62:63], v[134:135] op_sel_hi:[1,0]
	v_pk_mul_f32 v[60:61], v[60:61], v[134:135] op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59], v[58:59], v[134:135] op_sel_hi:[1,0]
	v_pk_mul_f32 v[56:57], v[56:57], v[134:135] op_sel_hi:[1,0]
	v_pk_mul_f32 v[54:55], v[54:55], v[134:135] op_sel_hi:[1,0]
	v_pk_mul_f32 v[52:53], v[52:53], v[134:135] op_sel_hi:[1,0]
	v_pk_mul_f32 v[50:51], v[50:51], v[134:135] op_sel_hi:[1,0]
	v_pk_mul_f32 v[48:49], v[48:49], v[134:135] op_sel_hi:[1,0]
	v_pk_mul_f32 v[46:47], v[46:47], v[134:135] op_sel_hi:[1,0]
	v_pk_mul_f32 v[44:45], v[44:45], v[134:135] op_sel_hi:[1,0]
	v_pk_mul_f32 v[42:43], v[42:43], v[134:135] op_sel_hi:[1,0]
	v_pk_mul_f32 v[40:41], v[40:41], v[134:135] op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39], v[38:39], v[134:135] op_sel_hi:[1,0]
	v_pk_mul_f32 v[36:37], v[36:37], v[134:135] op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35], v[34:35], v[134:135] op_sel_hi:[1,0]
	v_pk_mul_f32 v[32:33], v[32:33], v[134:135] op_sel_hi:[1,0]
	v_pk_mul_f32 v[30:31], v[30:31], v[134:135] op_sel_hi:[1,0]
	v_pk_mul_f32 v[28:29], v[28:29], v[134:135] op_sel_hi:[1,0]
	v_pk_mul_f32 v[26:27], v[26:27], v[134:135] op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25], v[24:25], v[134:135] op_sel_hi:[1,0]
	v_pk_mul_f32 v[22:23], v[22:23], v[134:135] op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21], v[20:21], v[134:135] op_sel_hi:[1,0]
	v_pk_mul_f32 v[18:19], v[18:19], v[134:135] op_sel_hi:[1,0]
	v_pk_mul_f32 v[16:17], v[16:17], v[134:135] op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[14:15], v[134:135] op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13], v[12:13], v[134:135] op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[10:11], v[134:135] op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[8:9], v[134:135] op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[6:7], v[134:135] op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[4:5], v[134:135] op_sel_hi:[1,0]
	v_pk_mul_f32 v[2:3], v[2:3], v[134:135] op_sel_hi:[1,0]
	v_pk_mul_f32 v[0:1], v[0:1], v[134:135] op_sel_hi:[1,0]
	s_branch .LBB0_26
.LBB0_33:
	v_dual_mov_b32 v1, v0 :: v_dual_mov_b32 v2, v0
	v_dual_mov_b32 v3, v0 :: v_dual_mov_b32 v4, v0
	v_dual_mov_b32 v5, v0 :: v_dual_mov_b32 v6, v0
	v_dual_mov_b32 v7, v0 :: v_dual_mov_b32 v205, v0
	v_dual_mov_b32 v131, 0xf149f2ca :: v_dual_mov_b32 v204, v0
	v_dual_mov_b32 v129, v194 :: v_dual_mov_b32 v132, 0xf149f2ca
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
	s_cmp_ge_u32 s44, s10
	s_cbranch_scc0 .LBB0_36
.LBB0_34:
	s_set_vgpr_msb 64
	v_dual_mov_b32 v51 /*v307*/, v131 :: v_dual_mov_b32 v50 /*v306*/, v132
	s_set_vgpr_msb 0x4000
	s_branch .LBB0_45
.LBB0_35:
	s_clause 0x4
	scratch_load_b32 v192, off, off offset:564 nv
	scratch_load_b32 v193, off, off offset:568 nv
	scratch_load_b32 v194, off, off offset:584 nv
	s_set_vgpr_msb 0xc0
	scratch_load_b32 v200 /*v968*/, off, off offset:8 nv
	s_cmp_ge_u32 s44, s10
	s_set_vgpr_msb 0xc000
	s_cbranch_scc1 .LBB0_34
.LBB0_36:
	s_wait_loadcnt 0x2
	v_dual_add_nc_u32 v133, s11, v192 :: v_dual_add_nc_u32 v134, s11, v193
	s_add_co_i32 s2, s58, -1
	s_mov_b32 s19, 0
	s_mov_b32 s4, 1
	v_min_i32_e32 v133, s2, v133
	v_min_i32_e32 v134, s2, v134
	s_mov_b32 s11, s19
	s_mov_b32 s16, 64
	s_mov_b32 s15, 0x800000
	s_mov_b32 s13, 0xffff0000
	s_mov_b32 s12, 0x7510000
	s_mov_b32 s20, 0xf510000
	s_mov_b32 s47, 0x76543210
	s_mov_b32 s46, 0x3fb8aa3b
	s_branch .LBB0_38
.LBB0_37:
	s_set_vgpr_msb 64
	s_clause 0x1
	scratch_load_b128 v[40:43] /*v[296:299]*/, off, off offset:524 th:TH_LOAD_LU nv
	scratch_load_b128 v[44:47] /*v[300:303]*/, off, off offset:540 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0x400f
	v_cvt_pk_bf16_f32 v199, v16 /*v784*/, v32 /*v800*/
	s_set_vgpr_msb 0xf0e
	v_cvt_pk_bf16_f32 v198, v252 /*v764*/, v10 /*v778*/
	s_set_vgpr_msb 0xe0a
	v_cvt_pk_bf16_f32 v197, v236 /*v748*/, v248 /*v760*/
	v_cvt_pk_bf16_f32 v196, v222 /*v734*/, v232 /*v744*/
	v_cvt_pk_bf16_f32 v195, v210 /*v722*/, v218 /*v730*/
	v_cvt_pk_bf16_f32 v194, v202 /*v714*/, v208 /*v720*/
	v_cvt_pk_bf16_f32 v193, v196 /*v708*/, v200 /*v712*/
	v_cvt_pk_bf16_f32 v192, v192 /*v704*/, v194 /*v706*/
	s_set_vgpr_msb 0xa0f
	v_cvt_pk_bf16_f32 v207, v17 /*v785*/, v33 /*v801*/
	s_set_vgpr_msb 0xf0e
	v_cvt_pk_bf16_f32 v206, v253 /*v765*/, v11 /*v779*/
	s_set_vgpr_msb 0xe0a
	v_cvt_pk_bf16_f32 v205, v237 /*v749*/, v249 /*v761*/
	v_cvt_pk_bf16_f32 v204, v223 /*v735*/, v233 /*v745*/
	v_cvt_pk_bf16_f32 v203, v211 /*v723*/, v219 /*v731*/
	v_cvt_pk_bf16_f32 v202, v203 /*v715*/, v209 /*v721*/
	v_cvt_pk_bf16_f32 v201, v197 /*v709*/, v201 /*v713*/
	v_cvt_pk_bf16_f32 v200, v193 /*v705*/, v195 /*v707*/
	s_set_vgpr_msb 0xa01
	v_wmma_f32_16x16x32_bf16 v[80:87], v[120:127] /*v[376:383]*/, v[192:199], v[80:87]
	s_set_vgpr_msb 0x10f
	v_cvt_pk_bf16_f32 v215, v48 /*v816*/, v64 /*v832*/
	v_cvt_pk_bf16_f32 v214, v26 /*v794*/, v42 /*v810*/
	v_cvt_pk_bf16_f32 v213, v6 /*v774*/, v22 /*v790*/
	s_set_vgpr_msb 0xf0e
	v_cvt_pk_bf16_f32 v212, v244 /*v756*/, v2 /*v770*/
	s_set_vgpr_msb 0xe0a
	v_cvt_pk_bf16_f32 v211, v228 /*v740*/, v240 /*v752*/
	v_cvt_pk_bf16_f32 v210, v216 /*v728*/, v226 /*v738*/
	v_cvt_pk_bf16_f32 v209, v206 /*v718*/, v214 /*v726*/
	s_set_vgpr_msb 0xa01
	v_wmma_f32_16x16x32_bf16 v[16:23], v[120:127] /*v[376:383]*/, v[200:207], v[16:23]
	s_set_vgpr_msb 0x10a
	v_cvt_pk_bf16_f32 v208, v198 /*v710*/, v204 /*v716*/
	s_set_vgpr_msb 0xa0f
	v_cvt_pk_bf16_f32 v223, v49 /*v817*/, v65 /*v833*/
	v_cvt_pk_bf16_f32 v222, v27 /*v795*/, v43 /*v811*/
	v_cvt_pk_bf16_f32 v221, v7 /*v775*/, v23 /*v791*/
	s_set_vgpr_msb 0xf0e
	v_cvt_pk_bf16_f32 v220, v245 /*v757*/, v3 /*v771*/
	s_set_vgpr_msb 0xe0a
	v_cvt_pk_bf16_f32 v219, v229 /*v741*/, v241 /*v753*/
	v_cvt_pk_bf16_f32 v218, v217 /*v729*/, v227 /*v739*/
	s_set_vgpr_msb 0xa02
	v_wmma_f32_16x16x32_bf16 v[120:127], v[184:191] /*v[696:703]*/, v[192:199], v[120:127]
	s_set_vgpr_msb 0x20a
	v_cvt_pk_bf16_f32 v217, v207 /*v719*/, v215 /*v727*/
	v_cvt_pk_bf16_f32 v216, v199 /*v711*/, v205 /*v717*/
	s_set_vgpr_msb 0xa0f
	v_cvt_pk_bf16_f32 v231, v80 /*v848*/, v96 /*v864*/
	v_cvt_pk_bf16_f32 v230, v58 /*v826*/, v74 /*v842*/
	v_cvt_pk_bf16_f32 v229, v38 /*v806*/, v54 /*v822*/
	v_cvt_pk_bf16_f32 v228, v18 /*v786*/, v34 /*v802*/
	s_set_vgpr_msb 0xf0e
	v_cvt_pk_bf16_f32 v227, v254 /*v766*/, v12 /*v780*/
	s_set_vgpr_msb 0xe02
	v_wmma_f32_16x16x32_bf16 v[56:63], v[184:191] /*v[696:703]*/, v[200:207], v[56:63]
	s_set_vgpr_msb 0x20a
	v_cvt_pk_bf16_f32 v226, v238 /*v750*/, v250 /*v762*/
	v_cvt_pk_bf16_f32 v225, v224 /*v736*/, v234 /*v746*/
	v_cvt_pk_bf16_f32 v224, v212 /*v724*/, v220 /*v732*/
	s_set_vgpr_msb 0xa0f
	v_cvt_pk_bf16_f32 v239, v81 /*v849*/, v97 /*v865*/
	v_cvt_pk_bf16_f32 v238, v59 /*v827*/, v75 /*v843*/
	v_cvt_pk_bf16_f32 v237, v39 /*v807*/, v55 /*v823*/
	v_cvt_pk_bf16_f32 v236, v19 /*v787*/, v35 /*v803*/
	s_set_vgpr_msb 0xf01
	v_wmma_f32_16x16x32_bf16 v[80:87], v[112:119] /*v[368:375]*/, v[208:215], v[80:87]
	s_set_vgpr_msb 0x10e
	v_cvt_pk_bf16_f32 v235, v255 /*v767*/, v13 /*v781*/
	s_set_vgpr_msb 0xe0a
	v_cvt_pk_bf16_f32 v234, v239 /*v751*/, v251 /*v763*/
	v_cvt_pk_bf16_f32 v233, v225 /*v737*/, v235 /*v747*/
	v_cvt_pk_bf16_f32 v232, v213 /*v725*/, v221 /*v733*/
	s_set_vgpr_msb 0xa0f
	v_cvt_pk_bf16_f32 v247, v112 /*v880*/, v126 /*v894*/
	v_cvt_pk_bf16_f32 v246, v90 /*v858*/, v106 /*v874*/
	v_cvt_pk_bf16_f32 v245, v70 /*v838*/, v86 /*v854*/
	s_set_vgpr_msb 0xf01
	v_wmma_f32_16x16x32_bf16 v[16:23], v[112:119] /*v[368:375]*/, v[216:223], v[16:23]
	s_set_vgpr_msb 0x10f
	v_cvt_pk_bf16_f32 v244, v50 /*v818*/, v66 /*v834*/
	v_cvt_pk_bf16_f32 v243, v28 /*v796*/, v44 /*v812*/
	v_cvt_pk_bf16_f32 v242, v8 /*v776*/, v24 /*v792*/
	s_set_vgpr_msb 0xf0e
	v_cvt_pk_bf16_f32 v241, v246 /*v758*/, v4 /*v772*/
	s_set_vgpr_msb 0xe0a
	v_cvt_pk_bf16_f32 v240, v230 /*v742*/, v242 /*v754*/
	s_set_vgpr_msb 0xa0f
	v_cvt_pk_bf16_f32 v255, v113 /*v881*/, v127 /*v895*/
	v_cvt_pk_bf16_f32 v254, v91 /*v859*/, v107 /*v875*/
	s_set_vgpr_msb 0xf02
	v_wmma_f32_16x16x32_bf16 v[120:127], v[176:183] /*v[688:695]*/, v[208:215], v[120:127]
	s_set_vgpr_msb 0x20f
	v_cvt_pk_bf16_f32 v253, v71 /*v839*/, v87 /*v855*/
	v_cvt_pk_bf16_f32 v252, v51 /*v819*/, v67 /*v835*/
	v_cvt_pk_bf16_f32 v251, v29 /*v797*/, v45 /*v813*/
	v_cvt_pk_bf16_f32 v250, v9 /*v777*/, v25 /*v793*/
	s_set_vgpr_msb 0xf0e
	v_cvt_pk_bf16_f32 v249, v247 /*v759*/, v5 /*v773*/
	s_set_vgpr_msb 0xe0a
	v_cvt_pk_bf16_f32 v248, v231 /*v743*/, v243 /*v755*/
	s_set_vgpr_msb 0xa4f
	v_cvt_pk_bf16_f32 v7 /*v263*/, v140 /*v908*/, v152 /*v920*/
	s_set_vgpr_msb 0x4f02
	v_wmma_f32_16x16x32_bf16 v[56:63], v[176:183] /*v[688:695]*/, v[216:223], v[56:63]
	s_set_vgpr_msb 0x24f
	v_cvt_pk_bf16_f32 v6 /*v262*/, v122 /*v890*/, v136 /*v904*/
	v_cvt_pk_bf16_f32 v5 /*v261*/, v102 /*v870*/, v118 /*v886*/
	v_cvt_pk_bf16_f32 v4 /*v260*/, v82 /*v850*/, v98 /*v866*/
	v_cvt_pk_bf16_f32 v3 /*v259*/, v60 /*v828*/, v76 /*v844*/
	v_cvt_pk_bf16_f32 v2 /*v258*/, v40 /*v808*/, v56 /*v824*/
	v_cvt_pk_bf16_f32 v1 /*v257*/, v20 /*v788*/, v36 /*v804*/
	v_cvt_pk_bf16_f32 v0 /*v256*/, v0 /*v768*/, v14 /*v782*/
	s_set_vgpr_msb 0x4f01
	v_wmma_f32_16x16x32_bf16 v[80:87], v[104:111] /*v[360:367]*/, v[224:231], v[80:87]
	s_set_vgpr_msb 0x14f
	v_cvt_pk_bf16_f32 v15 /*v271*/, v141 /*v909*/, v153 /*v921*/
	v_cvt_pk_bf16_f32 v14 /*v270*/, v123 /*v891*/, v137 /*v905*/
	v_cvt_pk_bf16_f32 v13 /*v269*/, v103 /*v871*/, v119 /*v887*/
	v_cvt_pk_bf16_f32 v12 /*v268*/, v83 /*v851*/, v99 /*v867*/
	v_cvt_pk_bf16_f32 v11 /*v267*/, v61 /*v829*/, v77 /*v845*/
	v_cvt_pk_bf16_f32 v10 /*v266*/, v41 /*v809*/, v57 /*v825*/
	v_cvt_pk_bf16_f32 v9 /*v265*/, v21 /*v789*/, v37 /*v805*/
	s_set_vgpr_msb 0x4f01
	v_wmma_f32_16x16x32_bf16 v[16:23], v[104:111] /*v[360:367]*/, v[232:239], v[16:23]
	s_set_vgpr_msb 0x14f
	v_cvt_pk_bf16_f32 v8 /*v264*/, v1 /*v769*/, v15 /*v783*/
	v_cvt_pk_bf16_f32 v23 /*v279*/, v162 /*v930*/, v170 /*v938*/
	v_cvt_pk_bf16_f32 v22 /*v278*/, v148 /*v916*/, v158 /*v926*/
	v_cvt_pk_bf16_f32 v21 /*v277*/, v132 /*v900*/, v144 /*v912*/
	v_cvt_pk_bf16_f32 v20 /*v276*/, v114 /*v882*/, v128 /*v896*/
	v_cvt_pk_bf16_f32 v19 /*v275*/, v92 /*v860*/, v108 /*v876*/
	v_cvt_pk_bf16_f32 v18 /*v274*/, v72 /*v840*/, v88 /*v856*/
	s_set_vgpr_msb 0x4f02
	v_wmma_f32_16x16x32_bf16 v[120:127], v[168:175] /*v[680:687]*/, v[224:231], v[120:127]
	s_set_vgpr_msb 0x24f
	v_cvt_pk_bf16_f32 v17 /*v273*/, v52 /*v820*/, v68 /*v836*/
	v_cvt_pk_bf16_f32 v16 /*v272*/, v30 /*v798*/, v46 /*v814*/
	v_cvt_pk_bf16_f32 v31 /*v287*/, v163 /*v931*/, v171 /*v939*/
	v_cvt_pk_bf16_f32 v30 /*v286*/, v149 /*v917*/, v159 /*v927*/
	v_cvt_pk_bf16_f32 v29 /*v285*/, v133 /*v901*/, v145 /*v913*/
	v_cvt_pk_bf16_f32 v28 /*v284*/, v115 /*v883*/, v129 /*v897*/
	v_cvt_pk_bf16_f32 v27 /*v283*/, v93 /*v861*/, v109 /*v877*/
	s_set_vgpr_msb 0x4f02
	v_wmma_f32_16x16x32_bf16 v[56:63], v[168:175] /*v[680:687]*/, v[232:239], v[56:63]
	s_set_vgpr_msb 0x24f
	v_cvt_pk_bf16_f32 v26 /*v282*/, v73 /*v841*/, v89 /*v857*/
	v_cvt_pk_bf16_f32 v25 /*v281*/, v53 /*v821*/, v69 /*v837*/
	v_cvt_pk_bf16_f32 v24 /*v280*/, v31 /*v799*/, v47 /*v815*/
	v_cvt_pk_bf16_f32 v39 /*v295*/, v180 /*v948*/, v186 /*v954*/
	v_cvt_pk_bf16_f32 v38 /*v294*/, v168 /*v936*/, v178 /*v946*/
	v_cvt_pk_bf16_f32 v37 /*v293*/, v156 /*v924*/, v166 /*v934*/
	v_cvt_pk_bf16_f32 v36 /*v292*/, v142 /*v910*/, v154 /*v922*/
	s_set_vgpr_msb 0x4f01
	v_wmma_f32_16x16x32_bf16 v[80:87], v[96:103] /*v[352:359]*/, v[240:247], v[80:87]
	s_set_vgpr_msb 0x14f
	v_cvt_pk_bf16_f32 v35 /*v291*/, v124 /*v892*/, v138 /*v906*/
	v_cvt_pk_bf16_f32 v34 /*v290*/, v104 /*v872*/, v120 /*v888*/
	v_cvt_pk_bf16_f32 v33 /*v289*/, v84 /*v852*/, v100 /*v868*/
	v_cvt_pk_bf16_f32 v32 /*v288*/, v62 /*v830*/, v78 /*v846*/
	s_add_nc_u64 s[44:45], s[44:45], 1
	s_set_vgpr_msb 0x4f01
	v_mov_b32_e32 v131, v51 /*v307*/
	v_cmp_ge_u64_e64 s2, s[44:45], s[10:11]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[96:103] /*v[352:359]*/, v[248:255], v[16:23]
	s_and_b32 vcc_lo, exec_lo, s2
	s_set_vgpr_msb 0x102
	v_wmma_f32_16x16x32_bf16 v[120:127], v[160:167] /*v[672:679]*/, v[240:247], v[120:127]
	v_wmma_f32_16x16x32_bf16 v[56:63], v[160:167] /*v[672:679]*/, v[248:255], v[56:63]
	s_set_vgpr_msb 0x205
	v_wmma_f32_16x16x32_bf16 v[80:87], v[88:95] /*v[344:351]*/, v[0:7] /*v[256:263]*/, v[80:87]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[88:95] /*v[344:351]*/, v[8:15] /*v[264:271]*/, v[16:23]
	s_set_vgpr_msb 0x506
	v_wmma_f32_16x16x32_bf16 v[120:127], v[152:159] /*v[664:671]*/, v[0:7] /*v[256:263]*/, v[120:127]
	v_wmma_f32_16x16x32_bf16 v[56:63], v[152:159] /*v[664:671]*/, v[8:15] /*v[264:271]*/, v[56:63]
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x68f
	v_cvt_pk_bf16_f32 v159 /*v671*/, v189 /*v957*/, v191 /*v959*/
	v_cvt_pk_bf16_f32 v158 /*v670*/, v185 /*v953*/, v177 /*v945*/
	v_cvt_pk_bf16_f32 v157 /*v669*/, v175 /*v943*/, v183 /*v951*/
	s_set_vgpr_msb 0x8f05
	v_wmma_f32_16x16x32_bf16 v[80:87], v[80:87] /*v[336:343]*/, v[16:23] /*v[272:279]*/, v[80:87]
	s_set_vgpr_msb 0x58f
	v_cvt_pk_bf16_f32 v156 /*v668*/, v165 /*v933*/, v173 /*v941*/
	v_cvt_pk_bf16_f32 v155 /*v667*/, v151 /*v919*/, v161 /*v929*/
	v_cvt_pk_bf16_f32 v154 /*v666*/, v135 /*v903*/, v147 /*v915*/
	v_cvt_pk_bf16_f32 v153 /*v665*/, v117 /*v885*/, v131 /*v899*/
	v_cvt_pk_bf16_f32 v152 /*v664*/, v95 /*v863*/, v111 /*v879*/
	s_set_vgpr_msb 0x8f05
	v_wmma_f32_16x16x32_bf16 v[16:23], v[80:87] /*v[336:343]*/, v[24:31] /*v[280:287]*/, v[16:23]
	s_set_vgpr_msb 0x506
	v_wmma_f32_16x16x32_bf16 v[120:127], v[144:151] /*v[656:663]*/, v[16:23] /*v[272:279]*/, v[120:127]
	v_wmma_f32_16x16x32_bf16 v[56:63], v[144:151] /*v[656:663]*/, v[24:31] /*v[280:287]*/, v[56:63]
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x68f
	v_cvt_pk_bf16_f32 v151 /*v663*/, v181 /*v949*/, v187 /*v955*/
	v_cvt_pk_bf16_f32 v150 /*v662*/, v169 /*v937*/, v179 /*v947*/
	v_cvt_pk_bf16_f32 v149 /*v661*/, v157 /*v925*/, v167 /*v935*/
	v_cvt_pk_bf16_f32 v148 /*v660*/, v143 /*v911*/, v155 /*v923*/
	v_cvt_pk_bf16_f32 v147 /*v659*/, v125 /*v893*/, v139 /*v907*/
	v_cvt_pk_bf16_f32 v146 /*v658*/, v105 /*v873*/, v121 /*v889*/
	v_cvt_pk_bf16_f32 v145 /*v657*/, v85 /*v853*/, v101 /*v869*/
	v_cvt_pk_bf16_f32 v144 /*v656*/, v63 /*v831*/, v79 /*v847*/
	s_set_vgpr_msb 0x8f05
	v_wmma_f32_16x16x32_bf16 v[80:87], v[64:71] /*v[320:327]*/, v[32:39] /*v[288:295]*/, v[80:87]
	s_set_vgpr_msb 0x509
	v_wmma_f32_16x16x32_bf16 v[16:23], v[64:71] /*v[320:327]*/, v[144:151] /*v[656:663]*/, v[16:23]
	s_set_vgpr_msb 0x906
	v_wmma_f32_16x16x32_bf16 v[120:127], v[136:143] /*v[648:655]*/, v[32:39] /*v[288:295]*/, v[120:127]
	s_set_vgpr_msb 0x60a
	v_wmma_f32_16x16x32_bf16 v[56:63], v[136:143] /*v[648:655]*/, v[144:151] /*v[656:663]*/, v[56:63]
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0xa8f
	v_cvt_pk_bf16_f32 v143 /*v655*/, v188 /*v956*/, v190 /*v958*/
	v_cvt_pk_bf16_f32 v142 /*v654*/, v184 /*v952*/, v176 /*v944*/
	v_cvt_pk_bf16_f32 v141 /*v653*/, v174 /*v942*/, v182 /*v950*/
	v_cvt_pk_bf16_f32 v140 /*v652*/, v164 /*v932*/, v172 /*v940*/
	v_cvt_pk_bf16_f32 v139 /*v651*/, v150 /*v918*/, v160 /*v928*/
	v_cvt_pk_bf16_f32 v138 /*v650*/, v134 /*v902*/, v146 /*v914*/
	v_cvt_pk_bf16_f32 v137 /*v649*/, v116 /*v884*/, v130 /*v898*/
	v_cvt_pk_bf16_f32 v136 /*v648*/, v94 /*v862*/, v110 /*v878*/
	s_set_vgpr_msb 0x8f01
	v_wmma_f32_16x16x32_bf16 v[72:79], v[72:79] /*v[328:335]*/, v[192:199], v[72:79]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[72:79] /*v[328:335]*/, v[200:207], v[8:15]
	v_wmma_f32_16x16x32_bf16 v[72:79], v[56:63] /*v[312:319]*/, v[208:215], v[72:79]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[56:63] /*v[312:319]*/, v[216:223], v[8:15]
	s_set_vgpr_msb 0x102
	v_wmma_f32_16x16x32_bf16 v[112:119], v[104:111] /*v[616:623]*/, v[192:199], v[112:119]
	v_wmma_f32_16x16x32_bf16 v[104:111], v[72:79] /*v[584:591]*/, v[192:199], v[104:111]
	s_set_vgpr_msb 0x201
	v_wmma_f32_16x16x32_bf16 v[96:103], v[248:255] /*v[504:511]*/, v[192:199], v[96:103]
	v_wmma_f32_16x16x32_bf16 v[88:95], v[200:207] /*v[456:463]*/, v[192:199], v[88:95]
	s_set_vgpr_msb 0x102
	v_wmma_f32_16x16x32_bf16 v[48:55], v[104:111] /*v[616:623]*/, v[200:207], v[48:55]
	s_set_vgpr_msb 0x209
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[80:87], v[40:47] /*v[296:303]*/, v[136:143] /*v[648:655]*/, v[80:87]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[40:47] /*v[296:303]*/, v[152:159] /*v[664:671]*/, v[16:23]
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x940
	s_clause 0x1
	scratch_load_b128 v[40:43] /*v[296:299]*/, off, off offset:492 th:TH_LOAD_LU nv
	scratch_load_b128 v[44:47] /*v[300:303]*/, off, off offset:508 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0x4002
	v_wmma_f32_16x16x32_bf16 v[40:47], v[72:79] /*v[584:591]*/, v[200:207], v[40:47]
	s_set_vgpr_msb 0x201
	v_wmma_f32_16x16x32_bf16 v[32:39], v[248:255] /*v[504:511]*/, v[200:207], v[32:39]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[200:207] /*v[456:463]*/, v[200:207], v[24:31]
	s_set_vgpr_msb 0x102
	v_wmma_f32_16x16x32_bf16 v[112:119], v[120:127] /*v[632:639]*/, v[208:215], v[112:119]
	v_wmma_f32_16x16x32_bf16 v[48:55], v[120:127] /*v[632:639]*/, v[216:223], v[48:55]
	v_wmma_f32_16x16x32_bf16 v[104:111], v[56:63] /*v[568:575]*/, v[208:215], v[104:111]
	v_wmma_f32_16x16x32_bf16 v[40:47], v[56:63] /*v[568:575]*/, v[216:223], v[40:47]
	s_set_vgpr_msb 0x201
	v_wmma_f32_16x16x32_bf16 v[96:103], v[240:247] /*v[496:503]*/, v[208:215], v[96:103]
	v_wmma_f32_16x16x32_bf16 v[32:39], v[240:247] /*v[496:503]*/, v[216:223], v[32:39]
	v_wmma_f32_16x16x32_bf16 v[88:95], v[184:191] /*v[440:447]*/, v[208:215], v[88:95]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[184:191] /*v[440:447]*/, v[216:223], v[24:31]
	s_set_vgpr_msb 0x102
	v_wmma_f32_16x16x32_bf16 v[112:119], v[112:119] /*v[624:631]*/, v[224:231], v[112:119]
	v_wmma_f32_16x16x32_bf16 v[48:55], v[112:119] /*v[624:631]*/, v[232:239], v[48:55]
	v_wmma_f32_16x16x32_bf16 v[104:111], v[40:47] /*v[552:559]*/, v[224:231], v[104:111]
	v_wmma_f32_16x16x32_bf16 v[40:47], v[40:47] /*v[552:559]*/, v[232:239], v[40:47]
	s_set_vgpr_msb 0x201
	v_wmma_f32_16x16x32_bf16 v[96:103], v[232:239] /*v[488:495]*/, v[224:231], v[96:103]
	v_wmma_f32_16x16x32_bf16 v[32:39], v[232:239] /*v[488:495]*/, v[232:239], v[32:39]
	v_wmma_f32_16x16x32_bf16 v[88:95], v[168:175] /*v[424:431]*/, v[224:231], v[88:95]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[168:175] /*v[424:431]*/, v[232:239], v[24:31]
	s_set_vgpr_msb 0x102
	v_wmma_f32_16x16x32_bf16 v[112:119], v[96:103] /*v[608:615]*/, v[240:247], v[112:119]
	v_wmma_f32_16x16x32_bf16 v[48:55], v[96:103] /*v[608:615]*/, v[248:255], v[48:55]
	v_wmma_f32_16x16x32_bf16 v[104:111], v[32:39] /*v[544:551]*/, v[240:247], v[104:111]
	v_wmma_f32_16x16x32_bf16 v[40:47], v[32:39] /*v[544:551]*/, v[248:255], v[40:47]
	s_set_vgpr_msb 0x201
	v_wmma_f32_16x16x32_bf16 v[96:103], v[224:231] /*v[480:487]*/, v[240:247], v[96:103]
	v_wmma_f32_16x16x32_bf16 v[32:39], v[224:231] /*v[480:487]*/, v[248:255], v[32:39]
	v_wmma_f32_16x16x32_bf16 v[88:95], v[160:167] /*v[416:423]*/, v[240:247], v[88:95]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[160:167] /*v[416:423]*/, v[248:255], v[24:31]
	s_set_vgpr_msb 0x106
	v_wmma_f32_16x16x32_bf16 v[112:119], v[88:95] /*v[600:607]*/, v[0:7] /*v[256:263]*/, v[112:119]
	v_wmma_f32_16x16x32_bf16 v[48:55], v[88:95] /*v[600:607]*/, v[8:15] /*v[264:271]*/, v[48:55]
	v_wmma_f32_16x16x32_bf16 v[104:111], v[24:31] /*v[536:543]*/, v[0:7] /*v[256:263]*/, v[104:111]
	v_wmma_f32_16x16x32_bf16 v[40:47], v[24:31] /*v[536:543]*/, v[8:15] /*v[264:271]*/, v[40:47]
	s_set_vgpr_msb 0x605
	v_wmma_f32_16x16x32_bf16 v[96:103], v[216:223] /*v[472:479]*/, v[0:7] /*v[256:263]*/, v[96:103]
	v_wmma_f32_16x16x32_bf16 v[32:39], v[216:223] /*v[472:479]*/, v[8:15] /*v[264:271]*/, v[32:39]
	v_wmma_f32_16x16x32_bf16 v[88:95], v[152:159] /*v[408:415]*/, v[0:7] /*v[256:263]*/, v[88:95]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[152:159] /*v[408:415]*/, v[8:15] /*v[264:271]*/, v[24:31]
	s_set_vgpr_msb 0x506
	v_wmma_f32_16x16x32_bf16 v[112:119], v[80:87] /*v[592:599]*/, v[16:23] /*v[272:279]*/, v[112:119]
	v_wmma_f32_16x16x32_bf16 v[48:55], v[80:87] /*v[592:599]*/, v[24:31] /*v[280:287]*/, v[48:55]
	v_wmma_f32_16x16x32_bf16 v[104:111], v[16:23] /*v[528:535]*/, v[16:23] /*v[272:279]*/, v[104:111]
	s_set_vgpr_msb 0x601
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[72:79], v[40:47] /*v[296:303]*/, v[224:231], v[72:79]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[40:47] /*v[296:303]*/, v[232:239], v[8:15]
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x140
	s_clause 0x1
	scratch_load_b128 v[40:43] /*v[296:299]*/, off, off offset:460 th:TH_LOAD_LU nv
	scratch_load_b128 v[44:47] /*v[300:303]*/, off, off offset:476 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0x4006
	v_wmma_f32_16x16x32_bf16 v[40:47], v[16:23] /*v[528:535]*/, v[24:31] /*v[280:287]*/, v[40:47]
	s_set_vgpr_msb 0x605
	v_wmma_f32_16x16x32_bf16 v[96:103], v[208:215] /*v[464:471]*/, v[16:23] /*v[272:279]*/, v[96:103]
	v_wmma_f32_16x16x32_bf16 v[32:39], v[208:215] /*v[464:471]*/, v[24:31] /*v[280:287]*/, v[32:39]
	v_wmma_f32_16x16x32_bf16 v[88:95], v[144:151] /*v[400:407]*/, v[16:23] /*v[272:279]*/, v[88:95]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[144:151] /*v[400:407]*/, v[24:31] /*v[280:287]*/, v[24:31]
	s_set_vgpr_msb 0x506
	v_wmma_f32_16x16x32_bf16 v[112:119], v[64:71] /*v[576:583]*/, v[32:39] /*v[288:295]*/, v[112:119]
	s_set_vgpr_msb 0x60a
	v_wmma_f32_16x16x32_bf16 v[48:55], v[64:71] /*v[576:583]*/, v[144:151] /*v[656:663]*/, v[48:55]
	s_set_vgpr_msb 0xa06
	v_wmma_f32_16x16x32_bf16 v[104:111], v[8:15] /*v[520:527]*/, v[32:39] /*v[288:295]*/, v[104:111]
	s_set_vgpr_msb 0x60a
	v_wmma_f32_16x16x32_bf16 v[40:47], v[8:15] /*v[520:527]*/, v[144:151] /*v[656:663]*/, v[40:47]
	s_set_vgpr_msb 0xa05
	v_wmma_f32_16x16x32_bf16 v[96:103], v[192:199] /*v[448:455]*/, v[32:39] /*v[288:295]*/, v[96:103]
	s_set_vgpr_msb 0x509
	v_wmma_f32_16x16x32_bf16 v[32:39], v[192:199] /*v[448:455]*/, v[144:151] /*v[656:663]*/, v[32:39]
	s_set_vgpr_msb 0x905
	v_wmma_f32_16x16x32_bf16 v[88:95], v[136:143] /*v[392:399]*/, v[32:39] /*v[288:295]*/, v[88:95]
	s_set_vgpr_msb 0x509
	v_wmma_f32_16x16x32_bf16 v[24:31], v[136:143] /*v[392:399]*/, v[144:151] /*v[656:663]*/, v[24:31]
	s_set_vgpr_msb 0x90a
	v_wmma_f32_16x16x32_bf16 v[120:127], v[128:135] /*v[640:647]*/, v[136:143] /*v[648:655]*/, v[120:127]
	v_wmma_f32_16x16x32_bf16 v[56:63], v[128:135] /*v[640:647]*/, v[152:159] /*v[664:671]*/, v[56:63]
	v_wmma_f32_16x16x32_bf16 v[112:119], v[48:55] /*v[560:567]*/, v[136:143] /*v[648:655]*/, v[112:119]
	v_wmma_f32_16x16x32_bf16 v[48:55], v[48:55] /*v[560:567]*/, v[152:159] /*v[664:671]*/, v[48:55]
	v_wmma_f32_16x16x32_bf16 v[104:111], v[0:7] /*v[512:519]*/, v[136:143] /*v[648:655]*/, v[104:111]
	v_wmma_f32_16x16x32_bf16 v[40:47], v[0:7] /*v[512:519]*/, v[152:159] /*v[664:671]*/, v[40:47]
	s_set_vgpr_msb 0xa09
	v_wmma_f32_16x16x32_bf16 v[96:103], v[176:183] /*v[432:439]*/, v[136:143] /*v[648:655]*/, v[96:103]
	v_wmma_f32_16x16x32_bf16 v[32:39], v[176:183] /*v[432:439]*/, v[152:159] /*v[664:671]*/, v[32:39]
	v_wmma_f32_16x16x32_bf16 v[88:95], v[128:135] /*v[384:391]*/, v[136:143] /*v[648:655]*/, v[88:95]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[128:135] /*v[384:391]*/, v[152:159] /*v[664:671]*/, v[24:31]
	s_set_vgpr_msb 0x901
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[72:79], v[40:47] /*v[296:303]*/, v[240:247], v[72:79]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[40:47] /*v[296:303]*/, v[248:255], v[8:15]
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x140
	s_clause 0x1
	scratch_load_b128 v[40:43] /*v[296:299]*/, off, off offset:428 th:TH_LOAD_LU nv
	scratch_load_b128 v[44:47] /*v[300:303]*/, off, off offset:444 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0x4005
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[72:79], v[40:47] /*v[296:303]*/, v[0:7] /*v[256:263]*/, v[72:79]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[40:47] /*v[296:303]*/, v[8:15] /*v[264:271]*/, v[8:15]
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x540
	s_clause 0x1
	scratch_load_b128 v[40:43] /*v[296:299]*/, off, off offset:396 th:TH_LOAD_LU nv
	scratch_load_b128 v[44:47] /*v[300:303]*/, off, off offset:412 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0x4005
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[72:79], v[40:47] /*v[296:303]*/, v[16:23] /*v[272:279]*/, v[72:79]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[40:47] /*v[296:303]*/, v[24:31] /*v[280:287]*/, v[8:15]
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x540
	s_clause 0x1
	scratch_load_b128 v[40:43] /*v[296:299]*/, off, off offset:364 th:TH_LOAD_LU nv
	scratch_load_b128 v[44:47] /*v[300:303]*/, off, off offset:380 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0x4005
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[72:79], v[40:47] /*v[296:303]*/, v[32:39] /*v[288:295]*/, v[72:79]
	s_set_vgpr_msb 0x509
	v_wmma_f32_16x16x32_bf16 v[8:15], v[40:47] /*v[296:303]*/, v[144:151] /*v[656:663]*/, v[8:15]
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x940
	s_clause 0x1
	scratch_load_b128 v[40:43] /*v[296:299]*/, off, off offset:332 th:TH_LOAD_LU nv
	scratch_load_b128 v[44:47] /*v[300:303]*/, off, off offset:348 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0x4009
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[72:79], v[40:47] /*v[296:303]*/, v[136:143] /*v[648:655]*/, v[72:79]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[40:47] /*v[296:303]*/, v[152:159] /*v[664:671]*/, v[8:15]
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x940
	s_clause 0x1
	scratch_load_b128 v[40:43] /*v[296:299]*/, off, off offset:300 th:TH_LOAD_LU nv
	scratch_load_b128 v[44:47] /*v[300:303]*/, off, off offset:316 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0x4001
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[64:71], v[40:47] /*v[296:303]*/, v[192:199], v[64:71]
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b128 v[192:195], off, off offset:268 th:TH_LOAD_LU nv
	scratch_load_b128 v[196:199], off, off offset:284 th:TH_LOAD_LU nv
	v_wmma_f32_16x16x32_bf16 v[0:7], v[40:47] /*v[296:303]*/, v[200:207], v[0:7]
	s_set_vgpr_msb 0x100
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[64:71], v[192:199], v[208:215], v[64:71]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[192:199], v[216:223], v[0:7]
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b128 v[192:195], off, off offset:236 th:TH_LOAD_LU nv
	scratch_load_b128 v[196:199], off, off offset:252 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[64:71], v[192:199], v[224:231], v[64:71]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[192:199], v[232:239], v[0:7]
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b128 v[192:195], off, off offset:204 th:TH_LOAD_LU nv
	scratch_load_b128 v[196:199], off, off offset:220 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[64:71], v[192:199], v[240:247], v[64:71]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[192:199], v[248:255], v[0:7]
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b128 v[192:195], off, off offset:172 th:TH_LOAD_LU nv
	scratch_load_b128 v[196:199], off, off offset:188 th:TH_LOAD_LU nv
	s_set_vgpr_msb 4
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[64:71], v[192:199], v[0:7] /*v[256:263]*/, v[64:71]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[192:199], v[8:15] /*v[264:271]*/, v[0:7]
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b128 v[192:195], off, off offset:140 th:TH_LOAD_LU nv
	scratch_load_b128 v[196:199], off, off offset:156 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[64:71], v[192:199], v[16:23] /*v[272:279]*/, v[64:71]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[192:199], v[24:31] /*v[280:287]*/, v[0:7]
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x2
	scratch_load_b128 v[194:197], off, off offset:108 th:TH_LOAD_LU nv
	scratch_load_b128 v[198:201], off, off offset:124 th:TH_LOAD_LU nv
	scratch_load_b64 v[192:193], off, off th:TH_LOAD_LU nv
	s_set_vgpr_msb 0x403
	s_wait_loadcnt_dscnt 0x0
	v_dual_fmac_f32 v130, v194 /*v962*/, v192 :: v_dual_fmac_f32 v132, v196 /*v964*/, v193
	s_set_vgpr_msb 0x304
	v_wmma_f32_16x16x32_bf16 v[64:71], v[194:201], v[32:39] /*v[288:295]*/, v[64:71]
	s_set_vgpr_msb 0x40c
	v_dual_add_f32 v204, v130, v192 /*v960*/ :: v_dual_add_f32 v205, v132, v193 /*v961*/
	v_dual_mov_b32 v130, v128 :: v_dual_mov_b32 v128, v135
	v_mov_b32_e32 v135, v129
	s_set_vgpr_msb 0xc08
	v_wmma_f32_16x16x32_bf16 v[0:7], v[194:201], v[144:151] /*v[656:663]*/, v[0:7]
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x2
	scratch_load_b128 v[192:195], off, off offset:76 th:TH_LOAD_LU nv
	scratch_load_b128 v[196:199], off, off offset:92 th:TH_LOAD_LU nv
	scratch_load_b32 v129, off, off offset:44 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0x801
	v_mov_b32_e32 v132, v50 /*v306*/
	s_set_vgpr_msb 0x108
	s_wait_loadcnt 0x1
	v_wmma_f32_16x16x32_bf16 v[64:71], v[192:199], v[136:143] /*v[648:655]*/, v[64:71]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[192:199], v[152:159] /*v[664:671]*/, v[0:7]
	s_set_vgpr_msb 0x800
	s_cbranch_vccnz .LBB0_44
.LBB0_38:
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_store_b64 off, v[204:205], off nv
	scratch_store_b32 off, v135, off offset:44 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	v_mov_b32_e32 v135, v130
	s_add_co_i32 s2, s44, 1
	s_wait_tensorcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	s_set_vgpr_msb 0x80
	ds_load_b128 v[184:187] /*v[696:699]*/, v129
	ds_load_b128 v[188:191] /*v[700:703]*/, v129 offset:32
	ds_load_b128 v[176:179] /*v[688:691]*/, v129 offset:64
	ds_load_b128 v[180:183] /*v[692:695]*/, v129 offset:96
	ds_load_b128 v[168:171] /*v[680:683]*/, v129 offset:128
	ds_load_b128 v[172:175] /*v[684:687]*/, v129 offset:160
	ds_load_b128 v[160:163] /*v[672:675]*/, v129 offset:192
	ds_load_b128 v[164:167] /*v[676:679]*/, v129 offset:224
	ds_load_b128 v[152:155] /*v[664:667]*/, v129 offset:4352
	ds_load_b128 v[156:159] /*v[668:671]*/, v129 offset:4384
	ds_load_b128 v[144:147] /*v[656:659]*/, v129 offset:4416
	ds_load_b128 v[148:151] /*v[660:663]*/, v129 offset:4448
	ds_load_b128 v[136:139] /*v[648:651]*/, v129 offset:4480
	ds_load_b128 v[140:143] /*v[652:655]*/, v129 offset:4512
	ds_load_b128 v[128:131] /*v[640:643]*/, v129 offset:4544
	ds_load_b128 v[132:135] /*v[644:647]*/, v129 offset:4576
	ds_load_b128 v[120:123] /*v[632:635]*/, v129 offset:8704
	ds_load_b128 v[124:127] /*v[636:639]*/, v129 offset:8736
	ds_load_b128 v[112:115] /*v[624:627]*/, v129 offset:8768
	ds_load_b128 v[116:119] /*v[628:631]*/, v129 offset:8800
	ds_load_b128 v[104:107] /*v[616:619]*/, v129 offset:8832
	ds_load_b128 v[108:111] /*v[620:623]*/, v129 offset:8864
	ds_load_b128 v[96:99] /*v[608:611]*/, v129 offset:8896
	ds_load_b128 v[100:103] /*v[612:615]*/, v129 offset:8928
	ds_load_b128 v[88:91] /*v[600:603]*/, v129 offset:13056
	ds_load_b128 v[92:95] /*v[604:607]*/, v129 offset:13088
	ds_load_b128 v[80:83] /*v[592:595]*/, v129 offset:13120
	ds_load_b128 v[84:87] /*v[596:599]*/, v129 offset:13152
	ds_load_b128 v[72:75] /*v[584:587]*/, v129 offset:13184
	ds_load_b128 v[76:79] /*v[588:591]*/, v129 offset:13216
	ds_load_b128 v[64:67] /*v[576:579]*/, v129 offset:13248
	ds_load_b128 v[68:71] /*v[580:583]*/, v129 offset:13280
	ds_load_b128 v[56:59] /*v[568:571]*/, v129 offset:17408
	ds_load_b128 v[60:63] /*v[572:575]*/, v129 offset:17440
	ds_load_b128 v[48:51] /*v[560:563]*/, v129 offset:17472
	ds_load_b128 v[52:55] /*v[564:567]*/, v129 offset:17504
	ds_load_b128 v[40:43] /*v[552:555]*/, v129 offset:17536
	ds_load_b128 v[44:47] /*v[556:559]*/, v129 offset:17568
	ds_load_b128 v[32:35] /*v[544:547]*/, v129 offset:17600
	ds_load_b128 v[36:39] /*v[548:551]*/, v129 offset:17632
	ds_load_b128 v[24:27] /*v[536:539]*/, v129 offset:21760
	ds_load_b128 v[28:31] /*v[540:543]*/, v129 offset:21792
	ds_load_b128 v[16:19] /*v[528:531]*/, v129 offset:21824
	ds_load_b128 v[20:23] /*v[532:535]*/, v129 offset:21856
	ds_load_b128 v[8:11] /*v[520:523]*/, v129 offset:21888
	ds_load_b128 v[12:15] /*v[524:527]*/, v129 offset:21920
	ds_load_b128 v[0:3] /*v[512:515]*/, v129 offset:21952
	ds_load_b128 v[4:7] /*v[516:519]*/, v129 offset:21984
	s_set_vgpr_msb 0x8040
	ds_load_b128 v[248:251] /*v[504:507]*/, v129 offset:26112
	ds_load_b128 v[252:255] /*v[508:511]*/, v129 offset:26144
	ds_load_b128 v[240:243] /*v[496:499]*/, v129 offset:26176
	ds_load_b128 v[244:247] /*v[500:503]*/, v129 offset:26208
	ds_load_b128 v[232:235] /*v[488:491]*/, v129 offset:26240
	ds_load_b128 v[236:239] /*v[492:495]*/, v129 offset:26272
	ds_load_b128 v[224:227] /*v[480:483]*/, v129 offset:26304
	ds_load_b128 v[228:231] /*v[484:487]*/, v129 offset:26336
	ds_load_b128 v[216:219] /*v[472:475]*/, v129 offset:30464
	ds_load_b128 v[220:223] /*v[476:479]*/, v129 offset:30496
	ds_load_b128 v[208:211] /*v[464:467]*/, v129 offset:30528
	ds_load_b128 v[212:215] /*v[468:471]*/, v129 offset:30560
	ds_load_b128 v[200:203] /*v[456:459]*/, v129 offset:30592
	ds_load_b128 v[204:207] /*v[460:463]*/, v129 offset:30624
	ds_load_b128 v[192:195] /*v[448:451]*/, v129 offset:30656
	ds_load_b128 v[196:199] /*v[452:455]*/, v129 offset:30688
	ds_load_b128 v[184:187] /*v[440:443]*/, v129 offset:34816
	ds_load_b128 v[188:191] /*v[444:447]*/, v129 offset:34848
	ds_load_b128 v[176:179] /*v[432:435]*/, v129 offset:34880
	ds_load_b128 v[180:183] /*v[436:439]*/, v129 offset:34912
	ds_load_b128 v[168:171] /*v[424:427]*/, v129 offset:34944
	ds_load_b128 v[172:175] /*v[428:431]*/, v129 offset:34976
	ds_load_b128 v[160:163] /*v[416:419]*/, v129 offset:35008
	ds_load_b128 v[164:167] /*v[420:423]*/, v129 offset:35040
	ds_load_b128 v[152:155] /*v[408:411]*/, v129 offset:39168
	ds_load_b128 v[156:159] /*v[412:415]*/, v129 offset:39200
	ds_load_b128 v[144:147] /*v[400:403]*/, v129 offset:39232
	ds_load_b128 v[148:151] /*v[404:407]*/, v129 offset:39264
	ds_load_b128 v[136:139] /*v[392:395]*/, v129 offset:39296
	ds_load_b128 v[140:143] /*v[396:399]*/, v129 offset:39328
	ds_load_b128 v[128:131] /*v[384:387]*/, v129 offset:39360
	ds_load_b128 v[132:135] /*v[388:391]*/, v129 offset:39392
	ds_load_b128 v[120:123] /*v[376:379]*/, v129 offset:43520
	ds_load_b128 v[124:127] /*v[380:383]*/, v129 offset:43552
	ds_load_b128 v[112:115] /*v[368:371]*/, v129 offset:43584
	ds_load_b128 v[116:119] /*v[372:375]*/, v129 offset:43616
	ds_load_b128 v[104:107] /*v[360:363]*/, v129 offset:43648
	ds_load_b128 v[108:111] /*v[364:367]*/, v129 offset:43680
	ds_load_b128 v[96:99] /*v[352:355]*/, v129 offset:43712
	ds_load_b128 v[100:103] /*v[356:359]*/, v129 offset:43744
	ds_load_b128 v[88:91] /*v[344:347]*/, v129 offset:47872
	ds_load_b128 v[92:95] /*v[348:351]*/, v129 offset:47904
	ds_load_b128 v[80:83] /*v[336:339]*/, v129 offset:47936
	ds_load_b128 v[84:87] /*v[340:343]*/, v129 offset:47968
	ds_load_b128 v[72:75] /*v[328:331]*/, v129 offset:48000
	ds_load_b128 v[76:79] /*v[332:335]*/, v129 offset:48032
	ds_load_b128 v[64:67] /*v[320:323]*/, v129 offset:48064
	ds_load_b128 v[68:71] /*v[324:327]*/, v129 offset:48096
	ds_load_b128 v[56:59] /*v[312:315]*/, v129 offset:52224
	ds_load_b128 v[60:63] /*v[316:319]*/, v129 offset:52256
	ds_load_b128 v[48:51] /*v[304:307]*/, v129 offset:52288
	ds_load_b128 v[52:55] /*v[308:311]*/, v129 offset:52320
	ds_load_b128 v[40:43] /*v[296:299]*/, v129 offset:52352
	ds_load_b128 v[44:47] /*v[300:303]*/, v129 offset:52384
	ds_load_b128 v[32:35] /*v[288:291]*/, v129 offset:52416
	ds_load_b128 v[36:39] /*v[292:295]*/, v129 offset:52448
	ds_load_b128 v[24:27] /*v[280:283]*/, v129 offset:56576
	ds_load_b128 v[28:31] /*v[284:287]*/, v129 offset:56608
	ds_load_b128 v[16:19] /*v[272:275]*/, v129 offset:56640
	ds_load_b128 v[20:23] /*v[276:279]*/, v129 offset:56672
	ds_load_b128 v[8:11] /*v[264:267]*/, v129 offset:56704
	ds_load_b128 v[12:15] /*v[268:271]*/, v129 offset:56736
	ds_load_b128 v[0:3] /*v[256:259]*/, v129 offset:56768
	ds_load_b128 v[4:7] /*v[260:263]*/, v129 offset:56800
	s_set_vgpr_msb 0x4000
	ds_load_b128 v[248:251], v129 offset:60928
	ds_load_b128 v[252:255], v129 offset:60960
	ds_load_b128 v[240:243], v129 offset:60992
	ds_load_b128 v[244:247], v129 offset:61024
	ds_load_b128 v[232:235], v129 offset:61056
	ds_load_b128 v[236:239], v129 offset:61088
	ds_load_b128 v[224:227], v129 offset:61120
	ds_load_b128 v[228:231], v129 offset:61152
	ds_load_b128 v[216:219], v129 offset:65280
	ds_load_b128 v[220:223], v129 offset:65312
	ds_load_b128 v[208:211], v129 offset:65344
	ds_load_b128 v[212:215], v129 offset:65376
	ds_load_b128 v[200:203], v129 offset:65408
	ds_load_b128 v[204:207], v129 offset:65440
	ds_load_b128 v[192:195], v129 offset:65472
	ds_load_b128 v[196:199], v129 offset:65504
	s_cmp_ge_i32 s2, s10
	s_cbranch_scc1 .LBB0_40
	s_lshl_b32 s5, s2, 8
	s_mov_b32 s21, s13
	s_add_co_i32 s6, s5, s64
	s_mov_b32 s24, s16
	s_ashr_i32 s7, s6, 31
	s_mov_b32 s27, s19
	s_mul_u64 s[22:23], s[6:7], s[62:63]
	s_mul_u64 s[6:7], s[6:7], s[60:61]
	s_lshl_b64 s[48:49], s[22:23], 1
	s_bitcmp1_b32 s2, 0
	s_add_nc_u64 s[48:49], s[40:41], s[48:49]
	s_cselect_b32 s2, 0x23000, 0
	s_sub_co_i32 s5, s35, s5
	s_lshl_b64 s[6:7], s[6:7], 1
	v_med3_i32 v130, s5, 0, 0x100
	s_add_nc_u64 s[6:7], s[42:43], s[6:7]
	s_mov_b32 s23, s15
	s_add_nc_u64 s[6:7], s[36:37], s[6:7]
	v_readfirstlane_b32 s5, v130
	s_bitset1_b32 s7, 31
	s_sub_co_i32 s14, s5, s53
	s_add_co_i32 s5, s54, s2
	s_max_i32 s14, s14, 0
	s_lshl_b32 s14, s14, 16
	s_addk_co_i32 s14, 0x7fff
	tensor_load_to_lds s[4:7], s[12:19]
	s_add_nc_u64 s[6:7], s[38:39], s[48:49]
	s_add_co_i32 s5, s55, s2
	s_mov_b32 s22, s14
	s_bitset1_b32 s7, 31
	tensor_load_to_lds s[4:7], s[20:27]
.LBB0_40:
	s_set_vgpr_msb 0xc0
	s_clause 0x1
	scratch_load_b128 v[56:59] /*v[824:827]*/, off, off offset:12 nv
	scratch_load_b128 v[60:63] /*v[828:831]*/, off, off offset:28 nv
	s_set_vgpr_msb 0xc082
	s_wait_dscnt 0x3e
	v_wmma_f32_16x16x32_bf16 v[192:199] /*v[704:711]*/, v[184:191] /*v[696:703]*/, v[160:167], 0
	v_wmma_f32_16x16x32_bf16 v[200:207] /*v[712:719]*/, v[152:159] /*v[664:671]*/, v[160:167], 0
	v_wmma_f32_16x16x32_bf16 v[208:215] /*v[720:727]*/, v[120:127] /*v[632:639]*/, v[160:167], 0
	v_wmma_f32_16x16x32_bf16 v[216:223] /*v[728:735]*/, v[88:95] /*v[600:607]*/, v[160:167], 0
	v_wmma_f32_16x16x32_bf16 v[224:231] /*v[736:743]*/, v[56:63] /*v[568:575]*/, v[160:167], 0
	v_wmma_f32_16x16x32_bf16 v[232:239] /*v[744:751]*/, v[24:31] /*v[536:543]*/, v[160:167], 0
	s_set_vgpr_msb 0x8281
	v_wmma_f32_16x16x32_bf16 v[240:247] /*v[752:759]*/, v[248:255] /*v[504:511]*/, v[160:167], 0
	v_wmma_f32_16x16x32_bf16 v[248:255] /*v[760:767]*/, v[216:223] /*v[472:479]*/, v[160:167], 0
	s_set_vgpr_msb 0x81c1
	v_wmma_f32_16x16x32_bf16 v[0:7] /*v[768:775]*/, v[184:191] /*v[440:447]*/, v[160:167], 0
	s_wait_dscnt 0x36
	v_wmma_f32_16x16x32_bf16 v[8:15] /*v[776:783]*/, v[152:159] /*v[408:415]*/, v[160:167], 0
	s_wait_dscnt 0x2e
	v_wmma_f32_16x16x32_bf16 v[16:23] /*v[784:791]*/, v[120:127] /*v[376:383]*/, v[160:167], 0
	s_wait_dscnt 0x26
	v_wmma_f32_16x16x32_bf16 v[24:31] /*v[792:799]*/, v[88:95] /*v[344:351]*/, v[160:167], 0
	s_wait_dscnt 0x1e
	v_wmma_f32_16x16x32_bf16 v[32:39] /*v[800:807]*/, v[56:63] /*v[312:319]*/, v[160:167], 0
	s_wait_dscnt 0x16
	v_wmma_f32_16x16x32_bf16 v[40:47] /*v[808:815]*/, v[24:31] /*v[280:287]*/, v[160:167], 0
	s_set_vgpr_msb 0xc1c0
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x32_bf16 v[48:55] /*v[816:823]*/, v[248:255], v[160:167], 0
	s_set_vgpr_msb 0xc0a2
	v_wmma_f32_16x16x32_bf16 v[192:199] /*v[704:711]*/, v[176:183] /*v[688:695]*/, v[168:175], v[192:199] /*v[704:711]*/
	v_wmma_f32_16x16x32_bf16 v[200:207] /*v[712:719]*/, v[144:151] /*v[656:663]*/, v[168:175], v[200:207] /*v[712:719]*/
	v_wmma_f32_16x16x32_bf16 v[208:215] /*v[720:727]*/, v[112:119] /*v[624:631]*/, v[168:175], v[208:215] /*v[720:727]*/
	v_wmma_f32_16x16x32_bf16 v[216:223] /*v[728:735]*/, v[80:87] /*v[592:599]*/, v[168:175], v[216:223] /*v[728:735]*/
	v_wmma_f32_16x16x32_bf16 v[224:231] /*v[736:743]*/, v[48:55] /*v[560:567]*/, v[168:175], v[224:231] /*v[736:743]*/
	v_wmma_f32_16x16x32_bf16 v[232:239] /*v[744:751]*/, v[16:23] /*v[528:535]*/, v[168:175], v[232:239] /*v[744:751]*/
	s_set_vgpr_msb 0xa2a1
	v_wmma_f32_16x16x32_bf16 v[240:247] /*v[752:759]*/, v[240:247] /*v[496:503]*/, v[168:175], v[240:247] /*v[752:759]*/
	v_wmma_f32_16x16x32_bf16 v[248:255] /*v[760:767]*/, v[208:215] /*v[464:471]*/, v[168:175], v[248:255] /*v[760:767]*/
	s_set_vgpr_msb 0xa1f1
	v_wmma_f32_16x16x32_bf16 v[0:7] /*v[768:775]*/, v[176:183] /*v[432:439]*/, v[168:175], v[0:7] /*v[768:775]*/
	v_wmma_f32_16x16x32_bf16 v[8:15] /*v[776:783]*/, v[144:151] /*v[400:407]*/, v[168:175], v[8:15] /*v[776:783]*/
	v_wmma_f32_16x16x32_bf16 v[16:23] /*v[784:791]*/, v[112:119] /*v[368:375]*/, v[168:175], v[16:23] /*v[784:791]*/
	v_wmma_f32_16x16x32_bf16 v[24:31] /*v[792:799]*/, v[80:87] /*v[336:343]*/, v[168:175], v[24:31] /*v[792:799]*/
	v_wmma_f32_16x16x32_bf16 v[32:39] /*v[800:807]*/, v[48:55] /*v[304:311]*/, v[168:175], v[32:39] /*v[800:807]*/
	v_wmma_f32_16x16x32_bf16 v[40:47] /*v[808:815]*/, v[16:23] /*v[272:279]*/, v[168:175], v[40:47] /*v[808:815]*/
	s_set_vgpr_msb 0xf1f0
	s_wait_dscnt 0xc
	v_wmma_f32_16x16x32_bf16 v[48:55] /*v[816:823]*/, v[240:247], v[168:175], v[48:55] /*v[816:823]*/
	s_set_vgpr_msb 0xf0a2
	v_wmma_f32_16x16x32_bf16 v[192:199] /*v[704:711]*/, v[168:175] /*v[680:687]*/, v[176:183], v[192:199] /*v[704:711]*/
	v_wmma_f32_16x16x32_bf16 v[200:207] /*v[712:719]*/, v[136:143] /*v[648:655]*/, v[176:183], v[200:207] /*v[712:719]*/
	v_wmma_f32_16x16x32_bf16 v[208:215] /*v[720:727]*/, v[104:111] /*v[616:623]*/, v[176:183], v[208:215] /*v[720:727]*/
	v_wmma_f32_16x16x32_bf16 v[216:223] /*v[728:735]*/, v[72:79] /*v[584:591]*/, v[176:183], v[216:223] /*v[728:735]*/
	v_wmma_f32_16x16x32_bf16 v[224:231] /*v[736:743]*/, v[40:47] /*v[552:559]*/, v[176:183], v[224:231] /*v[736:743]*/
	v_wmma_f32_16x16x32_bf16 v[232:239] /*v[744:751]*/, v[8:15] /*v[520:527]*/, v[176:183], v[232:239] /*v[744:751]*/
	s_set_vgpr_msb 0xa2a1
	v_wmma_f32_16x16x32_bf16 v[240:247] /*v[752:759]*/, v[232:239] /*v[488:495]*/, v[176:183], v[240:247] /*v[752:759]*/
	v_wmma_f32_16x16x32_bf16 v[248:255] /*v[760:767]*/, v[200:207] /*v[456:463]*/, v[176:183], v[248:255] /*v[760:767]*/
	s_set_vgpr_msb 0xa1f1
	v_wmma_f32_16x16x32_bf16 v[0:7] /*v[768:775]*/, v[168:175] /*v[424:431]*/, v[176:183], v[0:7] /*v[768:775]*/
	v_wmma_f32_16x16x32_bf16 v[8:15] /*v[776:783]*/, v[136:143] /*v[392:399]*/, v[176:183], v[8:15] /*v[776:783]*/
	s_set_vgpr_msb 0xf1ce
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[178:185] /*v[946:953]*/, v[184:191] /*v[696:703]*/, v[56:63] /*v[824:831]*/, 0
	v_wmma_f32_16x16x32_bf16 v[186:193] /*v[954:961]*/, v[152:159] /*v[664:671]*/, v[56:63] /*v[824:831]*/, 0
	v_wmma_f32_16x16x32_bf16 v[168:175] /*v[936:943]*/, v[120:127] /*v[632:639]*/, v[56:63] /*v[824:831]*/, 0
	v_wmma_f32_16x16x32_bf16 v[160:167] /*v[928:935]*/, v[88:95] /*v[600:607]*/, v[56:63] /*v[824:831]*/, 0
	v_wmma_f32_16x16x32_bf16 v[152:159] /*v[920:927]*/, v[56:63] /*v[568:575]*/, v[56:63] /*v[824:831]*/, 0
	v_wmma_f32_16x16x32_bf16 v[144:151] /*v[912:919]*/, v[24:31] /*v[536:543]*/, v[56:63] /*v[824:831]*/, 0
	s_set_vgpr_msb 0xcecd
	v_wmma_f32_16x16x32_bf16 v[136:143] /*v[904:911]*/, v[248:255] /*v[504:511]*/, v[56:63] /*v[824:831]*/, 0
	v_wmma_f32_16x16x32_bf16 v[128:135] /*v[896:903]*/, v[216:223] /*v[472:479]*/, v[56:63] /*v[824:831]*/, 0
	v_wmma_f32_16x16x32_bf16 v[120:127] /*v[888:895]*/, v[184:191] /*v[440:447]*/, v[56:63] /*v[824:831]*/, 0
	v_wmma_f32_16x16x32_bf16 v[112:119] /*v[880:887]*/, v[152:159] /*v[408:415]*/, v[56:63] /*v[824:831]*/, 0
	v_wmma_f32_16x16x32_bf16 v[104:111] /*v[872:879]*/, v[120:127] /*v[376:383]*/, v[56:63] /*v[824:831]*/, 0
	v_wmma_f32_16x16x32_bf16 v[96:103] /*v[864:871]*/, v[88:95] /*v[344:351]*/, v[56:63] /*v[824:831]*/, 0
	v_wmma_f32_16x16x32_bf16 v[88:95] /*v[856:863]*/, v[56:63] /*v[312:319]*/, v[56:63] /*v[824:831]*/, 0
	v_wmma_f32_16x16x32_bf16 v[80:87] /*v[848:855]*/, v[24:31] /*v[280:287]*/, v[56:63] /*v[824:831]*/, 0
	s_set_vgpr_msb 0xcdcc
	v_wmma_f32_16x16x32_bf16 v[72:79] /*v[840:847]*/, v[248:255], v[56:63] /*v[824:831]*/, 0
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x32_bf16 v[64:71] /*v[832:839]*/, v[216:223], v[56:63] /*v[824:831]*/, 0
	s_set_vgpr_msb 0xccc0
	v_wmma_f32_16x16x32_bf16 v[56:63] /*v[824:831]*/, v[216:223], v[160:167], 0
	s_set_vgpr_msb 0xc0f2
	v_wmma_f32_16x16x32_bf16 v[178:185] /*v[946:953]*/, v[176:183] /*v[688:695]*/, v[136:143], v[178:185] /*v[946:953]*/
	v_wmma_f32_16x16x32_bf16 v[186:193] /*v[954:961]*/, v[144:151] /*v[656:663]*/, v[136:143], v[186:193] /*v[954:961]*/
	v_wmma_f32_16x16x32_bf16 v[168:175] /*v[936:943]*/, v[112:119] /*v[624:631]*/, v[136:143], v[168:175] /*v[936:943]*/
	v_wmma_f32_16x16x32_bf16 v[160:167] /*v[928:935]*/, v[80:87] /*v[592:599]*/, v[136:143], v[160:167] /*v[928:935]*/
	v_wmma_f32_16x16x32_bf16 v[152:159] /*v[920:927]*/, v[48:55] /*v[560:567]*/, v[136:143], v[152:159] /*v[920:927]*/
	v_wmma_f32_16x16x32_bf16 v[144:151] /*v[912:919]*/, v[16:23] /*v[528:535]*/, v[136:143], v[144:151] /*v[912:919]*/
	s_set_vgpr_msb 0xf2f1
	v_wmma_f32_16x16x32_bf16 v[136:143] /*v[904:911]*/, v[240:247] /*v[496:503]*/, v[136:143], v[136:143] /*v[904:911]*/
	v_wmma_f32_16x16x32_bf16 v[128:135] /*v[896:903]*/, v[208:215] /*v[464:471]*/, v[136:143], v[128:135] /*v[896:903]*/
	v_wmma_f32_16x16x32_bf16 v[120:127] /*v[888:895]*/, v[176:183] /*v[432:439]*/, v[136:143], v[120:127] /*v[888:895]*/
	v_wmma_f32_16x16x32_bf16 v[112:119] /*v[880:887]*/, v[144:151] /*v[400:407]*/, v[136:143], v[112:119] /*v[880:887]*/
	v_wmma_f32_16x16x32_bf16 v[104:111] /*v[872:879]*/, v[112:119] /*v[368:375]*/, v[136:143], v[104:111] /*v[872:879]*/
	v_wmma_f32_16x16x32_bf16 v[96:103] /*v[864:871]*/, v[80:87] /*v[336:343]*/, v[136:143], v[96:103] /*v[864:871]*/
	v_wmma_f32_16x16x32_bf16 v[88:95] /*v[856:863]*/, v[48:55] /*v[304:311]*/, v[136:143], v[88:95] /*v[856:863]*/
	v_wmma_f32_16x16x32_bf16 v[80:87] /*v[848:855]*/, v[16:23] /*v[272:279]*/, v[136:143], v[80:87] /*v[848:855]*/
	s_set_vgpr_msb 0xf1f0
	v_wmma_f32_16x16x32_bf16 v[72:79] /*v[840:847]*/, v[240:247], v[136:143], v[72:79] /*v[840:847]*/
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x32_bf16 v[64:71] /*v[832:839]*/, v[208:215], v[136:143], v[64:71] /*v[832:839]*/
	v_wmma_f32_16x16x32_bf16 v[56:63] /*v[824:831]*/, v[208:215], v[168:175], v[56:63] /*v[824:831]*/
	s_set_vgpr_msb 0xf0f2
	v_wmma_f32_16x16x32_bf16 v[178:185] /*v[946:953]*/, v[168:175] /*v[680:687]*/, v[144:151], v[178:185] /*v[946:953]*/
	v_wmma_f32_16x16x32_bf16 v[186:193] /*v[954:961]*/, v[136:143] /*v[648:655]*/, v[144:151], v[186:193] /*v[954:961]*/
	v_wmma_f32_16x16x32_bf16 v[168:175] /*v[936:943]*/, v[104:111] /*v[616:623]*/, v[144:151], v[168:175] /*v[936:943]*/
	v_wmma_f32_16x16x32_bf16 v[160:167] /*v[928:935]*/, v[72:79] /*v[584:591]*/, v[144:151], v[160:167] /*v[928:935]*/
	v_wmma_f32_16x16x32_bf16 v[152:159] /*v[920:927]*/, v[40:47] /*v[552:559]*/, v[144:151], v[152:159] /*v[920:927]*/
	v_wmma_f32_16x16x32_bf16 v[144:151] /*v[912:919]*/, v[8:15] /*v[520:527]*/, v[144:151], v[144:151] /*v[912:919]*/
	s_set_vgpr_msb 0xf2f1
	v_wmma_f32_16x16x32_bf16 v[136:143] /*v[904:911]*/, v[232:239] /*v[488:495]*/, v[144:151], v[136:143] /*v[904:911]*/
	v_wmma_f32_16x16x32_bf16 v[128:135] /*v[896:903]*/, v[200:207] /*v[456:463]*/, v[144:151], v[128:135] /*v[896:903]*/
	v_wmma_f32_16x16x32_bf16 v[120:127] /*v[888:895]*/, v[168:175] /*v[424:431]*/, v[144:151], v[120:127] /*v[888:895]*/
	v_wmma_f32_16x16x32_bf16 v[112:119] /*v[880:887]*/, v[136:143] /*v[392:399]*/, v[144:151], v[112:119] /*v[880:887]*/
	v_wmma_f32_16x16x32_bf16 v[104:111] /*v[872:879]*/, v[104:111] /*v[360:367]*/, v[144:151], v[104:111] /*v[872:879]*/
	v_wmma_f32_16x16x32_bf16 v[16:23] /*v[784:791]*/, v[104:111] /*v[360:367]*/, v[176:183], v[16:23] /*v[784:791]*/
	v_wmma_f32_16x16x32_bf16 v[96:103] /*v[864:871]*/, v[72:79] /*v[328:335]*/, v[144:151], v[96:103] /*v[864:871]*/
	v_wmma_f32_16x16x32_bf16 v[24:31] /*v[792:799]*/, v[72:79] /*v[328:335]*/, v[176:183], v[24:31] /*v[792:799]*/
	v_wmma_f32_16x16x32_bf16 v[88:95] /*v[856:863]*/, v[40:47] /*v[296:303]*/, v[144:151], v[88:95] /*v[856:863]*/
	v_wmma_f32_16x16x32_bf16 v[32:39] /*v[800:807]*/, v[40:47] /*v[296:303]*/, v[176:183], v[32:39] /*v[800:807]*/
	v_wmma_f32_16x16x32_bf16 v[80:87] /*v[848:855]*/, v[8:15] /*v[264:271]*/, v[144:151], v[80:87] /*v[848:855]*/
	v_wmma_f32_16x16x32_bf16 v[40:47] /*v[808:815]*/, v[8:15] /*v[264:271]*/, v[176:183], v[40:47] /*v[808:815]*/
	s_set_vgpr_msb 0xf1f0
	v_wmma_f32_16x16x32_bf16 v[72:79] /*v[840:847]*/, v[232:239], v[144:151], v[72:79] /*v[840:847]*/
	v_wmma_f32_16x16x32_bf16 v[48:55] /*v[816:823]*/, v[232:239], v[176:183], v[48:55] /*v[816:823]*/
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[64:71] /*v[832:839]*/, v[200:207], v[144:151], v[64:71] /*v[832:839]*/
	v_wmma_f32_16x16x32_bf16 v[56:63] /*v[824:831]*/, v[200:207], v[176:183], v[56:63] /*v[824:831]*/
	s_set_vgpr_msb 0xf0f2
	v_wmma_f32_16x16x32_bf16 v[178:185] /*v[946:953]*/, v[160:167] /*v[672:679]*/, v[152:159], v[178:185] /*v[946:953]*/
	s_set_vgpr_msb 0xf2a2
	v_wmma_f32_16x16x32_bf16 v[192:199] /*v[704:711]*/, v[160:167] /*v[672:679]*/, v[184:191], v[192:199] /*v[704:711]*/
	s_set_vgpr_msb 0xa2f2
	v_wmma_f32_16x16x32_bf16 v[186:193] /*v[954:961]*/, v[128:135] /*v[640:647]*/, v[152:159], v[186:193] /*v[954:961]*/
	s_set_vgpr_msb 0xf2a2
	v_wmma_f32_16x16x32_bf16 v[200:207] /*v[712:719]*/, v[128:135] /*v[640:647]*/, v[184:191], v[200:207] /*v[712:719]*/
	s_set_vgpr_msb 0xa2f2
	v_wmma_f32_16x16x32_bf16 v[168:175] /*v[936:943]*/, v[96:103] /*v[608:615]*/, v[152:159], v[168:175] /*v[936:943]*/
	s_set_vgpr_msb 0xf2a2
	v_wmma_f32_16x16x32_bf16 v[208:215] /*v[720:727]*/, v[96:103] /*v[608:615]*/, v[184:191], v[208:215] /*v[720:727]*/
	s_set_vgpr_msb 0xa2f2
	v_wmma_f32_16x16x32_bf16 v[160:167] /*v[928:935]*/, v[64:71] /*v[576:583]*/, v[152:159], v[160:167] /*v[928:935]*/
	s_set_vgpr_msb 0xf2a2
	v_wmma_f32_16x16x32_bf16 v[216:223] /*v[728:735]*/, v[64:71] /*v[576:583]*/, v[184:191], v[216:223] /*v[728:735]*/
	s_set_vgpr_msb 0xa2f2
	v_wmma_f32_16x16x32_bf16 v[152:159] /*v[920:927]*/, v[32:39] /*v[544:551]*/, v[152:159], v[152:159] /*v[920:927]*/
	s_set_vgpr_msb 0xf2a2
	v_wmma_f32_16x16x32_bf16 v[224:231] /*v[736:743]*/, v[32:39] /*v[544:551]*/, v[184:191], v[224:231] /*v[736:743]*/
	s_set_vgpr_msb 0xa2f2
	v_wmma_f32_16x16x32_bf16 v[144:151] /*v[912:919]*/, v[0:7] /*v[512:519]*/, v[152:159], v[144:151] /*v[912:919]*/
	s_set_vgpr_msb 0xf2a2
	v_wmma_f32_16x16x32_bf16 v[232:239] /*v[744:751]*/, v[0:7] /*v[512:519]*/, v[184:191], v[232:239] /*v[744:751]*/
	s_set_vgpr_msb 0xa2f1
	v_wmma_f32_16x16x32_bf16 v[136:143] /*v[904:911]*/, v[224:231] /*v[480:487]*/, v[152:159], v[136:143] /*v[904:911]*/
	s_set_vgpr_msb 0xf1a1
	v_wmma_f32_16x16x32_bf16 v[240:247] /*v[752:759]*/, v[224:231] /*v[480:487]*/, v[184:191], v[240:247] /*v[752:759]*/
	s_set_vgpr_msb 0xa1f1
	v_wmma_f32_16x16x32_bf16 v[128:135] /*v[896:903]*/, v[192:199] /*v[448:455]*/, v[152:159], v[128:135] /*v[896:903]*/
	s_set_vgpr_msb 0xf1a1
	v_wmma_f32_16x16x32_bf16 v[248:255] /*v[760:767]*/, v[192:199] /*v[448:455]*/, v[184:191], v[248:255] /*v[760:767]*/
	s_set_vgpr_msb 0xa1f1
	v_wmma_f32_16x16x32_bf16 v[120:127] /*v[888:895]*/, v[160:167] /*v[416:423]*/, v[152:159], v[120:127] /*v[888:895]*/
	v_wmma_f32_16x16x32_bf16 v[0:7] /*v[768:775]*/, v[160:167] /*v[416:423]*/, v[184:191], v[0:7] /*v[768:775]*/
	v_wmma_f32_16x16x32_bf16 v[112:119] /*v[880:887]*/, v[128:135] /*v[384:391]*/, v[152:159], v[112:119] /*v[880:887]*/
	v_wmma_f32_16x16x32_bf16 v[8:15] /*v[776:783]*/, v[128:135] /*v[384:391]*/, v[184:191], v[8:15] /*v[776:783]*/
	v_wmma_f32_16x16x32_bf16 v[104:111] /*v[872:879]*/, v[96:103] /*v[352:359]*/, v[152:159], v[104:111] /*v[872:879]*/
	v_wmma_f32_16x16x32_bf16 v[16:23] /*v[784:791]*/, v[96:103] /*v[352:359]*/, v[184:191], v[16:23] /*v[784:791]*/
	v_wmma_f32_16x16x32_bf16 v[96:103] /*v[864:871]*/, v[64:71] /*v[320:327]*/, v[152:159], v[96:103] /*v[864:871]*/
	v_wmma_f32_16x16x32_bf16 v[24:31] /*v[792:799]*/, v[64:71] /*v[320:327]*/, v[184:191], v[24:31] /*v[792:799]*/
	v_wmma_f32_16x16x32_bf16 v[88:95] /*v[856:863]*/, v[32:39] /*v[288:295]*/, v[152:159], v[88:95] /*v[856:863]*/
	v_wmma_f32_16x16x32_bf16 v[32:39] /*v[800:807]*/, v[32:39] /*v[288:295]*/, v[184:191], v[32:39] /*v[800:807]*/
	v_wmma_f32_16x16x32_bf16 v[80:87] /*v[848:855]*/, v[0:7] /*v[256:263]*/, v[152:159], v[80:87] /*v[848:855]*/
	v_wmma_f32_16x16x32_bf16 v[40:47] /*v[808:815]*/, v[0:7] /*v[256:263]*/, v[184:191], v[40:47] /*v[808:815]*/
	s_set_vgpr_msb 0xf1f0
	v_wmma_f32_16x16x32_bf16 v[72:79] /*v[840:847]*/, v[224:231], v[152:159], v[72:79] /*v[840:847]*/
	v_wmma_f32_16x16x32_bf16 v[48:55] /*v[816:823]*/, v[224:231], v[184:191], v[48:55] /*v[816:823]*/
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[64:71] /*v[832:839]*/, v[192:199], v[152:159], v[64:71] /*v[832:839]*/
	v_wmma_f32_16x16x32_bf16 v[56:63] /*v[824:831]*/, v[192:199], v[184:191], v[56:63] /*v[824:831]*/
	s_set_vgpr_msb 0xf000
	v_add_nc_u32_e32 v130, 0x10e00, v128
	v_nop
	v_nop
	v_nop
	v_add_nc_u32_e32 v192, 0x10e20, v128
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x80
	ds_load_tr16_b128 v[140:143] /*v[652:655]*/, v128 offset:59904
	ds_load_tr16_b128 v[68:71] /*v[580:583]*/, v128 offset:59936
	ds_load_tr16_b128 v[128:131] /*v[640:643]*/, v128 offset:64512
	ds_load_tr16_b128 v[48:51] /*v[560:563]*/, v128 offset:64544
	ds_load_tr16_b128 v[132:135] /*v[644:647]*/, v130
	ds_load_tr16_b128 v[52:55] /*v[564:567]*/, v192
	s_wait_alu depctr_vm_vsrc(0)
	s_set_vgpr_msb 0x8000
	v_add_nc_u32_e32 v130, 0x10e40, v128
	v_add_nc_u32_e32 v192, 0x10e60, v128
	s_set_vgpr_msb 0x80
	ds_load_tr16_b128 v[12:15] /*v[524:527]*/, v128 offset:59968
	s_set_vgpr_msb 0x8040
	ds_load_tr16_b128 v[196:199] /*v[452:455]*/, v128 offset:60000
	s_set_vgpr_msb 0x4080
	ds_load_tr16_b128 v[0:3] /*v[512:515]*/, v128 offset:64576
	s_set_vgpr_msb 0x8040
	ds_load_tr16_b128 v[176:179] /*v[432:435]*/, v128 offset:64608
	s_wait_alu depctr_va_vdst(1)
	s_set_vgpr_msb 0x4080
	ds_load_tr16_b128 v[4:7] /*v[516:519]*/, v130
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x8040
	ds_load_tr16_b128 v[180:183] /*v[436:439]*/, v192
	s_wait_alu depctr_vm_vsrc(1)
	s_set_vgpr_msb 0x4000
	v_add_nc_u32_e32 v130, 0x10e80, v128
	s_wait_alu depctr_vm_vsrc(0)
	v_add_nc_u32_e32 v192, 0x10ea0, v128
	s_set_vgpr_msb 64
	ds_load_tr16_b128 v[140:143] /*v[396:399]*/, v128 offset:60032
	ds_load_tr16_b128 v[68:71] /*v[324:327]*/, v128 offset:60064
	ds_load_tr16_b128 v[128:131] /*v[384:387]*/, v128 offset:64640
	s_set_vgpr_msb 0x4000
	ds_load_tr16_b128 v[194:197], v128 offset:64672
	s_wait_alu depctr_va_vdst(1)
	s_set_vgpr_msb 64
	ds_load_tr16_b128 v[132:135] /*v[388:391]*/, v130
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x4000
	ds_load_tr16_b128 v[198:201], v192
	s_set_vgpr_msb 0x80
	ds_load_tr16_b128 v[188:191] /*v[700:703]*/, v128 offset:4608
	ds_load_tr16_b128 v[108:111] /*v[620:623]*/, v128 offset:4640
	ds_load_tr16_b128 v[176:179] /*v[688:691]*/, v128 offset:9216
	ds_load_tr16_b128 v[120:123] /*v[632:635]*/, v128 offset:9248
	ds_load_tr16_b128 v[180:183] /*v[692:695]*/, v128 offset:13824
	ds_load_tr16_b128 v[124:127] /*v[636:639]*/, v128 offset:13856
	ds_load_tr16_b128 v[168:171] /*v[680:683]*/, v128 offset:18432
	ds_load_tr16_b128 v[112:115] /*v[624:627]*/, v128 offset:18464
	ds_load_tr16_b128 v[172:175] /*v[684:687]*/, v128 offset:23040
	ds_load_tr16_b128 v[116:119] /*v[628:631]*/, v128 offset:23072
	ds_load_tr16_b128 v[160:163] /*v[672:675]*/, v128 offset:27648
	ds_load_tr16_b128 v[96:99] /*v[608:611]*/, v128 offset:27680
	ds_load_tr16_b128 v[164:167] /*v[676:679]*/, v128 offset:32256
	ds_load_tr16_b128 v[100:103] /*v[612:615]*/, v128 offset:32288
	ds_load_tr16_b128 v[152:155] /*v[664:667]*/, v128 offset:36864
	ds_load_tr16_b128 v[88:91] /*v[600:603]*/, v128 offset:36896
	ds_load_tr16_b128 v[156:159] /*v[668:671]*/, v128 offset:41472
	ds_load_tr16_b128 v[92:95] /*v[604:607]*/, v128 offset:41504
	ds_load_tr16_b128 v[144:147] /*v[656:659]*/, v128 offset:46080
	ds_load_tr16_b128 v[80:83] /*v[592:595]*/, v128 offset:46112
	ds_load_tr16_b128 v[148:151] /*v[660:663]*/, v128 offset:50688
	ds_load_tr16_b128 v[84:87] /*v[596:599]*/, v128 offset:50720
	ds_load_tr16_b128 v[136:139] /*v[648:651]*/, v128 offset:55296
	ds_load_tr16_b128 v[64:67] /*v[576:579]*/, v128 offset:55328
	ds_load_tr16_b128 v[72:75] /*v[584:587]*/, v128 offset:64
	s_set_vgpr_msb 0x8040
	ds_load_tr16_b128 v[248:251] /*v[504:507]*/, v128 offset:96
	s_set_vgpr_msb 0x4080
	ds_load_tr16_b128 v[76:79] /*v[588:591]*/, v128 offset:4672
	s_set_vgpr_msb 0x8040
	ds_load_tr16_b128 v[252:255] /*v[508:511]*/, v128 offset:4704
	s_set_vgpr_msb 0x4080
	ds_load_tr16_b128 v[56:59] /*v[568:571]*/, v128 offset:9280
	s_set_vgpr_msb 0x8040
	ds_load_tr16_b128 v[240:243] /*v[496:499]*/, v128 offset:9312
	s_set_vgpr_msb 0x4080
	ds_load_tr16_b128 v[60:63] /*v[572:575]*/, v128 offset:13888
	s_set_vgpr_msb 0x8040
	ds_load_tr16_b128 v[244:247] /*v[500:503]*/, v128 offset:13920
	s_set_vgpr_msb 0x4080
	ds_load_tr16_b128 v[40:43] /*v[552:555]*/, v128 offset:18496
	s_set_vgpr_msb 0x8040
	ds_load_tr16_b128 v[232:235] /*v[488:491]*/, v128 offset:18528
	s_set_vgpr_msb 0x4080
	ds_load_tr16_b128 v[44:47] /*v[556:559]*/, v128 offset:23104
	s_set_vgpr_msb 0x8040
	ds_load_tr16_b128 v[236:239] /*v[492:495]*/, v128 offset:23136
	s_set_vgpr_msb 0x4080
	ds_load_tr16_b128 v[32:35] /*v[544:547]*/, v128 offset:27712
	s_set_vgpr_msb 0x8040
	ds_load_tr16_b128 v[224:227] /*v[480:483]*/, v128 offset:27744
	s_set_vgpr_msb 0x4080
	ds_load_tr16_b128 v[36:39] /*v[548:551]*/, v128 offset:32320
	s_set_vgpr_msb 0x8040
	ds_load_tr16_b128 v[228:231] /*v[484:487]*/, v128 offset:32352
	s_set_vgpr_msb 0x4080
	ds_load_tr16_b128 v[24:27] /*v[536:539]*/, v128 offset:36928
	s_set_vgpr_msb 0x8040
	ds_load_tr16_b128 v[216:219] /*v[472:475]*/, v128 offset:36960
	s_set_vgpr_msb 0x4080
	ds_load_tr16_b128 v[28:31] /*v[540:543]*/, v128 offset:41536
	s_set_vgpr_msb 0x8040
	ds_load_tr16_b128 v[220:223] /*v[476:479]*/, v128 offset:41568
	s_set_vgpr_msb 0x4080
	ds_load_tr16_b128 v[16:19] /*v[528:531]*/, v128 offset:46144
	s_set_vgpr_msb 0x8040
	ds_load_tr16_b128 v[208:211] /*v[464:467]*/, v128 offset:46176
	s_set_vgpr_msb 0x4080
	ds_load_tr16_b128 v[20:23] /*v[532:535]*/, v128 offset:50752
	s_set_vgpr_msb 0x8040
	ds_load_tr16_b128 v[212:215] /*v[468:471]*/, v128 offset:50784
	s_set_vgpr_msb 0x4080
	ds_load_tr16_b128 v[8:11] /*v[520:523]*/, v128 offset:55360
	s_set_vgpr_msb 0x8040
	ds_load_tr16_b128 v[192:195] /*v[448:451]*/, v128 offset:55392
	ds_load_tr16_b128 v[200:203] /*v[456:459]*/, v128 offset:128
	ds_load_tr16_b128 v[120:123] /*v[376:379]*/, v128 offset:160
	ds_load_tr16_b128 v[204:207] /*v[460:463]*/, v128 offset:4736
	ds_load_tr16_b128 v[124:127] /*v[380:383]*/, v128 offset:4768
	ds_load_tr16_b128 v[184:187] /*v[440:443]*/, v128 offset:9344
	ds_load_tr16_b128 v[112:115] /*v[368:371]*/, v128 offset:9376
	ds_load_tr16_b128 v[188:191] /*v[444:447]*/, v128 offset:13952
	ds_load_tr16_b128 v[116:119] /*v[372:375]*/, v128 offset:13984
	ds_load_tr16_b128 v[168:171] /*v[424:427]*/, v128 offset:18560
	ds_load_tr16_b128 v[104:107] /*v[360:363]*/, v128 offset:18592
	ds_load_tr16_b128 v[172:175] /*v[428:431]*/, v128 offset:23168
	ds_load_tr16_b128 v[108:111] /*v[364:367]*/, v128 offset:23200
	ds_load_tr16_b128 v[160:163] /*v[416:419]*/, v128 offset:27776
	ds_load_tr16_b128 v[96:99] /*v[352:355]*/, v128 offset:27808
	ds_load_tr16_b128 v[164:167] /*v[420:423]*/, v128 offset:32384
	ds_load_tr16_b128 v[100:103] /*v[356:359]*/, v128 offset:32416
	ds_load_tr16_b128 v[152:155] /*v[408:411]*/, v128 offset:36992
	ds_load_tr16_b128 v[88:91] /*v[344:347]*/, v128 offset:37024
	ds_load_tr16_b128 v[156:159] /*v[412:415]*/, v128 offset:41600
	ds_load_tr16_b128 v[92:95] /*v[348:351]*/, v128 offset:41632
	ds_load_tr16_b128 v[144:147] /*v[400:403]*/, v128 offset:46208
	ds_load_tr16_b128 v[80:83] /*v[336:339]*/, v128 offset:46240
	ds_load_tr16_b128 v[148:151] /*v[404:407]*/, v128 offset:50816
	ds_load_tr16_b128 v[84:87] /*v[340:343]*/, v128 offset:50848
	ds_load_tr16_b128 v[136:139] /*v[392:395]*/, v128 offset:55424
	ds_load_tr16_b128 v[64:67] /*v[320:323]*/, v128 offset:55456
	s_wait_alu depctr_vm_vsrc(6)
	s_set_vgpr_msb 0x4000
	v_add_nc_u32_e32 v130, 0x10ec0, v128
	s_set_vgpr_msb 0x80
	ds_load_tr16_b128 v[184:187] /*v[696:699]*/, v128
	ds_load_tr16_b128 v[104:107] /*v[616:619]*/, v128 offset:32
	s_wait_dscnt 0x3e
	s_clause 0x1
	scratch_store_b128 off, v[194:197], off offset:524 nv
	scratch_store_b128 off, v[198:201], off offset:540 nv
	s_set_vgpr_msb 0x8040
	ds_load_tr16_b128 v[72:75] /*v[328:331]*/, v128 offset:192
	s_wait_alu depctr_vm_vsrc(0)
	s_set_vgpr_msb 0x4000
	ds_load_tr16_b128 v[192:195], v128 offset:224
	s_set_vgpr_msb 64
	ds_load_tr16_b128 v[76:79] /*v[332:335]*/, v128 offset:4800
	s_set_vgpr_msb 0x4000
	ds_load_tr16_b128 v[196:199], v128 offset:4832
	s_wait_dscnt 0x2
	scratch_store_b128 off, v[192:195], off offset:300 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[196:199], off offset:316 nv
	s_set_vgpr_msb 64
	ds_load_tr16_b128 v[56:59] /*v[312:315]*/, v128 offset:9408
	s_wait_alu depctr_vm_vsrc(0)
	s_set_vgpr_msb 0x4000
	ds_load_tr16_b128 v[192:195], v128 offset:9440
	s_set_vgpr_msb 64
	ds_load_tr16_b128 v[60:63] /*v[316:319]*/, v128 offset:14016
	s_set_vgpr_msb 0x4030
	ds_load_tr16_b128 v[196:199], v128 offset:14048
	s_wait_dscnt 0x2
	scratch_store_b128 off, v[192:195], off offset:268 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[196:199], off offset:284 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_tr16_b128 v[196:199], v128 offset:18624
	ds_load_tr16_b128 v[192:195], v128 offset:18656
	ds_load_tr16_b128 v[200:203], v128 offset:23232
	s_wait_dscnt 0x2
	scratch_store_b128 off, v[196:199], off offset:492 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[200:203], off offset:508 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_tr16_b128 v[196:199], v128 offset:23264
	scratch_store_b128 off, v[192:195], off offset:236 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[196:199], off offset:252 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_tr16_b128 v[196:199], v128 offset:27840
	ds_load_tr16_b128 v[192:195], v128 offset:27872
	ds_load_tr16_b128 v[200:203], v128 offset:32448
	s_wait_dscnt 0x2
	scratch_store_b128 off, v[196:199], off offset:460 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[200:203], off offset:476 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_tr16_b128 v[196:199], v128 offset:32480
	scratch_store_b128 off, v[192:195], off offset:204 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[196:199], off offset:220 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_tr16_b128 v[196:199], v128 offset:37056
	ds_load_tr16_b128 v[192:195], v128 offset:37088
	ds_load_tr16_b128 v[200:203], v128 offset:41664
	s_wait_dscnt 0x2
	scratch_store_b128 off, v[196:199], off offset:428 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[200:203], off offset:444 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_tr16_b128 v[196:199], v128 offset:41696
	scratch_store_b128 off, v[192:195], off offset:172 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[196:199], off offset:188 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_tr16_b128 v[196:199], v128 offset:46272
	ds_load_tr16_b128 v[192:195], v128 offset:46304
	ds_load_tr16_b128 v[200:203], v128 offset:50880
	s_wait_dscnt 0x2
	scratch_store_b128 off, v[196:199], off offset:396 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[200:203], off offset:412 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_tr16_b128 v[196:199], v128 offset:50912
	ds_load_tr16_b128 v[202:205], v128 offset:60096
	scratch_store_b128 off, v[192:195], off offset:140 nv
	s_wait_dscnt 0x1
	scratch_store_b128 off, v[196:199], off offset:156 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_tr16_b128 v[198:201], v128 offset:55488
	ds_load_tr16_b128 v[194:197], v128 offset:55520
	v_add_nc_u32_e32 v192, 0x10ee0, v128
	s_wait_dscnt 0x1
	s_clause 0x1
	scratch_store_b128 off, v[198:201], off offset:364 nv
	scratch_store_b128 off, v[202:205], off offset:380 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_tr16_b128 v[198:201], v128 offset:60128
	s_wait_dscnt 0x1
	scratch_store_b128 off, v[194:197], off offset:108 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[198:201], off offset:124 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_tr16_b128 v[198:201], v128 offset:64704
	ds_load_tr16_b128 v[194:197], v128 offset:64736
	s_wait_alu depctr_va_vdst(1)
	ds_load_tr16_b128 v[202:205], v130
	s_wait_dscnt 0x2
	scratch_store_b128 off, v[198:201], off offset:332 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[202:205], off offset:348 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_va_vdst(0) depctr_vm_vsrc(0)
	ds_load_tr16_b128 v[198:201], v192
	scratch_store_b128 off, v[194:197], off offset:76 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[198:201], off offset:92 nv
	v_lshl_or_b32 v130, s44, 8, v200 /*v968*/
	v_cmp_le_i32_e32 vcc_lo, v130, v133
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	v_dual_add_nc_u32 v199, 17, v130 :: v_dual_bitop2_b32 v192, 2, v130 bitop3:0x54
	v_dual_add_nc_u32 v200, 18, v130 :: v_dual_bitop2_b32 v193, 3, v130 bitop3:0x54
	s_set_vgpr_msb 0x30cc
	v_cndmask_b32_e32 v178 /*v946*/, 0xff800000, v178 /*v946*/, vcc_lo
	s_set_vgpr_msb 0xcc00
	v_cmp_lt_i32_e32 vcc_lo, v130, v133
	v_dual_add_nc_u32 v201, 19, v130 :: v_dual_bitop2_b32 v194, 4, v130 bitop3:0x54
	v_dual_add_nc_u32 v202, 20, v130 :: v_dual_bitop2_b32 v195, 5, v130 bitop3:0x54
	s_set_vgpr_msb 0xcc
	v_cndmask_b32_e32 v179 /*v947*/, 0xff800000, v179 /*v947*/, vcc_lo
	s_set_vgpr_msb 0xcc00
	v_cmp_le_i32_e32 vcc_lo, v192, v133
	v_dual_add_nc_u32 v203, 21, v130 :: v_dual_bitop2_b32 v196, 6, v130 bitop3:0x54
	v_dual_add_nc_u32 v204, 22, v130 :: v_dual_bitop2_b32 v197, 7, v130 bitop3:0x54
	s_set_vgpr_msb 0xcc
	v_cndmask_b32_e32 v180 /*v948*/, 0xff800000, v180 /*v948*/, vcc_lo
	s_set_vgpr_msb 0xcc00
	v_cmp_le_i32_e32 vcc_lo, v193, v133
	v_dual_add_nc_u32 v205, 23, v130 :: v_dual_bitop2_b32 v198, 16, v130 bitop3:0x54
	v_dual_add_nc_u32 v215, 49, v130 :: v_dual_bitop2_b32 v206, 32, v130 bitop3:0x54
	s_set_vgpr_msb 0xcc
	v_cndmask_b32_e32 v181 /*v949*/, 0xff800000, v181 /*v949*/, vcc_lo
	s_set_vgpr_msb 0xcc00
	v_cmp_le_i32_e32 vcc_lo, v194, v133
	v_dual_add_nc_u32 v216, 50, v130 :: v_dual_bitop2_b32 v207, 33, v130 bitop3:0x54
	v_dual_add_nc_u32 v217, 51, v130 :: v_dual_bitop2_b32 v208, 34, v130 bitop3:0x54
	s_set_vgpr_msb 0xcc
	v_cndmask_b32_e32 v182 /*v950*/, 0xff800000, v182 /*v950*/, vcc_lo
	s_set_vgpr_msb 0xcc00
	v_cmp_le_i32_e32 vcc_lo, v195, v133
	v_dual_add_nc_u32 v218, 52, v130 :: v_dual_bitop2_b32 v209, 35, v130 bitop3:0x54
	v_dual_add_nc_u32 v219, 53, v130 :: v_dual_bitop2_b32 v210, 36, v130 bitop3:0x54
	s_set_vgpr_msb 0xcc
	v_cndmask_b32_e32 v183 /*v951*/, 0xff800000, v183 /*v951*/, vcc_lo
	s_set_vgpr_msb 0xcc00
	v_cmp_le_i32_e32 vcc_lo, v196, v133
	v_dual_add_nc_u32 v220, 54, v130 :: v_dual_bitop2_b32 v211, 37, v130 bitop3:0x54
	v_dual_add_nc_u32 v221, 55, v130 :: v_dual_bitop2_b32 v212, 38, v130 bitop3:0x54
	s_set_vgpr_msb 0xcc
	v_cndmask_b32_e32 v184 /*v952*/, 0xff800000, v184 /*v952*/, vcc_lo
	s_set_vgpr_msb 0xcc00
	v_cmp_le_i32_e32 vcc_lo, v197, v133
	v_or_b32_e32 v213, 39, v130
	v_or_b32_e32 v214, 48, v130
	v_or_b32_e32 v222, 64, v130
	v_or_b32_e32 v223, 0x41, v130
	s_set_vgpr_msb 0xcc
	v_cndmask_b32_e32 v185 /*v953*/, 0xff800000, v185 /*v953*/, vcc_lo
	s_set_vgpr_msb 0xcc00
	v_cmp_le_i32_e32 vcc_lo, v198, v133
	v_or_b32_e32 v224, 0x42, v130
	v_or_b32_e32 v225, 0x43, v130
	v_or_b32_e32 v226, 0x44, v130
	v_or_b32_e32 v227, 0x45, v130
	s_set_vgpr_msb 0xcc
	v_cndmask_b32_e32 v186 /*v954*/, 0xff800000, v186 /*v954*/, vcc_lo
	s_set_vgpr_msb 0xcc00
	v_cmp_le_i32_e32 vcc_lo, v199, v133
	v_or_b32_e32 v228, 0x46, v130
	v_or_b32_e32 v229, 0x47, v130
	v_or_b32_e32 v230, 0x50, v130
	v_add_nc_u32_e32 v231, 0x51, v130
	s_set_vgpr_msb 0xcc
	v_cndmask_b32_e32 v187 /*v955*/, 0xff800000, v187 /*v955*/, vcc_lo
	s_set_vgpr_msb 0xcc00
	v_cmp_le_i32_e32 vcc_lo, v200, v133
	v_add_nc_u32_e32 v232, 0x52, v130
	v_add_nc_u32_e32 v233, 0x53, v130
	v_add_nc_u32_e32 v234, 0x54, v130
	v_add_nc_u32_e32 v235, 0x55, v130
	s_set_vgpr_msb 0xcc
	v_cndmask_b32_e32 v188 /*v956*/, 0xff800000, v188 /*v956*/, vcc_lo
	s_set_vgpr_msb 0xcc00
	v_cmp_le_i32_e32 vcc_lo, v201, v133
	v_add_nc_u32_e32 v236, 0x56, v130
	v_add_nc_u32_e32 v237, 0x57, v130
	v_or_b32_e32 v238, 0x60, v130
	v_or_b32_e32 v239, 0x61, v130
	s_set_vgpr_msb 0xcc
	v_cndmask_b32_e32 v189 /*v957*/, 0xff800000, v189 /*v957*/, vcc_lo
	s_set_vgpr_msb 0xcc00
	v_cmp_le_i32_e32 vcc_lo, v202, v133
	v_or_b32_e32 v240, 0x62, v130
	v_or_b32_e32 v241, 0x63, v130
	v_or_b32_e32 v242, 0x64, v130
	v_or_b32_e32 v243, 0x65, v130
	s_set_vgpr_msb 0xcc
	v_cndmask_b32_e32 v190 /*v958*/, 0xff800000, v190 /*v958*/, vcc_lo
	s_set_vgpr_msb 0xcc00
	v_cmp_le_i32_e32 vcc_lo, v203, v133
	v_or_b32_e32 v244, 0x66, v130
	v_or_b32_e32 v245, 0x67, v130
	v_or_b32_e32 v246, 0x70, v130
	v_add_nc_u32_e32 v247, 0x71, v130
	s_set_vgpr_msb 0xcc
	v_cndmask_b32_e32 v191 /*v959*/, 0xff800000, v191 /*v959*/, vcc_lo
	s_set_vgpr_msb 0xcc00
	v_cmp_le_i32_e32 vcc_lo, v204, v133
	v_add_nc_u32_e32 v248, 0x72, v130
	v_add_nc_u32_e32 v249, 0x73, v130
	v_add_nc_u32_e32 v250, 0x74, v130
	v_add_nc_u32_e32 v251, 0x75, v130
	s_set_vgpr_msb 0xcc
	v_cndmask_b32_e32 v192 /*v960*/, 0xff800000, v192 /*v960*/, vcc_lo
	s_set_vgpr_msb 0xcc00
	v_cmp_le_i32_e32 vcc_lo, v205, v133
	v_add_nc_u32_e32 v252, 0x76, v130
	v_add_nc_u32_e32 v253, 0x77, v130
	v_or_b32_e32 v254, 0x80, v130
	v_or_b32_e32 v255, 0x81, v130
	s_set_vgpr_msb 0xcc
	v_cndmask_b32_e32 v193 /*v961*/, 0xff800000, v193 /*v961*/, vcc_lo
	s_set_vgpr_msb 0xcc40
	v_cmp_le_i32_e32 vcc_lo, v206, v133
	v_or_b32_e32 v0 /*v256*/, 0x82, v130
	v_or_b32_e32 v1 /*v257*/, 0x83, v130
	v_or_b32_e32 v2 /*v258*/, 0x84, v130
	v_or_b32_e32 v3 /*v259*/, 0x85, v130
	s_set_vgpr_msb 0x40cc
	v_cndmask_b32_e32 v168 /*v936*/, 0xff800000, v168 /*v936*/, vcc_lo
	s_set_vgpr_msb 0xcc40
	v_cmp_le_i32_e32 vcc_lo, v207, v133
	v_or_b32_e32 v4 /*v260*/, 0x86, v130
	v_or_b32_e32 v5 /*v261*/, 0x87, v130
	v_or_b32_e32 v6 /*v262*/, 0x90, v130
	v_add_nc_u32_e32 v7 /*v263*/, 0x91, v130
	s_set_vgpr_msb 0x40cc
	v_cndmask_b32_e32 v169 /*v937*/, 0xff800000, v169 /*v937*/, vcc_lo
	s_set_vgpr_msb 0xcc40
	v_cmp_le_i32_e32 vcc_lo, v208, v133
	v_add_nc_u32_e32 v8 /*v264*/, 0x92, v130
	v_add_nc_u32_e32 v9 /*v265*/, 0x93, v130
	v_add_nc_u32_e32 v10 /*v266*/, 0x94, v130
	v_add_nc_u32_e32 v11 /*v267*/, 0x95, v130
	s_set_vgpr_msb 0x40cc
	v_cndmask_b32_e32 v170 /*v938*/, 0xff800000, v170 /*v938*/, vcc_lo
	s_set_vgpr_msb 0xcc40
	v_cmp_le_i32_e32 vcc_lo, v209, v133
	v_add_nc_u32_e32 v12 /*v268*/, 0x96, v130
	v_add_nc_u32_e32 v13 /*v269*/, 0x97, v130
	v_or_b32_e32 v14 /*v270*/, 0xa0, v130
	v_or_b32_e32 v15 /*v271*/, 0xa1, v130
	s_set_vgpr_msb 0x40cc
	v_cndmask_b32_e32 v171 /*v939*/, 0xff800000, v171 /*v939*/, vcc_lo
	s_set_vgpr_msb 0xcc40
	v_cmp_le_i32_e32 vcc_lo, v210, v133
	v_or_b32_e32 v16 /*v272*/, 0xa2, v130
	v_or_b32_e32 v17 /*v273*/, 0xa3, v130
	v_or_b32_e32 v18 /*v274*/, 0xa4, v130
	v_or_b32_e32 v19 /*v275*/, 0xa5, v130
	s_set_vgpr_msb 0x40cc
	v_cndmask_b32_e32 v172 /*v940*/, 0xff800000, v172 /*v940*/, vcc_lo
	s_set_vgpr_msb 0xcc40
	v_cmp_le_i32_e32 vcc_lo, v211, v133
	v_or_b32_e32 v20 /*v276*/, 0xa6, v130
	v_or_b32_e32 v21 /*v277*/, 0xa7, v130
	v_or_b32_e32 v22 /*v278*/, 0xb0, v130
	v_add_nc_u32_e32 v23 /*v279*/, 0xb1, v130
	s_set_vgpr_msb 0x40cc
	v_cndmask_b32_e32 v173 /*v941*/, 0xff800000, v173 /*v941*/, vcc_lo
	s_set_vgpr_msb 0xcc40
	v_cmp_le_i32_e32 vcc_lo, v212, v133
	v_add_nc_u32_e32 v24 /*v280*/, 0xb2, v130
	v_add_nc_u32_e32 v25 /*v281*/, 0xb3, v130
	v_add_nc_u32_e32 v26 /*v282*/, 0xb4, v130
	v_add_nc_u32_e32 v27 /*v283*/, 0xb5, v130
	s_set_vgpr_msb 0x40cc
	v_cndmask_b32_e32 v174 /*v942*/, 0xff800000, v174 /*v942*/, vcc_lo
	s_set_vgpr_msb 0xcc40
	v_cmp_le_i32_e32 vcc_lo, v213, v133
	v_add_nc_u32_e32 v28 /*v284*/, 0xb6, v130
	v_add_nc_u32_e32 v29 /*v285*/, 0xb7, v130
	v_or_b32_e32 v30 /*v286*/, 0xc0, v130
	v_or_b32_e32 v31 /*v287*/, 0xc1, v130
	s_set_vgpr_msb 0x40cc
	v_cndmask_b32_e32 v175 /*v943*/, 0xff800000, v175 /*v943*/, vcc_lo
	s_set_vgpr_msb 0xcc40
	v_cmp_le_i32_e32 vcc_lo, v214, v133
	v_or_b32_e32 v32 /*v288*/, 0xc2, v130
	v_or_b32_e32 v33 /*v289*/, 0xc3, v130
	v_or_b32_e32 v34 /*v290*/, 0xc4, v130
	v_or_b32_e32 v35 /*v291*/, 0xc5, v130
	s_set_vgpr_msb 0x40cc
	v_cndmask_b32_e32 v160 /*v928*/, 0xff800000, v160 /*v928*/, vcc_lo
	s_set_vgpr_msb 0xcc40
	v_cmp_le_i32_e32 vcc_lo, v215, v133
	v_or_b32_e32 v36 /*v292*/, 0xc6, v130
	v_or_b32_e32 v37 /*v293*/, 0xc7, v130
	v_or_b32_e32 v38 /*v294*/, 0xd0, v130
	v_add_nc_u32_e32 v39 /*v295*/, 0xd1, v130
	s_set_vgpr_msb 0x40cc
	v_cndmask_b32_e32 v161 /*v929*/, 0xff800000, v161 /*v929*/, vcc_lo
	s_set_vgpr_msb 0xcc40
	v_cmp_le_i32_e32 vcc_lo, v216, v133
	v_add_nc_u32_e32 v40 /*v296*/, 0xd2, v130
	v_add_nc_u32_e32 v41 /*v297*/, 0xd3, v130
	v_add_nc_u32_e32 v42 /*v298*/, 0xd4, v130
	v_add_nc_u32_e32 v43 /*v299*/, 0xd5, v130
	s_set_vgpr_msb 0x40cc
	v_cndmask_b32_e32 v162 /*v930*/, 0xff800000, v162 /*v930*/, vcc_lo
	s_set_vgpr_msb 0xcc40
	v_cmp_le_i32_e32 vcc_lo, v217, v133
	v_add_nc_u32_e32 v44 /*v300*/, 0xd6, v130
	v_add_nc_u32_e32 v45 /*v301*/, 0xd7, v130
	v_or_b32_e32 v46 /*v302*/, 0xe0, v130
	v_or_b32_e32 v47 /*v303*/, 0xe1, v130
	s_set_vgpr_msb 0x40cc
	v_cndmask_b32_e32 v163 /*v931*/, 0xff800000, v163 /*v931*/, vcc_lo
	s_set_vgpr_msb 0xcc40
	v_cmp_le_i32_e32 vcc_lo, v218, v133
	v_or_b32_e32 v48 /*v304*/, 0xe2, v130
	v_or_b32_e32 v49 /*v305*/, 0xe3, v130
	v_or_b32_e32 v50 /*v306*/, 0xe4, v130
	v_or_b32_e32 v51 /*v307*/, 0xe5, v130
	s_set_vgpr_msb 0x40cc
	v_cndmask_b32_e32 v164 /*v932*/, 0xff800000, v164 /*v932*/, vcc_lo
	s_set_vgpr_msb 0xcc40
	v_cmp_le_i32_e32 vcc_lo, v219, v133
	v_or_b32_e32 v52 /*v308*/, 0xe6, v130
	v_or_b32_e32 v53 /*v309*/, 0xe7, v130
	v_or_b32_e32 v54 /*v310*/, 0xf0, v130
	v_add_nc_u32_e32 v55 /*v311*/, 0xf1, v130
	s_set_vgpr_msb 0x40cc
	v_cndmask_b32_e32 v165 /*v933*/, 0xff800000, v165 /*v933*/, vcc_lo
	s_set_vgpr_msb 0xcc00
	v_cmp_le_i32_e32 vcc_lo, v220, v133
	s_set_vgpr_msb 0xcc
	v_cndmask_b32_e32 v166 /*v934*/, 0xff800000, v166 /*v934*/, vcc_lo
	s_set_vgpr_msb 0xcc00
	v_cmp_le_i32_e32 vcc_lo, v221, v133
	s_set_vgpr_msb 0xcc
	v_cndmask_b32_e32 v167 /*v935*/, 0xff800000, v167 /*v935*/, vcc_lo
	s_set_vgpr_msb 0xcc00
	v_cmp_le_i32_e32 vcc_lo, v222, v133
	s_set_vgpr_msb 0xcc
	v_cndmask_b32_e32 v152 /*v920*/, 0xff800000, v152 /*v920*/, vcc_lo
	s_set_vgpr_msb 0xcc00
	v_cmp_le_i32_e32 vcc_lo, v223, v133
	s_set_vgpr_msb 0xcc
	v_cndmask_b32_e32 v153 /*v921*/, 0xff800000, v153 /*v921*/, vcc_lo
	s_set_vgpr_msb 0xcc00
	v_cmp_le_i32_e32 vcc_lo, v224, v133
	s_set_vgpr_msb 0xcc
	v_cndmask_b32_e32 v154 /*v922*/, 0xff800000, v154 /*v922*/, vcc_lo
	s_set_vgpr_msb 0xcc00
	v_cmp_le_i32_e32 vcc_lo, v225, v133
	s_set_vgpr_msb 0xcc
	v_cndmask_b32_e32 v155 /*v923*/, 0xff800000, v155 /*v923*/, vcc_lo
	s_set_vgpr_msb 0xcc00
	v_cmp_le_i32_e32 vcc_lo, v226, v133
	s_set_vgpr_msb 0xcc
	v_cndmask_b32_e32 v156 /*v924*/, 0xff800000, v156 /*v924*/, vcc_lo
	s_set_vgpr_msb 0xcc00
	v_cmp_le_i32_e32 vcc_lo, v227, v133
	s_set_vgpr_msb 0xcc
	v_cndmask_b32_e32 v157 /*v925*/, 0xff800000, v157 /*v925*/, vcc_lo
	s_set_vgpr_msb 0xcc00
	v_cmp_le_i32_e32 vcc_lo, v228, v133
	s_set_vgpr_msb 0xcc
	v_cndmask_b32_e32 v158 /*v926*/, 0xff800000, v158 /*v926*/, vcc_lo
	s_set_vgpr_msb 0xcc00
	v_cmp_le_i32_e32 vcc_lo, v229, v133
	s_set_vgpr_msb 0xcc
	v_cndmask_b32_e32 v159 /*v927*/, 0xff800000, v159 /*v927*/, vcc_lo
	s_set_vgpr_msb 0xcc00
	v_cmp_le_i32_e32 vcc_lo, v230, v133
	s_set_vgpr_msb 0xcc
	v_cndmask_b32_e32 v144 /*v912*/, 0xff800000, v144 /*v912*/, vcc_lo
	s_set_vgpr_msb 0xcc00
	v_cmp_le_i32_e32 vcc_lo, v231, v133
	s_set_vgpr_msb 0xcc
	v_cndmask_b32_e32 v145 /*v913*/, 0xff800000, v145 /*v913*/, vcc_lo
	s_set_vgpr_msb 0xcc00
	v_cmp_le_i32_e32 vcc_lo, v232, v133
	s_set_vgpr_msb 0xcc
	v_cndmask_b32_e32 v146 /*v914*/, 0xff800000, v146 /*v914*/, vcc_lo
	s_set_vgpr_msb 0xcc00
	v_cmp_le_i32_e32 vcc_lo, v233, v133
	s_set_vgpr_msb 0xcc
	v_cndmask_b32_e32 v147 /*v915*/, 0xff800000, v147 /*v915*/, vcc_lo
	s_set_vgpr_msb 0xcc00
	v_cmp_le_i32_e32 vcc_lo, v234, v133
	s_set_vgpr_msb 0xcc
	v_cndmask_b32_e32 v148 /*v916*/, 0xff800000, v148 /*v916*/, vcc_lo
	s_set_vgpr_msb 0xcc00
	v_cmp_le_i32_e32 vcc_lo, v235, v133
	s_set_vgpr_msb 0xcc
	v_cndmask_b32_e32 v149 /*v917*/, 0xff800000, v149 /*v917*/, vcc_lo
	s_set_vgpr_msb 0xcc00
	v_cmp_le_i32_e32 vcc_lo, v236, v133
	s_set_vgpr_msb 0xcc
	v_cndmask_b32_e32 v150 /*v918*/, 0xff800000, v150 /*v918*/, vcc_lo
	s_set_vgpr_msb 0xcc00
	v_cmp_le_i32_e32 vcc_lo, v237, v133
	s_set_vgpr_msb 0xcc
	v_cndmask_b32_e32 v151 /*v919*/, 0xff800000, v151 /*v919*/, vcc_lo
	s_set_vgpr_msb 0xcc00
	v_cmp_le_i32_e32 vcc_lo, v238, v133
	s_set_vgpr_msb 0xcc
	v_cndmask_b32_e32 v136 /*v904*/, 0xff800000, v136 /*v904*/, vcc_lo
	s_set_vgpr_msb 0xcc00
	v_cmp_le_i32_e32 vcc_lo, v239, v133
	s_set_vgpr_msb 0xcc
	v_cndmask_b32_e32 v137 /*v905*/, 0xff800000, v137 /*v905*/, vcc_lo
	s_set_vgpr_msb 0xcc00
	v_cmp_le_i32_e32 vcc_lo, v240, v133
	s_set_vgpr_msb 0xcc
	v_cndmask_b32_e32 v138 /*v906*/, 0xff800000, v138 /*v906*/, vcc_lo
	s_set_vgpr_msb 0xcc00
	v_cmp_le_i32_e32 vcc_lo, v241, v133
	s_set_vgpr_msb 0xcc
	v_cndmask_b32_e32 v139 /*v907*/, 0xff800000, v139 /*v907*/, vcc_lo
	s_set_vgpr_msb 0xcc00
	v_cmp_le_i32_e32 vcc_lo, v242, v133
	s_set_vgpr_msb 0xcc
	v_cndmask_b32_e32 v140 /*v908*/, 0xff800000, v140 /*v908*/, vcc_lo
	s_set_vgpr_msb 0xcc00
	v_cmp_le_i32_e32 vcc_lo, v243, v133
	s_set_vgpr_msb 0xcc
	v_cndmask_b32_e32 v141 /*v909*/, 0xff800000, v141 /*v909*/, vcc_lo
	s_set_vgpr_msb 0xcc00
	v_cmp_le_i32_e32 vcc_lo, v244, v133
	s_set_vgpr_msb 0xcc
	v_cndmask_b32_e32 v142 /*v910*/, 0xff800000, v142 /*v910*/, vcc_lo
	s_set_vgpr_msb 0xcc00
	v_cmp_le_i32_e32 vcc_lo, v245, v133
	s_set_vgpr_msb 0xcc
	v_cndmask_b32_e32 v143 /*v911*/, 0xff800000, v143 /*v911*/, vcc_lo
	s_set_vgpr_msb 0xcc00
	v_cmp_le_i32_e32 vcc_lo, v246, v133
	s_set_vgpr_msb 0xcc
	v_cndmask_b32_e32 v128 /*v896*/, 0xff800000, v128 /*v896*/, vcc_lo
	s_set_vgpr_msb 0xcc00
	v_cmp_le_i32_e32 vcc_lo, v247, v133
	s_set_vgpr_msb 0xcc
	v_cndmask_b32_e32 v129 /*v897*/, 0xff800000, v129 /*v897*/, vcc_lo
	s_set_vgpr_msb 0xcc00
	v_cmp_le_i32_e32 vcc_lo, v248, v133
	s_set_vgpr_msb 0xcc
	v_cndmask_b32_e32 v130 /*v898*/, 0xff800000, v130 /*v898*/, vcc_lo
	s_set_vgpr_msb 0xcc00
	v_cmp_le_i32_e32 vcc_lo, v249, v133
	s_set_vgpr_msb 0xcc
	v_cndmask_b32_e32 v131 /*v899*/, 0xff800000, v131 /*v899*/, vcc_lo
	s_set_vgpr_msb 0xcc00
	v_cmp_le_i32_e32 vcc_lo, v250, v133
	s_set_vgpr_msb 0xcc
	v_cndmask_b32_e32 v132 /*v900*/, 0xff800000, v132 /*v900*/, vcc_lo
	s_set_vgpr_msb 0xcc00
	v_cmp_le_i32_e32 vcc_lo, v251, v133
	s_set_vgpr_msb 0xcc
	v_cndmask_b32_e32 v133 /*v901*/, 0xff800000, v133 /*v901*/, vcc_lo
	s_set_vgpr_msb 0xcc00
	v_cmp_le_i32_e32 vcc_lo, v252, v133
	s_set_vgpr_msb 0xcc
	v_cndmask_b32_e32 v134 /*v902*/, 0xff800000, v134 /*v902*/, vcc_lo
	s_set_vgpr_msb 0xcc00
	v_cmp_le_i32_e32 vcc_lo, v253, v133
	s_set_vgpr_msb 0xcc
	v_cndmask_b32_e32 v135 /*v903*/, 0xff800000, v135 /*v903*/, vcc_lo
	s_set_vgpr_msb 0xcc00
	v_cmp_le_i32_e32 vcc_lo, v254, v133
	s_set_vgpr_msb 0xcc
	v_cndmask_b32_e32 v120 /*v888*/, 0xff800000, v120 /*v888*/, vcc_lo
	s_set_vgpr_msb 0xcc00
	v_cmp_le_i32_e32 vcc_lo, v255, v133
	s_set_vgpr_msb 0xcc
	v_cndmask_b32_e32 v121 /*v889*/, 0xff800000, v121 /*v889*/, vcc_lo
	s_set_vgpr_msb 0xcc01
	v_cmp_le_i32_e32 vcc_lo, v0 /*v256*/, v133
	s_set_vgpr_msb 0x1cc
	v_cndmask_b32_e32 v122 /*v890*/, 0xff800000, v122 /*v890*/, vcc_lo
	s_set_vgpr_msb 0xcc01
	v_cmp_le_i32_e32 vcc_lo, v1 /*v257*/, v133
	s_set_vgpr_msb 0x1cc
	v_cndmask_b32_e32 v123 /*v891*/, 0xff800000, v123 /*v891*/, vcc_lo
	s_set_vgpr_msb 0xcc01
	v_cmp_le_i32_e32 vcc_lo, v2 /*v258*/, v133
	s_set_vgpr_msb 0x1cc
	v_cndmask_b32_e32 v124 /*v892*/, 0xff800000, v124 /*v892*/, vcc_lo
	s_set_vgpr_msb 0xcc01
	v_cmp_le_i32_e32 vcc_lo, v3 /*v259*/, v133
	s_set_vgpr_msb 0x1cc
	v_cndmask_b32_e32 v125 /*v893*/, 0xff800000, v125 /*v893*/, vcc_lo
	s_set_vgpr_msb 0xcc01
	v_cmp_le_i32_e32 vcc_lo, v4 /*v260*/, v133
	s_set_vgpr_msb 0x1cc
	v_cndmask_b32_e32 v176 /*v944*/, 0xff800000, v126 /*v894*/, vcc_lo
	s_set_vgpr_msb 0xcc01
	v_cmp_le_i32_e32 vcc_lo, v5 /*v261*/, v133
	s_set_vgpr_msb 0x1cc
	v_cndmask_b32_e32 v177 /*v945*/, 0xff800000, v127 /*v895*/, vcc_lo
	s_set_vgpr_msb 0xcc01
	v_cmp_le_i32_e32 vcc_lo, v6 /*v262*/, v133
	s_set_vgpr_msb 0x1cc
	v_cndmask_b32_e32 v194 /*v962*/, 0xff800000, v112 /*v880*/, vcc_lo
	s_set_vgpr_msb 0xcc01
	v_cmp_le_i32_e32 vcc_lo, v7 /*v263*/, v133
	s_set_vgpr_msb 0x1cc
	v_cndmask_b32_e32 v195 /*v963*/, 0xff800000, v113 /*v881*/, vcc_lo
	s_set_vgpr_msb 0xcc01
	v_cmp_le_i32_e32 vcc_lo, v8 /*v264*/, v133
	s_set_vgpr_msb 0x1cc
	v_cndmask_b32_e32 v114 /*v882*/, 0xff800000, v114 /*v882*/, vcc_lo
	s_set_vgpr_msb 0xcc01
	v_cmp_le_i32_e32 vcc_lo, v9 /*v265*/, v133
	s_set_vgpr_msb 0x1cc
	v_cndmask_b32_e32 v115 /*v883*/, 0xff800000, v115 /*v883*/, vcc_lo
	s_set_vgpr_msb 0xcc01
	v_cmp_le_i32_e32 vcc_lo, v10 /*v266*/, v133
	s_set_vgpr_msb 0x1cc
	v_cndmask_b32_e32 v116 /*v884*/, 0xff800000, v116 /*v884*/, vcc_lo
	s_set_vgpr_msb 0xcc01
	v_cmp_le_i32_e32 vcc_lo, v11 /*v267*/, v133
	s_set_vgpr_msb 0x1cc
	v_cndmask_b32_e32 v117 /*v885*/, 0xff800000, v117 /*v885*/, vcc_lo
	s_set_vgpr_msb 0xcc01
	v_cmp_le_i32_e32 vcc_lo, v12 /*v268*/, v133
	s_set_vgpr_msb 0x1cc
	v_cndmask_b32_e32 v118 /*v886*/, 0xff800000, v118 /*v886*/, vcc_lo
	s_set_vgpr_msb 0xcc01
	v_cmp_le_i32_e32 vcc_lo, v13 /*v269*/, v133
	s_set_vgpr_msb 0x1cc
	v_cndmask_b32_e32 v119 /*v887*/, 0xff800000, v119 /*v887*/, vcc_lo
	s_set_vgpr_msb 0xcc01
	v_cmp_le_i32_e32 vcc_lo, v14 /*v270*/, v133
	s_set_vgpr_msb 0x1cc
	v_cndmask_b32_e32 v104 /*v872*/, 0xff800000, v104 /*v872*/, vcc_lo
	s_set_vgpr_msb 0xcc01
	v_cmp_le_i32_e32 vcc_lo, v15 /*v271*/, v133
	s_set_vgpr_msb 0x1cc
	v_cndmask_b32_e32 v105 /*v873*/, 0xff800000, v105 /*v873*/, vcc_lo
	s_set_vgpr_msb 0xcc01
	v_cmp_le_i32_e32 vcc_lo, v16 /*v272*/, v133
	s_set_vgpr_msb 0x1cc
	v_cndmask_b32_e32 v196 /*v964*/, 0xff800000, v106 /*v874*/, vcc_lo
	s_set_vgpr_msb 0xcc01
	v_cmp_le_i32_e32 vcc_lo, v17 /*v273*/, v133
	s_set_vgpr_msb 0x1cc
	v_cndmask_b32_e32 v197 /*v965*/, 0xff800000, v107 /*v875*/, vcc_lo
	s_set_vgpr_msb 0xcc01
	v_cmp_le_i32_e32 vcc_lo, v18 /*v274*/, v133
	s_set_vgpr_msb 0x1cc
	v_cndmask_b32_e32 v108 /*v876*/, 0xff800000, v108 /*v876*/, vcc_lo
	s_set_vgpr_msb 0xcc01
	v_cmp_le_i32_e32 vcc_lo, v19 /*v275*/, v133
	s_set_vgpr_msb 0x1cc
	v_cndmask_b32_e32 v109 /*v877*/, 0xff800000, v109 /*v877*/, vcc_lo
	s_set_vgpr_msb 0xcc01
	v_cmp_le_i32_e32 vcc_lo, v20 /*v276*/, v133
	s_set_vgpr_msb 0x1cc
	v_cndmask_b32_e32 v110 /*v878*/, 0xff800000, v110 /*v878*/, vcc_lo
	s_set_vgpr_msb 0xcc01
	v_cmp_le_i32_e32 vcc_lo, v21 /*v277*/, v133
	s_set_vgpr_msb 0x1cc
	v_cndmask_b32_e32 v111 /*v879*/, 0xff800000, v111 /*v879*/, vcc_lo
	s_set_vgpr_msb 0xcc01
	v_cmp_le_i32_e32 vcc_lo, v22 /*v278*/, v133
	s_set_vgpr_msb 0x1cc
	v_cndmask_b32_e32 v198 /*v966*/, 0xff800000, v96 /*v864*/, vcc_lo
	s_set_vgpr_msb 0xcc01
	v_cmp_le_i32_e32 vcc_lo, v23 /*v279*/, v133
	s_set_vgpr_msb 0x1cc
	v_cndmask_b32_e32 v199 /*v967*/, 0xff800000, v97 /*v865*/, vcc_lo
	s_set_vgpr_msb 0xcc01
	v_cmp_le_i32_e32 vcc_lo, v24 /*v280*/, v133
	s_set_vgpr_msb 0x1cc
	v_cndmask_b32_e32 v200 /*v968*/, 0xff800000, v98 /*v866*/, vcc_lo
	s_set_vgpr_msb 0xcc01
	v_cmp_le_i32_e32 vcc_lo, v25 /*v281*/, v133
	s_set_vgpr_msb 0x1cc
	v_cndmask_b32_e32 v201 /*v969*/, 0xff800000, v99 /*v867*/, vcc_lo
	s_set_vgpr_msb 0xcc01
	v_cmp_le_i32_e32 vcc_lo, v26 /*v282*/, v133
	s_set_vgpr_msb 0x1cc
	v_cndmask_b32_e32 v100 /*v868*/, 0xff800000, v100 /*v868*/, vcc_lo
	s_set_vgpr_msb 0xcc01
	v_cmp_le_i32_e32 vcc_lo, v27 /*v283*/, v133
	s_set_vgpr_msb 0x1cc
	v_cndmask_b32_e32 v101 /*v869*/, 0xff800000, v101 /*v869*/, vcc_lo
	s_set_vgpr_msb 0xcc01
	v_cmp_le_i32_e32 vcc_lo, v28 /*v284*/, v133
	s_set_vgpr_msb 0x1cc
	v_cndmask_b32_e32 v202 /*v970*/, 0xff800000, v102 /*v870*/, vcc_lo
	s_set_vgpr_msb 0xcc01
	v_cmp_le_i32_e32 vcc_lo, v29 /*v285*/, v133
	s_set_vgpr_msb 0x1cc
	v_cndmask_b32_e32 v203 /*v971*/, 0xff800000, v103 /*v871*/, vcc_lo
	s_set_vgpr_msb 0xcc01
	v_cmp_le_i32_e32 vcc_lo, v30 /*v286*/, v133
	s_set_vgpr_msb 0x1cc
	v_cndmask_b32_e32 v204 /*v972*/, 0xff800000, v88 /*v856*/, vcc_lo
	s_set_vgpr_msb 0xcc01
	v_cmp_le_i32_e32 vcc_lo, v31 /*v287*/, v133
	s_set_vgpr_msb 0x1cc
	v_cndmask_b32_e32 v205 /*v973*/, 0xff800000, v89 /*v857*/, vcc_lo
	s_set_vgpr_msb 0xcc01
	v_cmp_le_i32_e32 vcc_lo, v32 /*v288*/, v133
	s_set_vgpr_msb 0x1cc
	v_cndmask_b32_e32 v206 /*v974*/, 0xff800000, v90 /*v858*/, vcc_lo
	s_set_vgpr_msb 0xcc01
	v_cmp_le_i32_e32 vcc_lo, v33 /*v289*/, v133
	s_set_vgpr_msb 0x1cc
	v_cndmask_b32_e32 v207 /*v975*/, 0xff800000, v91 /*v859*/, vcc_lo
	s_set_vgpr_msb 0xcc01
	v_cmp_le_i32_e32 vcc_lo, v34 /*v290*/, v133
	s_set_vgpr_msb 0x1cc
	v_cndmask_b32_e32 v208 /*v976*/, 0xff800000, v92 /*v860*/, vcc_lo
	s_set_vgpr_msb 0xcc01
	v_cmp_le_i32_e32 vcc_lo, v35 /*v291*/, v133
	s_set_vgpr_msb 0x1cc
	v_cndmask_b32_e32 v209 /*v977*/, 0xff800000, v93 /*v861*/, vcc_lo
	s_set_vgpr_msb 0xcc01
	v_cmp_le_i32_e32 vcc_lo, v36 /*v292*/, v133
	s_set_vgpr_msb 0x1cc
	v_cndmask_b32_e32 v94 /*v862*/, 0xff800000, v94 /*v862*/, vcc_lo
	s_set_vgpr_msb 0xcc01
	v_cmp_le_i32_e32 vcc_lo, v37 /*v293*/, v133
	s_set_vgpr_msb 0x1cc
	v_cndmask_b32_e32 v95 /*v863*/, 0xff800000, v95 /*v863*/, vcc_lo
	s_set_vgpr_msb 0xcc01
	v_cmp_le_i32_e32 vcc_lo, v38 /*v294*/, v133
	s_set_vgpr_msb 0x1cc
	v_cndmask_b32_e32 v210 /*v978*/, 0xff800000, v80 /*v848*/, vcc_lo
	s_set_vgpr_msb 0xcc01
	v_cmp_le_i32_e32 vcc_lo, v39 /*v295*/, v133
	s_set_vgpr_msb 0x1cc
	v_cndmask_b32_e32 v211 /*v979*/, 0xff800000, v81 /*v849*/, vcc_lo
	s_set_vgpr_msb 0xcc01
	v_cmp_le_i32_e32 vcc_lo, v40 /*v296*/, v133
	s_set_vgpr_msb 0x1cc
	v_cndmask_b32_e32 v212 /*v980*/, 0xff800000, v82 /*v850*/, vcc_lo
	s_set_vgpr_msb 0xcc01
	v_cmp_le_i32_e32 vcc_lo, v41 /*v297*/, v133
	s_set_vgpr_msb 0x1cc
	v_cndmask_b32_e32 v213 /*v981*/, 0xff800000, v83 /*v851*/, vcc_lo
	s_set_vgpr_msb 0xcc01
	v_cmp_le_i32_e32 vcc_lo, v42 /*v298*/, v133
	s_set_vgpr_msb 0x1cc
	v_cndmask_b32_e32 v214 /*v982*/, 0xff800000, v84 /*v852*/, vcc_lo
	s_set_vgpr_msb 0xcc01
	v_cmp_le_i32_e32 vcc_lo, v43 /*v299*/, v133
	s_set_vgpr_msb 0x1cc
	v_cndmask_b32_e32 v215 /*v983*/, 0xff800000, v85 /*v853*/, vcc_lo
	s_set_vgpr_msb 0xcc01
	v_cmp_le_i32_e32 vcc_lo, v44 /*v300*/, v133
	s_set_vgpr_msb 0x1cc
	v_cndmask_b32_e32 v216 /*v984*/, 0xff800000, v86 /*v854*/, vcc_lo
	s_set_vgpr_msb 0xcc01
	v_cmp_le_i32_e32 vcc_lo, v45 /*v301*/, v133
	s_set_vgpr_msb 0x1cc
	v_cndmask_b32_e32 v217 /*v985*/, 0xff800000, v87 /*v855*/, vcc_lo
	s_set_vgpr_msb 0xcc01
	v_cmp_le_i32_e32 vcc_lo, v46 /*v302*/, v133
	s_set_vgpr_msb 0x1cc
	v_cndmask_b32_e32 v218 /*v986*/, 0xff800000, v72 /*v840*/, vcc_lo
	s_set_vgpr_msb 0xccc1
	v_cmp_le_i32_e32 vcc_lo, v47 /*v303*/, v133
	v_add_nc_u32_e32 v72 /*v840*/, 0xf2, v130
	s_set_vgpr_msb 0xc1cc
	v_cndmask_b32_e32 v219 /*v987*/, 0xff800000, v73 /*v841*/, vcc_lo
	s_set_vgpr_msb 0xcc01
	v_cmp_le_i32_e32 vcc_lo, v48 /*v304*/, v133
	s_set_vgpr_msb 0x1cc
	v_cndmask_b32_e32 v220 /*v988*/, 0xff800000, v74 /*v842*/, vcc_lo
	s_set_vgpr_msb 0xcc01
	v_cmp_le_i32_e32 vcc_lo, v49 /*v305*/, v133
	s_set_vgpr_msb 0x1cc
	v_cndmask_b32_e32 v221 /*v989*/, 0xff800000, v75 /*v843*/, vcc_lo
	s_set_vgpr_msb 0xcc01
	v_cmp_le_i32_e32 vcc_lo, v50 /*v306*/, v133
	s_set_vgpr_msb 0x1cc
	v_cndmask_b32_e32 v222 /*v990*/, 0xff800000, v76 /*v844*/, vcc_lo
	s_set_vgpr_msb 0xcc01
	v_cmp_le_i32_e32 vcc_lo, v51 /*v307*/, v133
	s_set_vgpr_msb 0x1cc
	v_cndmask_b32_e32 v223 /*v991*/, 0xff800000, v77 /*v845*/, vcc_lo
	s_set_vgpr_msb 0xcc01
	v_cmp_le_i32_e32 vcc_lo, v52 /*v308*/, v133
	s_set_vgpr_msb 0x1cc
	v_cndmask_b32_e32 v224 /*v992*/, 0xff800000, v78 /*v846*/, vcc_lo
	s_set_vgpr_msb 0xcc01
	v_cmp_le_i32_e32 vcc_lo, v53 /*v309*/, v133
	s_set_vgpr_msb 0x1cc
	v_cndmask_b32_e32 v225 /*v993*/, 0xff800000, v79 /*v847*/, vcc_lo
	s_set_vgpr_msb 0xcc01
	v_cmp_le_i32_e32 vcc_lo, v54 /*v310*/, v133
	s_set_vgpr_msb 0x1cc
	v_cndmask_b32_e32 v226 /*v994*/, 0xff800000, v64 /*v832*/, vcc_lo
	s_set_vgpr_msb 0xccc1
	v_cmp_le_i32_e32 vcc_lo, v55 /*v311*/, v133
	v_add_nc_u32_e32 v64 /*v832*/, 0xf3, v130
	s_set_vgpr_msb 0xc1cc
	v_cndmask_b32_e32 v227 /*v995*/, 0xff800000, v65 /*v833*/, vcc_lo
	v_cmp_ge_i32_e32 vcc_lo, v133, v72 /*v840*/
	s_set_vgpr_msb 0xccc0
	v_add_nc_u32_e32 v65 /*v833*/, 0xf4, v130
	s_set_vgpr_msb 0xc0cc
	v_cndmask_b32_e32 v228 /*v996*/, 0xff800000, v66 /*v834*/, vcc_lo
	v_cmp_ge_i32_e32 vcc_lo, v133, v64 /*v832*/
	s_set_vgpr_msb 0xccc0
	v_add_nc_u32_e32 v66 /*v834*/, 0xf5, v130
	s_set_vgpr_msb 0xc0cc
	v_cndmask_b32_e32 v229 /*v997*/, 0xff800000, v67 /*v835*/, vcc_lo
	v_cmp_ge_i32_e32 vcc_lo, v133, v65 /*v833*/
	s_set_vgpr_msb 0xccc0
	v_add_nc_u32_e32 v67 /*v835*/, 0xf6, v130
	s_set_vgpr_msb 0xc0cc
	v_cndmask_b32_e32 v230 /*v998*/, 0xff800000, v68 /*v836*/, vcc_lo
	v_cmp_ge_i32_e32 vcc_lo, v133, v66 /*v834*/
	s_set_vgpr_msb 0xccc0
	v_add_nc_u32_e32 v68 /*v836*/, 0xf7, v130
	s_set_vgpr_msb 0xc0cc
	v_cndmask_b32_e32 v231 /*v999*/, 0xff800000, v69 /*v837*/, vcc_lo
	v_cmp_ge_i32_e32 vcc_lo, v133, v67 /*v835*/
	v_cndmask_b32_e32 v242 /*v1010*/, 0xff800000, v70 /*v838*/, vcc_lo
	v_cmp_ge_i32_e32 vcc_lo, v133, v68 /*v836*/
	v_cndmask_b32_e32 v243 /*v1011*/, 0xff800000, v71 /*v839*/, vcc_lo
	s_set_vgpr_msb 0xcc00
	v_cmp_le_i32_e32 vcc_lo, v130, v134
	s_set_vgpr_msb 0xc8
	v_cndmask_b32_e32 v232 /*v1000*/, 0xff800000, v192 /*v704*/, vcc_lo
	s_set_vgpr_msb 0xc800
	v_cmp_lt_i32_e32 vcc_lo, v130, v134
	s_set_vgpr_msb 15
	v_max_num_f32_e32 v130, v178 /*v946*/, v179 /*v947*/
	s_set_vgpr_msb 0xfc8
	v_cndmask_b32_e32 v233 /*v1001*/, 0xff800000, v193 /*v705*/, vcc_lo
	s_set_vgpr_msb 0xc800
	v_cmp_le_i32_e32 vcc_lo, v192, v134
	s_set_vgpr_msb 0xbf
	v_max3_num_f32 v193 /*v705*/, v160 /*v928*/, v161 /*v929*/, v162 /*v930*/
	s_set_vgpr_msb 0xbfc8
	v_cndmask_b32_e32 v252 /*v1020*/, 0xff800000, v194 /*v706*/, vcc_lo
	s_set_vgpr_msb 0xc800
	v_cmp_le_i32_e32 vcc_lo, v193, v134
	s_set_vgpr_msb 0xc8
	v_cndmask_b32_e32 v253 /*v1021*/, 0xff800000, v195 /*v707*/, vcc_lo
	s_set_vgpr_msb 0xc800
	v_cmp_le_i32_e32 vcc_lo, v194, v134
	s_set_vgpr_msb 0xbf
	v_max3_num_f32 v195 /*v707*/, v163 /*v931*/, v164 /*v932*/, v165 /*v933*/
	s_set_vgpr_msb 0xbfc8
	v_cndmask_b32_e32 v236 /*v1004*/, 0xff800000, v196 /*v708*/, vcc_lo
	s_set_vgpr_msb 0xc800
	v_cmp_le_i32_e32 vcc_lo, v195, v134
	s_set_vgpr_msb 0xc8
	v_cndmask_b32_e32 v237 /*v1005*/, 0xff800000, v197 /*v709*/, vcc_lo
	s_set_vgpr_msb 0xc800
	v_cmp_le_i32_e32 vcc_lo, v196, v134
	s_set_vgpr_msb 0xbf
	v_max3_num_f32 v197 /*v709*/, v166 /*v934*/, v167 /*v935*/, v152 /*v920*/
	s_set_vgpr_msb 0xbfc8
	v_cndmask_b32_e32 v234 /*v1002*/, 0xff800000, v198 /*v710*/, vcc_lo
	s_set_vgpr_msb 0xc800
	v_cmp_le_i32_e32 vcc_lo, v197, v134
	s_set_vgpr_msb 0xc8
	v_cndmask_b32_e32 v235 /*v1003*/, 0xff800000, v199 /*v711*/, vcc_lo
	s_set_vgpr_msb 0xc800
	v_cmp_le_i32_e32 vcc_lo, v198, v134
	s_set_vgpr_msb 0xbf
	v_max3_num_f32 v199 /*v711*/, v153 /*v921*/, v154 /*v922*/, v155 /*v923*/
	s_set_vgpr_msb 0xbfc8
	v_cndmask_b32_e32 v244 /*v1012*/, 0xff800000, v200 /*v712*/, vcc_lo
	s_set_vgpr_msb 0xc800
	v_cmp_le_i32_e32 vcc_lo, v199, v134
	s_set_vgpr_msb 0xc8
	v_cndmask_b32_e32 v245 /*v1013*/, 0xff800000, v201 /*v713*/, vcc_lo
	s_set_vgpr_msb 0xc800
	v_cmp_le_i32_e32 vcc_lo, v200, v134
	s_set_vgpr_msb 0xbf
	v_max3_num_f32 v201 /*v713*/, v156 /*v924*/, v157 /*v925*/, v158 /*v926*/
	s_set_vgpr_msb 0xbfc8
	v_cndmask_b32_e32 v238 /*v1006*/, 0xff800000, v202 /*v714*/, vcc_lo
	s_set_vgpr_msb 0xc800
	v_cmp_le_i32_e32 vcc_lo, v201, v134
	s_set_vgpr_msb 0xc8
	v_cndmask_b32_e32 v239 /*v1007*/, 0xff800000, v203 /*v715*/, vcc_lo
	s_set_vgpr_msb 0xc800
	v_cmp_le_i32_e32 vcc_lo, v202, v134
	s_set_vgpr_msb 0xbf
	v_max3_num_f32 v203 /*v715*/, v159 /*v927*/, v144 /*v912*/, v145 /*v913*/
	s_set_vgpr_msb 0xbf08
	v_cndmask_b32_e32 v198, 0xff800000, v204 /*v716*/, vcc_lo
	s_set_vgpr_msb 0x800
	v_cmp_le_i32_e32 vcc_lo, v203, v134
	s_set_vgpr_msb 8
	v_cndmask_b32_e32 v199, 0xff800000, v205 /*v717*/, vcc_lo
	s_set_vgpr_msb 0x800
	v_cmp_le_i32_e32 vcc_lo, v204, v134
	s_set_vgpr_msb 0xbf
	v_max3_num_f32 v205 /*v717*/, v146 /*v914*/, v147 /*v915*/, v148 /*v916*/
	s_set_vgpr_msb 0xbfc8
	v_cndmask_b32_e32 v246 /*v1014*/, 0xff800000, v206 /*v718*/, vcc_lo
	s_set_vgpr_msb 0xc800
	v_cmp_le_i32_e32 vcc_lo, v205, v134
	s_set_vgpr_msb 0xc8
	v_cndmask_b32_e32 v247 /*v1015*/, 0xff800000, v207 /*v719*/, vcc_lo
	s_set_vgpr_msb 0xc800
	v_cmp_le_i32_e32 vcc_lo, v206, v134
	s_set_vgpr_msb 0xbf
	v_max3_num_f32 v207 /*v719*/, v149 /*v917*/, v150 /*v918*/, v151 /*v919*/
	s_set_vgpr_msb 0xbfc8
	v_cndmask_b32_e32 v240 /*v1008*/, 0xff800000, v208 /*v720*/, vcc_lo
	s_set_vgpr_msb 0xc800
	v_cmp_le_i32_e32 vcc_lo, v207, v134
	s_set_vgpr_msb 0xc8
	v_cndmask_b32_e32 v241 /*v1009*/, 0xff800000, v209 /*v721*/, vcc_lo
	s_set_vgpr_msb 0xc800
	v_cmp_le_i32_e32 vcc_lo, v208, v134
	s_set_vgpr_msb 0xbf
	v_max3_num_f32 v209 /*v721*/, v136 /*v904*/, v137 /*v905*/, v138 /*v906*/
	s_set_vgpr_msb 0xbfc8
	v_cndmask_b32_e32 v254 /*v1022*/, 0xff800000, v210 /*v722*/, vcc_lo
	s_set_vgpr_msb 0xc800
	v_cmp_le_i32_e32 vcc_lo, v209, v134
	s_set_vgpr_msb 0xc8
	v_cndmask_b32_e32 v255 /*v1023*/, 0xff800000, v211 /*v723*/, vcc_lo
	s_set_vgpr_msb 0xc800
	v_cmp_le_i32_e32 vcc_lo, v210, v134
	s_set_vgpr_msb 0xbf
	v_max3_num_f32 v211 /*v723*/, v139 /*v907*/, v140 /*v908*/, v141 /*v909*/
	s_set_vgpr_msb 0xbfc8
	v_cndmask_b32_e32 v248 /*v1016*/, 0xff800000, v212 /*v724*/, vcc_lo
	s_set_vgpr_msb 0xc800
	v_cmp_le_i32_e32 vcc_lo, v211, v134
	s_set_vgpr_msb 0xc8
	v_cndmask_b32_e32 v249 /*v1017*/, 0xff800000, v213 /*v725*/, vcc_lo
	s_set_vgpr_msb 0xc800
	v_cmp_le_i32_e32 vcc_lo, v212, v134
	s_set_vgpr_msb 0xbf
	v_max3_num_f32 v213 /*v725*/, v142 /*v910*/, v143 /*v911*/, v128 /*v896*/
	s_set_vgpr_msb 0xbf08
	v_cndmask_b32_e32 v208, 0xff800000, v214 /*v726*/, vcc_lo
	s_set_vgpr_msb 0x800
	v_cmp_le_i32_e32 vcc_lo, v213, v134
	s_set_vgpr_msb 8
	v_cndmask_b32_e32 v209, 0xff800000, v215 /*v727*/, vcc_lo
	s_set_vgpr_msb 0x800
	v_cmp_le_i32_e32 vcc_lo, v214, v134
	s_set_vgpr_msb 0xbf
	v_max3_num_f32 v215 /*v727*/, v129 /*v897*/, v130 /*v898*/, v131 /*v899*/
	s_set_vgpr_msb 0xbf83
	v_max3_num_f32 v192 /*v704*/, v249 /*v1017*/, v208, v209
	s_set_vgpr_msb 0x8308
	v_cndmask_b32_e32 v192, 0xff800000, v216 /*v728*/, vcc_lo
	s_set_vgpr_msb 0x800
	v_cmp_le_i32_e32 vcc_lo, v215, v134
	s_set_vgpr_msb 8
	v_cndmask_b32_e32 v193, 0xff800000, v217 /*v729*/, vcc_lo
	s_set_vgpr_msb 0x800
	v_cmp_le_i32_e32 vcc_lo, v216, v134
	s_set_vgpr_msb 0xbf
	v_max3_num_f32 v217 /*v729*/, v132 /*v900*/, v133 /*v901*/, v134 /*v902*/
	s_set_vgpr_msb 0xbfc8
	v_cndmask_b32_e32 v250 /*v1018*/, 0xff800000, v218 /*v730*/, vcc_lo
	s_set_vgpr_msb 0xc800
	v_cmp_le_i32_e32 vcc_lo, v217, v134
	s_set_vgpr_msb 0xc8
	v_cndmask_b32_e32 v251 /*v1019*/, 0xff800000, v219 /*v731*/, vcc_lo
	s_set_vgpr_msb 0xc800
	v_cmp_le_i32_e32 vcc_lo, v218, v134
	s_set_vgpr_msb 0xbf
	v_max3_num_f32 v219 /*v731*/, v135 /*v903*/, v120 /*v888*/, v121 /*v889*/
	s_set_vgpr_msb 0xbfb0
	v_max3_num_f32 v194 /*v706*/, v192, v193, v250 /*v1018*/
	s_set_vgpr_msb 0xb008
	v_cndmask_b32_e32 v200, 0xff800000, v220 /*v732*/, vcc_lo
	s_set_vgpr_msb 0x800
	v_cmp_le_i32_e32 vcc_lo, v219, v134
	s_set_vgpr_msb 8
	v_cndmask_b32_e32 v201, 0xff800000, v221 /*v733*/, vcc_lo
	s_set_vgpr_msb 0x800
	v_cmp_le_i32_e32 vcc_lo, v220, v134
	s_set_vgpr_msb 0xbf
	v_max3_num_f32 v221 /*v733*/, v122 /*v890*/, v123 /*v891*/, v124 /*v892*/
	s_set_vgpr_msb 0xbf83
	v_max3_num_f32 v196 /*v708*/, v251 /*v1019*/, v200, v201
	s_set_vgpr_msb 0x8308
	v_cndmask_b32_e32 v194, 0xff800000, v222 /*v734*/, vcc_lo
	s_set_vgpr_msb 0x800
	v_cmp_le_i32_e32 vcc_lo, v221, v134
	s_set_vgpr_msb 8
	v_cndmask_b32_e32 v195, 0xff800000, v223 /*v735*/, vcc_lo
	s_set_vgpr_msb 0x800
	v_cmp_le_i32_e32 vcc_lo, v222, v134
	s_set_vgpr_msb 0xbf
	v_max3_num_f32 v223 /*v735*/, v125 /*v893*/, v176 /*v944*/, v177 /*v945*/
	s_set_vgpr_msb 0xbf08
	v_cndmask_b32_e32 v218, 0xff800000, v224 /*v736*/, vcc_lo
	s_set_vgpr_msb 0x800
	v_cmp_le_i32_e32 vcc_lo, v223, v134
	s_set_vgpr_msb 8
	v_cndmask_b32_e32 v219, 0xff800000, v225 /*v737*/, vcc_lo
	s_set_vgpr_msb 0x880
	v_cmp_le_i32_e32 vcc_lo, v224, v134
	v_max3_num_f32 v198 /*v710*/, v194, v195, v218
	s_set_vgpr_msb 0x80bf
	v_max3_num_f32 v225 /*v737*/, v194 /*v962*/, v195 /*v963*/, v114 /*v882*/
	s_set_vgpr_msb 0xbf08
	v_cndmask_b32_e32 v202, 0xff800000, v226 /*v738*/, vcc_lo
	s_set_vgpr_msb 0x800
	v_cmp_le_i32_e32 vcc_lo, v225, v134
	s_set_vgpr_msb 8
	v_cndmask_b32_e32 v203, 0xff800000, v227 /*v739*/, vcc_lo
	s_set_vgpr_msb 0x800
	v_cmp_le_i32_e32 vcc_lo, v226, v134
	s_set_vgpr_msb 0xbf
	v_max3_num_f32 v227 /*v739*/, v115 /*v883*/, v116 /*v884*/, v117 /*v885*/
	s_set_vgpr_msb 0xbf80
	v_max3_num_f32 v200 /*v712*/, v219, v202, v203
	s_set_vgpr_msb 0x8008
	v_cndmask_b32_e32 v196, 0xff800000, v228 /*v740*/, vcc_lo
	s_set_vgpr_msb 0x800
	v_cmp_le_i32_e32 vcc_lo, v227, v134
	s_set_vgpr_msb 8
	v_cndmask_b32_e32 v197, 0xff800000, v229 /*v741*/, vcc_lo
	s_set_vgpr_msb 0x800
	v_cmp_le_i32_e32 vcc_lo, v228, v134
	s_set_vgpr_msb 0xbf
	v_max3_num_f32 v229 /*v741*/, v118 /*v886*/, v119 /*v887*/, v104 /*v872*/
	s_set_vgpr_msb 0xbf08
	v_cndmask_b32_e32 v210, 0xff800000, v230 /*v742*/, vcc_lo
	s_set_vgpr_msb 0x800
	v_cmp_le_i32_e32 vcc_lo, v229, v134
	s_set_vgpr_msb 8
	v_cndmask_b32_e32 v211, 0xff800000, v231 /*v743*/, vcc_lo
	s_set_vgpr_msb 0x800
	v_cmp_le_i32_e32 vcc_lo, v230, v134
	s_set_vgpr_msb 0x8f
	v_max_num_f32_e32 v231 /*v743*/, v105 /*v873*/, v196 /*v964*/
	s_set_vgpr_msb 0x8f80
	v_max3_num_f32 v202 /*v714*/, v196, v197, v210
	s_set_vgpr_msb 0x8008
	v_cndmask_b32_e32 v204, 0xff800000, v232 /*v744*/, vcc_lo
	s_set_vgpr_msb 0x800
	v_cmp_le_i32_e32 vcc_lo, v231, v134
	s_set_vgpr_msb 8
	v_cndmask_b32_e32 v205, 0xff800000, v233 /*v745*/, vcc_lo
	s_set_vgpr_msb 0x800
	v_cmp_le_i32_e32 vcc_lo, v232, v134
	s_set_vgpr_msb 0xbf
	v_max3_num_f32 v233 /*v745*/, v108 /*v876*/, v109 /*v877*/, v110 /*v878*/
	s_set_vgpr_msb 0xbf80
	v_max3_num_f32 v204 /*v716*/, v211, v204, v205
	s_set_vgpr_msb 0x8008
	v_cndmask_b32_e32 v228, 0xff800000, v234 /*v746*/, vcc_lo
	s_set_vgpr_msb 0x800
	v_cmp_le_i32_e32 vcc_lo, v233, v134
	s_set_vgpr_msb 8
	v_cndmask_b32_e32 v229, 0xff800000, v235 /*v747*/, vcc_lo
	s_set_vgpr_msb 0x800
	v_cmp_le_i32_e32 vcc_lo, v234, v134
	s_set_vgpr_msb 0xbf
	v_max3_num_f32 v235 /*v747*/, v111 /*v879*/, v198 /*v966*/, v199 /*v967*/
	s_set_vgpr_msb 0xbf08
	v_cndmask_b32_e32 v212, 0xff800000, v236 /*v748*/, vcc_lo
	s_set_vgpr_msb 0x800
	v_cmp_le_i32_e32 vcc_lo, v235, v134
	s_set_vgpr_msb 8
	v_cndmask_b32_e32 v213, 0xff800000, v237 /*v749*/, vcc_lo
	s_set_vgpr_msb 0x800
	v_cmp_le_i32_e32 vcc_lo, v236, v134
	s_set_vgpr_msb 0xbf
	v_max3_num_f32 v237 /*v749*/, v200 /*v968*/, v201 /*v969*/, v100 /*v868*/
	s_set_vgpr_msb 0xbf80
	v_max3_num_f32 v206 /*v718*/, v228, v229, v212
	s_set_vgpr_msb 0x8008
	v_cndmask_b32_e32 v206, 0xff800000, v238 /*v750*/, vcc_lo
	s_set_vgpr_msb 0x800
	v_cmp_le_i32_e32 vcc_lo, v237, v134
	s_set_vgpr_msb 8
	v_cndmask_b32_e32 v207, 0xff800000, v239 /*v751*/, vcc_lo
	s_set_vgpr_msb 0x800
	v_cmp_le_i32_e32 vcc_lo, v238, v134
	s_set_vgpr_msb 0xbf
	v_max3_num_f32 v239 /*v751*/, v101 /*v869*/, v202 /*v970*/, v203 /*v971*/
	s_set_vgpr_msb 0xbf80
	v_max3_num_f32 v208 /*v720*/, v213, v206, v207
	s_set_vgpr_msb 0x8008
	v_cndmask_b32_e32 v220, 0xff800000, v240 /*v752*/, vcc_lo
	s_set_vgpr_msb 0x800
	v_cmp_le_i32_e32 vcc_lo, v239, v134
	s_set_vgpr_msb 8
	v_cndmask_b32_e32 v221, 0xff800000, v241 /*v753*/, vcc_lo
	s_set_vgpr_msb 0x800
	v_cmp_le_i32_e32 vcc_lo, v240, v134
	s_set_vgpr_msb 0xbf
	v_max3_num_f32 v241 /*v753*/, v204 /*v972*/, v205 /*v973*/, v206 /*v974*/
	s_set_vgpr_msb 0xbf08
	v_cndmask_b32_e32 v214, 0xff800000, v242 /*v754*/, vcc_lo
	s_set_vgpr_msb 0x800
	v_cmp_le_i32_e32 vcc_lo, v241, v134
	s_set_vgpr_msb 8
	v_cndmask_b32_e32 v215, 0xff800000, v243 /*v755*/, vcc_lo
	s_set_vgpr_msb 0x800
	v_cmp_le_i32_e32 vcc_lo, v242, v134
	s_set_vgpr_msb 0xbf
	v_max3_num_f32 v243 /*v755*/, v207 /*v975*/, v208 /*v976*/, v209 /*v977*/
	s_set_vgpr_msb 0xbf80
	v_max3_num_f32 v210 /*v722*/, v220, v221, v214
	s_set_vgpr_msb 0x8008
	v_cndmask_b32_e32 v238, 0xff800000, v244 /*v756*/, vcc_lo
	s_set_vgpr_msb 0x800
	v_cmp_le_i32_e32 vcc_lo, v243, v134
	s_set_vgpr_msb 8
	v_cndmask_b32_e32 v239, 0xff800000, v245 /*v757*/, vcc_lo
	s_set_vgpr_msb 0x800
	v_cmp_le_i32_e32 vcc_lo, v244, v134
	s_set_vgpr_msb 0xbf
	v_max3_num_f32 v245 /*v757*/, v94 /*v862*/, v95 /*v863*/, v210 /*v978*/
	s_set_vgpr_msb 0xbf80
	v_max3_num_f32 v212 /*v724*/, v215, v238, v239
	s_set_vgpr_msb 0x8008
	v_cndmask_b32_e32 v222, 0xff800000, v246 /*v758*/, vcc_lo
	s_set_vgpr_msb 0x800
	v_cmp_le_i32_e32 vcc_lo, v245, v134
	s_set_vgpr_msb 8
	v_cndmask_b32_e32 v223, 0xff800000, v247 /*v759*/, vcc_lo
	s_set_vgpr_msb 0x800
	v_cmp_le_i32_e32 vcc_lo, v246, v134
	s_set_vgpr_msb 0xbf
	v_max3_num_f32 v247 /*v759*/, v211 /*v979*/, v212 /*v980*/, v213 /*v981*/
	s_set_vgpr_msb 0xbf08
	v_cndmask_b32_e32 v216, 0xff800000, v248 /*v760*/, vcc_lo
	s_set_vgpr_msb 0x800
	v_cmp_le_i32_e32 vcc_lo, v247, v134
	s_set_vgpr_msb 8
	v_cndmask_b32_e32 v217, 0xff800000, v249 /*v761*/, vcc_lo
	s_set_vgpr_msb 0x800
	v_cmp_le_i32_e32 vcc_lo, v248, v134
	s_set_vgpr_msb 0x8f
	v_max_num_f32_e32 v249 /*v761*/, v214 /*v982*/, v215 /*v983*/
	s_set_vgpr_msb 0x8f80
	v_max3_num_f32 v214 /*v726*/, v222, v223, v216
	s_set_vgpr_msb 0x8008
	v_cndmask_b32_e32 v230, 0xff800000, v250 /*v762*/, vcc_lo
	s_set_vgpr_msb 0x800
	v_cmp_le_i32_e32 vcc_lo, v249, v134
	s_set_vgpr_msb 8
	v_cndmask_b32_e32 v231, 0xff800000, v251 /*v763*/, vcc_lo
	s_set_vgpr_msb 0x800
	v_cmp_le_i32_e32 vcc_lo, v250, v134
	s_set_vgpr_msb 0xbf
	v_max3_num_f32 v251 /*v763*/, v217 /*v985*/, v218 /*v986*/, v219 /*v987*/
	s_set_vgpr_msb 0xbf80
	v_max3_num_f32 v216 /*v728*/, v217, v230, v231
	s_set_vgpr_msb 0x8008
	v_cndmask_b32_e32 v224, 0xff800000, v252 /*v764*/, vcc_lo
	s_set_vgpr_msb 0x800
	v_cmp_le_i32_e32 vcc_lo, v251, v134
	s_set_vgpr_msb 8
	v_cndmask_b32_e32 v225, 0xff800000, v253 /*v765*/, vcc_lo
	s_set_vgpr_msb 0x800
	v_cmp_le_i32_e32 vcc_lo, v252, v134
	s_set_vgpr_msb 0xbf
	v_max3_num_f32 v253 /*v765*/, v220 /*v988*/, v221 /*v989*/, v222 /*v990*/
	s_set_vgpr_msb 0xbf08
	v_cndmask_b32_e32 v248, 0xff800000, v254 /*v766*/, vcc_lo
	s_set_vgpr_msb 0x800
	v_cmp_le_i32_e32 vcc_lo, v253, v134
	s_set_vgpr_msb 8
	v_cndmask_b32_e32 v249, 0xff800000, v255 /*v767*/, vcc_lo
	s_set_vgpr_msb 0x800
	v_cmp_le_i32_e32 vcc_lo, v254, v134
	s_set_vgpr_msb 0xbf
	v_max3_num_f32 v255 /*v767*/, v223 /*v991*/, v224 /*v992*/, v225 /*v993*/
	s_set_vgpr_msb 0xbf80
	v_max3_num_f32 v218 /*v730*/, v224, v225, v248
	s_set_vgpr_msb 0x800c
	v_cndmask_b32_e32 v232, 0xff800000, v0 /*v768*/, vcc_lo
	s_set_vgpr_msb 0xc00
	v_cmp_le_i32_e32 vcc_lo, v255, v134
	s_set_vgpr_msb 12
	v_cndmask_b32_e32 v233, 0xff800000, v1 /*v769*/, vcc_lo
	s_set_vgpr_msb 0xc01
	v_cmp_le_i32_e32 vcc_lo, v0 /*v256*/, v134
	s_set_vgpr_msb 0x1ff
	v_max3_num_f32 v1 /*v769*/, v226 /*v994*/, v227 /*v995*/, v228 /*v996*/
	s_set_vgpr_msb 0xff80
	v_max3_num_f32 v220 /*v732*/, v249, v232, v233
	s_set_vgpr_msb 0x800c
	v_cndmask_b32_e32 v226, 0xff800000, v2 /*v770*/, vcc_lo
	s_set_vgpr_msb 0xc01
	v_cmp_le_i32_e32 vcc_lo, v1 /*v257*/, v134
	s_set_vgpr_msb 0x10c
	v_cndmask_b32_e32 v227, 0xff800000, v3 /*v771*/, vcc_lo
	s_set_vgpr_msb 0xc01
	v_cmp_le_i32_e32 vcc_lo, v2 /*v258*/, v134
	s_set_vgpr_msb 0x1ff
	v_max3_num_f32 v3 /*v771*/, v229 /*v997*/, v230 /*v998*/, v231 /*v999*/
	s_set_vgpr_msb 0xff0c
	v_cndmask_b32_e32 v240, 0xff800000, v4 /*v772*/, vcc_lo
	s_set_vgpr_msb 0xc01
	v_cmp_le_i32_e32 vcc_lo, v3 /*v259*/, v134
	s_set_vgpr_msb 0x10c
	v_cndmask_b32_e32 v241, 0xff800000, v5 /*v773*/, vcc_lo
	s_set_vgpr_msb 0xc01
	v_cmp_le_i32_e32 vcc_lo, v4 /*v260*/, v134
	s_set_vgpr_msb 0x180
	v_max3_num_f32 v222 /*v734*/, v226, v227, v240
	s_set_vgpr_msb 0x800c
	v_cndmask_b32_e32 v234, 0xff800000, v6 /*v774*/, vcc_lo
	s_set_vgpr_msb 0xc01
	v_cmp_le_i32_e32 vcc_lo, v5 /*v261*/, v134
	s_set_vgpr_msb 0x10c
	v_cndmask_b32_e32 v235, 0xff800000, v7 /*v775*/, vcc_lo
	s_set_vgpr_msb 0xc01
	v_cmp_le_i32_e32 vcc_lo, v6 /*v262*/, v134
	s_set_vgpr_msb 0x180
	v_max3_num_f32 v224 /*v736*/, v241, v234, v235
	s_set_vgpr_msb 0x804c
	v_cndmask_b32_e32 v2 /*v258*/, 0xff800000, v8 /*v776*/, vcc_lo
	s_set_vgpr_msb 0x4c01
	v_cmp_le_i32_e32 vcc_lo, v7 /*v263*/, v134
	s_set_vgpr_msb 0x14c
	v_cndmask_b32_e32 v3 /*v259*/, 0xff800000, v9 /*v777*/, vcc_lo
	s_set_vgpr_msb 0x4c01
	v_cmp_le_i32_e32 vcc_lo, v8 /*v264*/, v134
	s_set_vgpr_msb 0x10c
	v_cndmask_b32_e32 v242, 0xff800000, v10 /*v778*/, vcc_lo
	s_set_vgpr_msb 0xc01
	v_cmp_le_i32_e32 vcc_lo, v9 /*v265*/, v134
	s_set_vgpr_msb 0x10c
	v_cndmask_b32_e32 v243, 0xff800000, v11 /*v779*/, vcc_lo
	s_set_vgpr_msb 0xc01
	v_cmp_le_i32_e32 vcc_lo, v10 /*v266*/, v134
	s_set_vgpr_msb 0x185
	v_max3_num_f32 v226 /*v738*/, v2 /*v258*/, v3 /*v259*/, v242
	s_set_vgpr_msb 0x850c
	v_cndmask_b32_e32 v236, 0xff800000, v12 /*v780*/, vcc_lo
	s_set_vgpr_msb 0xc01
	v_cmp_le_i32_e32 vcc_lo, v11 /*v267*/, v134
	s_set_vgpr_msb 0x10c
	v_cndmask_b32_e32 v237, 0xff800000, v13 /*v781*/, vcc_lo
	s_set_vgpr_msb 0xc01
	v_cmp_le_i32_e32 vcc_lo, v12 /*v268*/, v134
	s_set_vgpr_msb 0x180
	v_max3_num_f32 v228 /*v740*/, v243, v236, v237
	s_set_vgpr_msb 0x800c
	v_cndmask_b32_e32 v250, 0xff800000, v14 /*v782*/, vcc_lo
	s_set_vgpr_msb 0xc01
	v_cmp_le_i32_e32 vcc_lo, v13 /*v269*/, v134
	s_set_vgpr_msb 0x10c
	v_cndmask_b32_e32 v251, 0xff800000, v15 /*v783*/, vcc_lo
	s_set_vgpr_msb 0xc01
	v_cmp_le_i32_e32 vcc_lo, v14 /*v270*/, v134
	s_set_vgpr_msb 0x10c
	v_cndmask_b32_e32 v244, 0xff800000, v16 /*v784*/, vcc_lo
	s_set_vgpr_msb 0xc01
	v_cmp_le_i32_e32 vcc_lo, v15 /*v271*/, v134
	s_set_vgpr_msb 0x10c
	v_cndmask_b32_e32 v245, 0xff800000, v17 /*v785*/, vcc_lo
	s_set_vgpr_msb 0xc01
	v_cmp_le_i32_e32 vcc_lo, v16 /*v272*/, v134
	s_set_vgpr_msb 0x180
	v_max3_num_f32 v230 /*v742*/, v250, v251, v244
	s_set_vgpr_msb 0x804c
	v_cndmask_b32_e32 v12 /*v268*/, 0xff800000, v18 /*v786*/, vcc_lo
	s_set_vgpr_msb 0x4c01
	v_cmp_le_i32_e32 vcc_lo, v17 /*v273*/, v134
	s_set_vgpr_msb 0x14c
	v_cndmask_b32_e32 v13 /*v269*/, 0xff800000, v19 /*v787*/, vcc_lo
	s_set_vgpr_msb 0x4c01
	v_cmp_le_i32_e32 vcc_lo, v18 /*v274*/, v134
	s_set_vgpr_msb 0x184
	v_max_num_f32_e32 v232 /*v744*/, v245, v12 /*v268*/
	s_set_vgpr_msb 0x840c
	v_cndmask_b32_e32 v252, 0xff800000, v20 /*v788*/, vcc_lo
	s_set_vgpr_msb 0xc01
	v_cmp_le_i32_e32 vcc_lo, v19 /*v275*/, v134
	s_set_vgpr_msb 0x10c
	v_cndmask_b32_e32 v253, 0xff800000, v21 /*v789*/, vcc_lo
	s_set_vgpr_msb 0xc01
	v_cmp_le_i32_e32 vcc_lo, v20 /*v276*/, v134
	s_set_vgpr_msb 0x10c
	v_cndmask_b32_e32 v246, 0xff800000, v22 /*v790*/, vcc_lo
	s_set_vgpr_msb 0xc01
	v_cmp_le_i32_e32 vcc_lo, v21 /*v277*/, v134
	s_set_vgpr_msb 0x10c
	v_cndmask_b32_e32 v247, 0xff800000, v23 /*v791*/, vcc_lo
	s_set_vgpr_msb 0xc01
	v_cmp_le_i32_e32 vcc_lo, v22 /*v278*/, v134
	s_set_vgpr_msb 0x180
	v_max3_num_f32 v234 /*v746*/, v252, v253, v246
	s_set_vgpr_msb 0x804c
	v_cndmask_b32_e32 v4 /*v260*/, 0xff800000, v24 /*v792*/, vcc_lo
	s_set_vgpr_msb 0x4c01
	v_cmp_le_i32_e32 vcc_lo, v23 /*v279*/, v134
	s_set_vgpr_msb 0x14c
	v_cndmask_b32_e32 v5 /*v261*/, 0xff800000, v25 /*v793*/, vcc_lo
	s_set_vgpr_msb 0x4c01
	v_cmp_le_i32_e32 vcc_lo, v24 /*v280*/, v134
	s_set_vgpr_msb 0x194
	v_max3_num_f32 v236 /*v748*/, v247, v4 /*v260*/, v5 /*v261*/
	s_set_vgpr_msb 0x940c
	v_cndmask_b32_e32 v254, 0xff800000, v26 /*v794*/, vcc_lo
	s_set_vgpr_msb 0xc01
	v_cmp_le_i32_e32 vcc_lo, v25 /*v281*/, v134
	s_set_vgpr_msb 0x10c
	v_cndmask_b32_e32 v255, 0xff800000, v27 /*v795*/, vcc_lo
	s_set_vgpr_msb 0xc01
	v_cmp_le_i32_e32 vcc_lo, v26 /*v282*/, v134
	s_set_vgpr_msb 0x14c
	v_cndmask_b32_e32 v24 /*v280*/, 0xff800000, v28 /*v796*/, vcc_lo
	s_set_vgpr_msb 0x4c01
	v_cmp_le_i32_e32 vcc_lo, v27 /*v283*/, v134
	s_set_vgpr_msb 0x14c
	v_cndmask_b32_e32 v25 /*v281*/, 0xff800000, v29 /*v797*/, vcc_lo
	s_set_vgpr_msb 0x4c01
	v_cmp_le_i32_e32 vcc_lo, v28 /*v284*/, v134
	s_set_vgpr_msb 0x190
	v_max3_num_f32 v238 /*v750*/, v254, v255, v24 /*v280*/
	s_set_vgpr_msb 0x904c
	v_cndmask_b32_e32 v6 /*v262*/, 0xff800000, v30 /*v798*/, vcc_lo
	s_set_vgpr_msb 0x4c01
	v_cmp_le_i32_e32 vcc_lo, v29 /*v285*/, v134
	s_set_vgpr_msb 0x14c
	v_cndmask_b32_e32 v7 /*v263*/, 0xff800000, v31 /*v799*/, vcc_lo
	s_set_vgpr_msb 0x4c01
	v_cmp_le_i32_e32 vcc_lo, v30 /*v286*/, v134
	s_set_vgpr_msb 0x195
	v_max3_num_f32 v240 /*v752*/, v25 /*v281*/, v6 /*v262*/, v7 /*v263*/
	s_set_vgpr_msb 0x954c
	v_cndmask_b32_e32 v0 /*v256*/, 0xff800000, v32 /*v800*/, vcc_lo
	s_set_vgpr_msb 0x4c01
	v_cmp_le_i32_e32 vcc_lo, v31 /*v287*/, v134
	s_set_vgpr_msb 0x14c
	v_cndmask_b32_e32 v1 /*v257*/, 0xff800000, v33 /*v801*/, vcc_lo
	s_set_vgpr_msb 0x4c01
	v_cmp_le_i32_e32 vcc_lo, v32 /*v288*/, v134
	s_set_vgpr_msb 0x14c
	v_cndmask_b32_e32 v14 /*v270*/, 0xff800000, v34 /*v802*/, vcc_lo
	s_set_vgpr_msb 0x4c01
	v_cmp_le_i32_e32 vcc_lo, v33 /*v289*/, v134
	s_set_vgpr_msb 0x14c
	v_cndmask_b32_e32 v15 /*v271*/, 0xff800000, v35 /*v803*/, vcc_lo
	s_set_vgpr_msb 0x4c01
	v_cmp_le_i32_e32 vcc_lo, v34 /*v290*/, v134
	s_set_vgpr_msb 0x195
	v_max3_num_f32 v242 /*v754*/, v0 /*v256*/, v1 /*v257*/, v14 /*v270*/
	s_set_vgpr_msb 0x954c
	v_cndmask_b32_e32 v8 /*v264*/, 0xff800000, v36 /*v804*/, vcc_lo
	s_set_vgpr_msb 0x4c01
	v_cmp_le_i32_e32 vcc_lo, v35 /*v291*/, v134
	s_set_vgpr_msb 0x14c
	v_cndmask_b32_e32 v9 /*v265*/, 0xff800000, v37 /*v805*/, vcc_lo
	s_set_vgpr_msb 0x4c01
	v_cmp_le_i32_e32 vcc_lo, v36 /*v292*/, v134
	s_set_vgpr_msb 0x195
	v_max3_num_f32 v244 /*v756*/, v15 /*v271*/, v8 /*v264*/, v9 /*v265*/
	s_set_vgpr_msb 0x954c
	v_cndmask_b32_e32 v32 /*v288*/, 0xff800000, v38 /*v806*/, vcc_lo
	s_set_vgpr_msb 0x4c01
	v_cmp_le_i32_e32 vcc_lo, v37 /*v293*/, v134
	s_set_vgpr_msb 0x14c
	v_cndmask_b32_e32 v33 /*v289*/, 0xff800000, v39 /*v807*/, vcc_lo
	s_set_vgpr_msb 0x4c01
	v_cmp_le_i32_e32 vcc_lo, v38 /*v294*/, v134
	s_set_vgpr_msb 0x14c
	v_cndmask_b32_e32 v16 /*v272*/, 0xff800000, v40 /*v808*/, vcc_lo
	s_set_vgpr_msb 0x4c01
	v_cmp_le_i32_e32 vcc_lo, v39 /*v295*/, v134
	s_set_vgpr_msb 0x14c
	v_cndmask_b32_e32 v17 /*v273*/, 0xff800000, v41 /*v809*/, vcc_lo
	s_set_vgpr_msb 0x4c01
	v_cmp_le_i32_e32 vcc_lo, v40 /*v296*/, v134
	s_set_vgpr_msb 0x195
	v_max3_num_f32 v246 /*v758*/, v32 /*v288*/, v33 /*v289*/, v16 /*v272*/
	s_set_vgpr_msb 0x954c
	v_cndmask_b32_e32 v10 /*v266*/, 0xff800000, v42 /*v810*/, vcc_lo
	s_set_vgpr_msb 0x4c01
	v_cmp_le_i32_e32 vcc_lo, v41 /*v297*/, v134
	s_set_vgpr_msb 0x14c
	v_cndmask_b32_e32 v11 /*v267*/, 0xff800000, v43 /*v811*/, vcc_lo
	s_set_vgpr_msb 0x4c01
	v_cmp_le_i32_e32 vcc_lo, v42 /*v298*/, v134
	s_set_vgpr_msb 0x14f
	v_max_num_f32_e32 v42 /*v298*/, v232 /*v1000*/, v233 /*v1001*/
	s_set_vgpr_msb 0x4f95
	v_max3_num_f32 v248 /*v760*/, v17 /*v273*/, v10 /*v266*/, v11 /*v267*/
	s_set_vgpr_msb 0x954c
	v_cndmask_b32_e32 v26 /*v282*/, 0xff800000, v44 /*v812*/, vcc_lo
	s_set_vgpr_msb 0x4c01
	v_cmp_le_i32_e32 vcc_lo, v43 /*v299*/, v134
	s_set_vgpr_msb 0x17f
	v_max3_num_f32 v43 /*v299*/, v181 /*v949*/, v182 /*v950*/, v183 /*v951*/
	v_cndmask_b32_e32 v27 /*v283*/, 0xff800000, v45 /*v813*/, vcc_lo
	s_set_vgpr_msb 0x7f01
	v_cmp_le_i32_e32 vcc_lo, v44 /*v300*/, v134
	s_set_vgpr_msb 0x17f
	v_max3_num_f32 v44 /*v300*/, v253 /*v1021*/, v236 /*v1004*/, v237 /*v1005*/
	s_set_vgpr_msb 0x7f1c
	v_max3_num_f32 v130, v130, v180 /*v948*/, v43 /*v299*/
	s_set_vgpr_msb 0x1c85
	v_max_num_f32_e32 v250 /*v762*/, v26 /*v282*/, v27 /*v283*/
	s_set_vgpr_msb 0x854c
	v_cndmask_b32_e32 v18 /*v274*/, 0xff800000, v46 /*v814*/, vcc_lo
	s_set_vgpr_msb 0x4c01
	v_cmp_le_i32_e32 vcc_lo, v45 /*v301*/, v134
	s_set_vgpr_msb 0x15d
	v_max3_num_f32 v42 /*v298*/, v42 /*v298*/, v252 /*v1020*/, v44 /*v300*/
	s_set_vgpr_msb 0x5d7f
	v_max3_num_f32 v45 /*v301*/, v184 /*v952*/, v185 /*v953*/, v186 /*v954*/
	v_cndmask_b32_e32 v19 /*v275*/, 0xff800000, v47 /*v815*/, vcc_lo
	s_set_vgpr_msb 0x7f01
	v_cmp_le_i32_e32 vcc_lo, v46 /*v302*/, v134
	s_set_vgpr_msb 0x17f
	v_max3_num_f32 v46 /*v302*/, v234 /*v1002*/, v235 /*v1003*/, v244 /*v1012*/
	v_cndmask_b32_e32 v38 /*v294*/, 0xff800000, v48 /*v816*/, vcc_lo
	s_set_vgpr_msb 0x7f01
	v_cmp_le_i32_e32 vcc_lo, v47 /*v303*/, v134
	s_set_vgpr_msb 0x17f
	v_max3_num_f32 v47 /*v303*/, v187 /*v955*/, v188 /*v956*/, v189 /*v957*/
	v_cndmask_b32_e32 v39 /*v295*/, 0xff800000, v49 /*v817*/, vcc_lo
	s_set_vgpr_msb 0x7f01
	v_cmp_le_i32_e32 vcc_lo, v48 /*v304*/, v134
	s_set_vgpr_msb 0x17f
	v_max3_num_f32 v48 /*v304*/, v245 /*v1013*/, v238 /*v1006*/, v239 /*v1007*/
	s_set_vgpr_msb 0x7f95
	v_max3_num_f32 v252 /*v764*/, v19 /*v275*/, v38 /*v294*/, v39 /*v295*/
	s_set_vgpr_msb 0x954c
	v_cndmask_b32_e32 v28 /*v284*/, 0xff800000, v50 /*v818*/, vcc_lo
	s_set_vgpr_msb 0x4c01
	v_cmp_le_i32_e32 vcc_lo, v49 /*v305*/, v134
	s_set_vgpr_msb 0x17f
	v_max3_num_f32 v49 /*v305*/, v190 /*v958*/, v191 /*v959*/, v192 /*v960*/
	v_cndmask_b32_e32 v29 /*v285*/, 0xff800000, v51 /*v819*/, vcc_lo
	s_set_vgpr_msb 0x7f01
	v_cmp_le_i32_e32 vcc_lo, v50 /*v306*/, v134
	s_set_vgpr_msb 0x170
	v_max3_num_f32 v50 /*v306*/, v198, v199, v246 /*v1014*/
	s_set_vgpr_msb 0x704c
	v_cndmask_b32_e32 v20 /*v276*/, 0xff800000, v52 /*v820*/, vcc_lo
	s_set_vgpr_msb 0x4c01
	v_cmp_le_i32_e32 vcc_lo, v51 /*v307*/, v134
	s_set_vgpr_msb 0x17f
	v_max3_num_f32 v51 /*v307*/, v193 /*v961*/, v168 /*v936*/, v169 /*v937*/
	v_cndmask_b32_e32 v21 /*v277*/, 0xff800000, v53 /*v821*/, vcc_lo
	s_set_vgpr_msb 0x7f01
	v_cmp_le_i32_e32 vcc_lo, v52 /*v308*/, v134
	s_set_vgpr_msb 0x17f
	v_max3_num_f32 v52 /*v308*/, v247 /*v1015*/, v240 /*v1008*/, v241 /*v1009*/
	s_set_vgpr_msb 0x7f55
	v_max3_num_f32 v43 /*v299*/, v47 /*v303*/, v49 /*v305*/, v51 /*v307*/
	s_set_vgpr_msb 0x556a
	v_max3_num_f32 v49 /*v305*/, v195 /*v707*/, v197 /*v709*/, v199 /*v711*/
	s_set_vgpr_msb 0x6aae
	v_max3_num_f32 v199 /*v711*/, v249 /*v761*/, v216 /*v984*/, v251 /*v763*/
	s_set_vgpr_msb 0xae4c
	v_cndmask_b32_e32 v34 /*v290*/, 0xff800000, v54 /*v822*/, vcc_lo
	s_set_vgpr_msb 0x4c01
	v_cmp_le_i32_e32 vcc_lo, v53 /*v309*/, v134
	s_set_vgpr_msb 0x17f
	v_max3_num_f32 v53 /*v309*/, v170 /*v938*/, v171 /*v939*/, v172 /*v940*/
	s_set_vgpr_msb 0x7f55
	v_max3_num_f32 v44 /*v300*/, v48 /*v304*/, v50 /*v306*/, v52 /*v308*/
	s_set_vgpr_msb 0x556a
	v_max3_num_f32 v50 /*v306*/, v196 /*v708*/, v198 /*v710*/, v200 /*v712*/
	s_set_vgpr_msb 0x6abe
	v_max3_num_f32 v200 /*v712*/, v255 /*v767*/, v1 /*v769*/, v3 /*v771*/
	s_set_vgpr_msb 0xbe4c
	v_cndmask_b32_e32 v35 /*v291*/, 0xff800000, v55 /*v823*/, vcc_lo
	s_set_vgpr_msb 0x4c01
	v_cmp_le_i32_e32 vcc_lo, v54 /*v310*/, v134
	s_set_vgpr_msb 0x16a
	v_max3_num_f32 v51 /*v307*/, v201 /*v713*/, v203 /*v715*/, v205 /*v717*/
	s_set_vgpr_msb 0x6aae
	v_max3_num_f32 v197 /*v709*/, v231 /*v743*/, v197 /*v965*/, v233 /*v745*/
	s_set_vgpr_msb 0xaeaa
	v_max3_num_f32 v201 /*v713*/, v237 /*v749*/, v239 /*v751*/, v241 /*v753*/
	s_set_vgpr_msb 0xaad5
	v_max3_num_f32 v0 /*v768*/, v21 /*v277*/, v34 /*v290*/, v35 /*v291*/
	s_set_vgpr_msb 0xd54c
	v_cndmask_b32_e32 v30 /*v286*/, 0xff800000, v56 /*v824*/, vcc_lo
	s_set_vgpr_msb 0x4c01
	v_cmp_le_i32_e32 vcc_lo, v55 /*v311*/, v134
	s_set_vgpr_msb 0x17f
	v_max3_num_f32 v55 /*v311*/, v173 /*v941*/, v174 /*v942*/, v175 /*v943*/
	s_set_vgpr_msb 0x7faa
	v_max3_num_f32 v199 /*v711*/, v199 /*v711*/, v253 /*v765*/, v200 /*v712*/
	s_set_vgpr_msb 0xaa7f
	v_max3_num_f32 v54 /*v310*/, v254 /*v1022*/, v255 /*v1023*/, v248 /*v1016*/
	s_set_vgpr_msb 0x7f95
	v_max3_num_f32 v254 /*v766*/, v28 /*v284*/, v29 /*v285*/, v20 /*v276*/
	s_set_vgpr_msb 0x954c
	v_cndmask_b32_e32 v31 /*v287*/, 0xff800000, v57 /*v825*/, vcc_lo
	v_cmp_ge_i32_e32 vcc_lo, v134, v72 /*v840*/
	s_set_vgpr_msb 0x4c65
	v_max3_num_f32 v47 /*v303*/, v53 /*v309*/, v55 /*v311*/, v193 /*v705*/
	s_set_vgpr_msb 0x656a
	v_max3_num_f32 v53 /*v309*/, v207 /*v719*/, v209 /*v721*/, v211 /*v723*/
	v_max3_num_f32 v52 /*v308*/, v202 /*v714*/, v204 /*v716*/, v206 /*v718*/
	v_max3_num_f32 v55 /*v311*/, v213 /*v725*/, v215 /*v727*/, v217 /*v729*/
	s_set_vgpr_msb 0x6a4c
	v_cndmask_b32_e32 v40 /*v296*/, 0xff800000, v58 /*v826*/, vcc_lo
	v_cmp_ge_i32_e32 vcc_lo, v134, v64 /*v832*/
	s_set_vgpr_msb 0x4caa
	v_max3_num_f32 v193 /*v705*/, v219 /*v731*/, v221 /*v733*/, v223 /*v735*/
	v_max3_num_f32 v195 /*v707*/, v225 /*v737*/, v227 /*v739*/, v229 /*v741*/
	v_max3_num_f32 v203 /*v715*/, v243 /*v755*/, v245 /*v757*/, v247 /*v759*/
	s_set_vgpr_msb 0xaaa6
	v_max3_num_f32 v204 /*v716*/, v250 /*v762*/, v18 /*v274*/, v252 /*v764*/
	s_set_vgpr_msb 0xa64c
	v_cndmask_b32_e32 v41 /*v297*/, 0xff800000, v59 /*v827*/, vcc_lo
	v_cmp_ge_i32_e32 vcc_lo, v134, v65 /*v833*/
	s_set_vgpr_msb 0x4cd5
	v_max3_num_f32 v2 /*v770*/, v30 /*v286*/, v31 /*v287*/, v40 /*v296*/
	s_set_vgpr_msb 0xd514
	v_max3_num_f32 v130, v130, v45 /*v301*/, v43 /*v299*/
	s_set_vgpr_msb 0x1455
	v_max3_num_f32 v43 /*v299*/, v49 /*v305*/, v51 /*v307*/, v53 /*v309*/
	s_set_vgpr_msb 0x556a
	v_max3_num_f32 v45 /*v301*/, v197 /*v709*/, v235 /*v747*/, v201 /*v713*/
	s_set_vgpr_msb 0x6a4c
	v_cndmask_b32_e32 v36 /*v292*/, 0xff800000, v60 /*v828*/, vcc_lo
	v_cmp_ge_i32_e32 vcc_lo, v134, v66 /*v834*/
	s_set_vgpr_msb 0x4c7e
	v_max3_num_f32 v49 /*v305*/, v199 /*v711*/, v242 /*v1010*/, v243 /*v1011*/
	s_set_vgpr_msb 0x7e69
	v_max3_num_f32 v48 /*v304*/, v54 /*v310*/, v192 /*v704*/, v194 /*v706*/
	s_set_vgpr_msb 0x696a
	v_max3_num_f32 v54 /*v310*/, v208 /*v720*/, v210 /*v722*/, v212 /*v724*/
	s_set_vgpr_msb 0x6aa6
	v_max3_num_f32 v198 /*v710*/, v232 /*v744*/, v13 /*v269*/, v234 /*v746*/
	s_set_vgpr_msb 0xa64c
	v_cndmask_b32_e32 v37 /*v293*/, 0xff800000, v61 /*v829*/, vcc_lo
	v_cmp_ge_i32_e32 vcc_lo, v134, v67 /*v835*/
	s_set_vgpr_msb 0x4caa
	v_max3_num_f32 v202 /*v714*/, v238 /*v750*/, v240 /*v752*/, v242 /*v754*/
	s_set_vgpr_msb 0xaa69
	v_max3_num_f32 v53 /*v309*/, v55 /*v311*/, v193 /*v705*/, v195 /*v707*/
	s_set_vgpr_msb 0x6914
	v_max3_num_f32 v130, v130, v47 /*v303*/, v43 /*v299*/
	s_set_vgpr_msb 0x14d5
	v_max3_num_f32 v4 /*v772*/, v41 /*v297*/, v36 /*v292*/, v37 /*v293*/
	s_set_vgpr_msb 0xd54c
	v_cndmask_b32_e32 v22 /*v278*/, 0xff800000, v62 /*v830*/, vcc_lo
	v_cmp_ge_i32_e32 vcc_lo, v134, v68 /*v836*/
	s_set_vgpr_msb 0x4c59
	v_max3_num_f32 v43 /*v299*/, v45 /*v301*/, v203 /*v715*/, v49 /*v305*/
	s_set_vgpr_msb 0x596a
	v_max3_num_f32 v51 /*v307*/, v244 /*v756*/, v246 /*v758*/, v248 /*v760*/
	s_set_vgpr_msb 0x6abf
	v_max3_num_f32 v200 /*v712*/, v0 /*v768*/, v2 /*v770*/, v4 /*v772*/
	s_set_vgpr_msb 0xbf55
	v_max3_num_f32 v42 /*v298*/, v42 /*v298*/, v46 /*v302*/, v44 /*v300*/
	s_set_vgpr_msb 0x554c
	v_cndmask_b32_e32 v23 /*v279*/, 0xff800000, v63 /*v831*/, vcc_lo
	s_set_vgpr_msb 0x4c55
	v_max3_num_f32 v44 /*v300*/, v50 /*v306*/, v52 /*v308*/, v54 /*v310*/
	s_set_vgpr_msb 0x556a
	v_max3_num_f32 v45 /*v301*/, v198 /*v710*/, v236 /*v748*/, v202 /*v714*/
	v_max3_num_f32 v55 /*v311*/, v204 /*v716*/, v254 /*v766*/, v200 /*v712*/
	s_set_vgpr_msb 0x6a14
	v_max3_num_f32 v130, v130, v53 /*v309*/, v43 /*v299*/
	s_set_vgpr_msb 0x14aa
	v_max3_num_f32 v192 /*v704*/, v214 /*v726*/, v216 /*v728*/, v218 /*v730*/
	s_set_vgpr_msb 0xaa55
	v_max3_num_f32 v42 /*v298*/, v42 /*v298*/, v48 /*v304*/, v44 /*v300*/
	s_set_vgpr_msb 0x55aa
	v_max3_num_f32 v194 /*v706*/, v220 /*v732*/, v222 /*v734*/, v224 /*v736*/
	s_set_vgpr_msb 0xaa55
	v_max3_num_f32 v46 /*v302*/, v55 /*v311*/, v22 /*v278*/, v23 /*v279*/
	s_set_vgpr_msb 0x55aa
	v_max3_num_f32 v196 /*v708*/, v226 /*v738*/, v228 /*v740*/, v230 /*v742*/
	s_set_vgpr_msb 0xaa55
	v_max3_num_f32 v44 /*v300*/, v45 /*v301*/, v51 /*v307*/, v46 /*v302*/
	s_set_vgpr_msb 0x5540
	v_mov_b32_e32 v45 /*v301*/, v130
	s_set_vgpr_msb 0x406a
	v_max3_num_f32 v43 /*v299*/, v192 /*v704*/, v194 /*v706*/, v196 /*v708*/
	s_set_vgpr_msb 0x6a55
	v_permlanex16_b32 v45 /*v301*/, v45 /*v301*/, s47, 0xfedcba98
	v_max3_num_f32 v42 /*v298*/, v42 /*v298*/, v43 /*v299*/, v44 /*v300*/
	s_set_vgpr_msb 0x5504
	v_max_num_f32_e32 v130, v130, v45 /*v301*/
	s_set_vgpr_msb 0x441
	v_mov_b32_e32 v43 /*v299*/, v42 /*v298*/
	s_set_vgpr_msb 0x4140
	v_sub_f32_e32 v44 /*v300*/, v130, v132
	s_set_vgpr_msb 0x4000
	v_max_num_f32_e32 v130, v132, v130
	s_set_vgpr_msb 0x45
	v_permlanex16_b32 v43 /*v299*/, v43 /*v299*/, s47, 0xfedcba98
	v_cmp_lt_f32_e32 vcc_lo, 0x41000000, v44 /*v300*/
	v_max_num_f32_e32 v42 /*v298*/, v42 /*v298*/, v43 /*v299*/
	s_cmp_eq_u32 vcc_lo, 0
	s_cselect_b32 s2, -1, 0
	s_set_vgpr_msb 0x4541
	v_sub_f32_e32 v43 /*v299*/, v42 /*v298*/, v131
	s_set_vgpr_msb 0x4140
	v_cndmask_b32_e64 v50 /*v306*/, v130, v132, s2
	s_set_vgpr_msb 0x4044
	v_max_num_f32_e32 v42 /*v298*/, v131, v42 /*v298*/
	v_cmp_lt_f32_e64 s2, 0x41000000, v43 /*v299*/
	s_set_vgpr_msb 0x4404
	v_mul_f32_e32 v130, 0xbfb8aa3b, v50 /*v306*/
	s_cmp_lg_u32 s2, 0
	s_set_vgpr_msb 0x443
	v_pk_fma_f32 v[44:45] /*v[300:301]*/, v[182:183] /*v[950:951]*/, s[46:47], v[130:131] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[46:47] /*v[302:303]*/, v[184:185] /*v[952:953]*/, s[46:47], v[130:131] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[48:49] /*v[304:305]*/, v[186:187] /*v[954:955]*/, s[46:47], v[130:131] op_sel_hi:[1,0,0]
	s_cselect_b32 s5, -1, 0
	s_cmp_eq_u32 s2, 0
	s_set_vgpr_msb 0x4381
	v_exp_f32_e32 v202 /*v714*/, v44 /*v300*/
	v_exp_f32_e32 v208 /*v720*/, v45 /*v301*/
	v_nop
	s_set_vgpr_msb 0x8143
	v_pk_fma_f32 v[44:45] /*v[300:301]*/, v[188:189] /*v[956:957]*/, s[46:47], v[130:131] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x4381
	v_exp_f32_e32 v210 /*v722*/, v46 /*v302*/
	v_exp_f32_e32 v218 /*v730*/, v47 /*v303*/
	v_nop
	s_set_vgpr_msb 0x8143
	v_pk_fma_f32 v[46:47] /*v[302:303]*/, v[190:191] /*v[958:959]*/, s[46:47], v[130:131] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x4381
	v_exp_f32_e32 v222 /*v734*/, v48 /*v304*/
	v_exp_f32_e32 v236 /*v748*/, v44 /*v300*/
	v_exp_f32_e32 v248 /*v760*/, v45 /*v301*/
	v_nop
	s_set_vgpr_msb 0x8143
	v_pk_fma_f32 v[44:45] /*v[300:301]*/, v[168:169] /*v[936:937]*/, s[46:47], v[130:131] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x4381
	v_exp_f32_e32 v232 /*v744*/, v49 /*v305*/
	v_nop
	s_set_vgpr_msb 0x8143
	v_pk_fma_f32 v[48:49] /*v[304:305]*/, v[192:193] /*v[960:961]*/, s[46:47], v[130:131] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x4381
	v_exp_f32_e32 v252 /*v764*/, v46 /*v302*/
	s_set_vgpr_msb 0x81c1
	v_exp_f32_e32 v10 /*v778*/, v47 /*v303*/
	s_set_vgpr_msb 0xc181
	v_exp_f32_e32 v198 /*v710*/, v44 /*v300*/
	v_exp_f32_e32 v204 /*v716*/, v45 /*v301*/
	v_nop
	s_set_vgpr_msb 0x8143
	v_pk_fma_f32 v[44:45] /*v[300:301]*/, v[174:175] /*v[942:943]*/, s[46:47], v[130:131] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[46:47] /*v[302:303]*/, v[170:171] /*v[938:939]*/, s[46:47], v[130:131] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x43c1
	v_exp_f32_e32 v16 /*v784*/, v48 /*v304*/
	v_exp_f32_e32 v32 /*v800*/, v49 /*v305*/
	v_nop
	s_set_vgpr_msb 0xc143
	v_pk_fma_f32 v[48:49] /*v[304:305]*/, v[172:173] /*v[940:941]*/, s[46:47], v[130:131] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x4381
	v_exp_f32_e32 v228 /*v740*/, v44 /*v300*/
	v_exp_f32_e32 v240 /*v752*/, v45 /*v301*/
	v_nop
	s_set_vgpr_msb 0x8143
	v_pk_fma_f32 v[44:45] /*v[300:301]*/, v[164:165] /*v[932:933]*/, s[46:47], v[130:131] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x4381
	v_exp_f32_e32 v206 /*v718*/, v46 /*v302*/
	v_exp_f32_e32 v214 /*v726*/, v47 /*v303*/
	v_nop
	s_set_vgpr_msb 0x8143
	v_pk_fma_f32 v[46:47] /*v[302:303]*/, v[160:161] /*v[928:929]*/, s[46:47], v[130:131] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x4381
	v_exp_f32_e32 v216 /*v728*/, v48 /*v304*/
	s_set_vgpr_msb 0x81c1
	v_exp_f32_e32 v26 /*v794*/, v44 /*v300*/
	v_exp_f32_e32 v42 /*v810*/, v45 /*v301*/
	v_nop
	s_set_vgpr_msb 0xc143
	v_pk_fma_f32 v[44:45] /*v[300:301]*/, v[154:155] /*v[922:923]*/, s[46:47], v[130:131] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x4381
	v_exp_f32_e32 v226 /*v738*/, v49 /*v305*/
	v_nop
	s_set_vgpr_msb 0x8143
	v_pk_fma_f32 v[48:49] /*v[304:305]*/, v[162:163] /*v[930:931]*/, s[46:47], v[130:131] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x4381
	v_exp_f32_e32 v244 /*v756*/, v46 /*v302*/
	s_set_vgpr_msb 0x81c1
	v_exp_f32_e32 v2 /*v770*/, v47 /*v303*/
	v_nop
	s_set_vgpr_msb 0xc143
	v_pk_fma_f32 v[46:47] /*v[302:303]*/, v[166:167] /*v[934:935]*/, s[46:47], v[130:131] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x4381
	v_exp_f32_e32 v224 /*v736*/, v44 /*v300*/
	v_exp_f32_e32 v234 /*v746*/, v45 /*v301*/
	v_nop
	s_set_vgpr_msb 0x8143
	v_pk_fma_f32 v[44:45] /*v[300:301]*/, v[144:145] /*v[912:913]*/, s[46:47], v[130:131] op_sel_hi:[1,0,0]
	s_cselect_b32 s2, -1, 0
	s_set_vgpr_msb 0x43c1
	v_exp_f32_e32 v6 /*v774*/, v48 /*v304*/
	s_set_vgpr_msb 0xc141
	v_cndmask_b32_e64 v51 /*v307*/, v42 /*v298*/, v131, s2
	s_set_vgpr_msb 0x41c1
	v_exp_f32_e32 v22 /*v790*/, v49 /*v305*/
	v_nop
	s_set_vgpr_msb 0xc143
	v_pk_fma_f32 v[48:49] /*v[304:305]*/, v[152:153] /*v[920:921]*/, s[46:47], v[130:131] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x43c1
	v_exp_f32_e32 v48 /*v816*/, v46 /*v302*/
	v_exp_f32_e32 v64 /*v832*/, v47 /*v303*/
	v_nop
	s_set_vgpr_msb 0xc143
	v_pk_fma_f32 v[46:47] /*v[302:303]*/, v[156:157] /*v[924:925]*/, s[46:47], v[130:131] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x43c1
	v_exp_f32_e32 v18 /*v786*/, v44 /*v300*/
	v_exp_f32_e32 v34 /*v802*/, v45 /*v301*/
	v_nop
	s_set_vgpr_msb 0xc147
	v_pk_fma_f32 v[44:45] /*v[300:301]*/, v[150:151] /*v[918:919]*/, s[46:47], v[130:131] op_sel_hi:[1,0,0]
	v_mul_f32_e32 v42 /*v298*/, 0xbfb8aa3b, v51 /*v307*/
	s_set_vgpr_msb 0x4781
	v_exp_f32_e32 v212 /*v724*/, v48 /*v304*/
	v_exp_f32_e32 v220 /*v732*/, v49 /*v305*/
	v_nop
	s_set_vgpr_msb 0x8143
	v_pk_fma_f32 v[48:49] /*v[304:305]*/, v[158:159] /*v[926:927]*/, s[46:47], v[130:131] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x4381
	v_exp_f32_e32 v238 /*v750*/, v46 /*v302*/
	v_exp_f32_e32 v250 /*v762*/, v47 /*v303*/
	v_nop
	s_set_vgpr_msb 0x8143
	v_pk_fma_f32 v[46:47] /*v[302:303]*/, v[146:147] /*v[914:915]*/, s[46:47], v[130:131] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x43c1
	v_exp_f32_e32 v80 /*v848*/, v44 /*v300*/
	v_exp_f32_e32 v96 /*v864*/, v45 /*v301*/
	v_nop
	s_set_vgpr_msb 0xc143
	v_pk_fma_f32 v[44:45] /*v[300:301]*/, v[140:141] /*v[908:909]*/, s[46:47], v[130:131] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x4310
	v_pk_fma_f32 v[192:193], v[192:193], s[46:47], v[42:43] /*v[298:299]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1081
	v_exp_f32_e32 v254 /*v766*/, v48 /*v304*/
	s_set_vgpr_msb 0x81c1
	v_exp_f32_e32 v12 /*v780*/, v49 /*v305*/
	v_nop
	s_set_vgpr_msb 0xc143
	v_pk_fma_f32 v[48:49] /*v[304:305]*/, v[148:149] /*v[916:917]*/, s[46:47], v[130:131] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x43c1
	v_exp_f32_e32 v38 /*v806*/, v46 /*v302*/
	v_exp_f32_e32 v54 /*v822*/, v47 /*v303*/
	v_nop
	s_set_vgpr_msb 0xc143
	v_pk_fma_f32 v[46:47] /*v[302:303]*/, v[136:137] /*v[904:905]*/, s[46:47], v[130:131] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x43c1
	v_exp_f32_e32 v8 /*v776*/, v44 /*v300*/
	v_exp_f32_e32 v24 /*v792*/, v45 /*v301*/
	v_nop
	s_set_vgpr_msb 0xc143
	v_pk_fma_f32 v[44:45] /*v[300:301]*/, v[130:131] /*v[898:899]*/, s[46:47], v[130:131] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x4380
	v_exp_f32_e32 v245 /*v757*/, v192
	s_set_vgpr_msb 0x80c0
	v_exp_f32_e32 v3 /*v771*/, v193
	v_nop
	s_set_vgpr_msb 0xc010
	v_pk_fma_f32 v[192:193], v[194:195], s[46:47], v[42:43] /*v[298:299]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[194:195], v[218:219], s[46:47], v[42:43] /*v[298:299]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x10c1
	v_exp_f32_e32 v58 /*v826*/, v48 /*v304*/
	v_exp_f32_e32 v74 /*v842*/, v49 /*v305*/
	v_nop
	s_set_vgpr_msb 0xc143
	v_pk_fma_f32 v[48:49] /*v[304:305]*/, v[138:139] /*v[906:907]*/, s[46:47], v[130:131] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x4381
	v_exp_f32_e32 v230 /*v742*/, v46 /*v302*/
	v_exp_f32_e32 v242 /*v754*/, v47 /*v303*/
	v_nop
	s_set_vgpr_msb 0x8143
	v_pk_fma_f32 v[46:47] /*v[302:303]*/, v[142:143] /*v[910:911]*/, s[46:47], v[130:131] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x43c1
	v_exp_f32_e32 v70 /*v838*/, v44 /*v300*/
	v_exp_f32_e32 v86 /*v854*/, v45 /*v301*/
	v_nop
	s_set_vgpr_msb 0xc143
	v_pk_fma_f32 v[44:45] /*v[300:301]*/, v[120:121] /*v[888:889]*/, s[46:47], v[130:131] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x43c0
	v_exp_f32_e32 v49 /*v817*/, v192
	v_exp_f32_e32 v65 /*v833*/, v193
	s_set_vgpr_msb 0xc080
	v_exp_f32_e32 v213 /*v725*/, v194
	s_set_vgpr_msb 0x8010
	v_pk_fma_f32 v[192:193], v[196:197], s[46:47], v[42:43] /*v[298:299]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1080
	v_exp_f32_e32 v221 /*v733*/, v195
	v_nop
	s_set_vgpr_msb 0x8010
	v_pk_fma_f32 v[194:195], v[210:211], s[46:47], v[42:43] /*v[298:299]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[196:197], v[204:205], s[46:47], v[42:43] /*v[298:299]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1081
	v_exp_f32_e32 v246 /*v758*/, v48 /*v304*/
	s_set_vgpr_msb 0x81c1
	v_exp_f32_e32 v4 /*v772*/, v49 /*v305*/
	v_nop
	s_set_vgpr_msb 0xc143
	v_pk_fma_f32 v[48:49] /*v[304:305]*/, v[128:129] /*v[896:897]*/, s[46:47], v[130:131] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x43c1
	v_exp_f32_e32 v28 /*v796*/, v46 /*v302*/
	v_exp_f32_e32 v44 /*v812*/, v47 /*v303*/
	v_nop
	s_set_vgpr_msb 0xc143
	v_pk_fma_f32 v[46:47] /*v[302:303]*/, v[132:133] /*v[900:901]*/, s[46:47], v[130:131] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x43c1
	v_exp_f32_e32 v0 /*v768*/, v44 /*v300*/
	v_exp_f32_e32 v14 /*v782*/, v45 /*v301*/
	v_nop
	s_set_vgpr_msb 0xc143
	v_pk_fma_f32 v[44:45] /*v[300:301]*/, v[176:177] /*v[944:945]*/, s[46:47], v[130:131] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x4380
	v_exp_f32_e32 v239 /*v751*/, v192
	v_exp_f32_e32 v251 /*v763*/, v193
	v_exp_f32_e32 v255 /*v767*/, v194
	s_set_vgpr_msb 0x8010
	v_pk_fma_f32 v[192:193], v[228:229], s[46:47], v[42:43] /*v[298:299]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x10c0
	v_exp_f32_e32 v13 /*v781*/, v195
	v_exp_f32_e32 v19 /*v787*/, v196
	s_set_vgpr_msb 0xc010
	v_pk_fma_f32 v[194:195], v[212:213], s[46:47], v[42:43] /*v[298:299]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x10c0
	v_exp_f32_e32 v35 /*v803*/, v197
	v_nop
	s_set_vgpr_msb 0xc010
	v_pk_fma_f32 v[196:197], v[206:207], s[46:47], v[42:43] /*v[298:299]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x10c1
	v_exp_f32_e32 v50 /*v818*/, v48 /*v304*/
	v_exp_f32_e32 v66 /*v834*/, v49 /*v305*/
	v_nop
	s_set_vgpr_msb 0xc143
	v_pk_fma_f32 v[48:49] /*v[304:305]*/, v[134:135] /*v[902:903]*/, s[46:47], v[130:131] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x43c1
	v_exp_f32_e32 v90 /*v858*/, v46 /*v302*/
	v_exp_f32_e32 v106 /*v874*/, v47 /*v303*/
	v_nop
	s_set_vgpr_msb 0xc143
	v_pk_fma_f32 v[46:47] /*v[302:303]*/, v[122:123] /*v[890:891]*/, s[46:47], v[130:131] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x43c1
	v_exp_f32_e32 v60 /*v828*/, v44 /*v300*/
	v_exp_f32_e32 v76 /*v844*/, v45 /*v301*/
	v_nop
	s_set_vgpr_msb 0xc143
	v_pk_fma_f32 v[44:45] /*v[300:301]*/, v[116:117] /*v[884:885]*/, s[46:47], v[130:131] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x43c0
	v_exp_f32_e32 v39 /*v807*/, v192
	v_exp_f32_e32 v55 /*v823*/, v193
	v_exp_f32_e32 v59 /*v827*/, v194
	s_set_vgpr_msb 0xc010
	v_pk_fma_f32 v[192:193], v[220:221], s[46:47], v[42:43] /*v[298:299]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x10c0
	v_exp_f32_e32 v75 /*v843*/, v195
	v_exp_f32_e32 v81 /*v849*/, v196
	s_set_vgpr_msb 0xc010
	v_pk_fma_f32 v[194:195], v[214:215], s[46:47], v[42:43] /*v[298:299]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x10c0
	v_exp_f32_e32 v97 /*v865*/, v197
	v_nop
	s_set_vgpr_msb 0xc010
	v_pk_fma_f32 v[196:197], v[238:239], s[46:47], v[42:43] /*v[298:299]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x10c1
	v_exp_f32_e32 v112 /*v880*/, v48 /*v304*/
	v_exp_f32_e32 v126 /*v894*/, v49 /*v305*/
	v_nop
	s_set_vgpr_msb 0xc143
	v_pk_fma_f32 v[48:49] /*v[304:305]*/, v[124:125] /*v[892:893]*/, s[46:47], v[130:131] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x43c1
	v_exp_f32_e32 v20 /*v788*/, v46 /*v302*/
	v_exp_f32_e32 v36 /*v804*/, v47 /*v303*/
	v_nop
	s_set_vgpr_msb 0xc143
	v_pk_fma_f32 v[46:47] /*v[302:303]*/, v[194:195] /*v[962:963]*/, s[46:47], v[130:131] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x43c1
	v_exp_f32_e32 v122 /*v890*/, v44 /*v300*/
	v_exp_f32_e32 v136 /*v904*/, v45 /*v301*/
	v_nop
	s_set_vgpr_msb 0xc143
	v_pk_fma_f32 v[44:45] /*v[300:301]*/, v[196:197] /*v[964:965]*/, s[46:47], v[130:131] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x4380
	v_exp_f32_e32 v231 /*v743*/, v192
	v_exp_f32_e32 v243 /*v755*/, v193
	v_exp_f32_e32 v247 /*v759*/, v194
	s_set_vgpr_msb 0x8010
	v_pk_fma_f32 v[192:193], v[222:223], s[46:47], v[42:43] /*v[298:299]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x10c0
	v_exp_f32_e32 v5 /*v773*/, v195
	v_exp_f32_e32 v9 /*v777*/, v196
	s_set_vgpr_msb 0xc010
	v_pk_fma_f32 v[194:195], v[216:217], s[46:47], v[42:43] /*v[298:299]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x10c0
	v_exp_f32_e32 v25 /*v793*/, v197
	v_nop
	s_set_vgpr_msb 0xc010
	v_pk_fma_f32 v[196:197], v[230:231], s[46:47], v[42:43] /*v[298:299]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x10c1
	v_exp_f32_e32 v40 /*v808*/, v48 /*v304*/
	v_exp_f32_e32 v56 /*v824*/, v49 /*v305*/
	v_nop
	s_set_vgpr_msb 0xc143
	v_pk_fma_f32 v[48:49] /*v[304:305]*/, v[114:115] /*v[882:883]*/, s[46:47], v[130:131] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x43c1
	v_exp_f32_e32 v82 /*v850*/, v46 /*v302*/
	v_exp_f32_e32 v98 /*v866*/, v47 /*v303*/
	v_nop
	s_set_vgpr_msb 0xc143
	v_pk_fma_f32 v[46:47] /*v[302:303]*/, v[118:119] /*v[886:887]*/, s[46:47], v[130:131] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x43c1
	v_exp_f32_e32 v52 /*v820*/, v44 /*v300*/
	v_exp_f32_e32 v68 /*v836*/, v45 /*v301*/
	v_nop
	s_set_vgpr_msb 0xc143
	v_pk_fma_f32 v[44:45] /*v[300:301]*/, v[198:199] /*v[966:967]*/, s[46:47], v[130:131] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x43c0
	v_exp_f32_e32 v29 /*v797*/, v192
	v_exp_f32_e32 v45 /*v813*/, v193
	v_exp_f32_e32 v51 /*v819*/, v194
	s_set_vgpr_msb 0xc010
	v_pk_fma_f32 v[192:193], v[224:225], s[46:47], v[42:43] /*v[298:299]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x10c0
	v_exp_f32_e32 v67 /*v835*/, v195
	v_exp_f32_e32 v71 /*v839*/, v196
	s_set_vgpr_msb 0xc010
	v_pk_fma_f32 v[194:195], v[248:249], s[46:47], v[42:43] /*v[298:299]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x10c0
	v_exp_f32_e32 v87 /*v855*/, v197
	v_nop
	s_set_vgpr_msb 0xc010
	v_pk_fma_f32 v[196:197], v[232:233], s[46:47], v[42:43] /*v[298:299]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x10c1
	v_exp_f32_e32 v102 /*v870*/, v48 /*v304*/
	v_exp_f32_e32 v118 /*v886*/, v49 /*v305*/
	v_nop
	s_set_vgpr_msb 0xc143
	v_pk_fma_f32 v[48:49] /*v[304:305]*/, v[104:105] /*v[872:873]*/, s[46:47], v[130:131] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x43c1
	v_exp_f32_e32 v140 /*v908*/, v46 /*v302*/
	v_exp_f32_e32 v152 /*v920*/, v47 /*v303*/
	v_nop
	s_set_vgpr_msb 0xc143
	v_pk_fma_f32 v[46:47] /*v[302:303]*/, v[108:109] /*v[876:877]*/, s[46:47], v[130:131] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x43c1
	v_exp_f32_e32 v114 /*v882*/, v44 /*v300*/
	v_exp_f32_e32 v128 /*v896*/, v45 /*v301*/
	v_nop
	s_set_vgpr_msb 0xc143
	v_pk_fma_f32 v[44:45] /*v[300:301]*/, v[202:203] /*v[970:971]*/, s[46:47], v[130:131] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x43c0
	v_exp_f32_e32 v91 /*v859*/, v192
	v_exp_f32_e32 v107 /*v875*/, v193
	v_exp_f32_e32 v113 /*v881*/, v194
	s_set_vgpr_msb 0xc010
	v_pk_fma_f32 v[192:193], v[226:227], s[46:47], v[42:43] /*v[298:299]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x10c0
	v_exp_f32_e32 v127 /*v895*/, v195
	v_exp_f32_e32 v1 /*v769*/, v196
	s_set_vgpr_msb 0xc010
	v_pk_fma_f32 v[194:195], v[240:241], s[46:47], v[42:43] /*v[298:299]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x10c0
	v_exp_f32_e32 v15 /*v783*/, v197
	v_nop
	s_set_vgpr_msb 0xc010
	v_pk_fma_f32 v[196:197], v[234:235], s[46:47], v[42:43] /*v[298:299]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x10c1
	v_exp_f32_e32 v30 /*v798*/, v48 /*v304*/
	v_exp_f32_e32 v46 /*v814*/, v49 /*v305*/
	v_nop
	s_set_vgpr_msb 0xc143
	v_pk_fma_f32 v[48:49] /*v[304:305]*/, v[110:111] /*v[878:879]*/, s[46:47], v[130:131] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x43c1
	v_exp_f32_e32 v72 /*v840*/, v46 /*v302*/
	v_exp_f32_e32 v88 /*v856*/, v47 /*v303*/
	v_nop
	s_set_vgpr_msb 0xc143
	v_pk_fma_f32 v[46:47] /*v[302:303]*/, v[200:201] /*v[968:969]*/, s[46:47], v[130:131] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x43c1
	v_exp_f32_e32 v162 /*v930*/, v44 /*v300*/
	v_exp_f32_e32 v170 /*v938*/, v45 /*v301*/
	v_nop
	s_set_vgpr_msb 0xc143
	v_pk_fma_f32 v[44:45] /*v[300:301]*/, v[208:209] /*v[976:977]*/, s[46:47], v[130:131] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x43c0
	v_exp_f32_e32 v21 /*v789*/, v192
	v_exp_f32_e32 v37 /*v805*/, v193
	v_exp_f32_e32 v41 /*v809*/, v194
	s_set_vgpr_msb 0xc011
	v_pk_fma_f32 v[192:193], v[2:3] /*v[258:259]*/, s[46:47], v[42:43] /*v[298:299]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x11c0
	v_exp_f32_e32 v57 /*v825*/, v195
	v_exp_f32_e32 v61 /*v829*/, v196
	s_set_vgpr_msb 0xc010
	v_pk_fma_f32 v[194:195], v[242:243], s[46:47], v[42:43] /*v[298:299]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x10c0
	v_exp_f32_e32 v77 /*v845*/, v197
	v_nop
	s_set_vgpr_msb 0xc010
	v_pk_fma_f32 v[196:197], v[236:237], s[46:47], v[42:43] /*v[298:299]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x10c1
	v_exp_f32_e32 v92 /*v860*/, v48 /*v304*/
	v_exp_f32_e32 v108 /*v876*/, v49 /*v305*/
	v_nop
	s_set_vgpr_msb 0xc143
	v_pk_fma_f32 v[48:49] /*v[304:305]*/, v[100:101] /*v[868:869]*/, s[46:47], v[130:131] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x43c1
	v_exp_f32_e32 v132 /*v900*/, v46 /*v302*/
	v_exp_f32_e32 v144 /*v912*/, v47 /*v303*/
	v_nop
	s_set_vgpr_msb 0xc143
	v_pk_fma_f32 v[46:47] /*v[302:303]*/, v[204:205] /*v[972:973]*/, s[46:47], v[130:131] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x43c1
	v_exp_f32_e32 v104 /*v872*/, v44 /*v300*/
	v_exp_f32_e32 v120 /*v888*/, v45 /*v301*/
	v_nop
	s_set_vgpr_msb 0xc143
	v_pk_fma_f32 v[44:45] /*v[300:301]*/, v[212:213] /*v[980:981]*/, s[46:47], v[130:131] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x43c0
	v_exp_f32_e32 v83 /*v851*/, v192
	v_exp_f32_e32 v99 /*v867*/, v193
	v_exp_f32_e32 v103 /*v871*/, v194
	s_set_vgpr_msb 0xc010
	v_pk_fma_f32 v[192:193], v[250:251], s[46:47], v[42:43] /*v[298:299]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x10c0
	v_exp_f32_e32 v119 /*v887*/, v195
	v_exp_f32_e32 v123 /*v891*/, v196
	s_set_vgpr_msb 0xc010
	v_pk_fma_f32 v[194:195], v[244:245], s[46:47], v[42:43] /*v[298:299]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x10c0
	v_exp_f32_e32 v137 /*v905*/, v197
	v_nop
	s_set_vgpr_msb 0xc011
	v_pk_fma_f32 v[196:197], v[12:13] /*v[268:269]*/, s[46:47], v[42:43] /*v[298:299]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x11c1
	v_exp_f32_e32 v148 /*v916*/, v48 /*v304*/
	v_exp_f32_e32 v158 /*v926*/, v49 /*v305*/
	v_nop
	s_set_vgpr_msb 0xc143
	v_pk_fma_f32 v[48:49] /*v[304:305]*/, v[206:207] /*v[974:975]*/, s[46:47], v[130:131] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x43c1
	v_exp_f32_e32 v62 /*v830*/, v46 /*v302*/
	v_exp_f32_e32 v78 /*v846*/, v47 /*v303*/
	v_nop
	s_set_vgpr_msb 0xc143
	v_pk_fma_f32 v[46:47] /*v[302:303]*/, v[94:95] /*v[862:863]*/, s[46:47], v[130:131] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x43c1
	v_exp_f32_e32 v156 /*v924*/, v44 /*v300*/
	v_exp_f32_e32 v166 /*v934*/, v45 /*v301*/
	v_nop
	s_set_vgpr_msb 0xc143
	v_pk_fma_f32 v[44:45] /*v[300:301]*/, v[218:219] /*v[986:987]*/, s[46:47], v[130:131] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x43c0
	v_exp_f32_e32 v141 /*v909*/, v192
	v_exp_f32_e32 v153 /*v921*/, v193
	v_exp_f32_e32 v31 /*v799*/, v194
	s_set_vgpr_msb 0xc010
	v_pk_fma_f32 v[192:193], v[252:253], s[46:47], v[42:43] /*v[298:299]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x10c0
	v_exp_f32_e32 v47 /*v815*/, v195
	v_exp_f32_e32 v53 /*v821*/, v196
	s_set_vgpr_msb 0xc010
	v_pk_fma_f32 v[194:195], v[246:247], s[46:47], v[42:43] /*v[298:299]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x10c0
	v_exp_f32_e32 v69 /*v837*/, v197
	v_nop
	s_set_vgpr_msb 0xc011
	v_pk_fma_f32 v[196:197], v[4:5] /*v[260:261]*/, s[46:47], v[42:43] /*v[298:299]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x11c1
	v_exp_f32_e32 v84 /*v852*/, v48 /*v304*/
	v_exp_f32_e32 v100 /*v868*/, v49 /*v305*/
	v_nop
	s_set_vgpr_msb 0xc143
	v_pk_fma_f32 v[48:49] /*v[304:305]*/, v[210:211] /*v[978:979]*/, s[46:47], v[130:131] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x43c1
	v_exp_f32_e32 v124 /*v892*/, v46 /*v302*/
	v_exp_f32_e32 v138 /*v906*/, v47 /*v303*/
	v_nop
	s_set_vgpr_msb 0xc143
	v_pk_fma_f32 v[46:47] /*v[302:303]*/, v[214:215] /*v[982:983]*/, s[46:47], v[130:131] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x43c1
	v_exp_f32_e32 v94 /*v862*/, v44 /*v300*/
	v_exp_f32_e32 v110 /*v878*/, v45 /*v301*/
	v_nop
	s_set_vgpr_msb 0xc143
	v_pk_fma_f32 v[44:45] /*v[300:301]*/, v[224:225] /*v[992:993]*/, s[46:47], v[130:131] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x43c0
	v_exp_f32_e32 v73 /*v841*/, v192
	v_exp_f32_e32 v89 /*v857*/, v193
	v_exp_f32_e32 v93 /*v861*/, v194
	s_set_vgpr_msb 0xc010
	v_pk_fma_f32 v[192:193], v[254:255], s[46:47], v[42:43] /*v[298:299]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x10c0
	v_exp_f32_e32 v109 /*v877*/, v195
	v_exp_f32_e32 v115 /*v883*/, v196
	s_set_vgpr_msb 0xc011
	v_pk_fma_f32 v[194:195], v[24:25] /*v[280:281]*/, s[46:47], v[42:43] /*v[298:299]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x11c0
	v_exp_f32_e32 v129 /*v897*/, v197
	v_nop
	s_set_vgpr_msb 0xc011
	v_pk_fma_f32 v[196:197], v[6:7] /*v[262:263]*/, s[46:47], v[42:43] /*v[298:299]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1143
	v_pk_fma_f32 v[52:53] /*v[308:309]*/, v[178:179] /*v[946:947]*/, s[46:47], v[130:131] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x43c1
	v_exp_f32_e32 v142 /*v910*/, v48 /*v304*/
	v_exp_f32_e32 v154 /*v922*/, v49 /*v305*/
	v_nop
	s_set_vgpr_msb 0xc143
	v_pk_fma_f32 v[48:49] /*v[304:305]*/, v[216:217] /*v[984:985]*/, s[46:47], v[130:131] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x43c1
	v_exp_f32_e32 v168 /*v936*/, v46 /*v302*/
	v_exp_f32_e32 v178 /*v946*/, v47 /*v303*/
	v_nop
	s_set_vgpr_msb 0xc143
	v_pk_fma_f32 v[46:47] /*v[302:303]*/, v[220:221] /*v[988:989]*/, s[46:47], v[130:131] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x43c1
	v_exp_f32_e32 v150 /*v918*/, v44 /*v300*/
	v_exp_f32_e32 v160 /*v928*/, v45 /*v301*/
	v_nop
	s_set_vgpr_msb 0xc143
	v_pk_fma_f32 v[44:45] /*v[300:301]*/, v[230:231] /*v[998:999]*/, s[46:47], v[130:131] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x43c0
	v_exp_f32_e32 v133 /*v901*/, v192
	v_exp_f32_e32 v145 /*v913*/, v193
	v_exp_f32_e32 v149 /*v917*/, v194
	s_set_vgpr_msb 0xc011
	v_pk_fma_f32 v[192:193], v[0:1] /*v[256:257]*/, s[46:47], v[42:43] /*v[298:299]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x11c0
	v_exp_f32_e32 v159 /*v927*/, v195
	v_exp_f32_e32 v163 /*v931*/, v196
	s_set_vgpr_msb 0xc011
	v_pk_fma_f32 v[194:195], v[14:15] /*v[270:271]*/, s[46:47], v[42:43] /*v[298:299]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x11c0
	v_exp_f32_e32 v171 /*v939*/, v197
	v_nop
	s_set_vgpr_msb 0xc011
	v_pk_fma_f32 v[196:197], v[8:9] /*v[264:265]*/, s[46:47], v[42:43] /*v[298:299]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1143
	v_pk_fma_f32 v[54:55] /*v[310:311]*/, v[180:181] /*v[948:949]*/, s[46:47], v[130:131] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x43c1
	v_exp_f32_e32 v180 /*v948*/, v48 /*v304*/
	v_exp_f32_e32 v186 /*v954*/, v49 /*v305*/
	v_nop
	s_set_vgpr_msb 0xc143
	v_pk_fma_f32 v[48:49] /*v[304:305]*/, v[222:223] /*v[990:991]*/, s[46:47], v[130:131] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x43c1
	v_exp_f32_e32 v116 /*v884*/, v46 /*v302*/
	v_exp_f32_e32 v130 /*v898*/, v47 /*v303*/
	v_nop
	s_set_vgpr_msb 0xc143
	v_pk_fma_f32 v[46:47] /*v[302:303]*/, v[226:227] /*v[994:995]*/, s[46:47], v[130:131] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x43c1
	v_exp_f32_e32 v184 /*v952*/, v44 /*v300*/
	v_exp_f32_e32 v176 /*v944*/, v45 /*v301*/
	v_nop
	s_set_vgpr_msb 0xc153
	v_pk_fma_f32 v[44:45] /*v[300:301]*/, v[252:253] /*v[1020:1021]*/, s[46:47], v[42:43] /*v[298:299]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x53c0
	v_exp_f32_e32 v63 /*v831*/, v192
	v_exp_f32_e32 v79 /*v847*/, v193
	v_exp_f32_e32 v85 /*v853*/, v194
	s_set_vgpr_msb 0xc011
	v_pk_fma_f32 v[192:193], v[32:33] /*v[288:289]*/, s[46:47], v[42:43] /*v[298:299]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x11c0
	v_exp_f32_e32 v101 /*v869*/, v195
	v_exp_f32_e32 v105 /*v873*/, v196
	s_set_vgpr_msb 0xc011
	v_pk_fma_f32 v[194:195], v[16:17] /*v[272:273]*/, s[46:47], v[42:43] /*v[298:299]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x11c0
	v_exp_f32_e32 v121 /*v889*/, v197
	v_nop
	s_set_vgpr_msb 0xc011
	v_pk_fma_f32 v[196:197], v[10:11] /*v[266:267]*/, s[46:47], v[42:43] /*v[298:299]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x11c1
	v_exp_f32_e32 v134 /*v902*/, v48 /*v304*/
	v_exp_f32_e32 v146 /*v914*/, v49 /*v305*/
	v_nop
	s_set_vgpr_msb 0xc143
	v_pk_fma_f32 v[48:49] /*v[304:305]*/, v[228:229] /*v[996:997]*/, s[46:47], v[130:131] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x43c1
	v_exp_f32_e32 v164 /*v932*/, v46 /*v302*/
	v_exp_f32_e32 v172 /*v940*/, v47 /*v303*/
	v_nop
	s_set_vgpr_msb 0xc143
	v_pk_fma_f32 v[46:47] /*v[302:303]*/, v[242:243] /*v[1010:1011]*/, s[46:47], v[130:131] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x4381
	v_exp_f32_e32 v197 /*v709*/, v44 /*v300*/
	v_exp_f32_e32 v201 /*v713*/, v45 /*v301*/
	v_nop
	s_set_vgpr_msb 0x8153
	v_pk_fma_f32 v[44:45] /*v[300:301]*/, v[244:245] /*v[1012:1013]*/, s[46:47], v[42:43] /*v[298:299]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5310
	v_pk_fma_f32 v[198:199], v[198:199], s[46:47], v[42:43] /*v[298:299]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x10c0
	v_exp_f32_e32 v125 /*v893*/, v192
	v_exp_f32_e32 v139 /*v907*/, v193
	v_exp_f32_e32 v143 /*v911*/, v194
	s_set_vgpr_msb 0xc011
	v_pk_fma_f32 v[192:193], v[26:27] /*v[282:283]*/, s[46:47], v[42:43] /*v[298:299]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x11c0
	v_exp_f32_e32 v155 /*v923*/, v195
	v_exp_f32_e32 v157 /*v925*/, v196
	s_set_vgpr_msb 0xc011
	v_pk_fma_f32 v[194:195], v[18:19] /*v[274:275]*/, s[46:47], v[42:43] /*v[298:299]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x11c0
	v_exp_f32_e32 v167 /*v935*/, v197
	v_nop
	s_set_vgpr_msb 0xc011
	v_pk_fma_f32 v[196:197], v[38:39] /*v[294:295]*/, s[46:47], v[42:43] /*v[298:299]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x11c1
	v_exp_f32_e32 v174 /*v942*/, v48 /*v304*/
	v_exp_f32_e32 v182 /*v950*/, v49 /*v305*/
	v_nop
	s_set_vgpr_msb 0xc153
	v_pk_fma_f32 v[48:49] /*v[304:305]*/, v[232:233] /*v[1000:1001]*/, s[46:47], v[42:43] /*v[298:299]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x53c1
	v_exp_f32_e32 v188 /*v956*/, v46 /*v302*/
	v_exp_f32_e32 v190 /*v958*/, v47 /*v303*/
	v_nop
	s_set_vgpr_msb 0xc153
	v_pk_fma_f32 v[46:47] /*v[302:303]*/, v[236:237] /*v[1004:1005]*/, s[46:47], v[42:43] /*v[298:299]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5381
	v_exp_f32_e32 v223 /*v735*/, v44 /*v300*/
	v_exp_f32_e32 v233 /*v745*/, v45 /*v301*/
	v_nop
	s_set_vgpr_msb 0x8153
	v_pk_fma_f32 v[44:45] /*v[300:301]*/, v[246:247] /*v[1014:1015]*/, s[46:47], v[42:43] /*v[298:299]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5380
	v_exp_f32_e32 v253 /*v765*/, v198
	s_set_vgpr_msb 0x80c0
	v_exp_f32_e32 v11 /*v779*/, v199
	v_nop
	s_set_vgpr_msb 0xc013
	v_pk_fma_f32 v[198:199], v[254:255] /*v[1022:1023]*/, s[46:47], v[42:43] /*v[298:299]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x13c0
	v_exp_f32_e32 v169 /*v937*/, v192
	v_exp_f32_e32 v179 /*v947*/, v193
	v_exp_f32_e32 v181 /*v949*/, v194
	s_set_vgpr_msb 0xc011
	v_pk_fma_f32 v[192:193], v[28:29] /*v[284:285]*/, s[46:47], v[42:43] /*v[298:299]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x11c0
	v_exp_f32_e32 v187 /*v955*/, v195
	v_exp_f32_e32 v95 /*v863*/, v196
	s_set_vgpr_msb 0xc011
	v_pk_fma_f32 v[194:195], v[20:21] /*v[276:277]*/, s[46:47], v[42:43] /*v[298:299]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x11c0
	v_exp_f32_e32 v111 /*v879*/, v197
	v_nop
	s_set_vgpr_msb 0xc011
	v_pk_fma_f32 v[196:197], v[34:35] /*v[290:291]*/, s[46:47], v[42:43] /*v[298:299]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1181
	v_exp_f32_e32 v192 /*v704*/, v52 /*v308*/
	v_exp_f32_e32 v194 /*v706*/, v53 /*v309*/
	v_exp_f32_e32 v200 /*v712*/, v55 /*v311*/
	v_exp_f32_e32 v193 /*v705*/, v48 /*v304*/
	v_exp_f32_e32 v195 /*v707*/, v49 /*v305*/
	v_nop
	s_set_vgpr_msb 0x8153
	v_pk_fma_f32 v[48:49] /*v[304:305]*/, v[234:235] /*v[1002:1003]*/, s[46:47], v[42:43] /*v[298:299]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5381
	v_exp_f32_e32 v203 /*v715*/, v46 /*v302*/
	v_exp_f32_e32 v209 /*v721*/, v47 /*v303*/
	v_nop
	s_set_vgpr_msb 0x8153
	v_pk_fma_f32 v[46:47] /*v[302:303]*/, v[238:239] /*v[1006:1007]*/, s[46:47], v[42:43] /*v[298:299]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x53c1
	v_exp_f32_e32 v17 /*v785*/, v44 /*v300*/
	v_exp_f32_e32 v33 /*v801*/, v45 /*v301*/
	v_nop
	s_set_vgpr_msb 0xc153
	v_pk_fma_f32 v[44:45] /*v[300:301]*/, v[248:249] /*v[1016:1017]*/, s[46:47], v[42:43] /*v[298:299]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5380
	v_exp_f32_e32 v207 /*v719*/, v198
	s_set_vgpr_msb 0x8010
	v_pk_fma_f32 v[208:209], v[208:209], s[46:47], v[42:43] /*v[298:299]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1080
	v_exp_f32_e32 v215 /*v727*/, v199
	v_nop
	s_set_vgpr_msb 0x8013
	v_pk_fma_f32 v[198:199], v[250:251] /*v[1018:1019]*/, s[46:47], v[42:43] /*v[298:299]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1310
	v_pk_fma_f32 v[200:201], v[200:201], s[46:47], v[42:43] /*v[298:299]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x10c0
	v_exp_f32_e32 v117 /*v885*/, v192
	v_exp_f32_e32 v131 /*v899*/, v193
	v_exp_f32_e32 v135 /*v903*/, v194
	s_set_vgpr_msb 0xc011
	v_pk_fma_f32 v[192:193], v[30:31] /*v[286:287]*/, s[46:47], v[42:43] /*v[298:299]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x11c0
	v_exp_f32_e32 v147 /*v915*/, v195
	v_exp_f32_e32 v151 /*v919*/, v196
	s_set_vgpr_msb 0xc011
	v_pk_fma_f32 v[194:195], v[40:41] /*v[296:297]*/, s[46:47], v[42:43] /*v[298:299]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x11c0
	v_exp_f32_e32 v161 /*v929*/, v197
	v_nop
	s_set_vgpr_msb 0xc011
	v_pk_fma_f32 v[196:197], v[36:37] /*v[292:293]*/, s[46:47], v[42:43] /*v[298:299]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1181
	v_exp_f32_e32 v196 /*v708*/, v54 /*v310*/
	v_exp_f32_e32 v211 /*v723*/, v48 /*v304*/
	v_exp_f32_e32 v219 /*v731*/, v49 /*v305*/
	v_exp_f32_e32 v237 /*v749*/, v46 /*v302*/
	v_exp_f32_e32 v249 /*v761*/, v47 /*v303*/
	v_nop
	s_set_vgpr_msb 0x8153
	v_pk_fma_f32 v[46:47] /*v[302:303]*/, v[240:241] /*v[1008:1009]*/, s[46:47], v[42:43] /*v[298:299]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5381
	v_exp_f32_e32 v227 /*v739*/, v45 /*v301*/
	s_set_vgpr_msb 0x8180
	v_exp_f32_e32 v229 /*v741*/, v208
	s_set_vgpr_msb 0x80c0
	v_exp_f32_e32 v7 /*v775*/, v198
	v_exp_f32_e32 v23 /*v791*/, v199
	v_exp_f32_e32 v27 /*v795*/, v200
	s_set_vgpr_msb 0xc010
	v_pk_fma_f32 v[198:199], v[202:203], s[46:47], v[42:43] /*v[298:299]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x10c0
	v_exp_f32_e32 v165 /*v933*/, v192
	v_exp_f32_e32 v173 /*v941*/, v193
	v_exp_f32_e32 v175 /*v943*/, v194
	v_exp_f32_e32 v183 /*v951*/, v195
	s_set_vgpr_msb 0xc011
	v_pk_fma_f32 v[192:193], v[22:23] /*v[278:279]*/, s[46:47], v[42:43] /*v[298:299]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x11c0
	v_exp_f32_e32 v185 /*v953*/, v196
	s_set_vgpr_msb 0xc00a
	v_pk_add_f32 v[194:195], v[192:193] /*v[704:705]*/, v[194:195] /*v[706:707]*/
	s_set_vgpr_msb 0xac0
	v_exp_f32_e32 v177 /*v945*/, v197
	v_nop
	s_set_vgpr_msb 0xc00a
	v_pk_add_f32 v[196:197], v[200:201] /*v[712:713]*/, v[202:203] /*v[714:715]*/
	s_set_vgpr_msb 0xa81
	v_exp_f32_e32 v199 /*v711*/, v46 /*v302*/
	v_exp_f32_e32 v217 /*v729*/, v44 /*v300*/
	s_set_vgpr_msb 0x8180
	v_exp_f32_e32 v241 /*v753*/, v209
	s_set_vgpr_msb 0x80c0
	v_exp_f32_e32 v43 /*v811*/, v201
	s_set_vgpr_msb 0xc080
	v_exp_f32_e32 v225 /*v737*/, v198
	v_exp_f32_e32 v235 /*v747*/, v199
	s_set_vgpr_msb 0x80c0
	v_exp_f32_e32 v189 /*v957*/, v192
	v_exp_f32_e32 v191 /*v959*/, v193
	v_nop
	s_set_vgpr_msb 0xc002
	v_pk_add_f32 v[192:193], v[196:197] /*v[708:709]*/, v[194:195]
	s_set_vgpr_msb 0x20a
	v_pk_add_f32 v[194:195], v[210:211] /*v[722:723]*/, v[218:219] /*v[730:731]*/
	s_set_vgpr_msb 0xa02
	v_pk_add_f32 v[196:197], v[208:209] /*v[720:721]*/, v[196:197]
	s_set_vgpr_msb 0x20a
	v_pk_add_f32 v[198:199], v[232:233] /*v[744:745]*/, v[236:237] /*v[748:749]*/
	s_set_vgpr_msb 0xa0e
	v_pk_add_f32 v[200:201], v[252:253] /*v[764:765]*/, v[10:11] /*v[778:779]*/
	s_set_vgpr_msb 0xe0a
	v_pk_add_f32 v[204:205], v[206:207] /*v[718:719]*/, v[214:215] /*v[726:727]*/
	v_pk_add_f32 v[206:207], v[226:227] /*v[738:739]*/, v[228:229] /*v[740:741]*/
	s_set_vgpr_msb 0xa0f
	v_pk_add_f32 v[210:211], v[22:23] /*v[790:791]*/, v[26:27] /*v[794:795]*/
	v_pk_add_f32 v[212:213], v[48:49] /*v[816:817]*/, v[64:65] /*v[832:833]*/
	s_set_vgpr_msb 0xf0a
	v_pk_add_f32 v[216:217], v[238:239] /*v[750:751]*/, v[250:251] /*v[762:763]*/
	s_set_vgpr_msb 0xa0f
	v_pk_add_f32 v[218:219], v[12:13] /*v[780:781]*/, v[18:19] /*v[786:787]*/
	s_set_vgpr_msb 0xf81
	v_exp_f32_e32 v205 /*v717*/, v47 /*v303*/
	s_set_vgpr_msb 0x810b
	v_pk_add_f32 v[202:203], v[32:33] /*v[800:801]*/, v[198:199] /*v[710:711]*/
	s_set_vgpr_msb 0xb02
	v_pk_add_f32 v[194:195], v[222:223] /*v[734:735]*/, v[194:195]
	v_pk_add_f32 v[198:199], v[248:249] /*v[760:761]*/, v[198:199]
	s_set_vgpr_msb 0x203
	v_pk_add_f32 v[200:201], v[16:17] /*v[784:785]*/, v[200:201]
	s_set_vgpr_msb 0x302
	v_pk_add_f32 v[204:205], v[216:217] /*v[728:729]*/, v[204:205]
	s_set_vgpr_msb 0x20e
	v_pk_add_f32 v[208:209], v[244:245] /*v[756:757]*/, v[2:3] /*v[770:771]*/
	s_set_vgpr_msb 0xe02
	v_pk_add_f32 v[206:207], v[240:241] /*v[752:753]*/, v[206:207]
	s_set_vgpr_msb 0x20a
	v_pk_add_f32 v[214:215], v[220:221] /*v[732:733]*/, v[224:225] /*v[736:737]*/
	s_set_vgpr_msb 0xa03
	v_pk_add_f32 v[210:211], v[42:43] /*v[810:811]*/, v[210:211]
	s_set_vgpr_msb 0x302
	v_pk_add_f32 v[212:213], v[212:213] /*v[724:725]*/, v[212:213]
	s_set_vgpr_msb 0x20f
	v_pk_add_f32 v[220:221], v[38:39] /*v[806:807]*/, v[54:55] /*v[822:823]*/
	v_pk_add_f32 v[222:223], v[74:75] /*v[842:843]*/, v[80:81] /*v[848:849]*/
	s_set_vgpr_msb 0xf02
	v_pk_add_f32 v[216:217], v[254:255] /*v[766:767]*/, v[216:217]
	s_set_vgpr_msb 0x20a
	v_pk_add_f32 v[224:225], v[230:231] /*v[742:743]*/, v[242:243] /*v[754:755]*/
	s_set_vgpr_msb 0xa03
	v_pk_add_f32 v[218:219], v[34:35] /*v[802:803]*/, v[218:219]
	s_set_vgpr_msb 0x30f
	v_pk_add_f32 v[228:229], v[28:29] /*v[796:797]*/, v[44:45] /*v[812:813]*/
	v_pk_add_f32 v[230:231], v[66:67] /*v[834:835]*/, v[70:71] /*v[838:839]*/
	v_pk_add_f32 v[234:235], v[126:127] /*v[894:895]*/, v[0:1] /*v[768:769]*/
	v_pk_add_f32 v[236:237], v[20:21] /*v[788:789]*/, v[36:37] /*v[804:805]*/
	v_pk_add_f32 v[246:247], v[46:47] /*v[814:815]*/, v[52:53] /*v[820:821]*/
	v_pk_add_f32 v[248:249], v[72:73] /*v[840:841]*/, v[88:89] /*v[856:857]*/
	v_pk_add_f32 v[252:253], v[132:133] /*v[900:901]*/, v[144:145] /*v[912:913]*/
	v_pk_add_f32 v[254:255], v[158:159] /*v[926:927]*/, v[162:163] /*v[930:931]*/
	s_set_vgpr_msb 0xf4f
	v_pk_add_f32 v[2:3] /*v[258:259]*/, v[100:101] /*v[868:869]*/, v[104:105] /*v[872:873]*/
	v_pk_add_f32 v[4:5] /*v[260:261]*/, v[124:125] /*v[892:893]*/, v[138:139] /*v[906:907]*/
	v_pk_add_f32 v[8:9] /*v[264:265]*/, v[168:169] /*v[936:937]*/, v[178:179] /*v[946:947]*/
	v_pk_add_f32 v[10:11] /*v[266:267]*/, v[186:187] /*v[954:955]*/, v[94:95] /*v[862:863]*/
	v_pk_add_f32 v[14:15] /*v[270:271]*/, v[146:147] /*v[914:915]*/, v[150:151] /*v[918:919]*/
	v_pk_add_f32 v[16:17] /*v[272:273]*/, v[164:165] /*v[932:933]*/, v[172:173] /*v[940:941]*/
	s_set_vgpr_msb 0x4f00
	v_pk_add_f32 v[192:193], v[192:193], v[196:197]
	s_set_vgpr_msb 2
	v_pk_add_f32 v[202:203], v[204:205] /*v[716:717]*/, v[202:203]
	s_set_vgpr_msb 0x203
	v_pk_add_f32 v[208:209], v[6:7] /*v[774:775]*/, v[208:209]
	s_set_vgpr_msb 0x302
	v_pk_add_f32 v[214:215], v[234:235] /*v[746:747]*/, v[214:215]
	s_set_vgpr_msb 0x203
	v_pk_add_f32 v[220:221], v[58:59] /*v[826:827]*/, v[220:221]
	v_pk_add_f32 v[222:223], v[96:97] /*v[864:865]*/, v[222:223]
	s_set_vgpr_msb 0x30f
	v_pk_add_f32 v[226:227], v[4:5] /*v[772:773]*/, v[8:9] /*v[776:777]*/
	s_set_vgpr_msb 0xf02
	v_pk_add_f32 v[224:225], v[246:247] /*v[758:759]*/, v[224:225]
	s_set_vgpr_msb 0x20f
	v_pk_add_f32 v[232:233], v[90:91] /*v[858:859]*/, v[106:107] /*v[874:875]*/
	s_set_vgpr_msb 0xf03
	v_pk_add_f32 v[228:229], v[50:51] /*v[818:819]*/, v[228:229]
	v_pk_add_f32 v[230:231], v[86:87] /*v[854:855]*/, v[230:231]
	v_pk_add_f32 v[234:235], v[14:15] /*v[782:783]*/, v[234:235]
	s_set_vgpr_msb 0x30f
	v_pk_add_f32 v[238:239], v[56:57] /*v[824:825]*/, v[60:61] /*v[828:829]*/
	v_pk_add_f32 v[240:241], v[82:83] /*v[850:851]*/, v[98:99] /*v[866:867]*/
	v_pk_add_f32 v[242:243], v[118:119] /*v[886:887]*/, v[122:123] /*v[890:891]*/
	s_set_vgpr_msb 0xf03
	v_pk_add_f32 v[236:237], v[40:41] /*v[808:809]*/, v[236:237]
	s_set_vgpr_msb 0x30f
	v_pk_add_f32 v[250:251], v[108:109] /*v[876:877]*/, v[114:115] /*v[882:883]*/
	s_set_vgpr_msb 0xf03
	v_pk_add_f32 v[246:247], v[68:69] /*v[836:837]*/, v[246:247]
	v_pk_add_f32 v[248:249], v[92:93] /*v[860:861]*/, v[248:249]
	v_pk_add_f32 v[252:253], v[148:149] /*v[916:917]*/, v[252:253]
	s_set_vgpr_msb 0x34f
	v_pk_add_f32 v[0:1] /*v[256:257]*/, v[62:63] /*v[830:831]*/, v[78:79] /*v[846:847]*/
	s_set_vgpr_msb 0x4f03
	v_pk_add_f32 v[254:255], v[170:171] /*v[938:939]*/, v[254:255]
	s_set_vgpr_msb 0x34f
	v_pk_add_f32 v[6:7] /*v[262:263]*/, v[154:155] /*v[922:923]*/, v[156:157] /*v[924:925]*/
	s_set_vgpr_msb 0x4f47
	v_pk_add_f32 v[2:3] /*v[258:259]*/, v[120:121] /*v[888:889]*/, v[2:3] /*v[258:259]*/
	v_pk_add_f32 v[4:5] /*v[260:261]*/, v[142:143] /*v[910:911]*/, v[4:5] /*v[260:261]*/
	v_pk_add_f32 v[8:9] /*v[264:265]*/, v[180:181] /*v[948:949]*/, v[8:9] /*v[264:265]*/
	s_set_vgpr_msb 0x474f
	v_pk_add_f32 v[12:13] /*v[268:269]*/, v[116:117] /*v[884:885]*/, v[130:131] /*v[898:899]*/
	s_set_vgpr_msb 0x4f47
	v_pk_add_f32 v[10:11] /*v[266:267]*/, v[110:111] /*v[878:879]*/, v[10:11] /*v[266:267]*/
	s_set_vgpr_msb 0x474f
	v_pk_add_f32 v[18:19] /*v[274:275]*/, v[182:183] /*v[950:951]*/, v[184:185] /*v[952:953]*/
	s_set_vgpr_msb 0x4f47
	v_pk_add_f32 v[14:15] /*v[270:271]*/, v[160:161] /*v[928:929]*/, v[14:15] /*v[270:271]*/
	v_pk_add_f32 v[16:17] /*v[272:273]*/, v[174:175] /*v[942:943]*/, v[16:17] /*v[272:273]*/
	s_set_vgpr_msb 0x4700
	v_pk_add_f32 v[198:199], v[198:199], v[200:201]
	v_pk_add_f32 v[200:201], v[204:205], v[206:207]
	v_pk_add_f32 v[192:193], v[194:195], v[192:193]
	v_pk_add_f32 v[194:195], v[210:211], v[212:213]
	v_pk_add_f32 v[204:205], v[216:217], v[218:219]
	s_set_vgpr_msb 3
	v_pk_add_f32 v[226:227], v[24:25] /*v[792:793]*/, v[226:227]
	v_pk_add_f32 v[232:233], v[112:113] /*v[880:881]*/, v[232:233]
	s_set_vgpr_msb 0x30f
	v_pk_add_f32 v[244:245], v[140:141] /*v[908:909]*/, v[152:153] /*v[920:921]*/
	s_set_vgpr_msb 0xf03
	v_pk_add_f32 v[238:239], v[76:77] /*v[844:845]*/, v[238:239]
	v_pk_add_f32 v[240:241], v[102:103] /*v[870:871]*/, v[240:241]
	v_pk_add_f32 v[242:243], v[136:137] /*v[904:905]*/, v[242:243]
	v_pk_add_f32 v[250:251], v[128:129] /*v[896:897]*/, v[250:251]
	s_set_vgpr_msb 0x347
	v_pk_add_f32 v[0:1] /*v[256:257]*/, v[84:85] /*v[852:853]*/, v[0:1] /*v[256:257]*/
	v_pk_add_f32 v[6:7] /*v[262:263]*/, v[166:167] /*v[934:935]*/, v[6:7] /*v[262:263]*/
	v_pk_add_f32 v[12:13] /*v[268:269]*/, v[134:135] /*v[902:903]*/, v[12:13] /*v[268:269]*/
	s_set_vgpr_msb 0x4707
	v_pk_add_f32 v[196:197], v[176:177] /*v[944:945]*/, v[18:19] /*v[274:275]*/
	s_set_vgpr_msb 0x700
	v_pk_add_f32 v[198:199], v[202:203], v[198:199]
	v_pk_add_f32 v[200:201], v[208:209], v[200:201]
	v_pk_add_f32 v[202:203], v[222:223], v[224:225]
	v_pk_add_f32 v[194:195], v[214:215], v[194:195]
	v_pk_add_f32 v[204:205], v[220:221], v[204:205]
	v_pk_add_f32 v[206:207], v[228:229], v[230:231]
	v_pk_add_f32 v[208:209], v[234:235], v[236:237]
	v_pk_add_f32 v[212:213], v[246:247], v[248:249]
	v_pk_add_f32 v[214:215], v[252:253], v[254:255]
	s_set_vgpr_msb 5
	v_pk_add_f32 v[216:217], v[2:3] /*v[258:259]*/, v[4:5] /*v[260:261]*/
	v_pk_add_f32 v[218:219], v[8:9] /*v[264:265]*/, v[10:11] /*v[266:267]*/
	v_pk_add_f32 v[220:221], v[14:15] /*v[270:271]*/, v[16:17] /*v[272:273]*/
	s_set_vgpr_msb 0x503
	v_pk_add_f32 v[244:245], v[30:31] /*v[798:799]*/, v[244:245]
	s_set_vgpr_msb 0x300
	v_pk_add_f32 v[202:203], v[226:227], v[202:203]
	v_pk_add_f32 v[210:211], v[240:241], v[242:243]
	v_pk_add_f32 v[206:207], v[232:233], v[206:207]
	v_pk_add_f32 v[208:209], v[238:239], v[208:209]
	v_pk_add_f32 v[212:213], v[250:251], v[212:213]
	s_set_vgpr_msb 1
	v_pk_add_f32 v[214:215], v[0:1] /*v[256:257]*/, v[214:215]
	s_set_vgpr_msb 0x100
	v_pk_add_f32 v[192:193], v[192:193], v[198:199]
	s_set_vgpr_msb 1
	v_pk_add_f32 v[198:199], v[6:7] /*v[262:263]*/, v[216:217]
	v_pk_add_f32 v[216:217], v[12:13] /*v[268:269]*/, v[218:219]
	s_set_vgpr_msb 0x100
	v_pk_add_f32 v[194:195], v[194:195], v[204:205]
	v_pk_add_f32 v[196:197], v[196:197], v[220:221]
	s_set_vgpr_msb 0x4f
	v_pk_add_f32 v[18:19] /*v[274:275]*/, v[188:189] /*v[956:957]*/, v[190:191] /*v[958:959]*/
	s_set_vgpr_msb 0x4f00
	v_pk_add_f32 v[210:211], v[244:245], v[210:211]
	v_pk_add_f32 v[192:193], v[200:201], v[192:193]
	v_pk_add_f32 v[200:201], v[206:207], v[208:209]
	v_pk_add_f32 v[204:205], v[212:213], v[214:215]
	v_pk_add_f32 v[194:195], v[202:203], v[194:195]
	v_pk_add_f32 v[196:197], v[216:217], v[196:197]
	s_set_vgpr_msb 4
	v_sub_f32_e32 v130, v132, v50 /*v306*/
	s_set_vgpr_msb 0x400
	v_pk_add_f32 v[200:201], v[210:211], v[200:201]
	v_pk_add_f32 v[198:199], v[198:199], v[204:205]
	v_pk_add_f32 v[192:193], v[192:193], v[194:195]
	s_set_vgpr_msb 1
	v_pk_add_f32 v[194:195], v[18:19] /*v[274:275]*/, v[196:197]
	s_set_vgpr_msb 0x100
	v_pk_add_f32 v[192:193], v[200:201], v[192:193]
	v_pk_add_f32 v[194:195], v[198:199], v[194:195]
	s_set_vgpr_msb 0xc0
	v_pk_add_f32 v[192:193] /*v[960:961]*/, v[194:195], v[192:193]
	s_set_vgpr_msb 0xc003
	v_mul_f32_e32 v192, 0x3fb8aa3b, v130
	v_dual_mov_b32 v130, v192 /*v960*/ :: v_dual_mov_b32 v132, v193 /*v961*/
	s_set_vgpr_msb 0x3c0
	v_exp_f32_e32 v194 /*v962*/, v192
	s_set_vgpr_msb 0xc000
	v_permlanex16_b32 v130, v130, s47, 0xfedcba98
	v_permlanex16_b32 v132, v132, s47, 0xfedcba98
	s_cbranch_vccz .LBB0_42
	s_set_vgpr_msb 12
	v_pk_mul_f32 v[126:127], v[126:127], v[194:195] /*v[962:963]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[124:125], v[124:125], v[194:195] /*v[962:963]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[122:123], v[122:123], v[194:195] /*v[962:963]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[120:121], v[120:121], v[194:195] /*v[962:963]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[118:119], v[118:119], v[194:195] /*v[962:963]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[116:117], v[116:117], v[194:195] /*v[962:963]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[114:115], v[114:115], v[194:195] /*v[962:963]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[112:113], v[112:113], v[194:195] /*v[962:963]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[110:111], v[110:111], v[194:195] /*v[962:963]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[108:109], v[108:109], v[194:195] /*v[962:963]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[106:107], v[106:107], v[194:195] /*v[962:963]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[104:105], v[104:105], v[194:195] /*v[962:963]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[102:103], v[102:103], v[194:195] /*v[962:963]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[100:101], v[100:101], v[194:195] /*v[962:963]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[98:99], v[98:99], v[194:195] /*v[962:963]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[96:97], v[96:97], v[194:195] /*v[962:963]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[94:95], v[94:95], v[194:195] /*v[962:963]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[92:93], v[92:93], v[194:195] /*v[962:963]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[90:91], v[90:91], v[194:195] /*v[962:963]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[88:89], v[88:89], v[194:195] /*v[962:963]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[86:87], v[86:87], v[194:195] /*v[962:963]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[84:85], v[84:85], v[194:195] /*v[962:963]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[82:83], v[82:83], v[194:195] /*v[962:963]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[80:81], v[80:81], v[194:195] /*v[962:963]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[78:79], v[78:79], v[194:195] /*v[962:963]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[76:77], v[76:77], v[194:195] /*v[962:963]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[74:75], v[74:75], v[194:195] /*v[962:963]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[72:73], v[72:73], v[194:195] /*v[962:963]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[70:71], v[70:71], v[194:195] /*v[962:963]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[68:69], v[68:69], v[194:195] /*v[962:963]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[66:67], v[66:67], v[194:195] /*v[962:963]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[64:65], v[64:65], v[194:195] /*v[962:963]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0xc00
.LBB0_42:
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0xc0
	scratch_load_b32 v200 /*v968*/, off, off offset:8 nv
	s_set_vgpr_msb 0xc004
	v_sub_f32_e32 v131, v131, v51 /*v307*/
	s_and_b32 s2, s5, exec_lo
	s_cselect_b32 s2, 1, 0
	s_cmp_lg_u32 s2, 1
	s_set_vgpr_msb 0x400
	v_mul_f32_e32 v131, 0x3fb8aa3b, v131
	s_set_vgpr_msb 0xc0
	v_exp_f32_e32 v196 /*v964*/, v131
	s_set_vgpr_msb 0xc000
	s_cbranch_scc1 .LBB0_37
	v_nop
	s_set_vgpr_msb 12
	v_pk_mul_f32 v[62:63], v[62:63], v[196:197] /*v[964:965]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[60:61], v[60:61], v[196:197] /*v[964:965]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59], v[58:59], v[196:197] /*v[964:965]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[56:57], v[56:57], v[196:197] /*v[964:965]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[54:55], v[54:55], v[196:197] /*v[964:965]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[52:53], v[52:53], v[196:197] /*v[964:965]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[50:51], v[50:51], v[196:197] /*v[964:965]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[48:49], v[48:49], v[196:197] /*v[964:965]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[46:47], v[46:47], v[196:197] /*v[964:965]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[44:45], v[44:45], v[196:197] /*v[964:965]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[42:43], v[42:43], v[196:197] /*v[964:965]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[40:41], v[40:41], v[196:197] /*v[964:965]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39], v[38:39], v[196:197] /*v[964:965]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[36:37], v[36:37], v[196:197] /*v[964:965]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35], v[34:35], v[196:197] /*v[964:965]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[32:33], v[32:33], v[196:197] /*v[964:965]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[30:31], v[30:31], v[196:197] /*v[964:965]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[28:29], v[28:29], v[196:197] /*v[964:965]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[26:27], v[26:27], v[196:197] /*v[964:965]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25], v[24:25], v[196:197] /*v[964:965]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[22:23], v[22:23], v[196:197] /*v[964:965]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21], v[20:21], v[196:197] /*v[964:965]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[18:19], v[18:19], v[196:197] /*v[964:965]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[16:17], v[16:17], v[196:197] /*v[964:965]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[14:15], v[196:197] /*v[964:965]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13], v[12:13], v[196:197] /*v[964:965]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[10:11], v[196:197] /*v[964:965]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[8:9], v[196:197] /*v[964:965]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[6:7], v[196:197] /*v[964:965]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[4:5], v[196:197] /*v[964:965]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[2:3], v[2:3], v[196:197] /*v[964:965]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[0:1], v[0:1], v[196:197] /*v[964:965]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0xc00
	s_branch .LBB0_37
.LBB0_44:
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x2
	scratch_load_b32 v192, off, off offset:564 nv
	scratch_load_b32 v193, off, off offset:568 nv
	scratch_load_b32 v194, off, off offset:584 nv
.LBB0_45:
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b32 v154, off, off offset:576 nv
	scratch_load_b32 v155, off, off offset:580 nv
.LBB0_46:
	s_wait_loadcnt 0x5
	v_div_scale_f32 v129, null, v204, v204, 1.0
	v_div_scale_f32 v132, vcc_lo, 1.0, v204, 1.0
	v_div_scale_f32 v150, null, v205, v205, 1.0
	v_cmp_lt_f32_e64 s2, 0, v204
	s_bitcmp1_b32 s9, 8
	s_wait_loadcnt 0x3
	s_wait_alu depctr_vm_vsrc(3)
	v_rcp_f32_e32 v130, v129
	s_cselect_b32 s4, 0x23000, 0
	v_rcp_f32_e32 v151, v150
	s_add_co_i32 s89, s89, s4
	s_wait_loadcnt 0x0
	v_mul_u32_u24_e32 v128, 0x110, v154
	s_mov_b32 s4, 0
	s_wait_dscnt 0x0
	v_fma_f32 v131, -v129, v130, 1.0
	v_fmac_f32_e32 v130, v131, v130
	v_mul_f32_e32 v131, v132, v130
	v_fma_f32 v133, -v129, v131, v132
	v_fmac_f32_e32 v131, v133, v130
	v_fma_f32 v129, -v129, v131, v132
	v_div_fmas_f32 v129, v129, v130, v131
	v_fma_f32 v130, -v150, v151, 1.0
	v_div_scale_f32 v152, vcc_lo, 1.0, v205, 1.0
	v_div_fixup_f32 v129, v129, v204, 1.0
	v_dual_fmac_f32 v151, v130, v151 :: v_dual_cndmask_b32 v130, 0, v129, s2
	v_dual_add_nc_u32 v129, s89, v194 :: v_dual_mul_f32 v153, v152, v151
	v_pk_mul_f32 v[140:141], v[130:131], v[76:77] op_sel_hi:[0,1]
	v_pk_mul_f32 v[90:91], v[130:131], v[90:91] op_sel_hi:[0,1]
	v_fma_f32 v76, -v150, v153, v152
	v_pk_mul_f32 v[88:89], v[130:131], v[88:89] op_sel_hi:[0,1]
	v_pk_mul_f32 v[132:133], v[130:131], v[80:81] op_sel_hi:[0,1]
	v_pk_mul_f32 v[92:93], v[130:131], v[92:93] op_sel_hi:[0,1]
	v_cvt_pk_bf16_f32 v81, v90, v91
	v_fmac_f32_e32 v153, v76, v151
	v_cvt_pk_bf16_f32 v80, v88, v89
	v_pk_mul_f32 v[96:97], v[96:97], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[134:135], v[130:131], v[82:83] op_sel_hi:[0,1]
	v_cvt_pk_bf16_f32 v82, v92, v93
	v_fma_f32 v90, -v150, v153, v152
	v_pk_mul_f32 v[112:113], v[112:113], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[114:115], v[114:115], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[104:105], v[104:105], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[106:107], v[106:107], v[130:131] op_sel_hi:[1,0]
	v_div_fmas_f32 v88, v90, v151, v153
	v_cmp_lt_f32_e32 vcc_lo, 0, v205
	v_pk_mul_f32 v[108:109], v[108:109], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[110:111], v[110:111], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[98:99], v[98:99], v[130:131] op_sel_hi:[1,0]
	v_div_fixup_f32 v92, v88, v205, 1.0
	v_pk_mul_f32 v[100:101], v[100:101], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[102:103], v[102:103], v[130:131] op_sel_hi:[1,0]
	v_cvt_pk_bf16_f32 v76, v96, v97
	v_pk_mul_f32 v[120:121], v[120:121], v[130:131] op_sel_hi:[1,0]
	v_cndmask_b32_e32 v96, 0, v92, vcc_lo
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
	s_lshl_b32 s2, s57, 2
	s_wait_alu depctr_va_vdst(0)
	ds_store_b128 v129, v[64:67]
	ds_store_b128 v129, v[68:71] offset:32
	ds_store_b128 v129, v[72:75] offset:64
	ds_store_b128 v129, v[76:79] offset:96
	ds_store_b128 v129, v[80:83] offset:128
	ds_store_b128 v129, v[84:87] offset:160
	ds_store_b128 v129, v[88:91] offset:192
	ds_store_b128 v129, v[92:95] offset:224
	ds_store_b128 v129, v[0:3] offset:4352
	ds_store_b128 v129, v[4:7] offset:4384
	ds_store_b128 v129, v[8:11] offset:4416
	ds_store_b128 v129, v[12:15] offset:4448
	s_sub_co_i32 s2, s2, s88
	ds_store_b128 v129, v[16:19] offset:4480
	ds_store_b128 v129, v[20:23] offset:4512
	ds_store_b128 v129, v[24:27] offset:4544
	ds_store_b128 v129, v[28:31] offset:4576
	s_wait_dscnt 0x0
	scratch_load_b32 v57, off, off offset:572 nv
	s_cmp_lt_i32 s2, 1
	s_cbranch_scc1 .LBB0_48
	s_wait_loadcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	v_dual_lshlrev_b32 v4, 8, v154 :: v_dual_bitop2_b32 v0, 28, v57 bitop3:0x54
	s_add_co_i32 s2, s2, -1
	v_dual_lshrrev_b32 v26, 4, v155 :: v_dual_add_nc_u32 v2, s89, v128
	s_min_u32 s5, s2, 31
	s_load_b64 s[6:7], s[0:1], 0x0 nv
	v_min_u32_e32 v3, s5, v0
	v_lshlrev_b32_e32 v0, 3, v154
	v_dual_sub_nc_u32 v32, v2, v4 :: v_dual_bitop2_b32 v5, 26, v26 bitop3:0x54
	v_or_b32_e32 v30, 8, v57
	v_or_b32_e32 v7, s88, v3
	v_or_b32_e32 v1, 30, v26
	v_mad_u32_u24 v33, 0x110, v3, v32
	v_min_u32_e32 v41, s5, v30
	v_ashrrev_i32_e32 v10, 31, v7
	v_min_u32_e32 v6, s5, v1
	v_mov_b32_e32 v1, 0
	v_min_u32_e32 v9, s5, v5
	v_dual_lshrrev_b32 v4, 30, v10 :: v_dual_bitop2_b32 v30, s88, v41 bitop3:0x54
	v_or_b32_e32 v8, s88, v6
	v_or_b32_e32 v5, 24, v57
	v_or_b32_e32 v11, s88, v9
	v_mad_u32_u24 v34, 0x110, v6, v32
	v_add_nc_u32_e32 v3, v7, v4
	v_ashrrev_i32_e32 v2, 31, v8
	v_min_u32_e32 v14, s5, v5
	v_dual_ashrrev_i32 v10, 31, v11 :: v_dual_bitop2_b32 v5, 22, v26 bitop3:0x54
	v_mad_u32_u24 v35, 0x110, v9, v32
	v_lshrrev_b32_e32 v2, 30, v2
	v_or_b32_e32 v12, s88, v14
	v_min_u32_e32 v15, s5, v5
	v_dual_lshrrev_b32 v4, 30, v10 :: v_dual_bitop2_b32 v5, -4, v3 bitop3:0x40
	v_dual_add_nc_u32 v2, v8, v2 :: v_dual_ashrrev_i32 v3, 2, v3
	v_mad_u32_u24 v36, 0x110, v14, v32
	v_add_nc_u32_e32 v13, v11, v4
	v_dual_sub_nc_u32 v4, v7, v5 :: v_dual_bitop2_b32 v10, -4, v2 bitop3:0x40
	v_ashrrev_i32_e32 v2, 2, v2
	v_cmp_ne_u32_e32 vcc_lo, v7, v5
	v_add_nc_u32_e32 v7, s33, v3
	v_mad_u32_u24 v37, 0x110, v15, v32
	v_cmp_ne_u32_e64 s2, v8, v10
	v_sub_nc_u32_e32 v5, v8, v10
	v_dual_add_nc_u32 v8, s33, v2 :: v_dual_add_nc_u32 v2, s34, v4
	s_and_b32 s9, s59, vcc_lo
	s_and_b32 s2, s59, s2
	v_cndmask_b32_e64 v10, 0, 1, s9
	v_cndmask_b32_e64 v16, 0, 1, s2
	v_add_nc_u32_e32 v4, s34, v5
	v_mad_nc_i64_i32 v[2:3], v2, s28, v[0:1]
	v_mad_u32_u24 v41, 0x110, v41, v32
	v_dual_sub_nc_u32 v6, v7, v10 :: v_dual_sub_nc_u32 v7, v8, v16
	v_dual_ashrrev_i32 v10, 31, v12 :: v_dual_bitop2_b32 v8, -4, v13 bitop3:0x40
	v_dual_ashrrev_i32 v13, 2, v13 :: v_dual_bitop2_b32 v16, s88, v15 bitop3:0x54
	v_mad_nc_i64_i32 v[2:3], v6, s8, v[2:3]
	v_dual_sub_nc_u32 v6, v11, v8 :: v_dual_lshrrev_b32 v10, 30, v10
	v_cmp_ne_u32_e32 vcc_lo, v11, v8
	v_mad_nc_i64_i32 v[4:5], v4, s28, v[0:1]
	v_dual_add_nc_u32 v8, s33, v13 :: v_dual_add_nc_u32 v6, s34, v6
	v_dual_add_nc_u32 v10, v12, v10 :: v_dual_bitop2_b32 v13, 20, v57 bitop3:0x54
	s_and_b32 s2, s59, vcc_lo
	s_wait_kmcnt 0x0
	v_lshl_add_u64 v[2:3], v[2:3], 1, s[6:7]
	v_cndmask_b32_e64 v11, 0, 1, s2
	v_mad_nc_i64_i32 v[4:5], v7, s8, v[4:5]
	v_mad_nc_i64_i32 v[6:7], v6, s28, v[0:1]
	v_ashrrev_i32_e32 v17, 31, v16
	v_min_u32_e32 v18, s5, v13
	v_dual_sub_nc_u32 v8, v8, v11 :: v_dual_bitop2_b32 v9, -4, v10 bitop3:0x40
	v_dual_lshrrev_b32 v11, 30, v17 :: v_dual_bitop2_b32 v13, s88, v18 bitop3:0x54
	v_mad_nc_i64_i32 v[6:7], v8, s8, v[6:7]
	v_dual_ashrrev_i32 v8, 2, v10 :: v_dual_sub_nc_u32 v10, v12, v9
	v_cmp_ne_u32_e32 vcc_lo, v12, v9
	v_dual_add_nc_u32 v11, v16, v11 :: v_dual_ashrrev_i32 v9, 31, v13
	v_dual_add_nc_u32 v12, s33, v8 :: v_dual_add_nc_u32 v8, s34, v10
	s_and_b32 s2, s59, vcc_lo
	v_and_b32_e32 v10, -4, v11
	v_cndmask_b32_e64 v17, 0, 1, s2
	v_dual_ashrrev_i32 v11, 2, v11 :: v_dual_lshrrev_b32 v19, 30, v9
	v_mad_nc_i64_i32 v[8:9], v8, s28, v[0:1]
	v_cmp_ne_u32_e32 vcc_lo, v16, v10
	v_dual_sub_nc_u32 v12, v12, v17 :: v_dual_add_nc_u32 v11, s33, v11
	v_dual_add_nc_u32 v17, v13, v19 :: v_dual_sub_nc_u32 v10, v16, v10
	s_and_b32 s2, s59, vcc_lo
	v_mad_u32_u24 v38, 0x110, v18, v32
	v_mad_nc_i64_i32 v[8:9], v12, s8, v[8:9]
	v_dual_add_nc_u32 v10, s34, v10 :: v_dual_bitop2_b32 v12, -4, v17 bitop3:0x40
	v_cndmask_b32_e64 v16, 0, 1, s2
	v_dual_ashrrev_i32 v17, 2, v17 :: v_dual_bitop2_b32 v19, 18, v26 bitop3:0x54
	v_sub_nc_u32_e32 v20, v13, v12
	v_cmp_ne_u32_e32 vcc_lo, v13, v12
	v_sub_nc_u32_e32 v16, v11, v16
	v_mad_nc_i64_i32 v[10:11], v10, s28, v[0:1]
	v_min_u32_e32 v19, s5, v19
	v_dual_add_nc_u32 v17, s33, v17 :: v_dual_add_nc_u32 v12, s34, v20
	s_and_b32 s2, s59, vcc_lo
	v_lshl_add_u64 v[4:5], v[4:5], 1, s[6:7]
	v_cndmask_b32_e64 v22, 0, 1, s2
	v_or_b32_e32 v20, s88, v19
	v_mad_nc_i64_i32 v[10:11], v16, s8, v[10:11]
	v_mad_nc_i64_i32 v[12:13], v12, s28, v[0:1]
	v_mad_u32_u24 v39, 0x110, v19, v32
	v_dual_sub_nc_u32 v16, v17, v22 :: v_dual_bitop2_b32 v21, 16, v57 bitop3:0x54
	v_ashrrev_i32_e32 v23, 31, v20
	v_lshl_add_u64 v[6:7], v[6:7], 1, s[6:7]
	v_lshl_add_u64 v[8:9], v[8:9], 1, s[6:7]
	v_lshl_add_u64 v[10:11], v[10:11], 1, s[6:7]
	v_min_u32_e32 v21, s5, v21
	v_lshrrev_b32_e32 v17, 30, v23
	v_mad_nc_i64_i32 v[12:13], v16, s8, v[12:13]
	v_or_b32_e32 v22, s88, v21
	v_add_nc_u32_e32 v14, v20, v17
	v_mad_u32_u24 v40, 0x110, v21, v32
	v_ashrrev_i32_e32 v16, 31, v22
	v_and_b32_e32 v15, -4, v14
	v_lshl_add_u64 v[12:13], v[12:13], 1, s[6:7]
	v_dual_lshrrev_b32 v16, 30, v16 :: v_dual_bitop2_b32 v17, 14, v26 bitop3:0x54
	v_dual_sub_nc_u32 v18, v20, v15 :: v_dual_ashrrev_i32 v14, 2, v14
	v_cmp_ne_u32_e32 vcc_lo, v20, v15
	v_add_nc_u32_e32 v16, v22, v16
	v_min_u32_e32 v23, s5, v17
	v_add_nc_u32_e32 v20, s33, v14
	v_dual_add_nc_u32 v14, s34, v18 :: v_dual_bitop2_b32 v17, -4, v16 bitop3:0x40
	v_dual_ashrrev_i32 v16, 2, v16 :: v_dual_bitop2_b32 v18, s88, v23 bitop3:0x54
	s_and_b32 s2, s59, vcc_lo
	v_mad_nc_i64_i32 v[14:15], v14, s28, v[0:1]
	v_dual_sub_nc_u32 v25, v22, v17 :: v_dual_ashrrev_i32 v27, 31, v18
	v_cmp_ne_u32_e32 vcc_lo, v22, v17
	v_cndmask_b32_e64 v24, 0, 1, s2
	v_dual_add_nc_u32 v22, s33, v16 :: v_dual_add_nc_u32 v16, s34, v25
	v_dual_lshrrev_b32 v25, 30, v27 :: v_dual_bitop2_b32 v27, 12, v57 bitop3:0x54
	v_sub_nc_u32_e32 v20, v20, v24
	s_and_b32 s2, s59, vcc_lo
	v_mad_nc_i64_i32 v[16:17], v16, s28, v[0:1]
	v_cndmask_b32_e64 v28, 0, 1, s2
	v_min_u32_e32 v27, s5, v27
	v_add_nc_u32_e32 v25, v18, v25
	v_mad_nc_i64_i32 v[14:15], v20, s8, v[14:15]
	v_mad_u32_u24 v42, 0x110, v23, v32
	v_dual_sub_nc_u32 v22, v22, v28 :: v_dual_bitop2_b32 v24, s88, v27 bitop3:0x54
	v_dual_ashrrev_i32 v20, 2, v25 :: v_dual_bitop2_b32 v19, -4, v25 bitop3:0x40
	v_dual_ashrrev_i32 v25, 31, v24 :: v_dual_bitop2_b32 v28, 10, v26 bitop3:0x54
	v_mad_nc_i64_i32 v[16:17], v22, s8, v[16:17]
	v_sub_nc_u32_e32 v22, v18, v19
	v_cmp_ne_u32_e32 vcc_lo, v18, v19
	v_add_nc_u32_e32 v20, s33, v20
	v_mad_u32_u24 v44, 0x110, v27, v32
	v_lshl_add_u64 v[14:15], v[14:15], 1, s[6:7]
	v_dual_add_nc_u32 v18, s34, v22 :: v_dual_lshrrev_b32 v22, 30, v25
	v_min_u32_e32 v25, s5, v28
	s_and_b32 s2, s59, vcc_lo
	v_lshl_add_u64 v[16:17], v[16:17], 1, s[6:7]
	v_cndmask_b32_e64 v28, 0, 1, s2
	v_mad_nc_i64_i32 v[18:19], v18, s28, v[0:1]
	v_dual_add_nc_u32 v22, v24, v22 :: v_dual_bitop2_b32 v29, s88, v25 bitop3:0x54
	v_mad_u32_u24 v45, 0x110, v25, v32
	v_dual_sub_nc_u32 v20, v20, v28 :: v_dual_ashrrev_i32 v28, 31, v29
	v_and_b32_e32 v21, -4, v22
	v_mad_nc_i64_i32 v[18:19], v20, s8, v[18:19]
	v_dual_ashrrev_i32 v20, 2, v22 :: v_dual_lshrrev_b32 v28, 30, v28
	v_sub_nc_u32_e32 v22, v24, v21
	v_cmp_ne_u32_e32 vcc_lo, v24, v21
	v_dual_add_nc_u32 v24, s33, v20 :: v_dual_add_nc_u32 v20, s34, v22
	v_add_nc_u32_e32 v22, v29, v28
	s_and_b32 s2, s59, vcc_lo
	v_lshl_add_u64 v[18:19], v[18:19], 1, s[6:7]
	v_cndmask_b32_e64 v28, 0, 1, s2
	v_mad_nc_i64_i32 v[20:21], v20, s28, v[0:1]
	v_and_b32_e32 v23, -4, v22
	v_dual_ashrrev_i32 v22, 2, v22 :: v_dual_sub_nc_u32 v24, v24, v28
	v_dual_sub_nc_u32 v28, v29, v23 :: v_dual_ashrrev_i32 v31, 31, v30
	v_cmp_ne_u32_e32 vcc_lo, v29, v23
	v_mad_nc_i64_i32 v[20:21], v24, s8, v[20:21]
	v_dual_add_nc_u32 v24, s33, v22 :: v_dual_add_nc_u32 v22, s34, v28
	v_dual_lshrrev_b32 v28, 30, v31 :: v_dual_bitop2_b32 v29, 6, v26 bitop3:0x54
	s_and_b32 s2, s59, vcc_lo
	v_cndmask_b32_e64 v31, 0, 1, s2
	v_add_nc_u32_e32 v28, v30, v28
	v_min_u32_e32 v43, s5, v29
	v_mad_nc_i64_i32 v[22:23], v22, s28, v[0:1]
	v_lshl_add_u64 v[20:21], v[20:21], 1, s[6:7]
	v_dual_sub_nc_u32 v24, v24, v31 :: v_dual_bitop2_b32 v27, -4, v28 bitop3:0x40
	v_or_b32_e32 v29, s88, v43
	v_mad_u32_u24 v43, 0x110, v43, v32
	v_mad_nc_i64_i32 v[22:23], v24, s8, v[22:23]
	v_dual_ashrrev_i32 v24, 2, v28 :: v_dual_sub_nc_u32 v25, v30, v27
	v_ashrrev_i32_e32 v28, 31, v29
	v_or_b32_e32 v31, 4, v57
	v_cmp_ne_u32_e32 vcc_lo, v30, v27
	v_dual_add_nc_u32 v27, s33, v24 :: v_dual_add_nc_u32 v24, s34, v25
	v_lshrrev_b32_e32 v28, 30, v28
	v_min_u32_e32 v46, s5, v31
	s_and_b32 s2, s59, vcc_lo
	v_lshl_add_u64 v[22:23], v[22:23], 1, s[6:7]
	v_cndmask_b32_e64 v30, 0, 1, s2
	v_dual_add_nc_u32 v28, v29, v28 :: v_dual_bitop2_b32 v31, s88, v46 bitop3:0x54
	v_mad_nc_i64_i32 v[24:25], v24, s28, v[0:1]
	v_or_b32_e32 v26, 2, v26
	v_dual_sub_nc_u32 v27, v27, v30 :: v_dual_bitop2_b32 v30, -4, v28 bitop3:0x40
	v_dual_ashrrev_i32 v28, 2, v28 :: v_dual_ashrrev_i32 v47, 31, v31
	v_min_u32_e32 v48, s5, v26
	v_mad_u32_u24 v46, 0x110, v46, v32
	v_cmp_ne_u32_e32 vcc_lo, v29, v30
	v_mad_nc_i64_i32 v[24:25], v27, s8, v[24:25]
	v_dual_add_nc_u32 v26, s33, v28 :: v_dual_lshrrev_b32 v27, 30, v47
	v_min_i32_e32 v47, s5, v57
	s_and_b32 s2, s59, vcc_lo
	v_dual_sub_nc_u32 v29, v29, v30 :: v_dual_bitop2_b32 v28, s88, v48 bitop3:0x54
	v_cndmask_b32_e64 v49, 0, 1, s2
	v_or_b32_e32 v30, s88, v47
	v_dual_add_nc_u32 v27, v31, v27 :: v_dual_ashrrev_i32 v50, 31, v28
	v_mad_u32_u24 v47, 0x110, v47, v32
	v_sub_nc_u32_e32 v49, v26, v49
	v_dual_add_nc_u32 v26, s34, v29 :: v_dual_ashrrev_i32 v29, 31, v30
	v_dual_lshrrev_b32 v50, 30, v50 :: v_dual_bitop2_b32 v51, -4, v27 bitop3:0x40
	v_ashrrev_i32_e32 v52, 2, v27
	v_mad_nc_i64_i32 v[26:27], v26, s28, v[0:1]
	v_dual_lshrrev_b32 v29, 30, v29 :: v_dual_add_nc_u32 v50, v28, v50
	v_cmp_ne_u32_e32 vcc_lo, v31, v51
	v_dual_sub_nc_u32 v31, v31, v51 :: v_dual_add_nc_u32 v52, s33, v52
	v_dual_add_nc_u32 v29, v30, v29 :: v_dual_bitop2_b32 v51, -4, v50 bitop3:0x40
	s_and_b32 s2, s59, vcc_lo
	v_ashrrev_i32_e32 v50, 2, v50
	v_cndmask_b32_e64 v53, 0, 1, s2
	v_dual_add_nc_u32 v31, s34, v31 :: v_dual_bitop2_b32 v54, -4, v29 bitop3:0x40
	v_cmp_ne_u32_e32 vcc_lo, v28, v51
	v_dual_sub_nc_u32 v28, v28, v51 :: v_dual_ashrrev_i32 v29, 2, v29
	v_cmp_ne_u32_e64 s2, v30, v54
	v_sub_nc_u32_e32 v51, v30, v54
	v_dual_add_nc_u32 v50, s33, v50 :: v_dual_add_nc_u32 v54, s34, v28
	v_add_nc_u32_e32 v55, s33, v29
	s_and_b32 s2, s59, s2
	v_add_nc_u32_e32 v28, s34, v51
	v_cndmask_b32_e64 v51, 0, 1, s2
	s_and_b32 s2, s59, vcc_lo
	v_mad_nc_i64_i32 v[30:31], v31, s28, v[0:1]
	v_cndmask_b32_e64 v56, 0, 1, s2
	v_mad_nc_i64_i32 v[26:27], v49, s8, v[26:27]
	v_sub_nc_u32_e32 v51, v55, v51
	v_mad_nc_i64_i32 v[28:29], v28, s28, v[0:1]
	v_mad_nc_i64_i32 v[0:1], v54, s28, v[0:1]
	v_dual_sub_nc_u32 v49, v50, v56 :: v_dual_sub_nc_u32 v50, v52, v53
	v_mad_u32_u24 v32, 0x110, v48, v32
	v_lshl_add_u64 v[24:25], v[24:25], 1, s[6:7]
	v_lshl_add_u64 v[26:27], v[26:27], 1, s[6:7]
	v_mad_nc_i64_i32 v[28:29], v51, s8, v[28:29]
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
.LBB0_48:
	s_wait_alu depctr_vm_vsrc(0)
	s_clause 0x1
	scratch_load_b32 v0, off, off offset:560 th:TH_LOAD_LU nv
	scratch_load_b32 v5, off, off offset:556 th:TH_LOAD_LU nv
	v_cmp_gt_f32_e32 vcc_lo, 0x800000, v204
	v_cmp_gt_f32_e64 s2, 0x800000, v205
	s_load_b64 s[6:7], s[0:1], 0x40 nv
	s_mul_i32 s8, s3, s31
	s_wait_xcnt 0x0
	v_cmp_gt_i32_e64 s0, s57, v192
	v_cndmask_b32_e64 v2, 0, 32, vcc_lo
	v_cndmask_b32_e64 v4, 0, 32, s2
	v_cndmask_b32_e64 v1, 0, 0x42000000, vcc_lo
	v_cndmask_b32_e64 v3, 0, 0x42000000, s2
	v_mad_u32 v6, v192, s29, s8
	v_ldexp_f32 v2, v204, v2
	v_ldexp_f32 v4, v205, v4
	s_wait_loadcnt 0x2
	v_cmp_eq_u32_e32 vcc_lo, 0, v57
	v_cmp_gt_i32_e64 s1, s57, v193
	s_add_co_i32 s2, s8, s31
	v_log_f32_e32 v2, v2
	v_log_f32_e32 v4, v4
	s_and_b32 s0, vcc_lo, s0
	s_ashr_i32 s3, s2, 31
	s_and_b32 vcc_lo, vcc_lo, s1
	s_lshl_b32 s5, s2, 27
	s_lshr_b64 s[2:3], s[2:3], 5
	s_and_b64 s[2:3], s[2:3], 0x1ffffffffffffff
	v_nop
	v_dual_sub_f32 v1, v2, v1 :: v_dual_sub_f32 v2, v4, v3
	v_dual_mul_f32 v1, 0x3f317218, v1 :: v_dual_mul_f32 v2, 0x3f317218, v2
	s_set_vgpr_msb 4
	v_dual_add_f32 v1, v1, v50 /*v306*/ :: v_dual_add_f32 v2, v2, v51 /*v307*/
	s_set_vgpr_msb 0x400
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v0, 2, v0
	s_wait_loadcnt 0x0
	v_sub_nc_u32_e32 v0, v5, v0
	v_mul_lo_u32 v5, v193, s29
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
	s_endpgm
.Lfunc_end0:
	.size	kn_fmha_fwd_prefill_a16w16_m32x8_bshd_0, .Lfunc_end0-kn_fmha_fwd_prefill_a16w16_m32x8_bshd_0
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel kn_fmha_fwd_prefill_a16w16_m32x8_bshd_0
		.amdhsa_group_segment_fixed_size 327680
		.amdhsa_private_segment_fixed_size 592
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
		.amdhsa_enable_private_segment 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 1
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 1024
		.amdhsa_next_free_sgpr 98
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

	.set .Lkn_fmha_fwd_prefill_a16w16_m32x8_bshd_0.num_vgpr, 1024
	.set .Lkn_fmha_fwd_prefill_a16w16_m32x8_bshd_0.num_agpr, 0
	.set .Lkn_fmha_fwd_prefill_a16w16_m32x8_bshd_0.numbered_sgpr, 98
	.set .Lkn_fmha_fwd_prefill_a16w16_m32x8_bshd_0.num_named_barrier, 0
	.set .Lkn_fmha_fwd_prefill_a16w16_m32x8_bshd_0.private_seg_size, 592
	.set .Lkn_fmha_fwd_prefill_a16w16_m32x8_bshd_0.uses_vcc, 1
	.set .Lkn_fmha_fwd_prefill_a16w16_m32x8_bshd_0.uses_flat_scratch, 1
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
    .private_segment_fixed_size: 592
    .reqd_workgroup_size:
      - 128
      - 1
      - 1
    .sgpr_count:     100
    .sgpr_spill_count: 0
    .symbol:         kn_fmha_fwd_prefill_a16w16_m32x8_bshd_0.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     1024
    .vgpr_spill_count: 476
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa-unknown-gfx1250
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
