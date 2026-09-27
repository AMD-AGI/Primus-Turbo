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
	v_and_b32_e32 v156, 15, v0
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
	s_mov_b32 s40, 16
	s_mov_b32 s51, 0
	v_and_b32_e32 v157, 31, v0
	s_mul_f32 s7, s7, 0x4f7ffffe
	v_lshlrev_b32_e32 v129, 1, v157
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
	s_sub_co_i32 s21, s15, s14
	s_cmp_ge_u32 s15, s14
	s_cselect_b32 s3, s18, s3
	s_cselect_b32 s15, s21, s15
	s_add_co_i32 s18, s3, 1
	s_cmp_ge_u32 s15, s14
	s_clause 0x2
	s_load_b64 s[14:15], s[0:1], 0x10 nv
	s_load_b64 s[66:67], s[0:1], 0x20 nv
	s_load_b64 s[68:69], s[0:1], 0x30 nv
	s_cselect_b32 s3, s18, s3
	s_mov_b32 s21, 0xffff0000
	s_xor_b32 s3, s3, s20
	s_mov_b32 s18, -1
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
	s_and_b32 s19, s91, 28
	s_lshr_b32 s20, s91, 2
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
	s_sub_co_i32 s42, s13, s17
	s_add_co_i32 s37, s16, s2
	s_lshl_b32 s2, s91, 5
	s_lshl_b32 s90, s37, 8
	s_lshl_b32 s34, s42, 2
	s_add_co_i32 s88, s90, s2
	s_mov_b32 s60, s6
	s_cmp_lt_i32 s88, 0
	s_mov_b32 s38, s11
	s_cselect_b32 s59, -1, 0
	s_ashr_i32 s2, s88, 31
	s_add_co_i32 s13, s89, 0x20000
	s_lshr_b32 s2, s2, 30
	v_or_b32_e32 v6, s88, v156
	s_ashr_i32 s16, s88, 2
	s_ashr_i32 s23, s5, 31
	s_add_co_i32 s24, s33, s16
	s_ashr_i32 s35, s34, 31
	v_or_b32_e32 v2, 16, v6
	s_ashr_i32 s27, s9, 31
	s_ashr_i32 s25, s24, 31
	s_sub_co_i32 s17, s57, s16
	s_mul_u64 s[44:45], s[34:35], s[26:27]
	v_add_nc_u32_e32 v3, s2, v2
	v_dual_ashrrev_i32 v1, 31, v6 :: v_dual_bitop2_b32 v7, 16, v0 bitop3:0x40
	s_mul_u64 s[24:25], s[24:25], s[22:23]
	s_max_i32 s16, s17, 0
	v_dual_lshrrev_b32 v1, 30, v1 :: v_dual_bitop2_b32 v4, -4, v3 bitop3:0x40
	v_mad_u32_u24 v134, 0x110, v156, v7
	s_lshl_b64 s[44:45], s[44:45], 1
	s_lshl_b64 s[24:25], s[24:25], 1
	v_bfe_u32 v0, v0, 4, 1
	v_add_nc_u32_e32 v1, v6, v1
	v_cmp_ne_u32_e32 vcc_lo, v2, v4
	v_add_nc_u32_e32 v130, s13, v134
	s_add_nc_u64 s[14:15], s[14:15], s[24:25]
	v_dual_ashrrev_i32 v2, 2, v3 :: v_dual_bitop2_b32 v5, -4, v1 bitop3:0x40
	s_and_b32 vcc_lo, s59, vcc_lo
	s_add_nc_u64 s[14:15], s[14:15], s[44:45]
	v_ashrrev_i32_e32 v1, 2, v1
	v_subrev_co_ci_u32_e64 v155, null, 0, v2, vcc_lo
	v_cmp_ne_u32_e64 s2, v6, v5
	v_cvt_pk_bf16_f32 v128, s4, s4
	v_lshlrev_b32_e32 v152, 3, v0
	s_set_vgpr_msb 64
	v_or_b32_e32 v183 /*v439*/, 0x20000, v134
	s_bitset1_b32 s15, 31
	s_and_b32 s2, s59, s2
	s_cmp_lg_u32 s5, 0x80000000
	s_set_vgpr_msb 0x4000
	v_subrev_co_ci_u32_e64 v154, null, 0, v1, s2
	s_cselect_b32 s23, s23, 0
	s_cselect_b32 s22, s5, 0x200
	s_cmp_lg_u32 s9, 0x80000000
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x3
	scratch_store_b32 off, v6, off offset:908 nv
	scratch_store_b32 off, v1, off offset:912 nv
	scratch_store_b32 off, v7, off offset:932 nv
	scratch_store_b32 off, v0, off offset:924 nv
	s_cselect_b32 s25, s9, 0x80
	s_cselect_b32 s5, s27, 0
	s_bfe_i32 s24, s37, 0x10017
	s_lshl_b32 s9, s22, 16
	s_lshr_b64 s[26:27], s[22:23], 16
	s_and_b32 s5, s5, 0xffff
	s_lshr_b32 s35, s24, 30
	s_and_b32 s27, s26, 0xffff0000
	s_or_b32 s26, s5, s9
	s_or_b32 s5, s35, s90
	s_lshr_b32 s17, s22, 16
	s_addk_co_i32 s5, 0xff
	s_sub_co_i32 s22, s58, s57
	s_ashr_i32 s5, s5, 2
	s_add_co_i32 s41, s57, -1
	s_add_co_i32 s44, s56, s22
	s_add_co_i32 s5, s5, s24
	s_add_co_i32 s35, s44, 1
	s_min_i32 s5, s5, s41
	s_ashr_i32 s65, s64, 31
	s_ashr_i32 s43, s42, 31
	s_ashr_i32 s37, s10, 31
	s_ashr_i32 s63, s7, 31
	s_add_co_i32 s5, s35, s5
	s_ashr_i32 s61, s6, 31
	s_ashr_i32 s39, s11, 31
	s_mul_u64 s[10:11], s[42:43], s[36:37]
	s_mul_u64 s[22:23], s[64:65], s[62:63]
	s_min_i32 s5, s5, s58
	s_mul_u64 s[6:7], s[64:65], s[60:61]
	s_mul_u64 s[36:37], s[42:43], s[38:39]
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
	s_clause 0x5
	scratch_store_b32 off, v154, off offset:916 nv
	scratch_store_b32 off, v155, off offset:920 nv
	scratch_store_b32 off, v156, off offset:928 nv
	scratch_store_b32 off, v157, off offset:936 nv
	scratch_store_b32 off, v134, off offset:940 nv
	scratch_store_b32 off, v152, off offset:104 nv
	s_cbranch_scc0 .LBB0_10
	s_mov_b32 s17, s51
	s_mov_b32 s18, s51
	s_mov_b32 s19, s51
	s_mov_b32 s48, s51
	s_mov_b32 s49, s51
	s_mov_b32 s50, s51
	s_cmp_lg_u32 s60, 0x80000000
	tensor_load_to_lds s[12:15], s[20:27], s[16:19], s[48:51]
	s_mov_b64 s[6:7], s[14:15]
	s_cselect_b32 s7, s61, 0
	s_cselect_b32 s49, s60, 0x80
	s_and_b32 s2, s91, 7
	s_and_b32 s50, s7, 0xffff
	s_lshl_b32 s17, s2, 4
	s_lshl_b32 s78, s2, 5
	s_mul_i32 s92, s2, 0x1100
	s_mul_i32 s93, s2, 0x1200
	s_sub_co_i32 s2, s65, s17
	s_mov_b32 s79, s51
	s_max_i32 s2, s2, 0
	s_mov_b64 s[54:55], s[14:15]
	s_lshl_b32 s2, s2, 16
	s_mov_b32 s6, s49
	s_or_b32 s46, s2, 0x7fff
	s_cmp_lg_u32 s62, 0x80000000
	s_mul_u64 s[18:19], s[6:7], s[78:79]
	s_cselect_b32 s41, s62, 0x80
	s_cselect_b32 s55, s63, 0
	s_mov_b32 s54, s41
	s_add_nc_u64 s[6:7], s[18:19], s[76:77]
	s_mul_u64 s[78:79], s[54:55], s[78:79]
	s_mov_b64 s[4:5], s[12:13]
	s_mov_b32 s47, 0x800000
	s_bitset1_b32 s93, 16
	s_add_nc_u64 s[80:81], s[78:79], s[74:75]
	s_mov_b32 s44, s20
	s_mov_b32 s45, s21
	s_mov_b64 s[52:53], s[12:13]
	s_mov_b32 s48, s40
	s_mov_b32 s5, s92
	s_bitset1_b32 s7, 31
	s_mov_b32 s36, 0xf510000
	s_mov_b32 s37, s21
	s_mov_b32 s43, s51
	s_mov_b32 s39, s47
	s_mov_b32 s38, s46
	s_mov_b32 s53, s93
	s_and_b32 s42, s55, 0xffff
	s_or_b32 s55, s81, 0x80000000
	s_mov_b32 s54, s80
	s_ashr_i32 s2, s90, 2
	s_add_co_i32 s11, s35, -1
	v_and_or_b32 v64, v157, 7, v152
	s_add_co_i32 s2, s11, s2
	s_set_vgpr_msb 64
	v_or_b32_e32 v177 /*v433*/, 0x20000, v134
	s_max_i32 s2, s2, 0
	s_add_nc_u64 s[80:81], s[66:67], s[70:71]
	s_add_co_i32 s2, s2, 1
	s_set_vgpr_msb 0x4000
	v_mul_u32_u24_e32 v64, 0x120, v64
	v_and_or_b32 v64, v129, 16, v64
	s_set_vgpr_msb 64
	v_or_b32_e32 v182 /*v438*/, 0x10000, v64
	v_or_b32_e32 v180 /*v436*/, 0x30000, v64
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
	tensor_load_to_lds s[4:7], s[44:51]
	tensor_load_to_lds s[52:55], s[36:43]
	s_wait_dscnt 0xf
	v_pk_mul_bf16 v3, v128, v3
	s_wait_dscnt 0xe
	v_pk_mul_bf16 v7, v128, v7
	v_pk_mul_bf16 v6, v128, v6
	v_pk_mul_bf16 v5, v128, v5
	v_pk_mul_bf16 v4, v128, v4
	v_pk_mul_bf16 v2, v128, v2
	v_pk_mul_bf16 v1, v128, v1
	v_pk_mul_bf16 v0, v128, v0
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_store_b128 off, v[0:3], off offset:108 nv
	scratch_store_b128 off, v[4:7], off offset:124 nv
	s_wait_dscnt 0xc
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	v_pk_mul_bf16 v7, v128, v15
	v_pk_mul_bf16 v6, v128, v14
	v_pk_mul_bf16 v5, v128, v13
	v_pk_mul_bf16 v4, v128, v12
	v_pk_mul_bf16 v3, v128, v11
	v_pk_mul_bf16 v2, v128, v10
	v_pk_mul_bf16 v1, v128, v9
	v_pk_mul_bf16 v0, v128, v8
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_store_b128 off, v[0:3], off offset:140 nv
	scratch_store_b128 off, v[4:7], off offset:156 nv
	s_wait_dscnt 0xa
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	v_pk_mul_bf16 v7, v128, v23
	v_pk_mul_bf16 v6, v128, v22
	v_pk_mul_bf16 v5, v128, v21
	v_pk_mul_bf16 v4, v128, v20
	v_pk_mul_bf16 v3, v128, v19
	v_pk_mul_bf16 v2, v128, v18
	v_pk_mul_bf16 v1, v128, v17
	v_pk_mul_bf16 v0, v128, v16
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_store_b128 off, v[0:3], off offset:172 nv
	scratch_store_b128 off, v[4:7], off offset:188 nv
	s_wait_dscnt 0x8
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	v_pk_mul_bf16 v7, v128, v31
	v_pk_mul_bf16 v6, v128, v30
	v_pk_mul_bf16 v5, v128, v29
	v_pk_mul_bf16 v4, v128, v28
	v_pk_mul_bf16 v3, v128, v27
	v_pk_mul_bf16 v2, v128, v26
	v_pk_mul_bf16 v1, v128, v25
	v_pk_mul_bf16 v0, v128, v24
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_store_b128 off, v[0:3], off offset:204 nv
	scratch_store_b128 off, v[4:7], off offset:220 nv
	s_wait_dscnt 0x6
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	v_pk_mul_bf16 v7, v128, v39
	v_pk_mul_bf16 v6, v128, v38
	v_pk_mul_bf16 v5, v128, v37
	v_pk_mul_bf16 v4, v128, v36
	v_pk_mul_bf16 v3, v128, v35
	v_pk_mul_bf16 v2, v128, v34
	v_pk_mul_bf16 v1, v128, v33
	v_pk_mul_bf16 v0, v128, v32
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_store_b128 off, v[0:3], off offset:236 nv
	scratch_store_b128 off, v[4:7], off offset:252 nv
	s_wait_dscnt 0x4
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	v_pk_mul_bf16 v7, v128, v47
	v_pk_mul_bf16 v6, v128, v46
	v_pk_mul_bf16 v5, v128, v45
	v_pk_mul_bf16 v4, v128, v44
	v_pk_mul_bf16 v3, v128, v43
	v_pk_mul_bf16 v2, v128, v42
	v_pk_mul_bf16 v1, v128, v41
	v_pk_mul_bf16 v0, v128, v40
	s_ashr_i32 s4, s2, 31
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_store_b128 off, v[0:3], off offset:268 nv
	scratch_store_b128 off, v[4:7], off offset:284 nv
	s_lshr_b32 s4, s4, 25
	s_wait_dscnt 0x2
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	v_pk_mul_bf16 v7, v128, v55
	s_add_co_i32 s4, s2, s4
	v_pk_mul_bf16 v6, v128, v54
	v_pk_mul_bf16 v5, v128, v53
	v_pk_mul_bf16 v4, v128, v52
	v_pk_mul_bf16 v3, v128, v51
	v_pk_mul_bf16 v2, v128, v50
	v_pk_mul_bf16 v1, v128, v49
	v_pk_mul_bf16 v0, v128, v48
	s_and_b32 s6, s4, 0xffffff80
	s_add_co_i32 s5, s10, -1
	s_ashr_i32 s4, s4, 7
	s_cmp_lg_u32 s2, s6
	s_mov_b32 s53, s51
	s_cselect_b32 s6, -1, 0
	s_cmp_lt_i32 s2, 0
	s_add_nc_u64 s[54:55], s[68:69], s[72:73]
	s_cselect_b32 s2, -1, 0
	s_and_b32 s2, s2, s6
	s_sub_co_ci_u32 s2, s4, 0
	s_min_i32 s2, s2, s5
	s_max_i32 s52, s2, 0
	s_cmp_lt_i32 s2, 1
	s_wait_tensorcnt 0x0
	s_wait_storecnt_dscnt 0x0
	s_barrier_signal -1
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_store_b128 off, v[0:3], off offset:300 nv
	scratch_store_b128 off, v[4:7], off offset:316 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	v_pk_mul_bf16 v7, v128, v63
	v_pk_mul_bf16 v6, v128, v62
	v_pk_mul_bf16 v5, v128, v61
	v_pk_mul_bf16 v4, v128, v60
	v_pk_mul_bf16 v3, v128, v59
	v_pk_mul_bf16 v2, v128, v58
	v_pk_mul_bf16 v1, v128, v57
	v_pk_mul_bf16 v0, v128, v56
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_store_b128 off, v[0:3], off offset:332 nv
	scratch_store_b128 off, v[4:7], off offset:348 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	v_mov_b32_e32 v0, 0
	s_barrier_wait -1
	s_wait_storecnt 0x0
	s_clause 0x2
	scratch_store_b32 off, v128, off offset:944 nv
	scratch_store_b32 off, v129, off offset:948 nv
	scratch_store_b32 off, v130, off offset:952 nv
	s_cbranch_scc1 .LBB0_11
	v_dual_mov_b32 v1, v0 :: v_dual_mov_b32 v2, v0
	v_dual_mov_b32 v3, v0 :: v_dual_mov_b32 v4, v0
	v_dual_mov_b32 v5, v0 :: v_dual_mov_b32 v6, v0
	v_dual_mov_b32 v7, v0 :: v_dual_mov_b32 v234, v0
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
	v_dual_mov_b32 v178 /*v434*/, 0xf149f2ca :: v_dual_mov_b32 v181 /*v437*/, v134
	s_wait_alu depctr_vm_vsrc(1)
	s_set_vgpr_msb 0x4001
	v_dual_mov_b32 v128, v180 /*v436*/ :: v_dual_mov_b32 v129, v177 /*v433*/
	s_set_vgpr_msb 0x140
	v_mov_b32_e32 v179 /*v435*/, 0xf149f2ca
	s_set_vgpr_msb 0x4000
	v_mov_b32_e32 v235, v0
	s_lshl_b64 s[82:83], s[52:53], 7
	s_mov_b32 s4, 1
	s_add_co_i32 s94, s9, 0xffffff80
	s_add_co_i32 s84, s64, 0x80
	s_mov_b32 s48, 16
	s_mov_b32 s45, 0xffff0000
	s_mov_b32 s44, 0x7510000
	s_mov_b64 s[86:87], 0xffffffffffffff80
	s_mov_b32 s95, 0x76543210
	s_mov_b32 s56, 0x3fb8aa3b
	s_mov_b32 s96, 1
	s_branch .LBB0_4
.LBB0_3:
	s_set_vgpr_msb 64
	v_cvt_pk_bf16_f32 v23 /*v279*/, v247, v253
	v_cvt_pk_bf16_f32 v8 /*v264*/, v154, v156
	v_cvt_pk_bf16_f32 v16 /*v272*/, v155, v157
	v_cvt_pk_bf16_f32 v31 /*v287*/, v248, v254
	v_cvt_pk_bf16_f32 v25 /*v281*/, v132, v136
	v_cvt_pk_bf16_f32 v24 /*v280*/, v128, v130
	s_set_vgpr_msb 0x4000
	v_cvt_pk_bf16_f32 v156, v150, v162
	v_cvt_pk_bf16_f32 v155, v142, v146
	v_cvt_pk_bf16_f32 v254, v151, v163
	v_cvt_pk_bf16_f32 v253, v143, v147
	v_cvt_pk_bf16_f32 v128, v144, v148
	v_cvt_pk_bf16_f32 v136, v145, v149
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b128 v[144:147], off, off offset:748 th:TH_LOAD_LU nv
	scratch_load_b128 v[148:151], off, off offset:764 th:TH_LOAD_LU nv
	s_set_vgpr_msb 64
	v_cvt_pk_bf16_f32 v15 /*v271*/, v246, v252
	v_cvt_pk_bf16_f32 v14 /*v270*/, v232, v240
	v_cvt_pk_bf16_f32 v22 /*v278*/, v233, v241
	v_cvt_pk_bf16_f32 v13 /*v269*/, v222, v228
	v_cvt_pk_bf16_f32 v21 /*v277*/, v223, v229
	v_cvt_pk_bf16_f32 v12 /*v268*/, v204, v212
	v_cvt_pk_bf16_f32 v20 /*v276*/, v205, v213
	v_cvt_pk_bf16_f32 v11 /*v267*/, v180, v188
	v_cvt_pk_bf16_f32 v19 /*v275*/, v181, v189
	v_cvt_pk_bf16_f32 v10 /*v266*/, v164, v172
	v_cvt_pk_bf16_f32 v18 /*v274*/, v165, v173
	v_cvt_pk_bf16_f32 v9 /*v265*/, v192, v198
	v_cvt_pk_bf16_f32 v17 /*v273*/, v193, v199
	s_set_vgpr_msb 0x4005
	v_wmma_f32_16x16x32_bf16 v[88:95], v[56:63] /*v[312:319]*/, v[8:15] /*v[264:271]*/, v[88:95]
	s_set_vgpr_msb 0x540
	v_cvt_pk_bf16_f32 v30 /*v286*/, v234, v242
	s_set_vgpr_msb 0x4000
	v_cvt_pk_bf16_f32 v193, v249, v255
	v_cvt_pk_bf16_f32 v192, v235, v243
	s_set_vgpr_msb 64
	v_cvt_pk_bf16_f32 v29 /*v285*/, v190, v210
	v_cvt_pk_bf16_f32 v28 /*v284*/, v174, v186
	v_cvt_pk_bf16_f32 v27 /*v283*/, v194, v160
	v_cvt_pk_bf16_f32 v26 /*v282*/, v140, v158
	s_set_vgpr_msb 0x4005
	v_wmma_f32_16x16x32_bf16 v[24:31], v[56:63] /*v[312:319]*/, v[16:23] /*v[272:279]*/, v[24:31]
	s_set_vgpr_msb 0x500
	v_cvt_pk_bf16_f32 v191, v191, v211
	v_cvt_pk_bf16_f32 v190, v175, v187
	v_cvt_pk_bf16_f32 v189, v195, v161
	v_cvt_pk_bf16_f32 v188, v141, v159
	v_cvt_pk_bf16_f32 v187, v133, v137
	v_cvt_pk_bf16_f32 v186, v129, v131
	v_cvt_pk_bf16_f32 v161, v224, v236
	s_set_vgpr_msb 5
	v_wmma_f32_16x16x32_bf16 v[120:127], v[184:191] /*v[440:447]*/, v[8:15] /*v[264:271]*/, v[120:127]
	s_set_vgpr_msb 0x500
	v_cvt_pk_bf16_f32 v160, v218, v152
	v_cvt_pk_bf16_f32 v159, v206, v214
	v_cvt_pk_bf16_f32 v158, v182, v200
	v_cvt_pk_bf16_f32 v157, v166, v176
	v_cvt_pk_bf16_f32 v154, v134, v138
	v_cvt_pk_bf16_f32 v255, v167, v177
	v_cvt_pk_bf16_f32 v252, v135, v139
	s_set_vgpr_msb 5
	v_wmma_f32_16x16x32_bf16 v[56:63], v[184:191] /*v[440:447]*/, v[16:23] /*v[272:279]*/, v[56:63]
	s_set_vgpr_msb 0x500
	v_cvt_pk_bf16_f32 v135, v244, v250
	v_cvt_pk_bf16_f32 v134, v230, v238
	v_cvt_pk_bf16_f32 v133, v220, v226
	v_cvt_pk_bf16_f32 v132, v208, v216
	v_cvt_pk_bf16_f32 v131, v184, v202
	v_cvt_pk_bf16_f32 v130, v170, v178
	v_cvt_pk_bf16_f32 v129, v196, v168
	s_set_vgpr_msb 5
	v_wmma_f32_16x16x32_bf16 v[88:95], v[48:55] /*v[304:311]*/, v[24:31] /*v[280:287]*/, v[88:95]
	s_set_vgpr_msb 0x500
	v_cvt_pk_bf16_f32 v143, v245, v251
	v_cvt_pk_bf16_f32 v142, v231, v239
	v_cvt_pk_bf16_f32 v141, v221, v227
	v_cvt_pk_bf16_f32 v140, v209, v217
	v_cvt_pk_bf16_f32 v139, v185, v203
	v_cvt_pk_bf16_f32 v138, v171, v179
	v_cvt_pk_bf16_f32 v137, v197, v169
	s_set_vgpr_msb 1
	v_wmma_f32_16x16x32_bf16 v[24:31], v[48:55] /*v[304:311]*/, v[186:193], v[24:31]
	s_set_vgpr_msb 0x141
	v_dual_mov_b32 v178 /*v434*/, v32 /*v288*/ :: v_dual_mov_b32 v179 /*v435*/, v176 /*v432*/
	s_add_nc_u64 s[82:83], s[82:83], s[86:87]
	s_addk_co_i32 s94, 0xff80
	s_addk_co_i32 s84, 0x80
	s_add_co_i32 s96, s96, 1
	s_cmp_lg_u64 s[82:83], 0
	s_set_vgpr_msb 0x4105
	v_wmma_f32_16x16x32_bf16 v[120:127], v[0:7] /*v[256:263]*/, v[24:31] /*v[280:287]*/, v[120:127]
	s_set_vgpr_msb 0x501
	v_wmma_f32_16x16x32_bf16 v[56:63], v[0:7] /*v[256:263]*/, v[186:193], v[56:63]
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x140
	v_cvt_pk_bf16_f32 v3 /*v259*/, v225, v237
	v_cvt_pk_bf16_f32 v2 /*v258*/, v219, v153
	v_cvt_pk_bf16_f32 v1 /*v257*/, v207, v215
	v_cvt_pk_bf16_f32 v0 /*v256*/, v183, v201
	s_set_vgpr_msb 0x4001
	v_wmma_f32_16x16x32_bf16 v[88:95], v[40:47] /*v[296:303]*/, v[154:161], v[88:95]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[40:47] /*v[296:303]*/, v[252:259], v[24:31]
	s_set_vgpr_msb 0x105
	v_wmma_f32_16x16x32_bf16 v[112:119], v[152:159] /*v[408:415]*/, v[8:15] /*v[264:271]*/, v[112:119]
	v_wmma_f32_16x16x32_bf16 v[48:55], v[152:159] /*v[408:415]*/, v[16:23] /*v[272:279]*/, v[48:55]
	v_wmma_f32_16x16x32_bf16 v[104:111], v[120:127] /*v[376:383]*/, v[8:15] /*v[264:271]*/, v[104:111]
	v_wmma_f32_16x16x32_bf16 v[40:47], v[120:127] /*v[376:383]*/, v[16:23] /*v[272:279]*/, v[40:47]
	v_wmma_f32_16x16x32_bf16 v[96:103], v[88:95] /*v[344:351]*/, v[8:15] /*v[264:271]*/, v[96:103]
	v_wmma_f32_16x16x32_bf16 v[32:39], v[88:95] /*v[344:351]*/, v[16:23] /*v[272:279]*/, v[32:39]
	v_wmma_f32_16x16x32_bf16 v[112:119], v[144:151] /*v[400:407]*/, v[24:31] /*v[280:287]*/, v[112:119]
	s_set_vgpr_msb 0x501
	v_wmma_f32_16x16x32_bf16 v[48:55], v[144:151] /*v[400:407]*/, v[186:193], v[48:55]
	s_set_vgpr_msb 0x105
	v_wmma_f32_16x16x32_bf16 v[104:111], v[112:119] /*v[368:375]*/, v[24:31] /*v[280:287]*/, v[104:111]
	s_set_vgpr_msb 0x501
	v_wmma_f32_16x16x32_bf16 v[40:47], v[112:119] /*v[368:375]*/, v[186:193], v[40:47]
	s_set_vgpr_msb 0x105
	v_wmma_f32_16x16x32_bf16 v[96:103], v[80:87] /*v[336:343]*/, v[24:31] /*v[280:287]*/, v[96:103]
	s_set_vgpr_msb 0x501
	v_wmma_f32_16x16x32_bf16 v[32:39], v[80:87] /*v[336:343]*/, v[186:193], v[32:39]
	v_wmma_f32_16x16x32_bf16 v[120:127], v[168:175] /*v[424:431]*/, v[154:161], v[120:127]
	v_wmma_f32_16x16x32_bf16 v[56:63], v[168:175] /*v[424:431]*/, v[252:259], v[56:63]
	v_wmma_f32_16x16x32_bf16 v[112:119], v[136:143] /*v[392:399]*/, v[154:161], v[112:119]
	v_wmma_f32_16x16x32_bf16 v[48:55], v[136:143] /*v[392:399]*/, v[252:259], v[48:55]
	v_wmma_f32_16x16x32_bf16 v[104:111], v[104:111] /*v[360:367]*/, v[154:161], v[104:111]
	v_wmma_f32_16x16x32_bf16 v[40:47], v[104:111] /*v[360:367]*/, v[252:259], v[40:47]
	v_wmma_f32_16x16x32_bf16 v[96:103], v[72:79] /*v[328:335]*/, v[154:161], v[96:103]
	v_wmma_f32_16x16x32_bf16 v[32:39], v[72:79] /*v[328:335]*/, v[252:259], v[32:39]
	v_wmma_f32_16x16x32_bf16 v[120:127], v[160:167] /*v[416:423]*/, v[128:135], v[120:127]
	v_wmma_f32_16x16x32_bf16 v[56:63], v[160:167] /*v[416:423]*/, v[136:143], v[56:63]
	v_wmma_f32_16x16x32_bf16 v[112:119], v[128:135] /*v[384:391]*/, v[128:135], v[112:119]
	v_wmma_f32_16x16x32_bf16 v[48:55], v[128:135] /*v[384:391]*/, v[136:143], v[48:55]
	v_wmma_f32_16x16x32_bf16 v[104:111], v[96:103] /*v[352:359]*/, v[128:135], v[104:111]
	v_wmma_f32_16x16x32_bf16 v[40:47], v[96:103] /*v[352:359]*/, v[136:143], v[40:47]
	v_wmma_f32_16x16x32_bf16 v[96:103], v[64:71] /*v[320:327]*/, v[128:135], v[96:103]
	s_set_vgpr_msb 0x100
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[88:95], v[144:151], v[128:135], v[88:95]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[144:151], v[136:143], v[24:31]
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b128 v[144:147], off, off offset:716 th:TH_LOAD_LU nv
	scratch_load_b128 v[148:151], off, off offset:732 th:TH_LOAD_LU nv
	s_set_vgpr_msb 1
	v_wmma_f32_16x16x32_bf16 v[32:39], v[64:71] /*v[320:327]*/, v[136:143], v[32:39]
	s_set_vgpr_msb 0x104
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[80:87], v[144:151], v[8:15] /*v[264:271]*/, v[80:87]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[144:151], v[16:23] /*v[272:279]*/, v[16:23]
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b128 v[144:147], off, off offset:684 th:TH_LOAD_LU nv
	scratch_load_b128 v[148:151], off, off offset:700 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[80:87], v[144:151], v[24:31] /*v[280:287]*/, v[80:87]
	s_set_vgpr_msb 0x400
	v_wmma_f32_16x16x32_bf16 v[16:23], v[144:151], v[186:193], v[16:23]
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b128 v[144:147], off, off offset:652 th:TH_LOAD_LU nv
	scratch_load_b128 v[148:151], off, off offset:668 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[80:87], v[144:151], v[154:161], v[80:87]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[144:151], v[252:259], v[16:23]
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b128 v[144:147], off, off offset:620 th:TH_LOAD_LU nv
	scratch_load_b128 v[148:151], off, off offset:636 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[80:87], v[144:151], v[128:135], v[80:87]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[144:151], v[136:143], v[16:23]
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b128 v[144:147], off, off offset:588 th:TH_LOAD_LU nv
	scratch_load_b128 v[148:151], off, off offset:604 th:TH_LOAD_LU nv
	s_set_vgpr_msb 4
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[72:79], v[144:151], v[8:15] /*v[264:271]*/, v[72:79]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[144:151], v[16:23] /*v[272:279]*/, v[8:15]
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b128 v[144:147], off, off offset:556 th:TH_LOAD_LU nv
	scratch_load_b128 v[148:151], off, off offset:572 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[72:79], v[144:151], v[24:31] /*v[280:287]*/, v[72:79]
	s_set_vgpr_msb 0x400
	v_wmma_f32_16x16x32_bf16 v[8:15], v[144:151], v[186:193], v[8:15]
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b128 v[144:147], off, off offset:524 th:TH_LOAD_LU nv
	scratch_load_b128 v[148:151], off, off offset:540 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[72:79], v[144:151], v[154:161], v[72:79]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[144:151], v[252:259], v[8:15]
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b128 v[144:147], off, off offset:492 th:TH_LOAD_LU nv
	scratch_load_b128 v[148:151], off, off offset:508 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[72:79], v[144:151], v[128:135], v[72:79]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[144:151], v[136:143], v[8:15]
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b128 v[144:147], off, off offset:460 th:TH_LOAD_LU nv
	scratch_load_b128 v[148:151], off, off offset:476 th:TH_LOAD_LU nv
	s_set_vgpr_msb 4
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[64:71], v[144:151], v[8:15] /*v[264:271]*/, v[64:71]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[144:151], v[16:23] /*v[272:279]*/, v[0:7]
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b128 v[144:147], off, off offset:428 th:TH_LOAD_LU nv
	scratch_load_b128 v[148:151], off, off offset:444 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[64:71], v[144:151], v[24:31] /*v[280:287]*/, v[64:71]
	s_set_vgpr_msb 0x400
	v_wmma_f32_16x16x32_bf16 v[0:7], v[144:151], v[186:193], v[0:7]
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b128 v[144:147], off, off offset:396 th:TH_LOAD_LU nv
	scratch_load_b128 v[148:151], off, off offset:412 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[64:71], v[144:151], v[154:161], v[64:71]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[144:151], v[252:259], v[0:7]
	s_wait_alu depctr_va_vdst(0)
	scratch_load_b64 v[144:145], off, off th:TH_LOAD_LU nv
	s_wait_loadcnt_dscnt 0x0
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 17
	v_pk_fma_f32 v[144:145], v[196:197] /*v[452:453]*/, v[144:145], v[194:195] /*v[450:451]*/
	v_pk_add_f32 v[234:235], v[192:193] /*v[448:449]*/, v[144:145]
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b128 v[144:147], off, off offset:364 th:TH_LOAD_LU nv
	scratch_load_b128 v[148:151], off, off offset:380 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0x1100
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[64:71], v[144:151], v[128:135], v[64:71]
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 1
	v_dual_mov_b32 v128, v180 /*v436*/ :: v_dual_mov_b32 v129, v177 /*v433*/
	s_set_vgpr_msb 0x100
	v_wmma_f32_16x16x32_bf16 v[0:7], v[144:151], v[136:143], v[0:7]
	s_cbranch_scc0 .LBB0_13
.LBB0_4:
	s_set_vgpr_msb 0x41
	v_dual_mov_b32 v177 /*v433*/, v181 /*v437*/ :: v_dual_mov_b32 v180 /*v436*/, v182 /*v438*/
	s_set_vgpr_msb 0x4140
	v_dual_mov_b32 v181 /*v437*/, v129 :: v_dual_mov_b32 v182 /*v438*/, v128
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x4
	scratch_store_b128 off, v[16:19], off offset:72 nv
	scratch_store_b128 off, v[20:23], off offset:88 nv
	scratch_store_b128 off, v[8:11], off offset:40 nv
	scratch_store_b128 off, v[12:15], off offset:56 nv
	scratch_store_b64 off, v[234:235], off nv
	s_wait_tensorcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	s_cmp_ge_i32 s96, s10
	s_set_vgpr_msb 0x4000
	s_cbranch_scc1 .LBB0_6
	v_med3_i32 v128, s94, 0, 0x80
	s_ashr_i32 s85, s84, 31
	s_lshr_b32 s2, s96, 31
	s_mul_u64 s[6:7], s[84:85], s[62:63]
	s_add_co_i32 s2, s96, s2
	v_readfirstlane_b32 s5, v128
	s_lshl_b64 s[6:7], s[6:7], 1
	s_and_b32 s2, s2, 0x7ffe
	s_add_nc_u64 s[38:39], s[54:55], s[6:7]
	s_mul_u64 s[6:7], s[84:85], s[60:61]
	s_sub_co_i32 s5, s5, s17
	s_lshl_b64 s[6:7], s[6:7], 1
	s_sub_co_i32 s2, s96, s2
	s_add_nc_u64 s[6:7], s[80:81], s[6:7]
	s_max_i32 s37, s5, 0
	s_lshl_b32 s2, s2, 17
	s_add_nc_u64 s[6:7], s[18:19], s[6:7]
	s_lshl_b32 s37, s37, 16
	s_or_b32 s5, s92, s2
	s_bitset1_b32 s7, 31
	s_or_b32 s46, s37, 0x7fff
	s_mov_b32 s37, s45
	tensor_load_to_lds s[4:7], s[44:51]
	s_add_nc_u64 s[6:7], s[78:79], s[38:39]
	s_or_b32 s5, s93, s2
	s_bitset1_b32 s7, 31
	s_mov_b32 s38, s46
	s_mov_b32 s39, s47
	s_mov_b32 s40, s48
	s_mov_b32 s43, s51
	tensor_load_to_lds s[4:7], s[36:43]
.LBB0_6:
	s_wait_alu depctr_va_vdst(0) depctr_vm_vsrc(5)
	s_set_vgpr_msb 1
	ds_load_b128 v[128:131], v177 /*v433*/
	ds_load_b128 v[132:135], v177 /*v433*/ offset:32
	ds_load_b128 v[136:139], v177 /*v433*/ offset:64
	ds_load_b128 v[140:143], v177 /*v433*/ offset:96
	ds_load_b128 v[144:147], v177 /*v433*/ offset:128
	ds_load_b128 v[148:151], v177 /*v433*/ offset:160
	ds_load_b128 v[152:155], v177 /*v433*/ offset:192
	ds_load_b128 v[156:159], v177 /*v433*/ offset:224
	ds_load_b128 v[160:163], v177 /*v433*/ offset:4352
	ds_load_b128 v[164:167], v177 /*v433*/ offset:4384
	ds_load_b128 v[168:171], v177 /*v433*/ offset:4416
	ds_load_b128 v[172:175], v177 /*v433*/ offset:4448
	ds_load_b128 v[176:179], v177 /*v433*/ offset:4480
	ds_load_b128 v[180:183], v177 /*v433*/ offset:4512
	ds_load_b128 v[184:187], v177 /*v433*/ offset:4544
	ds_load_b128 v[188:191], v177 /*v433*/ offset:4576
	ds_load_b128 v[200:203], v177 /*v433*/ offset:8704
	ds_load_b128 v[204:207], v177 /*v433*/ offset:8736
	ds_load_b128 v[208:211], v177 /*v433*/ offset:8768
	ds_load_b128 v[212:215], v177 /*v433*/ offset:8800
	ds_load_b128 v[216:219], v177 /*v433*/ offset:8832
	ds_load_b128 v[220:223], v177 /*v433*/ offset:8864
	ds_load_b128 v[224:227], v177 /*v433*/ offset:8896
	ds_load_b128 v[228:231], v177 /*v433*/ offset:8928
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_b128 v[232:235], v177 /*v433*/ offset:13056
	ds_load_b128 v[236:239], v177 /*v433*/ offset:13088
	ds_load_b128 v[240:243], v177 /*v433*/ offset:13120
	ds_load_b128 v[244:247], v177 /*v433*/ offset:13152
	ds_load_b128 v[248:251], v177 /*v433*/ offset:13184
	ds_load_b128 v[252:255], v177 /*v433*/ offset:13216
	s_set_vgpr_msb 0x141
	ds_load_b128 v[0:3] /*v[256:259]*/, v177 /*v433*/ offset:13248
	ds_load_b128 v[4:7] /*v[260:263]*/, v177 /*v433*/ offset:13280
	ds_load_b128 v[8:11] /*v[264:267]*/, v177 /*v433*/ offset:17408
	ds_load_b128 v[12:15] /*v[268:271]*/, v177 /*v433*/ offset:17440
	ds_load_b128 v[16:19] /*v[272:275]*/, v177 /*v433*/ offset:17472
	ds_load_b128 v[20:23] /*v[276:279]*/, v177 /*v433*/ offset:17504
	ds_load_b128 v[24:27] /*v[280:283]*/, v177 /*v433*/ offset:17536
	ds_load_b128 v[28:31] /*v[284:287]*/, v177 /*v433*/ offset:17568
	ds_load_b128 v[32:35] /*v[288:291]*/, v177 /*v433*/ offset:17600
	ds_load_b128 v[36:39] /*v[292:295]*/, v177 /*v433*/ offset:17632
	ds_load_b128 v[40:43] /*v[296:299]*/, v177 /*v433*/ offset:21760
	ds_load_b128 v[44:47] /*v[300:303]*/, v177 /*v433*/ offset:21792
	ds_load_b128 v[48:51] /*v[304:307]*/, v177 /*v433*/ offset:21824
	ds_load_b128 v[52:55] /*v[308:311]*/, v177 /*v433*/ offset:21856
	ds_load_b128 v[56:59] /*v[312:315]*/, v177 /*v433*/ offset:21888
	ds_load_b128 v[60:63] /*v[316:319]*/, v177 /*v433*/ offset:21920
	ds_load_b128 v[64:67] /*v[320:323]*/, v177 /*v433*/ offset:21952
	ds_load_b128 v[68:71] /*v[324:327]*/, v177 /*v433*/ offset:21984
	ds_load_b128 v[72:75] /*v[328:331]*/, v177 /*v433*/ offset:26112
	ds_load_b128 v[76:79] /*v[332:335]*/, v177 /*v433*/ offset:26144
	ds_load_b128 v[80:83] /*v[336:339]*/, v177 /*v433*/ offset:26176
	ds_load_b128 v[84:87] /*v[340:343]*/, v177 /*v433*/ offset:26208
	ds_load_b128 v[88:91] /*v[344:347]*/, v177 /*v433*/ offset:26240
	ds_load_b128 v[92:95] /*v[348:351]*/, v177 /*v433*/ offset:26272
	ds_load_b128 v[96:99] /*v[352:355]*/, v177 /*v433*/ offset:26304
	ds_load_b128 v[100:103] /*v[356:359]*/, v177 /*v433*/ offset:26336
	ds_load_b128 v[104:107] /*v[360:363]*/, v177 /*v433*/ offset:30464
	ds_load_b128 v[108:111] /*v[364:367]*/, v177 /*v433*/ offset:30496
	ds_load_b128 v[112:115] /*v[368:371]*/, v177 /*v433*/ offset:30528
	ds_load_b128 v[116:119] /*v[372:375]*/, v177 /*v433*/ offset:30560
	ds_load_b128 v[120:123] /*v[376:379]*/, v177 /*v433*/ offset:30592
	ds_load_b128 v[124:127] /*v[380:383]*/, v177 /*v433*/ offset:30624
	ds_load_b128 v[128:131] /*v[384:387]*/, v177 /*v433*/ offset:30656
	ds_load_b128 v[132:135] /*v[388:391]*/, v177 /*v433*/ offset:30688
	s_clause 0x12
	scratch_load_b128 v[136:139] /*v[392:395]*/, off, off offset:108 nv
	scratch_load_b128 v[140:143] /*v[396:399]*/, off, off offset:124 nv
	scratch_load_b128 v[168:171] /*v[424:427]*/, off, off offset:236 nv
	scratch_load_b128 v[172:175] /*v[428:431]*/, off, off offset:252 nv
	scratch_load_b128 v[144:147] /*v[400:403]*/, off, off offset:140 nv
	scratch_load_b128 v[148:151] /*v[404:407]*/, off, off offset:156 nv
	scratch_load_b128 v[184:187] /*v[440:443]*/, off, off offset:268 nv
	scratch_load_b128 v[188:191] /*v[444:447]*/, off, off offset:284 nv
	scratch_load_b128 v[152:155] /*v[408:411]*/, off, off offset:172 nv
	scratch_load_b128 v[156:159] /*v[412:415]*/, off, off offset:188 nv
	s_set_vgpr_msb 0x4100
	scratch_load_b128 v[8:11], off, off offset:300 nv
	scratch_load_b128 v[12:15], off, off offset:316 nv
	s_set_vgpr_msb 64
	scratch_load_b128 v[160:163] /*v[416:419]*/, off, off offset:204 nv
	scratch_load_b128 v[164:167] /*v[420:423]*/, off, off offset:220 nv
	s_set_vgpr_msb 0x4004
	scratch_load_b128 v[16:19], off, off offset:332 nv
	scratch_load_b128 v[20:23], off, off offset:348 nv
	s_wait_loadcnt_dscnt 0xe3e
	v_wmma_f32_16x16x32_bf16 v[192:199], v[128:135], v[136:143] /*v[392:399]*/, 0
	s_set_vgpr_msb 0x444
	s_wait_loadcnt 0xc
	v_wmma_f32_16x16x32_bf16 v[192:199] /*v[448:455]*/, v[128:135], v[168:175] /*v[424:431]*/, 0
	s_set_vgpr_msb 0x4404
	s_wait_loadcnt_dscnt 0xa3c
	v_wmma_f32_16x16x32_bf16 v[192:199], v[136:143], v[144:151] /*v[400:407]*/, v[192:199]
	s_set_vgpr_msb 0x454
	s_wait_loadcnt 0x8
	v_wmma_f32_16x16x32_bf16 v[192:199] /*v[448:455]*/, v[136:143], v[184:191] /*v[440:447]*/, v[192:199] /*v[448:455]*/
	s_set_vgpr_msb 0x5404
	s_wait_dscnt 0x2e
	v_wmma_f32_16x16x32_bf16 v[136:143], v[200:207], v[136:143] /*v[392:399]*/, 0
	s_set_vgpr_msb 0x444
	v_wmma_f32_16x16x32_bf16 v[208:215] /*v[464:471]*/, v[200:207], v[168:175] /*v[424:431]*/, 0
	s_set_vgpr_msb 0x4404
	s_wait_dscnt 0x2c
	v_wmma_f32_16x16x32_bf16 v[136:143], v[208:215], v[144:151] /*v[400:407]*/, v[136:143]
	s_set_vgpr_msb 0x454
	v_wmma_f32_16x16x32_bf16 v[208:215] /*v[464:471]*/, v[208:215], v[184:191] /*v[440:447]*/, v[208:215] /*v[464:471]*/
	s_set_vgpr_msb 0x5404
	s_wait_loadcnt_dscnt 0x62a
	v_wmma_f32_16x16x32_bf16 v[136:143], v[216:223], v[152:159] /*v[408:415]*/, v[136:143]
	s_set_vgpr_msb 0x450
	s_wait_loadcnt 0x4
	v_wmma_f32_16x16x32_bf16 v[208:215] /*v[464:471]*/, v[216:223], v[8:15], v[208:215] /*v[464:471]*/
	s_set_vgpr_msb 0x5005
	s_wait_dscnt 0x1e
	v_wmma_f32_16x16x32_bf16 v[214:221], v[8:15] /*v[264:271]*/, v[136:143] /*v[392:399]*/, 0
	s_set_vgpr_msb 0x545
	v_wmma_f32_16x16x32_bf16 v[224:231] /*v[480:487]*/, v[8:15] /*v[264:271]*/, v[168:175] /*v[424:431]*/, 0
	s_set_vgpr_msb 0x4504
	v_wmma_f32_16x16x32_bf16 v[192:199], v[144:151], v[152:159] /*v[408:415]*/, v[192:199]
	s_set_vgpr_msb 0x450
	v_wmma_f32_16x16x32_bf16 v[192:199] /*v[448:455]*/, v[144:151], v[8:15], v[192:199] /*v[448:455]*/
	s_set_vgpr_msb 0x5004
	v_wmma_f32_16x16x32_bf16 v[128:135], v[160:167], v[136:143] /*v[392:399]*/, 0
	v_wmma_f32_16x16x32_bf16 v[144:151], v[232:239], v[136:143] /*v[392:399]*/, 0
	s_set_vgpr_msb 0x405
	s_wait_dscnt 0x1c
	v_wmma_f32_16x16x32_bf16 v[214:221], v[16:23] /*v[272:279]*/, v[144:151] /*v[400:407]*/, v[214:221]
	s_set_vgpr_msb 0x555
	v_wmma_f32_16x16x32_bf16 v[224:231] /*v[480:487]*/, v[16:23] /*v[272:279]*/, v[184:191] /*v[440:447]*/, v[224:231] /*v[480:487]*/
	s_set_vgpr_msb 0x5544
	v_wmma_f32_16x16x32_bf16 v[200:207] /*v[456:463]*/, v[160:167], v[168:175] /*v[424:431]*/, 0
	s_set_vgpr_msb 0x4404
	v_wmma_f32_16x16x32_bf16 v[128:135], v[168:175], v[144:151] /*v[400:407]*/, v[128:135]
	s_set_vgpr_msb 0x444
	v_wmma_f32_16x16x32_bf16 v[216:223] /*v[472:479]*/, v[232:239], v[168:175] /*v[424:431]*/, 0
	s_set_vgpr_msb 0x4404
	v_wmma_f32_16x16x32_bf16 v[144:151], v[240:247], v[144:151] /*v[400:407]*/, v[144:151]
	s_set_vgpr_msb 0x405
	s_wait_dscnt 0x1a
	v_wmma_f32_16x16x32_bf16 v[214:221], v[24:31] /*v[280:287]*/, v[152:159] /*v[408:415]*/, v[214:221]
	s_set_vgpr_msb 0x551
	v_wmma_f32_16x16x32_bf16 v[224:231] /*v[480:487]*/, v[24:31] /*v[280:287]*/, v[8:15], v[224:231] /*v[480:487]*/
	s_set_vgpr_msb 0x5145
	s_wait_dscnt 0x16
	v_wmma_f32_16x16x32_bf16 v[8:15] /*v[264:271]*/, v[40:47] /*v[296:303]*/, v[136:143] /*v[392:399]*/, 0
	v_wmma_f32_16x16x32_bf16 v[232:239] /*v[488:495]*/, v[40:47] /*v[296:303]*/, v[168:175] /*v[424:431]*/, 0
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x32_bf16 v[16:23] /*v[272:279]*/, v[72:79] /*v[328:335]*/, v[136:143] /*v[392:399]*/, 0
	v_wmma_f32_16x16x32_bf16 v[240:247] /*v[496:503]*/, v[72:79] /*v[328:335]*/, v[168:175] /*v[424:431]*/, 0
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x32_bf16 v[24:31] /*v[280:287]*/, v[104:111] /*v[360:367]*/, v[136:143] /*v[392:399]*/, 0
	v_wmma_f32_16x16x32_bf16 v[248:255] /*v[504:511]*/, v[104:111] /*v[360:367]*/, v[168:175] /*v[424:431]*/, 0
	s_set_vgpr_msb 0x4554
	v_wmma_f32_16x16x32_bf16 v[200:207] /*v[456:463]*/, v[168:175], v[184:191] /*v[440:447]*/, v[200:207] /*v[456:463]*/
	s_set_vgpr_msb 0x5404
	v_wmma_f32_16x16x32_bf16 v[128:135], v[176:183], v[152:159] /*v[408:415]*/, v[128:135]
	s_set_vgpr_msb 0x454
	v_wmma_f32_16x16x32_bf16 v[216:223] /*v[472:479]*/, v[240:247], v[184:191] /*v[440:447]*/, v[216:223] /*v[472:479]*/
	s_set_vgpr_msb 0x5404
	v_wmma_f32_16x16x32_bf16 v[144:151], v[248:255], v[152:159] /*v[408:415]*/, v[144:151]
	s_set_vgpr_msb 0x455
	v_wmma_f32_16x16x32_bf16 v[8:15] /*v[264:271]*/, v[48:55] /*v[304:311]*/, v[144:151] /*v[400:407]*/, v[8:15] /*v[264:271]*/
	v_wmma_f32_16x16x32_bf16 v[232:239] /*v[488:495]*/, v[48:55] /*v[304:311]*/, v[184:191] /*v[440:447]*/, v[232:239] /*v[488:495]*/
	v_wmma_f32_16x16x32_bf16 v[16:23] /*v[272:279]*/, v[80:87] /*v[336:343]*/, v[144:151] /*v[400:407]*/, v[16:23] /*v[272:279]*/
	v_wmma_f32_16x16x32_bf16 v[240:247] /*v[496:503]*/, v[80:87] /*v[336:343]*/, v[184:191] /*v[440:447]*/, v[240:247] /*v[496:503]*/
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x32_bf16 v[24:31] /*v[280:287]*/, v[112:119] /*v[368:375]*/, v[144:151] /*v[400:407]*/, v[24:31] /*v[280:287]*/
	v_wmma_f32_16x16x32_bf16 v[248:255] /*v[504:511]*/, v[112:119] /*v[368:375]*/, v[184:191] /*v[440:447]*/, v[248:255] /*v[504:511]*/
	s_set_vgpr_msb 0x5504
	s_wait_loadcnt 0x2
	v_wmma_f32_16x16x32_bf16 v[192:199], v[152:159], v[160:167] /*v[416:423]*/, v[192:199]
	s_set_vgpr_msb 0x450
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[192:199] /*v[448:455]*/, v[152:159], v[16:23], v[192:199] /*v[448:455]*/
	v_wmma_f32_16x16x32_bf16 v[200:207] /*v[456:463]*/, v[176:183], v[8:15], v[200:207] /*v[456:463]*/
	s_set_vgpr_msb 0x5004
	v_wmma_f32_16x16x32_bf16 v[128:135], v[184:191], v[160:167] /*v[416:423]*/, v[128:135]
	v_wmma_f32_16x16x32_bf16 v[136:143], v[224:231], v[160:167] /*v[416:423]*/, v[136:143]
	s_set_vgpr_msb 0x450
	v_wmma_f32_16x16x32_bf16 v[216:223] /*v[472:479]*/, v[248:255], v[8:15], v[216:223] /*v[472:479]*/
	s_set_vgpr_msb 0x5005
	v_wmma_f32_16x16x32_bf16 v[144:151], v[0:7] /*v[256:263]*/, v[160:167] /*v[416:423]*/, v[144:151]
	v_wmma_f32_16x16x32_bf16 v[214:221], v[32:39] /*v[288:295]*/, v[160:167] /*v[416:423]*/, v[214:221]
	s_set_vgpr_msb 0x555
	v_wmma_f32_16x16x32_bf16 v[8:15] /*v[264:271]*/, v[56:63] /*v[312:319]*/, v[152:159] /*v[408:415]*/, v[8:15] /*v[264:271]*/
	s_set_vgpr_msb 0x5551
	v_wmma_f32_16x16x32_bf16 v[232:239] /*v[488:495]*/, v[56:63] /*v[312:319]*/, v[8:15], v[232:239] /*v[488:495]*/
	s_set_vgpr_msb 0x5155
	v_wmma_f32_16x16x32_bf16 v[16:23] /*v[272:279]*/, v[88:95] /*v[344:351]*/, v[152:159] /*v[408:415]*/, v[16:23] /*v[272:279]*/
	s_set_vgpr_msb 0x5551
	v_wmma_f32_16x16x32_bf16 v[240:247] /*v[496:503]*/, v[88:95] /*v[344:351]*/, v[8:15], v[240:247] /*v[496:503]*/
	s_set_vgpr_msb 0x5155
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[24:31] /*v[280:287]*/, v[120:127] /*v[376:383]*/, v[152:159] /*v[408:415]*/, v[24:31] /*v[280:287]*/
	s_set_vgpr_msb 0x5551
	v_wmma_f32_16x16x32_bf16 v[248:255] /*v[504:511]*/, v[120:127] /*v[376:383]*/, v[8:15], v[248:255] /*v[504:511]*/
	s_set_vgpr_msb 0x5150
	v_wmma_f32_16x16x32_bf16 v[200:207] /*v[456:463]*/, v[184:191], v[16:23], v[200:207] /*v[456:463]*/
	v_wmma_f32_16x16x32_bf16 v[208:215] /*v[464:471]*/, v[224:231], v[16:23], v[208:215] /*v[464:471]*/
	s_set_vgpr_msb 0x5051
	v_wmma_f32_16x16x32_bf16 v[216:223] /*v[472:479]*/, v[0:7] /*v[256:263]*/, v[16:23], v[216:223] /*v[472:479]*/
	v_wmma_f32_16x16x32_bf16 v[224:231] /*v[480:487]*/, v[32:39] /*v[288:295]*/, v[16:23], v[224:231] /*v[480:487]*/
	s_set_vgpr_msb 0x5155
	v_wmma_f32_16x16x32_bf16 v[8:15] /*v[264:271]*/, v[64:71] /*v[320:327]*/, v[160:167] /*v[416:423]*/, v[8:15] /*v[264:271]*/
	s_set_vgpr_msb 0x5551
	v_wmma_f32_16x16x32_bf16 v[232:239] /*v[488:495]*/, v[64:71] /*v[320:327]*/, v[16:23], v[232:239] /*v[488:495]*/
	s_set_vgpr_msb 0x5155
	v_wmma_f32_16x16x32_bf16 v[16:23] /*v[272:279]*/, v[96:103] /*v[352:359]*/, v[160:167] /*v[416:423]*/, v[16:23] /*v[272:279]*/
	s_set_vgpr_msb 0x5551
	v_wmma_f32_16x16x32_bf16 v[240:247] /*v[496:503]*/, v[96:103] /*v[352:359]*/, v[16:23], v[240:247] /*v[496:503]*/
	s_set_vgpr_msb 0x5155
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[24:31] /*v[280:287]*/, v[128:135] /*v[384:391]*/, v[160:167] /*v[416:423]*/, v[24:31] /*v[280:287]*/
	s_set_vgpr_msb 0x5551
	v_wmma_f32_16x16x32_bf16 v[248:255] /*v[504:511]*/, v[128:135] /*v[384:391]*/, v[16:23], v[248:255] /*v[504:511]*/
	s_wait_alu depctr_va_vdst(14)
	ds_load_tr16_b128 v[184:187] /*v[440:443]*/, v180 /*v436*/
	s_wait_alu depctr_va_vdst(11)
	ds_load_tr16_b128 v[152:155] /*v[408:411]*/, v180 /*v436*/ offset:32
	ds_load_tr16_b128 v[188:191] /*v[444:447]*/, v180 /*v436*/ offset:4608
	ds_load_tr16_b128 v[156:159] /*v[412:415]*/, v180 /*v436*/ offset:4640
	s_wait_alu depctr_va_vdst(7)
	ds_load_tr16_b128 v[0:3] /*v[256:259]*/, v180 /*v436*/ offset:9216
	ds_load_tr16_b128 v[144:147] /*v[400:403]*/, v180 /*v436*/ offset:9248
	ds_load_tr16_b128 v[4:7] /*v[260:263]*/, v180 /*v436*/ offset:13824
	ds_load_tr16_b128 v[148:151] /*v[404:407]*/, v180 /*v436*/ offset:13856
	ds_load_tr16_b128 v[168:171] /*v[424:427]*/, v180 /*v436*/ offset:18432
	ds_load_tr16_b128 v[136:139] /*v[392:395]*/, v180 /*v436*/ offset:18464
	ds_load_tr16_b128 v[172:175] /*v[428:431]*/, v180 /*v436*/ offset:23040
	ds_load_tr16_b128 v[140:143] /*v[396:399]*/, v180 /*v436*/ offset:23072
	s_wait_alu depctr_va_vdst(1)
	ds_load_tr16_b128 v[160:163] /*v[416:419]*/, v180 /*v436*/ offset:27648
	s_wait_alu depctr_va_vdst(0)
	ds_load_tr16_b128 v[128:131] /*v[384:387]*/, v180 /*v436*/ offset:27680
	ds_load_tr16_b128 v[164:167] /*v[420:423]*/, v180 /*v436*/ offset:32256
	ds_load_tr16_b128 v[132:135] /*v[388:391]*/, v180 /*v436*/ offset:32288
	ds_load_tr16_b128 v[120:123] /*v[376:379]*/, v180 /*v436*/ offset:64
	ds_load_tr16_b128 v[88:91] /*v[344:347]*/, v180 /*v436*/ offset:96
	ds_load_tr16_b128 v[124:127] /*v[380:383]*/, v180 /*v436*/ offset:4672
	ds_load_tr16_b128 v[92:95] /*v[348:351]*/, v180 /*v436*/ offset:4704
	ds_load_tr16_b128 v[112:115] /*v[368:371]*/, v180 /*v436*/ offset:9280
	ds_load_tr16_b128 v[80:83] /*v[336:339]*/, v180 /*v436*/ offset:9312
	ds_load_tr16_b128 v[116:119] /*v[372:375]*/, v180 /*v436*/ offset:13888
	ds_load_tr16_b128 v[84:87] /*v[340:343]*/, v180 /*v436*/ offset:13920
	ds_load_tr16_b128 v[104:107] /*v[360:363]*/, v180 /*v436*/ offset:18496
	ds_load_tr16_b128 v[72:75] /*v[328:331]*/, v180 /*v436*/ offset:18528
	ds_load_tr16_b128 v[108:111] /*v[364:367]*/, v180 /*v436*/ offset:23104
	ds_load_tr16_b128 v[76:79] /*v[332:335]*/, v180 /*v436*/ offset:23136
	ds_load_tr16_b128 v[96:99] /*v[352:355]*/, v180 /*v436*/ offset:27712
	ds_load_tr16_b128 v[64:67] /*v[320:323]*/, v180 /*v436*/ offset:27744
	ds_load_tr16_b128 v[100:103] /*v[356:359]*/, v180 /*v436*/ offset:32320
	ds_load_tr16_b128 v[68:71] /*v[324:327]*/, v180 /*v436*/ offset:32352
	ds_load_tr16_b128 v[56:59] /*v[312:315]*/, v180 /*v436*/ offset:128
	s_set_vgpr_msb 0x5101
	ds_load_tr16_b128 v[8:11], v180 /*v436*/ offset:160
	s_set_vgpr_msb 0x141
	ds_load_tr16_b128 v[60:63] /*v[316:319]*/, v180 /*v436*/ offset:4736
	s_set_vgpr_msb 0x4101
	ds_load_tr16_b128 v[12:15], v180 /*v436*/ offset:4768
	s_wait_dscnt 0x2
	scratch_store_b128 off, v[8:11], off offset:716 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[12:15], off offset:732 nv
	s_set_vgpr_msb 0x141
	ds_load_tr16_b128 v[48:51] /*v[304:307]*/, v180 /*v436*/ offset:9344
	s_wait_alu depctr_vm_vsrc(0)
	s_set_vgpr_msb 0x4101
	ds_load_tr16_b128 v[8:11], v180 /*v436*/ offset:9376
	s_set_vgpr_msb 0x141
	ds_load_tr16_b128 v[52:55] /*v[308:311]*/, v180 /*v436*/ offset:13952
	s_set_vgpr_msb 0x4101
	ds_load_tr16_b128 v[12:15], v180 /*v436*/ offset:13984
	s_wait_dscnt 0x2
	scratch_store_b128 off, v[8:11], off offset:684 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[12:15], off offset:700 nv
	s_set_vgpr_msb 0x141
	ds_load_tr16_b128 v[40:43] /*v[296:299]*/, v180 /*v436*/ offset:18560
	s_wait_alu depctr_vm_vsrc(0)
	s_set_vgpr_msb 0x4101
	ds_load_tr16_b128 v[8:11], v180 /*v436*/ offset:18592
	s_set_vgpr_msb 0x141
	ds_load_tr16_b128 v[44:47] /*v[300:303]*/, v180 /*v436*/ offset:23168
	s_set_vgpr_msb 0x4101
	ds_load_tr16_b128 v[12:15], v180 /*v436*/ offset:23200
	s_wait_dscnt 0x2
	scratch_store_b128 off, v[8:11], off offset:652 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[12:15], off offset:668 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_tr16_b128 v[12:15], v180 /*v436*/ offset:27776
	ds_load_tr16_b128 v[8:11], v180 /*v436*/ offset:27808
	ds_load_tr16_b128 v[16:19], v180 /*v436*/ offset:32384
	s_wait_dscnt 0x2
	scratch_store_b128 off, v[12:15], off offset:748 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[16:19], off offset:764 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_tr16_b128 v[12:15], v180 /*v436*/ offset:32416
	scratch_store_b128 off, v[8:11], off offset:620 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[12:15], off offset:636 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_tr16_b128 v[12:15], v180 /*v436*/ offset:192
	ds_load_tr16_b128 v[8:11], v180 /*v436*/ offset:224
	ds_load_tr16_b128 v[16:19], v180 /*v436*/ offset:4800
	s_wait_dscnt 0x2
	scratch_store_b128 off, v[12:15], off offset:588 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[16:19], off offset:604 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_tr16_b128 v[12:15], v180 /*v436*/ offset:4832
	scratch_store_b128 off, v[8:11], off offset:460 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[12:15], off offset:476 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_tr16_b128 v[12:15], v180 /*v436*/ offset:9408
	ds_load_tr16_b128 v[8:11], v180 /*v436*/ offset:9440
	ds_load_tr16_b128 v[16:19], v180 /*v436*/ offset:14016
	s_wait_dscnt 0x2
	scratch_store_b128 off, v[12:15], off offset:556 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[16:19], off offset:572 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_tr16_b128 v[12:15], v180 /*v436*/ offset:14048
	scratch_store_b128 off, v[8:11], off offset:428 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[12:15], off offset:444 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_tr16_b128 v[12:15], v180 /*v436*/ offset:18624
	ds_load_tr16_b128 v[8:11], v180 /*v436*/ offset:18656
	ds_load_tr16_b128 v[16:19], v180 /*v436*/ offset:23232
	s_wait_dscnt 0x2
	scratch_store_b128 off, v[12:15], off offset:524 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[16:19], off offset:540 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_tr16_b128 v[12:15], v180 /*v436*/ offset:23264
	scratch_store_b128 off, v[8:11], off offset:396 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[12:15], off offset:412 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_tr16_b128 v[12:15], v180 /*v436*/ offset:27840
	ds_load_tr16_b128 v[8:11], v180 /*v436*/ offset:27872
	ds_load_tr16_b128 v[16:19], v180 /*v436*/ offset:32448
	s_wait_dscnt 0x2
	scratch_store_b128 off, v[12:15], off offset:492 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[16:19], off offset:508 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_tr16_b128 v[12:15], v180 /*v436*/ offset:32480
	scratch_store_b128 off, v[8:11], off offset:364 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[12:15], off offset:380 nv
	s_set_vgpr_msb 0x100
	v_max3_num_f32 v152, v192, v193, v194
	s_set_vgpr_msb 21
	v_max3_num_f32 v153, v192 /*v448*/, v193 /*v449*/, v194 /*v450*/
	s_set_vgpr_msb 0x1500
	v_max3_num_f32 v154, v195, v196, v197
	s_set_vgpr_msb 21
	v_max3_num_f32 v155, v195 /*v451*/, v196 /*v452*/, v197 /*v453*/
	s_set_vgpr_msb 0x1500
	v_max3_num_f32 v156, v198, v199, v128
	s_set_vgpr_msb 21
	v_max3_num_f32 v157, v198 /*v454*/, v199 /*v455*/, v200 /*v456*/
	s_set_vgpr_msb 0x1500
	v_max3_num_f32 v158, v129, v130, v131
	v_max3_num_f32 v160, v132, v133, v134
	v_max3_num_f32 v162, v135, v136, v137
	v_max3_num_f32 v164, v138, v139, v140
	v_max3_num_f32 v166, v141, v142, v143
	v_max3_num_f32 v168, v144, v145, v146
	v_max3_num_f32 v170, v147, v148, v149
	v_max3_num_f32 v172, v150, v151, v214
	v_max3_num_f32 v174, v215, v216, v217
	v_max3_num_f32 v176, v218, v219, v220
	s_set_vgpr_msb 20
	v_max3_num_f32 v178, v221, v8 /*v264*/, v9 /*v265*/
	s_set_vgpr_msb 0x1415
	v_max3_num_f32 v180, v10 /*v266*/, v11 /*v267*/, v12 /*v268*/
	v_max3_num_f32 v182, v13 /*v269*/, v14 /*v270*/, v15 /*v271*/
	v_max3_num_f32 v184, v16 /*v272*/, v17 /*v273*/, v18 /*v274*/
	v_max3_num_f32 v186, v19 /*v275*/, v20 /*v276*/, v21 /*v277*/
	v_max_num_f32_e32 v188, v22 /*v278*/, v23 /*v279*/
	v_max3_num_f32 v190, v25 /*v281*/, v26 /*v282*/, v27 /*v283*/
	v_max3_num_f32 v159, v201 /*v457*/, v202 /*v458*/, v203 /*v459*/
	v_max3_num_f32 v161, v204 /*v460*/, v205 /*v461*/, v206 /*v462*/
	v_max3_num_f32 v163, v207 /*v463*/, v208 /*v464*/, v209 /*v465*/
	v_max3_num_f32 v165, v210 /*v466*/, v211 /*v467*/, v212 /*v468*/
	v_max3_num_f32 v167, v213 /*v469*/, v214 /*v470*/, v215 /*v471*/
	v_max3_num_f32 v169, v216 /*v472*/, v217 /*v473*/, v218 /*v474*/
	v_max3_num_f32 v171, v219 /*v475*/, v220 /*v476*/, v221 /*v477*/
	v_max3_num_f32 v173, v222 /*v478*/, v223 /*v479*/, v224 /*v480*/
	v_max3_num_f32 v175, v225 /*v481*/, v226 /*v482*/, v227 /*v483*/
	v_max3_num_f32 v177, v228 /*v484*/, v229 /*v485*/, v230 /*v486*/
	v_max3_num_f32 v179, v231 /*v487*/, v232 /*v488*/, v233 /*v489*/
	v_max3_num_f32 v181, v234 /*v490*/, v235 /*v491*/, v236 /*v492*/
	v_max3_num_f32 v183, v237 /*v493*/, v238 /*v494*/, v239 /*v495*/
	v_max3_num_f32 v185, v240 /*v496*/, v241 /*v497*/, v242 /*v498*/
	v_max3_num_f32 v187, v243 /*v499*/, v244 /*v500*/, v245 /*v501*/
	v_max_num_f32_e32 v189, v246 /*v502*/, v247 /*v503*/
	v_max3_num_f32 v191, v249 /*v505*/, v250 /*v506*/, v251 /*v507*/
	v_max3_num_f32 v200, v28 /*v284*/, v29 /*v285*/, v30 /*v286*/
	s_set_vgpr_msb 0x1500
	v_max3_num_f32 v152, v152, v154, v156
	v_max3_num_f32 v153, v153, v155, v157
	v_max3_num_f32 v154, v158, v160, v162
	v_max3_num_f32 v155, v164, v166, v168
	v_max3_num_f32 v156, v170, v172, v174
	v_max3_num_f32 v157, v176, v178, v180
	v_max3_num_f32 v158, v182, v184, v186
	s_set_vgpr_msb 4
	v_max3_num_f32 v160, v188, v24 /*v280*/, v190
	s_set_vgpr_msb 0x415
	v_max3_num_f32 v201, v252 /*v508*/, v253 /*v509*/, v254 /*v510*/
	s_set_vgpr_msb 0x1500
	v_max3_num_f32 v159, v159, v161, v163
	v_max3_num_f32 v161, v165, v167, v169
	v_max3_num_f32 v152, v152, v154, v155
	v_max3_num_f32 v154, v156, v157, v158
	s_set_vgpr_msb 16
	v_max3_num_f32 v155, v160, v200, v31 /*v287*/
	s_set_vgpr_msb 0x1000
	v_max3_num_f32 v156, v171, v173, v175
	v_max3_num_f32 v157, v177, v179, v181
	v_max3_num_f32 v158, v183, v185, v187
	s_set_vgpr_msb 4
	v_max3_num_f32 v160, v189, v248 /*v504*/, v191
	s_set_vgpr_msb 0x400
	v_max3_num_f32 v152, v152, v154, v155
	v_max3_num_f32 v153, v153, v159, v161
	v_max3_num_f32 v154, v156, v157, v158
	s_set_vgpr_msb 16
	v_max3_num_f32 v155, v160, v201, v255 /*v511*/
	s_set_vgpr_msb 0x1000
	v_max3_num_f32 v153, v153, v154, v155
	v_dual_mov_b32 v156, v152 :: v_dual_mov_b32 v154, v153
	v_permlanex16_b32 v156, v156, s95, 0xfedcba98
	v_permlanex16_b32 v154, v154, s95, 0xfedcba98
	v_dual_max_num_f32 v152, v152, v156 :: v_dual_max_num_f32 v153, v153, v154
	s_set_vgpr_msb 4
	v_sub_f32_e32 v155, v152, v179 /*v435*/
	v_max_num_f32_e32 v152, v152, v179 /*v435*/
	v_sub_f32_e32 v154, v153, v178 /*v434*/
	s_set_vgpr_msb 0x400
	v_cmp_lt_f32_e32 vcc_lo, 0x41000000, v155
	s_cmp_eq_u32 vcc_lo, 0
	s_cselect_b32 s2, -1, 0
	s_set_vgpr_msb 0x44
	v_cndmask_b32_e64 v176 /*v432*/, v152, v179 /*v435*/, s2
	s_set_vgpr_msb 0x4401
	v_cmp_lt_f32_e64 s2, 0x41000000, v154
	v_max_num_f32_e32 v152, v178 /*v434*/, v153
	s_set_vgpr_msb 0x104
	v_mul_f32_e32 v226, 0xbfb8aa3b, v176 /*v432*/
	s_cmp_lg_u32 s2, 0
	s_cselect_b32 s5, -1, 0
	s_cmp_eq_u32 s2, 0
	v_pk_fma_f32 v[132:133], v[132:133], s[56:57], v[226:227] op_sel_hi:[1,0,0]
	s_cselect_b32 s2, -1, 0
	v_pk_fma_f32 v[128:129], v[128:129], s[56:57], v[226:227] op_sel_hi:[1,0,0]
	s_wait_alu depctr_vm_vsrc(0)
	v_cndmask_b32_e64 v8, v152, v178 /*v434*/, s2
	v_pk_fma_f32 v[152:153], v[192:193], s[56:57], v[226:227] op_sel_hi:[1,0,0]
	v_exp_f32_e32 v232, v132
	v_exp_f32_e32 v240, v133
	v_nop
	v_pk_fma_f32 v[132:133], v[138:139], s[56:57], v[226:227] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[138:139], v[142:143], s[56:57], v[226:227] op_sel_hi:[1,0,0]
	v_exp_f32_e32 v154, v152
	v_exp_f32_e32 v156, v153
	v_nop
	v_pk_fma_f32 v[152:153], v[198:199], s[56:57], v[226:227] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[142:143], v[144:145], s[56:57], v[226:227] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[158:159], v[194:195], s[56:57], v[226:227] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x440
	v_mul_f32_e32 v32 /*v288*/, 0xbfb8aa3b, v8
	s_set_vgpr_msb 0x4000
	v_exp_f32_e32 v204, v128
	v_exp_f32_e32 v180, v152
	v_exp_f32_e32 v188, v153
	v_exp_f32_e32 v212, v129
	v_nop
	v_pk_fma_f32 v[128:129], v[134:135], s[56:57], v[226:227] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[134:135], v[140:141], s[56:57], v[226:227] op_sel_hi:[1,0,0]
	v_exp_f32_e32 v174, v142
	v_exp_f32_e32 v186, v143
	v_nop
	v_pk_fma_f32 v[142:143], v[150:151], s[56:57], v[226:227] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[144:145], v[216:217], s[56:57], v[226:227] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[152:153], v[220:221], s[56:57], v[226:227] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[160:161], v[196:197], s[56:57], v[226:227] op_sel_hi:[1,0,0]
	v_exp_f32_e32 v192, v158
	v_exp_f32_e32 v140, v134
	v_exp_f32_e32 v158, v135
	v_nop
	v_pk_fma_f32 v[134:135], v[146:147], s[56:57], v[226:227] op_sel_hi:[1,0,0]
	v_exp_f32_e32 v248, v142
	v_exp_f32_e32 v142, v144
	v_exp_f32_e32 v146, v145
	v_nop
	s_set_vgpr_msb 1
	v_pk_fma_f32 v[144:145], v[8:9] /*v[264:265]*/, s[56:57], v[226:227] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x100
	v_exp_f32_e32 v166, v152
	v_exp_f32_e32 v176, v153
	v_nop
	s_set_vgpr_msb 1
	v_pk_fma_f32 v[152:153], v[12:13] /*v[268:269]*/, s[56:57], v[226:227] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x151
	v_pk_fma_f32 v[8:9] /*v[264:265]*/, v[192:193] /*v[448:449]*/, s[56:57], v[32:33] /*v[288:289]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[12:13] /*v[268:269]*/, v[196:197] /*v[452:453]*/, s[56:57], v[32:33] /*v[288:289]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5100
	v_exp_f32_e32 v164, v160
	v_exp_f32_e32 v194, v138
	v_exp_f32_e32 v160, v139
	v_nop
	v_pk_fma_f32 v[138:139], v[148:149], s[56:57], v[226:227] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[148:149], v[218:219], s[56:57], v[226:227] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[130:131], v[130:131], s[56:57], v[226:227] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 1
	v_exp_f32_e32 v155, v8 /*v264*/
	v_exp_f32_e32 v157, v9 /*v265*/
	v_nop
	s_set_vgpr_msb 0x151
	v_pk_fma_f32 v[8:9] /*v[264:265]*/, v[198:199] /*v[454:455]*/, s[56:57], v[32:33] /*v[288:289]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5101
	v_exp_f32_e32 v165, v12 /*v268*/
	v_exp_f32_e32 v173, v13 /*v269*/
	v_nop
	s_set_vgpr_msb 0x151
	v_pk_fma_f32 v[12:13] /*v[268:269]*/, v[202:203] /*v[458:459]*/, s[56:57], v[32:33] /*v[288:289]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5100
	v_exp_f32_e32 v150, v148
	v_exp_f32_e32 v162, v149
	v_nop
	s_set_vgpr_msb 1
	v_pk_fma_f32 v[148:149], v[10:11] /*v[266:267]*/, s[56:57], v[226:227] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x151
	v_pk_fma_f32 v[10:11] /*v[266:267]*/, v[194:195] /*v[450:451]*/, s[56:57], v[32:33] /*v[288:289]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5100
	v_exp_f32_e32 v222, v130
	v_exp_f32_e32 v228, v131
	v_nop
	v_pk_fma_f32 v[130:131], v[136:137], s[56:57], v[226:227] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 1
	v_exp_f32_e32 v181, v8 /*v264*/
	v_exp_f32_e32 v189, v9 /*v265*/
	v_nop
	s_set_vgpr_msb 0x151
	v_pk_fma_f32 v[8:9] /*v[264:265]*/, v[204:205] /*v[460:461]*/, s[56:57], v[32:33] /*v[288:289]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5101
	v_exp_f32_e32 v223, v12 /*v268*/
	v_exp_f32_e32 v229, v13 /*v269*/
	v_nop
	s_set_vgpr_msb 0x151
	v_pk_fma_f32 v[12:13] /*v[268:269]*/, v[208:209] /*v[464:465]*/, s[56:57], v[32:33] /*v[288:289]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5101
	v_exp_f32_e32 v193, v10 /*v266*/
	v_exp_f32_e32 v199, v11 /*v267*/
	v_nop
	s_set_vgpr_msb 0x151
	v_pk_fma_f32 v[10:11] /*v[266:267]*/, v[200:201] /*v[456:457]*/, s[56:57], v[32:33] /*v[288:289]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5100
	v_exp_f32_e32 v246, v128
	v_exp_f32_e32 v252, v129
	v_exp_f32_e32 v128, v130
	v_exp_f32_e32 v130, v131
	s_set_vgpr_msb 1
	v_exp_f32_e32 v233, v8 /*v264*/
	v_exp_f32_e32 v241, v9 /*v265*/
	v_nop
	s_set_vgpr_msb 0x151
	v_pk_fma_f32 v[8:9] /*v[264:265]*/, v[210:211] /*v[466:467]*/, s[56:57], v[32:33] /*v[288:289]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5101
	v_exp_f32_e32 v129, v12 /*v268*/
	v_exp_f32_e32 v131, v13 /*v269*/
	v_nop
	s_set_vgpr_msb 0x151
	v_pk_fma_f32 v[12:13] /*v[268:269]*/, v[214:215] /*v[470:471]*/, s[56:57], v[32:33] /*v[288:289]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5101
	v_exp_f32_e32 v205, v10 /*v266*/
	v_exp_f32_e32 v213, v11 /*v267*/
	v_nop
	s_set_vgpr_msb 0x151
	v_pk_fma_f32 v[10:11] /*v[266:267]*/, v[206:207] /*v[462:463]*/, s[56:57], v[32:33] /*v[288:289]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5100
	v_exp_f32_e32 v172, v161
	v_exp_f32_e32 v136, v133
	s_set_vgpr_msb 1
	v_exp_f32_e32 v133, v8 /*v264*/
	v_exp_f32_e32 v137, v9 /*v265*/
	v_nop
	s_set_vgpr_msb 0x151
	v_pk_fma_f32 v[8:9] /*v[264:265]*/, v[216:217] /*v[472:473]*/, s[56:57], v[32:33] /*v[288:289]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5101
	v_exp_f32_e32 v195, v12 /*v268*/
	v_exp_f32_e32 v161, v13 /*v269*/
	v_nop
	s_set_vgpr_msb 0x151
	v_pk_fma_f32 v[12:13] /*v[268:269]*/, v[220:221] /*v[476:477]*/, s[56:57], v[32:33] /*v[288:289]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5101
	v_exp_f32_e32 v247, v10 /*v266*/
	v_exp_f32_e32 v253, v11 /*v267*/
	v_nop
	s_set_vgpr_msb 0x151
	v_pk_fma_f32 v[10:11] /*v[266:267]*/, v[212:213] /*v[468:469]*/, s[56:57], v[32:33] /*v[288:289]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5101
	v_exp_f32_e32 v175, v8 /*v264*/
	v_exp_f32_e32 v187, v9 /*v265*/
	v_nop
	s_set_vgpr_msb 0x151
	v_pk_fma_f32 v[8:9] /*v[264:265]*/, v[222:223] /*v[478:479]*/, s[56:57], v[32:33] /*v[288:289]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5101
	v_exp_f32_e32 v235, v12 /*v268*/
	v_exp_f32_e32 v243, v13 /*v269*/
	v_nop
	s_set_vgpr_msb 0x151
	v_pk_fma_f32 v[12:13] /*v[268:269]*/, v[226:227] /*v[482:483]*/, s[56:57], v[32:33] /*v[288:289]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5100
	v_exp_f32_e32 v198, v159
	s_set_vgpr_msb 1
	v_exp_f32_e32 v141, v10 /*v266*/
	v_exp_f32_e32 v159, v11 /*v267*/
	v_nop
	s_set_vgpr_msb 0x151
	v_pk_fma_f32 v[10:11] /*v[266:267]*/, v[218:219] /*v[474:475]*/, s[56:57], v[32:33] /*v[288:289]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5100
	v_exp_f32_e32 v254, v143
	s_set_vgpr_msb 1
	v_exp_f32_e32 v249, v8 /*v264*/
	v_exp_f32_e32 v255, v9 /*v265*/
	v_nop
	s_set_vgpr_msb 0x151
	v_pk_fma_f32 v[8:9] /*v[264:265]*/, v[228:229] /*v[484:485]*/, s[56:57], v[32:33] /*v[288:289]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5101
	v_exp_f32_e32 v143, v12 /*v268*/
	v_exp_f32_e32 v147, v13 /*v269*/
	v_nop
	s_set_vgpr_msb 0x151
	v_pk_fma_f32 v[12:13] /*v[268:269]*/, v[232:233] /*v[488:489]*/, s[56:57], v[32:33] /*v[288:289]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5100
	v_exp_f32_e32 v190, v134
	v_exp_f32_e32 v210, v135
	v_nop
	v_pk_fma_f32 v[134:135], v[214:215], s[56:57], v[226:227] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 1
	v_exp_f32_e32 v191, v10 /*v266*/
	v_exp_f32_e32 v211, v11 /*v267*/
	v_nop
	s_set_vgpr_msb 0x151
	v_pk_fma_f32 v[10:11] /*v[266:267]*/, v[224:225] /*v[480:481]*/, s[56:57], v[32:33] /*v[288:289]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5101
	v_exp_f32_e32 v151, v8 /*v264*/
	v_exp_f32_e32 v163, v9 /*v265*/
	v_nop
	s_set_vgpr_msb 0x151
	v_pk_fma_f32 v[8:9] /*v[264:265]*/, v[234:235] /*v[490:491]*/, s[56:57], v[32:33] /*v[288:289]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5101
	v_exp_f32_e32 v183, v12 /*v268*/
	v_exp_f32_e32 v201, v13 /*v269*/
	v_nop
	s_set_vgpr_msb 0x151
	v_pk_fma_f32 v[12:13] /*v[268:269]*/, v[238:239] /*v[494:495]*/, s[56:57], v[32:33] /*v[288:289]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5100
	v_exp_f32_e32 v234, v138
	v_exp_f32_e32 v242, v139
	v_exp_f32_e32 v138, v135
	s_set_vgpr_msb 1
	v_exp_f32_e32 v135, v10 /*v266*/
	v_exp_f32_e32 v139, v11 /*v267*/
	v_nop
	s_set_vgpr_msb 0x151
	v_pk_fma_f32 v[10:11] /*v[266:267]*/, v[230:231] /*v[486:487]*/, s[56:57], v[32:33] /*v[288:289]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5100
	v_exp_f32_e32 v182, v144
	v_exp_f32_e32 v200, v145
	v_exp_f32_e32 v206, v148
	s_set_vgpr_msb 1
	v_pk_fma_f32 v[144:145], v[14:15] /*v[270:271]*/, s[56:57], v[226:227] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x100
	v_exp_f32_e32 v214, v149
	v_nop
	s_set_vgpr_msb 1
	v_pk_fma_f32 v[148:149], v[16:17] /*v[272:273]*/, s[56:57], v[226:227] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[170:171], v[20:21] /*v[276:277]*/, s[56:57], v[226:227] op_sel_hi:[1,0,0]
	v_exp_f32_e32 v207, v8 /*v264*/
	v_exp_f32_e32 v215, v9 /*v265*/
	v_nop
	s_set_vgpr_msb 0x151
	v_pk_fma_f32 v[8:9] /*v[264:265]*/, v[240:241] /*v[496:497]*/, s[56:57], v[32:33] /*v[288:289]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5101
	v_exp_f32_e32 v225, v12 /*v268*/
	v_exp_f32_e32 v237, v13 /*v269*/
	v_nop
	s_set_vgpr_msb 0x151
	v_pk_fma_f32 v[12:13] /*v[268:269]*/, v[244:245] /*v[500:501]*/, s[56:57], v[32:33] /*v[288:289]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5101
	v_exp_f32_e32 v167, v10 /*v266*/
	v_exp_f32_e32 v177, v11 /*v267*/
	v_nop
	s_set_vgpr_msb 0x151
	v_pk_fma_f32 v[10:11] /*v[266:267]*/, v[236:237] /*v[492:493]*/, s[56:57], v[32:33] /*v[288:289]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5100
	v_exp_f32_e32 v224, v144
	v_exp_f32_e32 v236, v145
	v_exp_f32_e32 v144, v148
	v_exp_f32_e32 v148, v149
	s_set_vgpr_msb 1
	v_pk_fma_f32 v[184:185], v[22:23] /*v[278:279]*/, s[56:57], v[226:227] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x100
	v_exp_f32_e32 v178, v171
	s_set_vgpr_msb 1
	v_pk_fma_f32 v[220:221], v[26:27] /*v[282:283]*/, s[56:57], v[226:227] op_sel_hi:[1,0,0]
	v_exp_f32_e32 v145, v8 /*v264*/
	v_exp_f32_e32 v149, v9 /*v265*/
	v_nop
	s_set_vgpr_msb 0x151
	v_pk_fma_f32 v[8:9] /*v[264:265]*/, v[246:247] /*v[502:503]*/, s[56:57], v[32:33] /*v[288:289]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5101
	v_exp_f32_e32 v171, v12 /*v268*/
	v_exp_f32_e32 v179, v13 /*v269*/
	v_nop
	s_set_vgpr_msb 0x151
	v_pk_fma_f32 v[12:13] /*v[268:269]*/, v[250:251] /*v[506:507]*/, s[56:57], v[32:33] /*v[288:289]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5100
	v_exp_f32_e32 v218, v152
	v_exp_f32_e32 v152, v153
	s_set_vgpr_msb 1
	v_pk_fma_f32 v[168:169], v[18:19] /*v[274:275]*/, s[56:57], v[226:227] op_sel_hi:[1,0,0]
	v_exp_f32_e32 v219, v10 /*v266*/
	v_exp_f32_e32 v153, v11 /*v267*/
	v_nop
	s_set_vgpr_msb 0x151
	v_pk_fma_f32 v[10:11] /*v[266:267]*/, v[242:243] /*v[498:499]*/, s[56:57], v[32:33] /*v[288:289]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5101
	v_pk_fma_f32 v[208:209], v[24:25] /*v[280:281]*/, s[56:57], v[226:227] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x100
	v_exp_f32_e32 v202, v185
	s_set_vgpr_msb 1
	v_pk_fma_f32 v[230:231], v[28:29] /*v[284:285]*/, s[56:57], v[226:227] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[244:245], v[30:31] /*v[286:287]*/, s[56:57], v[226:227] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x100
	v_exp_f32_e32 v226, v221
	s_set_vgpr_msb 1
	v_exp_f32_e32 v185, v8 /*v264*/
	v_exp_f32_e32 v203, v9 /*v265*/
	v_exp_f32_e32 v221, v12 /*v268*/
	s_set_vgpr_msb 0x151
	v_pk_fma_f32 v[8:9] /*v[264:265]*/, v[252:253] /*v[508:509]*/, s[56:57], v[32:33] /*v[288:289]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5101
	v_exp_f32_e32 v227, v13 /*v269*/
	v_nop
	s_set_vgpr_msb 0x140
	v_pk_add_f32 v[12:13] /*v[268:269]*/, v[154:155], v[156:157]
	v_pk_add_f32 v[14:15] /*v[270:271]*/, v[198:199], v[164:165]
	s_set_vgpr_msb 0x4000
	v_exp_f32_e32 v132, v132
	v_exp_f32_e32 v134, v134
	v_exp_f32_e32 v196, v168
	v_exp_f32_e32 v168, v169
	s_set_vgpr_msb 1
	v_exp_f32_e32 v197, v10 /*v266*/
	v_exp_f32_e32 v169, v11 /*v267*/
	v_nop
	s_set_vgpr_msb 0x151
	v_pk_fma_f32 v[10:11] /*v[266:267]*/, v[248:249] /*v[504:505]*/, s[56:57], v[32:33] /*v[288:289]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5100
	v_exp_f32_e32 v238, v231
	s_set_vgpr_msb 1
	v_exp_f32_e32 v231, v8 /*v264*/
	v_exp_f32_e32 v239, v9 /*v265*/
	v_nop
	s_set_vgpr_msb 0x144
	v_pk_add_f32 v[8:9] /*v[264:265]*/, v[192:193], v[12:13] /*v[268:269]*/
	v_pk_add_f32 v[12:13] /*v[268:269]*/, v[172:173], v[14:15] /*v[270:271]*/
	s_set_vgpr_msb 0x4440
	v_pk_add_f32 v[14:15] /*v[270:271]*/, v[180:181], v[188:189]
	v_pk_add_f32 v[16:17] /*v[272:273]*/, v[212:213], v[222:223]
	v_pk_add_f32 v[18:19] /*v[274:275]*/, v[232:233], v[240:241]
	v_pk_add_f32 v[28:29] /*v[284:285]*/, v[210:211], v[234:235]
	v_pk_add_f32 v[30:31] /*v[286:287]*/, v[248:249], v[254:255]
	v_pk_add_f32 v[34:35] /*v[290:291]*/, v[150:151], v[162:163]
	v_pk_add_f32 v[36:37] /*v[292:293]*/, v[176:177], v[182:183]
	s_set_vgpr_msb 0x4000
	v_exp_f32_e32 v170, v170
	v_exp_f32_e32 v184, v184
	v_exp_f32_e32 v216, v209
	v_exp_f32_e32 v220, v220
	s_set_vgpr_msb 1
	v_exp_f32_e32 v217, v11 /*v267*/
	v_exp_f32_e32 v209, v10 /*v266*/
	v_nop
	s_set_vgpr_msb 0x151
	v_pk_fma_f32 v[10:11] /*v[266:267]*/, v[254:255] /*v[510:511]*/, s[56:57], v[32:33] /*v[288:289]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5140
	v_pk_add_f32 v[20:21] /*v[276:277]*/, v[252:253], v[128:129]
	v_pk_add_f32 v[22:23] /*v[278:279]*/, v[132:133], v[136:137]
	s_set_vgpr_msb 0x4044
	v_pk_add_f32 v[14:15] /*v[270:271]*/, v[204:205], v[14:15] /*v[270:271]*/
	v_pk_add_f32 v[16:17] /*v[272:273]*/, v[228:229], v[16:17] /*v[272:273]*/
	v_pk_add_f32 v[18:19] /*v[274:275]*/, v[246:247], v[18:19] /*v[274:275]*/
	s_set_vgpr_msb 0x4440
	v_pk_add_f32 v[24:25] /*v[280:281]*/, v[158:159], v[194:195]
	v_pk_add_f32 v[32:33] /*v[288:289]*/, v[138:139], v[142:143]
	s_set_vgpr_msb 0x4044
	v_pk_add_f32 v[28:29] /*v[284:285]*/, v[242:243], v[28:29] /*v[284:285]*/
	v_pk_add_f32 v[30:31] /*v[286:287]*/, v[134:135], v[30:31] /*v[286:287]*/
	s_set_vgpr_msb 0x4440
	v_pk_add_f32 v[38:39] /*v[294:295]*/, v[206:207], v[214:215]
	v_pk_add_f32 v[192:193] /*v[448:449]*/, v[152:153], v[224:225]
	v_pk_add_f32 v[194:195] /*v[450:451]*/, v[144:145], v[148:149]
	s_set_vgpr_msb 0x4044
	v_pk_add_f32 v[34:35] /*v[290:291]*/, v[166:167], v[34:35] /*v[290:291]*/
	v_pk_add_f32 v[36:37] /*v[292:293]*/, v[200:201], v[36:37] /*v[292:293]*/
	s_set_vgpr_msb 0x4445
	v_pk_add_f32 v[8:9] /*v[264:265]*/, v[8:9] /*v[264:265]*/, v[12:13] /*v[268:269]*/
	s_set_vgpr_msb 0x4500
	v_exp_f32_e32 v208, v208
	v_exp_f32_e32 v230, v230
	s_set_vgpr_msb 0x44
	v_pk_add_f32 v[20:21] /*v[276:277]*/, v[130:131], v[20:21] /*v[276:277]*/
	v_pk_add_f32 v[22:23] /*v[278:279]*/, v[140:141], v[22:23] /*v[278:279]*/
	s_set_vgpr_msb 0x4440
	v_pk_add_f32 v[26:27] /*v[282:283]*/, v[174:175], v[186:187]
	s_set_vgpr_msb 0x4044
	v_pk_add_f32 v[24:25] /*v[280:281]*/, v[160:161], v[24:25] /*v[280:281]*/
	v_pk_add_f32 v[32:33] /*v[288:289]*/, v[146:147], v[32:33] /*v[288:289]*/
	v_pk_add_f32 v[38:39] /*v[294:295]*/, v[218:219], v[38:39] /*v[294:295]*/
	v_pk_add_f32 v[192:193] /*v[448:449]*/, v[236:237], v[192:193] /*v[448:449]*/
	v_pk_add_f32 v[194:195] /*v[450:451]*/, v[196:197], v[194:195] /*v[450:451]*/
	s_set_vgpr_msb 0x4440
	v_pk_add_f32 v[196:197] /*v[452:453]*/, v[168:169], v[170:171]
	v_pk_add_f32 v[198:199] /*v[454:455]*/, v[184:185], v[202:203]
	v_pk_add_f32 v[200:201] /*v[456:457]*/, v[216:217], v[220:221]
	s_set_vgpr_msb 0x4045
	v_pk_add_f32 v[8:9] /*v[264:265]*/, v[14:15] /*v[270:271]*/, v[8:9] /*v[264:265]*/
	v_pk_add_f32 v[14:15] /*v[270:271]*/, v[16:17] /*v[272:273]*/, v[18:19] /*v[274:275]*/
	v_pk_add_f32 v[16:17] /*v[272:273]*/, v[28:29] /*v[284:285]*/, v[30:31] /*v[286:287]*/
	v_pk_add_f32 v[18:19] /*v[274:275]*/, v[34:35] /*v[290:291]*/, v[36:37] /*v[292:293]*/
	s_set_vgpr_msb 0x4500
	v_exp_f32_e32 v244, v244
	v_exp_f32_e32 v250, v245
	s_set_vgpr_msb 1
	v_exp_f32_e32 v245, v10 /*v266*/
	s_set_vgpr_msb 0x144
	v_pk_add_f32 v[26:27] /*v[282:283]*/, v[190:191], v[26:27] /*v[282:283]*/
	s_set_vgpr_msb 0x4440
	v_pk_add_f32 v[202:203] /*v[458:459]*/, v[230:231], v[238:239]
	s_set_vgpr_msb 0x4044
	v_pk_add_f32 v[12:13] /*v[268:269]*/, v[178:179], v[196:197] /*v[452:453]*/
	v_pk_add_f32 v[196:197] /*v[452:453]*/, v[208:209], v[198:199] /*v[454:455]*/
	v_pk_add_f32 v[198:199] /*v[454:455]*/, v[226:227], v[200:201] /*v[456:457]*/
	s_set_vgpr_msb 0x4445
	v_pk_add_f32 v[22:23] /*v[278:279]*/, v[22:23] /*v[278:279]*/, v[24:25] /*v[280:281]*/
	v_pk_add_f32 v[24:25] /*v[280:281]*/, v[192:193] /*v[448:449]*/, v[194:195] /*v[450:451]*/
	v_pk_add_f32 v[14:15] /*v[270:271]*/, v[20:21] /*v[276:277]*/, v[14:15] /*v[270:271]*/
	v_pk_add_f32 v[16:17] /*v[272:273]*/, v[32:33] /*v[288:289]*/, v[16:17] /*v[272:273]*/
	v_pk_add_f32 v[18:19] /*v[274:275]*/, v[38:39] /*v[294:295]*/, v[18:19] /*v[274:275]*/
	s_set_vgpr_msb 0x4544
	v_pk_add_f32 v[200:201] /*v[456:457]*/, v[244:245], v[202:203] /*v[458:459]*/
	s_set_vgpr_msb 0x4445
	v_pk_add_f32 v[20:21] /*v[276:277]*/, v[26:27] /*v[282:283]*/, v[22:23] /*v[278:279]*/
	v_pk_add_f32 v[12:13] /*v[268:269]*/, v[12:13] /*v[268:269]*/, v[24:25] /*v[280:281]*/
	v_pk_add_f32 v[22:23] /*v[278:279]*/, v[196:197] /*v[452:453]*/, v[198:199] /*v[454:455]*/
	v_pk_add_f32 v[8:9] /*v[264:265]*/, v[8:9] /*v[264:265]*/, v[14:15] /*v[270:271]*/
	v_pk_add_f32 v[14:15] /*v[270:271]*/, v[16:17] /*v[272:273]*/, v[18:19] /*v[274:275]*/
	s_set_vgpr_msb 0x4501
	v_exp_f32_e32 v251, v11 /*v267*/
	s_wait_alu depctr_va_vdst(0)
	scratch_store_b32 off, v8, off offset:8 nv
	v_nop
	s_set_vgpr_msb 0x145
	v_pk_add_f32 v[10:11] /*v[266:267]*/, v[200:201] /*v[456:457]*/, v[22:23] /*v[278:279]*/
	v_pk_add_f32 v[8:9] /*v[264:265]*/, v[20:21] /*v[276:277]*/, v[8:9] /*v[264:265]*/
	v_pk_add_f32 v[12:13] /*v[268:269]*/, v[12:13] /*v[268:269]*/, v[14:15] /*v[270:271]*/
	s_set_vgpr_msb 0x4544
	v_pk_add_f32 v[10:11] /*v[266:267]*/, v[250:251], v[10:11] /*v[266:267]*/
	s_set_vgpr_msb 0x4445
	v_pk_add_f32 v[8:9] /*v[264:265]*/, v[8:9] /*v[264:265]*/, v[12:13] /*v[268:269]*/
	v_pk_add_f32 v[192:193] /*v[448:449]*/, v[10:11] /*v[266:267]*/, v[8:9] /*v[264:265]*/
	v_dual_sub_f32 v12 /*v268*/, v179 /*v435*/, v176 /*v432*/ :: v_dual_mov_b32 v194 /*v450*/, v192 /*v448*/
	v_dual_mul_f32 v8 /*v264*/, 0x3fb8aa3b, v12 /*v268*/ :: v_dual_mov_b32 v195 /*v451*/, v193 /*v449*/
	v_permlanex16_b32 v194 /*v450*/, v194 /*v450*/, s95, 0xfedcba98
	v_exp_f32_e32 v196 /*v452*/, v8 /*v264*/
	v_permlanex16_b32 v195 /*v451*/, v195 /*v451*/, s95, 0xfedcba98
	s_set_vgpr_msb 0x4500
	s_cbranch_vccz .LBB0_8
	s_set_vgpr_msb 4
	v_pk_mul_f32 v[126:127], v[126:127], v[196:197] /*v[452:453]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[124:125], v[124:125], v[196:197] /*v[452:453]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[122:123], v[122:123], v[196:197] /*v[452:453]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[120:121], v[120:121], v[196:197] /*v[452:453]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[118:119], v[118:119], v[196:197] /*v[452:453]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[116:117], v[116:117], v[196:197] /*v[452:453]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[114:115], v[114:115], v[196:197] /*v[452:453]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[112:113], v[112:113], v[196:197] /*v[452:453]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[110:111], v[110:111], v[196:197] /*v[452:453]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[108:109], v[108:109], v[196:197] /*v[452:453]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[106:107], v[106:107], v[196:197] /*v[452:453]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[104:105], v[104:105], v[196:197] /*v[452:453]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[102:103], v[102:103], v[196:197] /*v[452:453]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[100:101], v[100:101], v[196:197] /*v[452:453]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[98:99], v[98:99], v[196:197] /*v[452:453]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[96:97], v[96:97], v[196:197] /*v[452:453]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[94:95], v[94:95], v[196:197] /*v[452:453]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[92:93], v[92:93], v[196:197] /*v[452:453]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[90:91], v[90:91], v[196:197] /*v[452:453]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[88:89], v[88:89], v[196:197] /*v[452:453]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[86:87], v[86:87], v[196:197] /*v[452:453]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[84:85], v[84:85], v[196:197] /*v[452:453]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[82:83], v[82:83], v[196:197] /*v[452:453]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[80:81], v[80:81], v[196:197] /*v[452:453]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[78:79], v[78:79], v[196:197] /*v[452:453]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[76:77], v[76:77], v[196:197] /*v[452:453]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[74:75], v[74:75], v[196:197] /*v[452:453]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[72:73], v[72:73], v[196:197] /*v[452:453]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[70:71], v[70:71], v[196:197] /*v[452:453]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[68:69], v[68:69], v[196:197] /*v[452:453]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[66:67], v[66:67], v[196:197] /*v[452:453]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[64:65], v[64:65], v[196:197] /*v[452:453]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x400
.LBB0_8:
	s_wait_alu depctr_vm_vsrc(0)
	s_clause 0x5
	scratch_load_b128 v[8:11], off, off offset:40 nv
	scratch_load_b128 v[12:15], off, off offset:56 nv
	scratch_load_b128 v[16:19], off, off offset:72 nv
	scratch_load_b128 v[20:23], off, off offset:88 nv
	s_set_vgpr_msb 0x45
	scratch_load_b32 v32 /*v288*/, off, off offset:8 nv
	s_and_b32 s2, s5, exec_lo
	s_cselect_b32 s2, 1, 0
	s_cmp_lg_u32 s2, 1
	s_wait_loadcnt 0x0
	v_sub_f32_e32 v8 /*v264*/, v178 /*v434*/, v32 /*v288*/
	v_mul_f32_e32 v8 /*v264*/, 0x3fb8aa3b, v8 /*v264*/
	v_exp_f32_e32 v197 /*v453*/, v8 /*v264*/
	s_set_vgpr_msb 0x4500
	s_cbranch_scc1 .LBB0_3
	v_nop
	s_set_vgpr_msb 0x41
	v_mov_b32_e32 v8 /*v264*/, v197 /*v453*/
	s_set_vgpr_msb 0x4104
	v_pk_mul_f32 v[62:63], v[62:63], v[8:9] /*v[264:265]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[60:61], v[60:61], v[8:9] /*v[264:265]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59], v[58:59], v[8:9] /*v[264:265]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[56:57], v[56:57], v[8:9] /*v[264:265]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[54:55], v[54:55], v[8:9] /*v[264:265]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[52:53], v[52:53], v[8:9] /*v[264:265]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[50:51], v[50:51], v[8:9] /*v[264:265]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[48:49], v[48:49], v[8:9] /*v[264:265]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[46:47], v[46:47], v[8:9] /*v[264:265]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[44:45], v[44:45], v[8:9] /*v[264:265]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[42:43], v[42:43], v[8:9] /*v[264:265]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[40:41], v[40:41], v[8:9] /*v[264:265]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39], v[38:39], v[8:9] /*v[264:265]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[36:37], v[36:37], v[8:9] /*v[264:265]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35], v[34:35], v[8:9] /*v[264:265]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[32:33], v[32:33], v[8:9] /*v[264:265]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[30:31], v[30:31], v[8:9] /*v[264:265]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[28:29], v[28:29], v[8:9] /*v[264:265]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[26:27], v[26:27], v[8:9] /*v[264:265]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25], v[24:25], v[8:9] /*v[264:265]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[22:23], v[22:23], v[8:9] /*v[264:265]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21], v[20:21], v[8:9] /*v[264:265]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[18:19], v[18:19], v[8:9] /*v[264:265]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[16:17], v[16:17], v[8:9] /*v[264:265]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[14:15], v[8:9] /*v[264:265]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13], v[12:13], v[8:9] /*v[264:265]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[10:11], v[8:9] /*v[264:265]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[8:9], v[8:9] /*v[264:265]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[6:7], v[8:9] /*v[264:265]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[4:5], v[8:9] /*v[264:265]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[2:3], v[2:3], v[8:9] /*v[264:265]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[0:1], v[0:1], v[8:9] /*v[264:265]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x400
	s_branch .LBB0_3
.LBB0_10:
	s_and_b32 vcc_lo, exec_lo, s18
	s_cbranch_vccnz .LBB0_24
	s_branch .LBB0_45
.LBB0_11:
	v_dual_mov_b32 v1, v0 :: v_dual_mov_b32 v2, v0
	v_dual_mov_b32 v3, v0 :: v_dual_mov_b32 v4, v0
	v_dual_mov_b32 v5, v0 :: v_dual_mov_b32 v6, v0
	v_dual_mov_b32 v7, v0 :: v_dual_mov_b32 v235, v0
	s_set_vgpr_msb 64
	v_dual_mov_b32 v32 /*v288*/, 0xf149f2ca :: v_dual_mov_b32 v181 /*v437*/, v134
	s_set_vgpr_msb 0x4000
	v_mov_b32_e32 v234, v0
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
	s_set_vgpr_msb 64
	v_mov_b32_e32 v176 /*v432*/, 0xf149f2ca
	s_cmp_ge_u32 s52, s10
	s_set_vgpr_msb 0x4000
	s_cbranch_scc0 .LBB0_14
.LBB0_12:
	s_set_vgpr_msb 1
	v_mov_b32_e32 v236, v32 /*v288*/
	s_set_vgpr_msb 0x141
	v_mov_b32_e32 v55 /*v311*/, v176 /*v432*/
	s_set_vgpr_msb 0x4100
	s_branch .LBB0_23
.LBB0_13:
	s_clause 0x3
	scratch_load_b32 v154, off, off offset:916 nv
	scratch_load_b32 v155, off, off offset:920 nv
	scratch_load_b32 v156, off, off offset:928 nv
	scratch_load_b32 v157, off, off offset:936 nv
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x4
	scratch_load_b32 v134, off, off offset:940 nv
	scratch_load_b32 v152, off, off offset:104 nv
	scratch_load_b32 v128, off, off offset:944 nv
	scratch_load_b32 v129, off, off offset:948 nv
	scratch_load_b32 v130, off, off offset:952 nv
	s_cmp_ge_u32 s52, s10
	s_cbranch_scc1 .LBB0_12
.LBB0_14:
	s_wait_loadcnt 0x1
	s_wait_alu depctr_vm_vsrc(1)
	v_dual_add_nc_u32 v128, s11, v154 :: v_dual_add_nc_u32 v129, s11, v155
	s_add_co_i32 s2, s58, -1
	s_mov_b32 s51, 0
	s_mov_b32 s4, 1
	s_set_vgpr_msb 64
	v_min_i32_e32 v178 /*v434*/, s2, v128
	v_min_i32_e32 v179 /*v435*/, s2, v129
	s_mov_b32 s11, s51
	s_mov_b32 s48, 16
	s_mov_b32 s47, 0x800000
	s_mov_b32 s45, 0xffff0000
	s_mov_b32 s44, 0x7510000
	s_mov_b32 s36, 0xf510000
	s_mov_b32 s82, 0x76543210
	s_mov_b32 s56, 0x3fb8aa3b
	s_set_vgpr_msb 0x4000
	s_branch .LBB0_16
.LBB0_15:
	s_wait_dscnt 0x0
	v_cvt_pk_bf16_f32 v162, v156, v194
	v_cvt_pk_bf16_f32 v160, v144, v146
	v_cvt_pk_bf16_f32 v170, v157, v195
	v_cvt_pk_bf16_f32 v177, v192, v198
	v_cvt_pk_bf16_f32 v185, v193, v199
	v_cvt_pk_bf16_f32 v144, v196, v204
	v_cvt_pk_bf16_f32 v152, v197, v205
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b128 v[192:195], off, off offset:876 th:TH_LOAD_LU nv
	scratch_load_b128 v[196:199], off, off offset:892 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x6
	v_dual_fmac_f32 v131, v130, v234 :: v_dual_fmac_f32 v133, v132, v235
	v_cvt_pk_bf16_f32 v135, v142, v232
	s_set_vgpr_msb 5
	v_cvt_pk_bf16_f32 v134, v252 /*v508*/, v254 /*v510*/
	v_cvt_pk_bf16_f32 v132, v236 /*v492*/, v244 /*v500*/
	s_set_vgpr_msb 0x500
	v_dual_add_f32 v234, v131, v128 :: v_dual_add_f32 v235, v133, v129
	s_set_vgpr_msb 4
	v_cvt_pk_bf16_f32 v133, v140, v250 /*v506*/
	v_cvt_pk_bf16_f32 v131, v138, v228 /*v484*/
	s_set_vgpr_msb 0x405
	v_cvt_pk_bf16_f32 v130, v202 /*v458*/, v212 /*v468*/
	s_set_vgpr_msb 0x504
	v_cvt_pk_bf16_f32 v129, v136, v194 /*v450*/
	s_set_vgpr_msb 0x400
	v_cvt_pk_bf16_f32 v128, v208, v220
	v_cvt_pk_bf16_f32 v143, v143, v233
	s_set_vgpr_msb 5
	v_cvt_pk_bf16_f32 v142, v253 /*v509*/, v255 /*v511*/
	s_set_vgpr_msb 0x504
	v_cvt_pk_bf16_f32 v141, v141, v251 /*v507*/
	s_set_vgpr_msb 0x405
	v_cvt_pk_bf16_f32 v140, v237 /*v493*/, v245 /*v501*/
	s_set_vgpr_msb 0x504
	v_cvt_pk_bf16_f32 v139, v139, v229 /*v485*/
	s_set_vgpr_msb 0x405
	v_cvt_pk_bf16_f32 v138, v203 /*v459*/, v213 /*v469*/
	s_set_vgpr_msb 0x504
	v_cvt_pk_bf16_f32 v137, v137, v195 /*v451*/
	s_set_vgpr_msb 0x400
	v_cvt_pk_bf16_f32 v136, v209, v221
	s_set_vgpr_msb 1
	v_wmma_f32_16x16x32_bf16 v[88:95], v[56:63] /*v[312:319]*/, v[128:135], v[88:95]
	s_set_vgpr_msb 0x105
	v_cvt_pk_bf16_f32 v167, v238 /*v494*/, v246 /*v502*/
	v_cvt_pk_bf16_f32 v166, v224 /*v480*/, v234 /*v490*/
	v_cvt_pk_bf16_f32 v165, v206 /*v462*/, v216 /*v472*/
	s_set_vgpr_msb 0x500
	v_cvt_pk_bf16_f32 v164, v216, v226
	v_cvt_pk_bf16_f32 v163, v200, v210
	v_cvt_pk_bf16_f32 v161, v148, v154
	s_set_vgpr_msb 5
	v_cvt_pk_bf16_f32 v175, v239 /*v495*/, v247 /*v503*/
	s_set_vgpr_msb 0x501
	v_wmma_f32_16x16x32_bf16 v[24:31], v[56:63] /*v[312:319]*/, v[136:143], v[24:31]
	s_set_vgpr_msb 0x105
	v_cvt_pk_bf16_f32 v174, v225 /*v481*/, v235 /*v491*/
	v_cvt_pk_bf16_f32 v173, v207 /*v463*/, v217 /*v473*/
	s_set_vgpr_msb 0x500
	v_cvt_pk_bf16_f32 v172, v217, v227
	v_cvt_pk_bf16_f32 v171, v201, v211
	v_cvt_pk_bf16_f32 v169, v149, v155
	v_cvt_pk_bf16_f32 v168, v145, v147
	s_set_vgpr_msb 5
	v_cvt_pk_bf16_f32 v183, v222 /*v478*/, v232 /*v488*/
	v_cvt_pk_bf16_f32 v182, v210 /*v466*/, v220 /*v476*/
	v_cvt_pk_bf16_f32 v181, v198 /*v454*/, v208 /*v464*/
	s_set_vgpr_msb 0x504
	v_cvt_pk_bf16_f32 v180, v228, v196 /*v452*/
	s_set_vgpr_msb 0x400
	v_cvt_pk_bf16_f32 v179, v214, v224
	v_cvt_pk_bf16_f32 v178, v202, v212
	v_cvt_pk_bf16_f32 v176, v150, v158
	s_set_vgpr_msb 5
	v_cvt_pk_bf16_f32 v191, v223 /*v479*/, v233 /*v489*/
	v_cvt_pk_bf16_f32 v190, v211 /*v467*/, v221 /*v477*/
	v_cvt_pk_bf16_f32 v189, v199 /*v455*/, v209 /*v465*/
	s_set_vgpr_msb 0x504
	v_cvt_pk_bf16_f32 v188, v229, v197 /*v453*/
	s_set_vgpr_msb 0x400
	v_cvt_pk_bf16_f32 v187, v215, v225
	v_cvt_pk_bf16_f32 v186, v203, v213
	v_cvt_pk_bf16_f32 v184, v151, v159
	s_set_vgpr_msb 5
	v_cvt_pk_bf16_f32 v151, v242 /*v498*/, v248 /*v504*/
	v_cvt_pk_bf16_f32 v150, v230 /*v486*/, v240 /*v496*/
	v_cvt_pk_bf16_f32 v149, v218 /*v474*/, v226 /*v482*/
	v_cvt_pk_bf16_f32 v148, v204 /*v460*/, v214 /*v470*/
	v_cvt_pk_bf16_f32 v147, v192 /*v448*/, v200 /*v456*/
	s_set_vgpr_msb 0x500
	v_cvt_pk_bf16_f32 v146, v222, v230
	v_cvt_pk_bf16_f32 v145, v206, v218
	s_set_vgpr_msb 5
	v_cvt_pk_bf16_f32 v159, v243 /*v499*/, v249 /*v505*/
	v_cvt_pk_bf16_f32 v158, v231 /*v487*/, v241 /*v497*/
	v_cvt_pk_bf16_f32 v157, v219 /*v475*/, v227 /*v483*/
	v_cvt_pk_bf16_f32 v156, v205 /*v461*/, v215 /*v471*/
	v_cvt_pk_bf16_f32 v155, v193 /*v449*/, v201 /*v457*/
	s_set_vgpr_msb 0x500
	v_cvt_pk_bf16_f32 v154, v223, v231
	v_cvt_pk_bf16_f32 v153, v207, v219
	s_set_vgpr_msb 1
	v_wmma_f32_16x16x32_bf16 v[120:127], v[184:191] /*v[440:447]*/, v[128:135], v[120:127]
	s_set_vgpr_msb 0x141
	v_dual_mov_b32 v180 /*v436*/, v182 /*v438*/ :: v_dual_mov_b32 v177 /*v433*/, v181 /*v437*/
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b32 v182 /*v438*/, off, off offset:396 th:TH_LOAD_LU nv
	scratch_load_b32 v181 /*v437*/, off, off offset:364 th:TH_LOAD_LU nv
	s_add_nc_u64 s[52:53], s[52:53], 1
	s_set_vgpr_msb 0x4140
	v_mov_b32_e32 v32 /*v288*/, v236
	v_cmp_ge_u64_e64 s2, s[52:53], s[10:11]
	s_set_vgpr_msb 0x4001
	v_wmma_f32_16x16x32_bf16 v[112:119], v[152:159] /*v[408:415]*/, v[128:135], v[112:119]
	s_set_vgpr_msb 0x141
	v_mov_b32_e32 v176 /*v432*/, v55 /*v311*/
	s_and_b32 vcc_lo, exec_lo, s2
	s_set_vgpr_msb 0x4101
	v_wmma_f32_16x16x32_bf16 v[104:111], v[120:127] /*v[376:383]*/, v[128:135], v[104:111]
	v_wmma_f32_16x16x32_bf16 v[96:103], v[88:95] /*v[344:351]*/, v[128:135], v[96:103]
	v_wmma_f32_16x16x32_bf16 v[56:63], v[184:191] /*v[440:447]*/, v[136:143], v[56:63]
	v_wmma_f32_16x16x32_bf16 v[48:55], v[152:159] /*v[408:415]*/, v[136:143], v[48:55]
	v_wmma_f32_16x16x32_bf16 v[40:47], v[120:127] /*v[376:383]*/, v[136:143], v[40:47]
	v_wmma_f32_16x16x32_bf16 v[32:39], v[88:95] /*v[344:351]*/, v[136:143], v[32:39]
	v_wmma_f32_16x16x32_bf16 v[120:127], v[46:53] /*v[302:309]*/, v[160:167], v[120:127]
	v_wmma_f32_16x16x32_bf16 v[56:63], v[46:53] /*v[302:309]*/, v[168:175], v[56:63]
	v_wmma_f32_16x16x32_bf16 v[112:119], v[144:151] /*v[400:407]*/, v[160:167], v[112:119]
	v_wmma_f32_16x16x32_bf16 v[48:55], v[144:151] /*v[400:407]*/, v[168:175], v[48:55]
	v_wmma_f32_16x16x32_bf16 v[104:111], v[112:119] /*v[368:375]*/, v[160:167], v[104:111]
	v_wmma_f32_16x16x32_bf16 v[40:47], v[112:119] /*v[368:375]*/, v[168:175], v[40:47]
	v_wmma_f32_16x16x32_bf16 v[96:103], v[80:87] /*v[336:343]*/, v[160:167], v[96:103]
	v_wmma_f32_16x16x32_bf16 v[32:39], v[80:87] /*v[336:343]*/, v[168:175], v[32:39]
	v_wmma_f32_16x16x32_bf16 v[120:127], v[168:175] /*v[424:431]*/, v[176:183], v[120:127]
	v_wmma_f32_16x16x32_bf16 v[56:63], v[168:175] /*v[424:431]*/, v[184:191], v[56:63]
	v_wmma_f32_16x16x32_bf16 v[112:119], v[136:143] /*v[392:399]*/, v[176:183], v[112:119]
	v_wmma_f32_16x16x32_bf16 v[48:55], v[136:143] /*v[392:399]*/, v[184:191], v[48:55]
	v_wmma_f32_16x16x32_bf16 v[104:111], v[104:111] /*v[360:367]*/, v[176:183], v[104:111]
	v_wmma_f32_16x16x32_bf16 v[40:47], v[104:111] /*v[360:367]*/, v[184:191], v[40:47]
	v_wmma_f32_16x16x32_bf16 v[96:103], v[72:79] /*v[328:335]*/, v[176:183], v[96:103]
	v_wmma_f32_16x16x32_bf16 v[32:39], v[72:79] /*v[328:335]*/, v[184:191], v[32:39]
	v_wmma_f32_16x16x32_bf16 v[120:127], v[160:167] /*v[416:423]*/, v[144:151], v[120:127]
	v_wmma_f32_16x16x32_bf16 v[56:63], v[160:167] /*v[416:423]*/, v[152:159], v[56:63]
	v_wmma_f32_16x16x32_bf16 v[112:119], v[128:135] /*v[384:391]*/, v[144:151], v[112:119]
	v_wmma_f32_16x16x32_bf16 v[48:55], v[128:135] /*v[384:391]*/, v[152:159], v[48:55]
	v_wmma_f32_16x16x32_bf16 v[104:111], v[96:103] /*v[352:359]*/, v[144:151], v[104:111]
	v_wmma_f32_16x16x32_bf16 v[40:47], v[96:103] /*v[352:359]*/, v[152:159], v[40:47]
	v_wmma_f32_16x16x32_bf16 v[96:103], v[64:71] /*v[320:327]*/, v[144:151], v[96:103]
	s_set_vgpr_msb 0x100
	s_wait_loadcnt 0x2
	v_wmma_f32_16x16x32_bf16 v[88:95], v[192:199], v[160:167], v[88:95]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[192:199], v[168:175], v[24:31]
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b128 v[192:195], off, off offset:844 th:TH_LOAD_LU nv
	scratch_load_b128 v[196:199], off, off offset:860 th:TH_LOAD_LU nv
	s_set_vgpr_msb 1
	v_wmma_f32_16x16x32_bf16 v[32:39], v[64:71] /*v[320:327]*/, v[152:159], v[32:39]
	s_set_vgpr_msb 0x100
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[88:95], v[192:199], v[176:183], v[88:95]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[192:199], v[184:191], v[24:31]
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b128 v[192:195], off, off offset:812 th:TH_LOAD_LU nv
	scratch_load_b128 v[196:199], off, off offset:828 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[88:95], v[192:199], v[144:151], v[88:95]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[192:199], v[152:159], v[24:31]
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b128 v[192:195], off, off offset:780 th:TH_LOAD_LU nv
	scratch_load_b128 v[196:199], off, off offset:796 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[80:87], v[192:199], v[128:135], v[80:87]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[192:199], v[136:143], v[16:23]
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b128 v[192:195], off, off offset:748 th:TH_LOAD_LU nv
	scratch_load_b128 v[196:199], off, off offset:764 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[80:87], v[192:199], v[160:167], v[80:87]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[192:199], v[168:175], v[16:23]
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b128 v[192:195], off, off offset:716 th:TH_LOAD_LU nv
	scratch_load_b128 v[196:199], off, off offset:732 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[80:87], v[192:199], v[176:183], v[80:87]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[192:199], v[184:191], v[16:23]
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b128 v[192:195], off, off offset:684 th:TH_LOAD_LU nv
	scratch_load_b128 v[196:199], off, off offset:700 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[80:87], v[192:199], v[144:151], v[80:87]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[192:199], v[152:159], v[16:23]
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b128 v[192:195], off, off offset:652 th:TH_LOAD_LU nv
	scratch_load_b128 v[196:199], off, off offset:668 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[72:79], v[192:199], v[128:135], v[72:79]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[192:199], v[136:143], v[8:15]
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b128 v[192:195], off, off offset:620 th:TH_LOAD_LU nv
	scratch_load_b128 v[196:199], off, off offset:636 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[72:79], v[192:199], v[160:167], v[72:79]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[192:199], v[168:175], v[8:15]
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b128 v[192:195], off, off offset:588 th:TH_LOAD_LU nv
	scratch_load_b128 v[196:199], off, off offset:604 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[72:79], v[192:199], v[176:183], v[72:79]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[192:199], v[184:191], v[8:15]
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b128 v[192:195], off, off offset:556 th:TH_LOAD_LU nv
	scratch_load_b128 v[196:199], off, off offset:572 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[72:79], v[192:199], v[144:151], v[72:79]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[192:199], v[152:159], v[8:15]
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b128 v[192:195], off, off offset:524 th:TH_LOAD_LU nv
	scratch_load_b128 v[196:199], off, off offset:540 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[64:71], v[192:199], v[128:135], v[64:71]
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b128 v[128:131], off, off offset:492 th:TH_LOAD_LU nv
	scratch_load_b128 v[132:135], off, off offset:508 th:TH_LOAD_LU nv
	v_wmma_f32_16x16x32_bf16 v[0:7], v[192:199], v[136:143], v[0:7]
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[64:71], v[128:135], v[160:167], v[64:71]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[128:135], v[168:175], v[0:7]
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b128 v[128:131], off, off offset:428 th:TH_LOAD_LU nv
	scratch_load_b128 v[132:135], off, off offset:444 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[64:71], v[128:135], v[176:183], v[64:71]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[128:135], v[184:191], v[0:7]
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b128 v[128:131], off, off offset:460 th:TH_LOAD_LU nv
	scratch_load_b128 v[132:135], off, off offset:476 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[64:71], v[128:135], v[144:151], v[64:71]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[128:135], v[152:159], v[0:7]
	s_cbranch_vccnz .LBB0_22
.LBB0_16:
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 4
	s_clause 0x9
	scratch_store_b32 off, v32 /*v288*/, off offset:8 nv
	s_set_vgpr_msb 0x400
	scratch_store_b128 off, v[16:19], off offset:72 nv
	scratch_store_b128 off, v[20:23], off offset:88 nv
	scratch_store_b128 off, v[8:11], off offset:40 nv
	scratch_store_b128 off, v[12:15], off offset:56 nv
	scratch_store_b64 off, v[234:235], off nv
	s_set_vgpr_msb 4
	scratch_store_b32 off, v177 /*v433*/, off offset:364 nv
	scratch_store_b32 off, v180 /*v436*/, off offset:396 nv
	s_add_co_i32 s2, s52, 1
	s_wait_tensorcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	s_cmp_ge_i32 s2, s10
	s_set_vgpr_msb 0x400
	s_cbranch_scc1 .LBB0_18
	s_lshl_b32 s5, s2, 7
	s_lshl_b32 s2, s2, 17
	s_sub_co_i32 s7, s9, s5
	s_add_co_i32 s6, s5, s64
	v_nop
	v_nop
	v_nop
	v_nop
	v_med3_i32 v128, s7, 0, 0x80
	s_ashr_i32 s7, s6, 31
	s_and_b32 s2, s2, 0x20000
	s_mul_u64 s[38:39], s[6:7], s[62:63]
	s_mul_u64 s[6:7], s[6:7], s[60:61]
	v_readfirstlane_b32 s37, v128
	s_lshl_b64 s[6:7], s[6:7], 1
	s_lshl_b64 s[38:39], s[38:39], 1
	s_add_nc_u64 s[6:7], s[80:81], s[6:7]
	s_or_b32 s5, s92, s2
	s_sub_co_i32 s37, s37, s17
	s_add_nc_u64 s[6:7], s[18:19], s[6:7]
	s_max_i32 s37, s37, 0
	s_add_nc_u64 s[38:39], s[54:55], s[38:39]
	s_lshl_b32 s37, s37, 16
	s_bitset1_b32 s7, 31
	s_or_b32 s46, s37, 0x7fff
	s_mov_b32 s37, s45
	tensor_load_to_lds s[4:7], s[44:51]
	s_add_nc_u64 s[6:7], s[78:79], s[38:39]
	s_or_b32 s5, s93, s2
	s_bitset1_b32 s7, 31
	s_mov_b32 s38, s46
	s_mov_b32 s39, s47
	s_mov_b32 s40, s48
	s_mov_b32 s43, s51
	tensor_load_to_lds s[4:7], s[36:43]
.LBB0_18:
	s_wait_alu depctr_va_vdst(0) depctr_vm_vsrc(6)
	s_set_vgpr_msb 1
	ds_load_b128 v[128:131], v181 /*v437*/
	ds_load_b128 v[132:135], v181 /*v437*/ offset:32
	ds_load_b128 v[136:139], v181 /*v437*/ offset:64
	ds_load_b128 v[140:143], v181 /*v437*/ offset:96
	ds_load_b128 v[152:155], v181 /*v437*/ offset:128
	ds_load_b128 v[156:159], v181 /*v437*/ offset:160
	ds_load_b128 v[160:163], v181 /*v437*/ offset:192
	ds_load_b128 v[164:167], v181 /*v437*/ offset:224
	ds_load_b128 v[168:171], v181 /*v437*/ offset:4352
	ds_load_b128 v[172:175], v181 /*v437*/ offset:4384
	ds_load_b128 v[176:179], v181 /*v437*/ offset:4416
	ds_load_b128 v[180:183], v181 /*v437*/ offset:4448
	ds_load_b128 v[184:187], v181 /*v437*/ offset:4480
	ds_load_b128 v[188:191], v181 /*v437*/ offset:4512
	ds_load_b128 v[192:195], v181 /*v437*/ offset:4544
	ds_load_b128 v[196:199], v181 /*v437*/ offset:4576
	ds_load_b128 v[200:203], v181 /*v437*/ offset:8704
	ds_load_b128 v[204:207], v181 /*v437*/ offset:8736
	ds_load_b128 v[208:211], v181 /*v437*/ offset:8768
	ds_load_b128 v[212:215], v181 /*v437*/ offset:8800
	ds_load_b128 v[216:219], v181 /*v437*/ offset:8832
	ds_load_b128 v[220:223], v181 /*v437*/ offset:8864
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_b128 v[232:235], v181 /*v437*/ offset:8896
	ds_load_b128 v[236:239], v181 /*v437*/ offset:8928
	ds_load_b128 v[240:243], v181 /*v437*/ offset:13056
	ds_load_b128 v[244:247], v181 /*v437*/ offset:13088
	ds_load_b128 v[248:251], v181 /*v437*/ offset:13120
	ds_load_b128 v[252:255], v181 /*v437*/ offset:13152
	s_set_vgpr_msb 0x141
	ds_load_b128 v[0:3] /*v[256:259]*/, v181 /*v437*/ offset:13184
	ds_load_b128 v[4:7] /*v[260:263]*/, v181 /*v437*/ offset:13216
	ds_load_b128 v[8:11] /*v[264:267]*/, v181 /*v437*/ offset:13248
	ds_load_b128 v[12:15] /*v[268:271]*/, v181 /*v437*/ offset:13280
	ds_load_b128 v[16:19] /*v[272:275]*/, v181 /*v437*/ offset:17408
	ds_load_b128 v[20:23] /*v[276:279]*/, v181 /*v437*/ offset:17440
	ds_load_b128 v[24:27] /*v[280:283]*/, v181 /*v437*/ offset:17472
	ds_load_b128 v[28:31] /*v[284:287]*/, v181 /*v437*/ offset:17504
	ds_load_b128 v[32:35] /*v[288:291]*/, v181 /*v437*/ offset:17536
	ds_load_b128 v[36:39] /*v[292:295]*/, v181 /*v437*/ offset:17568
	ds_load_b128 v[40:43] /*v[296:299]*/, v181 /*v437*/ offset:17600
	ds_load_b128 v[44:47] /*v[300:303]*/, v181 /*v437*/ offset:17632
	ds_load_b128 v[48:51] /*v[304:307]*/, v181 /*v437*/ offset:21760
	ds_load_b128 v[52:55] /*v[308:311]*/, v181 /*v437*/ offset:21792
	ds_load_b128 v[56:59] /*v[312:315]*/, v181 /*v437*/ offset:21824
	ds_load_b128 v[60:63] /*v[316:319]*/, v181 /*v437*/ offset:21856
	ds_load_b128 v[64:67] /*v[320:323]*/, v181 /*v437*/ offset:21888
	ds_load_b128 v[68:71] /*v[324:327]*/, v181 /*v437*/ offset:21920
	ds_load_b128 v[72:75] /*v[328:331]*/, v181 /*v437*/ offset:21952
	ds_load_b128 v[76:79] /*v[332:335]*/, v181 /*v437*/ offset:21984
	ds_load_b128 v[80:83] /*v[336:339]*/, v181 /*v437*/ offset:26112
	ds_load_b128 v[84:87] /*v[340:343]*/, v181 /*v437*/ offset:26144
	ds_load_b128 v[88:91] /*v[344:347]*/, v181 /*v437*/ offset:26176
	ds_load_b128 v[92:95] /*v[348:351]*/, v181 /*v437*/ offset:26208
	ds_load_b128 v[96:99] /*v[352:355]*/, v181 /*v437*/ offset:26240
	ds_load_b128 v[100:103] /*v[356:359]*/, v181 /*v437*/ offset:26272
	ds_load_b128 v[104:107] /*v[360:363]*/, v181 /*v437*/ offset:26304
	ds_load_b128 v[108:111] /*v[364:367]*/, v181 /*v437*/ offset:26336
	ds_load_b128 v[112:115] /*v[368:371]*/, v181 /*v437*/ offset:30464
	ds_load_b128 v[116:119] /*v[372:375]*/, v181 /*v437*/ offset:30496
	ds_load_b128 v[120:123] /*v[376:379]*/, v181 /*v437*/ offset:30528
	ds_load_b128 v[124:127] /*v[380:383]*/, v181 /*v437*/ offset:30560
	ds_load_b128 v[128:131] /*v[384:387]*/, v181 /*v437*/ offset:30592
	ds_load_b128 v[132:135] /*v[388:391]*/, v181 /*v437*/ offset:30624
	ds_load_b128 v[136:139] /*v[392:395]*/, v181 /*v437*/ offset:30656
	ds_load_b128 v[140:143] /*v[396:399]*/, v181 /*v437*/ offset:30688
	s_clause 0xe
	scratch_load_b128 v[144:147] /*v[400:403]*/, off, off offset:108 nv
	scratch_load_b128 v[148:151] /*v[404:407]*/, off, off offset:124 nv
	scratch_load_b128 v[168:171] /*v[424:427]*/, off, off offset:236 nv
	scratch_load_b128 v[172:175] /*v[428:431]*/, off, off offset:252 nv
	scratch_load_b128 v[152:155] /*v[408:411]*/, off, off offset:140 nv
	scratch_load_b128 v[156:159] /*v[412:415]*/, off, off offset:156 nv
	scratch_load_b128 v[184:187] /*v[440:443]*/, off, off offset:268 nv
	scratch_load_b128 v[188:191] /*v[444:447]*/, off, off offset:284 nv
	scratch_load_b128 v[160:163] /*v[416:419]*/, off, off offset:172 nv
	scratch_load_b128 v[164:167] /*v[420:423]*/, off, off offset:188 nv
	s_set_vgpr_msb 0x4104
	scratch_load_b128 v[8:11], off, off offset:300 nv
	scratch_load_b128 v[12:15], off, off offset:316 nv
	scratch_load_b128 v[16:19], off, off offset:332 nv
	scratch_load_b128 v[20:23], off, off offset:348 nv
	s_wait_loadcnt_dscnt 0xc3e
	v_wmma_f32_16x16x32_bf16 v[144:151], v[128:135], v[144:151] /*v[400:407]*/, 0
	s_set_vgpr_msb 0x444
	s_wait_loadcnt 0xa
	v_wmma_f32_16x16x32_bf16 v[192:199] /*v[448:455]*/, v[128:135], v[168:175] /*v[424:431]*/, 0
	s_set_vgpr_msb 0x4404
	s_wait_loadcnt_dscnt 0x83c
	v_wmma_f32_16x16x32_bf16 v[144:151], v[136:143], v[152:159] /*v[408:415]*/, v[144:151]
	s_set_vgpr_msb 0x454
	s_wait_loadcnt 0x6
	v_wmma_f32_16x16x32_bf16 v[192:199] /*v[448:455]*/, v[136:143], v[184:191] /*v[440:447]*/, v[192:199] /*v[448:455]*/
	s_set_vgpr_msb 0x5404
	s_wait_loadcnt_dscnt 0x43a
	v_wmma_f32_16x16x32_bf16 v[144:151], v[152:159], v[160:167] /*v[416:423]*/, v[144:151]
	s_set_vgpr_msb 0x450
	s_wait_loadcnt 0x2
	v_wmma_f32_16x16x32_bf16 v[192:199] /*v[448:455]*/, v[152:159], v[8:15], v[192:199] /*v[448:455]*/
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x5004
	s_clause 0x1
	scratch_load_b128 v[152:155], off, off offset:204 nv
	scratch_load_b128 v[156:159], off, off offset:220 nv
	s_wait_dscnt 0x36
	v_wmma_f32_16x16x32_bf16 v[136:143], v[168:175], v[144:151] /*v[400:407]*/, 0
	s_set_vgpr_msb 0x444
	v_wmma_f32_16x16x32_bf16 v[200:207] /*v[456:463]*/, v[168:175], v[168:175] /*v[424:431]*/, 0
	s_set_vgpr_msb 0x4404
	s_wait_dscnt 0x34
	v_wmma_f32_16x16x32_bf16 v[136:143], v[176:183], v[152:159] /*v[408:415]*/, v[136:143]
	s_set_vgpr_msb 0x454
	v_wmma_f32_16x16x32_bf16 v[200:207] /*v[456:463]*/, v[176:183], v[184:191] /*v[440:447]*/, v[200:207] /*v[456:463]*/
	s_set_vgpr_msb 0x5404
	s_wait_dscnt 0x2e
	v_wmma_f32_16x16x32_bf16 v[224:231], v[200:207], v[144:151] /*v[400:407]*/, 0
	s_set_vgpr_msb 0x444
	v_wmma_f32_16x16x32_bf16 v[208:215] /*v[464:471]*/, v[200:207], v[168:175] /*v[424:431]*/, 0
	s_set_vgpr_msb 0x4404
	v_wmma_f32_16x16x32_bf16 v[136:143], v[184:191], v[160:167] /*v[416:423]*/, v[136:143]
	s_set_vgpr_msb 0x450
	v_wmma_f32_16x16x32_bf16 v[200:207] /*v[456:463]*/, v[184:191], v[8:15], v[200:207] /*v[456:463]*/
	s_set_vgpr_msb 0x5004
	s_wait_dscnt 0x2c
	v_wmma_f32_16x16x32_bf16 v[224:231], v[208:215], v[152:159] /*v[408:415]*/, v[224:231]
	s_set_vgpr_msb 0x454
	v_wmma_f32_16x16x32_bf16 v[208:215] /*v[464:471]*/, v[208:215], v[184:191] /*v[440:447]*/, v[208:215] /*v[464:471]*/
	s_set_vgpr_msb 0x5450
	s_wait_loadcnt 0x2
	v_wmma_f32_16x16x32_bf16 v[200:207] /*v[456:463]*/, v[192:199], v[16:23], v[200:207] /*v[456:463]*/
	s_set_vgpr_msb 0x5004
	s_wait_dscnt 0x2a
	v_wmma_f32_16x16x32_bf16 v[224:231], v[216:223], v[160:167] /*v[416:423]*/, v[224:231]
	s_set_vgpr_msb 0x450
	v_wmma_f32_16x16x32_bf16 v[208:215] /*v[464:471]*/, v[216:223], v[8:15], v[208:215] /*v[464:471]*/
	s_set_vgpr_msb 0x5004
	s_wait_dscnt 0x26
	v_wmma_f32_16x16x32_bf16 v[128:135], v[240:247], v[144:151] /*v[400:407]*/, 0
	s_set_vgpr_msb 0x444
	v_wmma_f32_16x16x32_bf16 v[216:223] /*v[472:479]*/, v[240:247], v[168:175] /*v[424:431]*/, 0
	s_set_vgpr_msb 0x4405
	s_wait_dscnt 0x1e
	v_wmma_f32_16x16x32_bf16 v[216:223], v[16:23] /*v[272:279]*/, v[144:151] /*v[400:407]*/, 0
	s_set_vgpr_msb 0x545
	v_wmma_f32_16x16x32_bf16 v[224:231] /*v[480:487]*/, v[16:23] /*v[272:279]*/, v[168:175] /*v[424:431]*/, 0
	s_set_vgpr_msb 0x4505
	s_wait_dscnt 0x16
	v_wmma_f32_16x16x32_bf16 v[208:215], v[48:55] /*v[304:311]*/, v[144:151] /*v[400:407]*/, 0
	s_set_vgpr_msb 0x545
	v_wmma_f32_16x16x32_bf16 v[232:239] /*v[488:495]*/, v[48:55] /*v[304:311]*/, v[168:175] /*v[424:431]*/, 0
	s_set_vgpr_msb 0x4505
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x32_bf16 v[200:207], v[80:87] /*v[336:343]*/, v[144:151] /*v[400:407]*/, 0
	s_set_vgpr_msb 0x545
	v_wmma_f32_16x16x32_bf16 v[240:247] /*v[496:503]*/, v[80:87] /*v[336:343]*/, v[168:175] /*v[424:431]*/, 0
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x32_bf16 v[248:255] /*v[504:511]*/, v[112:119] /*v[368:375]*/, v[168:175] /*v[424:431]*/, 0
	s_set_vgpr_msb 0x4504
	v_wmma_f32_16x16x32_bf16 v[128:135], v[248:255], v[152:159] /*v[408:415]*/, v[128:135]
	s_set_vgpr_msb 0x454
	v_wmma_f32_16x16x32_bf16 v[216:223] /*v[472:479]*/, v[248:255], v[184:191] /*v[440:447]*/, v[216:223] /*v[472:479]*/
	s_set_vgpr_msb 0x5405
	v_wmma_f32_16x16x32_bf16 v[216:223], v[24:31] /*v[280:287]*/, v[152:159] /*v[408:415]*/, v[216:223]
	s_set_vgpr_msb 0x555
	v_wmma_f32_16x16x32_bf16 v[224:231] /*v[480:487]*/, v[24:31] /*v[280:287]*/, v[184:191] /*v[440:447]*/, v[224:231] /*v[480:487]*/
	s_set_vgpr_msb 0x5505
	v_wmma_f32_16x16x32_bf16 v[208:215], v[56:63] /*v[312:319]*/, v[152:159] /*v[408:415]*/, v[208:215]
	s_set_vgpr_msb 0x555
	v_wmma_f32_16x16x32_bf16 v[232:239] /*v[488:495]*/, v[56:63] /*v[312:319]*/, v[184:191] /*v[440:447]*/, v[232:239] /*v[488:495]*/
	s_set_vgpr_msb 0x5505
	v_wmma_f32_16x16x32_bf16 v[200:207], v[88:95] /*v[344:351]*/, v[152:159] /*v[408:415]*/, v[200:207]
	s_set_vgpr_msb 0x555
	v_wmma_f32_16x16x32_bf16 v[240:247] /*v[496:503]*/, v[88:95] /*v[344:351]*/, v[184:191] /*v[440:447]*/, v[240:247] /*v[496:503]*/
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x32_bf16 v[248:255] /*v[504:511]*/, v[120:127] /*v[376:383]*/, v[184:191] /*v[440:447]*/, v[248:255] /*v[504:511]*/
	s_set_vgpr_msb 0x5505
	v_wmma_f32_16x16x32_bf16 v[128:135], v[0:7] /*v[256:263]*/, v[160:167] /*v[416:423]*/, v[128:135]
	s_set_vgpr_msb 0x551
	v_wmma_f32_16x16x32_bf16 v[216:223] /*v[472:479]*/, v[0:7] /*v[256:263]*/, v[8:15], v[216:223] /*v[472:479]*/
	s_set_vgpr_msb 0x5105
	v_wmma_f32_16x16x32_bf16 v[216:223], v[32:39] /*v[288:295]*/, v[160:167] /*v[416:423]*/, v[216:223]
	s_set_vgpr_msb 0x551
	v_wmma_f32_16x16x32_bf16 v[224:231] /*v[480:487]*/, v[32:39] /*v[288:295]*/, v[8:15], v[224:231] /*v[480:487]*/
	s_set_vgpr_msb 0x5105
	v_wmma_f32_16x16x32_bf16 v[208:215], v[64:71] /*v[320:327]*/, v[160:167] /*v[416:423]*/, v[208:215]
	s_set_vgpr_msb 0x500
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[136:143], v[192:199], v[152:159], v[136:143]
	s_set_vgpr_msb 5
	v_wmma_f32_16x16x32_bf16 v[192:199], v[112:119] /*v[368:375]*/, v[144:151] /*v[400:407]*/, 0
	v_wmma_f32_16x16x32_bf16 v[192:199], v[120:127] /*v[376:383]*/, v[152:159] /*v[408:415]*/, v[192:199]
	s_set_vgpr_msb 0x551
	v_wmma_f32_16x16x32_bf16 v[232:239] /*v[488:495]*/, v[64:71] /*v[320:327]*/, v[8:15], v[232:239] /*v[488:495]*/
	s_set_vgpr_msb 0x5105
	v_wmma_f32_16x16x32_bf16 v[200:207], v[96:103] /*v[352:359]*/, v[160:167] /*v[416:423]*/, v[200:207]
	s_set_vgpr_msb 0x551
	v_wmma_f32_16x16x32_bf16 v[240:247] /*v[496:503]*/, v[96:103] /*v[352:359]*/, v[8:15], v[240:247] /*v[496:503]*/
	s_set_vgpr_msb 0x5105
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[192:199], v[128:135] /*v[384:391]*/, v[160:167] /*v[416:423]*/, v[192:199]
	s_set_vgpr_msb 0x551
	v_wmma_f32_16x16x32_bf16 v[248:255] /*v[504:511]*/, v[128:135] /*v[384:391]*/, v[8:15], v[248:255] /*v[504:511]*/
	s_set_vgpr_msb 0x5100
	v_wmma_f32_16x16x32_bf16 v[144:151], v[160:167], v[152:159], v[144:151]
	s_set_vgpr_msb 0x50
	v_wmma_f32_16x16x32_bf16 v[192:199] /*v[448:455]*/, v[160:167], v[16:23], v[192:199] /*v[448:455]*/
	s_set_vgpr_msb 0x5000
	v_wmma_f32_16x16x32_bf16 v[224:231], v[232:239], v[152:159], v[224:231]
	s_set_vgpr_msb 0x50
	v_wmma_f32_16x16x32_bf16 v[208:215] /*v[464:471]*/, v[232:239], v[16:23], v[208:215] /*v[464:471]*/
	s_set_vgpr_msb 0x5001
	v_wmma_f32_16x16x32_bf16 v[128:135], v[8:15] /*v[264:271]*/, v[152:159], v[128:135]
	s_set_vgpr_msb 0x151
	v_wmma_f32_16x16x32_bf16 v[216:223] /*v[472:479]*/, v[8:15] /*v[264:271]*/, v[16:23], v[216:223] /*v[472:479]*/
	s_set_vgpr_msb 0x5101
	v_wmma_f32_16x16x32_bf16 v[216:223], v[40:47] /*v[296:303]*/, v[152:159], v[216:223]
	s_set_vgpr_msb 0x151
	v_wmma_f32_16x16x32_bf16 v[224:231] /*v[480:487]*/, v[40:47] /*v[296:303]*/, v[16:23], v[224:231] /*v[480:487]*/
	s_set_vgpr_msb 0x5101
	v_wmma_f32_16x16x32_bf16 v[208:215], v[72:79] /*v[328:335]*/, v[152:159], v[208:215]
	s_set_vgpr_msb 0x151
	v_wmma_f32_16x16x32_bf16 v[232:239] /*v[488:495]*/, v[72:79] /*v[328:335]*/, v[16:23], v[232:239] /*v[488:495]*/
	s_set_vgpr_msb 0x5101
	v_wmma_f32_16x16x32_bf16 v[200:207], v[104:111] /*v[360:367]*/, v[152:159], v[200:207]
	s_set_vgpr_msb 0x151
	v_wmma_f32_16x16x32_bf16 v[240:247] /*v[496:503]*/, v[104:111] /*v[360:367]*/, v[16:23], v[240:247] /*v[496:503]*/
	s_set_vgpr_msb 0x5101
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[192:199], v[136:143] /*v[392:399]*/, v[152:159], v[192:199]
	s_set_vgpr_msb 0x151
	v_wmma_f32_16x16x32_bf16 v[248:255] /*v[504:511]*/, v[136:143] /*v[392:399]*/, v[16:23], v[248:255] /*v[504:511]*/
	s_wait_alu depctr_va_vdst(14)
	ds_load_tr16_b128 v[184:187] /*v[440:443]*/, v182 /*v438*/
	ds_load_tr16_b128 v[152:155] /*v[408:411]*/, v182 /*v438*/ offset:32
	ds_load_tr16_b128 v[188:191] /*v[444:447]*/, v182 /*v438*/ offset:4608
	ds_load_tr16_b128 v[156:159] /*v[412:415]*/, v182 /*v438*/ offset:4640
	s_wait_alu depctr_va_vdst(6)
	ds_load_tr16_b128 v[46:49] /*v[302:305]*/, v182 /*v438*/ offset:9216
	ds_load_tr16_b128 v[144:147] /*v[400:403]*/, v182 /*v438*/ offset:9248
	ds_load_tr16_b128 v[50:53] /*v[306:309]*/, v182 /*v438*/ offset:13824
	ds_load_tr16_b128 v[148:151] /*v[404:407]*/, v182 /*v438*/ offset:13856
	ds_load_tr16_b128 v[168:171] /*v[424:427]*/, v182 /*v438*/ offset:18432
	s_wait_alu depctr_va_vdst(0)
	ds_load_tr16_b128 v[136:139] /*v[392:395]*/, v182 /*v438*/ offset:18464
	ds_load_tr16_b128 v[172:175] /*v[428:431]*/, v182 /*v438*/ offset:23040
	ds_load_tr16_b128 v[140:143] /*v[396:399]*/, v182 /*v438*/ offset:23072
	ds_load_tr16_b128 v[160:163] /*v[416:419]*/, v182 /*v438*/ offset:27648
	ds_load_tr16_b128 v[128:131] /*v[384:387]*/, v182 /*v438*/ offset:27680
	ds_load_tr16_b128 v[164:167] /*v[420:423]*/, v182 /*v438*/ offset:32256
	ds_load_tr16_b128 v[132:135] /*v[388:391]*/, v182 /*v438*/ offset:32288
	ds_load_tr16_b128 v[120:123] /*v[376:379]*/, v182 /*v438*/ offset:64
	ds_load_tr16_b128 v[88:91] /*v[344:347]*/, v182 /*v438*/ offset:96
	ds_load_tr16_b128 v[124:127] /*v[380:383]*/, v182 /*v438*/ offset:4672
	ds_load_tr16_b128 v[92:95] /*v[348:351]*/, v182 /*v438*/ offset:4704
	ds_load_tr16_b128 v[112:115] /*v[368:371]*/, v182 /*v438*/ offset:9280
	ds_load_tr16_b128 v[80:83] /*v[336:339]*/, v182 /*v438*/ offset:9312
	ds_load_tr16_b128 v[116:119] /*v[372:375]*/, v182 /*v438*/ offset:13888
	ds_load_tr16_b128 v[84:87] /*v[340:343]*/, v182 /*v438*/ offset:13920
	ds_load_tr16_b128 v[104:107] /*v[360:363]*/, v182 /*v438*/ offset:18496
	ds_load_tr16_b128 v[72:75] /*v[328:331]*/, v182 /*v438*/ offset:18528
	ds_load_tr16_b128 v[108:111] /*v[364:367]*/, v182 /*v438*/ offset:23104
	ds_load_tr16_b128 v[76:79] /*v[332:335]*/, v182 /*v438*/ offset:23136
	ds_load_tr16_b128 v[96:99] /*v[352:355]*/, v182 /*v438*/ offset:27712
	ds_load_tr16_b128 v[64:67] /*v[320:323]*/, v182 /*v438*/ offset:27744
	ds_load_tr16_b128 v[100:103] /*v[356:359]*/, v182 /*v438*/ offset:32320
	ds_load_tr16_b128 v[68:71] /*v[324:327]*/, v182 /*v438*/ offset:32352
	ds_load_tr16_b128 v[56:59] /*v[312:315]*/, v182 /*v438*/ offset:128
	s_set_vgpr_msb 0x5101
	ds_load_tr16_b128 v[8:11], v182 /*v438*/ offset:160
	s_set_vgpr_msb 0x141
	ds_load_tr16_b128 v[60:63] /*v[316:319]*/, v182 /*v438*/ offset:4736
	s_set_vgpr_msb 0x4101
	ds_load_tr16_b128 v[12:15], v182 /*v438*/ offset:4768
	s_wait_dscnt 0x2
	scratch_store_b128 off, v[8:11], off offset:780 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[12:15], off offset:796 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_tr16_b128 v[12:15], v182 /*v438*/ offset:9344
	ds_load_tr16_b128 v[8:11], v182 /*v438*/ offset:9376
	ds_load_tr16_b128 v[16:19], v182 /*v438*/ offset:13952
	s_wait_dscnt 0x2
	scratch_store_b128 off, v[12:15], off offset:876 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[16:19], off offset:892 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_tr16_b128 v[12:15], v182 /*v438*/ offset:13984
	scratch_store_b128 off, v[8:11], off offset:748 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[12:15], off offset:764 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_tr16_b128 v[12:15], v182 /*v438*/ offset:18560
	ds_load_tr16_b128 v[8:11], v182 /*v438*/ offset:18592
	ds_load_tr16_b128 v[16:19], v182 /*v438*/ offset:23168
	s_wait_dscnt 0x2
	scratch_store_b128 off, v[12:15], off offset:844 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[16:19], off offset:860 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_tr16_b128 v[12:15], v182 /*v438*/ offset:23200
	scratch_store_b128 off, v[8:11], off offset:716 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[12:15], off offset:732 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_tr16_b128 v[12:15], v182 /*v438*/ offset:27776
	ds_load_tr16_b128 v[8:11], v182 /*v438*/ offset:27808
	ds_load_tr16_b128 v[16:19], v182 /*v438*/ offset:32384
	s_wait_dscnt 0x2
	scratch_store_b128 off, v[12:15], off offset:812 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[16:19], off offset:828 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_tr16_b128 v[12:15], v182 /*v438*/ offset:32416
	scratch_store_b128 off, v[8:11], off offset:684 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[12:15], off offset:700 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_tr16_b128 v[12:15], v182 /*v438*/ offset:192
	ds_load_tr16_b128 v[8:11], v182 /*v438*/ offset:224
	ds_load_tr16_b128 v[16:19], v182 /*v438*/ offset:4800
	s_wait_dscnt 0x2
	scratch_store_b128 off, v[12:15], off offset:652 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[16:19], off offset:668 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_tr16_b128 v[12:15], v182 /*v438*/ offset:4832
	scratch_store_b128 off, v[8:11], off offset:524 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[12:15], off offset:540 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_tr16_b128 v[12:15], v182 /*v438*/ offset:9408
	ds_load_tr16_b128 v[8:11], v182 /*v438*/ offset:9440
	ds_load_tr16_b128 v[16:19], v182 /*v438*/ offset:14016
	s_wait_dscnt 0x2
	scratch_store_b128 off, v[12:15], off offset:620 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[16:19], off offset:636 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_tr16_b128 v[12:15], v182 /*v438*/ offset:14048
	scratch_store_b128 off, v[8:11], off offset:492 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[12:15], off offset:508 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_tr16_b128 v[12:15], v182 /*v438*/ offset:18624
	ds_load_tr16_b128 v[8:11], v182 /*v438*/ offset:18656
	ds_load_tr16_b128 v[16:19], v182 /*v438*/ offset:23232
	s_wait_dscnt 0x2
	scratch_store_b128 off, v[12:15], off offset:588 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[16:19], off offset:604 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_tr16_b128 v[12:15], v182 /*v438*/ offset:23264
	scratch_store_b128 off, v[8:11], off offset:428 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[12:15], off offset:444 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_tr16_b128 v[12:15], v182 /*v438*/ offset:27840
	ds_load_tr16_b128 v[8:11], v182 /*v438*/ offset:27872
	ds_load_tr16_b128 v[16:19], v182 /*v438*/ offset:32448
	s_wait_dscnt 0x2
	scratch_store_b128 off, v[12:15], off offset:556 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[16:19], off offset:572 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_tr16_b128 v[12:15], v182 /*v438*/ offset:32480
	scratch_store_b128 off, v[8:11], off offset:460 nv
	s_wait_dscnt 0x0
	s_clause 0x1
	scratch_store_b128 off, v[12:15], off offset:476 nv
	scratch_load_b32 v152, off, off offset:104 nv
	s_wait_alu depctr_vm_vsrc(0)
	scratch_load_b32 v8, off, off offset:8 nv
	s_wait_loadcnt 0x1
	v_lshl_or_b32 v232, s52, 7, v152
	v_cmp_ge_i32_e32 vcc_lo, v178 /*v434*/, v232
	v_dual_add_nc_u32 v247, 17, v232 :: v_dual_bitop2_b32 v233, 2, v232 bitop3:0x54
	v_dual_add_nc_u32 v248, 18, v232 :: v_dual_bitop2_b32 v237, 3, v232 bitop3:0x54
	v_cndmask_b32_e32 v144, 0xff800000, v144, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v178 /*v434*/, v232
	v_or_b32_e32 v240, 4, v232
	v_or_b32_e32 v241, 5, v232
	v_or_b32_e32 v242, 6, v232
	v_or_b32_e32 v243, 7, v232
	v_cndmask_b32_e32 v145, 0xff800000, v145, vcc_lo
	v_cmp_ge_i32_e32 vcc_lo, v178 /*v434*/, v233
	v_dual_add_nc_u32 v249, 23, v232 :: v_dual_bitop2_b32 v246, 16, v232 bitop3:0x54
	v_or_b32_e32 v250, 32, v232
	v_or_b32_e32 v251, 33, v232
	v_cndmask_b32_e32 v146, 0xff800000, v146, vcc_lo
	v_cmp_ge_i32_e32 vcc_lo, v178 /*v434*/, v237
	v_or_b32_e32 v254, 34, v232
	s_set_vgpr_msb 0x140
	v_dual_add_nc_u32 v1 /*v257*/, 51, v232 :: v_dual_add_nc_u32 v2 /*v258*/, 52, v232
	v_dual_add_nc_u32 v3 /*v259*/, 53, v232 :: v_dual_add_nc_u32 v10 /*v266*/, 54, v232
	s_set_vgpr_msb 0x4000
	v_cndmask_b32_e32 v147, 0xff800000, v147, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v240, v178 /*v434*/
	s_set_vgpr_msb 0x440
	v_dual_add_nc_u32 v11 /*v267*/, 55, v232 :: v_dual_bitop2_b32 v12 /*v268*/, 64, v232 bitop3:0x54
	v_or_b32_e32 v13 /*v269*/, 0x41, v232
	v_or_b32_e32 v14 /*v270*/, 0x42, v232
	s_set_vgpr_msb 0x4000
	v_cndmask_b32_e32 v148, 0xff800000, v148, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v241, v178 /*v434*/
	s_set_vgpr_msb 0x400
	v_cndmask_b32_e32 v149, 0xff800000, v149, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v242, v178 /*v434*/
	s_set_vgpr_msb 0x400
	v_cndmask_b32_e32 v150, 0xff800000, v150, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v243, v178 /*v434*/
	s_set_vgpr_msb 0x400
	v_cndmask_b32_e32 v151, 0xff800000, v151, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v246, v178 /*v434*/
	s_set_vgpr_msb 0x400
	v_cndmask_b32_e32 v152, 0xff800000, v136, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v247, v178 /*v434*/
	s_set_vgpr_msb 0x400
	v_dual_cndmask_b32 v153, 0xff800000, v137 :: v_dual_add_nc_u32 v136, 19, v232
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v248, v178 /*v434*/
	s_set_vgpr_msb 0x400
	v_dual_cndmask_b32 v154, 0xff800000, v138 :: v_dual_add_nc_u32 v137, 20, v232
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v136, v178 /*v434*/
	s_set_vgpr_msb 0x400
	v_dual_cndmask_b32 v155, 0xff800000, v139 :: v_dual_add_nc_u32 v138, 21, v232
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v137, v178 /*v434*/
	s_set_vgpr_msb 0x400
	v_add_nc_u32_e32 v139, 22, v232
	v_cndmask_b32_e32 v140, 0xff800000, v140, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v138, v178 /*v434*/
	s_set_vgpr_msb 0x400
	v_cndmask_b32_e32 v141, 0xff800000, v141, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v139, v178 /*v434*/
	s_set_vgpr_msb 0x400
	v_cndmask_b32_e32 v142, 0xff800000, v142, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v249, v178 /*v434*/
	s_set_vgpr_msb 0x400
	v_cndmask_b32_e32 v143, 0xff800000, v143, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v250, v178 /*v434*/
	s_set_vgpr_msb 0x400
	v_cndmask_b32_e32 v156, 0xff800000, v224, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v251, v178 /*v434*/
	s_set_vgpr_msb 0x400
	v_or_b32_e32 v224, 35, v232
	v_cndmask_b32_e32 v157, 0xff800000, v225, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v254, v178 /*v434*/
	s_set_vgpr_msb 0x400
	v_or_b32_e32 v225, 36, v232
	v_cndmask_b32_e32 v158, 0xff800000, v226, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v224, v178 /*v434*/
	s_set_vgpr_msb 0x400
	v_or_b32_e32 v226, 37, v232
	v_cndmask_b32_e32 v159, 0xff800000, v227, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v225, v178 /*v434*/
	s_set_vgpr_msb 0x400
	v_or_b32_e32 v227, 38, v232
	v_cndmask_b32_e32 v160, 0xff800000, v228, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v226, v178 /*v434*/
	s_set_vgpr_msb 0x400
	v_or_b32_e32 v228, 39, v232
	v_cndmask_b32_e32 v161, 0xff800000, v229, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v227, v178 /*v434*/
	s_set_vgpr_msb 0x400
	v_or_b32_e32 v229, 48, v232
	v_cndmask_b32_e32 v162, 0xff800000, v230, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v228, v178 /*v434*/
	s_set_vgpr_msb 0x400
	v_dual_cndmask_b32 v163, 0xff800000, v231 :: v_dual_add_nc_u32 v230, 49, v232
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v229, v178 /*v434*/
	s_set_vgpr_msb 0x400
	v_add_nc_u32_e32 v231, 50, v232
	v_cndmask_b32_e32 v128, 0xff800000, v128, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v230, v178 /*v434*/
	s_set_vgpr_msb 0x400
	v_cndmask_b32_e32 v129, 0xff800000, v129, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v231, v178 /*v434*/
	s_set_vgpr_msb 0x400
	v_cndmask_b32_e32 v130, 0xff800000, v130, vcc_lo
	s_set_vgpr_msb 5
	v_cmp_le_i32_e32 vcc_lo, v1 /*v257*/, v178 /*v434*/
	s_set_vgpr_msb 0x500
	v_cndmask_b32_e32 v131, 0xff800000, v131, vcc_lo
	s_set_vgpr_msb 5
	v_cmp_le_i32_e32 vcc_lo, v2 /*v258*/, v178 /*v434*/
	s_set_vgpr_msb 0x500
	v_cndmask_b32_e32 v132, 0xff800000, v132, vcc_lo
	s_set_vgpr_msb 5
	v_cmp_le_i32_e32 vcc_lo, v3 /*v259*/, v178 /*v434*/
	s_set_vgpr_msb 0x500
	v_cndmask_b32_e32 v133, 0xff800000, v133, vcc_lo
	s_set_vgpr_msb 5
	v_cmp_le_i32_e32 vcc_lo, v10 /*v266*/, v178 /*v434*/
	s_set_vgpr_msb 0x500
	v_cndmask_b32_e32 v134, 0xff800000, v134, vcc_lo
	s_set_vgpr_msb 5
	v_cmp_le_i32_e32 vcc_lo, v11 /*v267*/, v178 /*v434*/
	s_set_vgpr_msb 0x500
	v_cndmask_b32_e32 v135, 0xff800000, v135, vcc_lo
	s_set_vgpr_msb 5
	v_cmp_le_i32_e32 vcc_lo, v12 /*v268*/, v178 /*v434*/
	s_set_vgpr_msb 0x500
	v_cndmask_b32_e32 v164, 0xff800000, v216, vcc_lo
	s_set_vgpr_msb 5
	v_cmp_le_i32_e32 vcc_lo, v13 /*v269*/, v178 /*v434*/
	s_set_vgpr_msb 0x500
	v_or_b32_e32 v216, 0x43, v232
	v_cndmask_b32_e32 v165, 0xff800000, v217, vcc_lo
	s_set_vgpr_msb 5
	v_cmp_le_i32_e32 vcc_lo, v14 /*v270*/, v178 /*v434*/
	s_set_vgpr_msb 0x500
	v_or_b32_e32 v217, 0x44, v232
	v_cndmask_b32_e32 v166, 0xff800000, v218, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v216, v178 /*v434*/
	s_set_vgpr_msb 0x400
	v_or_b32_e32 v218, 0x45, v232
	v_cndmask_b32_e32 v167, 0xff800000, v219, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v217, v178 /*v434*/
	s_set_vgpr_msb 0x400
	v_or_b32_e32 v219, 0x46, v232
	v_cndmask_b32_e32 v168, 0xff800000, v220, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v218, v178 /*v434*/
	s_set_vgpr_msb 0x400
	v_or_b32_e32 v220, 0x47, v232
	v_cndmask_b32_e32 v169, 0xff800000, v221, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v219, v178 /*v434*/
	s_set_vgpr_msb 0x400
	v_or_b32_e32 v221, 0x50, v232
	v_cndmask_b32_e32 v170, 0xff800000, v222, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v220, v178 /*v434*/
	s_set_vgpr_msb 0x400
	v_add_nc_u32_e32 v222, 0x51, v232
	v_cndmask_b32_e32 v171, 0xff800000, v223, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v221, v178 /*v434*/
	s_set_vgpr_msb 0x400
	v_add_nc_u32_e32 v223, 0x52, v232
	v_cndmask_b32_e32 v172, 0xff800000, v208, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v222, v178 /*v434*/
	s_set_vgpr_msb 0x400
	v_add_nc_u32_e32 v208, 0x53, v232
	v_cndmask_b32_e32 v173, 0xff800000, v209, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v223, v178 /*v434*/
	s_set_vgpr_msb 0x400
	v_add_nc_u32_e32 v209, 0x54, v232
	v_cndmask_b32_e32 v174, 0xff800000, v210, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v208, v178 /*v434*/
	s_set_vgpr_msb 0x400
	v_add_nc_u32_e32 v210, 0x55, v232
	v_cndmask_b32_e32 v175, 0xff800000, v211, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v209, v178 /*v434*/
	s_set_vgpr_msb 0x400
	v_add_nc_u32_e32 v211, 0x56, v232
	v_cndmask_b32_e32 v176, 0xff800000, v212, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v210, v178 /*v434*/
	s_set_vgpr_msb 0x400
	v_add_nc_u32_e32 v212, 0x57, v232
	v_cndmask_b32_e32 v177, 0xff800000, v213, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v211, v178 /*v434*/
	s_set_vgpr_msb 0x400
	v_or_b32_e32 v213, 0x60, v232
	v_cndmask_b32_e32 v178, 0xff800000, v214, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v212, v178 /*v434*/
	s_set_vgpr_msb 0x400
	v_or_b32_e32 v214, 0x61, v232
	v_cndmask_b32_e32 v179, 0xff800000, v215, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v213, v178 /*v434*/
	s_set_vgpr_msb 0x400
	v_or_b32_e32 v215, 0x62, v232
	v_cndmask_b32_e32 v180, 0xff800000, v200, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v214, v178 /*v434*/
	s_set_vgpr_msb 0x400
	v_or_b32_e32 v200, 0x63, v232
	v_cndmask_b32_e32 v181, 0xff800000, v201, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v215, v178 /*v434*/
	s_set_vgpr_msb 0x400
	v_or_b32_e32 v201, 0x64, v232
	v_cndmask_b32_e32 v182, 0xff800000, v202, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v200, v178 /*v434*/
	s_set_vgpr_msb 0x400
	v_or_b32_e32 v202, 0x65, v232
	v_cndmask_b32_e32 v183, 0xff800000, v203, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v201, v178 /*v434*/
	s_set_vgpr_msb 0x400
	v_or_b32_e32 v203, 0x66, v232
	v_cndmask_b32_e32 v184, 0xff800000, v204, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v202, v178 /*v434*/
	s_set_vgpr_msb 0x400
	v_or_b32_e32 v204, 0x67, v232
	v_cndmask_b32_e32 v185, 0xff800000, v205, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v203, v178 /*v434*/
	s_set_vgpr_msb 0x400
	v_or_b32_e32 v205, 0x70, v232
	v_cndmask_b32_e32 v186, 0xff800000, v206, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v204, v178 /*v434*/
	s_set_vgpr_msb 0x400
	v_add_nc_u32_e32 v206, 0x71, v232
	v_cndmask_b32_e32 v187, 0xff800000, v207, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v205, v178 /*v434*/
	s_set_vgpr_msb 0x400
	v_add_nc_u32_e32 v207, 0x72, v232
	v_cndmask_b32_e32 v188, 0xff800000, v192, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v206, v178 /*v434*/
	s_set_vgpr_msb 0x400
	v_add_nc_u32_e32 v192, 0x73, v232
	v_cndmask_b32_e32 v189, 0xff800000, v193, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v207, v178 /*v434*/
	s_set_vgpr_msb 0x400
	v_add_nc_u32_e32 v193, 0x74, v232
	v_cndmask_b32_e32 v190, 0xff800000, v194, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v192, v178 /*v434*/
	s_set_vgpr_msb 0x400
	v_add_nc_u32_e32 v194, 0x75, v232
	v_cndmask_b32_e32 v191, 0xff800000, v195, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v193, v178 /*v434*/
	s_set_vgpr_msb 0x400
	v_add_nc_u32_e32 v195, 0x76, v232
	v_cndmask_b32_e32 v234, 0xff800000, v196, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v194, v178 /*v434*/
	s_set_vgpr_msb 0x400
	v_add_nc_u32_e32 v196, 0x77, v232
	v_cndmask_b32_e32 v235, 0xff800000, v197, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v195, v178 /*v434*/
	s_set_vgpr_msb 0x440
	v_cndmask_b32_e32 v4 /*v260*/, 0xff800000, v198, vcc_lo
	s_set_vgpr_msb 0x4004
	v_cmp_le_i32_e32 vcc_lo, v196, v178 /*v434*/
	s_set_vgpr_msb 0x400
	v_max3_num_f32 v198, v143, v156, v157
	s_set_vgpr_msb 64
	v_cndmask_b32_e32 v5 /*v261*/, 0xff800000, v199, vcc_lo
	s_set_vgpr_msb 0x4004
	v_cmp_le_i32_e32 vcc_lo, v232, v179 /*v435*/
	v_cndmask_b32_e32 v238, 0xff800000, v192 /*v448*/, vcc_lo
	v_cmp_lt_i32_e32 vcc_lo, v232, v179 /*v435*/
	v_cndmask_b32_e32 v239, 0xff800000, v193 /*v449*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v233, v179 /*v435*/
	v_cndmask_b32_e32 v236, 0xff800000, v194 /*v450*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v237, v179 /*v435*/
	v_cndmask_b32_e32 v237, 0xff800000, v195 /*v451*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v240, v179 /*v435*/
	v_cndmask_b32_e32 v244, 0xff800000, v196 /*v452*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v241, v179 /*v435*/
	v_cndmask_b32_e32 v245, 0xff800000, v197 /*v453*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v242, v179 /*v435*/
	v_cndmask_b32_e32 v240, 0xff800000, v198 /*v454*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v243, v179 /*v435*/
	v_cndmask_b32_e32 v241, 0xff800000, v199 /*v455*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v246, v179 /*v435*/
	s_set_vgpr_msb 0x444
	v_cndmask_b32_e32 v6 /*v262*/, 0xff800000, v200 /*v456*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v247, v179 /*v435*/
	v_cndmask_b32_e32 v7 /*v263*/, 0xff800000, v201 /*v457*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v248, v179 /*v435*/
	s_set_vgpr_msb 0x4404
	v_cndmask_b32_e32 v246, 0xff800000, v202 /*v458*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v136, v179 /*v435*/
	s_set_vgpr_msb 0x400
	v_max3_num_f32 v136, v144, v145, v146
	s_set_vgpr_msb 4
	v_cndmask_b32_e32 v247, 0xff800000, v203 /*v459*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v137, v179 /*v435*/
	s_set_vgpr_msb 0x400
	v_max3_num_f32 v137, v238, v239, v236
	s_set_vgpr_msb 4
	v_cndmask_b32_e32 v242, 0xff800000, v204 /*v460*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v138, v179 /*v435*/
	s_set_vgpr_msb 0x400
	v_max3_num_f32 v138, v147, v148, v149
	s_set_vgpr_msb 4
	v_cndmask_b32_e32 v243, 0xff800000, v205 /*v461*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v139, v179 /*v435*/
	s_set_vgpr_msb 0x400
	v_max3_num_f32 v139, v237, v244, v245
	s_set_vgpr_msb 4
	v_cndmask_b32_e32 v252, 0xff800000, v206 /*v462*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v249, v179 /*v435*/
	v_cndmask_b32_e32 v253, 0xff800000, v207 /*v463*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v250, v179 /*v435*/
	s_set_vgpr_msb 0x400
	v_max3_num_f32 v197, v242, v243, v252
	s_set_vgpr_msb 4
	v_cndmask_b32_e32 v248, 0xff800000, v208 /*v464*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v251, v179 /*v435*/
	v_cndmask_b32_e32 v249, 0xff800000, v209 /*v465*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v254, v179 /*v435*/
	s_set_vgpr_msb 0x400
	v_max3_num_f32 v199, v253, v248, v249
	s_set_vgpr_msb 0x44
	v_cndmask_b32_e32 v16 /*v272*/, 0xff800000, v210 /*v466*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v224, v179 /*v435*/
	s_set_vgpr_msb 0x4400
	v_max_num_f32_e32 v224, v186, v187
	s_set_vgpr_msb 0x44
	v_cndmask_b32_e32 v17 /*v273*/, 0xff800000, v211 /*v467*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v225, v179 /*v435*/
	s_set_vgpr_msb 0x4404
	v_cndmask_b32_e32 v254, 0xff800000, v212 /*v468*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v226, v179 /*v435*/
	s_set_vgpr_msb 0x400
	v_max3_num_f32 v226, v189, v190, v191
	s_set_vgpr_msb 4
	v_cndmask_b32_e32 v255, 0xff800000, v213 /*v469*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v227, v179 /*v435*/
	v_cndmask_b32_e32 v250, 0xff800000, v214 /*v470*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v228, v179 /*v435*/
	s_set_vgpr_msb 0x410
	v_max3_num_f32 v228, v234, v235, v4 /*v260*/
	s_set_vgpr_msb 0x1004
	v_cndmask_b32_e32 v251, 0xff800000, v215 /*v471*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v229, v179 /*v435*/
	s_set_vgpr_msb 0x444
	v_cndmask_b32_e32 v8 /*v264*/, 0xff800000, v216 /*v472*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v230, v179 /*v435*/
	v_cndmask_b32_e32 v9 /*v265*/, 0xff800000, v217 /*v473*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v231, v179 /*v435*/
	v_cndmask_b32_e32 v0 /*v256*/, 0xff800000, v218 /*v474*/, vcc_lo
	s_set_vgpr_msb 0x4445
	v_cmp_le_i32_e32 vcc_lo, v1 /*v257*/, v179 /*v435*/
	v_cndmask_b32_e32 v1 /*v257*/, 0xff800000, v219 /*v475*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v2 /*v258*/, v179 /*v435*/
	v_cndmask_b32_e32 v26 /*v282*/, 0xff800000, v220 /*v476*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v3 /*v259*/, v179 /*v435*/
	v_cndmask_b32_e32 v27 /*v283*/, 0xff800000, v221 /*v477*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v10 /*v266*/, v179 /*v435*/
	v_cndmask_b32_e32 v10 /*v266*/, 0xff800000, v222 /*v478*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v11 /*v267*/, v179 /*v435*/
	v_cndmask_b32_e32 v11 /*v267*/, 0xff800000, v223 /*v479*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v12 /*v268*/, v179 /*v435*/
	v_cndmask_b32_e32 v2 /*v258*/, 0xff800000, v224 /*v480*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v13 /*v269*/, v179 /*v435*/
	v_cndmask_b32_e32 v3 /*v259*/, 0xff800000, v225 /*v481*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v14 /*v270*/, v179 /*v435*/
	v_cndmask_b32_e32 v18 /*v274*/, 0xff800000, v226 /*v482*/, vcc_lo
	s_set_vgpr_msb 0x4504
	v_cmp_le_i32_e32 vcc_lo, v216, v179 /*v435*/
	s_set_vgpr_msb 0x400
	v_max3_num_f32 v216, v174, v175, v176
	s_set_vgpr_msb 0x44
	v_cndmask_b32_e32 v19 /*v275*/, 0xff800000, v227 /*v483*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v217, v179 /*v435*/
	v_cndmask_b32_e32 v12 /*v268*/, 0xff800000, v228 /*v484*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v218, v179 /*v435*/
	s_set_vgpr_msb 0x4400
	v_max3_num_f32 v218, v177, v178, v179
	s_set_vgpr_msb 0x44
	v_cndmask_b32_e32 v13 /*v269*/, 0xff800000, v229 /*v485*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v219, v179 /*v435*/
	v_cndmask_b32_e32 v40 /*v296*/, 0xff800000, v230 /*v486*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v220, v179 /*v435*/
	s_set_vgpr_msb 0x4400
	v_max3_num_f32 v220, v180, v181, v182
	s_set_vgpr_msb 0x44
	v_cndmask_b32_e32 v41 /*v297*/, 0xff800000, v231 /*v487*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v221, v179 /*v435*/
	v_cndmask_b32_e32 v20 /*v276*/, 0xff800000, v232 /*v488*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v222, v179 /*v435*/
	s_set_vgpr_msb 0x4400
	v_max3_num_f32 v222, v183, v184, v185
	s_set_vgpr_msb 0x44
	v_cndmask_b32_e32 v21 /*v277*/, 0xff800000, v233 /*v489*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v223, v179 /*v435*/
	v_cndmask_b32_e32 v14 /*v270*/, 0xff800000, v234 /*v490*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v208, v179 /*v435*/
	s_set_vgpr_msb 0x4400
	v_max3_num_f32 v208, v134, v135, v164
	s_set_vgpr_msb 0x44
	v_cndmask_b32_e32 v15 /*v271*/, 0xff800000, v235 /*v491*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v209, v179 /*v435*/
	s_set_vgpr_msb 0x4415
	v_max3_num_f32 v209, v10 /*v266*/, v11 /*v267*/, v2 /*v258*/
	s_set_vgpr_msb 0x1544
	v_cndmask_b32_e32 v28 /*v284*/, 0xff800000, v236 /*v492*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v210, v179 /*v435*/
	s_set_vgpr_msb 0x4400
	v_max3_num_f32 v210, v165, v166, v167
	s_set_vgpr_msb 0x44
	v_cndmask_b32_e32 v29 /*v285*/, 0xff800000, v237 /*v493*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v211, v179 /*v435*/
	s_set_vgpr_msb 0x4415
	v_max3_num_f32 v211, v3 /*v259*/, v18 /*v274*/, v19 /*v275*/
	v_max3_num_f32 v217, v14 /*v270*/, v15 /*v271*/, v28 /*v284*/
	s_set_vgpr_msb 0x1544
	v_cndmask_b32_e32 v22 /*v278*/, 0xff800000, v238 /*v494*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v212, v179 /*v435*/
	s_set_vgpr_msb 0x4400
	v_max3_num_f32 v212, v168, v169, v170
	s_set_vgpr_msb 0x44
	v_cndmask_b32_e32 v23 /*v279*/, 0xff800000, v239 /*v495*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v213, v179 /*v435*/
	s_set_vgpr_msb 0x4415
	v_max3_num_f32 v213, v12 /*v268*/, v13 /*v269*/, v40 /*v296*/
	v_max3_num_f32 v219, v29 /*v285*/, v22 /*v278*/, v23 /*v279*/
	s_set_vgpr_msb 0x1544
	v_cndmask_b32_e32 v42 /*v298*/, 0xff800000, v240 /*v496*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v214, v179 /*v435*/
	s_set_vgpr_msb 0x4400
	v_max3_num_f32 v214, v171, v172, v173
	s_set_vgpr_msb 0x44
	v_cndmask_b32_e32 v43 /*v299*/, 0xff800000, v241 /*v497*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v215, v179 /*v435*/
	s_set_vgpr_msb 0x4415
	v_max3_num_f32 v215, v41 /*v297*/, v20 /*v276*/, v21 /*v277*/
	s_set_vgpr_msb 0x1544
	v_cndmask_b32_e32 v30 /*v286*/, 0xff800000, v242 /*v498*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v200, v179 /*v435*/
	s_set_vgpr_msb 0x4400
	v_max3_num_f32 v200, v158, v159, v160
	s_set_vgpr_msb 0x44
	v_cndmask_b32_e32 v31 /*v287*/, 0xff800000, v243 /*v499*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v201, v179 /*v435*/
	s_set_vgpr_msb 0x4415
	v_max3_num_f32 v221, v42 /*v298*/, v43 /*v299*/, v30 /*v286*/
	s_set_vgpr_msb 0x1505
	v_max3_num_f32 v201, v16 /*v272*/, v17 /*v273*/, v254
	s_set_vgpr_msb 0x544
	v_cndmask_b32_e32 v24 /*v280*/, 0xff800000, v244 /*v500*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v202, v179 /*v435*/
	s_set_vgpr_msb 0x4400
	v_max3_num_f32 v202, v161, v162, v163
	s_set_vgpr_msb 0x44
	v_cndmask_b32_e32 v25 /*v281*/, 0xff800000, v245 /*v501*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v203, v179 /*v435*/
	s_set_vgpr_msb 0x4400
	v_max3_num_f32 v203, v255, v250, v251
	s_set_vgpr_msb 21
	v_max3_num_f32 v223, v31 /*v287*/, v24 /*v280*/, v25 /*v281*/
	s_set_vgpr_msb 0x1544
	v_cndmask_b32_e32 v32 /*v288*/, 0xff800000, v246 /*v502*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v204, v179 /*v435*/
	s_set_vgpr_msb 0x4400
	v_max3_num_f32 v204, v128, v129, v130
	s_set_vgpr_msb 0x44
	v_cndmask_b32_e32 v33 /*v289*/, 0xff800000, v247 /*v503*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v205, v179 /*v435*/
	s_set_vgpr_msb 0x4415
	v_max3_num_f32 v205, v8 /*v264*/, v9 /*v265*/, v0 /*v256*/
	v_max_num_f32_e32 v225, v32 /*v288*/, v33 /*v289*/
	s_set_vgpr_msb 0x1544
	v_cndmask_b32_e32 v34 /*v290*/, 0xff800000, v248 /*v504*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v206, v179 /*v435*/
	s_set_vgpr_msb 0x4400
	v_max3_num_f32 v206, v131, v132, v133
	s_set_vgpr_msb 0x44
	v_cndmask_b32_e32 v35 /*v291*/, 0xff800000, v249 /*v505*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v207, v179 /*v435*/
	s_set_vgpr_msb 0x4415
	v_max3_num_f32 v207, v1 /*v257*/, v26 /*v282*/, v27 /*v283*/
	s_set_vgpr_msb 0x1544
	v_cndmask_b32_e32 v44 /*v300*/, 0xff800000, v250 /*v506*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v192, v179 /*v435*/
	s_set_vgpr_msb 0x4400
	v_max3_num_f32 v192, v150, v151, v152
	s_set_vgpr_msb 0x44
	v_cndmask_b32_e32 v45 /*v301*/, 0xff800000, v251 /*v507*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v193, v179 /*v435*/
	s_set_vgpr_msb 0x4410
	v_max3_num_f32 v193, v240, v241, v6 /*v262*/
	s_set_vgpr_msb 0x1000
	v_max3_num_f32 v136, v136, v138, v192
	v_max3_num_f32 v192, v206, v208, v210
	s_set_vgpr_msb 21
	v_max3_num_f32 v227, v35 /*v291*/, v44 /*v300*/, v45 /*v301*/
	s_set_vgpr_msb 0x1544
	v_cndmask_b32_e32 v36 /*v292*/, 0xff800000, v252 /*v508*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v194, v179 /*v435*/
	s_set_vgpr_msb 0x4400
	v_max3_num_f32 v194, v153, v154, v155
	v_max3_num_f32 v137, v137, v139, v193
	v_max3_num_f32 v139, v200, v202, v204
	v_max3_num_f32 v193, v212, v214, v216
	s_set_vgpr_msb 0x44
	v_cndmask_b32_e32 v37 /*v293*/, 0xff800000, v253 /*v509*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v195, v179 /*v435*/
	s_set_vgpr_msb 0x4401
	v_max3_num_f32 v195, v7 /*v263*/, v246, v247
	s_set_vgpr_msb 0x144
	v_cndmask_b32_e32 v38 /*v294*/, 0xff800000, v254 /*v510*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v196, v179 /*v435*/
	s_set_vgpr_msb 0x4400
	v_max3_num_f32 v196, v140, v141, v142
	v_max3_num_f32 v195, v195, v197, v199
	v_max3_num_f32 v197, v201, v203, v205
	s_set_vgpr_msb 0x44
	v_cndmask_b32_e32 v39 /*v295*/, 0xff800000, v255 /*v511*/, vcc_lo
	s_set_vgpr_msb 0x4400
	v_max3_num_f32 v138, v194, v196, v198
	v_max3_num_f32 v194, v218, v220, v222
	v_max3_num_f32 v196, v224, v188, v226
	s_set_vgpr_msb 21
	v_max3_num_f32 v229, v36 /*v292*/, v37 /*v293*/, v38 /*v294*/
	s_set_vgpr_msb 0x1500
	v_max3_num_f32 v137, v137, v195, v197
	v_max3_num_f32 v136, v136, v138, v139
	v_max3_num_f32 v138, v192, v193, v194
	s_set_vgpr_msb 16
	v_max3_num_f32 v139, v196, v228, v5 /*v261*/
	s_set_vgpr_msb 0x1000
	v_max3_num_f32 v192, v207, v209, v211
	v_max3_num_f32 v193, v213, v215, v217
	v_max3_num_f32 v194, v219, v221, v223
	s_set_vgpr_msb 4
	v_max3_num_f32 v196, v225, v34 /*v290*/, v227
	s_set_vgpr_msb 0x400
	v_max3_num_f32 v136, v136, v138, v139
	v_max3_num_f32 v138, v192, v193, v194
	s_set_vgpr_msb 16
	v_max3_num_f32 v139, v196, v229, v39 /*v295*/
	v_mov_b32_e32 v192, v136
	s_set_vgpr_msb 0x1000
	v_max3_num_f32 v137, v137, v138, v139
	v_permlanex16_b32 v192, v192, s82, 0xfedcba98
	v_dual_mov_b32 v138, v137 :: v_dual_max_num_f32 v136, v136, v192
	v_permlanex16_b32 v138, v138, s82, 0xfedcba98
	s_set_vgpr_msb 4
	v_sub_f32_e32 v139, v136, v176 /*v432*/
	v_max_num_f32_e32 v136, v136, v176 /*v432*/
	s_set_vgpr_msb 0x400
	v_max_num_f32_e32 v137, v137, v138
	v_cmp_lt_f32_e32 vcc_lo, 0x41000000, v139
	s_cmp_eq_u32 vcc_lo, 0
	s_cselect_b32 s2, -1, 0
	s_set_vgpr_msb 0x44
	v_cndmask_b32_e64 v55 /*v311*/, v136, v176 /*v432*/, s2
	s_set_vgpr_msb 0x4400
	s_wait_loadcnt 0x0
	v_dual_max_num_f32 v136, v8, v137 :: v_dual_sub_f32 v138, v137, v8
	v_cmp_lt_f32_e64 s2, 0x41000000, v138
	s_cmp_lg_u32 s2, 0
	s_cselect_b32 s5, -1, 0
	s_cmp_eq_u32 s2, 0
	s_cselect_b32 s2, -1, 0
	v_cndmask_b32_e64 v8, v136, v8, s2
	s_set_vgpr_msb 0x44
	v_dual_mul_f32 v54 /*v310*/, 0xbfb8aa3b, v55 /*v311*/ :: v_dual_mov_b32 v177 /*v433*/, v8
	s_set_vgpr_msb 0x4410
	v_pk_fma_f32 v[128:129], v[128:129], s[56:57], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[132:133], v[132:133], s[56:57], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[136:137], v[144:145], s[56:57], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[144:145], v[148:149], s[56:57], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[130:131], v[130:131], s[56:57], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v216, v128
	v_exp_f32_e32 v226, v129
	v_nop
	v_pk_fma_f32 v[128:129], v[134:135], s[56:57], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1040
	v_exp_f32_e32 v224 /*v480*/, v132
	v_exp_f32_e32 v234 /*v490*/, v133
	v_nop
	s_set_vgpr_msb 0x4010
	v_pk_fma_f32 v[132:133], v[166:167], s[56:57], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[138:139], v[146:147], s[56:57], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1040
	v_exp_f32_e32 v238 /*v494*/, v128
	v_exp_f32_e32 v246 /*v502*/, v129
	v_nop
	s_set_vgpr_msb 0x4010
	v_pk_fma_f32 v[128:129], v[168:169], s[56:57], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[146:147], v[150:151], s[56:57], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1040
	v_exp_f32_e32 v202 /*v458*/, v144
	s_set_vgpr_msb 0x4010
	v_pk_fma_f32 v[148:149], v[152:153], s[56:57], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1040
	v_exp_f32_e32 v212 /*v468*/, v145
	v_nop
	s_set_vgpr_msb 0x4010
	v_pk_fma_f32 v[144:145], v[154:155], s[56:57], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[150:151], v[160:161], s[56:57], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1040
	v_exp_f32_e32 v206 /*v462*/, v130
	v_exp_f32_e32 v216 /*v472*/, v131
	v_nop
	s_set_vgpr_msb 0x4010
	v_pk_fma_f32 v[130:131], v[164:165], s[56:57], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v192, v132
	v_exp_f32_e32 v198, v133
	v_exp_f32_e32 v202, v128
	v_pk_fma_f32 v[132:133], v[172:173], s[56:57], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v212, v129
	v_nop
	v_pk_fma_f32 v[128:129], v[174:175], s[56:57], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v208, v136
	v_exp_f32_e32 v136, v138
	v_exp_f32_e32 v138, v146
	s_set_vgpr_msb 0x1040
	v_exp_f32_e32 v228 /*v484*/, v147
	v_exp_f32_e32 v236 /*v492*/, v148
	s_set_vgpr_msb 0x4010
	v_pk_fma_f32 v[146:147], v[140:141], s[56:57], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1040
	v_exp_f32_e32 v244 /*v500*/, v149
	s_set_vgpr_msb 0x4000
	v_exp_f32_e32 v140, v144
	s_set_vgpr_msb 64
	v_exp_f32_e32 v250 /*v506*/, v145
	v_nop
	s_set_vgpr_msb 0x4010
	v_pk_fma_f32 v[144:145], v[156:157], s[56:57], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[148:149], v[158:159], s[56:57], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v156, v150
	v_exp_f32_e32 v150, v130
	v_exp_f32_e32 v158, v131
	v_nop
	v_pk_fma_f32 v[130:131], v[170:171], s[56:57], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v228, v132
	s_set_vgpr_msb 0x1040
	v_exp_f32_e32 v196 /*v452*/, v133
	v_exp_f32_e32 v198 /*v454*/, v128
	s_set_vgpr_msb 0x4010
	v_pk_fma_f32 v[132:133], v[178:179], s[56:57], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1040
	v_exp_f32_e32 v208 /*v464*/, v129
	v_nop
	s_set_vgpr_msb 0x4010
	v_pk_fma_f32 v[128:129], v[180:181], s[56:57], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v214, v130
	v_exp_f32_e32 v224, v131
	v_nop
	v_pk_fma_f32 v[130:131], v[176:177], s[56:57], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1040
	v_exp_f32_e32 v222 /*v478*/, v132
	v_exp_f32_e32 v232 /*v488*/, v133
	s_set_vgpr_msb 0x4010
	v_exp_f32_e32 v196, v128
	v_pk_fma_f32 v[132:133], v[184:185], s[56:57], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v204, v129
	v_nop
	v_pk_fma_f32 v[128:129], v[186:187], s[56:57], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1040
	v_exp_f32_e32 v210 /*v466*/, v130
	v_exp_f32_e32 v220 /*v476*/, v131
	v_nop
	s_set_vgpr_msb 0x4010
	v_pk_fma_f32 v[130:131], v[182:183], s[56:57], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1040
	v_mul_f32_e32 v180 /*v436*/, 0xbfb8aa3b, v8
	s_set_vgpr_msb 0x4000
	v_exp_f32_e32 v222, v132
	v_exp_f32_e32 v230, v133
	s_set_vgpr_msb 64
	v_exp_f32_e32 v192 /*v448*/, v128
	s_set_vgpr_msb 0x4010
	v_pk_fma_f32 v[132:133], v[190:191], s[56:57], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1040
	v_exp_f32_e32 v200 /*v456*/, v129
	v_nop
	s_set_vgpr_msb 0x4010
	v_pk_fma_f32 v[128:129], v[234:235], s[56:57], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v206, v130
	v_exp_f32_e32 v218, v131
	v_nop
	v_pk_fma_f32 v[130:131], v[188:189], s[56:57], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1040
	v_exp_f32_e32 v218 /*v474*/, v132
	v_exp_f32_e32 v226 /*v482*/, v133
	v_exp_f32_e32 v230 /*v486*/, v128
	s_set_vgpr_msb 0x4010
	v_pk_fma_f32 v[132:133], v[238:239], s[56:57], v[180:181] /*v[436:437]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1040
	v_exp_f32_e32 v240 /*v496*/, v129
	v_nop
	s_set_vgpr_msb 0x4010
	v_pk_fma_f32 v[128:129], v[236:237], s[56:57], v[180:181] /*v[436:437]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1040
	v_exp_f32_e32 v204 /*v460*/, v130
	v_exp_f32_e32 v214 /*v470*/, v131
	v_nop
	s_set_vgpr_msb 0x4011
	v_pk_fma_f32 v[130:131], v[4:5] /*v[260:261]*/, s[56:57], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1110
	v_exp_f32_e32 v220, v137
	v_exp_f32_e32 v209, v132
	v_exp_f32_e32 v221, v133
	v_exp_f32_e32 v137, v128
	v_pk_fma_f32 v[132:133], v[240:241], s[56:57], v[180:181] /*v[436:437]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1040
	v_exp_f32_e32 v195 /*v451*/, v129
	v_nop
	s_set_vgpr_msb 0x4011
	v_pk_fma_f32 v[128:129], v[6:7] /*v[262:263]*/, s[56:57], v[180:181] /*v[436:437]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1140
	v_exp_f32_e32 v242 /*v498*/, v130
	v_exp_f32_e32 v248 /*v504*/, v131
	v_nop
	s_set_vgpr_msb 0x4010
	v_pk_fma_f32 v[130:131], v[244:245], s[56:57], v[180:181] /*v[436:437]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1040
	v_exp_f32_e32 v194 /*v450*/, v139
	s_set_vgpr_msb 0x4010
	v_pk_fma_f32 v[142:143], v[142:143], s[56:57], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v139, v132
	s_set_vgpr_msb 0x1040
	v_exp_f32_e32 v229 /*v485*/, v133
	v_exp_f32_e32 v237 /*v493*/, v128
	s_set_vgpr_msb 0x4010
	v_pk_fma_f32 v[132:133], v[242:243], s[56:57], v[180:181] /*v[436:437]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1040
	v_exp_f32_e32 v245 /*v501*/, v129
	v_nop
	s_set_vgpr_msb 0x4010
	v_pk_fma_f32 v[128:129], v[252:253], s[56:57], v[180:181] /*v[436:437]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1040
	v_exp_f32_e32 v203 /*v459*/, v130
	v_exp_f32_e32 v213 /*v469*/, v131
	v_nop
	s_set_vgpr_msb 0x4010
	v_pk_fma_f32 v[130:131], v[246:247], s[56:57], v[180:181] /*v[436:437]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v232, v143
	s_set_vgpr_msb 0x1040
	v_exp_f32_e32 v253 /*v509*/, v132
	v_exp_f32_e32 v255 /*v511*/, v133
	s_set_vgpr_msb 0x4000
	v_exp_f32_e32 v143, v128
	s_set_vgpr_msb 17
	v_pk_fma_f32 v[132:133], v[16:17] /*v[272:273]*/, s[56:57], v[180:181] /*v[436:437]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1110
	v_exp_f32_e32 v233, v129
	v_nop
	v_pk_fma_f32 v[128:129], v[254:255], s[56:57], v[180:181] /*v[436:437]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v141, v130
	s_set_vgpr_msb 0x1040
	v_exp_f32_e32 v251 /*v507*/, v131
	v_nop
	s_set_vgpr_msb 0x4010
	v_pk_fma_f32 v[130:131], v[248:249], s[56:57], v[180:181] /*v[436:437]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v154, v149
	v_exp_f32_e32 v149, v132
	v_exp_f32_e32 v155, v133
	v_exp_f32_e32 v157, v128
	s_set_vgpr_msb 0x1011
	v_pk_fma_f32 v[132:133], v[8:9] /*v[264:265]*/, s[56:57], v[180:181] /*v[436:437]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1100
	v_exp_f32_e32 v195, v129
	v_nop
	s_set_vgpr_msb 17
	v_pk_fma_f32 v[128:129], v[0:1] /*v[256:257]*/, s[56:57], v[180:181] /*v[436:437]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1140
	v_exp_f32_e32 v252 /*v508*/, v146
	v_exp_f32_e32 v254 /*v510*/, v147
	s_set_vgpr_msb 0x4010
	v_exp_f32_e32 v146, v145
	v_exp_f32_e32 v145, v130
	v_exp_f32_e32 v147, v131
	v_nop
	v_pk_fma_f32 v[130:131], v[250:251], s[56:57], v[180:181] /*v[436:437]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v217, v132
	v_exp_f32_e32 v227, v133
	s_set_vgpr_msb 0x1040
	v_exp_f32_e32 v207 /*v463*/, v128
	s_set_vgpr_msb 0x4011
	v_pk_fma_f32 v[132:133], v[10:11] /*v[266:267]*/, s[56:57], v[180:181] /*v[436:437]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1140
	v_exp_f32_e32 v217 /*v473*/, v129
	v_nop
	s_set_vgpr_msb 0x4011
	v_pk_fma_f32 v[128:129], v[2:3] /*v[258:259]*/, s[56:57], v[180:181] /*v[436:437]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1100
	v_exp_f32_e32 v201, v130
	v_exp_f32_e32 v211, v131
	v_nop
	s_set_vgpr_msb 17
	v_pk_fma_f32 v[130:131], v[26:27] /*v[282:283]*/, s[56:57], v[180:181] /*v[436:437]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1100
	v_exp_f32_e32 v194, v151
	s_set_vgpr_msb 64
	v_exp_f32_e32 v239 /*v495*/, v132
	v_exp_f32_e32 v247 /*v503*/, v133
	s_set_vgpr_msb 0x4000
	v_exp_f32_e32 v151, v128
	s_set_vgpr_msb 17
	v_pk_fma_f32 v[132:133], v[12:13] /*v[268:269]*/, s[56:57], v[180:181] /*v[436:437]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1100
	v_exp_f32_e32 v159, v129
	v_nop
	s_set_vgpr_msb 17
	v_pk_fma_f32 v[128:129], v[40:41] /*v[296:297]*/, s[56:57], v[180:181] /*v[436:437]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1140
	v_exp_f32_e32 v225 /*v481*/, v130
	v_exp_f32_e32 v235 /*v491*/, v131
	v_nop
	s_set_vgpr_msb 0x4011
	v_pk_fma_f32 v[130:131], v[18:19] /*v[274:275]*/, s[56:57], v[180:181] /*v[436:437]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1100
	v_exp_f32_e32 v203, v132
	v_exp_f32_e32 v213, v133
	v_exp_f32_e32 v215, v128
	s_set_vgpr_msb 17
	v_pk_fma_f32 v[132:133], v[14:15] /*v[270:271]*/, s[56:57], v[180:181] /*v[436:437]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1100
	v_exp_f32_e32 v225, v129
	v_nop
	s_set_vgpr_msb 17
	v_pk_fma_f32 v[128:129], v[28:29] /*v[284:285]*/, s[56:57], v[180:181] /*v[436:437]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1100
	v_exp_f32_e32 v193, v130
	v_exp_f32_e32 v199, v131
	v_nop
	s_set_vgpr_msb 17
	v_pk_fma_f32 v[130:131], v[20:21] /*v[276:277]*/, s[56:57], v[180:181] /*v[436:437]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1140
	v_exp_f32_e32 v199 /*v455*/, v132
	v_exp_f32_e32 v209 /*v465*/, v133
	v_exp_f32_e32 v211 /*v467*/, v128
	s_set_vgpr_msb 0x4011
	v_pk_fma_f32 v[132:133], v[42:43] /*v[298:299]*/, s[56:57], v[180:181] /*v[436:437]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1140
	v_exp_f32_e32 v221 /*v477*/, v129
	v_nop
	s_set_vgpr_msb 0x4011
	v_pk_fma_f32 v[128:129], v[30:31] /*v[286:287]*/, s[56:57], v[180:181] /*v[436:437]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1100
	v_exp_f32_e32 v229, v130
	s_set_vgpr_msb 64
	v_exp_f32_e32 v197 /*v453*/, v131
	v_nop
	s_set_vgpr_msb 0x4011
	v_pk_fma_f32 v[130:131], v[22:23] /*v[278:279]*/, s[56:57], v[180:181] /*v[436:437]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1100
	v_exp_f32_e32 v197, v132
	v_exp_f32_e32 v205, v133
	v_exp_f32_e32 v207, v128
	v_exp_f32_e32 v219, v129
	v_nop
	s_set_vgpr_msb 17
	v_pk_fma_f32 v[128:129], v[32:33] /*v[288:289]*/, s[56:57], v[180:181] /*v[436:437]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[132:133], v[34:35] /*v[290:291]*/, s[56:57], v[180:181] /*v[436:437]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1110
	v_pk_fma_f32 v[152:153], v[162:163], s[56:57], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1040
	v_exp_f32_e32 v223 /*v479*/, v130
	v_exp_f32_e32 v233 /*v489*/, v131
	v_nop
	s_set_vgpr_msb 0x4011
	v_pk_fma_f32 v[130:131], v[24:25] /*v[280:281]*/, s[56:57], v[180:181] /*v[436:437]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1140
	v_exp_f32_e32 v193 /*v449*/, v128
	v_exp_f32_e32 v201 /*v457*/, v129
	v_exp_f32_e32 v205 /*v461*/, v132
	v_exp_f32_e32 v215 /*v471*/, v133
	s_set_vgpr_msb 0x4011
	v_pk_fma_f32 v[128:129], v[36:37] /*v[292:293]*/, s[56:57], v[180:181] /*v[436:437]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1100
	v_pk_add_f32 v[132:133], v[208:209], v[220:221]
	s_set_vgpr_msb 5
	v_pk_add_f32 v[134:135], v[194:195] /*v[450:451]*/, v[202:203] /*v[458:459]*/
	s_set_vgpr_msb 0x500
	v_exp_f32_e32 v142, v142
	v_exp_f32_e32 v144, v144
	v_exp_f32_e32 v148, v148
	v_exp_f32_e32 v200, v152
	v_exp_f32_e32 v223, v130
	v_exp_f32_e32 v231, v131
	v_nop
	s_set_vgpr_msb 17
	v_pk_fma_f32 v[130:131], v[44:45] /*v[300:301]*/, s[56:57], v[180:181] /*v[436:437]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1100
	v_exp_f32_e32 v210, v153
	s_set_vgpr_msb 64
	v_exp_f32_e32 v231 /*v487*/, v128
	v_exp_f32_e32 v241 /*v497*/, v129
	v_nop
	s_set_vgpr_msb 0x4000
	v_pk_add_f32 v[128:129], v[136:137], v[132:133]
	s_set_vgpr_msb 1
	v_pk_add_f32 v[132:133], v[212:213] /*v[468:469]*/, v[134:135]
	v_pk_add_f32 v[134:135], v[228:229] /*v[484:485]*/, v[138:139]
	v_pk_add_f32 v[152:153], v[244:245] /*v[500:501]*/, v[140:141]
	s_set_vgpr_msb 0x105
	v_pk_add_f32 v[160:161], v[252:253] /*v[508:509]*/, v[254:255] /*v[510:511]*/
	v_pk_add_f32 v[170:171], v[216:217] /*v[472:473]*/, v[224:225] /*v[480:481]*/
	v_pk_add_f32 v[172:173], v[238:239] /*v[494:495]*/, v[246:247] /*v[502:503]*/
	s_set_vgpr_msb 0x500
	v_pk_add_f32 v[176:177], v[202:203], v[212:213]
	v_pk_add_f32 v[178:179], v[224:225], v[228:229]
	s_set_vgpr_msb 64
	v_exp_f32_e32 v219 /*v475*/, v130
	s_set_vgpr_msb 0x4000
	v_pk_add_f32 v[162:163], v[232:233], v[144:145]
	v_pk_add_f32 v[164:165], v[148:149], v[154:155]
	s_set_vgpr_msb 1
	v_pk_add_f32 v[134:135], v[236:237] /*v[492:493]*/, v[134:135]
	v_pk_add_f32 v[152:153], v[250:251] /*v[506:507]*/, v[152:153]
	s_set_vgpr_msb 0x100
	v_pk_add_f32 v[160:161], v[142:143], v[160:161]
	v_pk_add_f32 v[166:167], v[194:195], v[200:201]
	v_pk_add_f32 v[174:175], v[158:159], v[192:193]
	s_set_vgpr_msb 1
	v_pk_add_f32 v[170:171], v[234:235] /*v[490:491]*/, v[170:171]
	s_set_vgpr_msb 0x100
	v_pk_add_f32 v[172:173], v[150:151], v[172:173]
	s_set_vgpr_msb 5
	v_pk_add_f32 v[180:181], v[198:199] /*v[454:455]*/, v[208:209] /*v[464:465]*/
	v_pk_add_f32 v[182:183], v[220:221] /*v[476:477]*/, v[222:223] /*v[478:479]*/
	s_set_vgpr_msb 0x500
	v_pk_add_f32 v[184:185], v[196:197], v[204:205]
	v_pk_add_f32 v[176:177], v[214:215], v[176:177]
	s_set_vgpr_msb 1
	v_pk_add_f32 v[178:179], v[196:197] /*v[452:453]*/, v[178:179]
	s_set_vgpr_msb 0x100
	v_pk_add_f32 v[128:129], v[128:129], v[132:133]
	s_set_vgpr_msb 64
	v_exp_f32_e32 v227 /*v483*/, v131
	v_nop
	s_set_vgpr_msb 0x4011
	v_pk_fma_f32 v[130:131], v[38:39] /*v[294:295]*/, s[56:57], v[180:181] /*v[436:437]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1100
	v_pk_add_f32 v[162:163], v[146:147], v[162:163]
	v_pk_add_f32 v[164:165], v[156:157], v[164:165]
	v_pk_add_f32 v[168:169], v[216:217], v[226:227]
	v_pk_add_f32 v[166:167], v[210:211], v[166:167]
	v_pk_add_f32 v[174:175], v[198:199], v[174:175]
	s_set_vgpr_msb 1
	v_pk_add_f32 v[180:181], v[210:211] /*v[466:467]*/, v[180:181]
	v_pk_add_f32 v[182:183], v[232:233] /*v[488:489]*/, v[182:183]
	s_set_vgpr_msb 0x100
	v_pk_add_f32 v[184:185], v[206:207], v[184:185]
	v_pk_add_f32 v[186:187], v[218:219], v[222:223]
	s_set_vgpr_msb 5
	v_pk_add_f32 v[188:189], v[192:193] /*v[448:449]*/, v[200:201] /*v[456:457]*/
	v_pk_add_f32 v[190:191], v[214:215] /*v[470:471]*/, v[218:219] /*v[474:475]*/
	s_set_vgpr_msb 0x500
	v_pk_add_f32 v[128:129], v[134:135], v[128:129]
	v_pk_add_f32 v[134:135], v[152:153], v[160:161]
	v_pk_add_f32 v[152:153], v[170:171], v[172:173]
	v_pk_add_f32 v[160:161], v[176:177], v[178:179]
	s_set_vgpr_msb 64
	v_exp_f32_e32 v243 /*v499*/, v130
	s_set_vgpr_msb 0x4001
	v_pk_add_f32 v[168:169], v[206:207] /*v[462:463]*/, v[168:169]
	s_set_vgpr_msb 0x105
	v_pk_add_f32 v[234:235], v[230:231] /*v[486:487]*/, v[240:241] /*v[496:497]*/
	s_set_vgpr_msb 0x500
	v_pk_add_f32 v[132:133], v[230:231], v[186:187]
	s_set_vgpr_msb 1
	v_pk_add_f32 v[186:187], v[204:205] /*v[460:461]*/, v[188:189]
	v_pk_add_f32 v[188:189], v[226:227] /*v[482:483]*/, v[190:191]
	s_set_vgpr_msb 0x100
	v_pk_add_f32 v[164:165], v[164:165], v[166:167]
	v_pk_add_f32 v[166:167], v[182:183], v[184:185]
	v_pk_add_f32 v[134:135], v[162:163], v[134:135]
	v_pk_add_f32 v[152:153], v[174:175], v[152:153]
	v_pk_add_f32 v[160:161], v[180:181], v[160:161]
	s_set_vgpr_msb 1
	v_pk_add_f32 v[190:191], v[242:243] /*v[498:499]*/, v[234:235]
	s_set_vgpr_msb 0x100
	v_pk_add_f32 v[162:163], v[168:169], v[164:165]
	v_pk_add_f32 v[132:133], v[132:133], v[166:167]
	v_pk_add_f32 v[164:165], v[186:187], v[188:189]
	v_pk_add_f32 v[128:129], v[128:129], v[134:135]
	v_pk_add_f32 v[134:135], v[152:153], v[160:161]
	s_set_vgpr_msb 64
	v_exp_f32_e32 v249 /*v505*/, v131
	v_nop
	s_set_vgpr_msb 0x4000
	v_pk_add_f32 v[130:131], v[190:191], v[164:165]
	v_pk_add_f32 v[128:129], v[162:163], v[128:129]
	v_pk_add_f32 v[132:133], v[132:133], v[134:135]
	s_set_vgpr_msb 1
	v_pk_add_f32 v[130:131], v[248:249] /*v[504:505]*/, v[130:131]
	s_set_vgpr_msb 0x100
	v_pk_add_f32 v[128:129], v[128:129], v[132:133]
	s_set_vgpr_msb 5
	v_sub_f32_e32 v132, v176 /*v432*/, v55 /*v311*/
	s_set_vgpr_msb 0x500
	v_pk_add_f32 v[128:129], v[130:131], v[128:129]
	v_dual_mul_f32 v130, 0x3fb8aa3b, v132 :: v_dual_mov_b32 v131, v128
	v_mov_b32_e32 v133, v129
	v_exp_f32_e32 v130, v130
	v_permlanex16_b32 v131, v131, s82, 0xfedcba98
	v_permlanex16_b32 v133, v133, s82, 0xfedcba98
	s_cbranch_vccz .LBB0_20
	v_pk_mul_f32 v[126:127], v[126:127], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[124:125], v[124:125], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[122:123], v[122:123], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[120:121], v[120:121], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[118:119], v[118:119], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[116:117], v[116:117], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[114:115], v[114:115], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[112:113], v[112:113], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[110:111], v[110:111], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[108:109], v[108:109], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[106:107], v[106:107], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[104:105], v[104:105], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[102:103], v[102:103], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[100:101], v[100:101], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[98:99], v[98:99], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[96:97], v[96:97], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[94:95], v[94:95], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[92:93], v[92:93], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[90:91], v[90:91], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[88:89], v[88:89], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[86:87], v[86:87], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[84:85], v[84:85], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[82:83], v[82:83], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[80:81], v[80:81], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[78:79], v[78:79], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[76:77], v[76:77], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[74:75], v[74:75], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[72:73], v[72:73], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[70:71], v[70:71], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[68:69], v[68:69], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[66:67], v[66:67], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[64:65], v[64:65], v[130:131] op_sel_hi:[1,0]
.LBB0_20:
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x5
	scratch_load_b32 v132, off, off offset:8 th:TH_LOAD_LU nv
	scratch_load_b64 v[234:235], off, off nv
	scratch_load_b128 v[8:11], off, off offset:40 nv
	scratch_load_b128 v[12:15], off, off offset:56 nv
	scratch_load_b128 v[16:19], off, off offset:72 nv
	scratch_load_b128 v[20:23], off, off offset:88 nv
	s_set_vgpr_msb 1
	v_mov_b32_e32 v236, v177 /*v433*/
	s_and_b32 s2, s5, exec_lo
	s_cselect_b32 s2, 1, 0
	s_cmp_lg_u32 s2, 1
	s_set_vgpr_msb 0x100
	s_wait_loadcnt 0x5
	v_sub_f32_e32 v132, v132, v236
	v_mul_f32_e32 v132, 0x3fb8aa3b, v132
	v_exp_f32_e32 v132, v132
	s_cbranch_scc1 .LBB0_15
	v_nop
	v_pk_mul_f32 v[62:63], v[62:63], v[132:133] op_sel_hi:[1,0]
	v_pk_mul_f32 v[60:61], v[60:61], v[132:133] op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59], v[58:59], v[132:133] op_sel_hi:[1,0]
	v_pk_mul_f32 v[56:57], v[56:57], v[132:133] op_sel_hi:[1,0]
	v_pk_mul_f32 v[54:55], v[54:55], v[132:133] op_sel_hi:[1,0]
	v_pk_mul_f32 v[52:53], v[52:53], v[132:133] op_sel_hi:[1,0]
	v_pk_mul_f32 v[50:51], v[50:51], v[132:133] op_sel_hi:[1,0]
	v_pk_mul_f32 v[48:49], v[48:49], v[132:133] op_sel_hi:[1,0]
	v_pk_mul_f32 v[46:47], v[46:47], v[132:133] op_sel_hi:[1,0]
	v_pk_mul_f32 v[44:45], v[44:45], v[132:133] op_sel_hi:[1,0]
	v_pk_mul_f32 v[42:43], v[42:43], v[132:133] op_sel_hi:[1,0]
	v_pk_mul_f32 v[40:41], v[40:41], v[132:133] op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39], v[38:39], v[132:133] op_sel_hi:[1,0]
	v_pk_mul_f32 v[36:37], v[36:37], v[132:133] op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35], v[34:35], v[132:133] op_sel_hi:[1,0]
	v_pk_mul_f32 v[32:33], v[32:33], v[132:133] op_sel_hi:[1,0]
	v_pk_mul_f32 v[30:31], v[30:31], v[132:133] op_sel_hi:[1,0]
	v_pk_mul_f32 v[28:29], v[28:29], v[132:133] op_sel_hi:[1,0]
	v_pk_mul_f32 v[26:27], v[26:27], v[132:133] op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25], v[24:25], v[132:133] op_sel_hi:[1,0]
	s_wait_loadcnt 0x0
	v_pk_mul_f32 v[22:23], v[22:23], v[132:133] op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21], v[20:21], v[132:133] op_sel_hi:[1,0]
	v_pk_mul_f32 v[18:19], v[18:19], v[132:133] op_sel_hi:[1,0]
	v_pk_mul_f32 v[16:17], v[16:17], v[132:133] op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[14:15], v[132:133] op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13], v[12:13], v[132:133] op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[10:11], v[132:133] op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[8:9], v[132:133] op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[6:7], v[132:133] op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[4:5], v[132:133] op_sel_hi:[1,0]
	v_pk_mul_f32 v[2:3], v[2:3], v[132:133] op_sel_hi:[1,0]
	v_pk_mul_f32 v[0:1], v[0:1], v[132:133] op_sel_hi:[1,0]
	s_branch .LBB0_15
.LBB0_22:
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x8
	scratch_load_b32 v154, off, off offset:916 nv
	scratch_load_b32 v155, off, off offset:920 nv
	scratch_load_b32 v156, off, off offset:928 nv
	scratch_load_b32 v157, off, off offset:936 nv
	scratch_load_b32 v134, off, off offset:940 nv
	scratch_load_b32 v152, off, off offset:104 nv
	scratch_load_b32 v128, off, off offset:944 nv
	scratch_load_b32 v129, off, off offset:948 nv
	scratch_load_b32 v130, off, off offset:952 nv
.LBB0_23:
	s_branch .LBB0_45
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
	s_and_b32 s2, s91, 7
	s_and_b32 s18, s7, 0xffff
	s_lshl_b32 s53, s2, 4
	s_lshl_b32 s38, s2, 5
	s_mul_i32 s54, s2, 0x1100
	s_mul_i32 s55, s2, 0x1200
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
	s_mov_b32 s16, 16
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
	s_wait_loadcnt 0x3
	v_and_or_b32 v64, v157, 7, v152
	s_add_co_i32 s2, s11, s2
	s_add_nc_u64 s[40:41], s[68:69], s[72:73]
	s_max_i32 s2, s2, 0
	s_add_nc_u64 s[42:43], s[66:67], s[70:71]
	s_add_co_i32 s2, s2, 1
	v_mul_u32_u24_e32 v64, 0x120, v64
	s_mov_b32 s45, s19
	s_wait_loadcnt 0x1
	v_and_or_b32 v64, v129, 16, v64
	s_set_vgpr_msb 64
	v_or_b32_e32 v176 /*v432*/, 0x10000, v64
	v_or_b32_e32 v177 /*v433*/, 0x30000, v64
	s_wait_tensorcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_alu depctr_va_vdst(8) depctr_vm_vsrc(6)
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
	tensor_load_to_lds s[4:7], s[12:19]
	s_add_nc_u64 s[6:7], s[38:39], s[74:75]
	s_mov_b32 s5, s55
	s_bitset1_b32 s7, 31
	s_wait_dscnt 0xe
	v_pk_mul_bf16 v7, v128, v7
	tensor_load_to_lds s[4:7], s[20:27]
	v_pk_mul_bf16 v6, v128, v6
	v_pk_mul_bf16 v5, v128, v5
	v_pk_mul_bf16 v4, v128, v4
	v_pk_mul_bf16 v3, v128, v3
	v_pk_mul_bf16 v2, v128, v2
	v_pk_mul_bf16 v1, v128, v1
	v_pk_mul_bf16 v0, v128, v0
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_store_b128 off, v[0:3], off offset:8 nv
	scratch_store_b128 off, v[4:7], off offset:24 nv
	s_wait_dscnt 0xc
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	v_pk_mul_bf16 v7, v128, v15
	v_pk_mul_bf16 v6, v128, v14
	v_pk_mul_bf16 v5, v128, v13
	v_pk_mul_bf16 v4, v128, v12
	v_pk_mul_bf16 v3, v128, v11
	v_pk_mul_bf16 v2, v128, v10
	v_pk_mul_bf16 v1, v128, v9
	v_pk_mul_bf16 v0, v128, v8
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_store_b128 off, v[0:3], off offset:40 nv
	scratch_store_b128 off, v[4:7], off offset:56 nv
	s_wait_dscnt 0xa
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	v_pk_mul_bf16 v7, v128, v23
	v_pk_mul_bf16 v6, v128, v22
	v_pk_mul_bf16 v5, v128, v21
	v_pk_mul_bf16 v4, v128, v20
	v_pk_mul_bf16 v3, v128, v19
	v_pk_mul_bf16 v2, v128, v18
	v_pk_mul_bf16 v1, v128, v17
	v_pk_mul_bf16 v0, v128, v16
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_store_b128 off, v[0:3], off offset:72 nv
	scratch_store_b128 off, v[4:7], off offset:88 nv
	s_wait_dscnt 0x8
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	v_pk_mul_bf16 v7, v128, v31
	v_pk_mul_bf16 v6, v128, v30
	v_pk_mul_bf16 v5, v128, v29
	v_pk_mul_bf16 v4, v128, v28
	v_pk_mul_bf16 v3, v128, v27
	v_pk_mul_bf16 v2, v128, v26
	v_pk_mul_bf16 v1, v128, v25
	v_pk_mul_bf16 v0, v128, v24
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_store_b128 off, v[0:3], off offset:108 nv
	scratch_store_b128 off, v[4:7], off offset:124 nv
	s_wait_dscnt 0x6
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	v_pk_mul_bf16 v7, v128, v39
	v_pk_mul_bf16 v6, v128, v38
	v_pk_mul_bf16 v5, v128, v37
	v_pk_mul_bf16 v4, v128, v36
	v_pk_mul_bf16 v3, v128, v35
	v_pk_mul_bf16 v2, v128, v34
	v_pk_mul_bf16 v1, v128, v33
	v_pk_mul_bf16 v0, v128, v32
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_store_b128 off, v[0:3], off offset:140 nv
	scratch_store_b128 off, v[4:7], off offset:156 nv
	s_wait_dscnt 0x4
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	v_pk_mul_bf16 v7, v128, v47
	v_pk_mul_bf16 v6, v128, v46
	v_pk_mul_bf16 v5, v128, v45
	v_pk_mul_bf16 v4, v128, v44
	v_pk_mul_bf16 v3, v128, v43
	v_pk_mul_bf16 v2, v128, v42
	v_pk_mul_bf16 v1, v128, v41
	v_pk_mul_bf16 v0, v128, v40
	s_ashr_i32 s5, s2, 31
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_store_b128 off, v[0:3], off offset:172 nv
	scratch_store_b128 off, v[4:7], off offset:188 nv
	s_lshr_b32 s5, s5, 25
	s_wait_dscnt 0x2
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	v_pk_mul_bf16 v7, v128, v55
	s_add_co_i32 s5, s2, s5
	v_pk_mul_bf16 v6, v128, v54
	v_pk_mul_bf16 v5, v128, v53
	v_pk_mul_bf16 v4, v128, v52
	v_pk_mul_bf16 v3, v128, v51
	v_pk_mul_bf16 v2, v128, v50
	v_pk_mul_bf16 v1, v128, v49
	v_pk_mul_bf16 v0, v128, v48
	s_and_b32 s7, s5, 0xffffff80
	s_add_co_i32 s6, s10, -1
	s_ashr_i32 s5, s5, 7
	s_cmp_lg_u32 s2, s7
	s_cselect_b32 s7, -1, 0
	s_cmp_lt_i32 s2, 0
	s_cselect_b32 s2, -1, 0
	s_and_b32 s2, s2, s7
	s_sub_co_ci_u32 s2, s5, 0
	s_min_i32 s2, s2, s6
	s_max_i32 s44, s2, 0
	s_cmp_lt_i32 s2, 1
	s_wait_tensorcnt 0x0
	s_wait_storecnt_dscnt 0x0
	s_barrier_signal -1
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_store_b128 off, v[0:3], off offset:204 nv
	scratch_store_b128 off, v[4:7], off offset:220 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	v_pk_mul_bf16 v7, v128, v63
	v_pk_mul_bf16 v6, v128, v62
	v_pk_mul_bf16 v5, v128, v61
	v_pk_mul_bf16 v4, v128, v60
	v_pk_mul_bf16 v3, v128, v59
	v_pk_mul_bf16 v2, v128, v58
	v_pk_mul_bf16 v1, v128, v57
	v_pk_mul_bf16 v0, v128, v56
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_store_b128 off, v[0:3], off offset:236 nv
	scratch_store_b128 off, v[4:7], off offset:252 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	v_mov_b32_e32 v0, 0
	s_barrier_wait -1
	s_wait_storecnt 0x0
	s_cbranch_scc1 .LBB0_33
	v_dual_mov_b32 v1, v0 :: v_dual_mov_b32 v2, v0
	v_dual_mov_b32 v3, v0 :: v_dual_mov_b32 v4, v0
	v_dual_mov_b32 v5, v0 :: v_dual_mov_b32 v6, v0
	v_dual_mov_b32 v7, v0 :: v_dual_mov_b32 v234, v0
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
	v_dual_mov_b32 v181 /*v437*/, 0xf149f2ca :: v_dual_mov_b32 v178 /*v434*/, v134
	s_set_vgpr_msb 0x4001
	v_dual_mov_b32 v128, v177 /*v433*/ :: v_dual_mov_b32 v129, v183 /*v439*/
	s_set_vgpr_msb 0x140
	v_mov_b32_e32 v182 /*v438*/, 0xf149f2ca
	s_set_vgpr_msb 0x4000
	v_mov_b32_e32 v235, v0
	s_lshl_b64 s[46:47], s[44:45], 7
	s_add_co_i32 s35, s9, 0xffffff80
	s_add_co_i32 s48, s64, 0x80
	s_mov_b64 s[50:51], 0xffffffffffffff80
	s_mov_b32 s56, 0x76543210
	s_mov_b32 s52, 0x3fb8aa3b
	s_mov_b32 s65, 1
	s_branch .LBB0_27
.LBB0_26:
	s_set_vgpr_msb 64
	v_cvt_pk_bf16_f32 v39 /*v295*/, v247, v253
	v_cvt_pk_bf16_f32 v24 /*v280*/, v154, v156
	v_cvt_pk_bf16_f32 v32 /*v288*/, v155, v157
	v_cvt_pk_bf16_f32 v47 /*v303*/, v248, v254
	v_cvt_pk_bf16_f32 v41 /*v297*/, v132, v136
	v_cvt_pk_bf16_f32 v40 /*v296*/, v128, v130
	s_set_vgpr_msb 0x4000
	v_cvt_pk_bf16_f32 v156, v150, v162
	v_cvt_pk_bf16_f32 v155, v142, v146
	v_cvt_pk_bf16_f32 v254, v151, v163
	v_cvt_pk_bf16_f32 v253, v143, v147
	v_cvt_pk_bf16_f32 v128, v144, v148
	v_cvt_pk_bf16_f32 v136, v145, v149
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b128 v[144:147], off, off offset:684 th:TH_LOAD_LU nv
	scratch_load_b128 v[148:151], off, off offset:700 th:TH_LOAD_LU nv
	s_set_vgpr_msb 64
	v_cvt_pk_bf16_f32 v31 /*v287*/, v246, v252
	v_cvt_pk_bf16_f32 v30 /*v286*/, v232, v240
	v_cvt_pk_bf16_f32 v38 /*v294*/, v233, v241
	v_cvt_pk_bf16_f32 v29 /*v285*/, v222, v228
	v_cvt_pk_bf16_f32 v37 /*v293*/, v223, v229
	v_cvt_pk_bf16_f32 v28 /*v284*/, v204, v212
	v_cvt_pk_bf16_f32 v36 /*v292*/, v205, v213
	v_cvt_pk_bf16_f32 v27 /*v283*/, v180, v188
	v_cvt_pk_bf16_f32 v35 /*v291*/, v181, v189
	v_cvt_pk_bf16_f32 v26 /*v282*/, v164, v172
	v_cvt_pk_bf16_f32 v34 /*v290*/, v165, v173
	v_cvt_pk_bf16_f32 v25 /*v281*/, v192, v198
	v_cvt_pk_bf16_f32 v33 /*v289*/, v193, v199
	s_set_vgpr_msb 0x4005
	v_wmma_f32_16x16x32_bf16 v[88:95], v[56:63] /*v[312:319]*/, v[24:31] /*v[280:287]*/, v[88:95]
	s_set_vgpr_msb 0x540
	v_cvt_pk_bf16_f32 v46 /*v302*/, v234, v242
	s_set_vgpr_msb 0x4000
	v_cvt_pk_bf16_f32 v193, v249, v255
	v_cvt_pk_bf16_f32 v192, v235, v243
	s_set_vgpr_msb 64
	v_cvt_pk_bf16_f32 v45 /*v301*/, v190, v210
	v_cvt_pk_bf16_f32 v44 /*v300*/, v174, v186
	v_cvt_pk_bf16_f32 v43 /*v299*/, v194, v160
	v_cvt_pk_bf16_f32 v42 /*v298*/, v140, v158
	s_set_vgpr_msb 0x4005
	v_wmma_f32_16x16x32_bf16 v[24:31], v[56:63] /*v[312:319]*/, v[32:39] /*v[288:295]*/, v[24:31]
	s_set_vgpr_msb 0x500
	v_cvt_pk_bf16_f32 v191, v191, v211
	v_cvt_pk_bf16_f32 v190, v175, v187
	v_cvt_pk_bf16_f32 v189, v195, v161
	v_cvt_pk_bf16_f32 v188, v141, v159
	v_cvt_pk_bf16_f32 v187, v133, v137
	v_cvt_pk_bf16_f32 v186, v129, v131
	v_cvt_pk_bf16_f32 v161, v224, v236
	s_set_vgpr_msb 5
	v_wmma_f32_16x16x32_bf16 v[120:127], v[184:191] /*v[440:447]*/, v[24:31] /*v[280:287]*/, v[120:127]
	s_set_vgpr_msb 0x500
	v_cvt_pk_bf16_f32 v160, v218, v152
	v_cvt_pk_bf16_f32 v159, v206, v214
	v_cvt_pk_bf16_f32 v158, v182, v200
	v_cvt_pk_bf16_f32 v157, v166, v176
	v_cvt_pk_bf16_f32 v154, v134, v138
	v_cvt_pk_bf16_f32 v255, v167, v177
	v_cvt_pk_bf16_f32 v252, v135, v139
	s_set_vgpr_msb 5
	v_wmma_f32_16x16x32_bf16 v[56:63], v[184:191] /*v[440:447]*/, v[32:39] /*v[288:295]*/, v[56:63]
	s_set_vgpr_msb 0x500
	v_cvt_pk_bf16_f32 v135, v244, v250
	v_cvt_pk_bf16_f32 v134, v230, v238
	v_cvt_pk_bf16_f32 v133, v220, v226
	v_cvt_pk_bf16_f32 v132, v208, v216
	v_cvt_pk_bf16_f32 v131, v184, v202
	v_cvt_pk_bf16_f32 v130, v170, v178
	v_cvt_pk_bf16_f32 v129, v196, v168
	s_set_vgpr_msb 5
	v_wmma_f32_16x16x32_bf16 v[88:95], v[48:55] /*v[304:311]*/, v[40:47] /*v[296:303]*/, v[88:95]
	s_set_vgpr_msb 0x500
	v_cvt_pk_bf16_f32 v143, v245, v251
	v_cvt_pk_bf16_f32 v142, v231, v239
	v_cvt_pk_bf16_f32 v141, v221, v227
	v_cvt_pk_bf16_f32 v140, v209, v217
	v_cvt_pk_bf16_f32 v139, v185, v203
	v_cvt_pk_bf16_f32 v138, v171, v179
	v_cvt_pk_bf16_f32 v137, v197, v169
	s_set_vgpr_msb 1
	v_wmma_f32_16x16x32_bf16 v[24:31], v[48:55] /*v[304:311]*/, v[186:193], v[24:31]
	s_set_vgpr_msb 0x141
	v_dual_mov_b32 v181 /*v437*/, v179 /*v435*/ :: v_dual_mov_b32 v182 /*v438*/, v180 /*v436*/
	s_add_nc_u64 s[46:47], s[46:47], s[50:51]
	s_addk_co_i32 s35, 0xff80
	s_addk_co_i32 s48, 0x80
	s_add_co_i32 s65, s65, 1
	s_cmp_lg_u64 s[46:47], 0
	s_set_vgpr_msb 0x4105
	v_wmma_f32_16x16x32_bf16 v[120:127], v[0:7] /*v[256:263]*/, v[40:47] /*v[296:303]*/, v[120:127]
	s_set_vgpr_msb 0x501
	v_wmma_f32_16x16x32_bf16 v[56:63], v[0:7] /*v[256:263]*/, v[186:193], v[56:63]
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x140
	v_cvt_pk_bf16_f32 v3 /*v259*/, v225, v237
	v_cvt_pk_bf16_f32 v2 /*v258*/, v219, v153
	v_cvt_pk_bf16_f32 v1 /*v257*/, v207, v215
	v_cvt_pk_bf16_f32 v0 /*v256*/, v183, v201
	s_set_vgpr_msb 0x4005
	v_wmma_f32_16x16x32_bf16 v[112:119], v[152:159] /*v[408:415]*/, v[24:31] /*v[280:287]*/, v[112:119]
	v_wmma_f32_16x16x32_bf16 v[48:55], v[152:159] /*v[408:415]*/, v[32:39] /*v[288:295]*/, v[48:55]
	v_wmma_f32_16x16x32_bf16 v[104:111], v[120:127] /*v[376:383]*/, v[24:31] /*v[280:287]*/, v[104:111]
	v_wmma_f32_16x16x32_bf16 v[40:47], v[120:127] /*v[376:383]*/, v[32:39] /*v[288:295]*/, v[40:47]
	v_wmma_f32_16x16x32_bf16 v[96:103], v[88:95] /*v[344:351]*/, v[24:31] /*v[280:287]*/, v[96:103]
	v_wmma_f32_16x16x32_bf16 v[32:39], v[88:95] /*v[344:351]*/, v[32:39] /*v[288:295]*/, v[32:39]
	v_wmma_f32_16x16x32_bf16 v[112:119], v[144:151] /*v[400:407]*/, v[40:47] /*v[296:303]*/, v[112:119]
	s_set_vgpr_msb 0x501
	v_wmma_f32_16x16x32_bf16 v[48:55], v[144:151] /*v[400:407]*/, v[186:193], v[48:55]
	s_set_vgpr_msb 0x105
	v_wmma_f32_16x16x32_bf16 v[104:111], v[112:119] /*v[368:375]*/, v[40:47] /*v[296:303]*/, v[104:111]
	s_set_vgpr_msb 0x501
	v_wmma_f32_16x16x32_bf16 v[40:47], v[112:119] /*v[368:375]*/, v[186:193], v[40:47]
	s_set_vgpr_msb 0x105
	v_wmma_f32_16x16x32_bf16 v[96:103], v[80:87] /*v[336:343]*/, v[40:47] /*v[296:303]*/, v[96:103]
	s_set_vgpr_msb 0x501
	v_wmma_f32_16x16x32_bf16 v[32:39], v[80:87] /*v[336:343]*/, v[186:193], v[32:39]
	v_wmma_f32_16x16x32_bf16 v[120:127], v[168:175] /*v[424:431]*/, v[154:161], v[120:127]
	v_wmma_f32_16x16x32_bf16 v[56:63], v[168:175] /*v[424:431]*/, v[252:259], v[56:63]
	v_wmma_f32_16x16x32_bf16 v[112:119], v[136:143] /*v[392:399]*/, v[154:161], v[112:119]
	v_wmma_f32_16x16x32_bf16 v[48:55], v[136:143] /*v[392:399]*/, v[252:259], v[48:55]
	v_wmma_f32_16x16x32_bf16 v[104:111], v[104:111] /*v[360:367]*/, v[154:161], v[104:111]
	v_wmma_f32_16x16x32_bf16 v[40:47], v[104:111] /*v[360:367]*/, v[252:259], v[40:47]
	v_wmma_f32_16x16x32_bf16 v[96:103], v[8:15] /*v[264:271]*/, v[154:161], v[96:103]
	v_wmma_f32_16x16x32_bf16 v[32:39], v[8:15] /*v[264:271]*/, v[252:259], v[32:39]
	v_wmma_f32_16x16x32_bf16 v[120:127], v[160:167] /*v[416:423]*/, v[128:135], v[120:127]
	v_wmma_f32_16x16x32_bf16 v[56:63], v[160:167] /*v[416:423]*/, v[136:143], v[56:63]
	v_wmma_f32_16x16x32_bf16 v[112:119], v[128:135] /*v[384:391]*/, v[128:135], v[112:119]
	v_wmma_f32_16x16x32_bf16 v[48:55], v[128:135] /*v[384:391]*/, v[136:143], v[48:55]
	v_wmma_f32_16x16x32_bf16 v[104:111], v[96:103] /*v[352:359]*/, v[128:135], v[104:111]
	v_wmma_f32_16x16x32_bf16 v[40:47], v[96:103] /*v[352:359]*/, v[136:143], v[40:47]
	v_wmma_f32_16x16x32_bf16 v[96:103], v[64:71] /*v[320:327]*/, v[128:135], v[96:103]
	v_wmma_f32_16x16x32_bf16 v[32:39], v[64:71] /*v[320:327]*/, v[136:143], v[32:39]
	s_set_vgpr_msb 0x100
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[88:95], v[144:151], v[154:161], v[88:95]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[144:151], v[252:259], v[24:31]
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b128 v[144:147], off, off offset:652 th:TH_LOAD_LU nv
	scratch_load_b128 v[148:151], off, off offset:668 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[88:95], v[144:151], v[128:135], v[88:95]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[144:151], v[136:143], v[24:31]
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b128 v[144:147], off, off offset:620 th:TH_LOAD_LU nv
	scratch_load_b128 v[148:151], off, off offset:636 th:TH_LOAD_LU nv
	s_set_vgpr_msb 4
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[80:87], v[144:151], v[24:31] /*v[280:287]*/, v[80:87]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[144:151], v[32:39] /*v[288:295]*/, v[16:23]
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b128 v[144:147], off, off offset:588 th:TH_LOAD_LU nv
	scratch_load_b128 v[148:151], off, off offset:604 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[80:87], v[144:151], v[40:47] /*v[296:303]*/, v[80:87]
	s_set_vgpr_msb 0x400
	v_wmma_f32_16x16x32_bf16 v[16:23], v[144:151], v[186:193], v[16:23]
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b128 v[144:147], off, off offset:556 th:TH_LOAD_LU nv
	scratch_load_b128 v[148:151], off, off offset:572 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[80:87], v[144:151], v[154:161], v[80:87]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[144:151], v[252:259], v[16:23]
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b128 v[144:147], off, off offset:524 th:TH_LOAD_LU nv
	scratch_load_b128 v[148:151], off, off offset:540 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[80:87], v[144:151], v[128:135], v[80:87]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[144:151], v[136:143], v[16:23]
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b128 v[144:147], off, off offset:492 th:TH_LOAD_LU nv
	scratch_load_b128 v[148:151], off, off offset:508 th:TH_LOAD_LU nv
	s_set_vgpr_msb 4
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[72:79], v[144:151], v[24:31] /*v[280:287]*/, v[72:79]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[144:151], v[32:39] /*v[288:295]*/, v[8:15]
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b128 v[144:147], off, off offset:460 th:TH_LOAD_LU nv
	scratch_load_b128 v[148:151], off, off offset:476 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[72:79], v[144:151], v[40:47] /*v[296:303]*/, v[72:79]
	s_set_vgpr_msb 0x400
	v_wmma_f32_16x16x32_bf16 v[8:15], v[144:151], v[186:193], v[8:15]
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b128 v[144:147], off, off offset:428 th:TH_LOAD_LU nv
	scratch_load_b128 v[148:151], off, off offset:444 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[72:79], v[144:151], v[154:161], v[72:79]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[144:151], v[252:259], v[8:15]
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b128 v[144:147], off, off offset:396 th:TH_LOAD_LU nv
	scratch_load_b128 v[148:151], off, off offset:412 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[72:79], v[144:151], v[128:135], v[72:79]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[144:151], v[136:143], v[8:15]
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b128 v[144:147], off, off offset:364 th:TH_LOAD_LU nv
	scratch_load_b128 v[148:151], off, off offset:380 th:TH_LOAD_LU nv
	s_set_vgpr_msb 4
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[64:71], v[144:151], v[24:31] /*v[280:287]*/, v[64:71]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[144:151], v[32:39] /*v[288:295]*/, v[0:7]
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b128 v[144:147], off, off offset:332 th:TH_LOAD_LU nv
	scratch_load_b128 v[148:151], off, off offset:348 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[64:71], v[144:151], v[40:47] /*v[296:303]*/, v[64:71]
	s_set_vgpr_msb 0x400
	v_wmma_f32_16x16x32_bf16 v[0:7], v[144:151], v[186:193], v[0:7]
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b128 v[144:147], off, off offset:300 th:TH_LOAD_LU nv
	scratch_load_b128 v[148:151], off, off offset:316 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[64:71], v[144:151], v[154:161], v[64:71]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[144:151], v[252:259], v[0:7]
	s_wait_alu depctr_va_vdst(0)
	scratch_load_b64 v[144:145], off, off th:TH_LOAD_LU nv
	s_wait_loadcnt_dscnt 0x0
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 17
	v_pk_fma_f32 v[144:145], v[196:197] /*v[452:453]*/, v[144:145], v[194:195] /*v[450:451]*/
	v_pk_add_f32 v[234:235], v[192:193] /*v[448:449]*/, v[144:145]
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b128 v[144:147], off, off offset:268 th:TH_LOAD_LU nv
	scratch_load_b128 v[148:151], off, off offset:284 th:TH_LOAD_LU nv
	s_set_vgpr_msb 0x1100
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[64:71], v[144:151], v[128:135], v[64:71]
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 1
	v_dual_mov_b32 v128, v177 /*v433*/ :: v_dual_mov_b32 v129, v183 /*v439*/
	s_set_vgpr_msb 0x100
	v_wmma_f32_16x16x32_bf16 v[0:7], v[144:151], v[136:143], v[0:7]
	s_cbranch_scc0 .LBB0_35
.LBB0_27:
	s_set_vgpr_msb 0x41
	v_dual_mov_b32 v183 /*v439*/, v178 /*v434*/ :: v_dual_mov_b32 v177 /*v433*/, v176 /*v432*/
	s_set_vgpr_msb 0x4140
	v_dual_mov_b32 v178 /*v434*/, v129 :: v_dual_mov_b32 v176 /*v432*/, v128
	s_wait_alu depctr_va_vdst(0)
	scratch_store_b64 off, v[234:235], off nv
	s_wait_tensorcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	s_set_vgpr_msb 0x4041
	ds_load_b128 v[168:171] /*v[424:427]*/, v183 /*v439*/
	ds_load_b128 v[172:175] /*v[428:431]*/, v183 /*v439*/ offset:32
	ds_load_b128 v[160:163] /*v[416:419]*/, v183 /*v439*/ offset:64
	ds_load_b128 v[164:167] /*v[420:423]*/, v183 /*v439*/ offset:96
	ds_load_b128 v[152:155] /*v[408:411]*/, v183 /*v439*/ offset:128
	ds_load_b128 v[156:159] /*v[412:415]*/, v183 /*v439*/ offset:160
	s_set_vgpr_msb 0x4101
	ds_load_b128 v[128:131], v183 /*v439*/ offset:192
	ds_load_b128 v[132:135], v183 /*v439*/ offset:224
	s_set_vgpr_msb 0x141
	ds_load_b128 v[144:147] /*v[400:403]*/, v183 /*v439*/ offset:4352
	ds_load_b128 v[148:151] /*v[404:407]*/, v183 /*v439*/ offset:4384
	ds_load_b128 v[136:139] /*v[392:395]*/, v183 /*v439*/ offset:4416
	ds_load_b128 v[140:143] /*v[396:399]*/, v183 /*v439*/ offset:4448
	ds_load_b128 v[128:131] /*v[384:387]*/, v183 /*v439*/ offset:4480
	ds_load_b128 v[132:135] /*v[388:391]*/, v183 /*v439*/ offset:4512
	s_set_vgpr_msb 0x4101
	ds_load_b128 v[136:139], v183 /*v439*/ offset:4544
	ds_load_b128 v[140:143], v183 /*v439*/ offset:4576
	s_set_vgpr_msb 0x141
	ds_load_b128 v[120:123] /*v[376:379]*/, v183 /*v439*/ offset:8704
	ds_load_b128 v[124:127] /*v[380:383]*/, v183 /*v439*/ offset:8736
	ds_load_b128 v[112:115] /*v[368:371]*/, v183 /*v439*/ offset:8768
	ds_load_b128 v[116:119] /*v[372:375]*/, v183 /*v439*/ offset:8800
	ds_load_b128 v[104:107] /*v[360:363]*/, v183 /*v439*/ offset:8832
	ds_load_b128 v[108:111] /*v[364:367]*/, v183 /*v439*/ offset:8864
	s_set_vgpr_msb 0x4101
	ds_load_b128 v[144:147], v183 /*v439*/ offset:8896
	ds_load_b128 v[148:151], v183 /*v439*/ offset:8928
	s_set_vgpr_msb 0x141
	ds_load_b128 v[96:99] /*v[352:355]*/, v183 /*v439*/ offset:13056
	ds_load_b128 v[100:103] /*v[356:359]*/, v183 /*v439*/ offset:13088
	ds_load_b128 v[88:91] /*v[344:347]*/, v183 /*v439*/ offset:13120
	ds_load_b128 v[92:95] /*v[348:351]*/, v183 /*v439*/ offset:13152
	ds_load_b128 v[80:83] /*v[336:339]*/, v183 /*v439*/ offset:13184
	ds_load_b128 v[84:87] /*v[340:343]*/, v183 /*v439*/ offset:13216
	ds_load_b128 v[72:75] /*v[328:331]*/, v183 /*v439*/ offset:13248
	ds_load_b128 v[76:79] /*v[332:335]*/, v183 /*v439*/ offset:13280
	ds_load_b128 v[64:67] /*v[320:323]*/, v183 /*v439*/ offset:17408
	ds_load_b128 v[68:71] /*v[324:327]*/, v183 /*v439*/ offset:17440
	ds_load_b128 v[56:59] /*v[312:315]*/, v183 /*v439*/ offset:17472
	ds_load_b128 v[60:63] /*v[316:319]*/, v183 /*v439*/ offset:17504
	ds_load_b128 v[48:51] /*v[304:307]*/, v183 /*v439*/ offset:17536
	ds_load_b128 v[52:55] /*v[308:311]*/, v183 /*v439*/ offset:17568
	ds_load_b128 v[40:43] /*v[296:299]*/, v183 /*v439*/ offset:17600
	ds_load_b128 v[44:47] /*v[300:303]*/, v183 /*v439*/ offset:17632
	ds_load_b128 v[32:35] /*v[288:291]*/, v183 /*v439*/ offset:21760
	ds_load_b128 v[36:39] /*v[292:295]*/, v183 /*v439*/ offset:21792
	ds_load_b128 v[24:27] /*v[280:283]*/, v183 /*v439*/ offset:21824
	ds_load_b128 v[28:31] /*v[284:287]*/, v183 /*v439*/ offset:21856
	ds_load_b128 v[16:19] /*v[272:275]*/, v183 /*v439*/ offset:21888
	ds_load_b128 v[20:23] /*v[276:279]*/, v183 /*v439*/ offset:21920
	ds_load_b128 v[8:11] /*v[264:267]*/, v183 /*v439*/ offset:21952
	ds_load_b128 v[12:15] /*v[268:271]*/, v183 /*v439*/ offset:21984
	ds_load_b128 v[0:3] /*v[256:259]*/, v183 /*v439*/ offset:26112
	ds_load_b128 v[4:7] /*v[260:263]*/, v183 /*v439*/ offset:26144
	s_set_vgpr_msb 0x4101
	ds_load_b128 v[248:251], v183 /*v439*/ offset:26176
	ds_load_b128 v[252:255], v183 /*v439*/ offset:26208
	ds_load_b128 v[240:243], v183 /*v439*/ offset:26240
	ds_load_b128 v[244:247], v183 /*v439*/ offset:26272
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_b128 v[232:235], v183 /*v439*/ offset:26304
	ds_load_b128 v[236:239], v183 /*v439*/ offset:26336
	ds_load_b128 v[224:227], v183 /*v439*/ offset:30464
	ds_load_b128 v[228:231], v183 /*v439*/ offset:30496
	ds_load_b128 v[216:219], v183 /*v439*/ offset:30528
	ds_load_b128 v[220:223], v183 /*v439*/ offset:30560
	ds_load_b128 v[208:211], v183 /*v439*/ offset:30592
	ds_load_b128 v[212:215], v183 /*v439*/ offset:30624
	ds_load_b128 v[200:203], v183 /*v439*/ offset:30656
	ds_load_b128 v[204:207], v183 /*v439*/ offset:30688
	s_cmp_ge_i32 s65, s10
	s_set_vgpr_msb 0x100
	s_cbranch_scc1 .LBB0_29
	v_med3_i32 v152, s35, 0, 0x80
	s_ashr_i32 s49, s48, 31
	s_lshr_b32 s2, s65, 31
	s_mul_u64 s[6:7], s[48:49], s[62:63]
	s_add_co_i32 s2, s65, s2
	v_readfirstlane_b32 s5, v152
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
.LBB0_29:
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x9
	scratch_load_b128 v[152:155], off, off offset:8 nv
	scratch_load_b128 v[156:159], off, off offset:24 nv
	scratch_load_b128 v[184:187], off, off offset:140 nv
	scratch_load_b128 v[188:191], off, off offset:156 nv
	scratch_load_b128 v[160:163], off, off offset:40 nv
	scratch_load_b128 v[164:167], off, off offset:56 nv
	scratch_load_b128 v[168:171], off, off offset:72 nv
	scratch_load_b128 v[172:175], off, off offset:88 nv
	scratch_load_b128 v[176:179], off, off offset:108 nv
	scratch_load_b128 v[180:183], off, off offset:124 nv
	s_set_vgpr_msb 1
	s_wait_loadcnt_dscnt 0x83e
	v_wmma_f32_16x16x32_bf16 v[192:199], v[168:175] /*v[424:431]*/, v[152:159], 0
	s_set_vgpr_msb 0x141
	s_wait_loadcnt 0x6
	v_wmma_f32_16x16x32_bf16 v[192:199] /*v[448:455]*/, v[168:175] /*v[424:431]*/, v[184:191], 0
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b128 v[168:171] /*v[424:427]*/, off, off offset:172 nv
	scratch_load_b128 v[172:175] /*v[428:431]*/, off, off offset:188 nv
	s_set_vgpr_msb 0x4101
	s_wait_loadcnt_dscnt 0x63c
	v_wmma_f32_16x16x32_bf16 v[192:199], v[160:167] /*v[416:423]*/, v[160:167], v[192:199]
	s_wait_loadcnt_dscnt 0x43a
	v_wmma_f32_16x16x32_bf16 v[192:199], v[152:159] /*v[408:415]*/, v[168:175], v[192:199]
	s_set_vgpr_msb 0x100
	s_wait_loadcnt_dscnt 0x238
	v_wmma_f32_16x16x32_bf16 v[192:199], v[128:135], v[176:183], v[192:199]
	s_set_vgpr_msb 0x41
	s_wait_dscnt 0x36
	v_wmma_f32_16x16x32_bf16 v[200:207] /*v[456:463]*/, v[144:151] /*v[400:407]*/, v[184:191], 0
	s_wait_dscnt 0x2e
	v_wmma_f32_16x16x32_bf16 v[208:215] /*v[464:471]*/, v[120:127] /*v[376:383]*/, v[184:191], 0
	s_wait_dscnt 0x26
	v_wmma_f32_16x16x32_bf16 v[216:223] /*v[472:479]*/, v[96:103] /*v[352:359]*/, v[184:191], 0
	s_wait_dscnt 0x1e
	v_wmma_f32_16x16x32_bf16 v[224:231] /*v[480:487]*/, v[64:71] /*v[320:327]*/, v[184:191], 0
	s_wait_dscnt 0x16
	v_wmma_f32_16x16x32_bf16 v[232:239] /*v[488:495]*/, v[32:39] /*v[288:295]*/, v[184:191], 0
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x32_bf16 v[240:247] /*v[496:503]*/, v[0:7] /*v[256:263]*/, v[184:191], 0
	s_set_vgpr_msb 0x4140
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x32_bf16 v[248:255] /*v[504:511]*/, v[224:231], v[184:191], 0
	s_set_vgpr_msb 0x4055
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[192:199] /*v[448:455]*/, v[160:167] /*v[416:423]*/, v[168:175] /*v[424:431]*/, v[192:199] /*v[448:455]*/
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b128 v[160:163] /*v[416:419]*/, off, off offset:204 nv
	scratch_load_b128 v[164:167] /*v[420:423]*/, off, off offset:220 nv
	v_wmma_f32_16x16x32_bf16 v[200:207] /*v[456:463]*/, v[136:143] /*v[392:399]*/, v[168:175] /*v[424:431]*/, v[200:207] /*v[456:463]*/
	v_wmma_f32_16x16x32_bf16 v[208:215] /*v[464:471]*/, v[112:119] /*v[368:375]*/, v[168:175] /*v[424:431]*/, v[208:215] /*v[464:471]*/
	v_wmma_f32_16x16x32_bf16 v[216:223] /*v[472:479]*/, v[88:95] /*v[344:351]*/, v[168:175] /*v[424:431]*/, v[216:223] /*v[472:479]*/
	v_wmma_f32_16x16x32_bf16 v[224:231] /*v[480:487]*/, v[56:63] /*v[312:319]*/, v[168:175] /*v[424:431]*/, v[224:231] /*v[480:487]*/
	v_wmma_f32_16x16x32_bf16 v[232:239] /*v[488:495]*/, v[24:31] /*v[280:287]*/, v[168:175] /*v[424:431]*/, v[232:239] /*v[488:495]*/
	s_set_vgpr_msb 0x5554
	v_wmma_f32_16x16x32_bf16 v[240:247] /*v[496:503]*/, v[248:255], v[168:175] /*v[424:431]*/, v[240:247] /*v[496:503]*/
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x32_bf16 v[248:255] /*v[504:511]*/, v[216:223], v[168:175] /*v[424:431]*/, v[248:255] /*v[504:511]*/
	s_set_vgpr_msb 0x5455
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[192:199] /*v[448:455]*/, v[152:159] /*v[408:415]*/, v[160:167] /*v[416:423]*/, v[192:199] /*v[448:455]*/
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b128 v[152:155] /*v[408:411]*/, off, off offset:236 nv
	scratch_load_b128 v[156:159] /*v[412:415]*/, off, off offset:252 nv
	v_wmma_f32_16x16x32_bf16 v[200:207] /*v[456:463]*/, v[128:135] /*v[384:391]*/, v[160:167] /*v[416:423]*/, v[200:207] /*v[456:463]*/
	v_wmma_f32_16x16x32_bf16 v[208:215] /*v[464:471]*/, v[104:111] /*v[360:367]*/, v[160:167] /*v[416:423]*/, v[208:215] /*v[464:471]*/
	v_wmma_f32_16x16x32_bf16 v[216:223] /*v[472:479]*/, v[80:87] /*v[336:343]*/, v[160:167] /*v[416:423]*/, v[216:223] /*v[472:479]*/
	v_wmma_f32_16x16x32_bf16 v[224:231] /*v[480:487]*/, v[48:55] /*v[304:311]*/, v[160:167] /*v[416:423]*/, v[224:231] /*v[480:487]*/
	v_wmma_f32_16x16x32_bf16 v[232:239] /*v[488:495]*/, v[16:23] /*v[272:279]*/, v[160:167] /*v[416:423]*/, v[232:239] /*v[488:495]*/
	s_set_vgpr_msb 0x5554
	v_wmma_f32_16x16x32_bf16 v[240:247] /*v[496:503]*/, v[240:247], v[160:167] /*v[416:423]*/, v[240:247] /*v[496:503]*/
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[248:255] /*v[504:511]*/, v[208:215], v[160:167] /*v[416:423]*/, v[248:255] /*v[504:511]*/
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[192:199] /*v[448:455]*/, v[128:135], v[152:159] /*v[408:415]*/, v[192:199] /*v[448:455]*/
	s_set_vgpr_msb 0x5401
	v_wmma_f32_16x16x32_bf16 v[128:135], v[144:151] /*v[400:407]*/, v[152:159], 0
	v_wmma_f32_16x16x32_bf16 v[128:135], v[136:143] /*v[392:399]*/, v[160:167], v[128:135]
	v_wmma_f32_16x16x32_bf16 v[128:135], v[128:135] /*v[384:391]*/, v[168:175], v[128:135]
	s_set_vgpr_msb 0x100
	v_wmma_f32_16x16x32_bf16 v[128:135], v[136:143], v[176:183], v[128:135]
	s_set_vgpr_msb 0x54
	v_wmma_f32_16x16x32_bf16 v[200:207] /*v[456:463]*/, v[136:143], v[152:159] /*v[408:415]*/, v[200:207] /*v[456:463]*/
	s_set_vgpr_msb 0x5401
	v_wmma_f32_16x16x32_bf16 v[136:143], v[120:127] /*v[376:383]*/, v[152:159], 0
	v_wmma_f32_16x16x32_bf16 v[136:143], v[112:119] /*v[368:375]*/, v[160:167], v[136:143]
	v_wmma_f32_16x16x32_bf16 v[136:143], v[104:111] /*v[360:367]*/, v[168:175], v[136:143]
	s_set_vgpr_msb 0x100
	v_wmma_f32_16x16x32_bf16 v[136:143], v[144:151], v[176:183], v[136:143]
	s_set_vgpr_msb 0x54
	v_wmma_f32_16x16x32_bf16 v[208:215] /*v[464:471]*/, v[144:151], v[152:159] /*v[408:415]*/, v[208:215] /*v[464:471]*/
	s_set_vgpr_msb 0x5401
	v_wmma_f32_16x16x32_bf16 v[144:151], v[96:103] /*v[352:359]*/, v[152:159], 0
	v_wmma_f32_16x16x32_bf16 v[144:151], v[88:95] /*v[344:351]*/, v[160:167], v[144:151]
	v_wmma_f32_16x16x32_bf16 v[144:151], v[80:87] /*v[336:343]*/, v[168:175], v[144:151]
	v_wmma_f32_16x16x32_bf16 v[144:151], v[72:79] /*v[328:335]*/, v[176:183], v[144:151]
	s_set_vgpr_msb 0x155
	v_wmma_f32_16x16x32_bf16 v[216:223] /*v[472:479]*/, v[72:79] /*v[328:335]*/, v[152:159] /*v[408:415]*/, v[216:223] /*v[472:479]*/
	s_set_vgpr_msb 0x5551
	v_wmma_f32_16x16x32_bf16 v[72:79] /*v[328:335]*/, v[64:71] /*v[320:327]*/, v[152:159], 0
	v_wmma_f32_16x16x32_bf16 v[72:79] /*v[328:335]*/, v[56:63] /*v[312:319]*/, v[160:167], v[72:79] /*v[328:335]*/
	v_wmma_f32_16x16x32_bf16 v[72:79] /*v[328:335]*/, v[48:55] /*v[304:311]*/, v[168:175], v[72:79] /*v[328:335]*/
	v_wmma_f32_16x16x32_bf16 v[72:79] /*v[328:335]*/, v[40:47] /*v[296:303]*/, v[176:183], v[72:79] /*v[328:335]*/
	s_set_vgpr_msb 0x5155
	v_wmma_f32_16x16x32_bf16 v[224:231] /*v[480:487]*/, v[40:47] /*v[296:303]*/, v[152:159] /*v[408:415]*/, v[224:231] /*v[480:487]*/
	s_set_vgpr_msb 0x5551
	v_wmma_f32_16x16x32_bf16 v[40:47] /*v[296:303]*/, v[32:39] /*v[288:295]*/, v[152:159], 0
	v_wmma_f32_16x16x32_bf16 v[40:47] /*v[296:303]*/, v[24:31] /*v[280:287]*/, v[160:167], v[40:47] /*v[296:303]*/
	v_wmma_f32_16x16x32_bf16 v[24:31] /*v[280:287]*/, v[0:7] /*v[256:263]*/, v[152:159], 0
	s_set_vgpr_msb 0x5150
	v_wmma_f32_16x16x32_bf16 v[32:39] /*v[288:295]*/, v[224:231], v[152:159], 0
	v_wmma_f32_16x16x32_bf16 v[24:31] /*v[280:287]*/, v[248:255], v[160:167], v[24:31] /*v[280:287]*/
	v_wmma_f32_16x16x32_bf16 v[32:39] /*v[288:295]*/, v[216:223], v[160:167], v[32:39] /*v[288:295]*/
	s_set_vgpr_msb 0x5051
	v_wmma_f32_16x16x32_bf16 v[40:47] /*v[296:303]*/, v[16:23] /*v[272:279]*/, v[168:175], v[40:47] /*v[296:303]*/
	s_set_vgpr_msb 0x5150
	v_wmma_f32_16x16x32_bf16 v[24:31] /*v[280:287]*/, v[240:247], v[168:175], v[24:31] /*v[280:287]*/
	v_wmma_f32_16x16x32_bf16 v[32:39] /*v[288:295]*/, v[208:215], v[168:175], v[32:39] /*v[288:295]*/
	s_set_vgpr_msb 0x5051
	v_wmma_f32_16x16x32_bf16 v[40:47] /*v[296:303]*/, v[8:15] /*v[264:271]*/, v[176:183], v[40:47] /*v[296:303]*/
	s_set_vgpr_msb 0x5155
	v_wmma_f32_16x16x32_bf16 v[232:239] /*v[488:495]*/, v[8:15] /*v[264:271]*/, v[152:159] /*v[408:415]*/, v[232:239] /*v[488:495]*/
	s_set_vgpr_msb 0x5550
	v_wmma_f32_16x16x32_bf16 v[24:31] /*v[280:287]*/, v[232:239], v[176:183], v[24:31] /*v[280:287]*/
	s_set_vgpr_msb 0x5054
	v_wmma_f32_16x16x32_bf16 v[240:247] /*v[496:503]*/, v[232:239], v[152:159] /*v[408:415]*/, v[240:247] /*v[496:503]*/
	s_set_vgpr_msb 0x5450
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[32:39] /*v[288:295]*/, v[200:207], v[176:183], v[32:39] /*v[288:295]*/
	s_set_vgpr_msb 0x5054
	v_wmma_f32_16x16x32_bf16 v[248:255] /*v[504:511]*/, v[200:207], v[152:159] /*v[408:415]*/, v[248:255] /*v[504:511]*/
	s_set_vgpr_msb 0x5441
	ds_load_tr16_b128 v[184:187] /*v[440:443]*/, v177 /*v433*/
	s_wait_alu depctr_va_vdst(0)
	ds_load_tr16_b128 v[152:155] /*v[408:411]*/, v177 /*v433*/ offset:32
	ds_load_tr16_b128 v[188:191] /*v[444:447]*/, v177 /*v433*/ offset:4608
	ds_load_tr16_b128 v[156:159] /*v[412:415]*/, v177 /*v433*/ offset:4640
	ds_load_tr16_b128 v[0:3] /*v[256:259]*/, v177 /*v433*/ offset:9216
	ds_load_tr16_b128 v[144:147] /*v[400:403]*/, v177 /*v433*/ offset:9248
	ds_load_tr16_b128 v[4:7] /*v[260:263]*/, v177 /*v433*/ offset:13824
	ds_load_tr16_b128 v[148:151] /*v[404:407]*/, v177 /*v433*/ offset:13856
	ds_load_tr16_b128 v[168:171] /*v[424:427]*/, v177 /*v433*/ offset:18432
	ds_load_tr16_b128 v[136:139] /*v[392:395]*/, v177 /*v433*/ offset:18464
	ds_load_tr16_b128 v[172:175] /*v[428:431]*/, v177 /*v433*/ offset:23040
	ds_load_tr16_b128 v[140:143] /*v[396:399]*/, v177 /*v433*/ offset:23072
	ds_load_tr16_b128 v[160:163] /*v[416:419]*/, v177 /*v433*/ offset:27648
	ds_load_tr16_b128 v[128:131] /*v[384:387]*/, v177 /*v433*/ offset:27680
	ds_load_tr16_b128 v[164:167] /*v[420:423]*/, v177 /*v433*/ offset:32256
	ds_load_tr16_b128 v[132:135] /*v[388:391]*/, v177 /*v433*/ offset:32288
	ds_load_tr16_b128 v[120:123] /*v[376:379]*/, v177 /*v433*/ offset:64
	ds_load_tr16_b128 v[88:91] /*v[344:347]*/, v177 /*v433*/ offset:96
	ds_load_tr16_b128 v[124:127] /*v[380:383]*/, v177 /*v433*/ offset:4672
	ds_load_tr16_b128 v[92:95] /*v[348:351]*/, v177 /*v433*/ offset:4704
	ds_load_tr16_b128 v[112:115] /*v[368:371]*/, v177 /*v433*/ offset:9280
	ds_load_tr16_b128 v[80:83] /*v[336:339]*/, v177 /*v433*/ offset:9312
	ds_load_tr16_b128 v[116:119] /*v[372:375]*/, v177 /*v433*/ offset:13888
	ds_load_tr16_b128 v[84:87] /*v[340:343]*/, v177 /*v433*/ offset:13920
	ds_load_tr16_b128 v[104:107] /*v[360:363]*/, v177 /*v433*/ offset:18496
	ds_load_tr16_b128 v[8:11] /*v[264:267]*/, v177 /*v433*/ offset:18528
	ds_load_tr16_b128 v[108:111] /*v[364:367]*/, v177 /*v433*/ offset:23104
	ds_load_tr16_b128 v[12:15] /*v[268:271]*/, v177 /*v433*/ offset:23136
	ds_load_tr16_b128 v[96:99] /*v[352:355]*/, v177 /*v433*/ offset:27712
	ds_load_tr16_b128 v[64:67] /*v[320:323]*/, v177 /*v433*/ offset:27744
	ds_load_tr16_b128 v[100:103] /*v[356:359]*/, v177 /*v433*/ offset:32320
	ds_load_tr16_b128 v[68:71] /*v[324:327]*/, v177 /*v433*/ offset:32352
	ds_load_tr16_b128 v[56:59] /*v[312:315]*/, v177 /*v433*/ offset:128
	s_set_vgpr_msb 0x4101
	ds_load_tr16_b128 v[152:155], v177 /*v433*/ offset:160
	s_set_vgpr_msb 0x141
	ds_load_tr16_b128 v[60:63] /*v[316:319]*/, v177 /*v433*/ offset:4736
	s_set_vgpr_msb 0x4101
	ds_load_tr16_b128 v[156:159], v177 /*v433*/ offset:4768
	s_wait_dscnt 0x2
	scratch_store_b128 off, v[152:155], off offset:620 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[156:159], off offset:636 nv
	s_set_vgpr_msb 0x141
	ds_load_tr16_b128 v[48:51] /*v[304:307]*/, v177 /*v433*/ offset:9344
	s_wait_alu depctr_vm_vsrc(0)
	s_set_vgpr_msb 0x4101
	ds_load_tr16_b128 v[152:155], v177 /*v433*/ offset:9376
	s_set_vgpr_msb 0x141
	ds_load_tr16_b128 v[52:55] /*v[308:311]*/, v177 /*v433*/ offset:13952
	s_set_vgpr_msb 0x4101
	ds_load_tr16_b128 v[156:159], v177 /*v433*/ offset:13984
	s_wait_dscnt 0x2
	scratch_store_b128 off, v[152:155], off offset:588 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[156:159], off offset:604 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_tr16_b128 v[156:159], v177 /*v433*/ offset:18560
	ds_load_tr16_b128 v[152:155], v177 /*v433*/ offset:18592
	ds_load_tr16_b128 v[160:163], v177 /*v433*/ offset:23168
	s_wait_dscnt 0x2
	scratch_store_b128 off, v[156:159], off offset:684 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[160:163], off offset:700 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_tr16_b128 v[156:159], v177 /*v433*/ offset:23200
	scratch_store_b128 off, v[152:155], off offset:556 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[156:159], off offset:572 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_tr16_b128 v[156:159], v177 /*v433*/ offset:27776
	ds_load_tr16_b128 v[152:155], v177 /*v433*/ offset:27808
	ds_load_tr16_b128 v[160:163], v177 /*v433*/ offset:32384
	s_wait_dscnt 0x2
	scratch_store_b128 off, v[156:159], off offset:652 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[160:163], off offset:668 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_tr16_b128 v[156:159], v177 /*v433*/ offset:32416
	scratch_store_b128 off, v[152:155], off offset:524 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[156:159], off offset:540 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_tr16_b128 v[156:159], v177 /*v433*/ offset:192
	ds_load_tr16_b128 v[152:155], v177 /*v433*/ offset:224
	ds_load_tr16_b128 v[160:163], v177 /*v433*/ offset:4800
	s_wait_dscnt 0x2
	scratch_store_b128 off, v[156:159], off offset:492 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[160:163], off offset:508 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_tr16_b128 v[156:159], v177 /*v433*/ offset:4832
	scratch_store_b128 off, v[152:155], off offset:364 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[156:159], off offset:380 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_tr16_b128 v[156:159], v177 /*v433*/ offset:9408
	ds_load_tr16_b128 v[152:155], v177 /*v433*/ offset:9440
	ds_load_tr16_b128 v[160:163], v177 /*v433*/ offset:14016
	s_wait_dscnt 0x2
	scratch_store_b128 off, v[156:159], off offset:460 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[160:163], off offset:476 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_tr16_b128 v[156:159], v177 /*v433*/ offset:14048
	scratch_store_b128 off, v[152:155], off offset:332 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[156:159], off offset:348 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_tr16_b128 v[156:159], v177 /*v433*/ offset:18624
	ds_load_tr16_b128 v[152:155], v177 /*v433*/ offset:18656
	ds_load_tr16_b128 v[160:163], v177 /*v433*/ offset:23232
	s_wait_dscnt 0x2
	scratch_store_b128 off, v[156:159], off offset:428 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[160:163], off offset:444 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_tr16_b128 v[156:159], v177 /*v433*/ offset:23264
	scratch_store_b128 off, v[152:155], off offset:300 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[156:159], off offset:316 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_tr16_b128 v[156:159], v177 /*v433*/ offset:27840
	ds_load_tr16_b128 v[152:155], v177 /*v433*/ offset:27872
	ds_load_tr16_b128 v[160:163], v177 /*v433*/ offset:32448
	s_wait_dscnt 0x2
	scratch_store_b128 off, v[156:159], off offset:396 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[160:163], off offset:412 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_tr16_b128 v[156:159], v177 /*v433*/ offset:32480
	scratch_store_b128 off, v[152:155], off offset:268 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[156:159], off offset:284 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	s_set_vgpr_msb 0x100
	v_max3_num_f32 v152, v192, v193, v194
	s_set_vgpr_msb 21
	v_max3_num_f32 v153, v192 /*v448*/, v193 /*v449*/, v194 /*v450*/
	s_set_vgpr_msb 0x1500
	v_max3_num_f32 v154, v195, v196, v197
	s_set_vgpr_msb 21
	v_max3_num_f32 v155, v195 /*v451*/, v196 /*v452*/, v197 /*v453*/
	s_set_vgpr_msb 0x1500
	v_max3_num_f32 v156, v198, v199, v128
	s_set_vgpr_msb 21
	v_max3_num_f32 v157, v198 /*v454*/, v199 /*v455*/, v200 /*v456*/
	s_set_vgpr_msb 0x1500
	v_max3_num_f32 v158, v129, v130, v131
	v_max3_num_f32 v160, v132, v133, v134
	v_max3_num_f32 v162, v135, v136, v137
	v_max3_num_f32 v164, v138, v139, v140
	v_max3_num_f32 v166, v141, v142, v143
	v_max3_num_f32 v168, v144, v145, v146
	v_max3_num_f32 v170, v147, v148, v149
	s_set_vgpr_msb 16
	v_max3_num_f32 v172, v150, v151, v72 /*v328*/
	s_set_vgpr_msb 0x1015
	v_max3_num_f32 v174, v73 /*v329*/, v74 /*v330*/, v75 /*v331*/
	v_max3_num_f32 v176, v76 /*v332*/, v77 /*v333*/, v78 /*v334*/
	v_max3_num_f32 v178, v79 /*v335*/, v40 /*v296*/, v41 /*v297*/
	v_max3_num_f32 v180, v42 /*v298*/, v43 /*v299*/, v44 /*v300*/
	v_max3_num_f32 v182, v45 /*v301*/, v46 /*v302*/, v47 /*v303*/
	v_max3_num_f32 v184, v24 /*v280*/, v25 /*v281*/, v26 /*v282*/
	v_max3_num_f32 v186, v27 /*v283*/, v28 /*v284*/, v29 /*v285*/
	v_max_num_f32_e32 v188, v30 /*v286*/, v31 /*v287*/
	v_max3_num_f32 v190, v33 /*v289*/, v34 /*v290*/, v35 /*v291*/
	v_max3_num_f32 v159, v201 /*v457*/, v202 /*v458*/, v203 /*v459*/
	v_max3_num_f32 v161, v204 /*v460*/, v205 /*v461*/, v206 /*v462*/
	v_max3_num_f32 v163, v207 /*v463*/, v208 /*v464*/, v209 /*v465*/
	v_max3_num_f32 v165, v210 /*v466*/, v211 /*v467*/, v212 /*v468*/
	v_max3_num_f32 v167, v213 /*v469*/, v214 /*v470*/, v215 /*v471*/
	v_max3_num_f32 v169, v216 /*v472*/, v217 /*v473*/, v218 /*v474*/
	v_max3_num_f32 v171, v219 /*v475*/, v220 /*v476*/, v221 /*v477*/
	v_max3_num_f32 v173, v222 /*v478*/, v223 /*v479*/, v224 /*v480*/
	v_max3_num_f32 v175, v225 /*v481*/, v226 /*v482*/, v227 /*v483*/
	v_max3_num_f32 v177, v228 /*v484*/, v229 /*v485*/, v230 /*v486*/
	v_max3_num_f32 v179, v231 /*v487*/, v232 /*v488*/, v233 /*v489*/
	v_max3_num_f32 v181, v234 /*v490*/, v235 /*v491*/, v236 /*v492*/
	v_max3_num_f32 v183, v237 /*v493*/, v238 /*v494*/, v239 /*v495*/
	v_max3_num_f32 v185, v240 /*v496*/, v241 /*v497*/, v242 /*v498*/
	v_max3_num_f32 v187, v243 /*v499*/, v244 /*v500*/, v245 /*v501*/
	v_max_num_f32_e32 v189, v246 /*v502*/, v247 /*v503*/
	v_max3_num_f32 v191, v249 /*v505*/, v250 /*v506*/, v251 /*v507*/
	v_max3_num_f32 v200, v36 /*v292*/, v37 /*v293*/, v38 /*v294*/
	s_set_vgpr_msb 0x1500
	v_max3_num_f32 v152, v152, v154, v156
	v_max3_num_f32 v153, v153, v155, v157
	v_max3_num_f32 v154, v158, v160, v162
	v_max3_num_f32 v155, v164, v166, v168
	v_max3_num_f32 v156, v170, v172, v174
	v_max3_num_f32 v157, v176, v178, v180
	v_max3_num_f32 v158, v182, v184, v186
	s_set_vgpr_msb 4
	v_max3_num_f32 v160, v188, v32 /*v288*/, v190
	s_set_vgpr_msb 0x415
	v_max3_num_f32 v201, v252 /*v508*/, v253 /*v509*/, v254 /*v510*/
	s_set_vgpr_msb 0x1500
	v_max3_num_f32 v159, v159, v161, v163
	v_max3_num_f32 v161, v165, v167, v169
	v_max3_num_f32 v152, v152, v154, v155
	v_max3_num_f32 v154, v156, v157, v158
	s_set_vgpr_msb 16
	v_max3_num_f32 v155, v160, v200, v39 /*v295*/
	s_set_vgpr_msb 0x1000
	v_max3_num_f32 v156, v171, v173, v175
	v_max3_num_f32 v157, v177, v179, v181
	v_max3_num_f32 v158, v183, v185, v187
	s_set_vgpr_msb 4
	v_max3_num_f32 v160, v189, v248 /*v504*/, v191
	s_set_vgpr_msb 0x400
	v_max3_num_f32 v152, v152, v154, v155
	v_max3_num_f32 v153, v153, v159, v161
	v_max3_num_f32 v154, v156, v157, v158
	s_set_vgpr_msb 16
	v_max3_num_f32 v155, v160, v201, v255 /*v511*/
	s_set_vgpr_msb 0x1000
	v_max3_num_f32 v153, v153, v154, v155
	v_dual_mov_b32 v156, v152 :: v_dual_mov_b32 v154, v153
	v_permlanex16_b32 v156, v156, s56, 0xfedcba98
	v_permlanex16_b32 v154, v154, s56, 0xfedcba98
	v_dual_max_num_f32 v152, v152, v156 :: v_dual_max_num_f32 v153, v153, v154
	s_set_vgpr_msb 4
	v_sub_f32_e32 v155, v152, v182 /*v438*/
	v_max_num_f32_e32 v152, v152, v182 /*v438*/
	v_sub_f32_e32 v154, v153, v181 /*v437*/
	s_set_vgpr_msb 0x400
	v_cmp_lt_f32_e32 vcc_lo, 0x41000000, v155
	s_cmp_eq_u32 vcc_lo, 0
	s_cselect_b32 s2, -1, 0
	s_set_vgpr_msb 0x44
	v_cndmask_b32_e64 v180 /*v436*/, v152, v182 /*v438*/, s2
	s_set_vgpr_msb 0x4401
	v_cmp_lt_f32_e64 s2, 0x41000000, v154
	v_max_num_f32_e32 v152, v181 /*v437*/, v153
	s_cmp_lg_u32 s2, 0
	s_cselect_b32 s5, -1, 0
	s_cmp_eq_u32 s2, 0
	s_cselect_b32 s2, -1, 0
	s_set_vgpr_msb 0x144
	v_cndmask_b32_e64 v179 /*v435*/, v152, v181 /*v437*/, s2
	s_set_vgpr_msb 0x4404
	v_mul_f32_e32 v226, 0xbfb8aa3b, v180 /*v436*/
	s_set_vgpr_msb 0x444
	v_mul_f32_e32 v16 /*v272*/, 0xbfb8aa3b, v179 /*v435*/
	s_set_vgpr_msb 0x4400
	v_pk_fma_f32 v[152:153], v[192:193], s[52:53], v[226:227] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[128:129], v[128:129], s[52:53], v[226:227] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[130:131], v[130:131], s[52:53], v[226:227] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[132:133], v[132:133], s[52:53], v[226:227] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x51
	v_pk_fma_f32 v[18:19] /*v[274:275]*/, v[192:193] /*v[448:449]*/, s[52:53], v[16:17] /*v[272:273]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[20:21] /*v[276:277]*/, v[194:195] /*v[450:451]*/, s[52:53], v[16:17] /*v[272:273]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[22:23] /*v[278:279]*/, v[196:197] /*v[452:453]*/, s[52:53], v[16:17] /*v[272:273]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5100
	v_exp_f32_e32 v154, v152
	v_exp_f32_e32 v156, v153
	v_nop
	v_pk_fma_f32 v[152:153], v[198:199], s[52:53], v[226:227] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 1
	v_exp_f32_e32 v155, v18 /*v274*/
	v_exp_f32_e32 v157, v19 /*v275*/
	v_exp_f32_e32 v193, v20 /*v276*/
	s_set_vgpr_msb 0x151
	v_pk_fma_f32 v[18:19] /*v[274:275]*/, v[198:199] /*v[454:455]*/, s[52:53], v[16:17] /*v[272:273]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5101
	v_exp_f32_e32 v199, v21 /*v277*/
	v_exp_f32_e32 v165, v22 /*v278*/
	s_set_vgpr_msb 0x151
	v_pk_fma_f32 v[20:21] /*v[276:277]*/, v[200:201] /*v[456:457]*/, s[52:53], v[16:17] /*v[272:273]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5101
	v_exp_f32_e32 v173, v23 /*v279*/
	v_nop
	s_set_vgpr_msb 0x151
	v_pk_fma_f32 v[22:23] /*v[278:279]*/, v[202:203] /*v[458:459]*/, s[52:53], v[16:17] /*v[272:273]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5100
	v_exp_f32_e32 v204, v128
	v_exp_f32_e32 v212, v129
	v_exp_f32_e32 v222, v130
	v_pk_fma_f32 v[128:129], v[134:135], s[52:53], v[226:227] op_sel_hi:[1,0,0]
	v_exp_f32_e32 v228, v131
	v_nop
	v_pk_fma_f32 v[130:131], v[136:137], s[52:53], v[226:227] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 1
	v_exp_f32_e32 v181, v18 /*v274*/
	v_exp_f32_e32 v189, v19 /*v275*/
	v_exp_f32_e32 v205, v20 /*v276*/
	s_set_vgpr_msb 0x151
	v_pk_fma_f32 v[18:19] /*v[274:275]*/, v[204:205] /*v[460:461]*/, s[52:53], v[16:17] /*v[272:273]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5101
	v_exp_f32_e32 v213, v21 /*v277*/
	v_exp_f32_e32 v223, v22 /*v278*/
	s_set_vgpr_msb 0x151
	v_pk_fma_f32 v[20:21] /*v[276:277]*/, v[206:207] /*v[462:463]*/, s[52:53], v[16:17] /*v[272:273]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5101
	v_exp_f32_e32 v229, v23 /*v279*/
	v_nop
	s_set_vgpr_msb 0x151
	v_pk_fma_f32 v[22:23] /*v[278:279]*/, v[208:209] /*v[464:465]*/, s[52:53], v[16:17] /*v[272:273]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5100
	v_pk_fma_f32 v[158:159], v[194:195], s[52:53], v[226:227] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[160:161], v[196:197], s[52:53], v[226:227] op_sel_hi:[1,0,0]
	v_exp_f32_e32 v232, v132
	v_exp_f32_e32 v240, v133
	v_exp_f32_e32 v246, v128
	v_pk_fma_f32 v[132:133], v[138:139], s[52:53], v[226:227] op_sel_hi:[1,0,0]
	v_exp_f32_e32 v252, v129
	v_exp_f32_e32 v128, v130
	v_pk_fma_f32 v[134:135], v[140:141], s[52:53], v[226:227] op_sel_hi:[1,0,0]
	v_exp_f32_e32 v130, v131
	s_set_vgpr_msb 1
	v_exp_f32_e32 v233, v18 /*v274*/
	v_exp_f32_e32 v241, v19 /*v275*/
	v_exp_f32_e32 v247, v20 /*v276*/
	s_set_vgpr_msb 0x151
	v_pk_fma_f32 v[18:19] /*v[274:275]*/, v[210:211] /*v[466:467]*/, s[52:53], v[16:17] /*v[272:273]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5101
	v_exp_f32_e32 v253, v21 /*v277*/
	v_exp_f32_e32 v129, v22 /*v278*/
	s_set_vgpr_msb 0x151
	v_pk_fma_f32 v[20:21] /*v[276:277]*/, v[212:213] /*v[468:469]*/, s[52:53], v[16:17] /*v[272:273]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5101
	v_exp_f32_e32 v131, v23 /*v279*/
	v_nop
	s_set_vgpr_msb 0x151
	v_pk_fma_f32 v[22:23] /*v[278:279]*/, v[214:215] /*v[470:471]*/, s[52:53], v[16:17] /*v[272:273]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5100
	v_exp_f32_e32 v192, v158
	v_exp_f32_e32 v198, v159
	v_exp_f32_e32 v172, v161
	v_pk_fma_f32 v[138:139], v[142:143], s[52:53], v[226:227] op_sel_hi:[1,0,0]
	v_exp_f32_e32 v136, v133
	v_exp_f32_e32 v140, v134
	v_pk_fma_f32 v[142:143], v[144:145], s[52:53], v[226:227] op_sel_hi:[1,0,0]
	v_exp_f32_e32 v158, v135
	v_nop
	v_pk_fma_f32 v[134:135], v[146:147], s[52:53], v[226:227] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 1
	v_exp_f32_e32 v133, v18 /*v274*/
	v_exp_f32_e32 v137, v19 /*v275*/
	v_exp_f32_e32 v141, v20 /*v276*/
	s_set_vgpr_msb 0x151
	v_pk_fma_f32 v[18:19] /*v[274:275]*/, v[216:217] /*v[472:473]*/, s[52:53], v[16:17] /*v[272:273]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5101
	v_exp_f32_e32 v159, v21 /*v277*/
	v_exp_f32_e32 v195, v22 /*v278*/
	s_set_vgpr_msb 0x151
	v_pk_fma_f32 v[20:21] /*v[276:277]*/, v[218:219] /*v[474:475]*/, s[52:53], v[16:17] /*v[272:273]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5101
	v_exp_f32_e32 v161, v23 /*v279*/
	v_nop
	s_set_vgpr_msb 0x151
	v_pk_fma_f32 v[22:23] /*v[278:279]*/, v[220:221] /*v[476:477]*/, s[52:53], v[16:17] /*v[272:273]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5100
	v_exp_f32_e32 v164, v160
	v_exp_f32_e32 v194, v138
	v_exp_f32_e32 v160, v139
	v_exp_f32_e32 v174, v142
	v_pk_fma_f32 v[138:139], v[148:149], s[52:53], v[226:227] op_sel_hi:[1,0,0]
	v_exp_f32_e32 v186, v143
	v_exp_f32_e32 v190, v134
	v_pk_fma_f32 v[142:143], v[150:151], s[52:53], v[226:227] op_sel_hi:[1,0,0]
	v_exp_f32_e32 v210, v135
	v_nop
	s_set_vgpr_msb 1
	v_pk_fma_f32 v[134:135], v[72:73] /*v[328:329]*/, s[52:53], v[226:227] op_sel_hi:[1,0,0]
	v_exp_f32_e32 v175, v18 /*v274*/
	v_exp_f32_e32 v187, v19 /*v275*/
	v_exp_f32_e32 v191, v20 /*v276*/
	s_set_vgpr_msb 0x151
	v_pk_fma_f32 v[18:19] /*v[274:275]*/, v[222:223] /*v[478:479]*/, s[52:53], v[16:17] /*v[272:273]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5101
	v_exp_f32_e32 v211, v21 /*v277*/
	v_exp_f32_e32 v235, v22 /*v278*/
	s_set_vgpr_msb 0x151
	v_pk_fma_f32 v[20:21] /*v[276:277]*/, v[224:225] /*v[480:481]*/, s[52:53], v[16:17] /*v[272:273]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5101
	v_exp_f32_e32 v243, v23 /*v279*/
	v_nop
	s_set_vgpr_msb 0x151
	v_pk_fma_f32 v[22:23] /*v[278:279]*/, v[226:227] /*v[482:483]*/, s[52:53], v[16:17] /*v[272:273]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5100
	v_exp_f32_e32 v180, v152
	v_exp_f32_e32 v188, v153
	v_exp_f32_e32 v234, v138
	v_exp_f32_e32 v242, v139
	s_set_vgpr_msb 1
	v_pk_fma_f32 v[144:145], v[74:75] /*v[330:331]*/, s[52:53], v[226:227] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x100
	v_exp_f32_e32 v254, v143
	s_set_vgpr_msb 1
	v_pk_fma_f32 v[148:149], v[76:77] /*v[332:333]*/, s[52:53], v[226:227] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x100
	v_exp_f32_e32 v138, v135
	s_set_vgpr_msb 1
	v_pk_fma_f32 v[152:153], v[78:79] /*v[334:335]*/, s[52:53], v[226:227] op_sel_hi:[1,0,0]
	v_exp_f32_e32 v249, v18 /*v274*/
	v_exp_f32_e32 v255, v19 /*v275*/
	v_exp_f32_e32 v135, v20 /*v276*/
	s_set_vgpr_msb 0x151
	v_pk_fma_f32 v[18:19] /*v[274:275]*/, v[228:229] /*v[484:485]*/, s[52:53], v[16:17] /*v[272:273]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5101
	v_exp_f32_e32 v139, v21 /*v277*/
	v_exp_f32_e32 v143, v22 /*v278*/
	s_set_vgpr_msb 0x151
	v_pk_fma_f32 v[20:21] /*v[276:277]*/, v[230:231] /*v[486:487]*/, s[52:53], v[16:17] /*v[272:273]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5101
	v_exp_f32_e32 v147, v23 /*v279*/
	v_nop
	s_set_vgpr_msb 0x151
	v_pk_fma_f32 v[22:23] /*v[278:279]*/, v[232:233] /*v[488:489]*/, s[52:53], v[16:17] /*v[272:273]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5100
	v_exp_f32_e32 v248, v142
	v_exp_f32_e32 v142, v144
	v_exp_f32_e32 v146, v145
	v_exp_f32_e32 v150, v148
	s_set_vgpr_msb 1
	v_pk_fma_f32 v[144:145], v[40:41] /*v[296:297]*/, s[52:53], v[226:227] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x100
	v_exp_f32_e32 v162, v149
	v_exp_f32_e32 v166, v152
	s_set_vgpr_msb 1
	v_pk_fma_f32 v[148:149], v[42:43] /*v[298:299]*/, s[52:53], v[226:227] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x100
	v_exp_f32_e32 v176, v153
	v_nop
	s_set_vgpr_msb 1
	v_pk_fma_f32 v[152:153], v[44:45] /*v[300:301]*/, s[52:53], v[226:227] op_sel_hi:[1,0,0]
	v_exp_f32_e32 v151, v18 /*v274*/
	v_exp_f32_e32 v163, v19 /*v275*/
	v_exp_f32_e32 v167, v20 /*v276*/
	s_set_vgpr_msb 0x151
	v_pk_fma_f32 v[18:19] /*v[274:275]*/, v[234:235] /*v[490:491]*/, s[52:53], v[16:17] /*v[272:273]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5101
	v_exp_f32_e32 v177, v21 /*v277*/
	v_exp_f32_e32 v183, v22 /*v278*/
	s_set_vgpr_msb 0x151
	v_pk_fma_f32 v[20:21] /*v[276:277]*/, v[236:237] /*v[492:493]*/, s[52:53], v[16:17] /*v[272:273]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5101
	v_exp_f32_e32 v201, v23 /*v279*/
	v_nop
	s_set_vgpr_msb 0x151
	v_pk_fma_f32 v[22:23] /*v[278:279]*/, v[238:239] /*v[494:495]*/, s[52:53], v[16:17] /*v[272:273]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5100
	v_exp_f32_e32 v182, v144
	v_exp_f32_e32 v200, v145
	v_exp_f32_e32 v206, v148
	s_set_vgpr_msb 1
	v_pk_fma_f32 v[144:145], v[46:47] /*v[302:303]*/, s[52:53], v[226:227] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x100
	v_exp_f32_e32 v214, v149
	v_exp_f32_e32 v218, v152
	s_set_vgpr_msb 1
	v_pk_fma_f32 v[148:149], v[24:25] /*v[280:281]*/, s[52:53], v[226:227] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x100
	v_exp_f32_e32 v152, v153
	s_set_vgpr_msb 1
	v_pk_fma_f32 v[168:169], v[26:27] /*v[282:283]*/, s[52:53], v[226:227] op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[170:171], v[28:29] /*v[284:285]*/, s[52:53], v[226:227] op_sel_hi:[1,0,0]
	v_exp_f32_e32 v207, v18 /*v274*/
	v_exp_f32_e32 v215, v19 /*v275*/
	v_exp_f32_e32 v219, v20 /*v276*/
	s_set_vgpr_msb 0x151
	v_pk_fma_f32 v[18:19] /*v[274:275]*/, v[240:241] /*v[496:497]*/, s[52:53], v[16:17] /*v[272:273]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5101
	v_exp_f32_e32 v153, v21 /*v277*/
	v_exp_f32_e32 v225, v22 /*v278*/
	s_set_vgpr_msb 0x151
	v_pk_fma_f32 v[20:21] /*v[276:277]*/, v[242:243] /*v[498:499]*/, s[52:53], v[16:17] /*v[272:273]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5101
	v_exp_f32_e32 v237, v23 /*v279*/
	v_nop
	s_set_vgpr_msb 0x151
	v_pk_fma_f32 v[22:23] /*v[278:279]*/, v[244:245] /*v[500:501]*/, s[52:53], v[16:17] /*v[272:273]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5100
	v_exp_f32_e32 v224, v144
	v_exp_f32_e32 v236, v145
	v_exp_f32_e32 v144, v148
	v_exp_f32_e32 v148, v149
	v_exp_f32_e32 v196, v168
	s_set_vgpr_msb 1
	v_pk_fma_f32 v[184:185], v[30:31] /*v[286:287]*/, s[52:53], v[226:227] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x100
	v_exp_f32_e32 v168, v169
	s_set_vgpr_msb 1
	v_pk_fma_f32 v[208:209], v[32:33] /*v[288:289]*/, s[52:53], v[226:227] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x100
	v_exp_f32_e32 v178, v171
	s_set_vgpr_msb 1
	v_pk_fma_f32 v[220:221], v[34:35] /*v[290:291]*/, s[52:53], v[226:227] op_sel_hi:[1,0,0]
	v_exp_f32_e32 v145, v18 /*v274*/
	v_exp_f32_e32 v149, v19 /*v275*/
	v_exp_f32_e32 v197, v20 /*v276*/
	v_exp_f32_e32 v169, v21 /*v277*/
	s_set_vgpr_msb 0x151
	v_pk_fma_f32 v[18:19] /*v[274:275]*/, v[246:247] /*v[502:503]*/, s[52:53], v[16:17] /*v[272:273]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5101
	v_exp_f32_e32 v171, v22 /*v278*/
	s_set_vgpr_msb 0x151
	v_pk_fma_f32 v[20:21] /*v[276:277]*/, v[248:249] /*v[504:505]*/, s[52:53], v[16:17] /*v[272:273]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5101
	v_exp_f32_e32 v179, v23 /*v279*/
	v_nop
	s_set_vgpr_msb 0x151
	v_pk_fma_f32 v[22:23] /*v[278:279]*/, v[250:251] /*v[506:507]*/, s[52:53], v[16:17] /*v[272:273]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5100
	v_exp_f32_e32 v202, v185
	s_set_vgpr_msb 1
	v_pk_fma_f32 v[230:231], v[36:37] /*v[292:293]*/, s[52:53], v[226:227] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x100
	v_exp_f32_e32 v216, v209
	s_set_vgpr_msb 1
	v_pk_fma_f32 v[244:245], v[38:39] /*v[294:295]*/, s[52:53], v[226:227] op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x100
	v_exp_f32_e32 v226, v221
	s_set_vgpr_msb 1
	v_exp_f32_e32 v185, v18 /*v274*/
	v_exp_f32_e32 v203, v19 /*v275*/
	v_exp_f32_e32 v209, v20 /*v276*/
	v_exp_f32_e32 v217, v21 /*v277*/
	v_exp_f32_e32 v221, v22 /*v278*/
	s_set_vgpr_msb 0x151
	v_pk_fma_f32 v[18:19] /*v[274:275]*/, v[252:253] /*v[508:509]*/, s[52:53], v[16:17] /*v[272:273]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5101
	v_exp_f32_e32 v227, v23 /*v279*/
	s_set_vgpr_msb 0x140
	v_pk_add_f32 v[20:21] /*v[276:277]*/, v[154:155], v[156:157]
	v_pk_add_f32 v[22:23] /*v[278:279]*/, v[198:199], v[164:165]
	s_set_vgpr_msb 0x4000
	v_exp_f32_e32 v132, v132
	v_exp_f32_e32 v134, v134
	v_exp_f32_e32 v238, v231
	s_set_vgpr_msb 1
	v_exp_f32_e32 v231, v18 /*v274*/
	v_exp_f32_e32 v239, v19 /*v275*/
	v_nop
	s_set_vgpr_msb 0x144
	v_pk_add_f32 v[18:19] /*v[274:275]*/, v[192:193], v[20:21] /*v[276:277]*/
	v_pk_add_f32 v[20:21] /*v[276:277]*/, v[172:173], v[22:23] /*v[278:279]*/
	s_set_vgpr_msb 0x4440
	v_pk_add_f32 v[22:23] /*v[278:279]*/, v[180:181], v[188:189]
	v_pk_add_f32 v[24:25] /*v[280:281]*/, v[212:213], v[222:223]
	v_pk_add_f32 v[26:27] /*v[282:283]*/, v[232:233], v[240:241]
	v_pk_add_f32 v[36:37] /*v[292:293]*/, v[210:211], v[234:235]
	v_pk_add_f32 v[38:39] /*v[294:295]*/, v[248:249], v[254:255]
	v_pk_add_f32 v[42:43] /*v[298:299]*/, v[150:151], v[162:163]
	v_pk_add_f32 v[44:45] /*v[300:301]*/, v[176:177], v[182:183]
	s_set_vgpr_msb 0x4000
	v_exp_f32_e32 v170, v170
	v_exp_f32_e32 v184, v184
	v_exp_f32_e32 v220, v220
	s_set_vgpr_msb 64
	v_pk_add_f32 v[28:29] /*v[284:285]*/, v[252:253], v[128:129]
	v_pk_add_f32 v[30:31] /*v[286:287]*/, v[132:133], v[136:137]
	s_set_vgpr_msb 0x4044
	v_pk_add_f32 v[22:23] /*v[278:279]*/, v[204:205], v[22:23] /*v[278:279]*/
	v_pk_add_f32 v[24:25] /*v[280:281]*/, v[228:229], v[24:25] /*v[280:281]*/
	v_pk_add_f32 v[26:27] /*v[282:283]*/, v[246:247], v[26:27] /*v[282:283]*/
	s_set_vgpr_msb 0x4440
	v_pk_add_f32 v[32:33] /*v[288:289]*/, v[158:159], v[194:195]
	v_pk_add_f32 v[40:41] /*v[296:297]*/, v[138:139], v[142:143]
	s_set_vgpr_msb 0x4044
	v_pk_add_f32 v[36:37] /*v[292:293]*/, v[242:243], v[36:37] /*v[292:293]*/
	v_pk_add_f32 v[38:39] /*v[294:295]*/, v[134:135], v[38:39] /*v[294:295]*/
	s_set_vgpr_msb 0x4440
	v_pk_add_f32 v[46:47] /*v[302:303]*/, v[206:207], v[214:215]
	v_pk_add_f32 v[72:73] /*v[328:329]*/, v[152:153], v[224:225]
	v_pk_add_f32 v[74:75] /*v[330:331]*/, v[144:145], v[148:149]
	s_set_vgpr_msb 0x4044
	v_pk_add_f32 v[42:43] /*v[298:299]*/, v[166:167], v[42:43] /*v[298:299]*/
	v_pk_add_f32 v[44:45] /*v[300:301]*/, v[200:201], v[44:45] /*v[300:301]*/
	s_set_vgpr_msb 0x4445
	v_pk_add_f32 v[18:19] /*v[274:275]*/, v[18:19] /*v[274:275]*/, v[20:21] /*v[276:277]*/
	s_set_vgpr_msb 0x4500
	v_exp_f32_e32 v208, v208
	v_exp_f32_e32 v230, v230
	s_set_vgpr_msb 0x51
	v_pk_fma_f32 v[16:17] /*v[272:273]*/, v[254:255] /*v[510:511]*/, s[52:53], v[16:17] /*v[272:273]*/ op_sel_hi:[1,0,0]
	v_pk_add_f32 v[28:29] /*v[284:285]*/, v[28:29] /*v[284:285]*/, v[130:131]
	v_pk_add_f32 v[30:31] /*v[286:287]*/, v[30:31] /*v[286:287]*/, v[140:141]
	s_set_vgpr_msb 0x5140
	v_pk_add_f32 v[34:35] /*v[290:291]*/, v[174:175], v[186:187]
	s_set_vgpr_msb 0x4044
	v_pk_add_f32 v[32:33] /*v[288:289]*/, v[160:161], v[32:33] /*v[288:289]*/
	v_pk_add_f32 v[40:41] /*v[296:297]*/, v[146:147], v[40:41] /*v[296:297]*/
	v_pk_add_f32 v[46:47] /*v[302:303]*/, v[218:219], v[46:47] /*v[302:303]*/
	v_pk_add_f32 v[72:73] /*v[328:329]*/, v[236:237], v[72:73] /*v[328:329]*/
	v_pk_add_f32 v[74:75] /*v[330:331]*/, v[196:197], v[74:75] /*v[330:331]*/
	s_set_vgpr_msb 0x4440
	v_pk_add_f32 v[76:77] /*v[332:333]*/, v[168:169], v[170:171]
	v_pk_add_f32 v[78:79] /*v[334:335]*/, v[184:185], v[202:203]
	v_pk_add_f32 v[192:193] /*v[448:449]*/, v[216:217], v[220:221]
	s_set_vgpr_msb 0x4045
	v_pk_add_f32 v[18:19] /*v[274:275]*/, v[22:23] /*v[278:279]*/, v[18:19] /*v[274:275]*/
	v_pk_add_f32 v[22:23] /*v[278:279]*/, v[24:25] /*v[280:281]*/, v[26:27] /*v[282:283]*/
	v_pk_add_f32 v[24:25] /*v[280:281]*/, v[36:37] /*v[292:293]*/, v[38:39] /*v[294:295]*/
	v_pk_add_f32 v[26:27] /*v[282:283]*/, v[42:43] /*v[298:299]*/, v[44:45] /*v[300:301]*/
	s_set_vgpr_msb 0x4500
	v_exp_f32_e32 v244, v244
	v_exp_f32_e32 v250, v245
	s_set_vgpr_msb 1
	v_exp_f32_e32 v245, v16 /*v272*/
	s_set_vgpr_msb 0x144
	v_pk_add_f32 v[34:35] /*v[290:291]*/, v[190:191], v[34:35] /*v[290:291]*/
	s_set_vgpr_msb 0x4440
	v_pk_add_f32 v[194:195] /*v[450:451]*/, v[230:231], v[238:239]
	s_set_vgpr_msb 0x4044
	v_pk_add_f32 v[20:21] /*v[276:277]*/, v[178:179], v[76:77] /*v[332:333]*/
	v_pk_add_f32 v[76:77] /*v[332:333]*/, v[208:209], v[78:79] /*v[334:335]*/
	v_pk_add_f32 v[78:79] /*v[334:335]*/, v[226:227], v[192:193] /*v[448:449]*/
	s_set_vgpr_msb 0x4445
	v_pk_add_f32 v[30:31] /*v[286:287]*/, v[30:31] /*v[286:287]*/, v[32:33] /*v[288:289]*/
	v_pk_add_f32 v[32:33] /*v[288:289]*/, v[72:73] /*v[328:329]*/, v[74:75] /*v[330:331]*/
	v_pk_add_f32 v[22:23] /*v[278:279]*/, v[28:29] /*v[284:285]*/, v[22:23] /*v[278:279]*/
	v_pk_add_f32 v[24:25] /*v[280:281]*/, v[40:41] /*v[296:297]*/, v[24:25] /*v[280:281]*/
	v_pk_add_f32 v[26:27] /*v[282:283]*/, v[46:47] /*v[302:303]*/, v[26:27] /*v[282:283]*/
	s_set_vgpr_msb 0x4544
	v_pk_add_f32 v[192:193] /*v[448:449]*/, v[244:245], v[194:195] /*v[450:451]*/
	s_set_vgpr_msb 0x4445
	v_pk_add_f32 v[28:29] /*v[284:285]*/, v[34:35] /*v[290:291]*/, v[30:31] /*v[286:287]*/
	v_pk_add_f32 v[20:21] /*v[276:277]*/, v[20:21] /*v[276:277]*/, v[32:33] /*v[288:289]*/
	v_pk_add_f32 v[30:31] /*v[286:287]*/, v[76:77] /*v[332:333]*/, v[78:79] /*v[334:335]*/
	v_pk_add_f32 v[18:19] /*v[274:275]*/, v[18:19] /*v[274:275]*/, v[22:23] /*v[278:279]*/
	v_pk_add_f32 v[22:23] /*v[278:279]*/, v[24:25] /*v[280:281]*/, v[26:27] /*v[282:283]*/
	s_set_vgpr_msb 0x4501
	v_exp_f32_e32 v251, v17 /*v273*/
	v_nop
	s_set_vgpr_msb 0x145
	v_pk_add_f32 v[16:17] /*v[272:273]*/, v[192:193] /*v[448:449]*/, v[30:31] /*v[286:287]*/
	v_pk_add_f32 v[18:19] /*v[274:275]*/, v[28:29] /*v[284:285]*/, v[18:19] /*v[274:275]*/
	v_pk_add_f32 v[20:21] /*v[276:277]*/, v[20:21] /*v[276:277]*/, v[22:23] /*v[278:279]*/
	s_set_vgpr_msb 0x4544
	v_pk_add_f32 v[16:17] /*v[272:273]*/, v[250:251], v[16:17] /*v[272:273]*/
	s_set_vgpr_msb 0x4445
	v_pk_add_f32 v[18:19] /*v[274:275]*/, v[18:19] /*v[274:275]*/, v[20:21] /*v[276:277]*/
	v_pk_add_f32 v[192:193] /*v[448:449]*/, v[16:17] /*v[272:273]*/, v[18:19] /*v[274:275]*/
	v_dual_sub_f32 v20 /*v276*/, v182 /*v438*/, v180 /*v436*/ :: v_dual_mov_b32 v194 /*v450*/, v192 /*v448*/
	v_dual_mul_f32 v16 /*v272*/, 0x3fb8aa3b, v20 /*v276*/ :: v_dual_mov_b32 v195 /*v451*/, v193 /*v449*/
	v_permlanex16_b32 v194 /*v450*/, v194 /*v450*/, s56, 0xfedcba98
	v_exp_f32_e32 v196 /*v452*/, v16 /*v272*/
	v_permlanex16_b32 v195 /*v451*/, v195 /*v451*/, s56, 0xfedcba98
	s_set_vgpr_msb 0x4500
	s_cbranch_vccz .LBB0_31
	s_set_vgpr_msb 4
	v_pk_mul_f32 v[126:127], v[126:127], v[196:197] /*v[452:453]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[124:125], v[124:125], v[196:197] /*v[452:453]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[122:123], v[122:123], v[196:197] /*v[452:453]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[120:121], v[120:121], v[196:197] /*v[452:453]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[118:119], v[118:119], v[196:197] /*v[452:453]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[116:117], v[116:117], v[196:197] /*v[452:453]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[114:115], v[114:115], v[196:197] /*v[452:453]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[112:113], v[112:113], v[196:197] /*v[452:453]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[110:111], v[110:111], v[196:197] /*v[452:453]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[108:109], v[108:109], v[196:197] /*v[452:453]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[106:107], v[106:107], v[196:197] /*v[452:453]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[104:105], v[104:105], v[196:197] /*v[452:453]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[102:103], v[102:103], v[196:197] /*v[452:453]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[100:101], v[100:101], v[196:197] /*v[452:453]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[98:99], v[98:99], v[196:197] /*v[452:453]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[96:97], v[96:97], v[196:197] /*v[452:453]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[94:95], v[94:95], v[196:197] /*v[452:453]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[92:93], v[92:93], v[196:197] /*v[452:453]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[90:91], v[90:91], v[196:197] /*v[452:453]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[88:89], v[88:89], v[196:197] /*v[452:453]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[86:87], v[86:87], v[196:197] /*v[452:453]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[84:85], v[84:85], v[196:197] /*v[452:453]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[82:83], v[82:83], v[196:197] /*v[452:453]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[80:81], v[80:81], v[196:197] /*v[452:453]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[78:79], v[78:79], v[196:197] /*v[452:453]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[76:77], v[76:77], v[196:197] /*v[452:453]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[74:75], v[74:75], v[196:197] /*v[452:453]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[72:73], v[72:73], v[196:197] /*v[452:453]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[70:71], v[70:71], v[196:197] /*v[452:453]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[68:69], v[68:69], v[196:197] /*v[452:453]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[66:67], v[66:67], v[196:197] /*v[452:453]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[64:65], v[64:65], v[196:197] /*v[452:453]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x400
.LBB0_31:
	s_set_vgpr_msb 0x45
	v_sub_f32_e32 v16 /*v272*/, v181 /*v437*/, v179 /*v435*/
	s_and_b32 s2, s5, exec_lo
	s_cselect_b32 s2, 1, 0
	s_cmp_lg_u32 s2, 1
	v_mul_f32_e32 v16 /*v272*/, 0x3fb8aa3b, v16 /*v272*/
	v_exp_f32_e32 v197 /*v453*/, v16 /*v272*/
	s_set_vgpr_msb 0x4500
	s_cbranch_scc1 .LBB0_26
	v_nop
	s_set_vgpr_msb 0x41
	v_mov_b32_e32 v16 /*v272*/, v197 /*v453*/
	s_set_vgpr_msb 0x4104
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
	s_branch .LBB0_26
.LBB0_33:
	v_dual_mov_b32 v1, v0 :: v_dual_mov_b32 v2, v0
	v_dual_mov_b32 v3, v0 :: v_dual_mov_b32 v4, v0
	v_dual_mov_b32 v5, v0 :: v_dual_mov_b32 v6, v0
	v_dual_mov_b32 v7, v0 :: v_dual_mov_b32 v235, v0
	s_set_vgpr_msb 64
	v_dual_mov_b32 v179 /*v435*/, 0xf149f2ca :: v_dual_mov_b32 v178 /*v434*/, v134
	s_set_vgpr_msb 0x4000
	v_mov_b32_e32 v234, v0
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
	s_set_vgpr_msb 64
	v_mov_b32_e32 v180 /*v436*/, 0xf149f2ca
	s_cmp_ge_u32 s44, s10
	s_set_vgpr_msb 0x4000
	s_cbranch_scc0 .LBB0_36
.LBB0_34:
	s_set_vgpr_msb 1
	v_mov_b32_e32 v236, v179 /*v435*/
	s_set_vgpr_msb 0x141
	v_mov_b32_e32 v55 /*v311*/, v180 /*v436*/
	s_set_vgpr_msb 0x4100
	s_branch .LBB0_45
.LBB0_35:
	s_clause 0x3
	scratch_load_b32 v154, off, off offset:916 nv
	scratch_load_b32 v155, off, off offset:920 nv
	scratch_load_b32 v156, off, off offset:928 nv
	scratch_load_b32 v157, off, off offset:936 nv
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b32 v134, off, off offset:940 nv
	scratch_load_b32 v152, off, off offset:104 nv
	s_cmp_ge_u32 s44, s10
	s_cbranch_scc1 .LBB0_34
.LBB0_36:
	s_wait_loadcnt 0x4
	v_dual_add_nc_u32 v128, s11, v154 :: v_dual_add_nc_u32 v129, s11, v155
	s_add_co_i32 s2, s58, -1
	s_mov_b32 s19, 0
	s_mov_b32 s4, 1
	s_set_vgpr_msb 64
	v_min_i32_e32 v181 /*v437*/, s2, v128
	v_min_i32_e32 v182 /*v438*/, s2, v129
	s_mov_b32 s11, s19
	s_mov_b32 s16, 16
	s_mov_b32 s15, 0x800000
	s_mov_b32 s13, 0xffff0000
	s_mov_b32 s12, 0x7510000
	s_mov_b32 s20, 0xf510000
	s_mov_b32 s35, 0x76543210
	s_mov_b32 s46, 0x3fb8aa3b
	s_set_vgpr_msb 0x4000
	s_branch .LBB0_38
.LBB0_37:
	s_wait_dscnt 0x0
	v_cvt_pk_bf16_f32 v162, v156, v194
	v_cvt_pk_bf16_f32 v160, v144, v146
	v_cvt_pk_bf16_f32 v170, v157, v195
	v_cvt_pk_bf16_f32 v177, v192, v198
	v_cvt_pk_bf16_f32 v185, v193, v199
	v_cvt_pk_bf16_f32 v144, v196, v204
	v_cvt_pk_bf16_f32 v152, v197, v205
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b128 v[192:195], off, off offset:748 th:TH_LOAD_LU nv
	scratch_load_b128 v[196:199], off, off offset:764 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x2
	v_dual_fmac_f32 v131, v130, v234 :: v_dual_fmac_f32 v133, v132, v235
	v_cvt_pk_bf16_f32 v135, v142, v232
	s_set_vgpr_msb 5
	v_cvt_pk_bf16_f32 v134, v252 /*v508*/, v254 /*v510*/
	v_cvt_pk_bf16_f32 v132, v236 /*v492*/, v244 /*v500*/
	s_set_vgpr_msb 0x500
	v_dual_add_f32 v234, v131, v128 :: v_dual_add_f32 v235, v133, v129
	s_set_vgpr_msb 4
	v_cvt_pk_bf16_f32 v133, v140, v250 /*v506*/
	v_cvt_pk_bf16_f32 v131, v138, v228 /*v484*/
	s_set_vgpr_msb 0x405
	v_cvt_pk_bf16_f32 v130, v202 /*v458*/, v212 /*v468*/
	s_set_vgpr_msb 0x504
	v_cvt_pk_bf16_f32 v129, v136, v194 /*v450*/
	s_set_vgpr_msb 0x400
	v_cvt_pk_bf16_f32 v128, v208, v220
	v_cvt_pk_bf16_f32 v143, v143, v233
	s_set_vgpr_msb 5
	v_cvt_pk_bf16_f32 v142, v253 /*v509*/, v255 /*v511*/
	s_set_vgpr_msb 0x504
	v_cvt_pk_bf16_f32 v141, v141, v251 /*v507*/
	s_set_vgpr_msb 0x405
	v_cvt_pk_bf16_f32 v140, v237 /*v493*/, v245 /*v501*/
	s_set_vgpr_msb 0x504
	v_cvt_pk_bf16_f32 v139, v139, v229 /*v485*/
	s_set_vgpr_msb 0x405
	v_cvt_pk_bf16_f32 v138, v203 /*v459*/, v213 /*v469*/
	s_set_vgpr_msb 0x504
	v_cvt_pk_bf16_f32 v137, v137, v195 /*v451*/
	s_set_vgpr_msb 0x400
	v_cvt_pk_bf16_f32 v136, v209, v221
	s_set_vgpr_msb 1
	v_wmma_f32_16x16x32_bf16 v[88:95], v[56:63] /*v[312:319]*/, v[128:135], v[88:95]
	s_set_vgpr_msb 0x105
	v_cvt_pk_bf16_f32 v167, v238 /*v494*/, v246 /*v502*/
	v_cvt_pk_bf16_f32 v166, v224 /*v480*/, v234 /*v490*/
	v_cvt_pk_bf16_f32 v165, v206 /*v462*/, v216 /*v472*/
	s_set_vgpr_msb 0x500
	v_cvt_pk_bf16_f32 v164, v216, v226
	v_cvt_pk_bf16_f32 v163, v200, v210
	v_cvt_pk_bf16_f32 v161, v148, v154
	s_set_vgpr_msb 5
	v_cvt_pk_bf16_f32 v175, v239 /*v495*/, v247 /*v503*/
	s_set_vgpr_msb 0x501
	v_wmma_f32_16x16x32_bf16 v[24:31], v[56:63] /*v[312:319]*/, v[136:143], v[24:31]
	s_set_vgpr_msb 0x105
	v_cvt_pk_bf16_f32 v174, v225 /*v481*/, v235 /*v491*/
	v_cvt_pk_bf16_f32 v173, v207 /*v463*/, v217 /*v473*/
	s_set_vgpr_msb 0x500
	v_cvt_pk_bf16_f32 v172, v217, v227
	v_cvt_pk_bf16_f32 v171, v201, v211
	v_cvt_pk_bf16_f32 v169, v149, v155
	v_cvt_pk_bf16_f32 v168, v145, v147
	s_set_vgpr_msb 5
	v_cvt_pk_bf16_f32 v183, v222 /*v478*/, v232 /*v488*/
	v_cvt_pk_bf16_f32 v182, v210 /*v466*/, v220 /*v476*/
	v_cvt_pk_bf16_f32 v181, v198 /*v454*/, v208 /*v464*/
	s_set_vgpr_msb 0x504
	v_cvt_pk_bf16_f32 v180, v228, v196 /*v452*/
	s_set_vgpr_msb 0x400
	v_cvt_pk_bf16_f32 v179, v214, v224
	v_cvt_pk_bf16_f32 v178, v202, v212
	v_cvt_pk_bf16_f32 v176, v150, v158
	s_set_vgpr_msb 5
	v_cvt_pk_bf16_f32 v191, v223 /*v479*/, v233 /*v489*/
	v_cvt_pk_bf16_f32 v190, v211 /*v467*/, v221 /*v477*/
	v_cvt_pk_bf16_f32 v189, v199 /*v455*/, v209 /*v465*/
	s_set_vgpr_msb 0x504
	v_cvt_pk_bf16_f32 v188, v229, v197 /*v453*/
	s_set_vgpr_msb 0x400
	v_cvt_pk_bf16_f32 v187, v215, v225
	v_cvt_pk_bf16_f32 v186, v203, v213
	v_cvt_pk_bf16_f32 v184, v151, v159
	s_set_vgpr_msb 5
	v_cvt_pk_bf16_f32 v151, v242 /*v498*/, v248 /*v504*/
	v_cvt_pk_bf16_f32 v150, v230 /*v486*/, v240 /*v496*/
	v_cvt_pk_bf16_f32 v149, v218 /*v474*/, v226 /*v482*/
	v_cvt_pk_bf16_f32 v148, v204 /*v460*/, v214 /*v470*/
	v_cvt_pk_bf16_f32 v147, v192 /*v448*/, v200 /*v456*/
	s_set_vgpr_msb 0x500
	v_cvt_pk_bf16_f32 v146, v222, v230
	v_cvt_pk_bf16_f32 v145, v206, v218
	s_set_vgpr_msb 5
	v_cvt_pk_bf16_f32 v159, v243 /*v499*/, v249 /*v505*/
	v_cvt_pk_bf16_f32 v158, v231 /*v487*/, v241 /*v497*/
	v_cvt_pk_bf16_f32 v157, v219 /*v475*/, v227 /*v483*/
	v_cvt_pk_bf16_f32 v156, v205 /*v461*/, v215 /*v471*/
	v_cvt_pk_bf16_f32 v155, v193 /*v449*/, v201 /*v457*/
	s_set_vgpr_msb 0x500
	v_cvt_pk_bf16_f32 v154, v223, v231
	v_cvt_pk_bf16_f32 v153, v207, v219
	s_set_vgpr_msb 1
	v_wmma_f32_16x16x32_bf16 v[120:127], v[184:191] /*v[440:447]*/, v[128:135], v[120:127]
	s_set_vgpr_msb 0x141
	v_dual_mov_b32 v177 /*v433*/, v176 /*v432*/ :: v_dual_mov_b32 v176 /*v432*/, v183 /*v439*/
	v_dual_mov_b32 v183 /*v439*/, v178 /*v434*/ :: v_dual_mov_b32 v180 /*v436*/, v55 /*v311*/
	s_wait_alu depctr_va_vdst(0)
	scratch_load_b32 v178 /*v434*/, off, off offset:268 th:TH_LOAD_LU nv
	s_add_nc_u64 s[44:45], s[44:45], 1
	s_set_vgpr_msb 0x4140
	v_mov_b32_e32 v179 /*v435*/, v236
	s_set_vgpr_msb 0x4001
	v_wmma_f32_16x16x32_bf16 v[112:119], v[152:159] /*v[408:415]*/, v[128:135], v[112:119]
	v_cmp_ge_u64_e64 s2, s[44:45], s[10:11]
	s_and_b32 vcc_lo, exec_lo, s2
	v_wmma_f32_16x16x32_bf16 v[104:111], v[120:127] /*v[376:383]*/, v[128:135], v[104:111]
	v_wmma_f32_16x16x32_bf16 v[96:103], v[88:95] /*v[344:351]*/, v[128:135], v[96:103]
	v_wmma_f32_16x16x32_bf16 v[56:63], v[184:191] /*v[440:447]*/, v[136:143], v[56:63]
	v_wmma_f32_16x16x32_bf16 v[48:55], v[152:159] /*v[408:415]*/, v[136:143], v[48:55]
	v_wmma_f32_16x16x32_bf16 v[40:47], v[120:127] /*v[376:383]*/, v[136:143], v[40:47]
	v_wmma_f32_16x16x32_bf16 v[32:39], v[88:95] /*v[344:351]*/, v[136:143], v[32:39]
	v_wmma_f32_16x16x32_bf16 v[56:63], v[46:53] /*v[302:309]*/, v[168:175], v[56:63]
	v_wmma_f32_16x16x32_bf16 v[48:55], v[144:151] /*v[400:407]*/, v[168:175], v[48:55]
	v_wmma_f32_16x16x32_bf16 v[40:47], v[112:119] /*v[368:375]*/, v[168:175], v[40:47]
	v_wmma_f32_16x16x32_bf16 v[32:39], v[80:87] /*v[336:343]*/, v[168:175], v[32:39]
	v_wmma_f32_16x16x32_bf16 v[56:63], v[168:175] /*v[424:431]*/, v[184:191], v[56:63]
	v_wmma_f32_16x16x32_bf16 v[48:55], v[136:143] /*v[392:399]*/, v[184:191], v[48:55]
	v_wmma_f32_16x16x32_bf16 v[40:47], v[104:111] /*v[360:367]*/, v[184:191], v[40:47]
	v_wmma_f32_16x16x32_bf16 v[32:39], v[72:79] /*v[328:335]*/, v[184:191], v[32:39]
	v_wmma_f32_16x16x32_bf16 v[56:63], v[160:167] /*v[416:423]*/, v[152:159], v[56:63]
	v_wmma_f32_16x16x32_bf16 v[48:55], v[128:135] /*v[384:391]*/, v[152:159], v[48:55]
	v_wmma_f32_16x16x32_bf16 v[40:47], v[96:103] /*v[352:359]*/, v[152:159], v[40:47]
	v_wmma_f32_16x16x32_bf16 v[32:39], v[64:71] /*v[320:327]*/, v[152:159], v[32:39]
	v_wmma_f32_16x16x32_bf16 v[120:127], v[46:53] /*v[302:309]*/, v[160:167], v[120:127]
	v_wmma_f32_16x16x32_bf16 v[112:119], v[144:151] /*v[400:407]*/, v[160:167], v[112:119]
	v_wmma_f32_16x16x32_bf16 v[104:111], v[112:119] /*v[368:375]*/, v[160:167], v[104:111]
	v_wmma_f32_16x16x32_bf16 v[96:103], v[80:87] /*v[336:343]*/, v[160:167], v[96:103]
	v_wmma_f32_16x16x32_bf16 v[120:127], v[168:175] /*v[424:431]*/, v[176:183], v[120:127]
	v_wmma_f32_16x16x32_bf16 v[112:119], v[136:143] /*v[392:399]*/, v[176:183], v[112:119]
	v_wmma_f32_16x16x32_bf16 v[104:111], v[104:111] /*v[360:367]*/, v[176:183], v[104:111]
	v_wmma_f32_16x16x32_bf16 v[96:103], v[72:79] /*v[328:335]*/, v[176:183], v[96:103]
	v_wmma_f32_16x16x32_bf16 v[120:127], v[160:167] /*v[416:423]*/, v[144:151], v[120:127]
	v_wmma_f32_16x16x32_bf16 v[112:119], v[128:135] /*v[384:391]*/, v[144:151], v[112:119]
	v_wmma_f32_16x16x32_bf16 v[104:111], v[96:103] /*v[352:359]*/, v[144:151], v[104:111]
	s_set_vgpr_msb 0x100
	s_wait_loadcnt 0x1
	v_wmma_f32_16x16x32_bf16 v[88:95], v[192:199], v[160:167], v[88:95]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[192:199], v[168:175], v[24:31]
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b128 v[192:195], off, off offset:716 th:TH_LOAD_LU nv
	scratch_load_b128 v[196:199], off, off offset:732 th:TH_LOAD_LU nv
	s_set_vgpr_msb 1
	v_wmma_f32_16x16x32_bf16 v[96:103], v[64:71] /*v[320:327]*/, v[144:151], v[96:103]
	s_set_vgpr_msb 0x100
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[88:95], v[192:199], v[176:183], v[88:95]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[192:199], v[184:191], v[24:31]
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b128 v[192:195], off, off offset:684 th:TH_LOAD_LU nv
	scratch_load_b128 v[196:199], off, off offset:700 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[88:95], v[192:199], v[144:151], v[88:95]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[192:199], v[152:159], v[24:31]
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b128 v[192:195], off, off offset:652 th:TH_LOAD_LU nv
	scratch_load_b128 v[196:199], off, off offset:668 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[80:87], v[192:199], v[128:135], v[80:87]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[192:199], v[136:143], v[16:23]
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b128 v[192:195], off, off offset:620 th:TH_LOAD_LU nv
	scratch_load_b128 v[196:199], off, off offset:636 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[80:87], v[192:199], v[160:167], v[80:87]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[192:199], v[168:175], v[16:23]
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b128 v[192:195], off, off offset:588 th:TH_LOAD_LU nv
	scratch_load_b128 v[196:199], off, off offset:604 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[80:87], v[192:199], v[176:183], v[80:87]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[192:199], v[184:191], v[16:23]
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b128 v[192:195], off, off offset:556 th:TH_LOAD_LU nv
	scratch_load_b128 v[196:199], off, off offset:572 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[80:87], v[192:199], v[144:151], v[80:87]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[192:199], v[152:159], v[16:23]
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b128 v[192:195], off, off offset:524 th:TH_LOAD_LU nv
	scratch_load_b128 v[196:199], off, off offset:540 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[72:79], v[192:199], v[128:135], v[72:79]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[192:199], v[136:143], v[8:15]
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b128 v[192:195], off, off offset:492 th:TH_LOAD_LU nv
	scratch_load_b128 v[196:199], off, off offset:508 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[72:79], v[192:199], v[160:167], v[72:79]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[192:199], v[168:175], v[8:15]
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b128 v[192:195], off, off offset:460 th:TH_LOAD_LU nv
	scratch_load_b128 v[196:199], off, off offset:476 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[72:79], v[192:199], v[176:183], v[72:79]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[192:199], v[184:191], v[8:15]
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b128 v[192:195], off, off offset:428 th:TH_LOAD_LU nv
	scratch_load_b128 v[196:199], off, off offset:444 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[72:79], v[192:199], v[144:151], v[72:79]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[192:199], v[152:159], v[8:15]
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b128 v[192:195], off, off offset:396 th:TH_LOAD_LU nv
	scratch_load_b128 v[196:199], off, off offset:412 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[64:71], v[192:199], v[128:135], v[64:71]
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b128 v[128:131], off, off offset:364 th:TH_LOAD_LU nv
	scratch_load_b128 v[132:135], off, off offset:380 th:TH_LOAD_LU nv
	v_wmma_f32_16x16x32_bf16 v[0:7], v[192:199], v[136:143], v[0:7]
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[64:71], v[128:135], v[160:167], v[64:71]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[128:135], v[168:175], v[0:7]
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b128 v[128:131], off, off offset:300 th:TH_LOAD_LU nv
	scratch_load_b128 v[132:135], off, off offset:316 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[64:71], v[128:135], v[176:183], v[64:71]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[128:135], v[184:191], v[0:7]
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b128 v[128:131], off, off offset:332 th:TH_LOAD_LU nv
	scratch_load_b128 v[132:135], off, off offset:348 th:TH_LOAD_LU nv
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[0:7], v[128:135], v[152:159], v[0:7]
	s_wait_alu depctr_va_vdst(0)
	scratch_load_b32 v152, off, off offset:104 nv
	v_wmma_f32_16x16x32_bf16 v[64:71], v[128:135], v[144:151], v[64:71]
	s_cbranch_vccnz .LBB0_44
.LBB0_38:
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x2
	scratch_store_b64 off, v[234:235], off nv
	s_set_vgpr_msb 0x45
	scratch_store_b32 off, v183 /*v439*/, off offset:268 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	v_mov_b32_e32 v183 /*v439*/, v177 /*v433*/
	s_add_co_i32 s2, s44, 1
	s_wait_tensorcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	ds_load_b128 v[168:171] /*v[424:427]*/, v178 /*v434*/
	ds_load_b128 v[172:175] /*v[428:431]*/, v178 /*v434*/ offset:32
	ds_load_b128 v[160:163] /*v[416:419]*/, v178 /*v434*/ offset:64
	ds_load_b128 v[164:167] /*v[420:423]*/, v178 /*v434*/ offset:96
	ds_load_b128 v[152:155] /*v[408:411]*/, v178 /*v434*/ offset:128
	ds_load_b128 v[156:159] /*v[412:415]*/, v178 /*v434*/ offset:160
	s_set_vgpr_msb 0x4501
	ds_load_b128 v[136:139], v178 /*v434*/ offset:192
	ds_load_b128 v[140:143], v178 /*v434*/ offset:224
	s_set_vgpr_msb 0x141
	ds_load_b128 v[144:147] /*v[400:403]*/, v178 /*v434*/ offset:4352
	ds_load_b128 v[148:151] /*v[404:407]*/, v178 /*v434*/ offset:4384
	ds_load_b128 v[136:139] /*v[392:395]*/, v178 /*v434*/ offset:4416
	ds_load_b128 v[140:143] /*v[396:399]*/, v178 /*v434*/ offset:4448
	ds_load_b128 v[128:131] /*v[384:387]*/, v178 /*v434*/ offset:4480
	ds_load_b128 v[132:135] /*v[388:391]*/, v178 /*v434*/ offset:4512
	s_set_vgpr_msb 0x4101
	ds_load_b128 v[224:227], v178 /*v434*/ offset:4544
	ds_load_b128 v[228:231], v178 /*v434*/ offset:4576
	s_set_vgpr_msb 0x141
	ds_load_b128 v[120:123] /*v[376:379]*/, v178 /*v434*/ offset:8704
	ds_load_b128 v[124:127] /*v[380:383]*/, v178 /*v434*/ offset:8736
	ds_load_b128 v[112:115] /*v[368:371]*/, v178 /*v434*/ offset:8768
	ds_load_b128 v[116:119] /*v[372:375]*/, v178 /*v434*/ offset:8800
	ds_load_b128 v[104:107] /*v[360:363]*/, v178 /*v434*/ offset:8832
	ds_load_b128 v[108:111] /*v[364:367]*/, v178 /*v434*/ offset:8864
	s_set_vgpr_msb 0x4101
	ds_load_b128 v[128:131], v178 /*v434*/ offset:8896
	ds_load_b128 v[132:135], v178 /*v434*/ offset:8928
	s_set_vgpr_msb 0x141
	ds_load_b128 v[96:99] /*v[352:355]*/, v178 /*v434*/ offset:13056
	ds_load_b128 v[100:103] /*v[356:359]*/, v178 /*v434*/ offset:13088
	ds_load_b128 v[88:91] /*v[344:347]*/, v178 /*v434*/ offset:13120
	ds_load_b128 v[92:95] /*v[348:351]*/, v178 /*v434*/ offset:13152
	ds_load_b128 v[80:83] /*v[336:339]*/, v178 /*v434*/ offset:13184
	ds_load_b128 v[84:87] /*v[340:343]*/, v178 /*v434*/ offset:13216
	s_set_vgpr_msb 0x4101
	ds_load_b128 v[216:219], v178 /*v434*/ offset:13248
	ds_load_b128 v[220:223], v178 /*v434*/ offset:13280
	s_set_vgpr_msb 0x141
	ds_load_b128 v[72:75] /*v[328:331]*/, v178 /*v434*/ offset:17408
	ds_load_b128 v[76:79] /*v[332:335]*/, v178 /*v434*/ offset:17440
	ds_load_b128 v[64:67] /*v[320:323]*/, v178 /*v434*/ offset:17472
	ds_load_b128 v[68:71] /*v[324:327]*/, v178 /*v434*/ offset:17504
	ds_load_b128 v[56:59] /*v[312:315]*/, v178 /*v434*/ offset:17536
	ds_load_b128 v[60:63] /*v[316:319]*/, v178 /*v434*/ offset:17568
	s_set_vgpr_msb 0x4101
	ds_load_b128 v[208:211], v178 /*v434*/ offset:17600
	ds_load_b128 v[212:215], v178 /*v434*/ offset:17632
	s_set_vgpr_msb 0x141
	ds_load_b128 v[48:51] /*v[304:307]*/, v178 /*v434*/ offset:21760
	ds_load_b128 v[52:55] /*v[308:311]*/, v178 /*v434*/ offset:21792
	ds_load_b128 v[40:43] /*v[296:299]*/, v178 /*v434*/ offset:21824
	ds_load_b128 v[44:47] /*v[300:303]*/, v178 /*v434*/ offset:21856
	ds_load_b128 v[32:35] /*v[288:291]*/, v178 /*v434*/ offset:21888
	ds_load_b128 v[36:39] /*v[292:295]*/, v178 /*v434*/ offset:21920
	s_set_vgpr_msb 0x4101
	ds_load_b128 v[200:203], v178 /*v434*/ offset:21952
	ds_load_b128 v[204:207], v178 /*v434*/ offset:21984
	s_set_vgpr_msb 0x141
	ds_load_b128 v[24:27] /*v[280:283]*/, v178 /*v434*/ offset:26112
	ds_load_b128 v[28:31] /*v[284:287]*/, v178 /*v434*/ offset:26144
	ds_load_b128 v[16:19] /*v[272:275]*/, v178 /*v434*/ offset:26176
	ds_load_b128 v[20:23] /*v[276:279]*/, v178 /*v434*/ offset:26208
	ds_load_b128 v[8:11] /*v[264:267]*/, v178 /*v434*/ offset:26240
	ds_load_b128 v[12:15] /*v[268:271]*/, v178 /*v434*/ offset:26272
	s_set_vgpr_msb 0x4101
	ds_load_b128 v[192:195], v178 /*v434*/ offset:26304
	ds_load_b128 v[196:199], v178 /*v434*/ offset:26336
	s_set_vgpr_msb 0x141
	ds_load_b128 v[0:3] /*v[256:259]*/, v178 /*v434*/ offset:30464
	ds_load_b128 v[4:7] /*v[260:263]*/, v178 /*v434*/ offset:30496
	s_set_vgpr_msb 0x4101
	ds_load_b128 v[248:251], v178 /*v434*/ offset:30528
	ds_load_b128 v[252:255], v178 /*v434*/ offset:30560
	ds_load_b128 v[240:243], v178 /*v434*/ offset:30592
	ds_load_b128 v[244:247], v178 /*v434*/ offset:30624
	ds_load_b128 v[232:235], v178 /*v434*/ offset:30656
	ds_load_b128 v[236:239], v178 /*v434*/ offset:30688
	s_cmp_ge_i32 s2, s10
	s_set_vgpr_msb 0x100
	s_cbranch_scc1 .LBB0_40
	s_lshl_b32 s5, s2, 7
	s_lshl_b32 s2, s2, 17
	s_sub_co_i32 s7, s9, s5
	s_add_co_i32 s6, s5, s64
	v_nop
	v_nop
	v_nop
	v_med3_i32 v144, s7, 0, 0x80
	s_ashr_i32 s7, s6, 31
	s_and_b32 s2, s2, 0x20000
	s_mul_u64 s[22:23], s[6:7], s[62:63]
	s_mul_u64 s[6:7], s[6:7], s[60:61]
	v_readfirstlane_b32 s14, v144
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
.LBB0_40:
	s_clause 0xb
	scratch_load_b128 v[154:157], off, off offset:8 nv
	scratch_load_b128 v[158:161], off, off offset:24 nv
	s_set_vgpr_msb 64
	scratch_load_b128 v[184:187] /*v[440:443]*/, off, off offset:140 nv
	scratch_load_b128 v[188:191] /*v[444:447]*/, off, off offset:156 nv
	s_set_vgpr_msb 0x4001
	scratch_load_b128 v[162:165], off, off offset:40 nv
	scratch_load_b128 v[166:169], off, off offset:56 nv
	scratch_load_b128 v[170:173], off, off offset:72 nv
	scratch_load_b128 v[174:177], off, off offset:88 nv
	scratch_load_b128 v[178:181], off, off offset:108 nv
	scratch_load_b128 v[182:185], off, off offset:124 nv
	s_wait_loadcnt_dscnt 0x83e
	v_wmma_f32_16x16x32_bf16 v[144:151], v[168:175] /*v[424:431]*/, v[154:161], 0
	s_set_vgpr_msb 0x145
	s_wait_loadcnt 0x6
	v_wmma_f32_16x16x32_bf16 v[192:199] /*v[448:455]*/, v[168:175] /*v[424:431]*/, v[184:191] /*v[440:447]*/, 0
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b128 v[168:171] /*v[424:427]*/, off, off offset:172 nv
	scratch_load_b128 v[172:175] /*v[428:431]*/, off, off offset:188 nv
	s_set_vgpr_msb 0x4501
	s_wait_loadcnt_dscnt 0x63c
	v_wmma_f32_16x16x32_bf16 v[144:151], v[160:167] /*v[416:423]*/, v[162:169], v[144:151]
	s_wait_loadcnt_dscnt 0x43a
	v_wmma_f32_16x16x32_bf16 v[144:151], v[152:159] /*v[408:415]*/, v[170:177], v[144:151]
	s_set_vgpr_msb 0x100
	s_wait_loadcnt_dscnt 0x238
	v_wmma_f32_16x16x32_bf16 v[144:151], v[136:143], v[178:185], v[144:151]
	s_set_vgpr_msb 0x55
	s_wait_dscnt 0x36
	v_wmma_f32_16x16x32_bf16 v[200:207] /*v[456:463]*/, v[144:151] /*v[400:407]*/, v[184:191] /*v[440:447]*/, 0
	s_wait_dscnt 0x2e
	v_wmma_f32_16x16x32_bf16 v[208:215] /*v[464:471]*/, v[120:127] /*v[376:383]*/, v[184:191] /*v[440:447]*/, 0
	s_wait_dscnt 0x26
	v_wmma_f32_16x16x32_bf16 v[216:223] /*v[472:479]*/, v[96:103] /*v[352:359]*/, v[184:191] /*v[440:447]*/, 0
	s_wait_dscnt 0x1e
	v_wmma_f32_16x16x32_bf16 v[224:231] /*v[480:487]*/, v[72:79] /*v[328:335]*/, v[184:191] /*v[440:447]*/, 0
	s_wait_dscnt 0x16
	v_wmma_f32_16x16x32_bf16 v[232:239] /*v[488:495]*/, v[48:55] /*v[304:311]*/, v[184:191] /*v[440:447]*/, 0
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x32_bf16 v[240:247] /*v[496:503]*/, v[24:31] /*v[280:287]*/, v[184:191] /*v[440:447]*/, 0
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x32_bf16 v[248:255] /*v[504:511]*/, v[0:7] /*v[256:263]*/, v[184:191] /*v[440:447]*/, 0
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[192:199] /*v[448:455]*/, v[160:167] /*v[416:423]*/, v[168:175] /*v[424:431]*/, v[192:199] /*v[448:455]*/
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b128 v[160:163] /*v[416:419]*/, off, off offset:204 nv
	scratch_load_b128 v[164:167] /*v[420:423]*/, off, off offset:220 nv
	v_wmma_f32_16x16x32_bf16 v[200:207] /*v[456:463]*/, v[136:143] /*v[392:399]*/, v[168:175] /*v[424:431]*/, v[200:207] /*v[456:463]*/
	v_wmma_f32_16x16x32_bf16 v[208:215] /*v[464:471]*/, v[112:119] /*v[368:375]*/, v[168:175] /*v[424:431]*/, v[208:215] /*v[464:471]*/
	v_wmma_f32_16x16x32_bf16 v[216:223] /*v[472:479]*/, v[88:95] /*v[344:351]*/, v[168:175] /*v[424:431]*/, v[216:223] /*v[472:479]*/
	v_wmma_f32_16x16x32_bf16 v[224:231] /*v[480:487]*/, v[64:71] /*v[320:327]*/, v[168:175] /*v[424:431]*/, v[224:231] /*v[480:487]*/
	v_wmma_f32_16x16x32_bf16 v[232:239] /*v[488:495]*/, v[40:47] /*v[296:303]*/, v[168:175] /*v[424:431]*/, v[232:239] /*v[488:495]*/
	v_wmma_f32_16x16x32_bf16 v[240:247] /*v[496:503]*/, v[16:23] /*v[272:279]*/, v[168:175] /*v[424:431]*/, v[240:247] /*v[496:503]*/
	s_set_vgpr_msb 0x5554
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x32_bf16 v[248:255] /*v[504:511]*/, v[248:255], v[168:175] /*v[424:431]*/, v[248:255] /*v[504:511]*/
	s_set_vgpr_msb 0x5455
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[192:199] /*v[448:455]*/, v[152:159] /*v[408:415]*/, v[160:167] /*v[416:423]*/, v[192:199] /*v[448:455]*/
	s_wait_alu depctr_va_vdst(0)
	s_clause 0x1
	scratch_load_b128 v[152:155] /*v[408:411]*/, off, off offset:236 nv
	scratch_load_b128 v[156:159] /*v[412:415]*/, off, off offset:252 nv
	v_wmma_f32_16x16x32_bf16 v[200:207] /*v[456:463]*/, v[128:135] /*v[384:391]*/, v[160:167] /*v[416:423]*/, v[200:207] /*v[456:463]*/
	v_wmma_f32_16x16x32_bf16 v[208:215] /*v[464:471]*/, v[104:111] /*v[360:367]*/, v[160:167] /*v[416:423]*/, v[208:215] /*v[464:471]*/
	v_wmma_f32_16x16x32_bf16 v[216:223] /*v[472:479]*/, v[80:87] /*v[336:343]*/, v[160:167] /*v[416:423]*/, v[216:223] /*v[472:479]*/
	v_wmma_f32_16x16x32_bf16 v[224:231] /*v[480:487]*/, v[56:63] /*v[312:319]*/, v[160:167] /*v[416:423]*/, v[224:231] /*v[480:487]*/
	v_wmma_f32_16x16x32_bf16 v[232:239] /*v[488:495]*/, v[32:39] /*v[288:295]*/, v[160:167] /*v[416:423]*/, v[232:239] /*v[488:495]*/
	v_wmma_f32_16x16x32_bf16 v[240:247] /*v[496:503]*/, v[8:15] /*v[264:271]*/, v[160:167] /*v[416:423]*/, v[240:247] /*v[496:503]*/
	s_set_vgpr_msb 0x5554
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[248:255] /*v[504:511]*/, v[240:247], v[160:167] /*v[416:423]*/, v[248:255] /*v[504:511]*/
	s_wait_loadcnt 0x0
	v_wmma_f32_16x16x32_bf16 v[192:199] /*v[448:455]*/, v[136:143], v[152:159] /*v[408:415]*/, v[192:199] /*v[448:455]*/
	s_set_vgpr_msb 0x5401
	v_wmma_f32_16x16x32_bf16 v[136:143], v[144:151] /*v[400:407]*/, v[154:161], 0
	v_wmma_f32_16x16x32_bf16 v[136:143], v[136:143] /*v[392:399]*/, v[162:169], v[136:143]
	v_wmma_f32_16x16x32_bf16 v[136:143], v[128:135] /*v[384:391]*/, v[170:177], v[136:143]
	s_set_vgpr_msb 0x100
	v_wmma_f32_16x16x32_bf16 v[136:143], v[224:231], v[178:185], v[136:143]
	s_set_vgpr_msb 0x54
	v_wmma_f32_16x16x32_bf16 v[200:207] /*v[456:463]*/, v[224:231], v[152:159] /*v[408:415]*/, v[200:207] /*v[456:463]*/
	s_set_vgpr_msb 0x5401
	v_wmma_f32_16x16x32_bf16 v[224:231], v[120:127] /*v[376:383]*/, v[154:161], 0
	v_wmma_f32_16x16x32_bf16 v[224:231], v[112:119] /*v[368:375]*/, v[162:169], v[224:231]
	v_wmma_f32_16x16x32_bf16 v[224:231], v[104:111] /*v[360:367]*/, v[170:177], v[224:231]
	s_set_vgpr_msb 0x100
	v_wmma_f32_16x16x32_bf16 v[224:231], v[128:135], v[178:185], v[224:231]
	s_set_vgpr_msb 0x54
	v_wmma_f32_16x16x32_bf16 v[208:215] /*v[464:471]*/, v[128:135], v[152:159] /*v[408:415]*/, v[208:215] /*v[464:471]*/
	s_set_vgpr_msb 0x5401
	v_wmma_f32_16x16x32_bf16 v[128:135], v[96:103] /*v[352:359]*/, v[154:161], 0
	v_wmma_f32_16x16x32_bf16 v[128:135], v[88:95] /*v[344:351]*/, v[162:169], v[128:135]
	v_wmma_f32_16x16x32_bf16 v[128:135], v[80:87] /*v[336:343]*/, v[170:177], v[128:135]
	s_set_vgpr_msb 0x100
	v_wmma_f32_16x16x32_bf16 v[128:135], v[216:223], v[178:185], v[128:135]
	s_set_vgpr_msb 0x54
	v_wmma_f32_16x16x32_bf16 v[216:223] /*v[472:479]*/, v[216:223], v[152:159] /*v[408:415]*/, v[216:223] /*v[472:479]*/
	s_set_vgpr_msb 0x5401
	v_wmma_f32_16x16x32_bf16 v[216:223], v[72:79] /*v[328:335]*/, v[154:161], 0
	v_wmma_f32_16x16x32_bf16 v[216:223], v[64:71] /*v[320:327]*/, v[162:169], v[216:223]
	v_wmma_f32_16x16x32_bf16 v[216:223], v[56:63] /*v[312:319]*/, v[170:177], v[216:223]
	s_set_vgpr_msb 0x100
	v_wmma_f32_16x16x32_bf16 v[216:223], v[208:215], v[178:185], v[216:223]
	s_set_vgpr_msb 0x54
	v_wmma_f32_16x16x32_bf16 v[224:231] /*v[480:487]*/, v[208:215], v[152:159] /*v[408:415]*/, v[224:231] /*v[480:487]*/
	s_set_vgpr_msb 0x5401
	v_wmma_f32_16x16x32_bf16 v[208:215], v[48:55] /*v[304:311]*/, v[154:161], 0
	v_wmma_f32_16x16x32_bf16 v[208:215], v[40:47] /*v[296:303]*/, v[162:169], v[208:215]
	v_wmma_f32_16x16x32_bf16 v[208:215], v[32:39] /*v[288:295]*/, v[170:177], v[208:215]
	s_set_vgpr_msb 0x100
	v_wmma_f32_16x16x32_bf16 v[208:215], v[200:207], v[178:185], v[208:215]
	s_set_vgpr_msb 0x54
	v_wmma_f32_16x16x32_bf16 v[232:239] /*v[488:495]*/, v[200:207], v[152:159] /*v[408:415]*/, v[232:239] /*v[488:495]*/
	s_set_vgpr_msb 0x5401
	v_wmma_f32_16x16x32_bf16 v[200:207], v[24:31] /*v[280:287]*/, v[154:161], 0
	v_wmma_f32_16x16x32_bf16 v[200:207], v[16:23] /*v[272:279]*/, v[162:169], v[200:207]
	v_wmma_f32_16x16x32_bf16 v[200:207], v[8:15] /*v[264:271]*/, v[170:177], v[200:207]
	s_set_vgpr_msb 0x100
	v_wmma_f32_16x16x32_bf16 v[200:207], v[192:199], v[178:185], v[200:207]
	s_set_vgpr_msb 0x54
	v_wmma_f32_16x16x32_bf16 v[240:247] /*v[496:503]*/, v[192:199], v[152:159] /*v[408:415]*/, v[240:247] /*v[496:503]*/
	s_set_vgpr_msb 0x5401
	v_wmma_f32_16x16x32_bf16 v[192:199], v[0:7] /*v[256:263]*/, v[154:161], 0
	s_set_vgpr_msb 0x100
	v_wmma_f32_16x16x32_bf16 v[192:199], v[248:255], v[162:169], v[192:199]
	v_wmma_f32_16x16x32_bf16 v[192:199], v[240:247], v[170:177], v[192:199]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[192:199], v[232:239], v[178:185], v[192:199]
	s_set_vgpr_msb 0x54
	v_wmma_f32_16x16x32_bf16 v[248:255] /*v[504:511]*/, v[232:239], v[152:159] /*v[408:415]*/, v[248:255] /*v[504:511]*/
	s_set_vgpr_msb 0x5441
	ds_load_tr16_b128 v[184:187] /*v[440:443]*/, v176 /*v432*/
	s_wait_alu depctr_va_vdst(0)
	ds_load_tr16_b128 v[152:155] /*v[408:411]*/, v176 /*v432*/ offset:32
	ds_load_tr16_b128 v[188:191] /*v[444:447]*/, v176 /*v432*/ offset:4608
	ds_load_tr16_b128 v[156:159] /*v[412:415]*/, v176 /*v432*/ offset:4640
	ds_load_tr16_b128 v[46:49] /*v[302:305]*/, v176 /*v432*/ offset:9216
	ds_load_tr16_b128 v[144:147] /*v[400:403]*/, v176 /*v432*/ offset:9248
	ds_load_tr16_b128 v[50:53] /*v[306:309]*/, v176 /*v432*/ offset:13824
	ds_load_tr16_b128 v[148:151] /*v[404:407]*/, v176 /*v432*/ offset:13856
	ds_load_tr16_b128 v[168:171] /*v[424:427]*/, v176 /*v432*/ offset:18432
	ds_load_tr16_b128 v[136:139] /*v[392:395]*/, v176 /*v432*/ offset:18464
	ds_load_tr16_b128 v[172:175] /*v[428:431]*/, v176 /*v432*/ offset:23040
	ds_load_tr16_b128 v[140:143] /*v[396:399]*/, v176 /*v432*/ offset:23072
	ds_load_tr16_b128 v[160:163] /*v[416:419]*/, v176 /*v432*/ offset:27648
	ds_load_tr16_b128 v[128:131] /*v[384:387]*/, v176 /*v432*/ offset:27680
	ds_load_tr16_b128 v[164:167] /*v[420:423]*/, v176 /*v432*/ offset:32256
	ds_load_tr16_b128 v[132:135] /*v[388:391]*/, v176 /*v432*/ offset:32288
	ds_load_tr16_b128 v[120:123] /*v[376:379]*/, v176 /*v432*/ offset:64
	ds_load_tr16_b128 v[88:91] /*v[344:347]*/, v176 /*v432*/ offset:96
	ds_load_tr16_b128 v[124:127] /*v[380:383]*/, v176 /*v432*/ offset:4672
	ds_load_tr16_b128 v[92:95] /*v[348:351]*/, v176 /*v432*/ offset:4704
	ds_load_tr16_b128 v[112:115] /*v[368:371]*/, v176 /*v432*/ offset:9280
	ds_load_tr16_b128 v[80:83] /*v[336:339]*/, v176 /*v432*/ offset:9312
	ds_load_tr16_b128 v[116:119] /*v[372:375]*/, v176 /*v432*/ offset:13888
	ds_load_tr16_b128 v[84:87] /*v[340:343]*/, v176 /*v432*/ offset:13920
	ds_load_tr16_b128 v[104:107] /*v[360:363]*/, v176 /*v432*/ offset:18496
	ds_load_tr16_b128 v[72:75] /*v[328:331]*/, v176 /*v432*/ offset:18528
	ds_load_tr16_b128 v[108:111] /*v[364:367]*/, v176 /*v432*/ offset:23104
	ds_load_tr16_b128 v[76:79] /*v[332:335]*/, v176 /*v432*/ offset:23136
	ds_load_tr16_b128 v[96:99] /*v[352:355]*/, v176 /*v432*/ offset:27712
	ds_load_tr16_b128 v[64:67] /*v[320:323]*/, v176 /*v432*/ offset:27744
	ds_load_tr16_b128 v[100:103] /*v[356:359]*/, v176 /*v432*/ offset:32320
	ds_load_tr16_b128 v[68:71] /*v[324:327]*/, v176 /*v432*/ offset:32352
	ds_load_tr16_b128 v[56:59] /*v[312:315]*/, v176 /*v432*/ offset:128
	s_set_vgpr_msb 0x4101
	ds_load_tr16_b128 v[154:157], v176 /*v432*/ offset:160
	s_set_vgpr_msb 0x141
	ds_load_tr16_b128 v[60:63] /*v[316:319]*/, v176 /*v432*/ offset:4736
	s_set_vgpr_msb 0x4101
	ds_load_tr16_b128 v[158:161], v176 /*v432*/ offset:4768
	s_wait_dscnt 0x2
	scratch_store_b128 off, v[154:157], off offset:652 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[158:161], off offset:668 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_tr16_b128 v[158:161], v176 /*v432*/ offset:9344
	ds_load_tr16_b128 v[154:157], v176 /*v432*/ offset:9376
	ds_load_tr16_b128 v[162:165], v176 /*v432*/ offset:13952
	s_wait_dscnt 0x2
	scratch_store_b128 off, v[158:161], off offset:748 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[162:165], off offset:764 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_tr16_b128 v[158:161], v176 /*v432*/ offset:13984
	scratch_store_b128 off, v[154:157], off offset:620 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[158:161], off offset:636 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_tr16_b128 v[158:161], v176 /*v432*/ offset:18560
	ds_load_tr16_b128 v[154:157], v176 /*v432*/ offset:18592
	ds_load_tr16_b128 v[162:165], v176 /*v432*/ offset:23168
	s_wait_dscnt 0x2
	scratch_store_b128 off, v[158:161], off offset:716 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[162:165], off offset:732 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_tr16_b128 v[158:161], v176 /*v432*/ offset:23200
	scratch_store_b128 off, v[154:157], off offset:588 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[158:161], off offset:604 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_tr16_b128 v[158:161], v176 /*v432*/ offset:27776
	ds_load_tr16_b128 v[154:157], v176 /*v432*/ offset:27808
	ds_load_tr16_b128 v[162:165], v176 /*v432*/ offset:32384
	s_wait_dscnt 0x2
	scratch_store_b128 off, v[158:161], off offset:684 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[162:165], off offset:700 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_tr16_b128 v[158:161], v176 /*v432*/ offset:32416
	scratch_store_b128 off, v[154:157], off offset:556 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[158:161], off offset:572 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_tr16_b128 v[158:161], v176 /*v432*/ offset:192
	ds_load_tr16_b128 v[154:157], v176 /*v432*/ offset:224
	ds_load_tr16_b128 v[162:165], v176 /*v432*/ offset:4800
	s_wait_dscnt 0x2
	scratch_store_b128 off, v[158:161], off offset:524 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[162:165], off offset:540 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_tr16_b128 v[158:161], v176 /*v432*/ offset:4832
	scratch_store_b128 off, v[154:157], off offset:396 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[158:161], off offset:412 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_tr16_b128 v[158:161], v176 /*v432*/ offset:9408
	ds_load_tr16_b128 v[154:157], v176 /*v432*/ offset:9440
	ds_load_tr16_b128 v[162:165], v176 /*v432*/ offset:14016
	s_wait_dscnt 0x2
	scratch_store_b128 off, v[158:161], off offset:492 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[162:165], off offset:508 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_tr16_b128 v[158:161], v176 /*v432*/ offset:14048
	scratch_store_b128 off, v[154:157], off offset:364 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[158:161], off offset:380 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_tr16_b128 v[158:161], v176 /*v432*/ offset:18624
	ds_load_tr16_b128 v[154:157], v176 /*v432*/ offset:18656
	ds_load_tr16_b128 v[162:165], v176 /*v432*/ offset:23232
	s_wait_dscnt 0x2
	scratch_store_b128 off, v[158:161], off offset:460 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[162:165], off offset:476 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_tr16_b128 v[158:161], v176 /*v432*/ offset:23264
	scratch_store_b128 off, v[154:157], off offset:300 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[158:161], off offset:316 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_tr16_b128 v[158:161], v176 /*v432*/ offset:27840
	ds_load_tr16_b128 v[154:157], v176 /*v432*/ offset:27872
	ds_load_tr16_b128 v[162:165], v176 /*v432*/ offset:32448
	s_wait_dscnt 0x2
	scratch_store_b128 off, v[158:161], off offset:428 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[162:165], off offset:444 nv
	s_wait_xcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_tr16_b128 v[158:161], v176 /*v432*/ offset:32480
	scratch_store_b128 off, v[154:157], off offset:332 nv
	s_wait_dscnt 0x0
	scratch_store_b128 off, v[158:161], off offset:348 nv
	v_nop
	v_nop
	v_nop
	v_nop
	v_lshl_or_b32 v232, s44, 7, v152
	v_cmp_ge_i32_e32 vcc_lo, v181 /*v437*/, v232
	v_dual_add_nc_u32 v247, 17, v232 :: v_dual_bitop2_b32 v233, 2, v232 bitop3:0x54
	v_dual_add_nc_u32 v248, 18, v232 :: v_dual_bitop2_b32 v237, 3, v232 bitop3:0x54
	v_cndmask_b32_e32 v144, 0xff800000, v144, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v181 /*v437*/, v232
	v_or_b32_e32 v240, 4, v232
	v_or_b32_e32 v241, 5, v232
	v_or_b32_e32 v242, 6, v232
	v_or_b32_e32 v243, 7, v232
	v_cndmask_b32_e32 v145, 0xff800000, v145, vcc_lo
	v_cmp_ge_i32_e32 vcc_lo, v181 /*v437*/, v233
	v_dual_add_nc_u32 v249, 23, v232 :: v_dual_bitop2_b32 v246, 16, v232 bitop3:0x54
	v_or_b32_e32 v250, 32, v232
	v_or_b32_e32 v251, 33, v232
	v_cndmask_b32_e32 v146, 0xff800000, v146, vcc_lo
	v_cmp_ge_i32_e32 vcc_lo, v181 /*v437*/, v237
	v_or_b32_e32 v254, 34, v232
	s_set_vgpr_msb 0x140
	v_dual_add_nc_u32 v1 /*v257*/, 51, v232 :: v_dual_add_nc_u32 v2 /*v258*/, 52, v232
	v_dual_add_nc_u32 v3 /*v259*/, 53, v232 :: v_dual_add_nc_u32 v10 /*v266*/, 54, v232
	s_set_vgpr_msb 0x4000
	v_cndmask_b32_e32 v147, 0xff800000, v147, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v240, v181 /*v437*/
	s_set_vgpr_msb 0x440
	v_dual_add_nc_u32 v11 /*v267*/, 55, v232 :: v_dual_bitop2_b32 v12 /*v268*/, 64, v232 bitop3:0x54
	v_or_b32_e32 v13 /*v269*/, 0x41, v232
	v_or_b32_e32 v14 /*v270*/, 0x42, v232
	s_set_vgpr_msb 0x4000
	v_cndmask_b32_e32 v148, 0xff800000, v148, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v241, v181 /*v437*/
	s_set_vgpr_msb 0x400
	v_cndmask_b32_e32 v149, 0xff800000, v149, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v242, v181 /*v437*/
	s_set_vgpr_msb 0x400
	v_cndmask_b32_e32 v150, 0xff800000, v150, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v243, v181 /*v437*/
	s_set_vgpr_msb 0x400
	v_cndmask_b32_e32 v151, 0xff800000, v151, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v246, v181 /*v437*/
	s_set_vgpr_msb 0x400
	v_cndmask_b32_e32 v152, 0xff800000, v136, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v247, v181 /*v437*/
	s_set_vgpr_msb 0x400
	v_dual_cndmask_b32 v153, 0xff800000, v137 :: v_dual_add_nc_u32 v136, 19, v232
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v248, v181 /*v437*/
	s_wait_alu depctr_vm_vsrc(0)
	s_set_vgpr_msb 0x400
	v_dual_cndmask_b32 v154, 0xff800000, v138 :: v_dual_add_nc_u32 v137, 20, v232
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v136, v181 /*v437*/
	s_set_vgpr_msb 0x400
	v_dual_cndmask_b32 v155, 0xff800000, v139 :: v_dual_add_nc_u32 v138, 21, v232
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v137, v181 /*v437*/
	s_set_vgpr_msb 0x400
	v_add_nc_u32_e32 v139, 22, v232
	v_cndmask_b32_e32 v140, 0xff800000, v140, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v138, v181 /*v437*/
	s_set_vgpr_msb 0x400
	v_cndmask_b32_e32 v141, 0xff800000, v141, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v139, v181 /*v437*/
	s_set_vgpr_msb 0x400
	v_cndmask_b32_e32 v142, 0xff800000, v142, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v249, v181 /*v437*/
	s_set_vgpr_msb 0x400
	v_cndmask_b32_e32 v143, 0xff800000, v143, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v250, v181 /*v437*/
	s_set_vgpr_msb 0x400
	v_cndmask_b32_e32 v156, 0xff800000, v224, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v251, v181 /*v437*/
	s_set_vgpr_msb 0x400
	v_or_b32_e32 v224, 35, v232
	v_cndmask_b32_e32 v157, 0xff800000, v225, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v254, v181 /*v437*/
	s_set_vgpr_msb 0x400
	v_or_b32_e32 v225, 36, v232
	v_cndmask_b32_e32 v158, 0xff800000, v226, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v224, v181 /*v437*/
	s_set_vgpr_msb 0x400
	v_or_b32_e32 v226, 37, v232
	v_cndmask_b32_e32 v159, 0xff800000, v227, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v225, v181 /*v437*/
	s_set_vgpr_msb 0x400
	v_or_b32_e32 v227, 38, v232
	v_cndmask_b32_e32 v160, 0xff800000, v228, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v226, v181 /*v437*/
	s_set_vgpr_msb 0x400
	v_or_b32_e32 v228, 39, v232
	v_cndmask_b32_e32 v161, 0xff800000, v229, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v227, v181 /*v437*/
	s_set_vgpr_msb 0x400
	v_or_b32_e32 v229, 48, v232
	v_cndmask_b32_e32 v162, 0xff800000, v230, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v228, v181 /*v437*/
	s_set_vgpr_msb 0x400
	v_dual_cndmask_b32 v163, 0xff800000, v231 :: v_dual_add_nc_u32 v230, 49, v232
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v229, v181 /*v437*/
	s_set_vgpr_msb 0x400
	v_add_nc_u32_e32 v231, 50, v232
	v_cndmask_b32_e32 v128, 0xff800000, v128, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v230, v181 /*v437*/
	s_set_vgpr_msb 0x400
	v_cndmask_b32_e32 v129, 0xff800000, v129, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v231, v181 /*v437*/
	s_set_vgpr_msb 0x400
	v_cndmask_b32_e32 v130, 0xff800000, v130, vcc_lo
	s_set_vgpr_msb 5
	v_cmp_le_i32_e32 vcc_lo, v1 /*v257*/, v181 /*v437*/
	s_set_vgpr_msb 0x500
	v_cndmask_b32_e32 v131, 0xff800000, v131, vcc_lo
	s_set_vgpr_msb 5
	v_cmp_le_i32_e32 vcc_lo, v2 /*v258*/, v181 /*v437*/
	s_set_vgpr_msb 0x500
	v_cndmask_b32_e32 v132, 0xff800000, v132, vcc_lo
	s_set_vgpr_msb 5
	v_cmp_le_i32_e32 vcc_lo, v3 /*v259*/, v181 /*v437*/
	s_set_vgpr_msb 0x500
	v_cndmask_b32_e32 v133, 0xff800000, v133, vcc_lo
	s_set_vgpr_msb 5
	v_cmp_le_i32_e32 vcc_lo, v10 /*v266*/, v181 /*v437*/
	s_set_vgpr_msb 0x500
	v_cndmask_b32_e32 v134, 0xff800000, v134, vcc_lo
	s_set_vgpr_msb 5
	v_cmp_le_i32_e32 vcc_lo, v11 /*v267*/, v181 /*v437*/
	s_set_vgpr_msb 0x500
	v_cndmask_b32_e32 v135, 0xff800000, v135, vcc_lo
	s_set_vgpr_msb 5
	v_cmp_le_i32_e32 vcc_lo, v12 /*v268*/, v181 /*v437*/
	s_set_vgpr_msb 0x500
	v_cndmask_b32_e32 v164, 0xff800000, v216, vcc_lo
	s_set_vgpr_msb 5
	v_cmp_le_i32_e32 vcc_lo, v13 /*v269*/, v181 /*v437*/
	s_set_vgpr_msb 0x500
	v_or_b32_e32 v216, 0x43, v232
	v_cndmask_b32_e32 v165, 0xff800000, v217, vcc_lo
	s_set_vgpr_msb 5
	v_cmp_le_i32_e32 vcc_lo, v14 /*v270*/, v181 /*v437*/
	s_set_vgpr_msb 0x500
	v_or_b32_e32 v217, 0x44, v232
	v_cndmask_b32_e32 v166, 0xff800000, v218, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v216, v181 /*v437*/
	s_set_vgpr_msb 0x400
	v_or_b32_e32 v218, 0x45, v232
	v_cndmask_b32_e32 v167, 0xff800000, v219, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v217, v181 /*v437*/
	s_set_vgpr_msb 0x400
	v_or_b32_e32 v219, 0x46, v232
	v_cndmask_b32_e32 v168, 0xff800000, v220, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v218, v181 /*v437*/
	s_set_vgpr_msb 0x400
	v_or_b32_e32 v220, 0x47, v232
	v_cndmask_b32_e32 v169, 0xff800000, v221, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v219, v181 /*v437*/
	s_set_vgpr_msb 0x400
	v_or_b32_e32 v221, 0x50, v232
	v_cndmask_b32_e32 v170, 0xff800000, v222, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v220, v181 /*v437*/
	s_set_vgpr_msb 0x400
	v_add_nc_u32_e32 v222, 0x51, v232
	v_cndmask_b32_e32 v171, 0xff800000, v223, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v221, v181 /*v437*/
	s_set_vgpr_msb 0x400
	v_add_nc_u32_e32 v223, 0x52, v232
	v_cndmask_b32_e32 v172, 0xff800000, v208, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v222, v181 /*v437*/
	s_set_vgpr_msb 0x400
	v_add_nc_u32_e32 v208, 0x53, v232
	v_cndmask_b32_e32 v173, 0xff800000, v209, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v223, v181 /*v437*/
	s_set_vgpr_msb 0x400
	v_add_nc_u32_e32 v209, 0x54, v232
	v_cndmask_b32_e32 v174, 0xff800000, v210, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v208, v181 /*v437*/
	s_set_vgpr_msb 0x400
	v_add_nc_u32_e32 v210, 0x55, v232
	v_cndmask_b32_e32 v175, 0xff800000, v211, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v209, v181 /*v437*/
	s_set_vgpr_msb 0x400
	v_add_nc_u32_e32 v211, 0x56, v232
	v_cndmask_b32_e32 v176, 0xff800000, v212, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v210, v181 /*v437*/
	s_set_vgpr_msb 0x400
	v_add_nc_u32_e32 v212, 0x57, v232
	v_cndmask_b32_e32 v177, 0xff800000, v213, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v211, v181 /*v437*/
	s_set_vgpr_msb 0x400
	v_or_b32_e32 v213, 0x60, v232
	v_cndmask_b32_e32 v178, 0xff800000, v214, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v212, v181 /*v437*/
	s_set_vgpr_msb 0x400
	v_or_b32_e32 v214, 0x61, v232
	v_cndmask_b32_e32 v179, 0xff800000, v215, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v213, v181 /*v437*/
	s_set_vgpr_msb 0x400
	v_or_b32_e32 v215, 0x62, v232
	v_cndmask_b32_e32 v180, 0xff800000, v200, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v214, v181 /*v437*/
	s_set_vgpr_msb 0x400
	v_or_b32_e32 v200, 0x63, v232
	v_cndmask_b32_e32 v181, 0xff800000, v201, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v215, v181 /*v437*/
	s_set_vgpr_msb 0x400
	v_or_b32_e32 v201, 0x64, v232
	v_cndmask_b32_e32 v182, 0xff800000, v202, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v200, v181 /*v437*/
	s_set_vgpr_msb 0x400
	v_or_b32_e32 v202, 0x65, v232
	v_cndmask_b32_e32 v183, 0xff800000, v203, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v201, v181 /*v437*/
	s_set_vgpr_msb 0x400
	v_or_b32_e32 v203, 0x66, v232
	v_cndmask_b32_e32 v184, 0xff800000, v204, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v202, v181 /*v437*/
	s_set_vgpr_msb 0x400
	v_or_b32_e32 v204, 0x67, v232
	v_cndmask_b32_e32 v185, 0xff800000, v205, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v203, v181 /*v437*/
	s_set_vgpr_msb 0x400
	v_or_b32_e32 v205, 0x70, v232
	v_cndmask_b32_e32 v186, 0xff800000, v206, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v204, v181 /*v437*/
	s_set_vgpr_msb 0x400
	v_add_nc_u32_e32 v206, 0x71, v232
	v_cndmask_b32_e32 v187, 0xff800000, v207, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v205, v181 /*v437*/
	s_set_vgpr_msb 0x400
	v_add_nc_u32_e32 v207, 0x72, v232
	v_cndmask_b32_e32 v188, 0xff800000, v192, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v206, v181 /*v437*/
	s_set_vgpr_msb 0x400
	v_add_nc_u32_e32 v192, 0x73, v232
	v_cndmask_b32_e32 v189, 0xff800000, v193, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v207, v181 /*v437*/
	s_set_vgpr_msb 0x400
	v_add_nc_u32_e32 v193, 0x74, v232
	v_cndmask_b32_e32 v190, 0xff800000, v194, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v192, v181 /*v437*/
	s_set_vgpr_msb 0x400
	v_add_nc_u32_e32 v194, 0x75, v232
	v_cndmask_b32_e32 v191, 0xff800000, v195, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v193, v181 /*v437*/
	s_set_vgpr_msb 0x400
	v_add_nc_u32_e32 v195, 0x76, v232
	v_cndmask_b32_e32 v234, 0xff800000, v196, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v194, v181 /*v437*/
	s_set_vgpr_msb 0x400
	v_add_nc_u32_e32 v196, 0x77, v232
	v_cndmask_b32_e32 v235, 0xff800000, v197, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v195, v181 /*v437*/
	s_set_vgpr_msb 0x440
	v_cndmask_b32_e32 v4 /*v260*/, 0xff800000, v198, vcc_lo
	s_set_vgpr_msb 0x4004
	v_cmp_le_i32_e32 vcc_lo, v196, v181 /*v437*/
	s_set_vgpr_msb 0x400
	v_max3_num_f32 v198, v143, v156, v157
	s_set_vgpr_msb 64
	v_cndmask_b32_e32 v5 /*v261*/, 0xff800000, v199, vcc_lo
	s_set_vgpr_msb 0x4004
	v_cmp_le_i32_e32 vcc_lo, v232, v182 /*v438*/
	v_cndmask_b32_e32 v238, 0xff800000, v192 /*v448*/, vcc_lo
	v_cmp_lt_i32_e32 vcc_lo, v232, v182 /*v438*/
	v_cndmask_b32_e32 v239, 0xff800000, v193 /*v449*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v233, v182 /*v438*/
	v_cndmask_b32_e32 v236, 0xff800000, v194 /*v450*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v237, v182 /*v438*/
	v_cndmask_b32_e32 v237, 0xff800000, v195 /*v451*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v240, v182 /*v438*/
	v_cndmask_b32_e32 v244, 0xff800000, v196 /*v452*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v241, v182 /*v438*/
	v_cndmask_b32_e32 v245, 0xff800000, v197 /*v453*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v242, v182 /*v438*/
	v_cndmask_b32_e32 v240, 0xff800000, v198 /*v454*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v243, v182 /*v438*/
	v_cndmask_b32_e32 v241, 0xff800000, v199 /*v455*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v246, v182 /*v438*/
	s_set_vgpr_msb 0x444
	v_cndmask_b32_e32 v6 /*v262*/, 0xff800000, v200 /*v456*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v247, v182 /*v438*/
	v_cndmask_b32_e32 v7 /*v263*/, 0xff800000, v201 /*v457*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v248, v182 /*v438*/
	s_set_vgpr_msb 0x4404
	v_cndmask_b32_e32 v246, 0xff800000, v202 /*v458*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v136, v182 /*v438*/
	s_set_vgpr_msb 0x400
	v_max3_num_f32 v136, v144, v145, v146
	s_set_vgpr_msb 4
	v_cndmask_b32_e32 v247, 0xff800000, v203 /*v459*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v137, v182 /*v438*/
	s_set_vgpr_msb 0x400
	v_max3_num_f32 v137, v238, v239, v236
	s_set_vgpr_msb 4
	v_cndmask_b32_e32 v242, 0xff800000, v204 /*v460*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v138, v182 /*v438*/
	s_set_vgpr_msb 0x400
	v_max3_num_f32 v138, v147, v148, v149
	s_set_vgpr_msb 4
	v_cndmask_b32_e32 v243, 0xff800000, v205 /*v461*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v139, v182 /*v438*/
	s_set_vgpr_msb 0x400
	v_max3_num_f32 v139, v237, v244, v245
	s_set_vgpr_msb 4
	v_cndmask_b32_e32 v252, 0xff800000, v206 /*v462*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v249, v182 /*v438*/
	v_cndmask_b32_e32 v253, 0xff800000, v207 /*v463*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v250, v182 /*v438*/
	s_set_vgpr_msb 0x400
	v_max3_num_f32 v197, v242, v243, v252
	s_set_vgpr_msb 4
	v_cndmask_b32_e32 v248, 0xff800000, v208 /*v464*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v251, v182 /*v438*/
	v_cndmask_b32_e32 v249, 0xff800000, v209 /*v465*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v254, v182 /*v438*/
	s_set_vgpr_msb 0x400
	v_max3_num_f32 v199, v253, v248, v249
	s_set_vgpr_msb 0x44
	v_cndmask_b32_e32 v16 /*v272*/, 0xff800000, v210 /*v466*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v224, v182 /*v438*/
	s_set_vgpr_msb 0x4400
	v_max_num_f32_e32 v224, v186, v187
	s_set_vgpr_msb 0x44
	v_cndmask_b32_e32 v17 /*v273*/, 0xff800000, v211 /*v467*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v225, v182 /*v438*/
	s_set_vgpr_msb 0x4404
	v_cndmask_b32_e32 v254, 0xff800000, v212 /*v468*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v226, v182 /*v438*/
	s_set_vgpr_msb 0x400
	v_max3_num_f32 v226, v189, v190, v191
	s_set_vgpr_msb 4
	v_cndmask_b32_e32 v255, 0xff800000, v213 /*v469*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v227, v182 /*v438*/
	v_cndmask_b32_e32 v250, 0xff800000, v214 /*v470*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v228, v182 /*v438*/
	s_set_vgpr_msb 0x410
	v_max3_num_f32 v228, v234, v235, v4 /*v260*/
	s_set_vgpr_msb 0x1004
	v_cndmask_b32_e32 v251, 0xff800000, v215 /*v471*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v229, v182 /*v438*/
	s_set_vgpr_msb 0x444
	v_cndmask_b32_e32 v8 /*v264*/, 0xff800000, v216 /*v472*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v230, v182 /*v438*/
	v_cndmask_b32_e32 v9 /*v265*/, 0xff800000, v217 /*v473*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v231, v182 /*v438*/
	v_cndmask_b32_e32 v0 /*v256*/, 0xff800000, v218 /*v474*/, vcc_lo
	s_set_vgpr_msb 0x4445
	v_cmp_le_i32_e32 vcc_lo, v1 /*v257*/, v182 /*v438*/
	v_cndmask_b32_e32 v1 /*v257*/, 0xff800000, v219 /*v475*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v2 /*v258*/, v182 /*v438*/
	v_cndmask_b32_e32 v26 /*v282*/, 0xff800000, v220 /*v476*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v3 /*v259*/, v182 /*v438*/
	v_cndmask_b32_e32 v27 /*v283*/, 0xff800000, v221 /*v477*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v10 /*v266*/, v182 /*v438*/
	v_cndmask_b32_e32 v10 /*v266*/, 0xff800000, v222 /*v478*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v11 /*v267*/, v182 /*v438*/
	v_cndmask_b32_e32 v11 /*v267*/, 0xff800000, v223 /*v479*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v12 /*v268*/, v182 /*v438*/
	v_cndmask_b32_e32 v2 /*v258*/, 0xff800000, v224 /*v480*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v13 /*v269*/, v182 /*v438*/
	v_cndmask_b32_e32 v3 /*v259*/, 0xff800000, v225 /*v481*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v14 /*v270*/, v182 /*v438*/
	v_cndmask_b32_e32 v18 /*v274*/, 0xff800000, v226 /*v482*/, vcc_lo
	s_set_vgpr_msb 0x4504
	v_cmp_le_i32_e32 vcc_lo, v216, v182 /*v438*/
	s_set_vgpr_msb 0x400
	v_max3_num_f32 v216, v174, v175, v176
	s_set_vgpr_msb 0x44
	v_cndmask_b32_e32 v19 /*v275*/, 0xff800000, v227 /*v483*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v217, v182 /*v438*/
	v_cndmask_b32_e32 v12 /*v268*/, 0xff800000, v228 /*v484*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v218, v182 /*v438*/
	s_set_vgpr_msb 0x4400
	v_max3_num_f32 v218, v177, v178, v179
	s_set_vgpr_msb 0x44
	v_cndmask_b32_e32 v13 /*v269*/, 0xff800000, v229 /*v485*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v219, v182 /*v438*/
	v_cndmask_b32_e32 v40 /*v296*/, 0xff800000, v230 /*v486*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v220, v182 /*v438*/
	s_set_vgpr_msb 0x4400
	v_max3_num_f32 v220, v180, v181, v182
	s_set_vgpr_msb 0x44
	v_cndmask_b32_e32 v41 /*v297*/, 0xff800000, v231 /*v487*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v221, v182 /*v438*/
	v_cndmask_b32_e32 v20 /*v276*/, 0xff800000, v232 /*v488*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v222, v182 /*v438*/
	s_set_vgpr_msb 0x4400
	v_max3_num_f32 v222, v183, v184, v185
	s_set_vgpr_msb 0x44
	v_cndmask_b32_e32 v21 /*v277*/, 0xff800000, v233 /*v489*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v223, v182 /*v438*/
	v_cndmask_b32_e32 v14 /*v270*/, 0xff800000, v234 /*v490*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v208, v182 /*v438*/
	s_set_vgpr_msb 0x4400
	v_max3_num_f32 v208, v134, v135, v164
	s_set_vgpr_msb 0x44
	v_cndmask_b32_e32 v15 /*v271*/, 0xff800000, v235 /*v491*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v209, v182 /*v438*/
	s_set_vgpr_msb 0x4415
	v_max3_num_f32 v209, v10 /*v266*/, v11 /*v267*/, v2 /*v258*/
	s_set_vgpr_msb 0x1544
	v_cndmask_b32_e32 v28 /*v284*/, 0xff800000, v236 /*v492*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v210, v182 /*v438*/
	s_set_vgpr_msb 0x4400
	v_max3_num_f32 v210, v165, v166, v167
	s_set_vgpr_msb 0x44
	v_cndmask_b32_e32 v29 /*v285*/, 0xff800000, v237 /*v493*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v211, v182 /*v438*/
	s_set_vgpr_msb 0x4415
	v_max3_num_f32 v211, v3 /*v259*/, v18 /*v274*/, v19 /*v275*/
	v_max3_num_f32 v217, v14 /*v270*/, v15 /*v271*/, v28 /*v284*/
	s_set_vgpr_msb 0x1544
	v_cndmask_b32_e32 v22 /*v278*/, 0xff800000, v238 /*v494*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v212, v182 /*v438*/
	s_set_vgpr_msb 0x4400
	v_max3_num_f32 v212, v168, v169, v170
	s_set_vgpr_msb 0x44
	v_cndmask_b32_e32 v23 /*v279*/, 0xff800000, v239 /*v495*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v213, v182 /*v438*/
	s_set_vgpr_msb 0x4415
	v_max3_num_f32 v213, v12 /*v268*/, v13 /*v269*/, v40 /*v296*/
	v_max3_num_f32 v219, v29 /*v285*/, v22 /*v278*/, v23 /*v279*/
	s_set_vgpr_msb 0x1544
	v_cndmask_b32_e32 v42 /*v298*/, 0xff800000, v240 /*v496*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v214, v182 /*v438*/
	s_set_vgpr_msb 0x4400
	v_max3_num_f32 v214, v171, v172, v173
	s_set_vgpr_msb 0x44
	v_cndmask_b32_e32 v43 /*v299*/, 0xff800000, v241 /*v497*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v215, v182 /*v438*/
	s_set_vgpr_msb 0x4415
	v_max3_num_f32 v215, v41 /*v297*/, v20 /*v276*/, v21 /*v277*/
	s_set_vgpr_msb 0x1544
	v_cndmask_b32_e32 v30 /*v286*/, 0xff800000, v242 /*v498*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v200, v182 /*v438*/
	s_set_vgpr_msb 0x4400
	v_max3_num_f32 v200, v158, v159, v160
	s_set_vgpr_msb 0x44
	v_cndmask_b32_e32 v31 /*v287*/, 0xff800000, v243 /*v499*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v201, v182 /*v438*/
	s_set_vgpr_msb 0x4405
	v_max3_num_f32 v201, v16 /*v272*/, v17 /*v273*/, v254
	s_set_vgpr_msb 0x515
	v_max3_num_f32 v221, v42 /*v298*/, v43 /*v299*/, v30 /*v286*/
	s_set_vgpr_msb 0x1544
	v_cndmask_b32_e32 v24 /*v280*/, 0xff800000, v244 /*v500*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v202, v182 /*v438*/
	s_set_vgpr_msb 0x4400
	v_max3_num_f32 v202, v161, v162, v163
	s_set_vgpr_msb 0x44
	v_cndmask_b32_e32 v25 /*v281*/, 0xff800000, v245 /*v501*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v203, v182 /*v438*/
	s_set_vgpr_msb 0x4400
	v_max3_num_f32 v203, v255, v250, v251
	s_set_vgpr_msb 21
	v_max3_num_f32 v223, v31 /*v287*/, v24 /*v280*/, v25 /*v281*/
	s_set_vgpr_msb 0x1544
	v_cndmask_b32_e32 v32 /*v288*/, 0xff800000, v246 /*v502*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v204, v182 /*v438*/
	s_set_vgpr_msb 0x4400
	v_max3_num_f32 v204, v128, v129, v130
	s_set_vgpr_msb 0x44
	v_cndmask_b32_e32 v33 /*v289*/, 0xff800000, v247 /*v503*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v205, v182 /*v438*/
	s_set_vgpr_msb 0x4415
	v_max3_num_f32 v205, v8 /*v264*/, v9 /*v265*/, v0 /*v256*/
	v_max_num_f32_e32 v225, v32 /*v288*/, v33 /*v289*/
	s_set_vgpr_msb 0x1544
	v_cndmask_b32_e32 v34 /*v290*/, 0xff800000, v248 /*v504*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v206, v182 /*v438*/
	s_set_vgpr_msb 0x4400
	v_max3_num_f32 v206, v131, v132, v133
	s_set_vgpr_msb 0x44
	v_cndmask_b32_e32 v35 /*v291*/, 0xff800000, v249 /*v505*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v207, v182 /*v438*/
	s_set_vgpr_msb 0x4415
	v_max3_num_f32 v207, v1 /*v257*/, v26 /*v282*/, v27 /*v283*/
	s_set_vgpr_msb 0x1544
	v_cndmask_b32_e32 v44 /*v300*/, 0xff800000, v250 /*v506*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v192, v182 /*v438*/
	s_set_vgpr_msb 0x4400
	v_max3_num_f32 v192, v150, v151, v152
	s_set_vgpr_msb 0x44
	v_cndmask_b32_e32 v45 /*v301*/, 0xff800000, v251 /*v507*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v193, v182 /*v438*/
	s_set_vgpr_msb 0x4410
	v_max3_num_f32 v193, v240, v241, v6 /*v262*/
	s_set_vgpr_msb 0x1000
	v_max3_num_f32 v136, v136, v138, v192
	v_max3_num_f32 v192, v206, v208, v210
	s_set_vgpr_msb 21
	v_max3_num_f32 v227, v35 /*v291*/, v44 /*v300*/, v45 /*v301*/
	s_set_vgpr_msb 0x1544
	v_cndmask_b32_e32 v36 /*v292*/, 0xff800000, v252 /*v508*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v194, v182 /*v438*/
	s_set_vgpr_msb 0x4400
	v_max3_num_f32 v194, v153, v154, v155
	v_max3_num_f32 v137, v137, v139, v193
	v_max3_num_f32 v139, v200, v202, v204
	v_max3_num_f32 v193, v212, v214, v216
	s_set_vgpr_msb 0x44
	v_cndmask_b32_e32 v37 /*v293*/, 0xff800000, v253 /*v509*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v195, v182 /*v438*/
	s_set_vgpr_msb 0x4401
	v_max3_num_f32 v195, v7 /*v263*/, v246, v247
	s_set_vgpr_msb 0x144
	v_cndmask_b32_e32 v38 /*v294*/, 0xff800000, v254 /*v510*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v196, v182 /*v438*/
	s_set_vgpr_msb 0x4400
	v_max3_num_f32 v196, v140, v141, v142
	v_max3_num_f32 v195, v195, v197, v199
	v_max3_num_f32 v197, v201, v203, v205
	s_set_vgpr_msb 0x44
	v_cndmask_b32_e32 v39 /*v295*/, 0xff800000, v255 /*v511*/, vcc_lo
	s_set_vgpr_msb 0x4400
	v_max3_num_f32 v138, v194, v196, v198
	v_max3_num_f32 v194, v218, v220, v222
	v_max3_num_f32 v196, v224, v188, v226
	s_set_vgpr_msb 21
	v_max3_num_f32 v229, v36 /*v292*/, v37 /*v293*/, v38 /*v294*/
	s_set_vgpr_msb 0x1500
	v_max3_num_f32 v137, v137, v195, v197
	v_max3_num_f32 v136, v136, v138, v139
	v_max3_num_f32 v138, v192, v193, v194
	s_set_vgpr_msb 16
	v_max3_num_f32 v139, v196, v228, v5 /*v261*/
	s_set_vgpr_msb 0x1000
	v_max3_num_f32 v192, v207, v209, v211
	v_max3_num_f32 v193, v213, v215, v217
	v_max3_num_f32 v194, v219, v221, v223
	s_set_vgpr_msb 4
	v_max3_num_f32 v196, v225, v34 /*v290*/, v227
	s_set_vgpr_msb 0x400
	v_max3_num_f32 v136, v136, v138, v139
	v_max3_num_f32 v138, v192, v193, v194
	s_set_vgpr_msb 16
	v_max3_num_f32 v139, v196, v229, v39 /*v295*/
	v_mov_b32_e32 v192, v136
	s_set_vgpr_msb 0x1000
	v_max3_num_f32 v137, v137, v138, v139
	v_permlanex16_b32 v192, v192, s35, 0xfedcba98
	v_dual_mov_b32 v138, v137 :: v_dual_max_num_f32 v136, v136, v192
	v_permlanex16_b32 v138, v138, s35, 0xfedcba98
	v_max_num_f32_e32 v137, v137, v138
	s_set_vgpr_msb 4
	v_sub_f32_e32 v139, v136, v180 /*v436*/
	v_max_num_f32_e32 v136, v136, v180 /*v436*/
	v_sub_f32_e32 v138, v137, v179 /*v435*/
	s_set_vgpr_msb 0x400
	v_cmp_lt_f32_e32 vcc_lo, 0x41000000, v139
	s_cmp_eq_u32 vcc_lo, 0
	s_cselect_b32 s2, -1, 0
	s_set_vgpr_msb 0x44
	v_cndmask_b32_e64 v55 /*v311*/, v136, v180 /*v436*/, s2
	s_set_vgpr_msb 0x4401
	v_cmp_lt_f32_e64 s2, 0x41000000, v138
	v_max_num_f32_e32 v136, v179 /*v435*/, v137
	s_cmp_lg_u32 s2, 0
	s_cselect_b32 s5, -1, 0
	s_cmp_eq_u32 s2, 0
	s_cselect_b32 s2, -1, 0
	s_set_vgpr_msb 0x104
	v_cndmask_b32_e64 v192, v136, v179 /*v435*/, s2
	s_set_vgpr_msb 0x444
	v_dual_mul_f32 v226 /*v482*/, 0xbfb8aa3b, v55 /*v311*/ :: v_dual_mov_b32 v177 /*v433*/, v192
	s_set_vgpr_msb 0x4410
	v_pk_fma_f32 v[128:129], v[128:129], s[46:47], v[226:227] /*v[482:483]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[136:137], v[144:145], s[46:47], v[226:227] /*v[482:483]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[144:145], v[148:149], s[46:47], v[226:227] /*v[482:483]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[130:131], v[130:131], s[46:47], v[226:227] /*v[482:483]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[132:133], v[132:133], s[46:47], v[226:227] /*v[482:483]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v216, v128
	v_exp_f32_e32 v226, v129
	v_nop
	v_pk_fma_f32 v[128:129], v[134:135], s[46:47], v[226:227] /*v[482:483]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[138:139], v[146:147], s[46:47], v[226:227] /*v[482:483]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[146:147], v[150:151], s[46:47], v[226:227] /*v[482:483]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1040
	v_exp_f32_e32 v202 /*v458*/, v144
	s_set_vgpr_msb 0x4010
	v_pk_fma_f32 v[148:149], v[152:153], s[46:47], v[226:227] /*v[482:483]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1040
	v_exp_f32_e32 v212 /*v468*/, v145
	v_nop
	s_set_vgpr_msb 0x4010
	v_pk_fma_f32 v[144:145], v[154:155], s[46:47], v[226:227] /*v[482:483]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[150:151], v[160:161], s[46:47], v[226:227] /*v[482:483]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1040
	v_exp_f32_e32 v206 /*v462*/, v130
	v_exp_f32_e32 v216 /*v472*/, v131
	v_exp_f32_e32 v224 /*v480*/, v132
	s_set_vgpr_msb 0x4010
	v_pk_fma_f32 v[130:131], v[164:165], s[46:47], v[226:227] /*v[482:483]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1040
	v_exp_f32_e32 v234 /*v490*/, v133
	v_exp_f32_e32 v238 /*v494*/, v128
	s_set_vgpr_msb 0x4010
	v_pk_fma_f32 v[132:133], v[166:167], s[46:47], v[226:227] /*v[482:483]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1040
	v_exp_f32_e32 v246 /*v502*/, v129
	v_nop
	s_set_vgpr_msb 0x4010
	v_pk_fma_f32 v[128:129], v[168:169], s[46:47], v[226:227] /*v[482:483]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1040
	v_mul_f32_e32 v54 /*v310*/, 0xbfb8aa3b, v192
	s_set_vgpr_msb 0x4000
	v_exp_f32_e32 v208, v136
	v_exp_f32_e32 v136, v138
	v_exp_f32_e32 v138, v146
	s_set_vgpr_msb 64
	v_exp_f32_e32 v228 /*v484*/, v147
	v_exp_f32_e32 v236 /*v492*/, v148
	s_set_vgpr_msb 0x4010
	v_pk_fma_f32 v[146:147], v[140:141], s[46:47], v[226:227] /*v[482:483]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1040
	v_exp_f32_e32 v244 /*v500*/, v149
	s_set_vgpr_msb 0x4000
	v_exp_f32_e32 v140, v144
	s_set_vgpr_msb 64
	v_exp_f32_e32 v250 /*v506*/, v145
	v_nop
	s_set_vgpr_msb 0x4010
	v_pk_fma_f32 v[144:145], v[156:157], s[46:47], v[226:227] /*v[482:483]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[148:149], v[158:159], s[46:47], v[226:227] /*v[482:483]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v156, v150
	v_exp_f32_e32 v150, v130
	v_exp_f32_e32 v158, v131
	v_exp_f32_e32 v192, v132
	v_pk_fma_f32 v[130:131], v[170:171], s[46:47], v[226:227] /*v[482:483]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v198, v133
	v_exp_f32_e32 v202, v128
	v_pk_fma_f32 v[132:133], v[172:173], s[46:47], v[226:227] /*v[482:483]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v212, v129
	v_nop
	v_pk_fma_f32 v[128:129], v[174:175], s[46:47], v[226:227] /*v[482:483]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v214, v130
	v_exp_f32_e32 v224, v131
	v_exp_f32_e32 v228, v132
	v_pk_fma_f32 v[130:131], v[176:177], s[46:47], v[226:227] /*v[482:483]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1040
	v_exp_f32_e32 v196 /*v452*/, v133
	v_exp_f32_e32 v198 /*v454*/, v128
	s_set_vgpr_msb 0x4010
	v_pk_fma_f32 v[132:133], v[178:179], s[46:47], v[226:227] /*v[482:483]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1040
	v_exp_f32_e32 v208 /*v464*/, v129
	v_nop
	s_set_vgpr_msb 0x4010
	v_pk_fma_f32 v[128:129], v[180:181], s[46:47], v[226:227] /*v[482:483]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1040
	v_exp_f32_e32 v210 /*v466*/, v130
	v_exp_f32_e32 v220 /*v476*/, v131
	v_exp_f32_e32 v222 /*v478*/, v132
	s_set_vgpr_msb 0x4010
	v_pk_fma_f32 v[130:131], v[182:183], s[46:47], v[226:227] /*v[482:483]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1040
	v_exp_f32_e32 v232 /*v488*/, v133
	s_set_vgpr_msb 0x4010
	v_exp_f32_e32 v196, v128
	v_pk_fma_f32 v[132:133], v[184:185], s[46:47], v[226:227] /*v[482:483]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v204, v129
	v_nop
	v_pk_fma_f32 v[128:129], v[186:187], s[46:47], v[226:227] /*v[482:483]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v206, v130
	v_exp_f32_e32 v218, v131
	v_exp_f32_e32 v222, v132
	v_pk_fma_f32 v[130:131], v[188:189], s[46:47], v[226:227] /*v[482:483]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v230, v133
	s_set_vgpr_msb 0x1040
	v_exp_f32_e32 v192 /*v448*/, v128
	s_set_vgpr_msb 0x4010
	v_pk_fma_f32 v[132:133], v[190:191], s[46:47], v[226:227] /*v[482:483]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1040
	v_exp_f32_e32 v200 /*v456*/, v129
	v_nop
	s_set_vgpr_msb 0x4010
	v_pk_fma_f32 v[128:129], v[234:235], s[46:47], v[226:227] /*v[482:483]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[142:143], v[142:143], s[46:47], v[226:227] /*v[482:483]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[152:153], v[162:163], s[46:47], v[226:227] /*v[482:483]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1040
	v_exp_f32_e32 v204 /*v460*/, v130
	v_exp_f32_e32 v214 /*v470*/, v131
	v_exp_f32_e32 v218 /*v474*/, v132
	s_set_vgpr_msb 0x4011
	v_pk_fma_f32 v[130:131], v[4:5] /*v[260:261]*/, s[46:47], v[226:227] /*v[482:483]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1140
	v_exp_f32_e32 v226 /*v482*/, v133
	v_exp_f32_e32 v230 /*v486*/, v128
	s_set_vgpr_msb 0x4010
	v_pk_fma_f32 v[132:133], v[238:239], s[46:47], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1040
	v_exp_f32_e32 v240 /*v496*/, v129
	v_nop
	s_set_vgpr_msb 0x4010
	v_pk_fma_f32 v[128:129], v[236:237], s[46:47], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v220, v137
	s_set_vgpr_msb 0x1040
	v_exp_f32_e32 v242 /*v498*/, v130
	s_set_vgpr_msb 0x4010
	v_exp_f32_e32 v209, v132
	v_exp_f32_e32 v221, v133
	v_exp_f32_e32 v137, v128
	v_pk_fma_f32 v[132:133], v[240:241], s[46:47], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1040
	v_exp_f32_e32 v195 /*v451*/, v129
	v_nop
	s_set_vgpr_msb 0x4011
	v_pk_fma_f32 v[128:129], v[6:7] /*v[262:263]*/, s[46:47], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1140
	v_exp_f32_e32 v248 /*v504*/, v131
	v_nop
	s_set_vgpr_msb 0x4010
	v_pk_fma_f32 v[130:131], v[244:245], s[46:47], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1040
	v_exp_f32_e32 v194 /*v450*/, v139
	s_set_vgpr_msb 0x4000
	v_exp_f32_e32 v139, v132
	s_set_vgpr_msb 64
	v_exp_f32_e32 v229 /*v485*/, v133
	v_exp_f32_e32 v237 /*v493*/, v128
	s_set_vgpr_msb 0x4010
	v_pk_fma_f32 v[132:133], v[242:243], s[46:47], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1040
	v_exp_f32_e32 v245 /*v501*/, v129
	v_nop
	s_set_vgpr_msb 0x4010
	v_pk_fma_f32 v[128:129], v[252:253], s[46:47], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1040
	v_exp_f32_e32 v203 /*v459*/, v130
	v_exp_f32_e32 v213 /*v469*/, v131
	v_nop
	s_set_vgpr_msb 0x4010
	v_pk_fma_f32 v[130:131], v[246:247], s[46:47], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v232, v143
	s_set_vgpr_msb 0x1040
	v_exp_f32_e32 v253 /*v509*/, v132
	v_exp_f32_e32 v255 /*v511*/, v133
	s_set_vgpr_msb 0x4000
	v_exp_f32_e32 v143, v128
	s_set_vgpr_msb 17
	v_pk_fma_f32 v[132:133], v[16:17] /*v[272:273]*/, s[46:47], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1110
	v_exp_f32_e32 v233, v129
	v_nop
	v_pk_fma_f32 v[128:129], v[254:255], s[46:47], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v141, v130
	s_set_vgpr_msb 0x1040
	v_exp_f32_e32 v251 /*v507*/, v131
	v_nop
	s_set_vgpr_msb 0x4010
	v_pk_fma_f32 v[130:131], v[248:249], s[46:47], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v154, v149
	v_exp_f32_e32 v149, v132
	v_exp_f32_e32 v155, v133
	v_exp_f32_e32 v157, v128
	s_set_vgpr_msb 0x1011
	v_pk_fma_f32 v[132:133], v[8:9] /*v[264:265]*/, s[46:47], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1100
	v_exp_f32_e32 v195, v129
	v_nop
	s_set_vgpr_msb 17
	v_pk_fma_f32 v[128:129], v[0:1] /*v[256:257]*/, s[46:47], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1140
	v_exp_f32_e32 v252 /*v508*/, v146
	v_exp_f32_e32 v254 /*v510*/, v147
	s_set_vgpr_msb 0x4010
	v_exp_f32_e32 v146, v145
	v_exp_f32_e32 v145, v130
	v_exp_f32_e32 v147, v131
	v_nop
	v_pk_fma_f32 v[130:131], v[250:251], s[46:47], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v217, v132
	v_exp_f32_e32 v227, v133
	s_set_vgpr_msb 0x1040
	v_exp_f32_e32 v207 /*v463*/, v128
	s_set_vgpr_msb 0x4011
	v_pk_fma_f32 v[132:133], v[10:11] /*v[266:267]*/, s[46:47], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1140
	v_exp_f32_e32 v217 /*v473*/, v129
	v_nop
	s_set_vgpr_msb 0x4011
	v_pk_fma_f32 v[128:129], v[2:3] /*v[258:259]*/, s[46:47], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1100
	v_exp_f32_e32 v201, v130
	v_exp_f32_e32 v211, v131
	v_nop
	s_set_vgpr_msb 17
	v_pk_fma_f32 v[130:131], v[26:27] /*v[282:283]*/, s[46:47], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1100
	v_exp_f32_e32 v194, v151
	s_set_vgpr_msb 64
	v_exp_f32_e32 v239 /*v495*/, v132
	v_exp_f32_e32 v247 /*v503*/, v133
	s_set_vgpr_msb 0x4000
	v_exp_f32_e32 v151, v128
	s_set_vgpr_msb 17
	v_pk_fma_f32 v[132:133], v[12:13] /*v[268:269]*/, s[46:47], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1100
	v_exp_f32_e32 v159, v129
	v_nop
	s_set_vgpr_msb 17
	v_pk_fma_f32 v[128:129], v[40:41] /*v[296:297]*/, s[46:47], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1140
	v_exp_f32_e32 v225 /*v481*/, v130
	v_exp_f32_e32 v235 /*v491*/, v131
	v_nop
	s_set_vgpr_msb 0x4011
	v_pk_fma_f32 v[130:131], v[18:19] /*v[274:275]*/, s[46:47], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1100
	v_exp_f32_e32 v203, v132
	v_exp_f32_e32 v213, v133
	v_exp_f32_e32 v215, v128
	s_set_vgpr_msb 17
	v_pk_fma_f32 v[132:133], v[14:15] /*v[270:271]*/, s[46:47], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1100
	v_exp_f32_e32 v225, v129
	v_nop
	s_set_vgpr_msb 17
	v_pk_fma_f32 v[128:129], v[28:29] /*v[284:285]*/, s[46:47], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1100
	v_exp_f32_e32 v193, v130
	v_exp_f32_e32 v199, v131
	v_nop
	s_set_vgpr_msb 17
	v_pk_fma_f32 v[130:131], v[20:21] /*v[276:277]*/, s[46:47], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1140
	v_exp_f32_e32 v199 /*v455*/, v132
	v_exp_f32_e32 v209 /*v465*/, v133
	v_exp_f32_e32 v211 /*v467*/, v128
	s_set_vgpr_msb 0x4011
	v_pk_fma_f32 v[132:133], v[42:43] /*v[298:299]*/, s[46:47], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1140
	v_exp_f32_e32 v221 /*v477*/, v129
	v_nop
	s_set_vgpr_msb 0x4011
	v_pk_fma_f32 v[128:129], v[30:31] /*v[286:287]*/, s[46:47], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1100
	v_exp_f32_e32 v229, v130
	s_set_vgpr_msb 64
	v_exp_f32_e32 v197 /*v453*/, v131
	v_nop
	s_set_vgpr_msb 0x4011
	v_pk_fma_f32 v[130:131], v[22:23] /*v[278:279]*/, s[46:47], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1100
	v_exp_f32_e32 v197, v132
	v_exp_f32_e32 v205, v133
	v_exp_f32_e32 v207, v128
	v_exp_f32_e32 v219, v129
	v_nop
	s_set_vgpr_msb 17
	v_pk_fma_f32 v[128:129], v[32:33] /*v[288:289]*/, s[46:47], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[132:133], v[34:35] /*v[290:291]*/, s[46:47], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1140
	v_exp_f32_e32 v223 /*v479*/, v130
	v_exp_f32_e32 v233 /*v489*/, v131
	v_nop
	s_set_vgpr_msb 0x4011
	v_pk_fma_f32 v[130:131], v[24:25] /*v[280:281]*/, s[46:47], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1140
	v_exp_f32_e32 v193 /*v449*/, v128
	v_exp_f32_e32 v201 /*v457*/, v129
	v_exp_f32_e32 v205 /*v461*/, v132
	v_exp_f32_e32 v215 /*v471*/, v133
	s_set_vgpr_msb 0x4011
	v_pk_fma_f32 v[128:129], v[36:37] /*v[292:293]*/, s[46:47], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1100
	v_pk_add_f32 v[132:133], v[208:209], v[220:221]
	s_set_vgpr_msb 5
	v_pk_add_f32 v[134:135], v[194:195] /*v[450:451]*/, v[202:203] /*v[458:459]*/
	s_set_vgpr_msb 0x500
	v_exp_f32_e32 v142, v142
	v_exp_f32_e32 v144, v144
	v_exp_f32_e32 v148, v148
	v_exp_f32_e32 v200, v152
	v_exp_f32_e32 v223, v130
	v_exp_f32_e32 v231, v131
	v_nop
	s_set_vgpr_msb 17
	v_pk_fma_f32 v[130:131], v[44:45] /*v[300:301]*/, s[46:47], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1100
	v_exp_f32_e32 v210, v153
	s_set_vgpr_msb 64
	v_exp_f32_e32 v231 /*v487*/, v128
	v_exp_f32_e32 v241 /*v497*/, v129
	v_nop
	s_set_vgpr_msb 0x4000
	v_pk_add_f32 v[128:129], v[136:137], v[132:133]
	s_set_vgpr_msb 1
	v_pk_add_f32 v[132:133], v[212:213] /*v[468:469]*/, v[134:135]
	v_pk_add_f32 v[134:135], v[228:229] /*v[484:485]*/, v[138:139]
	v_pk_add_f32 v[152:153], v[244:245] /*v[500:501]*/, v[140:141]
	s_set_vgpr_msb 0x105
	v_pk_add_f32 v[160:161], v[252:253] /*v[508:509]*/, v[254:255] /*v[510:511]*/
	v_pk_add_f32 v[170:171], v[216:217] /*v[472:473]*/, v[224:225] /*v[480:481]*/
	v_pk_add_f32 v[172:173], v[238:239] /*v[494:495]*/, v[246:247] /*v[502:503]*/
	s_set_vgpr_msb 0x500
	v_pk_add_f32 v[176:177], v[202:203], v[212:213]
	v_pk_add_f32 v[178:179], v[224:225], v[228:229]
	s_set_vgpr_msb 64
	v_exp_f32_e32 v219 /*v475*/, v130
	s_set_vgpr_msb 0x4000
	v_pk_add_f32 v[162:163], v[232:233], v[144:145]
	v_pk_add_f32 v[164:165], v[148:149], v[154:155]
	s_set_vgpr_msb 1
	v_pk_add_f32 v[134:135], v[236:237] /*v[492:493]*/, v[134:135]
	v_pk_add_f32 v[152:153], v[250:251] /*v[506:507]*/, v[152:153]
	s_set_vgpr_msb 0x100
	v_pk_add_f32 v[160:161], v[142:143], v[160:161]
	v_pk_add_f32 v[166:167], v[194:195], v[200:201]
	v_pk_add_f32 v[174:175], v[158:159], v[192:193]
	s_set_vgpr_msb 1
	v_pk_add_f32 v[170:171], v[234:235] /*v[490:491]*/, v[170:171]
	s_set_vgpr_msb 0x100
	v_pk_add_f32 v[172:173], v[150:151], v[172:173]
	s_set_vgpr_msb 5
	v_pk_add_f32 v[180:181], v[198:199] /*v[454:455]*/, v[208:209] /*v[464:465]*/
	v_pk_add_f32 v[182:183], v[220:221] /*v[476:477]*/, v[222:223] /*v[478:479]*/
	s_set_vgpr_msb 0x500
	v_pk_add_f32 v[184:185], v[196:197], v[204:205]
	v_pk_add_f32 v[176:177], v[214:215], v[176:177]
	s_set_vgpr_msb 1
	v_pk_add_f32 v[178:179], v[196:197] /*v[452:453]*/, v[178:179]
	s_set_vgpr_msb 0x100
	v_pk_add_f32 v[128:129], v[128:129], v[132:133]
	s_set_vgpr_msb 64
	v_exp_f32_e32 v227 /*v483*/, v131
	v_nop
	s_set_vgpr_msb 0x4011
	v_pk_fma_f32 v[130:131], v[38:39] /*v[294:295]*/, s[46:47], v[54:55] /*v[310:311]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1100
	v_pk_add_f32 v[162:163], v[146:147], v[162:163]
	v_pk_add_f32 v[164:165], v[156:157], v[164:165]
	v_pk_add_f32 v[168:169], v[216:217], v[226:227]
	v_pk_add_f32 v[166:167], v[210:211], v[166:167]
	v_pk_add_f32 v[174:175], v[198:199], v[174:175]
	s_set_vgpr_msb 1
	v_pk_add_f32 v[180:181], v[210:211] /*v[466:467]*/, v[180:181]
	v_pk_add_f32 v[182:183], v[232:233] /*v[488:489]*/, v[182:183]
	s_set_vgpr_msb 0x100
	v_pk_add_f32 v[184:185], v[206:207], v[184:185]
	v_pk_add_f32 v[186:187], v[218:219], v[222:223]
	s_set_vgpr_msb 5
	v_pk_add_f32 v[188:189], v[192:193] /*v[448:449]*/, v[200:201] /*v[456:457]*/
	v_pk_add_f32 v[190:191], v[214:215] /*v[470:471]*/, v[218:219] /*v[474:475]*/
	s_set_vgpr_msb 0x500
	v_pk_add_f32 v[128:129], v[134:135], v[128:129]
	v_pk_add_f32 v[134:135], v[152:153], v[160:161]
	v_pk_add_f32 v[152:153], v[170:171], v[172:173]
	v_pk_add_f32 v[160:161], v[176:177], v[178:179]
	s_set_vgpr_msb 64
	v_exp_f32_e32 v243 /*v499*/, v130
	s_set_vgpr_msb 0x4001
	v_pk_add_f32 v[168:169], v[206:207] /*v[462:463]*/, v[168:169]
	s_set_vgpr_msb 0x105
	v_pk_add_f32 v[234:235], v[230:231] /*v[486:487]*/, v[240:241] /*v[496:497]*/
	s_set_vgpr_msb 0x500
	v_pk_add_f32 v[132:133], v[230:231], v[186:187]
	s_set_vgpr_msb 1
	v_pk_add_f32 v[186:187], v[204:205] /*v[460:461]*/, v[188:189]
	v_pk_add_f32 v[188:189], v[226:227] /*v[482:483]*/, v[190:191]
	s_set_vgpr_msb 0x100
	v_pk_add_f32 v[164:165], v[164:165], v[166:167]
	v_pk_add_f32 v[166:167], v[182:183], v[184:185]
	v_pk_add_f32 v[134:135], v[162:163], v[134:135]
	v_pk_add_f32 v[152:153], v[174:175], v[152:153]
	v_pk_add_f32 v[160:161], v[180:181], v[160:161]
	s_set_vgpr_msb 1
	v_pk_add_f32 v[190:191], v[242:243] /*v[498:499]*/, v[234:235]
	s_set_vgpr_msb 0x100
	v_pk_add_f32 v[162:163], v[168:169], v[164:165]
	v_pk_add_f32 v[132:133], v[132:133], v[166:167]
	v_pk_add_f32 v[164:165], v[186:187], v[188:189]
	v_pk_add_f32 v[128:129], v[128:129], v[134:135]
	v_pk_add_f32 v[134:135], v[152:153], v[160:161]
	s_set_vgpr_msb 64
	v_exp_f32_e32 v249 /*v505*/, v131
	v_nop
	s_set_vgpr_msb 0x4000
	v_pk_add_f32 v[130:131], v[190:191], v[164:165]
	v_pk_add_f32 v[128:129], v[162:163], v[128:129]
	v_pk_add_f32 v[132:133], v[132:133], v[134:135]
	s_set_vgpr_msb 1
	v_pk_add_f32 v[130:131], v[248:249] /*v[504:505]*/, v[130:131]
	s_set_vgpr_msb 0x100
	v_pk_add_f32 v[128:129], v[128:129], v[132:133]
	s_set_vgpr_msb 5
	v_sub_f32_e32 v132, v180 /*v436*/, v55 /*v311*/
	s_set_vgpr_msb 0x500
	v_pk_add_f32 v[128:129], v[130:131], v[128:129]
	v_dual_mul_f32 v130, 0x3fb8aa3b, v132 :: v_dual_mov_b32 v131, v128
	v_mov_b32_e32 v133, v129
	v_exp_f32_e32 v130, v130
	v_permlanex16_b32 v131, v131, s35, 0xfedcba98
	v_permlanex16_b32 v133, v133, s35, 0xfedcba98
	s_cbranch_vccz .LBB0_42
	v_pk_mul_f32 v[126:127], v[126:127], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[124:125], v[124:125], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[122:123], v[122:123], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[120:121], v[120:121], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[118:119], v[118:119], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[116:117], v[116:117], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[114:115], v[114:115], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[112:113], v[112:113], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[110:111], v[110:111], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[108:109], v[108:109], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[106:107], v[106:107], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[104:105], v[104:105], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[102:103], v[102:103], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[100:101], v[100:101], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[98:99], v[98:99], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[96:97], v[96:97], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[94:95], v[94:95], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[92:93], v[92:93], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[90:91], v[90:91], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[88:89], v[88:89], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[86:87], v[86:87], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[84:85], v[84:85], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[82:83], v[82:83], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[80:81], v[80:81], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[78:79], v[78:79], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[76:77], v[76:77], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[74:75], v[74:75], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[72:73], v[72:73], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[70:71], v[70:71], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[68:69], v[68:69], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[66:67], v[66:67], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[64:65], v[64:65], v[130:131] op_sel_hi:[1,0]
.LBB0_42:
	s_wait_alu depctr_va_vdst(0)
	scratch_load_b64 v[234:235], off, off nv
	s_set_vgpr_msb 1
	v_mov_b32_e32 v236, v177 /*v433*/
	s_and_b32 s2, s5, exec_lo
	s_cselect_b32 s2, 1, 0
	s_cmp_lg_u32 s2, 1
	v_sub_f32_e32 v132, v179 /*v435*/, v236
	v_mul_f32_e32 v132, 0x3fb8aa3b, v132
	s_set_vgpr_msb 0x100
	v_exp_f32_e32 v132, v132
	s_cbranch_scc1 .LBB0_37
	v_nop
	v_pk_mul_f32 v[62:63], v[62:63], v[132:133] op_sel_hi:[1,0]
	v_pk_mul_f32 v[60:61], v[60:61], v[132:133] op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59], v[58:59], v[132:133] op_sel_hi:[1,0]
	v_pk_mul_f32 v[56:57], v[56:57], v[132:133] op_sel_hi:[1,0]
	v_pk_mul_f32 v[54:55], v[54:55], v[132:133] op_sel_hi:[1,0]
	v_pk_mul_f32 v[52:53], v[52:53], v[132:133] op_sel_hi:[1,0]
	v_pk_mul_f32 v[50:51], v[50:51], v[132:133] op_sel_hi:[1,0]
	v_pk_mul_f32 v[48:49], v[48:49], v[132:133] op_sel_hi:[1,0]
	v_pk_mul_f32 v[46:47], v[46:47], v[132:133] op_sel_hi:[1,0]
	v_pk_mul_f32 v[44:45], v[44:45], v[132:133] op_sel_hi:[1,0]
	v_pk_mul_f32 v[42:43], v[42:43], v[132:133] op_sel_hi:[1,0]
	v_pk_mul_f32 v[40:41], v[40:41], v[132:133] op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39], v[38:39], v[132:133] op_sel_hi:[1,0]
	v_pk_mul_f32 v[36:37], v[36:37], v[132:133] op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35], v[34:35], v[132:133] op_sel_hi:[1,0]
	v_pk_mul_f32 v[32:33], v[32:33], v[132:133] op_sel_hi:[1,0]
	v_pk_mul_f32 v[30:31], v[30:31], v[132:133] op_sel_hi:[1,0]
	v_pk_mul_f32 v[28:29], v[28:29], v[132:133] op_sel_hi:[1,0]
	v_pk_mul_f32 v[26:27], v[26:27], v[132:133] op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25], v[24:25], v[132:133] op_sel_hi:[1,0]
	v_pk_mul_f32 v[22:23], v[22:23], v[132:133] op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21], v[20:21], v[132:133] op_sel_hi:[1,0]
	v_pk_mul_f32 v[18:19], v[18:19], v[132:133] op_sel_hi:[1,0]
	v_pk_mul_f32 v[16:17], v[16:17], v[132:133] op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[14:15], v[132:133] op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13], v[12:13], v[132:133] op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[10:11], v[132:133] op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[8:9], v[132:133] op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[6:7], v[132:133] op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[4:5], v[132:133] op_sel_hi:[1,0]
	v_pk_mul_f32 v[2:3], v[2:3], v[132:133] op_sel_hi:[1,0]
	v_pk_mul_f32 v[0:1], v[0:1], v[132:133] op_sel_hi:[1,0]
	s_branch .LBB0_37
.LBB0_44:
	s_clause 0x3
	scratch_load_b32 v154, off, off offset:916 nv
	scratch_load_b32 v155, off, off offset:920 nv
	scratch_load_b32 v156, off, off offset:928 nv
	scratch_load_b32 v157, off, off offset:936 nv
	s_wait_alu depctr_va_vdst(0)
	scratch_load_b32 v134, off, off offset:940 nv
.LBB0_45:
	s_wait_loadcnt 0x1
	s_wait_alu depctr_vm_vsrc(1)
	v_nop
	v_nop
	v_nop
	v_nop
	v_div_scale_f32 v129, null, v234, v234, 1.0
	v_div_scale_f32 v132, vcc_lo, 1.0, v234, 1.0
	v_div_scale_f32 v150, null, v235, v235, 1.0
	s_lshl_b32 s2, s10, 17
	v_mul_u32_u24_e32 v128, 0x110, v156
	s_wait_loadcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	v_rcp_f32_e32 v130, v129
	s_and_b32 s4, s2, 0x20000
	v_rcp_f32_e32 v151, v150
	v_cmp_lt_f32_e64 s2, 0, v234
	v_fma_f32 v131, -v129, v130, 1.0
	v_fmac_f32_e32 v130, v131, v130
	v_mul_f32_e32 v131, v132, v130
	v_fma_f32 v133, -v129, v131, v132
	v_fmac_f32_e32 v131, v133, v130
	v_fma_f32 v129, -v129, v131, v132
	v_div_fmas_f32 v129, v129, v130, v131
	v_fma_f32 v130, -v150, v151, 1.0
	v_div_scale_f32 v152, vcc_lo, 1.0, v235, 1.0
	v_div_fixup_f32 v129, v129, v234, 1.0
	v_dual_fmac_f32 v151, v130, v151 :: v_dual_cndmask_b32 v130, 0, v129, s2
	s_add_co_i32 s2, s4, s89
	v_dual_mul_f32 v153, v152, v151 :: v_dual_add_nc_u32 v129, s2, v134
	s_mov_b32 s4, 0
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
	v_pk_mul_f32 v[110:111], v[110:111], v[130:131] op_sel_hi:[1,0]
	v_cvt_pk_bf16_f32 v76, v96, v97
	v_pk_mul_f32 v[138:139], v[130:131], v[74:75] op_sel_hi:[0,1]
	v_pk_mul_f32 v[112:113], v[112:113], v[130:131] op_sel_hi:[1,0]
	v_div_fmas_f32 v88, v90, v151, v153
	v_cmp_lt_f32_e32 vcc_lo, 0, v235
	v_cvt_pk_bf16_f32 v75, v110, v111
	v_pk_mul_f32 v[114:115], v[114:115], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[104:105], v[104:105], v[130:131] op_sel_hi:[1,0]
	v_div_fixup_f32 v92, v88, v235, 1.0
	v_pk_mul_f32 v[106:107], v[106:107], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[108:109], v[108:109], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[98:99], v[98:99], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[100:101], v[100:101], v[130:131] op_sel_hi:[1,0]
	v_cndmask_b32_e32 v96, 0, v92, vcc_lo
	v_pk_mul_f32 v[102:103], v[102:103], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[120:121], v[120:121], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[122:123], v[122:123], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[124:125], v[124:125], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[110:111], v[96:97], v[0:1] op_sel_hi:[0,1]
	s_wait_alu depctr_va_vdst(0)
	scratch_load_b32 v0, off, off offset:932 th:TH_LOAD_LU nv
	v_pk_mul_f32 v[126:127], v[126:127], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[116:117], v[116:117], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[118:119], v[118:119], v[130:131] op_sel_hi:[1,0]
	v_pk_mul_f32 v[94:95], v[130:131], v[94:95] op_sel_hi:[0,1]
	v_pk_mul_f32 v[84:85], v[130:131], v[84:85] op_sel_hi:[0,1]
	v_pk_mul_f32 v[86:87], v[130:131], v[86:87] op_sel_hi:[0,1]
	v_pk_mul_f32 v[136:137], v[130:131], v[72:73] op_sel_hi:[0,1]
	v_pk_mul_f32 v[142:143], v[130:131], v[78:79] op_sel_hi:[0,1]
	v_pk_mul_f32 v[144:145], v[130:131], v[64:65] op_sel_hi:[0,1]
	v_pk_mul_f32 v[146:147], v[130:131], v[66:67] op_sel_hi:[0,1]
	v_pk_mul_f32 v[148:149], v[130:131], v[68:69] op_sel_hi:[0,1]
	v_pk_mul_f32 v[130:131], v[130:131], v[70:71] op_sel_hi:[0,1]
	v_cvt_pk_bf16_f32 v69, v114, v115
	v_cvt_pk_bf16_f32 v68, v112, v113
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
	s_wait_loadcnt_dscnt 0x0
	v_add3_u32 v116, v128, s2, v0
	v_cvt_pk_bf16_f32 v0, v56, v57
	s_lshl_b32 s5, s57, 2
	s_wait_alu depctr_va_vdst(14)
	ds_store_b128 v129, v[64:67]
	ds_store_b128 v129, v[68:71] offset:32
	ds_store_b128 v129, v[72:75] offset:64
	ds_store_b128 v129, v[76:79] offset:96
	ds_store_b128 v129, v[80:83] offset:128
	ds_store_b128 v129, v[84:87] offset:160
	ds_store_b128 v129, v[88:91] offset:192
	ds_store_b128 v129, v[92:95] offset:224
	s_wait_alu depctr_va_vdst(0)
	ds_store_b128 v116, v[0:3] offset:4352
	ds_store_b128 v116, v[4:7] offset:4384
	ds_store_b128 v116, v[8:11] offset:4416
	ds_store_b128 v116, v[12:15] offset:4448
	s_sub_co_i32 s5, s5, s88
	ds_store_b128 v116, v[16:19] offset:4480
	ds_store_b128 v116, v[20:23] offset:4512
	ds_store_b128 v116, v[24:27] offset:4544
	ds_store_b128 v116, v[28:31] offset:4576
	s_wait_dscnt 0x0
	scratch_load_b32 v57, off, off offset:924 nv
	s_cmp_lt_i32 s5, 1
	s_cbranch_scc1 .LBB0_47
	s_wait_loadcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	v_or_b32_e32 v0, 28, v57
	s_add_co_i32 s5, s5, -1
	v_lshrrev_b32_e32 v26, 4, v157
	s_min_u32 s5, s5, 31
	v_lshl_add_u32 v32, v156, 4, s2
	v_min_u32_e32 v2, s5, v0
	v_lshlrev_b32_e32 v0, 3, v156
	s_load_b64 s[6:7], s[0:1], 0x0 nv
	v_or_b32_e32 v5, s88, v2
	v_or_b32_e32 v1, 30, v26
	v_mad_u32_u24 v33, 0x110, v2, v32
	v_dual_ashrrev_i32 v7, 31, v5 :: v_dual_bitop2_b32 v3, 26, v26 bitop3:0x54
	v_min_u32_e32 v4, s5, v1
	v_dual_mov_b32 v1, 0 :: v_dual_lshrrev_b32 v7, 30, v7
	v_min_u32_e32 v8, s5, v3
	v_or_b32_e32 v3, s88, v4
	v_mad_u32_u24 v34, 0x110, v4, v32
	v_mad_u32_u24 v35, 0x110, v8, v32
	v_dual_ashrrev_i32 v9, 31, v3 :: v_dual_bitop2_b32 v6, 24, v57 bitop3:0x54
	v_dual_add_nc_u32 v7, v5, v7 :: v_dual_lshrrev_b32 v9, 30, v9
	v_min_u32_e32 v14, s5, v6
	v_or_b32_e32 v6, s88, v8
	v_and_b32_e32 v4, -4, v7
	v_dual_ashrrev_i32 v7, 2, v7 :: v_dual_bitop2_b32 v11, s88, v14 bitop3:0x54
	v_ashrrev_i32_e32 v2, 31, v6
	v_dual_add_nc_u32 v9, v3, v9 :: v_dual_bitop2_b32 v10, 22, v26 bitop3:0x54
	v_mad_u32_u24 v36, 0x110, v14, v32
	v_cmp_ne_u32_e32 vcc_lo, v5, v4
	v_lshrrev_b32_e32 v2, 30, v2
	v_min_u32_e32 v15, s5, v10
	v_add_nc_u32_e32 v7, s33, v7
	s_and_b32 s9, s59, vcc_lo
	v_dual_add_nc_u32 v10, v6, v2 :: v_dual_bitop2_b32 v2, -4, v9 bitop3:0x40
	v_dual_sub_nc_u32 v4, v5, v4 :: v_dual_ashrrev_i32 v9, 2, v9
	v_cndmask_b32_e64 v13, 0, 1, s9
	v_and_b32_e32 v12, -4, v10
	v_sub_nc_u32_e32 v5, v3, v2
	v_cmp_ne_u32_e64 s2, v3, v2
	v_dual_add_nc_u32 v2, s34, v4 :: v_dual_add_nc_u32 v9, s33, v9
	v_dual_ashrrev_i32 v17, 31, v11 :: v_dual_sub_nc_u32 v7, v7, v13
	v_add_nc_u32_e32 v4, s34, v5
	s_and_b32 s2, s59, s2
	v_ashrrev_i32_e32 v10, 2, v10
	v_cndmask_b32_e64 v16, 0, 1, s2
	v_cmp_ne_u32_e32 vcc_lo, v6, v12
	v_mad_nc_i64_i32 v[4:5], v4, s28, v[0:1]
	v_dual_sub_nc_u32 v6, v6, v12 :: v_dual_bitop2_b32 v12, s88, v15 bitop3:0x54
	v_sub_nc_u32_e32 v9, v9, v16
	v_mad_nc_i64_i32 v[2:3], v2, s28, v[0:1]
	v_add_nc_u32_e32 v10, s33, v10
	s_and_b32 s2, s59, vcc_lo
	v_or_b32_e32 v16, 20, v57
	v_mad_nc_i64_i32 v[4:5], v9, s8, v[4:5]
	v_dual_lshrrev_b32 v9, 30, v17 :: v_dual_add_nc_u32 v6, s34, v6
	v_cndmask_b32_e64 v13, 0, 1, s2
	v_mad_nc_i64_i32 v[2:3], v7, s8, v[2:3]
	v_dual_ashrrev_i32 v17, 31, v12 :: v_dual_add_nc_u32 v9, v11, v9
	v_mad_nc_i64_i32 v[6:7], v6, s28, v[0:1]
	v_min_u32_e32 v16, s5, v16
	v_sub_nc_u32_e32 v10, v10, v13
	v_mad_u32_u24 v37, 0x110, v15, v32
	v_and_b32_e32 v8, -4, v9
	v_dual_lshrrev_b32 v13, 30, v17 :: v_dual_bitop2_b32 v17, s88, v16 bitop3:0x54
	v_mad_u32_u24 v38, 0x110, v16, v32
	v_mad_nc_i64_i32 v[6:7], v10, s8, v[6:7]
	v_cmp_ne_u32_e32 vcc_lo, v11, v8
	v_dual_sub_nc_u32 v10, v11, v8 :: v_dual_ashrrev_i32 v9, 2, v9
	s_wait_kmcnt 0x0
	v_lshl_add_u64 v[2:3], v[2:3], 1, s[6:7]
	v_lshl_add_u64 v[4:5], v[4:5], 1, s[6:7]
	s_and_b32 s2, s59, vcc_lo
	v_dual_add_nc_u32 v8, s34, v10 :: v_dual_add_nc_u32 v11, s33, v9
	v_cndmask_b32_e64 v18, 0, 1, s2
	v_add_nc_u32_e32 v13, v12, v13
	v_ashrrev_i32_e32 v9, 31, v17
	v_lshl_add_u64 v[6:7], v[6:7], 1, s[6:7]
	v_dual_sub_nc_u32 v11, v11, v18 :: v_dual_bitop2_b32 v10, -4, v13 bitop3:0x40
	v_lshrrev_b32_e32 v19, 30, v9
	v_mad_nc_i64_i32 v[8:9], v8, s28, v[0:1]
	v_ashrrev_i32_e32 v13, 2, v13
	v_cmp_ne_u32_e32 vcc_lo, v12, v10
	v_sub_nc_u32_e32 v10, v12, v10
	s_and_b32 s2, s59, vcc_lo
	v_mad_nc_i64_i32 v[8:9], v11, s8, v[8:9]
	v_dual_add_nc_u32 v18, v17, v19 :: v_dual_bitop2_b32 v11, 18, v26 bitop3:0x54
	v_add_nc_u32_e32 v13, s33, v13
	v_cndmask_b32_e64 v12, 0, 1, s2
	v_and_b32_e32 v19, -4, v18
	v_dual_add_nc_u32 v10, s34, v10 :: v_dual_sub_nc_u32 v20, v13, v12
	v_ashrrev_i32_e32 v12, 2, v18
	v_min_u32_e32 v18, s5, v11
	v_cmp_ne_u32_e32 vcc_lo, v17, v19
	v_sub_nc_u32_e32 v13, v17, v19
	v_mad_nc_i64_i32 v[10:11], v10, s28, v[0:1]
	v_lshl_add_u64 v[8:9], v[8:9], 1, s[6:7]
	v_or_b32_e32 v19, s88, v18
	s_and_b32 s2, s59, vcc_lo
	v_dual_add_nc_u32 v17, s33, v12 :: v_dual_add_nc_u32 v12, s34, v13
	v_dual_ashrrev_i32 v23, 31, v19 :: v_dual_bitop2_b32 v21, 16, v57 bitop3:0x54
	v_cndmask_b32_e64 v22, 0, 1, s2
	v_mad_nc_i64_i32 v[10:11], v20, s8, v[10:11]
	v_mad_nc_i64_i32 v[12:13], v12, s28, v[0:1]
	v_min_u32_e32 v21, s5, v21
	v_dual_lshrrev_b32 v20, 30, v23 :: v_dual_sub_nc_u32 v17, v17, v22
	v_mad_u32_u24 v39, 0x110, v18, v32
	v_dual_add_nc_u32 v14, v19, v20 :: v_dual_bitop2_b32 v22, s88, v21 bitop3:0x54
	v_mad_nc_i64_i32 v[12:13], v17, s8, v[12:13]
	v_mad_u32_u24 v40, 0x110, v21, v32
	v_lshl_add_u64 v[10:11], v[10:11], 1, s[6:7]
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
	s_and_b32 s2, s59, vcc_lo
	v_mad_nc_i64_i32 v[14:15], v14, s28, v[0:1]
	v_dual_lshrrev_b32 v25, 30, v27 :: v_dual_bitop2_b32 v27, 12, v57 bitop3:0x54
	v_cndmask_b32_e64 v28, 0, 1, s2
	v_mad_nc_i64_i32 v[16:17], v16, s28, v[0:1]
	v_dual_sub_nc_u32 v19, v19, v24 :: v_dual_add_nc_u32 v25, v20, v25
	v_min_u32_e32 v27, s5, v27
	v_sub_nc_u32_e32 v22, v22, v28
	v_mad_u32_u24 v42, 0x110, v23, v32
	v_mad_nc_i64_i32 v[14:15], v19, s8, v[14:15]
	v_and_b32_e32 v18, -4, v25
	v_dual_ashrrev_i32 v19, 2, v25 :: v_dual_bitop2_b32 v24, s88, v27 bitop3:0x54
	v_mad_nc_i64_i32 v[16:17], v22, s8, v[16:17]
	v_or_b32_e32 v28, 10, v26
	v_dual_sub_nc_u32 v22, v20, v18 :: v_dual_ashrrev_i32 v25, 31, v24
	v_cmp_ne_u32_e32 vcc_lo, v20, v18
	v_add_nc_u32_e32 v20, s33, v19
	v_mad_u32_u24 v44, 0x110, v27, v32
	v_dual_add_nc_u32 v18, s34, v22 :: v_dual_lshrrev_b32 v22, 30, v25
	v_min_u32_e32 v25, s5, v28
	s_and_b32 s2, s59, vcc_lo
	v_lshl_add_u64 v[14:15], v[14:15], 1, s[6:7]
	v_cndmask_b32_e64 v28, 0, 1, s2
	v_dual_add_nc_u32 v22, v24, v22 :: v_dual_bitop2_b32 v29, s88, v25 bitop3:0x54
	v_mad_nc_i64_i32 v[18:19], v18, s28, v[0:1]
	v_mad_u32_u24 v45, 0x110, v25, v32
	v_dual_sub_nc_u32 v20, v20, v28 :: v_dual_bitop2_b32 v21, -4, v22 bitop3:0x40
	v_ashrrev_i32_e32 v28, 31, v29
	v_lshl_add_u64 v[16:17], v[16:17], 1, s[6:7]
	v_mad_nc_i64_i32 v[18:19], v20, s8, v[18:19]
	v_dual_ashrrev_i32 v20, 2, v22 :: v_dual_sub_nc_u32 v22, v24, v21
	v_dual_lshrrev_b32 v28, 30, v28 :: v_dual_bitop2_b32 v30, 8, v57 bitop3:0x54
	v_cmp_ne_u32_e32 vcc_lo, v24, v21
	v_dual_add_nc_u32 v24, s33, v20 :: v_dual_add_nc_u32 v20, s34, v22
	v_add_nc_u32_e32 v22, v29, v28
	v_min_u32_e32 v41, s5, v30
	s_and_b32 s2, s59, vcc_lo
	v_lshl_add_u64 v[18:19], v[18:19], 1, s[6:7]
	v_cndmask_b32_e64 v28, 0, 1, s2
	v_and_b32_e32 v23, -4, v22
	v_mad_nc_i64_i32 v[20:21], v20, s28, v[0:1]
	v_dual_ashrrev_i32 v22, 2, v22 :: v_dual_bitop2_b32 v30, s88, v41 bitop3:0x54
	v_dual_sub_nc_u32 v24, v24, v28 :: v_dual_sub_nc_u32 v28, v29, v23
	v_cmp_ne_u32_e32 vcc_lo, v29, v23
	v_ashrrev_i32_e32 v31, 31, v30
	v_or_b32_e32 v29, 6, v26
	v_mad_nc_i64_i32 v[20:21], v24, s8, v[20:21]
	v_dual_add_nc_u32 v24, s33, v22 :: v_dual_add_nc_u32 v22, s34, v28
	s_and_b32 s2, s59, vcc_lo
	v_lshrrev_b32_e32 v28, 30, v31
	v_cndmask_b32_e64 v31, 0, 1, s2
	v_min_u32_e32 v43, s5, v29
	v_mad_nc_i64_i32 v[22:23], v22, s28, v[0:1]
	v_mad_u32_u24 v41, 0x110, v41, v32
	v_dual_add_nc_u32 v28, v30, v28 :: v_dual_sub_nc_u32 v24, v24, v31
	v_lshl_add_u64 v[20:21], v[20:21], 1, s[6:7]
	v_and_b32_e32 v27, -4, v28
	v_mad_nc_i64_i32 v[22:23], v24, s8, v[22:23]
	v_dual_ashrrev_i32 v24, 2, v28 :: v_dual_bitop2_b32 v29, s88, v43 bitop3:0x54
	v_mad_u32_u24 v43, 0x110, v43, v32
	v_sub_nc_u32_e32 v25, v30, v27
	v_cmp_ne_u32_e32 vcc_lo, v30, v27
	v_dual_add_nc_u32 v27, s33, v24 :: v_dual_ashrrev_i32 v28, 31, v29
	v_or_b32_e32 v31, 4, v57
	v_lshl_add_u64 v[22:23], v[22:23], 1, s[6:7]
	s_and_b32 s2, s59, vcc_lo
	v_lshrrev_b32_e32 v28, 30, v28
	v_min_u32_e32 v46, s5, v31
	v_add_nc_u32_e32 v24, s34, v25
	v_cndmask_b32_e64 v30, 0, 1, s2
	v_dual_add_nc_u32 v28, v29, v28 :: v_dual_bitop2_b32 v31, s88, v46 bitop3:0x54
	v_mad_nc_i64_i32 v[24:25], v24, s28, v[0:1]
	v_sub_nc_u32_e32 v27, v27, v30
	v_mad_u32_u24 v46, 0x110, v46, v32
	v_and_b32_e32 v30, -4, v28
	v_dual_ashrrev_i32 v28, 2, v28 :: v_dual_bitop2_b32 v26, 2, v26 bitop3:0x54
	v_ashrrev_i32_e32 v47, 31, v31
	v_cmp_ne_u32_e32 vcc_lo, v29, v30
	v_mad_nc_i64_i32 v[24:25], v27, s8, v[24:25]
	v_min_u32_e32 v48, s5, v26
	v_dual_add_nc_u32 v26, s33, v28 :: v_dual_lshrrev_b32 v27, 30, v47
	v_dual_sub_nc_u32 v29, v29, v30 :: v_dual_min_i32 v47, s5, v57
	s_and_b32 s2, s59, vcc_lo
	v_dual_add_nc_u32 v27, v31, v27 :: v_dual_bitop2_b32 v28, s88, v48 bitop3:0x54
	v_cndmask_b32_e64 v49, 0, 1, s2
	v_or_b32_e32 v30, s88, v47
	v_mad_u32_u24 v47, 0x110, v47, v32
	v_dual_ashrrev_i32 v50, 31, v28 :: v_dual_bitop2_b32 v51, -4, v27 bitop3:0x40
	v_dual_ashrrev_i32 v52, 2, v27 :: v_dual_sub_nc_u32 v49, v26, v49
	v_dual_add_nc_u32 v26, s34, v29 :: v_dual_lshrrev_b32 v50, 30, v50
	v_ashrrev_i32_e32 v29, 31, v30
	v_cmp_ne_u32_e32 vcc_lo, v31, v51
	v_dual_add_nc_u32 v52, s33, v52 :: v_dual_sub_nc_u32 v31, v31, v51
	v_dual_add_nc_u32 v50, v28, v50 :: v_dual_lshrrev_b32 v29, 30, v29
	s_and_b32 s2, s59, vcc_lo
	v_mad_nc_i64_i32 v[26:27], v26, s28, v[0:1]
	v_cndmask_b32_e64 v53, 0, 1, s2
	v_dual_add_nc_u32 v29, v30, v29 :: v_dual_bitop2_b32 v51, -4, v50 bitop3:0x40
	v_dual_add_nc_u32 v31, s34, v31 :: v_dual_ashrrev_i32 v50, 2, v50
	v_mad_u32_u24 v32, 0x110, v48, v32
	v_cmp_ne_u32_e32 vcc_lo, v28, v51
	v_dual_sub_nc_u32 v28, v28, v51 :: v_dual_bitop2_b32 v54, -4, v29 bitop3:0x40
	v_dual_ashrrev_i32 v29, 2, v29 :: v_dual_add_nc_u32 v50, s33, v50
	v_mad_nc_i64_i32 v[26:27], v49, s8, v[26:27]
	v_sub_nc_u32_e32 v51, v30, v54
	v_cmp_ne_u32_e64 s2, v30, v54
	v_dual_add_nc_u32 v54, s34, v28 :: v_dual_add_nc_u32 v55, s33, v29
	v_mad_nc_i64_i32 v[30:31], v31, s28, v[0:1]
	v_add_nc_u32_e32 v28, s34, v51
	s_and_b32 s2, s59, s2
	v_lshl_add_u64 v[26:27], v[26:27], 1, s[6:7]
	v_cndmask_b32_e64 v51, 0, 1, s2
	s_and_b32 s2, s59, vcc_lo
	v_mad_nc_i64_i32 v[28:29], v28, s28, v[0:1]
	v_cndmask_b32_e64 v56, 0, 1, s2
	v_mad_nc_i64_i32 v[0:1], v54, s28, v[0:1]
	v_sub_nc_u32_e32 v51, v55, v51
	v_lshl_add_u64 v[24:25], v[24:25], 1, s[6:7]
	v_dual_sub_nc_u32 v49, v50, v56 :: v_dual_sub_nc_u32 v50, v52, v53
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
.LBB0_47:
	s_wait_alu depctr_vm_vsrc(0)
	s_clause 0x1
	scratch_load_b32 v0, off, off offset:912 th:TH_LOAD_LU nv
	scratch_load_b32 v5, off, off offset:908 th:TH_LOAD_LU nv
	v_cmp_gt_f32_e32 vcc_lo, 0x800000, v234
	v_cmp_gt_f32_e64 s2, 0x800000, v235
	s_load_b64 s[6:7], s[0:1], 0x40 nv
	s_mul_i32 s8, s3, s31
	s_wait_xcnt 0x0
	v_cmp_gt_i32_e64 s0, s57, v154
	v_cndmask_b32_e64 v2, 0, 32, vcc_lo
	v_cndmask_b32_e64 v4, 0, 32, s2
	v_cndmask_b32_e64 v1, 0, 0x42000000, vcc_lo
	v_cndmask_b32_e64 v3, 0, 0x42000000, s2
	v_mad_u32 v6, v154, s29, s8
	v_ldexp_f32 v2, v234, v2
	v_ldexp_f32 v4, v235, v4
	s_wait_loadcnt 0x2
	v_cmp_eq_u32_e32 vcc_lo, 0, v57
	v_cmp_gt_i32_e64 s1, s57, v155
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
	v_add_f32_e32 v1, v1, v55 /*v311*/
	s_set_vgpr_msb 0x400
	s_wait_loadcnt 0x1
	v_lshlrev_b32_e32 v0, 2, v0
	s_wait_loadcnt 0x0
	v_sub_nc_u32_e32 v0, v5, v0
	v_mul_lo_u32 v5, v155, s29
	v_add_nc_u32_e32 v0, s34, v0
	v_mul_lo_u32 v0, v0, s30
	v_add_nc_u32_e32 v3, s8, v0
	v_add_lshl_u32 v0, v6, v0, 2
	v_add_lshl_u32 v3, v3, v5, 2
	v_cndmask_b32_e64 v0, 0x7fffffff, v0, s0
	v_add_f32_e32 v2, v2, v236
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
		.amdhsa_private_segment_fixed_size 960
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
		.amdhsa_next_free_vgpr 512
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

	.set .Lkn_fmha_fwd_prefill_a16w16_m32x8_bshd_0.num_vgpr, 512
	.set .Lkn_fmha_fwd_prefill_a16w16_m32x8_bshd_0.num_agpr, 0
	.set .Lkn_fmha_fwd_prefill_a16w16_m32x8_bshd_0.numbered_sgpr, 97
	.set .Lkn_fmha_fwd_prefill_a16w16_m32x8_bshd_0.num_named_barrier, 0
	.set .Lkn_fmha_fwd_prefill_a16w16_m32x8_bshd_0.private_seg_size, 960
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
    .max_flat_workgroup_size: 256
    .name:           kn_fmha_fwd_prefill_a16w16_m32x8_bshd_0
    .private_segment_fixed_size: 960
    .reqd_workgroup_size:
      - 256
      - 1
      - 1
    .sgpr_count:     99
    .sgpr_spill_count: 0
    .symbol:         kn_fmha_fwd_prefill_a16w16_m32x8_bshd_0.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     512
    .vgpr_spill_count: 642
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa-unknown-gfx1250
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
