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
	s_load_b96 s[12:14], s[0:1], 0xa0 nv
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
	s_load_b512 s[48:63], s[0:1], 0x5c nv
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
	s_clause 0x2
	s_load_b64 s[14:15], s[0:1], 0x10 nv
	s_load_b64 s[18:19], s[0:1], 0x20 nv
	s_load_b64 s[16:17], s[0:1], 0x30 nv
	s_mov_b32 s80, 1
	s_mov_b32 s65, 0xffff0000
	s_mov_b32 s68, 0x80004
	s_mov_b32 s67, 0x807fff
	s_mov_b32 s66, 0xffff7fff
	s_mul_f32 s7, s7, 0x4f7ffffe
	s_mov_b32 s22, s53
	s_mov_b32 s64, 0x7510000
	s_set_vgpr_msb 0x80
	v_and_b32_e32 v135 /*v647*/, 15, v0
	s_cvt_u32_f32 s7, s7
	s_set_vgpr_msb 0x8000
	v_and_b32_e32 v1, 16, v0
	s_mov_b32 s90, s51
	s_mov_b32 s72, 0xf510000
	s_mul_i32 s5, s5, s7
	s_mov_b32 s73, s65
	s_mul_hi_u32 s2, s7, s5
	s_abs_i32 s5, s3
	s_add_co_i32 s7, s7, s2
	s_set_vgpr_msb 0x88
	v_mad_u32_u24 v203 /*v715*/, 0x110, v135 /*v647*/, v1
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
	s_sub_co_ci_u32 s4, s2, s9
	s_abs_i32 s2, s13
	s_mul_i32 s6, s4, s6
	s_cvt_f32_u32 s5, s2
	s_sub_co_i32 s8, 0, s2
	s_sub_co_i32 s3, s3, s6
	v_s_rcp_f32 s7, s5
	s_xor_b32 s20, s3, s13
	s_mov_b32 s5, 0
	s_ashr_i32 s21, s20, 31
	s_mov_b32 s11, s5
	s_mov_b32 s79, s5
	s_mul_f32 s7, s7, 0x4f7ffffe
	s_cvt_u32_f32 s7, s7
	s_mul_i32 s8, s8, s7
	s_mul_hi_u32 s6, s7, s8
	s_abs_i32 s8, s3
	s_add_co_i32 s7, s7, s6
	s_mul_hi_u32 s6, s8, s7
	s_mul_i32 s7, s6, s2
	s_sub_co_i32 s7, s8, s7
	s_add_co_i32 s8, s6, 1
	s_sub_co_i32 s9, s7, s2
	s_cmp_ge_u32 s7, s2
	s_cselect_b32 s8, s8, s6
	s_cselect_b32 s6, s9, s7
	s_add_co_i32 s7, s8, 1
	s_cmp_ge_u32 s6, s2
	s_mov_b32 s6, s5
	s_cselect_b32 s2, s7, s8
	s_mov_b32 s7, s5
	s_xor_b32 s2, s2, s21
	s_mov_b32 s8, s5
	s_sub_co_i32 s10, s2, s21
	s_mov_b32 s9, s5
	s_mul_i32 s13, s10, s13
	s_mov_b32 s10, s5
	s_cmp_lg_u32 s3, s13
	s_cselect_b32 s23, -1, 0
	s_cmp_lt_i32 s20, 0
	s_mov_b32 s20, s49
	s_cselect_b32 s24, -1, 0
	s_and_b32 s23, s23, s24
	s_sub_co_ci_u32 s30, s2, s21
	s_sub_co_i32 s2, s3, s13
	s_not_b32 s3, s4
	s_bfe_u32 s29, ttmp8, 0x50019
	s_add_co_i32 s3, s12, s3
	s_lshl_b32 s4, s29, 5
	s_lshl_b32 s27, s3, 7
	s_mul_i32 s93, s30, s62
	s_add_co_i32 s104, s27, s4
	s_lshl_b32 s88, s2, 2
	s_ashr_i32 s4, s104, 2
	s_ashr_i32 s21, s49, 31
	s_add_co_i32 s24, s93, s4
	s_ashr_i32 s23, s53, 31
	s_mul_i32 s12, s29, 0x2200
	s_ashr_i32 s89, s88, 31
	s_ashr_i32 s25, s24, 31
	s_set_vgpr_msb 0x8840
	v_writelane_b32 v0 /*v256*/, s12, 0
	s_add_co_i32 s81, s12, 0x46000
	s_mul_u64 s[12:13], s[88:89], s[22:23]
	s_sub_co_i32 s4, s62, s4
	s_mul_u64 s[24:25], s[24:25], s[20:21]
	s_lshl_b64 s[12:13], s[12:13], 1
	s_max_i32 s4, s4, 0
	s_lshl_b64 s[24:25], s[24:25], 1
	s_cmp_lg_u32 s49, 0x80000000
	s_wait_kmcnt 0x0
	s_add_nc_u64 s[14:15], s[14:15], s[24:25]
	s_cselect_b32 s21, s21, 0
	s_cselect_b32 s20, s49, 0x200
	s_cmp_lg_u32 s53, 0x80000000
	s_add_nc_u64 s[82:83], s[14:15], s[12:13]
	s_cselect_b32 s69, s53, 0x80
	s_cselect_b32 s14, s23, 0
	s_lshr_b64 s[12:13], s[20:21], 16
	s_lshl_b32 s15, s20, 16
	s_lshr_b32 s13, s20, 16
	s_and_b32 s14, s14, 0xffff
	s_and_b32 s12, s12, 0xffff0000
	s_bitset1_b32 s83, 31
	s_or_b32 s70, s14, s15
	s_or_b32 s71, s12, s13
	s_add_co_i32 s20, s62, -1
	tensor_load_to_lds s[80:83], s[64:71], s[4:7], s[8:11]
	s_bfe_i32 s4, s3, 0x10018
	s_mov_b32 s6, s54
	s_lshr_b32 s10, s4, 30
	s_sub_co_i32 s26, s63, s62
	s_or_b32 s10, s10, s27
	s_ashr_i32 s24, s27, 2
	s_addk_co_i32 s10, 0x7f
	s_ashr_i32 s3, s2, 31
	s_ashr_i32 s12, s10, 2
	s_ashr_i32 s7, s54, 31
	s_add_co_i32 s4, s12, s4
	s_mul_u64 s[6:7], s[2:3], s[6:7]
	s_min_i32 s27, s4, s20
	s_mov_b32 s8, s55
	s_add_co_i32 s27, s27, s26
	s_ashr_i32 s9, s55, 31
	s_lshl_b64 s[20:21], s[6:7], 1
	s_add_co_i32 s6, s61, s27
	s_mul_u64 s[2:3], s[2:3], s[8:9]
	s_add_co_i32 s6, s6, 1
	s_lshl_b64 s[22:23], s[2:3], 1
	s_min_i32 s3, s6, s63
	s_add_co_i32 s28, s24, s26
	s_max_i32 s89, s3, 1
	s_sub_co_i32 s4, s28, s60
	s_add_co_i32 s3, s89, 0x7f
	s_max_i32 s2, s4, 0
	s_lshr_b32 s54, s3, 7
	s_lshr_b32 s2, s2, 7
	s_add_co_i32 s25, s54, -1
	s_mul_i32 s53, s30, s63
	s_min_u32 s92, s2, s25
	s_mov_b32 s82, s50
	s_lshl_b32 s3, s92, 7
	s_ashr_i32 s83, s50, 31
	s_add_co_i32 s6, s3, s53
	s_sub_co_i32 s3, s89, s3
	s_ashr_i32 s91, s51, 31
	s_ashr_i32 s7, s6, 31
	s_mov_b64 s[10:11], s[82:83]
	s_set_vgpr_msb 0x4000
	v_med3_i32 v1, s3, 0, 0x80
	s_mul_u64 s[10:11], s[6:7], s[82:83]
	s_mul_u64 s[6:7], s[6:7], s[90:91]
	s_add_co_i32 s2, s92, 1
	s_lshl_b64 s[10:11], s[10:11], 1
	s_lshl_b64 s[6:7], s[6:7], 1
	s_cmp_lg_u32 s50, 0x80000000
	s_mov_b64 s[8:9], s[80:81]
	s_mov_b64 s[14:15], s[82:83]
	v_readfirstlane_b32 s3, v1
	s_cselect_b32 s15, s83, 0
	s_cselect_b32 s69, s50, 0x80
	s_and_b32 s9, s29, 3
	s_and_b32 s70, s15, 0xffff
	s_lshl_b32 vcc_hi, s9, 5
	s_lshl_b32 s4, s9, 6
	s_sub_co_i32 s3, s3, vcc_hi
	s_mov_b32 s14, s69
	s_max_i32 s3, s3, 0
	s_add_nc_u64 s[10:11], s[18:19], s[10:11]
	s_lshl_b32 s3, s3, 16
	s_add_nc_u64 s[6:7], s[16:17], s[6:7]
	s_or_b32 s66, s3, 0x7fff
	s_cmp_lg_u32 s51, 0x80000000
	s_mul_u64 s[94:95], s[14:15], s[4:5]
	s_cselect_b32 s77, s51, 0x80
	s_cselect_b32 s15, s91, 0
	s_mov_b32 s14, s77
	s_add_nc_u64 s[10:11], s[10:11], s[20:21]
	s_set_vgpr_msb 0x88
	v_add_nc_u32_e32 v202 /*v714*/, s81, v203 /*v715*/
	s_add_nc_u64 s[6:7], s[6:7], s[22:23]
	s_mul_i32 s103, s9, 0x2400
	s_mul_u64 s[50:51], s[14:15], s[4:5]
	s_mul_i32 s102, s9, 0x2200
	s_add_nc_u64 s[10:11], s[94:95], s[10:11]
	s_mov_b32 s68, 32
	s_mov_b32 s67, 0x800000
	s_add_co_i32 s103, s103, 0x8800
	s_add_nc_u64 s[6:7], s[50:51], s[6:7]
	s_mov_b64 s[12:13], s[80:81]
	s_mov_b32 s71, s5
	s_mov_b32 s9, s102
	s_bitset1_b32 s11, 31
	s_mov_b32 s75, s67
	s_mov_b32 s76, s68
	s_mov_b32 s74, s66
	s_mov_b32 s13, s103
	s_and_b32 s78, s15, 0xffff
	s_or_b32 s15, s7, 0x80000000
	s_mov_b32 s14, s6
	s_cmp_ge_u32 s2, s54
	s_wait_tensorcnt 0x0
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x8802
	ds_load_b128 v[58:61], v202 /*v714*/
	ds_load_b128 v[62:65], v202 /*v714*/ offset:32
	ds_load_b128 v[50:53], v202 /*v714*/ offset:64
	ds_load_b128 v[54:57], v202 /*v714*/ offset:96
	ds_load_b128 v[42:45], v202 /*v714*/ offset:128
	ds_load_b128 v[46:49], v202 /*v714*/ offset:160
	ds_load_b128 v[34:37], v202 /*v714*/ offset:192
	ds_load_b128 v[38:41], v202 /*v714*/ offset:224
	ds_load_b128 v[26:29], v202 /*v714*/ offset:4352
	ds_load_b128 v[30:33], v202 /*v714*/ offset:4384
	ds_load_b128 v[18:21], v202 /*v714*/ offset:4416
	ds_load_b128 v[22:25], v202 /*v714*/ offset:4448
	ds_load_b128 v[10:13], v202 /*v714*/ offset:4480
	ds_load_b128 v[14:17], v202 /*v714*/ offset:4512
	ds_load_b128 v[2:5], v202 /*v714*/ offset:4544
	ds_load_b128 v[6:9], v202 /*v714*/ offset:4576
	tensor_load_to_lds s[8:11], s[64:71]
	tensor_load_to_lds s[12:15], s[72:79]
	s_set_vgpr_msb 0x200
	s_cbranch_scc1 .LBB0_2
	s_lshl_b32 s3, s2, 7
	s_mov_b64 s[10:11], s[82:83]
	s_sub_co_i32 s6, s89, s3
	s_add_co_i32 s2, s3, s53
	v_med3_i32 v1, s6, 0, 0x80
	s_ashr_i32 s3, s2, 31
	s_add_co_i32 s4, s102, 0x11800
	s_mul_u64 s[6:7], s[2:3], s[82:83]
	s_mul_u64 s[2:3], s[2:3], s[90:91]
	v_readfirstlane_b32 s12, v1
	s_lshl_b64 s[6:7], s[6:7], 1
	s_lshl_b64 s[2:3], s[2:3], 1
	s_add_nc_u64 s[6:7], s[18:19], s[6:7]
	s_add_nc_u64 s[2:3], s[16:17], s[2:3]
	s_add_nc_u64 s[6:7], s[6:7], s[20:21]
	s_sub_co_i32 s10, s12, vcc_hi
	s_mov_b64 s[8:9], s[80:81]
	s_add_nc_u64 s[6:7], s[94:95], s[6:7]
	s_mov_b32 s9, s4
	s_max_i32 s4, s10, 0
	s_add_nc_u64 s[2:3], s[2:3], s[22:23]
	s_bitset1_b32 s7, 31
	s_lshl_b32 s4, s4, 16
	s_add_nc_u64 s[2:3], s[50:51], s[2:3]
	s_mov_b32 s10, s6
	s_mov_b32 s11, s7
	s_or_b32 s66, s4, 0x7fff
	s_mov_b32 s71, s5
	s_add_co_i32 s4, s103, 0x11800
	s_bitset1_b32 s3, 31
	tensor_load_to_lds s[8:11], s[64:71]
	s_mov_b64 s[8:9], s[80:81]
	s_mov_b64 s[10:11], s[82:83]
	s_mov_b32 s9, s4
	s_mov_b32 s10, s2
	s_mov_b32 s11, s3
	s_mov_b32 s73, s65
	s_mov_b32 s74, s66
	s_mov_b32 s75, s67
	s_mov_b32 s76, s68
	s_mov_b32 s79, s5
	tensor_load_to_lds s[8:11], s[72:79]
.LBB0_2:
	s_wait_tensorcnt 0x0
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_add_co_i32 s2, s92, 2
	s_cmp_ge_u32 s2, s54
	s_barrier_wait -1
	s_cbranch_scc1 .LBB0_4
	s_lshl_b32 s3, s2, 7
	s_mov_b32 s4, 1
	s_sub_co_i32 s5, s89, s3
	s_add_co_i32 s2, s3, s53
	v_med3_i32 v1, s5, 0, 0x80
	s_ashr_i32 s3, s2, 31
	s_mov_b32 s71, 0
	s_mul_u64 s[6:7], s[2:3], s[90:91]
	s_mul_u64 s[2:3], s[2:3], s[82:83]
	v_readfirstlane_b32 s5, v1
	s_lshl_b64 s[2:3], s[2:3], 1
	s_lshl_b64 s[6:7], s[6:7], 1
	s_add_nc_u64 s[2:3], s[18:19], s[2:3]
	s_add_nc_u64 s[6:7], s[16:17], s[6:7]
	s_add_nc_u64 s[2:3], s[2:3], s[20:21]
	s_sub_co_i32 s5, s5, vcc_hi
	s_add_nc_u64 s[8:9], s[6:7], s[22:23]
	s_add_nc_u64 s[6:7], s[94:95], s[2:3]
	s_max_i32 s2, s5, 0
	s_add_co_i32 s5, s102, 0x23000
	s_lshl_b32 s2, s2, 16
	s_bitset1_b32 s7, 31
	s_or_b32 s66, s2, 0x7fff
	s_mov_b32 s73, s65
	tensor_load_to_lds s[4:7], s[64:71]
	s_add_nc_u64 s[6:7], s[50:51], s[8:9]
	s_add_co_i32 s5, s103, 0x23000
	s_bitset1_b32 s7, 31
	s_mov_b32 s74, s66
	s_mov_b32 s75, s67
	s_mov_b32 s76, s68
	s_mov_b32 s79, s71
	tensor_load_to_lds s[4:7], s[72:79]
.LBB0_4:
	v_and_b32_e32 v0, 31, v0
	s_add_co_i32 s2, s92, 3
	s_cmp_ge_u32 s2, s54
	s_cbranch_scc1 .LBB0_6
	s_lshl_b32 s3, s2, 7
	s_mov_b32 s4, 1
	s_sub_co_i32 s5, s89, s3
	s_add_co_i32 s2, s3, s53
	v_med3_i32 v1, s5, 0, 0x80
	s_ashr_i32 s3, s2, 31
	s_mov_b32 s71, 0
	s_mul_u64 s[6:7], s[2:3], s[90:91]
	s_mul_u64 s[2:3], s[2:3], s[82:83]
	v_readfirstlane_b32 s5, v1
	s_lshl_b64 s[2:3], s[2:3], 1
	s_lshl_b64 s[6:7], s[6:7], 1
	s_add_nc_u64 s[2:3], s[18:19], s[2:3]
	s_add_nc_u64 s[6:7], s[16:17], s[6:7]
	s_add_nc_u64 s[2:3], s[2:3], s[20:21]
	s_sub_co_i32 s5, s5, vcc_hi
	s_add_nc_u64 s[8:9], s[6:7], s[22:23]
	s_add_nc_u64 s[6:7], s[94:95], s[2:3]
	s_max_i32 s2, s5, 0
	s_add_co_i32 s5, s102, 0x34800
	s_lshl_b32 s2, s2, 16
	s_bitset1_b32 s7, 31
	s_or_b32 s66, s2, 0x7fff
	s_mov_b32 s73, s65
	tensor_load_to_lds s[4:7], s[64:71]
	s_add_nc_u64 s[6:7], s[50:51], s[8:9]
	s_add_co_i32 s5, s103, 0x34800
	s_bitset1_b32 s7, 31
	s_mov_b32 s74, s66
	s_mov_b32 s75, s67
	s_mov_b32 s76, s68
	s_mov_b32 s79, s71
	tensor_load_to_lds s[4:7], s[72:79]
.LBB0_6:
	s_set_vgpr_msb 8
	v_or_b32_e32 v1, s104, v135 /*v647*/
	s_load_b64 s[2:3], s[0:1], 0x50 nv
	s_cmp_lt_i32 s104, 0
	s_set_vgpr_msb 0x880
	v_lshrrev_b32_e32 v201 /*v713*/, 4, v0
	s_cselect_b32 s49, -1, 0
	s_set_vgpr_msb 0x8000
	v_ashrrev_i32_e32 v66, 31, v1
	s_add_co_i32 s28, s28, s61
	s_set_vgpr_msb 0x88
	v_add_nc_u32_e32 v210 /*v722*/, 0x11800, v203 /*v715*/
	v_lshlrev_b32_e32 v204 /*v716*/, 3, v201 /*v713*/
	v_add_nc_u32_e32 v10 /*v522*/, 0x23000, v203 /*v715*/
	s_set_vgpr_msb 0x8800
	v_lshrrev_b32_e32 v66, 30, v66
	s_set_vgpr_msb 0x88
	v_add_nc_u32_e32 v12 /*v524*/, 0x34800, v203 /*v715*/
	s_mov_b32 s71, 0
	s_mov_b32 s8, 1
	s_mov_b32 s13, s71
	s_set_vgpr_msb 0x8800
	v_add_nc_u32_e32 v66, v1, v66
	s_add_nc_u64 s[96:97], s[18:19], s[20:21]
	s_add_nc_u64 s[98:99], s[16:17], s[22:23]
	v_and_b32_e32 v67, -4, v66
	v_sub_nc_u32_e32 v68, v1, v67
	v_cmp_ne_u32_e32 vcc_lo, v1, v67
	s_set_vgpr_msb 64
	v_add_nc_u32_e32 v1 /*v257*/, s88, v68
	s_set_vgpr_msb 0x4000
	v_or_b32_e32 v68, 16, v1
	s_and_b32 vcc_lo, s49, vcc_lo
	s_wait_kmcnt 0x0
	s_wait_alu depctr_va_vdst(1)
	s_set_vgpr_msb 0x81
	global_load_b32 v211 /*v723*/, v1 /*v257*/, s[2:3] scale_offset
	s_wait_xcnt 0x0
	s_ashr_i32 s2, s104, 31
	s_lshr_b32 s2, s2, 30
	s_set_vgpr_msb 0x8100
	v_dual_add_nc_u32 v69, s2, v68 :: v_dual_ashrrev_i32 v1, 2, v66
	v_dual_ashrrev_i32 v66, 2, v69 :: v_dual_bitop2_b32 v70, -4, v69 bitop3:0x40
	s_set_vgpr_msb 0x80
	v_subrev_co_ci_u32_e64 v200 /*v712*/, null, 0, v1, vcc_lo
	s_set_vgpr_msb 0x8020
	v_cvt_pk_bf16_f32 v1, s48, s0
	v_cmp_ne_u32_e64 s2, v68, v70
	v_pk_mul_bf16 v135, v1, v65 op_sel_hi:[0,1]
	s_and_b32 vcc_lo, s49, s2
	s_max_i32 s2, s28, 0
	v_pk_mul_bf16 v134, v1, v64 op_sel_hi:[0,1]
	s_add_co_i32 s2, s2, 1
	v_pk_mul_bf16 v133, v1, v63 op_sel_hi:[0,1]
	s_ashr_i32 s3, s2, 31
	v_pk_mul_bf16 v132, v1, v62 op_sel_hi:[0,1]
	s_lshr_b32 s3, s3, 25
	v_pk_mul_bf16 v131, v1, v61 op_sel_hi:[0,1]
	s_add_co_i32 s3, s2, s3
	v_pk_mul_bf16 v130, v1, v60 op_sel_hi:[0,1]
	s_and_b32 s4, s3, 0xffffff80
	s_ashr_i32 s3, s3, 7
	s_cmp_lg_u32 s2, s4
	v_pk_mul_bf16 v129, v1, v59 op_sel_hi:[0,1]
	s_cselect_b32 s4, -1, 0
	s_cmp_lt_i32 s2, 0
	v_pk_mul_bf16 v128, v1, v58 op_sel_hi:[0,1]
	s_cselect_b32 s2, -1, 0
	v_pk_mul_bf16 v143, v1, v57 op_sel_hi:[0,1]
	s_and_b32 s2, s2, s4
	s_sub_co_ci_u32 s2, s3, 0
	s_sub_co_i32 s3, s27, s60
	v_pk_mul_bf16 v142, v1, v56 op_sel_hi:[0,1]
	s_max_i32 s3, s3, 0
	v_pk_mul_bf16 v141, v1, v55 op_sel_hi:[0,1]
	s_addk_co_i32 s3, 0x7f
	v_pk_mul_bf16 v140, v1, v54 op_sel_hi:[0,1]
	s_ashr_i32 s4, s3, 31
	v_pk_mul_bf16 v139, v1, v53 op_sel_hi:[0,1]
	s_lshr_b32 s4, s4, 25
	v_pk_mul_bf16 v138, v1, v52 op_sel_hi:[0,1]
	v_pk_mul_bf16 v137, v1, v51 op_sel_hi:[0,1]
	v_pk_mul_bf16 v136, v1, v50 op_sel_hi:[0,1]
	v_pk_mul_bf16 v151, v1, v49 op_sel_hi:[0,1]
	v_pk_mul_bf16 v150, v1, v48 op_sel_hi:[0,1]
	v_pk_mul_bf16 v149, v1, v47 op_sel_hi:[0,1]
	v_pk_mul_bf16 v148, v1, v46 op_sel_hi:[0,1]
	v_pk_mul_bf16 v147, v1, v45 op_sel_hi:[0,1]
	v_pk_mul_bf16 v146, v1, v44 op_sel_hi:[0,1]
	v_pk_mul_bf16 v145, v1, v43 op_sel_hi:[0,1]
	v_pk_mul_bf16 v144, v1, v42 op_sel_hi:[0,1]
	v_pk_mul_bf16 v159, v1, v41 op_sel_hi:[0,1]
	v_pk_mul_bf16 v158, v1, v40 op_sel_hi:[0,1]
	v_pk_mul_bf16 v157, v1, v39 op_sel_hi:[0,1]
	v_pk_mul_bf16 v156, v1, v38 op_sel_hi:[0,1]
	v_pk_mul_bf16 v155, v1, v37 op_sel_hi:[0,1]
	v_pk_mul_bf16 v154, v1, v36 op_sel_hi:[0,1]
	v_pk_mul_bf16 v153, v1, v35 op_sel_hi:[0,1]
	v_pk_mul_bf16 v152, v1, v34 op_sel_hi:[0,1]
	v_pk_mul_bf16 v167, v1, v33 op_sel_hi:[0,1]
	v_pk_mul_bf16 v166, v1, v32 op_sel_hi:[0,1]
	v_pk_mul_bf16 v165, v1, v31 op_sel_hi:[0,1]
	v_pk_mul_bf16 v164, v1, v30 op_sel_hi:[0,1]
	v_pk_mul_bf16 v163, v1, v29 op_sel_hi:[0,1]
	v_pk_mul_bf16 v162, v1, v28 op_sel_hi:[0,1]
	v_pk_mul_bf16 v161, v1, v27 op_sel_hi:[0,1]
	v_pk_mul_bf16 v160, v1, v26 op_sel_hi:[0,1]
	v_pk_mul_bf16 v175, v1, v25 op_sel_hi:[0,1]
	v_pk_mul_bf16 v174, v1, v24 op_sel_hi:[0,1]
	v_pk_mul_bf16 v173, v1, v23 op_sel_hi:[0,1]
	v_pk_mul_bf16 v172, v1, v22 op_sel_hi:[0,1]
	v_pk_mul_bf16 v171, v1, v21 op_sel_hi:[0,1]
	v_pk_mul_bf16 v170, v1, v20 op_sel_hi:[0,1]
	v_pk_mul_bf16 v169, v1, v19 op_sel_hi:[0,1]
	v_pk_mul_bf16 v168, v1, v18 op_sel_hi:[0,1]
	v_pk_mul_bf16 v183, v1, v17 op_sel_hi:[0,1]
	v_pk_mul_bf16 v182, v1, v16 op_sel_hi:[0,1]
	v_pk_mul_bf16 v181, v1, v15 op_sel_hi:[0,1]
	v_pk_mul_bf16 v180, v1, v14 op_sel_hi:[0,1]
	v_pk_mul_bf16 v179, v1, v13 op_sel_hi:[0,1]
	v_pk_mul_bf16 v178, v1, v12 op_sel_hi:[0,1]
	v_pk_mul_bf16 v177, v1, v11 op_sel_hi:[0,1]
	v_pk_mul_bf16 v176, v1, v10 op_sel_hi:[0,1]
	v_pk_mul_bf16 v191, v1, v9 op_sel_hi:[0,1]
	v_pk_mul_bf16 v190, v1, v8 op_sel_hi:[0,1]
	v_pk_mul_bf16 v189, v1, v7 op_sel_hi:[0,1]
	v_pk_mul_bf16 v188, v1, v6 op_sel_hi:[0,1]
	v_pk_mul_bf16 v187, v1, v5 op_sel_hi:[0,1]
	v_pk_mul_bf16 v186, v1, v4 op_sel_hi:[0,1]
	v_pk_mul_bf16 v185, v1, v3 op_sel_hi:[0,1]
	v_pk_mul_bf16 v184, v1, v2 op_sel_hi:[0,1]
	v_and_or_b32 v1, v0, 7, v204 /*v716*/
	s_min_i32 s2, s2, s25
	s_add_co_i32 s4, s3, s4
	s_max_i32 s7, s2, s92
	s_and_b32 s2, s4, 0xffffff80
	s_ashr_i32 s4, s4, 7
	s_set_vgpr_msb 0x2080
	v_subrev_co_ci_u32_e64 v137 /*v649*/, null, 0, v66, vcc_lo
	s_set_vgpr_msb 0x8000
	v_mul_u32_u24_e32 v1, 0x120, v1
	v_lshlrev_b32_e32 v0, 1, v0
	s_cmp_lg_u32 s3, s2
	s_set_vgpr_msb 8
	v_add_nc_u32_e32 v2, s26, v137 /*v649*/
	s_cselect_b32 s2, -1, 0
	s_cmp_lt_i32 s3, 0
	v_and_or_b32 v0, v0, 16, v1
	s_cselect_b32 s3, -1, 0
	v_add_nc_u32_e32 v1, s26, v200 /*v712*/
	s_and_b32 s2, s3, s2
	s_sub_co_ci_u32 s2, s4, 0
	s_set_vgpr_msb 0x880
	v_add_nc_u32_e32 v205 /*v717*/, 0x8800, v0
	s_max_i32 s2, s2, s92
	v_or_b32_e32 v208 /*v720*/, 0x1a000, v0
	v_add_nc_u32_e32 v11 /*v523*/, 0x2b800, v0
	v_add_nc_u32_e32 v13 /*v525*/, 0x3d000, v0
	v_dual_add_nc_u32 v212 /*v724*/, s61, v1 :: v_dual_add_nc_u32 v213 /*v725*/, s61, v2
	v_subrev_nc_u32_e32 v206 /*v718*/, s60, v1
	v_subrev_nc_u32_e32 v207 /*v719*/, s60, v2
	s_min_i32 s12, s2, s7
	s_cmp_ge_u32 s92, s12
	s_set_vgpr_msb 0x8000
	s_cbranch_scc1 .LBB0_15
	v_mov_b32_e32 v0, 0
	s_set_vgpr_msb 0x82
	v_dual_mov_b32 v130 /*v642*/, 1.0 :: v_dual_mov_b32 v215 /*v727*/, v203 /*v715*/
	s_set_vgpr_msb 0x8242
	v_dual_mov_b32 v148 /*v404*/, v13 /*v525*/ :: v_dual_mov_b32 v149 /*v405*/, v12 /*v524*/
	s_set_vgpr_msb 0x4200
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
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v209 /*v721*/, v211 /*v723*/ :: v_dual_mov_b32 v131 /*v643*/, v130 /*v642*/
	s_mov_b32 s18, s93
	s_mov_b32 s93, s71
	s_mov_b32 s15, 0x76543210
	s_mov_b32 s14, 0x3fb8aa3b
	s_mov_b64 s[16:17], s[92:93]
	s_set_vgpr_msb 0x8200
	s_branch .LBB0_9
.LBB0_8:
	s_add_nc_u64 s[16:17], s[16:17], 1
	s_set_vgpr_msb 25
	v_pk_fma_f32 v[192:193], v[70:71] /*v[326:327]*/, v[130:131] /*v[642:643]*/, v[68:69] /*v[324:325]*/
	v_cmp_lt_u64_e64 s2, s[16:17], s[12:13]
	s_set_vgpr_msb 0x1942
	v_dual_mov_b32 v148 /*v404*/, v13 /*v525*/ :: v_dual_mov_b32 v149 /*v405*/, v12 /*v524*/
	s_set_vgpr_msb 0x4281
	v_mov_b32_e32 v211 /*v723*/, v72 /*v328*/
	v_pk_add_f32 v[130:131] /*v[642:643]*/, v[66:67] /*v[322:323]*/, v[192:193]
	s_and_b32 vcc_lo, exec_lo, s2
	s_set_vgpr_msb 0x8100
	s_cbranch_vccz .LBB0_16
.LBB0_9:
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 2
	ds_load_b128 v[200:203], v215 /*v727*/
	ds_load_b128 v[204:207], v215 /*v727*/ offset:32
	ds_load_b128 v[208:211], v215 /*v727*/ offset:64
	ds_load_b128 v[212:215], v215 /*v727*/ offset:96
	ds_load_b128 v[216:219], v215 /*v727*/ offset:128
	ds_load_b128 v[220:223], v215 /*v727*/ offset:160
	ds_load_b128 v[224:227], v215 /*v727*/ offset:192
	ds_load_b128 v[228:231], v215 /*v727*/ offset:224
	s_set_vgpr_msb 0x266
	ds_load_b128 v[10:13] /*v[266:269]*/, v215 /*v727*/ offset:4352
	ds_load_b128 v[14:17] /*v[270:273]*/, v215 /*v727*/ offset:4384
	ds_load_b128 v[58:61] /*v[314:317]*/, v215 /*v727*/ offset:4416
	ds_load_b128 v[62:65] /*v[318:321]*/, v215 /*v727*/ offset:4448
	v_lshl_or_b32 v206 /*v462*/, s16, 7, v204 /*v716*/
	ds_load_b128 v[94:97] /*v[350:353]*/, v215 /*v727*/ offset:4480
	ds_load_b128 v[98:101] /*v[354:357]*/, v215 /*v727*/ offset:4512
	ds_load_b128 v[102:105] /*v[358:361]*/, v215 /*v727*/ offset:4544
	ds_load_b128 v[106:109] /*v[362:365]*/, v215 /*v727*/ offset:4576
	ds_load_b128 v[110:113] /*v[366:369]*/, v215 /*v727*/ offset:8704
	ds_load_b128 v[114:117] /*v[370:373]*/, v215 /*v727*/ offset:8736
	ds_load_b128 v[118:121] /*v[374:377]*/, v215 /*v727*/ offset:8768
	ds_load_b128 v[122:125] /*v[378:381]*/, v215 /*v727*/ offset:8800
	ds_load_b128 v[126:129] /*v[382:385]*/, v215 /*v727*/ offset:8832
	ds_load_b128 v[130:133] /*v[386:389]*/, v215 /*v727*/ offset:8864
	ds_load_b128 v[134:137] /*v[390:393]*/, v215 /*v727*/ offset:8896
	ds_load_b128 v[138:141] /*v[394:397]*/, v215 /*v727*/ offset:8928
	ds_load_b128 v[150:153] /*v[406:409]*/, v215 /*v727*/ offset:13056
	ds_load_b128 v[154:157] /*v[410:413]*/, v215 /*v727*/ offset:13088
	ds_load_b128 v[158:161] /*v[414:417]*/, v215 /*v727*/ offset:13120
	ds_load_b128 v[162:165] /*v[418:421]*/, v215 /*v727*/ offset:13152
	ds_load_b128 v[166:169] /*v[422:425]*/, v215 /*v727*/ offset:13184
	ds_load_b128 v[170:173] /*v[426:429]*/, v215 /*v727*/ offset:13216
	ds_load_b128 v[174:177] /*v[430:433]*/, v215 /*v727*/ offset:13248
	ds_load_b128 v[178:181] /*v[434:437]*/, v215 /*v727*/ offset:13280
	ds_load_b128 v[182:185] /*v[438:441]*/, v215 /*v727*/ offset:17408
	ds_load_b128 v[186:189] /*v[442:445]*/, v215 /*v727*/ offset:17440
	ds_load_b128 v[190:193] /*v[446:449]*/, v215 /*v727*/ offset:17472
	ds_load_b128 v[194:197] /*v[450:453]*/, v215 /*v727*/ offset:17504
	ds_load_b128 v[82:85] /*v[338:341]*/, v215 /*v727*/ offset:17536
	ds_load_b128 v[86:89] /*v[342:345]*/, v215 /*v727*/ offset:17568
	ds_load_b128 v[74:77] /*v[330:333]*/, v215 /*v727*/ offset:17600
	ds_load_b128 v[78:81] /*v[334:337]*/, v215 /*v727*/ offset:17632
	v_dual_add_nc_u32 v214 /*v470*/, 16, v206 /*v462*/ :: v_dual_bitop2_b32 v207 /*v463*/, 1, v206 /*v462*/ bitop3:0x54
	v_cmp_lt_i32_e32 vcc_lo, v212 /*v724*/, v206 /*v462*/
	v_cmp_gt_i32_e64 s2, v206 /*v718*/, v206 /*v462*/
	v_cmp_le_i32_e64 s3, v212 /*v724*/, v206 /*v462*/
	v_cmp_gt_i32_e64 s4, v206 /*v718*/, v207 /*v463*/
	v_dual_add_nc_u32 v215 /*v471*/, 17, v206 /*v462*/ :: v_dual_bitop2_b32 v208 /*v464*/, 2, v206 /*v462*/ bitop3:0x54
	s_or_b32 s2, s2, vcc_lo
	v_dual_add_nc_u32 v216 /*v472*/, 18, v206 /*v462*/ :: v_dual_bitop2_b32 v209 /*v465*/, 3, v206 /*v462*/ bitop3:0x54
	s_set_vgpr_msb 0x6640
	s_wait_dscnt 0x26
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[258:265]*/, v[200:207], v[128:135], 0
	s_set_vgpr_msb 0x4009
	v_cmp_gt_i32_e32 vcc_lo, v208 /*v464*/, v212 /*v724*/
	s_set_vgpr_msb 0x944
	v_or_b32_e32 v210 /*v466*/, 4, v206 /*v462*/
	v_dual_add_nc_u32 v217 /*v473*/, 20, v206 /*v462*/ :: v_dual_bitop2_b32 v211 /*v467*/, 5, v206 /*v462*/ bitop3:0x54
	v_or_b32_e32 v212 /*v468*/, 6, v206 /*v462*/
	v_or_b32_e32 v213 /*v469*/, 7, v206 /*v462*/
	v_dual_add_nc_u32 v225 /*v481*/, 51, v206 /*v462*/ :: v_dual_bitop2_b32 v218 /*v474*/, 36, v206 /*v462*/ bitop3:0x54
	s_set_vgpr_msb 0x4450
	s_wait_dscnt 0x24
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[258:265]*/, v[208:215], v[136:143], v[2:9] /*v[258:265]*/
	s_set_vgpr_msb 0x5046
	v_dual_add_nc_u32 v226 /*v482*/, 52, v206 /*v462*/ :: v_dual_bitop2_b32 v219 /*v475*/, 37, v206 /*v462*/ bitop3:0x54
	v_dual_add_nc_u32 v227 /*v483*/, 53, v206 /*v462*/ :: v_dual_bitop2_b32 v220 /*v476*/, 38, v206 /*v462*/ bitop3:0x54
	v_dual_add_nc_u32 v228 /*v484*/, 54, v206 /*v462*/ :: v_dual_bitop2_b32 v221 /*v477*/, 39, v206 /*v462*/ bitop3:0x54
	ds_load_b128 v[66:69] /*v[322:325]*/, v215 /*v727*/ offset:21760
	ds_load_b128 v[70:73] /*v[326:329]*/, v215 /*v727*/ offset:21792
	ds_load_b128 v[50:53] /*v[306:309]*/, v215 /*v727*/ offset:21824
	ds_load_b128 v[54:57] /*v[310:313]*/, v215 /*v727*/ offset:21856
	ds_load_b128 v[42:45] /*v[298:301]*/, v215 /*v727*/ offset:21888
	ds_load_b128 v[46:49] /*v[302:305]*/, v215 /*v727*/ offset:21920
	ds_load_b128 v[34:37] /*v[290:293]*/, v215 /*v727*/ offset:21952
	s_set_vgpr_msb 0x4650
	s_wait_dscnt 0x29
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[258:265]*/, v[216:223], v[144:151], v[2:9] /*v[258:265]*/
	s_set_vgpr_msb 0x5046
	v_dual_add_nc_u32 v229 /*v485*/, 55, v206 /*v462*/ :: v_dual_bitop2_b32 v230 /*v486*/, 64, v206 /*v462*/ bitop3:0x54
	v_or_b32_e32 v231 /*v487*/, 0x41, v206 /*v462*/
	v_or_b32_e32 v232 /*v488*/, 0x42, v206 /*v462*/
	v_or_b32_e32 v233 /*v489*/, 0x43, v206 /*v462*/
	v_or_b32_e32 v234 /*v490*/, 0x44, v206 /*v462*/
	ds_load_b128 v[38:41] /*v[294:297]*/, v215 /*v727*/ offset:21984
	ds_load_b128 v[26:29] /*v[282:285]*/, v215 /*v727*/ offset:26112
	ds_load_b128 v[30:33] /*v[286:289]*/, v215 /*v727*/ offset:26144
	ds_load_b128 v[18:21] /*v[274:277]*/, v215 /*v727*/ offset:26176
	ds_load_b128 v[22:25] /*v[278:281]*/, v215 /*v727*/ offset:26208
	s_set_vgpr_msb 0x4602
	ds_load_b128 v[232:235], v215 /*v727*/ offset:26240
	ds_load_b128 v[236:239], v215 /*v727*/ offset:26272
	s_set_vgpr_msb 0x250
	s_wait_dscnt 0x2e
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[258:265]*/, v[224:231], v[152:159], v[2:9] /*v[258:265]*/
	s_set_vgpr_msb 0x5044
	v_or_b32_e32 v236 /*v492*/, 0x46, v206 /*v462*/
	v_or_b32_e32 v235 /*v491*/, 0x45, v206 /*v462*/
	v_or_b32_e32 v237 /*v493*/, 0x47, v206 /*v462*/
	v_add_nc_u32_e32 v238 /*v494*/, 0x50, v206 /*v462*/
	v_add_nc_u32_e32 v239 /*v495*/, 0x51, v206 /*v462*/
	v_add_nc_u32_e32 v240 /*v496*/, 0x52, v206 /*v462*/
	v_add_nc_u32_e32 v241 /*v497*/, 0x53, v206 /*v462*/
	s_set_vgpr_msb 0x4441
	s_wait_dscnt 0x2c
	v_wmma_f32_16x16x32_bf16 v[198:205] /*v[454:461]*/, v[10:17] /*v[266:273]*/, v[128:135], 0
	v_cndmask_b32_e64 v90 /*v346*/, v2 /*v258*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	s_set_vgpr_msb 0x4149
	v_cmp_gt_i32_e64 s3, v209 /*v465*/, v212 /*v724*/
	v_cndmask_b32_e64 v91 /*v347*/, v3 /*v259*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v208 /*v464*/, v206 /*v718*/
	v_cmp_lt_i32_e64 s4, v209 /*v465*/, v206 /*v718*/
	s_set_vgpr_msb 0x4944
	v_add_nc_u32_e32 v242 /*v498*/, 0x54, v206 /*v462*/
	s_set_vgpr_msb 0x4401
	v_wmma_f32_16x16x32_bf16 v[240:247], v[10:17] /*v[266:273]*/, v[160:167], 0
	s_set_vgpr_msb 0x144
	v_add_nc_u32_e32 v243 /*v499*/, 0x55, v206 /*v462*/
	s_or_b32 s5, s2, vcc_lo
	s_set_vgpr_msb 0x4409
	v_cmp_gt_i32_e32 vcc_lo, v210 /*v466*/, v212 /*v724*/
	v_cmp_lt_i32_e64 s2, v210 /*v466*/, v206 /*v718*/
	s_or_b32 s6, s4, s3
	v_cmp_gt_i32_e64 s3, v211 /*v467*/, v212 /*v724*/
	v_cmp_lt_i32_e64 s4, v211 /*v467*/, v206 /*v718*/
	s_set_vgpr_msb 0x951
	s_wait_dscnt 0x2a
	v_wmma_f32_16x16x32_bf16 v[198:205] /*v[454:461]*/, v[58:65] /*v[314:321]*/, v[136:143], v[198:205] /*v[454:461]*/
	v_cndmask_b32_e64 v92 /*v348*/, v4 /*v260*/, 0xff800000, s5
	s_or_b32 s5, s2, vcc_lo
	s_set_vgpr_msb 0x5149
	v_cmp_gt_i32_e32 vcc_lo, v212 /*v468*/, v212 /*v724*/
	v_cmp_lt_i32_e64 s2, v212 /*v468*/, v206 /*v718*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s4, v215 /*v471*/, v206 /*v718*/
	v_cndmask_b32_e64 v93 /*v349*/, v5 /*v261*/, 0xff800000, s6
	s_set_vgpr_msb 0x4901
	v_wmma_f32_16x16x32_bf16 v[240:247], v[58:65] /*v[314:321]*/, v[168:175], v[240:247]
	s_set_vgpr_msb 0x102
	ds_load_b128 v[248:251], v215 /*v727*/ offset:30656
	ds_load_b128 v[252:255], v215 /*v727*/ offset:30688
	s_set_vgpr_msb 0x244
	v_add_nc_u32_e32 v244 /*v500*/, 0x76, v206 /*v462*/
	s_wait_alu depctr_vm_vsrc(0)
	s_set_vgpr_msb 0x4482
	v_dual_mov_b32 v12 /*v524*/, v215 /*v727*/ :: v_dual_mov_b32 v13 /*v525*/, v205 /*v717*/
	v_dual_mov_b32 v205 /*v717*/, v208 /*v720*/ :: v_dual_mov_b32 v208 /*v720*/, v11 /*v523*/
	s_set_vgpr_msb 0x8251
	s_wait_dscnt 0x2a
	v_wmma_f32_16x16x32_bf16 v[198:205] /*v[454:461]*/, v[94:101] /*v[350:357]*/, v[144:151], v[198:205] /*v[454:461]*/
	s_set_vgpr_msb 0x5181
	v_mov_b32_e32 v11 /*v523*/, v148 /*v404*/
	s_set_vgpr_msb 0x8144
	v_dual_add_nc_u32 v222 /*v478*/, 48, v206 /*v462*/ :: v_dual_add_nc_u32 v223 /*v479*/, 49, v206 /*v462*/
	v_add_nc_u32_e32 v224 /*v480*/, 50, v206 /*v462*/
	s_set_vgpr_msb 0x4401
	v_wmma_f32_16x16x32_bf16 v[240:247], v[94:101] /*v[350:357]*/, v[176:183], v[240:247]
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x149
	v_cndmask_b32_e64 v95 /*v351*/, v7 /*v263*/, 0xff800000, s3
	s_or_b32 s3, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v213 /*v469*/, v212 /*v724*/
	v_cmp_lt_i32_e64 s2, v213 /*v469*/, v206 /*v718*/
	s_set_vgpr_msb 0x4951
	s_wait_dscnt 0x28
	v_wmma_f32_16x16x32_bf16 v[198:205] /*v[454:461]*/, v[102:109] /*v[358:365]*/, v[152:159], v[198:205] /*v[454:461]*/
	v_cndmask_b32_e64 v94 /*v350*/, v6 /*v262*/, 0xff800000, s5
	v_cndmask_b32_e64 v96 /*v352*/, v8 /*v264*/, 0xff800000, s3
	s_set_vgpr_msb 0x5149
	v_cmp_gt_i32_e64 s3, v215 /*v471*/, v212 /*v724*/
	s_or_b32 s5, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v214 /*v470*/, v212 /*v724*/
	v_cmp_lt_i32_e64 s2, v214 /*v470*/, v206 /*v718*/
	v_cndmask_b32_e64 v97 /*v353*/, v9 /*v265*/, 0xff800000, s5
	s_set_vgpr_msb 0x4941
	s_wait_dscnt 0x26
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[266:273]*/, v[110:117] /*v[366:373]*/, v[128:135], 0
	s_or_b32 s3, s4, s3
	s_set_vgpr_msb 0x4149
	v_cmp_lt_i32_e64 s4, v217 /*v473*/, v206 /*v718*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v216 /*v472*/, v212 /*v724*/
	v_cndmask_b32_e64 v98 /*v354*/, v198 /*v454*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v216 /*v472*/, v206 /*v718*/
	s_set_vgpr_msb 0x4945
	v_add_nc_u32_e32 v198 /*v454*/, 19, v206 /*v462*/
	v_cndmask_b32_e64 v99 /*v355*/, v199 /*v455*/, 0xff800000, s3
	v_add_nc_u32_e32 v199 /*v455*/, 21, v206 /*v462*/
	s_set_vgpr_msb 0x4509
	v_cmp_gt_i32_e64 s3, v217 /*v473*/, v212 /*v724*/
	s_or_b32 s5, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v198 /*v454*/, v212 /*v724*/
	v_cmp_lt_i32_e64 s2, v198 /*v454*/, v206 /*v718*/
	s_set_vgpr_msb 0x951
	s_wait_dscnt 0x24
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[266:273]*/, v[118:125] /*v[374:381]*/, v[136:143], v[10:17] /*v[266:273]*/
	v_cndmask_b32_e64 v100 /*v356*/, v200 /*v456*/, 0xff800000, s5
	s_set_vgpr_msb 0x5144
	v_add_nc_u32_e32 v200 /*v456*/, 22, v206 /*v462*/
	s_or_b32 s3, s4, s3
	s_or_b32 s2, s2, vcc_lo
	s_set_vgpr_msb 0x4449
	v_cmp_gt_i32_e32 vcc_lo, v199 /*v455*/, v212 /*v724*/
	v_cndmask_b32_e64 v101 /*v357*/, v201 /*v457*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v199 /*v455*/, v206 /*v718*/
	s_set_vgpr_msb 0x4944
	v_add_nc_u32_e32 v201 /*v457*/, 23, v206 /*v462*/
	s_set_vgpr_msb 0x4401
	v_wmma_f32_16x16x32_bf16 v[240:247], v[102:109] /*v[358:365]*/, v[184:191], v[240:247]
	s_set_vgpr_msb 0x149
	v_cmp_lt_i32_e64 s4, v200 /*v456*/, v206 /*v718*/
	s_or_b32 s5, s2, vcc_lo
	v_nop
	v_nop
	v_nop
	v_cndmask_b32_e64 v104 /*v360*/, v202 /*v458*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v200 /*v456*/, v212 /*v724*/
	s_set_vgpr_msb 0x4944
	v_or_b32_e32 v202 /*v458*/, 32, v206 /*v462*/
	s_set_vgpr_msb 0x4409
	v_cmp_gt_i32_e32 vcc_lo, v201 /*v457*/, v212 /*v724*/
	v_cmp_lt_i32_e64 s2, v201 /*v457*/, v206 /*v718*/
	s_set_vgpr_msb 0x951
	s_wait_dscnt 0x22
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[266:273]*/, v[126:133] /*v[382:389]*/, v[144:151], v[10:17] /*v[266:273]*/
	s_or_b32 s3, s4, s3
	v_cndmask_b32_e64 v105 /*v361*/, v203 /*v459*/, 0xff800000, s5
	s_set_vgpr_msb 0x5145
	v_or_b32_e32 v203 /*v459*/, 33, v206 /*v462*/
	s_or_b32 s2, s2, vcc_lo
	v_cndmask_b32_e64 v102 /*v358*/, v204 /*v460*/, 0xff800000, s3
	v_cndmask_b32_e64 v103 /*v359*/, v205 /*v461*/, 0xff800000, s2
	s_set_vgpr_msb 0x4509
	v_cmp_gt_i32_e32 vcc_lo, v202 /*v458*/, v212 /*v724*/
	v_cmp_lt_i32_e64 s2, v202 /*v458*/, v206 /*v718*/
	s_set_vgpr_msb 0x944
	v_or_b32_e32 v204 /*v460*/, 34, v206 /*v462*/
	s_set_vgpr_msb 0x4451
	s_wait_dscnt 0x20
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[266:273]*/, v[134:141] /*v[390:397]*/, v[152:159], v[10:17] /*v[266:273]*/
	s_set_vgpr_msb 0x5109
	v_cmp_gt_i32_e64 s3, v203 /*v459*/, v212 /*v724*/
	v_cmp_lt_i32_e64 s4, v203 /*v459*/, v206 /*v718*/
	s_set_vgpr_msb 0x944
	v_or_b32_e32 v205 /*v461*/, 35, v206 /*v462*/
	s_or_b32 s5, s2, vcc_lo
	s_set_vgpr_msb 0x4409
	v_cmp_gt_i32_e32 vcc_lo, v204 /*v460*/, v212 /*v724*/
	v_cmp_lt_i32_e64 s2, v204 /*v460*/, v206 /*v718*/
	s_or_b32 s6, s4, s3
	s_set_vgpr_msb 0x941
	s_wait_dscnt 0x1e
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[314:321]*/, v[150:157] /*v[406:413]*/, v[128:135], 0
	s_set_vgpr_msb 0x4149
	v_cmp_gt_i32_e64 s3, v205 /*v461*/, v212 /*v724*/
	v_cmp_lt_i32_e64 s4, v205 /*v461*/, v206 /*v718*/
	v_cndmask_b32_e64 v108 /*v364*/, v10 /*v266*/, 0xff800000, s5
	s_or_b32 s5, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v218 /*v474*/, v212 /*v724*/
	v_cmp_lt_i32_e64 s2, v218 /*v474*/, v206 /*v718*/
	s_or_b32 s3, s4, s3
	s_set_vgpr_msb 0x4951
	s_wait_dscnt 0x1c
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[314:321]*/, v[158:165] /*v[414:421]*/, v[136:143], v[58:65] /*v[314:321]*/
	v_cndmask_b32_e64 v107 /*v363*/, v13 /*v269*/, 0xff800000, s3
	v_cndmask_b32_e64 v106 /*v362*/, v12 /*v268*/, 0xff800000, s5
	s_or_b32 s3, s2, vcc_lo
	s_set_vgpr_msb 0x5149
	v_cmp_gt_i32_e32 vcc_lo, v219 /*v475*/, v212 /*v724*/
	v_cmp_lt_i32_e64 s2, v219 /*v475*/, v206 /*v718*/
	v_cmp_lt_i32_e64 s4, v221 /*v477*/, v206 /*v718*/
	v_cndmask_b32_e64 v109 /*v365*/, v11 /*v267*/, 0xff800000, s6
	s_set_vgpr_msb 0x4941
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[258:265]*/, v[110:117] /*v[366:373]*/, v[160:167], 0
	s_or_b32 s5, s2, vcc_lo
	s_set_vgpr_msb 0x4109
	v_cmp_gt_i32_e32 vcc_lo, v220 /*v476*/, v212 /*v724*/
	v_cmp_lt_i32_e64 s2, v220 /*v476*/, v206 /*v718*/
	s_or_b32 s2, s2, vcc_lo
	s_set_vgpr_msb 0x951
	s_wait_dscnt 0x1a
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[314:321]*/, v[166:173] /*v[422:429]*/, v[144:151], v[58:65] /*v[314:321]*/
	s_set_vgpr_msb 0x5149
	v_cmp_gt_i32_e32 vcc_lo, v222 /*v478*/, v212 /*v724*/
	v_cndmask_b32_e64 v112 /*v368*/, v16 /*v272*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v222 /*v478*/, v206 /*v718*/
	v_cndmask_b32_e64 v111 /*v367*/, v15 /*v271*/, 0xff800000, s5
	v_cndmask_b32_e64 v110 /*v366*/, v14 /*v270*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v221 /*v477*/, v212 /*v724*/
	s_or_b32 s5, s2, vcc_lo
	s_set_vgpr_msb 0x4951
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[258:265]*/, v[118:125] /*v[374:381]*/, v[168:175], v[2:9] /*v[258:265]*/
	s_set_vgpr_msb 0x5149
	v_cmp_gt_i32_e32 vcc_lo, v223 /*v479*/, v212 /*v724*/
	v_cmp_lt_i32_e64 s2, v223 /*v479*/, v206 /*v718*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s4, v224 /*v480*/, v206 /*v718*/
	v_cndmask_b32_e64 v113 /*v369*/, v17 /*v273*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v224 /*v480*/, v212 /*v724*/
	s_or_b32 s2, s2, vcc_lo
	s_set_vgpr_msb 0x4951
	s_wait_dscnt 0x18
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[314:321]*/, v[174:181] /*v[430:437]*/, v[152:159], v[58:65] /*v[314:321]*/
	s_set_vgpr_msb 0x5149
	v_cmp_gt_i32_e32 vcc_lo, v225 /*v481*/, v212 /*v724*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s4, v226 /*v482*/, v206 /*v718*/
	v_nop
	v_nop
	v_cndmask_b32_e64 v115 /*v371*/, v59 /*v315*/, 0xff800000, s2
	s_set_vgpr_msb 0x4951
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[258:265]*/, v[126:133] /*v[382:389]*/, v[176:183], v[2:9] /*v[258:265]*/
	s_set_vgpr_msb 0x5149
	v_cmp_lt_i32_e64 s2, v225 /*v481*/, v206 /*v718*/
	v_cndmask_b32_e64 v114 /*v370*/, v58 /*v314*/, 0xff800000, s5
	v_cndmask_b32_e64 v116 /*v372*/, v60 /*v316*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v226 /*v482*/, v212 /*v724*/
	s_or_b32 s5, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v227 /*v483*/, v212 /*v724*/
	s_set_vgpr_msb 0x4941
	s_wait_dscnt 0x16
	v_wmma_f32_16x16x32_bf16 v[122:129] /*v[378:385]*/, v[182:189] /*v[438:445]*/, v[128:135], 0
	s_set_vgpr_msb 0x4149
	v_cmp_lt_i32_e64 s2, v227 /*v483*/, v206 /*v718*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s4, v229 /*v485*/, v206 /*v718*/
	v_cndmask_b32_e64 v118 /*v374*/, v62 /*v318*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v229 /*v485*/, v212 /*v724*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v228 /*v484*/, v212 /*v724*/
	s_set_vgpr_msb 0x4951
	s_wait_dscnt 0x14
	v_wmma_f32_16x16x32_bf16 v[122:129] /*v[378:385]*/, v[190:197] /*v[446:453]*/, v[136:143], v[122:129] /*v[378:385]*/
	v_cndmask_b32_e64 v119 /*v375*/, v63 /*v319*/, 0xff800000, s2
	s_set_vgpr_msb 0x5149
	v_cmp_lt_i32_e64 s2, v228 /*v484*/, v206 /*v718*/
	v_cndmask_b32_e64 v117 /*v373*/, v61 /*v317*/, 0xff800000, s5
	s_or_b32 s6, s4, s3
	v_cmp_gt_i32_e64 s3, v231 /*v487*/, v212 /*v724*/
	v_cmp_lt_i32_e64 s4, v231 /*v487*/, v206 /*v718*/
	s_or_b32 s5, s2, vcc_lo
	s_set_vgpr_msb 0x4951
	s_wait_dscnt 0x12
	v_wmma_f32_16x16x32_bf16 v[122:129] /*v[378:385]*/, v[82:89] /*v[338:345]*/, v[144:151], v[122:129] /*v[378:385]*/
	s_set_vgpr_msb 0x5149
	v_cmp_gt_i32_e32 vcc_lo, v230 /*v486*/, v212 /*v724*/
	v_cmp_lt_i32_e64 s2, v230 /*v486*/, v206 /*v718*/
	v_cndmask_b32_e64 v120 /*v376*/, v64 /*v320*/, 0xff800000, s5
	v_cndmask_b32_e64 v121 /*v377*/, v65 /*v321*/, 0xff800000, s6
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s4, v235 /*v491*/, v206 /*v718*/
	s_or_b32 s5, s2, vcc_lo
	s_set_vgpr_msb 0x4951
	s_wait_dscnt 0x10
	v_wmma_f32_16x16x32_bf16 v[122:129] /*v[378:385]*/, v[74:81] /*v[330:337]*/, v[152:159], v[122:129] /*v[378:385]*/
	s_set_vgpr_msb 0x5149
	v_cmp_gt_i32_e32 vcc_lo, v232 /*v488*/, v212 /*v724*/
	v_cmp_lt_i32_e64 s2, v232 /*v488*/, v206 /*v718*/
	v_nop
	v_nop
	v_cndmask_b32_e64 v123 /*v379*/, v123 /*v379*/, 0xff800000, s3
	s_set_vgpr_msb 0x4951
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[258:265]*/, v[134:141] /*v[390:397]*/, v[184:191], v[2:9] /*v[258:265]*/
	s_or_b32 s3, s2, vcc_lo
	s_set_vgpr_msb 0x5149
	v_cmp_gt_i32_e32 vcc_lo, v233 /*v489*/, v212 /*v724*/
	v_cmp_lt_i32_e64 s2, v233 /*v489*/, v206 /*v718*/
	v_cndmask_b32_e64 v122 /*v378*/, v122 /*v378*/, 0xff800000, s5
	v_cndmask_b32_e64 v124 /*v380*/, v124 /*v380*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v235 /*v491*/, v212 /*v724*/
	s_or_b32 s5, s2, vcc_lo
	s_set_vgpr_msb 0x4941
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x32_bf16 v[130:137] /*v[386:393]*/, v[66:73] /*v[322:329]*/, v[128:135], 0
	s_set_vgpr_msb 0x4149
	v_cmp_gt_i32_e32 vcc_lo, v234 /*v490*/, v212 /*v724*/
	v_cmp_lt_i32_e64 s2, v234 /*v490*/, v206 /*v718*/
	v_cndmask_b32_e64 v125 /*v381*/, v125 /*v381*/, 0xff800000, s5
	s_or_b32 s3, s4, s3
	v_cndmask_b32_e64 v127 /*v383*/, v127 /*v383*/, 0xff800000, s3
	s_or_b32 s2, s2, vcc_lo
	s_set_vgpr_msb 0x4941
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[314:321]*/, v[182:189] /*v[438:445]*/, v[160:167], 0
	v_cndmask_b32_e64 v126 /*v382*/, v126 /*v382*/, 0xff800000, s2
	s_set_vgpr_msb 0x4109
	v_cmp_gt_i32_e32 vcc_lo, v236 /*v492*/, v212 /*v724*/
	v_cmp_lt_i32_e64 s2, v236 /*v492*/, v206 /*v718*/
	s_or_b32 s5, s2, vcc_lo
	s_set_vgpr_msb 0x900
	v_wmma_f32_16x16x32_bf16 v[192:199], v[200:207], v[160:167], 0
	s_set_vgpr_msb 0x49
	v_cmp_gt_i32_e32 vcc_lo, v237 /*v493*/, v212 /*v724*/
	v_cmp_lt_i32_e64 s2, v237 /*v493*/, v206 /*v718*/
	v_cndmask_b32_e64 v128 /*v384*/, v128 /*v384*/, 0xff800000, s5
	s_set_vgpr_msb 0x4944
	v_add_nc_u32_e32 v182 /*v438*/, 0x56, v206 /*v462*/
	v_add_nc_u32_e32 v183 /*v439*/, 0x57, v206 /*v462*/
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x4402
	ds_load_b128 v[200:203], v215 /*v727*/ offset:30592
	ds_load_b128 v[204:207], v215 /*v727*/ offset:30624
	s_or_b32 s2, s2, vcc_lo
	s_set_vgpr_msb 0x251
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x32_bf16 v[130:137] /*v[386:393]*/, v[50:57] /*v[306:313]*/, v[136:143], v[130:137] /*v[386:393]*/
	v_cndmask_b32_e64 v129 /*v385*/, v129 /*v385*/, 0xff800000, s2
	s_set_vgpr_msb 0x5109
	v_cmp_gt_i32_e32 vcc_lo, v239 /*v495*/, v212 /*v724*/
	v_cmp_lt_i32_e64 s2, v239 /*v495*/, v206 /*v718*/
	v_cmp_lt_i32_e64 s6, v182 /*v438*/, v206 /*v718*/
	s_set_vgpr_msb 0x944
	v_or_b32_e32 v184 /*v440*/, 0x60, v206 /*v462*/
	v_or_b32_e32 v185 /*v441*/, 0x61, v206 /*v462*/
	v_or_b32_e32 v186 /*v442*/, 0x62, v206 /*v462*/
	s_set_vgpr_msb 0x4451
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[314:321]*/, v[190:197] /*v[446:453]*/, v[168:175], v[58:65] /*v[314:321]*/
	s_or_b32 s5, s2, vcc_lo
	s_set_vgpr_msb 0x5109
	v_cmp_gt_i32_e32 vcc_lo, v241 /*v497*/, v212 /*v724*/
	v_cmp_lt_i32_e64 s2, v241 /*v497*/, v206 /*v718*/
	s_set_vgpr_msb 0x944
	v_or_b32_e32 v187 /*v443*/, 0x63, v206 /*v462*/
	v_or_b32_e32 v188 /*v444*/, 0x64, v206 /*v462*/
	v_or_b32_e32 v189 /*v445*/, 0x65, v206 /*v462*/
	v_or_b32_e32 v190 /*v446*/, 0x66, v206 /*v462*/
	s_set_vgpr_msb 0x4400
	v_wmma_f32_16x16x32_bf16 v[192:199], v[208:215], v[168:175], v[192:199]
	s_or_b32 s9, s2, vcc_lo
	s_set_vgpr_msb 9
	v_cmp_gt_i32_e32 vcc_lo, v242 /*v498*/, v212 /*v724*/
	v_cmp_lt_i32_e64 s2, v242 /*v498*/, v206 /*v718*/
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x902
	ds_load_b128 v[212:215], v215 /*v727*/ offset:30560
	s_set_vgpr_msb 0x244
	v_or_b32_e32 v191 /*v447*/, 0x67, v206 /*v462*/
	v_add_nc_u32_e32 v192 /*v448*/, 0x70, v206 /*v462*/
	v_add_nc_u32_e32 v193 /*v449*/, 0x71, v206 /*v462*/
	s_set_vgpr_msb 0x4451
	s_wait_dscnt 0xd
	v_wmma_f32_16x16x32_bf16 v[130:137] /*v[386:393]*/, v[42:49] /*v[298:305]*/, v[144:151], v[130:137] /*v[386:393]*/
	s_or_b32 s2, s2, vcc_lo
	s_set_vgpr_msb 0x5109
	v_cmp_gt_i32_e32 vcc_lo, v183 /*v439*/, v212 /*v724*/
	s_set_vgpr_msb 0x944
	v_add_nc_u32_e32 v194 /*v450*/, 0x72, v206 /*v462*/
	v_add_nc_u32_e32 v196 /*v452*/, 0x74, v206 /*v462*/
	v_add_nc_u32_e32 v195 /*v451*/, 0x73, v206 /*v462*/
	v_add_nc_u32_e32 v197 /*v453*/, 0x75, v206 /*v462*/
	s_set_vgpr_msb 0x4451
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[314:321]*/, v[82:89] /*v[338:345]*/, v[176:183], v[58:65] /*v[314:321]*/
	s_set_vgpr_msb 0x5100
	v_wmma_f32_16x16x32_bf16 v[192:199], v[216:223], v[176:183], v[192:199]
	s_set_vgpr_msb 0x51
	s_wait_dscnt 0xb
	v_wmma_f32_16x16x32_bf16 v[130:137] /*v[386:393]*/, v[34:41] /*v[290:297]*/, v[152:159], v[130:137] /*v[386:393]*/
	v_nop
	v_nop
	v_nop
	v_nop
	v_cndmask_b32_e64 v134 /*v390*/, v134 /*v390*/, 0xff800000, s2
	v_wmma_f32_16x16x32_bf16 v[58:65] /*v[314:321]*/, v[74:81] /*v[330:337]*/, v[184:191], v[58:65] /*v[314:321]*/
	v_cndmask_b32_e64 v131 /*v387*/, v131 /*v387*/, 0xff800000, s5
	s_set_vgpr_msb 0x5149
	v_cmp_gt_i32_e64 s5, v182 /*v438*/, v212 /*v724*/
	v_cndmask_b32_e64 v133 /*v389*/, v133 /*v389*/, 0xff800000, s9
	s_set_vgpr_msb 0x4941
	v_wmma_f32_16x16x32_bf16 v[74:81] /*v[330:337]*/, v[66:73] /*v[322:329]*/, v[160:167], 0
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x4142
	v_mov_b32_e32 v73 /*v329*/, v209 /*v721*/
	s_set_vgpr_msb 0x4209
	v_cmp_gt_i32_e64 s3, v238 /*v494*/, v212 /*v724*/
	v_cmp_lt_i32_e64 s4, v238 /*v494*/, v206 /*v718*/
	s_set_vgpr_msb 0x941
	s_wait_dscnt 0x9
	v_wmma_f32_16x16x32_bf16 v[138:145] /*v[394:401]*/, v[26:33] /*v[282:289]*/, v[128:135], 0
	s_set_vgpr_msb 0x4144
	v_add_nc_u32_e32 v72 /*v328*/, 0x77, v206 /*v462*/
	s_or_b32 s3, s4, s3
	s_set_vgpr_msb 0x4449
	v_cmp_lt_i32_e64 s4, v240 /*v496*/, v206 /*v718*/
	v_cndmask_b32_e64 v130 /*v386*/, v130 /*v386*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v240 /*v496*/, v212 /*v724*/
	s_set_vgpr_msb 0x4900
	v_wmma_f32_16x16x32_bf16 v[192:199], v[224:231], v[184:191], v[192:199]
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 2
	ds_load_b128 v[224:227], v215 /*v727*/ offset:26304
	ds_load_b128 v[228:231], v215 /*v727*/ offset:26336
	ds_load_b128 v[216:219], v215 /*v727*/ offset:30464
	ds_load_b128 v[220:223], v215 /*v727*/ offset:30496
	ds_load_b128 v[208:211], v215 /*v727*/ offset:30528
	s_wait_alu depctr_vm_vsrc(0)
	s_set_vgpr_msb 0x282
	v_dual_mov_b32 v215 /*v727*/, v210 /*v722*/ :: v_dual_mov_b32 v210 /*v722*/, v10 /*v522*/
	s_or_b32 s3, s4, s3
	s_set_vgpr_msb 0x8249
	v_cmp_lt_i32_e64 s4, v243 /*v499*/, v206 /*v718*/
	v_cndmask_b32_e64 v132 /*v388*/, v132 /*v388*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v243 /*v499*/, v212 /*v724*/
	s_set_vgpr_msb 0x4951
	s_wait_dscnt 0xc
	v_wmma_f32_16x16x32_bf16 v[138:145] /*v[394:401]*/, v[18:25] /*v[274:281]*/, v[136:143], v[138:145] /*v[394:401]*/
	s_set_vgpr_msb 0x5181
	v_mov_b32_e32 v10 /*v522*/, v149 /*v405*/
	s_or_b32 s2, s4, s3
	s_or_b32 s3, s6, s5
	s_set_vgpr_msb 0x8149
	v_cndmask_b32_e64 v135 /*v391*/, v135 /*v391*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v183 /*v439*/, v206 /*v718*/
	s_set_vgpr_msb 0x4950
	s_wait_dscnt 0xa
	v_wmma_f32_16x16x32_bf16 v[138:145] /*v[394:401]*/, v[232:239], v[144:151], v[138:145] /*v[394:401]*/
	s_set_vgpr_msb 0x5049
	v_cndmask_b32_e64 v136 /*v392*/, v136 /*v392*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v184 /*v440*/, v212 /*v724*/
	v_cmp_lt_i32_e64 s4, v184 /*v440*/, v206 /*v718*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v185 /*v441*/, v212 /*v724*/
	v_cndmask_b32_e64 v137 /*v393*/, v137 /*v393*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v185 /*v441*/, v206 /*v718*/
	s_set_vgpr_msb 0x4941
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[266:273]*/, v[150:157] /*v[406:413]*/, v[160:167], 0
	s_or_b32 s5, s4, s3
	s_set_vgpr_msb 0x4109
	v_cmp_gt_i32_e64 s3, v186 /*v442*/, v212 /*v724*/
	v_cmp_lt_i32_e64 s4, v186 /*v442*/, v206 /*v718*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v187 /*v443*/, v212 /*v724*/
	v_cmp_lt_i32_e64 s6, v189 /*v445*/, v206 /*v718*/
	s_or_b32 s9, s4, s3
	s_set_vgpr_msb 0x940
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x32_bf16 v[150:157] /*v[406:413]*/, v[216:223], v[128:135], 0
	s_set_vgpr_msb 0x4009
	v_cmp_gt_i32_e64 s3, v188 /*v444*/, v212 /*v724*/
	v_cmp_lt_i32_e64 s4, v188 /*v444*/, v206 /*v718*/
	s_set_vgpr_msb 0x950
	v_wmma_f32_16x16x32_bf16 v[138:145] /*v[394:401]*/, v[224:231], v[152:159], v[138:145] /*v[394:401]*/
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x5041
	v_cndmask_b32_e64 v139 /*v395*/, v139 /*v395*/, 0xff800000, s2
	s_set_vgpr_msb 0x4150
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[150:157] /*v[406:413]*/, v[208:215], v[136:143], v[150:157] /*v[406:413]*/
	s_set_vgpr_msb 0x5049
	v_cmp_lt_i32_e64 s2, v187 /*v443*/, v206 /*v718*/
	v_cndmask_b32_e64 v138 /*v394*/, v138 /*v394*/, 0xff800000, s5
	v_cmp_gt_i32_e64 s5, v189 /*v445*/, v212 /*v724*/
	v_cndmask_b32_e64 v140 /*v396*/, v140 /*v396*/, 0xff800000, s9
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v190 /*v446*/, v212 /*v724*/
	v_cndmask_b32_e64 v141 /*v397*/, v141 /*v397*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	s_set_vgpr_msb 0x4950
	v_wmma_f32_16x16x32_bf16 v[150:157] /*v[406:413]*/, v[200:207], v[144:151], v[150:157] /*v[406:413]*/
	s_set_vgpr_msb 0x5049
	v_cndmask_b32_e64 v142 /*v398*/, v142 /*v398*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v190 /*v446*/, v206 /*v718*/
	s_or_b32 s3, s6, s5
	v_cmp_lt_i32_e64 s4, v191 /*v447*/, v206 /*v718*/
	v_cndmask_b32_e64 v143 /*v399*/, v143 /*v399*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v191 /*v447*/, v212 /*v724*/
	s_or_b32 s2, s2, vcc_lo
	s_set_vgpr_msb 0x4950
	v_wmma_f32_16x16x32_bf16 v[150:157] /*v[406:413]*/, v[248:255], v[152:159], v[150:157] /*v[406:413]*/
	s_set_vgpr_msb 0x5049
	v_cndmask_b32_e64 v144 /*v400*/, v144 /*v400*/, 0xff800000, s2
	v_cmp_gt_i32_e32 vcc_lo, v192 /*v448*/, v212 /*v724*/
	v_cmp_lt_i32_e64 s2, v192 /*v448*/, v206 /*v718*/
	s_or_b32 s5, s4, s3
	v_cmp_gt_i32_e64 s3, v193 /*v449*/, v212 /*v724*/
	v_cmp_lt_i32_e64 s4, v193 /*v449*/, v206 /*v718*/
	v_cndmask_b32_e64 v145 /*v401*/, v145 /*v401*/, 0xff800000, s5
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v194 /*v450*/, v212 /*v724*/
	v_cndmask_b32_e64 v146 /*v402*/, v150 /*v406*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	s_set_vgpr_msb 0x4951
	v_wmma_f32_16x16x32_bf16 v[74:81] /*v[330:337]*/, v[50:57] /*v[306:313]*/, v[168:175], v[74:81] /*v[330:337]*/
	v_cndmask_b32_e64 v147 /*v403*/, v151 /*v407*/, 0xff800000, s2
	s_set_vgpr_msb 0x5149
	v_cmp_lt_i32_e64 s2, v194 /*v450*/, v206 /*v718*/
	v_cmp_gt_i32_e64 s3, v195 /*v451*/, v212 /*v724*/
	v_cmp_lt_i32_e64 s4, v195 /*v451*/, v206 /*v718*/
	s_or_b32 s5, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v196 /*v452*/, v212 /*v724*/
	v_cmp_lt_i32_e64 s2, v196 /*v452*/, v206 /*v718*/
	v_cndmask_b32_e64 v66 /*v322*/, v152 /*v408*/, 0xff800000, s5
	s_or_b32 s6, s4, s3
	v_cmp_gt_i32_e64 s3, v197 /*v453*/, v212 /*v724*/
	v_cmp_lt_i32_e64 s4, v197 /*v453*/, v206 /*v718*/
	s_or_b32 s5, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v244 /*v500*/, v212 /*v724*/
	v_cmp_lt_i32_e64 s2, v244 /*v500*/, v206 /*v718*/
	s_set_vgpr_msb 0x4951
	v_wmma_f32_16x16x32_bf16 v[74:81] /*v[330:337]*/, v[42:49] /*v[298:305]*/, v[176:183], v[74:81] /*v[330:337]*/
	v_cndmask_b32_e64 v67 /*v323*/, v153 /*v409*/, 0xff800000, s6
	v_cndmask_b32_e64 v50 /*v306*/, v154 /*v410*/, 0xff800000, s5
	s_set_vgpr_msb 0x5149
	v_cmp_ge_i32_e64 s5, v206 /*v462*/, v213 /*v725*/
	s_or_b32 s9, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v72 /*v328*/, v212 /*v724*/
	v_cmp_lt_i32_e64 s2, v72 /*v328*/, v206 /*v718*/
	v_cmp_lt_i32_e64 s6, v207 /*v463*/, v207 /*v719*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s4, v206 /*v462*/, v207 /*v719*/
	v_cndmask_b32_e64 v51 /*v307*/, v155 /*v411*/, 0xff800000, s3
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v208 /*v464*/, v213 /*v725*/
	v_cndmask_b32_e64 v43 /*v299*/, v157 /*v413*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v208 /*v464*/, v207 /*v719*/
	v_cmp_gt_i32_e64 s3, v206 /*v462*/, v213 /*v725*/
	s_or_b32 s5, s6, s5
	s_set_vgpr_msb 0x4941
	v_wmma_f32_16x16x32_bf16 v[82:89] /*v[338:345]*/, v[26:33] /*v[282:289]*/, v[160:167], 0
	s_set_vgpr_msb 0x4149
	v_cmp_lt_i32_e64 s6, v212 /*v468*/, v207 /*v719*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v210 /*v466*/, v213 /*v725*/
	v_cndmask_b32_e64 v42 /*v298*/, v156 /*v412*/, 0xff800000, s9
	s_or_b32 s9, s4, s3
	v_cmp_gt_i32_e64 s3, v209 /*v465*/, v213 /*v725*/
	s_set_vgpr_msb 0x4940
	v_cndmask_b32_e64 v26 /*v282*/, v194, 0xff800000, s2
	s_set_vgpr_msb 0x4009
	v_cmp_lt_i32_e64 s2, v210 /*v466*/, v207 /*v719*/
	s_set_vgpr_msb 0x951
	v_wmma_f32_16x16x32_bf16 v[74:81] /*v[330:337]*/, v[34:41] /*v[290:297]*/, v[184:191], v[74:81] /*v[330:337]*/
	s_set_vgpr_msb 0x5109
	v_cmp_lt_i32_e64 s4, v209 /*v465*/, v207 /*v719*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v213 /*v469*/, v213 /*v725*/
	s_set_vgpr_msb 0x940
	v_cndmask_b32_e64 v70 /*v326*/, v196, 0xff800000, s2
	s_or_b32 s3, s4, s3
	s_set_vgpr_msb 0x4009
	v_cmp_lt_i32_e64 s4, v211 /*v467*/, v207 /*v719*/
	s_set_vgpr_msb 0x940
	v_cndmask_b32_e64 v35 /*v291*/, v193, 0xff800000, s5
	s_set_vgpr_msb 0x4009
	v_cmp_gt_i32_e64 s5, v212 /*v468*/, v213 /*v725*/
	s_set_vgpr_msb 0x940
	v_cndmask_b32_e64 v27 /*v283*/, v195, 0xff800000, s3
	s_set_vgpr_msb 0x4009
	v_cmp_gt_i32_e64 s3, v211 /*v467*/, v213 /*v725*/
	s_set_vgpr_msb 0x951
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[266:273]*/, v[158:165] /*v[414:421]*/, v[168:175], v[10:17] /*v[266:273]*/
	s_set_vgpr_msb 0x5140
	v_cndmask_b32_e64 v34 /*v290*/, v192, 0xff800000, s9
	s_or_b32 s2, s6, s5
	s_set_vgpr_msb 0x4009
	v_cmp_gt_i32_e64 s5, v215 /*v471*/, v213 /*v725*/
	s_set_vgpr_msb 0x940
	v_cndmask_b32_e64 v68 /*v324*/, v198, 0xff800000, s2
	s_set_vgpr_msb 0x4009
	v_cmp_lt_i32_e64 s2, v213 /*v469*/, v207 /*v719*/
	v_cmp_lt_i32_e64 s6, v215 /*v471*/, v207 /*v719*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s4, v214 /*v470*/, v207 /*v719*/
	s_set_vgpr_msb 0x940
	v_cndmask_b32_e64 v71 /*v327*/, v197, 0xff800000, s3
	s_or_b32 s2, s2, vcc_lo
	s_set_vgpr_msb 0x4009
	v_cmp_gt_i32_e32 vcc_lo, v216 /*v472*/, v213 /*v725*/
	s_set_vgpr_msb 0x940
	v_cndmask_b32_e64 v69 /*v325*/, v199, 0xff800000, s2
	s_set_vgpr_msb 0x4009
	v_cmp_lt_i32_e64 s2, v216 /*v472*/, v207 /*v719*/
	v_cmp_gt_i32_e64 s3, v214 /*v470*/, v213 /*v725*/
	s_or_b32 s5, s6, s5
	v_cmp_lt_i32_e64 s6, v204 /*v460*/, v207 /*v719*/
	s_set_vgpr_msb 0x940
	v_cndmask_b32_e64 v151 /*v407*/, v241, 0xff800000, s5
	s_or_b32 s5, s2, vcc_lo
	s_set_vgpr_msb 0x4009
	v_cmp_gt_i32_e32 vcc_lo, v217 /*v473*/, v213 /*v725*/
	v_cmp_lt_i32_e64 s2, v217 /*v473*/, v207 /*v719*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s4, v198 /*v454*/, v207 /*v719*/
	s_set_vgpr_msb 0x940
	v_cndmask_b32_e64 v150 /*v406*/, v240, 0xff800000, s3
	s_set_vgpr_msb 0x4009
	v_cmp_gt_i32_e64 s3, v198 /*v454*/, v213 /*v725*/
	s_set_vgpr_msb 0x940
	v_cndmask_b32_e64 v152 /*v408*/, v242, 0xff800000, s5
	s_or_b32 s5, s2, vcc_lo
	s_set_vgpr_msb 0x4009
	v_cmp_gt_i32_e32 vcc_lo, v199 /*v455*/, v213 /*v725*/
	v_cmp_lt_i32_e64 s2, v199 /*v455*/, v207 /*v719*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s4, v200 /*v456*/, v207 /*v719*/
	s_set_vgpr_msb 0x940
	v_cndmask_b32_e64 v153 /*v409*/, v243, 0xff800000, s3
	s_set_vgpr_msb 0x4009
	v_cmp_gt_i32_e64 s3, v200 /*v456*/, v213 /*v725*/
	s_set_vgpr_msb 0x940
	v_cndmask_b32_e64 v154 /*v410*/, v244, 0xff800000, s5
	s_or_b32 s5, s2, vcc_lo
	s_set_vgpr_msb 0x4009
	v_cmp_gt_i32_e32 vcc_lo, v201 /*v457*/, v213 /*v725*/
	v_cmp_lt_i32_e64 s2, v201 /*v457*/, v207 /*v719*/
	s_or_b32 s3, s4, s3
	s_set_vgpr_msb 0x940
	v_cndmask_b32_e64 v155 /*v411*/, v245, 0xff800000, s5
	v_cndmask_b32_e64 v156 /*v412*/, v246, 0xff800000, s3
	s_set_vgpr_msb 0x4009
	v_cmp_gt_i32_e64 s3, v203 /*v459*/, v213 /*v725*/
	s_or_b32 s5, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v202 /*v458*/, v213 /*v725*/
	v_cmp_lt_i32_e64 s2, v202 /*v458*/, v207 /*v719*/
	v_cmp_lt_i32_e64 s4, v203 /*v459*/, v207 /*v719*/
	s_set_vgpr_msb 0x940
	v_cndmask_b32_e64 v157 /*v413*/, v247, 0xff800000, s5
	s_set_vgpr_msb 0x4009
	v_cmp_gt_i32_e64 s5, v204 /*v460*/, v213 /*v725*/
	s_set_vgpr_msb 0x951
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[266:273]*/, v[166:173] /*v[422:429]*/, v[176:183], v[10:17] /*v[266:273]*/
	s_or_b32 s2, s2, vcc_lo
	s_or_b32 s3, s4, s3
	v_cndmask_b32_e64 v158 /*v414*/, v2 /*v258*/, 0xff800000, s2
	s_set_vgpr_msb 0x5149
	v_cmp_gt_i32_e32 vcc_lo, v205 /*v461*/, v213 /*v725*/
	v_cmp_lt_i32_e64 s2, v205 /*v461*/, v207 /*v719*/
	v_cndmask_b32_e64 v159 /*v415*/, v3 /*v259*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v218 /*v474*/, v213 /*v725*/
	v_cmp_lt_i32_e64 s4, v218 /*v474*/, v207 /*v719*/
	s_or_b32 s5, s6, s5
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v220 /*v476*/, v213 /*v725*/
	v_cndmask_b32_e64 v161 /*v417*/, v5 /*v261*/, 0xff800000, s2
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s2, v220 /*v476*/, v207 /*v719*/
	v_cndmask_b32_e64 v160 /*v416*/, v4 /*v260*/, 0xff800000, s5
	v_cmp_gt_i32_e64 s5, v219 /*v475*/, v213 /*v725*/
	v_cmp_lt_i32_e64 s6, v219 /*v475*/, v207 /*v719*/
	v_cndmask_b32_e64 v162 /*v418*/, v6 /*v262*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v221 /*v477*/, v213 /*v725*/
	v_cmp_lt_i32_e64 s4, v221 /*v477*/, v207 /*v719*/
	s_or_b32 s2, s2, vcc_lo
	s_set_vgpr_msb 0x4951
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[266:273]*/, v[174:181] /*v[430:437]*/, v[184:191], v[10:17] /*v[266:273]*/
	s_or_b32 s5, s6, s5
	v_cndmask_b32_e64 v164 /*v420*/, v8 /*v264*/, 0xff800000, s2
	s_or_b32 s3, s4, s3
	s_set_vgpr_msb 0x5149
	v_cmp_gt_i32_e32 vcc_lo, v223 /*v479*/, v213 /*v725*/
	v_cmp_lt_i32_e64 s2, v223 /*v479*/, v207 /*v719*/
	v_cndmask_b32_e64 v163 /*v419*/, v7 /*v263*/, 0xff800000, s5
	v_cmp_gt_i32_e64 s5, v222 /*v478*/, v213 /*v725*/
	v_cmp_lt_i32_e64 s6, v222 /*v478*/, v207 /*v719*/
	v_cndmask_b32_e64 v165 /*v421*/, v9 /*v265*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v224 /*v480*/, v213 /*v725*/
	v_cmp_lt_i32_e64 s4, v224 /*v480*/, v207 /*v719*/
	s_or_b32 s2, s2, vcc_lo
	s_or_b32 s5, s6, s5
	v_cndmask_b32_e64 v167 /*v423*/, v11 /*v267*/, 0xff800000, s2
	v_cmp_gt_i32_e32 vcc_lo, v226 /*v482*/, v213 /*v725*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s2, v226 /*v482*/, v207 /*v719*/
	v_cndmask_b32_e64 v166 /*v422*/, v10 /*v266*/, 0xff800000, s5
	v_cmp_gt_i32_e64 s5, v225 /*v481*/, v213 /*v725*/
	v_cmp_lt_i32_e64 s6, v225 /*v481*/, v207 /*v719*/
	v_cndmask_b32_e64 v168 /*v424*/, v12 /*v268*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v227 /*v483*/, v213 /*v725*/
	v_cmp_lt_i32_e64 s4, v227 /*v483*/, v207 /*v719*/
	s_or_b32 s2, s2, vcc_lo
	s_or_b32 s5, s6, s5
	v_cndmask_b32_e64 v170 /*v426*/, v14 /*v270*/, 0xff800000, s2
	v_cmp_gt_i32_e32 vcc_lo, v229 /*v485*/, v213 /*v725*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s2, v229 /*v485*/, v207 /*v719*/
	v_cndmask_b32_e64 v169 /*v425*/, v13 /*v269*/, 0xff800000, s5
	v_cmp_gt_i32_e64 s5, v228 /*v484*/, v213 /*v725*/
	v_cmp_lt_i32_e64 s6, v228 /*v484*/, v207 /*v719*/
	v_cndmask_b32_e64 v171 /*v427*/, v15 /*v271*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v230 /*v486*/, v213 /*v725*/
	v_cmp_lt_i32_e64 s4, v230 /*v486*/, v207 /*v719*/
	s_or_b32 s2, s2, vcc_lo
	s_or_b32 s5, s6, s5
	v_cndmask_b32_e64 v173 /*v429*/, v17 /*v273*/, 0xff800000, s2
	v_cmp_gt_i32_e32 vcc_lo, v232 /*v488*/, v213 /*v725*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s2, v232 /*v488*/, v207 /*v719*/
	v_cndmask_b32_e64 v172 /*v428*/, v16 /*v272*/, 0xff800000, s5
	v_cmp_gt_i32_e64 s5, v231 /*v487*/, v213 /*v725*/
	v_cmp_lt_i32_e64 s6, v231 /*v487*/, v207 /*v719*/
	v_cndmask_b32_e64 v174 /*v430*/, v58 /*v314*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v233 /*v489*/, v213 /*v725*/
	v_cmp_lt_i32_e64 s4, v233 /*v489*/, v207 /*v719*/
	s_or_b32 s2, s2, vcc_lo
	s_or_b32 s5, s6, s5
	v_cndmask_b32_e64 v176 /*v432*/, v60 /*v316*/, 0xff800000, s2
	v_cmp_gt_i32_e32 vcc_lo, v235 /*v491*/, v213 /*v725*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s2, v235 /*v491*/, v207 /*v719*/
	v_cndmask_b32_e64 v175 /*v431*/, v59 /*v315*/, 0xff800000, s5
	v_cmp_gt_i32_e64 s5, v234 /*v490*/, v213 /*v725*/
	v_cmp_lt_i32_e64 s6, v234 /*v490*/, v207 /*v719*/
	v_cndmask_b32_e64 v177 /*v433*/, v61 /*v317*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v236 /*v492*/, v213 /*v725*/
	v_cmp_lt_i32_e64 s4, v236 /*v492*/, v207 /*v719*/
	s_or_b32 s2, s2, vcc_lo
	s_or_b32 s5, s6, s5
	v_cndmask_b32_e64 v179 /*v435*/, v63 /*v319*/, 0xff800000, s2
	v_cmp_gt_i32_e32 vcc_lo, v238 /*v494*/, v213 /*v725*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s2, v238 /*v494*/, v207 /*v719*/
	v_cndmask_b32_e64 v178 /*v434*/, v62 /*v318*/, 0xff800000, s5
	v_cmp_gt_i32_e64 s5, v237 /*v493*/, v213 /*v725*/
	v_cmp_lt_i32_e64 s6, v237 /*v493*/, v207 /*v719*/
	v_cndmask_b32_e64 v180 /*v436*/, v64 /*v320*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v239 /*v495*/, v213 /*v725*/
	v_cmp_lt_i32_e64 s4, v239 /*v495*/, v207 /*v719*/
	s_set_vgpr_msb 0x4951
	v_wmma_f32_16x16x32_bf16 v[82:89] /*v[338:345]*/, v[18:25] /*v[274:281]*/, v[168:175], v[82:89] /*v[338:345]*/
	s_or_b32 s2, s2, vcc_lo
	s_or_b32 s5, s6, s5
	v_cndmask_b32_e64 v74 /*v330*/, v74 /*v330*/, 0xff800000, s2
	s_or_b32 s3, s4, s3
	s_set_vgpr_msb 0x5149
	v_cmp_gt_i32_e32 vcc_lo, v241 /*v497*/, v213 /*v725*/
	v_cmp_lt_i32_e64 s2, v241 /*v497*/, v207 /*v719*/
	v_cndmask_b32_e64 v181 /*v437*/, v65 /*v321*/, 0xff800000, s5
	v_cmp_gt_i32_e64 s5, v240 /*v496*/, v213 /*v725*/
	v_cmp_lt_i32_e64 s6, v240 /*v496*/, v207 /*v719*/
	v_cndmask_b32_e64 v75 /*v331*/, v75 /*v331*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v242 /*v498*/, v213 /*v725*/
	v_cmp_lt_i32_e64 s4, v242 /*v498*/, v207 /*v719*/
	s_set_vgpr_msb 0x4950
	v_wmma_f32_16x16x32_bf16 v[82:89] /*v[338:345]*/, v[232:239], v[176:183], v[82:89] /*v[338:345]*/
	s_or_b32 s2, s2, vcc_lo
	s_or_b32 s5, s6, s5
	s_set_vgpr_msb 0x5049
	v_cndmask_b32_e64 v77 /*v333*/, v77 /*v333*/, 0xff800000, s2
	s_or_b32 s3, s4, s3
	v_cmp_gt_i32_e32 vcc_lo, v182 /*v438*/, v213 /*v725*/
	v_cmp_lt_i32_e64 s2, v182 /*v438*/, v207 /*v719*/
	v_cndmask_b32_e64 v76 /*v332*/, v76 /*v332*/, 0xff800000, s5
	s_set_vgpr_msb 0x4900
	v_wmma_f32_16x16x32_bf16 v[192:199], v[216:223], v[160:167], 0
	s_set_vgpr_msb 0x49
	v_cmp_gt_i32_e64 s5, v243 /*v499*/, v213 /*v725*/
	v_cmp_lt_i32_e64 s6, v243 /*v499*/, v207 /*v719*/
	v_cndmask_b32_e64 v78 /*v334*/, v78 /*v334*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v183 /*v439*/, v213 /*v725*/
	v_cmp_lt_i32_e64 s4, v183 /*v439*/, v207 /*v719*/
	s_or_b32 s2, s2, vcc_lo
	s_or_b32 s5, s6, s5
	s_set_vgpr_msb 0x4950
	v_wmma_f32_16x16x32_bf16 v[82:89] /*v[338:345]*/, v[224:231], v[184:191], v[82:89] /*v[338:345]*/
	s_set_vgpr_msb 0x5049
	v_cndmask_b32_e64 v80 /*v336*/, v80 /*v336*/, 0xff800000, s2
	s_or_b32 s3, s4, s3
	v_cmp_gt_i32_e32 vcc_lo, v185 /*v441*/, v213 /*v725*/
	v_cmp_lt_i32_e64 s2, v185 /*v441*/, v207 /*v719*/
	v_cndmask_b32_e64 v79 /*v335*/, v79 /*v335*/, 0xff800000, s5
	v_cmp_gt_i32_e64 s5, v184 /*v440*/, v213 /*v725*/
	v_cmp_lt_i32_e64 s6, v184 /*v440*/, v207 /*v719*/
	s_set_vgpr_msb 0x4900
	v_wmma_f32_16x16x32_bf16 v[192:199], v[208:215], v[168:175], v[192:199]
	s_set_vgpr_msb 0x49
	v_cndmask_b32_e64 v81 /*v337*/, v81 /*v337*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v186 /*v442*/, v213 /*v725*/
	v_cmp_lt_i32_e64 s4, v186 /*v442*/, v207 /*v719*/
	s_or_b32 s2, s2, vcc_lo
	s_or_b32 s5, s6, s5
	v_cndmask_b32_e64 v83 /*v339*/, v83 /*v339*/, 0xff800000, s2
	v_cmp_gt_i32_e32 vcc_lo, v188 /*v444*/, v213 /*v725*/
	s_set_vgpr_msb 0x4900
	v_wmma_f32_16x16x32_bf16 v[192:199], v[200:207], v[176:183], v[192:199]
	s_or_b32 s3, s4, s3
	s_set_vgpr_msb 0x49
	v_cmp_lt_i32_e64 s2, v188 /*v444*/, v207 /*v719*/
	v_cndmask_b32_e64 v82 /*v338*/, v82 /*v338*/, 0xff800000, s5
	v_cmp_gt_i32_e64 s5, v187 /*v443*/, v213 /*v725*/
	v_cmp_lt_i32_e64 s6, v187 /*v443*/, v207 /*v719*/
	v_cndmask_b32_e64 v84 /*v340*/, v84 /*v340*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v189 /*v445*/, v213 /*v725*/
	v_cmp_lt_i32_e64 s4, v189 /*v445*/, v207 /*v719*/
	s_or_b32 s2, s2, vcc_lo
	s_or_b32 s5, s6, s5
	v_cndmask_b32_e64 v86 /*v342*/, v86 /*v342*/, 0xff800000, s2
	v_cmp_gt_i32_e32 vcc_lo, v191 /*v447*/, v213 /*v725*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s2, v191 /*v447*/, v207 /*v719*/
	s_set_vgpr_msb 0x4900
	v_wmma_f32_16x16x32_bf16 v[192:199], v[248:255], v[184:191], v[192:199]
	s_set_vgpr_msb 0x49
	v_cndmask_b32_e64 v85 /*v341*/, v85 /*v341*/, 0xff800000, s5
	v_cmp_gt_i32_e64 s5, v190 /*v446*/, v213 /*v725*/
	v_cmp_lt_i32_e64 s6, v190 /*v446*/, v207 /*v719*/
	v_cndmask_b32_e64 v87 /*v343*/, v87 /*v343*/, 0xff800000, s3
	v_cmp_gt_i32_e64 s3, v192 /*v448*/, v213 /*v725*/
	v_cmp_lt_i32_e64 s4, v192 /*v448*/, v207 /*v719*/
	s_or_b32 s2, s2, vcc_lo
	s_or_b32 s5, s6, s5
	v_cndmask_b32_e64 v89 /*v345*/, v89 /*v345*/, 0xff800000, s2
	v_cmp_gt_i32_e32 vcc_lo, v194 /*v450*/, v213 /*v725*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s2, v194 /*v450*/, v207 /*v719*/
	v_cndmask_b32_e64 v88 /*v344*/, v88 /*v344*/, 0xff800000, s5
	v_cmp_gt_i32_e64 s5, v193 /*v449*/, v213 /*v725*/
	v_cmp_lt_i32_e64 s6, v193 /*v449*/, v207 /*v719*/
	s_set_vgpr_msb 0x4940
	v_cndmask_b32_e64 v182 /*v438*/, v192, 0xff800000, s3
	s_set_vgpr_msb 0x4009
	v_cmp_gt_i32_e64 s3, v195 /*v451*/, v213 /*v725*/
	v_cmp_lt_i32_e64 s4, v195 /*v451*/, v207 /*v719*/
	s_or_b32 s2, s2, vcc_lo
	s_or_b32 s5, s6, s5
	s_set_vgpr_msb 0x940
	v_cndmask_b32_e64 v184 /*v440*/, v194, 0xff800000, s2
	s_set_vgpr_msb 0x4009
	v_cmp_gt_i32_e32 vcc_lo, v197 /*v453*/, v213 /*v725*/
	s_or_b32 s3, s4, s3
	v_cmp_lt_i32_e64 s2, v197 /*v453*/, v207 /*v719*/
	s_set_vgpr_msb 0x940
	v_cndmask_b32_e64 v183 /*v439*/, v193, 0xff800000, s5
	s_set_vgpr_msb 0x4009
	v_cmp_gt_i32_e64 s5, v196 /*v452*/, v213 /*v725*/
	v_cmp_lt_i32_e64 s6, v196 /*v452*/, v207 /*v719*/
	s_set_vgpr_msb 0x940
	v_cndmask_b32_e64 v185 /*v441*/, v195, 0xff800000, s3
	s_set_vgpr_msb 0x4009
	v_cmp_gt_i32_e64 s3, v244 /*v500*/, v213 /*v725*/
	v_cmp_lt_i32_e64 s4, v244 /*v500*/, v207 /*v719*/
	s_or_b32 s2, s2, vcc_lo
	s_or_b32 s9, s6, s5
	s_set_vgpr_msb 0x940
	v_cndmask_b32_e64 v187 /*v443*/, v197, 0xff800000, s2
	v_cndmask_b32_e64 v186 /*v442*/, v196, 0xff800000, s9
	s_or_b32 s2, s4, s3
	s_set_vgpr_msb 0x4015
	v_max3_num_f32 v192, v90 /*v346*/, v91 /*v347*/, v92 /*v348*/
	s_set_vgpr_msb 0x1540
	v_cndmask_b32_e64 v188 /*v444*/, v198, 0xff800000, s2
	s_set_vgpr_msb 0x4015
	v_max3_num_f32 v194, v93 /*v349*/, v94 /*v350*/, v95 /*v351*/
	v_max3_num_f32 v196, v96 /*v352*/, v97 /*v353*/, v98 /*v354*/
	v_max3_num_f32 v198, v99 /*v355*/, v100 /*v356*/, v101 /*v357*/
	v_max3_num_f32 v200, v104 /*v360*/, v105 /*v361*/, v102 /*v358*/
	v_max3_num_f32 v202, v103 /*v359*/, v108 /*v364*/, v109 /*v365*/
	v_max3_num_f32 v204, v106 /*v362*/, v107 /*v363*/, v110 /*v366*/
	v_max3_num_f32 v206, v111 /*v367*/, v112 /*v368*/, v113 /*v369*/
	v_max3_num_f32 v208, v114 /*v370*/, v115 /*v371*/, v116 /*v372*/
	v_max3_num_f32 v210, v117 /*v373*/, v118 /*v374*/, v119 /*v375*/
	v_max3_num_f32 v212, v120 /*v376*/, v121 /*v377*/, v122 /*v378*/
	v_max3_num_f32 v214, v123 /*v379*/, v124 /*v380*/, v125 /*v381*/
	v_max3_num_f32 v216, v126 /*v382*/, v127 /*v383*/, v128 /*v384*/
	v_max3_num_f32 v218, v129 /*v385*/, v130 /*v386*/, v131 /*v387*/
	v_max3_num_f32 v220, v132 /*v388*/, v133 /*v389*/, v134 /*v390*/
	v_max3_num_f32 v222, v135 /*v391*/, v136 /*v392*/, v137 /*v393*/
	v_max3_num_f32 v224, v138 /*v394*/, v139 /*v395*/, v140 /*v396*/
	v_max3_num_f32 v226, v141 /*v397*/, v142 /*v398*/, v143 /*v399*/
	v_max_num_f32_e32 v228, v144 /*v400*/, v145 /*v401*/
	v_max3_num_f32 v230, v147 /*v403*/, v66 /*v322*/, v67 /*v323*/
	s_set_vgpr_msb 0x1509
	v_cmp_gt_i32_e64 s5, v72 /*v328*/, v213 /*v725*/
	v_cmp_lt_i32_e64 s6, v72 /*v328*/, v207 /*v719*/
	s_set_vgpr_msb 0x915
	v_max3_num_f32 v232, v50 /*v306*/, v51 /*v307*/, v42 /*v298*/
	s_set_vgpr_msb 0x1500
	v_max3_num_f32 v192, v192, v194, v196
	v_max3_num_f32 v194, v198, v200, v202
	v_max3_num_f32 v196, v204, v206, v208
	v_max3_num_f32 v198, v210, v212, v214
	v_max3_num_f32 v200, v216, v218, v220
	v_max3_num_f32 v202, v222, v224, v226
	s_set_vgpr_msb 4
	v_max3_num_f32 v204, v228, v146 /*v402*/, v230
	s_or_b32 s3, s6, s5
	s_set_vgpr_msb 0x415
	v_max3_num_f32 v193, v34 /*v290*/, v35 /*v291*/, v26 /*v282*/
	s_set_vgpr_msb 0x1540
	v_cndmask_b32_e64 v189 /*v445*/, v199, 0xff800000, s3
	s_set_vgpr_msb 0x4015
	v_max3_num_f32 v195, v27 /*v283*/, v70 /*v326*/, v71 /*v327*/
	v_max3_num_f32 v197, v68 /*v324*/, v69 /*v325*/, v150 /*v406*/
	v_max3_num_f32 v199, v151 /*v407*/, v152 /*v408*/, v153 /*v409*/
	v_max3_num_f32 v201, v154 /*v410*/, v155 /*v411*/, v156 /*v412*/
	v_max3_num_f32 v203, v157 /*v413*/, v158 /*v414*/, v159 /*v415*/
	v_max3_num_f32 v205, v160 /*v416*/, v161 /*v417*/, v162 /*v418*/
	v_max3_num_f32 v207, v163 /*v419*/, v164 /*v420*/, v165 /*v421*/
	v_max3_num_f32 v209, v166 /*v422*/, v167 /*v423*/, v168 /*v424*/
	v_max3_num_f32 v211, v169 /*v425*/, v170 /*v426*/, v171 /*v427*/
	v_max3_num_f32 v213, v172 /*v428*/, v173 /*v429*/, v174 /*v430*/
	v_max3_num_f32 v215, v175 /*v431*/, v176 /*v432*/, v177 /*v433*/
	v_max3_num_f32 v217, v178 /*v434*/, v179 /*v435*/, v180 /*v436*/
	v_max3_num_f32 v219, v181 /*v437*/, v74 /*v330*/, v75 /*v331*/
	v_max3_num_f32 v221, v76 /*v332*/, v77 /*v333*/, v78 /*v334*/
	v_max3_num_f32 v223, v79 /*v335*/, v80 /*v336*/, v81 /*v337*/
	v_max3_num_f32 v225, v82 /*v338*/, v83 /*v339*/, v84 /*v340*/
	v_max3_num_f32 v227, v85 /*v341*/, v86 /*v342*/, v87 /*v343*/
	v_max_num_f32_e32 v229, v88 /*v344*/, v89 /*v345*/
	v_max3_num_f32 v231, v183 /*v439*/, v184 /*v440*/, v185 /*v441*/
	s_set_vgpr_msb 0x1500
	v_max3_num_f32 v192, v192, v194, v196
	v_max3_num_f32 v194, v198, v200, v202
	s_set_vgpr_msb 16
	v_max3_num_f32 v196, v204, v232, v43 /*v299*/
	s_set_vgpr_msb 0x1015
	v_max3_num_f32 v233, v186 /*v442*/, v187 /*v443*/, v188 /*v444*/
	s_set_vgpr_msb 0x1500
	v_max3_num_f32 v193, v193, v195, v197
	v_max3_num_f32 v195, v199, v201, v203
	v_max3_num_f32 v197, v205, v207, v209
	v_max3_num_f32 v198, v211, v213, v215
	v_max3_num_f32 v199, v217, v219, v221
	v_max3_num_f32 v200, v223, v225, v227
	v_max3_num_f32 v192, v192, v194, v196
	s_set_vgpr_msb 4
	v_max3_num_f32 v194, v229, v182 /*v438*/, v231
	s_set_vgpr_msb 0x400
	v_max3_num_f32 v193, v193, v195, v197
	v_max3_num_f32 v195, v198, v199, v200
	s_set_vgpr_msb 16
	v_max3_num_f32 v194, v194, v233, v189 /*v445*/
	s_set_vgpr_msb 0x1000
	v_max3_num_f32 v193, v193, v195, v194
	v_dual_mov_b32 v196, v192 :: v_dual_mov_b32 v194, v193
	v_permlanex16_b32 v196, v196, s15, 0xfedcba98
	v_permlanex16_b32 v194, v194, s15, 0xfedcba98
	v_dual_max_num_f32 v192, v192, v196 :: v_dual_max_num_f32 v193, v193, v194
	s_set_vgpr_msb 4
	v_sub_f32_e32 v195, v192, v73 /*v329*/
	v_max_num_f32_e32 v192, v192, v73 /*v329*/
	s_set_vgpr_msb 0x408
	v_sub_f32_e32 v194, v193, v211 /*v723*/
	s_set_vgpr_msb 0x800
	v_cmp_lt_f32_e32 vcc_lo, 0x41000000, v195
	s_cmp_eq_u32 vcc_lo, 0
	s_cselect_b32 s2, -1, 0
	s_set_vgpr_msb 0x84
	v_cndmask_b32_e64 v209 /*v721*/, v192, v73 /*v329*/, s2
	s_set_vgpr_msb 0x8402
	v_cmp_lt_f32_e64 s2, 0x41000000, v194
	v_max_num_f32_e32 v192, v211 /*v723*/, v193
	s_set_vgpr_msb 0x248
	v_mul_f32_e32 v62 /*v318*/, 0xbfb8aa3b, v209 /*v721*/
	s_cmp_lg_u32 s2, 0
	s_cselect_b32 s3, -1, 0
	s_cmp_eq_u32 s2, 0
	s_set_vgpr_msb 0x4811
	v_pk_fma_f32 v[204:205], v[96:97] /*v[352:353]*/, s[14:15], v[62:63] /*v[318:319]*/ op_sel_hi:[1,0,0]
	s_cselect_b32 s2, -1, 0
	v_pk_fma_f32 v[206:207], v[98:99] /*v[354:355]*/, s[14:15], v[62:63] /*v[318:319]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1148
	v_cndmask_b32_e64 v72 /*v328*/, v192, v211 /*v723*/, s2
	s_set_vgpr_msb 0x4811
	v_pk_fma_f32 v[192:193], v[90:91] /*v[346:347]*/, s[14:15], v[62:63] /*v[318:319]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[208:209], v[100:101] /*v[356:357]*/, s[14:15], v[62:63] /*v[318:319]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[228:229], v[116:117] /*v[372:373]*/, s[14:15], v[62:63] /*v[318:319]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1155
	v_pk_fma_f32 v[20:21] /*v[276:277]*/, v[132:133] /*v[388:389]*/, s[14:15], v[62:63] /*v[318:319]*/ op_sel_hi:[1,0,0]
	v_mul_f32_e32 v90 /*v346*/, 0xbfb8aa3b, v72 /*v328*/
	v_pk_fma_f32 v[60:61] /*v[316:317]*/, v[66:67] /*v[322:323]*/, s[14:15], v[62:63] /*v[318:319]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[64:65] /*v[320:321]*/, v[50:51] /*v[306:307]*/, s[14:15], v[62:63] /*v[318:319]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[66:67] /*v[322:323]*/, v[42:43] /*v[298:299]*/, s[14:15], v[62:63] /*v[318:319]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5511
	v_pk_fma_f32 v[196:197], v[92:93] /*v[348:349]*/, s[14:15], v[62:63] /*v[318:319]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[200:201], v[94:95] /*v[350:351]*/, s[14:15], v[62:63] /*v[318:319]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1100
	v_exp_f32_e32 v216, v204
	v_exp_f32_e32 v220, v205
	v_exp_f32_e32 v222, v206
	s_set_vgpr_msb 17
	v_pk_fma_f32 v[204:205], v[104:105] /*v[360:361]*/, s[14:15], v[62:63] /*v[318:319]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1100
	v_exp_f32_e32 v230, v207
	v_exp_f32_e32 v232, v208
	s_set_vgpr_msb 17
	v_pk_fma_f32 v[206:207], v[102:103] /*v[358:359]*/, s[14:15], v[62:63] /*v[318:319]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1100
	v_exp_f32_e32 v242, v209
	v_nop
	s_set_vgpr_msb 17
	v_pk_fma_f32 v[208:209], v[108:109] /*v[364:365]*/, s[14:15], v[62:63] /*v[318:319]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[210:211], v[106:107] /*v[362:363]*/, s[14:15], v[62:63] /*v[318:319]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[212:213], v[110:111] /*v[366:367]*/, s[14:15], v[62:63] /*v[318:319]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[218:219], v[112:113] /*v[368:369]*/, s[14:15], v[62:63] /*v[318:319]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[226:227], v[114:115] /*v[370:371]*/, s[14:15], v[62:63] /*v[318:319]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[238:239], v[118:119] /*v[374:375]*/, s[14:15], v[62:63] /*v[318:319]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1100
	v_exp_f32_e32 v236, v228
	s_set_vgpr_msb 17
	v_pk_fma_f32 v[240:241], v[120:121] /*v[376:377]*/, s[14:15], v[62:63] /*v[318:319]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1100
	v_exp_f32_e32 v246, v229
	v_nop
	s_set_vgpr_msb 17
	v_pk_fma_f32 v[228:229], v[122:123] /*v[378:379]*/, s[14:15], v[62:63] /*v[318:319]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[250:251], v[124:125] /*v[380:381]*/, s[14:15], v[62:63] /*v[318:319]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[252:253], v[126:127] /*v[382:383]*/, s[14:15], v[62:63] /*v[318:319]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1151
	v_pk_fma_f32 v[14:15] /*v[270:271]*/, v[128:129] /*v[384:385]*/, s[14:15], v[62:63] /*v[318:319]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[18:19] /*v[274:275]*/, v[130:131] /*v[386:387]*/, s[14:15], v[62:63] /*v[318:319]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[28:29] /*v[284:285]*/, v[134:135] /*v[390:391]*/, s[14:15], v[62:63] /*v[318:319]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v24 /*v280*/, v20 /*v276*/
	v_pk_fma_f32 v[30:31] /*v[286:287]*/, v[136:137] /*v[392:393]*/, s[14:15], v[62:63] /*v[318:319]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v32 /*v288*/, v21 /*v277*/
	v_nop
	v_pk_fma_f32 v[20:21] /*v[276:277]*/, v[138:139] /*v[394:395]*/, s[14:15], v[62:63] /*v[318:319]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[38:39] /*v[294:295]*/, v[140:141] /*v[396:397]*/, s[14:15], v[62:63] /*v[318:319]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[40:41] /*v[296:297]*/, v[142:143] /*v[398:399]*/, s[14:15], v[62:63] /*v[318:319]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[54:55] /*v[310:311]*/, v[144:145] /*v[400:401]*/, s[14:15], v[62:63] /*v[318:319]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[58:59] /*v[314:315]*/, v[146:147] /*v[402:403]*/, s[14:15], v[62:63] /*v[318:319]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v62 /*v318*/, v64 /*v320*/
	v_exp_f32_e32 v64 /*v320*/, v66 /*v322*/
	v_pk_fma_f32 v[94:95] /*v[350:351]*/, v[26:27] /*v[282:283]*/, s[14:15], v[90:91] /*v[346:347]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v26 /*v282*/, v67 /*v323*/
	v_nop
	v_pk_fma_f32 v[66:67] /*v[322:323]*/, v[70:71] /*v[326:327]*/, s[14:15], v[90:91] /*v[346:347]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[68:69] /*v[324:325]*/, v[68:69] /*v[324:325]*/, s[14:15], v[90:91] /*v[346:347]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[70:71] /*v[326:327]*/, v[150:151] /*v[406:407]*/, s[14:15], v[90:91] /*v[346:347]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5100
	v_exp_f32_e32 v202, v201
	v_exp_f32_e32 v244, v204
	s_set_vgpr_msb 1
	v_exp_f32_e32 v201, v66 /*v322*/
	v_exp_f32_e32 v203, v67 /*v323*/
	v_exp_f32_e32 v217, v68 /*v324*/
	s_set_vgpr_msb 0x151
	v_pk_fma_f32 v[66:67] /*v[322:323]*/, v[152:153] /*v[408:409]*/, s[14:15], v[90:91] /*v[346:347]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5101
	v_exp_f32_e32 v221, v69 /*v325*/
	v_exp_f32_e32 v223, v70 /*v326*/
	s_set_vgpr_msb 0x151
	v_pk_fma_f32 v[68:69] /*v[324:325]*/, v[154:155] /*v[410:411]*/, s[14:15], v[90:91] /*v[346:347]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5101
	v_exp_f32_e32 v231, v71 /*v327*/
	v_nop
	s_set_vgpr_msb 0x151
	v_pk_fma_f32 v[70:71] /*v[326:327]*/, v[156:157] /*v[412:413]*/, s[14:15], v[90:91] /*v[346:347]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5101
	v_exp_f32_e32 v233, v66 /*v322*/
	v_exp_f32_e32 v243, v67 /*v323*/
	v_exp_f32_e32 v245, v68 /*v324*/
	s_set_vgpr_msb 0x151
	v_pk_fma_f32 v[66:67] /*v[322:323]*/, v[158:159] /*v[414:415]*/, s[14:15], v[90:91] /*v[346:347]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5101
	v_exp_f32_e32 v255, v69 /*v325*/
	s_set_vgpr_msb 0x151
	v_exp_f32_e32 v3 /*v259*/, v70 /*v326*/
	v_pk_fma_f32 v[68:69] /*v[324:325]*/, v[160:161] /*v[416:417]*/, s[14:15], v[90:91] /*v[346:347]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v11 /*v267*/, v71 /*v327*/
	v_nop
	v_pk_fma_f32 v[70:71] /*v[326:327]*/, v[162:163] /*v[418:419]*/, s[14:15], v[90:91] /*v[346:347]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5100
	v_exp_f32_e32 v254, v205
	s_set_vgpr_msb 64
	v_exp_f32_e32 v2 /*v258*/, v206
	v_exp_f32_e32 v10 /*v266*/, v207
	s_set_vgpr_msb 0x4000
	v_exp_f32_e32 v204, v208
	v_exp_f32_e32 v206, v209
	v_exp_f32_e32 v208, v210
	v_exp_f32_e32 v210, v211
	v_exp_f32_e32 v214, v213
	s_set_vgpr_msb 1
	v_exp_f32_e32 v205, v66 /*v322*/
	v_exp_f32_e32 v207, v67 /*v323*/
	v_exp_f32_e32 v209, v68 /*v324*/
	s_set_vgpr_msb 0x151
	v_pk_fma_f32 v[66:67] /*v[322:323]*/, v[164:165] /*v[420:421]*/, s[14:15], v[90:91] /*v[346:347]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5101
	v_exp_f32_e32 v211, v69 /*v325*/
	v_exp_f32_e32 v213, v70 /*v326*/
	s_set_vgpr_msb 0x151
	v_pk_fma_f32 v[68:69] /*v[324:325]*/, v[166:167] /*v[422:423]*/, s[14:15], v[90:91] /*v[346:347]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5101
	v_exp_f32_e32 v215, v71 /*v327*/
	v_nop
	s_set_vgpr_msb 0x151
	v_pk_fma_f32 v[70:71] /*v[326:327]*/, v[168:169] /*v[424:425]*/, s[14:15], v[90:91] /*v[346:347]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5100
	v_exp_f32_e32 v224, v219
	v_exp_f32_e32 v234, v227
	s_set_vgpr_msb 1
	v_exp_f32_e32 v219, v66 /*v322*/
	v_exp_f32_e32 v225, v67 /*v323*/
	v_exp_f32_e32 v227, v68 /*v324*/
	s_set_vgpr_msb 0x151
	v_pk_fma_f32 v[66:67] /*v[322:323]*/, v[170:171] /*v[426:427]*/, s[14:15], v[90:91] /*v[346:347]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5101
	v_exp_f32_e32 v235, v69 /*v325*/
	v_exp_f32_e32 v237, v70 /*v326*/
	s_set_vgpr_msb 0x151
	v_pk_fma_f32 v[68:69] /*v[324:325]*/, v[172:173] /*v[428:429]*/, s[14:15], v[90:91] /*v[346:347]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5101
	v_exp_f32_e32 v247, v71 /*v327*/
	v_nop
	s_set_vgpr_msb 0x151
	v_pk_fma_f32 v[70:71] /*v[326:327]*/, v[174:175] /*v[430:431]*/, s[14:15], v[90:91] /*v[346:347]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5100
	v_exp_f32_e32 v248, v238
	s_set_vgpr_msb 64
	v_exp_f32_e32 v4 /*v260*/, v239
	s_set_vgpr_msb 0x4000
	v_exp_f32_e32 v238, v229
	s_set_vgpr_msb 1
	v_exp_f32_e32 v249, v66 /*v322*/
	s_set_vgpr_msb 0x151
	v_exp_f32_e32 v5 /*v261*/, v67 /*v323*/
	v_exp_f32_e32 v7 /*v263*/, v68 /*v324*/
	v_pk_fma_f32 v[66:67] /*v[322:323]*/, v[176:177] /*v[432:433]*/, s[14:15], v[90:91] /*v[346:347]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v13 /*v269*/, v69 /*v325*/
	s_set_vgpr_msb 0x5101
	v_exp_f32_e32 v229, v70 /*v326*/
	s_set_vgpr_msb 0x151
	v_pk_fma_f32 v[68:69] /*v[324:325]*/, v[178:179] /*v[434:435]*/, s[14:15], v[90:91] /*v[346:347]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5101
	v_exp_f32_e32 v239, v71 /*v327*/
	v_nop
	s_set_vgpr_msb 0x151
	v_pk_fma_f32 v[70:71] /*v[326:327]*/, v[180:181] /*v[436:437]*/, s[14:15], v[90:91] /*v[346:347]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5140
	v_exp_f32_e32 v6 /*v262*/, v240
	v_exp_f32_e32 v12 /*v268*/, v241
	s_set_vgpr_msb 0x4000
	v_exp_f32_e32 v240, v250
	v_exp_f32_e32 v250, v251
	s_set_vgpr_msb 64
	v_exp_f32_e32 v8 /*v264*/, v253
	s_set_vgpr_msb 0x4041
	v_exp_f32_e32 v16 /*v272*/, v15 /*v271*/
	s_set_vgpr_msb 0x4101
	v_exp_f32_e32 v241, v66 /*v322*/
	v_exp_f32_e32 v251, v67 /*v323*/
	v_exp_f32_e32 v253, v68 /*v324*/
	s_set_vgpr_msb 0x151
	v_pk_fma_f32 v[66:67] /*v[322:323]*/, v[74:75] /*v[330:331]*/, s[14:15], v[90:91] /*v[346:347]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v9 /*v265*/, v69 /*v325*/
	v_exp_f32_e32 v15 /*v271*/, v70 /*v326*/
	v_pk_fma_f32 v[68:69] /*v[324:325]*/, v[76:77] /*v[332:333]*/, s[14:15], v[90:91] /*v[346:347]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v17 /*v273*/, v71 /*v327*/
	v_nop
	v_pk_fma_f32 v[70:71] /*v[326:327]*/, v[78:79] /*v[334:335]*/, s[14:15], v[90:91] /*v[346:347]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[92:93] /*v[348:349]*/, v[34:35] /*v[290:291]*/, s[14:15], v[90:91] /*v[346:347]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v22 /*v278*/, v19 /*v275*/
	v_exp_f32_e32 v19 /*v275*/, v66 /*v322*/
	v_exp_f32_e32 v23 /*v279*/, v67 /*v323*/
	v_exp_f32_e32 v25 /*v281*/, v68 /*v324*/
	v_pk_fma_f32 v[66:67] /*v[322:323]*/, v[80:81] /*v[336:337]*/, s[14:15], v[90:91] /*v[346:347]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v33 /*v289*/, v69 /*v325*/
	v_exp_f32_e32 v37 /*v293*/, v70 /*v326*/
	v_pk_fma_f32 v[68:69] /*v[324:325]*/, v[82:83] /*v[338:339]*/, s[14:15], v[90:91] /*v[346:347]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v45 /*v301*/, v71 /*v327*/
	v_nop
	v_pk_fma_f32 v[70:71] /*v[326:327]*/, v[84:85] /*v[340:341]*/, s[14:15], v[90:91] /*v[346:347]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5100
	v_exp_f32_e32 v192, v192
	v_exp_f32_e32 v194, v193
	v_exp_f32_e32 v198, v197
	v_exp_f32_e32 v200, v200
	s_set_vgpr_msb 1
	v_exp_f32_e32 v193, v92 /*v348*/
	v_exp_f32_e32 v195, v93 /*v349*/
	v_exp_f32_e32 v199, v95 /*v351*/
	s_set_vgpr_msb 0x151
	v_exp_f32_e32 v36 /*v292*/, v28 /*v284*/
	v_exp_f32_e32 v44 /*v300*/, v29 /*v285*/
	v_exp_f32_e32 v46 /*v302*/, v30 /*v286*/
	v_exp_f32_e32 v52 /*v308*/, v31 /*v287*/
	v_exp_f32_e32 v28 /*v284*/, v21 /*v277*/
	v_exp_f32_e32 v30 /*v286*/, v38 /*v294*/
	v_exp_f32_e32 v38 /*v294*/, v39 /*v295*/
	v_exp_f32_e32 v47 /*v303*/, v66 /*v322*/
	v_exp_f32_e32 v53 /*v309*/, v67 /*v323*/
	v_exp_f32_e32 v21 /*v277*/, v68 /*v324*/
	v_pk_fma_f32 v[66:67] /*v[322:323]*/, v[86:87] /*v[342:343]*/, s[14:15], v[90:91] /*v[346:347]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v29 /*v285*/, v69 /*v325*/
	v_exp_f32_e32 v31 /*v287*/, v70 /*v326*/
	v_exp_f32_e32 v39 /*v295*/, v71 /*v327*/
	v_pk_fma_f32 v[68:69] /*v[324:325]*/, v[88:89] /*v[344:345]*/, s[14:15], v[90:91] /*v[346:347]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[70:71] /*v[326:327]*/, v[182:183] /*v[438:439]*/, s[14:15], v[90:91] /*v[346:347]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5100
	v_exp_f32_e32 v196, v196
	v_exp_f32_e32 v252, v252
	s_set_vgpr_msb 0x41
	v_exp_f32_e32 v18 /*v274*/, v18 /*v274*/
	s_set_vgpr_msb 0x4101
	v_exp_f32_e32 v197, v94 /*v350*/
	s_set_vgpr_msb 0x151
	v_exp_f32_e32 v48 /*v304*/, v41 /*v297*/
	v_exp_f32_e32 v56 /*v312*/, v55 /*v311*/
	v_exp_f32_e32 v50 /*v306*/, v59 /*v315*/
	v_exp_f32_e32 v41 /*v297*/, v66 /*v322*/
	v_exp_f32_e32 v49 /*v305*/, v67 /*v323*/
	v_nop
	v_pk_fma_f32 v[66:67] /*v[322:323]*/, v[184:185] /*v[440:441]*/, s[14:15], v[90:91] /*v[346:347]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v55 /*v311*/, v68 /*v324*/
	v_exp_f32_e32 v57 /*v313*/, v69 /*v325*/
	v_exp_f32_e32 v59 /*v315*/, v70 /*v326*/
	v_exp_f32_e32 v51 /*v307*/, v71 /*v327*/
	v_pk_fma_f32 v[68:69] /*v[324:325]*/, v[186:187] /*v[442:443]*/, s[14:15], v[90:91] /*v[346:347]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5140
	v_pk_add_f32 v[70:71] /*v[326:327]*/, v[192:193], v[194:195]
	v_pk_add_f32 v[74:75] /*v[330:331]*/, v[198:199], v[200:201]
	s_set_vgpr_msb 0x4000
	v_exp_f32_e32 v218, v218
	v_exp_f32_e32 v228, v228
	s_set_vgpr_msb 0x51
	v_exp_f32_e32 v14 /*v270*/, v14 /*v270*/
	v_exp_f32_e32 v20 /*v276*/, v20 /*v276*/
	v_exp_f32_e32 v42 /*v298*/, v61 /*v317*/
	v_exp_f32_e32 v61 /*v317*/, v66 /*v322*/
	v_exp_f32_e32 v43 /*v299*/, v67 /*v323*/
	v_nop
	v_pk_fma_f32 v[66:67] /*v[322:323]*/, v[188:189] /*v[444:445]*/, s[14:15], v[90:91] /*v[346:347]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v63 /*v319*/, v68 /*v324*/
	v_exp_f32_e32 v35 /*v291*/, v69 /*v325*/
	v_nop
	v_pk_add_f32 v[68:69] /*v[324:325]*/, v[70:71] /*v[326:327]*/, v[196:197]
	v_pk_add_f32 v[70:71] /*v[326:327]*/, v[74:75] /*v[330:331]*/, v[202:203]
	s_set_vgpr_msb 0x5140
	v_pk_add_f32 v[74:75] /*v[330:331]*/, v[216:217], v[220:221]
	v_pk_add_f32 v[76:77] /*v[332:333]*/, v[230:231], v[232:233]
	v_pk_add_f32 v[78:79] /*v[334:335]*/, v[244:245], v[254:255]
	v_pk_add_f32 v[88:89] /*v[344:345]*/, v[246:247], v[248:249]
	s_set_vgpr_msb 0x4045
	v_pk_add_f32 v[90:91] /*v[346:347]*/, v[6:7] /*v[262:263]*/, v[12:13] /*v[268:269]*/
	s_set_vgpr_msb 0x4544
	v_pk_add_f32 v[94:95] /*v[350:351]*/, v[252:253], v[8:9] /*v[264:265]*/
	s_set_vgpr_msb 0x4445
	v_pk_add_f32 v[96:97] /*v[352:353]*/, v[16:17] /*v[272:273]*/, v[18:19] /*v[274:275]*/
	s_set_vgpr_msb 0x4500
	v_exp_f32_e32 v212, v212
	v_exp_f32_e32 v226, v226
	s_set_vgpr_msb 0x41
	v_exp_f32_e32 v40 /*v296*/, v40 /*v296*/
	v_exp_f32_e32 v54 /*v310*/, v54 /*v310*/
	v_exp_f32_e32 v60 /*v316*/, v60 /*v316*/
	v_pk_add_f32 v[80:81] /*v[336:337]*/, v[10:11] /*v[266:267]*/, v[204:205]
	s_set_vgpr_msb 0x4140
	v_pk_add_f32 v[82:83] /*v[338:339]*/, v[208:209], v[210:211]
	s_set_vgpr_msb 0x4044
	v_pk_add_f32 v[74:75] /*v[330:331]*/, v[222:223], v[74:75] /*v[330:331]*/
	v_pk_add_f32 v[76:77] /*v[332:333]*/, v[242:243], v[76:77] /*v[332:333]*/
	s_set_vgpr_msb 0x4445
	v_pk_add_f32 v[78:79] /*v[334:335]*/, v[2:3] /*v[258:259]*/, v[78:79] /*v[334:335]*/
	s_set_vgpr_msb 0x4540
	v_pk_add_f32 v[84:85] /*v[340:341]*/, v[214:215], v[218:219]
	v_pk_add_f32 v[92:93] /*v[348:349]*/, v[238:239], v[240:241]
	s_set_vgpr_msb 0x4045
	v_pk_add_f32 v[88:89] /*v[344:345]*/, v[4:5] /*v[260:261]*/, v[88:89] /*v[344:345]*/
	s_set_vgpr_msb 0x4544
	v_pk_add_f32 v[90:91] /*v[346:347]*/, v[228:229], v[90:91] /*v[346:347]*/
	s_set_vgpr_msb 0x4445
	v_pk_add_f32 v[98:99] /*v[354:355]*/, v[24:25] /*v[280:281]*/, v[32:33] /*v[288:289]*/
	v_pk_add_f32 v[100:101] /*v[356:357]*/, v[44:45] /*v[300:301]*/, v[46:47] /*v[302:303]*/
	v_pk_add_f32 v[102:103] /*v[358:359]*/, v[20:21] /*v[276:277]*/, v[28:29] /*v[284:285]*/
	v_pk_add_f32 v[94:95] /*v[350:351]*/, v[14:15] /*v[270:271]*/, v[94:95] /*v[350:351]*/
	v_pk_add_f32 v[96:97] /*v[352:353]*/, v[22:23] /*v[278:279]*/, v[96:97] /*v[352:353]*/
	v_pk_add_f32 v[68:69] /*v[324:325]*/, v[68:69] /*v[324:325]*/, v[70:71] /*v[326:327]*/
	v_exp_f32_e32 v58 /*v314*/, v58 /*v314*/
	v_exp_f32_e32 v34 /*v290*/, v65 /*v321*/
	s_set_vgpr_msb 0x4544
	v_pk_add_f32 v[80:81] /*v[336:337]*/, v[206:207], v[80:81] /*v[336:337]*/
	v_pk_add_f32 v[82:83] /*v[338:339]*/, v[212:213], v[82:83] /*v[338:339]*/
	s_set_vgpr_msb 0x4440
	v_pk_add_f32 v[86:87] /*v[342:343]*/, v[226:227], v[234:235]
	s_set_vgpr_msb 0x4044
	v_pk_add_f32 v[84:85] /*v[340:341]*/, v[224:225], v[84:85] /*v[340:341]*/
	v_pk_add_f32 v[92:93] /*v[348:349]*/, v[250:251], v[92:93] /*v[348:349]*/
	s_set_vgpr_msb 0x4445
	v_pk_add_f32 v[98:99] /*v[354:355]*/, v[36:37] /*v[292:293]*/, v[98:99] /*v[354:355]*/
	v_pk_add_f32 v[100:101] /*v[356:357]*/, v[52:53] /*v[308:309]*/, v[100:101] /*v[356:357]*/
	v_pk_add_f32 v[102:103] /*v[358:359]*/, v[30:31] /*v[286:287]*/, v[102:103] /*v[358:359]*/
	v_pk_add_f32 v[104:105] /*v[360:361]*/, v[38:39] /*v[294:295]*/, v[40:41] /*v[296:297]*/
	v_pk_add_f32 v[106:107] /*v[362:363]*/, v[54:55] /*v[310:311]*/, v[56:57] /*v[312:313]*/
	v_pk_add_f32 v[108:109] /*v[364:365]*/, v[50:51] /*v[306:307]*/, v[60:61] /*v[316:317]*/
	v_pk_add_f32 v[68:69] /*v[324:325]*/, v[74:75] /*v[330:331]*/, v[68:69] /*v[324:325]*/
	v_pk_add_f32 v[74:75] /*v[330:331]*/, v[76:77] /*v[332:333]*/, v[78:79] /*v[334:335]*/
	v_pk_add_f32 v[76:77] /*v[332:333]*/, v[88:89] /*v[344:345]*/, v[90:91] /*v[346:347]*/
	v_pk_add_f32 v[78:79] /*v[334:335]*/, v[94:95] /*v[350:351]*/, v[96:97] /*v[352:353]*/
	v_exp_f32_e32 v65 /*v321*/, v66 /*v322*/
	s_set_vgpr_msb 0x4544
	v_pk_add_f32 v[86:87] /*v[342:343]*/, v[236:237], v[86:87] /*v[342:343]*/
	s_set_vgpr_msb 0x4445
	v_pk_add_f32 v[110:111] /*v[366:367]*/, v[62:63] /*v[318:319]*/, v[34:35] /*v[290:291]*/
	v_pk_add_f32 v[70:71] /*v[326:327]*/, v[48:49] /*v[304:305]*/, v[104:105] /*v[360:361]*/
	v_pk_add_f32 v[104:105] /*v[360:361]*/, v[58:59] /*v[314:315]*/, v[106:107] /*v[362:363]*/
	v_pk_add_f32 v[106:107] /*v[362:363]*/, v[42:43] /*v[298:299]*/, v[108:109] /*v[364:365]*/
	v_pk_add_f32 v[82:83] /*v[338:339]*/, v[82:83] /*v[338:339]*/, v[84:85] /*v[340:341]*/
	v_pk_add_f32 v[84:85] /*v[340:341]*/, v[100:101] /*v[356:357]*/, v[102:103] /*v[358:359]*/
	v_pk_add_f32 v[74:75] /*v[330:331]*/, v[80:81] /*v[336:337]*/, v[74:75] /*v[330:331]*/
	v_pk_add_f32 v[76:77] /*v[332:333]*/, v[92:93] /*v[348:349]*/, v[76:77] /*v[332:333]*/
	v_pk_add_f32 v[78:79] /*v[334:335]*/, v[98:99] /*v[354:355]*/, v[78:79] /*v[334:335]*/
	v_pk_add_f32 v[108:109] /*v[364:365]*/, v[64:65] /*v[320:321]*/, v[110:111] /*v[366:367]*/
	v_pk_add_f32 v[80:81] /*v[336:337]*/, v[86:87] /*v[342:343]*/, v[82:83] /*v[338:339]*/
	v_pk_add_f32 v[70:71] /*v[326:327]*/, v[70:71] /*v[326:327]*/, v[84:85] /*v[340:341]*/
	v_pk_add_f32 v[82:83] /*v[338:339]*/, v[104:105] /*v[360:361]*/, v[106:107] /*v[362:363]*/
	v_pk_add_f32 v[68:69] /*v[324:325]*/, v[68:69] /*v[324:325]*/, v[74:75] /*v[330:331]*/
	v_pk_add_f32 v[74:75] /*v[330:331]*/, v[76:77] /*v[332:333]*/, v[78:79] /*v[334:335]*/
	v_exp_f32_e32 v27 /*v283*/, v67 /*v323*/
	v_nop
	v_pk_add_f32 v[66:67] /*v[322:323]*/, v[108:109] /*v[364:365]*/, v[82:83] /*v[338:339]*/
	v_pk_add_f32 v[68:69] /*v[324:325]*/, v[80:81] /*v[336:337]*/, v[68:69] /*v[324:325]*/
	v_pk_add_f32 v[70:71] /*v[326:327]*/, v[70:71] /*v[326:327]*/, v[74:75] /*v[330:331]*/
	v_pk_add_f32 v[66:67] /*v[322:323]*/, v[26:27] /*v[282:283]*/, v[66:67] /*v[322:323]*/
	v_pk_add_f32 v[68:69] /*v[324:325]*/, v[68:69] /*v[324:325]*/, v[70:71] /*v[326:327]*/
	s_set_vgpr_msb 0x4549
	v_sub_f32_e32 v70 /*v326*/, v73 /*v329*/, v209 /*v721*/
	s_set_vgpr_msb 0x4945
	v_pk_add_f32 v[66:67] /*v[322:323]*/, v[66:67] /*v[322:323]*/, v[68:69] /*v[324:325]*/
	v_mul_f32_e32 v70 /*v326*/, 0x3fb8aa3b, v70 /*v326*/
	v_dual_mov_b32 v68 /*v324*/, v66 /*v322*/ :: v_dual_mov_b32 v69 /*v325*/, v67 /*v323*/
	v_exp_f32_e32 v70 /*v326*/, v70 /*v326*/
	v_permlanex16_b32 v68 /*v324*/, v68 /*v324*/, s15, 0xfedcba98
	v_permlanex16_b32 v69 /*v325*/, v69 /*v325*/, s15, 0xfedcba98
	s_set_vgpr_msb 0x4500
	s_cbranch_vccz .LBB0_11
	s_set_vgpr_msb 4
	v_pk_mul_f32 v[126:127], v[126:127], v[70:71] /*v[326:327]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[124:125], v[124:125], v[70:71] /*v[326:327]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[122:123], v[122:123], v[70:71] /*v[326:327]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[120:121], v[120:121], v[70:71] /*v[326:327]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[118:119], v[118:119], v[70:71] /*v[326:327]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[116:117], v[116:117], v[70:71] /*v[326:327]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[114:115], v[114:115], v[70:71] /*v[326:327]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[112:113], v[112:113], v[70:71] /*v[326:327]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[110:111], v[110:111], v[70:71] /*v[326:327]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[108:109], v[108:109], v[70:71] /*v[326:327]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[106:107], v[106:107], v[70:71] /*v[326:327]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[104:105], v[104:105], v[70:71] /*v[326:327]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[102:103], v[102:103], v[70:71] /*v[326:327]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[100:101], v[100:101], v[70:71] /*v[326:327]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[98:99], v[98:99], v[70:71] /*v[326:327]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[96:97], v[96:97], v[70:71] /*v[326:327]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[94:95], v[94:95], v[70:71] /*v[326:327]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[92:93], v[92:93], v[70:71] /*v[326:327]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[90:91], v[90:91], v[70:71] /*v[326:327]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[88:89], v[88:89], v[70:71] /*v[326:327]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[86:87], v[86:87], v[70:71] /*v[326:327]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[84:85], v[84:85], v[70:71] /*v[326:327]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[82:83], v[82:83], v[70:71] /*v[326:327]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[80:81], v[80:81], v[70:71] /*v[326:327]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[78:79], v[78:79], v[70:71] /*v[326:327]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[76:77], v[76:77], v[70:71] /*v[326:327]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[74:75], v[74:75], v[70:71] /*v[326:327]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[72:73], v[72:73], v[70:71] /*v[326:327]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[70:71], v[70:71], v[70:71] /*v[326:327]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[68:69], v[68:69], v[70:71] /*v[326:327]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[66:67], v[66:67], v[70:71] /*v[326:327]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[64:65], v[64:65], v[70:71] /*v[326:327]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x400
.LBB0_11:
	s_set_vgpr_msb 0x46
	v_sub_f32_e32 v71 /*v327*/, v211 /*v723*/, v72 /*v328*/
	s_and_b32 s2, s3, exec_lo
	s_cselect_b32 s2, 1, 0
	s_cmp_lg_u32 s2, 1
	v_mul_f32_e32 v71 /*v327*/, 0x3fb8aa3b, v71 /*v327*/
	s_set_vgpr_msb 0x4641
	v_exp_f32_e32 v71 /*v327*/, v71 /*v327*/
	s_set_vgpr_msb 0x4100
	s_cbranch_scc1 .LBB0_13
	v_nop
	s_set_vgpr_msb 0x41
	v_mov_b32_e32 v74 /*v330*/, v71 /*v327*/
	s_set_vgpr_msb 0x4104
	v_pk_mul_f32 v[62:63], v[62:63], v[74:75] /*v[330:331]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[60:61], v[60:61], v[74:75] /*v[330:331]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59], v[58:59], v[74:75] /*v[330:331]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[56:57], v[56:57], v[74:75] /*v[330:331]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[54:55], v[54:55], v[74:75] /*v[330:331]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[52:53], v[52:53], v[74:75] /*v[330:331]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[50:51], v[50:51], v[74:75] /*v[330:331]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[48:49], v[48:49], v[74:75] /*v[330:331]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[46:47], v[46:47], v[74:75] /*v[330:331]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[44:45], v[44:45], v[74:75] /*v[330:331]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[42:43], v[42:43], v[74:75] /*v[330:331]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[40:41], v[40:41], v[74:75] /*v[330:331]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39], v[38:39], v[74:75] /*v[330:331]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[36:37], v[36:37], v[74:75] /*v[330:331]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35], v[34:35], v[74:75] /*v[330:331]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[32:33], v[32:33], v[74:75] /*v[330:331]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[30:31], v[30:31], v[74:75] /*v[330:331]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[28:29], v[28:29], v[74:75] /*v[330:331]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[26:27], v[26:27], v[74:75] /*v[330:331]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25], v[24:25], v[74:75] /*v[330:331]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[22:23], v[22:23], v[74:75] /*v[330:331]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21], v[20:21], v[74:75] /*v[330:331]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[18:19], v[18:19], v[74:75] /*v[330:331]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[16:17], v[16:17], v[74:75] /*v[330:331]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[14:15], v[74:75] /*v[330:331]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13], v[12:13], v[74:75] /*v[330:331]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[10:11], v[74:75] /*v[330:331]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[8:9], v[74:75] /*v[330:331]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[6:7], v[74:75] /*v[330:331]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[4:5], v[74:75] /*v[330:331]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[2:3], v[2:3], v[74:75] /*v[330:331]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[0:1], v[0:1], v[74:75] /*v[330:331]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x400
.LBB0_13:
	s_set_vgpr_msb 64
	v_cvt_pk_bf16_f32 v138 /*v394*/, v192, v194
	v_cvt_pk_bf16_f32 v151 /*v407*/, v236, v246
	v_cvt_pk_bf16_f32 v157 /*v413*/, v217, v221
	v_cvt_pk_bf16_f32 v147 /*v403*/, v208, v210
	s_set_vgpr_msb 0x4000
	v_cvt_pk_bf16_f32 v221, v237, v247
	v_cvt_pk_bf16_f32 v192, v228, v238
	v_cvt_pk_bf16_f32 v208, v229, v239
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 2
	ds_load_tr16_b128 v[236:239], v13 /*v525*/ offset:4672
	s_set_vgpr_msb 0x240
	v_cvt_pk_bf16_f32 v144 /*v400*/, v244, v254
	v_cvt_pk_bf16_f32 v143 /*v399*/, v232, v242
	v_cvt_pk_bf16_f32 v141 /*v397*/, v216, v220
	v_cvt_pk_bf16_f32 v150 /*v406*/, v226, v234
	v_cvt_pk_bf16_f32 v160 /*v416*/, v245, v255
	v_cvt_pk_bf16_f32 v159 /*v415*/, v233, v243
	v_cvt_pk_bf16_f32 v154 /*v410*/, v193, v195
	s_set_vgpr_msb 0x4000
	v_cvt_pk_bf16_f32 v220, v227, v235
	v_cvt_pk_bf16_f32 v217, v209, v211
	v_cvt_pk_bf16_f32 v193, v240, v250
	v_cvt_pk_bf16_f32 v209, v241, v251
	s_wait_alu depctr_va_vdst(3)
	s_set_vgpr_msb 2
	ds_load_tr16_b128 v[232:235], v13 /*v525*/ offset:64
	s_wait_alu depctr_va_vdst(0)
	ds_load_tr16_b128 v[240:243], v13 /*v525*/ offset:96
	ds_load_tr16_b128 v[244:247], v13 /*v525*/ offset:4704
	s_set_vgpr_msb 0x245
	v_cvt_pk_bf16_f32 v145 /*v401*/, v2 /*v258*/, v10 /*v266*/
	s_set_vgpr_msb 0x4540
	v_cvt_pk_bf16_f32 v142 /*v398*/, v222, v230
	v_cvt_pk_bf16_f32 v140 /*v396*/, v200, v202
	v_cvt_pk_bf16_f32 v139 /*v395*/, v196, v198
	s_set_vgpr_msb 0x4045
	v_cvt_pk_bf16_f32 v161 /*v417*/, v3 /*v259*/, v11 /*v267*/
	s_set_vgpr_msb 0x4540
	v_cvt_pk_bf16_f32 v158 /*v414*/, v223, v231
	v_cvt_pk_bf16_f32 v156 /*v412*/, v201, v203
	v_cvt_pk_bf16_f32 v155 /*v411*/, v197, v199
	s_set_vgpr_msb 0x4004
	v_cvt_pk_bf16_f32 v194, v252, v8 /*v264*/
	v_cvt_pk_bf16_f32 v210, v253, v9 /*v265*/
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x402
	ds_load_tr16_b128 v[252:255], v13 /*v525*/ offset:13888
	s_set_vgpr_msb 0x244
	v_cvt_pk_bf16_f32 v152 /*v408*/, v248, v4 /*v260*/
	s_set_vgpr_msb 0x4404
	v_cvt_pk_bf16_f32 v222, v249, v5 /*v261*/
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x32_bf16 v[104:111], v[232:239], v[138:145] /*v[394:401]*/, v[104:111]
	s_set_vgpr_msb 0x445
	v_cvt_pk_bf16_f32 v153 /*v409*/, v6 /*v262*/, v12 /*v268*/
	s_set_vgpr_msb 0x4540
	v_cvt_pk_bf16_f32 v149 /*v405*/, v218, v224
	v_cvt_pk_bf16_f32 v148 /*v404*/, v212, v214
	v_cvt_pk_bf16_f32 v146 /*v402*/, v204, v206
	s_set_vgpr_msb 0x4005
	v_cvt_pk_bf16_f32 v223, v7 /*v263*/, v13 /*v269*/
	s_set_vgpr_msb 0x500
	v_cvt_pk_bf16_f32 v219, v219, v225
	v_cvt_pk_bf16_f32 v218, v213, v215
	s_set_vgpr_msb 4
	v_wmma_f32_16x16x32_bf16 v[40:47], v[232:239], v[154:161] /*v[410:417]*/, v[40:47]
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x402
	ds_load_tr16_b128 v[248:251], v13 /*v525*/ offset:9280
	ds_load_tr16_b128 v[232:235], v13 /*v525*/ offset:9312
	ds_load_tr16_b128 v[236:239], v13 /*v525*/ offset:13920
	s_set_vgpr_msb 0x200
	v_cvt_pk_bf16_f32 v216, v205, v207
	s_set_vgpr_msb 0x42
	ds_load_tr16_b128 v[6:9] /*v[262:265]*/, v13 /*v525*/ offset:23104
	ds_load_tr16_b128 v[10:13] /*v[266:269]*/, v13 /*v525*/ offset:32320
	s_set_vgpr_msb 0x4205
	v_cvt_pk_bf16_f32 v199, v46 /*v302*/, v52 /*v308*/
	v_cvt_pk_bf16_f32 v198, v36 /*v292*/, v44 /*v300*/
	v_cvt_pk_bf16_f32 v197, v24 /*v280*/, v32 /*v288*/
	s_set_vgpr_msb 0x504
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x32_bf16 v[96:103], v[240:247], v[138:145] /*v[394:401]*/, v[96:103]
	s_set_vgpr_msb 0x405
	v_cvt_pk_bf16_f32 v196, v18 /*v274*/, v22 /*v278*/
	v_cvt_pk_bf16_f32 v195, v14 /*v270*/, v16 /*v272*/
	v_cvt_pk_bf16_f32 v215, v47 /*v303*/, v53 /*v309*/
	v_cvt_pk_bf16_f32 v214, v37 /*v293*/, v45 /*v301*/
	v_cvt_pk_bf16_f32 v213, v25 /*v281*/, v33 /*v289*/
	v_cvt_pk_bf16_f32 v212, v19 /*v275*/, v23 /*v279*/
	v_cvt_pk_bf16_f32 v211, v15 /*v271*/, v17 /*v273*/
	s_set_vgpr_msb 0x504
	v_wmma_f32_16x16x32_bf16 v[32:39], v[240:247], v[154:161] /*v[410:417]*/, v[32:39]
	s_set_vgpr_msb 0x405
	v_cvt_pk_bf16_f32 v200, v20 /*v276*/, v28 /*v284*/
	v_cvt_pk_bf16_f32 v224, v21 /*v277*/, v29 /*v285*/
	v_cvt_pk_bf16_f32 v207, v64 /*v320*/, v26 /*v282*/
	v_cvt_pk_bf16_f32 v206, v62 /*v318*/, v34 /*v290*/
	v_cvt_pk_bf16_f32 v205, v60 /*v316*/, v42 /*v298*/
	v_cvt_pk_bf16_f32 v204, v58 /*v314*/, v50 /*v306*/
	v_cvt_pk_bf16_f32 v203, v54 /*v310*/, v56 /*v312*/
	s_set_vgpr_msb 0x504
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x32_bf16 v[104:111], v[248:255], v[146:153] /*v[402:409]*/, v[104:111]
	s_set_vgpr_msb 0x405
	v_cvt_pk_bf16_f32 v202, v40 /*v296*/, v48 /*v304*/
	v_cvt_pk_bf16_f32 v201, v30 /*v286*/, v38 /*v294*/
	v_cvt_pk_bf16_f32 v231, v65 /*v321*/, v27 /*v283*/
	v_cvt_pk_bf16_f32 v230, v63 /*v319*/, v35 /*v291*/
	v_cvt_pk_bf16_f32 v229, v61 /*v317*/, v43 /*v299*/
	v_cvt_pk_bf16_f32 v228, v59 /*v315*/, v51 /*v307*/
	v_cvt_pk_bf16_f32 v227, v55 /*v311*/, v57 /*v313*/
	s_set_vgpr_msb 0x500
	v_wmma_f32_16x16x32_bf16 v[40:47], v[248:255], v[216:223], v[40:47]
	s_set_vgpr_msb 0x42
	ds_load_tr16_b128 v[2:5] /*v[258:261]*/, v13 /*v525*/ offset:18496
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x4202
	ds_load_tr16_b128 v[248:251], v13 /*v525*/ offset:18528
	ds_load_tr16_b128 v[252:255], v13 /*v525*/ offset:23136
	s_set_vgpr_msb 0x205
	v_cvt_pk_bf16_f32 v226, v41 /*v297*/, v49 /*v305*/
	v_cvt_pk_bf16_f32 v225, v31 /*v287*/, v39 /*v295*/
	s_set_vgpr_msb 0x542
	ds_load_tr16_b128 v[74:77] /*v[330:333]*/, v13 /*v525*/
	ds_load_tr16_b128 v[82:85] /*v[338:341]*/, v13 /*v525*/ offset:32
	ds_load_tr16_b128 v[78:81] /*v[334:337]*/, v13 /*v525*/ offset:4608
	ds_load_tr16_b128 v[86:89] /*v[342:345]*/, v13 /*v525*/ offset:4640
	ds_load_tr16_b128 v[90:93] /*v[346:349]*/, v13 /*v525*/ offset:9216
	ds_load_tr16_b128 v[98:101] /*v[354:357]*/, v13 /*v525*/ offset:9248
	ds_load_tr16_b128 v[94:97] /*v[350:353]*/, v13 /*v525*/ offset:13824
	ds_load_tr16_b128 v[102:105] /*v[358:361]*/, v13 /*v525*/ offset:13856
	ds_load_tr16_b128 v[106:109] /*v[362:365]*/, v13 /*v525*/ offset:18432
	ds_load_tr16_b128 v[114:117] /*v[370:373]*/, v13 /*v525*/ offset:18464
	ds_load_tr16_b128 v[110:113] /*v[366:369]*/, v13 /*v525*/ offset:23040
	ds_load_tr16_b128 v[118:121] /*v[374:377]*/, v13 /*v525*/ offset:23072
	ds_load_tr16_b128 v[122:125] /*v[378:381]*/, v13 /*v525*/ offset:27648
	ds_load_tr16_b128 v[130:133] /*v[386:389]*/, v13 /*v525*/ offset:27680
	ds_load_tr16_b128 v[126:129] /*v[382:385]*/, v13 /*v525*/ offset:32256
	ds_load_tr16_b128 v[134:137] /*v[390:393]*/, v13 /*v525*/ offset:32288
	s_add_co_i32 s2, s16, 4
	s_cmp_ge_i32 s2, s54
	s_set_vgpr_msb 0x4204
	s_wait_dscnt 0x15
	v_wmma_f32_16x16x32_bf16 v[96:103], v[232:239], v[146:153] /*v[402:409]*/, v[96:103]
	s_set_vgpr_msb 0x400
	v_wmma_f32_16x16x32_bf16 v[32:39], v[232:239], v[216:223], v[32:39]
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 2
	ds_load_tr16_b128 v[236:239], v13 /*v525*/ offset:4736
	ds_load_tr16_b128 v[232:235], v13 /*v525*/ offset:128
	ds_load_tr16_b128 v[240:243], v13 /*v525*/ offset:160
	ds_load_tr16_b128 v[244:247], v13 /*v525*/ offset:4768
	s_set_vgpr_msb 0x200
	s_wait_dscnt 0x14
	v_wmma_f32_16x16x32_bf16 v[96:103], v[248:255], v[192:199], v[96:103]
	v_wmma_f32_16x16x32_bf16 v[32:39], v[248:255], v[208:215], v[32:39]
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 2
	ds_load_tr16_b128 v[252:255], v13 /*v525*/ offset:13952
	s_set_vgpr_msb 0x201
	v_wmma_f32_16x16x32_bf16 v[104:111], v[2:9] /*v[258:265]*/, v[192:199], v[104:111]
	v_wmma_f32_16x16x32_bf16 v[40:47], v[2:9] /*v[258:265]*/, v[208:215], v[40:47]
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x142
	ds_load_tr16_b128 v[6:9] /*v[262:265]*/, v13 /*v525*/ offset:27712
	ds_load_tr16_b128 v[14:17] /*v[270:273]*/, v13 /*v525*/ offset:27744
	ds_load_tr16_b128 v[18:21] /*v[274:277]*/, v13 /*v525*/ offset:32352
	s_set_vgpr_msb 0x4204
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x32_bf16 v[88:95], v[232:239], v[138:145] /*v[394:401]*/, v[88:95]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[232:239], v[154:161] /*v[410:417]*/, v[24:31]
	s_set_vgpr_msb 0x402
	ds_load_tr16_b128 v[248:251], v13 /*v525*/ offset:9344
	s_wait_alu depctr_va_vdst(0)
	ds_load_tr16_b128 v[232:235], v13 /*v525*/ offset:9376
	ds_load_tr16_b128 v[236:239], v13 /*v525*/ offset:13984
	s_set_vgpr_msb 0x204
	s_wait_dscnt 0x7
	v_wmma_f32_16x16x32_bf16 v[80:87], v[240:247], v[138:145] /*v[394:401]*/, v[80:87]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[240:247], v[154:161] /*v[410:417]*/, v[16:23]
	s_set_vgpr_msb 0x401
	s_wait_dscnt 0x5
	v_wmma_f32_16x16x32_bf16 v[104:111], v[6:13] /*v[262:269]*/, v[200:207], v[104:111]
	v_wmma_f32_16x16x32_bf16 v[40:47], v[6:13] /*v[262:269]*/, v[224:231], v[40:47]
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x142
	ds_load_tr16_b128 v[6:9] /*v[262:265]*/, v13 /*v525*/ offset:23168
	ds_load_tr16_b128 v[10:13] /*v[266:269]*/, v13 /*v525*/ offset:32384
	s_set_vgpr_msb 0x4204
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x32_bf16 v[88:95], v[248:255], v[146:153] /*v[402:409]*/, v[88:95]
	s_set_vgpr_msb 0x400
	v_wmma_f32_16x16x32_bf16 v[24:31], v[248:255], v[216:223], v[24:31]
	s_set_vgpr_msb 0x42
	ds_load_tr16_b128 v[2:5] /*v[258:261]*/, v13 /*v525*/ offset:18560
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x4202
	ds_load_tr16_b128 v[248:251], v13 /*v525*/ offset:18592
	ds_load_tr16_b128 v[252:255], v13 /*v525*/ offset:23200
	s_set_vgpr_msb 0x204
	s_wait_dscnt 0x5
	v_wmma_f32_16x16x32_bf16 v[80:87], v[232:239], v[146:153] /*v[402:409]*/, v[80:87]
	s_set_vgpr_msb 0x400
	v_wmma_f32_16x16x32_bf16 v[16:23], v[232:239], v[216:223], v[16:23]
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 2
	ds_load_tr16_b128 v[236:239], v13 /*v525*/ offset:4800
	ds_load_tr16_b128 v[232:235], v13 /*v525*/ offset:192
	ds_load_tr16_b128 v[240:243], v13 /*v525*/ offset:224
	ds_load_tr16_b128 v[244:247], v13 /*v525*/ offset:4832
	s_set_vgpr_msb 0x201
	v_wmma_f32_16x16x32_bf16 v[96:103], v[14:21] /*v[270:277]*/, v[200:207], v[96:103]
	v_wmma_f32_16x16x32_bf16 v[32:39], v[14:21] /*v[270:277]*/, v[224:231], v[32:39]
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x32_bf16 v[88:95], v[2:9] /*v[258:265]*/, v[192:199], v[88:95]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[2:9] /*v[258:265]*/, v[208:215], v[24:31]
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x142
	ds_load_tr16_b128 v[6:9] /*v[262:265]*/, v13 /*v525*/ offset:27776
	ds_load_tr16_b128 v[14:17] /*v[270:273]*/, v13 /*v525*/ offset:27808
	ds_load_tr16_b128 v[18:21] /*v[274:277]*/, v13 /*v525*/ offset:32416
	s_set_vgpr_msb 0x4200
	s_wait_dscnt 0x7
	v_wmma_f32_16x16x32_bf16 v[80:87], v[248:255], v[192:199], v[80:87]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[248:255], v[208:215], v[16:23]
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 2
	ds_load_tr16_b128 v[252:255], v13 /*v525*/ offset:14016
	s_set_vgpr_msb 0x204
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x32_bf16 v[72:79], v[232:239], v[138:145] /*v[394:401]*/, v[72:79]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[232:239], v[154:161] /*v[410:417]*/, v[8:15]
	s_set_vgpr_msb 0x402
	ds_load_tr16_b128 v[248:251], v13 /*v525*/ offset:9408
	s_wait_alu depctr_va_vdst(0)
	ds_load_tr16_b128 v[232:235], v13 /*v525*/ offset:9440
	ds_load_tr16_b128 v[236:239], v13 /*v525*/ offset:14048
	s_set_vgpr_msb 0x201
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x32_bf16 v[88:95], v[6:13] /*v[262:269]*/, v[200:207], v[88:95]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[6:13] /*v[262:269]*/, v[224:231], v[24:31]
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x142
	ds_load_tr16_b128 v[6:9] /*v[262:265]*/, v13 /*v525*/ offset:23232
	ds_load_tr16_b128 v[10:13] /*v[266:269]*/, v13 /*v525*/ offset:32448
	s_set_vgpr_msb 0x4204
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x32_bf16 v[72:79], v[248:255], v[146:153] /*v[402:409]*/, v[72:79]
	s_set_vgpr_msb 0x400
	v_wmma_f32_16x16x32_bf16 v[8:15], v[248:255], v[216:223], v[8:15]
	s_set_vgpr_msb 0x42
	ds_load_tr16_b128 v[2:5] /*v[258:261]*/, v13 /*v525*/ offset:18624
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x4202
	ds_load_tr16_b128 v[248:251], v13 /*v525*/ offset:18656
	ds_load_tr16_b128 v[252:255], v13 /*v525*/ offset:23264
	s_set_vgpr_msb 0x205
	v_wmma_f32_16x16x32_bf16 v[120:127], v[74:81] /*v[330:337]*/, v[138:145] /*v[394:401]*/, v[120:127]
	v_wmma_f32_16x16x32_bf16 v[56:63], v[74:81] /*v[330:337]*/, v[154:161] /*v[410:417]*/, v[56:63]
	v_wmma_f32_16x16x32_bf16 v[112:119], v[82:89] /*v[338:345]*/, v[138:145] /*v[394:401]*/, v[112:119]
	v_wmma_f32_16x16x32_bf16 v[48:55], v[82:89] /*v[338:345]*/, v[154:161] /*v[410:417]*/, v[48:55]
	s_set_vgpr_msb 0x504
	v_wmma_f32_16x16x32_bf16 v[64:71], v[240:247], v[138:145] /*v[394:401]*/, v[64:71]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[240:247], v[154:161] /*v[410:417]*/, v[0:7]
	s_set_vgpr_msb 0x401
	v_wmma_f32_16x16x32_bf16 v[80:87], v[14:21] /*v[270:277]*/, v[200:207], v[80:87]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[14:21] /*v[270:277]*/, v[224:231], v[16:23]
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[72:79], v[2:9] /*v[258:265]*/, v[192:199], v[72:79]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[2:9] /*v[258:265]*/, v[208:215], v[8:15]
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x142
	ds_load_tr16_b128 v[6:9] /*v[262:265]*/, v13 /*v525*/ offset:27840
	ds_load_tr16_b128 v[14:17] /*v[270:273]*/, v13 /*v525*/ offset:27872
	ds_load_tr16_b128 v[18:21] /*v[274:277]*/, v13 /*v525*/ offset:32480
	s_wait_tensorcnt 0x0
	s_set_vgpr_msb 0x4205
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	v_wmma_f32_16x16x32_bf16 v[120:127], v[90:97] /*v[346:353]*/, v[146:153] /*v[402:409]*/, v[120:127]
	s_set_vgpr_msb 0x501
	v_wmma_f32_16x16x32_bf16 v[56:63], v[90:97] /*v[346:353]*/, v[216:223], v[56:63]
	s_set_vgpr_msb 0x105
	v_wmma_f32_16x16x32_bf16 v[112:119], v[98:105] /*v[354:361]*/, v[146:153] /*v[402:409]*/, v[112:119]
	s_set_vgpr_msb 0x501
	v_wmma_f32_16x16x32_bf16 v[48:55], v[98:105] /*v[354:361]*/, v[216:223], v[48:55]
	s_set_vgpr_msb 0x104
	v_wmma_f32_16x16x32_bf16 v[64:71], v[232:239], v[146:153] /*v[402:409]*/, v[64:71]
	s_set_vgpr_msb 0x400
	v_wmma_f32_16x16x32_bf16 v[0:7], v[232:239], v[216:223], v[0:7]
	s_set_vgpr_msb 1
	v_wmma_f32_16x16x32_bf16 v[120:127], v[106:113] /*v[362:369]*/, v[192:199], v[120:127]
	v_wmma_f32_16x16x32_bf16 v[56:63], v[106:113] /*v[362:369]*/, v[208:215], v[56:63]
	v_wmma_f32_16x16x32_bf16 v[112:119], v[114:121] /*v[370:377]*/, v[192:199], v[112:119]
	v_wmma_f32_16x16x32_bf16 v[48:55], v[114:121] /*v[370:377]*/, v[208:215], v[48:55]
	s_set_vgpr_msb 0x100
	v_wmma_f32_16x16x32_bf16 v[64:71], v[248:255], v[192:199], v[64:71]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[248:255], v[208:215], v[0:7]
	s_set_vgpr_msb 1
	v_wmma_f32_16x16x32_bf16 v[120:127], v[122:129] /*v[378:385]*/, v[200:207], v[120:127]
	v_wmma_f32_16x16x32_bf16 v[56:63], v[122:129] /*v[378:385]*/, v[224:231], v[56:63]
	v_wmma_f32_16x16x32_bf16 v[112:119], v[130:137] /*v[386:393]*/, v[200:207], v[112:119]
	v_wmma_f32_16x16x32_bf16 v[48:55], v[130:137] /*v[386:393]*/, v[224:231], v[48:55]
	v_wmma_f32_16x16x32_bf16 v[72:79], v[6:13] /*v[262:269]*/, v[200:207], v[72:79]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[6:13] /*v[262:269]*/, v[224:231], v[8:15]
	v_wmma_f32_16x16x32_bf16 v[64:71], v[14:21] /*v[270:277]*/, v[200:207], v[64:71]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[14:21] /*v[270:277]*/, v[224:231], v[0:7]
	s_set_vgpr_msb 0x100
	s_cbranch_scc1 .LBB0_8
	s_sub_co_i32 s4, s16, s92
	s_lshl_b32 s2, s2, 7
	s_ashr_i32 s3, s4, 31
	s_sub_co_i32 s5, s89, s2
	s_lshr_b32 s3, s3, 30
	v_med3_i32 v192, s5, 0, 0x80
	s_add_co_i32 s2, s2, s53
	s_add_co_i32 s3, s4, s3
	s_mov_b32 s73, s65
	s_and_b32 s5, s3, 0x1ffffc
	s_ashr_i32 s3, s2, 31
	v_readfirstlane_b32 s9, v192
	s_sub_co_i32 s6, s4, s5
	s_mul_u64 s[4:5], s[2:3], s[82:83]
	s_mul_u64 s[2:3], s[2:3], s[90:91]
	s_lshl_b64 s[4:5], s[4:5], 1
	s_sub_co_i32 s9, s9, vcc_hi
	s_add_nc_u64 s[4:5], s[96:97], s[4:5]
	s_mul_i32 s6, s6, 0x11800
	s_add_nc_u64 s[10:11], s[94:95], s[4:5]
	s_max_i32 s4, s9, 0
	s_lshl_b64 s[2:3], s[2:3], 1
	s_lshl_b32 s4, s4, 16
	s_add_nc_u64 s[2:3], s[98:99], s[2:3]
	s_add_co_i32 s9, s102, s6
	s_bitset1_b32 s11, 31
	s_or_b32 s66, s4, 0x7fff
	s_mov_b32 s75, s67
	tensor_load_to_lds s[8:11], s[64:71]
	s_add_nc_u64 s[10:11], s[50:51], s[2:3]
	s_add_co_i32 s9, s103, s6
	s_bitset1_b32 s11, 31
	s_mov_b32 s74, s66
	s_mov_b32 s76, s68
	s_mov_b32 s79, s71
	tensor_load_to_lds s[8:11], s[72:79]
	s_branch .LBB0_8
.LBB0_15:
	v_mov_b32_e32 v0, 0
	s_set_vgpr_msb 0x82
	v_dual_mov_b32 v131 /*v643*/, 1.0 :: v_dual_mov_b32 v215 /*v727*/, v203 /*v715*/
	s_wait_loadcnt 0x0
	v_mov_b32_e32 v209 /*v721*/, v211 /*v723*/
	s_set_vgpr_msb 0x8200
	v_dual_mov_b32 v1, v0 :: v_dual_mov_b32 v2, v0
	v_dual_mov_b32 v3, v0 :: v_dual_mov_b32 v4, v0
	v_dual_mov_b32 v5, v0 :: v_dual_mov_b32 v6, v0
	v_mov_b32_e32 v7, v0
	s_set_vgpr_msb 0x82
	v_mov_b32_e32 v130 /*v642*/, v131 /*v643*/
	s_set_vgpr_msb 0x8200
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
	s_branch .LBB0_17
.LBB0_16:
	s_set_vgpr_msb 0x81
	v_mov_b32_e32 v211 /*v723*/, v72 /*v328*/
	s_mov_b32 s93, s18
	s_set_vgpr_msb 0x8100
.LBB0_17:
	s_sub_co_i32 s2, s7, s12
	s_and_b32 s3, s2, -2
	s_add_co_i32 s100, s3, s12
	s_ashr_i32 s101, s100, 31
	s_cmp_lt_i32 s2, 2
	s_cbranch_scc1 .LBB0_32
	s_add_co_i32 s2, s63, s24
	s_lshl_b32 s3, s12, 7
	s_sub_co_i32 s2, s2, s62
	s_sub_co_i32 s5, s89, s3
	s_sub_co_i32 s2, s2, s60
	s_add_co_i32 s3, s53, s3
	s_max_i32 s2, s2, 0
	s_mov_b32 s71, 0
	s_lshr_b32 s2, s2, 7
	s_mov_b32 s4, 1
	s_min_u32 s2, s2, s25
	s_add_co_i32 s15, s5, 0xfffffd80
	s_add_co_i32 s8, s3, 0x280
	s_sub_co_i32 s10, 0, s2
	s_mov_b32 s11, s71
	s_mov_b32 s16, 0x76543210
	s_mov_b32 s14, 0x3fb8aa3b
	s_mov_b32 s68, 32
	s_mov_b32 s67, 0x800000
	s_mov_b32 s65, 0xffff0000
	s_mov_b32 s64, 0x7510000
	s_mov_b32 s72, 0xf510000
	s_branch .LBB0_20
.LBB0_19:
	s_set_vgpr_msb 0x8a
	v_dual_fmac_f32 v219 /*v731*/, v134 /*v646*/, v130 /*v642*/ :: v_dual_fmac_f32 v220 /*v732*/, v136 /*v648*/, v131 /*v643*/
	s_add_nc_u64 s[12:13], s[12:13], 2
	v_dual_mov_b32 v13 /*v525*/, v218 /*v730*/ :: v_dual_mov_b32 v11 /*v523*/, v217 /*v729*/
	v_nop
	v_nop
	s_set_vgpr_msb 0x8a0a
	v_dual_add_f32 v192, v219 /*v731*/, v132 /*v644*/ :: v_dual_add_f32 v193, v220 /*v732*/, v133 /*v645*/
	v_cmp_lt_i64_e64 s2, s[12:13], s[100:101]
	s_set_vgpr_msb 0xa82
	v_dual_mov_b32 v12 /*v524*/, v216 /*v728*/ :: v_dual_mov_b32 v10 /*v522*/, v214 /*v726*/
	s_set_vgpr_msb 0x8200
	v_dual_fmac_f32 v211, v210, v192 :: v_dual_fmac_f32 v213, v212, v193
	s_addk_co_i32 s15, 0xff00
	s_and_b32 vcc_lo, exec_lo, s2
	s_addk_co_i32 s8, 0x100
	s_set_vgpr_msb 0x80
	v_dual_add_f32 v130 /*v642*/, v211, v208 :: v_dual_add_f32 v131 /*v643*/, v213, v209
	s_set_vgpr_msb 0x8000
	s_cbranch_vccz .LBB0_33
.LBB0_20:
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 2
	ds_load_b128 v[192:195], v215 /*v727*/
	ds_load_b128 v[196:199], v215 /*v727*/ offset:32
	ds_load_b128 v[200:203], v215 /*v727*/ offset:64
	ds_load_b128 v[204:207], v215 /*v727*/ offset:96
	ds_load_b128 v[208:211], v215 /*v727*/ offset:128
	ds_load_b128 v[212:215], v215 /*v727*/ offset:160
	ds_load_b128 v[240:243], v215 /*v727*/ offset:4544
	ds_load_b128 v[244:247], v215 /*v727*/ offset:4576
	s_set_vgpr_msb 0x242
	ds_load_b128 v[10:13] /*v[266:269]*/, v215 /*v727*/ offset:8704
	ds_load_b128 v[14:17] /*v[270:273]*/, v215 /*v727*/ offset:8736
	ds_load_b128 v[82:85] /*v[338:341]*/, v215 /*v727*/ offset:8768
	ds_load_b128 v[86:89] /*v[342:345]*/, v215 /*v727*/ offset:8800
	ds_load_b128 v[106:109] /*v[362:365]*/, v215 /*v727*/ offset:8832
	ds_load_b128 v[110:113] /*v[366:369]*/, v215 /*v727*/ offset:8864
	ds_load_b128 v[130:133] /*v[386:389]*/, v215 /*v727*/ offset:8896
	ds_load_b128 v[134:137] /*v[390:393]*/, v215 /*v727*/ offset:8928
	ds_load_b128 v[146:149] /*v[402:405]*/, v215 /*v727*/ offset:13056
	ds_load_b128 v[150:153] /*v[406:409]*/, v215 /*v727*/ offset:13088
	ds_load_b128 v[170:173] /*v[426:429]*/, v215 /*v727*/ offset:13120
	ds_load_b128 v[174:177] /*v[430:433]*/, v215 /*v727*/ offset:13152
	ds_load_b128 v[226:229] /*v[482:485]*/, v215 /*v727*/ offset:21824
	ds_load_b128 v[230:233] /*v[486:489]*/, v215 /*v727*/ offset:21856
	ds_load_b128 v[234:237] /*v[490:493]*/, v215 /*v727*/ offset:21888
	ds_load_b128 v[238:241] /*v[494:497]*/, v215 /*v727*/ offset:21920
	ds_load_b128 v[242:245] /*v[498:501]*/, v215 /*v727*/ offset:21952
	ds_load_b128 v[246:249] /*v[502:505]*/, v215 /*v727*/ offset:21984
	ds_load_b128 v[250:253] /*v[506:509]*/, v215 /*v727*/ offset:26112
	ds_load_b128 v[254:257] /*v[510:513]*/, v215 /*v727*/ offset:26144
	s_set_vgpr_msb 0x4282
	ds_load_b128 v[2:5] /*v[514:517]*/, v215 /*v727*/ offset:26176
	ds_load_b128 v[6:9] /*v[518:521]*/, v215 /*v727*/ offset:26208
	ds_load_b128 v[14:17] /*v[526:529]*/, v215 /*v727*/ offset:26240
	ds_load_b128 v[18:21] /*v[530:533]*/, v215 /*v727*/ offset:26272
	ds_load_b128 v[22:25] /*v[534:537]*/, v215 /*v727*/ offset:26304
	ds_load_b128 v[26:29] /*v[538:541]*/, v215 /*v727*/ offset:26336
	ds_load_b128 v[46:49] /*v[558:561]*/, v215 /*v727*/ offset:30656
	ds_load_b128 v[50:53] /*v[562:565]*/, v215 /*v727*/ offset:30688
	ds_load_b128 v[54:57] /*v[566:569]*/, v210 /*v722*/
	ds_load_b128 v[58:61] /*v[570:573]*/, v210 /*v722*/ offset:32
	ds_load_b128 v[62:65] /*v[574:577]*/, v210 /*v722*/ offset:64
	ds_load_b128 v[66:69] /*v[578:581]*/, v210 /*v722*/ offset:96
	ds_load_b128 v[70:73] /*v[582:585]*/, v210 /*v722*/ offset:128
	ds_load_b128 v[74:77] /*v[586:589]*/, v210 /*v722*/ offset:160
	ds_load_b128 v[146:149] /*v[658:661]*/, v210 /*v722*/ offset:8832
	ds_load_b128 v[150:153] /*v[662:665]*/, v210 /*v722*/ offset:8864
	ds_load_b128 v[154:157] /*v[666:669]*/, v210 /*v722*/ offset:8896
	ds_load_b128 v[158:161] /*v[670:673]*/, v210 /*v722*/ offset:8928
	ds_load_b128 v[162:165] /*v[674:677]*/, v210 /*v722*/ offset:13056
	ds_load_b128 v[166:169] /*v[678:681]*/, v210 /*v722*/ offset:13088
	ds_load_b128 v[78:81] /*v[590:593]*/, v210 /*v722*/ offset:192
	ds_load_b128 v[82:85] /*v[594:597]*/, v210 /*v722*/ offset:224
	ds_load_b128 v[86:89] /*v[598:601]*/, v210 /*v722*/ offset:4352
	ds_load_b128 v[90:93] /*v[602:605]*/, v210 /*v722*/ offset:4384
	ds_load_b128 v[94:97] /*v[606:609]*/, v210 /*v722*/ offset:4416
	ds_load_b128 v[98:101] /*v[610:613]*/, v210 /*v722*/ offset:4448
	v_dual_mov_b32 v214 /*v726*/, v215 /*v727*/ :: v_dual_mov_b32 v216 /*v728*/, v210 /*v722*/
	s_set_vgpr_msb 0x8242
	ds_load_b128 v[178:181] /*v[434:437]*/, v210 /*v722*/ offset:13120
	ds_load_b128 v[182:185] /*v[438:441]*/, v210 /*v722*/ offset:13152
	ds_load_b128 v[162:165] /*v[418:421]*/, v210 /*v722*/ offset:13184
	ds_load_b128 v[166:169] /*v[422:425]*/, v210 /*v722*/ offset:13216
	ds_load_b128 v[138:141] /*v[394:397]*/, v210 /*v722*/ offset:13248
	ds_load_b128 v[142:145] /*v[398:401]*/, v210 /*v722*/ offset:13280
	ds_load_b128 v[114:117] /*v[370:373]*/, v210 /*v722*/ offset:17408
	ds_load_b128 v[118:121] /*v[374:377]*/, v210 /*v722*/ offset:17440
	ds_load_b128 v[98:101] /*v[354:357]*/, v210 /*v722*/ offset:17472
	ds_load_b128 v[102:105] /*v[358:361]*/, v210 /*v722*/ offset:17504
	ds_load_b128 v[90:93] /*v[346:349]*/, v210 /*v722*/ offset:17536
	ds_load_b128 v[94:97] /*v[350:353]*/, v210 /*v722*/ offset:17568
	ds_load_b128 v[74:77] /*v[330:333]*/, v210 /*v722*/ offset:17600
	ds_load_b128 v[78:81] /*v[334:337]*/, v210 /*v722*/ offset:17632
	ds_load_b128 v[58:61] /*v[314:317]*/, v210 /*v722*/ offset:21760
	ds_load_b128 v[62:65] /*v[318:321]*/, v210 /*v722*/ offset:21792
	ds_load_b128 v[50:53] /*v[306:309]*/, v210 /*v722*/ offset:21824
	ds_load_b128 v[54:57] /*v[310:313]*/, v210 /*v722*/ offset:21856
	ds_load_b128 v[42:45] /*v[298:301]*/, v210 /*v722*/ offset:21888
	ds_load_b128 v[46:49] /*v[302:305]*/, v210 /*v722*/ offset:21920
	ds_load_b128 v[34:37] /*v[290:293]*/, v210 /*v722*/ offset:21952
	ds_load_b128 v[38:41] /*v[294:297]*/, v210 /*v722*/ offset:21984
	s_set_vgpr_msb 0x4241
	s_wait_dscnt 0x30
	v_wmma_f32_16x16x32_bf16 v[122:129] /*v[378:385]*/, v[250:257] /*v[506:513]*/, v[128:135], 0
	s_set_vgpr_msb 0x4182
	ds_load_b128 v[102:105] /*v[614:617]*/, v210 /*v722*/ offset:4480
	ds_load_b128 v[106:109] /*v[618:621]*/, v210 /*v722*/ offset:4512
	ds_load_b128 v[110:113] /*v[622:625]*/, v210 /*v722*/ offset:4544
	ds_load_b128 v[114:117] /*v[626:629]*/, v210 /*v722*/ offset:4576
	ds_load_b128 v[118:121] /*v[630:633]*/, v210 /*v722*/ offset:8704
	ds_load_b128 v[122:125] /*v[634:637]*/, v210 /*v722*/ offset:8736
	ds_load_b128 v[138:141] /*v[650:653]*/, v210 /*v722*/ offset:8768
	ds_load_b128 v[142:145] /*v[654:657]*/, v210 /*v722*/ offset:8800
	s_wait_alu depctr_vm_vsrc(0)
	v_dual_mov_b32 v218 /*v730*/, v208 /*v720*/ :: v_dual_mov_b32 v217 /*v729*/, v205 /*v717*/
	v_dual_mov_b32 v205 /*v717*/, v11 /*v523*/ :: v_dual_mov_b32 v208 /*v720*/, v13 /*v525*/
	s_set_vgpr_msb 0x8240
	v_wmma_f32_16x16x32_bf16 v[26:33] /*v[282:289]*/, v[192:199], v[128:135], 0
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[258:265]*/, v[192:199], v[160:167], 0
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x4002
	ds_load_b128 v[192:195], v215 /*v727*/ offset:192
	ds_load_b128 v[196:199], v215 /*v727*/ offset:224
	ds_load_b128 v[216:219], v215 /*v727*/ offset:4352
	ds_load_b128 v[220:223], v215 /*v727*/ offset:4384
	ds_load_b128 v[224:227], v215 /*v727*/ offset:4416
	ds_load_b128 v[228:231], v215 /*v727*/ offset:4448
	ds_load_b128 v[232:235], v215 /*v727*/ offset:4480
	ds_load_b128 v[236:239], v215 /*v727*/ offset:4512
	s_set_vgpr_msb 0x250
	v_wmma_f32_16x16x32_bf16 v[26:33] /*v[282:289]*/, v[200:207], v[136:143], v[26:33] /*v[282:289]*/
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[258:265]*/, v[200:207], v[168:175], v[2:9] /*v[258:265]*/
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x5002
	ds_load_b128 v[200:203], v215 /*v727*/ offset:13184
	ds_load_b128 v[204:207], v215 /*v727*/ offset:13216
	s_set_vgpr_msb 0x242
	ds_load_b128 v[186:189] /*v[442:445]*/, v215 /*v727*/ offset:13248
	ds_load_b128 v[190:193] /*v[446:449]*/, v215 /*v727*/ offset:13280
	ds_load_b128 v[194:197] /*v[450:453]*/, v215 /*v727*/ offset:17408
	ds_load_b128 v[198:201] /*v[454:457]*/, v215 /*v727*/ offset:17440
	s_set_vgpr_msb 0x4281
	v_wmma_f32_16x16x32_bf16 v[170:177] /*v[682:689]*/, v[10:17] /*v[266:273]*/, v[128:135], 0
	s_set_vgpr_msb 0x8141
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[322:329]*/, v[10:17] /*v[266:273]*/, v[160:167], 0
	s_set_vgpr_msb 0x4150
	v_wmma_f32_16x16x32_bf16 v[26:33] /*v[282:289]*/, v[208:215], v[144:151], v[26:33] /*v[282:289]*/
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[258:265]*/, v[208:215], v[176:183], v[2:9] /*v[258:265]*/
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x5002
	ds_load_b128 v[208:211], v215 /*v727*/ offset:17472
	ds_load_b128 v[212:215], v215 /*v727*/ offset:17504
	s_set_vgpr_msb 0x242
	ds_load_b128 v[202:205] /*v[458:461]*/, v215 /*v727*/ offset:17536
	ds_load_b128 v[206:209] /*v[462:465]*/, v215 /*v727*/ offset:17568
	ds_load_b128 v[210:213] /*v[466:469]*/, v215 /*v727*/ offset:17600
	ds_load_b128 v[214:217] /*v[470:473]*/, v215 /*v727*/ offset:17632
	ds_load_b128 v[218:221] /*v[474:477]*/, v215 /*v727*/ offset:21760
	ds_load_b128 v[222:225] /*v[478:481]*/, v215 /*v727*/ offset:21792
	s_set_vgpr_msb 0x42a1
	v_wmma_f32_16x16x32_bf16 v[170:177] /*v[682:689]*/, v[82:89] /*v[338:345]*/, v[136:143], v[170:177] /*v[682:689]*/
	s_set_vgpr_msb 0xa151
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[322:329]*/, v[82:89] /*v[338:345]*/, v[168:175], v[66:73] /*v[322:329]*/
	s_set_vgpr_msb 0x5101
	s_wait_dscnt 0x8
	v_wmma_f32_16x16x32_bf16 v[248:255], v[194:201] /*v[450:457]*/, v[128:135], 0
	s_set_vgpr_msb 0x1a1
	v_wmma_f32_16x16x32_bf16 v[178:185] /*v[690:697]*/, v[146:153] /*v[402:409]*/, v[128:135], 0
	v_wmma_f32_16x16x32_bf16 v[170:177] /*v[682:689]*/, v[106:113] /*v[362:369]*/, v[144:151], v[170:177] /*v[682:689]*/
	s_set_vgpr_msb 0xa151
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[322:329]*/, v[106:113] /*v[362:369]*/, v[176:183], v[66:73] /*v[322:329]*/
	v_wmma_f32_16x16x32_bf16 v[82:89] /*v[338:345]*/, v[146:153] /*v[402:409]*/, v[160:167], 0
	v_wmma_f32_16x16x32_bf16 v[106:113] /*v[362:369]*/, v[194:201] /*v[450:457]*/, v[160:167], 0
	s_set_vgpr_msb 0x51a1
	v_wmma_f32_16x16x32_bf16 v[178:185] /*v[690:697]*/, v[170:177] /*v[426:433]*/, v[136:143], v[178:185] /*v[690:697]*/
	s_set_vgpr_msb 0xa100
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x32_bf16 v[248:255], v[208:215], v[136:143], v[248:255]
	s_set_vgpr_msb 0x50
	v_wmma_f32_16x16x32_bf16 v[26:33] /*v[282:289]*/, v[192:199], v[152:159], v[26:33] /*v[282:289]*/
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[258:265]*/, v[192:199], v[184:191], v[2:9] /*v[258:265]*/
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x5002
	ds_load_b128 v[192:195], v215 /*v727*/ offset:30464
	ds_load_b128 v[196:199], v215 /*v727*/ offset:30496
	s_set_vgpr_msb 0x282
	ds_load_b128 v[30:33] /*v[542:545]*/, v215 /*v727*/ offset:30528
	ds_load_b128 v[34:37] /*v[546:549]*/, v215 /*v727*/ offset:30560
	ds_load_b128 v[38:41] /*v[550:553]*/, v215 /*v727*/ offset:30592
	ds_load_b128 v[42:45] /*v[554:557]*/, v215 /*v727*/ offset:30624
	s_wait_alu depctr_vm_vsrc(0)
	v_mov_b32_e32 v215 /*v727*/, v10 /*v522*/
	s_set_vgpr_msb 0x8281
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x32_bf16 v[186:193] /*v[698:705]*/, v[218:225] /*v[474:481]*/, v[128:135], 0
	s_set_vgpr_msb 0x8151
	v_wmma_f32_16x16x32_bf16 v[82:89] /*v[338:345]*/, v[170:177] /*v[426:433]*/, v[168:175], v[82:89] /*v[338:345]*/
	s_set_vgpr_msb 0x5150
	v_wmma_f32_16x16x32_bf16 v[106:113] /*v[362:369]*/, v[208:215], v[168:175], v[106:113] /*v[362:369]*/
	s_set_vgpr_msb 0x50a1
	v_wmma_f32_16x16x32_bf16 v[170:177] /*v[682:689]*/, v[130:137] /*v[386:393]*/, v[152:159], v[170:177] /*v[682:689]*/
	s_set_vgpr_msb 0xa151
	v_wmma_f32_16x16x32_bf16 v[66:73] /*v[322:329]*/, v[130:137] /*v[386:393]*/, v[184:191], v[66:73] /*v[322:329]*/
	v_wmma_f32_16x16x32_bf16 v[130:137] /*v[386:393]*/, v[218:225] /*v[474:481]*/, v[160:167], 0
	s_set_vgpr_msb 0x51a0
	v_wmma_f32_16x16x32_bf16 v[178:185] /*v[690:697]*/, v[200:207], v[144:151], v[178:185] /*v[690:697]*/
	s_set_vgpr_msb 0xa001
	v_wmma_f32_16x16x32_bf16 v[248:255], v[202:209] /*v[458:465]*/, v[144:151], v[248:255]
	s_set_vgpr_msb 0x141
	v_wmma_f32_16x16x32_bf16 v[146:153] /*v[402:409]*/, v[250:257] /*v[506:513]*/, v[160:167], 0
	s_set_vgpr_msb 0x41a1
	v_wmma_f32_16x16x32_bf16 v[186:193] /*v[698:705]*/, v[226:233] /*v[482:489]*/, v[136:143], v[186:193] /*v[698:705]*/
	s_set_vgpr_msb 0xa140
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[274:281]*/, v[216:223], v[160:167], 0
	v_wmma_f32_16x16x32_bf16 v[154:161] /*v[410:417]*/, v[216:223], v[128:135], 0
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x4015
	v_max3_num_f32 v216, v26 /*v282*/, v27 /*v283*/, v28 /*v284*/
	v_max3_num_f32 v217, v2 /*v258*/, v3 /*v259*/, v4 /*v260*/
	v_max3_num_f32 v218, v29 /*v285*/, v30 /*v286*/, v31 /*v287*/
	s_set_vgpr_msb 0x1550
	v_wmma_f32_16x16x32_bf16 v[82:89] /*v[338:345]*/, v[200:207], v[176:183], v[82:89] /*v[338:345]*/
	s_set_vgpr_msb 0x5015
	v_max3_num_f32 v219, v5 /*v261*/, v6 /*v262*/, v7 /*v263*/
	s_set_vgpr_msb 0x1551
	v_wmma_f32_16x16x32_bf16 v[106:113] /*v[362:369]*/, v[202:209] /*v[458:465]*/, v[176:183], v[106:113] /*v[362:369]*/
	v_wmma_f32_16x16x32_bf16 v[130:137] /*v[386:393]*/, v[226:233] /*v[482:489]*/, v[168:175], v[130:137] /*v[386:393]*/
	s_set_vgpr_msb 0x51a1
	v_wmma_f32_16x16x32_bf16 v[178:185] /*v[690:697]*/, v[186:193] /*v[442:449]*/, v[152:159], v[178:185] /*v[690:697]*/
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0xa12a
	v_max3_num_f32 v201, v181 /*v693*/, v182 /*v694*/, v183 /*v695*/
	s_set_vgpr_msb 0x2a01
	v_wmma_f32_16x16x32_bf16 v[248:255], v[210:217] /*v[466:473]*/, v[152:159], v[248:255]
	s_set_vgpr_msb 0x12a
	v_max3_num_f32 v200, v178 /*v690*/, v179 /*v691*/, v180 /*v692*/
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x2a0a
	v_max3_num_f32 v204, v184 /*v696*/, v185 /*v697*/, v248
	s_set_vgpr_msb 0xa52
	v_wmma_f32_16x16x32_bf16 v[122:129] /*v[378:385]*/, v[2:9] /*v[514:521]*/, v[136:143], v[122:129] /*v[378:385]*/
	s_set_vgpr_msb 0x5200
	v_max3_num_f32 v205, v249, v250, v251
	v_max3_num_f32 v206, v252, v253, v254
	v_max3_num_f32 v201, v201, v204, v205
	s_set_vgpr_msb 0x52
	v_wmma_f32_16x16x32_bf16 v[146:153] /*v[402:409]*/, v[2:9] /*v[514:521]*/, v[168:175], v[146:153] /*v[402:409]*/
	s_set_vgpr_msb 0x5280
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x32_bf16 v[0:7] /*v[512:519]*/, v[192:199], v[128:135], 0
	s_set_vgpr_msb 0x80a1
	v_wmma_f32_16x16x32_bf16 v[186:193] /*v[698:705]*/, v[234:241] /*v[490:497]*/, v[144:151], v[186:193] /*v[698:705]*/
	s_set_vgpr_msb 0xa150
	v_wmma_f32_16x16x32_bf16 v[154:161] /*v[410:417]*/, v[224:231], v[136:143], v[154:161] /*v[410:417]*/
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[274:281]*/, v[224:231], v[168:175], v[18:25] /*v[274:281]*/
	s_set_vgpr_msb 0x5051
	v_wmma_f32_16x16x32_bf16 v[82:89] /*v[338:345]*/, v[186:193] /*v[442:449]*/, v[184:191], v[82:89] /*v[338:345]*/
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x5115
	v_max3_num_f32 v203, v85 /*v341*/, v86 /*v342*/, v87 /*v343*/
	s_set_vgpr_msb 0x1551
	v_wmma_f32_16x16x32_bf16 v[106:113] /*v[362:369]*/, v[210:217] /*v[466:473]*/, v[184:191], v[106:113] /*v[362:369]*/
	s_set_vgpr_msb 0x5115
	v_max3_num_f32 v202, v82 /*v338*/, v83 /*v339*/, v84 /*v340*/
	v_nop
	v_nop
	v_nop
	v_max3_num_f32 v204, v88 /*v344*/, v89 /*v345*/, v106 /*v362*/
	s_set_vgpr_msb 0x1540
	v_wmma_f32_16x16x32_bf16 v[170:177] /*v[426:433]*/, v[192:199], v[160:167], 0
	s_set_vgpr_msb 0x4015
	v_max3_num_f32 v205, v107 /*v363*/, v108 /*v364*/, v109 /*v365*/
	v_max3_num_f32 v207, v110 /*v366*/, v111 /*v367*/, v112 /*v368*/
	s_set_vgpr_msb 0x1500
	v_max3_num_f32 v203, v203, v204, v205
	s_set_vgpr_msb 0x51
	v_wmma_f32_16x16x32_bf16 v[130:137] /*v[386:393]*/, v[234:241] /*v[490:497]*/, v[176:183], v[130:137] /*v[386:393]*/
	s_set_vgpr_msb 0x51a2
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[0:7] /*v[512:519]*/, v[30:37] /*v[542:549]*/, v[136:143], v[0:7] /*v[512:519]*/
	s_set_vgpr_msb 0xa2a1
	v_wmma_f32_16x16x32_bf16 v[186:193] /*v[698:705]*/, v[242:249] /*v[498:505]*/, v[152:159], v[186:193] /*v[698:705]*/
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0xa128
	v_max3_num_f32 v204, v255, v186 /*v698*/, v187 /*v699*/
	s_set_vgpr_msb 0x2850
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[274:281]*/, v[232:239], v[176:183], v[18:25] /*v[274:281]*/
	s_set_vgpr_msb 0x502a
	v_max3_num_f32 v205, v188 /*v700*/, v189 /*v701*/, v190 /*v702*/
	v_max3_num_f32 v208, v191 /*v703*/, v192 /*v704*/, v193 /*v705*/
	s_set_vgpr_msb 0x2a00
	v_max3_num_f32 v204, v206, v204, v205
	s_set_vgpr_msb 0x50
	v_wmma_f32_16x16x32_bf16 v[154:161] /*v[410:417]*/, v[232:239], v[144:151], v[154:161] /*v[410:417]*/
	s_set_vgpr_msb 0x5052
	v_wmma_f32_16x16x32_bf16 v[122:129] /*v[378:385]*/, v[14:21] /*v[526:533]*/, v[144:151], v[122:129] /*v[378:385]*/
	v_wmma_f32_16x16x32_bf16 v[170:177] /*v[426:433]*/, v[30:37] /*v[542:549]*/, v[168:175], v[170:177] /*v[426:433]*/
	s_set_vgpr_msb 0x5251
	v_wmma_f32_16x16x32_bf16 v[130:137] /*v[386:393]*/, v[242:249] /*v[498:505]*/, v[184:191], v[130:137] /*v[386:393]*/
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x5115
	v_max3_num_f32 v205, v113 /*v369*/, v130 /*v386*/, v131 /*v387*/
	s_set_vgpr_msb 0x1552
	v_wmma_f32_16x16x32_bf16 v[146:153] /*v[402:409]*/, v[14:21] /*v[526:533]*/, v[176:183], v[146:153] /*v[402:409]*/
	s_set_vgpr_msb 0x5215
	v_max3_num_f32 v206, v132 /*v388*/, v133 /*v389*/, v134 /*v390*/
	v_max3_num_f32 v209, v135 /*v391*/, v136 /*v392*/, v137 /*v393*/
	s_set_vgpr_msb 0x1500
	v_max3_num_f32 v205, v207, v205, v206
	s_set_vgpr_msb 0xa2
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[0:7] /*v[512:519]*/, v[38:45] /*v[550:557]*/, v[144:151], v[0:7] /*v[512:519]*/
	s_set_vgpr_msb 0xa250
	v_wmma_f32_16x16x32_bf16 v[154:161] /*v[410:417]*/, v[240:247], v[152:159], v[154:161] /*v[410:417]*/
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x5015
	v_max3_num_f32 v220, v32 /*v288*/, v33 /*v289*/, v154 /*v410*/
	s_set_vgpr_msb 0x1550
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[274:281]*/, v[240:247], v[184:191], v[18:25] /*v[274:281]*/
	s_set_vgpr_msb 0x5015
	v_max3_num_f32 v221, v155 /*v411*/, v156 /*v412*/, v157 /*v413*/
	v_max3_num_f32 v224, v158 /*v414*/, v159 /*v415*/, v160 /*v416*/
	s_set_vgpr_msb 0x1500
	v_max3_num_f32 v216, v216, v218, v220
	s_set_vgpr_msb 41
	v_max3_num_f32 v218, v161 /*v417*/, v170 /*v682*/, v171 /*v683*/
	s_set_vgpr_msb 0x292a
	v_max3_num_f32 v220, v175 /*v687*/, v176 /*v688*/, v177 /*v689*/
	s_set_vgpr_msb 0x2a15
	v_max3_num_f32 v222, v8 /*v264*/, v9 /*v265*/, v18 /*v274*/
	s_set_vgpr_msb 0x1552
	v_wmma_f32_16x16x32_bf16 v[122:129] /*v[378:385]*/, v[22:29] /*v[534:541]*/, v[152:159], v[122:129] /*v[378:385]*/
	s_set_vgpr_msb 0x5215
	v_max3_num_f32 v223, v19 /*v275*/, v20 /*v276*/, v21 /*v277*/
	v_max3_num_f32 v225, v22 /*v278*/, v23 /*v279*/, v24 /*v280*/
	s_set_vgpr_msb 0x1500
	v_max3_num_f32 v218, v221, v224, v218
	v_max3_num_f32 v217, v217, v219, v222
	s_set_vgpr_msb 42
	v_max3_num_f32 v219, v172 /*v684*/, v173 /*v685*/, v174 /*v686*/
	s_set_vgpr_msb 0x2a15
	v_max3_num_f32 v221, v25 /*v281*/, v66 /*v322*/, v67 /*v323*/
	v_max3_num_f32 v222, v68 /*v324*/, v69 /*v325*/, v70 /*v326*/
	s_set_vgpr_msb 0x1552
	v_wmma_f32_16x16x32_bf16 v[170:177] /*v[426:433]*/, v[38:45] /*v[550:557]*/, v[176:183], v[170:177] /*v[426:433]*/
	s_set_vgpr_msb 0x5215
	v_max3_num_f32 v206, v122 /*v378*/, v123 /*v379*/, v124 /*v380*/
	v_max3_num_f32 v207, v125 /*v381*/, v126 /*v382*/, v127 /*v383*/
	v_max_num_f32_e32 v210, v128 /*v384*/, v129 /*v385*/
	v_max3_num_f32 v224, v71 /*v327*/, v72 /*v328*/, v73 /*v329*/
	s_set_vgpr_msb 0x1500
	v_max3_num_f32 v200, v219, v220, v200
	v_max3_num_f32 v221, v223, v225, v221
	v_max3_num_f32 v206, v208, v206, v207
	s_set_vgpr_msb 0x52
	v_wmma_f32_16x16x32_bf16 v[146:153] /*v[402:409]*/, v[22:29] /*v[534:541]*/, v[184:191], v[146:153] /*v[402:409]*/
	s_set_vgpr_msb 0x5200
	v_max3_num_f32 v202, v222, v224, v202
	v_max3_num_f32 v200, v216, v218, v200
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x82
	ds_load_b128 v[14:17] /*v[526:529]*/, v210 /*v722*/ offset:26112
	ds_load_b128 v[18:21] /*v[530:533]*/, v210 /*v722*/ offset:26144
	ds_load_b128 v[22:25] /*v[534:537]*/, v210 /*v722*/ offset:26176
	ds_load_b128 v[26:29] /*v[538:541]*/, v210 /*v722*/ offset:26208
	s_set_vgpr_msb 0x8200
	v_max3_num_f32 v201, v201, v204, v206
	s_set_vgpr_msb 0x82
	ds_load_b128 v[30:33] /*v[542:545]*/, v210 /*v722*/ offset:26240
	ds_load_b128 v[34:37] /*v[546:549]*/, v210 /*v722*/ offset:26272
	ds_load_b128 v[38:41] /*v[550:553]*/, v210 /*v722*/ offset:26304
	ds_load_b128 v[42:45] /*v[554:557]*/, v210 /*v722*/ offset:26336
	s_set_vgpr_msb 0x8200
	v_max3_num_f32 v202, v217, v221, v202
	s_set_vgpr_msb 21
	v_max3_num_f32 v207, v146 /*v402*/, v147 /*v403*/, v148 /*v404*/
	s_set_vgpr_msb 0x15a2
	v_wmma_f32_16x16x32_bf16 v[0:7] /*v[512:519]*/, v[46:53] /*v[558:565]*/, v[152:159], v[0:7] /*v[512:519]*/
	s_set_vgpr_msb 0xa215
	v_max3_num_f32 v208, v149 /*v405*/, v150 /*v406*/, v151 /*v407*/
	v_max_num_f32_e32 v211, v152 /*v408*/, v153 /*v409*/
	s_set_vgpr_msb 0x1500
	v_max3_num_f32 v204, v209, v207, v208
	v_nop
	s_set_vgpr_msb 42
	v_max3_num_f32 v192, v1 /*v513*/, v2 /*v514*/, v3 /*v515*/
	s_set_vgpr_msb 0x2a52
	v_wmma_f32_16x16x32_bf16 v[170:177] /*v[426:433]*/, v[46:53] /*v[558:565]*/, v[184:191], v[170:177] /*v[426:433]*/
	s_set_vgpr_msb 0x522a
	v_max3_num_f32 v193, v4 /*v516*/, v5 /*v517*/, v6 /*v518*/
	s_set_vgpr_msb 0x2a00
	v_max3_num_f32 v203, v203, v205, v204
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x82
	ds_load_b128 v[46:49] /*v[558:561]*/, v210 /*v722*/ offset:30464
	ds_load_b128 v[50:53] /*v[562:565]*/, v210 /*v722*/ offset:30496
	s_set_vgpr_msb 0x8208
	v_max3_num_f32 v192, v210, v0 /*v512*/, v192
	s_set_vgpr_msb 0x820
	v_max3_num_f32 v192, v192, v193, v7 /*v519*/
	s_set_vgpr_msb 0x2015
	v_max3_num_f32 v194, v171 /*v427*/, v172 /*v428*/, v173 /*v429*/
	v_max3_num_f32 v204, v174 /*v430*/, v175 /*v431*/, v176 /*v432*/
	s_set_vgpr_msb 0x1502
	v_wmma_f32_16x16x32_bf16 v[216:223], v[54:61] /*v[566:573]*/, v[128:135], 0
	s_set_vgpr_msb 0x200
	v_max3_num_f32 v200, v200, v201, v192
	s_set_vgpr_msb 4
	v_max3_num_f32 v205, v211, v170 /*v426*/, v194
	s_set_vgpr_msb 0x410
	v_max3_num_f32 v201, v205, v204, v177 /*v433*/
	s_set_vgpr_msb 0x1002
	v_wmma_f32_16x16x32_bf16 v[192:199], v[54:61] /*v[566:573]*/, v[160:167], 0
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x282
	ds_load_b128 v[54:57] /*v[566:569]*/, v210 /*v722*/ offset:30528
	ds_load_b128 v[58:61] /*v[570:573]*/, v210 /*v722*/ offset:30560
	s_set_vgpr_msb 0x8200
	v_max3_num_f32 v240, v202, v203, v201
	v_dual_mov_b32 v204, v200 :: v_dual_mov_b32 v241, v240
	v_permlanex16_b32 v204, v204, s16, 0xfedcba98
	s_set_vgpr_msb 2
	v_wmma_f32_16x16x32_bf16 v[216:223], v[62:69] /*v[574:581]*/, v[136:143], v[216:223]
	s_set_vgpr_msb 0x200
	v_permlanex16_b32 v241, v241, s16, 0xfedcba98
	v_max_num_f32_e32 v242, v200, v204
	v_max_num_f32_e32 v240, v240, v241
	s_set_vgpr_msb 8
	v_sub_f32_e32 v243, v242, v209 /*v721*/
	v_max_num_f32_e32 v242, v242, v209 /*v721*/
	s_set_vgpr_msb 0x802
	v_wmma_f32_16x16x32_bf16 v[192:199], v[62:69] /*v[574:581]*/, v[168:175], v[192:199]
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x282
	ds_load_b128 v[62:65] /*v[574:577]*/, v210 /*v722*/ offset:30592
	ds_load_b128 v[66:69] /*v[578:581]*/, v210 /*v722*/ offset:30624
	s_set_vgpr_msb 0x8208
	v_sub_f32_e32 v241, v240, v211 /*v723*/
	s_set_vgpr_msb 0x802
	v_cmp_lt_f32_e32 vcc_lo, 0x41000000, v243
	v_max_num_f32_e32 v240, v211 /*v723*/, v240
	v_cmp_lt_f32_e64 s2, 0x41000000, v241
	s_cmp_eq_u32 vcc_lo, 0
	v_wmma_f32_16x16x32_bf16 v[216:223], v[70:77] /*v[582:589]*/, v[144:151], v[216:223]
	s_cselect_b32 s3, -1, 0
	s_cmp_lg_u32 s2, 0
	s_set_vgpr_msb 0x288
	v_cndmask_b32_e64 v222 /*v734*/, v242, v209 /*v721*/, s3
	s_cselect_b32 s3, -1, 0
	s_cmp_eq_u32 s2, 0
	s_cselect_b32 s2, -1, 0
	s_set_vgpr_msb 0x8802
	v_wmma_f32_16x16x32_bf16 v[192:199], v[70:77] /*v[582:589]*/, v[176:183], v[192:199]
	s_set_vgpr_msb 0x288
	v_cndmask_b32_e64 v221 /*v733*/, v240, v211 /*v723*/, s2
	v_mul_f32_e32 v8 /*v520*/, 0xbfb8aa3b, v222 /*v734*/
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x8882
	ds_load_b128 v[70:73] /*v[582:585]*/, v210 /*v722*/ offset:30656
	ds_load_b128 v[74:77] /*v[586:589]*/, v210 /*v722*/ offset:30688
	s_wait_alu depctr_vm_vsrc(0)
	v_mov_b32_e32 v210 /*v722*/, v12 /*v524*/
	s_set_vgpr_msb 0x8261
	v_pk_fma_f32 v[26:27] /*v[282:283]*/, v[26:27] /*v[282:283]*/, s[14:15], v[8:9] /*v[520:521]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6142
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[266:273]*/, v[162:169] /*v[674:681]*/, v[128:135], 0
	s_set_vgpr_msb 0x4261
	v_pk_fma_f32 v[28:29] /*v[284:285]*/, v[28:29] /*v[284:285]*/, s[14:15], v[8:9] /*v[520:521]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[30:31] /*v[286:287]*/, v[30:31] /*v[286:287]*/, s[14:15], v[8:9] /*v[520:521]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6120
	v_pk_fma_f32 v[248:249], v[248:249], s[14:15], v[8:9] /*v[520:521]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x2061
	v_exp_f32_e32 v186 /*v442*/, v26 /*v282*/
	v_exp_f32_e32 v188 /*v444*/, v27 /*v283*/
	v_nop
	v_pk_fma_f32 v[26:27] /*v[282:283]*/, v[32:33] /*v[288:289]*/, s[14:15], v[8:9] /*v[520:521]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v190 /*v446*/, v28 /*v284*/
	s_set_vgpr_msb 0x6102
	v_wmma_f32_16x16x32_bf16 v[240:247], v[162:169] /*v[674:681]*/, v[160:167], 0
	s_set_vgpr_msb 0x261
	v_exp_f32_e32 v192 /*v448*/, v29 /*v285*/
	v_nop
	v_pk_fma_f32 v[28:29] /*v[284:285]*/, v[154:155] /*v[410:411]*/, s[14:15], v[8:9] /*v[520:521]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v196 /*v452*/, v26 /*v282*/
	v_exp_f32_e32 v198 /*v454*/, v27 /*v283*/
	v_nop
	v_pk_fma_f32 v[26:27] /*v[282:283]*/, v[156:157] /*v[412:413]*/, s[14:15], v[8:9] /*v[520:521]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v194 /*v450*/, v30 /*v286*/
	v_exp_f32_e32 v154 /*v410*/, v31 /*v287*/
	s_set_vgpr_msb 0x6102
	v_wmma_f32_16x16x32_bf16 v[216:223], v[78:85] /*v[590:597]*/, v[152:159], v[216:223]
	s_set_vgpr_msb 0x261
	v_exp_f32_e32 v156 /*v412*/, v28 /*v284*/
	v_pk_fma_f32 v[30:31] /*v[286:287]*/, v[158:159] /*v[414:415]*/, s[14:15], v[8:9] /*v[520:521]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v158 /*v414*/, v29 /*v285*/
	v_nop
	s_set_vgpr_msb 0x6162
	v_pk_fma_f32 v[28:29] /*v[284:285]*/, v[170:171] /*v[682:683]*/, s[14:15], v[8:9] /*v[520:521]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6220
	v_pk_fma_f32 v[250:251], v[250:251], s[14:15], v[8:9] /*v[520:521]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x2060
	v_pk_fma_f32 v[226:227] /*v[482:483]*/, v[252:253], s[14:15], v[8:9] /*v[520:521]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6041
	v_exp_f32_e32 v200 /*v456*/, v30 /*v286*/
	s_set_vgpr_msb 0x4102
	v_wmma_f32_16x16x32_bf16 v[192:199], v[78:85] /*v[590:597]*/, v[184:191], v[192:199]
	s_set_vgpr_msb 0x241
	v_exp_f32_e32 v202 /*v458*/, v31 /*v287*/
	v_nop
	s_set_vgpr_msb 0x4162
	v_pk_fma_f32 v[30:31] /*v[286:287]*/, v[172:173] /*v[684:685]*/, s[14:15], v[8:9] /*v[520:521]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6260
	v_exp_f32_e32 v218 /*v474*/, v251
	v_pk_fma_f32 v[228:229] /*v[484:485]*/, v[254:255], s[14:15], v[8:9] /*v[520:521]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6088
	v_mul_f32_e32 v78 /*v590*/, 0xbfb8aa3b, v221 /*v733*/
	s_set_vgpr_msb 0x8862
	v_pk_fma_f32 v[230:231] /*v[486:487]*/, v[186:187] /*v[698:699]*/, s[14:15], v[8:9] /*v[520:521]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[232:233] /*v[488:489]*/, v[188:189] /*v[700:701]*/, s[14:15], v[8:9] /*v[520:521]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6251
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[266:273]*/, v[178:185] /*v[434:441]*/, v[136:143], v[10:17] /*v[266:273]*/
	s_set_vgpr_msb 0x5162
	v_pk_fma_f32 v[234:235] /*v[490:491]*/, v[192:193] /*v[704:705]*/, s[14:15], v[8:9] /*v[520:521]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6261
	v_pk_fma_f32 v[4:5] /*v[260:261]*/, v[4:5] /*v[260:261]*/, s[14:15], v[78:79] /*v[590:591]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[2:3] /*v[258:259]*/, v[2:3] /*v[258:259]*/, s[14:15], v[78:79] /*v[590:591]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[6:7] /*v[262:263]*/, v[6:7] /*v[262:263]*/, s[14:15], v[78:79] /*v[590:591]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[122:123] /*v[378:379]*/, v[122:123] /*v[378:379]*/, s[14:15], v[8:9] /*v[520:521]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v236 /*v492*/, v233 /*v489*/
	v_exp_f32_e32 v191 /*v447*/, v4 /*v260*/
	s_set_vgpr_msb 0x6101
	v_wmma_f32_16x16x32_bf16 v[240:247], v[178:185] /*v[434:441]*/, v[168:175], v[240:247]
	s_set_vgpr_msb 0x161
	v_exp_f32_e32 v193 /*v449*/, v5 /*v261*/
	v_nop
	v_pk_fma_f32 v[4:5] /*v[260:261]*/, v[18:19] /*v[274:275]*/, s[14:15], v[78:79] /*v[590:591]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[18:19] /*v[274:275]*/, v[20:21] /*v[276:277]*/, s[14:15], v[78:79] /*v[590:591]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[20:21] /*v[276:277]*/, v[22:23] /*v[278:279]*/, s[14:15], v[78:79] /*v[590:591]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v178 /*v434*/, v26 /*v282*/
	v_exp_f32_e32 v182 /*v438*/, v27 /*v283*/
	v_nop
	v_pk_fma_f32 v[26:27] /*v[282:283]*/, v[160:161] /*v[416:417]*/, s[14:15], v[8:9] /*v[520:521]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6151
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[266:273]*/, v[162:169] /*v[418:425]*/, v[144:151], v[10:17] /*v[266:273]*/
	v_exp_f32_e32 v160 /*v416*/, v28 /*v284*/
	v_exp_f32_e32 v180 /*v436*/, v29 /*v285*/
	v_nop
	s_set_vgpr_msb 0x5162
	v_pk_fma_f32 v[28:29] /*v[284:285]*/, v[176:177] /*v[688:689]*/, s[14:15], v[8:9] /*v[520:521]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6241
	v_exp_f32_e32 v204 /*v460*/, v26 /*v282*/
	v_exp_f32_e32 v206 /*v462*/, v27 /*v283*/
	v_nop
	s_set_vgpr_msb 0x4162
	v_pk_fma_f32 v[26:27] /*v[282:283]*/, v[174:175] /*v[686:687]*/, s[14:15], v[8:9] /*v[520:521]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6241
	v_exp_f32_e32 v179 /*v435*/, v18 /*v274*/
	s_set_vgpr_msb 0x4101
	v_wmma_f32_16x16x32_bf16 v[240:247], v[162:169] /*v[418:425]*/, v[176:183], v[240:247]
	s_set_vgpr_msb 0x161
	v_exp_f32_e32 v183 /*v439*/, v19 /*v275*/
	v_nop
	v_pk_fma_f32 v[18:19] /*v[274:275]*/, v[66:67] /*v[322:323]*/, s[14:15], v[78:79] /*v[590:591]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v184 /*v440*/, v30 /*v286*/
	v_exp_f32_e32 v208 /*v464*/, v29 /*v285*/
	v_exp_f32_e32 v164 /*v420*/, v26 /*v282*/
	v_exp_f32_e32 v166 /*v422*/, v27 /*v283*/
	v_nop
	s_set_vgpr_msb 0x6162
	v_pk_fma_f32 v[26:27] /*v[282:283]*/, v[178:179] /*v[690:691]*/, s[14:15], v[8:9] /*v[520:521]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6241
	v_exp_f32_e32 v162 /*v418*/, v31 /*v287*/
	v_exp_f32_e32 v168 /*v424*/, v28 /*v284*/
	s_set_vgpr_msb 0x4162
	v_pk_fma_f32 v[30:31] /*v[286:287]*/, v[180:181] /*v[692:693]*/, s[14:15], v[8:9] /*v[520:521]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[28:29] /*v[284:285]*/, v[184:185] /*v[696:697]*/, s[14:15], v[8:9] /*v[520:521]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6241
	v_exp_f32_e32 v210 /*v466*/, v26 /*v282*/
	v_exp_f32_e32 v212 /*v468*/, v27 /*v283*/
	v_nop
	s_set_vgpr_msb 0x4162
	v_pk_fma_f32 v[26:27] /*v[282:283]*/, v[182:183] /*v[694:695]*/, s[14:15], v[8:9] /*v[520:521]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6261
	v_exp_f32_e32 v161 /*v417*/, v18 /*v274*/
	v_exp_f32_e32 v181 /*v437*/, v19 /*v275*/
	v_nop
	v_pk_fma_f32 v[18:19] /*v[274:275]*/, v[70:71] /*v[326:327]*/, s[14:15], v[78:79] /*v[590:591]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v201 /*v457*/, v20 /*v276*/
	v_exp_f32_e32 v203 /*v459*/, v21 /*v277*/
	v_nop
	v_pk_fma_f32 v[20:21] /*v[276:277]*/, v[68:69] /*v[324:325]*/, s[14:15], v[78:79] /*v[590:591]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6151
	v_wmma_f32_16x16x32_bf16 v[10:17] /*v[266:273]*/, v[138:145] /*v[394:401]*/, v[152:159], v[10:17] /*v[266:273]*/
	v_exp_f32_e32 v214 /*v470*/, v30 /*v286*/
	v_exp_f32_e32 v216 /*v472*/, v31 /*v287*/
	v_exp_f32_e32 v220 /*v476*/, v27 /*v283*/
	v_exp_f32_e32 v222 /*v478*/, v28 /*v284*/
	v_exp_f32_e32 v224 /*v480*/, v29 /*v285*/
	v_exp_f32_e32 v165 /*v421*/, v18 /*v274*/
	v_exp_f32_e32 v167 /*v423*/, v19 /*v275*/
	s_set_vgpr_msb 0x5101
	v_wmma_f32_16x16x32_bf16 v[240:247], v[138:145] /*v[394:401]*/, v[184:191], v[240:247]
	s_set_vgpr_msb 0x161
	v_pk_fma_f32 v[18:19] /*v[274:275]*/, v[84:85] /*v[340:341]*/, s[14:15], v[78:79] /*v[590:591]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[22:23] /*v[278:279]*/, v[24:25] /*v[280:281]*/, s[14:15], v[78:79] /*v[590:591]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v185 /*v441*/, v20 /*v276*/
	v_exp_f32_e32 v163 /*v419*/, v21 /*v277*/
	v_exp_f32_e32 v144 /*v400*/, v26 /*v282*/
	s_set_vgpr_msb 0x6140
	v_exp_f32_e32 v138 /*v394*/, v248
	v_exp_f32_e32 v140 /*v396*/, v249
	s_set_vgpr_msb 0x4041
	v_wmma_f32_16x16x32_bf16 v[26:33] /*v[282:289]*/, v[114:121] /*v[370:377]*/, v[128:135], 0
	s_set_vgpr_msb 0x4140
	v_exp_f32_e32 v142 /*v398*/, v250
	s_set_vgpr_msb 0x4061
	v_pk_fma_f32 v[20:21] /*v[276:277]*/, v[72:73] /*v[328:329]*/, s[14:15], v[78:79] /*v[590:591]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v215 /*v471*/, v18 /*v274*/
	v_exp_f32_e32 v217 /*v473*/, v19 /*v275*/
	v_nop
	v_pk_fma_f32 v[18:19] /*v[274:275]*/, v[88:89] /*v[344:345]*/, s[14:15], v[78:79] /*v[590:591]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v205 /*v461*/, v22 /*v278*/
	v_exp_f32_e32 v207 /*v463*/, v23 /*v279*/
	s_set_vgpr_msb 0x6101
	v_wmma_f32_16x16x32_bf16 v[248:255], v[114:121] /*v[370:377]*/, v[160:167], 0
	s_set_vgpr_msb 0x161
	v_pk_fma_f32 v[22:23] /*v[278:279]*/, v[82:83] /*v[338:339]*/, s[14:15], v[78:79] /*v[590:591]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v169 /*v425*/, v20 /*v276*/
	v_exp_f32_e32 v209 /*v465*/, v21 /*v277*/
	v_nop
	v_pk_fma_f32 v[20:21] /*v[276:277]*/, v[86:87] /*v[342:343]*/, s[14:15], v[78:79] /*v[590:591]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v223 /*v479*/, v18 /*v274*/
	v_exp_f32_e32 v225 /*v481*/, v19 /*v275*/
	v_nop
	v_pk_fma_f32 v[18:19] /*v[274:275]*/, v[108:109] /*v[364:365]*/, s[14:15], v[78:79] /*v[590:591]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6151
	v_wmma_f32_16x16x32_bf16 v[26:33] /*v[282:289]*/, v[98:105] /*v[354:361]*/, v[136:143], v[26:33] /*v[282:289]*/
	v_exp_f32_e32 v211 /*v467*/, v22 /*v278*/
	v_exp_f32_e32 v213 /*v469*/, v23 /*v279*/
	v_exp_f32_e32 v145 /*v401*/, v20 /*v276*/
	s_set_vgpr_msb 0x5161
	v_pk_fma_f32 v[22:23] /*v[278:279]*/, v[106:107] /*v[362:363]*/, s[14:15], v[78:79] /*v[590:591]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v221 /*v477*/, v21 /*v277*/
	v_nop
	v_pk_fma_f32 v[20:21] /*v[276:277]*/, v[110:111] /*v[366:367]*/, s[14:15], v[78:79] /*v[590:591]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v143 /*v399*/, v18 /*v274*/
	s_set_vgpr_msb 0x6101
	v_wmma_f32_16x16x32_bf16 v[248:255], v[98:105] /*v[354:361]*/, v[168:175], v[248:255]
	s_set_vgpr_msb 0x161
	v_exp_f32_e32 v219 /*v475*/, v19 /*v275*/
	v_nop
	v_pk_fma_f32 v[18:19] /*v[274:275]*/, v[130:131] /*v[386:387]*/, s[14:15], v[78:79] /*v[590:591]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v187 /*v443*/, v2 /*v258*/
	v_exp_f32_e32 v189 /*v445*/, v3 /*v259*/
	v_pk_fma_f32 v[98:99] /*v[354:355]*/, v[124:125] /*v[380:381]*/, s[14:15], v[8:9] /*v[520:521]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[100:101] /*v[356:357]*/, v[126:127] /*v[382:383]*/, s[14:15], v[8:9] /*v[520:521]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[102:103] /*v[358:359]*/, v[128:129] /*v[384:385]*/, s[14:15], v[8:9] /*v[520:521]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6151
	v_wmma_f32_16x16x32_bf16 v[26:33] /*v[282:289]*/, v[90:97] /*v[346:353]*/, v[144:151], v[26:33] /*v[282:289]*/
	s_set_vgpr_msb 0x5161
	v_pk_fma_f32 v[2:3] /*v[258:259]*/, v[8:9] /*v[264:265]*/, s[14:15], v[78:79] /*v[590:591]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v238 /*v494*/, v98 /*v354*/
	v_exp_f32_e32 v246 /*v502*/, v99 /*v355*/
	v_nop
	s_set_vgpr_msb 0x6162
	v_pk_fma_f32 v[98:99] /*v[354:355]*/, v[0:1] /*v[512:513]*/, s[14:15], v[8:9] /*v[520:521]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6241
	v_exp_f32_e32 v248 /*v504*/, v100 /*v356*/
	v_exp_f32_e32 v250 /*v506*/, v101 /*v357*/
	v_exp_f32_e32 v254 /*v510*/, v102 /*v358*/
	s_set_vgpr_msb 0x4181
	v_exp_f32_e32 v0 /*v512*/, v103 /*v359*/
	s_set_vgpr_msb 0x8162
	v_pk_fma_f32 v[100:101] /*v[356:357]*/, v[2:3] /*v[514:515]*/, s[14:15], v[8:9] /*v[520:521]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6201
	v_wmma_f32_16x16x32_bf16 v[248:255], v[90:97] /*v[346:353]*/, v[176:183], v[248:255]
	s_set_vgpr_msb 0x162
	v_pk_fma_f32 v[102:103] /*v[358:359]*/, v[4:5] /*v[516:517]*/, s[14:15], v[8:9] /*v[520:521]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6241
	v_exp_f32_e32 v114 /*v370*/, v227 /*v483*/
	v_exp_f32_e32 v139 /*v395*/, v22 /*v278*/
	v_exp_f32_e32 v141 /*v397*/, v23 /*v279*/
	v_exp_f32_e32 v90 /*v346*/, v98 /*v354*/
	v_exp_f32_e32 v92 /*v348*/, v99 /*v355*/
	v_nop
	s_set_vgpr_msb 0x4162
	v_pk_fma_f32 v[98:99] /*v[354:355]*/, v[6:7] /*v[518:519]*/, s[14:15], v[8:9] /*v[520:521]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6261
	v_pk_fma_f32 v[22:23] /*v[278:279]*/, v[112:113] /*v[368:369]*/, s[14:15], v[78:79] /*v[590:591]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v227 /*v483*/, v20 /*v276*/
	v_exp_f32_e32 v115 /*v371*/, v21 /*v277*/
	v_nop
	v_pk_fma_f32 v[20:21] /*v[276:277]*/, v[132:133] /*v[388:389]*/, s[14:15], v[78:79] /*v[590:591]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v118 /*v374*/, v229 /*v485*/
	v_exp_f32_e32 v121 /*v377*/, v18 /*v274*/
	v_exp_f32_e32 v229 /*v485*/, v19 /*v275*/
	v_nop
	v_pk_fma_f32 v[18:19] /*v[274:275]*/, v[134:135] /*v[390:391]*/, s[14:15], v[78:79] /*v[590:591]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v116 /*v372*/, v228 /*v484*/
	v_exp_f32_e32 v120 /*v376*/, v230 /*v486*/
	v_exp_f32_e32 v228 /*v484*/, v231 /*v487*/
	v_nop
	s_set_vgpr_msb 0x6162
	v_pk_fma_f32 v[230:231] /*v[486:487]*/, v[190:191] /*v[702:703]*/, s[14:15], v[8:9] /*v[520:521]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6241
	v_exp_f32_e32 v94 /*v350*/, v100 /*v356*/
	v_exp_f32_e32 v96 /*v352*/, v101 /*v357*/
	s_set_vgpr_msb 0x4181
	v_exp_f32_e32 v2 /*v514*/, v102 /*v358*/
	v_exp_f32_e32 v4 /*v516*/, v103 /*v359*/
	v_exp_f32_e32 v6 /*v518*/, v98 /*v354*/
	v_exp_f32_e32 v8 /*v520*/, v99 /*v355*/
	s_set_vgpr_msb 0x8161
	v_exp_f32_e32 v195 /*v451*/, v6 /*v262*/
	v_exp_f32_e32 v155 /*v411*/, v7 /*v263*/
	v_wmma_f32_16x16x32_bf16 v[98:105] /*v[354:361]*/, v[58:65] /*v[314:321]*/, v[128:135], 0
	v_exp_f32_e32 v197 /*v453*/, v2 /*v258*/
	v_exp_f32_e32 v199 /*v455*/, v3 /*v259*/
	v_exp_f32_e32 v157 /*v413*/, v4 /*v260*/
	v_exp_f32_e32 v159 /*v415*/, v5 /*v261*/
	v_exp_f32_e32 v117 /*v373*/, v22 /*v278*/
	v_exp_f32_e32 v119 /*v375*/, v23 /*v279*/
	v_exp_f32_e32 v233 /*v489*/, v20 /*v276*/
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[258:265]*/, v[58:65] /*v[314:321]*/, v[160:167], 0
	v_pk_fma_f32 v[22:23] /*v[278:279]*/, v[136:137] /*v[392:393]*/, s[14:15], v[78:79] /*v[590:591]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v237 /*v493*/, v21 /*v277*/
	v_nop
	v_pk_fma_f32 v[20:21] /*v[276:277]*/, v[148:149] /*v[404:405]*/, s[14:15], v[78:79] /*v[590:591]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v241 /*v497*/, v18 /*v274*/
	v_exp_f32_e32 v243 /*v499*/, v19 /*v275*/
	v_nop
	v_pk_fma_f32 v[18:19] /*v[274:275]*/, v[146:147] /*v[402:403]*/, s[14:15], v[78:79] /*v[590:591]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v245 /*v501*/, v22 /*v278*/
	v_exp_f32_e32 v253 /*v509*/, v23 /*v279*/
	v_nop
	v_pk_fma_f32 v[22:23] /*v[278:279]*/, v[150:151] /*v[406:407]*/, s[14:15], v[78:79] /*v[590:591]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v239 /*v495*/, v20 /*v276*/
	v_exp_f32_e32 v247 /*v503*/, v21 /*v277*/
	v_nop
	v_pk_fma_f32 v[20:21] /*v[276:277]*/, v[170:171] /*v[426:427]*/, s[14:15], v[78:79] /*v[590:591]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v226 /*v482*/, v226 /*v482*/
	v_exp_f32_e32 v242 /*v498*/, v231 /*v487*/
	v_exp_f32_e32 v252 /*v508*/, v235 /*v491*/
	s_set_vgpr_msb 0x6151
	v_wmma_f32_16x16x32_bf16 v[98:105] /*v[354:361]*/, v[50:57] /*v[306:313]*/, v[136:143], v[98:105] /*v[354:361]*/
	v_exp_f32_e32 v231 /*v487*/, v18 /*v274*/
	v_exp_f32_e32 v235 /*v491*/, v19 /*v275*/
	v_nop
	s_set_vgpr_msb 0x5161
	v_pk_fma_f32 v[18:19] /*v[274:275]*/, v[152:153] /*v[408:409]*/, s[14:15], v[78:79] /*v[590:591]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v249 /*v505*/, v22 /*v278*/
	v_exp_f32_e32 v251 /*v507*/, v23 /*v279*/
	v_exp_f32_e32 v91 /*v347*/, v20 /*v276*/
	v_pk_fma_f32 v[22:23] /*v[278:279]*/, v[174:175] /*v[430:431]*/, s[14:15], v[78:79] /*v[590:591]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6151
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[258:265]*/, v[50:57] /*v[306:313]*/, v[168:175], v[2:9] /*v[258:265]*/
	v_exp_f32_e32 v93 /*v349*/, v21 /*v277*/
	v_nop
	s_set_vgpr_msb 0x5145
	v_pk_add_f32 v[20:21] /*v[276:277]*/, v[186:187] /*v[442:443]*/, v[188:189] /*v[444:445]*/
	v_pk_add_f32 v[24:25] /*v[280:281]*/, v[192:193] /*v[448:449]*/, v[194:195] /*v[450:451]*/
	v_exp_f32_e32 v232 /*v488*/, v232 /*v488*/
	v_exp_f32_e32 v240 /*v496*/, v230 /*v486*/
	v_exp_f32_e32 v244 /*v500*/, v234 /*v490*/
	v_exp_f32_e32 v230 /*v486*/, v122 /*v378*/
	v_exp_f32_e32 v234 /*v490*/, v123 /*v379*/
	v_exp_f32_e32 v255 /*v511*/, v18 /*v274*/
	s_set_vgpr_msb 0x4581
	v_exp_f32_e32 v1 /*v513*/, v19 /*v275*/
	v_nop
	s_set_vgpr_msb 0x8161
	v_pk_fma_f32 v[18:19] /*v[274:275]*/, v[172:173] /*v[428:429]*/, s[14:15], v[78:79] /*v[590:591]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6151
	v_wmma_f32_16x16x32_bf16 v[98:105] /*v[354:361]*/, v[42:49] /*v[298:305]*/, v[144:151], v[98:105] /*v[354:361]*/
	s_set_vgpr_msb 0x5181
	v_exp_f32_e32 v3 /*v515*/, v22 /*v278*/
	v_exp_f32_e32 v5 /*v517*/, v23 /*v279*/
	s_set_vgpr_msb 0x8145
	v_pk_add_f32 v[20:21] /*v[276:277]*/, v[190:191] /*v[446:447]*/, v[20:21] /*v[276:277]*/
	v_pk_add_f32 v[22:23] /*v[278:279]*/, v[196:197] /*v[452:453]*/, v[198:199] /*v[454:455]*/
	v_pk_add_f32 v[24:25] /*v[280:281]*/, v[154:155] /*v[410:411]*/, v[24:25] /*v[280:281]*/
	v_pk_add_f32 v[50:51] /*v[306:307]*/, v[166:167] /*v[422:423]*/, v[168:169] /*v[424:425]*/
	v_pk_add_f32 v[54:55] /*v[310:311]*/, v[216:217] /*v[472:473]*/, v[144:145] /*v[400:401]*/
	s_set_vgpr_msb 0x4551
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[258:265]*/, v[42:49] /*v[298:305]*/, v[176:183], v[2:9] /*v[258:265]*/
	s_set_vgpr_msb 0x5165
	v_pk_add_f32 v[56:57] /*v[312:313]*/, v[222:223] /*v[478:479]*/, v[224:225] /*v[480:481]*/
	v_pk_add_f32 v[60:61] /*v[316:317]*/, v[226:227] /*v[482:483]*/, v[114:115] /*v[370:371]*/
	v_pk_add_f32 v[62:63] /*v[318:319]*/, v[118:119] /*v[374:375]*/, v[120:121] /*v[376:377]*/
	v_exp_f32_e32 v95 /*v351*/, v18 /*v274*/
	v_pk_add_f32 v[42:43] /*v[298:299]*/, v[158:159] /*v[414:415]*/, v[178:179] /*v[434:435]*/
	v_pk_add_f32 v[44:45] /*v[300:301]*/, v[200:201] /*v[456:457]*/, v[202:203] /*v[458:459]*/
	v_pk_add_f32 v[48:49] /*v[304:305]*/, v[184:185] /*v[440:441]*/, v[162:163] /*v[418:419]*/
	v_pk_add_f32 v[22:23] /*v[278:279]*/, v[156:157] /*v[412:413]*/, v[22:23] /*v[278:279]*/
	v_pk_add_f32 v[46:47] /*v[302:303]*/, v[206:207] /*v[462:463]*/, v[160:161] /*v[416:417]*/
	v_pk_add_f32 v[42:43] /*v[298:299]*/, v[182:183] /*v[438:439]*/, v[42:43] /*v[298:299]*/
	v_pk_add_f32 v[44:45] /*v[300:301]*/, v[204:205] /*v[460:461]*/, v[44:45] /*v[300:301]*/
	v_pk_add_f32 v[48:49] /*v[304:305]*/, v[164:165] /*v[420:421]*/, v[48:49] /*v[304:305]*/
	v_pk_add_f32 v[50:51] /*v[306:307]*/, v[208:209] /*v[464:465]*/, v[50:51] /*v[306:307]*/
	v_pk_add_f32 v[58:59] /*v[314:315]*/, v[140:141] /*v[396:397]*/, v[142:143] /*v[398:399]*/
	v_pk_add_f32 v[54:55] /*v[310:311]*/, v[220:221] /*v[476:477]*/, v[54:55] /*v[310:311]*/
	v_pk_add_f32 v[56:57] /*v[312:313]*/, v[138:139] /*v[394:395]*/, v[56:57] /*v[312:313]*/
	v_pk_add_f32 v[64:65] /*v[320:321]*/, v[232:233] /*v[488:489]*/, v[236:237] /*v[492:493]*/
	v_pk_add_f32 v[66:67] /*v[322:323]*/, v[242:243] /*v[498:499]*/, v[244:245] /*v[500:501]*/
	v_pk_add_f32 v[60:61] /*v[316:317]*/, v[116:117] /*v[372:373]*/, v[60:61] /*v[316:317]*/
	v_pk_add_f32 v[68:69] /*v[324:325]*/, v[230:231] /*v[486:487]*/, v[234:235] /*v[490:491]*/
	v_pk_add_f32 v[62:63] /*v[318:319]*/, v[228:229] /*v[484:485]*/, v[62:63] /*v[318:319]*/
	v_pk_add_f32 v[20:21] /*v[276:277]*/, v[20:21] /*v[276:277]*/, v[24:25] /*v[280:281]*/
	v_exp_f32_e32 v97 /*v353*/, v19 /*v275*/
	v_nop
	v_pk_fma_f32 v[18:19] /*v[274:275]*/, v[176:177] /*v[432:433]*/, s[14:15], v[78:79] /*v[590:591]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6551
	v_wmma_f32_16x16x32_bf16 v[26:33] /*v[282:289]*/, v[74:81] /*v[330:337]*/, v[152:159], v[26:33] /*v[282:289]*/
	s_set_vgpr_msb 0x5145
	v_pk_add_f32 v[46:47] /*v[302:303]*/, v[180:181] /*v[436:437]*/, v[46:47] /*v[302:303]*/
	v_pk_add_f32 v[52:53] /*v[308:309]*/, v[210:211] /*v[466:467]*/, v[212:213] /*v[468:469]*/
	v_pk_add_f32 v[58:59] /*v[314:315]*/, v[218:219] /*v[474:475]*/, v[58:59] /*v[314:315]*/
	v_pk_add_f32 v[64:65] /*v[320:321]*/, v[240:241] /*v[496:497]*/, v[64:65] /*v[320:321]*/
	v_pk_add_f32 v[66:67] /*v[322:323]*/, v[252:253] /*v[508:509]*/, v[66:67] /*v[322:323]*/
	v_pk_add_f32 v[70:71] /*v[326:327]*/, v[246:247] /*v[502:503]*/, v[248:249] /*v[504:505]*/
	v_pk_add_f32 v[68:69] /*v[324:325]*/, v[238:239] /*v[494:495]*/, v[68:69] /*v[324:325]*/
	s_set_vgpr_msb 0x4501
	v_wmma_f32_16x16x32_bf16 v[248:255], v[74:81] /*v[330:337]*/, v[184:191], v[248:255]
	s_set_vgpr_msb 0x149
	v_pk_add_f32 v[72:73] /*v[328:329]*/, v[254:255] /*v[510:511]*/, v[0:1] /*v[512:513]*/
	s_set_vgpr_msb 0x4945
	v_pk_add_f32 v[20:21] /*v[276:277]*/, v[22:23] /*v[278:279]*/, v[20:21] /*v[276:277]*/
	v_pk_add_f32 v[22:23] /*v[278:279]*/, v[42:43] /*v[298:299]*/, v[44:45] /*v[300:301]*/
	v_pk_add_f32 v[42:43] /*v[298:299]*/, v[48:49] /*v[304:305]*/, v[50:51] /*v[306:307]*/
	v_pk_add_f32 v[74:75] /*v[330:331]*/, v[92:93] /*v[348:349]*/, v[94:95] /*v[350:351]*/
	v_pk_add_f32 v[44:45] /*v[300:301]*/, v[54:55] /*v[310:311]*/, v[56:57] /*v[312:313]*/
	v_pk_add_f32 v[48:49] /*v[304:305]*/, v[60:61] /*v[316:317]*/, v[62:63] /*v[318:319]*/
	s_set_vgpr_msb 0x4551
	v_wmma_f32_16x16x32_bf16 v[98:105] /*v[354:361]*/, v[34:41] /*v[290:297]*/, v[152:159], v[98:105] /*v[354:361]*/
	s_set_vgpr_msb 0x5181
	v_exp_f32_e32 v7 /*v519*/, v18 /*v274*/
	s_set_vgpr_msb 0x8145
	v_pk_add_f32 v[52:53] /*v[308:309]*/, v[214:215] /*v[470:471]*/, v[52:53] /*v[308:309]*/
	v_pk_add_f32 v[70:71] /*v[326:327]*/, v[250:251] /*v[506:507]*/, v[70:71] /*v[326:327]*/
	s_set_vgpr_msb 0x454a
	v_pk_add_f32 v[76:77] /*v[332:333]*/, v[2:3] /*v[514:515]*/, v[4:5] /*v[516:517]*/
	s_set_vgpr_msb 0x4a45
	v_pk_add_f32 v[24:25] /*v[280:281]*/, v[90:91] /*v[346:347]*/, v[72:73] /*v[328:329]*/
	v_pk_add_f32 v[72:73] /*v[328:329]*/, v[96:97] /*v[352:353]*/, v[74:75] /*v[330:331]*/
	v_pk_add_f32 v[50:51] /*v[306:307]*/, v[66:67] /*v[322:323]*/, v[68:69] /*v[324:325]*/
	s_set_vgpr_msb 0x4551
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[258:265]*/, v[34:41] /*v[290:297]*/, v[184:191], v[2:9] /*v[258:265]*/
	s_set_vgpr_msb 0x5145
	v_pk_add_f32 v[22:23] /*v[278:279]*/, v[46:47] /*v[302:303]*/, v[22:23] /*v[278:279]*/
	v_pk_add_f32 v[44:45] /*v[300:301]*/, v[58:59] /*v[314:315]*/, v[44:45] /*v[300:301]*/
	v_pk_add_f32 v[46:47] /*v[302:303]*/, v[64:65] /*v[320:321]*/, v[48:49] /*v[304:305]*/
	s_set_vgpr_msb 0x4546
	v_pk_add_f32 v[74:75] /*v[330:331]*/, v[6:7] /*v[518:519]*/, v[76:77] /*v[332:333]*/
	s_set_vgpr_msb 0x4645
	v_pk_add_f32 v[42:43] /*v[298:299]*/, v[52:53] /*v[308:309]*/, v[42:43] /*v[298:299]*/
	v_pk_add_f32 v[48:49] /*v[304:305]*/, v[70:71] /*v[326:327]*/, v[50:51] /*v[306:307]*/
	v_pk_add_f32 v[24:25] /*v[280:281]*/, v[24:25] /*v[280:281]*/, v[72:73] /*v[328:329]*/
	s_set_vgpr_msb 0x4542
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x32_bf16 v[122:129] /*v[378:385]*/, v[14:21] /*v[526:533]*/, v[128:135], 0
	s_set_vgpr_msb 0x4245
	v_pk_add_f32 v[20:21] /*v[276:277]*/, v[20:21] /*v[276:277]*/, v[22:23] /*v[278:279]*/
	v_pk_add_f32 v[22:23] /*v[278:279]*/, v[44:45] /*v[300:301]*/, v[46:47] /*v[302:303]*/
	s_set_vgpr_msb 0x4581
	v_exp_f32_e32 v9 /*v521*/, v19 /*v275*/
	v_nop
	s_set_vgpr_msb 0x8145
	v_pk_add_f32 v[18:19] /*v[274:275]*/, v[74:75] /*v[330:331]*/, v[24:25] /*v[280:281]*/
	v_pk_add_f32 v[20:21] /*v[276:277]*/, v[42:43] /*v[298:299]*/, v[20:21] /*v[276:277]*/
	v_pk_add_f32 v[22:23] /*v[278:279]*/, v[48:49] /*v[304:305]*/, v[22:23] /*v[278:279]*/
	s_set_vgpr_msb 0x4542
	v_wmma_f32_16x16x32_bf16 v[34:41] /*v[290:297]*/, v[14:21] /*v[526:533]*/, v[160:167], 0
	s_set_vgpr_msb 0x4246
	v_pk_add_f32 v[18:19] /*v[274:275]*/, v[8:9] /*v[520:521]*/, v[18:19] /*v[274:275]*/
	s_set_vgpr_msb 0x4645
	v_pk_add_f32 v[20:21] /*v[276:277]*/, v[20:21] /*v[276:277]*/, v[22:23] /*v[278:279]*/
	s_set_vgpr_msb 0x454a
	v_sub_f32_e32 v22 /*v278*/, v209 /*v721*/, v222 /*v734*/
	s_set_vgpr_msb 0x4a85
	v_pk_add_f32 v[132:133] /*v[644:645]*/, v[18:19] /*v[274:275]*/, v[20:21] /*v[276:277]*/
	s_set_vgpr_msb 0x8502
	v_wmma_f32_16x16x32_bf16 v[224:231], v[86:93] /*v[598:605]*/, v[128:135], 0
	s_set_vgpr_msb 0x244
	v_mul_f32_e32 v18 /*v274*/, 0x3fb8aa3b, v22 /*v278*/
	s_set_vgpr_msb 0x4482
	v_dual_mov_b32 v219 /*v731*/, v132 /*v644*/ :: v_dual_mov_b32 v220 /*v732*/, v133 /*v645*/
	s_set_vgpr_msb 0x8281
	v_exp_f32_e32 v134 /*v646*/, v18 /*v274*/
	s_set_vgpr_msb 0x8182
	v_permlanex16_b32 v219 /*v731*/, v219 /*v731*/, s16, 0xfedcba98
	s_set_vgpr_msb 0x8202
	v_wmma_f32_16x16x32_bf16 v[200:207], v[86:93] /*v[598:605]*/, v[160:167], 0
	s_set_vgpr_msb 0x282
	v_permlanex16_b32 v220 /*v732*/, v220 /*v732*/, s16, 0xfedcba98
	s_set_vgpr_msb 0x8202
	v_wmma_f32_16x16x32_bf16 v[232:239], v[118:125] /*v[630:637]*/, v[128:135], 0
	v_wmma_f32_16x16x32_bf16 v[208:215], v[118:125] /*v[630:637]*/, v[160:167], 0
	s_set_vgpr_msb 0x252
	s_wait_dscnt 0xc
	v_wmma_f32_16x16x32_bf16 v[122:129] /*v[378:385]*/, v[22:29] /*v[534:541]*/, v[136:143], v[122:129] /*v[378:385]*/
	v_wmma_f32_16x16x32_bf16 v[34:41] /*v[290:297]*/, v[22:29] /*v[534:541]*/, v[168:175], v[34:41] /*v[290:297]*/
	s_set_vgpr_msb 0x5282
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[530:537]*/, v[46:53] /*v[558:565]*/, v[128:135], 0
	s_set_vgpr_msb 0x8242
	v_wmma_f32_16x16x32_bf16 v[130:137] /*v[386:393]*/, v[46:53] /*v[558:565]*/, v[160:167], 0
	s_set_vgpr_msb 0x4202
	v_wmma_f32_16x16x32_bf16 v[224:231], v[94:101] /*v[606:613]*/, v[136:143], v[224:231]
	v_wmma_f32_16x16x32_bf16 v[200:207], v[94:101] /*v[606:613]*/, v[168:175], v[200:207]
	v_wmma_f32_16x16x32_bf16 v[232:239], v[138:145] /*v[650:657]*/, v[136:143], v[232:239]
	v_wmma_f32_16x16x32_bf16 v[208:215], v[138:145] /*v[650:657]*/, v[168:175], v[208:215]
	s_set_vgpr_msb 0x2a2
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[530:537]*/, v[54:61] /*v[566:573]*/, v[136:143], v[18:25] /*v[530:537]*/
	s_set_vgpr_msb 0xa252
	v_wmma_f32_16x16x32_bf16 v[130:137] /*v[386:393]*/, v[54:61] /*v[566:573]*/, v[168:175], v[130:137] /*v[386:393]*/
	s_set_vgpr_msb 0x5202
	v_wmma_f32_16x16x32_bf16 v[224:231], v[102:109] /*v[614:621]*/, v[144:151], v[224:231]
	v_wmma_f32_16x16x32_bf16 v[200:207], v[102:109] /*v[614:621]*/, v[176:183], v[200:207]
	v_wmma_f32_16x16x32_bf16 v[232:239], v[146:153] /*v[658:665]*/, v[144:151], v[232:239]
	v_wmma_f32_16x16x32_bf16 v[208:215], v[146:153] /*v[658:665]*/, v[176:183], v[208:215]
	s_set_vgpr_msb 0x252
	v_wmma_f32_16x16x32_bf16 v[122:129] /*v[378:385]*/, v[30:37] /*v[542:549]*/, v[144:151], v[122:129] /*v[378:385]*/
	v_wmma_f32_16x16x32_bf16 v[34:41] /*v[290:297]*/, v[30:37] /*v[542:549]*/, v[176:183], v[34:41] /*v[290:297]*/
	s_set_vgpr_msb 0x52a2
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[530:537]*/, v[62:69] /*v[574:581]*/, v[144:151], v[18:25] /*v[530:537]*/
	s_set_vgpr_msb 0xa252
	v_wmma_f32_16x16x32_bf16 v[130:137] /*v[386:393]*/, v[62:69] /*v[574:581]*/, v[176:183], v[130:137] /*v[386:393]*/
	s_set_vgpr_msb 0x5202
	v_wmma_f32_16x16x32_bf16 v[224:231], v[110:117] /*v[622:629]*/, v[152:159], v[224:231]
	v_wmma_f32_16x16x32_bf16 v[200:207], v[110:117] /*v[622:629]*/, v[184:191], v[200:207]
	v_wmma_f32_16x16x32_bf16 v[232:239], v[154:161] /*v[666:673]*/, v[152:159], v[232:239]
	v_wmma_f32_16x16x32_bf16 v[208:215], v[154:161] /*v[666:673]*/, v[184:191], v[208:215]
	s_set_vgpr_msb 0x252
	v_wmma_f32_16x16x32_bf16 v[122:129] /*v[378:385]*/, v[38:45] /*v[550:557]*/, v[152:159], v[122:129] /*v[378:385]*/
	v_wmma_f32_16x16x32_bf16 v[34:41] /*v[290:297]*/, v[38:45] /*v[550:557]*/, v[184:191], v[34:41] /*v[290:297]*/
	s_set_vgpr_msb 0x52a2
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[18:25] /*v[530:537]*/, v[70:77] /*v[582:589]*/, v[152:159], v[18:25] /*v[530:537]*/
	s_set_vgpr_msb 0xa252
	v_wmma_f32_16x16x32_bf16 v[130:137] /*v[386:393]*/, v[70:77] /*v[582:589]*/, v[184:191], v[130:137] /*v[386:393]*/
	s_set_vgpr_msb 0x5200
	s_cbranch_vccz .LBB0_22
	s_set_vgpr_msb 8
	v_pk_mul_f32 v[126:127], v[126:127], v[134:135] /*v[646:647]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[124:125], v[124:125], v[134:135] /*v[646:647]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[122:123], v[122:123], v[134:135] /*v[646:647]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[120:121], v[120:121], v[134:135] /*v[646:647]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[118:119], v[118:119], v[134:135] /*v[646:647]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[116:117], v[116:117], v[134:135] /*v[646:647]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[114:115], v[114:115], v[134:135] /*v[646:647]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[112:113], v[112:113], v[134:135] /*v[646:647]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[110:111], v[110:111], v[134:135] /*v[646:647]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[108:109], v[108:109], v[134:135] /*v[646:647]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[106:107], v[106:107], v[134:135] /*v[646:647]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[104:105], v[104:105], v[134:135] /*v[646:647]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[102:103], v[102:103], v[134:135] /*v[646:647]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[100:101], v[100:101], v[134:135] /*v[646:647]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[98:99], v[98:99], v[134:135] /*v[646:647]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[96:97], v[96:97], v[134:135] /*v[646:647]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[94:95], v[94:95], v[134:135] /*v[646:647]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[92:93], v[92:93], v[134:135] /*v[646:647]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[90:91], v[90:91], v[134:135] /*v[646:647]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[88:89], v[88:89], v[134:135] /*v[646:647]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[86:87], v[86:87], v[134:135] /*v[646:647]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[84:85], v[84:85], v[134:135] /*v[646:647]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[82:83], v[82:83], v[134:135] /*v[646:647]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[80:81], v[80:81], v[134:135] /*v[646:647]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[78:79], v[78:79], v[134:135] /*v[646:647]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[76:77], v[76:77], v[134:135] /*v[646:647]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[74:75], v[74:75], v[134:135] /*v[646:647]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[72:73], v[72:73], v[134:135] /*v[646:647]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[70:71], v[70:71], v[134:135] /*v[646:647]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[68:69], v[68:69], v[134:135] /*v[646:647]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[66:67], v[66:67], v[134:135] /*v[646:647]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[64:65], v[64:65], v[134:135] /*v[646:647]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x800
.LBB0_22:
	s_set_vgpr_msb 0x4a
	v_sub_f32_e32 v18 /*v274*/, v211 /*v723*/, v221 /*v733*/
	s_and_b32 s2, s3, exec_lo
	s_cselect_b32 s2, 1, 0
	s_cmp_lg_u32 s2, 1
	s_set_vgpr_msb 0x4a44
	v_mul_f32_e32 v18 /*v274*/, 0x3fb8aa3b, v18 /*v274*/
	s_set_vgpr_msb 0x4481
	v_exp_f32_e32 v136 /*v648*/, v18 /*v274*/
	s_set_vgpr_msb 0x8100
	s_cbranch_scc1 .LBB0_24
	v_nop
	s_set_vgpr_msb 8
	v_pk_mul_f32 v[62:63], v[62:63], v[136:137] /*v[648:649]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[60:61], v[60:61], v[136:137] /*v[648:649]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59], v[58:59], v[136:137] /*v[648:649]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[56:57], v[56:57], v[136:137] /*v[648:649]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[54:55], v[54:55], v[136:137] /*v[648:649]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[52:53], v[52:53], v[136:137] /*v[648:649]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[50:51], v[50:51], v[136:137] /*v[648:649]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[48:49], v[48:49], v[136:137] /*v[648:649]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[46:47], v[46:47], v[136:137] /*v[648:649]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[44:45], v[44:45], v[136:137] /*v[648:649]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[42:43], v[42:43], v[136:137] /*v[648:649]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[40:41], v[40:41], v[136:137] /*v[648:649]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39], v[38:39], v[136:137] /*v[648:649]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[36:37], v[36:37], v[136:137] /*v[648:649]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35], v[34:35], v[136:137] /*v[648:649]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[32:33], v[32:33], v[136:137] /*v[648:649]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[30:31], v[30:31], v[136:137] /*v[648:649]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[28:29], v[28:29], v[136:137] /*v[648:649]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[26:27], v[26:27], v[136:137] /*v[648:649]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25], v[24:25], v[136:137] /*v[648:649]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[22:23], v[22:23], v[136:137] /*v[648:649]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21], v[20:21], v[136:137] /*v[648:649]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[18:19], v[18:19], v[136:137] /*v[648:649]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[16:17], v[16:17], v[136:137] /*v[648:649]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[14:15], v[136:137] /*v[648:649]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13], v[12:13], v[136:137] /*v[648:649]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[10:11], v[136:137] /*v[648:649]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[8:9], v[136:137] /*v[648:649]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[6:7], v[136:137] /*v[648:649]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[4:5], v[136:137] /*v[648:649]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[2:3], v[2:3], v[136:137] /*v[648:649]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[0:1], v[0:1], v[136:137] /*v[648:649]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x800
.LBB0_24:
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x42
	ds_load_tr16_b128 v[22:25] /*v[278:281]*/, v217 /*v729*/ offset:4608
	ds_load_tr16_b128 v[46:49] /*v[302:305]*/, v217 /*v729*/ offset:4640
	ds_load_tr16_b128 v[18:21] /*v[274:277]*/, v217 /*v729*/
	ds_load_tr16_b128 v[42:45] /*v[298:301]*/, v217 /*v729*/ offset:32
	s_set_vgpr_msb 0x4285
	v_cvt_pk_bf16_f32 v97 /*v609*/, v204 /*v460*/, v206 /*v462*/
	v_cvt_pk_bf16_f32 v96 /*v608*/, v200 /*v456*/, v202 /*v458*/
	v_cvt_pk_bf16_f32 v95 /*v607*/, v178 /*v434*/, v182 /*v438*/
	v_cvt_pk_bf16_f32 v94 /*v606*/, v156 /*v412*/, v158 /*v414*/
	v_cvt_pk_bf16_f32 v93 /*v605*/, v196 /*v452*/, v198 /*v454*/
	v_cvt_pk_bf16_f32 v92 /*v604*/, v194 /*v450*/, v154 /*v410*/
	v_cvt_pk_bf16_f32 v91 /*v603*/, v190 /*v446*/, v192 /*v448*/
	v_cvt_pk_bf16_f32 v90 /*v602*/, v186 /*v442*/, v188 /*v444*/
	v_cvt_pk_bf16_f32 v121 /*v633*/, v205 /*v461*/, v207 /*v463*/
	v_cvt_pk_bf16_f32 v120 /*v632*/, v201 /*v457*/, v203 /*v459*/
	v_cvt_pk_bf16_f32 v119 /*v631*/, v179 /*v435*/, v183 /*v439*/
	v_cvt_pk_bf16_f32 v118 /*v630*/, v157 /*v413*/, v159 /*v415*/
	v_cvt_pk_bf16_f32 v117 /*v629*/, v197 /*v453*/, v199 /*v455*/
	v_cvt_pk_bf16_f32 v116 /*v628*/, v195 /*v451*/, v155 /*v411*/
	v_cvt_pk_bf16_f32 v115 /*v627*/, v191 /*v447*/, v193 /*v449*/
	v_cvt_pk_bf16_f32 v114 /*v626*/, v187 /*v443*/, v189 /*v445*/
	v_cvt_pk_bf16_f32 v85 /*v597*/, v168 /*v424*/, v208 /*v464*/
	v_cvt_pk_bf16_f32 v84 /*v596*/, v164 /*v420*/, v166 /*v422*/
	v_cvt_pk_bf16_f32 v83 /*v595*/, v184 /*v440*/, v162 /*v418*/
	v_cvt_pk_bf16_f32 v82 /*v594*/, v160 /*v416*/, v180 /*v436*/
	v_cvt_pk_bf16_f32 v109 /*v621*/, v169 /*v425*/, v209 /*v465*/
	v_cvt_pk_bf16_f32 v108 /*v620*/, v165 /*v421*/, v167 /*v423*/
	v_cvt_pk_bf16_f32 v107 /*v619*/, v185 /*v441*/, v163 /*v419*/
	v_cvt_pk_bf16_f32 v106 /*v618*/, v161 /*v417*/, v181 /*v437*/
	s_set_vgpr_msb 0x8509
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x32_bf16 v[120:127], v[18:25] /*v[274:281]*/, v[90:97] /*v[602:609]*/, v[120:127]
	s_set_vgpr_msb 0x985
	v_cvt_pk_bf16_f32 v89 /*v601*/, v222 /*v478*/, v224 /*v480*/
	v_cvt_pk_bf16_f32 v88 /*v600*/, v144 /*v400*/, v220 /*v476*/
	v_cvt_pk_bf16_f32 v87 /*v599*/, v214 /*v470*/, v216 /*v472*/
	v_cvt_pk_bf16_f32 v86 /*v598*/, v210 /*v466*/, v212 /*v468*/
	v_cvt_pk_bf16_f32 v75 /*v587*/, v142 /*v398*/, v218 /*v474*/
	v_cvt_pk_bf16_f32 v113 /*v625*/, v223 /*v479*/, v225 /*v481*/
	v_cvt_pk_bf16_f32 v112 /*v624*/, v145 /*v401*/, v221 /*v477*/
	s_set_vgpr_msb 0x8509
	v_wmma_f32_16x16x32_bf16 v[56:63], v[18:25] /*v[274:281]*/, v[114:121] /*v[626:633]*/, v[56:63]
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x942
	ds_load_tr16_b128 v[18:21] /*v[274:277]*/, v217 /*v729*/ offset:27648
	ds_load_tr16_b128 v[154:157] /*v[410:413]*/, v217 /*v729*/ offset:27680
	ds_load_tr16_b128 v[22:25] /*v[278:281]*/, v217 /*v729*/ offset:32256
	ds_load_tr16_b128 v[158:161] /*v[414:417]*/, v217 /*v729*/ offset:32288
	ds_load_tr16_b128 v[162:165] /*v[418:421]*/, v217 /*v729*/ offset:64
	ds_load_tr16_b128 v[170:173] /*v[426:429]*/, v217 /*v729*/ offset:96
	ds_load_tr16_b128 v[166:169] /*v[422:425]*/, v217 /*v729*/ offset:4672
	ds_load_tr16_b128 v[174:177] /*v[430:433]*/, v217 /*v729*/ offset:4704
	s_set_vgpr_msb 0x4285
	v_cvt_pk_bf16_f32 v111 /*v623*/, v215 /*v471*/, v217 /*v473*/
	v_cvt_pk_bf16_f32 v110 /*v622*/, v211 /*v467*/, v213 /*v469*/
	v_cvt_pk_bf16_f32 v99 /*v611*/, v143 /*v399*/, v219 /*v475*/
	s_set_vgpr_msb 0x8542
	ds_load_tr16_b128 v[178:181] /*v[434:437]*/, v217 /*v729*/ offset:9280
	s_wait_alu depctr_va_vdst(1)
	ds_load_tr16_b128 v[210:213] /*v[466:469]*/, v217 /*v729*/ offset:9312
	ds_load_tr16_b128 v[182:185] /*v[438:441]*/, v217 /*v729*/ offset:13888
	ds_load_tr16_b128 v[214:217] /*v[470:473]*/, v217 /*v729*/ offset:13920
	ds_load_tr16_b128 v[186:189] /*v[442:445]*/, v217 /*v729*/ offset:18496
	s_wait_alu depctr_va_vdst(0)
	ds_load_tr16_b128 v[218:221] /*v[474:477]*/, v217 /*v729*/ offset:18528
	ds_load_tr16_b128 v[190:193] /*v[446:449]*/, v217 /*v729*/ offset:23104
	ds_load_tr16_b128 v[222:225] /*v[478:481]*/, v217 /*v729*/ offset:23136
	s_set_vgpr_msb 0x4285
	v_cvt_pk_bf16_f32 v81 /*v593*/, v244 /*v500*/, v252 /*v508*/
	v_cvt_pk_bf16_f32 v80 /*v592*/, v240 /*v496*/, v242 /*v498*/
	s_set_vgpr_msb 0x8509
	s_wait_dscnt 0x9
	v_wmma_f32_16x16x32_bf16 v[104:111], v[162:169] /*v[418:425]*/, v[90:97] /*v[602:609]*/, v[104:111]
	s_set_vgpr_msb 0x985
	v_cvt_pk_bf16_f32 v79 /*v591*/, v232 /*v488*/, v236 /*v492*/
	v_cvt_pk_bf16_f32 v78 /*v590*/, v120 /*v376*/, v228 /*v484*/
	v_cvt_pk_bf16_f32 v76 /*v588*/, v226 /*v482*/, v114 /*v370*/
	v_cvt_pk_bf16_f32 v68 /*v580*/, v248 /*v504*/, v250 /*v506*/
	v_cvt_pk_bf16_f32 v67 /*v579*/, v238 /*v494*/, v246 /*v502*/
	v_cvt_pk_bf16_f32 v66 /*v578*/, v230 /*v486*/, v234 /*v490*/
	v_cvt_pk_bf16_f32 v105 /*v617*/, v245 /*v501*/, v253 /*v509*/
	s_set_vgpr_msb 0x8509
	v_wmma_f32_16x16x32_bf16 v[40:47], v[162:169] /*v[418:425]*/, v[114:121] /*v[626:633]*/, v[40:47]
	s_set_vgpr_msb 0x985
	v_cvt_pk_bf16_f32 v104 /*v616*/, v241 /*v497*/, v243 /*v499*/
	v_cvt_pk_bf16_f32 v103 /*v615*/, v233 /*v489*/, v237 /*v493*/
	v_cvt_pk_bf16_f32 v102 /*v614*/, v121 /*v377*/, v229 /*v485*/
	v_cvt_pk_bf16_f32 v100 /*v612*/, v227 /*v483*/, v115 /*v371*/
	v_cvt_pk_bf16_f32 v52 /*v564*/, v249 /*v505*/, v251 /*v507*/
	v_cvt_pk_bf16_f32 v51 /*v563*/, v239 /*v495*/, v247 /*v503*/
	v_cvt_pk_bf16_f32 v50 /*v562*/, v231 /*v487*/, v235 /*v491*/
	s_set_vgpr_msb 0x8509
	s_wait_dscnt 0x5
	v_wmma_f32_16x16x32_bf16 v[104:111], v[178:185] /*v[434:441]*/, v[82:89] /*v[594:601]*/, v[104:111]
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x942
	ds_load_tr16_b128 v[226:229] /*v[482:485]*/, v217 /*v729*/ offset:27712
	ds_load_tr16_b128 v[234:237] /*v[490:493]*/, v217 /*v729*/ offset:27744
	ds_load_tr16_b128 v[230:233] /*v[486:489]*/, v217 /*v729*/ offset:32320
	ds_load_tr16_b128 v[238:241] /*v[494:497]*/, v217 /*v729*/ offset:32352
	ds_load_tr16_b128 v[242:245] /*v[498:501]*/, v217 /*v729*/ offset:128
	s_set_vgpr_msb 0x4282
	ds_load_tr16_b128 v[142:145] /*v[654:657]*/, v217 /*v729*/ offset:160
	s_set_vgpr_msb 0x8242
	ds_load_tr16_b128 v[246:249] /*v[502:505]*/, v217 /*v729*/ offset:4736
	s_set_vgpr_msb 0x4282
	ds_load_tr16_b128 v[146:149] /*v[658:661]*/, v217 /*v729*/ offset:4768
	s_set_vgpr_msb 0x8285
	v_cvt_pk_bf16_f32 v77 /*v589*/, v116 /*v372*/, v118 /*v374*/
	v_cvt_pk_bf16_f32 v74 /*v586*/, v138 /*v394*/, v140 /*v396*/
	v_cvt_pk_bf16_f32 v101 /*v613*/, v117 /*v373*/, v119 /*v375*/
	v_cvt_pk_bf16_f32 v98 /*v610*/, v139 /*v395*/, v141 /*v397*/
	s_set_vgpr_msb 0x858a
	v_cvt_pk_bf16_f32 v73 /*v585*/, v6 /*v518*/, v8 /*v520*/
	v_cvt_pk_bf16_f32 v72 /*v584*/, v2 /*v514*/, v4 /*v516*/
	s_set_vgpr_msb 0x8a09
	v_wmma_f32_16x16x32_bf16 v[40:47], v[178:185] /*v[434:441]*/, v[106:113] /*v[618:625]*/, v[40:47]
	s_set_vgpr_msb 0x985
	v_cvt_pk_bf16_f32 v71 /*v583*/, v94 /*v350*/, v96 /*v352*/
	v_cvt_pk_bf16_f32 v70 /*v582*/, v90 /*v346*/, v92 /*v348*/
	s_set_vgpr_msb 0x8589
	v_cvt_pk_bf16_f32 v69 /*v581*/, v254 /*v510*/, v0 /*v512*/
	s_set_vgpr_msb 0x898a
	v_cvt_pk_bf16_f32 v57 /*v569*/, v7 /*v519*/, v9 /*v521*/
	v_cvt_pk_bf16_f32 v56 /*v568*/, v3 /*v515*/, v5 /*v517*/
	s_set_vgpr_msb 0x8a85
	v_cvt_pk_bf16_f32 v55 /*v567*/, v95 /*v351*/, v97 /*v353*/
	v_cvt_pk_bf16_f32 v54 /*v566*/, v91 /*v347*/, v93 /*v349*/
	s_set_vgpr_msb 0x8509
	s_wait_dscnt 0x9
	v_wmma_f32_16x16x32_bf16 v[104:111], v[186:193] /*v[442:449]*/, v[74:81] /*v[586:593]*/, v[104:111]
	s_set_vgpr_msb 0x989
	v_cvt_pk_bf16_f32 v53 /*v565*/, v255 /*v511*/, v1 /*v513*/
	s_set_vgpr_msb 0x8940
	v_max3_num_f32 v162 /*v418*/, v216, v217, v218
	v_max3_num_f32 v163 /*v419*/, v192, v193, v194
	v_max3_num_f32 v164 /*v420*/, v219, v220, v221
	v_max3_num_f32 v165 /*v421*/, v195, v196, v197
	v_max3_num_f32 v166 /*v422*/, v222, v223, v224
	v_max3_num_f32 v167 /*v423*/, v198, v199, v200
	s_set_vgpr_msb 0x4009
	v_wmma_f32_16x16x32_bf16 v[40:47], v[186:193] /*v[442:449]*/, v[98:105] /*v[610:617]*/, v[40:47]
	s_set_vgpr_msb 0x940
	v_max3_num_f32 v168 /*v424*/, v225, v226, v227
	v_max3_num_f32 v178 /*v434*/, v228, v229, v230
	v_max3_num_f32 v180 /*v436*/, v231, v232, v233
	v_max3_num_f32 v182 /*v438*/, v234, v235, v236
	v_max3_num_f32 v184 /*v440*/, v237, v238, v239
	s_set_vgpr_msb 0x40aa
	v_max3_num_f32 v26 /*v538*/, v22 /*v534*/, v23 /*v535*/, v24 /*v536*/
	s_set_vgpr_msb 0xaa55
	v_max3_num_f32 v162 /*v418*/, v162 /*v418*/, v164 /*v420*/, v166 /*v422*/
	s_set_vgpr_msb 0x5509
	v_wmma_f32_16x16x32_bf16 v[96:103], v[170:177] /*v[426:433]*/, v[90:97] /*v[602:609]*/, v[96:103]
	s_set_vgpr_msb 0x955
	v_max3_num_f32 v163 /*v419*/, v163 /*v419*/, v165 /*v421*/, v167 /*v423*/
	v_max3_num_f32 v164 /*v420*/, v168 /*v424*/, v178 /*v434*/, v180 /*v436*/
	s_set_vgpr_msb 0x5540
	v_max3_num_f32 v169 /*v425*/, v201, v202, v203
	v_max3_num_f32 v179 /*v435*/, v204, v205, v206
	v_max3_num_f32 v181 /*v437*/, v207, v208, v209
	v_max3_num_f32 v183 /*v439*/, v210, v211, v212
	v_max3_num_f32 v185 /*v441*/, v213, v214, v215
	s_set_vgpr_msb 0x4009
	v_wmma_f32_16x16x32_bf16 v[32:39], v[170:177] /*v[426:433]*/, v[114:121] /*v[626:633]*/, v[32:39]
	s_set_vgpr_msb 0x995
	v_max3_num_f32 v30 /*v542*/, v134 /*v390*/, v135 /*v391*/, v136 /*v392*/
	s_set_vgpr_msb 0x9555
	v_max3_num_f32 v169 /*v425*/, v169 /*v425*/, v179 /*v435*/, v181 /*v437*/
	s_set_vgpr_msb 0x5542
	ds_load_tr16_b128 v[250:253] /*v[506:509]*/, v217 /*v729*/ offset:9344
	s_set_vgpr_msb 0x4282
	ds_load_tr16_b128 v[224:227] /*v[736:739]*/, v217 /*v729*/ offset:9376
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x8242
	ds_load_tr16_b128 v[254:257] /*v[510:513]*/, v217 /*v729*/ offset:13952
	s_set_vgpr_msb 0x4282
	ds_load_tr16_b128 v[228:231] /*v[740:743]*/, v217 /*v729*/ offset:13984
	ds_load_tr16_b128 v[150:153] /*v[662:665]*/, v217 /*v729*/ offset:18560
	ds_load_tr16_b128 v[232:235] /*v[744:747]*/, v217 /*v729*/ offset:18592
	ds_load_tr16_b128 v[154:157] /*v[666:669]*/, v217 /*v729*/ offset:23168
	ds_load_tr16_b128 v[236:239] /*v[748:751]*/, v217 /*v729*/ offset:23200
	ds_load_tr16_b128 v[158:161] /*v[670:673]*/, v217 /*v729*/ offset:27776
	ds_load_tr16_b128 v[240:243] /*v[752:755]*/, v217 /*v729*/ offset:27808
	ds_load_tr16_b128 v[162:165] /*v[674:677]*/, v217 /*v729*/ offset:32384
	ds_load_tr16_b128 v[244:247] /*v[756:759]*/, v217 /*v729*/ offset:32416
	ds_load_tr16_b128 v[248:251] /*v[760:763]*/, v217 /*v729*/ offset:192
	s_set_vgpr_msb 0x82c2
	ds_load_tr16_b128 v[0:3] /*v[768:771]*/, v217 /*v729*/ offset:224
	s_set_vgpr_msb 0xc282
	ds_load_tr16_b128 v[252:255] /*v[764:767]*/, v217 /*v729*/ offset:4800
	s_set_vgpr_msb 0x82c2
	ds_load_tr16_b128 v[4:7] /*v[772:775]*/, v217 /*v729*/ offset:4832
	s_set_vgpr_msb 0xc242
	ds_load_tr16_b128 v[50:53] /*v[306:309]*/, v217 /*v729*/ offset:9216
	ds_load_tr16_b128 v[138:141] /*v[394:397]*/, v217 /*v729*/ offset:9248
	ds_load_tr16_b128 v[54:57] /*v[310:313]*/, v217 /*v729*/ offset:13824
	ds_load_tr16_b128 v[142:145] /*v[398:401]*/, v217 /*v729*/ offset:13856
	ds_load_tr16_b128 v[58:61] /*v[314:317]*/, v217 /*v729*/ offset:18432
	ds_load_tr16_b128 v[146:149] /*v[402:405]*/, v217 /*v729*/ offset:18464
	ds_load_tr16_b128 v[62:65] /*v[318:321]*/, v217 /*v729*/ offset:23040
	ds_load_tr16_b128 v[150:153] /*v[406:409]*/, v217 /*v729*/ offset:23072
	s_set_vgpr_msb 0x42c2
	ds_load_tr16_b128 v[8:11] /*v[776:779]*/, v217 /*v729*/ offset:9408
	ds_load_tr16_b128 v[16:19] /*v[784:787]*/, v217 /*v729*/ offset:9440
	ds_load_tr16_b128 v[12:15] /*v[780:783]*/, v217 /*v729*/ offset:14016
	ds_load_tr16_b128 v[20:23] /*v[788:791]*/, v217 /*v729*/ offset:14048
	ds_load_tr16_b128 v[24:27] /*v[792:795]*/, v217 /*v729*/ offset:18624
	s_set_vgpr_msb 0xc282
	ds_load_tr16_b128 v[122:125] /*v[634:637]*/, v217 /*v729*/ offset:18656
	s_set_vgpr_msb 0x82c2
	ds_load_tr16_b128 v[28:31] /*v[796:799]*/, v217 /*v729*/ offset:23232
	s_set_vgpr_msb 0xc282
	ds_load_tr16_b128 v[126:129] /*v[638:641]*/, v217 /*v729*/ offset:23264
	s_set_vgpr_msb 0x82c2
	ds_load_tr16_b128 v[32:35] /*v[800:803]*/, v217 /*v729*/ offset:27840
	s_set_vgpr_msb 0xc282
	ds_load_tr16_b128 v[58:61] /*v[570:573]*/, v217 /*v729*/ offset:27872
	s_set_vgpr_msb 0x82c2
	ds_load_tr16_b128 v[36:39] /*v[804:807]*/, v217 /*v729*/ offset:32448
	s_set_vgpr_msb 0xc282
	ds_load_tr16_b128 v[62:65] /*v[574:577]*/, v217 /*v729*/ offset:32480
	s_set_vgpr_msb 0x8242
	ds_load_tr16_b128 v[114:117] /*v[370:373]*/, v218 /*v730*/
	ds_load_tr16_b128 v[66:69] /*v[322:325]*/, v218 /*v730*/ offset:32
	ds_load_tr16_b128 v[118:121] /*v[374:377]*/, v218 /*v730*/ offset:4608
	ds_load_tr16_b128 v[70:73] /*v[326:329]*/, v218 /*v730*/ offset:4640
	s_set_vgpr_msb 0x4209
	s_wait_dscnt 0x2d
	v_wmma_f32_16x16x32_bf16 v[104:111], v[226:233] /*v[482:489]*/, v[66:73] /*v[578:585]*/, v[104:111]
	v_nop
	s_set_vgpr_msb 0x955
	v_max3_num_f32 v171 /*v427*/, v13 /*v269*/, v14 /*v270*/, v15 /*v271*/
	v_max3_num_f32 v173 /*v429*/, v16 /*v272*/, v17 /*v273*/, v26 /*v282*/
	v_max3_num_f32 v175 /*v431*/, v27 /*v283*/, v28 /*v284*/, v29 /*v285*/
	v_max3_num_f32 v177 /*v433*/, v30 /*v286*/, v31 /*v287*/, v32 /*v288*/
	s_set_vgpr_msb 0x5540
	v_max3_num_f32 v170 /*v426*/, v240, v241, v242
	v_max3_num_f32 v172 /*v428*/, v243, v244, v245
	s_set_vgpr_msb 0x4009
	v_wmma_f32_16x16x32_bf16 v[40:47], v[226:233] /*v[482:489]*/, v[50:57] /*v[562:569]*/, v[40:47]
	s_set_vgpr_msb 0x955
	v_max3_num_f32 v166 /*v422*/, v171 /*v427*/, v173 /*v429*/, v175 /*v431*/
	s_set_vgpr_msb 0x5540
	v_max3_num_f32 v174 /*v430*/, v246, v247, v248
	v_max3_num_f32 v176 /*v432*/, v249, v250, v251
	v_nop
	s_set_vgpr_msb 0x4055
	v_max3_num_f32 v226 /*v482*/, v10 /*v266*/, v11 /*v267*/, v12 /*v268*/
	v_max3_num_f32 v228 /*v484*/, v33 /*v289*/, v98 /*v354*/, v99 /*v355*/
	v_max3_num_f32 v230 /*v486*/, v100 /*v356*/, v101 /*v357*/, v102 /*v358*/
	s_set_vgpr_msb 0x5509
	v_wmma_f32_16x16x32_bf16 v[96:103], v[210:217] /*v[466:473]*/, v[82:89] /*v[594:601]*/, v[96:103]
	s_set_vgpr_msb 0x955
	v_max3_num_f32 v232 /*v488*/, v103 /*v359*/, v104 /*v360*/, v105 /*v361*/
	v_max3_num_f32 v165 /*v421*/, v182 /*v438*/, v184 /*v440*/, v226 /*v482*/
	s_set_vgpr_msb 0x5540
	v_max3_num_f32 v227 /*v483*/, v252, v253, v254
	s_set_vgpr_msb 0x4055
	v_max3_num_f32 v167 /*v423*/, v177 /*v433*/, v228 /*v484*/, v230 /*v486*/
	s_set_vgpr_msb 0x5554
	v_max3_num_f32 v229 /*v485*/, v255, v2 /*v258*/, v3 /*v259*/
	s_set_vgpr_msb 0x5455
	v_max3_num_f32 v231 /*v487*/, v4 /*v260*/, v5 /*v261*/, v6 /*v262*/
	v_max3_num_f32 v233 /*v489*/, v131 /*v387*/, v132 /*v388*/, v133 /*v389*/
	s_set_vgpr_msb 0x5509
	v_wmma_f32_16x16x32_bf16 v[32:39], v[210:217] /*v[466:473]*/, v[106:113] /*v[618:625]*/, v[32:39]
	s_set_vgpr_msb 0x955
	v_max3_num_f32 v162 /*v418*/, v162 /*v418*/, v164 /*v420*/, v165 /*v421*/
	v_nop
	v_nop
	v_nop
	v_max3_num_f32 v211 /*v467*/, v122 /*v378*/, v123 /*v379*/, v124 /*v380*/
	v_max3_num_f32 v213 /*v469*/, v125 /*v381*/, v126 /*v382*/, v127 /*v383*/
	v_max_num_f32_e32 v215 /*v471*/, v128 /*v384*/, v129 /*v385*/
	s_set_vgpr_msb 0x556a
	v_max3_num_f32 v217 /*v473*/, v19 /*v531*/, v20 /*v532*/, v21 /*v533*/
	s_set_vgpr_msb 0x6a55
	v_max3_num_f32 v210 /*v466*/, v7 /*v263*/, v8 /*v264*/, v9 /*v265*/
	v_max3_num_f32 v212 /*v468*/, v34 /*v290*/, v35 /*v291*/, v36 /*v292*/
	v_max3_num_f32 v168 /*v424*/, v232 /*v488*/, v211 /*v467*/, v213 /*v469*/
	v_max3_num_f32 v214 /*v470*/, v37 /*v293*/, v38 /*v294*/, v39 /*v295*/
	s_set_vgpr_msb 0x5559
	v_max3_num_f32 v171 /*v427*/, v215 /*v471*/, v18 /*v530*/, v217 /*v473*/
	s_set_vgpr_msb 0x5945
	v_max_num_f32_e32 v216 /*v472*/, v40 /*v296*/, v41 /*v297*/
	s_set_vgpr_msb 0x4509
	s_wait_dscnt 0x29
	v_wmma_f32_16x16x32_bf16 v[88:95], v[242:249] /*v[498:505]*/, v[90:97] /*v[602:609]*/, v[88:95]
	s_set_vgpr_msb 0x955
	v_max3_num_f32 v164 /*v420*/, v166 /*v422*/, v167 /*v423*/, v168 /*v424*/
	v_max3_num_f32 v166 /*v422*/, v183 /*v439*/, v185 /*v441*/, v170 /*v426*/
	s_set_vgpr_msb 0x5569
	v_max3_num_f32 v165 /*v421*/, v171 /*v427*/, v26 /*v538*/, v25 /*v537*/
	s_set_vgpr_msb 0x6955
	v_max3_num_f32 v167 /*v423*/, v172 /*v428*/, v174 /*v430*/, v176 /*v432*/
	v_max3_num_f32 v168 /*v424*/, v227 /*v483*/, v229 /*v485*/, v231 /*v487*/
	v_max3_num_f32 v170 /*v426*/, v210 /*v466*/, v212 /*v468*/, v214 /*v470*/
	v_max3_num_f32 v163 /*v419*/, v163 /*v419*/, v169 /*v425*/, v166 /*v422*/
	v_max3_num_f32 v162 /*v418*/, v162 /*v418*/, v164 /*v420*/, v165 /*v421*/
	v_max3_num_f32 v164 /*v420*/, v216 /*v472*/, v130 /*v386*/, v233 /*v489*/
	s_set_vgpr_msb 0x5509
	v_wmma_f32_16x16x32_bf16 v[24:31], v[242:249] /*v[498:505]*/, v[114:121] /*v[626:633]*/, v[24:31]
	s_set_vgpr_msb 0x955
	v_max3_num_f32 v166 /*v422*/, v167 /*v423*/, v168 /*v424*/, v170 /*v426*/
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x5582
	ds_load_tr16_b128 v[26:29] /*v[538:541]*/, v218 /*v730*/ offset:9344
	s_set_vgpr_msb 0x8242
	ds_load_tr16_b128 v[210:213] /*v[466:469]*/, v218 /*v730*/ offset:9376
	s_set_vgpr_msb 0x4259
	v_mov_b32_e32 v165 /*v421*/, v162 /*v418*/
	v_max3_num_f32 v164 /*v420*/, v164 /*v420*/, v30 /*v542*/, v137 /*v393*/
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x5982
	ds_load_tr16_b128 v[30:33] /*v[542:545]*/, v218 /*v730*/ offset:13952
	s_set_vgpr_msb 0x8242
	ds_load_tr16_b128 v[214:217] /*v[470:473]*/, v218 /*v730*/ offset:13984
	s_set_vgpr_msb 0x4282
	ds_load_tr16_b128 v[46:49] /*v[558:561]*/, v218 /*v730*/ offset:32384
	s_set_vgpr_msb 0x8242
	ds_load_tr16_b128 v[246:249] /*v[502:505]*/, v218 /*v730*/ offset:32416
	s_set_vgpr_msb 0x4255
	v_permlanex16_b32 v165 /*v421*/, v165 /*v421*/, s16, 0xfedcba98
	v_max3_num_f32 v163 /*v419*/, v163 /*v419*/, v166 /*v422*/, v164 /*v420*/
	s_set_vgpr_msb 0x5509
	s_wait_dscnt 0x2b
	v_wmma_f32_16x16x32_bf16 v[88:95], v[250:257] /*v[506:513]*/, v[82:89] /*v[594:601]*/, v[88:95]
	s_set_vgpr_msb 0x945
	v_dual_max_num_f32 v166 /*v422*/, v162 /*v418*/, v165 /*v421*/ :: v_dual_mov_b32 v162 /*v418*/, v163 /*v419*/
	v_permlanex16_b32 v162 /*v418*/, v162 /*v418*/, s16, 0xfedcba98
	s_set_vgpr_msb 0x4509
	v_wmma_f32_16x16x32_bf16 v[24:31], v[250:257] /*v[506:513]*/, v[106:113] /*v[618:625]*/, v[24:31]
	s_set_vgpr_msb 0x945
	v_max_num_f32_e32 v174 /*v430*/, v163 /*v419*/, v162 /*v418*/
	s_set_vgpr_msb 0x4549
	v_sub_f32_e32 v164 /*v420*/, v166 /*v422*/, v222 /*v734*/
	v_max_num_f32_e32 v170 /*v426*/, v166 /*v422*/, v222 /*v734*/
	s_set_vgpr_msb 0x4942
	ds_load_tr16_b128 v[230:233] /*v[486:489]*/, v218 /*v730*/ offset:4800
	s_wait_alu depctr_va_vdst(0)
	ds_load_tr16_b128 v[166:169] /*v[422:425]*/, v218 /*v730*/ offset:4832
	s_set_vgpr_msb 0x420a
	s_wait_dscnt 0x29
	v_wmma_f32_16x16x32_bf16 v[88:95], v[150:157] /*v[662:669]*/, v[74:81] /*v[586:593]*/, v[88:95]
	s_set_vgpr_msb 0xa49
	v_sub_f32_e32 v171 /*v427*/, v174 /*v430*/, v221 /*v733*/
	s_set_vgpr_msb 0x4946
	v_cmp_lt_f32_e32 vcc_lo, 0x41000000, v164 /*v420*/
	v_max_num_f32_e32 v178 /*v434*/, v221 /*v733*/, v174 /*v430*/
	ds_load_tr16_b128 v[226:229] /*v[482:485]*/, v218 /*v730*/ offset:192
	s_wait_alu depctr_va_vdst(0)
	ds_load_tr16_b128 v[162:165] /*v[418:421]*/, v218 /*v730*/ offset:224
	s_cmp_eq_u32 vcc_lo, 0
	s_set_vgpr_msb 0x460a
	v_wmma_f32_16x16x32_bf16 v[24:31], v[150:157] /*v[662:669]*/, v[98:105] /*v[610:617]*/, v[24:31]
	s_cselect_b32 s2, -1, 0
	s_set_vgpr_msb 0xa89
	v_cndmask_b32_e64 v209 /*v721*/, v170 /*v426*/, v222 /*v734*/, s2
	s_set_vgpr_msb 0x8904
	v_cmp_lt_f32_e64 s2, 0x41000000, v171 /*v427*/
	s_set_vgpr_msb 0x448
	v_mul_f32_e32 v182 /*v438*/, 0xbfb8aa3b, v209 /*v721*/
	s_cmp_lg_u32 s2, 0
	s_set_vgpr_msb 0x480a
	s_wait_dscnt 0x27
	v_wmma_f32_16x16x32_bf16 v[88:95], v[158:165] /*v[670:677]*/, v[66:73] /*v[578:585]*/, v[88:95]
	s_cselect_b32 s3, -1, 0
	s_cmp_eq_u32 s2, 0
	s_set_vgpr_msb 0xa10
	v_pk_fma_f32 v[216:217], v[216:217], s[14:15], v[182:183] /*v[438:439]*/ op_sel_hi:[1,0,0]
	s_cselect_b32 s2, -1, 0
	s_set_vgpr_msb 0x1050
	v_pk_fma_f32 v[184:185] /*v[440:441]*/, v[218:219], s[14:15], v[182:183] /*v[438:439]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5089
	v_cndmask_b32_e64 v211 /*v723*/, v178 /*v434*/, v221 /*v733*/, s2
	s_set_vgpr_msb 0x8910
	v_pk_fma_f32 v[224:225], v[224:225], s[14:15], v[182:183] /*v[438:439]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v218, v217
	s_set_vgpr_msb 0x100a
	v_wmma_f32_16x16x32_bf16 v[24:31], v[158:165] /*v[670:677]*/, v[50:57] /*v[562:569]*/, v[24:31]
	s_set_vgpr_msb 0xa10
	v_pk_fma_f32 v[226:227], v[226:227], s[14:15], v[182:183] /*v[438:439]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1048
	v_mul_f32_e32 v254 /*v510*/, 0xbfb8aa3b, v211 /*v723*/
	s_set_vgpr_msb 0x4880
	v_exp_f32_e32 v168 /*v680*/, v225
	s_set_vgpr_msb 0x8010
	v_pk_fma_f32 v[228:229], v[228:229], s[14:15], v[182:183] /*v[438:439]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1080
	v_exp_f32_e32 v164 /*v676*/, v224
	v_exp_f32_e32 v172 /*v684*/, v226
	s_set_vgpr_msb 0x8010
	v_pk_fma_f32 v[192:193], v[192:193], s[14:15], v[254:255] /*v[510:511]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1080
	v_exp_f32_e32 v176 /*v688*/, v227
	s_set_vgpr_msb 0x8010
	v_pk_fma_f32 v[224:225], v[230:231], s[14:15], v[182:183] /*v[438:439]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[226:227], v[232:233], s[14:15], v[182:183] /*v[438:439]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[194:195], v[194:195], s[14:15], v[254:255] /*v[510:511]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v217, v192
	v_exp_f32_e32 v219, v193
	v_nop
	v_pk_fma_f32 v[192:193], v[198:199], s[14:15], v[254:255] /*v[510:511]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1080
	v_exp_f32_e32 v188 /*v700*/, v224
	v_exp_f32_e32 v192 /*v704*/, v225
	s_set_vgpr_msb 0x8010
	v_exp_f32_e32 v224, v226
	v_pk_fma_f32 v[232:233], v[236:237], s[14:15], v[182:183] /*v[438:439]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1080
	v_exp_f32_e32 v157 /*v669*/, v192
	v_exp_f32_e32 v159 /*v671*/, v193
	v_nop
	s_set_vgpr_msb 0x8010
	v_pk_fma_f32 v[192:193], v[202:203], s[14:15], v[254:255] /*v[510:511]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v226, v227
	v_pk_fma_f32 v[196:197], v[196:197], s[14:15], v[254:255] /*v[510:511]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1090
	v_pk_fma_f32 v[0:1] /*v[512:513]*/, v[220:221], s[14:15], v[182:183] /*v[438:439]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[150:151] /*v[662:663]*/, v[222:223], s[14:15], v[182:183] /*v[438:439]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v173 /*v685*/, v192
	v_exp_f32_e32 v177 /*v689*/, v193
	v_nop
	s_set_vgpr_msb 0x9010
	v_pk_fma_f32 v[192:193], v[208:209], s[14:15], v[254:255] /*v[510:511]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v221, v194
	v_exp_f32_e32 v223, v195
	v_nop
	v_pk_fma_f32 v[194:195], v[200:201], s[14:15], v[254:255] /*v[510:511]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1080
	v_exp_f32_e32 v180 /*v692*/, v228
	s_set_vgpr_msb 0x8010
	v_exp_f32_e32 v225, v192
	v_exp_f32_e32 v227, v193
	v_nop
	v_pk_fma_f32 v[192:193], v[212:213], s[14:15], v[254:255] /*v[510:511]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1080
	v_exp_f32_e32 v184 /*v696*/, v229
	v_nop
	s_set_vgpr_msb 0x8010
	v_pk_fma_f32 v[228:229], v[234:235], s[14:15], v[182:183] /*v[438:439]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v234, v233
	s_set_vgpr_msb 0x1080
	v_exp_f32_e32 v139 /*v651*/, v196
	v_exp_f32_e32 v141 /*v653*/, v197
	v_nop
	s_set_vgpr_msb 0x8010
	v_pk_fma_f32 v[196:197], v[206:207], s[14:15], v[254:255] /*v[510:511]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v233, v192
	v_exp_f32_e32 v235, v193
	v_nop
	v_pk_fma_f32 v[192:193], v[240:241], s[14:15], v[254:255] /*v[510:511]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1080
	v_exp_f32_e32 v165 /*v677*/, v194
	v_exp_f32_e32 v169 /*v681*/, v195
	v_nop
	s_set_vgpr_msb 0x8010
	v_pk_fma_f32 v[194:195], v[204:205], s[14:15], v[254:255] /*v[510:511]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x100a
	v_wmma_f32_16x16x32_bf16 v[80:87], v[142:149] /*v[654:661]*/, v[90:97] /*v[602:609]*/, v[80:87]
	s_set_vgpr_msb 0xa51
	v_pk_fma_f32 v[12:13] /*v[268:269]*/, v[12:13] /*v[268:269]*/, s[14:15], v[182:183] /*v[438:439]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5180
	v_exp_f32_e32 v189 /*v701*/, v196
	v_exp_f32_e32 v193 /*v705*/, v197
	v_nop
	s_set_vgpr_msb 0x8010
	v_pk_fma_f32 v[196:197], v[214:215], s[14:15], v[254:255] /*v[510:511]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[236:237], v[238:239], s[14:15], v[182:183] /*v[438:439]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1080
	v_exp_f32_e32 v181 /*v693*/, v194
	v_exp_f32_e32 v185 /*v697*/, v195
	s_set_vgpr_msb 0x800a
	v_wmma_f32_16x16x32_bf16 v[16:23], v[142:149] /*v[654:661]*/, v[114:121] /*v[626:633]*/, v[16:23]
	s_set_vgpr_msb 0xa10
	v_pk_fma_f32 v[194:195], v[210:211], s[14:15], v[254:255] /*v[510:511]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1082
	v_exp_f32_e32 v156 /*v668*/, v150 /*v662*/
	s_set_vgpr_msb 0x8211
	v_pk_fma_f32 v[238:239], v[10:11] /*v[266:267]*/, s[14:15], v[182:183] /*v[438:439]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1181
	v_exp_f32_e32 v150 /*v662*/, v13 /*v269*/
	s_set_vgpr_msb 0x8180
	v_exp_f32_e32 v145 /*v657*/, v192
	v_exp_f32_e32 v147 /*v659*/, v193
	v_nop
	s_set_vgpr_msb 0x8010
	v_pk_fma_f32 v[192:193], v[246:247], s[14:15], v[254:255] /*v[510:511]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1081
	v_exp_f32_e32 v148 /*v660*/, v12 /*v268*/
	v_nop
	s_set_vgpr_msb 0x8151
	v_pk_fma_f32 v[12:13] /*v[268:269]*/, v[28:29] /*v[284:285]*/, s[14:15], v[182:183] /*v[438:439]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5140
	v_exp_f32_e32 v11 /*v267*/, v196
	s_set_vgpr_msb 0x4080
	v_exp_f32_e32 v143 /*v655*/, v197
	v_nop
	s_set_vgpr_msb 0x8010
	v_pk_fma_f32 v[196:197], v[244:245], s[14:15], v[254:255] /*v[510:511]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1080
	v_exp_f32_e32 v161 /*v673*/, v192
	v_exp_f32_e32 v167 /*v679*/, v193
	v_nop
	s_set_vgpr_msb 0x8010
	v_pk_fma_f32 v[192:193], v[250:251], s[14:15], v[254:255] /*v[510:511]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v230, v229
	s_set_vgpr_msb 0x1040
	v_exp_f32_e32 v10 /*v266*/, v236
	s_set_vgpr_msb 0x4080
	v_exp_f32_e32 v142 /*v654*/, v237
	v_nop
	s_set_vgpr_msb 0x8011
	v_pk_fma_f32 v[236:237], v[14:15] /*v[270:271]*/, s[14:15], v[182:183] /*v[438:439]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1110
	v_exp_f32_e32 v229, v194
	v_exp_f32_e32 v231, v195
	v_nop
	v_pk_fma_f32 v[194:195], v[242:243], s[14:15], v[254:255] /*v[510:511]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1080
	v_exp_f32_e32 v144 /*v656*/, v238
	v_exp_f32_e32 v146 /*v658*/, v239
	v_nop
	s_set_vgpr_msb 0x8011
	v_pk_fma_f32 v[238:239], v[16:17] /*v[272:273]*/, s[14:15], v[182:183] /*v[438:439]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1151
	v_pk_fma_f32 v[16:17] /*v[272:273]*/, v[30:31] /*v[286:287]*/, s[14:15], v[182:183] /*v[438:439]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v14 /*v270*/, v13 /*v269*/
	v_pk_fma_f32 v[28:29] /*v[284:285]*/, v[32:33] /*v[288:289]*/, s[14:15], v[182:183] /*v[438:439]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5180
	v_exp_f32_e32 v153 /*v665*/, v196
	v_exp_f32_e32 v155 /*v667*/, v197
	v_nop
	s_set_vgpr_msb 0x8010
	v_pk_fma_f32 v[196:197], v[252:253], s[14:15], v[254:255] /*v[510:511]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1040
	v_exp_f32_e32 v13 /*v269*/, v192
	v_exp_f32_e32 v15 /*v271*/, v193
	v_nop
	s_set_vgpr_msb 0x4010
	v_pk_fma_f32 v[192:193], v[254:255], s[14:15], v[254:255] /*v[510:511]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1082
	v_exp_f32_e32 v158 /*v670*/, v151 /*v663*/
	s_set_vgpr_msb 0x8280
	v_exp_f32_e32 v152 /*v664*/, v236
	v_exp_f32_e32 v154 /*v666*/, v237
	v_nop
	s_set_vgpr_msb 0x8011
	v_pk_fma_f32 v[236:237], v[26:27] /*v[282:283]*/, s[14:15], v[182:183] /*v[438:439]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1180
	v_exp_f32_e32 v149 /*v661*/, v194
	v_exp_f32_e32 v151 /*v663*/, v195
	v_nop
	s_set_vgpr_msb 0x8010
	v_pk_fma_f32 v[194:195], v[248:249], s[14:15], v[254:255] /*v[510:511]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1051
	v_pk_fma_f32 v[98:99] /*v[354:355]*/, v[98:99] /*v[354:355]*/, s[14:15], v[182:183] /*v[438:439]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v26 /*v282*/, v17 /*v273*/
	v_exp_f32_e32 v30 /*v286*/, v28 /*v284*/
	v_exp_f32_e32 v32 /*v288*/, v29 /*v285*/
	v_nop
	v_pk_fma_f32 v[28:29] /*v[284:285]*/, v[100:101] /*v[356:357]*/, s[14:15], v[182:183] /*v[438:439]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[100:101] /*v[356:357]*/, v[102:103] /*v[358:359]*/, s[14:15], v[182:183] /*v[438:439]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5140
	v_exp_f32_e32 v17 /*v273*/, v196
	v_exp_f32_e32 v27 /*v283*/, v197
	v_exp_f32_e32 v31 /*v287*/, v192
	s_set_vgpr_msb 0x4011
	v_pk_fma_f32 v[196:197], v[4:5] /*v[260:261]*/, s[14:15], v[254:255] /*v[510:511]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1140
	v_exp_f32_e32 v33 /*v289*/, v193
	v_nop
	s_set_vgpr_msb 0x4011
	v_pk_fma_f32 v[192:193], v[6:7] /*v[262:263]*/, s[14:15], v[254:255] /*v[510:511]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1180
	v_exp_f32_e32 v160 /*v672*/, v238
	v_exp_f32_e32 v166 /*v678*/, v239
	s_set_vgpr_msb 0x8000
	v_exp_f32_e32 v238, v237
	v_exp_f32_e32 v237, v194
	v_exp_f32_e32 v239, v195
	v_nop
	s_set_vgpr_msb 17
	v_pk_fma_f32 v[194:195], v[2:3] /*v[258:259]*/, s[14:15], v[254:255] /*v[510:511]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1181
	v_exp_f32_e32 v162 /*v674*/, v98 /*v354*/
	v_exp_f32_e32 v170 /*v682*/, v99 /*v355*/
	v_exp_f32_e32 v174 /*v686*/, v28 /*v284*/
	v_exp_f32_e32 v178 /*v690*/, v29 /*v285*/
	v_nop
	s_set_vgpr_msb 0x8151
	v_pk_fma_f32 v[28:29] /*v[284:285]*/, v[104:105] /*v[360:361]*/, s[14:15], v[182:183] /*v[438:439]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5181
	v_exp_f32_e32 v182 /*v694*/, v100 /*v356*/
	s_set_vgpr_msb 0x8151
	v_pk_fma_f32 v[98:99] /*v[354:355]*/, v[122:123] /*v[378:379]*/, s[14:15], v[182:183] /*v[438:439]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5181
	v_exp_f32_e32 v186 /*v698*/, v101 /*v357*/
	v_nop
	s_set_vgpr_msb 0x8151
	v_pk_fma_f32 v[100:101] /*v[356:357]*/, v[124:125] /*v[380:381]*/, s[14:15], v[182:183] /*v[438:439]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5180
	v_exp_f32_e32 v175 /*v687*/, v196
	v_exp_f32_e32 v179 /*v691*/, v197
	v_exp_f32_e32 v183 /*v695*/, v192
	v_exp_f32_e32 v187 /*v699*/, v193
	v_nop
	s_set_vgpr_msb 0x8011
	v_pk_fma_f32 v[192:193], v[34:35] /*v[290:291]*/, s[14:15], v[254:255] /*v[510:511]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[196:197], v[36:37] /*v[292:293]*/, s[14:15], v[254:255] /*v[510:511]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1100
	v_exp_f32_e32 v216, v216
	s_set_vgpr_msb 1
	v_exp_f32_e32 v222, v185 /*v441*/
	s_set_vgpr_msb 0x182
	v_exp_f32_e32 v138 /*v650*/, v0 /*v512*/
	s_set_vgpr_msb 0x8280
	v_exp_f32_e32 v163 /*v675*/, v194
	v_exp_f32_e32 v171 /*v683*/, v195
	v_nop
	s_set_vgpr_msb 0x8011
	v_pk_fma_f32 v[194:195], v[8:9] /*v[264:265]*/, s[14:15], v[254:255] /*v[510:511]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1181
	v_exp_f32_e32 v190 /*v702*/, v28 /*v284*/
	v_exp_f32_e32 v194 /*v706*/, v29 /*v285*/
	s_set_vgpr_msb 0x8151
	v_exp_f32_e32 v28 /*v284*/, v98 /*v354*/
	v_exp_f32_e32 v98 /*v354*/, v99 /*v355*/
	v_exp_f32_e32 v102 /*v358*/, v101 /*v357*/
	v_pk_fma_f32 v[124:125] /*v[380:381]*/, v[128:129] /*v[384:385]*/, s[14:15], v[182:183] /*v[438:439]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5152
	v_pk_fma_f32 v[128:129] /*v[384:385]*/, v[18:19] /*v[530:531]*/, s[14:15], v[182:183] /*v[438:439]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5240
	v_exp_f32_e32 v29 /*v285*/, v192
	v_exp_f32_e32 v99 /*v355*/, v193
	v_exp_f32_e32 v101 /*v357*/, v196
	v_exp_f32_e32 v103 /*v359*/, v197
	s_set_vgpr_msb 0x4011
	v_pk_fma_f32 v[192:193], v[40:41] /*v[296:297]*/, s[14:15], v[254:255] /*v[510:511]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[196:197], v[130:131] /*v[386:387]*/, s[14:15], v[254:255] /*v[510:511]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v220, v184 /*v440*/
	s_set_vgpr_msb 0x1182
	v_exp_f32_e32 v140 /*v652*/, v1 /*v513*/
	s_set_vgpr_msb 0x8251
	v_exp_f32_e32 v16 /*v272*/, v16 /*v272*/
	v_pk_fma_f32 v[104:105] /*v[360:361]*/, v[126:127] /*v[382:383]*/, s[14:15], v[182:183] /*v[438:439]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5180
	v_exp_f32_e32 v191 /*v703*/, v194
	v_exp_f32_e32 v195 /*v707*/, v195
	v_nop
	s_set_vgpr_msb 0x8011
	v_pk_fma_f32 v[194:195], v[38:39] /*v[294:295]*/, s[14:15], v[254:255] /*v[510:511]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1141
	v_exp_f32_e32 v126 /*v382*/, v125 /*v381*/
	s_set_vgpr_msb 0x4181
	v_exp_f32_e32 v18 /*v530*/, v129 /*v385*/
	s_set_vgpr_msb 0x8140
	v_exp_f32_e32 v125 /*v381*/, v192
	v_exp_f32_e32 v127 /*v383*/, v193
	v_exp_f32_e32 v129 /*v385*/, v196
	s_set_vgpr_msb 0x4080
	v_exp_f32_e32 v19 /*v531*/, v197
	s_set_vgpr_msb 0x8011
	v_pk_fma_f32 v[192:193], v[134:135] /*v[390:391]*/, s[14:15], v[254:255] /*v[510:511]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1100
	v_pk_add_f32 v[196:197], v[216:217], v[218:219]
	s_set_vgpr_msb 8
	v_pk_add_f32 v[198:199], v[222:223], v[138:139] /*v[650:651]*/
	v_exp_f32_e32 v228, v228
	v_exp_f32_e32 v236, v236
	s_set_vgpr_msb 0x841
	v_exp_f32_e32 v12 /*v268*/, v12 /*v268*/
	v_exp_f32_e32 v122 /*v378*/, v105 /*v361*/
	s_set_vgpr_msb 0x4152
	v_pk_fma_f32 v[184:185] /*v[440:441]*/, v[20:21] /*v[532:533]*/, s[14:15], v[182:183] /*v[438:439]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5240
	v_exp_f32_e32 v105 /*v361*/, v194
	v_exp_f32_e32 v123 /*v379*/, v195
	v_nop
	s_set_vgpr_msb 0x4011
	v_pk_fma_f32 v[194:195], v[132:133] /*v[388:389]*/, s[14:15], v[254:255] /*v[510:511]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1109
	v_wmma_f32_16x16x32_bf16 v[112:119], v[42:49] /*v[298:305]*/, v[90:97] /*v[602:609]*/, v[112:119]
	s_set_vgpr_msb 0x992
	v_pk_fma_f32 v[0:1] /*v[512:513]*/, v[22:23] /*v[534:535]*/, s[14:15], v[182:183] /*v[438:439]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x9252
	v_pk_fma_f32 v[182:183] /*v[438:439]*/, v[24:25] /*v[536:537]*/, s[14:15], v[182:183] /*v[438:439]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5280
	v_exp_f32_e32 v25 /*v537*/, v192
	v_exp_f32_e32 v197 /*v709*/, v193
	v_nop
	s_set_vgpr_msb 0x8000
	v_pk_add_f32 v[192:193], v[220:221], v[196:197]
	s_set_vgpr_msb 2
	v_pk_add_f32 v[196:197], v[140:141] /*v[652:653]*/, v[198:199]
	s_set_vgpr_msb 0x20a
	v_pk_add_f32 v[198:199], v[156:157] /*v[668:669]*/, v[158:159] /*v[670:671]*/
	s_set_vgpr_msb 0xa09
	v_wmma_f32_16x16x32_bf16 v[48:55], v[42:49] /*v[298:305]*/, v[114:121] /*v[626:633]*/, v[48:55]
	s_set_vgpr_msb 0x90a
	v_pk_add_f32 v[200:201], v[168:169] /*v[680:681]*/, v[172:173] /*v[684:685]*/
	v_pk_add_f32 v[202:203], v[180:181] /*v[692:693]*/, v[184:185] /*v[696:697]*/
	v_pk_add_f32 v[212:213], v[150:151] /*v[662:663]*/, v[152:153] /*v[664:665]*/
	v_pk_add_f32 v[214:215], v[160:161] /*v[672:673]*/, v[166:167] /*v[678:679]*/
	s_set_vgpr_msb 0xa05
	v_pk_add_f32 v[242:243], v[16:17] /*v[272:273]*/, v[26:27] /*v[282:283]*/
	s_set_vgpr_msb 0x509
	v_pk_add_f32 v[244:245], v[32:33] /*v[288:289]*/, v[162:163] /*v[674:675]*/
	s_set_vgpr_msb 0x900
	v_exp_f32_e32 v232, v232
	s_set_vgpr_msb 0x41
	v_exp_f32_e32 v100 /*v356*/, v100 /*v356*/
	v_exp_f32_e32 v104 /*v360*/, v104 /*v360*/
	v_exp_f32_e32 v124 /*v380*/, v124 /*v380*/
	s_set_vgpr_msb 0x4181
	v_exp_f32_e32 v20 /*v532*/, v184 /*v440*/
	s_set_vgpr_msb 0x8180
	v_exp_f32_e32 v21 /*v533*/, v194
	s_set_vgpr_msb 0x8002
	v_pk_add_f32 v[204:205], v[192:193] /*v[704:705]*/, v[224:225]
	s_set_vgpr_msb 0x200
	v_pk_add_f32 v[206:207], v[228:229], v[230:231]
	s_set_vgpr_msb 2
	v_pk_add_f32 v[198:199], v[164:165] /*v[676:677]*/, v[198:199]
	v_pk_add_f32 v[200:201], v[176:177] /*v[688:689]*/, v[200:201]
	v_pk_add_f32 v[202:203], v[188:189] /*v[700:701]*/, v[202:203]
	s_set_vgpr_msb 0x204
	v_pk_add_f32 v[208:209], v[234:235], v[10:11] /*v[266:267]*/
	v_pk_add_f32 v[240:241], v[238:239], v[12:13] /*v[268:269]*/
	s_set_vgpr_msb 0x402
	v_pk_add_f32 v[212:213], v[154:155] /*v[666:667]*/, v[212:213]
	s_set_vgpr_msb 0x200
	v_pk_add_f32 v[214:215], v[236:237], v[214:215]
	s_set_vgpr_msb 10
	v_pk_add_f32 v[246:247], v[174:175] /*v[686:687]*/, v[178:179] /*v[690:691]*/
	v_pk_add_f32 v[248:249], v[186:187] /*v[698:699]*/, v[190:191] /*v[702:703]*/
	s_set_vgpr_msb 0xa05
	v_pk_add_f32 v[250:251], v[28:29] /*v[284:285]*/, v[98:99] /*v[354:355]*/
	s_set_vgpr_msb 0x501
	v_pk_add_f32 v[242:243], v[30:31] /*v[286:287]*/, v[242:243]
	s_set_vgpr_msb 0x102
	v_pk_add_f32 v[244:245], v[170:171] /*v[682:683]*/, v[244:245]
	s_set_vgpr_msb 0x200
	v_pk_add_f32 v[192:193], v[192:193], v[196:197]
	s_set_vgpr_msb 0x41
	v_exp_f32_e32 v128 /*v384*/, v128 /*v384*/
	s_set_vgpr_msb 0x4181
	v_exp_f32_e32 v22 /*v534*/, v185 /*v441*/
	s_set_vgpr_msb 0x8182
	v_exp_f32_e32 v24 /*v536*/, v0 /*v512*/
	v_exp_f32_e32 v196 /*v708*/, v1 /*v513*/
	s_set_vgpr_msb 0x8280
	v_exp_f32_e32 v23 /*v535*/, v195
	v_nop
	s_set_vgpr_msb 0x8011
	v_pk_fma_f32 v[194:195], v[136:137] /*v[392:393]*/, s[14:15], v[254:255] /*v[510:511]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1109
	s_wait_dscnt 0x1e
	v_wmma_f32_16x16x32_bf16 v[112:119], v[138:145] /*v[394:401]*/, v[82:89] /*v[594:601]*/, v[112:119]
	s_set_vgpr_msb 0x900
	v_pk_add_f32 v[204:205], v[226:227], v[204:205]
	v_pk_add_f32 v[206:207], v[232:233], v[206:207]
	s_set_vgpr_msb 10
	v_pk_add_f32 v[210:211], v[144:145] /*v[656:657]*/, v[146:147] /*v[658:659]*/
	s_set_vgpr_msb 0xa02
	v_pk_add_f32 v[208:209], v[142:143] /*v[654:655]*/, v[208:209]
	s_set_vgpr_msb 0x201
	v_pk_add_f32 v[240:241], v[14:15] /*v[270:271]*/, v[240:241]
	s_set_vgpr_msb 0x102
	v_pk_add_f32 v[246:247], v[182:183] /*v[694:695]*/, v[246:247]
	v_pk_add_f32 v[248:249], v[194:195] /*v[706:707]*/, v[248:249]
	s_set_vgpr_msb 0x209
	v_wmma_f32_16x16x32_bf16 v[48:55], v[138:145] /*v[394:401]*/, v[106:113] /*v[618:625]*/, v[48:55]
	s_set_vgpr_msb 0x901
	v_pk_add_f32 v[250:251], v[100:101] /*v[356:357]*/, v[250:251]
	s_set_vgpr_msb 0x105
	v_pk_add_f32 v[252:253], v[102:103] /*v[358:359]*/, v[104:105] /*v[360:361]*/
	v_pk_add_f32 v[254:255], v[124:125] /*v[380:381]*/, v[126:127] /*v[382:383]*/
	s_set_vgpr_msb 0x54a
	v_pk_add_f32 v[2:3] /*v[258:259]*/, v[18:19] /*v[530:531]*/, v[20:21] /*v[532:533]*/
	s_set_vgpr_msb 0x4a00
	v_pk_add_f32 v[192:193], v[198:199], v[192:193]
	v_pk_add_f32 v[198:199], v[200:201], v[202:203]
	v_pk_add_f32 v[200:201], v[212:213], v[214:215]
	s_set_vgpr_msb 9
	v_wmma_f32_16x16x32_bf16 v[120:127], v[50:57] /*v[306:313]*/, v[82:89] /*v[594:601]*/, v[120:127]
	s_set_vgpr_msb 0x900
	v_pk_add_f32 v[202:203], v[242:243], v[244:245]
	s_set_vgpr_msb 0x81
	v_exp_f32_e32 v198 /*v710*/, v182 /*v438*/
	s_set_vgpr_msb 0x8180
	v_exp_f32_e32 v199 /*v711*/, v194
	s_set_vgpr_msb 0x8002
	v_pk_add_f32 v[210:211], v[148:149] /*v[660:661]*/, v[210:211]
	s_set_vgpr_msb 0x24a
	v_pk_add_f32 v[4:5] /*v[260:261]*/, v[24:25] /*v[536:537]*/, v[196:197] /*v[708:709]*/
	s_set_vgpr_msb 0x4a01
	v_pk_add_f32 v[196:197], v[122:123] /*v[378:379]*/, v[252:253]
	v_pk_add_f32 v[252:253], v[128:129] /*v[384:385]*/, v[254:255]
	s_set_vgpr_msb 0x109
	v_wmma_f32_16x16x32_bf16 v[56:63], v[50:57] /*v[306:313]*/, v[106:113] /*v[618:625]*/, v[56:63]
	v_pk_add_f32 v[254:255], v[2:3] /*v[258:259]*/, v[22:23] /*v[534:535]*/
	s_set_vgpr_msb 0x900
	v_pk_add_f32 v[206:207], v[206:207], v[208:209]
	v_pk_add_f32 v[208:209], v[248:249], v[250:251]
	v_pk_add_f32 v[198:199], v[204:205], v[198:199]
	v_pk_add_f32 v[200:201], v[240:241], v[200:201]
	v_pk_add_f32 v[202:203], v[246:247], v[202:203]
	s_set_vgpr_msb 0x46
	v_pk_add_f32 v[2:3] /*v[258:259]*/, v[198:199] /*v[710:711]*/, v[4:5] /*v[260:261]*/
	s_set_vgpr_msb 0x4609
	s_wait_dscnt 0x1a
	v_wmma_f32_16x16x32_bf16 v[112:119], v[146:153] /*v[402:409]*/, v[74:81] /*v[586:593]*/, v[112:119]
	s_set_vgpr_msb 0x900
	v_pk_add_f32 v[204:205], v[210:211], v[206:207]
	v_pk_add_f32 v[196:197], v[196:197], v[208:209]
	v_pk_add_f32 v[206:207], v[252:253], v[254:255]
	v_pk_add_f32 v[192:193], v[192:193], v[198:199]
	v_pk_add_f32 v[198:199], v[200:201], v[202:203]
	s_set_vgpr_msb 1
	v_exp_f32_e32 v214, v183 /*v439*/
	s_set_vgpr_msb 0x100
	v_exp_f32_e32 v215, v195
	s_set_vgpr_msb 9
	v_wmma_f32_16x16x32_bf16 v[48:55], v[146:153] /*v[402:409]*/, v[98:105] /*v[610:617]*/, v[48:55]
	s_set_vgpr_msb 0x901
	v_pk_add_f32 v[206:207], v[2:3] /*v[258:259]*/, v[206:207]
	s_set_vgpr_msb 0x100
	v_pk_add_f32 v[204:205], v[204:205], v[192:193]
	v_pk_add_f32 v[196:197], v[196:197], v[198:199]
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x42
	ds_load_tr16_b128 v[146:149] /*v[402:405]*/, v218 /*v730*/ offset:128
	ds_load_tr16_b128 v[138:141] /*v[394:397]*/, v218 /*v730*/ offset:160
	ds_load_tr16_b128 v[150:153] /*v[406:409]*/, v218 /*v730*/ offset:4736
	ds_load_tr16_b128 v[142:145] /*v[398:401]*/, v218 /*v730*/ offset:4768
	ds_load_tr16_b128 v[254:257] /*v[510:513]*/, v218 /*v730*/ offset:23232
	ds_load_tr16_b128 v[182:185] /*v[438:441]*/, v218 /*v730*/ offset:23264
	s_set_vgpr_msb 0x4202
	ds_load_tr16_b128 v[200:203], v218 /*v730*/ offset:27840
	ds_load_tr16_b128 v[192:195], v218 /*v730*/ offset:27872
	s_set_vgpr_msb 0x242
	ds_load_tr16_b128 v[250:253] /*v[506:509]*/, v218 /*v730*/ offset:18624
	ds_load_tr16_b128 v[178:181] /*v[434:437]*/, v218 /*v730*/ offset:18656
	s_set_vgpr_msb 0x4200
	v_pk_add_f32 v[208:209], v[214:215], v[206:207]
	s_set_vgpr_msb 9
	v_wmma_f32_16x16x32_bf16 v[120:127], v[58:65] /*v[314:321]*/, v[74:81] /*v[586:593]*/, v[120:127]
	s_set_vgpr_msb 0x900
	v_pk_add_f32 v[210:211], v[204:205], v[196:197]
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 10
	ds_load_tr16_b128 v[204:207], v218 /*v730*/ offset:32448
	ds_load_tr16_b128 v[196:199], v218 /*v730*/ offset:32480
	v_sub_f32_e32 v212, v222 /*v734*/, v209 /*v721*/
	s_set_vgpr_msb 0xa00
	v_pk_add_f32 v[208:209], v[208:209], v[210:211]
	v_mul_f32_e32 v210, 0x3fb8aa3b, v212
	s_set_vgpr_msb 9
	v_wmma_f32_16x16x32_bf16 v[56:63], v[58:65] /*v[314:321]*/, v[98:105] /*v[610:617]*/, v[56:63]
	s_set_vgpr_msb 0x942
	ds_load_tr16_b128 v[106:109] /*v[362:365]*/, v218 /*v730*/ offset:9216
	ds_load_tr16_b128 v[74:77] /*v[330:333]*/, v218 /*v730*/ offset:9248
	ds_load_tr16_b128 v[110:113] /*v[366:369]*/, v218 /*v730*/ offset:13824
	ds_load_tr16_b128 v[78:81] /*v[334:337]*/, v218 /*v730*/ offset:13856
	ds_load_tr16_b128 v[90:93] /*v[346:349]*/, v218 /*v730*/ offset:18432
	s_wait_alu depctr_va_vdst(0)
	ds_load_tr16_b128 v[58:61] /*v[314:317]*/, v218 /*v730*/ offset:18464
	ds_load_tr16_b128 v[94:97] /*v[350:353]*/, v218 /*v730*/ offset:23040
	ds_load_tr16_b128 v[62:65] /*v[318:321]*/, v218 /*v730*/ offset:23072
	s_set_vgpr_msb 0x4200
	v_dual_mov_b32 v211, v208 :: v_dual_mov_b32 v213, v209
	v_exp_f32_e32 v210, v210
	v_permlanex16_b32 v211, v211, s16, 0xfedcba98
	s_set_vgpr_msb 10
	v_wmma_f32_16x16x32_bf16 v[72:79], v[248:255] /*v[760:767]*/, v[90:97] /*v[602:609]*/, v[72:79]
	s_set_vgpr_msb 0xa00
	v_permlanex16_b32 v213, v213, s16, 0xfedcba98
	s_set_vgpr_msb 10
	v_wmma_f32_16x16x32_bf16 v[8:15], v[248:255] /*v[760:767]*/, v[114:121] /*v[626:633]*/, v[8:15]
	s_set_vgpr_msb 0xa0b
	v_wmma_f32_16x16x32_bf16 v[64:71], v[0:7] /*v[768:775]*/, v[90:97] /*v[602:609]*/, v[64:71]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[0:7] /*v[768:775]*/, v[114:121] /*v[626:633]*/, v[0:7]
	s_set_vgpr_msb 0xb09
	v_wmma_f32_16x16x32_bf16 v[96:103], v[218:225] /*v[474:481]*/, v[74:81] /*v[586:593]*/, v[96:103]
	v_wmma_f32_16x16x32_bf16 v[32:39], v[218:225] /*v[474:481]*/, v[98:105] /*v[610:617]*/, v[32:39]
	s_set_vgpr_msb 0x982
	ds_load_tr16_b128 v[38:41] /*v[550:553]*/, v218 /*v730*/ offset:23168
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x8242
	ds_load_tr16_b128 v[222:225] /*v[478:481]*/, v218 /*v730*/ offset:23200
	s_set_vgpr_msb 0x4282
	ds_load_tr16_b128 v[42:45] /*v[554:557]*/, v218 /*v730*/ offset:27776
	s_set_vgpr_msb 0x8242
	ds_load_tr16_b128 v[242:245] /*v[498:501]*/, v218 /*v730*/ offset:27808
	s_set_vgpr_msb 0x4282
	ds_load_tr16_b128 v[34:37] /*v[546:549]*/, v218 /*v730*/ offset:18560
	s_set_vgpr_msb 0x8242
	ds_load_tr16_b128 v[218:221] /*v[474:477]*/, v218 /*v730*/ offset:18592
	s_set_vgpr_msb 0x4209
	v_wmma_f32_16x16x32_bf16 v[112:119], v[154:161] /*v[410:417]*/, v[66:73] /*v[578:585]*/, v[112:119]
	v_wmma_f32_16x16x32_bf16 v[48:55], v[154:161] /*v[410:417]*/, v[50:57] /*v[562:569]*/, v[48:55]
	s_set_vgpr_msb 0x982
	ds_load_tr16_b128 v[2:5] /*v[514:517]*/, v218 /*v730*/ offset:18496
	s_set_vgpr_msb 0x8242
	ds_load_tr16_b128 v[186:189] /*v[442:445]*/, v218 /*v730*/ offset:18528
	s_set_vgpr_msb 0x4282
	ds_load_tr16_b128 v[6:9] /*v[518:521]*/, v218 /*v730*/ offset:23104
	s_set_vgpr_msb 0x8242
	ds_load_tr16_b128 v[190:193] /*v[446:449]*/, v218 /*v730*/ offset:23136
	ds_load_tr16_b128 v[202:205] /*v[458:461]*/, v218 /*v730*/ offset:27712
	s_wait_alu depctr_va_vdst(0)
	ds_load_tr16_b128 v[154:157] /*v[410:413]*/, v218 /*v730*/ offset:27744
	ds_load_tr16_b128 v[206:209] /*v[462:465]*/, v218 /*v730*/ offset:32320
	ds_load_tr16_b128 v[158:161] /*v[414:417]*/, v218 /*v730*/ offset:32352
	s_set_vgpr_msb 0x4209
	v_wmma_f32_16x16x32_bf16 v[120:127], v[18:25] /*v[274:281]*/, v[66:73] /*v[578:585]*/, v[120:127]
	v_wmma_f32_16x16x32_bf16 v[56:63], v[18:25] /*v[274:281]*/, v[50:57] /*v[562:569]*/, v[56:63]
	s_set_vgpr_msb 0x942
	ds_load_tr16_b128 v[82:85] /*v[338:341]*/, v218 /*v730*/ offset:27648
	ds_load_tr16_b128 v[50:53] /*v[306:309]*/, v218 /*v730*/ offset:27680
	ds_load_tr16_b128 v[86:89] /*v[342:345]*/, v218 /*v730*/ offset:32256
	ds_load_tr16_b128 v[54:57] /*v[310:313]*/, v218 /*v730*/ offset:32288
	ds_load_tr16_b128 v[42:45] /*v[298:301]*/, v218 /*v730*/ offset:64
	s_wait_alu depctr_va_vdst(0)
	ds_load_tr16_b128 v[18:21] /*v[274:277]*/, v218 /*v730*/ offset:96
	ds_load_tr16_b128 v[46:49] /*v[302:305]*/, v218 /*v730*/ offset:4672
	ds_load_tr16_b128 v[22:25] /*v[278:281]*/, v218 /*v730*/ offset:4704
	s_set_vgpr_msb 0x4282
	ds_load_tr16_b128 v[10:13] /*v[522:525]*/, v218 /*v730*/ offset:9280
	s_set_vgpr_msb 0x8242
	ds_load_tr16_b128 v[194:197] /*v[450:453]*/, v218 /*v730*/ offset:9312
	s_set_vgpr_msb 0x4282
	ds_load_tr16_b128 v[14:17] /*v[526:529]*/, v218 /*v730*/ offset:13888
	s_set_vgpr_msb 0x8242
	ds_load_tr16_b128 v[198:201] /*v[454:457]*/, v218 /*v730*/ offset:13920
	s_set_vgpr_msb 0x420a
	v_wmma_f32_16x16x32_bf16 v[80:87], v[224:231] /*v[736:743]*/, v[82:89] /*v[594:601]*/, v[80:87]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[224:231] /*v[736:743]*/, v[106:113] /*v[618:625]*/, v[16:23]
	s_set_vgpr_msb 0xa0b
	s_wait_dscnt 0x3e
	v_wmma_f32_16x16x32_bf16 v[72:79], v[8:15] /*v[776:783]*/, v[82:89] /*v[594:601]*/, v[72:79]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[8:15] /*v[776:783]*/, v[106:113] /*v[618:625]*/, v[8:15]
	v_wmma_f32_16x16x32_bf16 v[64:71], v[16:23] /*v[784:791]*/, v[82:89] /*v[594:601]*/, v[64:71]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[16:23] /*v[784:791]*/, v[106:113] /*v[618:625]*/, v[0:7]
	s_set_vgpr_msb 0xb09
	v_wmma_f32_16x16x32_bf16 v[96:103], v[234:241] /*v[490:497]*/, v[66:73] /*v[578:585]*/, v[96:103]
	v_wmma_f32_16x16x32_bf16 v[32:39], v[234:241] /*v[490:497]*/, v[50:57] /*v[562:569]*/, v[32:39]
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x942
	ds_load_tr16_b128 v[234:237] /*v[490:493]*/, v218 /*v730*/ offset:9408
	ds_load_tr16_b128 v[170:173] /*v[426:429]*/, v218 /*v730*/ offset:9440
	ds_load_tr16_b128 v[238:241] /*v[494:497]*/, v218 /*v730*/ offset:14016
	ds_load_tr16_b128 v[174:177] /*v[430:433]*/, v218 /*v730*/ offset:14048
	s_wait_dscnt 0x0
	s_wait_tensorcnt 0x0
	s_set_vgpr_msb 0x420a
	s_barrier_signal -1
	v_wmma_f32_16x16x32_bf16 v[80:87], v[232:239] /*v[744:751]*/, v[74:81] /*v[586:593]*/, v[80:87]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[232:239] /*v[744:751]*/, v[98:105] /*v[610:617]*/, v[16:23]
	s_set_vgpr_msb 0xa0b
	v_wmma_f32_16x16x32_bf16 v[72:79], v[24:31] /*v[792:799]*/, v[74:81] /*v[586:593]*/, v[72:79]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[24:31] /*v[792:799]*/, v[98:105] /*v[610:617]*/, v[8:15]
	s_set_vgpr_msb 0xb0a
	v_wmma_f32_16x16x32_bf16 v[64:71], v[122:129] /*v[634:641]*/, v[74:81] /*v[586:593]*/, v[64:71]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[122:129] /*v[634:641]*/, v[98:105] /*v[610:617]*/, v[0:7]
	v_wmma_f32_16x16x32_bf16 v[80:87], v[240:247] /*v[752:759]*/, v[66:73] /*v[578:585]*/, v[80:87]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[240:247] /*v[752:759]*/, v[50:57] /*v[562:569]*/, v[16:23]
	s_set_vgpr_msb 0xa0b
	v_wmma_f32_16x16x32_bf16 v[72:79], v[32:39] /*v[800:807]*/, v[66:73] /*v[578:585]*/, v[72:79]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[32:39] /*v[800:807]*/, v[50:57] /*v[562:569]*/, v[8:15]
	s_set_vgpr_msb 0xb0a
	v_wmma_f32_16x16x32_bf16 v[64:71], v[58:65] /*v[570:577]*/, v[66:73] /*v[578:585]*/, v[64:71]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[58:65] /*v[570:577]*/, v[50:57] /*v[562:569]*/, v[0:7]
	s_set_vgpr_msb 0xa00
	s_cbranch_vccz .LBB0_26
	v_pk_mul_f32 v[126:127], v[126:127], v[210:211] op_sel_hi:[1,0]
	v_pk_mul_f32 v[124:125], v[124:125], v[210:211] op_sel_hi:[1,0]
	v_pk_mul_f32 v[122:123], v[122:123], v[210:211] op_sel_hi:[1,0]
	v_pk_mul_f32 v[120:121], v[120:121], v[210:211] op_sel_hi:[1,0]
	v_pk_mul_f32 v[118:119], v[118:119], v[210:211] op_sel_hi:[1,0]
	v_pk_mul_f32 v[116:117], v[116:117], v[210:211] op_sel_hi:[1,0]
	v_pk_mul_f32 v[114:115], v[114:115], v[210:211] op_sel_hi:[1,0]
	v_pk_mul_f32 v[112:113], v[112:113], v[210:211] op_sel_hi:[1,0]
	v_pk_mul_f32 v[110:111], v[110:111], v[210:211] op_sel_hi:[1,0]
	v_pk_mul_f32 v[108:109], v[108:109], v[210:211] op_sel_hi:[1,0]
	v_pk_mul_f32 v[106:107], v[106:107], v[210:211] op_sel_hi:[1,0]
	v_pk_mul_f32 v[104:105], v[104:105], v[210:211] op_sel_hi:[1,0]
	v_pk_mul_f32 v[102:103], v[102:103], v[210:211] op_sel_hi:[1,0]
	v_pk_mul_f32 v[100:101], v[100:101], v[210:211] op_sel_hi:[1,0]
	v_pk_mul_f32 v[98:99], v[98:99], v[210:211] op_sel_hi:[1,0]
	v_pk_mul_f32 v[96:97], v[96:97], v[210:211] op_sel_hi:[1,0]
	v_pk_mul_f32 v[94:95], v[94:95], v[210:211] op_sel_hi:[1,0]
	v_pk_mul_f32 v[92:93], v[92:93], v[210:211] op_sel_hi:[1,0]
	v_pk_mul_f32 v[90:91], v[90:91], v[210:211] op_sel_hi:[1,0]
	v_pk_mul_f32 v[88:89], v[88:89], v[210:211] op_sel_hi:[1,0]
	v_pk_mul_f32 v[86:87], v[86:87], v[210:211] op_sel_hi:[1,0]
	v_pk_mul_f32 v[84:85], v[84:85], v[210:211] op_sel_hi:[1,0]
	v_pk_mul_f32 v[82:83], v[82:83], v[210:211] op_sel_hi:[1,0]
	v_pk_mul_f32 v[80:81], v[80:81], v[210:211] op_sel_hi:[1,0]
	v_pk_mul_f32 v[78:79], v[78:79], v[210:211] op_sel_hi:[1,0]
	v_pk_mul_f32 v[76:77], v[76:77], v[210:211] op_sel_hi:[1,0]
	v_pk_mul_f32 v[74:75], v[74:75], v[210:211] op_sel_hi:[1,0]
	v_pk_mul_f32 v[72:73], v[72:73], v[210:211] op_sel_hi:[1,0]
	v_pk_mul_f32 v[70:71], v[70:71], v[210:211] op_sel_hi:[1,0]
	v_pk_mul_f32 v[68:69], v[68:69], v[210:211] op_sel_hi:[1,0]
	v_pk_mul_f32 v[66:67], v[66:67], v[210:211] op_sel_hi:[1,0]
	v_pk_mul_f32 v[64:65], v[64:65], v[210:211] op_sel_hi:[1,0]
.LBB0_26:
	s_set_vgpr_msb 10
	v_sub_f32_e32 v212, v221 /*v733*/, v211 /*v723*/
	s_and_b32 s2, s3, exec_lo
	s_cselect_b32 s2, 1, 0
	s_cmp_lg_u32 s2, 1
	s_set_vgpr_msb 0xa00
	v_mul_f32_e32 v212, 0x3fb8aa3b, v212
	v_exp_f32_e32 v212, v212
	s_cbranch_scc1 .LBB0_28
	v_nop
	v_pk_mul_f32 v[62:63], v[62:63], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[60:61], v[60:61], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59], v[58:59], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[56:57], v[56:57], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[54:55], v[54:55], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[52:53], v[52:53], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[50:51], v[50:51], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[48:49], v[48:49], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[46:47], v[46:47], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[44:45], v[44:45], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[42:43], v[42:43], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[40:41], v[40:41], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39], v[38:39], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[36:37], v[36:37], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35], v[34:35], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[32:33], v[32:33], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[30:31], v[30:31], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[28:29], v[28:29], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[26:27], v[26:27], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25], v[24:25], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[22:23], v[22:23], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21], v[20:21], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[18:19], v[18:19], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[16:17], v[16:17], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[14:15], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13], v[12:13], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[10:11], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[8:9], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[6:7], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[4:5], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[2:3], v[2:3], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[0:1], v[0:1], v[212:213] op_sel_hi:[1,0]
.LBB0_28:
	s_set_vgpr_msb 10
	v_cvt_pk_bf16_f32 v247, v188 /*v700*/, v192 /*v704*/
	v_cvt_pk_bf16_f32 v246, v180 /*v692*/, v184 /*v696*/
	v_cvt_pk_bf16_f32 v245, v172 /*v684*/, v176 /*v688*/
	v_cvt_pk_bf16_f32 v244, v164 /*v676*/, v168 /*v680*/
	v_cvt_pk_bf16_f32 v243, v156 /*v668*/, v158 /*v670*/
	v_cvt_pk_bf16_f32 v242, v138 /*v650*/, v140 /*v652*/
	s_set_vgpr_msb 0xa00
	v_cvt_pk_bf16_f32 v241, v220, v222
	v_cvt_pk_bf16_f32 v240, v216, v218
	s_set_vgpr_msb 0x4a
	v_cvt_pk_bf16_f32 v9 /*v265*/, v189 /*v701*/, v193 /*v705*/
	v_cvt_pk_bf16_f32 v8 /*v264*/, v181 /*v693*/, v185 /*v697*/
	v_cvt_pk_bf16_f32 v7 /*v263*/, v173 /*v685*/, v177 /*v689*/
	v_cvt_pk_bf16_f32 v6 /*v262*/, v165 /*v677*/, v169 /*v681*/
	v_cvt_pk_bf16_f32 v5 /*v261*/, v157 /*v669*/, v159 /*v671*/
	v_cvt_pk_bf16_f32 v4 /*v260*/, v139 /*v651*/, v141 /*v653*/
	s_set_vgpr_msb 0x4a40
	v_cvt_pk_bf16_f32 v3 /*v259*/, v221, v223
	v_cvt_pk_bf16_f32 v2 /*v258*/, v217, v219
	s_set_vgpr_msb 0x4001
	v_wmma_f32_16x16x32_bf16 v[120:127], v[114:121] /*v[370:377]*/, v[240:247], v[120:127]
	s_set_vgpr_msb 0x10a
	v_cvt_pk_bf16_f32 v255, v160 /*v672*/, v166 /*v678*/
	v_cvt_pk_bf16_f32 v254, v152 /*v664*/, v154 /*v666*/
	v_cvt_pk_bf16_f32 v253, v148 /*v660*/, v150 /*v662*/
	v_cvt_pk_bf16_f32 v252, v144 /*v656*/, v146 /*v658*/
	s_set_vgpr_msb 0xa09
	v_cvt_pk_bf16_f32 v251, v10 /*v266*/, v142 /*v654*/
	s_set_vgpr_msb 0x900
	v_cvt_pk_bf16_f32 v250, v232, v234
	v_cvt_pk_bf16_f32 v249, v228, v230
	s_set_vgpr_msb 5
	v_wmma_f32_16x16x32_bf16 v[56:63], v[114:121] /*v[370:377]*/, v[2:9] /*v[258:265]*/, v[56:63]
	s_set_vgpr_msb 0x500
	v_cvt_pk_bf16_f32 v248, v224, v226
	s_set_vgpr_msb 0x4a
	v_cvt_pk_bf16_f32 v41 /*v297*/, v161 /*v673*/, v167 /*v679*/
	v_cvt_pk_bf16_f32 v40 /*v296*/, v153 /*v665*/, v155 /*v667*/
	v_cvt_pk_bf16_f32 v39 /*v295*/, v149 /*v661*/, v151 /*v663*/
	v_cvt_pk_bf16_f32 v38 /*v294*/, v145 /*v657*/, v147 /*v659*/
	s_set_vgpr_msb 0x4a49
	v_cvt_pk_bf16_f32 v37 /*v293*/, v11 /*v267*/, v143 /*v655*/
	s_set_vgpr_msb 0x4940
	v_cvt_pk_bf16_f32 v36 /*v292*/, v233, v235
	s_set_vgpr_msb 0x4001
	v_wmma_f32_16x16x32_bf16 v[112:119], v[66:73] /*v[322:329]*/, v[240:247], v[112:119]
	s_set_vgpr_msb 0x140
	v_cvt_pk_bf16_f32 v35 /*v291*/, v229, v231
	v_cvt_pk_bf16_f32 v34 /*v290*/, v225, v227
	s_set_vgpr_msb 0x400a
	v_cvt_pk_bf16_f32 v223, v190 /*v702*/, v194 /*v706*/
	v_cvt_pk_bf16_f32 v222, v182 /*v694*/, v186 /*v698*/
	v_cvt_pk_bf16_f32 v221, v174 /*v686*/, v178 /*v690*/
	v_cvt_pk_bf16_f32 v220, v162 /*v674*/, v170 /*v682*/
	s_set_vgpr_msb 0xa05
	v_cvt_pk_bf16_f32 v219, v30 /*v286*/, v32 /*v288*/
	v_wmma_f32_16x16x32_bf16 v[48:55], v[66:73] /*v[322:329]*/, v[2:9] /*v[258:265]*/, v[48:55]
	v_cvt_pk_bf16_f32 v218, v16 /*v272*/, v26 /*v282*/
	v_cvt_pk_bf16_f32 v217, v12 /*v268*/, v14 /*v270*/
	s_set_vgpr_msb 0x500
	v_cvt_pk_bf16_f32 v216, v236, v238
	s_set_vgpr_msb 2
	v_cvt_pk_bf16_f32 v231, v198 /*v710*/, v214
	s_set_vgpr_msb 0x20a
	v_cvt_pk_bf16_f32 v230, v24 /*v536*/, v196 /*v708*/
	v_cvt_pk_bf16_f32 v229, v20 /*v532*/, v22 /*v534*/
	s_set_vgpr_msb 0xa09
	v_cvt_pk_bf16_f32 v228, v128 /*v384*/, v18 /*v530*/
	s_set_vgpr_msb 0x901
	v_wmma_f32_16x16x32_bf16 v[104:111], v[42:49] /*v[298:305]*/, v[240:247], v[104:111]
	s_set_vgpr_msb 0x105
	v_cvt_pk_bf16_f32 v227, v124 /*v380*/, v126 /*v382*/
	v_cvt_pk_bf16_f32 v226, v104 /*v360*/, v122 /*v378*/
	v_cvt_pk_bf16_f32 v225, v100 /*v356*/, v102 /*v358*/
	v_cvt_pk_bf16_f32 v224, v28 /*v284*/, v98 /*v354*/
	s_set_vgpr_msb 0x50a
	v_cvt_pk_bf16_f32 v238, v25 /*v537*/, v197 /*v709*/
	s_set_vgpr_msb 0xa09
	v_cvt_pk_bf16_f32 v236, v129 /*v385*/, v19 /*v531*/
	s_set_vgpr_msb 0x905
	v_cvt_pk_bf16_f32 v235, v125 /*v381*/, v127 /*v383*/
	v_wmma_f32_16x16x32_bf16 v[40:47], v[42:49] /*v[298:305]*/, v[2:9] /*v[258:265]*/, v[40:47]
	v_cvt_pk_bf16_f32 v234, v105 /*v361*/, v123 /*v379*/
	v_cvt_pk_bf16_f32 v233, v101 /*v357*/, v103 /*v359*/
	v_cvt_pk_bf16_f32 v232, v29 /*v285*/, v99 /*v355*/
	s_add_co_i32 s5, s12, 4
	s_add_nc_u64 s[2:3], s[10:11], s[12:13]
	s_cmp_ge_i32 s5, s54
	s_set_vgpr_msb 0x501
	s_barrier_wait -1
	v_wmma_f32_16x16x32_bf16 v[96:103], v[18:25] /*v[274:281]*/, v[240:247], v[96:103]
	s_set_vgpr_msb 0x105
	v_wmma_f32_16x16x32_bf16 v[32:39], v[18:25] /*v[274:281]*/, v[2:9] /*v[258:265]*/, v[32:39]
	s_set_vgpr_msb 0x501
	v_wmma_f32_16x16x32_bf16 v[88:95], v[146:153] /*v[402:409]*/, v[240:247], v[88:95]
	s_set_vgpr_msb 0x105
	v_wmma_f32_16x16x32_bf16 v[24:31], v[146:153] /*v[402:409]*/, v[2:9] /*v[258:265]*/, v[24:31]
	s_set_vgpr_msb 0x501
	v_wmma_f32_16x16x32_bf16 v[80:87], v[138:145] /*v[394:401]*/, v[240:247], v[80:87]
	s_set_vgpr_msb 0x105
	v_wmma_f32_16x16x32_bf16 v[16:23], v[138:145] /*v[394:401]*/, v[2:9] /*v[258:265]*/, v[16:23]
	s_set_vgpr_msb 0x501
	v_wmma_f32_16x16x32_bf16 v[72:79], v[226:233] /*v[482:489]*/, v[240:247], v[72:79]
	s_set_vgpr_msb 0x105
	v_wmma_f32_16x16x32_bf16 v[8:15], v[226:233] /*v[482:489]*/, v[2:9] /*v[258:265]*/, v[8:15]
	s_set_vgpr_msb 0x501
	v_wmma_f32_16x16x32_bf16 v[64:71], v[162:169] /*v[418:425]*/, v[240:247], v[64:71]
	s_set_vgpr_msb 0x105
	v_wmma_f32_16x16x32_bf16 v[0:7], v[162:169] /*v[418:425]*/, v[2:9] /*v[258:265]*/, v[0:7]
	s_set_vgpr_msb 0x501
	v_wmma_f32_16x16x32_bf16 v[120:127], v[106:113] /*v[362:369]*/, v[248:255], v[120:127]
	s_set_vgpr_msb 0x105
	v_wmma_f32_16x16x32_bf16 v[56:63], v[106:113] /*v[362:369]*/, v[34:41] /*v[290:297]*/, v[56:63]
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x54a
	v_cvt_pk_bf16_f32 v113 /*v369*/, v191 /*v703*/, v195 /*v707*/
	v_cvt_pk_bf16_f32 v112 /*v368*/, v183 /*v695*/, v187 /*v699*/
	v_cvt_pk_bf16_f32 v111 /*v367*/, v175 /*v687*/, v179 /*v691*/
	s_set_vgpr_msb 0x4a01
	v_wmma_f32_16x16x32_bf16 v[112:119], v[74:81] /*v[330:337]*/, v[248:255], v[112:119]
	s_set_vgpr_msb 0x14a
	v_cvt_pk_bf16_f32 v110 /*v366*/, v163 /*v675*/, v171 /*v683*/
	s_set_vgpr_msb 0x4a45
	v_cvt_pk_bf16_f32 v109 /*v365*/, v31 /*v287*/, v33 /*v289*/
	v_cvt_pk_bf16_f32 v108 /*v364*/, v17 /*v273*/, v27 /*v283*/
	v_cvt_pk_bf16_f32 v107 /*v363*/, v13 /*v269*/, v15 /*v271*/
	s_set_vgpr_msb 0x4540
	v_cvt_pk_bf16_f32 v106 /*v362*/, v237, v239
	s_set_vgpr_msb 0x4002
	v_cvt_pk_bf16_f32 v239, v199 /*v711*/, v215
	s_set_vgpr_msb 0x20a
	v_cvt_pk_bf16_f32 v237, v21 /*v533*/, v23 /*v535*/
	s_set_vgpr_msb 0xa05
	v_wmma_f32_16x16x32_bf16 v[48:55], v[74:81] /*v[330:337]*/, v[34:41] /*v[290:297]*/, v[48:55]
	s_set_vgpr_msb 0x502
	v_wmma_f32_16x16x32_bf16 v[104:111], v[10:17] /*v[522:529]*/, v[248:255], v[104:111]
	s_set_vgpr_msb 0x206
	v_wmma_f32_16x16x32_bf16 v[40:47], v[10:17] /*v[522:529]*/, v[34:41] /*v[290:297]*/, v[40:47]
	s_set_vgpr_msb 0x601
	v_wmma_f32_16x16x32_bf16 v[96:103], v[194:201] /*v[450:457]*/, v[248:255], v[96:103]
	s_set_vgpr_msb 0x105
	v_wmma_f32_16x16x32_bf16 v[32:39], v[194:201] /*v[450:457]*/, v[34:41] /*v[290:297]*/, v[32:39]
	s_set_vgpr_msb 0x502
	v_wmma_f32_16x16x32_bf16 v[88:95], v[26:33] /*v[538:545]*/, v[248:255], v[88:95]
	s_set_vgpr_msb 0x206
	v_wmma_f32_16x16x32_bf16 v[24:31], v[26:33] /*v[538:545]*/, v[34:41] /*v[290:297]*/, v[24:31]
	s_set_vgpr_msb 0x601
	v_wmma_f32_16x16x32_bf16 v[80:87], v[210:217] /*v[466:473]*/, v[248:255], v[80:87]
	s_set_vgpr_msb 0x105
	v_wmma_f32_16x16x32_bf16 v[16:23], v[210:217] /*v[466:473]*/, v[34:41] /*v[290:297]*/, v[16:23]
	s_set_vgpr_msb 0x501
	v_wmma_f32_16x16x32_bf16 v[72:79], v[234:241] /*v[490:497]*/, v[248:255], v[72:79]
	s_set_vgpr_msb 0x105
	v_wmma_f32_16x16x32_bf16 v[8:15], v[234:241] /*v[490:497]*/, v[34:41] /*v[290:297]*/, v[8:15]
	s_set_vgpr_msb 0x501
	v_wmma_f32_16x16x32_bf16 v[64:71], v[170:177] /*v[426:433]*/, v[248:255], v[64:71]
	s_set_vgpr_msb 0x105
	v_wmma_f32_16x16x32_bf16 v[0:7], v[170:177] /*v[426:433]*/, v[34:41] /*v[290:297]*/, v[0:7]
	s_set_vgpr_msb 0x501
	v_wmma_f32_16x16x32_bf16 v[120:127], v[90:97] /*v[346:353]*/, v[216:223], v[120:127]
	s_set_vgpr_msb 0x105
	v_wmma_f32_16x16x32_bf16 v[56:63], v[90:97] /*v[346:353]*/, v[106:113] /*v[362:369]*/, v[56:63]
	s_set_vgpr_msb 0x501
	v_wmma_f32_16x16x32_bf16 v[112:119], v[58:65] /*v[314:321]*/, v[216:223], v[112:119]
	s_set_vgpr_msb 0x105
	v_wmma_f32_16x16x32_bf16 v[48:55], v[58:65] /*v[314:321]*/, v[106:113] /*v[362:369]*/, v[48:55]
	s_set_vgpr_msb 0x502
	v_wmma_f32_16x16x32_bf16 v[104:111], v[2:9] /*v[514:521]*/, v[216:223], v[104:111]
	s_set_vgpr_msb 0x206
	v_wmma_f32_16x16x32_bf16 v[40:47], v[2:9] /*v[514:521]*/, v[106:113] /*v[362:369]*/, v[40:47]
	s_set_vgpr_msb 0x601
	v_wmma_f32_16x16x32_bf16 v[96:103], v[186:193] /*v[442:449]*/, v[216:223], v[96:103]
	s_set_vgpr_msb 0x105
	v_wmma_f32_16x16x32_bf16 v[32:39], v[186:193] /*v[442:449]*/, v[106:113] /*v[362:369]*/, v[32:39]
	s_set_vgpr_msb 0x502
	v_wmma_f32_16x16x32_bf16 v[88:95], v[34:41] /*v[546:553]*/, v[216:223], v[88:95]
	s_set_vgpr_msb 0x206
	v_wmma_f32_16x16x32_bf16 v[24:31], v[34:41] /*v[546:553]*/, v[106:113] /*v[362:369]*/, v[24:31]
	s_set_vgpr_msb 0x601
	v_wmma_f32_16x16x32_bf16 v[80:87], v[218:225] /*v[474:481]*/, v[216:223], v[80:87]
	s_set_vgpr_msb 0x105
	v_wmma_f32_16x16x32_bf16 v[16:23], v[218:225] /*v[474:481]*/, v[106:113] /*v[362:369]*/, v[16:23]
	s_set_vgpr_msb 0x501
	v_wmma_f32_16x16x32_bf16 v[72:79], v[250:257] /*v[506:513]*/, v[216:223], v[72:79]
	s_set_vgpr_msb 0x105
	v_wmma_f32_16x16x32_bf16 v[8:15], v[250:257] /*v[506:513]*/, v[106:113] /*v[362:369]*/, v[8:15]
	s_set_vgpr_msb 0x501
	v_wmma_f32_16x16x32_bf16 v[64:71], v[178:185] /*v[434:441]*/, v[216:223], v[64:71]
	s_set_vgpr_msb 0x105
	v_wmma_f32_16x16x32_bf16 v[0:7], v[178:185] /*v[434:441]*/, v[106:113] /*v[362:369]*/, v[0:7]
	s_set_vgpr_msb 0x501
	v_wmma_f32_16x16x32_bf16 v[120:127], v[82:89] /*v[338:345]*/, v[224:231], v[120:127]
	v_wmma_f32_16x16x32_bf16 v[56:63], v[82:89] /*v[338:345]*/, v[232:239], v[56:63]
	v_wmma_f32_16x16x32_bf16 v[112:119], v[50:57] /*v[306:313]*/, v[224:231], v[112:119]
	v_wmma_f32_16x16x32_bf16 v[48:55], v[50:57] /*v[306:313]*/, v[232:239], v[48:55]
	v_wmma_f32_16x16x32_bf16 v[104:111], v[202:209] /*v[458:465]*/, v[224:231], v[104:111]
	v_wmma_f32_16x16x32_bf16 v[40:47], v[202:209] /*v[458:465]*/, v[232:239], v[40:47]
	v_wmma_f32_16x16x32_bf16 v[96:103], v[154:161] /*v[410:417]*/, v[224:231], v[96:103]
	v_wmma_f32_16x16x32_bf16 v[32:39], v[154:161] /*v[410:417]*/, v[232:239], v[32:39]
	s_set_vgpr_msb 0x102
	v_wmma_f32_16x16x32_bf16 v[88:95], v[42:49] /*v[554:561]*/, v[224:231], v[88:95]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[42:49] /*v[554:561]*/, v[232:239], v[24:31]
	s_set_vgpr_msb 0x201
	v_wmma_f32_16x16x32_bf16 v[80:87], v[242:249] /*v[498:505]*/, v[224:231], v[80:87]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[242:249] /*v[498:505]*/, v[232:239], v[16:23]
	s_set_vgpr_msb 0x100
	v_wmma_f32_16x16x32_bf16 v[72:79], v[200:207], v[224:231], v[72:79]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[200:207], v[232:239], v[8:15]
	v_wmma_f32_16x16x32_bf16 v[64:71], v[192:199], v[224:231], v[64:71]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[192:199], v[232:239], v[0:7]
	s_cbranch_scc1 .LBB0_30
	s_add_co_i32 s5, s15, 0x80
	s_ashr_i32 s3, s2, 31
	v_nop
	v_nop
	v_nop
	v_nop
	v_med3_i32 v192, s5, 0, 0x80
	s_add_co_i32 s6, s8, 0xffffff80
	s_lshr_b32 s3, s3, 30
	s_ashr_i32 s7, s6, 31
	s_add_co_i32 s3, s2, s3
	v_readfirstlane_b32 s5, v192
	s_mul_u64 s[18:19], s[6:7], s[82:83]
	s_and_b32 s3, s3, 0x1ffffc
	s_mul_u64 s[6:7], s[6:7], s[90:91]
	s_lshl_b64 s[18:19], s[18:19], 1
	s_sub_co_i32 s5, s5, vcc_hi
	s_sub_co_i32 s3, s2, s3
	s_lshl_b64 s[6:7], s[6:7], 1
	s_add_nc_u64 s[18:19], s[96:97], s[18:19]
	s_max_i32 s9, s5, 0
	s_mul_i32 s3, s3, 0x11800
	s_add_nc_u64 s[20:21], s[98:99], s[6:7]
	s_add_nc_u64 s[6:7], s[94:95], s[18:19]
	s_lshl_b32 s9, s9, 16
	s_add_co_i32 s5, s102, s3
	s_bitset1_b32 s7, 31
	s_or_b32 s66, s9, 0x7fff
	s_mov_b32 s73, s65
	tensor_load_to_lds s[4:7], s[64:71]
	s_add_nc_u64 s[6:7], s[50:51], s[20:21]
	s_add_co_i32 s5, s103, s3
	s_bitset1_b32 s7, 31
	s_mov_b32 s74, s66
	s_mov_b32 s75, s67
	s_mov_b32 s76, s68
	s_mov_b32 s79, s71
	tensor_load_to_lds s[4:7], s[72:79]
.LBB0_30:
	s_add_co_i32 s3, s12, 5
	s_cmp_ge_i32 s3, s54
	s_cbranch_scc1 .LBB0_19
	s_add_co_i32 s5, s2, 1
	v_nop
	v_nop
	v_nop
	v_nop
	v_med3_i32 v192, s15, 0, 0x80
	s_ashr_i32 s2, s5, 31
	s_ashr_i32 s9, s8, 31
	s_lshr_b32 s6, s2, 30
	s_mul_u64 s[2:3], s[8:9], s[82:83]
	s_add_co_i32 s17, s5, s6
	s_mul_u64 s[6:7], s[8:9], s[90:91]
	s_and_b32 s9, s17, 0x1ffffc
	s_lshl_b64 s[2:3], s[2:3], 1
	s_sub_co_i32 s5, s5, s9
	v_readfirstlane_b32 s9, v192
	s_mul_i32 s17, s5, 0x11800
	s_lshl_b64 s[6:7], s[6:7], 1
	s_add_nc_u64 s[2:3], s[96:97], s[2:3]
	s_add_nc_u64 s[18:19], s[98:99], s[6:7]
	s_sub_co_i32 s5, s9, vcc_hi
	s_add_nc_u64 s[6:7], s[94:95], s[2:3]
	s_max_i32 s2, s5, 0
	s_add_co_i32 s5, s102, s17
	s_lshl_b32 s2, s2, 16
	s_bitset1_b32 s7, 31
	s_or_b32 s66, s2, 0x7fff
	s_mov_b32 s73, s65
	tensor_load_to_lds s[4:7], s[64:71]
	s_add_nc_u64 s[6:7], s[50:51], s[18:19]
	s_add_co_i32 s5, s103, s17
	s_bitset1_b32 s7, 31
	s_mov_b32 s74, s66
	s_mov_b32 s75, s67
	s_mov_b32 s76, s68
	s_mov_b32 s79, s71
	tensor_load_to_lds s[4:7], s[72:79]
	s_branch .LBB0_19
.LBB0_32:
	s_set_vgpr_msb 0x82
	v_dual_mov_b32 v218 /*v730*/, v13 /*v525*/ :: v_dual_mov_b32 v217 /*v729*/, v11 /*v523*/
	v_dual_mov_b32 v216 /*v728*/, v12 /*v524*/ :: v_dual_mov_b32 v214 /*v726*/, v10 /*v522*/
	s_set_vgpr_msb 0x8200
.LBB0_33:
	s_cmp_ge_i32 s100, s54
	s_cbranch_scc1 .LBB0_42
	s_add_co_i32 s2, s63, -1
	s_mov_b32 s71, 0
	s_set_vgpr_msb 0x48
	v_min_i32_e32 v80 /*v336*/, s2, v212 /*v724*/
	v_min_i32_e32 v81 /*v337*/, s2, v213 /*v725*/
	s_mov_b32 s80, s30
	s_mov_b32 s55, s71
	s_mov_b32 s84, 1
	s_mov_b32 s68, 32
	s_mov_b32 s61, 0x76543210
	s_mov_b32 s60, 0x3fb8aa3b
	s_mov_b32 s67, 0x800000
	s_mov_b32 s65, 0xffff0000
	s_mov_b32 s64, 0x7510000
	s_mov_b32 s72, 0xf510000
	s_set_vgpr_msb 0x4800
	s_branch .LBB0_36
.LBB0_35:
	s_set_vgpr_msb 0x49
	v_dual_fmac_f32 v60 /*v316*/, v14 /*v270*/, v130 /*v642*/ :: v_dual_fmac_f32 v61 /*v317*/, v40 /*v296*/, v131 /*v643*/
	s_add_nc_u64 s[100:101], s[100:101], 1
	s_set_vgpr_msb 0x4982
	v_dual_mov_b32 v218 /*v730*/, v205 /*v717*/ :: v_dual_mov_b32 v216 /*v728*/, v215 /*v727*/
	v_cmp_lt_i64_e64 s2, s[100:101], s[54:55]
	s_wait_alu depctr_vm_vsrc(0)
	s_set_vgpr_msb 0x8281
	v_dual_mov_b32 v205 /*v717*/, v41 /*v297*/ :: v_dual_add_f32 v130 /*v642*/, v60 /*v316*/, v240
	v_dual_add_f32 v131 /*v643*/, v61 /*v317*/, v241 :: v_dual_mov_b32 v215 /*v727*/, v15 /*v271*/
	v_mov_b32_e32 v211 /*v723*/, v82 /*v338*/
	s_and_b32 vcc_lo, exec_lo, s2
	s_set_vgpr_msb 0x8100
	s_cbranch_vccz .LBB0_43
.LBB0_36:
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 2
	ds_load_b128 v[192:195], v215 /*v727*/
	ds_load_b128 v[196:199], v215 /*v727*/ offset:32
	ds_load_b128 v[200:203], v215 /*v727*/ offset:64
	ds_load_b128 v[204:207], v215 /*v727*/ offset:96
	ds_load_b128 v[216:219], v215 /*v727*/ offset:128
	ds_load_b128 v[220:223], v215 /*v727*/ offset:160
	ds_load_b128 v[224:227], v215 /*v727*/ offset:192
	ds_load_b128 v[228:231], v215 /*v727*/ offset:224
	ds_load_b128 v[232:235], v215 /*v727*/ offset:4352
	ds_load_b128 v[236:239], v215 /*v727*/ offset:4384
	s_set_vgpr_msb 0x260
	v_lshl_or_b32 v18 /*v274*/, s100, 7, v204 /*v716*/
	s_set_vgpr_msb 0x6002
	ds_load_b128 v[240:243], v215 /*v727*/ offset:4416
	ds_load_b128 v[244:247], v215 /*v727*/ offset:4448
	ds_load_b128 v[248:251], v215 /*v727*/ offset:4480
	ds_load_b128 v[252:255], v215 /*v727*/ offset:4512
	s_set_vgpr_msb 0x242
	ds_load_b128 v[2:5] /*v[258:261]*/, v215 /*v727*/ offset:4544
	ds_load_b128 v[6:9] /*v[262:265]*/, v215 /*v727*/ offset:4576
	ds_load_b128 v[10:13] /*v[266:269]*/, v215 /*v727*/ offset:8704
	ds_load_b128 v[14:17] /*v[270:273]*/, v215 /*v727*/ offset:8736
	ds_load_b128 v[30:33] /*v[286:289]*/, v215 /*v727*/ offset:8768
	ds_load_b128 v[34:37] /*v[290:293]*/, v215 /*v727*/ offset:8800
	ds_load_b128 v[38:41] /*v[294:297]*/, v215 /*v727*/ offset:8832
	ds_load_b128 v[42:45] /*v[298:301]*/, v215 /*v727*/ offset:8864
	ds_load_b128 v[46:49] /*v[302:305]*/, v215 /*v727*/ offset:8896
	ds_load_b128 v[50:53] /*v[306:309]*/, v215 /*v727*/ offset:8928
	ds_load_b128 v[54:57] /*v[310:313]*/, v215 /*v727*/ offset:13056
	ds_load_b128 v[58:61] /*v[314:317]*/, v215 /*v727*/ offset:13088
	ds_load_b128 v[62:65] /*v[318:321]*/, v215 /*v727*/ offset:13120
	ds_load_b128 v[66:69] /*v[322:325]*/, v215 /*v727*/ offset:13152
	ds_load_b128 v[70:73] /*v[326:329]*/, v215 /*v727*/ offset:13184
	ds_load_b128 v[74:77] /*v[330:333]*/, v215 /*v727*/ offset:13216
	ds_load_b128 v[82:85] /*v[338:341]*/, v215 /*v727*/ offset:13248
	ds_load_b128 v[86:89] /*v[342:345]*/, v215 /*v727*/ offset:13280
	ds_load_b128 v[90:93] /*v[346:349]*/, v215 /*v727*/ offset:17408
	ds_load_b128 v[94:97] /*v[350:353]*/, v215 /*v727*/ offset:17440
	ds_load_b128 v[98:101] /*v[354:357]*/, v215 /*v727*/ offset:17472
	ds_load_b128 v[102:105] /*v[358:361]*/, v215 /*v727*/ offset:17504
	ds_load_b128 v[106:109] /*v[362:365]*/, v215 /*v727*/ offset:17536
	ds_load_b128 v[110:113] /*v[366:369]*/, v215 /*v727*/ offset:17568
	ds_load_b128 v[114:117] /*v[370:373]*/, v215 /*v727*/ offset:17600
	ds_load_b128 v[118:121] /*v[374:377]*/, v215 /*v727*/ offset:17632
	ds_load_b128 v[122:125] /*v[378:381]*/, v215 /*v727*/ offset:21760
	ds_load_b128 v[126:129] /*v[382:385]*/, v215 /*v727*/ offset:21792
	s_set_vgpr_msb 0x4205
	v_cmp_gt_i32_e32 vcc_lo, v18 /*v274*/, v80 /*v336*/
	s_set_vgpr_msb 0x509
	v_cmp_lt_i32_e64 s2, v18 /*v274*/, v206 /*v718*/
	s_set_vgpr_msb 0x944
	v_dual_add_nc_u32 v215 /*v471*/, 17, v18 /*v274*/ :: v_dual_bitop2_b32 v78 /*v334*/, 2, v18 /*v274*/ bitop3:0x54
	v_dual_add_nc_u32 v214 /*v470*/, 16, v18 /*v274*/ :: v_dual_bitop2_b32 v19 /*v275*/, 1, v18 /*v274*/ bitop3:0x54
	s_or_b32 s2, s2, vcc_lo
	s_set_vgpr_msb 0x4409
	v_cmp_lt_i32_e64 s4, v78 /*v334*/, v206 /*v718*/
	s_set_vgpr_msb 0x905
	v_cmp_ge_i32_e64 s3, v18 /*v274*/, v80 /*v336*/
	s_set_vgpr_msb 0x509
	v_cmp_lt_i32_e32 vcc_lo, v19 /*v275*/, v206 /*v718*/
	s_set_vgpr_msb 0x944
	v_dual_add_nc_u32 v216 /*v472*/, 18, v18 /*v274*/ :: v_dual_bitop2_b32 v79 /*v335*/, 3, v18 /*v274*/ bitop3:0x54
	v_dual_add_nc_u32 v217 /*v473*/, 19, v18 /*v274*/ :: v_dual_bitop2_b32 v210 /*v466*/, 4, v18 /*v274*/ bitop3:0x54
	s_set_vgpr_msb 0x4400
	s_wait_dscnt 0x28
	v_wmma_f32_16x16x32_bf16 v[208:215], v[192:199], v[128:135], 0
	s_or_b32 s3, vcc_lo, s3
	s_set_vgpr_msb 0x45
	v_cmp_gt_i32_e32 vcc_lo, v79 /*v335*/, v80 /*v336*/
	v_dual_add_nc_u32 v218 /*v474*/, 20, v18 /*v274*/ :: v_dual_bitop2_b32 v211 /*v467*/, 5, v18 /*v274*/ bitop3:0x54
	v_dual_add_nc_u32 v219 /*v475*/, 21, v18 /*v274*/ :: v_dual_bitop2_b32 v212 /*v468*/, 6, v18 /*v274*/ bitop3:0x54
	v_dual_add_nc_u32 v220 /*v476*/, 22, v18 /*v274*/ :: v_dual_bitop2_b32 v213 /*v469*/, 7, v18 /*v274*/ bitop3:0x54
	s_set_vgpr_msb 0x4500
	s_wait_dscnt 0x26
	v_wmma_f32_16x16x32_bf16 v[208:215], v[200:207], v[136:143], v[208:215]
	s_set_vgpr_msb 0x45
	v_cmp_gt_i32_e64 s5, v212 /*v468*/, v80 /*v336*/
	v_dual_add_nc_u32 v221 /*v477*/, 23, v18 /*v274*/ :: v_dual_bitop2_b32 v222 /*v478*/, 32, v18 /*v274*/ bitop3:0x54
	v_dual_add_nc_u32 v230 /*v486*/, 48, v18 /*v274*/ :: v_dual_bitop2_b32 v223 /*v479*/, 33, v18 /*v274*/ bitop3:0x54
	v_dual_add_nc_u32 v231 /*v487*/, 49, v18 /*v274*/ :: v_dual_bitop2_b32 v224 /*v480*/, 34, v18 /*v274*/ bitop3:0x54
	s_set_vgpr_msb 0x4540
	v_wmma_f32_16x16x32_bf16 v[22:29] /*v[278:285]*/, v[192:199], v[160:167], 0
	s_set_vgpr_msb 0x4046
	v_dual_add_nc_u32 v235 /*v491*/, 53, v18 /*v274*/ :: v_dual_bitop2_b32 v228 /*v484*/, 38, v18 /*v274*/ bitop3:0x54
	ds_load_b128 v[130:133] /*v[386:389]*/, v215 /*v727*/ offset:21824
	ds_load_b128 v[134:137] /*v[390:393]*/, v215 /*v727*/ offset:21856
	ds_load_b128 v[138:141] /*v[394:397]*/, v215 /*v727*/ offset:21888
	ds_load_b128 v[142:145] /*v[398:401]*/, v215 /*v727*/ offset:21920
	ds_load_b128 v[146:149] /*v[402:405]*/, v215 /*v727*/ offset:21952
	ds_load_b128 v[150:153] /*v[406:409]*/, v215 /*v727*/ offset:21984
	ds_load_b128 v[154:157] /*v[410:413]*/, v215 /*v727*/ offset:26112
	ds_load_b128 v[158:161] /*v[414:417]*/, v215 /*v727*/ offset:26144
	v_cmp_gt_i32_e64 s6, v206 /*v718*/, v224 /*v480*/
	v_dual_add_nc_u32 v232 /*v488*/, 50, v18 /*v274*/ :: v_dual_bitop2_b32 v225 /*v481*/, 35, v18 /*v274*/ bitop3:0x54
	v_dual_add_nc_u32 v233 /*v489*/, 51, v18 /*v274*/ :: v_dual_bitop2_b32 v226 /*v482*/, 36, v18 /*v274*/ bitop3:0x54
	s_set_vgpr_msb 0x4600
	s_wait_dscnt 0x2c
	v_wmma_f32_16x16x32_bf16 v[208:215], v[216:223], v[144:151], v[208:215]
	s_set_vgpr_msb 0x45
	v_dual_add_nc_u32 v236 /*v492*/, 54, v18 /*v274*/ :: v_dual_bitop2_b32 v229 /*v485*/, 39, v18 /*v274*/ bitop3:0x54
	v_dual_add_nc_u32 v234 /*v490*/, 52, v18 /*v274*/ :: v_dual_bitop2_b32 v227 /*v483*/, 37, v18 /*v274*/ bitop3:0x54
	v_cmp_gt_i32_e64 s7, v225 /*v481*/, v80 /*v336*/
	s_set_vgpr_msb 0x4509
	v_cmp_lt_i32_e64 s8, v225 /*v481*/, v206 /*v718*/
	s_set_vgpr_msb 0x905
	v_cmp_gt_i32_e64 s9, v226 /*v482*/, v80 /*v336*/
	s_set_vgpr_msb 0x550
	v_wmma_f32_16x16x32_bf16 v[22:29] /*v[278:285]*/, v[200:207], v[168:175], v[22:29] /*v[278:285]*/
	s_set_vgpr_msb 0x5009
	v_cmp_lt_i32_e64 s10, v226 /*v482*/, v206 /*v718*/
	s_set_vgpr_msb 0x905
	v_cmp_gt_i32_e64 s11, v227 /*v483*/, v80 /*v336*/
	s_set_vgpr_msb 0x509
	v_cmp_lt_i32_e64 s12, v227 /*v483*/, v206 /*v718*/
	s_set_vgpr_msb 0x946
	v_dual_add_nc_u32 v237 /*v493*/, 55, v18 /*v274*/ :: v_dual_bitop2_b32 v238 /*v494*/, 64, v18 /*v274*/ bitop3:0x54
	ds_load_b128 v[162:165] /*v[418:421]*/, v215 /*v727*/ offset:26176
	ds_load_b128 v[166:169] /*v[422:425]*/, v215 /*v727*/ offset:26208
	ds_load_b128 v[170:173] /*v[426:429]*/, v215 /*v727*/ offset:26240
	ds_load_b128 v[174:177] /*v[430:433]*/, v215 /*v727*/ offset:26272
	ds_load_b128 v[178:181] /*v[434:437]*/, v215 /*v727*/ offset:26304
	ds_load_b128 v[182:185] /*v[438:441]*/, v215 /*v727*/ offset:26336
	ds_load_b128 v[186:189] /*v[442:445]*/, v215 /*v727*/ offset:30464
	ds_load_b128 v[190:193] /*v[446:449]*/, v215 /*v727*/ offset:30496
	s_set_vgpr_msb 0x4605
	v_cmp_gt_i32_e64 s13, v236 /*v492*/, v80 /*v336*/
	s_set_vgpr_msb 0x500
	s_wait_dscnt 0x32
	v_wmma_f32_16x16x32_bf16 v[208:215], v[224:231], v[152:159], v[208:215]
	s_set_vgpr_msb 5
	v_cmp_gt_i32_e64 s15, v237 /*v493*/, v80 /*v336*/
	s_set_vgpr_msb 0x509
	v_cmp_lt_i32_e64 s16, v237 /*v493*/, v206 /*v718*/
	v_cmp_lt_i32_e64 s14, v236 /*v492*/, v206 /*v718*/
	s_set_vgpr_msb 0x944
	v_or_b32_e32 v240 /*v496*/, 0x42, v18 /*v274*/
	v_or_b32_e32 v241 /*v497*/, 0x43, v18 /*v274*/
	v_or_b32_e32 v242 /*v498*/, 0x44, v18 /*v274*/
	v_or_b32_e32 v243 /*v499*/, 0x45, v18 /*v274*/
	s_set_vgpr_msb 0x4450
	v_wmma_f32_16x16x32_bf16 v[22:29] /*v[278:285]*/, v[216:223], v[176:183], v[22:29] /*v[278:285]*/
	s_set_vgpr_msb 0x5045
	v_or_b32_e32 v244 /*v500*/, 0x46, v18 /*v274*/
	v_or_b32_e32 v245 /*v501*/, 0x47, v18 /*v274*/
	v_or_b32_e32 v239 /*v495*/, 0x41, v18 /*v274*/
	v_cmp_gt_i32_e64 s17, v238 /*v494*/, v80 /*v336*/
	s_set_vgpr_msb 0x4500
	v_cndmask_b32_e64 v216, v208, 0xff800000, s2
	s_set_vgpr_msb 5
	v_cmp_gt_i32_e64 s2, v78 /*v334*/, v80 /*v336*/
	s_set_vgpr_msb 0x500
	v_cndmask_b32_e64 v217, v209, 0xff800000, s3
	s_set_vgpr_msb 0x50
	v_wmma_f32_16x16x32_bf16 v[22:29] /*v[278:285]*/, v[224:231], v[184:191], v[22:29] /*v[278:285]*/
	s_set_vgpr_msb 0x5005
	v_cmp_gt_i32_e64 s3, v211 /*v467*/, v80 /*v336*/
	s_set_vgpr_msb 0x509
	v_cmp_lt_i32_e64 s18, v238 /*v494*/, v206 /*v718*/
	s_or_b32 s2, s4, s2
	v_cmp_lt_i32_e64 s4, v211 /*v467*/, v206 /*v718*/
	s_set_vgpr_msb 0x900
	v_cndmask_b32_e64 v218, v210, 0xff800000, s2
	s_set_vgpr_msb 9
	v_cmp_lt_i32_e64 s2, v79 /*v335*/, v206 /*v718*/
	s_set_vgpr_msb 0x905
	v_cmp_gt_i32_e64 s19, v239 /*v495*/, v80 /*v336*/
	s_set_vgpr_msb 0x500
	s_wait_dscnt 0x30
	v_wmma_f32_16x16x32_bf16 v[224:231], v[232:239], v[128:135], 0
	s_set_vgpr_msb 9
	v_cmp_lt_i32_e64 s20, v239 /*v495*/, v206 /*v718*/
	s_set_vgpr_msb 0x945
	v_or_b32_e32 v254 /*v510*/, 0x60, v18 /*v274*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v210 /*v466*/, v80 /*v336*/
	s_set_vgpr_msb 0x4500
	v_cndmask_b32_e64 v219, v211, 0xff800000, s2
	s_set_vgpr_msb 9
	v_cmp_lt_i32_e64 s2, v210 /*v466*/, v206 /*v718*/
	s_set_vgpr_msb 0x984
	v_or_b32_e32 v3 /*v515*/, 0x67, v18 /*v274*/
	s_set_vgpr_msb 0x8400
	s_wait_dscnt 0x2e
	v_wmma_f32_16x16x32_bf16 v[224:231], v[240:247], v[136:143], v[224:231]
	s_or_b32 s17, s18, s17
	s_or_b32 s18, s20, s19
	s_or_b32 s2, s2, vcc_lo
	s_set_vgpr_msb 9
	v_cmp_lt_i32_e32 vcc_lo, v212 /*v468*/, v206 /*v718*/
	s_set_vgpr_msb 0x900
	v_cndmask_b32_e64 v220, v212, 0xff800000, s2
	s_or_b32 s2, s4, s3
	s_set_vgpr_msb 5
	v_cmp_gt_i32_e64 s3, v214 /*v470*/, v80 /*v336*/
	s_set_vgpr_msb 0x500
	s_wait_dscnt 0x2c
	v_wmma_f32_16x16x32_bf16 v[224:231], v[248:255], v[144:151], v[224:231]
	v_cndmask_b32_e64 v221, v213, 0xff800000, s2
	s_or_b32 s2, vcc_lo, s5
	s_set_vgpr_msb 5
	v_cmp_gt_i32_e32 vcc_lo, v213 /*v469*/, v80 /*v336*/
	s_set_vgpr_msb 0x500
	v_cndmask_b32_e64 v222, v214, 0xff800000, s2
	s_set_vgpr_msb 9
	v_cmp_lt_i32_e64 s2, v213 /*v469*/, v206 /*v718*/
	v_cmp_lt_i32_e64 s4, v214 /*v470*/, v206 /*v718*/
	s_set_vgpr_msb 0x905
	v_cmp_gt_i32_e64 s5, v215 /*v471*/, v80 /*v336*/
	s_set_vgpr_msb 0x501
	s_wait_dscnt 0x2a
	v_wmma_f32_16x16x32_bf16 v[224:231], v[2:9] /*v[258:265]*/, v[152:159], v[224:231]
	s_set_vgpr_msb 0x106
	v_cmp_gt_i32_e64 s44, v3 /*v515*/, v80 /*v336*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v206 /*v718*/, v215 /*v471*/
	s_set_vgpr_msb 0x600
	v_cndmask_b32_e64 v223, v215, 0xff800000, s2
	s_or_b32 s2, s4, s3
	s_set_vgpr_msb 5
	v_cmp_gt_i32_e64 s3, v217 /*v473*/, v80 /*v336*/
	s_set_vgpr_msb 0x509
	v_cmp_lt_i32_e64 s4, v217 /*v473*/, v206 /*v718*/
	s_set_vgpr_msb 0x900
	v_wmma_f32_16x16x32_bf16 v[200:207], v[232:239], v[160:167], 0
	v_cndmask_b32_e64 v224, v224, 0xff800000, s2
	s_or_b32 s2, vcc_lo, s5
	s_set_vgpr_msb 5
	v_cmp_gt_i32_e32 vcc_lo, v216 /*v472*/, v80 /*v336*/
	s_set_vgpr_msb 0x500
	v_cndmask_b32_e64 v225, v225, 0xff800000, s2
	s_set_vgpr_msb 9
	v_cmp_lt_i32_e64 s2, v216 /*v472*/, v206 /*v718*/
	s_set_vgpr_msb 0x905
	v_cmp_gt_i32_e64 s5, v218 /*v474*/, v80 /*v336*/
	s_set_vgpr_msb 0x50a
	v_cmp_lt_i32_e64 s45, v3 /*v515*/, v206 /*v718*/
	s_set_vgpr_msb 0xa00
	v_wmma_f32_16x16x32_bf16 v[200:207], v[240:247], v[168:175], v[200:207]
	s_set_vgpr_msb 0x44
	v_add_nc_u32_e32 v246 /*v502*/, 0x50, v18 /*v274*/
	s_or_b32 s2, s2, vcc_lo
	s_set_vgpr_msb 0x4409
	v_cmp_lt_i32_e32 vcc_lo, v218 /*v474*/, v206 /*v718*/
	s_set_vgpr_msb 0x900
	v_cndmask_b32_e64 v226, v226, 0xff800000, s2
	s_or_b32 s2, s4, s3
	s_set_vgpr_msb 5
	v_cmp_gt_i32_e64 s3, v220 /*v476*/, v80 /*v336*/
	s_set_vgpr_msb 0x500
	v_cndmask_b32_e64 v227, v227, 0xff800000, s2
	s_set_vgpr_msb 1
	s_wait_dscnt 0x28
	v_wmma_f32_16x16x32_bf16 v[236:243], v[10:17] /*v[266:273]*/, v[128:135], 0
	s_or_b32 s2, vcc_lo, s5
	s_set_vgpr_msb 0x105
	v_cmp_gt_i32_e32 vcc_lo, v219 /*v475*/, v80 /*v336*/
	s_set_vgpr_msb 0x500
	v_cndmask_b32_e64 v228, v228, 0xff800000, s2
	s_set_vgpr_msb 9
	v_cmp_lt_i32_e64 s2, v219 /*v475*/, v206 /*v718*/
	v_cmp_lt_i32_e64 s4, v220 /*v476*/, v206 /*v718*/
	s_set_vgpr_msb 0x945
	v_cmp_gt_i32_e64 s5, v221 /*v477*/, v80 /*v336*/
	v_add_nc_u32_e32 v248 /*v504*/, 0x52, v18 /*v274*/
	s_set_vgpr_msb 0x4501
	s_wait_dscnt 0x26
	v_wmma_f32_16x16x32_bf16 v[236:243], v[30:37] /*v[286:293]*/, v[136:143], v[236:243]
	s_or_b32 s2, s2, vcc_lo
	s_set_vgpr_msb 0x109
	v_cmp_lt_i32_e32 vcc_lo, v221 /*v477*/, v206 /*v718*/
	s_set_vgpr_msb 0x900
	v_cndmask_b32_e64 v229, v229, 0xff800000, s2
	s_or_b32 s2, s4, s3
	s_set_vgpr_msb 5
	v_cmp_gt_i32_e64 s3, v223 /*v479*/, v80 /*v336*/
	s_set_vgpr_msb 0x500
	v_cndmask_b32_e64 v230, v230, 0xff800000, s2
	s_or_b32 s2, vcc_lo, s5
	s_set_vgpr_msb 1
	s_wait_dscnt 0x24
	v_wmma_f32_16x16x32_bf16 v[236:243], v[38:45] /*v[294:301]*/, v[144:151], v[236:243]
	s_set_vgpr_msb 0x100
	v_cndmask_b32_e64 v231, v231, 0xff800000, s2
	s_set_vgpr_msb 5
	v_cmp_gt_i32_e32 vcc_lo, v222 /*v478*/, v80 /*v336*/
	s_set_vgpr_msb 0x509
	v_cmp_lt_i32_e64 s2, v222 /*v478*/, v206 /*v718*/
	v_cmp_lt_i32_e64 s4, v223 /*v479*/, v206 /*v718*/
	s_set_vgpr_msb 0x945
	v_cmp_gt_i32_e64 s5, v224 /*v480*/, v80 /*v336*/
	v_add_nc_u32_e32 v250 /*v506*/, 0x54, v18 /*v274*/
	v_cmp_gt_i32_e64 s21, v248 /*v504*/, v80 /*v336*/
	s_set_vgpr_msb 0x4501
	v_wmma_f32_16x16x32_bf16 v[208:215], v[10:17] /*v[266:273]*/, v[160:167], 0
	s_or_b32 s2, s2, vcc_lo
	s_set_vgpr_msb 0x105
	v_cmp_gt_i32_e32 vcc_lo, v228 /*v484*/, v80 /*v336*/
	s_or_b32 s3, s4, s3
	s_or_b32 s4, s6, s5
	s_or_b32 s5, s8, s7
	s_or_b32 s6, s10, s9
	s_or_b32 s7, s12, s11
	s_set_vgpr_msb 0x500
	v_wmma_f32_16x16x32_bf16 v[200:207], v[248:255], v[176:183], v[200:207]
	s_set_vgpr_msb 9
	v_cmp_lt_i32_e64 s8, v233 /*v489*/, v206 /*v718*/
	s_set_vgpr_msb 0x905
	v_cmp_gt_i32_e64 s9, v234 /*v490*/, v80 /*v336*/
	s_set_vgpr_msb 0x509
	v_cmp_lt_i32_e64 s10, v234 /*v490*/, v206 /*v718*/
	s_set_vgpr_msb 0x905
	v_cmp_gt_i32_e64 s11, v235 /*v491*/, v80 /*v336*/
	s_set_vgpr_msb 0x509
	v_cmp_lt_i32_e64 s12, v235 /*v491*/, v206 /*v718*/
	v_cmp_lt_i32_e64 s22, v248 /*v504*/, v206 /*v718*/
	s_set_vgpr_msb 0x905
	v_cmp_gt_i32_e64 s25, v250 /*v506*/, v80 /*v336*/
	s_set_vgpr_msb 0x501
	s_wait_dscnt 0x20
	v_wmma_f32_16x16x32_bf16 v[246:253], v[54:61] /*v[310:317]*/, v[128:135], 0
	s_set_vgpr_msb 0x109
	v_cmp_lt_i32_e64 s26, v250 /*v506*/, v206 /*v718*/
	s_set_vgpr_msb 0x905
	v_cmp_gt_i32_e64 s46, v18 /*v274*/, v81 /*v337*/
	s_set_vgpr_msb 0x509
	v_cmp_lt_i32_e64 s47, v18 /*v274*/, v207 /*v719*/
	s_set_vgpr_msb 0x945
	v_add_nc_u32_e32 v249 /*v505*/, 0x53, v18 /*v274*/
	v_add_nc_u32_e32 v251 /*v507*/, 0x55, v18 /*v274*/
	v_add_nc_u32_e32 v252 /*v508*/, 0x56, v18 /*v274*/
	v_cmp_ge_i32_e64 s48, v18 /*v274*/, v81 /*v337*/
	s_set_vgpr_msb 0x4501
	v_wmma_f32_16x16x32_bf16 v[236:243], v[46:53] /*v[302:309]*/, v[152:159], v[236:243]
	s_set_vgpr_msb 0x105
	v_cmp_gt_i32_e64 s23, v249 /*v505*/, v80 /*v336*/
	s_set_vgpr_msb 0x509
	v_cmp_lt_i32_e64 s24, v249 /*v505*/, v206 /*v718*/
	s_set_vgpr_msb 0x905
	v_cmp_gt_i32_e64 s27, v251 /*v507*/, v80 /*v336*/
	s_set_vgpr_msb 0x509
	v_cmp_lt_i32_e64 s28, v251 /*v507*/, v206 /*v718*/
	s_set_vgpr_msb 0x905
	v_cmp_gt_i32_e64 s29, v252 /*v508*/, v80 /*v336*/
	s_set_vgpr_msb 0x509
	v_cmp_lt_i32_e64 s30, v252 /*v508*/, v206 /*v718*/
	s_set_vgpr_msb 0x944
	v_add_nc_u32_e32 v253 /*v509*/, 0x57, v18 /*v274*/
	s_set_vgpr_msb 0x4401
	v_wmma_f32_16x16x32_bf16 v[208:215], v[30:37] /*v[286:293]*/, v[168:175], v[208:215]
	s_set_vgpr_msb 0x100
	v_cndmask_b32_e64 v236, v236, 0xff800000, s2
	s_set_vgpr_msb 9
	v_cmp_lt_i32_e64 s2, v228 /*v484*/, v206 /*v718*/
	s_set_vgpr_msb 0x900
	v_cndmask_b32_e64 v234, v238, 0xff800000, s4
	v_cndmask_b32_e64 v235, v239, 0xff800000, s5
	v_cndmask_b32_e64 v232, v240, 0xff800000, s6
	s_set_vgpr_msb 9
	v_cmp_lt_i32_e64 s6, v232 /*v488*/, v206 /*v718*/
	s_or_b32 s2, s2, vcc_lo
	s_set_vgpr_msb 0x901
	s_wait_dscnt 0x1e
	v_wmma_f32_16x16x32_bf16 v[246:253], v[62:69] /*v[318:325]*/, v[136:143], v[246:253]
	s_set_vgpr_msb 0x100
	v_cndmask_b32_e64 v238, v242, 0xff800000, s2
	s_set_vgpr_msb 5
	v_cmp_gt_i32_e32 vcc_lo, v229 /*v485*/, v80 /*v336*/
	s_set_vgpr_msb 0x509
	v_cmp_lt_i32_e64 s2, v229 /*v485*/, v206 /*v718*/
	s_set_vgpr_msb 0x900
	v_cndmask_b32_e64 v237, v237, 0xff800000, s3
	v_cndmask_b32_e64 v233, v241, 0xff800000, s7
	s_set_vgpr_msb 5
	v_cmp_gt_i32_e64 s3, v230 /*v486*/, v80 /*v336*/
	s_set_vgpr_msb 0x509
	v_cmp_lt_i32_e64 s4, v230 /*v486*/, v206 /*v718*/
	s_set_vgpr_msb 0x901
	v_wmma_f32_16x16x32_bf16 v[200:207], v[2:9] /*v[258:265]*/, v[184:191], v[200:207]
	s_or_b32 s2, s2, vcc_lo
	s_set_vgpr_msb 0x105
	v_cmp_gt_i32_e64 s5, v231 /*v487*/, v80 /*v336*/
	s_set_vgpr_msb 0x500
	v_cndmask_b32_e64 v239, v243, 0xff800000, s2
	s_set_vgpr_msb 5
	v_cmp_gt_i32_e64 s2, v232 /*v488*/, v80 /*v336*/
	s_set_vgpr_msb 0x509
	v_cmp_lt_i32_e32 vcc_lo, v231 /*v487*/, v206 /*v718*/
	s_set_vgpr_msb 0x905
	v_cmp_gt_i32_e64 s7, v233 /*v489*/, v80 /*v336*/
	s_or_b32 s3, s4, s3
	s_set_vgpr_msb 0x541
	s_wait_dscnt 0x18
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[258:265]*/, v[90:97] /*v[346:353]*/, v[128:135], 0
	s_or_b32 s2, s6, s2
	s_or_b32 s4, vcc_lo, s5
	s_or_b32 s5, s8, s7
	s_or_b32 s6, s10, s9
	s_or_b32 s7, s12, s11
	s_or_b32 s8, s14, s13
	s_set_vgpr_msb 0x4105
	v_cmp_gt_i32_e32 vcc_lo, v240 /*v496*/, v80 /*v336*/
	s_set_vgpr_msb 0x501
	v_wmma_f32_16x16x32_bf16 v[208:215], v[38:45] /*v[294:301]*/, v[176:183], v[208:215]
	s_set_vgpr_msb 0x105
	v_cmp_gt_i32_e64 s9, v244 /*v500*/, v80 /*v336*/
	s_set_vgpr_msb 0x509
	v_cmp_lt_i32_e64 s10, v244 /*v500*/, v206 /*v718*/
	s_set_vgpr_msb 0x905
	v_cmp_gt_i32_e64 s11, v245 /*v501*/, v80 /*v336*/
	s_set_vgpr_msb 0x509
	v_cmp_lt_i32_e64 s12, v245 /*v501*/, v206 /*v718*/
	s_set_vgpr_msb 0x905
	v_cmp_gt_i32_e64 s13, v246 /*v502*/, v80 /*v336*/
	s_set_vgpr_msb 0x509
	v_cmp_lt_i32_e64 s14, v246 /*v502*/, v206 /*v718*/
	s_set_vgpr_msb 0x944
	v_or_b32_e32 v255 /*v511*/, 0x61, v18 /*v274*/
	s_set_vgpr_msb 0x4441
	v_wmma_f32_16x16x32_bf16 v[32:39] /*v[288:295]*/, v[54:61] /*v[310:317]*/, v[160:167], 0
	s_set_vgpr_msb 0x4184
	v_or_b32_e32 v0 /*v512*/, 0x62, v18 /*v274*/
	v_or_b32_e32 v1 /*v513*/, 0x63, v18 /*v274*/
	s_or_b32 s13, s14, s13
	v_or_b32_e32 v2 /*v514*/, 0x64, v18 /*v274*/
	v_add_nc_u32_e32 v4 /*v516*/, 0x70, v18 /*v274*/
	v_add_nc_u32_e32 v5 /*v517*/, 0x74, v18 /*v274*/
	v_add_nc_u32_e32 v6 /*v518*/, 0x75, v18 /*v274*/
	s_set_vgpr_msb 0x8401
	v_wmma_f32_16x16x32_bf16 v[246:253], v[70:77] /*v[326:333]*/, v[144:151], v[246:253]
	s_set_vgpr_msb 0x142
	ds_load_b128 v[194:197] /*v[450:453]*/, v215 /*v727*/ offset:30528
	ds_load_b128 v[198:201] /*v[454:457]*/, v215 /*v727*/ offset:30560
	ds_load_b128 v[202:205] /*v[458:461]*/, v215 /*v727*/ offset:30592
	ds_load_b128 v[206:209] /*v[462:465]*/, v215 /*v727*/ offset:30624
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x4202
	ds_load_b128 v[192:195], v215 /*v727*/ offset:30656
	ds_load_b128 v[196:199], v215 /*v727*/ offset:30688
	s_set_vgpr_msb 0x205
	v_cmp_gt_i32_e64 s31, v253 /*v509*/, v80 /*v336*/
	s_set_vgpr_msb 0x509
	v_cmp_lt_i32_e64 s33, v253 /*v509*/, v206 /*v718*/
	s_set_vgpr_msb 0x905
	v_cmp_gt_i32_e64 s19, v255 /*v511*/, v80 /*v336*/
	s_set_vgpr_msb 0x509
	v_cmp_lt_i32_e64 s20, v255 /*v511*/, v206 /*v718*/
	v_cmp_lt_i32_e64 s34, v80 /*v336*/, v0 /*v512*/
	s_set_vgpr_msb 0x90a
	v_cmp_lt_i32_e64 s35, v0 /*v512*/, v206 /*v718*/
	s_set_vgpr_msb 0xa51
	s_wait_dscnt 0x1c
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[258:265]*/, v[98:105] /*v[354:361]*/, v[136:143], v[2:9] /*v[258:265]*/
	s_set_vgpr_msb 0x5106
	v_cmp_gt_i32_e64 s36, v1 /*v513*/, v80 /*v336*/
	s_set_vgpr_msb 0x60a
	v_cmp_lt_i32_e64 s37, v1 /*v513*/, v206 /*v718*/
	s_set_vgpr_msb 0xa06
	v_cmp_gt_i32_e64 s38, v2 /*v514*/, v80 /*v336*/
	s_set_vgpr_msb 0x60a
	v_cmp_lt_i32_e64 s39, v2 /*v514*/, v206 /*v718*/
	s_set_vgpr_msb 0xa51
	v_wmma_f32_16x16x32_bf16 v[32:39] /*v[288:295]*/, v[62:69] /*v[318:325]*/, v[168:175], v[32:39] /*v[288:295]*/
	s_set_vgpr_msb 0x5101
	v_wmma_f32_16x16x32_bf16 v[246:253], v[82:89] /*v[338:345]*/, v[152:159], v[246:253]
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x100
	v_cndmask_b32_e64 v244, v248, 0xff800000, s2
	s_set_vgpr_msb 0x51
	s_wait_dscnt 0x1a
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[258:265]*/, v[106:113] /*v[362:369]*/, v[144:151], v[2:9] /*v[258:265]*/
	s_or_b32 s2, s16, s15
	s_set_vgpr_msb 0x5100
	v_cndmask_b32_e64 v240, v252, 0xff800000, s8
	v_cndmask_b32_e64 v241, v253, 0xff800000, s2
	s_set_vgpr_msb 9
	v_cmp_lt_i32_e64 s2, v240 /*v496*/, v206 /*v718*/
	s_set_vgpr_msb 0x900
	v_cndmask_b32_e64 v246, v246, 0xff800000, s3
	v_cndmask_b32_e64 v247, v247, 0xff800000, s4
	v_cndmask_b32_e64 v245, v249, 0xff800000, s5
	s_set_vgpr_msb 0x51
	v_wmma_f32_16x16x32_bf16 v[32:39] /*v[288:295]*/, v[70:77] /*v[326:333]*/, v[176:183], v[32:39] /*v[288:295]*/
	s_set_vgpr_msb 0x5100
	v_cndmask_b32_e64 v242, v250, 0xff800000, s6
	v_cndmask_b32_e64 v243, v251, 0xff800000, s7
	s_set_vgpr_msb 5
	v_cmp_gt_i32_e64 s3, v241 /*v497*/, v80 /*v336*/
	s_set_vgpr_msb 0x509
	v_cmp_lt_i32_e64 s4, v241 /*v497*/, v206 /*v718*/
	s_set_vgpr_msb 0x905
	v_cmp_gt_i32_e64 s5, v242 /*v498*/, v80 /*v336*/
	s_set_vgpr_msb 0x509
	v_cmp_lt_i32_e64 s6, v242 /*v498*/, v206 /*v718*/
	s_set_vgpr_msb 0x905
	v_cmp_gt_i32_e64 s7, v243 /*v499*/, v80 /*v336*/
	s_set_vgpr_msb 0x541
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x32_bf16 v[64:71] /*v[320:327]*/, v[154:161] /*v[410:417]*/, v[128:135], 0
	s_set_vgpr_msb 0x4109
	v_cmp_lt_i32_e64 s8, v243 /*v499*/, v206 /*v718*/
	s_or_b32 s2, s2, vcc_lo
	s_or_b32 s3, s4, s3
	s_or_b32 s4, s6, s5
	s_or_b32 s6, s10, s9
	s_or_b32 s5, s8, s7
	v_cmp_lt_i32_e32 vcc_lo, v80 /*v336*/, v4 /*v516*/
	s_set_vgpr_msb 0x951
	v_wmma_f32_16x16x32_bf16 v[2:9] /*v[258:265]*/, v[114:121] /*v[370:377]*/, v[152:159], v[2:9] /*v[258:265]*/
	s_set_vgpr_msb 0x510a
	v_cmp_lt_i32_e64 s7, v5 /*v517*/, v206 /*v718*/
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0xa01
	v_cndmask_b32_e64 v252, v4 /*v260*/, 0xff800000, s2
	s_set_vgpr_msb 0x151
	s_wait_dscnt 0xc
	v_wmma_f32_16x16x32_bf16 v[64:71] /*v[320:327]*/, v[162:169] /*v[418:425]*/, v[136:143], v[64:71] /*v[320:327]*/
	s_or_b32 s2, s12, s11
	s_set_vgpr_msb 0x5101
	v_cndmask_b32_e64 v248, v8 /*v264*/, 0xff800000, s6
	v_cndmask_b32_e64 v253, v5 /*v261*/, 0xff800000, s3
	v_cndmask_b32_e64 v250, v6 /*v262*/, 0xff800000, s4
	v_cndmask_b32_e64 v251, v7 /*v263*/, 0xff800000, s5
	v_cndmask_b32_e64 v249, v9 /*v265*/, 0xff800000, s2
	v_cndmask_b32_e64 v254, v2 /*v258*/, 0xff800000, s17
	s_set_vgpr_msb 0x141
	v_wmma_f32_16x16x32_bf16 v[4:11] /*v[260:267]*/, v[122:129] /*v[378:385]*/, v[128:135], 0
	s_set_vgpr_msb 0x4105
	v_cndmask_b32_e64 v255, v3 /*v259*/, 0xff800000, s18
	v_cmp_gt_i32_e64 s17, v254 /*v510*/, v80 /*v336*/
	s_set_vgpr_msb 0x509
	v_cmp_lt_i32_e64 s18, v254 /*v510*/, v206 /*v718*/
	s_set_vgpr_msb 0x944
	v_or_b32_e32 v2 /*v258*/, 0x65, v18 /*v274*/
	v_or_b32_e32 v3 /*v259*/, 0x66, v18 /*v274*/
	s_set_vgpr_msb 0x440a
	v_cmp_lt_i32_e64 s2, v4 /*v516*/, v206 /*v718*/
	s_set_vgpr_msb 0xa06
	v_cmp_gt_i32_e64 s5, v5 /*v517*/, v80 /*v336*/
	s_set_vgpr_msb 0x651
	s_wait_dscnt 0xa
	v_wmma_f32_16x16x32_bf16 v[64:71] /*v[320:327]*/, v[170:177] /*v[426:433]*/, v[144:151], v[64:71] /*v[320:327]*/
	s_or_b32 s17, s18, s17
	s_set_vgpr_msb 0x5106
	v_cmp_gt_i32_e64 s4, v6 /*v518*/, v80 /*v336*/
	s_or_b32 s2, s2, vcc_lo
	s_set_vgpr_msb 0x60a
	v_cmp_lt_i32_e64 s6, v6 /*v518*/, v206 /*v718*/
	s_or_b32 s18, s20, s19
	s_or_b32 s19, s35, s34
	s_or_b32 s20, s37, s36
	s_set_vgpr_msb 0xa51
	v_wmma_f32_16x16x32_bf16 v[4:11] /*v[260:267]*/, v[130:137] /*v[386:393]*/, v[136:143], v[4:11] /*v[260:267]*/
	s_set_vgpr_msb 0x5105
	v_cmp_gt_i32_e32 vcc_lo, v221 /*v477*/, v81 /*v337*/
	v_cmp_gt_i32_e64 s42, v3 /*v259*/, v80 /*v336*/
	s_set_vgpr_msb 0x509
	v_cmp_lt_i32_e64 s43, v3 /*v259*/, v206 /*v718*/
	s_set_vgpr_msb 0x905
	v_cmp_gt_i32_e64 s40, v2 /*v258*/, v80 /*v336*/
	s_set_vgpr_msb 0x509
	v_cmp_lt_i32_e64 s41, v2 /*v258*/, v206 /*v718*/
	s_set_vgpr_msb 0x951
	s_wait_dscnt 0x8
	v_wmma_f32_16x16x32_bf16 v[64:71] /*v[320:327]*/, v[178:185] /*v[434:441]*/, v[152:159], v[64:71] /*v[320:327]*/
	v_nop
	v_nop
	v_nop
	v_nop
	v_cndmask_b32_e64 v58 /*v314*/, v64 /*v320*/, 0xff800000, s17
	v_wmma_f32_16x16x32_bf16 v[4:11] /*v[260:267]*/, v[138:145] /*v[394:401]*/, v[144:151], v[4:11] /*v[260:267]*/
	s_or_b32 s17, s45, s44
	v_cndmask_b32_e64 v59 /*v315*/, v65 /*v321*/, 0xff800000, s18
	v_cndmask_b32_e64 v31 /*v287*/, v71 /*v327*/, 0xff800000, s17
	s_set_vgpr_msb 0x5145
	v_add_nc_u32_e32 v247 /*v503*/, 0x51, v18 /*v274*/
	v_cndmask_b32_e64 v72 /*v328*/, v66 /*v322*/, 0xff800000, s19
	v_cmp_gt_i32_e64 s18, v217 /*v473*/, v81 /*v337*/
	s_set_vgpr_msb 0x4509
	v_cmp_lt_i32_e64 s19, v217 /*v473*/, v207 /*v719*/
	s_set_vgpr_msb 0x951
	v_wmma_f32_16x16x32_bf16 v[4:11] /*v[260:267]*/, v[146:153] /*v[402:409]*/, v[152:159], v[4:11] /*v[260:267]*/
	s_set_vgpr_msb 0x5105
	v_cmp_gt_i32_e64 s15, v247 /*v503*/, v80 /*v336*/
	s_set_vgpr_msb 0x549
	v_cmp_lt_i32_e64 s16, v247 /*v503*/, v206 /*v718*/
	v_cndmask_b32_e64 v73 /*v329*/, v67 /*v323*/, 0xff800000, s20
	s_set_vgpr_msb 0x4905
	v_cmp_gt_i32_e64 s20, v218 /*v474*/, v81 /*v337*/
	s_or_b32 s18, s19, s18
	s_set_vgpr_msb 0x509
	v_cmp_lt_i32_e64 s19, v219 /*v475*/, v207 /*v719*/
	s_or_b32 s14, s16, s15
	s_or_b32 s15, s22, s21
	s_or_b32 s21, s26, s25
	s_set_vgpr_msb 0x901
	v_wmma_f32_16x16x32_bf16 v[208:215], v[46:53] /*v[302:309]*/, v[184:191], v[208:215]
	s_set_vgpr_msb 0x149
	v_cndmask_b32_e64 v20 /*v276*/, v8 /*v264*/, 0xff800000, s21
	s_or_b32 s21, s47, s46
	s_or_b32 s16, s24, s23
	v_cndmask_b32_e64 v14 /*v270*/, v22 /*v278*/, 0xff800000, s21
	v_cmp_lt_i32_e64 s21, v19 /*v275*/, v207 /*v719*/
	s_or_b32 s22, s28, s27
	s_or_b32 s23, s30, s29
	s_set_vgpr_msb 0x4941
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[306:313]*/, v[122:129] /*v[378:385]*/, v[160:167], 0
	v_cndmask_b32_e64 v16 /*v272*/, v10 /*v266*/, 0xff800000, s23
	s_or_b32 s21, s21, s48
	v_cndmask_b32_e64 v21 /*v277*/, v9 /*v265*/, 0xff800000, s22
	s_set_vgpr_msb 0x4105
	v_cmp_gt_i32_e64 s22, v78 /*v334*/, v81 /*v337*/
	s_set_vgpr_msb 0x509
	v_cmp_lt_i32_e64 s23, v78 /*v334*/, v207 /*v719*/
	s_set_vgpr_msb 0x945
	v_cmp_gt_i32_e64 s24, v79 /*v335*/, v81 /*v337*/
	v_cndmask_b32_e64 v15 /*v271*/, v23 /*v279*/, 0xff800000, s21
	s_set_vgpr_msb 0x4551
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[306:313]*/, v[130:137] /*v[386:393]*/, v[168:175], v[50:57] /*v[306:313]*/
	s_set_vgpr_msb 0x5109
	v_cmp_lt_i32_e64 s21, v79 /*v335*/, v207 /*v719*/
	s_or_b32 s22, s23, s22
	s_set_vgpr_msb 0x945
	v_cmp_gt_i32_e64 s23, v211 /*v467*/, v81 /*v337*/
	v_cmp_gt_i32_e64 s25, v212 /*v468*/, v81 /*v337*/
	v_cndmask_b32_e64 v12 /*v268*/, v4 /*v260*/, 0xff800000, s13
	s_or_b32 s21, s21, s24
	v_add_nc_u32_e32 v134 /*v390*/, 0x71, v18 /*v274*/
	v_add_nc_u32_e32 v136 /*v392*/, 0x72, v18 /*v274*/
	v_add_nc_u32_e32 v137 /*v393*/, 0x73, v18 /*v274*/
	s_set_vgpr_msb 0x4551
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[306:313]*/, v[138:145] /*v[394:401]*/, v[176:183], v[50:57] /*v[306:313]*/
	v_cndmask_b32_e64 v19 /*v275*/, v25 /*v281*/, 0xff800000, s21
	s_set_vgpr_msb 0x5109
	v_cmp_lt_i32_e64 s21, v210 /*v466*/, v207 /*v719*/
	v_cmp_lt_i32_e64 s24, v211 /*v467*/, v207 /*v719*/
	s_set_vgpr_msb 0x945
	v_cmp_gt_i32_e64 s3, v134 /*v390*/, v80 /*v336*/
	v_add_nc_u32_e32 v139 /*v395*/, 0x76, v18 /*v274*/
	v_add_nc_u32_e32 v140 /*v396*/, 0x77, v18 /*v274*/
	v_cndmask_b32_e64 v18 /*v274*/, v24 /*v280*/, 0xff800000, s22
	v_cmp_gt_i32_e64 s22, v210 /*v466*/, v81 /*v337*/
	s_set_vgpr_msb 0x4509
	v_cmp_lt_i32_e64 s12, v134 /*v390*/, v206 /*v718*/
	s_set_vgpr_msb 0x905
	v_cmp_gt_i32_e64 s10, v136 /*v392*/, v80 /*v336*/
	s_set_vgpr_msb 0x509
	v_cmp_lt_i32_e64 s11, v136 /*v392*/, v206 /*v718*/
	s_set_vgpr_msb 0x945
	v_cmp_gt_i32_e64 s8, v137 /*v393*/, v80 /*v336*/
	s_or_b32 s21, s21, s22
	v_cmp_gt_i32_e64 s22, v213 /*v469*/, v81 /*v337*/
	v_cndmask_b32_e64 v40 /*v296*/, v26 /*v282*/, 0xff800000, s21
	s_or_b32 s21, s24, s23
	s_set_vgpr_msb 0x4549
	v_cmp_lt_i32_e64 s23, v213 /*v469*/, v207 /*v719*/
	v_cndmask_b32_e64 v41 /*v297*/, v27 /*v283*/, 0xff800000, s21
	v_cmp_lt_i32_e64 s21, v212 /*v468*/, v207 /*v719*/
	s_set_vgpr_msb 0x4905
	v_cmp_gt_i32_e64 s24, v214 /*v470*/, v81 /*v337*/
	s_set_vgpr_msb 0x509
	v_cmp_lt_i32_e64 s9, v137 /*v393*/, v206 /*v718*/
	s_set_vgpr_msb 0x941
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[298:305]*/, v[90:97] /*v[346:353]*/, v[160:167], 0
	v_cndmask_b32_e64 v6 /*v262*/, v6 /*v262*/, 0xff800000, s15
	s_or_b32 s21, s21, s25
	s_set_vgpr_msb 0x4145
	v_cmp_gt_i32_e64 s25, v216 /*v472*/, v81 /*v337*/
	v_cndmask_b32_e64 v60 /*v316*/, v28 /*v284*/, 0xff800000, s21
	s_or_b32 s21, s23, s22
	v_cndmask_b32_e64 v7 /*v263*/, v7 /*v263*/, 0xff800000, s16
	v_cndmask_b32_e64 v61 /*v317*/, v29 /*v285*/, 0xff800000, s21
	s_set_vgpr_msb 0x4541
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x32_bf16 v[22:29] /*v[278:285]*/, v[186:193] /*v[442:449]*/, v[128:135], 0
	s_set_vgpr_msb 0x4109
	v_cmp_lt_i32_e64 s21, v214 /*v470*/, v207 /*v719*/
	s_or_b32 s13, s33, s31
	s_set_vgpr_msb 0x905
	v_cmp_gt_i32_e64 s15, v139 /*v395*/, v80 /*v336*/
	s_set_vgpr_msb 0x549
	v_cmp_lt_i32_e64 s16, v139 /*v395*/, v206 /*v718*/
	v_cndmask_b32_e64 v13 /*v269*/, v5 /*v261*/, 0xff800000, s14
	s_or_b32 s21, s21, s24
	v_cndmask_b32_e64 v17 /*v273*/, v11 /*v267*/, 0xff800000, s13
	s_set_vgpr_msb 0x4951
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x32_bf16 v[22:29] /*v[278:285]*/, v[194:201] /*v[450:457]*/, v[136:143], v[22:29] /*v[278:285]*/
	s_set_vgpr_msb 0x5140
	v_cndmask_b32_e64 v62 /*v318*/, v200, 0xff800000, s21
	s_set_vgpr_msb 0x4009
	v_cmp_lt_i32_e64 s21, v216 /*v472*/, v207 /*v719*/
	s_set_vgpr_msb 0x905
	v_cmp_gt_i32_e64 s13, v140 /*v396*/, v80 /*v336*/
	s_set_vgpr_msb 0x509
	v_cmp_lt_i32_e64 s14, v140 /*v396*/, v206 /*v718*/
	s_set_vgpr_msb 0x940
	v_cndmask_b32_e64 v77 /*v333*/, v203, 0xff800000, s18
	s_or_b32 s24, s43, s42
	s_or_b32 s17, s21, s25
	s_set_vgpr_msb 0x4051
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[22:29] /*v[278:285]*/, v[202:209] /*v[458:465]*/, v[144:151], v[22:29] /*v[278:285]*/
	s_set_vgpr_msb 0x5109
	v_cmp_lt_i32_e64 s21, v218 /*v474*/, v207 /*v719*/
	s_set_vgpr_msb 0x940
	v_cndmask_b32_e64 v76 /*v332*/, v202, 0xff800000, s17
	s_set_vgpr_msb 0x4045
	v_cmp_gt_i32_e64 s17, v219 /*v475*/, v81 /*v337*/
	v_cndmask_b32_e64 v30 /*v286*/, v70 /*v326*/, 0xff800000, s24
	v_cmp_gt_i32_e64 s22, v215 /*v471*/, v81 /*v337*/
	s_or_b32 s18, s21, s20
	s_set_vgpr_msb 0x4509
	v_cmp_lt_i32_e64 s20, v220 /*v476*/, v207 /*v719*/
	s_set_vgpr_msb 0x950
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[22:29] /*v[278:285]*/, v[192:199], v[152:159], v[22:29] /*v[278:285]*/
	v_cndmask_b32_e64 v78 /*v334*/, v204, 0xff800000, s18
	s_set_vgpr_msb 0x5005
	v_cmp_gt_i32_e64 s18, v220 /*v476*/, v81 /*v337*/
	s_set_vgpr_msb 0x509
	v_cmp_lt_i32_e64 s23, v215 /*v471*/, v207 /*v719*/
	s_set_vgpr_msb 0x900
	v_max3_num_f32 v200, v228, v229, v230
	v_max3_num_f32 v202, v231, v236, v237
	v_max3_num_f32 v204, v234, v235, v232
	s_set_vgpr_msb 0x54
	v_max3_num_f32 v4 /*v260*/, v249, v12 /*v268*/, v13 /*v269*/
	s_set_vgpr_msb 0x5451
	v_cndmask_b32_e64 v90 /*v346*/, v22 /*v278*/, 0xff800000, s2
	s_or_b32 s2, s12, s3
	s_or_b32 s3, s19, s17
	v_cndmask_b32_e64 v91 /*v347*/, v23 /*v279*/, 0xff800000, s2
	s_or_b32 s2, s11, s10
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[298:305]*/, v[98:105] /*v[354:361]*/, v[168:175], v[42:49] /*v[298:305]*/
	v_cndmask_b32_e64 v92 /*v348*/, v24 /*v280*/, 0xff800000, s2
	s_or_b32 s2, s9, s8
	s_set_vgpr_msb 0x5140
	v_cndmask_b32_e64 v79 /*v335*/, v205, 0xff800000, s3
	s_set_vgpr_msb 0x4041
	v_cndmask_b32_e64 v93 /*v349*/, v25 /*v281*/, 0xff800000, s2
	s_or_b32 s2, s7, s5
	s_or_b32 s3, s20, s18
	v_cndmask_b32_e64 v94 /*v350*/, v26 /*v282*/, 0xff800000, s2
	s_or_b32 s2, s6, s4
	s_set_vgpr_msb 0x4140
	v_cndmask_b32_e64 v98 /*v354*/, v206, 0xff800000, s3
	s_set_vgpr_msb 0x4045
	v_cndmask_b32_e64 v95 /*v351*/, v27 /*v283*/, 0xff800000, s2
	s_or_b32 s2, s16, s15
	v_cmp_gt_i32_e64 s3, v223 /*v479*/, v81 /*v337*/
	v_cndmask_b32_e64 v96 /*v352*/, v28 /*v284*/, 0xff800000, s2
	s_or_b32 s2, s14, s13
	s_set_vgpr_msb 0x4549
	v_cmp_lt_i32_e64 s4, v223 /*v479*/, v207 /*v719*/
	v_cndmask_b32_e64 v97 /*v353*/, v29 /*v285*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v221 /*v477*/, v207 /*v719*/
	s_set_vgpr_msb 0x4905
	v_cmp_gt_i32_e64 s5, v224 /*v480*/, v81 /*v337*/
	s_set_vgpr_msb 0x509
	v_cmp_lt_i32_e64 s6, v224 /*v480*/, v207 /*v719*/
	s_set_vgpr_msb 0x951
	v_wmma_f32_16x16x32_bf16 v[32:39] /*v[288:295]*/, v[82:89] /*v[338:345]*/, v[184:191], v[32:39] /*v[288:295]*/
	s_set_vgpr_msb 0x5145
	v_max_num_f32_e32 v26 /*v282*/, v30 /*v286*/, v31 /*v287*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v222 /*v478*/, v81 /*v337*/
	s_set_vgpr_msb 0x4540
	v_cndmask_b32_e64 v99 /*v355*/, v207, 0xff800000, s2
	s_set_vgpr_msb 0x4009
	v_cmp_lt_i32_e64 s2, v222 /*v478*/, v207 /*v719*/
	s_or_b32 s22, s23, s22
	s_or_b32 s23, s41, s40
	s_set_vgpr_msb 0x951
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[298:305]*/, v[106:113] /*v[362:369]*/, v[176:183], v[42:49] /*v[298:305]*/
	s_set_vgpr_msb 0x5140
	v_cndmask_b32_e64 v63 /*v319*/, v201, 0xff800000, s22
	s_or_b32 s2, s2, vcc_lo
	s_set_vgpr_msb 0x4005
	v_cmp_gt_i32_e32 vcc_lo, v225 /*v481*/, v81 /*v337*/
	s_set_vgpr_msb 0x540
	v_cndmask_b32_e64 v100 /*v356*/, v208, 0xff800000, s2
	s_or_b32 s2, s4, s3
	s_set_vgpr_msb 0x4005
	v_cmp_gt_i32_e64 s3, v226 /*v482*/, v81 /*v337*/
	s_set_vgpr_msb 0x540
	v_cndmask_b32_e64 v101 /*v357*/, v209, 0xff800000, s2
	s_or_b32 s2, s6, s5
	s_set_vgpr_msb 0x4009
	v_cmp_lt_i32_e64 s4, v226 /*v482*/, v207 /*v719*/
	s_set_vgpr_msb 0x940
	v_cndmask_b32_e64 v102 /*v358*/, v210, 0xff800000, s2
	s_set_vgpr_msb 0x4009
	v_cmp_lt_i32_e64 s2, v225 /*v481*/, v207 /*v719*/
	s_set_vgpr_msb 0x905
	v_cmp_gt_i32_e64 s5, v227 /*v483*/, v81 /*v337*/
	s_set_vgpr_msb 0x509
	v_cmp_lt_i32_e64 s6, v227 /*v483*/, v207 /*v719*/
	s_set_vgpr_msb 0x951
	v_wmma_f32_16x16x32_bf16 v[42:49] /*v[298:305]*/, v[114:121] /*v[370:377]*/, v[184:191], v[42:49] /*v[298:305]*/
	s_or_b32 s22, s39, s38
	s_or_b32 s2, s2, vcc_lo
	s_set_vgpr_msb 0x5105
	v_cmp_gt_i32_e32 vcc_lo, v228 /*v484*/, v81 /*v337*/
	s_set_vgpr_msb 0x540
	v_cndmask_b32_e64 v103 /*v359*/, v211, 0xff800000, s2
	s_or_b32 s2, s4, s3
	s_set_vgpr_msb 0x4005
	v_cmp_gt_i32_e64 s3, v229 /*v485*/, v81 /*v337*/
	s_set_vgpr_msb 0x540
	v_cndmask_b32_e64 v104 /*v360*/, v212, 0xff800000, s2
	s_or_b32 s2, s6, s5
	s_set_vgpr_msb 0x4009
	v_cmp_lt_i32_e64 s4, v229 /*v485*/, v207 /*v719*/
	s_set_vgpr_msb 0x940
	v_cndmask_b32_e64 v105 /*v361*/, v213, 0xff800000, s2
	s_set_vgpr_msb 0x4009
	v_cmp_lt_i32_e64 s2, v228 /*v484*/, v207 /*v719*/
	s_set_vgpr_msb 0x905
	v_cmp_gt_i32_e64 s5, v230 /*v486*/, v81 /*v337*/
	s_set_vgpr_msb 0x509
	v_cmp_lt_i32_e64 s6, v230 /*v486*/, v207 /*v719*/
	s_set_vgpr_msb 0x951
	v_wmma_f32_16x16x32_bf16 v[50:57] /*v[306:313]*/, v[146:153] /*v[402:409]*/, v[184:191], v[50:57] /*v[306:313]*/
	v_cndmask_b32_e64 v74 /*v330*/, v68 /*v324*/, 0xff800000, s22
	s_or_b32 s2, s2, vcc_lo
	s_set_vgpr_msb 0x5105
	v_cmp_gt_i32_e32 vcc_lo, v231 /*v487*/, v81 /*v337*/
	s_set_vgpr_msb 0x540
	v_cndmask_b32_e64 v106 /*v362*/, v214, 0xff800000, s2
	s_or_b32 s2, s4, s3
	s_set_vgpr_msb 0x4005
	v_cmp_gt_i32_e64 s3, v232 /*v488*/, v81 /*v337*/
	s_set_vgpr_msb 0x540
	v_cndmask_b32_e64 v107 /*v363*/, v215, 0xff800000, s2
	s_or_b32 s2, s6, s5
	s_set_vgpr_msb 0x4049
	v_cmp_lt_i32_e64 s4, v232 /*v488*/, v207 /*v719*/
	v_cndmask_b32_e64 v108 /*v364*/, v32 /*v288*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v231 /*v487*/, v207 /*v719*/
	s_set_vgpr_msb 0x4905
	v_cmp_gt_i32_e64 s5, v233 /*v489*/, v81 /*v337*/
	s_set_vgpr_msb 0x509
	v_cmp_lt_i32_e64 s6, v233 /*v489*/, v207 /*v719*/
	s_set_vgpr_msb 0x941
	v_wmma_f32_16x16x32_bf16 v[82:89] /*v[338:345]*/, v[154:161] /*v[410:417]*/, v[160:167], 0
	v_cndmask_b32_e64 v75 /*v331*/, v69 /*v325*/, 0xff800000, s23
	s_or_b32 s2, s2, vcc_lo
	s_set_vgpr_msb 0x4145
	v_cmp_gt_i32_e32 vcc_lo, v234 /*v490*/, v81 /*v337*/
	v_cndmask_b32_e64 v109 /*v365*/, v33 /*v289*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	v_cmp_gt_i32_e64 s3, v235 /*v491*/, v81 /*v337*/
	v_cndmask_b32_e64 v110 /*v366*/, v34 /*v290*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	s_set_vgpr_msb 0x4549
	v_cmp_lt_i32_e64 s4, v235 /*v491*/, v207 /*v719*/
	v_cndmask_b32_e64 v111 /*v367*/, v35 /*v291*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v234 /*v490*/, v207 /*v719*/
	s_set_vgpr_msb 0x4905
	v_cmp_gt_i32_e64 s5, v236 /*v492*/, v81 /*v337*/
	s_set_vgpr_msb 0x509
	v_cmp_lt_i32_e64 s6, v236 /*v492*/, v207 /*v719*/
	s_set_vgpr_msb 0x951
	v_wmma_f32_16x16x32_bf16 v[82:89] /*v[338:345]*/, v[162:169] /*v[418:425]*/, v[168:175], v[82:89] /*v[338:345]*/
	s_set_vgpr_msb 0x5100
	v_max3_num_f32 v206, v233, v238, v239
	s_or_b32 s2, s2, vcc_lo
	s_set_vgpr_msb 0x45
	v_cmp_gt_i32_e32 vcc_lo, v237 /*v493*/, v81 /*v337*/
	v_cndmask_b32_e64 v112 /*v368*/, v36 /*v292*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	v_cmp_gt_i32_e64 s3, v238 /*v494*/, v81 /*v337*/
	v_cndmask_b32_e64 v113 /*v369*/, v37 /*v293*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	s_set_vgpr_msb 0x4549
	v_cmp_lt_i32_e64 s4, v238 /*v494*/, v207 /*v719*/
	v_cndmask_b32_e64 v114 /*v370*/, v38 /*v294*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v237 /*v493*/, v207 /*v719*/
	s_set_vgpr_msb 0x4905
	v_cmp_gt_i32_e64 s5, v239 /*v495*/, v81 /*v337*/
	s_set_vgpr_msb 0x509
	v_cmp_lt_i32_e64 s6, v239 /*v495*/, v207 /*v719*/
	s_set_vgpr_msb 0x951
	v_wmma_f32_16x16x32_bf16 v[82:89] /*v[338:345]*/, v[170:177] /*v[426:433]*/, v[176:183], v[82:89] /*v[338:345]*/
	s_set_vgpr_msb 0x5100
	v_max3_num_f32 v208, v246, v247, v244
	s_or_b32 s2, s2, vcc_lo
	s_set_vgpr_msb 0x45
	v_cmp_gt_i32_e32 vcc_lo, v240 /*v496*/, v81 /*v337*/
	v_cndmask_b32_e64 v115 /*v371*/, v39 /*v295*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	v_cmp_gt_i32_e64 s3, v241 /*v497*/, v81 /*v337*/
	v_cndmask_b32_e64 v116 /*v372*/, v42 /*v298*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	s_set_vgpr_msb 0x4549
	v_cmp_lt_i32_e64 s4, v241 /*v497*/, v207 /*v719*/
	v_cndmask_b32_e64 v117 /*v373*/, v43 /*v299*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v240 /*v496*/, v207 /*v719*/
	s_set_vgpr_msb 0x4905
	v_cmp_gt_i32_e64 s5, v242 /*v498*/, v81 /*v337*/
	s_set_vgpr_msb 0x509
	v_cmp_lt_i32_e64 s6, v242 /*v498*/, v207 /*v719*/
	s_set_vgpr_msb 0x951
	v_wmma_f32_16x16x32_bf16 v[82:89] /*v[338:345]*/, v[178:185] /*v[434:441]*/, v[184:191], v[82:89] /*v[338:345]*/
	s_set_vgpr_msb 0x5100
	v_max3_num_f32 v210, v245, v242, v243
	s_or_b32 s2, s2, vcc_lo
	s_set_vgpr_msb 0x45
	v_cmp_gt_i32_e32 vcc_lo, v243 /*v499*/, v81 /*v337*/
	v_cndmask_b32_e64 v118 /*v374*/, v44 /*v300*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	v_cmp_gt_i32_e64 s3, v244 /*v500*/, v81 /*v337*/
	v_cndmask_b32_e64 v119 /*v375*/, v45 /*v301*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	s_set_vgpr_msb 0x4549
	v_cmp_lt_i32_e64 s4, v244 /*v500*/, v207 /*v719*/
	v_cndmask_b32_e64 v120 /*v376*/, v46 /*v302*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v243 /*v499*/, v207 /*v719*/
	s_set_vgpr_msb 0x4905
	v_cmp_gt_i32_e64 s5, v245 /*v501*/, v81 /*v337*/
	s_set_vgpr_msb 0x509
	v_cmp_lt_i32_e64 s6, v245 /*v501*/, v207 /*v719*/
	s_set_vgpr_msb 0x941
	v_wmma_f32_16x16x32_bf16 v[64:71] /*v[320:327]*/, v[186:193] /*v[442:449]*/, v[160:167], 0
	s_set_vgpr_msb 0x4100
	v_max3_num_f32 v212, v240, v241, v254
	s_or_b32 s2, s2, vcc_lo
	s_set_vgpr_msb 0x45
	v_cmp_gt_i32_e32 vcc_lo, v246 /*v502*/, v81 /*v337*/
	v_cndmask_b32_e64 v121 /*v377*/, v47 /*v303*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	v_cmp_gt_i32_e64 s3, v247 /*v503*/, v81 /*v337*/
	v_cndmask_b32_e64 v122 /*v378*/, v48 /*v304*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	s_set_vgpr_msb 0x4549
	v_cmp_lt_i32_e64 s4, v247 /*v503*/, v207 /*v719*/
	v_cndmask_b32_e64 v123 /*v379*/, v49 /*v305*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v246 /*v502*/, v207 /*v719*/
	s_set_vgpr_msb 0x4905
	v_cmp_gt_i32_e64 s5, v248 /*v504*/, v81 /*v337*/
	s_set_vgpr_msb 0x509
	v_cmp_lt_i32_e64 s6, v248 /*v504*/, v207 /*v719*/
	s_set_vgpr_msb 0x951
	v_wmma_f32_16x16x32_bf16 v[64:71] /*v[320:327]*/, v[194:201] /*v[450:457]*/, v[168:175], v[64:71] /*v[320:327]*/
	s_set_vgpr_msb 0x5100
	v_max3_num_f32 v214, v255, v252, v253
	s_or_b32 s2, s2, vcc_lo
	s_set_vgpr_msb 0x45
	v_cmp_gt_i32_e32 vcc_lo, v249 /*v505*/, v81 /*v337*/
	v_cndmask_b32_e64 v124 /*v380*/, v50 /*v306*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	v_cmp_gt_i32_e64 s3, v250 /*v506*/, v81 /*v337*/
	v_cndmask_b32_e64 v125 /*v381*/, v51 /*v307*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	s_set_vgpr_msb 0x4549
	v_cmp_lt_i32_e64 s4, v250 /*v506*/, v207 /*v719*/
	v_cndmask_b32_e64 v126 /*v382*/, v52 /*v308*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v249 /*v505*/, v207 /*v719*/
	s_set_vgpr_msb 0x4905
	v_cmp_gt_i32_e64 s5, v251 /*v507*/, v81 /*v337*/
	s_set_vgpr_msb 0x509
	v_cmp_lt_i32_e64 s6, v251 /*v507*/, v207 /*v719*/
	s_set_vgpr_msb 0x951
	v_wmma_f32_16x16x32_bf16 v[64:71] /*v[320:327]*/, v[202:209] /*v[458:465]*/, v[176:183], v[64:71] /*v[320:327]*/
	s_set_vgpr_msb 0x5155
	v_max3_num_f32 v8 /*v264*/, v6 /*v262*/, v7 /*v263*/, v20 /*v276*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v252 /*v508*/, v81 /*v337*/
	v_cndmask_b32_e64 v127 /*v383*/, v53 /*v309*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	v_cmp_gt_i32_e64 s3, v253 /*v509*/, v81 /*v337*/
	v_cndmask_b32_e64 v128 /*v384*/, v54 /*v310*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	s_set_vgpr_msb 0x5549
	v_cmp_lt_i32_e64 s4, v253 /*v509*/, v207 /*v719*/
	v_cndmask_b32_e64 v129 /*v385*/, v55 /*v311*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v252 /*v508*/, v207 /*v719*/
	s_set_vgpr_msb 0x4905
	v_cmp_gt_i32_e64 s5, v254 /*v510*/, v81 /*v337*/
	s_set_vgpr_msb 0x509
	v_cmp_lt_i32_e64 s6, v254 /*v510*/, v207 /*v719*/
	s_set_vgpr_msb 0x950
	v_wmma_f32_16x16x32_bf16 v[64:71] /*v[320:327]*/, v[192:199], v[184:191], v[64:71] /*v[320:327]*/
	s_set_vgpr_msb 0x5055
	v_max3_num_f32 v10 /*v266*/, v21 /*v277*/, v16 /*v272*/, v17 /*v273*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v255 /*v511*/, v81 /*v337*/
	v_cndmask_b32_e64 v130 /*v386*/, v56 /*v312*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	s_set_vgpr_msb 0x5506
	v_cmp_gt_i32_e64 s3, v0 /*v512*/, v81 /*v337*/
	s_set_vgpr_msb 0x641
	v_cndmask_b32_e64 v131 /*v387*/, v57 /*v313*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	s_set_vgpr_msb 0x410a
	v_cmp_lt_i32_e64 s4, v0 /*v512*/, v207 /*v719*/
	s_set_vgpr_msb 0xa49
	v_cndmask_b32_e64 v132 /*v388*/, v82 /*v338*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v255 /*v511*/, v207 /*v719*/
	v_cmp_lt_i32_e64 s5, v81 /*v337*/, v1 /*v513*/
	s_set_vgpr_msb 0x490a
	v_cmp_lt_i32_e64 s6, v1 /*v513*/, v207 /*v719*/
	s_set_vgpr_msb 0xa00
	v_max3_num_f32 v192, v216, v217, v218
	s_set_vgpr_msb 21
	v_max3_num_f32 v193, v14 /*v270*/, v15 /*v271*/, v18 /*v274*/
	s_or_b32 s2, s2, vcc_lo
	s_set_vgpr_msb 0x1506
	v_cmp_gt_i32_e32 vcc_lo, v2 /*v514*/, v81 /*v337*/
	s_set_vgpr_msb 0x645
	v_cndmask_b32_e64 v133 /*v389*/, v83 /*v339*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	v_cmp_gt_i32_e64 s3, v2 /*v258*/, v81 /*v337*/
	v_cndmask_b32_e64 v84 /*v340*/, v84 /*v340*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	s_set_vgpr_msb 0x4549
	v_cmp_lt_i32_e64 s4, v2 /*v258*/, v207 /*v719*/
	v_cndmask_b32_e64 v85 /*v341*/, v85 /*v341*/, 0xff800000, s2
	s_set_vgpr_msb 0x490a
	v_cmp_lt_i32_e64 s2, v2 /*v514*/, v207 /*v719*/
	s_set_vgpr_msb 0xa05
	v_cmp_gt_i32_e64 s5, v3 /*v259*/, v81 /*v337*/
	s_set_vgpr_msb 0x509
	v_cmp_lt_i32_e64 s6, v3 /*v259*/, v207 /*v719*/
	s_set_vgpr_msb 0x900
	v_max3_num_f32 v194, v219, v220, v221
	s_set_vgpr_msb 21
	v_max3_num_f32 v195, v19 /*v275*/, v40 /*v296*/, v41 /*v297*/
	s_or_b32 s2, s2, vcc_lo
	s_set_vgpr_msb 0x1506
	v_cmp_gt_i32_e32 vcc_lo, v3 /*v515*/, v81 /*v337*/
	s_set_vgpr_msb 0x641
	v_cndmask_b32_e64 v86 /*v342*/, v86 /*v342*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	s_set_vgpr_msb 0x4106
	v_cmp_gt_i32_e64 s3, v4 /*v516*/, v81 /*v337*/
	s_set_vgpr_msb 0x641
	v_cndmask_b32_e64 v87 /*v343*/, v87 /*v343*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	s_set_vgpr_msb 0x410a
	v_cmp_lt_i32_e64 s4, v4 /*v516*/, v207 /*v719*/
	s_set_vgpr_msb 0xa41
	v_cndmask_b32_e64 v88 /*v344*/, v88 /*v344*/, 0xff800000, s2
	s_set_vgpr_msb 0x410a
	v_cmp_lt_i32_e64 s2, v3 /*v515*/, v207 /*v719*/
	s_set_vgpr_msb 0xa05
	v_cmp_gt_i32_e64 s5, v134 /*v390*/, v81 /*v337*/
	s_set_vgpr_msb 0x509
	v_cmp_lt_i32_e64 s6, v134 /*v390*/, v207 /*v719*/
	s_set_vgpr_msb 0x900
	v_max3_num_f32 v196, v222, v223, v224
	s_set_vgpr_msb 21
	v_max3_num_f32 v197, v60 /*v316*/, v61 /*v317*/, v62 /*v318*/
	s_or_b32 s2, s2, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, v136 /*v392*/, v81 /*v337*/
	s_set_vgpr_msb 0x1545
	v_cndmask_b32_e64 v89 /*v345*/, v89 /*v345*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	v_cmp_gt_i32_e64 s3, v137 /*v393*/, v81 /*v337*/
	v_cndmask_b32_e64 v134 /*v390*/, v64 /*v320*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	s_set_vgpr_msb 0x4549
	v_cmp_lt_i32_e64 s4, v137 /*v393*/, v207 /*v719*/
	v_cndmask_b32_e64 v135 /*v391*/, v65 /*v321*/, 0xff800000, s2
	v_cmp_lt_i32_e64 s2, v136 /*v392*/, v207 /*v719*/
	v_cmp_lt_i32_e64 s5, v81 /*v337*/, v5 /*v517*/
	s_set_vgpr_msb 0x490a
	v_cmp_lt_i32_e64 s6, v5 /*v517*/, v207 /*v719*/
	s_set_vgpr_msb 0xa00
	v_max3_num_f32 v198, v225, v226, v227
	s_set_vgpr_msb 64
	v_max3_num_f32 v2 /*v258*/, v250, v251, v248
	s_or_b32 s2, s2, vcc_lo
	s_set_vgpr_msb 0x4006
	v_cmp_gt_i32_e32 vcc_lo, v6 /*v518*/, v81 /*v337*/
	s_set_vgpr_msb 0x645
	v_cndmask_b32_e64 v136 /*v392*/, v66 /*v322*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	v_cmp_gt_i32_e64 s3, v139 /*v395*/, v81 /*v337*/
	v_cndmask_b32_e64 v137 /*v393*/, v67 /*v323*/, 0xff800000, s2
	s_or_b32 s2, s6, s5
	s_set_vgpr_msb 0x4549
	v_cmp_lt_i32_e64 s4, v139 /*v395*/, v207 /*v719*/
	v_cndmask_b32_e64 v138 /*v394*/, v68 /*v324*/, 0xff800000, s2
	s_set_vgpr_msb 0x490a
	v_cmp_lt_i32_e64 s2, v6 /*v518*/, v207 /*v719*/
	s_set_vgpr_msb 0xa55
	v_max3_num_f32 v22 /*v278*/, v58 /*v314*/, v59 /*v315*/, v72 /*v328*/
	v_max3_num_f32 v24 /*v280*/, v73 /*v329*/, v74 /*v330*/, v75 /*v331*/
	v_max3_num_f32 v28 /*v284*/, v91 /*v347*/, v92 /*v348*/, v93 /*v349*/
	v_cmp_gt_i32_e64 s5, v140 /*v396*/, v81 /*v337*/
	s_or_b32 s2, s2, vcc_lo
	s_set_vgpr_msb 0x5549
	v_cmp_lt_i32_e64 s6, v140 /*v396*/, v207 /*v719*/
	v_cndmask_b32_e64 v139 /*v395*/, v69 /*v325*/, 0xff800000, s2
	s_or_b32 s2, s4, s3
	s_set_vgpr_msb 0x4915
	v_max3_num_f32 v199, v63 /*v319*/, v76 /*v332*/, v77 /*v333*/
	s_set_vgpr_msb 0x1541
	v_cndmask_b32_e64 v140 /*v396*/, v70 /*v326*/, 0xff800000, s2
	s_set_vgpr_msb 0x4115
	v_max3_num_f32 v201, v78 /*v334*/, v79 /*v335*/, v98 /*v354*/
	v_max3_num_f32 v203, v99 /*v355*/, v100 /*v356*/, v101 /*v357*/
	v_max3_num_f32 v205, v102 /*v358*/, v103 /*v359*/, v104 /*v360*/
	v_max3_num_f32 v207, v105 /*v361*/, v106 /*v362*/, v107 /*v363*/
	v_max3_num_f32 v209, v108 /*v364*/, v109 /*v365*/, v110 /*v366*/
	v_max3_num_f32 v211, v111 /*v367*/, v112 /*v368*/, v113 /*v369*/
	v_max3_num_f32 v213, v114 /*v370*/, v115 /*v371*/, v116 /*v372*/
	v_max3_num_f32 v215, v117 /*v373*/, v118 /*v374*/, v119 /*v375*/
	s_set_vgpr_msb 0x1555
	v_max3_num_f32 v3 /*v259*/, v120 /*v376*/, v121 /*v377*/, v122 /*v378*/
	v_max3_num_f32 v5 /*v261*/, v123 /*v379*/, v124 /*v380*/, v125 /*v381*/
	v_max3_num_f32 v9 /*v265*/, v126 /*v382*/, v127 /*v383*/, v128 /*v384*/
	v_max3_num_f32 v11 /*v267*/, v129 /*v385*/, v130 /*v386*/, v131 /*v387*/
	v_max3_num_f32 v23 /*v279*/, v132 /*v388*/, v133 /*v389*/, v84 /*v340*/
	v_max3_num_f32 v25 /*v281*/, v85 /*v341*/, v86 /*v342*/, v87 /*v343*/
	v_max_num_f32_e32 v27 /*v283*/, v88 /*v344*/, v89 /*v345*/
	v_max3_num_f32 v29 /*v285*/, v135 /*v391*/, v136 /*v392*/, v137 /*v393*/
	v_max3_num_f32 v32 /*v288*/, v94 /*v350*/, v95 /*v351*/, v96 /*v352*/
	s_set_vgpr_msb 0x5500
	v_max3_num_f32 v192, v192, v194, v196
	v_max3_num_f32 v193, v193, v195, v197
	v_max3_num_f32 v194, v198, v200, v202
	v_max3_num_f32 v195, v204, v206, v208
	v_max3_num_f32 v196, v210, v212, v214
	s_set_vgpr_msb 21
	v_max3_num_f32 v197, v2 /*v258*/, v4 /*v260*/, v8 /*v264*/
	v_max3_num_f32 v198, v10 /*v266*/, v22 /*v278*/, v24 /*v280*/
	v_max3_num_f32 v200, v26 /*v282*/, v90 /*v346*/, v28 /*v284*/
	s_or_b32 s2, s6, s5
	s_set_vgpr_msb 0x1555
	v_max3_num_f32 v33 /*v289*/, v138 /*v394*/, v139 /*v395*/, v140 /*v396*/
	v_cndmask_b32_e64 v141 /*v397*/, v71 /*v327*/, 0xff800000, s2
	s_set_vgpr_msb 0x5500
	v_max3_num_f32 v199, v199, v201, v203
	v_max3_num_f32 v201, v205, v207, v209
	v_max3_num_f32 v192, v192, v194, v195
	v_max3_num_f32 v194, v196, v197, v198
	s_set_vgpr_msb 20
	v_max3_num_f32 v195, v200, v32 /*v288*/, v97 /*v353*/
	s_set_vgpr_msb 0x1400
	v_max3_num_f32 v196, v211, v213, v215
	s_set_vgpr_msb 21
	v_max3_num_f32 v197, v3 /*v259*/, v5 /*v261*/, v9 /*v265*/
	v_max3_num_f32 v198, v11 /*v267*/, v23 /*v279*/, v25 /*v281*/
	v_max3_num_f32 v200, v27 /*v283*/, v134 /*v390*/, v29 /*v285*/
	s_set_vgpr_msb 0x1500
	v_max3_num_f32 v192, v192, v194, v195
	v_max3_num_f32 v193, v193, v199, v201
	s_set_vgpr_msb 0x42
	v_mov_b32_e32 v83 /*v339*/, v209 /*v721*/
	s_set_vgpr_msb 0x4200
	v_max3_num_f32 v194, v196, v197, v198
	s_set_vgpr_msb 20
	v_max3_num_f32 v195, v200, v33 /*v289*/, v141 /*v397*/
	s_set_vgpr_msb 0x1400
	v_max3_num_f32 v193, v193, v194, v195
	v_dual_mov_b32 v196, v192 :: v_dual_mov_b32 v194, v193
	v_permlanex16_b32 v196, v196, s61, 0xfedcba98
	v_permlanex16_b32 v194, v194, s61, 0xfedcba98
	v_dual_max_num_f32 v192, v192, v196 :: v_dual_max_num_f32 v193, v193, v194
	s_set_vgpr_msb 4
	v_sub_f32_e32 v195, v192, v83 /*v339*/
	v_max_num_f32_e32 v192, v192, v83 /*v339*/
	s_set_vgpr_msb 0x408
	v_sub_f32_e32 v194, v193, v211 /*v723*/
	s_set_vgpr_msb 0x800
	v_cmp_lt_f32_e32 vcc_lo, 0x41000000, v195
	s_cmp_eq_u32 vcc_lo, 0
	s_cselect_b32 s2, -1, 0
	s_set_vgpr_msb 0x84
	v_cndmask_b32_e64 v209 /*v721*/, v192, v83 /*v339*/, s2
	s_set_vgpr_msb 0x8402
	v_cmp_lt_f32_e64 s2, 0x41000000, v194
	v_max_num_f32_e32 v192, v211 /*v723*/, v193
	s_set_vgpr_msb 0x248
	v_mul_f32_e32 v68 /*v324*/, 0xbfb8aa3b, v209 /*v721*/
	s_cmp_lg_u32 s2, 0
	s_cselect_b32 s3, -1, 0
	s_cmp_eq_u32 s2, 0
	s_set_vgpr_msb 0x4810
	v_pk_fma_f32 v[200:201], v[220:221], s[60:61], v[68:69] /*v[324:325]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[220:221], v[246:247], s[60:61], v[68:69] /*v[324:325]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[206:207], v[224:225], s[60:61], v[68:69] /*v[324:325]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[208:209], v[226:227], s[60:61], v[68:69] /*v[324:325]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[212:213], v[232:233], s[60:61], v[68:69] /*v[324:325]*/ op_sel_hi:[1,0,0]
	s_cselect_b32 s2, -1, 0
	v_exp_f32_e32 v226, v220
	v_exp_f32_e32 v232, v221
	v_nop
	v_pk_fma_f32 v[220:221], v[240:241], s[60:61], v[68:69] /*v[324:325]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[240:241], v[248:249], s[60:61], v[68:69] /*v[324:325]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1040
	v_exp_f32_e32 v8 /*v264*/, v206
	v_exp_f32_e32 v22 /*v278*/, v207
	v_nop
	s_set_vgpr_msb 0x4010
	v_pk_fma_f32 v[206:207], v[230:231], s[60:61], v[68:69] /*v[324:325]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[230:231], v[250:251], s[60:61], v[68:69] /*v[324:325]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v248, v240
	s_set_vgpr_msb 0x1011
	v_pk_fma_f32 v[250:251], v[6:7] /*v[262:263]*/, s[60:61], v[68:69] /*v[324:325]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1140
	v_exp_f32_e32 v6 /*v262*/, v241
	v_nop
	s_set_vgpr_msb 0x4011
	v_pk_fma_f32 v[240:241], v[20:21] /*v[276:277]*/, s[60:61], v[68:69] /*v[324:325]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1148
	v_cndmask_b32_e64 v82 /*v338*/, v192, v211 /*v723*/, s2
	s_set_vgpr_msb 0x4810
	v_pk_fma_f32 v[204:205], v[222:223], s[60:61], v[68:69] /*v[324:325]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[222:223], v[244:245], s[60:61], v[68:69] /*v[324:325]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1011
	v_pk_fma_f32 v[244:245], v[12:13] /*v[268:269]*/, s[60:61], v[68:69] /*v[324:325]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1140
	v_exp_f32_e32 v38 /*v294*/, v240
	v_exp_f32_e32 v44 /*v300*/, v241
	v_nop
	s_set_vgpr_msb 0x4011
	v_pk_fma_f32 v[240:241], v[72:73] /*v[328:329]*/, s[60:61], v[68:69] /*v[324:325]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1144
	v_mul_f32_e32 v142 /*v398*/, 0xbfb8aa3b, v82 /*v338*/
	v_exp_f32_e32 v2 /*v258*/, v204
	v_exp_f32_e32 v4 /*v260*/, v205
	v_nop
	s_set_vgpr_msb 0x4410
	v_pk_fma_f32 v[204:205], v[228:229], s[60:61], v[68:69] /*v[324:325]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[228:229], v[252:253], s[60:61], v[68:69] /*v[324:325]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1040
	v_exp_f32_e32 v12 /*v268*/, v244
	v_exp_f32_e32 v24 /*v280*/, v245
	v_nop
	s_set_vgpr_msb 0x4011
	v_pk_fma_f32 v[244:245], v[16:17] /*v[272:273]*/, s[60:61], v[68:69] /*v[324:325]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1100
	v_exp_f32_e32 v252, v240
	s_set_vgpr_msb 64
	v_exp_f32_e32 v16 /*v272*/, v241
	v_nop
	s_set_vgpr_msb 0x4011
	v_pk_fma_f32 v[240:241], v[90:91] /*v[346:347]*/, s[60:61], v[68:69] /*v[324:325]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1110
	v_pk_fma_f32 v[192:193], v[216:217], s[60:61], v[68:69] /*v[324:325]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1040
	v_exp_f32_e32 v28 /*v284*/, v250
	v_exp_f32_e32 v34 /*v290*/, v251
	v_nop
	s_set_vgpr_msb 0x4011
	v_pk_fma_f32 v[250:251], v[58:59] /*v[314:315]*/, s[60:61], v[68:69] /*v[324:325]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1140
	v_exp_f32_e32 v58 /*v314*/, v240
	v_exp_f32_e32 v64 /*v320*/, v241
	v_nop
	s_set_vgpr_msb 0x4011
	v_pk_fma_f32 v[240:241], v[96:97] /*v[352:353]*/, s[60:61], v[68:69] /*v[324:325]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1151
	v_pk_fma_f32 v[14:15] /*v[270:271]*/, v[14:15] /*v[270:271]*/, s[60:61], v[142:143] /*v[398:399]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5100
	v_exp_f32_e32 v194, v193
	s_set_vgpr_msb 0x51
	v_pk_fma_f32 v[20:21] /*v[276:277]*/, v[74:75] /*v[330:331]*/, s[60:61], v[68:69] /*v[324:325]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[90:91] /*v[346:347]*/, v[18:19] /*v[274:275]*/, s[60:61], v[142:143] /*v[398:399]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5140
	v_exp_f32_e32 v74 /*v330*/, v240
	v_exp_f32_e32 v18 /*v274*/, v241
	s_set_vgpr_msb 0x4011
	v_exp_f32_e32 v193, v14 /*v270*/
	v_pk_fma_f32 v[240:241], v[40:41] /*v[296:297]*/, s[60:61], v[142:143] /*v[398:399]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v195, v15 /*v271*/
	v_nop
	s_set_vgpr_msb 0x1151
	v_pk_fma_f32 v[14:15] /*v[270:271]*/, v[60:61] /*v[316:317]*/, s[60:61], v[142:143] /*v[398:399]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[40:41] /*v[296:297]*/, v[62:63] /*v[318:319]*/, s[60:61], v[142:143] /*v[398:399]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5100
	v_exp_f32_e32 v202, v201
	v_exp_f32_e32 v201, v240
	v_exp_f32_e32 v203, v241
	s_set_vgpr_msb 0x51
	v_exp_f32_e32 v3 /*v259*/, v14 /*v270*/
	v_exp_f32_e32 v5 /*v261*/, v15 /*v271*/
	v_exp_f32_e32 v9 /*v265*/, v40 /*v296*/
	v_pk_fma_f32 v[14:15] /*v[270:271]*/, v[78:79] /*v[334:335]*/, s[60:61], v[142:143] /*v[398:399]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v23 /*v279*/, v41 /*v297*/
	v_nop
	v_pk_fma_f32 v[40:41] /*v[296:297]*/, v[98:99] /*v[354:355]*/, s[60:61], v[142:143] /*v[398:399]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5111
	v_pk_fma_f32 v[240:241], v[76:77] /*v[332:333]*/, s[60:61], v[142:143] /*v[398:399]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1140
	v_exp_f32_e32 v26 /*v282*/, v208
	v_exp_f32_e32 v32 /*v288*/, v209
	v_nop
	s_set_vgpr_msb 0x4010
	v_pk_fma_f32 v[208:209], v[236:237], s[60:61], v[68:69] /*v[324:325]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[210:211], v[234:235], s[60:61], v[68:69] /*v[324:325]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1051
	v_exp_f32_e32 v37 /*v293*/, v14 /*v270*/
	v_exp_f32_e32 v43 /*v299*/, v15 /*v271*/
	v_exp_f32_e32 v49 /*v305*/, v40 /*v296*/
	v_pk_fma_f32 v[14:15] /*v[270:271]*/, v[102:103] /*v[358:359]*/, s[60:61], v[142:143] /*v[398:399]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v55 /*v311*/, v41 /*v297*/
	v_nop
	v_pk_fma_f32 v[40:41] /*v[296:297]*/, v[104:105] /*v[360:361]*/, s[60:61], v[142:143] /*v[398:399]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5140
	v_exp_f32_e32 v27 /*v283*/, v240
	v_exp_f32_e32 v33 /*v289*/, v241
	v_nop
	s_set_vgpr_msb 0x4011
	v_pk_fma_f32 v[240:241], v[100:101] /*v[356:357]*/, s[60:61], v[142:143] /*v[398:399]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1140
	v_exp_f32_e32 v36 /*v292*/, v204
	v_exp_f32_e32 v48 /*v304*/, v206
	s_set_vgpr_msb 0x4010
	v_exp_f32_e32 v204, v208
	v_exp_f32_e32 v206, v209
	v_exp_f32_e32 v208, v210
	v_pk_fma_f32 v[216:217], v[238:239], s[60:61], v[68:69] /*v[324:325]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v210, v211
	v_exp_f32_e32 v214, v213
	s_set_vgpr_msb 0x1001
	v_exp_f32_e32 v209, v14 /*v270*/
	v_exp_f32_e32 v211, v15 /*v271*/
	v_exp_f32_e32 v213, v40 /*v296*/
	s_set_vgpr_msb 0x151
	v_pk_fma_f32 v[14:15] /*v[270:271]*/, v[108:109] /*v[364:365]*/, s[60:61], v[142:143] /*v[398:399]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5101
	v_exp_f32_e32 v215, v41 /*v297*/
	v_nop
	s_set_vgpr_msb 0x151
	v_pk_fma_f32 v[40:41] /*v[296:297]*/, v[110:111] /*v[366:367]*/, s[60:61], v[142:143] /*v[398:399]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5140
	v_exp_f32_e32 v42 /*v298*/, v205
	v_exp_f32_e32 v54 /*v310*/, v207
	s_set_vgpr_msb 0x4000
	v_exp_f32_e32 v205, v240
	v_exp_f32_e32 v207, v241
	v_nop
	s_set_vgpr_msb 17
	v_pk_fma_f32 v[240:241], v[106:107] /*v[362:363]*/, s[60:61], v[142:143] /*v[398:399]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1110
	v_pk_fma_f32 v[196:197], v[218:219], s[60:61], v[68:69] /*v[324:325]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v218, v216
	v_exp_f32_e32 v224, v217
	v_nop
	v_pk_fma_f32 v[216:217], v[242:243], s[60:61], v[68:69] /*v[324:325]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1001
	v_exp_f32_e32 v227, v14 /*v270*/
	v_exp_f32_e32 v233, v15 /*v271*/
	v_exp_f32_e32 v235, v40 /*v296*/
	s_set_vgpr_msb 0x151
	v_pk_fma_f32 v[14:15] /*v[270:271]*/, v[114:115] /*v[370:371]*/, s[60:61], v[142:143] /*v[398:399]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5101
	v_exp_f32_e32 v239, v41 /*v297*/
	v_nop
	s_set_vgpr_msb 0x151
	v_pk_fma_f32 v[40:41] /*v[296:297]*/, v[116:117] /*v[372:373]*/, s[60:61], v[142:143] /*v[398:399]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5100
	v_exp_f32_e32 v219, v240
	v_exp_f32_e32 v225, v241
	v_nop
	s_set_vgpr_msb 17
	v_pk_fma_f32 v[240:241], v[112:113] /*v[368:369]*/, s[60:61], v[142:143] /*v[398:399]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1110
	v_exp_f32_e32 v234, v222
	v_exp_f32_e32 v238, v223
	v_nop
	v_pk_fma_f32 v[222:223], v[254:255], s[60:61], v[68:69] /*v[324:325]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v246, v217
	s_set_vgpr_msb 0x1040
	v_exp_f32_e32 v10 /*v266*/, v221
	s_set_vgpr_msb 0x4001
	v_exp_f32_e32 v255, v14 /*v270*/
	s_set_vgpr_msb 0x141
	v_exp_f32_e32 v11 /*v267*/, v15 /*v271*/
	s_set_vgpr_msb 0x4101
	v_exp_f32_e32 v217, v40 /*v296*/
	s_set_vgpr_msb 0x151
	v_pk_fma_f32 v[14:15] /*v[270:271]*/, v[120:121] /*v[376:377]*/, s[60:61], v[142:143] /*v[398:399]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5101
	v_exp_f32_e32 v221, v41 /*v297*/
	v_nop
	s_set_vgpr_msb 0x151
	v_pk_fma_f32 v[40:41] /*v[296:297]*/, v[122:123] /*v[378:379]*/, s[60:61], v[142:143] /*v[398:399]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5100
	v_exp_f32_e32 v243, v240
	v_exp_f32_e32 v247, v241
	v_nop
	s_set_vgpr_msb 17
	v_pk_fma_f32 v[240:241], v[118:119] /*v[374:375]*/, s[60:61], v[142:143] /*v[398:399]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1100
	v_exp_f32_e32 v236, v231
	s_set_vgpr_msb 1
	v_exp_f32_e32 v231, v14 /*v270*/
	v_exp_f32_e32 v237, v15 /*v271*/
	v_exp_f32_e32 v249, v40 /*v296*/
	s_set_vgpr_msb 0x151
	v_pk_fma_f32 v[14:15] /*v[270:271]*/, v[126:127] /*v[382:383]*/, s[60:61], v[142:143] /*v[398:399]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v7 /*v263*/, v41 /*v297*/
	v_nop
	v_pk_fma_f32 v[40:41] /*v[296:297]*/, v[128:129] /*v[384:385]*/, s[60:61], v[142:143] /*v[398:399]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5100
	v_exp_f32_e32 v242, v216
	v_exp_f32_e32 v254, v220
	v_exp_f32_e32 v216, v222
	v_exp_f32_e32 v220, v223
	v_exp_f32_e32 v222, v228
	v_exp_f32_e32 v228, v229
	v_exp_f32_e32 v223, v240
	v_exp_f32_e32 v229, v241
	v_nop
	s_set_vgpr_msb 17
	v_pk_fma_f32 v[240:241], v[124:125] /*v[380:381]*/, s[60:61], v[142:143] /*v[398:399]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1151
	v_exp_f32_e32 v29 /*v285*/, v14 /*v270*/
	v_exp_f32_e32 v35 /*v291*/, v15 /*v271*/
	v_exp_f32_e32 v39 /*v295*/, v40 /*v296*/
	v_pk_fma_f32 v[14:15] /*v[270:271]*/, v[132:133] /*v[388:389]*/, s[60:61], v[142:143] /*v[398:399]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v45 /*v301*/, v41 /*v297*/
	v_nop
	v_pk_fma_f32 v[40:41] /*v[296:297]*/, v[84:85] /*v[340:341]*/, s[60:61], v[142:143] /*v[398:399]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5100
	v_exp_f32_e32 v192, v192
	v_exp_f32_e32 v198, v197
	v_exp_f32_e32 v200, v200
	s_set_vgpr_msb 1
	v_exp_f32_e32 v199, v91 /*v347*/
	s_set_vgpr_msb 0x140
	v_exp_f32_e32 v13 /*v269*/, v240
	v_exp_f32_e32 v25 /*v281*/, v241
	v_nop
	s_set_vgpr_msb 0x4011
	v_pk_fma_f32 v[240:241], v[130:131] /*v[386:387]*/, s[60:61], v[142:143] /*v[398:399]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1140
	v_exp_f32_e32 v50 /*v306*/, v244
	v_exp_f32_e32 v56 /*v312*/, v245
	s_set_vgpr_msb 0x4000
	v_exp_f32_e32 v244, v250
	v_exp_f32_e32 v250, v251
	s_set_vgpr_msb 0x51
	v_pk_fma_f32 v[46:47] /*v[302:303]*/, v[30:31] /*v[286:287]*/, s[60:61], v[68:69] /*v[324:325]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5101
	v_exp_f32_e32 v245, v14 /*v270*/
	v_exp_f32_e32 v251, v15 /*v271*/
	v_exp_f32_e32 v253, v40 /*v296*/
	s_set_vgpr_msb 0x151
	v_exp_f32_e32 v17 /*v273*/, v41 /*v297*/
	v_pk_fma_f32 v[14:15] /*v[270:271]*/, v[88:89] /*v[344:345]*/, s[60:61], v[142:143] /*v[398:399]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[40:41] /*v[296:297]*/, v[134:135] /*v[390:391]*/, s[60:61], v[142:143] /*v[398:399]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5100
	v_exp_f32_e32 v196, v196
	v_exp_f32_e32 v230, v230
	s_set_vgpr_msb 1
	v_exp_f32_e32 v197, v90 /*v346*/
	s_set_vgpr_msb 0x140
	v_exp_f32_e32 v51 /*v307*/, v240
	v_exp_f32_e32 v57 /*v313*/, v241
	v_nop
	s_set_vgpr_msb 0x4011
	v_pk_fma_f32 v[240:241], v[86:87] /*v[342:343]*/, s[60:61], v[142:143] /*v[398:399]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1151
	v_exp_f32_e32 v52 /*v308*/, v47 /*v303*/
	v_pk_fma_f32 v[70:71] /*v[326:327]*/, v[94:95] /*v[350:351]*/, s[60:61], v[68:69] /*v[324:325]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v47 /*v303*/, v14 /*v270*/
	v_exp_f32_e32 v53 /*v309*/, v15 /*v271*/
	v_exp_f32_e32 v59 /*v315*/, v40 /*v296*/
	v_exp_f32_e32 v65 /*v321*/, v41 /*v297*/
	v_pk_fma_f32 v[14:15] /*v[270:271]*/, v[138:139] /*v[394:395]*/, s[60:61], v[142:143] /*v[398:399]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5140
	v_pk_add_f32 v[40:41] /*v[296:297]*/, v[192:193], v[194:195]
	v_pk_add_f32 v[60:61] /*v[316:317]*/, v[198:199], v[200:201]
	s_set_vgpr_msb 0x4051
	v_exp_f32_e32 v30 /*v286*/, v21 /*v277*/
	v_pk_fma_f32 v[66:67] /*v[322:323]*/, v[92:93] /*v[348:349]*/, s[60:61], v[68:69] /*v[324:325]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5140
	v_exp_f32_e32 v21 /*v277*/, v240
	v_exp_f32_e32 v31 /*v287*/, v241
	v_nop
	s_set_vgpr_msb 0x4011
	v_pk_fma_f32 v[240:241], v[136:137] /*v[392:393]*/, s[60:61], v[142:143] /*v[398:399]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1141
	v_exp_f32_e32 v72 /*v328*/, v71 /*v327*/
	v_exp_f32_e32 v71 /*v327*/, v14 /*v270*/
	v_exp_f32_e32 v73 /*v329*/, v15 /*v271*/
	v_nop
	v_pk_add_f32 v[14:15] /*v[270:271]*/, v[40:41] /*v[296:297]*/, v[196:197]
	v_pk_add_f32 v[40:41] /*v[296:297]*/, v[60:61] /*v[316:317]*/, v[202:203]
	s_set_vgpr_msb 0x4145
	v_pk_add_f32 v[60:61] /*v[316:317]*/, v[2:3] /*v[258:259]*/, v[4:5] /*v[260:261]*/
	v_pk_add_f32 v[62:63] /*v[318:319]*/, v[22:23] /*v[278:279]*/, v[26:27] /*v[282:283]*/
	v_pk_add_f32 v[76:77] /*v[332:333]*/, v[36:37] /*v[292:293]*/, v[42:43] /*v[298:299]*/
	s_set_vgpr_msb 0x4540
	v_pk_add_f32 v[90:91] /*v[346:347]*/, v[238:239], v[242:243]
	s_set_vgpr_msb 0x4044
	v_pk_add_f32 v[92:93] /*v[348:349]*/, v[254:255], v[10:11] /*v[266:267]*/
	s_set_vgpr_msb 0x4440
	v_pk_add_f32 v[96:97] /*v[352:353]*/, v[230:231], v[236:237]
	s_set_vgpr_msb 0x4045
	v_pk_add_f32 v[98:99] /*v[354:355]*/, v[6:7] /*v[262:263]*/, v[12:13] /*v[268:269]*/
	s_set_vgpr_msb 0x4500
	v_exp_f32_e32 v212, v212
	s_set_vgpr_msb 0x41
	v_exp_f32_e32 v20 /*v276*/, v20 /*v276*/
	v_exp_f32_e32 v46 /*v302*/, v46 /*v302*/
	v_exp_f32_e32 v66 /*v322*/, v66 /*v322*/
	v_exp_f32_e32 v68 /*v324*/, v67 /*v323*/
	s_set_vgpr_msb 0x4140
	v_exp_f32_e32 v67 /*v323*/, v240
	s_set_vgpr_msb 0x4041
	v_pk_add_f32 v[78:79] /*v[334:335]*/, v[54:55] /*v[310:311]*/, v[204:205]
	s_set_vgpr_msb 0x4140
	v_pk_add_f32 v[84:85] /*v[340:341]*/, v[208:209], v[210:211]
	s_set_vgpr_msb 0x4045
	v_pk_add_f32 v[60:61] /*v[316:317]*/, v[8:9] /*v[264:265]*/, v[60:61] /*v[316:317]*/
	v_pk_add_f32 v[62:63] /*v[318:319]*/, v[32:33] /*v[288:289]*/, v[62:63] /*v[318:319]*/
	v_pk_add_f32 v[76:77] /*v[332:333]*/, v[48:49] /*v[304:305]*/, v[76:77] /*v[332:333]*/
	s_set_vgpr_msb 0x4540
	v_pk_add_f32 v[86:87] /*v[342:343]*/, v[214:215], v[218:219]
	v_pk_add_f32 v[94:95] /*v[350:351]*/, v[220:221], v[222:223]
	s_set_vgpr_msb 0x4044
	v_pk_add_f32 v[90:91] /*v[346:347]*/, v[246:247], v[90:91] /*v[346:347]*/
	v_pk_add_f32 v[92:93] /*v[348:349]*/, v[216:217], v[92:93] /*v[348:349]*/
	s_set_vgpr_msb 0x4445
	v_pk_add_f32 v[100:101] /*v[356:357]*/, v[28:29] /*v[284:285]*/, v[34:35] /*v[290:291]*/
	v_pk_add_f32 v[102:103] /*v[358:359]*/, v[44:45] /*v[300:301]*/, v[50:51] /*v[306:307]*/
	s_set_vgpr_msb 0x4540
	v_pk_add_f32 v[104:105] /*v[360:361]*/, v[244:245], v[250:251]
	s_set_vgpr_msb 0x4044
	v_pk_add_f32 v[96:97] /*v[352:353]*/, v[248:249], v[96:97] /*v[352:353]*/
	s_set_vgpr_msb 0x4445
	v_pk_add_f32 v[98:99] /*v[354:355]*/, v[24:25] /*v[280:281]*/, v[98:99] /*v[354:355]*/
	v_pk_add_f32 v[14:15] /*v[270:271]*/, v[14:15] /*v[270:271]*/, v[40:41] /*v[296:297]*/
	v_exp_f32_e32 v70 /*v326*/, v70 /*v326*/
	s_set_vgpr_msb 0x4540
	v_exp_f32_e32 v69 /*v325*/, v241
	v_nop
	s_set_vgpr_msb 0x4011
	v_pk_fma_f32 v[240:241], v[140:141] /*v[396:397]*/, s[60:61], v[142:143] /*v[398:399]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1144
	v_pk_add_f32 v[78:79] /*v[334:335]*/, v[206:207], v[78:79] /*v[334:335]*/
	v_pk_add_f32 v[84:85] /*v[340:341]*/, v[212:213], v[84:85] /*v[340:341]*/
	s_set_vgpr_msb 0x4440
	v_pk_add_f32 v[88:89] /*v[344:345]*/, v[226:227], v[232:233]
	s_set_vgpr_msb 0x4044
	v_pk_add_f32 v[86:87] /*v[342:343]*/, v[224:225], v[86:87] /*v[342:343]*/
	v_pk_add_f32 v[94:95] /*v[350:351]*/, v[228:229], v[94:95] /*v[350:351]*/
	s_set_vgpr_msb 0x4445
	v_pk_add_f32 v[100:101] /*v[356:357]*/, v[38:39] /*v[294:295]*/, v[100:101] /*v[356:357]*/
	v_pk_add_f32 v[102:103] /*v[358:359]*/, v[56:57] /*v[312:313]*/, v[102:103] /*v[358:359]*/
	s_set_vgpr_msb 0x4544
	v_pk_add_f32 v[104:105] /*v[360:361]*/, v[252:253], v[104:105] /*v[360:361]*/
	s_set_vgpr_msb 0x4445
	v_pk_add_f32 v[106:107] /*v[362:363]*/, v[16:17] /*v[272:273]*/, v[20:21] /*v[276:277]*/
	v_pk_add_f32 v[108:109] /*v[364:365]*/, v[46:47] /*v[302:303]*/, v[52:53] /*v[308:309]*/
	v_pk_add_f32 v[110:111] /*v[366:367]*/, v[64:65] /*v[320:321]*/, v[66:67] /*v[322:323]*/
	v_pk_add_f32 v[14:15] /*v[270:271]*/, v[60:61] /*v[316:317]*/, v[14:15] /*v[270:271]*/
	v_pk_add_f32 v[60:61] /*v[316:317]*/, v[62:63] /*v[318:319]*/, v[76:77] /*v[332:333]*/
	v_pk_add_f32 v[62:63] /*v[318:319]*/, v[90:91] /*v[346:347]*/, v[92:93] /*v[348:349]*/
	v_pk_add_f32 v[76:77] /*v[332:333]*/, v[96:97] /*v[352:353]*/, v[98:99] /*v[354:355]*/
	s_set_vgpr_msb 0x4544
	v_exp_f32_e32 v75 /*v331*/, v240
	v_pk_add_f32 v[88:89] /*v[344:345]*/, v[234:235], v[88:89] /*v[344:345]*/
	s_set_vgpr_msb 0x4445
	v_pk_add_f32 v[112:113] /*v[368:369]*/, v[70:71] /*v[326:327]*/, v[72:73] /*v[328:329]*/
	v_pk_add_f32 v[40:41] /*v[296:297]*/, v[30:31] /*v[286:287]*/, v[106:107] /*v[362:363]*/
	v_pk_add_f32 v[106:107] /*v[362:363]*/, v[58:59] /*v[314:315]*/, v[108:109] /*v[364:365]*/
	v_pk_add_f32 v[108:109] /*v[364:365]*/, v[68:69] /*v[324:325]*/, v[110:111] /*v[366:367]*/
	v_pk_add_f32 v[84:85] /*v[340:341]*/, v[84:85] /*v[340:341]*/, v[86:87] /*v[342:343]*/
	v_pk_add_f32 v[86:87] /*v[342:343]*/, v[102:103] /*v[358:359]*/, v[104:105] /*v[360:361]*/
	v_pk_add_f32 v[60:61] /*v[316:317]*/, v[78:79] /*v[334:335]*/, v[60:61] /*v[316:317]*/
	v_pk_add_f32 v[62:63] /*v[318:319]*/, v[94:95] /*v[350:351]*/, v[62:63] /*v[318:319]*/
	v_pk_add_f32 v[76:77] /*v[332:333]*/, v[100:101] /*v[356:357]*/, v[76:77] /*v[332:333]*/
	v_pk_add_f32 v[110:111] /*v[366:367]*/, v[74:75] /*v[330:331]*/, v[112:113] /*v[368:369]*/
	v_pk_add_f32 v[78:79] /*v[334:335]*/, v[88:89] /*v[344:345]*/, v[84:85] /*v[340:341]*/
	v_pk_add_f32 v[40:41] /*v[296:297]*/, v[40:41] /*v[296:297]*/, v[86:87] /*v[342:343]*/
	v_pk_add_f32 v[84:85] /*v[340:341]*/, v[106:107] /*v[362:363]*/, v[108:109] /*v[364:365]*/
	v_pk_add_f32 v[14:15] /*v[270:271]*/, v[14:15] /*v[270:271]*/, v[60:61] /*v[316:317]*/
	v_pk_add_f32 v[60:61] /*v[316:317]*/, v[62:63] /*v[318:319]*/, v[76:77] /*v[332:333]*/
	s_set_vgpr_msb 0x4540
	v_exp_f32_e32 v19 /*v275*/, v241
	v_nop
	s_set_vgpr_msb 0x4005
	v_pk_add_f32 v[240:241], v[110:111] /*v[366:367]*/, v[84:85] /*v[340:341]*/
	s_set_vgpr_msb 0x545
	v_pk_add_f32 v[14:15] /*v[270:271]*/, v[78:79] /*v[334:335]*/, v[14:15] /*v[270:271]*/
	v_pk_add_f32 v[40:41] /*v[296:297]*/, v[40:41] /*v[296:297]*/, v[60:61] /*v[316:317]*/
	s_set_vgpr_msb 0x4501
	v_pk_add_f32 v[240:241], v[18:19] /*v[274:275]*/, v[240:241]
	s_set_vgpr_msb 0x145
	v_pk_add_f32 v[40:41] /*v[296:297]*/, v[14:15] /*v[270:271]*/, v[40:41] /*v[296:297]*/
	s_set_vgpr_msb 0x4549
	v_sub_f32_e32 v14 /*v270*/, v83 /*v339*/, v209 /*v721*/
	s_set_vgpr_msb 0x4942
	v_mov_b32_e32 v15 /*v271*/, v210 /*v722*/
	s_set_vgpr_msb 0x4282
	v_dual_mov_b32 v210 /*v722*/, v214 /*v726*/ :: v_dual_mov_b32 v214 /*v726*/, v216 /*v728*/
	s_set_vgpr_msb 0x8204
	v_pk_add_f32 v[240:241], v[240:241], v[40:41] /*v[296:297]*/
	s_set_vgpr_msb 0x446
	v_dual_mul_f32 v14 /*v270*/, 0x3fb8aa3b, v14 /*v270*/ :: v_dual_mov_b32 v41 /*v297*/, v208 /*v720*/
	s_wait_alu depctr_vm_vsrc(0)
	s_set_vgpr_msb 0x4682
	v_dual_mov_b32 v208 /*v720*/, v217 /*v729*/ :: v_dual_mov_b32 v217 /*v729*/, v218 /*v730*/
	s_set_vgpr_msb 0x8240
	v_dual_mov_b32 v60 /*v316*/, v240 :: v_dual_mov_b32 v61 /*v317*/, v241
	s_set_vgpr_msb 0x4041
	v_exp_f32_e32 v14 /*v270*/, v14 /*v270*/
	v_permlanex16_b32 v60 /*v316*/, v60 /*v316*/, s61, 0xfedcba98
	v_permlanex16_b32 v61 /*v317*/, v61 /*v317*/, s61, 0xfedcba98
	s_set_vgpr_msb 0x4100
	s_cbranch_vccz .LBB0_38
	s_set_vgpr_msb 4
	v_pk_mul_f32 v[126:127], v[126:127], v[14:15] /*v[270:271]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[124:125], v[124:125], v[14:15] /*v[270:271]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[122:123], v[122:123], v[14:15] /*v[270:271]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[120:121], v[120:121], v[14:15] /*v[270:271]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[118:119], v[118:119], v[14:15] /*v[270:271]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[116:117], v[116:117], v[14:15] /*v[270:271]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[114:115], v[114:115], v[14:15] /*v[270:271]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[112:113], v[112:113], v[14:15] /*v[270:271]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[110:111], v[110:111], v[14:15] /*v[270:271]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[108:109], v[108:109], v[14:15] /*v[270:271]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[106:107], v[106:107], v[14:15] /*v[270:271]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[104:105], v[104:105], v[14:15] /*v[270:271]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[102:103], v[102:103], v[14:15] /*v[270:271]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[100:101], v[100:101], v[14:15] /*v[270:271]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[98:99], v[98:99], v[14:15] /*v[270:271]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[96:97], v[96:97], v[14:15] /*v[270:271]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[94:95], v[94:95], v[14:15] /*v[270:271]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[92:93], v[92:93], v[14:15] /*v[270:271]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[90:91], v[90:91], v[14:15] /*v[270:271]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[88:89], v[88:89], v[14:15] /*v[270:271]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[86:87], v[86:87], v[14:15] /*v[270:271]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[84:85], v[84:85], v[14:15] /*v[270:271]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[82:83], v[82:83], v[14:15] /*v[270:271]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[80:81], v[80:81], v[14:15] /*v[270:271]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[78:79], v[78:79], v[14:15] /*v[270:271]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[76:77], v[76:77], v[14:15] /*v[270:271]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[74:75], v[74:75], v[14:15] /*v[270:271]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[72:73], v[72:73], v[14:15] /*v[270:271]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[70:71], v[70:71], v[14:15] /*v[270:271]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[68:69], v[68:69], v[14:15] /*v[270:271]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[66:67], v[66:67], v[14:15] /*v[270:271]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[64:65], v[64:65], v[14:15] /*v[270:271]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x400
.LBB0_38:
	s_set_vgpr_msb 0x46
	v_sub_f32_e32 v40 /*v296*/, v211 /*v723*/, v82 /*v338*/
	s_and_b32 s2, s3, exec_lo
	s_cselect_b32 s2, 1, 0
	s_cmp_lg_u32 s2, 1
	v_mul_f32_e32 v40 /*v296*/, 0x3fb8aa3b, v40 /*v296*/
	s_set_vgpr_msb 0x4641
	v_exp_f32_e32 v40 /*v296*/, v40 /*v296*/
	s_set_vgpr_msb 0x4100
	s_cbranch_scc1 .LBB0_40
	v_nop
	s_set_vgpr_msb 4
	v_pk_mul_f32 v[62:63], v[62:63], v[40:41] /*v[296:297]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[60:61], v[60:61], v[40:41] /*v[296:297]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59], v[58:59], v[40:41] /*v[296:297]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[56:57], v[56:57], v[40:41] /*v[296:297]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[54:55], v[54:55], v[40:41] /*v[296:297]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[52:53], v[52:53], v[40:41] /*v[296:297]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[50:51], v[50:51], v[40:41] /*v[296:297]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[48:49], v[48:49], v[40:41] /*v[296:297]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[46:47], v[46:47], v[40:41] /*v[296:297]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[44:45], v[44:45], v[40:41] /*v[296:297]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[42:43], v[42:43], v[40:41] /*v[296:297]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[40:41], v[40:41], v[40:41] /*v[296:297]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39], v[38:39], v[40:41] /*v[296:297]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[36:37], v[36:37], v[40:41] /*v[296:297]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35], v[34:35], v[40:41] /*v[296:297]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[32:33], v[32:33], v[40:41] /*v[296:297]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[30:31], v[30:31], v[40:41] /*v[296:297]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[28:29], v[28:29], v[40:41] /*v[296:297]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[26:27], v[26:27], v[40:41] /*v[296:297]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25], v[24:25], v[40:41] /*v[296:297]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[22:23], v[22:23], v[40:41] /*v[296:297]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21], v[20:21], v[40:41] /*v[296:297]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[18:19], v[18:19], v[40:41] /*v[296:297]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[16:17], v[16:17], v[40:41] /*v[296:297]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[14:15], v[40:41] /*v[296:297]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13], v[12:13], v[40:41] /*v[296:297]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[10:11], v[40:41] /*v[296:297]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[8:9], v[40:41] /*v[296:297]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[6:7], v[40:41] /*v[296:297]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[4:5], v[40:41] /*v[296:297]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[2:3], v[2:3], v[40:41] /*v[296:297]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[0:1], v[0:1], v[40:41] /*v[296:297]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x400
.LBB0_40:
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x42
	ds_load_tr16_b128 v[84:87] /*v[340:343]*/, v205 /*v717*/
	ds_load_tr16_b128 v[92:95] /*v[348:351]*/, v205 /*v717*/ offset:32
	ds_load_tr16_b128 v[88:91] /*v[344:347]*/, v205 /*v717*/ offset:4608
	ds_load_tr16_b128 v[96:99] /*v[352:355]*/, v205 /*v717*/ offset:4640
	ds_load_tr16_b128 v[100:103] /*v[356:359]*/, v205 /*v717*/ offset:9216
	ds_load_tr16_b128 v[108:111] /*v[364:367]*/, v205 /*v717*/ offset:9248
	ds_load_tr16_b128 v[104:107] /*v[360:363]*/, v205 /*v717*/ offset:13824
	ds_load_tr16_b128 v[112:115] /*v[368:371]*/, v205 /*v717*/ offset:13856
	ds_load_tr16_b128 v[116:119] /*v[372:375]*/, v205 /*v717*/ offset:18432
	ds_load_tr16_b128 v[124:127] /*v[380:383]*/, v205 /*v717*/ offset:18464
	ds_load_tr16_b128 v[120:123] /*v[376:379]*/, v205 /*v717*/ offset:23040
	ds_load_tr16_b128 v[128:131] /*v[384:387]*/, v205 /*v717*/ offset:23072
	ds_load_tr16_b128 v[132:135] /*v[388:391]*/, v205 /*v717*/ offset:27648
	ds_load_tr16_b128 v[140:143] /*v[396:399]*/, v205 /*v717*/ offset:27680
	ds_load_tr16_b128 v[136:139] /*v[392:395]*/, v205 /*v717*/ offset:32256
	ds_load_tr16_b128 v[144:147] /*v[400:403]*/, v205 /*v717*/ offset:32288
	s_set_vgpr_msb 0x4245
	v_cvt_pk_bf16_f32 v155 /*v411*/, v48 /*v304*/, v54 /*v310*/
	v_cvt_pk_bf16_f32 v154 /*v410*/, v36 /*v292*/, v42 /*v298*/
	v_cvt_pk_bf16_f32 v153 /*v409*/, v26 /*v282*/, v32 /*v288*/
	v_cvt_pk_bf16_f32 v152 /*v408*/, v8 /*v264*/, v22 /*v278*/
	v_cvt_pk_bf16_f32 v151 /*v407*/, v2 /*v258*/, v4 /*v260*/
	s_set_vgpr_msb 0x4540
	v_cvt_pk_bf16_f32 v150 /*v406*/, v200, v202
	v_cvt_pk_bf16_f32 v149 /*v405*/, v196, v198
	v_cvt_pk_bf16_f32 v148 /*v404*/, v192, v194
	s_set_vgpr_msb 0x4045
	v_cvt_pk_bf16_f32 v171 /*v427*/, v49 /*v305*/, v55 /*v311*/
	v_cvt_pk_bf16_f32 v170 /*v426*/, v37 /*v293*/, v43 /*v299*/
	v_cvt_pk_bf16_f32 v169 /*v425*/, v27 /*v283*/, v33 /*v289*/
	v_cvt_pk_bf16_f32 v168 /*v424*/, v9 /*v265*/, v23 /*v279*/
	v_cvt_pk_bf16_f32 v167 /*v423*/, v3 /*v259*/, v5 /*v261*/
	s_set_vgpr_msb 0x4540
	v_cvt_pk_bf16_f32 v166 /*v422*/, v201, v203
	v_cvt_pk_bf16_f32 v165 /*v421*/, v197, v199
	v_cvt_pk_bf16_f32 v164 /*v420*/, v193, v195
	s_set_vgpr_msb 0x4005
	s_wait_dscnt 0xd
	v_wmma_f32_16x16x32_bf16 v[120:127], v[84:91] /*v[340:347]*/, v[148:155] /*v[404:411]*/, v[120:127]
	s_set_vgpr_msb 0x540
	v_cvt_pk_bf16_f32 v157 /*v413*/, v208, v210
	s_set_vgpr_msb 0x4000
	v_cvt_pk_bf16_f32 v194, v230, v236
	v_cvt_pk_bf16_f32 v193, v222, v228
	v_cvt_pk_bf16_f32 v210, v231, v237
	s_set_vgpr_msb 64
	v_cvt_pk_bf16_f32 v161 /*v417*/, v234, v238
	v_cvt_pk_bf16_f32 v160 /*v416*/, v226, v232
	v_cvt_pk_bf16_f32 v159 /*v415*/, v218, v224
	s_set_vgpr_msb 0x4005
	v_wmma_f32_16x16x32_bf16 v[56:63], v[84:91] /*v[340:347]*/, v[164:171] /*v[420:427]*/, v[56:63]
	s_set_vgpr_msb 0x540
	v_cvt_pk_bf16_f32 v162 /*v418*/, v242, v246
	s_set_vgpr_msb 0x4004
	v_cvt_pk_bf16_f32 v195, v248, v6 /*v262*/
	s_set_vgpr_msb 0x400
	v_cvt_pk_bf16_f32 v192, v216, v220
	v_cvt_pk_bf16_f32 v200, v244, v250
	s_set_vgpr_msb 64
	v_cvt_pk_bf16_f32 v85 /*v341*/, v209, v211
	s_set_vgpr_msb 0x4000
	v_cvt_pk_bf16_f32 v209, v223, v229
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 2
	ds_load_tr16_b128 v[228:231], v205 /*v717*/ offset:4672
	s_set_vgpr_msb 0x240
	v_cvt_pk_bf16_f32 v89 /*v345*/, v235, v239
	v_cvt_pk_bf16_f32 v88 /*v344*/, v227, v233
	v_cvt_pk_bf16_f32 v87 /*v343*/, v219, v225
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x4002
	ds_load_tr16_b128 v[224:227], v205 /*v717*/ offset:64
	ds_load_tr16_b128 v[232:235], v205 /*v717*/ offset:96
	ds_load_tr16_b128 v[236:239], v205 /*v717*/ offset:4704
	s_set_vgpr_msb 0x240
	v_cvt_pk_bf16_f32 v90 /*v346*/, v243, v247
	s_set_vgpr_msb 0x4004
	v_cvt_pk_bf16_f32 v211, v249, v7 /*v263*/
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x402
	ds_load_tr16_b128 v[246:249], v205 /*v717*/ offset:13888
	s_set_vgpr_msb 0x200
	v_cvt_pk_bf16_f32 v216, v245, v251
	s_set_vgpr_msb 4
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x32_bf16 v[104:111], v[224:231], v[148:155] /*v[404:411]*/, v[104:111]
	s_set_vgpr_msb 0x444
	v_cvt_pk_bf16_f32 v163 /*v419*/, v254, v10 /*v266*/
	s_set_vgpr_msb 0x4440
	v_cvt_pk_bf16_f32 v158 /*v414*/, v212, v214
	v_cvt_pk_bf16_f32 v156 /*v412*/, v204, v206
	s_set_vgpr_msb 0x4044
	v_cvt_pk_bf16_f32 v91 /*v347*/, v255, v11 /*v267*/
	s_set_vgpr_msb 0x4440
	v_cvt_pk_bf16_f32 v86 /*v342*/, v213, v215
	v_cvt_pk_bf16_f32 v84 /*v340*/, v205, v207
	s_set_vgpr_msb 0x4000
	v_cvt_pk_bf16_f32 v208, v217, v221
	s_set_vgpr_msb 4
	v_wmma_f32_16x16x32_bf16 v[40:47], v[224:231], v[164:171] /*v[420:427]*/, v[40:47]
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x402
	ds_load_tr16_b128 v[242:245], v205 /*v717*/ offset:9280
	ds_load_tr16_b128 v[224:227], v205 /*v717*/ offset:9312
	ds_load_tr16_b128 v[228:231], v205 /*v717*/ offset:13920
	s_set_vgpr_msb 0x204
	v_cvt_pk_bf16_f32 v201, v252, v16 /*v272*/
	v_cvt_pk_bf16_f32 v217, v253, v17 /*v273*/
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x402
	ds_load_tr16_b128 v[250:253], v205 /*v717*/ offset:23104
	s_set_vgpr_msb 0x205
	v_cvt_pk_bf16_f32 v199, v50 /*v306*/, v56 /*v312*/
	v_cvt_pk_bf16_f32 v198, v38 /*v294*/, v44 /*v300*/
	v_cvt_pk_bf16_f32 v197, v28 /*v284*/, v34 /*v290*/
	s_set_vgpr_msb 0x504
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x32_bf16 v[104:111], v[242:249], v[156:163] /*v[412:419]*/, v[104:111]
	s_set_vgpr_msb 0x405
	v_cvt_pk_bf16_f32 v196, v12 /*v268*/, v24 /*v280*/
	v_cvt_pk_bf16_f32 v215, v51 /*v307*/, v57 /*v313*/
	v_cvt_pk_bf16_f32 v214, v39 /*v295*/, v45 /*v301*/
	v_cvt_pk_bf16_f32 v213, v29 /*v285*/, v35 /*v291*/
	v_cvt_pk_bf16_f32 v212, v13 /*v269*/, v25 /*v281*/
	v_cvt_pk_bf16_f32 v207, v74 /*v330*/, v18 /*v274*/
	v_cvt_pk_bf16_f32 v202, v20 /*v276*/, v30 /*v286*/
	s_set_vgpr_msb 0x504
	v_wmma_f32_16x16x32_bf16 v[40:47], v[242:249], v[84:91] /*v[340:347]*/, v[40:47]
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x402
	ds_load_tr16_b128 v[246:249], v205 /*v717*/ offset:18496
	s_set_vgpr_msb 0x242
	ds_load_tr16_b128 v[2:5] /*v[258:261]*/, v205 /*v717*/ offset:18528
	ds_load_tr16_b128 v[6:9] /*v[262:265]*/, v205 /*v717*/ offset:23136
	s_set_vgpr_msb 0x4205
	v_cvt_pk_bf16_f32 v223, v75 /*v331*/, v19 /*v275*/
	v_cvt_pk_bf16_f32 v218, v21 /*v277*/, v31 /*v287*/
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x542
	ds_load_tr16_b128 v[20:23] /*v[276:279]*/, v205 /*v717*/ offset:32320
	s_set_vgpr_msb 0x4205
	v_cvt_pk_bf16_f32 v206, v70 /*v326*/, v72 /*v328*/
	v_cvt_pk_bf16_f32 v205, v66 /*v322*/, v68 /*v324*/
	v_cvt_pk_bf16_f32 v204, v58 /*v314*/, v64 /*v320*/
	s_set_vgpr_msb 0x504
	v_wmma_f32_16x16x32_bf16 v[96:103], v[232:239], v[148:155] /*v[404:411]*/, v[96:103]
	s_set_vgpr_msb 0x405
	v_cvt_pk_bf16_f32 v203, v46 /*v302*/, v52 /*v308*/
	v_cvt_pk_bf16_f32 v222, v71 /*v327*/, v73 /*v329*/
	v_cvt_pk_bf16_f32 v221, v67 /*v323*/, v69 /*v325*/
	v_cvt_pk_bf16_f32 v220, v59 /*v315*/, v65 /*v321*/
	v_cvt_pk_bf16_f32 v219, v47 /*v303*/, v53 /*v309*/
	s_add_co_i32 s2, s100, 4
	s_cmp_ge_i32 s2, s54
	s_set_vgpr_msb 0x504
	v_wmma_f32_16x16x32_bf16 v[32:39], v[232:239], v[164:171] /*v[420:427]*/, v[32:39]
	s_set_vgpr_msb 0x400
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x32_bf16 v[104:111], v[246:253], v[192:199], v[104:111]
	v_wmma_f32_16x16x32_bf16 v[40:47], v[246:253], v[208:215], v[40:47]
	s_set_vgpr_msb 0x42
	ds_load_tr16_b128 v[16:19] /*v[272:275]*/, v205 /*v717*/ offset:27712
	s_set_vgpr_msb 0x4202
	ds_load_tr16_b128 v[242:245], v205 /*v717*/ offset:27744
	s_wait_alu depctr_va_vdst(0)
	ds_load_tr16_b128 v[246:249], v205 /*v717*/ offset:32352
	ds_load_tr16_b128 v[250:253], v205 /*v717*/ offset:23168
	s_set_vgpr_msb 0x204
	v_wmma_f32_16x16x32_bf16 v[96:103], v[224:231], v[156:163] /*v[412:419]*/, v[96:103]
	v_wmma_f32_16x16x32_bf16 v[32:39], v[224:231], v[84:91] /*v[340:347]*/, v[32:39]
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x402
	ds_load_tr16_b128 v[228:231], v205 /*v717*/ offset:4736
	ds_load_tr16_b128 v[224:227], v205 /*v717*/ offset:128
	ds_load_tr16_b128 v[232:235], v205 /*v717*/ offset:160
	ds_load_tr16_b128 v[236:239], v205 /*v717*/ offset:4768
	s_set_vgpr_msb 0x201
	s_wait_dscnt 0x9
	v_wmma_f32_16x16x32_bf16 v[96:103], v[2:9] /*v[258:265]*/, v[192:199], v[96:103]
	v_wmma_f32_16x16x32_bf16 v[32:39], v[2:9] /*v[258:265]*/, v[208:215], v[32:39]
	s_set_vgpr_msb 0x100
	s_wait_dscnt 0x5
	v_wmma_f32_16x16x32_bf16 v[96:103], v[242:249], v[200:207], v[96:103]
	v_wmma_f32_16x16x32_bf16 v[32:39], v[242:249], v[216:223], v[32:39]
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 2
	ds_load_tr16_b128 v[246:249], v205 /*v717*/ offset:13952
	s_set_vgpr_msb 0x204
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x32_bf16 v[88:95], v[224:231], v[148:155] /*v[404:411]*/, v[88:95]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[224:231], v[164:171] /*v[420:427]*/, v[24:31]
	s_set_vgpr_msb 0x402
	ds_load_tr16_b128 v[242:245], v205 /*v717*/ offset:9344
	s_wait_alu depctr_va_vdst(0)
	ds_load_tr16_b128 v[224:227], v205 /*v717*/ offset:9376
	ds_load_tr16_b128 v[228:231], v205 /*v717*/ offset:13984
	s_set_vgpr_msb 0x204
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[88:95], v[242:249], v[156:163] /*v[412:419]*/, v[88:95]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[242:249], v[84:91] /*v[340:347]*/, v[24:31]
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x402
	ds_load_tr16_b128 v[246:249], v205 /*v717*/ offset:18560
	s_set_vgpr_msb 0x242
	ds_load_tr16_b128 v[2:5] /*v[258:261]*/, v205 /*v717*/ offset:18592
	ds_load_tr16_b128 v[6:9] /*v[262:265]*/, v205 /*v717*/ offset:23200
	s_set_vgpr_msb 0x4204
	v_wmma_f32_16x16x32_bf16 v[80:87], v[232:239], v[148:155] /*v[404:411]*/, v[80:87]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[232:239], v[164:171] /*v[420:427]*/, v[16:23]
	s_set_vgpr_msb 0x401
	v_wmma_f32_16x16x32_bf16 v[104:111], v[16:23] /*v[272:279]*/, v[200:207], v[104:111]
	v_wmma_f32_16x16x32_bf16 v[40:47], v[16:23] /*v[272:279]*/, v[216:223], v[40:47]
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x142
	ds_load_tr16_b128 v[20:23] /*v[276:279]*/, v205 /*v717*/ offset:32384
	s_set_vgpr_msb 0x4200
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x32_bf16 v[88:95], v[246:253], v[192:199], v[88:95]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[246:253], v[208:215], v[24:31]
	s_set_vgpr_msb 0x42
	ds_load_tr16_b128 v[16:19] /*v[272:275]*/, v205 /*v717*/ offset:27776
	s_set_vgpr_msb 0x4202
	ds_load_tr16_b128 v[242:245], v205 /*v717*/ offset:27808
	s_wait_alu depctr_va_vdst(0)
	ds_load_tr16_b128 v[246:249], v205 /*v717*/ offset:32416
	ds_load_tr16_b128 v[250:253], v205 /*v717*/ offset:23232
	s_set_vgpr_msb 0x204
	v_wmma_f32_16x16x32_bf16 v[80:87], v[224:231], v[156:163] /*v[412:419]*/, v[80:87]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[224:231], v[84:91] /*v[340:347]*/, v[16:23]
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x402
	ds_load_tr16_b128 v[228:231], v205 /*v717*/ offset:4800
	ds_load_tr16_b128 v[224:227], v205 /*v717*/ offset:192
	ds_load_tr16_b128 v[232:235], v205 /*v717*/ offset:224
	ds_load_tr16_b128 v[236:239], v205 /*v717*/ offset:4832
	s_set_vgpr_msb 0x201
	s_wait_dscnt 0x9
	v_wmma_f32_16x16x32_bf16 v[80:87], v[2:9] /*v[258:265]*/, v[192:199], v[80:87]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[2:9] /*v[258:265]*/, v[208:215], v[16:23]
	s_set_vgpr_msb 0x100
	s_wait_dscnt 0x5
	v_wmma_f32_16x16x32_bf16 v[80:87], v[242:249], v[200:207], v[80:87]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[242:249], v[216:223], v[16:23]
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 2
	ds_load_tr16_b128 v[246:249], v205 /*v717*/ offset:14016
	s_set_vgpr_msb 0x204
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x32_bf16 v[72:79], v[224:231], v[148:155] /*v[404:411]*/, v[72:79]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[224:231], v[164:171] /*v[420:427]*/, v[8:15]
	s_set_vgpr_msb 0x402
	ds_load_tr16_b128 v[242:245], v205 /*v717*/ offset:9408
	s_wait_alu depctr_va_vdst(0)
	ds_load_tr16_b128 v[224:227], v205 /*v717*/ offset:9440
	ds_load_tr16_b128 v[228:231], v205 /*v717*/ offset:14048
	s_set_vgpr_msb 0x204
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[72:79], v[242:249], v[156:163] /*v[412:419]*/, v[72:79]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[242:249], v[84:91] /*v[340:347]*/, v[8:15]
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x402
	ds_load_tr16_b128 v[246:249], v205 /*v717*/ offset:18624
	s_set_vgpr_msb 0x242
	ds_load_tr16_b128 v[2:5] /*v[258:261]*/, v205 /*v717*/ offset:18656
	ds_load_tr16_b128 v[6:9] /*v[262:265]*/, v205 /*v717*/ offset:23264
	s_set_vgpr_msb 0x4205
	v_wmma_f32_16x16x32_bf16 v[112:119], v[92:99] /*v[348:355]*/, v[148:155] /*v[404:411]*/, v[112:119]
	v_wmma_f32_16x16x32_bf16 v[48:55], v[92:99] /*v[348:355]*/, v[164:171] /*v[420:427]*/, v[48:55]
	s_set_vgpr_msb 0x504
	v_wmma_f32_16x16x32_bf16 v[64:71], v[232:239], v[148:155] /*v[404:411]*/, v[64:71]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[232:239], v[164:171] /*v[420:427]*/, v[0:7]
	s_set_vgpr_msb 0x401
	v_wmma_f32_16x16x32_bf16 v[88:95], v[16:23] /*v[272:279]*/, v[200:207], v[88:95]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[16:23] /*v[272:279]*/, v[216:223], v[24:31]
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x142
	ds_load_tr16_b128 v[20:23] /*v[276:279]*/, v205 /*v717*/ offset:32448
	s_set_vgpr_msb 0x4200
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x32_bf16 v[72:79], v[246:253], v[192:199], v[72:79]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[246:253], v[208:215], v[8:15]
	s_set_vgpr_msb 0x42
	ds_load_tr16_b128 v[16:19] /*v[272:275]*/, v205 /*v717*/ offset:27840
	s_set_vgpr_msb 0x4202
	ds_load_tr16_b128 v[242:245], v205 /*v717*/ offset:27872
	s_wait_alu depctr_va_vdst(0)
	ds_load_tr16_b128 v[246:249], v205 /*v717*/ offset:32480
	s_wait_tensorcnt 0x0
	s_set_vgpr_msb 0x205
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	v_wmma_f32_16x16x32_bf16 v[120:127], v[100:107] /*v[356:363]*/, v[156:163] /*v[412:419]*/, v[120:127]
	v_wmma_f32_16x16x32_bf16 v[56:63], v[100:107] /*v[356:363]*/, v[84:91] /*v[340:347]*/, v[56:63]
	v_wmma_f32_16x16x32_bf16 v[112:119], v[108:115] /*v[364:371]*/, v[156:163] /*v[412:419]*/, v[112:119]
	v_wmma_f32_16x16x32_bf16 v[48:55], v[108:115] /*v[364:371]*/, v[84:91] /*v[340:347]*/, v[48:55]
	s_set_vgpr_msb 0x504
	v_wmma_f32_16x16x32_bf16 v[64:71], v[224:231], v[156:163] /*v[412:419]*/, v[64:71]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[224:231], v[84:91] /*v[340:347]*/, v[0:7]
	s_set_vgpr_msb 0x401
	v_wmma_f32_16x16x32_bf16 v[120:127], v[116:123] /*v[372:379]*/, v[192:199], v[120:127]
	v_wmma_f32_16x16x32_bf16 v[56:63], v[116:123] /*v[372:379]*/, v[208:215], v[56:63]
	v_wmma_f32_16x16x32_bf16 v[112:119], v[124:131] /*v[380:387]*/, v[192:199], v[112:119]
	v_wmma_f32_16x16x32_bf16 v[48:55], v[124:131] /*v[380:387]*/, v[208:215], v[48:55]
	v_wmma_f32_16x16x32_bf16 v[64:71], v[2:9] /*v[258:265]*/, v[192:199], v[64:71]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[2:9] /*v[258:265]*/, v[208:215], v[0:7]
	v_wmma_f32_16x16x32_bf16 v[120:127], v[132:139] /*v[388:395]*/, v[200:207], v[120:127]
	v_wmma_f32_16x16x32_bf16 v[56:63], v[132:139] /*v[388:395]*/, v[216:223], v[56:63]
	v_wmma_f32_16x16x32_bf16 v[112:119], v[140:147] /*v[396:403]*/, v[200:207], v[112:119]
	v_wmma_f32_16x16x32_bf16 v[48:55], v[140:147] /*v[396:403]*/, v[216:223], v[48:55]
	v_wmma_f32_16x16x32_bf16 v[72:79], v[16:23] /*v[272:279]*/, v[200:207], v[72:79]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[16:23] /*v[272:279]*/, v[216:223], v[8:15]
	s_set_vgpr_msb 0x100
	v_wmma_f32_16x16x32_bf16 v[64:71], v[242:249], v[200:207], v[64:71]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[242:249], v[216:223], v[0:7]
	s_cbranch_scc1 .LBB0_35
	s_sub_co_i32 s4, s100, s92
	s_lshl_b32 s2, s2, 7
	s_ashr_i32 s3, s4, 31
	s_sub_co_i32 s5, s89, s2
	s_lshr_b32 s3, s3, 30
	v_med3_i32 v192, s5, 0, 0x80
	s_add_co_i32 s2, s2, s53
	s_add_co_i32 s3, s4, s3
	s_mov_b32 s73, s65
	s_and_b32 s5, s3, 0x1ffffc
	s_ashr_i32 s3, s2, 31
	v_readfirstlane_b32 s7, v192
	s_sub_co_i32 s6, s4, s5
	s_mul_u64 s[4:5], s[2:3], s[82:83]
	s_mul_u64 s[2:3], s[2:3], s[90:91]
	s_lshl_b64 s[4:5], s[4:5], 1
	s_sub_co_i32 s7, s7, vcc_hi
	s_add_nc_u64 s[4:5], s[96:97], s[4:5]
	s_mul_i32 s6, s6, 0x11800
	s_add_nc_u64 s[86:87], s[94:95], s[4:5]
	s_max_i32 s4, s7, 0
	s_lshl_b64 s[2:3], s[2:3], 1
	s_lshl_b32 s4, s4, 16
	s_add_nc_u64 s[2:3], s[98:99], s[2:3]
	s_add_co_i32 s85, s102, s6
	s_bitset1_b32 s87, 31
	s_or_b32 s66, s4, 0x7fff
	s_mov_b32 s75, s67
	tensor_load_to_lds s[84:87], s[64:71]
	s_add_nc_u64 s[86:87], s[50:51], s[2:3]
	s_add_co_i32 s85, s103, s6
	s_bitset1_b32 s87, 31
	s_mov_b32 s74, s66
	s_mov_b32 s76, s68
	s_mov_b32 s79, s71
	tensor_load_to_lds s[84:87], s[72:79]
	s_branch .LBB0_35
.LBB0_42:
	s_set_vgpr_msb 0x42
	v_mov_b32_e32 v82 /*v338*/, v211 /*v723*/
	s_set_vgpr_msb 0x4200
	s_branch .LBB0_44
.LBB0_43:
	s_mov_b32 s30, s80
.LBB0_44:
	s_set_vgpr_msb 10
	v_div_scale_f32 v128, null, v130 /*v642*/, v130 /*v642*/, 1.0
	v_div_scale_f32 v131, vcc_lo, 1.0, v130 /*v642*/, 1.0
	v_cmp_lt_f32_e64 s2, 0, v130 /*v642*/
	v_div_scale_f32 v148, null, v131 /*v643*/, v131 /*v643*/, 1.0
	s_mov_b32 s4, 0
	s_set_vgpr_msb 0xa00
	v_rcp_f32_e32 v129, v128
	s_wait_dscnt 0x0
	v_rcp_f32_e32 v149, v148
	v_fma_f32 v130, -v128, v129, 1.0
	v_fmac_f32_e32 v129, v130, v129
	v_mul_f32_e32 v130, v131, v129
	v_fma_f32 v132, -v128, v130, v131
	v_fmac_f32_e32 v130, v132, v129
	v_fma_f32 v128, -v128, v130, v131
	v_fma_f32 v131, -v148, v149, 1.0
	v_fmac_f32_e32 v149, v131, v149
	v_div_fmas_f32 v128, v128, v129, v130
	s_set_vgpr_msb 8
	v_div_scale_f32 v150, vcc_lo, 1.0, v131 /*v643*/, 1.0
	v_div_fixup_f32 v128, v128, v130 /*v642*/, 1.0
	s_set_vgpr_msb 0x800
	v_dual_mul_f32 v151, v150, v149 :: v_dual_cndmask_b32 v128, 0, v128, s2
	v_fma_f32 v152, -v148, v151, v150
	s_set_vgpr_msb 1
	v_readlane_b32 s2, v0 /*v256*/, 0
	s_set_vgpr_msb 0x100
	v_pk_mul_f32 v[90:91], v[128:129], v[90:91] op_sel_hi:[0,1]
	v_fmac_f32_e32 v151, v152, v149
	v_pk_mul_f32 v[88:89], v[128:129], v[88:89] op_sel_hi:[0,1]
	v_pk_mul_f32 v[130:131], v[128:129], v[80:81] op_sel_hi:[0,1]
	v_pk_mul_f32 v[92:93], v[128:129], v[92:93] op_sel_hi:[0,1]
	v_cvt_pk_bf16_f32 v81, v90, v91
	v_fma_f32 v90, -v148, v151, v150
	v_cvt_pk_bf16_f32 v80, v88, v89
	v_pk_mul_f32 v[96:97], v[96:97], v[128:129] op_sel_hi:[1,0]
	v_pk_mul_f32 v[132:133], v[128:129], v[82:83] op_sel_hi:[0,1]
	v_cvt_pk_bf16_f32 v82, v92, v93
	v_div_fmas_f32 v88, v90, v149, v151
	s_set_vgpr_msb 8
	v_cmp_lt_f32_e32 vcc_lo, 0, v131 /*v643*/
	s_set_vgpr_msb 0x800
	v_pk_mul_f32 v[112:113], v[112:113], v[128:129] op_sel_hi:[1,0]
	v_pk_mul_f32 v[114:115], v[114:115], v[128:129] op_sel_hi:[1,0]
	v_pk_mul_f32 v[104:105], v[104:105], v[128:129] op_sel_hi:[1,0]
	s_set_vgpr_msb 8
	v_div_fixup_f32 v92, v88, v131 /*v643*/, 1.0
	s_set_vgpr_msb 0x800
	v_pk_mul_f32 v[106:107], v[106:107], v[128:129] op_sel_hi:[1,0]
	v_pk_mul_f32 v[108:109], v[108:109], v[128:129] op_sel_hi:[1,0]
	v_pk_mul_f32 v[110:111], v[110:111], v[128:129] op_sel_hi:[1,0]
	v_pk_mul_f32 v[98:99], v[98:99], v[128:129] op_sel_hi:[1,0]
	v_pk_mul_f32 v[100:101], v[100:101], v[128:129] op_sel_hi:[1,0]
	v_pk_mul_f32 v[102:103], v[102:103], v[128:129] op_sel_hi:[1,0]
	v_pk_mul_f32 v[138:139], v[128:129], v[76:77] op_sel_hi:[0,1]
	v_cvt_pk_bf16_f32 v76, v96, v97
	v_cndmask_b32_e32 v96, 0, v92, vcc_lo
	v_pk_mul_f32 v[120:121], v[120:121], v[128:129] op_sel_hi:[1,0]
	v_pk_mul_f32 v[122:123], v[122:123], v[128:129] op_sel_hi:[1,0]
	v_pk_mul_f32 v[124:125], v[124:125], v[128:129] op_sel_hi:[1,0]
	v_pk_mul_f32 v[126:127], v[126:127], v[128:129] op_sel_hi:[1,0]
	v_pk_mul_f32 v[116:117], v[116:117], v[128:129] op_sel_hi:[1,0]
	v_pk_mul_f32 v[118:119], v[118:119], v[128:129] op_sel_hi:[1,0]
	v_pk_mul_f32 v[94:95], v[128:129], v[94:95] op_sel_hi:[0,1]
	v_pk_mul_f32 v[84:85], v[128:129], v[84:85] op_sel_hi:[0,1]
	v_pk_mul_f32 v[86:87], v[128:129], v[86:87] op_sel_hi:[0,1]
	v_pk_mul_f32 v[134:135], v[128:129], v[72:73] op_sel_hi:[0,1]
	v_pk_mul_f32 v[136:137], v[128:129], v[74:75] op_sel_hi:[0,1]
	v_pk_mul_f32 v[140:141], v[128:129], v[78:79] op_sel_hi:[0,1]
	v_pk_mul_f32 v[142:143], v[128:129], v[64:65] op_sel_hi:[0,1]
	v_pk_mul_f32 v[144:145], v[128:129], v[66:67] op_sel_hi:[0,1]
	v_pk_mul_f32 v[146:147], v[128:129], v[68:69] op_sel_hi:[0,1]
	v_pk_mul_f32 v[128:129], v[128:129], v[70:71] op_sel_hi:[0,1]
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
	v_cvt_pk_bf16_f32 v85, v132, v133
	v_cvt_pk_bf16_f32 v84, v130, v131
	v_cvt_pk_bf16_f32 v91, v140, v141
	v_cvt_pk_bf16_f32 v90, v138, v139
	v_cvt_pk_bf16_f32 v89, v136, v137
	v_cvt_pk_bf16_f32 v88, v134, v135
	v_cvt_pk_bf16_f32 v95, v128, v129
	v_cvt_pk_bf16_f32 v94, v146, v147
	v_cvt_pk_bf16_f32 v93, v144, v145
	v_cvt_pk_bf16_f32 v92, v142, v143
	s_set_vgpr_msb 2
	v_add3_u32 v116, v203 /*v715*/, s2, 0x47100
	s_set_vgpr_msb 0x200
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
	s_lshl_b32 s2, s62, 2
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 2
	ds_store_b128 v202 /*v714*/, v[64:67]
	ds_store_b128 v202 /*v714*/, v[68:71] offset:32
	ds_store_b128 v202 /*v714*/, v[72:75] offset:64
	ds_store_b128 v202 /*v714*/, v[76:79] offset:96
	ds_store_b128 v202 /*v714*/, v[80:83] offset:128
	ds_store_b128 v202 /*v714*/, v[84:87] offset:160
	ds_store_b128 v202 /*v714*/, v[88:91] offset:192
	ds_store_b128 v202 /*v714*/, v[92:95] offset:224
	s_set_vgpr_msb 0x200
	ds_store_b128 v116, v[0:3]
	ds_store_b128 v116, v[4:7] offset:32
	ds_store_b128 v116, v[8:11] offset:64
	ds_store_b128 v116, v[12:15] offset:96
	s_sub_co_i32 s2, s2, s104
	ds_store_b128 v116, v[16:19] offset:128
	ds_store_b128 v116, v[20:23] offset:160
	ds_store_b128 v116, v[24:27] offset:192
	ds_store_b128 v116, v[28:31] offset:224
	s_cmp_lt_i32 s2, 1
	s_wait_dscnt 0x0
	s_cbranch_scc1 .LBB0_46
	s_wait_alu depctr_vm_vsrc(0)
	s_set_vgpr_msb 8
	v_or_b32_e32 v0, 30, v201 /*v713*/
	v_or_b32_e32 v1, 28, v201 /*v713*/
	s_add_co_i32 s2, s2, -1
	v_or_b32_e32 v11, 24, v201 /*v713*/
	s_min_u32 s3, s2, 31
	v_or_b32_e32 v12, 22, v201 /*v713*/
	s_set_vgpr_msb 0x800
	v_dual_mov_b32 v1, 0 :: v_dual_min_i32 v7, s3, v1
	v_min_u32_e32 v6, s3, v0
	s_set_vgpr_msb 8
	v_or_b32_e32 v0, 26, v201 /*v713*/
	s_set_vgpr_msb 0x802
	v_min_u32_e32 v15, s3, v12
	v_lshl_add_u32 v32, v135 /*v647*/, 4, s81
	s_set_vgpr_msb 0x208
	v_or_b32_e32 v21, 16, v201 /*v713*/
	s_set_vgpr_msb 0x800
	v_or_b32_e32 v2, s104, v6
	v_min_u32_e32 v8, s3, v0
	s_set_vgpr_msb 8
	v_lshlrev_b32_e32 v0, 3, v135 /*v647*/
	s_set_vgpr_msb 0x800
	v_mad_u32_u24 v33, 0x110, v6, v32
	v_mad_u32_u24 v37, 0x110, v15, v32
	v_dual_ashrrev_i32 v4, 31, v2 :: v_dual_bitop2_b32 v3, s104, v7 bitop3:0x54
	v_mad_u32_u24 v34, 0x110, v7, v32
	v_mad_u32_u24 v35, 0x110, v8, v32
	s_set_vgpr_msb 8
	v_or_b32_e32 v29, 8, v201 /*v713*/
	s_set_vgpr_msb 0x800
	v_dual_lshrrev_b32 v4, 30, v4 :: v_dual_ashrrev_i32 v5, 31, v3
	v_or_b32_e32 v9, s104, v8
	s_load_b64 s[6:7], s[0:1], 0x0 nv
	v_dual_add_nc_u32 v4, v2, v4 :: v_dual_min_i32 v21, s3, v21
	v_lshrrev_b32_e32 v5, 30, v5
	v_ashrrev_i32_e32 v10, 31, v9
	v_mad_u32_u24 v40, 0x110, v21, v32
	v_dual_add_nc_u32 v5, v3, v5 :: v_dual_lshrrev_b32 v10, 30, v10
	v_min_i32_e32 v14, s3, v11
	v_and_b32_e32 v11, -4, v4
	v_dual_ashrrev_i32 v4, 2, v4 :: v_dual_bitop2_b32 v12, -4, v5 bitop3:0x40
	v_dual_ashrrev_i32 v5, 2, v5 :: v_dual_add_nc_u32 v10, v9, v10
	v_cmp_ne_u32_e32 vcc_lo, v2, v11
	v_dual_sub_nc_u32 v2, v2, v11 :: v_dual_sub_nc_u32 v11, v3, v12
	v_cmp_ne_u32_e64 s2, v3, v12
	v_dual_add_nc_u32 v13, s93, v4 :: v_dual_add_nc_u32 v12, s93, v5
	v_dual_add_nc_u32 v2, s88, v2 :: v_dual_add_nc_u32 v4, s88, v11
	s_and_b32 s5, s49, vcc_lo
	s_and_b32 s2, s49, s2
	v_cndmask_b32_e64 v11, 0, 1, s5
	v_cndmask_b32_e64 v16, 0, 1, s2
	v_mad_nc_i64_i32 v[2:3], v2, s56, v[0:1]
	v_mad_nc_i64_i32 v[4:5], v4, s56, v[0:1]
	v_mad_u32_u24 v36, 0x110, v14, v32
	v_dual_sub_nc_u32 v6, v13, v11 :: v_dual_sub_nc_u32 v11, v12, v16
	v_or_b32_e32 v13, s104, v14
	v_and_b32_e32 v12, -4, v10
	v_ashrrev_i32_e32 v10, 2, v10
	v_mad_nc_i64_i32 v[2:3], v6, s52, v[2:3]
	v_mad_nc_i64_i32 v[4:5], v11, s52, v[4:5]
	v_dual_ashrrev_i32 v7, 31, v13 :: v_dual_sub_nc_u32 v6, v9, v12
	v_cmp_ne_u32_e32 vcc_lo, v9, v12
	v_dual_add_nc_u32 v9, s93, v10 :: v_dual_bitop2_b32 v11, s104, v15 bitop3:0x54
	v_dual_lshrrev_b32 v10, 30, v7 :: v_dual_add_nc_u32 v6, s88, v6
	s_and_b32 s2, s49, vcc_lo
	s_set_vgpr_msb 8
	v_or_b32_e32 v16, 20, v201 /*v713*/
	v_cndmask_b32_e64 v12, 0, 1, s2
	s_set_vgpr_msb 0x800
	v_ashrrev_i32_e32 v17, 31, v11
	v_mad_nc_i64_i32 v[6:7], v6, s56, v[0:1]
	v_dual_add_nc_u32 v10, v13, v10 :: v_dual_min_i32 v16, s3, v16
	v_dual_sub_nc_u32 v9, v9, v12 :: v_dual_lshrrev_b32 v12, 30, v17
	s_wait_kmcnt 0x0
	v_lshl_add_u64 v[2:3], v[2:3], 1, s[6:7]
	v_and_b32_e32 v8, -4, v10
	v_mad_u32_u24 v38, 0x110, v16, v32
	v_mad_nc_i64_i32 v[6:7], v9, s52, v[6:7]
	v_ashrrev_i32_e32 v9, 2, v10
	v_lshl_add_u64 v[4:5], v[4:5], 1, s[6:7]
	v_sub_nc_u32_e32 v10, v13, v8
	v_cmp_ne_u32_e32 vcc_lo, v13, v8
	v_dual_add_nc_u32 v13, s93, v9 :: v_dual_bitop2_b32 v17, s104, v16 bitop3:0x54
	v_dual_add_nc_u32 v8, s88, v10 :: v_dual_add_nc_u32 v12, v11, v12
	s_and_b32 s2, s49, vcc_lo
	v_lshl_add_u64 v[6:7], v[6:7], 1, s[6:7]
	v_ashrrev_i32_e32 v9, 31, v17
	v_cndmask_b32_e64 v18, 0, 1, s2
	v_and_b32_e32 v10, -4, v12
	v_dual_ashrrev_i32 v12, 2, v12 :: v_dual_lshrrev_b32 v19, 30, v9
	v_mad_nc_i64_i32 v[8:9], v8, s56, v[0:1]
	v_cmp_ne_u32_e32 vcc_lo, v11, v10
	v_dual_sub_nc_u32 v13, v13, v18 :: v_dual_add_nc_u32 v12, s93, v12
	v_dual_add_nc_u32 v18, v17, v19 :: v_dual_sub_nc_u32 v10, v11, v10
	s_and_b32 s2, s49, vcc_lo
	s_set_vgpr_msb 8
	v_or_b32_e32 v19, 18, v201 /*v713*/
	v_cndmask_b32_e64 v11, 0, 1, s2
	v_mad_nc_i64_i32 v[8:9], v13, s52, v[8:9]
	s_set_vgpr_msb 0x800
	v_and_b32_e32 v13, -4, v18
	v_min_u32_e32 v19, s3, v19
	v_dual_sub_nc_u32 v20, v12, v11 :: v_dual_ashrrev_i32 v12, 2, v18
	v_cmp_ne_u32_e32 vcc_lo, v17, v13
	v_dual_add_nc_u32 v10, s88, v10 :: v_dual_sub_nc_u32 v18, v17, v13
	v_mad_u32_u24 v39, 0x110, v19, v32
	v_add_nc_u32_e32 v17, s93, v12
	s_and_b32 s2, s49, vcc_lo
	v_mad_nc_i64_i32 v[10:11], v10, s56, v[0:1]
	v_add_nc_u32_e32 v12, s88, v18
	v_cndmask_b32_e64 v22, 0, 1, s2
	v_or_b32_e32 v18, s104, v19
	v_lshl_add_u64 v[8:9], v[8:9], 1, s[6:7]
	v_mad_nc_i64_i32 v[12:13], v12, s56, v[0:1]
	v_sub_nc_u32_e32 v17, v17, v22
	v_ashrrev_i32_e32 v23, 31, v18
	v_mad_nc_i64_i32 v[10:11], v20, s52, v[10:11]
	v_dual_lshrrev_b32 v20, 30, v23 :: v_dual_bitop2_b32 v22, s104, v21 bitop3:0x54
	v_mad_nc_i64_i32 v[12:13], v17, s52, v[12:13]
	v_ashrrev_i32_e32 v17, 31, v22
	v_lshl_add_u64 v[10:11], v[10:11], 1, s[6:7]
	v_dual_add_nc_u32 v14, v18, v20 :: v_dual_lshrrev_b32 v16, 30, v17
	s_set_vgpr_msb 8
	v_or_b32_e32 v17, 14, v201 /*v713*/
	v_lshl_add_u64 v[12:13], v[12:13], 1, s[6:7]
	s_set_vgpr_msb 0x800
	v_and_b32_e32 v15, -4, v14
	v_dual_ashrrev_i32 v14, 2, v14 :: v_dual_add_nc_u32 v16, v22, v16
	v_min_u32_e32 v23, s3, v17
	v_sub_nc_u32_e32 v20, v18, v15
	v_cmp_ne_u32_e32 vcc_lo, v18, v15
	v_dual_add_nc_u32 v18, s93, v14 :: v_dual_bitop2_b32 v17, -4, v16 bitop3:0x40
	v_ashrrev_i32_e32 v16, 2, v16
	v_dual_add_nc_u32 v14, s88, v20 :: v_dual_bitop2_b32 v20, s104, v23 bitop3:0x54
	s_and_b32 s2, s49, vcc_lo
	v_sub_nc_u32_e32 v25, v22, v17
	v_cmp_ne_u32_e32 vcc_lo, v22, v17
	v_add_nc_u32_e32 v22, s93, v16
	v_ashrrev_i32_e32 v26, 31, v20
	v_cndmask_b32_e64 v24, 0, 1, s2
	v_add_nc_u32_e32 v16, s88, v25
	s_and_b32 s2, s49, vcc_lo
	v_mad_nc_i64_i32 v[14:15], v14, s56, v[0:1]
	v_lshrrev_b32_e32 v25, 30, v26
	s_set_vgpr_msb 8
	v_or_b32_e32 v26, 12, v201 /*v713*/
	v_cndmask_b32_e64 v27, 0, 1, s2
	v_mad_nc_i64_i32 v[16:17], v16, s56, v[0:1]
	s_set_vgpr_msb 0x800
	v_sub_nc_u32_e32 v18, v18, v24
	v_mad_u32_u24 v42, 0x110, v23, v32
	v_dual_add_nc_u32 v25, v20, v25 :: v_dual_min_i32 v26, s3, v26
	v_sub_nc_u32_e32 v22, v22, v27
	v_mad_nc_i64_i32 v[14:15], v18, s52, v[14:15]
	s_set_vgpr_msb 8
	v_or_b32_e32 v27, 10, v201 /*v713*/
	s_set_vgpr_msb 0x800
	v_dual_ashrrev_i32 v18, 2, v25 :: v_dual_bitop2_b32 v24, s104, v26 bitop3:0x54
	v_and_b32_e32 v19, -4, v25
	v_mad_nc_i64_i32 v[16:17], v22, s52, v[16:17]
	v_mad_u32_u24 v44, 0x110, v26, v32
	v_ashrrev_i32_e32 v25, 31, v24
	v_lshl_add_u64 v[14:15], v[14:15], 1, s[6:7]
	v_sub_nc_u32_e32 v22, v20, v19
	v_cmp_ne_u32_e32 vcc_lo, v20, v19
	v_add_nc_u32_e32 v20, s93, v18
	v_lshl_add_u64 v[16:17], v[16:17], 1, s[6:7]
	v_dual_add_nc_u32 v18, s88, v22 :: v_dual_lshrrev_b32 v22, 30, v25
	v_min_u32_e32 v25, s3, v27
	s_and_b32 s2, s49, vcc_lo
	v_cndmask_b32_e64 v27, 0, 1, s2
	v_dual_add_nc_u32 v22, v24, v22 :: v_dual_bitop2_b32 v28, s104, v25 bitop3:0x54
	v_mad_nc_i64_i32 v[18:19], v18, s56, v[0:1]
	v_mad_u32_u24 v45, 0x110, v25, v32
	v_dual_sub_nc_u32 v20, v20, v27 :: v_dual_bitop2_b32 v21, -4, v22 bitop3:0x40
	v_ashrrev_i32_e32 v27, 31, v28
	v_cmp_ne_u32_e32 vcc_lo, v24, v21
	v_mad_nc_i64_i32 v[18:19], v20, s52, v[18:19]
	v_dual_ashrrev_i32 v20, 2, v22 :: v_dual_sub_nc_u32 v22, v24, v21
	v_lshrrev_b32_e32 v27, 30, v27
	s_and_b32 s2, s49, vcc_lo
	v_dual_add_nc_u32 v24, s93, v20 :: v_dual_add_nc_u32 v20, s88, v22
	v_add_nc_u32_e32 v22, v28, v27
	v_cndmask_b32_e64 v27, 0, 1, s2
	v_min_i32_e32 v41, s3, v29
	v_lshl_add_u64 v[18:19], v[18:19], 1, s[6:7]
	v_mad_nc_i64_i32 v[20:21], v20, s56, v[0:1]
	v_dual_sub_nc_u32 v24, v24, v27 :: v_dual_bitop2_b32 v29, s104, v41 bitop3:0x54
	v_and_b32_e32 v23, -4, v22
	v_ashrrev_i32_e32 v22, 2, v22
	v_mad_u32_u24 v41, 0x110, v41, v32
	v_dual_ashrrev_i32 v30, 31, v29 :: v_dual_sub_nc_u32 v27, v28, v23
	v_cmp_ne_u32_e32 vcc_lo, v28, v23
	v_mad_nc_i64_i32 v[20:21], v24, s52, v[20:21]
	v_add_nc_u32_e32 v24, s93, v22
	s_set_vgpr_msb 8
	v_or_b32_e32 v28, 6, v201 /*v713*/
	s_set_vgpr_msb 0x800
	v_dual_add_nc_u32 v22, s88, v27 :: v_dual_lshrrev_b32 v27, 30, v30
	s_and_b32 s2, s49, vcc_lo
	v_cndmask_b32_e64 v30, 0, 1, s2
	v_mad_nc_i64_i32 v[22:23], v22, s56, v[0:1]
	v_add_nc_u32_e32 v27, v29, v27
	v_min_u32_e32 v43, s3, v28
	v_lshl_add_u64 v[20:21], v[20:21], 1, s[6:7]
	v_sub_nc_u32_e32 v24, v24, v30
	s_set_vgpr_msb 8
	v_or_b32_e32 v30, 4, v201 /*v713*/
	s_set_vgpr_msb 0x800
	v_and_b32_e32 v26, -4, v27
	v_mad_nc_i64_i32 v[22:23], v24, s52, v[22:23]
	v_ashrrev_i32_e32 v24, 2, v27
	v_dual_sub_nc_u32 v25, v29, v26 :: v_dual_bitop2_b32 v28, s104, v43 bitop3:0x54
	v_cmp_ne_u32_e32 vcc_lo, v29, v26
	v_mad_u32_u24 v43, 0x110, v43, v32
	v_ashrrev_i32_e32 v27, 31, v28
	v_dual_add_nc_u32 v26, s93, v24 :: v_dual_add_nc_u32 v24, s88, v25
	s_and_b32 s2, s49, vcc_lo
	v_lshl_add_u64 v[22:23], v[22:23], 1, s[6:7]
	v_dual_lshrrev_b32 v27, 30, v27 :: v_dual_min_i32 v46, s3, v30
	v_cndmask_b32_e64 v29, 0, 1, s2
	v_mad_nc_i64_i32 v[24:25], v24, s56, v[0:1]
	v_dual_add_nc_u32 v27, v28, v27 :: v_dual_bitop2_b32 v30, s104, v46 bitop3:0x54
	v_sub_nc_u32_e32 v26, v26, v29
	s_set_vgpr_msb 8
	v_or_b32_e32 v29, 2, v201 /*v713*/
	s_set_vgpr_msb 0x800
	v_mad_u32_u24 v46, 0x110, v46, v32
	v_dual_ashrrev_i32 v47, 31, v30 :: v_dual_bitop2_b32 v31, -4, v27 bitop3:0x40
	v_mad_nc_i64_i32 v[24:25], v26, s52, v[24:25]
	v_min_u32_e32 v48, s3, v29
	v_ashrrev_i32_e32 v27, 2, v27
	v_cmp_ne_u32_e32 vcc_lo, v28, v31
	v_dual_add_nc_u32 v26, s93, v27 :: v_dual_bitop2_b32 v29, s104, v48 bitop3:0x54
	v_lshrrev_b32_e32 v27, 30, v47
	s_set_vgpr_msb 8
	v_min_i32_e32 v47, s3, v201 /*v713*/
	s_and_b32 s2, s49, vcc_lo
	s_set_vgpr_msb 0x800
	v_ashrrev_i32_e32 v50, 31, v29
	v_cndmask_b32_e64 v49, 0, 1, s2
	v_sub_nc_u32_e32 v28, v28, v31
	v_or_b32_e32 v31, s104, v47
	v_dual_add_nc_u32 v27, v30, v27 :: v_dual_lshrrev_b32 v50, 30, v50
	v_dual_sub_nc_u32 v49, v26, v49 :: v_dual_add_nc_u32 v26, s88, v28
	v_ashrrev_i32_e32 v28, 31, v31
	v_dual_ashrrev_i32 v52, 2, v27 :: v_dual_bitop2_b32 v51, -4, v27 bitop3:0x40
	v_add_nc_u32_e32 v50, v29, v50
	v_mad_nc_i64_i32 v[26:27], v26, s56, v[0:1]
	v_lshrrev_b32_e32 v28, 30, v28
	v_cmp_ne_u32_e32 vcc_lo, v30, v51
	v_dual_add_nc_u32 v52, s93, v52 :: v_dual_sub_nc_u32 v30, v30, v51
	v_dual_add_nc_u32 v28, v31, v28 :: v_dual_bitop2_b32 v51, -4, v50 bitop3:0x40
	s_and_b32 s2, s49, vcc_lo
	v_ashrrev_i32_e32 v50, 2, v50
	v_add_nc_u32_e32 v30, s88, v30
	v_cndmask_b32_e64 v53, 0, 1, s2
	v_and_b32_e32 v54, -4, v28
	v_cmp_ne_u32_e32 vcc_lo, v29, v51
	v_dual_sub_nc_u32 v29, v29, v51 :: v_dual_ashrrev_i32 v28, 2, v28
	v_add_nc_u32_e32 v50, s93, v50
	v_sub_nc_u32_e32 v51, v31, v54
	v_cmp_ne_u32_e64 s2, v31, v54
	v_dual_add_nc_u32 v54, s88, v29 :: v_dual_add_nc_u32 v55, s93, v28
	v_mad_nc_i64_i32 v[30:31], v30, s56, v[0:1]
	v_add_nc_u32_e32 v28, s88, v51
	s_and_b32 s2, s49, s2
	v_mad_nc_i64_i32 v[26:27], v49, s52, v[26:27]
	v_cndmask_b32_e64 v51, 0, 1, s2
	s_and_b32 s2, s49, vcc_lo
	v_mad_nc_i64_i32 v[28:29], v28, s56, v[0:1]
	v_cndmask_b32_e64 v56, 0, 1, s2
	v_mad_nc_i64_i32 v[0:1], v54, s56, v[0:1]
	v_sub_nc_u32_e32 v51, v55, v51
	v_mad_u32_u24 v47, 0x110, v47, v32
	v_mad_u32_u24 v32, 0x110, v48, v32
	v_dual_sub_nc_u32 v49, v50, v56 :: v_dual_sub_nc_u32 v50, v52, v53
	v_mad_nc_i64_i32 v[28:29], v51, s52, v[28:29]
	v_lshl_add_u64 v[26:27], v[26:27], 1, s[6:7]
	v_lshl_add_u64 v[24:25], v[24:25], 1, s[6:7]
	v_mad_nc_i64_i32 v[0:1], v49, s52, v[0:1]
	v_mad_nc_i64_i32 v[30:31], v50, s52, v[30:31]
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
	global_store_async_from_lds_b128 v[4:5], v34, off
	global_store_async_from_lds_b128 v[2:3], v33, off
.LBB0_46:
	s_set_vgpr_msb 8
	v_cmp_gt_f32_e32 vcc_lo, 0x800000, v130 /*v642*/
	v_cmp_gt_f32_e64 s2, 0x800000, v131 /*v643*/
	s_load_b64 s[6:7], s[0:1], 0x40 nv
	s_wait_xcnt 0x0
	s_mul_i32 s1, s30, s59
	s_wait_alu depctr_vm_vsrc(0)
	v_mul_lo_u32 v4, s57, v200 /*v712*/
	v_cndmask_b32_e64 v2, 0, 32, vcc_lo
	v_cndmask_b32_e64 v3, 0, 32, s2
	v_cndmask_b32_e64 v0, 0, 0x42000000, vcc_lo
	v_cndmask_b32_e64 v1, 0, 0x42000000, s2
	v_mul_lo_u32 v5, s57, v137 /*v649*/
	s_set_vgpr_msb 0x802
	v_ldexp_f32 v2, v130 /*v642*/, v2
	v_ldexp_f32 v3, v131 /*v643*/, v3
	s_set_vgpr_msb 0x209
	v_mad_u32 v6, v1 /*v257*/, s58, s1
	v_cmp_eq_u32_e32 vcc_lo, 0, v201 /*v713*/
	v_cmp_gt_i32_e64 s0, s62, v200 /*v712*/
	s_set_vgpr_msb 0x908
	v_log_f32_e32 v2, v2
	v_log_f32_e32 v3, v3
	s_add_co_i32 s2, s1, s59
	v_cmp_gt_i32_e64 s1, s62, v137 /*v649*/
	s_and_b32 s0, vcc_lo, s0
	s_ashr_i32 s3, s2, 31
	s_lshl_b32 s5, s2, 27
	s_lshr_b64 s[2:3], s[2:3], 5
	s_set_vgpr_msb 0x800
	v_dual_sub_f32 v0, v2, v0 :: v_dual_sub_f32 v1, v3, v1
	v_add_lshl_u32 v2, v6, v4, 2
	v_add_lshl_u32 v3, v6, v5, 2
	s_and_b32 vcc_lo, vcc_lo, s1
	v_dual_mul_f32 v0, 0x3f317218, v0 :: v_dual_mul_f32 v1, 0x3f317218, v1
	v_cndmask_b32_e64 v2, 0x7fffffff, v2, s0
	v_cndmask_b32_e32 v3, 0x7fffffff, v3, vcc_lo
	s_and_b64 s[2:3], s[2:3], 0x1ffffffffffffff
	s_set_vgpr_msb 2
	v_add_f32_e32 v0, v209 /*v721*/, v0
	s_set_vgpr_msb 0x201
	v_add_f32_e32 v1, v82 /*v338*/, v1
	s_wait_kmcnt 0x0
	s_or_b64 s[0:1], s[6:7], s[4:5]
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x100
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
		.amdhsa_next_free_vgpr 808
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

	.set .Lkn_fmha_fwd_prefill_a16w16_m32x8_bshd_0.num_vgpr, 808
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
    .sgpr_spill_count: 1
    .symbol:         kn_fmha_fwd_prefill_a16w16_m32x8_bshd_0.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     808
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa-unknown-gfx1250
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
