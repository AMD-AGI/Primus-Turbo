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
	s_load_b96 s[52:54], s[0:1], 0xa0 nv
	s_load_b96 s[48:50], s[0:1], 0x90 nv
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
	s_mov_b32 s41, 0
	s_cselect_b32 s5, s7, s6
	s_wait_kmcnt 0x0
	s_mul_i32 s6, s54, s53
	s_cselect_b32 s2, ttmp9, s2
	s_cselect_b32 s3, s3, s4
	s_abs_i32 s4, s6
	s_mul_i32 s5, s53, s5
	s_cvt_f32_u32 s7, s4
	s_add_co_i32 s3, s5, s3
	s_sub_co_i32 s5, 0, s4
	s_mul_i32 s3, s3, s52
	v_s_rcp_f32 s7, s7
	s_add_co_i32 s8, s3, s2
	s_mov_b32 s28, 1
	s_abs_i32 s3, s8
	s_mov_b32 s13, 0xffff0000
	s_mov_b32 s16, 0x200001
	s_mov_b32 s15, 0x807fff
	s_mov_b32 s14, 0xffff7fff
	s_mul_f32 s7, s7, 0x4f7ffffe
	s_mov_b32 s12, 0x7510000
	s_mov_b32 s42, s41
	s_mov_b32 s43, s41
	s_cvt_u32_f32 s7, s7
	s_set_vgpr_msb 0x80
	v_and_b32_e32 v190 /*v702*/, 15, v0
	s_set_vgpr_msb 0x8000
	v_and_b32_e32 v1, 16, v0
	s_mul_i32 s5, s5, s7
	s_mul_hi_u32 s2, s7, s5
	s_xor_b32 s5, s8, s6
	s_add_co_i32 s7, s7, s2
	s_ashr_i32 s68, s5, 31
	s_mul_hi_u32 s2, s3, s7
	s_set_vgpr_msb 0x88
	v_mad_u32_u24 v192 /*v704*/, 0x110, v190 /*v702*/, v1
	s_mul_i32 s7, s2, s4
	s_sub_co_i32 s3, s3, s7
	s_add_co_i32 s7, s2, 1
	s_sub_co_i32 s9, s3, s4
	s_cmp_ge_u32 s3, s4
	s_cselect_b32 s2, s7, s2
	s_cselect_b32 s3, s9, s3
	s_add_co_i32 s7, s2, 1
	s_cmp_ge_u32 s3, s4
	s_cselect_b32 s2, s7, s2
	s_xor_b32 s69, s2, s68
	s_sub_co_i32 s2, s69, s68
	s_mul_i32 s2, s2, s6
	s_cmp_lg_u32 s8, s2
	s_cselect_b32 s2, -1, 0
	s_cmp_lt_i32 s5, 0
	s_cselect_b32 s3, -1, 0
	s_and_b32 s2, s2, s3
	s_and_b32 s3, s2, exec_lo
	s_cselect_b32 s70, -1, 0
	s_cmp_lg_u32 s2, 0
	s_sub_co_ci_u32 s17, s69, s68
	s_abs_i32 s20, s53
	s_mul_i32 s6, s17, s6
	s_cvt_f32_u32 s2, s20
	s_sub_co_i32 s5, 0, s20
	s_sub_co_i32 s25, s8, s6
	v_s_rcp_f32 s2, s2
	s_abs_i32 s21, s25
	s_xor_b32 s24, s25, s53
	s_ashr_i32 s27, s24, 31
	s_mul_f32 s2, s2, 0x4f7ffffe
	s_cvt_u32_f32 s4, s2
	s_clause 0x2
	s_load_b64 s[18:19], s[0:1], 0x10 nv
	s_load_b64 s[60:61], s[0:1], 0x20 nv
	s_load_b64 s[2:3], s[0:1], 0x30 nv
	s_mul_i32 s5, s5, s4
	s_mul_hi_u32 s5, s4, s5
	s_add_co_i32 s22, s4, s5
	s_load_b256 s[4:11], s[0:1], 0x5c nv
	s_mul_hi_u32 s22, s21, s22
	s_mul_i32 s23, s22, s20
	s_sub_co_i32 s21, s21, s23
	s_add_co_i32 s23, s22, 1
	s_sub_co_i32 s26, s21, s20
	s_cmp_ge_u32 s21, s20
	s_cselect_b32 s22, s23, s22
	s_cselect_b32 s21, s26, s21
	s_add_co_i32 s23, s22, 1
	s_cmp_ge_u32 s21, s20
	s_mov_b32 s21, s41
	s_cselect_b32 s20, s23, s22
	s_mov_b32 s23, s41
	s_xor_b32 s29, s20, s27
	s_mov_b32 s20, s41
	s_sub_co_i32 s22, s29, s27
	s_wait_kmcnt 0x0
	s_mov_b32 s26, s9
	s_mul_i32 s30, s22, s53
	s_mov_b32 s22, s41
	s_cmp_lg_u32 s25, s30
	s_mov_b32 s56, s7
	s_cselect_b32 s31, -1, 0
	s_cmp_lt_i32 s24, 0
	s_mov_b32 s24, s5
	s_cselect_b32 s33, -1, 0
	s_and_b32 s31, s31, s33
	s_sub_co_ci_u32 s33, s29, s27
	s_not_b32 s17, s17
	s_bfe_u32 s58, ttmp8, 0x50019
	s_add_co_i32 s17, s52, s17
	s_lshl_b32 s29, s58, 5
	s_lshl_b32 s74, s17, 7
	s_mul_i32 s36, s33, s49
	s_add_co_i32 s71, s74, s29
	s_sub_co_i32 s34, s25, s30
	s_add_co_i32 s54, s36, s71
	s_ashr_i32 s25, s5, 31
	s_ashr_i32 s27, s9, 31
	s_ashr_i32 s35, s34, 31
	s_ashr_i32 s55, s54, 31
	s_mul_i32 s53, s58, 0x2200
	s_mul_u64 s[30:31], s[34:35], s[26:27]
	s_sub_co_i32 s51, s49, s71
	s_mul_u64 s[36:37], s[54:55], s[24:25]
	s_add_co_i32 s29, s53, 0x46000
	s_lshl_b64 s[30:31], s[30:31], 1
	s_max_i32 s40, s51, 0
	s_lshl_b64 s[36:37], s[36:37], 1
	s_cmp_lg_u32 s5, 0x80000000
	s_add_nc_u64 s[18:19], s[18:19], s[36:37]
	s_cselect_b32 s25, s25, 0
	s_cselect_b32 s24, s5, 0x80
	s_cmp_lg_u32 s9, 0x80000000
	s_add_nc_u64 s[30:31], s[18:19], s[30:31]
	s_cselect_b32 s17, s9, 0x80
	s_cselect_b32 s5, s27, 0
	s_lshr_b64 s[18:19], s[24:25], 16
	s_lshl_b32 s9, s24, 16
	s_lshr_b32 s19, s24, 16
	s_and_b32 s5, s5, 0xffff
	s_and_b32 s24, s18, 0xffff0000
	s_bitset1_b32 s31, 31
	s_or_b32 s18, s5, s9
	s_or_b32 s19, s24, s19
	s_sub_co_i32 s73, s50, s49
	tensor_load_to_lds s[28:31], s[12:19], s[40:43], s[20:23]
	s_or_b32 s5, s74, 0x7f
	s_add_co_i32 s72, s49, -1
	s_add_co_i32 s73, s73, s48
	s_min_i32 s5, s5, s72
	s_mul_i32 s64, s33, s50
	s_add_co_i32 s5, s73, s5
	s_mov_b32 s30, s6
	s_mov_b32 s14, s10
	s_ashr_i32 s65, s64, 31
	s_ashr_i32 s31, s6, 31
	s_ashr_i32 s15, s10, 31
	s_ashr_i32 s57, s7, 31
	s_add_co_i32 s5, s5, 1
	s_mov_b32 s16, s11
	s_ashr_i32 s17, s11, 31
	s_mul_u64 s[10:11], s[64:65], s[30:31]
	s_mul_u64 s[14:15], s[34:35], s[14:15]
	s_mul_u64 s[18:19], s[64:65], s[56:57]
	s_min_i32 s65, s5, s50
	s_mul_u64 s[16:17], s[34:35], s[16:17]
	s_lshl_b64 s[66:67], s[14:15], 1
	s_lshl_b64 s[14:15], s[18:19], 1
	s_max_i32 s5, s65, 1
	s_lshl_b64 s[10:11], s[10:11], 1
	s_lshl_b64 s[62:63], s[16:17], 1
	s_add_nc_u64 s[22:23], s[2:3], s[14:15]
	s_min_i32 s14, s5, 0x80
	s_add_nc_u64 s[10:11], s[60:61], s[10:11]
	s_cmp_lg_u32 s6, 0x80000000
	s_mov_b64 s[38:39], s[30:31]
	s_add_nc_u64 s[38:39], s[10:11], s[66:67]
	s_cselect_b32 s11, s31, 0
	s_cselect_b32 s17, s6, 0x80
	s_and_b32 s6, s58, 3
	s_and_b32 s18, s11, 0xffff
	s_lshl_b32 s9, s6, 5
	s_lshl_b32 s40, s6, 6
	s_mul_i32 s35, s6, 0x2200
	s_mul_i32 s55, s6, 0x2400
	s_sub_co_i32 s6, s14, s9
	s_mov_b32 s10, s17
	s_max_i32 s6, s6, 0
	s_mul_u64 s[10:11], s[10:11], s[40:41]
	s_lshl_b32 s6, s6, 16
	v_add_nc_u32_e32 v191 /*v703*/, s29, v192 /*v704*/
	s_or_b32 s14, s6, 0x7fff
	s_cmp_lg_u32 s7, 0x80000000
	s_add_nc_u64 s[42:43], s[22:23], s[62:63]
	s_cselect_b32 s25, s7, 0x80
	s_cselect_b32 s7, s57, 0
	s_mov_b32 s6, s25
	s_add_nc_u64 s[38:39], s[10:11], s[38:39]
	s_mul_u64 s[58:59], s[6:7], s[40:41]
	s_mov_b64 s[36:37], s[28:29]
	s_mov_b32 s16, 32
	s_mov_b32 s15, 0x800000
	s_add_co_i32 s55, s55, 0x8800
	s_add_nc_u64 s[42:43], s[58:59], s[42:43]
	s_mov_b64 s[46:47], s[30:31]
	s_mov_b64 s[44:45], s[28:29]
	s_mov_b32 s19, s41
	s_mov_b32 s37, s35
	s_bitset1_b32 s39, 31
	s_mov_b32 s20, 0xf510000
	s_mov_b32 s21, s13
	s_mov_b32 s27, s41
	s_mov_b32 s23, s15
	s_mov_b32 s24, s16
	s_mov_b32 s22, s14
	s_mov_b32 s45, s55
	s_and_b32 s26, s7, 0xffff
	s_or_b32 s47, s43, 0x80000000
	s_mov_b32 s46, s42
	s_cmp_lt_i32 s65, 0x81
	s_wait_tensorcnt 0x0
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x8802
	ds_load_b128 v[58:61], v191 /*v703*/
	ds_load_b128 v[62:65], v191 /*v703*/ offset:32
	ds_load_b128 v[50:53], v191 /*v703*/ offset:64
	ds_load_b128 v[54:57], v191 /*v703*/ offset:96
	ds_load_b128 v[42:45], v191 /*v703*/ offset:128
	ds_load_b128 v[46:49], v191 /*v703*/ offset:160
	ds_load_b128 v[34:37], v191 /*v703*/ offset:192
	ds_load_b128 v[38:41], v191 /*v703*/ offset:224
	ds_load_b128 v[26:29], v191 /*v703*/ offset:4352
	ds_load_b128 v[30:33], v191 /*v703*/ offset:4384
	ds_load_b128 v[18:21], v191 /*v703*/ offset:4416
	ds_load_b128 v[22:25], v191 /*v703*/ offset:4448
	ds_load_b128 v[10:13], v191 /*v703*/ offset:4480
	ds_load_b128 v[14:17], v191 /*v703*/ offset:4512
	ds_load_b128 v[2:5], v191 /*v703*/ offset:4544
	ds_load_b128 v[6:9], v191 /*v703*/ offset:4576
	tensor_load_to_lds s[36:39], s[12:19]
	tensor_load_to_lds s[44:47], s[20:27]
	s_load_b128 s[36:39], s[0:1], 0x7c nv
	s_set_vgpr_msb 0x200
	s_cbranch_scc1 .LBB0_2
	s_add_co_i32 s6, s64, 0x80
	s_min_u32 s14, s65, 0x100
	s_ashr_i32 s7, s6, 31
	s_sub_co_i32 s14, s14, s9
	s_mul_u64 s[22:23], s[6:7], s[30:31]
	s_mul_u64 s[6:7], s[6:7], s[56:57]
	s_lshl_b64 s[22:23], s[22:23], 1
	s_lshl_b64 s[6:7], s[6:7], 1
	s_add_nc_u64 s[22:23], s[60:61], s[22:23]
	s_add_nc_u64 s[6:7], s[2:3], s[6:7]
	s_add_nc_u64 s[22:23], s[22:23], s[66:67]
	s_max_u32 s14, s14, 0x80
	s_add_co_i32 s19, s35, 0x11800
	s_add_nc_u64 s[22:23], s[10:11], s[22:23]
	s_add_nc_u64 s[6:7], s[6:7], s[62:63]
	s_lshl_b32 s14, s14, 16
	s_mov_b64 s[46:47], s[30:31]
	s_mov_b64 s[44:45], s[28:29]
	s_or_b32 s47, s23, 0x80000000
	s_mov_b32 s45, s19
	s_mov_b32 s46, s22
	s_add_co_i32 s14, s14, 0xff807fff
	s_mov_b32 s19, s41
	s_add_nc_u64 s[6:7], s[58:59], s[6:7]
	tensor_load_to_lds s[44:47], s[12:19]
	s_add_co_i32 s12, s55, 0x11800
	s_bitset1_b32 s7, 31
	s_mov_b64 s[46:47], s[30:31]
	s_mov_b64 s[44:45], s[28:29]
	s_mov_b32 s45, s12
	s_mov_b32 s46, s6
	s_mov_b32 s47, s7
	s_mov_b32 s21, s13
	s_mov_b32 s22, s14
	s_mov_b32 s23, s15
	s_mov_b32 s24, s16
	s_mov_b32 s27, s41
	tensor_load_to_lds s[44:47], s[20:27]
.LBB0_2:
	s_wait_tensorcnt 0x0
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_cmp_lt_i32 s65, 0x101
	s_barrier_wait -1
	s_cbranch_scc1 .LBB0_4
	s_add_co_i32 s6, s64, 0x100
	s_min_u32 s14, s65, 0x180
	s_ashr_i32 s7, s6, 31
	s_mov_b32 s40, 1
	s_mul_u64 s[12:13], s[6:7], s[56:57]
	s_mul_u64 s[6:7], s[6:7], s[30:31]
	s_lshl_b64 s[12:13], s[12:13], 1
	s_lshl_b64 s[6:7], s[6:7], 1
	s_add_nc_u64 s[12:13], s[2:3], s[12:13]
	s_add_nc_u64 s[6:7], s[60:61], s[6:7]
	s_add_nc_u64 s[20:21], s[12:13], s[62:63]
	s_add_nc_u64 s[6:7], s[6:7], s[66:67]
	s_sub_co_i32 s12, s14, s9
	s_add_nc_u64 s[42:43], s[10:11], s[6:7]
	s_max_u32 s6, s12, 0x100
	s_add_co_i32 s41, s35, 0x23000
	s_lshl_b32 s6, s6, 16
	s_bitset1_b32 s43, 31
	s_add_co_i32 s14, s6, 0xff007fff
	s_mov_b32 s19, 0
	s_mov_b32 s13, 0xffff0000
	s_mov_b32 s12, 0x7510000
	s_mov_b32 s22, s14
	tensor_load_to_lds s[40:43], s[12:19]
	s_add_nc_u64 s[42:43], s[58:59], s[20:21]
	s_add_co_i32 s41, s55, 0x23000
	s_bitset1_b32 s43, 31
	s_mov_b32 s20, 0xf510000
	s_mov_b32 s21, s13
	s_mov_b32 s23, s15
	s_mov_b32 s24, s16
	s_mov_b32 s27, s19
	tensor_load_to_lds s[40:43], s[20:27]
.LBB0_4:
	v_and_b32_e32 v0, 31, v0
	s_cmp_lt_i32 s65, 0x181
	s_cbranch_scc1 .LBB0_6
	s_add_co_i32 s6, s64, 0x180
	s_min_u32 s14, s65, 0x200
	s_ashr_i32 s7, s6, 31
	s_mov_b32 s40, 1
	s_mul_u64 s[12:13], s[6:7], s[56:57]
	s_mul_u64 s[6:7], s[6:7], s[30:31]
	s_lshl_b64 s[12:13], s[12:13], 1
	s_lshl_b64 s[6:7], s[6:7], 1
	s_add_nc_u64 s[12:13], s[2:3], s[12:13]
	s_add_nc_u64 s[6:7], s[60:61], s[6:7]
	s_add_nc_u64 s[20:21], s[12:13], s[62:63]
	s_add_nc_u64 s[6:7], s[6:7], s[66:67]
	s_sub_co_i32 s12, s14, s9
	s_add_nc_u64 s[42:43], s[10:11], s[6:7]
	s_max_u32 s6, s12, 0x180
	s_add_co_i32 s41, s35, 0x34800
	s_lshl_b32 s6, s6, 16
	s_bitset1_b32 s43, 31
	s_add_co_i32 s14, s6, 0xfe807fff
	s_mov_b32 s19, 0
	s_mov_b32 s13, 0xffff0000
	s_mov_b32 s12, 0x7510000
	s_mov_b32 s22, s14
	tensor_load_to_lds s[40:43], s[12:19]
	s_add_nc_u64 s[42:43], s[58:59], s[20:21]
	s_add_co_i32 s41, s55, 0x34800
	s_bitset1_b32 s43, 31
	s_mov_b32 s20, 0xf510000
	s_mov_b32 s21, s13
	s_mov_b32 s23, s15
	s_mov_b32 s24, s16
	s_mov_b32 s27, s19
	tensor_load_to_lds s[40:43], s[20:27]
.LBB0_6:
	v_cvt_pk_bf16_f32 v1, s4, s0
	s_add_co_i32 s4, s73, s74
	s_set_vgpr_msb 0x80
	v_lshrrev_b32_e32 v193 /*v705*/, 4, v0
	s_addk_co_i32 s5, 0x7f
	s_max_i32 s4, s4, 0
	s_lshr_b32 s44, s5, 7
	s_add_co_i32 s5, s4, 1
	s_set_vgpr_msb 0x8088
	v_lshlrev_b32_e32 v194 /*v706*/, 3, v193 /*v705*/
	s_ashr_i32 s4, s5, 31
	s_set_vgpr_msb 0x8820
	v_pk_mul_bf16 v186, v1, v4 op_sel_hi:[0,1]
	s_lshr_b32 s4, s4, 25
	s_add_co_i32 s6, s44, -1
	s_add_co_i32 s4, s5, s4
	v_and_or_b32 v4, v0, 7, v194 /*v706*/
	s_and_b32 s7, s4, 0xffffff80
	s_ashr_i32 s12, s4, 7
	s_cmp_lg_u32 s5, s7
	v_pk_mul_bf16 v71, v1, v65 op_sel_hi:[0,1]
	v_pk_mul_bf16 v70, v1, v64 op_sel_hi:[0,1]
	v_pk_mul_bf16 v69, v1, v63 op_sel_hi:[0,1]
	v_pk_mul_bf16 v68, v1, v62 op_sel_hi:[0,1]
	v_pk_mul_bf16 v67, v1, v61 op_sel_hi:[0,1]
	v_pk_mul_bf16 v66, v1, v60 op_sel_hi:[0,1]
	v_pk_mul_bf16 v65, v1, v59 op_sel_hi:[0,1]
	v_pk_mul_bf16 v64, v1, v58 op_sel_hi:[0,1]
	v_pk_mul_bf16 v79, v1, v57 op_sel_hi:[0,1]
	v_pk_mul_bf16 v78, v1, v56 op_sel_hi:[0,1]
	v_pk_mul_bf16 v77, v1, v55 op_sel_hi:[0,1]
	v_pk_mul_bf16 v76, v1, v54 op_sel_hi:[0,1]
	v_pk_mul_bf16 v75, v1, v53 op_sel_hi:[0,1]
	v_pk_mul_bf16 v74, v1, v52 op_sel_hi:[0,1]
	v_pk_mul_bf16 v73, v1, v51 op_sel_hi:[0,1]
	v_pk_mul_bf16 v72, v1, v50 op_sel_hi:[0,1]
	v_pk_mul_bf16 v111, v1, v49 op_sel_hi:[0,1]
	v_pk_mul_bf16 v110, v1, v48 op_sel_hi:[0,1]
	v_pk_mul_bf16 v109, v1, v47 op_sel_hi:[0,1]
	v_pk_mul_bf16 v108, v1, v46 op_sel_hi:[0,1]
	v_pk_mul_bf16 v107, v1, v45 op_sel_hi:[0,1]
	v_pk_mul_bf16 v106, v1, v44 op_sel_hi:[0,1]
	v_pk_mul_bf16 v105, v1, v43 op_sel_hi:[0,1]
	v_pk_mul_bf16 v104, v1, v42 op_sel_hi:[0,1]
	v_pk_mul_bf16 v135, v1, v41 op_sel_hi:[0,1]
	v_pk_mul_bf16 v134, v1, v40 op_sel_hi:[0,1]
	v_pk_mul_bf16 v133, v1, v39 op_sel_hi:[0,1]
	v_pk_mul_bf16 v132, v1, v38 op_sel_hi:[0,1]
	v_pk_mul_bf16 v131, v1, v37 op_sel_hi:[0,1]
	v_pk_mul_bf16 v130, v1, v36 op_sel_hi:[0,1]
	v_pk_mul_bf16 v129, v1, v35 op_sel_hi:[0,1]
	v_pk_mul_bf16 v128, v1, v34 op_sel_hi:[0,1]
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
	v_pk_mul_bf16 v185, v1, v3 op_sel_hi:[0,1]
	v_pk_mul_bf16 v184, v1, v2 op_sel_hi:[0,1]
	v_mul_u32_u24_e32 v1, 0x120, v4
	v_lshlrev_b32_e32 v0, 1, v0
	s_cselect_b32 s7, -1, 0
	s_cmp_lt_i32 s5, 0
	s_set_vgpr_msb 0x2088
	v_add_nc_u32_e32 v200 /*v712*/, 0x11800, v192 /*v704*/
	s_cselect_b32 s5, -1, 0
	s_set_vgpr_msb 0x8800
	v_and_or_b32 v0, v0, 16, v1
	s_and_b32 s5, s5, s7
	s_sub_co_ci_u32 s5, s12, 0
	s_set_vgpr_msb 0x88
	v_add_nc_u32_e32 v197 /*v709*/, 0x23000, v192 /*v704*/
	s_min_i32 s5, s5, s6
	v_add_nc_u32_e32 v199 /*v711*/, 0x34800, v192 /*v704*/
	s_max_i32 s65, s5, 0
	s_set_vgpr_msb 0x8880
	v_add_nc_u32_e32 v196 /*v708*/, 0x8800, v0
	v_or_b32_e32 v195 /*v707*/, 0x1a000, v0
	v_add_nc_u32_e32 v201 /*v713*/, 0x2b800, v0
	v_add_nc_u32_e32 v204 /*v716*/, 0x3d000, v0
	s_mov_b32 s19, 0
	s_and_b32 s46, s65, 0x7ffffffe
	s_mov_b32 s4, 1
	s_mov_b32 s47, s19
	s_add_nc_u64 s[60:61], s[60:61], s[66:67]
	s_cmp_eq_u32 s46, 0
	s_add_nc_u64 s[62:63], s[2:3], s[62:63]
	s_set_vgpr_msb 0x8000
	s_cbranch_scc1 .LBB0_21
	s_add_co_i32 s2, s69, s70
	s_add_co_i32 s3, s52, s68
	s_lshl_b32 s2, s2, 7
	s_lshl_b32 s3, s3, 7
	s_not_b32 s2, s2
	v_mov_b32_e32 v0, 0
	s_add_co_i32 s2, s2, s3
	s_add_co_i32 s3, s50, s48
	s_min_i32 s2, s2, s72
	s_set_vgpr_msb 0x82
	v_dual_mov_b32 v203 /*v715*/, 0xf149f2ca :: v_dual_mov_b32 v8 /*v520*/, v204 /*v716*/
	s_add_co_i32 s3, s3, s2
	s_set_vgpr_msb 0x8200
	v_dual_mov_b32 v1, v0 :: v_dual_mov_b32 v2, v0
	v_dual_mov_b32 v3, v0 :: v_dual_mov_b32 v4, v0
	v_dual_mov_b32 v5, v0 :: v_dual_mov_b32 v6, v0
	v_mov_b32_e32 v7, v0
	s_sub_co_i32 s2, s3, s49
	v_mov_b64_e32 v[10:11], v[2:3]
	s_add_co_i32 s2, s2, 1
	v_mov_b64_e32 v[12:13], v[4:5]
	s_min_i32 s2, s50, s2
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
	v_mov_b64_e32 v[118:119], v[6:7]
	v_mov_b64_e32 v[116:117], v[4:5]
	v_mov_b64_e32 v[114:115], v[2:3]
	v_mov_b64_e32 v[112:113], v[0:1]
	v_mov_b64_e32 v[126:127], v[6:7]
	v_mov_b64_e32 v[124:125], v[4:5]
	v_mov_b64_e32 v[122:123], v[2:3]
	v_mov_b64_e32 v[120:121], v[0:1]
	v_mov_b64_e32 v[142:143], v[6:7]
	v_mov_b64_e32 v[140:141], v[4:5]
	v_mov_b64_e32 v[138:139], v[2:3]
	v_mov_b64_e32 v[136:137], v[0:1]
	v_mov_b64_e32 v[150:151], v[6:7]
	v_mov_b64_e32 v[148:149], v[4:5]
	v_mov_b64_e32 v[146:147], v[2:3]
	v_mov_b64_e32 v[144:145], v[0:1]
	v_mov_b64_e32 v[158:159], v[6:7]
	v_mov_b64_e32 v[156:157], v[4:5]
	v_mov_b64_e32 v[154:155], v[2:3]
	v_mov_b64_e32 v[152:153], v[0:1]
	s_set_vgpr_msb 0x82
	v_dual_mov_b32 v9 /*v521*/, v201 /*v713*/ :: v_dual_mov_b32 v10 /*v522*/, v199 /*v711*/
	v_dual_mov_b32 v11 /*v523*/, v197 /*v709*/ :: v_dual_mov_b32 v202 /*v714*/, v192 /*v704*/
	s_set_vgpr_msb 0x8280
	v_dual_mov_b32 v198 /*v710*/, 0xf149f2ca :: v_dual_mov_b32 v121 /*v633*/, v0
	v_mov_b32_e32 v120 /*v632*/, v0
	s_max_i32 s2, s2, 1
	s_add_co_i32 s40, s64, 0x280
	s_add_co_i32 s45, s2, 0xfffffd80
	s_mov_b64 s[42:43], 0
	s_mov_b32 s66, 0x76543210
	s_mov_b32 s28, 0x3fb8aa3b
	s_mov_b32 s12, 0x7510000
	s_set_vgpr_msb 0x8000
	s_branch .LBB0_9
.LBB0_8:
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 42
	v_pk_fma_f32 v[192:193], v[126:127] /*v[638:639]*/, v[120:121] /*v[632:633]*/, v[124:125] /*v[636:637]*/
	s_add_nc_u64 s[42:43], s[42:43], 2
	s_set_vgpr_msb 0x2a82
	v_dual_mov_b32 v8 /*v520*/, v204 /*v716*/ :: v_dual_mov_b32 v9 /*v521*/, v201 /*v713*/
	v_cmp_lt_u64_e64 s2, s[42:43], s[46:47]
	s_set_vgpr_msb 0x8208
	v_pk_add_f32 v[192:193], v[192:193], v[122:123] /*v[634:635]*/
	s_set_vgpr_msb 0x882
	v_dual_mov_b32 v10 /*v522*/, v199 /*v711*/ :: v_dual_mov_b32 v11 /*v523*/, v197 /*v709*/
	s_addk_co_i32 s45, 0xff00
	s_addk_co_i32 s40, 0x100
	s_set_vgpr_msb 0x8200
	v_pk_fma_f32 v[192:193], v[212:213], v[192:193], v[210:211]
	s_and_b32 vcc_lo, exec_lo, s2
	s_set_vgpr_msb 0x80
	v_pk_add_f32 v[120:121] /*v[632:633]*/, v[192:193], v[208:209]
	s_set_vgpr_msb 0x8000
	s_cbranch_vccz .LBB0_22
.LBB0_9:
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 2
	ds_load_b128 v[192:195], v202 /*v714*/
	ds_load_b128 v[196:199], v202 /*v714*/ offset:32
	ds_load_b128 v[200:203], v202 /*v714*/ offset:64
	ds_load_b128 v[204:207], v202 /*v714*/ offset:96
	ds_load_b128 v[208:211], v202 /*v714*/ offset:128
	ds_load_b128 v[212:215], v202 /*v714*/ offset:160
	ds_load_b128 v[240:243], v202 /*v714*/ offset:4544
	ds_load_b128 v[244:247], v202 /*v714*/ offset:4576
	s_set_vgpr_msb 0x242
	ds_load_b128 v[8:11] /*v[264:267]*/, v202 /*v714*/ offset:8704
	ds_load_b128 v[12:15] /*v[268:271]*/, v202 /*v714*/ offset:8736
	ds_load_b128 v[80:83] /*v[336:339]*/, v202 /*v714*/ offset:8768
	ds_load_b128 v[84:87] /*v[340:343]*/, v202 /*v714*/ offset:8800
	ds_load_b128 v[104:107] /*v[360:363]*/, v202 /*v714*/ offset:8832
	ds_load_b128 v[108:111] /*v[364:367]*/, v202 /*v714*/ offset:8864
	ds_load_b128 v[128:131] /*v[384:387]*/, v202 /*v714*/ offset:8896
	ds_load_b128 v[132:135] /*v[388:391]*/, v202 /*v714*/ offset:8928
	ds_load_b128 v[144:147] /*v[400:403]*/, v202 /*v714*/ offset:13056
	ds_load_b128 v[148:151] /*v[404:407]*/, v202 /*v714*/ offset:13088
	ds_load_b128 v[168:171] /*v[424:427]*/, v202 /*v714*/ offset:13120
	ds_load_b128 v[172:175] /*v[428:431]*/, v202 /*v714*/ offset:13152
	ds_load_b128 v[224:227] /*v[480:483]*/, v202 /*v714*/ offset:21824
	ds_load_b128 v[228:231] /*v[484:487]*/, v202 /*v714*/ offset:21856
	ds_load_b128 v[232:235] /*v[488:491]*/, v202 /*v714*/ offset:21888
	ds_load_b128 v[236:239] /*v[492:495]*/, v202 /*v714*/ offset:21920
	ds_load_b128 v[240:243] /*v[496:499]*/, v202 /*v714*/ offset:21952
	ds_load_b128 v[244:247] /*v[500:503]*/, v202 /*v714*/ offset:21984
	ds_load_b128 v[248:251] /*v[504:507]*/, v202 /*v714*/ offset:26112
	ds_load_b128 v[252:255] /*v[508:511]*/, v202 /*v714*/ offset:26144
	s_set_vgpr_msb 0x4282
	ds_load_b128 v[0:3] /*v[512:515]*/, v202 /*v714*/ offset:26176
	ds_load_b128 v[4:7] /*v[516:519]*/, v202 /*v714*/ offset:26208
	ds_load_b128 v[12:15] /*v[524:527]*/, v202 /*v714*/ offset:26240
	ds_load_b128 v[16:19] /*v[528:531]*/, v202 /*v714*/ offset:26272
	ds_load_b128 v[20:23] /*v[532:535]*/, v202 /*v714*/ offset:26304
	ds_load_b128 v[24:27] /*v[536:539]*/, v202 /*v714*/ offset:26336
	ds_load_b128 v[44:47] /*v[556:559]*/, v202 /*v714*/ offset:30656
	ds_load_b128 v[48:51] /*v[560:563]*/, v202 /*v714*/ offset:30688
	ds_load_b128 v[52:55] /*v[564:567]*/, v200 /*v712*/
	ds_load_b128 v[56:59] /*v[568:571]*/, v200 /*v712*/ offset:32
	ds_load_b128 v[60:63] /*v[572:575]*/, v200 /*v712*/ offset:64
	ds_load_b128 v[64:67] /*v[576:579]*/, v200 /*v712*/ offset:96
	ds_load_b128 v[68:71] /*v[580:583]*/, v200 /*v712*/ offset:128
	ds_load_b128 v[72:75] /*v[584:587]*/, v200 /*v712*/ offset:160
	ds_load_b128 v[138:141] /*v[650:653]*/, v200 /*v712*/ offset:8832
	ds_load_b128 v[142:145] /*v[654:657]*/, v200 /*v712*/ offset:8864
	ds_load_b128 v[146:149] /*v[658:661]*/, v200 /*v712*/ offset:8896
	ds_load_b128 v[150:153] /*v[662:665]*/, v200 /*v712*/ offset:8928
	ds_load_b128 v[154:157] /*v[666:669]*/, v200 /*v712*/ offset:13056
	ds_load_b128 v[158:161] /*v[670:673]*/, v200 /*v712*/ offset:13088
	ds_load_b128 v[76:79] /*v[588:591]*/, v200 /*v712*/ offset:192
	ds_load_b128 v[80:83] /*v[592:595]*/, v200 /*v712*/ offset:224
	ds_load_b128 v[84:87] /*v[596:599]*/, v200 /*v712*/ offset:4352
	ds_load_b128 v[88:91] /*v[600:603]*/, v200 /*v712*/ offset:4384
	ds_load_b128 v[92:95] /*v[604:607]*/, v200 /*v712*/ offset:4416
	ds_load_b128 v[96:99] /*v[608:611]*/, v200 /*v712*/ offset:4448
	s_set_vgpr_msb 0x8242
	ds_load_b128 v[176:179] /*v[432:435]*/, v200 /*v712*/ offset:13120
	ds_load_b128 v[180:183] /*v[436:439]*/, v200 /*v712*/ offset:13152
	ds_load_b128 v[160:163] /*v[416:419]*/, v200 /*v712*/ offset:13184
	ds_load_b128 v[164:167] /*v[420:423]*/, v200 /*v712*/ offset:13216
	ds_load_b128 v[136:139] /*v[392:395]*/, v200 /*v712*/ offset:13248
	ds_load_b128 v[140:143] /*v[396:399]*/, v200 /*v712*/ offset:13280
	ds_load_b128 v[112:115] /*v[368:371]*/, v200 /*v712*/ offset:17408
	ds_load_b128 v[116:119] /*v[372:375]*/, v200 /*v712*/ offset:17440
	ds_load_b128 v[96:99] /*v[352:355]*/, v200 /*v712*/ offset:17472
	ds_load_b128 v[100:103] /*v[356:359]*/, v200 /*v712*/ offset:17504
	ds_load_b128 v[88:91] /*v[344:347]*/, v200 /*v712*/ offset:17536
	ds_load_b128 v[92:95] /*v[348:351]*/, v200 /*v712*/ offset:17568
	ds_load_b128 v[72:75] /*v[328:331]*/, v200 /*v712*/ offset:17600
	ds_load_b128 v[76:79] /*v[332:335]*/, v200 /*v712*/ offset:17632
	ds_load_b128 v[56:59] /*v[312:315]*/, v200 /*v712*/ offset:21760
	ds_load_b128 v[60:63] /*v[316:319]*/, v200 /*v712*/ offset:21792
	ds_load_b128 v[48:51] /*v[304:307]*/, v200 /*v712*/ offset:21824
	ds_load_b128 v[52:55] /*v[308:311]*/, v200 /*v712*/ offset:21856
	ds_load_b128 v[40:43] /*v[296:299]*/, v200 /*v712*/ offset:21888
	ds_load_b128 v[44:47] /*v[300:303]*/, v200 /*v712*/ offset:21920
	ds_load_b128 v[32:35] /*v[288:291]*/, v200 /*v712*/ offset:21952
	ds_load_b128 v[36:39] /*v[292:295]*/, v200 /*v712*/ offset:21984
	s_set_vgpr_msb 0x4282
	ds_load_b128 v[100:103] /*v[612:615]*/, v200 /*v712*/ offset:4480
	ds_load_b128 v[104:107] /*v[616:619]*/, v200 /*v712*/ offset:4512
	ds_load_b128 v[108:111] /*v[620:623]*/, v200 /*v712*/ offset:4544
	ds_load_b128 v[112:115] /*v[624:627]*/, v200 /*v712*/ offset:4576
	ds_load_b128 v[122:125] /*v[634:637]*/, v200 /*v712*/ offset:8704
	ds_load_b128 v[126:129] /*v[638:641]*/, v200 /*v712*/ offset:8736
	ds_load_b128 v[130:133] /*v[642:645]*/, v200 /*v712*/ offset:8768
	ds_load_b128 v[134:137] /*v[646:649]*/, v200 /*v712*/ offset:8800
	v_dual_mov_b32 v197 /*v709*/, v202 /*v714*/ :: v_dual_mov_b32 v199 /*v711*/, v200 /*v712*/
	s_set_vgpr_msb 0x8241
	s_wait_dscnt 0x38
	v_wmma_f32_16x16x32_bf16 v[120:127] /*v[376:383]*/, v[248:255] /*v[504:511]*/, v[64:71], 0
	s_set_vgpr_msb 0x4182
	v_dual_mov_b32 v201 /*v713*/, v196 /*v708*/ :: v_dual_mov_b32 v196 /*v708*/, v9 /*v521*/
	s_set_vgpr_msb 0x8240
	v_wmma_f32_16x16x32_bf16 v[24:31] /*v[280:287]*/, v[192:199], v[64:71], 0
	v_wmma_f32_16x16x32_bf16 v[0:7] /*v[256:263]*/, v[192:199], v[160:167], 0
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x4002
	ds_load_b128 v[192:195], v202 /*v714*/ offset:192
	ds_load_b128 v[196:199], v202 /*v714*/ offset:224
	ds_load_b128 v[216:219], v202 /*v714*/ offset:4352
	ds_load_b128 v[220:223], v202 /*v714*/ offset:4384
	ds_load_b128 v[224:227], v202 /*v714*/ offset:4416
	ds_load_b128 v[228:231], v202 /*v714*/ offset:4448
	ds_load_b128 v[232:235], v202 /*v714*/ offset:4480
	ds_load_b128 v[236:239], v202 /*v714*/ offset:4512
	s_set_vgpr_msb 0x250
	v_wmma_f32_16x16x32_bf16 v[24:31] /*v[280:287]*/, v[200:207], v[72:79], v[24:31] /*v[280:287]*/
	v_wmma_f32_16x16x32_bf16 v[0:7] /*v[256:263]*/, v[200:207], v[168:175], v[0:7] /*v[256:263]*/
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x5002
	ds_load_b128 v[200:203], v202 /*v714*/ offset:13184
	ds_load_b128 v[204:207], v202 /*v714*/ offset:13216
	s_set_vgpr_msb 0x242
	ds_load_b128 v[184:187] /*v[440:443]*/, v202 /*v714*/ offset:13248
	ds_load_b128 v[188:191] /*v[444:447]*/, v202 /*v714*/ offset:13280
	ds_load_b128 v[192:195] /*v[448:451]*/, v202 /*v714*/ offset:17408
	ds_load_b128 v[196:199] /*v[452:455]*/, v202 /*v714*/ offset:17440
	s_set_vgpr_msb 0x4281
	v_wmma_f32_16x16x32_bf16 v[162:169] /*v[674:681]*/, v[8:15] /*v[264:271]*/, v[64:71], 0
	s_set_vgpr_msb 0x8141
	v_wmma_f32_16x16x32_bf16 v[64:71] /*v[320:327]*/, v[8:15] /*v[264:271]*/, v[160:167], 0
	s_set_vgpr_msb 0x4150
	v_wmma_f32_16x16x32_bf16 v[24:31] /*v[280:287]*/, v[208:215], v[104:111], v[24:31] /*v[280:287]*/
	v_wmma_f32_16x16x32_bf16 v[0:7] /*v[256:263]*/, v[208:215], v[176:183], v[0:7] /*v[256:263]*/
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x5002
	ds_load_b128 v[208:211], v202 /*v714*/ offset:17472
	ds_load_b128 v[212:215], v202 /*v714*/ offset:17504
	s_set_vgpr_msb 0x242
	ds_load_b128 v[200:203] /*v[456:459]*/, v202 /*v714*/ offset:17536
	ds_load_b128 v[204:207] /*v[460:463]*/, v202 /*v714*/ offset:17568
	ds_load_b128 v[208:211] /*v[464:467]*/, v202 /*v714*/ offset:17600
	ds_load_b128 v[212:215] /*v[468:471]*/, v202 /*v714*/ offset:17632
	ds_load_b128 v[216:219] /*v[472:475]*/, v202 /*v714*/ offset:21760
	ds_load_b128 v[220:223] /*v[476:479]*/, v202 /*v714*/ offset:21792
	s_set_vgpr_msb 0x42a1
	v_wmma_f32_16x16x32_bf16 v[162:169] /*v[674:681]*/, v[80:87] /*v[336:343]*/, v[72:79], v[162:169] /*v[674:681]*/
	s_set_vgpr_msb 0xa151
	v_wmma_f32_16x16x32_bf16 v[64:71] /*v[320:327]*/, v[80:87] /*v[336:343]*/, v[168:175], v[64:71] /*v[320:327]*/
	s_set_vgpr_msb 0x5101
	s_wait_dscnt 0x8
	v_wmma_f32_16x16x32_bf16 v[248:255], v[192:199] /*v[448:455]*/, v[64:71], 0
	s_set_vgpr_msb 0x1a1
	v_wmma_f32_16x16x32_bf16 v[170:177] /*v[682:689]*/, v[144:151] /*v[400:407]*/, v[64:71], 0
	v_wmma_f32_16x16x32_bf16 v[162:169] /*v[674:681]*/, v[104:111] /*v[360:367]*/, v[104:111], v[162:169] /*v[674:681]*/
	s_set_vgpr_msb 0xa151
	v_wmma_f32_16x16x32_bf16 v[64:71] /*v[320:327]*/, v[104:111] /*v[360:367]*/, v[176:183], v[64:71] /*v[320:327]*/
	v_wmma_f32_16x16x32_bf16 v[80:87] /*v[336:343]*/, v[144:151] /*v[400:407]*/, v[160:167], 0
	v_wmma_f32_16x16x32_bf16 v[104:111] /*v[360:367]*/, v[192:199] /*v[448:455]*/, v[160:167], 0
	s_set_vgpr_msb 0x51a1
	v_wmma_f32_16x16x32_bf16 v[170:177] /*v[682:689]*/, v[168:175] /*v[424:431]*/, v[72:79], v[170:177] /*v[682:689]*/
	s_set_vgpr_msb 0xa100
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x32_bf16 v[248:255], v[208:215], v[72:79], v[248:255]
	s_set_vgpr_msb 0x50
	v_wmma_f32_16x16x32_bf16 v[24:31] /*v[280:287]*/, v[192:199], v[128:135], v[24:31] /*v[280:287]*/
	v_wmma_f32_16x16x32_bf16 v[0:7] /*v[256:263]*/, v[192:199], v[184:191], v[0:7] /*v[256:263]*/
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x5002
	ds_load_b128 v[192:195], v202 /*v714*/ offset:30464
	ds_load_b128 v[196:199], v202 /*v714*/ offset:30496
	s_set_vgpr_msb 0x282
	ds_load_b128 v[28:31] /*v[540:543]*/, v202 /*v714*/ offset:30528
	ds_load_b128 v[32:35] /*v[544:547]*/, v202 /*v714*/ offset:30560
	ds_load_b128 v[36:39] /*v[548:551]*/, v202 /*v714*/ offset:30592
	ds_load_b128 v[40:43] /*v[552:555]*/, v202 /*v714*/ offset:30624
	s_wait_alu depctr_vm_vsrc(0)
	v_mov_b32_e32 v202 /*v714*/, v11 /*v523*/
	s_set_vgpr_msb 0x8281
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x32_bf16 v[178:185] /*v[690:697]*/, v[216:223] /*v[472:479]*/, v[64:71], 0
	s_set_vgpr_msb 0x8151
	v_wmma_f32_16x16x32_bf16 v[80:87] /*v[336:343]*/, v[168:175] /*v[424:431]*/, v[168:175], v[80:87] /*v[336:343]*/
	s_set_vgpr_msb 0x5150
	v_wmma_f32_16x16x32_bf16 v[104:111] /*v[360:367]*/, v[208:215], v[168:175], v[104:111] /*v[360:367]*/
	s_set_vgpr_msb 0x50a1
	v_wmma_f32_16x16x32_bf16 v[162:169] /*v[674:681]*/, v[128:135] /*v[384:391]*/, v[128:135], v[162:169] /*v[674:681]*/
	s_set_vgpr_msb 0xa151
	v_wmma_f32_16x16x32_bf16 v[64:71] /*v[320:327]*/, v[128:135] /*v[384:391]*/, v[184:191], v[64:71] /*v[320:327]*/
	v_wmma_f32_16x16x32_bf16 v[128:135] /*v[384:391]*/, v[216:223] /*v[472:479]*/, v[160:167], 0
	s_set_vgpr_msb 0x51a0
	v_wmma_f32_16x16x32_bf16 v[170:177] /*v[682:689]*/, v[200:207], v[104:111], v[170:177] /*v[682:689]*/
	s_set_vgpr_msb 0xa001
	v_wmma_f32_16x16x32_bf16 v[248:255], v[200:207] /*v[456:463]*/, v[104:111], v[248:255]
	s_set_vgpr_msb 0x141
	v_wmma_f32_16x16x32_bf16 v[144:151] /*v[400:407]*/, v[248:255] /*v[504:511]*/, v[160:167], 0
	s_set_vgpr_msb 0x41a1
	v_wmma_f32_16x16x32_bf16 v[178:185] /*v[690:697]*/, v[224:231] /*v[480:487]*/, v[72:79], v[178:185] /*v[690:697]*/
	s_set_vgpr_msb 0xa140
	v_wmma_f32_16x16x32_bf16 v[16:23] /*v[272:279]*/, v[216:223], v[160:167], 0
	v_wmma_f32_16x16x32_bf16 v[152:159] /*v[408:415]*/, v[216:223], v[64:71], 0
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x4015
	v_max3_num_f32 v216, v24 /*v280*/, v25 /*v281*/, v26 /*v282*/
	v_max3_num_f32 v217, v0 /*v256*/, v1 /*v257*/, v2 /*v258*/
	v_max3_num_f32 v218, v27 /*v283*/, v28 /*v284*/, v29 /*v285*/
	s_set_vgpr_msb 0x1550
	v_wmma_f32_16x16x32_bf16 v[80:87] /*v[336:343]*/, v[200:207], v[176:183], v[80:87] /*v[336:343]*/
	s_set_vgpr_msb 0x5015
	v_max3_num_f32 v219, v3 /*v259*/, v4 /*v260*/, v5 /*v261*/
	s_set_vgpr_msb 0x1551
	v_wmma_f32_16x16x32_bf16 v[104:111] /*v[360:367]*/, v[200:207] /*v[456:463]*/, v[176:183], v[104:111] /*v[360:367]*/
	v_wmma_f32_16x16x32_bf16 v[128:135] /*v[384:391]*/, v[224:231] /*v[480:487]*/, v[168:175], v[128:135] /*v[384:391]*/
	s_set_vgpr_msb 0x51a1
	v_wmma_f32_16x16x32_bf16 v[170:177] /*v[682:689]*/, v[184:191] /*v[440:447]*/, v[128:135], v[170:177] /*v[682:689]*/
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0xa12a
	v_max3_num_f32 v201, v173 /*v685*/, v174 /*v686*/, v175 /*v687*/
	s_set_vgpr_msb 0x2a01
	v_wmma_f32_16x16x32_bf16 v[248:255], v[208:215] /*v[464:471]*/, v[128:135], v[248:255]
	s_set_vgpr_msb 0x12a
	v_max3_num_f32 v200, v170 /*v682*/, v171 /*v683*/, v172 /*v684*/
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x2a0a
	v_max3_num_f32 v204, v176 /*v688*/, v177 /*v689*/, v248
	s_set_vgpr_msb 0xa52
	v_wmma_f32_16x16x32_bf16 v[120:127] /*v[376:383]*/, v[0:7] /*v[512:519]*/, v[72:79], v[120:127] /*v[376:383]*/
	s_set_vgpr_msb 0x5200
	v_max3_num_f32 v205, v249, v250, v251
	v_max3_num_f32 v206, v252, v253, v254
	v_max3_num_f32 v201, v201, v204, v205
	s_set_vgpr_msb 0x52
	v_wmma_f32_16x16x32_bf16 v[144:151] /*v[400:407]*/, v[0:7] /*v[512:519]*/, v[168:175], v[144:151] /*v[400:407]*/
	s_set_vgpr_msb 0x5240
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x32_bf16 v[254:261] /*v[510:517]*/, v[192:199], v[64:71], 0
	s_set_vgpr_msb 0x40a1
	v_wmma_f32_16x16x32_bf16 v[178:185] /*v[690:697]*/, v[232:239] /*v[488:495]*/, v[104:111], v[178:185] /*v[690:697]*/
	s_set_vgpr_msb 0xa150
	v_wmma_f32_16x16x32_bf16 v[152:159] /*v[408:415]*/, v[224:231], v[72:79], v[152:159] /*v[408:415]*/
	v_wmma_f32_16x16x32_bf16 v[16:23] /*v[272:279]*/, v[224:231], v[168:175], v[16:23] /*v[272:279]*/
	s_set_vgpr_msb 0x5051
	v_wmma_f32_16x16x32_bf16 v[80:87] /*v[336:343]*/, v[184:191] /*v[440:447]*/, v[184:191], v[80:87] /*v[336:343]*/
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x5115
	v_max3_num_f32 v203, v83 /*v339*/, v84 /*v340*/, v85 /*v341*/
	s_set_vgpr_msb 0x1551
	v_wmma_f32_16x16x32_bf16 v[104:111] /*v[360:367]*/, v[208:215] /*v[464:471]*/, v[184:191], v[104:111] /*v[360:367]*/
	s_set_vgpr_msb 0x5115
	v_max3_num_f32 v202, v80 /*v336*/, v81 /*v337*/, v82 /*v338*/
	v_nop
	v_nop
	v_nop
	v_max3_num_f32 v204, v86 /*v342*/, v87 /*v343*/, v104 /*v360*/
	s_set_vgpr_msb 0x1540
	v_wmma_f32_16x16x32_bf16 v[168:175] /*v[424:431]*/, v[192:199], v[160:167], 0
	s_set_vgpr_msb 0x4015
	v_max3_num_f32 v205, v105 /*v361*/, v106 /*v362*/, v107 /*v363*/
	v_max3_num_f32 v207, v108 /*v364*/, v109 /*v365*/, v110 /*v366*/
	s_set_vgpr_msb 0x1500
	v_max3_num_f32 v203, v203, v204, v205
	s_set_vgpr_msb 0x51
	v_wmma_f32_16x16x32_bf16 v[128:135] /*v[384:391]*/, v[232:239] /*v[488:495]*/, v[176:183], v[128:135] /*v[384:391]*/
	s_set_vgpr_msb 0x5152
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[254:261] /*v[510:517]*/, v[28:35] /*v[540:547]*/, v[72:79], v[254:261] /*v[510:517]*/
	s_set_vgpr_msb 0x52a1
	v_wmma_f32_16x16x32_bf16 v[178:185] /*v[690:697]*/, v[240:247] /*v[496:503]*/, v[128:135], v[178:185] /*v[690:697]*/
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0xa128
	v_max3_num_f32 v204, v255, v178 /*v690*/, v179 /*v691*/
	s_set_vgpr_msb 0x2850
	v_wmma_f32_16x16x32_bf16 v[16:23] /*v[272:279]*/, v[232:239], v[176:183], v[16:23] /*v[272:279]*/
	s_set_vgpr_msb 0x502a
	v_max3_num_f32 v205, v180 /*v692*/, v181 /*v693*/, v182 /*v694*/
	v_max3_num_f32 v208, v183 /*v695*/, v184 /*v696*/, v185 /*v697*/
	s_set_vgpr_msb 0x2a00
	v_max3_num_f32 v204, v206, v204, v205
	s_set_vgpr_msb 0x50
	v_wmma_f32_16x16x32_bf16 v[152:159] /*v[408:415]*/, v[232:239], v[104:111], v[152:159] /*v[408:415]*/
	s_set_vgpr_msb 0x5052
	v_wmma_f32_16x16x32_bf16 v[120:127] /*v[376:383]*/, v[12:19] /*v[524:531]*/, v[104:111], v[120:127] /*v[376:383]*/
	v_wmma_f32_16x16x32_bf16 v[168:175] /*v[424:431]*/, v[28:35] /*v[540:547]*/, v[168:175], v[168:175] /*v[424:431]*/
	s_set_vgpr_msb 0x5251
	v_wmma_f32_16x16x32_bf16 v[128:135] /*v[384:391]*/, v[240:247] /*v[496:503]*/, v[184:191], v[128:135] /*v[384:391]*/
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x5115
	v_max3_num_f32 v205, v111 /*v367*/, v128 /*v384*/, v129 /*v385*/
	s_set_vgpr_msb 0x1552
	v_wmma_f32_16x16x32_bf16 v[144:151] /*v[400:407]*/, v[12:19] /*v[524:531]*/, v[176:183], v[144:151] /*v[400:407]*/
	s_set_vgpr_msb 0x5215
	v_max3_num_f32 v206, v130 /*v386*/, v131 /*v387*/, v132 /*v388*/
	v_max3_num_f32 v209, v133 /*v389*/, v134 /*v390*/, v135 /*v391*/
	s_set_vgpr_msb 0x1500
	v_max3_num_f32 v205, v207, v205, v206
	s_set_vgpr_msb 0x52
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[254:261] /*v[510:517]*/, v[36:43] /*v[548:555]*/, v[104:111], v[254:261] /*v[510:517]*/
	s_set_vgpr_msb 0x5250
	v_wmma_f32_16x16x32_bf16 v[152:159] /*v[408:415]*/, v[240:247], v[128:135], v[152:159] /*v[408:415]*/
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x5015
	v_max3_num_f32 v220, v30 /*v286*/, v31 /*v287*/, v152 /*v408*/
	s_set_vgpr_msb 0x1550
	v_wmma_f32_16x16x32_bf16 v[16:23] /*v[272:279]*/, v[240:247], v[184:191], v[16:23] /*v[272:279]*/
	s_set_vgpr_msb 0x5015
	v_max3_num_f32 v221, v153 /*v409*/, v154 /*v410*/, v155 /*v411*/
	v_max3_num_f32 v224, v156 /*v412*/, v157 /*v413*/, v158 /*v414*/
	s_set_vgpr_msb 0x1500
	v_max3_num_f32 v216, v216, v218, v220
	s_set_vgpr_msb 41
	v_max3_num_f32 v218, v159 /*v415*/, v162 /*v674*/, v163 /*v675*/
	s_set_vgpr_msb 0x292a
	v_max3_num_f32 v220, v167 /*v679*/, v168 /*v680*/, v169 /*v681*/
	s_set_vgpr_msb 0x2a15
	v_max3_num_f32 v222, v6 /*v262*/, v7 /*v263*/, v16 /*v272*/
	s_set_vgpr_msb 0x1552
	v_wmma_f32_16x16x32_bf16 v[120:127] /*v[376:383]*/, v[20:27] /*v[532:539]*/, v[128:135], v[120:127] /*v[376:383]*/
	s_set_vgpr_msb 0x5215
	v_max3_num_f32 v223, v17 /*v273*/, v18 /*v274*/, v19 /*v275*/
	v_max3_num_f32 v225, v20 /*v276*/, v21 /*v277*/, v22 /*v278*/
	s_set_vgpr_msb 0x1500
	v_max3_num_f32 v218, v221, v224, v218
	v_max3_num_f32 v217, v217, v219, v222
	s_set_vgpr_msb 42
	v_max3_num_f32 v219, v164 /*v676*/, v165 /*v677*/, v166 /*v678*/
	s_set_vgpr_msb 0x2a15
	v_max3_num_f32 v221, v23 /*v279*/, v64 /*v320*/, v65 /*v321*/
	v_max3_num_f32 v222, v66 /*v322*/, v67 /*v323*/, v68 /*v324*/
	s_set_vgpr_msb 0x1552
	v_wmma_f32_16x16x32_bf16 v[168:175] /*v[424:431]*/, v[36:43] /*v[548:555]*/, v[176:183], v[168:175] /*v[424:431]*/
	s_set_vgpr_msb 0x5215
	v_max3_num_f32 v206, v120 /*v376*/, v121 /*v377*/, v122 /*v378*/
	v_max3_num_f32 v207, v123 /*v379*/, v124 /*v380*/, v125 /*v381*/
	v_max_num_f32_e32 v210, v126 /*v382*/, v127 /*v383*/
	v_max3_num_f32 v224, v69 /*v325*/, v70 /*v326*/, v71 /*v327*/
	s_set_vgpr_msb 0x1500
	v_max3_num_f32 v200, v219, v220, v200
	v_max3_num_f32 v221, v223, v225, v221
	v_max3_num_f32 v206, v208, v206, v207
	s_set_vgpr_msb 0x52
	v_wmma_f32_16x16x32_bf16 v[144:151] /*v[400:407]*/, v[20:27] /*v[532:539]*/, v[184:191], v[144:151] /*v[400:407]*/
	s_set_vgpr_msb 0x5200
	v_max3_num_f32 v202, v222, v224, v202
	v_max3_num_f32 v200, v216, v218, v200
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x82
	ds_load_b128 v[12:15] /*v[524:527]*/, v200 /*v712*/ offset:26112
	ds_load_b128 v[16:19] /*v[528:531]*/, v200 /*v712*/ offset:26144
	ds_load_b128 v[20:23] /*v[532:535]*/, v200 /*v712*/ offset:26176
	ds_load_b128 v[24:27] /*v[536:539]*/, v200 /*v712*/ offset:26208
	s_set_vgpr_msb 0x8200
	v_max3_num_f32 v201, v201, v204, v206
	s_set_vgpr_msb 0x82
	ds_load_b128 v[28:31] /*v[540:543]*/, v200 /*v712*/ offset:26240
	ds_load_b128 v[32:35] /*v[544:547]*/, v200 /*v712*/ offset:26272
	ds_load_b128 v[36:39] /*v[548:551]*/, v200 /*v712*/ offset:26304
	ds_load_b128 v[40:43] /*v[552:555]*/, v200 /*v712*/ offset:26336
	s_set_vgpr_msb 0x8200
	v_max3_num_f32 v202, v217, v221, v202
	s_set_vgpr_msb 21
	v_max3_num_f32 v207, v144 /*v400*/, v145 /*v401*/, v146 /*v402*/
	s_set_vgpr_msb 0x1552
	v_wmma_f32_16x16x32_bf16 v[254:261] /*v[510:517]*/, v[44:51] /*v[556:563]*/, v[128:135], v[254:261] /*v[510:517]*/
	s_set_vgpr_msb 0x5215
	v_max3_num_f32 v208, v147 /*v403*/, v148 /*v404*/, v149 /*v405*/
	v_max_num_f32_e32 v211, v150 /*v406*/, v151 /*v407*/
	s_set_vgpr_msb 0x1500
	v_max3_num_f32 v204, v209, v207, v208
	v_nop
	s_set_vgpr_msb 41
	v_max3_num_f32 v192, v255 /*v511*/, v0 /*v512*/, v1 /*v513*/
	s_set_vgpr_msb 0x2952
	v_wmma_f32_16x16x32_bf16 v[168:175] /*v[424:431]*/, v[44:51] /*v[556:563]*/, v[184:191], v[168:175] /*v[424:431]*/
	s_set_vgpr_msb 0x522a
	v_max3_num_f32 v193, v2 /*v514*/, v3 /*v515*/, v4 /*v516*/
	s_set_vgpr_msb 0x2a00
	v_max3_num_f32 v203, v203, v205, v204
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x82
	ds_load_b128 v[44:47] /*v[556:559]*/, v200 /*v712*/ offset:30464
	ds_load_b128 v[48:51] /*v[560:563]*/, v200 /*v712*/ offset:30496
	s_set_vgpr_msb 0x8204
	v_max3_num_f32 v192, v210, v254 /*v510*/, v192
	s_set_vgpr_msb 0x420
	v_max3_num_f32 v192, v192, v193, v5 /*v517*/
	s_set_vgpr_msb 0x2015
	v_max3_num_f32 v194, v169 /*v425*/, v170 /*v426*/, v171 /*v427*/
	v_max3_num_f32 v204, v172 /*v428*/, v173 /*v429*/, v174 /*v430*/
	s_set_vgpr_msb 0x1502
	v_wmma_f32_16x16x32_bf16 v[216:223], v[52:59] /*v[564:571]*/, v[64:71], 0
	s_set_vgpr_msb 0x200
	v_max3_num_f32 v200, v200, v201, v192
	s_set_vgpr_msb 4
	v_max3_num_f32 v205, v211, v168 /*v424*/, v194
	s_set_vgpr_msb 0x410
	v_max3_num_f32 v201, v205, v204, v175 /*v431*/
	s_set_vgpr_msb 0x1002
	v_wmma_f32_16x16x32_bf16 v[192:199], v[52:59] /*v[564:571]*/, v[160:167], 0
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x282
	ds_load_b128 v[52:55] /*v[564:567]*/, v200 /*v712*/ offset:30528
	ds_load_b128 v[56:59] /*v[568:571]*/, v200 /*v712*/ offset:30560
	s_set_vgpr_msb 0x8200
	v_max3_num_f32 v240, v202, v203, v201
	v_dual_mov_b32 v204, v200 :: v_dual_mov_b32 v241, v240
	v_permlanex16_b32 v204, v204, s66, 0xfedcba98
	s_set_vgpr_msb 2
	v_wmma_f32_16x16x32_bf16 v[216:223], v[60:67] /*v[572:579]*/, v[72:79], v[216:223]
	s_set_vgpr_msb 0x200
	v_permlanex16_b32 v241, v241, s66, 0xfedcba98
	v_max_num_f32_e32 v242, v200, v204
	v_max_num_f32_e32 v240, v240, v241
	s_set_vgpr_msb 8
	v_sub_f32_e32 v243, v242, v198 /*v710*/
	v_max_num_f32_e32 v242, v242, v198 /*v710*/
	s_set_vgpr_msb 0x802
	v_wmma_f32_16x16x32_bf16 v[192:199], v[60:67] /*v[572:579]*/, v[168:175], v[192:199]
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x282
	ds_load_b128 v[60:63] /*v[572:575]*/, v200 /*v712*/ offset:30592
	ds_load_b128 v[64:67] /*v[576:579]*/, v200 /*v712*/ offset:30624
	s_set_vgpr_msb 0x8208
	v_sub_f32_e32 v241, v240, v203 /*v715*/
	s_set_vgpr_msb 0x802
	v_cmp_lt_f32_e32 vcc_lo, 0x41000000, v243
	v_max_num_f32_e32 v240, v203 /*v715*/, v240
	v_cmp_lt_f32_e64 s2, 0x41000000, v241
	s_cmp_eq_u32 vcc_lo, 0
	v_wmma_f32_16x16x32_bf16 v[216:223], v[68:75] /*v[580:587]*/, v[104:111], v[216:223]
	s_cselect_b32 s3, -1, 0
	s_cmp_lg_u32 s2, 0
	s_set_vgpr_msb 0x288
	v_cndmask_b32_e64 v206 /*v718*/, v242, v198 /*v710*/, s3
	s_cselect_b32 s3, -1, 0
	s_cmp_eq_u32 s2, 0
	s_cselect_b32 s2, -1, 0
	s_set_vgpr_msb 0x8802
	v_wmma_f32_16x16x32_bf16 v[192:199], v[68:75] /*v[580:587]*/, v[176:183], v[192:199]
	s_set_vgpr_msb 0x288
	v_cndmask_b32_e64 v205 /*v717*/, v240, v203 /*v715*/, s2
	v_mul_f32_e32 v6 /*v518*/, 0xbfb8aa3b, v206 /*v718*/
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x8882
	ds_load_b128 v[68:71] /*v[580:583]*/, v200 /*v712*/ offset:30656
	ds_load_b128 v[72:75] /*v[584:587]*/, v200 /*v712*/ offset:30688
	s_wait_alu depctr_vm_vsrc(0)
	v_mov_b32_e32 v200 /*v712*/, v10 /*v522*/
	s_set_vgpr_msb 0x8261
	v_pk_fma_f32 v[24:25] /*v[280:281]*/, v[24:25] /*v[280:281]*/, s[28:29], v[6:7] /*v[518:519]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6142
	v_wmma_f32_16x16x32_bf16 v[8:15] /*v[264:271]*/, v[154:161] /*v[666:673]*/, v[64:71], 0
	s_set_vgpr_msb 0x4261
	v_pk_fma_f32 v[26:27] /*v[282:283]*/, v[26:27] /*v[282:283]*/, s[28:29], v[6:7] /*v[518:519]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[28:29] /*v[284:285]*/, v[28:29] /*v[284:285]*/, s[28:29], v[6:7] /*v[518:519]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6120
	v_pk_fma_f32 v[248:249], v[248:249], s[28:29], v[6:7] /*v[518:519]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x2061
	v_exp_f32_e32 v184 /*v440*/, v24 /*v280*/
	v_exp_f32_e32 v186 /*v442*/, v25 /*v281*/
	v_nop
	v_pk_fma_f32 v[24:25] /*v[280:281]*/, v[30:31] /*v[286:287]*/, s[28:29], v[6:7] /*v[518:519]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v188 /*v444*/, v26 /*v282*/
	s_set_vgpr_msb 0x6102
	v_wmma_f32_16x16x32_bf16 v[240:247], v[154:161] /*v[666:673]*/, v[160:167], 0
	s_set_vgpr_msb 0x261
	v_exp_f32_e32 v190 /*v446*/, v27 /*v283*/
	v_nop
	v_pk_fma_f32 v[26:27] /*v[282:283]*/, v[152:153] /*v[408:409]*/, s[28:29], v[6:7] /*v[518:519]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v194 /*v450*/, v24 /*v280*/
	v_exp_f32_e32 v196 /*v452*/, v25 /*v281*/
	v_nop
	v_pk_fma_f32 v[24:25] /*v[280:281]*/, v[154:155] /*v[410:411]*/, s[28:29], v[6:7] /*v[518:519]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v192 /*v448*/, v28 /*v284*/
	v_exp_f32_e32 v152 /*v408*/, v29 /*v285*/
	s_set_vgpr_msb 0x6102
	v_wmma_f32_16x16x32_bf16 v[216:223], v[76:83] /*v[588:595]*/, v[128:135], v[216:223]
	s_set_vgpr_msb 0x261
	v_exp_f32_e32 v154 /*v410*/, v26 /*v282*/
	v_pk_fma_f32 v[28:29] /*v[284:285]*/, v[156:157] /*v[412:413]*/, s[28:29], v[6:7] /*v[518:519]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v156 /*v412*/, v27 /*v283*/
	v_nop
	s_set_vgpr_msb 0x6162
	v_pk_fma_f32 v[26:27] /*v[282:283]*/, v[162:163] /*v[674:675]*/, s[28:29], v[6:7] /*v[518:519]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6220
	v_pk_fma_f32 v[250:251], v[250:251], s[28:29], v[6:7] /*v[518:519]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x2060
	v_pk_fma_f32 v[224:225] /*v[480:481]*/, v[252:253], s[28:29], v[6:7] /*v[518:519]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6041
	v_exp_f32_e32 v198 /*v454*/, v28 /*v284*/
	s_set_vgpr_msb 0x4102
	v_wmma_f32_16x16x32_bf16 v[192:199], v[76:83] /*v[588:595]*/, v[184:191], v[192:199]
	s_set_vgpr_msb 0x241
	v_exp_f32_e32 v200 /*v456*/, v29 /*v285*/
	v_nop
	s_set_vgpr_msb 0x4162
	v_pk_fma_f32 v[28:29] /*v[284:285]*/, v[164:165] /*v[676:677]*/, s[28:29], v[6:7] /*v[518:519]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6260
	v_exp_f32_e32 v216 /*v472*/, v251
	v_pk_fma_f32 v[226:227] /*v[482:483]*/, v[254:255], s[28:29], v[6:7] /*v[518:519]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6088
	v_mul_f32_e32 v76 /*v588*/, 0xbfb8aa3b, v205 /*v717*/
	s_set_vgpr_msb 0x8862
	v_pk_fma_f32 v[228:229] /*v[484:485]*/, v[178:179] /*v[690:691]*/, s[28:29], v[6:7] /*v[518:519]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[230:231] /*v[486:487]*/, v[180:181] /*v[692:693]*/, s[28:29], v[6:7] /*v[518:519]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6251
	v_wmma_f32_16x16x32_bf16 v[8:15] /*v[264:271]*/, v[176:183] /*v[432:439]*/, v[72:79], v[8:15] /*v[264:271]*/
	s_set_vgpr_msb 0x5162
	v_pk_fma_f32 v[232:233] /*v[488:489]*/, v[184:185] /*v[696:697]*/, s[28:29], v[6:7] /*v[518:519]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6261
	v_pk_fma_f32 v[2:3] /*v[258:259]*/, v[2:3] /*v[258:259]*/, s[28:29], v[76:77] /*v[588:589]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[0:1] /*v[256:257]*/, v[0:1] /*v[256:257]*/, s[28:29], v[76:77] /*v[588:589]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[4:5] /*v[260:261]*/, v[4:5] /*v[260:261]*/, s[28:29], v[76:77] /*v[588:589]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[120:121] /*v[376:377]*/, v[120:121] /*v[376:377]*/, s[28:29], v[6:7] /*v[518:519]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v234 /*v490*/, v231 /*v487*/
	v_exp_f32_e32 v189 /*v445*/, v2 /*v258*/
	s_set_vgpr_msb 0x6101
	v_wmma_f32_16x16x32_bf16 v[240:247], v[176:183] /*v[432:439]*/, v[168:175], v[240:247]
	s_set_vgpr_msb 0x161
	v_exp_f32_e32 v191 /*v447*/, v3 /*v259*/
	v_nop
	v_pk_fma_f32 v[2:3] /*v[258:259]*/, v[16:17] /*v[272:273]*/, s[28:29], v[76:77] /*v[588:589]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[16:17] /*v[272:273]*/, v[18:19] /*v[274:275]*/, s[28:29], v[76:77] /*v[588:589]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[18:19] /*v[274:275]*/, v[20:21] /*v[276:277]*/, s[28:29], v[76:77] /*v[588:589]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v176 /*v432*/, v24 /*v280*/
	v_exp_f32_e32 v180 /*v436*/, v25 /*v281*/
	v_nop
	v_pk_fma_f32 v[24:25] /*v[280:281]*/, v[158:159] /*v[414:415]*/, s[28:29], v[6:7] /*v[518:519]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6151
	v_wmma_f32_16x16x32_bf16 v[8:15] /*v[264:271]*/, v[160:167] /*v[416:423]*/, v[104:111], v[8:15] /*v[264:271]*/
	v_exp_f32_e32 v158 /*v414*/, v26 /*v282*/
	v_exp_f32_e32 v178 /*v434*/, v27 /*v283*/
	v_nop
	s_set_vgpr_msb 0x5162
	v_pk_fma_f32 v[26:27] /*v[282:283]*/, v[168:169] /*v[680:681]*/, s[28:29], v[6:7] /*v[518:519]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6241
	v_exp_f32_e32 v202 /*v458*/, v24 /*v280*/
	v_exp_f32_e32 v204 /*v460*/, v25 /*v281*/
	v_nop
	s_set_vgpr_msb 0x4162
	v_pk_fma_f32 v[24:25] /*v[280:281]*/, v[166:167] /*v[678:679]*/, s[28:29], v[6:7] /*v[518:519]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6241
	v_exp_f32_e32 v177 /*v433*/, v16 /*v272*/
	s_set_vgpr_msb 0x4101
	v_wmma_f32_16x16x32_bf16 v[240:247], v[160:167] /*v[416:423]*/, v[176:183], v[240:247]
	s_set_vgpr_msb 0x161
	v_exp_f32_e32 v181 /*v437*/, v17 /*v273*/
	v_nop
	v_pk_fma_f32 v[16:17] /*v[272:273]*/, v[64:65] /*v[320:321]*/, s[28:29], v[76:77] /*v[588:589]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v182 /*v438*/, v28 /*v284*/
	v_exp_f32_e32 v206 /*v462*/, v27 /*v283*/
	v_exp_f32_e32 v162 /*v418*/, v24 /*v280*/
	v_exp_f32_e32 v164 /*v420*/, v25 /*v281*/
	v_nop
	s_set_vgpr_msb 0x6162
	v_pk_fma_f32 v[24:25] /*v[280:281]*/, v[170:171] /*v[682:683]*/, s[28:29], v[6:7] /*v[518:519]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6241
	v_exp_f32_e32 v160 /*v416*/, v29 /*v285*/
	v_exp_f32_e32 v166 /*v422*/, v26 /*v282*/
	s_set_vgpr_msb 0x4162
	v_pk_fma_f32 v[28:29] /*v[284:285]*/, v[172:173] /*v[684:685]*/, s[28:29], v[6:7] /*v[518:519]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[26:27] /*v[282:283]*/, v[176:177] /*v[688:689]*/, s[28:29], v[6:7] /*v[518:519]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6241
	v_exp_f32_e32 v208 /*v464*/, v24 /*v280*/
	v_exp_f32_e32 v210 /*v466*/, v25 /*v281*/
	v_nop
	s_set_vgpr_msb 0x4162
	v_pk_fma_f32 v[24:25] /*v[280:281]*/, v[174:175] /*v[686:687]*/, s[28:29], v[6:7] /*v[518:519]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6261
	v_exp_f32_e32 v159 /*v415*/, v16 /*v272*/
	v_exp_f32_e32 v179 /*v435*/, v17 /*v273*/
	v_nop
	v_pk_fma_f32 v[16:17] /*v[272:273]*/, v[68:69] /*v[324:325]*/, s[28:29], v[76:77] /*v[588:589]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v199 /*v455*/, v18 /*v274*/
	v_exp_f32_e32 v201 /*v457*/, v19 /*v275*/
	v_nop
	v_pk_fma_f32 v[18:19] /*v[274:275]*/, v[66:67] /*v[322:323]*/, s[28:29], v[76:77] /*v[588:589]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6151
	v_wmma_f32_16x16x32_bf16 v[8:15] /*v[264:271]*/, v[136:143] /*v[392:399]*/, v[128:135], v[8:15] /*v[264:271]*/
	v_exp_f32_e32 v212 /*v468*/, v28 /*v284*/
	v_exp_f32_e32 v214 /*v470*/, v29 /*v285*/
	v_exp_f32_e32 v218 /*v474*/, v25 /*v281*/
	v_exp_f32_e32 v220 /*v476*/, v26 /*v282*/
	v_exp_f32_e32 v222 /*v478*/, v27 /*v283*/
	v_exp_f32_e32 v163 /*v419*/, v16 /*v272*/
	v_exp_f32_e32 v165 /*v421*/, v17 /*v273*/
	s_set_vgpr_msb 0x5101
	v_wmma_f32_16x16x32_bf16 v[240:247], v[136:143] /*v[392:399]*/, v[184:191], v[240:247]
	s_set_vgpr_msb 0x161
	v_pk_fma_f32 v[16:17] /*v[272:273]*/, v[82:83] /*v[338:339]*/, s[28:29], v[76:77] /*v[588:589]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[20:21] /*v[276:277]*/, v[22:23] /*v[278:279]*/, s[28:29], v[76:77] /*v[588:589]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v183 /*v439*/, v18 /*v274*/
	v_exp_f32_e32 v161 /*v417*/, v19 /*v275*/
	v_exp_f32_e32 v142 /*v398*/, v24 /*v280*/
	s_set_vgpr_msb 0x6140
	v_exp_f32_e32 v136 /*v392*/, v248
	v_exp_f32_e32 v138 /*v394*/, v249
	s_set_vgpr_msb 0x4041
	v_wmma_f32_16x16x32_bf16 v[24:31] /*v[280:287]*/, v[112:119] /*v[368:375]*/, v[64:71], 0
	s_set_vgpr_msb 0x4140
	v_exp_f32_e32 v140 /*v396*/, v250
	s_set_vgpr_msb 0x4061
	v_pk_fma_f32 v[18:19] /*v[274:275]*/, v[70:71] /*v[326:327]*/, s[28:29], v[76:77] /*v[588:589]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v213 /*v469*/, v16 /*v272*/
	v_exp_f32_e32 v215 /*v471*/, v17 /*v273*/
	v_nop
	v_pk_fma_f32 v[16:17] /*v[272:273]*/, v[86:87] /*v[342:343]*/, s[28:29], v[76:77] /*v[588:589]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v203 /*v459*/, v20 /*v276*/
	v_exp_f32_e32 v205 /*v461*/, v21 /*v277*/
	s_set_vgpr_msb 0x6101
	v_wmma_f32_16x16x32_bf16 v[248:255], v[112:119] /*v[368:375]*/, v[160:167], 0
	s_set_vgpr_msb 0x161
	v_pk_fma_f32 v[20:21] /*v[276:277]*/, v[80:81] /*v[336:337]*/, s[28:29], v[76:77] /*v[588:589]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v167 /*v423*/, v18 /*v274*/
	v_exp_f32_e32 v207 /*v463*/, v19 /*v275*/
	v_nop
	v_pk_fma_f32 v[18:19] /*v[274:275]*/, v[84:85] /*v[340:341]*/, s[28:29], v[76:77] /*v[588:589]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v221 /*v477*/, v16 /*v272*/
	v_exp_f32_e32 v223 /*v479*/, v17 /*v273*/
	v_nop
	v_pk_fma_f32 v[16:17] /*v[272:273]*/, v[106:107] /*v[362:363]*/, s[28:29], v[76:77] /*v[588:589]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6151
	v_wmma_f32_16x16x32_bf16 v[24:31] /*v[280:287]*/, v[96:103] /*v[352:359]*/, v[72:79], v[24:31] /*v[280:287]*/
	v_exp_f32_e32 v209 /*v465*/, v20 /*v276*/
	v_exp_f32_e32 v211 /*v467*/, v21 /*v277*/
	v_exp_f32_e32 v143 /*v399*/, v18 /*v274*/
	s_set_vgpr_msb 0x5161
	v_pk_fma_f32 v[20:21] /*v[276:277]*/, v[104:105] /*v[360:361]*/, s[28:29], v[76:77] /*v[588:589]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v219 /*v475*/, v19 /*v275*/
	v_nop
	v_pk_fma_f32 v[18:19] /*v[274:275]*/, v[108:109] /*v[364:365]*/, s[28:29], v[76:77] /*v[588:589]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v141 /*v397*/, v16 /*v272*/
	s_set_vgpr_msb 0x6101
	v_wmma_f32_16x16x32_bf16 v[248:255], v[96:103] /*v[352:359]*/, v[168:175], v[248:255]
	s_set_vgpr_msb 0x161
	v_exp_f32_e32 v217 /*v473*/, v17 /*v273*/
	v_nop
	v_pk_fma_f32 v[16:17] /*v[272:273]*/, v[128:129] /*v[384:385]*/, s[28:29], v[76:77] /*v[588:589]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v185 /*v441*/, v0 /*v256*/
	v_exp_f32_e32 v187 /*v443*/, v1 /*v257*/
	v_pk_fma_f32 v[96:97] /*v[352:353]*/, v[122:123] /*v[378:379]*/, s[28:29], v[6:7] /*v[518:519]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[98:99] /*v[354:355]*/, v[124:125] /*v[380:381]*/, s[28:29], v[6:7] /*v[518:519]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[100:101] /*v[356:357]*/, v[126:127] /*v[382:383]*/, s[28:29], v[6:7] /*v[518:519]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6151
	v_wmma_f32_16x16x32_bf16 v[24:31] /*v[280:287]*/, v[88:95] /*v[344:351]*/, v[104:111], v[24:31] /*v[280:287]*/
	s_set_vgpr_msb 0x5161
	v_pk_fma_f32 v[0:1] /*v[256:257]*/, v[6:7] /*v[262:263]*/, s[28:29], v[76:77] /*v[588:589]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v236 /*v492*/, v96 /*v352*/
	v_exp_f32_e32 v244 /*v500*/, v97 /*v353*/
	v_nop
	v_pk_fma_f32 v[96:97] /*v[352:353]*/, v[254:255] /*v[510:511]*/, s[28:29], v[6:7] /*v[518:519]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v246 /*v502*/, v98 /*v354*/
	v_exp_f32_e32 v248 /*v504*/, v99 /*v355*/
	v_exp_f32_e32 v252 /*v508*/, v100 /*v356*/
	v_exp_f32_e32 v254 /*v510*/, v101 /*v357*/
	s_set_vgpr_msb 0x6162
	v_pk_fma_f32 v[98:99] /*v[354:355]*/, v[0:1] /*v[512:513]*/, s[28:29], v[6:7] /*v[518:519]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6201
	v_wmma_f32_16x16x32_bf16 v[248:255], v[88:95] /*v[344:351]*/, v[176:183], v[248:255]
	s_set_vgpr_msb 0x162
	v_pk_fma_f32 v[100:101] /*v[356:357]*/, v[2:3] /*v[514:515]*/, s[28:29], v[6:7] /*v[518:519]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6241
	v_exp_f32_e32 v112 /*v368*/, v225 /*v481*/
	v_exp_f32_e32 v137 /*v393*/, v20 /*v276*/
	v_exp_f32_e32 v139 /*v395*/, v21 /*v277*/
	v_exp_f32_e32 v88 /*v344*/, v96 /*v352*/
	v_exp_f32_e32 v90 /*v346*/, v97 /*v353*/
	v_nop
	s_set_vgpr_msb 0x4162
	v_pk_fma_f32 v[96:97] /*v[352:353]*/, v[4:5] /*v[516:517]*/, s[28:29], v[6:7] /*v[518:519]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6261
	v_pk_fma_f32 v[20:21] /*v[276:277]*/, v[110:111] /*v[366:367]*/, s[28:29], v[76:77] /*v[588:589]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v225 /*v481*/, v18 /*v274*/
	v_exp_f32_e32 v113 /*v369*/, v19 /*v275*/
	v_nop
	v_pk_fma_f32 v[18:19] /*v[274:275]*/, v[130:131] /*v[386:387]*/, s[28:29], v[76:77] /*v[588:589]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v116 /*v372*/, v227 /*v483*/
	v_exp_f32_e32 v119 /*v375*/, v16 /*v272*/
	v_exp_f32_e32 v227 /*v483*/, v17 /*v273*/
	v_nop
	v_pk_fma_f32 v[16:17] /*v[272:273]*/, v[132:133] /*v[388:389]*/, s[28:29], v[76:77] /*v[588:589]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v114 /*v370*/, v226 /*v482*/
	v_exp_f32_e32 v118 /*v374*/, v228 /*v484*/
	v_exp_f32_e32 v226 /*v482*/, v229 /*v485*/
	v_nop
	s_set_vgpr_msb 0x6162
	v_pk_fma_f32 v[228:229] /*v[484:485]*/, v[182:183] /*v[694:695]*/, s[28:29], v[6:7] /*v[518:519]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6241
	v_exp_f32_e32 v92 /*v348*/, v98 /*v354*/
	v_exp_f32_e32 v94 /*v350*/, v99 /*v355*/
	s_set_vgpr_msb 0x4181
	v_exp_f32_e32 v0 /*v512*/, v100 /*v356*/
	v_exp_f32_e32 v2 /*v514*/, v101 /*v357*/
	v_exp_f32_e32 v4 /*v516*/, v96 /*v352*/
	v_exp_f32_e32 v6 /*v518*/, v97 /*v353*/
	s_set_vgpr_msb 0x8161
	v_exp_f32_e32 v193 /*v449*/, v4 /*v260*/
	v_exp_f32_e32 v153 /*v409*/, v5 /*v261*/
	v_wmma_f32_16x16x32_bf16 v[96:103] /*v[352:359]*/, v[56:63] /*v[312:319]*/, v[64:71], 0
	v_exp_f32_e32 v195 /*v451*/, v0 /*v256*/
	v_exp_f32_e32 v197 /*v453*/, v1 /*v257*/
	v_exp_f32_e32 v155 /*v411*/, v2 /*v258*/
	v_exp_f32_e32 v157 /*v413*/, v3 /*v259*/
	v_exp_f32_e32 v115 /*v371*/, v20 /*v276*/
	v_exp_f32_e32 v117 /*v373*/, v21 /*v277*/
	v_exp_f32_e32 v231 /*v487*/, v18 /*v274*/
	v_wmma_f32_16x16x32_bf16 v[0:7] /*v[256:263]*/, v[56:63] /*v[312:319]*/, v[160:167], 0
	v_pk_fma_f32 v[20:21] /*v[276:277]*/, v[134:135] /*v[390:391]*/, s[28:29], v[76:77] /*v[588:589]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v235 /*v491*/, v19 /*v275*/
	v_nop
	v_pk_fma_f32 v[18:19] /*v[274:275]*/, v[146:147] /*v[402:403]*/, s[28:29], v[76:77] /*v[588:589]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v239 /*v495*/, v16 /*v272*/
	v_exp_f32_e32 v241 /*v497*/, v17 /*v273*/
	v_nop
	v_pk_fma_f32 v[16:17] /*v[272:273]*/, v[144:145] /*v[400:401]*/, s[28:29], v[76:77] /*v[588:589]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v243 /*v499*/, v20 /*v276*/
	v_exp_f32_e32 v251 /*v507*/, v21 /*v277*/
	v_nop
	v_pk_fma_f32 v[20:21] /*v[276:277]*/, v[148:149] /*v[404:405]*/, s[28:29], v[76:77] /*v[588:589]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v237 /*v493*/, v18 /*v274*/
	v_exp_f32_e32 v245 /*v501*/, v19 /*v275*/
	v_nop
	v_pk_fma_f32 v[18:19] /*v[274:275]*/, v[168:169] /*v[424:425]*/, s[28:29], v[76:77] /*v[588:589]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v224 /*v480*/, v224 /*v480*/
	v_exp_f32_e32 v240 /*v496*/, v229 /*v485*/
	v_exp_f32_e32 v250 /*v506*/, v233 /*v489*/
	s_set_vgpr_msb 0x6151
	v_wmma_f32_16x16x32_bf16 v[96:103] /*v[352:359]*/, v[48:55] /*v[304:311]*/, v[72:79], v[96:103] /*v[352:359]*/
	v_exp_f32_e32 v229 /*v485*/, v16 /*v272*/
	v_exp_f32_e32 v233 /*v489*/, v17 /*v273*/
	v_nop
	s_set_vgpr_msb 0x5161
	v_pk_fma_f32 v[16:17] /*v[272:273]*/, v[150:151] /*v[406:407]*/, s[28:29], v[76:77] /*v[588:589]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v247 /*v503*/, v20 /*v276*/
	v_exp_f32_e32 v249 /*v505*/, v21 /*v277*/
	v_exp_f32_e32 v89 /*v345*/, v18 /*v274*/
	v_pk_fma_f32 v[20:21] /*v[276:277]*/, v[172:173] /*v[428:429]*/, s[28:29], v[76:77] /*v[588:589]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6151
	v_wmma_f32_16x16x32_bf16 v[0:7] /*v[256:263]*/, v[48:55] /*v[304:311]*/, v[168:175], v[0:7] /*v[256:263]*/
	v_exp_f32_e32 v91 /*v347*/, v19 /*v275*/
	v_nop
	s_set_vgpr_msb 0x5165
	v_pk_add_f32 v[18:19] /*v[274:275]*/, v[184:185] /*v[440:441]*/, v[186:187] /*v[442:443]*/
	v_pk_add_f32 v[22:23] /*v[278:279]*/, v[190:191] /*v[446:447]*/, v[192:193] /*v[448:449]*/
	v_exp_f32_e32 v230 /*v486*/, v230 /*v486*/
	v_exp_f32_e32 v238 /*v494*/, v228 /*v484*/
	v_exp_f32_e32 v242 /*v498*/, v232 /*v488*/
	v_exp_f32_e32 v228 /*v484*/, v120 /*v376*/
	v_exp_f32_e32 v232 /*v488*/, v121 /*v377*/
	v_exp_f32_e32 v253 /*v509*/, v16 /*v272*/
	v_exp_f32_e32 v255 /*v511*/, v17 /*v273*/
	v_nop
	v_pk_fma_f32 v[16:17] /*v[272:273]*/, v[170:171] /*v[426:427]*/, s[28:29], v[76:77] /*v[588:589]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6551
	v_wmma_f32_16x16x32_bf16 v[96:103] /*v[352:359]*/, v[40:47] /*v[296:303]*/, v[104:111], v[96:103] /*v[352:359]*/
	s_set_vgpr_msb 0x5181
	v_exp_f32_e32 v1 /*v513*/, v20 /*v276*/
	v_exp_f32_e32 v3 /*v515*/, v21 /*v277*/
	s_set_vgpr_msb 0x8145
	v_pk_add_f32 v[18:19] /*v[274:275]*/, v[188:189] /*v[444:445]*/, v[18:19] /*v[274:275]*/
	v_pk_add_f32 v[20:21] /*v[276:277]*/, v[194:195] /*v[450:451]*/, v[196:197] /*v[452:453]*/
	v_pk_add_f32 v[22:23] /*v[278:279]*/, v[152:153] /*v[408:409]*/, v[22:23] /*v[278:279]*/
	v_pk_add_f32 v[48:49] /*v[304:305]*/, v[164:165] /*v[420:421]*/, v[166:167] /*v[422:423]*/
	v_pk_add_f32 v[52:53] /*v[308:309]*/, v[214:215] /*v[470:471]*/, v[142:143] /*v[398:399]*/
	s_set_vgpr_msb 0x4551
	v_wmma_f32_16x16x32_bf16 v[0:7] /*v[256:263]*/, v[40:47] /*v[296:303]*/, v[176:183], v[0:7] /*v[256:263]*/
	s_set_vgpr_msb 0x5165
	v_pk_add_f32 v[54:55] /*v[310:311]*/, v[220:221] /*v[476:477]*/, v[222:223] /*v[478:479]*/
	v_pk_add_f32 v[58:59] /*v[314:315]*/, v[224:225] /*v[480:481]*/, v[112:113] /*v[368:369]*/
	v_pk_add_f32 v[60:61] /*v[316:317]*/, v[116:117] /*v[372:373]*/, v[118:119] /*v[374:375]*/
	v_exp_f32_e32 v93 /*v349*/, v16 /*v272*/
	v_pk_add_f32 v[40:41] /*v[296:297]*/, v[156:157] /*v[412:413]*/, v[176:177] /*v[432:433]*/
	v_pk_add_f32 v[42:43] /*v[298:299]*/, v[198:199] /*v[454:455]*/, v[200:201] /*v[456:457]*/
	v_pk_add_f32 v[46:47] /*v[302:303]*/, v[182:183] /*v[438:439]*/, v[160:161] /*v[416:417]*/
	v_pk_add_f32 v[20:21] /*v[276:277]*/, v[154:155] /*v[410:411]*/, v[20:21] /*v[276:277]*/
	v_pk_add_f32 v[44:45] /*v[300:301]*/, v[204:205] /*v[460:461]*/, v[158:159] /*v[414:415]*/
	v_pk_add_f32 v[40:41] /*v[296:297]*/, v[180:181] /*v[436:437]*/, v[40:41] /*v[296:297]*/
	v_pk_add_f32 v[42:43] /*v[298:299]*/, v[202:203] /*v[458:459]*/, v[42:43] /*v[298:299]*/
	v_pk_add_f32 v[46:47] /*v[302:303]*/, v[162:163] /*v[418:419]*/, v[46:47] /*v[302:303]*/
	v_pk_add_f32 v[48:49] /*v[304:305]*/, v[206:207] /*v[462:463]*/, v[48:49] /*v[304:305]*/
	v_pk_add_f32 v[56:57] /*v[312:313]*/, v[138:139] /*v[394:395]*/, v[140:141] /*v[396:397]*/
	v_pk_add_f32 v[52:53] /*v[308:309]*/, v[218:219] /*v[474:475]*/, v[52:53] /*v[308:309]*/
	v_pk_add_f32 v[54:55] /*v[310:311]*/, v[136:137] /*v[392:393]*/, v[54:55] /*v[310:311]*/
	v_pk_add_f32 v[62:63] /*v[318:319]*/, v[230:231] /*v[486:487]*/, v[234:235] /*v[490:491]*/
	v_pk_add_f32 v[64:65] /*v[320:321]*/, v[240:241] /*v[496:497]*/, v[242:243] /*v[498:499]*/
	v_pk_add_f32 v[58:59] /*v[314:315]*/, v[114:115] /*v[370:371]*/, v[58:59] /*v[314:315]*/
	v_pk_add_f32 v[66:67] /*v[322:323]*/, v[228:229] /*v[484:485]*/, v[232:233] /*v[488:489]*/
	v_pk_add_f32 v[60:61] /*v[316:317]*/, v[226:227] /*v[482:483]*/, v[60:61] /*v[316:317]*/
	v_pk_add_f32 v[18:19] /*v[274:275]*/, v[18:19] /*v[274:275]*/, v[22:23] /*v[278:279]*/
	v_exp_f32_e32 v95 /*v351*/, v17 /*v273*/
	v_nop
	v_pk_fma_f32 v[16:17] /*v[272:273]*/, v[174:175] /*v[430:431]*/, s[28:29], v[76:77] /*v[588:589]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x6551
	v_wmma_f32_16x16x32_bf16 v[24:31] /*v[280:287]*/, v[72:79] /*v[328:335]*/, v[128:135], v[24:31] /*v[280:287]*/
	s_set_vgpr_msb 0x5145
	v_pk_add_f32 v[44:45] /*v[300:301]*/, v[178:179] /*v[434:435]*/, v[44:45] /*v[300:301]*/
	v_pk_add_f32 v[50:51] /*v[306:307]*/, v[208:209] /*v[464:465]*/, v[210:211] /*v[466:467]*/
	v_pk_add_f32 v[56:57] /*v[312:313]*/, v[216:217] /*v[472:473]*/, v[56:57] /*v[312:313]*/
	v_pk_add_f32 v[62:63] /*v[318:319]*/, v[238:239] /*v[494:495]*/, v[62:63] /*v[318:319]*/
	v_pk_add_f32 v[64:65] /*v[320:321]*/, v[250:251] /*v[506:507]*/, v[64:65] /*v[320:321]*/
	v_pk_add_f32 v[68:69] /*v[324:325]*/, v[244:245] /*v[500:501]*/, v[246:247] /*v[502:503]*/
	v_pk_add_f32 v[66:67] /*v[322:323]*/, v[236:237] /*v[492:493]*/, v[66:67] /*v[322:323]*/
	s_set_vgpr_msb 0x4501
	v_wmma_f32_16x16x32_bf16 v[248:255], v[72:79] /*v[328:335]*/, v[184:191], v[248:255]
	s_set_vgpr_msb 0x145
	v_pk_add_f32 v[70:71] /*v[326:327]*/, v[252:253] /*v[508:509]*/, v[254:255] /*v[510:511]*/
	v_pk_add_f32 v[18:19] /*v[274:275]*/, v[20:21] /*v[276:277]*/, v[18:19] /*v[274:275]*/
	v_pk_add_f32 v[20:21] /*v[276:277]*/, v[40:41] /*v[296:297]*/, v[42:43] /*v[298:299]*/
	v_pk_add_f32 v[40:41] /*v[296:297]*/, v[46:47] /*v[302:303]*/, v[48:49] /*v[304:305]*/
	v_pk_add_f32 v[72:73] /*v[328:329]*/, v[90:91] /*v[346:347]*/, v[92:93] /*v[348:349]*/
	v_pk_add_f32 v[42:43] /*v[298:299]*/, v[52:53] /*v[308:309]*/, v[54:55] /*v[310:311]*/
	v_pk_add_f32 v[46:47] /*v[302:303]*/, v[58:59] /*v[314:315]*/, v[60:61] /*v[316:317]*/
	s_set_vgpr_msb 0x4551
	v_wmma_f32_16x16x32_bf16 v[96:103] /*v[352:359]*/, v[32:39] /*v[288:295]*/, v[128:135], v[96:103] /*v[352:359]*/
	s_set_vgpr_msb 0x5181
	v_exp_f32_e32 v5 /*v517*/, v16 /*v272*/
	s_set_vgpr_msb 0x8145
	v_pk_add_f32 v[50:51] /*v[306:307]*/, v[212:213] /*v[468:469]*/, v[50:51] /*v[306:307]*/
	v_pk_add_f32 v[68:69] /*v[324:325]*/, v[248:249] /*v[504:505]*/, v[68:69] /*v[324:325]*/
	s_set_vgpr_msb 0x454a
	v_pk_add_f32 v[74:75] /*v[330:331]*/, v[0:1] /*v[512:513]*/, v[2:3] /*v[514:515]*/
	s_set_vgpr_msb 0x4a45
	v_pk_add_f32 v[22:23] /*v[278:279]*/, v[88:89] /*v[344:345]*/, v[70:71] /*v[326:327]*/
	v_pk_add_f32 v[70:71] /*v[326:327]*/, v[94:95] /*v[350:351]*/, v[72:73] /*v[328:329]*/
	v_pk_add_f32 v[48:49] /*v[304:305]*/, v[64:65] /*v[320:321]*/, v[66:67] /*v[322:323]*/
	s_set_vgpr_msb 0x4551
	v_wmma_f32_16x16x32_bf16 v[0:7] /*v[256:263]*/, v[32:39] /*v[288:295]*/, v[184:191], v[0:7] /*v[256:263]*/
	s_set_vgpr_msb 0x5145
	v_pk_add_f32 v[20:21] /*v[276:277]*/, v[44:45] /*v[300:301]*/, v[20:21] /*v[276:277]*/
	v_pk_add_f32 v[42:43] /*v[298:299]*/, v[56:57] /*v[312:313]*/, v[42:43] /*v[298:299]*/
	v_pk_add_f32 v[44:45] /*v[300:301]*/, v[62:63] /*v[318:319]*/, v[46:47] /*v[302:303]*/
	s_set_vgpr_msb 0x4546
	v_pk_add_f32 v[72:73] /*v[328:329]*/, v[4:5] /*v[516:517]*/, v[74:75] /*v[330:331]*/
	s_set_vgpr_msb 0x4645
	v_pk_add_f32 v[40:41] /*v[296:297]*/, v[50:51] /*v[306:307]*/, v[40:41] /*v[296:297]*/
	v_pk_add_f32 v[46:47] /*v[302:303]*/, v[68:69] /*v[324:325]*/, v[48:49] /*v[304:305]*/
	v_pk_add_f32 v[22:23] /*v[278:279]*/, v[22:23] /*v[278:279]*/, v[70:71] /*v[326:327]*/
	s_set_vgpr_msb 0x4542
	s_wait_dscnt 0xe
	v_wmma_f32_16x16x32_bf16 v[120:127] /*v[376:383]*/, v[12:19] /*v[524:531]*/, v[64:71], 0
	s_set_vgpr_msb 0x4245
	v_pk_add_f32 v[18:19] /*v[274:275]*/, v[18:19] /*v[274:275]*/, v[20:21] /*v[276:277]*/
	v_pk_add_f32 v[20:21] /*v[276:277]*/, v[42:43] /*v[298:299]*/, v[44:45] /*v[300:301]*/
	s_set_vgpr_msb 0x4581
	v_exp_f32_e32 v7 /*v519*/, v17 /*v273*/
	v_nop
	s_set_vgpr_msb 0x8145
	v_pk_add_f32 v[16:17] /*v[272:273]*/, v[72:73] /*v[328:329]*/, v[22:23] /*v[278:279]*/
	v_pk_add_f32 v[18:19] /*v[274:275]*/, v[40:41] /*v[296:297]*/, v[18:19] /*v[274:275]*/
	v_pk_add_f32 v[20:21] /*v[276:277]*/, v[46:47] /*v[302:303]*/, v[20:21] /*v[276:277]*/
	s_set_vgpr_msb 0x4542
	v_wmma_f32_16x16x32_bf16 v[32:39] /*v[288:295]*/, v[12:19] /*v[524:531]*/, v[160:167], 0
	s_set_vgpr_msb 0x4246
	v_pk_add_f32 v[16:17] /*v[272:273]*/, v[6:7] /*v[518:519]*/, v[16:17] /*v[272:273]*/
	s_set_vgpr_msb 0x4645
	v_pk_add_f32 v[18:19] /*v[274:275]*/, v[18:19] /*v[274:275]*/, v[20:21] /*v[276:277]*/
	s_set_vgpr_msb 0x454a
	v_sub_f32_e32 v20 /*v276*/, v198 /*v710*/, v206 /*v718*/
	s_set_vgpr_msb 0x4a02
	v_wmma_f32_16x16x32_bf16 v[224:231], v[84:91] /*v[596:603]*/, v[64:71], 0
	v_wmma_f32_16x16x32_bf16 v[200:207], v[84:91] /*v[596:603]*/, v[160:167], 0
	v_wmma_f32_16x16x32_bf16 v[232:239], v[122:129] /*v[634:641]*/, v[64:71], 0
	v_wmma_f32_16x16x32_bf16 v[208:215], v[122:129] /*v[634:641]*/, v[160:167], 0
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x285
	v_pk_add_f32 v[122:123] /*v[634:635]*/, v[16:17] /*v[272:273]*/, v[18:19] /*v[274:275]*/
	s_set_vgpr_msb 0x8544
	v_mul_f32_e32 v16 /*v272*/, 0x3fb8aa3b, v20 /*v276*/
	s_set_vgpr_msb 0x4482
	v_mov_b32_e32 v204 /*v716*/, v195 /*v707*/
	s_set_vgpr_msb 0x8252
	s_wait_dscnt 0xc
	v_wmma_f32_16x16x32_bf16 v[120:127] /*v[376:383]*/, v[20:27] /*v[532:539]*/, v[72:79], v[120:127] /*v[376:383]*/
	s_set_vgpr_msb 0x5282
	v_dual_mov_b32 v195 /*v707*/, v8 /*v520*/ :: v_dual_mov_b32 v124 /*v636*/, v122 /*v634*/
	v_mov_b32_e32 v125 /*v637*/, v123 /*v635*/
	s_set_vgpr_msb 0x8281
	v_exp_f32_e32 v126 /*v638*/, v16 /*v272*/
	s_set_vgpr_msb 0x8182
	v_permlanex16_b32 v124 /*v636*/, v124 /*v636*/, s66, 0xfedcba98
	s_set_vgpr_msb 0x8252
	v_wmma_f32_16x16x32_bf16 v[32:39] /*v[288:295]*/, v[20:27] /*v[532:539]*/, v[168:175], v[32:39] /*v[288:295]*/
	s_set_vgpr_msb 0x5282
	v_permlanex16_b32 v125 /*v637*/, v125 /*v637*/, s66, 0xfedcba98
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x32_bf16 v[16:23] /*v[528:535]*/, v[44:51] /*v[556:563]*/, v[64:71], 0
	s_set_vgpr_msb 0x8242
	v_wmma_f32_16x16x32_bf16 v[128:135] /*v[384:391]*/, v[44:51] /*v[556:563]*/, v[160:167], 0
	s_set_vgpr_msb 0x4202
	v_wmma_f32_16x16x32_bf16 v[224:231], v[92:99] /*v[604:611]*/, v[72:79], v[224:231]
	v_wmma_f32_16x16x32_bf16 v[200:207], v[92:99] /*v[604:611]*/, v[168:175], v[200:207]
	v_wmma_f32_16x16x32_bf16 v[232:239], v[130:137] /*v[642:649]*/, v[72:79], v[232:239]
	v_wmma_f32_16x16x32_bf16 v[208:215], v[130:137] /*v[642:649]*/, v[168:175], v[208:215]
	s_set_vgpr_msb 0x2a2
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x32_bf16 v[16:23] /*v[528:535]*/, v[52:59] /*v[564:571]*/, v[72:79], v[16:23] /*v[528:535]*/
	s_set_vgpr_msb 0xa252
	v_wmma_f32_16x16x32_bf16 v[128:135] /*v[384:391]*/, v[52:59] /*v[564:571]*/, v[168:175], v[128:135] /*v[384:391]*/
	s_set_vgpr_msb 0x5202
	v_wmma_f32_16x16x32_bf16 v[224:231], v[100:107] /*v[612:619]*/, v[104:111], v[224:231]
	v_wmma_f32_16x16x32_bf16 v[200:207], v[100:107] /*v[612:619]*/, v[176:183], v[200:207]
	v_wmma_f32_16x16x32_bf16 v[232:239], v[138:145] /*v[650:657]*/, v[104:111], v[232:239]
	v_wmma_f32_16x16x32_bf16 v[208:215], v[138:145] /*v[650:657]*/, v[176:183], v[208:215]
	s_set_vgpr_msb 0x252
	v_wmma_f32_16x16x32_bf16 v[120:127] /*v[376:383]*/, v[28:35] /*v[540:547]*/, v[104:111], v[120:127] /*v[376:383]*/
	v_wmma_f32_16x16x32_bf16 v[32:39] /*v[288:295]*/, v[28:35] /*v[540:547]*/, v[176:183], v[32:39] /*v[288:295]*/
	s_set_vgpr_msb 0x52a2
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[16:23] /*v[528:535]*/, v[60:67] /*v[572:579]*/, v[104:111], v[16:23] /*v[528:535]*/
	s_set_vgpr_msb 0xa252
	v_wmma_f32_16x16x32_bf16 v[128:135] /*v[384:391]*/, v[60:67] /*v[572:579]*/, v[176:183], v[128:135] /*v[384:391]*/
	s_set_vgpr_msb 0x5202
	v_wmma_f32_16x16x32_bf16 v[224:231], v[108:115] /*v[620:627]*/, v[128:135], v[224:231]
	v_wmma_f32_16x16x32_bf16 v[200:207], v[108:115] /*v[620:627]*/, v[184:191], v[200:207]
	v_wmma_f32_16x16x32_bf16 v[232:239], v[146:153] /*v[658:665]*/, v[128:135], v[232:239]
	v_wmma_f32_16x16x32_bf16 v[208:215], v[146:153] /*v[658:665]*/, v[184:191], v[208:215]
	s_set_vgpr_msb 0x252
	v_wmma_f32_16x16x32_bf16 v[120:127] /*v[376:383]*/, v[36:43] /*v[548:555]*/, v[128:135], v[120:127] /*v[376:383]*/
	v_wmma_f32_16x16x32_bf16 v[32:39] /*v[288:295]*/, v[36:43] /*v[548:555]*/, v[184:191], v[32:39] /*v[288:295]*/
	s_set_vgpr_msb 0x52a2
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[16:23] /*v[528:535]*/, v[68:75] /*v[580:587]*/, v[128:135], v[16:23] /*v[528:535]*/
	s_set_vgpr_msb 0xa252
	v_wmma_f32_16x16x32_bf16 v[128:135] /*v[384:391]*/, v[68:75] /*v[580:587]*/, v[184:191], v[128:135] /*v[384:391]*/
	s_set_vgpr_msb 0x5200
	s_cbranch_vccz .LBB0_11
	s_set_vgpr_msb 8
	v_pk_mul_f32 v[158:159], v[158:159], v[126:127] /*v[638:639]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[156:157], v[156:157], v[126:127] /*v[638:639]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[154:155], v[154:155], v[126:127] /*v[638:639]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[152:153], v[152:153], v[126:127] /*v[638:639]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[150:151], v[150:151], v[126:127] /*v[638:639]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[148:149], v[148:149], v[126:127] /*v[638:639]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[146:147], v[146:147], v[126:127] /*v[638:639]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[144:145], v[144:145], v[126:127] /*v[638:639]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[142:143], v[142:143], v[126:127] /*v[638:639]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[140:141], v[140:141], v[126:127] /*v[638:639]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[138:139], v[138:139], v[126:127] /*v[638:639]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[136:137], v[136:137], v[126:127] /*v[638:639]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[126:127], v[126:127], v[126:127] /*v[638:639]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[124:125], v[124:125], v[126:127] /*v[638:639]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[122:123], v[122:123], v[126:127] /*v[638:639]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[120:121], v[120:121], v[126:127] /*v[638:639]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[118:119], v[118:119], v[126:127] /*v[638:639]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[116:117], v[116:117], v[126:127] /*v[638:639]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[114:115], v[114:115], v[126:127] /*v[638:639]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[112:113], v[112:113], v[126:127] /*v[638:639]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[102:103], v[102:103], v[126:127] /*v[638:639]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[100:101], v[100:101], v[126:127] /*v[638:639]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[98:99], v[98:99], v[126:127] /*v[638:639]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[96:97], v[96:97], v[126:127] /*v[638:639]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[94:95], v[94:95], v[126:127] /*v[638:639]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[92:93], v[92:93], v[126:127] /*v[638:639]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[90:91], v[90:91], v[126:127] /*v[638:639]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[88:89], v[88:89], v[126:127] /*v[638:639]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[86:87], v[86:87], v[126:127] /*v[638:639]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[84:85], v[84:85], v[126:127] /*v[638:639]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[82:83], v[82:83], v[126:127] /*v[638:639]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[80:81], v[80:81], v[126:127] /*v[638:639]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x800
.LBB0_11:
	s_set_vgpr_msb 0x4a
	v_sub_f32_e32 v16 /*v272*/, v203 /*v715*/, v205 /*v717*/
	s_and_b32 s2, s3, exec_lo
	s_cselect_b32 s2, 1, 0
	s_cmp_lg_u32 s2, 1
	s_set_vgpr_msb 0x4a44
	v_mul_f32_e32 v16 /*v272*/, 0x3fb8aa3b, v16 /*v272*/
	s_set_vgpr_msb 0x4481
	v_exp_f32_e32 v127 /*v639*/, v16 /*v272*/
	s_set_vgpr_msb 0x8100
	s_cbranch_scc1 .LBB0_13
	v_nop
	s_set_vgpr_msb 0x42
	v_mov_b32_e32 v16 /*v272*/, v127 /*v639*/
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
.LBB0_13:
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x42
	ds_load_tr16_b128 v[20:23] /*v[276:279]*/, v201 /*v713*/ offset:4608
	ds_load_tr16_b128 v[44:47] /*v[300:303]*/, v201 /*v713*/ offset:4640
	ds_load_tr16_b128 v[16:19] /*v[272:275]*/, v201 /*v713*/
	ds_load_tr16_b128 v[40:43] /*v[296:299]*/, v201 /*v713*/ offset:32
	s_set_vgpr_msb 0x4285
	v_cvt_pk_bf16_f32 v95 /*v607*/, v202 /*v458*/, v204 /*v460*/
	v_cvt_pk_bf16_f32 v94 /*v606*/, v198 /*v454*/, v200 /*v456*/
	v_cvt_pk_bf16_f32 v93 /*v605*/, v176 /*v432*/, v180 /*v436*/
	v_cvt_pk_bf16_f32 v92 /*v604*/, v154 /*v410*/, v156 /*v412*/
	v_cvt_pk_bf16_f32 v91 /*v603*/, v194 /*v450*/, v196 /*v452*/
	v_cvt_pk_bf16_f32 v90 /*v602*/, v192 /*v448*/, v152 /*v408*/
	v_cvt_pk_bf16_f32 v89 /*v601*/, v188 /*v444*/, v190 /*v446*/
	v_cvt_pk_bf16_f32 v88 /*v600*/, v184 /*v440*/, v186 /*v442*/
	v_cvt_pk_bf16_f32 v119 /*v631*/, v203 /*v459*/, v205 /*v461*/
	v_cvt_pk_bf16_f32 v118 /*v630*/, v199 /*v455*/, v201 /*v457*/
	v_cvt_pk_bf16_f32 v117 /*v629*/, v177 /*v433*/, v181 /*v437*/
	v_cvt_pk_bf16_f32 v116 /*v628*/, v155 /*v411*/, v157 /*v413*/
	v_cvt_pk_bf16_f32 v115 /*v627*/, v195 /*v451*/, v197 /*v453*/
	v_cvt_pk_bf16_f32 v114 /*v626*/, v193 /*v449*/, v153 /*v409*/
	v_cvt_pk_bf16_f32 v113 /*v625*/, v189 /*v445*/, v191 /*v447*/
	v_cvt_pk_bf16_f32 v112 /*v624*/, v185 /*v441*/, v187 /*v443*/
	v_cvt_pk_bf16_f32 v83 /*v595*/, v166 /*v422*/, v206 /*v462*/
	v_cvt_pk_bf16_f32 v82 /*v594*/, v162 /*v418*/, v164 /*v420*/
	v_cvt_pk_bf16_f32 v81 /*v593*/, v182 /*v438*/, v160 /*v416*/
	v_cvt_pk_bf16_f32 v80 /*v592*/, v158 /*v414*/, v178 /*v434*/
	v_cvt_pk_bf16_f32 v107 /*v619*/, v167 /*v423*/, v207 /*v463*/
	v_cvt_pk_bf16_f32 v106 /*v618*/, v163 /*v419*/, v165 /*v421*/
	v_cvt_pk_bf16_f32 v105 /*v617*/, v183 /*v439*/, v161 /*v417*/
	v_cvt_pk_bf16_f32 v104 /*v616*/, v159 /*v415*/, v179 /*v435*/
	s_set_vgpr_msb 0x8509
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x32_bf16 v[152:159], v[16:23] /*v[272:279]*/, v[88:95] /*v[600:607]*/, v[152:159]
	s_set_vgpr_msb 0x985
	v_cvt_pk_bf16_f32 v87 /*v599*/, v220 /*v476*/, v222 /*v478*/
	v_cvt_pk_bf16_f32 v86 /*v598*/, v142 /*v398*/, v218 /*v474*/
	v_cvt_pk_bf16_f32 v85 /*v597*/, v212 /*v468*/, v214 /*v470*/
	v_cvt_pk_bf16_f32 v84 /*v596*/, v208 /*v464*/, v210 /*v466*/
	v_cvt_pk_bf16_f32 v73 /*v585*/, v140 /*v396*/, v216 /*v472*/
	v_cvt_pk_bf16_f32 v111 /*v623*/, v221 /*v477*/, v223 /*v479*/
	v_cvt_pk_bf16_f32 v110 /*v622*/, v143 /*v399*/, v219 /*v475*/
	s_set_vgpr_msb 0x8509
	v_wmma_f32_16x16x32_bf16 v[56:63], v[16:23] /*v[272:279]*/, v[112:119] /*v[624:631]*/, v[56:63]
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x942
	ds_load_tr16_b128 v[16:19] /*v[272:275]*/, v201 /*v713*/ offset:27648
	ds_load_tr16_b128 v[152:155] /*v[408:411]*/, v201 /*v713*/ offset:27680
	ds_load_tr16_b128 v[20:23] /*v[276:279]*/, v201 /*v713*/ offset:32256
	ds_load_tr16_b128 v[156:159] /*v[412:415]*/, v201 /*v713*/ offset:32288
	ds_load_tr16_b128 v[160:163] /*v[416:419]*/, v201 /*v713*/ offset:64
	ds_load_tr16_b128 v[168:171] /*v[424:427]*/, v201 /*v713*/ offset:96
	ds_load_tr16_b128 v[164:167] /*v[420:423]*/, v201 /*v713*/ offset:4672
	ds_load_tr16_b128 v[172:175] /*v[428:431]*/, v201 /*v713*/ offset:4704
	s_set_vgpr_msb 0x4285
	v_cvt_pk_bf16_f32 v109 /*v621*/, v213 /*v469*/, v215 /*v471*/
	v_cvt_pk_bf16_f32 v108 /*v620*/, v209 /*v465*/, v211 /*v467*/
	v_cvt_pk_bf16_f32 v97 /*v609*/, v141 /*v397*/, v217 /*v473*/
	s_set_vgpr_msb 0x8542
	ds_load_tr16_b128 v[176:179] /*v[432:435]*/, v201 /*v713*/ offset:9280
	s_wait_alu depctr_va_vdst(1)
	ds_load_tr16_b128 v[208:211] /*v[464:467]*/, v201 /*v713*/ offset:9312
	ds_load_tr16_b128 v[180:183] /*v[436:439]*/, v201 /*v713*/ offset:13888
	ds_load_tr16_b128 v[212:215] /*v[468:471]*/, v201 /*v713*/ offset:13920
	ds_load_tr16_b128 v[184:187] /*v[440:443]*/, v201 /*v713*/ offset:18496
	s_wait_alu depctr_va_vdst(0)
	ds_load_tr16_b128 v[216:219] /*v[472:475]*/, v201 /*v713*/ offset:18528
	ds_load_tr16_b128 v[188:191] /*v[444:447]*/, v201 /*v713*/ offset:23104
	ds_load_tr16_b128 v[220:223] /*v[476:479]*/, v201 /*v713*/ offset:23136
	s_set_vgpr_msb 0x4285
	v_cvt_pk_bf16_f32 v79 /*v591*/, v242 /*v498*/, v250 /*v506*/
	v_cvt_pk_bf16_f32 v78 /*v590*/, v238 /*v494*/, v240 /*v496*/
	s_set_vgpr_msb 0x8509
	s_wait_dscnt 0x9
	v_wmma_f32_16x16x32_bf16 v[136:143], v[160:167] /*v[416:423]*/, v[88:95] /*v[600:607]*/, v[136:143]
	s_set_vgpr_msb 0x985
	v_cvt_pk_bf16_f32 v77 /*v589*/, v230 /*v486*/, v234 /*v490*/
	v_cvt_pk_bf16_f32 v76 /*v588*/, v118 /*v374*/, v226 /*v482*/
	v_cvt_pk_bf16_f32 v74 /*v586*/, v224 /*v480*/, v112 /*v368*/
	v_cvt_pk_bf16_f32 v66 /*v578*/, v246 /*v502*/, v248 /*v504*/
	v_cvt_pk_bf16_f32 v65 /*v577*/, v236 /*v492*/, v244 /*v500*/
	v_cvt_pk_bf16_f32 v64 /*v576*/, v228 /*v484*/, v232 /*v488*/
	v_cvt_pk_bf16_f32 v103 /*v615*/, v243 /*v499*/, v251 /*v507*/
	s_set_vgpr_msb 0x8509
	v_wmma_f32_16x16x32_bf16 v[40:47], v[160:167] /*v[416:423]*/, v[112:119] /*v[624:631]*/, v[40:47]
	s_set_vgpr_msb 0x985
	v_cvt_pk_bf16_f32 v102 /*v614*/, v239 /*v495*/, v241 /*v497*/
	v_cvt_pk_bf16_f32 v101 /*v613*/, v231 /*v487*/, v235 /*v491*/
	v_cvt_pk_bf16_f32 v100 /*v612*/, v119 /*v375*/, v227 /*v483*/
	v_cvt_pk_bf16_f32 v98 /*v610*/, v225 /*v481*/, v113 /*v369*/
	v_cvt_pk_bf16_f32 v50 /*v562*/, v247 /*v503*/, v249 /*v505*/
	v_cvt_pk_bf16_f32 v49 /*v561*/, v237 /*v493*/, v245 /*v501*/
	v_cvt_pk_bf16_f32 v48 /*v560*/, v229 /*v485*/, v233 /*v489*/
	s_set_vgpr_msb 0x8509
	s_wait_dscnt 0x5
	v_wmma_f32_16x16x32_bf16 v[136:143], v[176:183] /*v[432:439]*/, v[80:87] /*v[592:599]*/, v[136:143]
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x942
	ds_load_tr16_b128 v[224:227] /*v[480:483]*/, v201 /*v713*/ offset:27712
	ds_load_tr16_b128 v[232:235] /*v[488:491]*/, v201 /*v713*/ offset:27744
	ds_load_tr16_b128 v[228:231] /*v[484:487]*/, v201 /*v713*/ offset:32320
	ds_load_tr16_b128 v[236:239] /*v[492:495]*/, v201 /*v713*/ offset:32352
	ds_load_tr16_b128 v[240:243] /*v[496:499]*/, v201 /*v713*/ offset:128
	s_set_vgpr_msb 0x4282
	ds_load_tr16_b128 v[132:135] /*v[644:647]*/, v201 /*v713*/ offset:160
	s_set_vgpr_msb 0x8242
	ds_load_tr16_b128 v[244:247] /*v[500:503]*/, v201 /*v713*/ offset:4736
	s_set_vgpr_msb 0x4282
	ds_load_tr16_b128 v[136:139] /*v[648:651]*/, v201 /*v713*/ offset:4768
	s_set_vgpr_msb 0x8285
	v_cvt_pk_bf16_f32 v75 /*v587*/, v114 /*v370*/, v116 /*v372*/
	v_cvt_pk_bf16_f32 v72 /*v584*/, v136 /*v392*/, v138 /*v394*/
	v_cvt_pk_bf16_f32 v99 /*v611*/, v115 /*v371*/, v117 /*v373*/
	v_cvt_pk_bf16_f32 v96 /*v608*/, v137 /*v393*/, v139 /*v395*/
	s_set_vgpr_msb 0x858a
	v_cvt_pk_bf16_f32 v71 /*v583*/, v4 /*v516*/, v6 /*v518*/
	v_cvt_pk_bf16_f32 v70 /*v582*/, v0 /*v512*/, v2 /*v514*/
	s_set_vgpr_msb 0x8a09
	v_wmma_f32_16x16x32_bf16 v[40:47], v[176:183] /*v[432:439]*/, v[104:111] /*v[616:623]*/, v[40:47]
	s_set_vgpr_msb 0x985
	v_cvt_pk_bf16_f32 v69 /*v581*/, v92 /*v348*/, v94 /*v350*/
	v_cvt_pk_bf16_f32 v68 /*v580*/, v88 /*v344*/, v90 /*v346*/
	v_cvt_pk_bf16_f32 v67 /*v579*/, v252 /*v508*/, v254 /*v510*/
	s_set_vgpr_msb 0x858a
	v_cvt_pk_bf16_f32 v55 /*v567*/, v5 /*v517*/, v7 /*v519*/
	v_cvt_pk_bf16_f32 v54 /*v566*/, v1 /*v513*/, v3 /*v515*/
	s_set_vgpr_msb 0x8a85
	v_cvt_pk_bf16_f32 v53 /*v565*/, v93 /*v349*/, v95 /*v351*/
	v_cvt_pk_bf16_f32 v52 /*v564*/, v89 /*v345*/, v91 /*v347*/
	s_set_vgpr_msb 0x8509
	s_wait_dscnt 0x9
	v_wmma_f32_16x16x32_bf16 v[136:143], v[184:191] /*v[440:447]*/, v[72:79] /*v[584:591]*/, v[136:143]
	s_set_vgpr_msb 0x985
	v_cvt_pk_bf16_f32 v51 /*v563*/, v253 /*v509*/, v255 /*v511*/
	s_set_vgpr_msb 0x8540
	v_max3_num_f32 v160 /*v416*/, v216, v217, v218
	v_max3_num_f32 v161 /*v417*/, v192, v193, v194
	v_max3_num_f32 v162 /*v418*/, v219, v220, v221
	v_max3_num_f32 v163 /*v419*/, v195, v196, v197
	v_max3_num_f32 v164 /*v420*/, v222, v223, v224
	v_max3_num_f32 v165 /*v421*/, v198, v199, v200
	s_set_vgpr_msb 0x4009
	v_wmma_f32_16x16x32_bf16 v[40:47], v[184:191] /*v[440:447]*/, v[96:103] /*v[608:615]*/, v[40:47]
	s_set_vgpr_msb 0x940
	v_max3_num_f32 v166 /*v422*/, v225, v226, v227
	v_max3_num_f32 v176 /*v432*/, v228, v229, v230
	v_max3_num_f32 v178 /*v434*/, v231, v232, v233
	v_max3_num_f32 v180 /*v436*/, v234, v235, v236
	v_max3_num_f32 v182 /*v438*/, v237, v238, v239
	s_set_vgpr_msb 0x40aa
	v_max3_num_f32 v24 /*v536*/, v20 /*v532*/, v21 /*v533*/, v22 /*v534*/
	s_set_vgpr_msb 0xaa55
	v_max3_num_f32 v160 /*v416*/, v160 /*v416*/, v162 /*v418*/, v164 /*v420*/
	s_set_vgpr_msb 0x5509
	v_wmma_f32_16x16x32_bf16 v[120:127], v[168:175] /*v[424:431]*/, v[88:95] /*v[600:607]*/, v[120:127]
	s_set_vgpr_msb 0x955
	v_max3_num_f32 v161 /*v417*/, v161 /*v417*/, v163 /*v419*/, v165 /*v421*/
	v_max3_num_f32 v162 /*v418*/, v166 /*v422*/, v176 /*v432*/, v178 /*v434*/
	s_set_vgpr_msb 0x5540
	v_max3_num_f32 v167 /*v423*/, v201, v202, v203
	v_max3_num_f32 v177 /*v433*/, v204, v205, v206
	v_max3_num_f32 v179 /*v435*/, v207, v208, v209
	v_max3_num_f32 v181 /*v437*/, v210, v211, v212
	v_max3_num_f32 v183 /*v439*/, v213, v214, v215
	s_set_vgpr_msb 0x4009
	v_wmma_f32_16x16x32_bf16 v[32:39], v[168:175] /*v[424:431]*/, v[112:119] /*v[624:631]*/, v[32:39]
	s_set_vgpr_msb 0x995
	v_max3_num_f32 v28 /*v540*/, v132 /*v388*/, v133 /*v389*/, v134 /*v390*/
	s_set_vgpr_msb 0x9555
	v_max3_num_f32 v167 /*v423*/, v167 /*v423*/, v177 /*v433*/, v179 /*v435*/
	s_set_vgpr_msb 0x5542
	ds_load_tr16_b128 v[248:251] /*v[504:507]*/, v201 /*v713*/ offset:9344
	s_set_vgpr_msb 0x4282
	ds_load_tr16_b128 v[208:211] /*v[720:723]*/, v201 /*v713*/ offset:9376
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x8242
	ds_load_tr16_b128 v[252:255] /*v[508:511]*/, v201 /*v713*/ offset:13952
	s_set_vgpr_msb 0x4282
	ds_load_tr16_b128 v[212:215] /*v[724:727]*/, v201 /*v713*/ offset:13984
	ds_load_tr16_b128 v[140:143] /*v[652:655]*/, v201 /*v713*/ offset:18560
	ds_load_tr16_b128 v[216:219] /*v[728:731]*/, v201 /*v713*/ offset:18592
	ds_load_tr16_b128 v[144:147] /*v[656:659]*/, v201 /*v713*/ offset:23168
	ds_load_tr16_b128 v[220:223] /*v[732:735]*/, v201 /*v713*/ offset:23200
	ds_load_tr16_b128 v[148:151] /*v[660:663]*/, v201 /*v713*/ offset:27776
	ds_load_tr16_b128 v[224:227] /*v[736:739]*/, v201 /*v713*/ offset:27808
	ds_load_tr16_b128 v[152:155] /*v[664:667]*/, v201 /*v713*/ offset:32384
	ds_load_tr16_b128 v[228:231] /*v[740:743]*/, v201 /*v713*/ offset:32416
	ds_load_tr16_b128 v[232:235] /*v[744:747]*/, v201 /*v713*/ offset:192
	ds_load_tr16_b128 v[240:243] /*v[752:755]*/, v201 /*v713*/ offset:224
	ds_load_tr16_b128 v[236:239] /*v[748:751]*/, v201 /*v713*/ offset:4800
	ds_load_tr16_b128 v[244:247] /*v[756:759]*/, v201 /*v713*/ offset:4832
	s_set_vgpr_msb 0x8242
	ds_load_tr16_b128 v[48:51] /*v[304:307]*/, v201 /*v713*/ offset:9216
	ds_load_tr16_b128 v[136:139] /*v[392:395]*/, v201 /*v713*/ offset:9248
	ds_load_tr16_b128 v[52:55] /*v[308:311]*/, v201 /*v713*/ offset:13824
	ds_load_tr16_b128 v[140:143] /*v[396:399]*/, v201 /*v713*/ offset:13856
	ds_load_tr16_b128 v[56:59] /*v[312:315]*/, v201 /*v713*/ offset:18432
	ds_load_tr16_b128 v[144:147] /*v[400:403]*/, v201 /*v713*/ offset:18464
	ds_load_tr16_b128 v[60:63] /*v[316:319]*/, v201 /*v713*/ offset:23040
	ds_load_tr16_b128 v[148:151] /*v[404:407]*/, v201 /*v713*/ offset:23072
	s_set_vgpr_msb 0x4282
	ds_load_tr16_b128 v[248:251] /*v[760:763]*/, v201 /*v713*/ offset:9408
	s_set_vgpr_msb 0x82c2
	ds_load_tr16_b128 v[0:3] /*v[768:771]*/, v201 /*v713*/ offset:9440
	s_set_vgpr_msb 0xc282
	ds_load_tr16_b128 v[252:255] /*v[764:767]*/, v201 /*v713*/ offset:14016
	s_set_vgpr_msb 0x82c2
	ds_load_tr16_b128 v[4:7] /*v[772:775]*/, v201 /*v713*/ offset:14048
	ds_load_tr16_b128 v[8:11] /*v[776:779]*/, v201 /*v713*/ offset:18624
	ds_load_tr16_b128 v[16:19] /*v[784:787]*/, v201 /*v713*/ offset:18656
	ds_load_tr16_b128 v[12:15] /*v[780:783]*/, v201 /*v713*/ offset:23232
	ds_load_tr16_b128 v[20:23] /*v[788:791]*/, v201 /*v713*/ offset:23264
	ds_load_tr16_b128 v[24:27] /*v[792:795]*/, v201 /*v713*/ offset:27840
	s_set_vgpr_msb 0xc282
	ds_load_tr16_b128 v[56:59] /*v[568:571]*/, v201 /*v713*/ offset:27872
	s_set_vgpr_msb 0x82c2
	ds_load_tr16_b128 v[28:31] /*v[796:799]*/, v201 /*v713*/ offset:32448
	s_set_vgpr_msb 0xc282
	ds_load_tr16_b128 v[60:63] /*v[572:575]*/, v201 /*v713*/ offset:32480
	s_set_vgpr_msb 0x8242
	ds_load_tr16_b128 v[112:115] /*v[368:371]*/, v204 /*v716*/
	ds_load_tr16_b128 v[64:67] /*v[320:323]*/, v204 /*v716*/ offset:32
	ds_load_tr16_b128 v[116:119] /*v[372:375]*/, v204 /*v716*/ offset:4608
	ds_load_tr16_b128 v[68:71] /*v[324:327]*/, v204 /*v716*/ offset:4640
	s_set_vgpr_msb 0x4209
	s_wait_dscnt 0x2d
	v_wmma_f32_16x16x32_bf16 v[136:143], v[224:231] /*v[480:487]*/, v[64:71] /*v[576:583]*/, v[136:143]
	v_nop
	s_set_vgpr_msb 0x955
	v_max3_num_f32 v169 /*v425*/, v11 /*v267*/, v12 /*v268*/, v13 /*v269*/
	v_max3_num_f32 v171 /*v427*/, v14 /*v270*/, v15 /*v271*/, v24 /*v280*/
	v_max3_num_f32 v173 /*v429*/, v25 /*v281*/, v26 /*v282*/, v27 /*v283*/
	v_max3_num_f32 v175 /*v431*/, v28 /*v284*/, v29 /*v285*/, v30 /*v286*/
	s_set_vgpr_msb 0x5540
	v_max3_num_f32 v168 /*v424*/, v240, v241, v242
	v_max3_num_f32 v170 /*v426*/, v243, v244, v245
	s_set_vgpr_msb 0x4009
	v_wmma_f32_16x16x32_bf16 v[40:47], v[224:231] /*v[480:487]*/, v[48:55] /*v[560:567]*/, v[40:47]
	s_set_vgpr_msb 0x955
	v_max3_num_f32 v164 /*v420*/, v169 /*v425*/, v171 /*v427*/, v173 /*v429*/
	s_set_vgpr_msb 0x5540
	v_max3_num_f32 v172 /*v428*/, v246, v247, v248
	v_max3_num_f32 v174 /*v430*/, v249, v250, v251
	v_nop
	s_set_vgpr_msb 0x4055
	v_max3_num_f32 v224 /*v480*/, v8 /*v264*/, v9 /*v265*/, v10 /*v266*/
	v_max3_num_f32 v226 /*v482*/, v31 /*v287*/, v96 /*v352*/, v97 /*v353*/
	v_max3_num_f32 v228 /*v484*/, v98 /*v354*/, v99 /*v355*/, v100 /*v356*/
	s_set_vgpr_msb 0x5509
	v_wmma_f32_16x16x32_bf16 v[120:127], v[208:215] /*v[464:471]*/, v[80:87] /*v[592:599]*/, v[120:127]
	s_set_vgpr_msb 0x955
	v_max3_num_f32 v230 /*v486*/, v101 /*v357*/, v102 /*v358*/, v103 /*v359*/
	v_max3_num_f32 v163 /*v419*/, v180 /*v436*/, v182 /*v438*/, v224 /*v480*/
	s_set_vgpr_msb 0x5540
	v_max3_num_f32 v225 /*v481*/, v252, v253, v254
	s_set_vgpr_msb 0x4055
	v_max3_num_f32 v165 /*v421*/, v175 /*v431*/, v226 /*v482*/, v228 /*v484*/
	s_set_vgpr_msb 0x5554
	v_max3_num_f32 v227 /*v483*/, v255, v0 /*v256*/, v1 /*v257*/
	s_set_vgpr_msb 0x5455
	v_max3_num_f32 v229 /*v485*/, v2 /*v258*/, v3 /*v259*/, v4 /*v260*/
	v_max3_num_f32 v231 /*v487*/, v129 /*v385*/, v130 /*v386*/, v131 /*v387*/
	s_set_vgpr_msb 0x5509
	v_wmma_f32_16x16x32_bf16 v[32:39], v[208:215] /*v[464:471]*/, v[104:111] /*v[616:623]*/, v[32:39]
	s_set_vgpr_msb 0x955
	v_max3_num_f32 v160 /*v416*/, v160 /*v416*/, v162 /*v418*/, v163 /*v419*/
	v_nop
	v_nop
	v_nop
	v_max3_num_f32 v209 /*v465*/, v120 /*v376*/, v121 /*v377*/, v122 /*v378*/
	v_max3_num_f32 v211 /*v467*/, v123 /*v379*/, v124 /*v380*/, v125 /*v381*/
	v_max_num_f32_e32 v213 /*v469*/, v126 /*v382*/, v127 /*v383*/
	s_set_vgpr_msb 0x556a
	v_max3_num_f32 v215 /*v471*/, v17 /*v529*/, v18 /*v530*/, v19 /*v531*/
	s_set_vgpr_msb 0x6a55
	v_max3_num_f32 v208 /*v464*/, v5 /*v261*/, v6 /*v262*/, v7 /*v263*/
	v_max3_num_f32 v210 /*v466*/, v32 /*v288*/, v33 /*v289*/, v34 /*v290*/
	v_max3_num_f32 v166 /*v422*/, v230 /*v486*/, v209 /*v465*/, v211 /*v467*/
	v_max3_num_f32 v212 /*v468*/, v35 /*v291*/, v36 /*v292*/, v37 /*v293*/
	s_set_vgpr_msb 0x5559
	v_max3_num_f32 v169 /*v425*/, v213 /*v469*/, v16 /*v528*/, v215 /*v471*/
	s_set_vgpr_msb 0x5945
	v_max_num_f32_e32 v214 /*v470*/, v38 /*v294*/, v39 /*v295*/
	s_set_vgpr_msb 0x4509
	s_wait_dscnt 0x29
	v_wmma_f32_16x16x32_bf16 v[112:119], v[240:247] /*v[496:503]*/, v[88:95] /*v[600:607]*/, v[112:119]
	s_set_vgpr_msb 0x955
	v_max3_num_f32 v162 /*v418*/, v164 /*v420*/, v165 /*v421*/, v166 /*v422*/
	v_max3_num_f32 v164 /*v420*/, v181 /*v437*/, v183 /*v439*/, v168 /*v424*/
	s_set_vgpr_msb 0x5569
	v_max3_num_f32 v163 /*v419*/, v169 /*v425*/, v24 /*v536*/, v23 /*v535*/
	s_set_vgpr_msb 0x6955
	v_max3_num_f32 v165 /*v421*/, v170 /*v426*/, v172 /*v428*/, v174 /*v430*/
	v_max3_num_f32 v166 /*v422*/, v225 /*v481*/, v227 /*v483*/, v229 /*v485*/
	v_max3_num_f32 v168 /*v424*/, v208 /*v464*/, v210 /*v466*/, v212 /*v468*/
	v_max3_num_f32 v161 /*v417*/, v161 /*v417*/, v167 /*v423*/, v164 /*v420*/
	v_max3_num_f32 v160 /*v416*/, v160 /*v416*/, v162 /*v418*/, v163 /*v419*/
	v_max3_num_f32 v162 /*v418*/, v214 /*v470*/, v128 /*v384*/, v231 /*v487*/
	s_set_vgpr_msb 0x5509
	v_wmma_f32_16x16x32_bf16 v[24:31], v[240:247] /*v[496:503]*/, v[112:119] /*v[624:631]*/, v[24:31]
	s_set_vgpr_msb 0x955
	v_max3_num_f32 v164 /*v420*/, v165 /*v421*/, v166 /*v422*/, v168 /*v424*/
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x5582
	ds_load_tr16_b128 v[24:27] /*v[536:539]*/, v204 /*v716*/ offset:9344
	s_set_vgpr_msb 0x8242
	ds_load_tr16_b128 v[208:211] /*v[464:467]*/, v204 /*v716*/ offset:9376
	s_set_vgpr_msb 0x4259
	v_mov_b32_e32 v163 /*v419*/, v160 /*v416*/
	v_max3_num_f32 v162 /*v418*/, v162 /*v418*/, v28 /*v540*/, v135 /*v391*/
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x5982
	ds_load_tr16_b128 v[28:31] /*v[540:543]*/, v204 /*v716*/ offset:13952
	s_set_vgpr_msb 0x8242
	ds_load_tr16_b128 v[212:215] /*v[468:471]*/, v204 /*v716*/ offset:13984
	s_set_vgpr_msb 0x4282
	ds_load_tr16_b128 v[44:47] /*v[556:559]*/, v204 /*v716*/ offset:32384
	s_set_vgpr_msb 0x8242
	ds_load_tr16_b128 v[244:247] /*v[500:503]*/, v204 /*v716*/ offset:32416
	s_set_vgpr_msb 0x4255
	v_permlanex16_b32 v163 /*v419*/, v163 /*v419*/, s66, 0xfedcba98
	v_max3_num_f32 v161 /*v417*/, v161 /*v417*/, v164 /*v420*/, v162 /*v418*/
	s_set_vgpr_msb 0x5509
	s_wait_dscnt 0x2b
	v_wmma_f32_16x16x32_bf16 v[112:119], v[248:255] /*v[504:511]*/, v[80:87] /*v[592:599]*/, v[112:119]
	s_set_vgpr_msb 0x945
	v_dual_max_num_f32 v164 /*v420*/, v160 /*v416*/, v163 /*v419*/ :: v_dual_mov_b32 v160 /*v416*/, v161 /*v417*/
	v_permlanex16_b32 v160 /*v416*/, v160 /*v416*/, s66, 0xfedcba98
	s_set_vgpr_msb 0x4509
	v_wmma_f32_16x16x32_bf16 v[24:31], v[248:255] /*v[504:511]*/, v[104:111] /*v[616:623]*/, v[24:31]
	s_set_vgpr_msb 0x945
	v_max_num_f32_e32 v172 /*v428*/, v161 /*v417*/, v160 /*v416*/
	s_set_vgpr_msb 0x4549
	v_sub_f32_e32 v162 /*v418*/, v164 /*v420*/, v206 /*v718*/
	v_max_num_f32_e32 v168 /*v424*/, v164 /*v420*/, v206 /*v718*/
	s_set_vgpr_msb 0x4942
	ds_load_tr16_b128 v[228:231] /*v[484:487]*/, v204 /*v716*/ offset:4800
	s_wait_alu depctr_va_vdst(0)
	ds_load_tr16_b128 v[164:167] /*v[420:423]*/, v204 /*v716*/ offset:4832
	s_set_vgpr_msb 0x420a
	s_wait_dscnt 0x29
	v_wmma_f32_16x16x32_bf16 v[112:119], v[140:147] /*v[652:659]*/, v[72:79] /*v[584:591]*/, v[112:119]
	s_set_vgpr_msb 0xa46
	v_max_num_f32_e32 v176 /*v432*/, v205 /*v717*/, v172 /*v428*/
	v_cmp_lt_f32_e32 vcc_lo, 0x41000000, v162 /*v418*/
	ds_load_tr16_b128 v[224:227] /*v[480:483]*/, v204 /*v716*/ offset:192
	s_wait_alu depctr_va_vdst(0)
	ds_load_tr16_b128 v[160:163] /*v[416:419]*/, v204 /*v716*/ offset:224
	s_cmp_eq_u32 vcc_lo, 0
	s_set_vgpr_msb 0x460a
	v_wmma_f32_16x16x32_bf16 v[24:31], v[140:147] /*v[652:659]*/, v[96:103] /*v[608:615]*/, v[24:31]
	s_cselect_b32 s2, -1, 0
	s_set_vgpr_msb 0xa89
	v_cndmask_b32_e64 v198 /*v710*/, v168 /*v424*/, v206 /*v718*/, s2
	s_set_vgpr_msb 0x8949
	v_dual_sub_f32 v169 /*v425*/, v172 /*v428*/, v205 /*v717*/ :: v_dual_mul_f32 v180 /*v436*/, 0xbfb8aa3b, v198 /*v710*/
	s_set_vgpr_msb 0x4904
	v_cmp_lt_f32_e64 s2, 0x41000000, v169 /*v425*/
	s_set_vgpr_msb 0x40a
	s_wait_dscnt 0x27
	v_wmma_f32_16x16x32_bf16 v[112:119], v[148:155] /*v[660:667]*/, v[64:71] /*v[576:583]*/, v[112:119]
	s_set_vgpr_msb 0xa10
	v_pk_fma_f32 v[216:217], v[216:217], s[28:29], v[180:181] /*v[436:437]*/ op_sel_hi:[1,0,0]
	s_cmp_lg_u32 s2, 0
	s_set_vgpr_msb 0x1050
	v_pk_fma_f32 v[182:183] /*v[438:439]*/, v[218:219], s[28:29], v[180:181] /*v[436:437]*/ op_sel_hi:[1,0,0]
	s_cselect_b32 s3, -1, 0
	s_cmp_eq_u32 s2, 0
	s_set_vgpr_msb 0x5000
	v_exp_f32_e32 v218, v217
	s_cselect_b32 s2, -1, 0
	s_set_vgpr_msb 10
	v_wmma_f32_16x16x32_bf16 v[24:31], v[148:155] /*v[660:667]*/, v[48:55] /*v[560:567]*/, v[24:31]
	s_set_vgpr_msb 0xa89
	v_cndmask_b32_e64 v203 /*v715*/, v176 /*v432*/, v205 /*v717*/, s2
	s_set_vgpr_msb 0x8910
	v_pk_fma_f32 v[224:225], v[224:225], s[28:29], v[180:181] /*v[436:437]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[226:227], v[226:227], s[28:29], v[180:181] /*v[436:437]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[228:229], v[228:229], s[28:29], v[180:181] /*v[436:437]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1058
	v_pk_fma_f32 v[254:255] /*v[510:511]*/, v[220:221], s[28:29], v[180:181] /*v[436:437]*/ op_sel_hi:[1,0,0]
	v_mul_f32_e32 v252 /*v508*/, 0xbfb8aa3b, v203 /*v715*/
	s_set_vgpr_msb 0x5880
	v_exp_f32_e32 v154 /*v666*/, v224
	v_exp_f32_e32 v158 /*v670*/, v225
	v_exp_f32_e32 v162 /*v674*/, v226
	v_exp_f32_e32 v166 /*v678*/, v227
	s_set_vgpr_msb 0x8010
	v_pk_fma_f32 v[192:193], v[192:193], s[28:29], v[252:253] /*v[508:509]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[224:225], v[230:231], s[28:29], v[180:181] /*v[436:437]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[226:227], v[232:233], s[28:29], v[180:181] /*v[436:437]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[194:195], v[194:195], s[28:29], v[252:253] /*v[508:509]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[232:233], v[236:237], s[28:29], v[180:181] /*v[436:437]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v217, v192
	v_exp_f32_e32 v219, v193
	v_nop
	v_pk_fma_f32 v[192:193], v[198:199], s[28:29], v[252:253] /*v[508:509]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1080
	v_exp_f32_e32 v178 /*v690*/, v224
	v_exp_f32_e32 v182 /*v694*/, v225
	s_set_vgpr_msb 0x8000
	v_exp_f32_e32 v224, v226
	v_exp_f32_e32 v226, v227
	s_set_vgpr_msb 0x80
	v_exp_f32_e32 v147 /*v659*/, v192
	v_exp_f32_e32 v149 /*v661*/, v193
	v_nop
	s_set_vgpr_msb 0x8010
	v_pk_fma_f32 v[192:193], v[202:203], s[28:29], v[252:253] /*v[508:509]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[196:197], v[196:197], s[28:29], v[252:253] /*v[508:509]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1090
	v_pk_fma_f32 v[140:141] /*v[652:653]*/, v[222:223], s[28:29], v[180:181] /*v[436:437]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x9000
	v_exp_f32_e32 v221, v194
	v_exp_f32_e32 v223, v195
	s_set_vgpr_msb 0x80
	v_exp_f32_e32 v163 /*v675*/, v192
	v_exp_f32_e32 v167 /*v679*/, v193
	v_nop
	s_set_vgpr_msb 0x8010
	v_pk_fma_f32 v[192:193], v[208:209], s[28:29], v[252:253] /*v[508:509]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[194:195], v[200:201], s[28:29], v[252:253] /*v[508:509]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1080
	v_exp_f32_e32 v170 /*v682*/, v228
	v_exp_f32_e32 v174 /*v686*/, v229
	v_nop
	s_set_vgpr_msb 0x8010
	v_pk_fma_f32 v[228:229], v[234:235], s[28:29], v[180:181] /*v[436:437]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v225, v192
	v_exp_f32_e32 v227, v193
	v_nop
	v_pk_fma_f32 v[192:193], v[212:213], s[28:29], v[252:253] /*v[508:509]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v234, v233
	s_set_vgpr_msb 0x1080
	v_exp_f32_e32 v129 /*v641*/, v196
	v_exp_f32_e32 v131 /*v643*/, v197
	v_nop
	s_set_vgpr_msb 0x8010
	v_pk_fma_f32 v[196:197], v[206:207], s[28:29], v[252:253] /*v[508:509]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v233, v192
	v_exp_f32_e32 v235, v193
	v_nop
	v_pk_fma_f32 v[192:193], v[240:241], s[28:29], v[252:253] /*v[508:509]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1080
	v_exp_f32_e32 v155 /*v667*/, v194
	v_exp_f32_e32 v159 /*v671*/, v195
	v_nop
	s_set_vgpr_msb 0x8010
	v_pk_fma_f32 v[194:195], v[204:205], s[28:29], v[252:253] /*v[508:509]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x100a
	v_wmma_f32_16x16x32_bf16 v[96:103], v[132:139] /*v[644:651]*/, v[88:95] /*v[600:607]*/, v[96:103]
	s_set_vgpr_msb 0xa51
	v_pk_fma_f32 v[10:11] /*v[266:267]*/, v[10:11] /*v[266:267]*/, s[28:29], v[180:181] /*v[436:437]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5180
	v_exp_f32_e32 v179 /*v691*/, v196
	v_exp_f32_e32 v183 /*v695*/, v197
	v_nop
	s_set_vgpr_msb 0x8010
	v_pk_fma_f32 v[196:197], v[214:215], s[28:29], v[252:253] /*v[508:509]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[236:237], v[238:239], s[28:29], v[180:181] /*v[436:437]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1080
	v_exp_f32_e32 v171 /*v683*/, v194
	v_exp_f32_e32 v175 /*v687*/, v195
	s_set_vgpr_msb 0x800a
	v_wmma_f32_16x16x32_bf16 v[16:23], v[132:139] /*v[644:651]*/, v[112:119] /*v[624:631]*/, v[16:23]
	s_set_vgpr_msb 0xa10
	v_pk_fma_f32 v[194:195], v[210:211], s[28:29], v[252:253] /*v[508:509]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1082
	v_exp_f32_e32 v146 /*v658*/, v140 /*v652*/
	s_set_vgpr_msb 0x8211
	v_pk_fma_f32 v[238:239], v[8:9] /*v[264:265]*/, s[28:29], v[180:181] /*v[436:437]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1181
	v_exp_f32_e32 v140 /*v652*/, v11 /*v267*/
	s_set_vgpr_msb 0x8180
	v_exp_f32_e32 v135 /*v647*/, v192
	v_exp_f32_e32 v137 /*v649*/, v193
	v_nop
	s_set_vgpr_msb 0x8010
	v_pk_fma_f32 v[192:193], v[246:247], s[28:29], v[252:253] /*v[508:509]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1081
	v_exp_f32_e32 v138 /*v650*/, v10 /*v266*/
	v_nop
	s_set_vgpr_msb 0x8151
	v_pk_fma_f32 v[10:11] /*v[266:267]*/, v[26:27] /*v[282:283]*/, s[28:29], v[180:181] /*v[436:437]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5140
	v_exp_f32_e32 v9 /*v265*/, v196
	s_set_vgpr_msb 0x4080
	v_exp_f32_e32 v133 /*v645*/, v197
	v_nop
	s_set_vgpr_msb 0x8010
	v_pk_fma_f32 v[196:197], v[244:245], s[28:29], v[252:253] /*v[508:509]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1080
	v_exp_f32_e32 v151 /*v663*/, v192
	v_exp_f32_e32 v157 /*v669*/, v193
	v_nop
	s_set_vgpr_msb 0x8010
	v_pk_fma_f32 v[192:193], v[250:251], s[28:29], v[252:253] /*v[508:509]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v230, v229
	s_set_vgpr_msb 0x1040
	v_exp_f32_e32 v8 /*v264*/, v236
	s_set_vgpr_msb 0x4080
	v_exp_f32_e32 v132 /*v644*/, v237
	v_nop
	s_set_vgpr_msb 0x8011
	v_pk_fma_f32 v[236:237], v[12:13] /*v[268:269]*/, s[28:29], v[180:181] /*v[436:437]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1110
	v_exp_f32_e32 v229, v194
	v_exp_f32_e32 v231, v195
	v_nop
	v_pk_fma_f32 v[194:195], v[242:243], s[28:29], v[252:253] /*v[508:509]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1080
	v_exp_f32_e32 v134 /*v646*/, v238
	v_exp_f32_e32 v136 /*v648*/, v239
	v_nop
	s_set_vgpr_msb 0x8011
	v_pk_fma_f32 v[238:239], v[14:15] /*v[270:271]*/, s[28:29], v[180:181] /*v[436:437]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1151
	v_pk_fma_f32 v[14:15] /*v[270:271]*/, v[28:29] /*v[284:285]*/, s[28:29], v[180:181] /*v[436:437]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v12 /*v268*/, v11 /*v267*/
	v_pk_fma_f32 v[26:27] /*v[282:283]*/, v[30:31] /*v[286:287]*/, s[28:29], v[180:181] /*v[436:437]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5180
	v_exp_f32_e32 v143 /*v655*/, v196
	v_exp_f32_e32 v145 /*v657*/, v197
	v_nop
	s_set_vgpr_msb 0x8010
	v_pk_fma_f32 v[196:197], v[252:253], s[28:29], v[252:253] /*v[508:509]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1040
	v_exp_f32_e32 v11 /*v267*/, v192
	v_exp_f32_e32 v13 /*v269*/, v193
	v_nop
	s_set_vgpr_msb 0x4010
	v_pk_fma_f32 v[192:193], v[254:255], s[28:29], v[252:253] /*v[508:509]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1082
	v_exp_f32_e32 v148 /*v660*/, v141 /*v653*/
	s_set_vgpr_msb 0x8280
	v_exp_f32_e32 v142 /*v654*/, v236
	v_exp_f32_e32 v144 /*v656*/, v237
	v_nop
	s_set_vgpr_msb 0x8011
	v_pk_fma_f32 v[236:237], v[24:25] /*v[280:281]*/, s[28:29], v[180:181] /*v[436:437]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1180
	v_exp_f32_e32 v139 /*v651*/, v194
	v_exp_f32_e32 v141 /*v653*/, v195
	v_nop
	s_set_vgpr_msb 0x8010
	v_pk_fma_f32 v[194:195], v[248:249], s[28:29], v[252:253] /*v[508:509]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1051
	v_pk_fma_f32 v[96:97] /*v[352:353]*/, v[96:97] /*v[352:353]*/, s[28:29], v[180:181] /*v[436:437]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v24 /*v280*/, v15 /*v271*/
	v_exp_f32_e32 v28 /*v284*/, v26 /*v282*/
	v_exp_f32_e32 v30 /*v286*/, v27 /*v283*/
	v_nop
	v_pk_fma_f32 v[26:27] /*v[282:283]*/, v[98:99] /*v[354:355]*/, s[28:29], v[180:181] /*v[436:437]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[98:99] /*v[354:355]*/, v[100:101] /*v[356:357]*/, s[28:29], v[180:181] /*v[436:437]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5140
	v_exp_f32_e32 v15 /*v271*/, v196
	v_exp_f32_e32 v25 /*v281*/, v197
	v_exp_f32_e32 v29 /*v285*/, v192
	s_set_vgpr_msb 0x4011
	v_pk_fma_f32 v[196:197], v[2:3] /*v[258:259]*/, s[28:29], v[252:253] /*v[508:509]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1140
	v_exp_f32_e32 v31 /*v287*/, v193
	v_nop
	s_set_vgpr_msb 0x4011
	v_pk_fma_f32 v[192:193], v[4:5] /*v[260:261]*/, s[28:29], v[252:253] /*v[508:509]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1180
	v_exp_f32_e32 v150 /*v662*/, v238
	v_exp_f32_e32 v156 /*v668*/, v239
	s_set_vgpr_msb 0x8000
	v_exp_f32_e32 v238, v237
	v_exp_f32_e32 v237, v194
	v_exp_f32_e32 v239, v195
	v_nop
	s_set_vgpr_msb 17
	v_pk_fma_f32 v[194:195], v[0:1] /*v[256:257]*/, s[28:29], v[252:253] /*v[508:509]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1181
	v_exp_f32_e32 v152 /*v664*/, v96 /*v352*/
	v_exp_f32_e32 v160 /*v672*/, v97 /*v353*/
	v_exp_f32_e32 v164 /*v676*/, v26 /*v282*/
	v_exp_f32_e32 v168 /*v680*/, v27 /*v283*/
	v_nop
	s_set_vgpr_msb 0x8151
	v_pk_fma_f32 v[26:27] /*v[282:283]*/, v[102:103] /*v[358:359]*/, s[28:29], v[180:181] /*v[436:437]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5181
	v_exp_f32_e32 v172 /*v684*/, v98 /*v354*/
	s_set_vgpr_msb 0x8151
	v_pk_fma_f32 v[96:97] /*v[352:353]*/, v[120:121] /*v[376:377]*/, s[28:29], v[180:181] /*v[436:437]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5181
	v_exp_f32_e32 v176 /*v688*/, v99 /*v355*/
	v_nop
	s_set_vgpr_msb 0x8151
	v_pk_fma_f32 v[98:99] /*v[354:355]*/, v[122:123] /*v[378:379]*/, s[28:29], v[180:181] /*v[436:437]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5180
	v_exp_f32_e32 v165 /*v677*/, v196
	v_exp_f32_e32 v169 /*v681*/, v197
	v_exp_f32_e32 v173 /*v685*/, v192
	v_exp_f32_e32 v177 /*v689*/, v193
	v_nop
	s_set_vgpr_msb 0x8011
	v_pk_fma_f32 v[192:193], v[32:33] /*v[288:289]*/, s[28:29], v[252:253] /*v[508:509]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[196:197], v[34:35] /*v[290:291]*/, s[28:29], v[252:253] /*v[508:509]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1100
	v_exp_f32_e32 v216, v216
	s_set_vgpr_msb 1
	v_exp_f32_e32 v222, v183 /*v439*/
	s_set_vgpr_msb 0x181
	v_exp_f32_e32 v128 /*v640*/, v254 /*v510*/
	s_set_vgpr_msb 0x8180
	v_exp_f32_e32 v153 /*v665*/, v194
	v_exp_f32_e32 v161 /*v673*/, v195
	v_nop
	s_set_vgpr_msb 0x8011
	v_pk_fma_f32 v[194:195], v[6:7] /*v[262:263]*/, s[28:29], v[252:253] /*v[508:509]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1181
	v_exp_f32_e32 v180 /*v692*/, v26 /*v282*/
	v_exp_f32_e32 v184 /*v696*/, v27 /*v283*/
	s_set_vgpr_msb 0x8151
	v_exp_f32_e32 v26 /*v282*/, v96 /*v352*/
	v_exp_f32_e32 v96 /*v352*/, v97 /*v353*/
	v_exp_f32_e32 v100 /*v356*/, v99 /*v355*/
	v_pk_fma_f32 v[122:123] /*v[378:379]*/, v[126:127] /*v[382:383]*/, s[28:29], v[180:181] /*v[436:437]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5152
	v_pk_fma_f32 v[126:127] /*v[382:383]*/, v[16:17] /*v[528:529]*/, s[28:29], v[180:181] /*v[436:437]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5240
	v_exp_f32_e32 v27 /*v283*/, v192
	v_exp_f32_e32 v97 /*v353*/, v193
	v_exp_f32_e32 v99 /*v355*/, v196
	v_exp_f32_e32 v101 /*v357*/, v197
	s_set_vgpr_msb 0x4011
	v_pk_fma_f32 v[192:193], v[38:39] /*v[294:295]*/, s[28:29], v[252:253] /*v[508:509]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[196:197], v[128:129] /*v[384:385]*/, s[28:29], v[252:253] /*v[508:509]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v220, v182 /*v438*/
	s_set_vgpr_msb 0x1181
	v_exp_f32_e32 v130 /*v642*/, v255 /*v511*/
	s_set_vgpr_msb 0x8151
	v_exp_f32_e32 v14 /*v270*/, v14 /*v270*/
	v_pk_fma_f32 v[102:103] /*v[358:359]*/, v[124:125] /*v[380:381]*/, s[28:29], v[180:181] /*v[436:437]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5180
	v_exp_f32_e32 v181 /*v693*/, v194
	v_exp_f32_e32 v185 /*v697*/, v195
	v_nop
	s_set_vgpr_msb 0x8011
	v_pk_fma_f32 v[194:195], v[36:37] /*v[292:293]*/, s[28:29], v[252:253] /*v[508:509]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1141
	v_exp_f32_e32 v124 /*v380*/, v123 /*v379*/
	s_set_vgpr_msb 0x4181
	v_exp_f32_e32 v16 /*v528*/, v127 /*v383*/
	s_set_vgpr_msb 0x8140
	v_exp_f32_e32 v123 /*v379*/, v192
	v_exp_f32_e32 v125 /*v381*/, v193
	v_exp_f32_e32 v127 /*v383*/, v196
	s_set_vgpr_msb 0x4080
	v_exp_f32_e32 v17 /*v529*/, v197
	s_set_vgpr_msb 0x8011
	v_pk_fma_f32 v[192:193], v[132:133] /*v[388:389]*/, s[28:29], v[252:253] /*v[508:509]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1100
	v_pk_add_f32 v[196:197], v[216:217], v[218:219]
	s_set_vgpr_msb 8
	v_pk_add_f32 v[198:199], v[222:223], v[128:129] /*v[640:641]*/
	v_exp_f32_e32 v228, v228
	v_exp_f32_e32 v236, v236
	s_set_vgpr_msb 0x841
	v_exp_f32_e32 v10 /*v266*/, v10 /*v266*/
	v_exp_f32_e32 v120 /*v376*/, v103 /*v359*/
	s_set_vgpr_msb 0x4152
	v_pk_fma_f32 v[182:183] /*v[438:439]*/, v[18:19] /*v[530:531]*/, s[28:29], v[180:181] /*v[436:437]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5240
	v_exp_f32_e32 v103 /*v359*/, v194
	v_exp_f32_e32 v121 /*v377*/, v195
	v_nop
	s_set_vgpr_msb 0x4011
	v_pk_fma_f32 v[194:195], v[130:131] /*v[386:387]*/, s[28:29], v[252:253] /*v[508:509]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1109
	v_wmma_f32_16x16x32_bf16 v[144:151], v[40:47] /*v[296:303]*/, v[88:95] /*v[600:607]*/, v[144:151]
	s_set_vgpr_msb 0x952
	v_pk_fma_f32 v[254:255] /*v[510:511]*/, v[20:21] /*v[532:533]*/, s[28:29], v[180:181] /*v[436:437]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[180:181] /*v[436:437]*/, v[22:23] /*v[534:535]*/, s[28:29], v[180:181] /*v[436:437]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5280
	v_exp_f32_e32 v23 /*v535*/, v192
	v_exp_f32_e32 v187 /*v699*/, v193
	v_nop
	s_set_vgpr_msb 0x8000
	v_pk_add_f32 v[192:193], v[220:221], v[196:197]
	s_set_vgpr_msb 2
	v_pk_add_f32 v[196:197], v[130:131] /*v[642:643]*/, v[198:199]
	s_set_vgpr_msb 0x20a
	v_pk_add_f32 v[198:199], v[146:147] /*v[658:659]*/, v[148:149] /*v[660:661]*/
	s_set_vgpr_msb 0xa09
	v_wmma_f32_16x16x32_bf16 v[48:55], v[40:47] /*v[296:303]*/, v[112:119] /*v[624:631]*/, v[48:55]
	s_set_vgpr_msb 0x90a
	v_pk_add_f32 v[200:201], v[158:159] /*v[670:671]*/, v[162:163] /*v[674:675]*/
	v_pk_add_f32 v[202:203], v[170:171] /*v[682:683]*/, v[174:175] /*v[686:687]*/
	v_pk_add_f32 v[212:213], v[140:141] /*v[652:653]*/, v[142:143] /*v[654:655]*/
	v_pk_add_f32 v[214:215], v[150:151] /*v[662:663]*/, v[156:157] /*v[668:669]*/
	s_set_vgpr_msb 0xa05
	v_pk_add_f32 v[242:243], v[14:15] /*v[270:271]*/, v[24:25] /*v[280:281]*/
	s_set_vgpr_msb 0x509
	v_pk_add_f32 v[244:245], v[30:31] /*v[286:287]*/, v[152:153] /*v[664:665]*/
	s_set_vgpr_msb 0x900
	v_exp_f32_e32 v232, v232
	s_set_vgpr_msb 0x41
	v_exp_f32_e32 v98 /*v354*/, v98 /*v354*/
	v_exp_f32_e32 v102 /*v358*/, v102 /*v358*/
	v_exp_f32_e32 v122 /*v378*/, v122 /*v378*/
	s_set_vgpr_msb 0x4181
	v_exp_f32_e32 v18 /*v530*/, v182 /*v438*/
	s_set_vgpr_msb 0x8180
	v_exp_f32_e32 v19 /*v531*/, v194
	s_set_vgpr_msb 0x8002
	v_pk_add_f32 v[204:205], v[182:183] /*v[694:695]*/, v[224:225]
	s_set_vgpr_msb 0x200
	v_pk_add_f32 v[206:207], v[228:229], v[230:231]
	s_set_vgpr_msb 2
	v_pk_add_f32 v[198:199], v[154:155] /*v[666:667]*/, v[198:199]
	v_pk_add_f32 v[200:201], v[166:167] /*v[678:679]*/, v[200:201]
	v_pk_add_f32 v[202:203], v[178:179] /*v[690:691]*/, v[202:203]
	s_set_vgpr_msb 0x204
	v_pk_add_f32 v[208:209], v[234:235], v[8:9] /*v[264:265]*/
	v_pk_add_f32 v[240:241], v[238:239], v[10:11] /*v[266:267]*/
	s_set_vgpr_msb 0x402
	v_pk_add_f32 v[212:213], v[144:145] /*v[656:657]*/, v[212:213]
	s_set_vgpr_msb 0x200
	v_pk_add_f32 v[214:215], v[236:237], v[214:215]
	s_set_vgpr_msb 10
	v_pk_add_f32 v[246:247], v[164:165] /*v[676:677]*/, v[168:169] /*v[680:681]*/
	v_pk_add_f32 v[248:249], v[176:177] /*v[688:689]*/, v[180:181] /*v[692:693]*/
	s_set_vgpr_msb 0xa05
	v_pk_add_f32 v[250:251], v[26:27] /*v[282:283]*/, v[96:97] /*v[352:353]*/
	s_set_vgpr_msb 0x501
	v_pk_add_f32 v[242:243], v[28:29] /*v[284:285]*/, v[242:243]
	s_set_vgpr_msb 0x102
	v_pk_add_f32 v[244:245], v[160:161] /*v[672:673]*/, v[244:245]
	s_set_vgpr_msb 0x200
	v_pk_add_f32 v[192:193], v[192:193], v[196:197]
	s_set_vgpr_msb 0x41
	v_exp_f32_e32 v126 /*v382*/, v126 /*v382*/
	s_set_vgpr_msb 0x4181
	v_exp_f32_e32 v20 /*v532*/, v183 /*v439*/
	v_exp_f32_e32 v22 /*v534*/, v254 /*v510*/
	v_exp_f32_e32 v186 /*v698*/, v255 /*v511*/
	s_set_vgpr_msb 0x8180
	v_exp_f32_e32 v21 /*v533*/, v195
	v_nop
	s_set_vgpr_msb 0x8011
	v_pk_fma_f32 v[194:195], v[134:135] /*v[390:391]*/, s[28:29], v[252:253] /*v[508:509]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1109
	s_wait_dscnt 0x1e
	v_wmma_f32_16x16x32_bf16 v[144:151], v[136:143] /*v[392:399]*/, v[80:87] /*v[592:599]*/, v[144:151]
	s_set_vgpr_msb 0x900
	v_pk_add_f32 v[204:205], v[226:227], v[204:205]
	v_pk_add_f32 v[206:207], v[232:233], v[206:207]
	s_set_vgpr_msb 10
	v_pk_add_f32 v[210:211], v[134:135] /*v[646:647]*/, v[136:137] /*v[648:649]*/
	s_set_vgpr_msb 0xa02
	v_pk_add_f32 v[208:209], v[132:133] /*v[644:645]*/, v[208:209]
	s_set_vgpr_msb 0x201
	v_pk_add_f32 v[240:241], v[12:13] /*v[268:269]*/, v[240:241]
	s_set_vgpr_msb 0x102
	v_pk_add_f32 v[246:247], v[172:173] /*v[684:685]*/, v[246:247]
	v_pk_add_f32 v[248:249], v[184:185] /*v[696:697]*/, v[248:249]
	s_set_vgpr_msb 0x209
	v_wmma_f32_16x16x32_bf16 v[48:55], v[136:143] /*v[392:399]*/, v[104:111] /*v[616:623]*/, v[48:55]
	s_set_vgpr_msb 0x901
	v_pk_add_f32 v[250:251], v[98:99] /*v[354:355]*/, v[250:251]
	s_set_vgpr_msb 0x105
	v_pk_add_f32 v[252:253], v[100:101] /*v[356:357]*/, v[102:103] /*v[358:359]*/
	v_pk_add_f32 v[254:255], v[122:123] /*v[378:379]*/, v[124:125] /*v[380:381]*/
	s_set_vgpr_msb 0x54a
	v_pk_add_f32 v[0:1] /*v[256:257]*/, v[16:17] /*v[528:529]*/, v[18:19] /*v[530:531]*/
	s_set_vgpr_msb 0x4a00
	v_pk_add_f32 v[192:193], v[198:199], v[192:193]
	v_pk_add_f32 v[198:199], v[200:201], v[202:203]
	v_pk_add_f32 v[200:201], v[212:213], v[214:215]
	s_set_vgpr_msb 9
	v_wmma_f32_16x16x32_bf16 v[152:159], v[48:55] /*v[304:311]*/, v[80:87] /*v[592:599]*/, v[152:159]
	s_set_vgpr_msb 0x900
	v_pk_add_f32 v[202:203], v[242:243], v[244:245]
	s_set_vgpr_msb 0x81
	v_exp_f32_e32 v188 /*v700*/, v180 /*v436*/
	s_set_vgpr_msb 0x8180
	v_exp_f32_e32 v189 /*v701*/, v194
	s_set_vgpr_msb 0x8002
	v_pk_add_f32 v[210:211], v[138:139] /*v[650:651]*/, v[210:211]
	s_set_vgpr_msb 0x24a
	v_pk_add_f32 v[2:3] /*v[258:259]*/, v[22:23] /*v[534:535]*/, v[186:187] /*v[698:699]*/
	s_set_vgpr_msb 0x4a01
	v_pk_add_f32 v[196:197], v[120:121] /*v[376:377]*/, v[252:253]
	v_pk_add_f32 v[252:253], v[126:127] /*v[382:383]*/, v[254:255]
	s_set_vgpr_msb 0x109
	v_wmma_f32_16x16x32_bf16 v[56:63], v[48:55] /*v[304:311]*/, v[104:111] /*v[616:623]*/, v[56:63]
	v_pk_add_f32 v[254:255], v[0:1] /*v[256:257]*/, v[20:21] /*v[532:533]*/
	s_set_vgpr_msb 0x900
	v_pk_add_f32 v[206:207], v[206:207], v[208:209]
	v_pk_add_f32 v[208:209], v[248:249], v[250:251]
	v_pk_add_f32 v[198:199], v[204:205], v[198:199]
	v_pk_add_f32 v[200:201], v[240:241], v[200:201]
	v_pk_add_f32 v[202:203], v[246:247], v[202:203]
	s_set_vgpr_msb 0x46
	v_pk_add_f32 v[0:1] /*v[256:257]*/, v[188:189] /*v[700:701]*/, v[2:3] /*v[258:259]*/
	s_set_vgpr_msb 0x4609
	s_wait_dscnt 0x1a
	v_wmma_f32_16x16x32_bf16 v[144:151], v[144:151] /*v[400:407]*/, v[72:79] /*v[584:591]*/, v[144:151]
	s_set_vgpr_msb 0x900
	v_pk_add_f32 v[204:205], v[210:211], v[206:207]
	v_pk_add_f32 v[196:197], v[196:197], v[208:209]
	v_pk_add_f32 v[206:207], v[252:253], v[254:255]
	v_pk_add_f32 v[192:193], v[192:193], v[198:199]
	v_pk_add_f32 v[198:199], v[200:201], v[202:203]
	s_set_vgpr_msb 1
	v_exp_f32_e32 v214, v181 /*v437*/
	s_set_vgpr_msb 0x100
	v_exp_f32_e32 v215, v195
	s_set_vgpr_msb 9
	v_wmma_f32_16x16x32_bf16 v[48:55], v[144:151] /*v[400:407]*/, v[96:103] /*v[608:615]*/, v[48:55]
	s_set_vgpr_msb 0x901
	v_pk_add_f32 v[206:207], v[0:1] /*v[256:257]*/, v[206:207]
	s_set_vgpr_msb 0x100
	v_pk_add_f32 v[204:205], v[204:205], v[192:193]
	v_pk_add_f32 v[196:197], v[196:197], v[198:199]
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x42
	ds_load_tr16_b128 v[144:147] /*v[400:403]*/, v204 /*v716*/ offset:128
	ds_load_tr16_b128 v[136:139] /*v[392:395]*/, v204 /*v716*/ offset:160
	ds_load_tr16_b128 v[148:151] /*v[404:407]*/, v204 /*v716*/ offset:4736
	ds_load_tr16_b128 v[140:143] /*v[396:399]*/, v204 /*v716*/ offset:4768
	ds_load_tr16_b128 v[252:255] /*v[508:511]*/, v204 /*v716*/ offset:23232
	ds_load_tr16_b128 v[180:183] /*v[436:439]*/, v204 /*v716*/ offset:23264
	s_set_vgpr_msb 0x4202
	ds_load_tr16_b128 v[200:203], v204 /*v716*/ offset:27840
	ds_load_tr16_b128 v[192:195], v204 /*v716*/ offset:27872
	s_set_vgpr_msb 0x242
	ds_load_tr16_b128 v[248:251] /*v[504:507]*/, v204 /*v716*/ offset:18624
	ds_load_tr16_b128 v[176:179] /*v[432:435]*/, v204 /*v716*/ offset:18656
	s_set_vgpr_msb 0x4200
	v_pk_add_f32 v[208:209], v[214:215], v[206:207]
	s_set_vgpr_msb 9
	v_wmma_f32_16x16x32_bf16 v[152:159], v[56:63] /*v[312:319]*/, v[72:79] /*v[584:591]*/, v[152:159]
	s_set_vgpr_msb 0x900
	v_pk_add_f32 v[210:211], v[204:205], v[196:197]
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 10
	ds_load_tr16_b128 v[204:207], v204 /*v716*/ offset:32448
	ds_load_tr16_b128 v[196:199], v204 /*v716*/ offset:32480
	v_sub_f32_e32 v212, v206 /*v718*/, v198 /*v710*/
	s_set_vgpr_msb 0xa00
	v_pk_add_f32 v[208:209], v[208:209], v[210:211]
	v_mul_f32_e32 v212, 0x3fb8aa3b, v212
	s_set_vgpr_msb 9
	v_wmma_f32_16x16x32_bf16 v[56:63], v[56:63] /*v[312:319]*/, v[96:103] /*v[608:615]*/, v[56:63]
	s_set_vgpr_msb 0x942
	ds_load_tr16_b128 v[104:107] /*v[360:363]*/, v204 /*v716*/ offset:9216
	ds_load_tr16_b128 v[72:75] /*v[328:331]*/, v204 /*v716*/ offset:9248
	ds_load_tr16_b128 v[108:111] /*v[364:367]*/, v204 /*v716*/ offset:13824
	ds_load_tr16_b128 v[76:79] /*v[332:335]*/, v204 /*v716*/ offset:13856
	ds_load_tr16_b128 v[88:91] /*v[344:347]*/, v204 /*v716*/ offset:18432
	s_wait_alu depctr_va_vdst(0)
	ds_load_tr16_b128 v[56:59] /*v[312:315]*/, v204 /*v716*/ offset:18464
	ds_load_tr16_b128 v[92:95] /*v[348:351]*/, v204 /*v716*/ offset:23040
	ds_load_tr16_b128 v[60:63] /*v[316:319]*/, v204 /*v716*/ offset:23072
	s_set_vgpr_msb 0x4200
	v_dual_mov_b32 v210, v208 :: v_dual_mov_b32 v211, v209
	v_exp_f32_e32 v212, v212
	v_permlanex16_b32 v210, v210, s66, 0xfedcba98
	s_set_vgpr_msb 10
	v_wmma_f32_16x16x32_bf16 v[88:95], v[232:239] /*v[744:751]*/, v[88:95] /*v[600:607]*/, v[88:95]
	s_set_vgpr_msb 0xa00
	v_permlanex16_b32 v211, v211, s66, 0xfedcba98
	s_set_vgpr_msb 10
	v_wmma_f32_16x16x32_bf16 v[8:15], v[232:239] /*v[744:751]*/, v[112:119] /*v[624:631]*/, v[8:15]
	v_wmma_f32_16x16x32_bf16 v[80:87], v[240:247] /*v[752:759]*/, v[88:95] /*v[600:607]*/, v[80:87]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[240:247] /*v[752:759]*/, v[112:119] /*v[624:631]*/, v[0:7]
	s_set_vgpr_msb 0xa09
	v_wmma_f32_16x16x32_bf16 v[120:127], v[216:223] /*v[472:479]*/, v[72:79] /*v[584:591]*/, v[120:127]
	v_wmma_f32_16x16x32_bf16 v[32:39], v[216:223] /*v[472:479]*/, v[96:103] /*v[608:615]*/, v[32:39]
	s_set_vgpr_msb 0x982
	ds_load_tr16_b128 v[36:39] /*v[548:551]*/, v204 /*v716*/ offset:23168
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x8242
	ds_load_tr16_b128 v[220:223] /*v[476:479]*/, v204 /*v716*/ offset:23200
	s_set_vgpr_msb 0x4282
	ds_load_tr16_b128 v[40:43] /*v[552:555]*/, v204 /*v716*/ offset:27776
	s_set_vgpr_msb 0x8242
	ds_load_tr16_b128 v[240:243] /*v[496:499]*/, v204 /*v716*/ offset:27808
	s_set_vgpr_msb 0x4282
	ds_load_tr16_b128 v[32:35] /*v[544:547]*/, v204 /*v716*/ offset:18560
	s_set_vgpr_msb 0x8242
	ds_load_tr16_b128 v[216:219] /*v[472:475]*/, v204 /*v716*/ offset:18592
	s_set_vgpr_msb 0x4209
	v_wmma_f32_16x16x32_bf16 v[144:151], v[152:159] /*v[408:415]*/, v[64:71] /*v[576:583]*/, v[144:151]
	v_wmma_f32_16x16x32_bf16 v[48:55], v[152:159] /*v[408:415]*/, v[48:55] /*v[560:567]*/, v[48:55]
	s_set_vgpr_msb 0x982
	ds_load_tr16_b128 v[0:3] /*v[512:515]*/, v204 /*v716*/ offset:18496
	s_set_vgpr_msb 0x8242
	ds_load_tr16_b128 v[184:187] /*v[440:443]*/, v204 /*v716*/ offset:18528
	s_set_vgpr_msb 0x4282
	ds_load_tr16_b128 v[4:7] /*v[516:519]*/, v204 /*v716*/ offset:23104
	s_set_vgpr_msb 0x8242
	ds_load_tr16_b128 v[188:191] /*v[444:447]*/, v204 /*v716*/ offset:23136
	ds_load_tr16_b128 v[200:203] /*v[456:459]*/, v204 /*v716*/ offset:27712
	s_wait_alu depctr_va_vdst(0)
	ds_load_tr16_b128 v[152:155] /*v[408:411]*/, v204 /*v716*/ offset:27744
	ds_load_tr16_b128 v[204:207] /*v[460:463]*/, v204 /*v716*/ offset:32320
	ds_load_tr16_b128 v[156:159] /*v[412:415]*/, v204 /*v716*/ offset:32352
	s_set_vgpr_msb 0x4209
	v_wmma_f32_16x16x32_bf16 v[152:159], v[16:23] /*v[272:279]*/, v[64:71] /*v[576:583]*/, v[152:159]
	v_wmma_f32_16x16x32_bf16 v[56:63], v[16:23] /*v[272:279]*/, v[48:55] /*v[560:567]*/, v[56:63]
	s_set_vgpr_msb 0x942
	ds_load_tr16_b128 v[80:83] /*v[336:339]*/, v204 /*v716*/ offset:27648
	ds_load_tr16_b128 v[48:51] /*v[304:307]*/, v204 /*v716*/ offset:27680
	ds_load_tr16_b128 v[84:87] /*v[340:343]*/, v204 /*v716*/ offset:32256
	ds_load_tr16_b128 v[52:55] /*v[308:311]*/, v204 /*v716*/ offset:32288
	ds_load_tr16_b128 v[40:43] /*v[296:299]*/, v204 /*v716*/ offset:64
	s_wait_alu depctr_va_vdst(0)
	ds_load_tr16_b128 v[16:19] /*v[272:275]*/, v204 /*v716*/ offset:96
	ds_load_tr16_b128 v[44:47] /*v[300:303]*/, v204 /*v716*/ offset:4672
	ds_load_tr16_b128 v[20:23] /*v[276:279]*/, v204 /*v716*/ offset:4704
	s_set_vgpr_msb 0x4282
	ds_load_tr16_b128 v[8:11] /*v[520:523]*/, v204 /*v716*/ offset:9280
	s_set_vgpr_msb 0x8242
	ds_load_tr16_b128 v[192:195] /*v[448:451]*/, v204 /*v716*/ offset:9312
	s_set_vgpr_msb 0x4282
	ds_load_tr16_b128 v[12:15] /*v[524:527]*/, v204 /*v716*/ offset:13888
	s_set_vgpr_msb 0x8242
	ds_load_tr16_b128 v[196:199] /*v[452:455]*/, v204 /*v716*/ offset:13920
	s_set_vgpr_msb 0x420a
	v_wmma_f32_16x16x32_bf16 v[96:103], v[208:215] /*v[720:727]*/, v[80:87] /*v[592:599]*/, v[96:103]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[208:215] /*v[720:727]*/, v[104:111] /*v[616:623]*/, v[16:23]
	s_wait_dscnt 0x3e
	v_wmma_f32_16x16x32_bf16 v[88:95], v[248:255] /*v[760:767]*/, v[80:87] /*v[592:599]*/, v[88:95]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[248:255] /*v[760:767]*/, v[104:111] /*v[616:623]*/, v[8:15]
	s_set_vgpr_msb 0xa0b
	v_wmma_f32_16x16x32_bf16 v[80:87], v[0:7] /*v[768:775]*/, v[80:87] /*v[592:599]*/, v[80:87]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[0:7] /*v[768:775]*/, v[104:111] /*v[616:623]*/, v[0:7]
	s_set_vgpr_msb 0xb09
	v_wmma_f32_16x16x32_bf16 v[120:127], v[232:239] /*v[488:495]*/, v[64:71] /*v[576:583]*/, v[120:127]
	v_wmma_f32_16x16x32_bf16 v[32:39], v[232:239] /*v[488:495]*/, v[48:55] /*v[560:567]*/, v[32:39]
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x942
	ds_load_tr16_b128 v[232:235] /*v[488:491]*/, v204 /*v716*/ offset:9408
	ds_load_tr16_b128 v[168:171] /*v[424:427]*/, v204 /*v716*/ offset:9440
	ds_load_tr16_b128 v[236:239] /*v[492:495]*/, v204 /*v716*/ offset:14016
	ds_load_tr16_b128 v[172:175] /*v[428:431]*/, v204 /*v716*/ offset:14048
	s_wait_dscnt 0x0
	s_wait_tensorcnt 0x0
	s_set_vgpr_msb 0x420a
	s_barrier_signal -1
	v_wmma_f32_16x16x32_bf16 v[96:103], v[216:223] /*v[728:735]*/, v[72:79] /*v[584:591]*/, v[96:103]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[216:223] /*v[728:735]*/, v[96:103] /*v[608:615]*/, v[16:23]
	s_set_vgpr_msb 0xa0b
	v_wmma_f32_16x16x32_bf16 v[88:95], v[8:15] /*v[776:783]*/, v[72:79] /*v[584:591]*/, v[88:95]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[8:15] /*v[776:783]*/, v[96:103] /*v[608:615]*/, v[8:15]
	v_wmma_f32_16x16x32_bf16 v[80:87], v[16:23] /*v[784:791]*/, v[72:79] /*v[584:591]*/, v[80:87]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[16:23] /*v[784:791]*/, v[96:103] /*v[608:615]*/, v[0:7]
	s_set_vgpr_msb 0xb0a
	v_wmma_f32_16x16x32_bf16 v[96:103], v[224:231] /*v[736:743]*/, v[64:71] /*v[576:583]*/, v[96:103]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[224:231] /*v[736:743]*/, v[48:55] /*v[560:567]*/, v[16:23]
	s_set_vgpr_msb 0xa0b
	v_wmma_f32_16x16x32_bf16 v[88:95], v[24:31] /*v[792:799]*/, v[64:71] /*v[576:583]*/, v[88:95]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[24:31] /*v[792:799]*/, v[48:55] /*v[560:567]*/, v[8:15]
	s_set_vgpr_msb 0xb0a
	v_wmma_f32_16x16x32_bf16 v[80:87], v[56:63] /*v[568:575]*/, v[64:71] /*v[576:583]*/, v[80:87]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[56:63] /*v[568:575]*/, v[48:55] /*v[560:567]*/, v[0:7]
	s_set_vgpr_msb 0xa00
	s_cbranch_vccz .LBB0_15
	v_pk_mul_f32 v[158:159], v[158:159], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[156:157], v[156:157], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[154:155], v[154:155], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[152:153], v[152:153], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[150:151], v[150:151], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[148:149], v[148:149], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[146:147], v[146:147], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[144:145], v[144:145], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[142:143], v[142:143], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[140:141], v[140:141], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[138:139], v[138:139], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[136:137], v[136:137], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[126:127], v[126:127], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[124:125], v[124:125], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[122:123], v[122:123], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[120:121], v[120:121], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[118:119], v[118:119], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[116:117], v[116:117], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[114:115], v[114:115], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[112:113], v[112:113], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[102:103], v[102:103], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[100:101], v[100:101], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[98:99], v[98:99], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[96:97], v[96:97], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[94:95], v[94:95], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[92:93], v[92:93], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[90:91], v[90:91], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[88:89], v[88:89], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[86:87], v[86:87], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[84:85], v[84:85], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[82:83], v[82:83], v[212:213] op_sel_hi:[1,0]
	v_pk_mul_f32 v[80:81], v[80:81], v[212:213] op_sel_hi:[1,0]
.LBB0_15:
	s_set_vgpr_msb 10
	v_sub_f32_e32 v213, v205 /*v717*/, v203 /*v715*/
	s_and_b32 s2, s3, exec_lo
	s_cselect_b32 s2, 1, 0
	s_cmp_lg_u32 s2, 1
	s_set_vgpr_msb 0xa00
	v_mul_f32_e32 v213, 0x3fb8aa3b, v213
	v_exp_f32_e32 v213, v213
	s_cbranch_scc1 .LBB0_17
	v_nop
	v_mov_b32_e32 v240, v213
	v_pk_mul_f32 v[62:63], v[62:63], v[240:241] op_sel_hi:[1,0]
	v_pk_mul_f32 v[60:61], v[60:61], v[240:241] op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59], v[58:59], v[240:241] op_sel_hi:[1,0]
	v_pk_mul_f32 v[56:57], v[56:57], v[240:241] op_sel_hi:[1,0]
	v_pk_mul_f32 v[54:55], v[54:55], v[240:241] op_sel_hi:[1,0]
	v_pk_mul_f32 v[52:53], v[52:53], v[240:241] op_sel_hi:[1,0]
	v_pk_mul_f32 v[50:51], v[50:51], v[240:241] op_sel_hi:[1,0]
	v_pk_mul_f32 v[48:49], v[48:49], v[240:241] op_sel_hi:[1,0]
	v_pk_mul_f32 v[46:47], v[46:47], v[240:241] op_sel_hi:[1,0]
	v_pk_mul_f32 v[44:45], v[44:45], v[240:241] op_sel_hi:[1,0]
	v_pk_mul_f32 v[42:43], v[42:43], v[240:241] op_sel_hi:[1,0]
	v_pk_mul_f32 v[40:41], v[40:41], v[240:241] op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39], v[38:39], v[240:241] op_sel_hi:[1,0]
	v_pk_mul_f32 v[36:37], v[36:37], v[240:241] op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35], v[34:35], v[240:241] op_sel_hi:[1,0]
	v_pk_mul_f32 v[32:33], v[32:33], v[240:241] op_sel_hi:[1,0]
	v_pk_mul_f32 v[30:31], v[30:31], v[240:241] op_sel_hi:[1,0]
	v_pk_mul_f32 v[28:29], v[28:29], v[240:241] op_sel_hi:[1,0]
	v_pk_mul_f32 v[26:27], v[26:27], v[240:241] op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25], v[24:25], v[240:241] op_sel_hi:[1,0]
	v_pk_mul_f32 v[22:23], v[22:23], v[240:241] op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21], v[20:21], v[240:241] op_sel_hi:[1,0]
	v_pk_mul_f32 v[18:19], v[18:19], v[240:241] op_sel_hi:[1,0]
	v_pk_mul_f32 v[16:17], v[16:17], v[240:241] op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[14:15], v[240:241] op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13], v[12:13], v[240:241] op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[10:11], v[240:241] op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[8:9], v[240:241] op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[6:7], v[240:241] op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[4:5], v[240:241] op_sel_hi:[1,0]
	v_pk_mul_f32 v[2:3], v[2:3], v[240:241] op_sel_hi:[1,0]
	v_pk_mul_f32 v[0:1], v[0:1], v[240:241] op_sel_hi:[1,0]
.LBB0_17:
	s_set_vgpr_msb 10
	v_cvt_pk_bf16_f32 v247, v178 /*v690*/, v182 /*v694*/
	v_cvt_pk_bf16_f32 v246, v170 /*v682*/, v174 /*v686*/
	v_cvt_pk_bf16_f32 v245, v162 /*v674*/, v166 /*v678*/
	v_cvt_pk_bf16_f32 v244, v154 /*v666*/, v158 /*v670*/
	v_cvt_pk_bf16_f32 v243, v146 /*v658*/, v148 /*v660*/
	v_cvt_pk_bf16_f32 v242, v128 /*v640*/, v130 /*v642*/
	s_set_vgpr_msb 0xa00
	v_cvt_pk_bf16_f32 v241, v220, v222
	v_cvt_pk_bf16_f32 v240, v216, v218
	s_set_vgpr_msb 0x4a
	v_cvt_pk_bf16_f32 v7 /*v263*/, v179 /*v691*/, v183 /*v695*/
	v_cvt_pk_bf16_f32 v6 /*v262*/, v171 /*v683*/, v175 /*v687*/
	v_cvt_pk_bf16_f32 v5 /*v261*/, v163 /*v675*/, v167 /*v679*/
	v_cvt_pk_bf16_f32 v4 /*v260*/, v155 /*v667*/, v159 /*v671*/
	v_cvt_pk_bf16_f32 v3 /*v259*/, v147 /*v659*/, v149 /*v661*/
	v_cvt_pk_bf16_f32 v2 /*v258*/, v129 /*v641*/, v131 /*v643*/
	s_set_vgpr_msb 0x4a40
	v_cvt_pk_bf16_f32 v1 /*v257*/, v221, v223
	v_cvt_pk_bf16_f32 v0 /*v256*/, v217, v219
	s_set_vgpr_msb 0x4001
	v_wmma_f32_16x16x32_bf16 v[152:159], v[112:119] /*v[368:375]*/, v[240:247], v[152:159]
	s_set_vgpr_msb 0x10a
	v_cvt_pk_bf16_f32 v255, v150 /*v662*/, v156 /*v668*/
	v_cvt_pk_bf16_f32 v254, v142 /*v654*/, v144 /*v656*/
	v_cvt_pk_bf16_f32 v253, v138 /*v650*/, v140 /*v652*/
	v_cvt_pk_bf16_f32 v252, v134 /*v646*/, v136 /*v648*/
	s_set_vgpr_msb 0xa09
	v_cvt_pk_bf16_f32 v251, v8 /*v264*/, v132 /*v644*/
	s_set_vgpr_msb 0x900
	v_cvt_pk_bf16_f32 v250, v232, v234
	v_cvt_pk_bf16_f32 v249, v228, v230
	s_set_vgpr_msb 5
	v_wmma_f32_16x16x32_bf16 v[56:63], v[112:119] /*v[368:375]*/, v[0:7] /*v[256:263]*/, v[56:63]
	s_set_vgpr_msb 0x500
	v_cvt_pk_bf16_f32 v248, v224, v226
	s_set_vgpr_msb 0x4a
	v_cvt_pk_bf16_f32 v39 /*v295*/, v151 /*v663*/, v157 /*v669*/
	v_cvt_pk_bf16_f32 v38 /*v294*/, v143 /*v655*/, v145 /*v657*/
	v_cvt_pk_bf16_f32 v37 /*v293*/, v139 /*v651*/, v141 /*v653*/
	v_cvt_pk_bf16_f32 v36 /*v292*/, v135 /*v647*/, v137 /*v649*/
	s_set_vgpr_msb 0x4a49
	v_cvt_pk_bf16_f32 v35 /*v291*/, v9 /*v265*/, v133 /*v645*/
	s_set_vgpr_msb 0x4940
	v_cvt_pk_bf16_f32 v34 /*v290*/, v233, v235
	s_set_vgpr_msb 0x4001
	v_wmma_f32_16x16x32_bf16 v[144:151], v[64:71] /*v[320:327]*/, v[240:247], v[144:151]
	s_set_vgpr_msb 0x140
	v_cvt_pk_bf16_f32 v33 /*v289*/, v229, v231
	v_cvt_pk_bf16_f32 v32 /*v288*/, v225, v227
	s_set_vgpr_msb 0x400a
	v_cvt_pk_bf16_f32 v223, v180 /*v692*/, v184 /*v696*/
	v_cvt_pk_bf16_f32 v222, v172 /*v684*/, v176 /*v688*/
	v_cvt_pk_bf16_f32 v221, v164 /*v676*/, v168 /*v680*/
	v_cvt_pk_bf16_f32 v220, v152 /*v664*/, v160 /*v672*/
	s_set_vgpr_msb 0xa05
	v_cvt_pk_bf16_f32 v219, v28 /*v284*/, v30 /*v286*/
	v_wmma_f32_16x16x32_bf16 v[48:55], v[64:71] /*v[320:327]*/, v[0:7] /*v[256:263]*/, v[48:55]
	v_cvt_pk_bf16_f32 v218, v14 /*v270*/, v24 /*v280*/
	v_cvt_pk_bf16_f32 v217, v10 /*v266*/, v12 /*v268*/
	s_set_vgpr_msb 0x500
	v_cvt_pk_bf16_f32 v216, v236, v238
	s_set_vgpr_msb 2
	v_cvt_pk_bf16_f32 v231, v188 /*v700*/, v214
	s_set_vgpr_msb 0x20a
	v_cvt_pk_bf16_f32 v230, v22 /*v534*/, v186 /*v698*/
	v_cvt_pk_bf16_f32 v229, v18 /*v530*/, v20 /*v532*/
	s_set_vgpr_msb 0xa09
	v_cvt_pk_bf16_f32 v228, v126 /*v382*/, v16 /*v528*/
	s_set_vgpr_msb 0x901
	v_wmma_f32_16x16x32_bf16 v[136:143], v[40:47] /*v[296:303]*/, v[240:247], v[136:143]
	s_set_vgpr_msb 0x105
	v_cvt_pk_bf16_f32 v227, v122 /*v378*/, v124 /*v380*/
	v_cvt_pk_bf16_f32 v226, v102 /*v358*/, v120 /*v376*/
	v_cvt_pk_bf16_f32 v225, v98 /*v354*/, v100 /*v356*/
	v_cvt_pk_bf16_f32 v224, v26 /*v282*/, v96 /*v352*/
	s_set_vgpr_msb 0x50a
	v_cvt_pk_bf16_f32 v238, v23 /*v535*/, v187 /*v699*/
	s_set_vgpr_msb 0xa09
	v_cvt_pk_bf16_f32 v236, v127 /*v383*/, v17 /*v529*/
	s_set_vgpr_msb 0x905
	v_cvt_pk_bf16_f32 v235, v123 /*v379*/, v125 /*v381*/
	v_wmma_f32_16x16x32_bf16 v[40:47], v[40:47] /*v[296:303]*/, v[0:7] /*v[256:263]*/, v[40:47]
	v_cvt_pk_bf16_f32 v234, v103 /*v359*/, v121 /*v377*/
	v_cvt_pk_bf16_f32 v233, v99 /*v355*/, v101 /*v357*/
	v_cvt_pk_bf16_f32 v232, v27 /*v283*/, v97 /*v353*/
	s_add_co_i32 s2, s42, 4
	s_barrier_wait -1
	s_cmp_ge_i32 s2, s44
	s_set_vgpr_msb 0x501
	v_wmma_f32_16x16x32_bf16 v[120:127], v[16:23] /*v[272:279]*/, v[240:247], v[120:127]
	s_set_vgpr_msb 0x105
	v_wmma_f32_16x16x32_bf16 v[32:39], v[16:23] /*v[272:279]*/, v[0:7] /*v[256:263]*/, v[32:39]
	s_set_vgpr_msb 0x501
	v_wmma_f32_16x16x32_bf16 v[112:119], v[144:151] /*v[400:407]*/, v[240:247], v[112:119]
	s_set_vgpr_msb 0x105
	v_wmma_f32_16x16x32_bf16 v[24:31], v[144:151] /*v[400:407]*/, v[0:7] /*v[256:263]*/, v[24:31]
	s_set_vgpr_msb 0x501
	v_wmma_f32_16x16x32_bf16 v[96:103], v[136:143] /*v[392:399]*/, v[240:247], v[96:103]
	s_set_vgpr_msb 0x105
	v_wmma_f32_16x16x32_bf16 v[16:23], v[136:143] /*v[392:399]*/, v[0:7] /*v[256:263]*/, v[16:23]
	s_set_vgpr_msb 0x501
	v_wmma_f32_16x16x32_bf16 v[88:95], v[224:231] /*v[480:487]*/, v[240:247], v[88:95]
	s_set_vgpr_msb 0x105
	v_wmma_f32_16x16x32_bf16 v[8:15], v[224:231] /*v[480:487]*/, v[0:7] /*v[256:263]*/, v[8:15]
	s_set_vgpr_msb 0x501
	v_wmma_f32_16x16x32_bf16 v[80:87], v[160:167] /*v[416:423]*/, v[240:247], v[80:87]
	s_set_vgpr_msb 0x105
	v_wmma_f32_16x16x32_bf16 v[0:7], v[160:167] /*v[416:423]*/, v[0:7] /*v[256:263]*/, v[0:7]
	s_set_vgpr_msb 0x501
	v_wmma_f32_16x16x32_bf16 v[152:159], v[104:111] /*v[360:367]*/, v[248:255], v[152:159]
	s_set_vgpr_msb 0x105
	v_wmma_f32_16x16x32_bf16 v[56:63], v[104:111] /*v[360:367]*/, v[32:39] /*v[288:295]*/, v[56:63]
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x54a
	v_cvt_pk_bf16_f32 v111 /*v367*/, v181 /*v693*/, v185 /*v697*/
	v_cvt_pk_bf16_f32 v110 /*v366*/, v173 /*v685*/, v177 /*v689*/
	v_cvt_pk_bf16_f32 v109 /*v365*/, v165 /*v677*/, v169 /*v681*/
	s_set_vgpr_msb 0x4a01
	v_wmma_f32_16x16x32_bf16 v[144:151], v[72:79] /*v[328:335]*/, v[248:255], v[144:151]
	s_set_vgpr_msb 0x14a
	v_cvt_pk_bf16_f32 v108 /*v364*/, v153 /*v665*/, v161 /*v673*/
	s_set_vgpr_msb 0x4a45
	v_cvt_pk_bf16_f32 v107 /*v363*/, v29 /*v285*/, v31 /*v287*/
	v_cvt_pk_bf16_f32 v106 /*v362*/, v15 /*v271*/, v25 /*v281*/
	v_cvt_pk_bf16_f32 v105 /*v361*/, v11 /*v267*/, v13 /*v269*/
	s_set_vgpr_msb 0x4540
	v_cvt_pk_bf16_f32 v104 /*v360*/, v237, v239
	s_set_vgpr_msb 0x4002
	v_cvt_pk_bf16_f32 v239, v189 /*v701*/, v215
	s_set_vgpr_msb 0x20a
	v_cvt_pk_bf16_f32 v237, v19 /*v531*/, v21 /*v533*/
	s_set_vgpr_msb 0xa05
	v_wmma_f32_16x16x32_bf16 v[48:55], v[72:79] /*v[328:335]*/, v[32:39] /*v[288:295]*/, v[48:55]
	s_set_vgpr_msb 0x502
	v_wmma_f32_16x16x32_bf16 v[136:143], v[8:15] /*v[520:527]*/, v[248:255], v[136:143]
	s_set_vgpr_msb 0x206
	v_wmma_f32_16x16x32_bf16 v[40:47], v[8:15] /*v[520:527]*/, v[32:39] /*v[288:295]*/, v[40:47]
	s_set_vgpr_msb 0x601
	v_wmma_f32_16x16x32_bf16 v[120:127], v[192:199] /*v[448:455]*/, v[248:255], v[120:127]
	s_set_vgpr_msb 0x105
	v_wmma_f32_16x16x32_bf16 v[32:39], v[192:199] /*v[448:455]*/, v[32:39] /*v[288:295]*/, v[32:39]
	s_set_vgpr_msb 0x502
	v_wmma_f32_16x16x32_bf16 v[112:119], v[24:31] /*v[536:543]*/, v[248:255], v[112:119]
	s_set_vgpr_msb 0x206
	v_wmma_f32_16x16x32_bf16 v[24:31], v[24:31] /*v[536:543]*/, v[32:39] /*v[288:295]*/, v[24:31]
	s_set_vgpr_msb 0x601
	v_wmma_f32_16x16x32_bf16 v[96:103], v[208:215] /*v[464:471]*/, v[248:255], v[96:103]
	s_set_vgpr_msb 0x105
	v_wmma_f32_16x16x32_bf16 v[16:23], v[208:215] /*v[464:471]*/, v[32:39] /*v[288:295]*/, v[16:23]
	s_set_vgpr_msb 0x501
	v_wmma_f32_16x16x32_bf16 v[88:95], v[232:239] /*v[488:495]*/, v[248:255], v[88:95]
	s_set_vgpr_msb 0x105
	v_wmma_f32_16x16x32_bf16 v[8:15], v[232:239] /*v[488:495]*/, v[32:39] /*v[288:295]*/, v[8:15]
	s_set_vgpr_msb 0x501
	v_wmma_f32_16x16x32_bf16 v[80:87], v[168:175] /*v[424:431]*/, v[248:255], v[80:87]
	s_set_vgpr_msb 0x105
	v_wmma_f32_16x16x32_bf16 v[0:7], v[168:175] /*v[424:431]*/, v[32:39] /*v[288:295]*/, v[0:7]
	s_set_vgpr_msb 0x501
	v_wmma_f32_16x16x32_bf16 v[152:159], v[88:95] /*v[344:351]*/, v[216:223], v[152:159]
	s_set_vgpr_msb 0x105
	v_wmma_f32_16x16x32_bf16 v[56:63], v[88:95] /*v[344:351]*/, v[104:111] /*v[360:367]*/, v[56:63]
	s_set_vgpr_msb 0x501
	v_wmma_f32_16x16x32_bf16 v[144:151], v[56:63] /*v[312:319]*/, v[216:223], v[144:151]
	s_set_vgpr_msb 0x105
	v_wmma_f32_16x16x32_bf16 v[48:55], v[56:63] /*v[312:319]*/, v[104:111] /*v[360:367]*/, v[48:55]
	s_set_vgpr_msb 0x502
	v_wmma_f32_16x16x32_bf16 v[136:143], v[0:7] /*v[512:519]*/, v[216:223], v[136:143]
	s_set_vgpr_msb 0x206
	v_wmma_f32_16x16x32_bf16 v[40:47], v[0:7] /*v[512:519]*/, v[104:111] /*v[360:367]*/, v[40:47]
	s_set_vgpr_msb 0x601
	v_wmma_f32_16x16x32_bf16 v[120:127], v[184:191] /*v[440:447]*/, v[216:223], v[120:127]
	s_set_vgpr_msb 0x105
	v_wmma_f32_16x16x32_bf16 v[32:39], v[184:191] /*v[440:447]*/, v[104:111] /*v[360:367]*/, v[32:39]
	s_set_vgpr_msb 0x502
	v_wmma_f32_16x16x32_bf16 v[112:119], v[32:39] /*v[544:551]*/, v[216:223], v[112:119]
	s_set_vgpr_msb 0x206
	v_wmma_f32_16x16x32_bf16 v[24:31], v[32:39] /*v[544:551]*/, v[104:111] /*v[360:367]*/, v[24:31]
	s_set_vgpr_msb 0x601
	v_wmma_f32_16x16x32_bf16 v[96:103], v[216:223] /*v[472:479]*/, v[216:223], v[96:103]
	s_set_vgpr_msb 0x105
	v_wmma_f32_16x16x32_bf16 v[16:23], v[216:223] /*v[472:479]*/, v[104:111] /*v[360:367]*/, v[16:23]
	s_set_vgpr_msb 0x501
	v_wmma_f32_16x16x32_bf16 v[88:95], v[248:255] /*v[504:511]*/, v[216:223], v[88:95]
	s_set_vgpr_msb 0x105
	v_wmma_f32_16x16x32_bf16 v[8:15], v[248:255] /*v[504:511]*/, v[104:111] /*v[360:367]*/, v[8:15]
	s_set_vgpr_msb 0x501
	v_wmma_f32_16x16x32_bf16 v[80:87], v[176:183] /*v[432:439]*/, v[216:223], v[80:87]
	s_set_vgpr_msb 0x105
	v_wmma_f32_16x16x32_bf16 v[0:7], v[176:183] /*v[432:439]*/, v[104:111] /*v[360:367]*/, v[0:7]
	s_set_vgpr_msb 0x501
	v_wmma_f32_16x16x32_bf16 v[152:159], v[80:87] /*v[336:343]*/, v[224:231], v[152:159]
	v_wmma_f32_16x16x32_bf16 v[56:63], v[80:87] /*v[336:343]*/, v[232:239], v[56:63]
	v_wmma_f32_16x16x32_bf16 v[144:151], v[48:55] /*v[304:311]*/, v[224:231], v[144:151]
	v_wmma_f32_16x16x32_bf16 v[48:55], v[48:55] /*v[304:311]*/, v[232:239], v[48:55]
	v_wmma_f32_16x16x32_bf16 v[136:143], v[200:207] /*v[456:463]*/, v[224:231], v[136:143]
	v_wmma_f32_16x16x32_bf16 v[40:47], v[200:207] /*v[456:463]*/, v[232:239], v[40:47]
	v_wmma_f32_16x16x32_bf16 v[120:127], v[152:159] /*v[408:415]*/, v[224:231], v[120:127]
	v_wmma_f32_16x16x32_bf16 v[32:39], v[152:159] /*v[408:415]*/, v[232:239], v[32:39]
	s_set_vgpr_msb 0x102
	v_wmma_f32_16x16x32_bf16 v[112:119], v[40:47] /*v[552:559]*/, v[224:231], v[112:119]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[40:47] /*v[552:559]*/, v[232:239], v[24:31]
	s_set_vgpr_msb 0x201
	v_wmma_f32_16x16x32_bf16 v[96:103], v[240:247] /*v[496:503]*/, v[224:231], v[96:103]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[240:247] /*v[496:503]*/, v[232:239], v[16:23]
	s_set_vgpr_msb 0x100
	v_wmma_f32_16x16x32_bf16 v[88:95], v[200:207], v[224:231], v[88:95]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[200:207], v[232:239], v[8:15]
	v_wmma_f32_16x16x32_bf16 v[80:87], v[192:199], v[224:231], v[80:87]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[192:199], v[232:239], v[0:7]
	s_cbranch_scc1 .LBB0_19
	s_ashr_i32 s2, s42, 31
	s_add_co_i32 s3, s45, 0x80
	s_lshr_b32 s5, s2, 30
	v_nop
	v_nop
	v_nop
	v_nop
	v_med3_i32 v192, s3, 0, 0x80
	s_add_co_i32 s5, s42, s5
	s_add_co_i32 s2, s40, 0xffffff80
	s_and_b32 s5, s5, 0x1ffffc
	s_ashr_i32 s3, s2, 31
	s_sub_co_i32 s5, s42, s5
	s_mul_u64 s[6:7], s[2:3], s[30:31]
	s_mul_i32 s21, s5, 0x11800
	v_readfirstlane_b32 s5, v192
	s_lshl_b64 s[6:7], s[6:7], 1
	s_mul_u64 s[2:3], s[2:3], s[56:57]
	s_add_nc_u64 s[6:7], s[60:61], s[6:7]
	s_lshl_b64 s[2:3], s[2:3], 1
	s_sub_co_i32 s5, s5, s9
	s_add_nc_u64 s[6:7], s[10:11], s[6:7]
	s_max_i32 s14, s5, 0
	s_add_nc_u64 s[2:3], s[62:63], s[2:3]
	s_lshl_b32 s14, s14, 16
	s_add_co_i32 s5, s35, s21
	s_bitset1_b32 s7, 31
	s_addk_co_i32 s14, 0x7fff
	s_mov_b32 s23, s15
	tensor_load_to_lds s[4:7], s[12:19]
	s_add_nc_u64 s[6:7], s[58:59], s[2:3]
	s_add_co_i32 s5, s55, s21
	s_bitset1_b32 s7, 31
	s_mov_b32 s21, s13
	s_mov_b32 s22, s14
	s_mov_b32 s24, s16
	s_mov_b32 s27, s19
	tensor_load_to_lds s[4:7], s[20:27]
.LBB0_19:
	s_add_co_i32 s2, s42, 5
	s_cmp_ge_i32 s2, s44
	s_cbranch_scc1 .LBB0_8
	s_add_co_i32 s5, s42, 1
	v_nop
	v_nop
	v_nop
	v_nop
	v_med3_i32 v192, s45, 0, 0x80
	s_ashr_i32 s2, s5, 31
	s_ashr_i32 s41, s40, 31
	s_lshr_b32 s6, s2, 30
	s_mul_u64 s[2:3], s[40:41], s[30:31]
	s_add_co_i32 s14, s5, s6
	s_mul_u64 s[6:7], s[40:41], s[56:57]
	s_and_b32 s14, s14, 0x1ffffc
	s_lshl_b64 s[2:3], s[2:3], 1
	s_sub_co_i32 s5, s5, s14
	v_readfirstlane_b32 s14, v192
	s_mul_i32 s21, s5, 0x11800
	s_lshl_b64 s[6:7], s[6:7], 1
	s_add_nc_u64 s[2:3], s[60:61], s[2:3]
	s_add_nc_u64 s[22:23], s[62:63], s[6:7]
	s_sub_co_i32 s5, s14, s9
	s_add_nc_u64 s[6:7], s[10:11], s[2:3]
	s_max_i32 s2, s5, 0
	s_add_co_i32 s5, s35, s21
	s_lshl_b32 s2, s2, 16
	s_bitset1_b32 s7, 31
	s_or_b32 s14, s2, 0x7fff
	s_mov_b32 s24, s16
	tensor_load_to_lds s[4:7], s[12:19]
	s_add_nc_u64 s[6:7], s[58:59], s[22:23]
	s_add_co_i32 s5, s55, s21
	s_bitset1_b32 s7, 31
	s_mov_b32 s21, s13
	s_mov_b32 s22, s14
	s_mov_b32 s23, s15
	s_mov_b32 s27, s19
	tensor_load_to_lds s[4:7], s[20:27]
	s_branch .LBB0_8
.LBB0_21:
	v_mov_b32_e32 v0, 0
	s_set_vgpr_msb 0x82
	v_dual_mov_b32 v203 /*v715*/, 0xf149f2ca :: v_dual_mov_b32 v202 /*v714*/, v192 /*v704*/
	s_set_vgpr_msb 0x8280
	v_dual_mov_b32 v198 /*v710*/, 0xf149f2ca :: v_dual_mov_b32 v121 /*v633*/, v0
	s_set_vgpr_msb 0x8000
	v_dual_mov_b32 v1, v0 :: v_dual_mov_b32 v2, v0
	v_dual_mov_b32 v3, v0 :: v_dual_mov_b32 v4, v0
	v_dual_mov_b32 v5, v0 :: v_dual_mov_b32 v6, v0
	v_mov_b32_e32 v7, v0
	s_set_vgpr_msb 0x80
	v_mov_b32_e32 v120 /*v632*/, v0
	s_set_vgpr_msb 0x8000
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
	v_mov_b64_e32 v[118:119], v[6:7]
	v_mov_b64_e32 v[116:117], v[4:5]
	v_mov_b64_e32 v[114:115], v[2:3]
	v_mov_b64_e32 v[112:113], v[0:1]
	v_mov_b64_e32 v[126:127], v[6:7]
	v_mov_b64_e32 v[124:125], v[4:5]
	v_mov_b64_e32 v[122:123], v[2:3]
	v_mov_b64_e32 v[120:121], v[0:1]
	v_mov_b64_e32 v[142:143], v[6:7]
	v_mov_b64_e32 v[140:141], v[4:5]
	v_mov_b64_e32 v[138:139], v[2:3]
	v_mov_b64_e32 v[136:137], v[0:1]
	v_mov_b64_e32 v[150:151], v[6:7]
	v_mov_b64_e32 v[148:149], v[4:5]
	v_mov_b64_e32 v[146:147], v[2:3]
	v_mov_b64_e32 v[144:145], v[0:1]
	v_mov_b64_e32 v[158:159], v[6:7]
	v_mov_b64_e32 v[156:157], v[4:5]
	v_mov_b64_e32 v[154:155], v[2:3]
	v_mov_b64_e32 v[152:153], v[0:1]
.LBB0_22:
	s_set_vgpr_msb 0x48
	v_or_b32_e32 v95 /*v351*/, s71, v190 /*v702*/
	s_cmp_ge_u32 s46, s44
	s_set_vgpr_msb 0x4844
	v_or_b32_e32 v94 /*v350*/, 16, v95 /*v351*/
	s_set_vgpr_msb 0x4400
	s_cbranch_scc1 .LBB0_31
	s_add_co_i32 s69, s69, s70
	s_add_co_i32 s2, s52, s68
	s_lshl_b32 s3, s69, 7
	s_lshl_b32 s2, s2, 7
	s_not_b32 s3, s3
	s_set_vgpr_msb 4
	v_dual_add_nc_u32 v192, s73, v95 /*v351*/ :: v_dual_add_nc_u32 v193, s73, v94 /*v350*/
	s_add_co_i32 s3, s3, s2
	s_add_co_i32 s2, s50, s48
	s_min_i32 s3, s3, s72
	s_lshl_b32 s4, s65, 7
	s_add_co_i32 s2, s2, s3
	s_add_co_i32 s3, s50, -1
	s_sub_co_i32 s2, s2, s49
	s_and_b32 s48, s4, 0xffffff00
	s_add_co_i32 s2, s2, 1
	s_set_vgpr_msb 0x440
	v_min_i32_e32 v96 /*v352*/, s3, v192
	s_min_i32 s2, s50, s2
	v_min_i32_e32 v97 /*v353*/, s3, v193
	s_max_i32 s2, s2, 1
	s_mov_b32 s19, 0
	s_sub_co_i32 s2, s2, s48
	s_mov_b32 s45, s19
	s_mov_b32 s40, 1
	s_add_co_i32 s50, s2, 0xfffffe00
	s_add_co_i32 s52, s64, 0x200
	s_mov_b32 s16, 32
	s_mov_b32 s64, 0x76543210
	s_mov_b32 s28, 0x3fb8aa3b
	s_mov_b32 s15, 0x800000
	s_mov_b32 s13, 0xffff0000
	s_mov_b32 s12, 0x7510000
	s_mov_b32 s20, 0xf510000
	s_set_vgpr_msb 0x4000
	s_branch .LBB0_25
.LBB0_24:
	s_set_vgpr_msb 0x49
	v_fmac_f32_e32 v64 /*v320*/, v34 /*v290*/, v120 /*v632*/
	s_add_nc_u64 s[46:47], s[46:47], 1
	v_fmac_f32_e32 v65 /*v321*/, v62 /*v318*/, v121 /*v633*/
	v_cmp_lt_u64_e64 s2, s[46:47], s[44:45]
	s_set_vgpr_msb 0x4982
	v_dual_mov_b32 v204 /*v716*/, v196 /*v708*/ :: v_dual_mov_b32 v199 /*v711*/, v202 /*v714*/
	s_set_vgpr_msb 0x8281
	v_dual_add_f32 v120 /*v632*/, v64 /*v320*/, v254 :: v_dual_add_f32 v121 /*v633*/, v65 /*v321*/, v255
	s_wait_alu depctr_vm_vsrc(0)
	v_dual_mov_b32 v196 /*v708*/, v63 /*v319*/ :: v_dual_mov_b32 v202 /*v714*/, v35 /*v291*/
	v_mov_b32_e32 v203 /*v715*/, v98 /*v354*/
	s_addk_co_i32 s48, 0x80
	s_and_b32 vcc_lo, exec_lo, s2
	s_addk_co_i32 s50, 0xff80
	s_set_vgpr_msb 0x8100
	s_cbranch_vccz .LBB0_32
.LBB0_25:
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 2
	ds_load_b128 v[192:195], v202 /*v714*/
	ds_load_b128 v[196:199], v202 /*v714*/ offset:32
	ds_load_b128 v[208:211], v202 /*v714*/ offset:64
	ds_load_b128 v[212:215], v202 /*v714*/ offset:96
	ds_load_b128 v[216:219], v202 /*v714*/ offset:128
	ds_load_b128 v[220:223], v202 /*v714*/ offset:160
	ds_load_b128 v[224:227], v202 /*v714*/ offset:192
	ds_load_b128 v[228:231], v202 /*v714*/ offset:224
	ds_load_b128 v[232:235], v202 /*v714*/ offset:4352
	ds_load_b128 v[236:239], v202 /*v714*/ offset:4384
	s_set_vgpr_msb 0x248
	v_add_nc_u32_e32 v72 /*v328*/, s48, v194 /*v706*/
	s_set_vgpr_msb 0x4802
	ds_load_b128 v[248:251], v202 /*v714*/ offset:4416
	ds_load_b128 v[252:255], v202 /*v714*/ offset:4448
	s_set_vgpr_msb 0x242
	ds_load_b128 v[0:3] /*v[256:259]*/, v202 /*v714*/ offset:4480
	ds_load_b128 v[4:7] /*v[260:263]*/, v202 /*v714*/ offset:4512
	ds_load_b128 v[8:11] /*v[264:267]*/, v202 /*v714*/ offset:4544
	ds_load_b128 v[12:15] /*v[268:271]*/, v202 /*v714*/ offset:4576
	ds_load_b128 v[16:19] /*v[272:275]*/, v202 /*v714*/ offset:8704
	ds_load_b128 v[20:23] /*v[276:279]*/, v202 /*v714*/ offset:8736
	ds_load_b128 v[24:27] /*v[280:283]*/, v202 /*v714*/ offset:8768
	ds_load_b128 v[28:31] /*v[284:287]*/, v202 /*v714*/ offset:8800
	ds_load_b128 v[32:35] /*v[288:291]*/, v202 /*v714*/ offset:8832
	ds_load_b128 v[36:39] /*v[292:295]*/, v202 /*v714*/ offset:8864
	ds_load_b128 v[40:43] /*v[296:299]*/, v202 /*v714*/ offset:8896
	ds_load_b128 v[44:47] /*v[300:303]*/, v202 /*v714*/ offset:8928
	ds_load_b128 v[48:51] /*v[304:307]*/, v202 /*v714*/ offset:13056
	ds_load_b128 v[52:55] /*v[308:311]*/, v202 /*v714*/ offset:13088
	ds_load_b128 v[56:59] /*v[312:315]*/, v202 /*v714*/ offset:13120
	ds_load_b128 v[60:63] /*v[316:319]*/, v202 /*v714*/ offset:13152
	ds_load_b128 v[86:89] /*v[342:345]*/, v202 /*v714*/ offset:13184
	ds_load_b128 v[90:93] /*v[346:349]*/, v202 /*v714*/ offset:13216
	ds_load_b128 v[98:101] /*v[354:357]*/, v202 /*v714*/ offset:13248
	ds_load_b128 v[102:105] /*v[358:361]*/, v202 /*v714*/ offset:13280
	ds_load_b128 v[106:109] /*v[362:365]*/, v202 /*v714*/ offset:17408
	ds_load_b128 v[110:113] /*v[366:369]*/, v202 /*v714*/ offset:17440
	ds_load_b128 v[114:117] /*v[370:373]*/, v202 /*v714*/ offset:17472
	ds_load_b128 v[118:121] /*v[374:377]*/, v202 /*v714*/ offset:17504
	ds_load_b128 v[122:125] /*v[378:381]*/, v202 /*v714*/ offset:17536
	ds_load_b128 v[126:129] /*v[382:385]*/, v202 /*v714*/ offset:17568
	ds_load_b128 v[130:133] /*v[386:389]*/, v202 /*v714*/ offset:17600
	ds_load_b128 v[134:137] /*v[390:393]*/, v202 /*v714*/ offset:17632
	ds_load_b128 v[138:141] /*v[394:397]*/, v202 /*v714*/ offset:21760
	ds_load_b128 v[142:145] /*v[398:401]*/, v202 /*v714*/ offset:21792
	s_set_vgpr_msb 0x4245
	v_cmp_le_i32_e32 vcc_lo, v72 /*v328*/, v96 /*v352*/
	v_dual_add_nc_u32 v73 /*v329*/, 6, v72 /*v328*/ :: v_dual_add_nc_u32 v74 /*v330*/, 17, v72 /*v328*/
	v_dual_add_nc_u32 v81 /*v337*/, 22, v72 /*v328*/ :: v_dual_add_nc_u32 v84 /*v340*/, 23, v72 /*v328*/
	v_dual_add_nc_u32 v85 /*v341*/, 32, v72 /*v328*/ :: v_dual_add_nc_u32 v226 /*v482*/, 33, v72 /*v328*/
	v_dual_add_nc_u32 v231 /*v487*/, 38, v72 /*v328*/ :: v_dual_add_nc_u32 v232 /*v488*/, 39, v72 /*v328*/
	v_dual_add_nc_u32 v75 /*v331*/, 18, v72 /*v328*/ :: v_dual_add_nc_u32 v78 /*v334*/, 19, v72 /*v328*/
	v_dual_add_nc_u32 v79 /*v335*/, 20, v72 /*v328*/ :: v_dual_add_nc_u32 v80 /*v336*/, 21, v72 /*v328*/
	s_set_vgpr_msb 0x4500
	s_wait_dscnt 0x28
	v_wmma_f32_16x16x32_bf16 v[200:207], v[192:199], v[64:71], 0
	s_set_vgpr_msb 0x45
	v_dual_add_nc_u32 v227 /*v483*/, 34, v72 /*v328*/ :: v_dual_add_nc_u32 v228 /*v484*/, 35, v72 /*v328*/
	v_dual_add_nc_u32 v229 /*v485*/, 36, v72 /*v328*/ :: v_dual_add_nc_u32 v230 /*v486*/, 37, v72 /*v328*/
	v_cmp_le_i32_e64 s2, v74 /*v330*/, v96 /*v352*/
	v_cmp_le_i32_e64 s3, v75 /*v331*/, v96 /*v352*/
	v_cmp_le_i32_e64 s4, v78 /*v334*/, v96 /*v352*/
	s_set_vgpr_msb 0x4500
	s_wait_dscnt 0x26
	v_wmma_f32_16x16x32_bf16 v[200:207], v[208:215], v[72:79], v[200:207]
	s_set_vgpr_msb 5
	v_cmp_le_i32_e64 s5, v79 /*v335*/, v96 /*v352*/
	v_cmp_le_i32_e64 s6, v80 /*v336*/, v96 /*v352*/
	v_cmp_le_i32_e64 s7, v81 /*v337*/, v96 /*v352*/
	s_set_vgpr_msb 0x542
	ds_load_b128 v[170:173] /*v[426:429]*/, v202 /*v714*/ offset:26176
	ds_load_b128 v[174:177] /*v[430:433]*/, v202 /*v714*/ offset:26208
	ds_load_b128 v[178:181] /*v[434:437]*/, v202 /*v714*/ offset:26240
	ds_load_b128 v[182:185] /*v[438:441]*/, v202 /*v714*/ offset:26272
	ds_load_b128 v[186:189] /*v[442:445]*/, v202 /*v714*/ offset:26304
	ds_load_b128 v[190:193] /*v[446:449]*/, v202 /*v714*/ offset:26336
	ds_load_b128 v[194:197] /*v[450:453]*/, v202 /*v714*/ offset:30464
	ds_load_b128 v[198:201] /*v[454:457]*/, v202 /*v714*/ offset:30496
	s_set_vgpr_msb 0x4200
	s_wait_dscnt 0x2c
	v_wmma_f32_16x16x32_bf16 v[200:207], v[216:223], v[104:111], v[200:207]
	s_wait_dscnt 0x2a
	v_wmma_f32_16x16x32_bf16 v[200:207], v[224:231], v[128:135], v[200:207]
	v_wmma_f32_16x16x32_bf16 v[240:247], v[192:199], v[160:167], 0
	v_nop
	v_nop
	v_nop
	v_nop
	v_cndmask_b32_e32 v192, 0xff800000, v200, vcc_lo
	s_set_vgpr_msb 5
	v_cmp_lt_i32_e32 vcc_lo, v72 /*v328*/, v96 /*v352*/
	v_add_nc_u32_e32 v200, 2, v72 /*v328*/
	s_set_vgpr_msb 0x540
	s_wait_dscnt 0x28
	v_wmma_f32_16x16x32_bf16 v[64:71] /*v[320:327]*/, v[232:239], v[64:71], 0
	s_set_vgpr_msb 0x4000
	v_cndmask_b32_e32 v193, 0xff800000, v201, vcc_lo
	s_set_vgpr_msb 4
	v_add_nc_u32_e32 v201, 3, v72 /*v328*/
	v_cmp_le_i32_e32 vcc_lo, v200, v96 /*v352*/
	s_set_vgpr_msb 0x400
	v_cndmask_b32_e32 v194, 0xff800000, v202, vcc_lo
	s_set_vgpr_msb 64
	v_wmma_f32_16x16x32_bf16 v[218:225] /*v[474:481]*/, v[232:239], v[160:167], 0
	s_set_vgpr_msb 0x4004
	v_cmp_le_i32_e32 vcc_lo, v201, v96 /*v352*/
	s_set_vgpr_msb 0x400
	v_cndmask_b32_e32 v195, 0xff800000, v203, vcc_lo
	v_nop
	v_nop
	s_set_vgpr_msb 4
	v_dual_add_nc_u32 v234, 4, v72 /*v328*/ :: v_dual_add_nc_u32 v235, 5, v72 /*v328*/
	s_set_vgpr_msb 0x400
	v_wmma_f32_16x16x32_bf16 v[240:247], v[208:215], v[168:175], v[240:247]
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v234, v96 /*v352*/
	s_set_vgpr_msb 0x400
	v_cndmask_b32_e32 v196, 0xff800000, v204, vcc_lo
	s_set_vgpr_msb 0x50
	s_wait_dscnt 0x26
	v_wmma_f32_16x16x32_bf16 v[64:71] /*v[320:327]*/, v[248:255], v[72:79], v[64:71] /*v[320:327]*/
	s_set_vgpr_msb 0x5004
	v_cmp_le_i32_e32 vcc_lo, v235, v96 /*v352*/
	s_set_vgpr_msb 0x400
	v_cndmask_b32_e32 v197, 0xff800000, v205, vcc_lo
	s_set_vgpr_msb 5
	v_cmp_le_i32_e32 vcc_lo, v73 /*v329*/, v96 /*v352*/
	s_set_vgpr_msb 0x500
	v_wmma_f32_16x16x32_bf16 v[240:247], v[216:223], v[176:183], v[240:247]
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 2
	ds_load_b128 v[210:213], v202 /*v714*/ offset:21824
	ds_load_b128 v[214:217], v202 /*v714*/ offset:21856
	s_set_vgpr_msb 0x242
	ds_load_b128 v[146:149] /*v[402:405]*/, v202 /*v714*/ offset:21888
	ds_load_b128 v[150:153] /*v[406:409]*/, v202 /*v714*/ offset:21920
	ds_load_b128 v[154:157] /*v[410:413]*/, v202 /*v714*/ offset:21952
	ds_load_b128 v[158:161] /*v[414:417]*/, v202 /*v714*/ offset:21984
	ds_load_b128 v[162:165] /*v[418:421]*/, v202 /*v714*/ offset:26112
	ds_load_b128 v[166:169] /*v[422:425]*/, v202 /*v714*/ offset:26144
	s_set_vgpr_msb 0x4200
	v_cndmask_b32_e32 v198, 0xff800000, v206, vcc_lo
	s_set_vgpr_msb 0x51
	s_wait_dscnt 0x2c
	v_wmma_f32_16x16x32_bf16 v[64:71] /*v[320:327]*/, v[0:7] /*v[256:263]*/, v[104:111], v[64:71] /*v[320:327]*/
	s_set_vgpr_msb 0x5150
	v_wmma_f32_16x16x32_bf16 v[218:225] /*v[474:481]*/, v[248:255], v[168:175], v[218:225] /*v[474:481]*/
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x5004
	v_dual_add_nc_u32 v248, 7, v72 /*v328*/ :: v_dual_add_nc_u32 v249, 16, v72 /*v328*/
	v_cmp_le_i32_e32 vcc_lo, v248, v96 /*v352*/
	s_set_vgpr_msb 0x400
	v_wmma_f32_16x16x32_bf16 v[240:247], v[224:231], v[184:191], v[240:247]
	s_set_vgpr_msb 2
	ds_load_b128 v[218:221], v202 /*v714*/ offset:30528
	s_wait_alu depctr_va_vdst(0)
	ds_load_b128 v[222:225], v202 /*v714*/ offset:30560
	s_set_vgpr_msb 0x242
	ds_load_b128 v[202:205] /*v[458:461]*/, v202 /*v714*/ offset:30592
	ds_load_b128 v[206:209] /*v[462:465]*/, v202 /*v714*/ offset:30624
	ds_load_b128 v[210:213] /*v[466:469]*/, v202 /*v714*/ offset:30656
	ds_load_b128 v[214:217] /*v[470:473]*/, v202 /*v714*/ offset:30688
	s_set_vgpr_msb 0x4200
	v_cndmask_b32_e32 v199, 0xff800000, v207, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v249, v96 /*v352*/
	s_set_vgpr_msb 0x401
	s_wait_dscnt 0x2e
	v_wmma_f32_16x16x32_bf16 v[226:233], v[16:23] /*v[272:279]*/, v[64:71], 0
	s_set_vgpr_msb 0x151
	v_wmma_f32_16x16x32_bf16 v[64:71] /*v[320:327]*/, v[8:15] /*v[264:271]*/, v[128:135], v[64:71] /*v[320:327]*/
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x5104
	v_cndmask_b32_e32 v202, 0xff800000, v64 /*v320*/, vcc_lo
	s_set_vgpr_msb 0x401
	s_wait_dscnt 0x2c
	v_wmma_f32_16x16x32_bf16 v[226:233], v[24:31] /*v[280:287]*/, v[72:79], v[226:233]
	s_set_vgpr_msb 0x105
	v_cmp_le_i32_e32 vcc_lo, v84 /*v340*/, v96 /*v352*/
	v_cndmask_b32_e64 v204, 0xff800000, v70 /*v326*/, s7
	v_cndmask_b32_e64 v203, 0xff800000, v65 /*v321*/, s2
	v_cndmask_b32_e64 v208, 0xff800000, v66 /*v322*/, s3
	v_cndmask_b32_e64 v209, 0xff800000, v67 /*v323*/, s4
	v_cndmask_b32_e32 v205, 0xff800000, v71 /*v327*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v72 /*v328*/, v97 /*v353*/
	s_set_vgpr_msb 0x501
	s_wait_dscnt 0x2a
	v_wmma_f32_16x16x32_bf16 v[226:233], v[32:39] /*v[288:295]*/, v[104:111], v[226:233]
	s_set_vgpr_msb 0x105
	v_cndmask_b32_e64 v206, 0xff800000, v68 /*v324*/, s5
	v_cndmask_b32_e64 v207, 0xff800000, v69 /*v325*/, s6
	v_cmp_le_i32_e64 s2, v226 /*v482*/, v96 /*v352*/
	s_set_vgpr_msb 0x500
	v_cndmask_b32_e32 v254, 0xff800000, v240, vcc_lo
	s_set_vgpr_msb 5
	v_cmp_lt_i32_e32 vcc_lo, v72 /*v328*/, v97 /*v353*/
	v_cmp_le_i32_e64 s3, v227 /*v483*/, v96 /*v352*/
	v_cmp_le_i32_e64 s4, v228 /*v484*/, v96 /*v352*/
	s_set_vgpr_msb 0x501
	s_wait_dscnt 0x28
	v_wmma_f32_16x16x32_bf16 v[226:233], v[40:47] /*v[296:303]*/, v[128:135], v[226:233]
	s_set_vgpr_msb 0x105
	v_cmp_le_i32_e64 s5, v229 /*v485*/, v96 /*v352*/
	s_set_vgpr_msb 0x500
	v_cndmask_b32_e32 v255, 0xff800000, v241, vcc_lo
	s_set_vgpr_msb 5
	v_cmp_le_i32_e32 vcc_lo, v85 /*v341*/, v96 /*v352*/
	v_cmp_le_i32_e64 s6, v230 /*v486*/, v96 /*v352*/
	v_cmp_le_i32_e64 s7, v231 /*v487*/, v96 /*v352*/
	s_set_vgpr_msb 0x500
	v_cndmask_b32_e32 v252, 0xff800000, v226, vcc_lo
	s_set_vgpr_msb 5
	v_cmp_le_i32_e32 vcc_lo, v232 /*v488*/, v96 /*v352*/
	s_set_vgpr_msb 0x551
	v_wmma_f32_16x16x32_bf16 v[218:225] /*v[474:481]*/, v[0:7] /*v[256:263]*/, v[176:183], v[218:225] /*v[474:481]*/
	s_set_vgpr_msb 0x5100
	v_cndmask_b32_e64 v250, 0xff800000, v232, s7
	v_cndmask_b32_e64 v253, 0xff800000, v227, s2
	v_cndmask_b32_e32 v251, 0xff800000, v233, vcc_lo
	s_set_vgpr_msb 4
	v_cmp_le_i32_e32 vcc_lo, v200, v97 /*v353*/
	s_set_vgpr_msb 0x400
	v_max3_num_f32 v200, v192, v193, v194
	s_set_vgpr_msb 64
	v_cndmask_b32_e32 v76 /*v332*/, 0xff800000, v242, vcc_lo
	s_set_vgpr_msb 0x4004
	v_cmp_le_i32_e32 vcc_lo, v201, v97 /*v353*/
	s_set_vgpr_msb 0x451
	v_wmma_f32_16x16x32_bf16 v[218:225] /*v[474:481]*/, v[8:15] /*v[264:271]*/, v[184:191], v[218:225] /*v[474:481]*/
	v_cndmask_b32_e32 v77 /*v333*/, 0xff800000, v243, vcc_lo
	v_cmp_ge_i32_e32 vcc_lo, v97 /*v353*/, v234
	v_nop
	v_nop
	v_cndmask_b32_e64 v8 /*v264*/, 0xff800000, v228, s3
	v_cndmask_b32_e64 v9 /*v265*/, 0xff800000, v229, s4
	v_cndmask_b32_e64 v10 /*v266*/, 0xff800000, v230, s5
	v_cndmask_b32_e64 v11 /*v267*/, 0xff800000, v231, s6
	s_set_vgpr_msb 0x5101
	s_wait_dscnt 0x26
	v_wmma_f32_16x16x32_bf16 v[226:233], v[48:55] /*v[304:311]*/, v[64:71], 0
	s_set_vgpr_msb 0x140
	v_cndmask_b32_e32 v66 /*v322*/, 0xff800000, v244, vcc_lo
	s_set_vgpr_msb 0x4004
	v_cmp_le_i32_e32 vcc_lo, v235, v97 /*v353*/
	s_set_vgpr_msb 0x410
	v_max3_num_f32 v201, v254, v255, v76 /*v332*/
	s_set_vgpr_msb 0x1040
	v_cndmask_b32_e32 v67 /*v323*/, 0xff800000, v245, vcc_lo
	s_set_vgpr_msb 0x4001
	v_wmma_f32_16x16x32_bf16 v[234:241], v[48:55] /*v[304:311]*/, v[160:167], 0
	s_set_vgpr_msb 0x105
	v_cmp_le_i32_e32 vcc_lo, v73 /*v329*/, v97 /*v353*/
	s_set_vgpr_msb 0x540
	v_cndmask_b32_e32 v64 /*v320*/, 0xff800000, v246, vcc_lo
	s_set_vgpr_msb 0x4004
	v_cmp_le_i32_e32 vcc_lo, v248, v97 /*v353*/
	s_set_vgpr_msb 0x441
	v_wmma_f32_16x16x32_bf16 v[0:7] /*v[256:263]*/, v[16:23] /*v[272:279]*/, v[160:167], 0
	v_cndmask_b32_e32 v65 /*v321*/, 0xff800000, v247, vcc_lo
	v_cmp_ge_i32_e32 vcc_lo, v97 /*v353*/, v249
	s_set_vgpr_msb 0x4101
	s_wait_dscnt 0x24
	v_wmma_f32_16x16x32_bf16 v[226:233], v[56:63] /*v[312:319]*/, v[72:79], v[226:233]
	v_wmma_f32_16x16x32_bf16 v[234:241], v[56:63] /*v[312:319]*/, v[168:175], v[234:241]
	s_set_vgpr_msb 0x151
	v_wmma_f32_16x16x32_bf16 v[0:7] /*v[256:263]*/, v[24:31] /*v[280:287]*/, v[168:175], v[0:7] /*v[256:263]*/
	s_set_vgpr_msb 0x5101
	s_wait_dscnt 0x22
	v_wmma_f32_16x16x32_bf16 v[226:233], v[86:93] /*v[342:349]*/, v[104:111], v[226:233]
	v_wmma_f32_16x16x32_bf16 v[234:241], v[86:93] /*v[342:349]*/, v[176:183], v[234:241]
	s_set_vgpr_msb 0x151
	v_wmma_f32_16x16x32_bf16 v[0:7] /*v[256:263]*/, v[32:39] /*v[288:295]*/, v[176:183], v[0:7] /*v[256:263]*/
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x5145
	v_cndmask_b32_e32 v34 /*v290*/, 0xff800000, v218 /*v474*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v74 /*v330*/, v97 /*v353*/
	v_cndmask_b32_e32 v35 /*v291*/, 0xff800000, v219 /*v475*/, vcc_lo
	s_set_vgpr_msb 0x4501
	s_wait_dscnt 0x20
	v_wmma_f32_16x16x32_bf16 v[226:233], v[98:105] /*v[354:361]*/, v[128:135], v[226:233]
	s_set_vgpr_msb 0x145
	v_cmp_le_i32_e32 vcc_lo, v75 /*v331*/, v97 /*v353*/
	v_cndmask_b32_e32 v82 /*v338*/, 0xff800000, v220 /*v476*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v78 /*v334*/, v97 /*v353*/
	s_set_vgpr_msb 0x4501
	v_wmma_f32_16x16x32_bf16 v[234:241], v[98:105] /*v[354:361]*/, v[184:191], v[234:241]
	s_set_vgpr_msb 0x146
	v_cndmask_b32_e32 v83 /*v339*/, 0xff800000, v221 /*v477*/, vcc_lo
	v_nop
	v_nop
	v_nop
	v_dual_mov_b32 v99 /*v355*/, v198 /*v710*/ :: v_dual_add_nc_u32 v241 /*v497*/, 64, v72 /*v328*/
	s_set_vgpr_msb 0x4605
	v_cmp_le_i32_e32 vcc_lo, v79 /*v335*/, v97 /*v353*/
	s_set_vgpr_msb 0x551
	v_wmma_f32_16x16x32_bf16 v[0:7] /*v[256:263]*/, v[40:47] /*v[296:303]*/, v[184:191], v[0:7] /*v[256:263]*/
	s_set_vgpr_msb 0x5145
	v_dual_add_nc_u32 v233 /*v489*/, 48, v72 /*v328*/ :: v_dual_add_nc_u32 v234 /*v490*/, 49, v72 /*v328*/
	v_dual_add_nc_u32 v239 /*v495*/, 54, v72 /*v328*/ :: v_dual_add_nc_u32 v240 /*v496*/, 55, v72 /*v328*/
	v_cndmask_b32_e32 v70 /*v326*/, 0xff800000, v222 /*v478*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v80 /*v336*/, v97 /*v353*/
	v_dual_add_nc_u32 v235 /*v491*/, 50, v72 /*v328*/ :: v_dual_add_nc_u32 v236 /*v492*/, 51, v72 /*v328*/
	v_dual_add_nc_u32 v237 /*v493*/, 52, v72 /*v328*/ :: v_dual_add_nc_u32 v238 /*v494*/, 53, v72 /*v328*/
	v_cndmask_b32_e32 v71 /*v327*/, 0xff800000, v223 /*v479*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v81 /*v337*/, v97 /*v353*/
	v_cmp_le_i32_e64 s2, v234 /*v490*/, v96 /*v352*/
	v_cmp_le_i32_e64 s3, v235 /*v491*/, v96 /*v352*/
	v_cmp_le_i32_e64 s4, v236 /*v492*/, v96 /*v352*/
	v_cmp_le_i32_e64 s5, v237 /*v493*/, v96 /*v352*/
	v_cndmask_b32_e32 v68 /*v324*/, 0xff800000, v224 /*v480*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v84 /*v340*/, v97 /*v353*/
	v_cmp_le_i32_e64 s6, v238 /*v494*/, v96 /*v352*/
	v_cmp_le_i32_e64 s7, v239 /*v495*/, v96 /*v352*/
	s_set_vgpr_msb 0x4540
	v_cndmask_b32_e64 v12 /*v268*/, 0xff800000, v228, s3
	v_cndmask_b32_e64 v13 /*v269*/, 0xff800000, v229, s4
	s_set_vgpr_msb 0x4045
	v_cndmask_b32_e32 v69 /*v325*/, 0xff800000, v225 /*v481*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v85 /*v341*/, v97 /*v353*/
	s_set_vgpr_msb 0x4540
	v_cndmask_b32_e64 v14 /*v270*/, 0xff800000, v230, s5
	v_cndmask_b32_e64 v15 /*v271*/, 0xff800000, v231, s6
	s_set_vgpr_msb 0x4001
	s_wait_dscnt 0x1e
	v_wmma_f32_16x16x32_bf16 v[242:249], v[106:113] /*v[362:369]*/, v[160:167], 0
	s_set_vgpr_msb 0x145
	v_add_nc_u32_e32 v244 /*v500*/, 0x47, v72 /*v328*/
	v_cndmask_b32_e32 v62 /*v318*/, 0xff800000, v0 /*v256*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v226 /*v482*/, v97 /*v353*/
	s_set_vgpr_msb 0x4540
	v_cndmask_b32_e64 v0 /*v256*/, 0xff800000, v232, s7
	s_set_vgpr_msb 0x4045
	v_add_nc_u32_e32 v16 /*v272*/, 0x41, v72 /*v328*/
	v_add_nc_u32_e32 v17 /*v273*/, 0x42, v72 /*v328*/
	v_add_nc_u32_e32 v20 /*v276*/, 0x43, v72 /*v328*/
	v_cndmask_b32_e32 v63 /*v319*/, 0xff800000, v1 /*v257*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v227 /*v483*/, v97 /*v353*/
	s_set_vgpr_msb 0x4501
	s_wait_dscnt 0x1c
	v_wmma_f32_16x16x32_bf16 v[242:249], v[114:121] /*v[370:377]*/, v[168:175], v[242:249]
	s_set_vgpr_msb 0x145
	v_add_nc_u32_e32 v21 /*v277*/, 0x44, v72 /*v328*/
	v_add_nc_u32_e32 v242 /*v498*/, 0x45, v72 /*v328*/
	v_add_nc_u32_e32 v243 /*v499*/, 0x46, v72 /*v328*/
	v_cndmask_b32_e32 v84 /*v340*/, 0xff800000, v2 /*v258*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v228 /*v484*/, v97 /*v353*/
	v_cmp_le_i32_e64 s3, v17 /*v273*/, v96 /*v352*/
	v_cmp_le_i32_e64 s4, v20 /*v276*/, v96 /*v352*/
	s_set_vgpr_msb 0x4501
	s_wait_dscnt 0x1a
	v_wmma_f32_16x16x32_bf16 v[242:249], v[122:129] /*v[378:385]*/, v[176:183], v[242:249]
	s_set_vgpr_msb 0x145
	v_cmp_le_i32_e64 s5, v21 /*v277*/, v96 /*v352*/
	v_cndmask_b32_e32 v85 /*v341*/, 0xff800000, v3 /*v259*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v229 /*v485*/, v97 /*v353*/
	v_cmp_le_i32_e64 s6, v242 /*v498*/, v96 /*v352*/
	v_cmp_le_i32_e64 s7, v243 /*v499*/, v96 /*v352*/
	v_add_nc_u32_e32 v245 /*v501*/, 0x50, v72 /*v328*/
	v_add_nc_u32_e32 v26 /*v282*/, 0x57, v72 /*v328*/
	v_cndmask_b32_e32 v78 /*v334*/, 0xff800000, v4 /*v260*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v230 /*v486*/, v97 /*v353*/
	s_set_vgpr_msb 0x4501
	s_wait_dscnt 0x18
	v_wmma_f32_16x16x32_bf16 v[242:249], v[130:137] /*v[386:393]*/, v[184:191], v[242:249]
	s_set_vgpr_msb 0x145
	v_add_nc_u32_e32 v246 /*v502*/, 0x51, v72 /*v328*/
	v_add_nc_u32_e32 v247 /*v503*/, 0x52, v72 /*v328*/
	v_add_nc_u32_e32 v248 /*v504*/, 0x53, v72 /*v328*/
	v_cndmask_b32_e32 v79 /*v335*/, 0xff800000, v5 /*v261*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v231 /*v487*/, v97 /*v353*/
	s_set_vgpr_msb 0x4540
	v_cndmask_b32_e64 v5 /*v261*/, 0xff800000, v227, s2
	s_set_vgpr_msb 0x4045
	v_cmp_le_i32_e64 s2, v16 /*v272*/, v96 /*v352*/
	v_add_nc_u32_e32 v249 /*v505*/, 0x54, v72 /*v328*/
	v_add_nc_u32_e32 v250 /*v506*/, 0x55, v72 /*v328*/
	v_cndmask_b32_e32 v74 /*v330*/, 0xff800000, v6 /*v262*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v232 /*v488*/, v97 /*v353*/
	v_add_nc_u32_e32 v251 /*v507*/, 0x56, v72 /*v328*/
	v_add_nc_u32_e32 v27 /*v283*/, 0x60, v72 /*v328*/
	s_set_vgpr_msb 0x4584
	v_add_nc_u32_e32 v0 /*v512*/, 0x67, v72 /*v328*/
	s_set_vgpr_msb 0x8445
	v_add_nc_u32_e32 v30 /*v286*/, 0x61, v72 /*v328*/
	v_cndmask_b32_e32 v75 /*v331*/, 0xff800000, v7 /*v263*/, vcc_lo
	v_cmp_le_i32_e32 vcc_lo, v233 /*v489*/, v96 /*v352*/
	v_add_nc_u32_e32 v31 /*v287*/, 0x62, v72 /*v328*/
	v_add_nc_u32_e32 v252 /*v508*/, 0x63, v72 /*v328*/
	v_add_nc_u32_e32 v253 /*v509*/, 0x64, v72 /*v328*/
	v_add_nc_u32_e32 v254 /*v510*/, 0x65, v72 /*v328*/
	s_set_vgpr_msb 0x4540
	v_cndmask_b32_e32 v4 /*v260*/, 0xff800000, v226, vcc_lo
	s_set_vgpr_msb 0x4045
	v_cmp_le_i32_e32 vcc_lo, v240 /*v496*/, v96 /*v352*/
	v_add_nc_u32_e32 v255 /*v511*/, 0x66, v72 /*v328*/
	s_set_vgpr_msb 0x4584
	v_add_nc_u32_e32 v1 /*v513*/, 0x70, v72 /*v328*/
	v_add_nc_u32_e32 v2 /*v514*/, 0x71, v72 /*v328*/
	v_add_nc_u32_e32 v3 /*v515*/, 0x72, v72 /*v328*/
	s_set_vgpr_msb 0x8440
	v_cndmask_b32_e32 v1 /*v257*/, 0xff800000, v233, vcc_lo
	s_set_vgpr_msb 0x4005
	v_cmp_le_i32_e32 vcc_lo, v233 /*v489*/, v97 /*v353*/
	s_set_vgpr_msb 0x501
	v_wmma_f32_16x16x32_bf16 v[226:233], v[106:113] /*v[362:369]*/, v[64:71], 0
	s_set_vgpr_msb 0x184
	v_add_nc_u32_e32 v4 /*v516*/, 0x73, v72 /*v328*/
	v_add_nc_u32_e32 v5 /*v517*/, 0x74, v72 /*v328*/
	s_set_vgpr_msb 0x8444
	v_add_nc_u32_e32 v36 /*v292*/, 0x75, v72 /*v328*/
	s_set_vgpr_msb 0x4440
	v_cndmask_b32_e32 v92 /*v348*/, 0xff800000, v234, vcc_lo
	s_set_vgpr_msb 0x4045
	v_cmp_le_i32_e32 vcc_lo, v234 /*v490*/, v97 /*v353*/
	v_add_nc_u32_e32 v37 /*v293*/, 0x76, v72 /*v328*/
	s_set_vgpr_msb 0x4584
	v_add_nc_u32_e32 v6 /*v518*/, 0x77, v72 /*v328*/
	s_set_vgpr_msb 0x8401
	v_wmma_f32_16x16x32_bf16 v[226:233], v[114:121] /*v[370:377]*/, v[72:79], v[226:233]
	s_set_vgpr_msb 0x140
	v_cndmask_b32_e32 v93 /*v349*/, 0xff800000, v235, vcc_lo
	s_set_vgpr_msb 0x4005
	v_cmp_le_i32_e32 vcc_lo, v235 /*v491*/, v97 /*v353*/
	s_set_vgpr_msb 0x540
	v_cndmask_b32_e32 v90 /*v346*/, 0xff800000, v236, vcc_lo
	s_set_vgpr_msb 0x4005
	v_cmp_le_i32_e32 vcc_lo, v236 /*v492*/, v97 /*v353*/
	s_set_vgpr_msb 0x501
	v_wmma_f32_16x16x32_bf16 v[226:233], v[122:129] /*v[378:385]*/, v[104:111], v[226:233]
	s_set_vgpr_msb 0x140
	v_cndmask_b32_e32 v91 /*v347*/, 0xff800000, v237, vcc_lo
	s_set_vgpr_msb 0x4005
	v_cmp_le_i32_e32 vcc_lo, v237 /*v493*/, v97 /*v353*/
	s_set_vgpr_msb 0x540
	v_cndmask_b32_e32 v88 /*v344*/, 0xff800000, v238, vcc_lo
	s_set_vgpr_msb 0x4005
	v_cmp_le_i32_e32 vcc_lo, v238 /*v494*/, v97 /*v353*/
	s_set_vgpr_msb 0x501
	v_wmma_f32_16x16x32_bf16 v[226:233], v[130:137] /*v[386:393]*/, v[128:135], v[226:233]
	s_set_vgpr_msb 0x140
	v_cndmask_b32_e32 v89 /*v345*/, 0xff800000, v239, vcc_lo
	s_set_vgpr_msb 0x4005
	v_cmp_le_i32_e32 vcc_lo, v239 /*v495*/, v97 /*v353*/
	v_nop
	v_nop
	s_set_vgpr_msb 0x540
	v_cndmask_b32_e64 v18 /*v274*/, 0xff800000, v232, s7
	v_cndmask_b32_e64 v23 /*v279*/, 0xff800000, v227, s2
	v_cndmask_b32_e32 v86 /*v342*/, 0xff800000, v240, vcc_lo
	s_set_vgpr_msb 0x4005
	v_cmp_le_i32_e32 vcc_lo, v240 /*v496*/, v97 /*v353*/
	s_set_vgpr_msb 0x540
	v_cndmask_b32_e64 v24 /*v280*/, 0xff800000, v228, s3
	v_cndmask_b32_e64 v25 /*v281*/, 0xff800000, v229, s4
	v_cndmask_b32_e64 v28 /*v284*/, 0xff800000, v230, s5
	v_cndmask_b32_e64 v29 /*v285*/, 0xff800000, v231, s6
	v_cndmask_b32_e32 v87 /*v343*/, 0xff800000, v241, vcc_lo
	s_set_vgpr_msb 0x4005
	v_cmp_le_i32_e32 vcc_lo, v241 /*v497*/, v96 /*v352*/
	s_set_vgpr_msb 0x501
	s_wait_dscnt 0x16
	v_wmma_f32_16x16x32_bf16 v[234:241], v[138:145] /*v[394:401]*/, v[160:167], 0
	s_set_vgpr_msb 0x105
	v_cmp_le_i32_e64 s2, v246 /*v502*/, v96 /*v352*/
	v_cmp_le_i32_e64 s3, v247 /*v503*/, v96 /*v352*/
	v_cmp_le_i32_e64 s4, v248 /*v504*/, v96 /*v352*/
	s_set_vgpr_msb 0x540
	v_cndmask_b32_e32 v22 /*v278*/, 0xff800000, v226, vcc_lo
	s_set_vgpr_msb 0x4005
	v_cmp_le_i32_e32 vcc_lo, v244 /*v500*/, v96 /*v352*/
	v_cmp_le_i32_e64 s5, v249 /*v505*/, v96 /*v352*/
	v_cmp_le_i32_e64 s6, v250 /*v506*/, v96 /*v352*/
	s_set_vgpr_msb 0x500
	s_wait_dscnt 0xc
	v_wmma_f32_16x16x32_bf16 v[234:241], v[210:217], v[168:175], v[234:241]
	s_set_vgpr_msb 5
	v_cmp_le_i32_e64 s7, v251 /*v507*/, v96 /*v352*/
	s_set_vgpr_msb 0x540
	v_cndmask_b32_e32 v19 /*v275*/, 0xff800000, v233, vcc_lo
	s_set_vgpr_msb 0x4005
	v_cmp_le_i32_e32 vcc_lo, v241 /*v497*/, v97 /*v353*/
	s_set_vgpr_msb 0x540
	v_cndmask_b32_e32 v100 /*v356*/, 0xff800000, v242, vcc_lo
	s_set_vgpr_msb 0x4005
	v_cmp_le_i32_e32 vcc_lo, v16 /*v272*/, v97 /*v353*/
	s_set_vgpr_msb 0x501
	v_wmma_f32_16x16x32_bf16 v[226:233], v[138:145] /*v[394:401]*/, v[64:71], 0
	s_set_vgpr_msb 0x140
	v_cndmask_b32_e32 v101 /*v357*/, 0xff800000, v243, vcc_lo
	s_set_vgpr_msb 0x4005
	v_cmp_le_i32_e32 vcc_lo, v17 /*v273*/, v97 /*v353*/
	s_set_vgpr_msb 0x540
	v_cndmask_b32_e32 v102 /*v358*/, 0xff800000, v244, vcc_lo
	s_set_vgpr_msb 0x4005
	v_cmp_le_i32_e32 vcc_lo, v20 /*v276*/, v97 /*v353*/
	s_set_vgpr_msb 0x500
	v_wmma_f32_16x16x32_bf16 v[226:233], v[210:217], v[72:79], v[226:233]
	s_set_vgpr_msb 64
	v_cndmask_b32_e32 v103 /*v359*/, 0xff800000, v245, vcc_lo
	s_set_vgpr_msb 0x4005
	v_cmp_le_i32_e32 vcc_lo, v21 /*v277*/, v97 /*v353*/
	s_set_vgpr_msb 0x540
	v_cndmask_b32_e32 v104 /*v360*/, 0xff800000, v246, vcc_lo
	s_set_vgpr_msb 0x4005
	v_cmp_le_i32_e32 vcc_lo, v242 /*v498*/, v97 /*v353*/
	s_set_vgpr_msb 0x501
	s_wait_dscnt 0xa
	v_wmma_f32_16x16x32_bf16 v[226:233], v[146:153] /*v[402:409]*/, v[104:111], v[226:233]
	s_set_vgpr_msb 0x140
	v_cndmask_b32_e32 v105 /*v361*/, 0xff800000, v247, vcc_lo
	s_set_vgpr_msb 0x4005
	v_cmp_le_i32_e32 vcc_lo, v243 /*v499*/, v97 /*v353*/
	s_set_vgpr_msb 0x540
	v_cndmask_b32_e32 v106 /*v362*/, 0xff800000, v248, vcc_lo
	s_set_vgpr_msb 0x4005
	v_cmp_le_i32_e32 vcc_lo, v244 /*v500*/, v97 /*v353*/
	s_set_vgpr_msb 0x501
	s_wait_dscnt 0x8
	v_wmma_f32_16x16x32_bf16 v[226:233], v[154:161] /*v[410:417]*/, v[128:135], v[226:233]
	s_set_vgpr_msb 0x140
	v_cndmask_b32_e32 v107 /*v363*/, 0xff800000, v249, vcc_lo
	s_set_vgpr_msb 0x4005
	v_cmp_le_i32_e32 vcc_lo, v245 /*v501*/, v96 /*v352*/
	v_nop
	v_nop
	s_set_vgpr_msb 0x501
	v_cndmask_b32_e64 v242, 0xff800000, v232, s7
	v_wmma_f32_16x16x32_bf16 v[234:241], v[146:153] /*v[402:409]*/, v[176:183], v[234:241]
	v_cndmask_b32_e32 v248, 0xff800000, v226, vcc_lo
	s_set_vgpr_msb 0x105
	v_cmp_le_i32_e32 vcc_lo, v26 /*v282*/, v96 /*v352*/
	s_set_vgpr_msb 0x500
	v_cndmask_b32_e64 v249, 0xff800000, v227, s2
	s_set_vgpr_msb 64
	v_cndmask_b32_e64 v32 /*v288*/, 0xff800000, v228, s3
	v_cndmask_b32_e64 v33 /*v289*/, 0xff800000, v229, s4
	v_cndmask_b32_e64 v38 /*v294*/, 0xff800000, v230, s5
	s_set_vgpr_msb 0x4001
	v_cndmask_b32_e32 v243, 0xff800000, v233, vcc_lo
	v_wmma_f32_16x16x32_bf16 v[234:241], v[154:161] /*v[410:417]*/, v[184:191], v[234:241]
	s_set_vgpr_msb 0x105
	v_cmp_le_i32_e32 vcc_lo, v245 /*v501*/, v97 /*v353*/
	s_set_vgpr_msb 0x540
	v_cndmask_b32_e64 v39 /*v295*/, 0xff800000, v231, s6
	s_set_vgpr_msb 0x4005
	v_cmp_le_i32_e64 s2, v30 /*v286*/, v96 /*v352*/
	v_cmp_le_i32_e64 s3, v31 /*v287*/, v96 /*v352*/
	v_cmp_le_i32_e64 s4, v252 /*v508*/, v96 /*v352*/
	v_cmp_le_i32_e64 s5, v253 /*v509*/, v96 /*v352*/
	v_cmp_le_i32_e64 s6, v254 /*v510*/, v96 /*v352*/
	s_set_vgpr_msb 0x540
	v_cndmask_b32_e32 v108 /*v364*/, 0xff800000, v234, vcc_lo
	s_set_vgpr_msb 0x4005
	v_cmp_le_i32_e32 vcc_lo, v246 /*v502*/, v97 /*v353*/
	s_set_vgpr_msb 0x501
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x32_bf16 v[210:217], v[162:169] /*v[418:425]*/, v[64:71], 0
	s_set_vgpr_msb 0x105
	v_cmp_le_i32_e64 s7, v255 /*v511*/, v96 /*v352*/
	s_set_vgpr_msb 0x540
	v_cndmask_b32_e32 v109 /*v365*/, 0xff800000, v235, vcc_lo
	s_set_vgpr_msb 0x4005
	v_cmp_le_i32_e32 vcc_lo, v247 /*v503*/, v97 /*v353*/
	s_set_vgpr_msb 0x540
	v_cndmask_b32_e32 v110 /*v366*/, 0xff800000, v236, vcc_lo
	s_set_vgpr_msb 0x4005
	v_cmp_le_i32_e32 vcc_lo, v248 /*v504*/, v97 /*v353*/
	s_set_vgpr_msb 0x501
	v_wmma_f32_16x16x32_bf16 v[210:217], v[170:177] /*v[426:433]*/, v[72:79], v[210:217]
	s_set_vgpr_msb 0x140
	v_cndmask_b32_e32 v111 /*v367*/, 0xff800000, v237, vcc_lo
	s_set_vgpr_msb 0x4005
	v_cmp_le_i32_e32 vcc_lo, v249 /*v505*/, v97 /*v353*/
	s_set_vgpr_msb 0x540
	v_cndmask_b32_e32 v112 /*v368*/, 0xff800000, v238, vcc_lo
	s_set_vgpr_msb 0x4001
	v_wmma_f32_16x16x32_bf16 v[226:233], v[162:169] /*v[418:425]*/, v[160:167], 0
	s_set_vgpr_msb 0x105
	v_cmp_le_i32_e32 vcc_lo, v250 /*v506*/, v97 /*v353*/
	s_set_vgpr_msb 0x540
	v_cndmask_b32_e32 v113 /*v369*/, 0xff800000, v239, vcc_lo
	s_set_vgpr_msb 0x4005
	v_cmp_le_i32_e32 vcc_lo, v251 /*v507*/, v97 /*v353*/
	s_set_vgpr_msb 0x501
	v_wmma_f32_16x16x32_bf16 v[210:217], v[178:185] /*v[434:441]*/, v[104:111], v[210:217]
	s_set_vgpr_msb 0x140
	v_cndmask_b32_e32 v114 /*v370*/, 0xff800000, v240, vcc_lo
	s_set_vgpr_msb 0x4005
	v_cmp_le_i32_e32 vcc_lo, v26 /*v282*/, v97 /*v353*/
	s_set_vgpr_msb 0x540
	v_cndmask_b32_e32 v115 /*v371*/, 0xff800000, v241, vcc_lo
	s_set_vgpr_msb 0x4001
	v_wmma_f32_16x16x32_bf16 v[226:233], v[170:177] /*v[426:433]*/, v[168:175], v[226:233]
	s_set_vgpr_msb 0x105
	v_cmp_le_i32_e32 vcc_lo, v27 /*v283*/, v96 /*v352*/
	s_set_vgpr_msb 0x501
	v_wmma_f32_16x16x32_bf16 v[210:217], v[186:193] /*v[442:449]*/, v[128:135], v[210:217]
	v_nop
	v_nop
	v_nop
	v_nop
	s_set_vgpr_msb 0x140
	v_cndmask_b32_e32 v44 /*v300*/, 0xff800000, v210, vcc_lo
	s_set_vgpr_msb 0x4001
	v_wmma_f32_16x16x32_bf16 v[226:233], v[178:185] /*v[434:441]*/, v[176:183], v[226:233]
	s_set_vgpr_msb 0x106
	v_cmp_le_i32_e32 vcc_lo, v0 /*v512*/, v96 /*v352*/
	s_set_vgpr_msb 0x640
	v_cndmask_b32_e64 v40 /*v296*/, 0xff800000, v216, s7
	v_cndmask_b32_e64 v45 /*v301*/, 0xff800000, v211, s2
	v_cndmask_b32_e64 v46 /*v302*/, 0xff800000, v212, s3
	v_cndmask_b32_e64 v47 /*v303*/, 0xff800000, v213, s4
	v_cndmask_b32_e32 v41 /*v297*/, 0xff800000, v217, vcc_lo
	s_set_vgpr_msb 0x4005
	v_cmp_le_i32_e32 vcc_lo, v27 /*v283*/, v97 /*v353*/
	s_set_vgpr_msb 0x501
	v_wmma_f32_16x16x32_bf16 v[226:233], v[186:193] /*v[442:449]*/, v[184:191], v[226:233]
	s_set_vgpr_msb 0x140
	v_cndmask_b32_e64 v52 /*v308*/, 0xff800000, v214, s5
	v_cndmask_b32_e64 v53 /*v309*/, 0xff800000, v215, s6
	s_set_vgpr_msb 0x4015
	v_max3_num_f32 v244, v47 /*v303*/, v52 /*v308*/, v53 /*v309*/
	v_nop
	s_set_vgpr_msb 0x1540
	v_cndmask_b32_e32 v116 /*v372*/, 0xff800000, v226, vcc_lo
	s_set_vgpr_msb 0x4005
	v_cmp_le_i32_e32 vcc_lo, v30 /*v286*/, v97 /*v353*/
	s_set_vgpr_msb 0x501
	v_wmma_f32_16x16x32_bf16 v[210:217], v[194:201] /*v[450:457]*/, v[64:71], 0
	s_set_vgpr_msb 0x115
	v_max3_num_f32 v226, v13 /*v269*/, v14 /*v270*/, v15 /*v271*/
	s_set_vgpr_msb 0x1540
	v_cndmask_b32_e32 v117 /*v373*/, 0xff800000, v227, vcc_lo
	s_set_vgpr_msb 0x4015
	v_cmp_le_i32_e32 vcc_lo, v31 /*v287*/, v97 /*v353*/
	v_max3_num_f32 v227, v91 /*v347*/, v88 /*v344*/, v89 /*v345*/
	s_set_vgpr_msb 0x1540
	v_cndmask_b32_e32 v118 /*v374*/, 0xff800000, v228, vcc_lo
	s_set_vgpr_msb 0x4005
	v_cmp_le_i32_e32 vcc_lo, v252 /*v508*/, v97 /*v353*/
	s_set_vgpr_msb 0x500
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x32_bf16 v[210:217], v[218:225], v[72:79], v[210:217]
	s_set_vgpr_msb 21
	v_max3_num_f32 v228, v0 /*v256*/, v1 /*v257*/, v22 /*v278*/
	s_set_vgpr_msb 0x1540
	v_cndmask_b32_e32 v119 /*v375*/, 0xff800000, v229, vcc_lo
	s_set_vgpr_msb 0x4015
	v_cmp_le_i32_e32 vcc_lo, v253 /*v509*/, v97 /*v353*/
	v_max3_num_f32 v229, v86 /*v342*/, v87 /*v343*/, v100 /*v356*/
	s_set_vgpr_msb 0x1540
	v_cndmask_b32_e32 v120 /*v376*/, 0xff800000, v230, vcc_lo
	s_set_vgpr_msb 0x4005
	v_cmp_le_i32_e32 vcc_lo, v254 /*v510*/, v97 /*v353*/
	s_set_vgpr_msb 0x501
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[210:217], v[202:209] /*v[458:465]*/, v[104:111], v[210:217]
	s_set_vgpr_msb 0x115
	v_max3_num_f32 v230, v23 /*v279*/, v24 /*v280*/, v25 /*v281*/
	s_set_vgpr_msb 0x1540
	v_cndmask_b32_e32 v121 /*v377*/, 0xff800000, v231, vcc_lo
	s_set_vgpr_msb 0x4015
	v_cmp_le_i32_e32 vcc_lo, v255 /*v511*/, v97 /*v353*/
	v_max3_num_f32 v231, v101 /*v357*/, v102 /*v358*/, v103 /*v359*/
	v_max3_num_f32 v245, v119 /*v375*/, v120 /*v376*/, v121 /*v377*/
	s_set_vgpr_msb 0x1540
	v_cndmask_b32_e32 v122 /*v378*/, 0xff800000, v232, vcc_lo
	s_set_vgpr_msb 0x4006
	v_cmp_le_i32_e32 vcc_lo, v0 /*v512*/, v97 /*v353*/
	s_set_vgpr_msb 0x601
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x32_bf16 v[210:217], v[210:217] /*v[466:473]*/, v[128:135], v[210:217]
	s_set_vgpr_msb 0x115
	v_max3_num_f32 v232, v28 /*v284*/, v29 /*v285*/, v18 /*v274*/
	s_set_vgpr_msb 0x1540
	v_cndmask_b32_e32 v123 /*v379*/, 0xff800000, v233, vcc_lo
	s_set_vgpr_msb 0x4006
	v_cmp_le_i32_e32 vcc_lo, v1 /*v513*/, v96 /*v352*/
	s_set_vgpr_msb 0x615
	v_max_num_f32_e32 v246, v40 /*v296*/, v41 /*v297*/
	v_max3_num_f32 v233, v104 /*v360*/, v105 /*v361*/, v106 /*v362*/
	s_set_vgpr_msb 0x1501
	v_wmma_f32_16x16x32_bf16 v[234:241], v[194:201] /*v[450:457]*/, v[160:167], 0
	s_set_vgpr_msb 0x105
	v_max_num_f32_e32 v247, v122 /*v378*/, v123 /*v379*/
	s_set_vgpr_msb 0x540
	v_cndmask_b32_e32 v54 /*v310*/, 0xff800000, v210, vcc_lo
	s_set_vgpr_msb 0x4006
	v_cmp_le_i32_e32 vcc_lo, v2 /*v514*/, v96 /*v352*/
	s_set_vgpr_msb 0x600
	v_max3_num_f32 v210, v195, v196, v197
	s_set_vgpr_msb 64
	v_cndmask_b32_e32 v55 /*v311*/, 0xff800000, v211, vcc_lo
	s_set_vgpr_msb 0x4006
	v_cmp_le_i32_e32 vcc_lo, v3 /*v515*/, v96 /*v352*/
	s_set_vgpr_msb 0x600
	v_wmma_f32_16x16x32_bf16 v[234:241], v[218:225], v[168:175], v[234:241]
	s_set_vgpr_msb 21
	v_max3_num_f32 v211, v77 /*v333*/, v66 /*v322*/, v67 /*v323*/
	s_set_vgpr_msb 0x1540
	v_cndmask_b32_e32 v56 /*v312*/, 0xff800000, v212, vcc_lo
	s_set_vgpr_msb 0x4006
	v_cmp_le_i32_e32 vcc_lo, v4 /*v516*/, v96 /*v352*/
	s_set_vgpr_msb 0x600
	v_max3_num_f32 v212, v198, v199, v202
	v_max3_num_f32 v218, v205, v252, v253
	s_set_vgpr_msb 1
	v_wmma_f32_16x16x32_bf16 v[234:241], v[202:209] /*v[458:465]*/, v[176:183], v[234:241]
	s_set_vgpr_msb 0x115
	v_max3_num_f32 v220, v8 /*v264*/, v9 /*v265*/, v10 /*v266*/
	s_set_vgpr_msb 0x1540
	v_cndmask_b32_e32 v57 /*v313*/, 0xff800000, v213, vcc_lo
	s_set_vgpr_msb 0x4006
	v_cmp_le_i32_e32 vcc_lo, v5 /*v517*/, v96 /*v352*/
	s_set_vgpr_msb 0x615
	v_max3_num_f32 v213, v64 /*v320*/, v65 /*v321*/, v34 /*v290*/
	s_set_vgpr_msb 0x1501
	v_max3_num_f32 v222, v11 /*v267*/, v250, v251
	s_set_vgpr_msb 0x115
	v_max3_num_f32 v224, v4 /*v260*/, v5 /*v261*/, v12 /*v268*/
	s_set_vgpr_msb 0x1555
	v_max3_num_f32 v2 /*v258*/, v55 /*v311*/, v56 /*v312*/, v57 /*v313*/
	s_set_vgpr_msb 0x5540
	v_cndmask_b32_e32 v58 /*v314*/, 0xff800000, v214, vcc_lo
	s_set_vgpr_msb 0x4005
	v_cmp_le_i32_e32 vcc_lo, v36 /*v292*/, v96 /*v352*/
	s_set_vgpr_msb 0x501
	v_wmma_f32_16x16x32_bf16 v[234:241], v[210:217] /*v[466:473]*/, v[184:191], v[234:241]
	s_set_vgpr_msb 0x100
	v_max3_num_f32 v214, v203, v208, v209
	s_set_vgpr_msb 21
	v_max3_num_f32 v219, v69 /*v325*/, v62 /*v318*/, v63 /*v319*/
	v_max3_num_f32 v221, v84 /*v340*/, v85 /*v341*/, v78 /*v334*/
	s_set_vgpr_msb 0x1540
	v_cndmask_b32_e32 v59 /*v315*/, 0xff800000, v215, vcc_lo
	s_set_vgpr_msb 0x4015
	v_cmp_le_i32_e32 vcc_lo, v37 /*v293*/, v96 /*v352*/
	v_max3_num_f32 v215, v35 /*v291*/, v82 /*v338*/, v83 /*v339*/
	v_max3_num_f32 v223, v79 /*v335*/, v74 /*v330*/, v75 /*v331*/
	v_max3_num_f32 v225, v92 /*v348*/, v93 /*v349*/, v90 /*v346*/
	s_set_vgpr_msb 0x1500
	v_max3_num_f32 v200, v200, v210, v212
	s_set_vgpr_msb 64
	v_cndmask_b32_e32 v60 /*v316*/, 0xff800000, v216, vcc_lo
	s_set_vgpr_msb 0x4006
	v_cmp_le_i32_e32 vcc_lo, v6 /*v518*/, v96 /*v352*/
	s_set_vgpr_msb 0x600
	v_max3_num_f32 v216, v206, v207, v204
	v_max3_num_f32 v201, v201, v211, v213
	v_max3_num_f32 v211, v220, v222, v224
	v_max3_num_f32 v212, v226, v228, v230
	s_set_vgpr_msb 64
	v_cndmask_b32_e32 v61 /*v317*/, 0xff800000, v217, vcc_lo
	s_set_vgpr_msb 0x4006
	v_cmp_le_i32_e32 vcc_lo, v1 /*v513*/, v97 /*v353*/
	s_set_vgpr_msb 0x615
	v_max3_num_f32 v217, v70 /*v326*/, v71 /*v327*/, v68 /*v324*/
	s_set_vgpr_msb 0x1555
	v_max3_num_f32 v6 /*v262*/, v58 /*v314*/, v59 /*v315*/, v60 /*v316*/
	s_set_vgpr_msb 0x5500
	v_max3_num_f32 v210, v214, v216, v218
	s_set_vgpr_msb 20
	v_max3_num_f32 v216, v246, v54 /*v310*/, v2 /*v258*/
	s_set_vgpr_msb 0x1440
	v_cndmask_b32_e32 v124 /*v380*/, 0xff800000, v234, vcc_lo
	s_set_vgpr_msb 0x4006
	v_cmp_le_i32_e32 vcc_lo, v2 /*v514*/, v97 /*v353*/
	s_set_vgpr_msb 0x601
	v_max3_num_f32 v234, v19 /*v275*/, v248, v249
	s_set_vgpr_msb 0x100
	v_max3_num_f32 v215, v215, v217, v219
	v_max3_num_f32 v217, v221, v223, v225
	v_max3_num_f32 v200, v200, v210, v211
	s_set_vgpr_msb 64
	v_cndmask_b32_e32 v125 /*v381*/, 0xff800000, v235, vcc_lo
	s_set_vgpr_msb 0x4006
	v_cmp_le_i32_e32 vcc_lo, v3 /*v515*/, v97 /*v353*/
	s_set_vgpr_msb 0x615
	v_max3_num_f32 v235, v107 /*v363*/, v108 /*v364*/, v109 /*v365*/
	s_set_vgpr_msb 0x1514
	v_max3_num_f32 v211, v216, v6 /*v262*/, v61 /*v317*/
	s_set_vgpr_msb 0x1400
	v_max3_num_f32 v201, v201, v215, v217
	s_set_vgpr_msb 64
	v_cndmask_b32_e32 v126 /*v382*/, 0xff800000, v236, vcc_lo
	s_set_vgpr_msb 0x4006
	v_cmp_le_i32_e32 vcc_lo, v4 /*v516*/, v97 /*v353*/
	s_set_vgpr_msb 0x615
	v_max3_num_f32 v236, v32 /*v288*/, v33 /*v289*/, v38 /*v294*/
	s_set_vgpr_msb 0x1540
	v_cndmask_b32_e32 v127 /*v383*/, 0xff800000, v237, vcc_lo
	s_set_vgpr_msb 0x4006
	v_cmp_le_i32_e32 vcc_lo, v5 /*v517*/, v97 /*v353*/
	s_set_vgpr_msb 0x615
	v_max3_num_f32 v237, v110 /*v366*/, v111 /*v367*/, v112 /*v368*/
	s_set_vgpr_msb 0x1500
	v_max3_num_f32 v213, v232, v234, v236
	s_set_vgpr_msb 0x55
	v_max3_num_f32 v3 /*v259*/, v125 /*v381*/, v126 /*v382*/, v127 /*v383*/
	s_set_vgpr_msb 0x5540
	v_cndmask_b32_e32 v128 /*v384*/, 0xff800000, v238, vcc_lo
	s_set_vgpr_msb 0x4005
	v_cmp_le_i32_e32 vcc_lo, v36 /*v292*/, v97 /*v353*/
	s_set_vgpr_msb 0x501
	v_max3_num_f32 v238, v39 /*v295*/, v242, v243
	s_set_vgpr_msb 0x114
	v_max3_num_f32 v216, v247, v124 /*v380*/, v3 /*v259*/
	s_set_vgpr_msb 0x1440
	v_cndmask_b32_e32 v129 /*v385*/, 0xff800000, v239, vcc_lo
	s_set_vgpr_msb 0x4015
	v_cmp_le_i32_e32 vcc_lo, v37 /*v293*/, v97 /*v353*/
	v_max3_num_f32 v239, v113 /*v369*/, v114 /*v370*/, v115 /*v371*/
	s_set_vgpr_msb 0x1540
	v_cndmask_b32_e32 v130 /*v386*/, 0xff800000, v240, vcc_lo
	s_set_vgpr_msb 0x4006
	v_cmp_le_i32_e32 vcc_lo, v6 /*v518*/, v97 /*v353*/
	s_set_vgpr_msb 0x615
	v_max3_num_f32 v240, v44 /*v300*/, v45 /*v301*/, v46 /*v302*/
	s_set_vgpr_msb 0x1540
	v_cndmask_b32_e32 v131 /*v387*/, 0xff800000, v241, vcc_lo
	s_set_vgpr_msb 0x4015
	v_max3_num_f32 v241, v116 /*v372*/, v117 /*v373*/, v118 /*v374*/
	s_set_vgpr_msb 0x1500
	v_max3_num_f32 v214, v238, v240, v244
	s_set_vgpr_msb 0x55
	v_max3_num_f32 v7 /*v263*/, v128 /*v384*/, v129 /*v385*/, v130 /*v386*/
	s_set_vgpr_msb 0x5500
	v_max3_num_f32 v210, v212, v213, v214
	v_max3_num_f32 v212, v227, v229, v231
	v_max3_num_f32 v213, v233, v235, v237
	v_max3_num_f32 v214, v239, v241, v245
	v_max3_num_f32 v200, v200, v210, v211
	s_set_vgpr_msb 20
	v_max3_num_f32 v211, v216, v7 /*v263*/, v131 /*v387*/
	s_set_vgpr_msb 0x1400
	v_max3_num_f32 v210, v212, v213, v214
	v_max3_num_f32 v201, v201, v210, v211
	v_dual_mov_b32 v212, v200 :: v_dual_mov_b32 v210, v201
	v_permlanex16_b32 v212, v212, s64, 0xfedcba98
	v_permlanex16_b32 v210, v210, s64, 0xfedcba98
	v_dual_max_num_f32 v200, v200, v212 :: v_dual_max_num_f32 v201, v201, v210
	s_set_vgpr_msb 4
	v_sub_f32_e32 v211, v200, v99 /*v355*/
	v_max_num_f32_e32 v200, v200, v99 /*v355*/
	s_set_vgpr_msb 0x408
	v_sub_f32_e32 v210, v201, v203 /*v715*/
	s_set_vgpr_msb 0x800
	v_cmp_lt_f32_e32 vcc_lo, 0x41000000, v211
	s_cmp_eq_u32 vcc_lo, 0
	s_cselect_b32 s2, -1, 0
	s_set_vgpr_msb 0x84
	v_cndmask_b32_e64 v198 /*v710*/, v200, v99 /*v355*/, s2
	s_set_vgpr_msb 0x8402
	v_cmp_lt_f32_e64 s2, 0x41000000, v210
	v_max_num_f32_e32 v200, v203 /*v715*/, v201
	s_set_vgpr_msb 0x248
	v_mul_f32_e32 v72 /*v328*/, 0xbfb8aa3b, v198 /*v710*/
	s_cmp_lg_u32 s2, 0
	s_cselect_b32 s3, -1, 0
	s_cmp_eq_u32 s2, 0
	s_set_vgpr_msb 0x4810
	v_pk_fma_f32 v[192:193], v[192:193], s[28:29], v[72:73] /*v[328:329]*/ op_sel_hi:[1,0,0]
	s_cselect_b32 s2, -1, 0
	v_pk_fma_f32 v[210:211], v[196:197], s[28:29], v[72:73] /*v[328:329]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1048
	v_cndmask_b32_e64 v98 /*v354*/, v200, v203 /*v715*/, s2
	s_set_vgpr_msb 0x4810
	v_pk_fma_f32 v[200:201], v[194:195], s[28:29], v[72:73] /*v[328:329]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v194, v193
	v_pk_fma_f32 v[208:209], v[208:209], s[28:29], v[72:73] /*v[328:329]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[206:207], v[206:207], s[28:29], v[72:73] /*v[328:329]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1044
	v_mul_f32_e32 v132 /*v388*/, 0xbfb8aa3b, v98 /*v354*/
	s_set_vgpr_msb 0x4410
	v_pk_fma_f32 v[204:205], v[204:205], s[28:29], v[72:73] /*v[328:329]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v196, v200
	v_exp_f32_e32 v200, v210
	v_pk_fma_f32 v[214:215], v[202:203], s[28:29], v[72:73] /*v[328:329]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[254:255], v[254:255], s[28:29], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1051
	v_pk_fma_f32 v[62:63] /*v[318:319]*/, v[62:63] /*v[318:319]*/, s[28:29], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[34:35] /*v[290:291]*/, v[34:35] /*v[290:291]*/, s[28:29], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5100
	v_exp_f32_e32 v202, v211
	s_set_vgpr_msb 64
	v_exp_f32_e32 v20 /*v276*/, v208
	s_set_vgpr_msb 0x4000
	v_exp_f32_e32 v193, v254
	v_exp_f32_e32 v195, v255
	v_nop
	s_set_vgpr_msb 17
	v_pk_fma_f32 v[254:255], v[64:65] /*v[320:321]*/, s[28:29], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1140
	v_exp_f32_e32 v26 /*v282*/, v209
	v_nop
	s_set_vgpr_msb 0x4010
	v_pk_fma_f32 v[208:209], v[252:253], s[28:29], v[72:73] /*v[328:329]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1040
	v_exp_f32_e32 v36 /*v292*/, v207
	s_set_vgpr_msb 0x4011
	v_pk_fma_f32 v[210:211], v[8:9] /*v[264:265]*/, s[28:29], v[72:73] /*v[328:329]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1100
	v_exp_f32_e32 v245, v254
	s_set_vgpr_msb 64
	v_exp_f32_e32 v3 /*v259*/, v255
	v_nop
	s_set_vgpr_msb 0x4011
	v_pk_fma_f32 v[254:255], v[70:71] /*v[326:327]*/, s[28:29], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1140
	v_exp_f32_e32 v48 /*v304*/, v205
	s_set_vgpr_msb 0x4001
	v_exp_f32_e32 v205, v62 /*v318*/
	v_exp_f32_e32 v207, v63 /*v319*/
	v_nop
	s_set_vgpr_msb 0x151
	v_pk_fma_f32 v[62:63] /*v[318:319]*/, v[74:75] /*v[330:331]*/, s[28:29], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5140
	v_exp_f32_e32 v31 /*v287*/, v254
	v_exp_f32_e32 v37 /*v293*/, v255
	v_nop
	s_set_vgpr_msb 0x4011
	v_pk_fma_f32 v[254:255], v[84:85] /*v[340:341]*/, s[28:29], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1110
	v_pk_fma_f32 v[212:213], v[198:199], s[28:29], v[72:73] /*v[328:329]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1051
	v_exp_f32_e32 v7 /*v263*/, v34 /*v290*/
	v_exp_f32_e32 v17 /*v273*/, v35 /*v291*/
	v_nop
	v_pk_fma_f32 v[34:35] /*v[290:291]*/, v[68:69] /*v[324:325]*/, s[28:29], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5140
	v_exp_f32_e32 v30 /*v286*/, v206
	v_exp_f32_e32 v42 /*v298*/, v204
	s_set_vgpr_msb 0x4010
	v_exp_f32_e32 v204, v208
	v_exp_f32_e32 v206, v209
	v_exp_f32_e32 v208, v210
	v_pk_fma_f32 v[216:217], v[250:251], s[28:29], v[72:73] /*v[328:329]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v210, v211
	s_set_vgpr_msb 0x1011
	v_pk_fma_f32 v[220:221], v[4:5] /*v[260:261]*/, s[28:29], v[72:73] /*v[328:329]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[222:223], v[12:13] /*v[268:269]*/, s[28:29], v[72:73] /*v[328:329]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1100
	v_exp_f32_e32 v209, v254
	v_exp_f32_e32 v211, v255
	v_nop
	s_set_vgpr_msb 17
	v_pk_fma_f32 v[254:255], v[92:93] /*v[348:349]*/, s[28:29], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v219, v62 /*v318*/
	v_exp_f32_e32 v225, v63 /*v319*/
	v_nop
	s_set_vgpr_msb 0x1151
	v_pk_fma_f32 v[62:63] /*v[318:319]*/, v[88:89] /*v[344:345]*/, s[28:29], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5100
	v_exp_f32_e32 v244, v212
	s_set_vgpr_msb 64
	v_exp_f32_e32 v2 /*v258*/, v213
	v_nop
	s_set_vgpr_msb 0x4011
	v_pk_fma_f32 v[212:213], v[10:11] /*v[266:267]*/, s[28:29], v[72:73] /*v[328:329]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1151
	v_exp_f32_e32 v43 /*v299*/, v34 /*v290*/
	v_exp_f32_e32 v49 /*v305*/, v35 /*v291*/
	v_nop
	v_pk_fma_f32 v[34:35] /*v[290:291]*/, v[78:79] /*v[334:335]*/, s[28:29], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5100
	v_exp_f32_e32 v218, v216
	v_exp_f32_e32 v224, v217
	v_exp_f32_e32 v226, v220
	s_set_vgpr_msb 17
	v_pk_fma_f32 v[216:217], v[14:15] /*v[270:271]*/, s[28:29], v[72:73] /*v[328:329]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1100
	v_exp_f32_e32 v232, v221
	v_exp_f32_e32 v234, v222
	s_set_vgpr_msb 17
	v_pk_fma_f32 v[220:221], v[0:1] /*v[256:257]*/, s[28:29], v[72:73] /*v[328:329]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1100
	v_exp_f32_e32 v238, v223
	v_nop
	s_set_vgpr_msb 17
	v_pk_fma_f32 v[222:223], v[22:23] /*v[278:279]*/, s[28:29], v[72:73] /*v[328:329]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[228:229], v[24:25] /*v[280:281]*/, s[28:29], v[72:73] /*v[328:329]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1100
	v_exp_f32_e32 v227, v254
	v_exp_f32_e32 v233, v255
	v_nop
	s_set_vgpr_msb 17
	v_pk_fma_f32 v[254:255], v[86:87] /*v[342:343]*/, s[28:29], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v241, v62 /*v318*/
	v_exp_f32_e32 v247, v63 /*v319*/
	v_nop
	s_set_vgpr_msb 0x1151
	v_pk_fma_f32 v[62:63] /*v[318:319]*/, v[102:103] /*v[358:359]*/, s[28:29], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5140
	v_exp_f32_e32 v6 /*v262*/, v214
	v_exp_f32_e32 v16 /*v272*/, v215
	s_set_vgpr_msb 0x4000
	v_exp_f32_e32 v214, v213
	s_set_vgpr_msb 1
	v_exp_f32_e32 v213, v34 /*v290*/
	v_exp_f32_e32 v215, v35 /*v291*/
	v_nop
	s_set_vgpr_msb 0x151
	v_pk_fma_f32 v[34:35] /*v[290:291]*/, v[90:91] /*v[346:347]*/, s[28:29], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5100
	v_exp_f32_e32 v240, v216
	v_exp_f32_e32 v250, v220
	v_exp_f32_e32 v216, v222
	s_set_vgpr_msb 17
	v_pk_fma_f32 v[230:231], v[28:29] /*v[284:285]*/, s[28:29], v[72:73] /*v[328:329]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1100
	v_exp_f32_e32 v220, v223
	v_exp_f32_e32 v222, v228
	v_exp_f32_e32 v228, v229
	v_exp_f32_e32 v251, v254
	s_set_vgpr_msb 64
	v_exp_f32_e32 v9 /*v265*/, v255
	v_nop
	s_set_vgpr_msb 0x4011
	v_pk_fma_f32 v[254:255], v[104:105] /*v[360:361]*/, s[28:29], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v223, v62 /*v318*/
	v_exp_f32_e32 v229, v63 /*v319*/
	v_nop
	s_set_vgpr_msb 0x1151
	v_pk_fma_f32 v[62:63] /*v[318:319]*/, v[108:109] /*v[364:365]*/, s[28:29], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5101
	v_exp_f32_e32 v235, v34 /*v290*/
	v_exp_f32_e32 v239, v35 /*v291*/
	v_nop
	s_set_vgpr_msb 0x151
	v_pk_fma_f32 v[34:35] /*v[290:291]*/, v[100:101] /*v[356:357]*/, s[28:29], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5111
	v_pk_fma_f32 v[252:253], v[18:19] /*v[274:275]*/, s[28:29], v[72:73] /*v[328:329]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1150
	v_pk_fma_f32 v[0:1] /*v[256:257]*/, v[248:249], s[28:29], v[72:73] /*v[328:329]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5000
	v_exp_f32_e32 v236, v231
	v_exp_f32_e32 v231, v254
	v_exp_f32_e32 v237, v255
	v_nop
	s_set_vgpr_msb 17
	v_pk_fma_f32 v[254:255], v[110:111] /*v[366:367]*/, s[28:29], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1151
	v_exp_f32_e32 v11 /*v267*/, v62 /*v318*/
	v_exp_f32_e32 v19 /*v275*/, v63 /*v319*/
	v_nop
	v_pk_fma_f32 v[62:63] /*v[318:319]*/, v[114:115] /*v[370:371]*/, s[28:29], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5100
	v_exp_f32_e32 v246, v217
	s_set_vgpr_msb 64
	v_exp_f32_e32 v8 /*v264*/, v221
	s_set_vgpr_msb 0x4051
	v_pk_fma_f32 v[136:137] /*v[392:393]*/, v[76:77] /*v[332:333]*/, s[28:29], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[66:67] /*v[322:323]*/, v[66:67] /*v[322:323]*/, s[28:29], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5101
	v_exp_f32_e32 v217, v34 /*v290*/
	v_exp_f32_e32 v221, v35 /*v291*/
	v_nop
	s_set_vgpr_msb 0x151
	v_pk_fma_f32 v[34:35] /*v[290:291]*/, v[106:107] /*v[362:363]*/, s[28:29], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5100
	v_exp_f32_e32 v248, v252
	s_set_vgpr_msb 64
	v_exp_f32_e32 v4 /*v260*/, v253
	s_set_vgpr_msb 0x4041
	v_exp_f32_e32 v10 /*v266*/, v0 /*v256*/
	s_set_vgpr_msb 0x4111
	v_pk_fma_f32 v[252:253], v[38:39] /*v[294:295]*/, s[28:29], v[72:73] /*v[328:329]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1141
	v_exp_f32_e32 v18 /*v274*/, v1 /*v257*/
	s_set_vgpr_msb 0x4110
	v_pk_fma_f32 v[242:243], v[242:243], s[28:29], v[72:73] /*v[328:329]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1051
	v_pk_fma_f32 v[0:1] /*v[256:257]*/, v[44:45] /*v[300:301]*/, s[28:29], v[72:73] /*v[328:329]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[14:15] /*v[270:271]*/, v[52:53] /*v[308:309]*/, s[28:29], v[72:73] /*v[328:329]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5140
	v_exp_f32_e32 v23 /*v279*/, v254
	v_exp_f32_e32 v29 /*v285*/, v255
	v_nop
	s_set_vgpr_msb 0x4011
	v_pk_fma_f32 v[254:255], v[116:117] /*v[372:373]*/, s[28:29], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1151
	v_exp_f32_e32 v45 /*v301*/, v62 /*v318*/
	v_exp_f32_e32 v51 /*v307*/, v63 /*v319*/
	v_nop
	v_pk_fma_f32 v[62:63] /*v[318:319]*/, v[120:121] /*v[376:377]*/, s[28:29], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5100
	v_exp_f32_e32 v192, v192
	v_exp_f32_e32 v198, v201
	s_set_vgpr_msb 0x51
	v_pk_fma_f32 v[12:13] /*v[268:269]*/, v[32:33] /*v[288:289]*/, s[28:29], v[72:73] /*v[328:329]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5101
	v_exp_f32_e32 v199, v137 /*v393*/
	v_exp_f32_e32 v201, v66 /*v322*/
	s_set_vgpr_msb 0x151
	v_pk_fma_f32 v[64:65] /*v[320:321]*/, v[82:83] /*v[338:339]*/, s[28:29], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5101
	v_exp_f32_e32 v249, v34 /*v290*/
	s_set_vgpr_msb 0x151
	v_exp_f32_e32 v5 /*v261*/, v35 /*v291*/
	v_nop
	v_pk_fma_f32 v[34:35] /*v[290:291]*/, v[112:113] /*v[368:369]*/, s[28:29], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5140
	v_exp_f32_e32 v38 /*v294*/, v253
	v_exp_f32_e32 v50 /*v306*/, v243
	s_set_vgpr_msb 0x4051
	v_pk_fma_f32 v[40:41] /*v[296:297]*/, v[40:41] /*v[296:297]*/, s[28:29], v[72:73] /*v[328:329]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v24 /*v280*/, v15 /*v271*/
	v_pk_fma_f32 v[56:57] /*v[312:313]*/, v[56:57] /*v[312:313]*/, s[28:29], v[72:73] /*v[328:329]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5100
	v_exp_f32_e32 v243, v254
	v_exp_f32_e32 v253, v255
	v_nop
	s_set_vgpr_msb 17
	v_pk_fma_f32 v[254:255], v[122:123] /*v[378:379]*/, s[28:29], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1151
	v_exp_f32_e32 v15 /*v271*/, v62 /*v318*/
	v_exp_f32_e32 v25 /*v281*/, v63 /*v319*/
	v_nop
	v_pk_fma_f32 v[62:63] /*v[318:319]*/, v[126:127] /*v[382:383]*/, s[28:29], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5100
	v_exp_f32_e32 v230, v230
	s_set_vgpr_msb 0x51
	v_exp_f32_e32 v22 /*v278*/, v12 /*v268*/
	v_exp_f32_e32 v28 /*v284*/, v13 /*v269*/
	v_nop
	v_pk_fma_f32 v[12:13] /*v[268:269]*/, v[46:47] /*v[302:303]*/, s[28:29], v[72:73] /*v[328:329]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5101
	v_exp_f32_e32 v197, v136 /*v392*/
	v_exp_f32_e32 v203, v67 /*v323*/
	s_set_vgpr_msb 0x151
	v_exp_f32_e32 v21 /*v277*/, v64 /*v320*/
	v_exp_f32_e32 v33 /*v289*/, v34 /*v290*/
	v_exp_f32_e32 v39 /*v295*/, v35 /*v291*/
	v_nop
	v_pk_fma_f32 v[34:35] /*v[290:291]*/, v[118:119] /*v[374:375]*/, s[28:29], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v46 /*v302*/, v41 /*v297*/
	v_pk_fma_f32 v[80:81] /*v[336:337]*/, v[58:59] /*v[314:315]*/, s[28:29], v[72:73] /*v[328:329]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v58 /*v314*/, v57 /*v313*/
	v_exp_f32_e32 v27 /*v283*/, v65 /*v321*/
	s_set_vgpr_msb 0x5140
	v_exp_f32_e32 v41 /*v297*/, v254
	v_exp_f32_e32 v47 /*v303*/, v255
	s_set_vgpr_msb 0x4041
	v_exp_f32_e32 v57 /*v313*/, v62 /*v318*/
	s_set_vgpr_msb 0x4111
	v_pk_fma_f32 v[254:255], v[128:129] /*v[384:385]*/, s[28:29], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x1141
	v_exp_f32_e32 v59 /*v315*/, v63 /*v319*/
	v_nop
	s_set_vgpr_msb 0x4140
	v_pk_add_f32 v[62:63] /*v[318:319]*/, v[192:193], v[194:195]
	v_pk_add_f32 v[64:65] /*v[320:321]*/, v[198:199], v[200:201]
	v_exp_f32_e32 v32 /*v288*/, v252
	v_exp_f32_e32 v44 /*v300*/, v242
	s_set_vgpr_msb 0x4001
	v_exp_f32_e32 v242, v0 /*v256*/
	v_exp_f32_e32 v252, v1 /*v257*/
	s_set_vgpr_msb 0x151
	v_exp_f32_e32 v0 /*v256*/, v12 /*v268*/
	v_exp_f32_e32 v12 /*v268*/, v13 /*v269*/
	v_pk_fma_f32 v[52:53] /*v[308:309]*/, v[54:55] /*v[310:311]*/, s[28:29], v[72:73] /*v[328:329]*/ op_sel_hi:[1,0,0]
	v_exp_f32_e32 v1 /*v257*/, v34 /*v290*/
	v_exp_f32_e32 v13 /*v269*/, v35 /*v291*/
	v_nop
	v_pk_fma_f32 v[34:35] /*v[290:291]*/, v[124:125] /*v[380:381]*/, s[28:29], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_pk_fma_f32 v[134:135] /*v[390:391]*/, v[60:61] /*v[316:317]*/, s[28:29], v[72:73] /*v[328:329]*/ op_sel_hi:[1,0,0]
	s_set_vgpr_msb 0x5140
	v_exp_f32_e32 v61 /*v317*/, v254
	v_exp_f32_e32 v73 /*v329*/, v255
	v_nop
	s_set_vgpr_msb 0x4004
	v_pk_add_f32 v[254:255], v[196:197], v[62:63] /*v[318:319]*/
	s_set_vgpr_msb 0x444
	v_pk_add_f32 v[62:63] /*v[318:319]*/, v[202:203], v[64:65] /*v[320:321]*/
	v_pk_add_f32 v[64:65] /*v[320:321]*/, v[244:245], v[2:3] /*v[258:259]*/
	s_set_vgpr_msb 0x4445
	v_pk_add_f32 v[66:67] /*v[322:323]*/, v[16:17] /*v[272:273]*/, v[20:21] /*v[276:277]*/
	v_pk_add_f32 v[68:69] /*v[324:325]*/, v[30:31] /*v[286:287]*/, v[36:37] /*v[292:293]*/
	s_set_vgpr_msb 0x4540
	v_pk_add_f32 v[84:85] /*v[340:341]*/, v[238:239], v[240:241]
	s_set_vgpr_msb 0x4044
	v_pk_add_f32 v[86:87] /*v[342:343]*/, v[250:251], v[8:9] /*v[264:265]*/
	s_set_vgpr_msb 0x4440
	v_pk_add_f32 v[90:91] /*v[346:347]*/, v[230:231], v[236:237]
	s_set_vgpr_msb 0x4045
	v_pk_add_f32 v[92:93] /*v[348:349]*/, v[4:5] /*v[260:261]*/, v[10:11] /*v[266:267]*/
	s_set_vgpr_msb 0x4500
	v_exp_f32_e32 v212, v212
	s_set_vgpr_msb 0x41
	v_exp_f32_e32 v14 /*v270*/, v14 /*v270*/
	v_exp_f32_e32 v40 /*v296*/, v40 /*v296*/
	v_exp_f32_e32 v54 /*v310*/, v53 /*v309*/
	v_exp_f32_e32 v56 /*v312*/, v56 /*v312*/
	v_exp_f32_e32 v55 /*v311*/, v35 /*v291*/
	v_pk_add_f32 v[70:71] /*v[326:327]*/, v[48:49] /*v[304:305]*/, v[204:205]
	s_set_vgpr_msb 0x4140
	v_pk_add_f32 v[74:75] /*v[330:331]*/, v[208:209], v[210:211]
	s_set_vgpr_msb 0x4045
	v_pk_add_f32 v[64:65] /*v[320:321]*/, v[6:7] /*v[262:263]*/, v[64:65] /*v[320:321]*/
	v_pk_add_f32 v[66:67] /*v[322:323]*/, v[26:27] /*v[282:283]*/, v[66:67] /*v[322:323]*/
	v_pk_add_f32 v[68:69] /*v[324:325]*/, v[42:43] /*v[298:299]*/, v[68:69] /*v[324:325]*/
	s_set_vgpr_msb 0x4540
	v_pk_add_f32 v[78:79] /*v[334:335]*/, v[214:215], v[218:219]
	v_pk_add_f32 v[88:89] /*v[344:345]*/, v[220:221], v[222:223]
	s_set_vgpr_msb 0x4044
	v_pk_add_f32 v[84:85] /*v[340:341]*/, v[246:247], v[84:85] /*v[340:341]*/
	v_pk_add_f32 v[86:87] /*v[342:343]*/, v[216:217], v[86:87] /*v[342:343]*/
	s_set_vgpr_msb 0x4445
	v_pk_add_f32 v[100:101] /*v[356:357]*/, v[22:23] /*v[278:279]*/, v[28:29] /*v[284:285]*/
	v_pk_add_f32 v[102:103] /*v[358:359]*/, v[38:39] /*v[294:295]*/, v[44:45] /*v[300:301]*/
	s_set_vgpr_msb 0x4540
	v_pk_add_f32 v[104:105] /*v[360:361]*/, v[242:243], v[252:253]
	s_set_vgpr_msb 0x4044
	v_pk_add_f32 v[90:91] /*v[346:347]*/, v[248:249], v[90:91] /*v[346:347]*/
	s_set_vgpr_msb 0x4445
	v_pk_add_f32 v[92:93] /*v[348:349]*/, v[18:19] /*v[274:275]*/, v[92:93] /*v[348:349]*/
	s_set_vgpr_msb 0x4504
	v_pk_add_f32 v[254:255], v[254:255], v[62:63] /*v[318:319]*/
	s_set_vgpr_msb 0x451
	v_exp_f32_e32 v52 /*v308*/, v52 /*v308*/
	v_exp_f32_e32 v60 /*v316*/, v80 /*v336*/
	v_exp_f32_e32 v72 /*v328*/, v81 /*v337*/
	v_exp_f32_e32 v53 /*v309*/, v34 /*v290*/
	v_nop
	v_pk_fma_f32 v[34:35] /*v[290:291]*/, v[130:131] /*v[386:387]*/, s[28:29], v[132:133] /*v[388:389]*/ op_sel_hi:[1,0,0]
	v_pk_add_f32 v[70:71] /*v[326:327]*/, v[70:71] /*v[326:327]*/, v[206:207]
	v_pk_add_f32 v[74:75] /*v[330:331]*/, v[74:75] /*v[330:331]*/, v[212:213]
	s_set_vgpr_msb 0x5140
	v_pk_add_f32 v[82:83] /*v[338:339]*/, v[226:227], v[232:233]
	s_set_vgpr_msb 0x4044
	v_pk_add_f32 v[78:79] /*v[334:335]*/, v[224:225], v[78:79] /*v[334:335]*/
	v_pk_add_f32 v[88:89] /*v[344:345]*/, v[228:229], v[88:89] /*v[344:345]*/
	s_set_vgpr_msb 0x4445
	v_pk_add_f32 v[100:101] /*v[356:357]*/, v[32:33] /*v[288:289]*/, v[100:101] /*v[356:357]*/
	v_pk_add_f32 v[102:103] /*v[358:359]*/, v[50:51] /*v[306:307]*/, v[102:103] /*v[358:359]*/
	v_pk_add_f32 v[104:105] /*v[360:361]*/, v[0:1] /*v[256:257]*/, v[104:105] /*v[360:361]*/
	v_pk_add_f32 v[106:107] /*v[362:363]*/, v[12:13] /*v[268:269]*/, v[14:15] /*v[270:271]*/
	v_pk_add_f32 v[108:109] /*v[364:365]*/, v[40:41] /*v[296:297]*/, v[46:47] /*v[302:303]*/
	v_pk_add_f32 v[110:111] /*v[366:367]*/, v[54:55] /*v[310:311]*/, v[56:57] /*v[312:313]*/
	s_set_vgpr_msb 0x4501
	v_pk_add_f32 v[254:255], v[64:65] /*v[320:321]*/, v[254:255]
	s_set_vgpr_msb 0x145
	v_pk_add_f32 v[64:65] /*v[320:321]*/, v[66:67] /*v[322:323]*/, v[68:69] /*v[324:325]*/
	v_pk_add_f32 v[66:67] /*v[322:323]*/, v[84:85] /*v[340:341]*/, v[86:87] /*v[342:343]*/
	v_pk_add_f32 v[68:69] /*v[324:325]*/, v[90:91] /*v[346:347]*/, v[92:93] /*v[348:349]*/
	v_exp_f32_e32 v80 /*v336*/, v134 /*v390*/
	v_exp_f32_e32 v81 /*v337*/, v34 /*v290*/
	s_set_vgpr_msb 0x4544
	v_pk_add_f32 v[82:83] /*v[338:339]*/, v[234:235], v[82:83] /*v[338:339]*/
	s_set_vgpr_msb 0x4445
	v_pk_add_f32 v[112:113] /*v[368:369]*/, v[60:61] /*v[316:317]*/, v[72:73] /*v[328:329]*/
	v_pk_add_f32 v[62:63] /*v[318:319]*/, v[24:25] /*v[280:281]*/, v[106:107] /*v[362:363]*/
	v_pk_add_f32 v[106:107] /*v[362:363]*/, v[52:53] /*v[308:309]*/, v[108:109] /*v[364:365]*/
	v_pk_add_f32 v[108:109] /*v[364:365]*/, v[58:59] /*v[314:315]*/, v[110:111] /*v[366:367]*/
	v_pk_add_f32 v[74:75] /*v[330:331]*/, v[74:75] /*v[330:331]*/, v[78:79] /*v[334:335]*/
	v_pk_add_f32 v[78:79] /*v[334:335]*/, v[102:103] /*v[358:359]*/, v[104:105] /*v[360:361]*/
	v_pk_add_f32 v[64:65] /*v[320:321]*/, v[70:71] /*v[326:327]*/, v[64:65] /*v[320:321]*/
	v_pk_add_f32 v[66:67] /*v[322:323]*/, v[88:89] /*v[344:345]*/, v[66:67] /*v[322:323]*/
	v_pk_add_f32 v[68:69] /*v[324:325]*/, v[100:101] /*v[356:357]*/, v[68:69] /*v[324:325]*/
	v_pk_add_f32 v[110:111] /*v[366:367]*/, v[80:81] /*v[336:337]*/, v[112:113] /*v[368:369]*/
	v_pk_add_f32 v[70:71] /*v[326:327]*/, v[82:83] /*v[338:339]*/, v[74:75] /*v[330:331]*/
	v_pk_add_f32 v[62:63] /*v[318:319]*/, v[62:63] /*v[318:319]*/, v[78:79] /*v[334:335]*/
	v_pk_add_f32 v[74:75] /*v[330:331]*/, v[106:107] /*v[362:363]*/, v[108:109] /*v[364:365]*/
	s_set_vgpr_msb 0x4504
	v_pk_add_f32 v[254:255], v[254:255], v[64:65] /*v[320:321]*/
	s_set_vgpr_msb 0x445
	v_pk_add_f32 v[64:65] /*v[320:321]*/, v[66:67] /*v[322:323]*/, v[68:69] /*v[324:325]*/
	v_exp_f32_e32 v76 /*v332*/, v135 /*v391*/
	v_exp_f32_e32 v77 /*v333*/, v35 /*v291*/
	v_nop
	v_pk_add_f32 v[34:35] /*v[290:291]*/, v[110:111] /*v[366:367]*/, v[74:75] /*v[330:331]*/
	s_set_vgpr_msb 0x4501
	v_pk_add_f32 v[254:255], v[70:71] /*v[326:327]*/, v[254:255]
	s_set_vgpr_msb 0x145
	v_pk_add_f32 v[62:63] /*v[318:319]*/, v[62:63] /*v[318:319]*/, v[64:65] /*v[320:321]*/
	v_pk_add_f32 v[64:65] /*v[320:321]*/, v[76:77] /*v[332:333]*/, v[34:35] /*v[290:291]*/
	s_set_vgpr_msb 0x4504
	v_pk_add_f32 v[254:255], v[254:255], v[62:63] /*v[318:319]*/
	s_set_vgpr_msb 0x449
	v_sub_f32_e32 v34 /*v290*/, v99 /*v355*/, v198 /*v710*/
	s_set_vgpr_msb 0x4942
	v_mov_b32_e32 v35 /*v291*/, v200 /*v712*/
	s_set_vgpr_msb 0x4282
	v_dual_mov_b32 v200 /*v712*/, v197 /*v709*/ :: v_dual_mov_b32 v197 /*v709*/, v199 /*v711*/
	s_set_vgpr_msb 0x8201
	v_pk_add_f32 v[254:255], v[64:65] /*v[320:321]*/, v[254:255]
	s_set_vgpr_msb 0x146
	v_dual_mul_f32 v34 /*v290*/, 0x3fb8aa3b, v34 /*v290*/ :: v_dual_mov_b32 v63 /*v319*/, v195 /*v707*/
	s_wait_alu depctr_vm_vsrc(6)
	s_set_vgpr_msb 0x4682
	v_dual_mov_b32 v195 /*v707*/, v201 /*v713*/ :: v_dual_mov_b32 v201 /*v713*/, v204 /*v716*/
	s_set_vgpr_msb 0x8240
	v_dual_mov_b32 v64 /*v320*/, v254 :: v_dual_mov_b32 v65 /*v321*/, v255
	s_set_vgpr_msb 0x4041
	v_exp_f32_e32 v34 /*v290*/, v34 /*v290*/
	v_permlanex16_b32 v64 /*v320*/, v64 /*v320*/, s64, 0xfedcba98
	v_permlanex16_b32 v65 /*v321*/, v65 /*v321*/, s64, 0xfedcba98
	s_set_vgpr_msb 0x4100
	s_cbranch_vccz .LBB0_27
	s_set_vgpr_msb 4
	v_pk_mul_f32 v[158:159], v[158:159], v[34:35] /*v[290:291]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[156:157], v[156:157], v[34:35] /*v[290:291]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[154:155], v[154:155], v[34:35] /*v[290:291]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[152:153], v[152:153], v[34:35] /*v[290:291]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[150:151], v[150:151], v[34:35] /*v[290:291]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[148:149], v[148:149], v[34:35] /*v[290:291]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[146:147], v[146:147], v[34:35] /*v[290:291]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[144:145], v[144:145], v[34:35] /*v[290:291]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[142:143], v[142:143], v[34:35] /*v[290:291]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[140:141], v[140:141], v[34:35] /*v[290:291]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[138:139], v[138:139], v[34:35] /*v[290:291]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[136:137], v[136:137], v[34:35] /*v[290:291]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[126:127], v[126:127], v[34:35] /*v[290:291]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[124:125], v[124:125], v[34:35] /*v[290:291]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[122:123], v[122:123], v[34:35] /*v[290:291]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[120:121], v[120:121], v[34:35] /*v[290:291]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[118:119], v[118:119], v[34:35] /*v[290:291]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[116:117], v[116:117], v[34:35] /*v[290:291]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[114:115], v[114:115], v[34:35] /*v[290:291]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[112:113], v[112:113], v[34:35] /*v[290:291]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[102:103], v[102:103], v[34:35] /*v[290:291]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[100:101], v[100:101], v[34:35] /*v[290:291]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[98:99], v[98:99], v[34:35] /*v[290:291]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[96:97], v[96:97], v[34:35] /*v[290:291]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[94:95], v[94:95], v[34:35] /*v[290:291]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[92:93], v[92:93], v[34:35] /*v[290:291]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[90:91], v[90:91], v[34:35] /*v[290:291]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[88:89], v[88:89], v[34:35] /*v[290:291]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[86:87], v[86:87], v[34:35] /*v[290:291]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[84:85], v[84:85], v[34:35] /*v[290:291]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[82:83], v[82:83], v[34:35] /*v[290:291]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[80:81], v[80:81], v[34:35] /*v[290:291]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x400
.LBB0_27:
	s_set_vgpr_msb 0x46
	v_sub_f32_e32 v62 /*v318*/, v203 /*v715*/, v98 /*v354*/
	s_and_b32 s2, s3, exec_lo
	s_cselect_b32 s2, 1, 0
	s_cmp_lg_u32 s2, 1
	v_mul_f32_e32 v62 /*v318*/, 0x3fb8aa3b, v62 /*v318*/
	s_set_vgpr_msb 0x4641
	v_exp_f32_e32 v62 /*v318*/, v62 /*v318*/
	s_set_vgpr_msb 0x4100
	s_cbranch_scc1 .LBB0_29
	v_nop
	s_set_vgpr_msb 4
	v_pk_mul_f32 v[62:63], v[62:63], v[62:63] /*v[318:319]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[60:61], v[60:61], v[62:63] /*v[318:319]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59], v[58:59], v[62:63] /*v[318:319]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[56:57], v[56:57], v[62:63] /*v[318:319]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[54:55], v[54:55], v[62:63] /*v[318:319]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[52:53], v[52:53], v[62:63] /*v[318:319]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[50:51], v[50:51], v[62:63] /*v[318:319]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[48:49], v[48:49], v[62:63] /*v[318:319]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[46:47], v[46:47], v[62:63] /*v[318:319]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[44:45], v[44:45], v[62:63] /*v[318:319]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[42:43], v[42:43], v[62:63] /*v[318:319]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[40:41], v[40:41], v[62:63] /*v[318:319]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39], v[38:39], v[62:63] /*v[318:319]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[36:37], v[36:37], v[62:63] /*v[318:319]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35], v[34:35], v[62:63] /*v[318:319]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[32:33], v[32:33], v[62:63] /*v[318:319]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[30:31], v[30:31], v[62:63] /*v[318:319]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[28:29], v[28:29], v[62:63] /*v[318:319]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[26:27], v[26:27], v[62:63] /*v[318:319]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25], v[24:25], v[62:63] /*v[318:319]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[22:23], v[22:23], v[62:63] /*v[318:319]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21], v[20:21], v[62:63] /*v[318:319]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[18:19], v[18:19], v[62:63] /*v[318:319]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[16:17], v[16:17], v[62:63] /*v[318:319]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[14:15], v[62:63] /*v[318:319]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13], v[12:13], v[62:63] /*v[318:319]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[10:11], v[62:63] /*v[318:319]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[8:9], v[62:63] /*v[318:319]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[6:7], v[62:63] /*v[318:319]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[4:5], v[62:63] /*v[318:319]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[2:3], v[2:3], v[62:63] /*v[318:319]*/ op_sel_hi:[1,0]
	v_pk_mul_f32 v[0:1], v[0:1], v[62:63] /*v[318:319]*/ op_sel_hi:[1,0]
	s_set_vgpr_msb 0x400
.LBB0_29:
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x42
	ds_load_tr16_b128 v[82:85] /*v[338:341]*/, v196 /*v708*/
	ds_load_tr16_b128 v[100:103] /*v[356:359]*/, v196 /*v708*/ offset:32
	ds_load_tr16_b128 v[86:89] /*v[342:345]*/, v196 /*v708*/ offset:4608
	ds_load_tr16_b128 v[104:107] /*v[360:363]*/, v196 /*v708*/ offset:4640
	ds_load_tr16_b128 v[108:111] /*v[364:367]*/, v196 /*v708*/ offset:9216
	ds_load_tr16_b128 v[116:119] /*v[372:375]*/, v196 /*v708*/ offset:9248
	ds_load_tr16_b128 v[112:115] /*v[368:371]*/, v196 /*v708*/ offset:13824
	ds_load_tr16_b128 v[120:123] /*v[376:379]*/, v196 /*v708*/ offset:13856
	ds_load_tr16_b128 v[124:127] /*v[380:383]*/, v196 /*v708*/ offset:18432
	ds_load_tr16_b128 v[132:135] /*v[388:391]*/, v196 /*v708*/ offset:18464
	ds_load_tr16_b128 v[128:131] /*v[384:387]*/, v196 /*v708*/ offset:23040
	ds_load_tr16_b128 v[136:139] /*v[392:395]*/, v196 /*v708*/ offset:23072
	ds_load_tr16_b128 v[140:143] /*v[396:399]*/, v196 /*v708*/ offset:27648
	ds_load_tr16_b128 v[148:151] /*v[404:407]*/, v196 /*v708*/ offset:27680
	ds_load_tr16_b128 v[144:147] /*v[400:403]*/, v196 /*v708*/ offset:32256
	ds_load_tr16_b128 v[152:155] /*v[408:411]*/, v196 /*v708*/ offset:32288
	s_set_vgpr_msb 0x4245
	v_cvt_pk_bf16_f32 v163 /*v419*/, v42 /*v298*/, v48 /*v304*/
	v_cvt_pk_bf16_f32 v162 /*v418*/, v30 /*v286*/, v36 /*v292*/
	v_cvt_pk_bf16_f32 v161 /*v417*/, v20 /*v276*/, v26 /*v282*/
	v_cvt_pk_bf16_f32 v160 /*v416*/, v6 /*v262*/, v16 /*v272*/
	s_set_vgpr_msb 0x4544
	v_cvt_pk_bf16_f32 v159 /*v415*/, v244, v2 /*v258*/
	s_set_vgpr_msb 0x4440
	v_cvt_pk_bf16_f32 v158 /*v414*/, v200, v202
	v_cvt_pk_bf16_f32 v157 /*v413*/, v196, v198
	v_cvt_pk_bf16_f32 v156 /*v412*/, v192, v194
	s_set_vgpr_msb 0x4045
	v_cvt_pk_bf16_f32 v179 /*v435*/, v43 /*v299*/, v49 /*v305*/
	v_cvt_pk_bf16_f32 v178 /*v434*/, v31 /*v287*/, v37 /*v293*/
	v_cvt_pk_bf16_f32 v177 /*v433*/, v21 /*v277*/, v27 /*v283*/
	v_cvt_pk_bf16_f32 v176 /*v432*/, v7 /*v263*/, v17 /*v273*/
	s_set_vgpr_msb 0x4544
	v_cvt_pk_bf16_f32 v175 /*v431*/, v245, v3 /*v259*/
	s_set_vgpr_msb 0x4440
	v_cvt_pk_bf16_f32 v174 /*v430*/, v201, v203
	v_cvt_pk_bf16_f32 v173 /*v429*/, v197, v199
	v_cvt_pk_bf16_f32 v172 /*v428*/, v193, v195
	s_set_vgpr_msb 0x4005
	s_wait_dscnt 0xd
	v_wmma_f32_16x16x32_bf16 v[152:159], v[82:89] /*v[338:345]*/, v[156:163] /*v[412:419]*/, v[152:159]
	s_set_vgpr_msb 0x540
	v_cvt_pk_bf16_f32 v165 /*v421*/, v208, v210
	s_set_vgpr_msb 0x4000
	v_cvt_pk_bf16_f32 v194, v230, v236
	v_cvt_pk_bf16_f32 v193, v222, v228
	v_cvt_pk_bf16_f32 v210, v231, v237
	s_set_vgpr_msb 64
	v_cvt_pk_bf16_f32 v169 /*v425*/, v234, v238
	v_cvt_pk_bf16_f32 v168 /*v424*/, v226, v232
	v_cvt_pk_bf16_f32 v167 /*v423*/, v218, v224
	s_set_vgpr_msb 0x4005
	v_wmma_f32_16x16x32_bf16 v[56:63], v[82:89] /*v[338:345]*/, v[172:179] /*v[428:435]*/, v[56:63]
	s_set_vgpr_msb 0x540
	v_cvt_pk_bf16_f32 v170 /*v426*/, v240, v246
	s_set_vgpr_msb 0x4000
	v_cvt_pk_bf16_f32 v192, v216, v220
	v_cvt_pk_bf16_f32 v200, v242, v252
	v_cvt_pk_bf16_f32 v216, v243, v253
	s_set_vgpr_msb 64
	v_cvt_pk_bf16_f32 v83 /*v339*/, v209, v211
	s_set_vgpr_msb 0x4000
	v_cvt_pk_bf16_f32 v209, v223, v229
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 2
	ds_load_tr16_b128 v[228:231], v196 /*v708*/ offset:4672
	s_set_vgpr_msb 0x240
	v_cvt_pk_bf16_f32 v87 /*v343*/, v235, v239
	v_cvt_pk_bf16_f32 v86 /*v342*/, v227, v233
	v_cvt_pk_bf16_f32 v85 /*v341*/, v219, v225
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x4002
	ds_load_tr16_b128 v[224:227], v196 /*v708*/ offset:64
	ds_load_tr16_b128 v[232:235], v196 /*v708*/ offset:96
	ds_load_tr16_b128 v[236:239], v196 /*v708*/ offset:4704
	s_set_vgpr_msb 0x240
	v_cvt_pk_bf16_f32 v88 /*v344*/, v241, v247
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x4002
	ds_load_tr16_b128 v[244:247], v196 /*v708*/ offset:13888
	s_set_vgpr_msb 0x204
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x32_bf16 v[136:143], v[224:231], v[156:163] /*v[412:419]*/, v[136:143]
	s_set_vgpr_msb 0x444
	v_cvt_pk_bf16_f32 v171 /*v427*/, v250, v8 /*v264*/
	s_set_vgpr_msb 0x4440
	v_cvt_pk_bf16_f32 v166 /*v422*/, v212, v214
	v_cvt_pk_bf16_f32 v164 /*v420*/, v204, v206
	s_set_vgpr_msb 0x4044
	v_cvt_pk_bf16_f32 v89 /*v345*/, v251, v9 /*v265*/
	s_set_vgpr_msb 0x4440
	v_cvt_pk_bf16_f32 v84 /*v340*/, v213, v215
	v_cvt_pk_bf16_f32 v82 /*v338*/, v205, v207
	s_set_vgpr_msb 0x4004
	v_cvt_pk_bf16_f32 v195, v248, v4 /*v260*/
	v_wmma_f32_16x16x32_bf16 v[40:47], v[224:231], v[172:179] /*v[428:435]*/, v[40:47]
	s_set_vgpr_msb 0x402
	ds_load_tr16_b128 v[240:243], v196 /*v708*/ offset:9280
	s_wait_alu depctr_va_vdst(0)
	ds_load_tr16_b128 v[224:227], v196 /*v708*/ offset:9312
	ds_load_tr16_b128 v[228:231], v196 /*v708*/ offset:13920
	s_set_vgpr_msb 0x204
	v_cvt_pk_bf16_f32 v211, v249, v5 /*v261*/
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x402
	ds_load_tr16_b128 v[248:251], v196 /*v708*/ offset:23104
	s_set_vgpr_msb 0x200
	v_cvt_pk_bf16_f32 v208, v217, v221
	s_set_vgpr_msb 5
	v_cvt_pk_bf16_f32 v201, v0 /*v256*/, v12 /*v268*/
	v_cvt_pk_bf16_f32 v217, v1 /*v257*/, v13 /*v269*/
	v_cvt_pk_bf16_f32 v199, v44 /*v300*/, v50 /*v306*/
	s_set_vgpr_msb 0x504
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x32_bf16 v[136:143], v[240:247], v[164:171] /*v[420:427]*/, v[136:143]
	s_set_vgpr_msb 0x405
	v_cvt_pk_bf16_f32 v198, v32 /*v288*/, v38 /*v294*/
	v_cvt_pk_bf16_f32 v197, v22 /*v278*/, v28 /*v284*/
	v_cvt_pk_bf16_f32 v196, v10 /*v266*/, v18 /*v274*/
	v_cvt_pk_bf16_f32 v215, v45 /*v301*/, v51 /*v307*/
	v_cvt_pk_bf16_f32 v214, v33 /*v289*/, v39 /*v295*/
	v_cvt_pk_bf16_f32 v213, v23 /*v279*/, v29 /*v285*/
	v_cvt_pk_bf16_f32 v212, v11 /*v267*/, v19 /*v275*/
	s_set_vgpr_msb 0x504
	v_wmma_f32_16x16x32_bf16 v[40:47], v[240:247], v[82:89] /*v[338:345]*/, v[40:47]
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x402
	ds_load_tr16_b128 v[244:247], v196 /*v708*/ offset:18496
	s_set_vgpr_msb 0x242
	ds_load_tr16_b128 v[0:3] /*v[256:259]*/, v196 /*v708*/ offset:18528
	ds_load_tr16_b128 v[4:7] /*v[260:263]*/, v196 /*v708*/ offset:23136
	s_set_vgpr_msb 0x4205
	v_cvt_pk_bf16_f32 v202, v14 /*v270*/, v24 /*v280*/
	v_cvt_pk_bf16_f32 v218, v15 /*v271*/, v25 /*v281*/
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x542
	ds_load_tr16_b128 v[12:15] /*v[268:271]*/, v196 /*v708*/ offset:32320
	s_set_vgpr_msb 0x4205
	v_cvt_pk_bf16_f32 v207, v80 /*v336*/, v76 /*v332*/
	v_cvt_pk_bf16_f32 v206, v60 /*v316*/, v72 /*v328*/
	v_cvt_pk_bf16_f32 v205, v56 /*v312*/, v58 /*v314*/
	s_set_vgpr_msb 0x504
	v_wmma_f32_16x16x32_bf16 v[120:127], v[232:239], v[156:163] /*v[412:419]*/, v[120:127]
	s_set_vgpr_msb 0x405
	v_cvt_pk_bf16_f32 v204, v52 /*v308*/, v54 /*v310*/
	v_cvt_pk_bf16_f32 v203, v40 /*v296*/, v46 /*v302*/
	v_cvt_pk_bf16_f32 v223, v81 /*v337*/, v77 /*v333*/
	v_cvt_pk_bf16_f32 v222, v61 /*v317*/, v73 /*v329*/
	v_cvt_pk_bf16_f32 v221, v57 /*v313*/, v59 /*v315*/
	v_cvt_pk_bf16_f32 v220, v53 /*v309*/, v55 /*v311*/
	v_cvt_pk_bf16_f32 v219, v41 /*v297*/, v47 /*v303*/
	s_set_vgpr_msb 0x504
	v_wmma_f32_16x16x32_bf16 v[32:39], v[232:239], v[172:179] /*v[428:435]*/, v[32:39]
	s_add_co_i32 s2, s46, 4
	s_cmp_ge_i32 s2, s44
	s_set_vgpr_msb 0x400
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x32_bf16 v[136:143], v[244:251], v[192:199], v[136:143]
	v_wmma_f32_16x16x32_bf16 v[40:47], v[244:251], v[208:215], v[40:47]
	s_set_vgpr_msb 0x42
	ds_load_tr16_b128 v[8:11] /*v[264:267]*/, v196 /*v708*/ offset:27712
	s_set_vgpr_msb 0x4202
	ds_load_tr16_b128 v[240:243], v196 /*v708*/ offset:27744
	s_wait_alu depctr_va_vdst(0)
	ds_load_tr16_b128 v[244:247], v196 /*v708*/ offset:32352
	ds_load_tr16_b128 v[248:251], v196 /*v708*/ offset:23168
	s_set_vgpr_msb 0x204
	v_wmma_f32_16x16x32_bf16 v[120:127], v[224:231], v[164:171] /*v[420:427]*/, v[120:127]
	v_wmma_f32_16x16x32_bf16 v[32:39], v[224:231], v[82:89] /*v[338:345]*/, v[32:39]
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x402
	ds_load_tr16_b128 v[228:231], v196 /*v708*/ offset:4736
	ds_load_tr16_b128 v[224:227], v196 /*v708*/ offset:128
	ds_load_tr16_b128 v[232:235], v196 /*v708*/ offset:160
	ds_load_tr16_b128 v[236:239], v196 /*v708*/ offset:4768
	s_set_vgpr_msb 0x201
	s_wait_dscnt 0x9
	v_wmma_f32_16x16x32_bf16 v[120:127], v[0:7] /*v[256:263]*/, v[192:199], v[120:127]
	v_wmma_f32_16x16x32_bf16 v[32:39], v[0:7] /*v[256:263]*/, v[208:215], v[32:39]
	s_set_vgpr_msb 0x100
	s_wait_dscnt 0x5
	v_wmma_f32_16x16x32_bf16 v[120:127], v[240:247], v[200:207], v[120:127]
	v_wmma_f32_16x16x32_bf16 v[32:39], v[240:247], v[216:223], v[32:39]
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 2
	ds_load_tr16_b128 v[244:247], v196 /*v708*/ offset:13952
	s_set_vgpr_msb 0x204
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x32_bf16 v[112:119], v[224:231], v[156:163] /*v[412:419]*/, v[112:119]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[224:231], v[172:179] /*v[428:435]*/, v[24:31]
	s_set_vgpr_msb 0x402
	ds_load_tr16_b128 v[240:243], v196 /*v708*/ offset:9344
	s_wait_alu depctr_va_vdst(0)
	ds_load_tr16_b128 v[224:227], v196 /*v708*/ offset:9376
	ds_load_tr16_b128 v[228:231], v196 /*v708*/ offset:13984
	s_set_vgpr_msb 0x204
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[112:119], v[240:247], v[164:171] /*v[420:427]*/, v[112:119]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[240:247], v[82:89] /*v[338:345]*/, v[24:31]
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x402
	ds_load_tr16_b128 v[244:247], v196 /*v708*/ offset:18560
	s_set_vgpr_msb 0x242
	ds_load_tr16_b128 v[0:3] /*v[256:259]*/, v196 /*v708*/ offset:18592
	ds_load_tr16_b128 v[4:7] /*v[260:263]*/, v196 /*v708*/ offset:23200
	s_set_vgpr_msb 0x4204
	v_wmma_f32_16x16x32_bf16 v[96:103], v[232:239], v[156:163] /*v[412:419]*/, v[96:103]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[232:239], v[172:179] /*v[428:435]*/, v[16:23]
	s_set_vgpr_msb 0x401
	v_wmma_f32_16x16x32_bf16 v[136:143], v[8:15] /*v[264:271]*/, v[200:207], v[136:143]
	v_wmma_f32_16x16x32_bf16 v[40:47], v[8:15] /*v[264:271]*/, v[216:223], v[40:47]
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x142
	ds_load_tr16_b128 v[12:15] /*v[268:271]*/, v196 /*v708*/ offset:32384
	s_set_vgpr_msb 0x4200
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x32_bf16 v[112:119], v[244:251], v[192:199], v[112:119]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[244:251], v[208:215], v[24:31]
	s_set_vgpr_msb 0x42
	ds_load_tr16_b128 v[8:11] /*v[264:267]*/, v196 /*v708*/ offset:27776
	s_set_vgpr_msb 0x4202
	ds_load_tr16_b128 v[240:243], v196 /*v708*/ offset:27808
	s_wait_alu depctr_va_vdst(0)
	ds_load_tr16_b128 v[244:247], v196 /*v708*/ offset:32416
	ds_load_tr16_b128 v[248:251], v196 /*v708*/ offset:23232
	s_set_vgpr_msb 0x204
	v_wmma_f32_16x16x32_bf16 v[96:103], v[224:231], v[164:171] /*v[420:427]*/, v[96:103]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[224:231], v[82:89] /*v[338:345]*/, v[16:23]
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x402
	ds_load_tr16_b128 v[228:231], v196 /*v708*/ offset:4800
	ds_load_tr16_b128 v[224:227], v196 /*v708*/ offset:192
	ds_load_tr16_b128 v[232:235], v196 /*v708*/ offset:224
	ds_load_tr16_b128 v[236:239], v196 /*v708*/ offset:4832
	s_set_vgpr_msb 0x201
	s_wait_dscnt 0x9
	v_wmma_f32_16x16x32_bf16 v[96:103], v[0:7] /*v[256:263]*/, v[192:199], v[96:103]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[0:7] /*v[256:263]*/, v[208:215], v[16:23]
	s_set_vgpr_msb 0x100
	s_wait_dscnt 0x5
	v_wmma_f32_16x16x32_bf16 v[96:103], v[240:247], v[200:207], v[96:103]
	v_wmma_f32_16x16x32_bf16 v[16:23], v[240:247], v[216:223], v[16:23]
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 2
	ds_load_tr16_b128 v[244:247], v196 /*v708*/ offset:14016
	s_set_vgpr_msb 0x204
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x32_bf16 v[88:95], v[224:231], v[156:163] /*v[412:419]*/, v[88:95]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[224:231], v[172:179] /*v[428:435]*/, v[8:15]
	s_set_vgpr_msb 0x402
	ds_load_tr16_b128 v[240:243], v196 /*v708*/ offset:9408
	s_wait_alu depctr_va_vdst(0)
	ds_load_tr16_b128 v[224:227], v196 /*v708*/ offset:9440
	ds_load_tr16_b128 v[228:231], v196 /*v708*/ offset:14048
	s_set_vgpr_msb 0x204
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x32_bf16 v[88:95], v[240:247], v[164:171] /*v[420:427]*/, v[88:95]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[240:247], v[82:89] /*v[338:345]*/, v[8:15]
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x402
	ds_load_tr16_b128 v[244:247], v196 /*v708*/ offset:18624
	s_set_vgpr_msb 0x242
	ds_load_tr16_b128 v[0:3] /*v[256:259]*/, v196 /*v708*/ offset:18656
	ds_load_tr16_b128 v[4:7] /*v[260:263]*/, v196 /*v708*/ offset:23264
	s_set_vgpr_msb 0x4205
	v_wmma_f32_16x16x32_bf16 v[144:151], v[100:107] /*v[356:363]*/, v[156:163] /*v[412:419]*/, v[144:151]
	v_wmma_f32_16x16x32_bf16 v[48:55], v[100:107] /*v[356:363]*/, v[172:179] /*v[428:435]*/, v[48:55]
	s_set_vgpr_msb 0x504
	v_wmma_f32_16x16x32_bf16 v[80:87], v[232:239], v[156:163] /*v[412:419]*/, v[80:87]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[232:239], v[172:179] /*v[428:435]*/, v[0:7]
	s_set_vgpr_msb 0x401
	v_wmma_f32_16x16x32_bf16 v[112:119], v[8:15] /*v[264:271]*/, v[200:207], v[112:119]
	v_wmma_f32_16x16x32_bf16 v[24:31], v[8:15] /*v[264:271]*/, v[216:223], v[24:31]
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x142
	ds_load_tr16_b128 v[12:15] /*v[268:271]*/, v196 /*v708*/ offset:32448
	s_set_vgpr_msb 0x4200
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x32_bf16 v[88:95], v[244:251], v[192:199], v[88:95]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[244:251], v[208:215], v[8:15]
	s_set_vgpr_msb 0x42
	ds_load_tr16_b128 v[8:11] /*v[264:267]*/, v196 /*v708*/ offset:27840
	s_set_vgpr_msb 0x4202
	ds_load_tr16_b128 v[240:243], v196 /*v708*/ offset:27872
	s_wait_alu depctr_va_vdst(0)
	ds_load_tr16_b128 v[244:247], v196 /*v708*/ offset:32480
	s_wait_tensorcnt 0x0
	s_set_vgpr_msb 0x205
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	v_wmma_f32_16x16x32_bf16 v[152:159], v[108:115] /*v[364:371]*/, v[164:171] /*v[420:427]*/, v[152:159]
	v_wmma_f32_16x16x32_bf16 v[56:63], v[108:115] /*v[364:371]*/, v[82:89] /*v[338:345]*/, v[56:63]
	v_wmma_f32_16x16x32_bf16 v[144:151], v[116:123] /*v[372:379]*/, v[164:171] /*v[420:427]*/, v[144:151]
	v_wmma_f32_16x16x32_bf16 v[48:55], v[116:123] /*v[372:379]*/, v[82:89] /*v[338:345]*/, v[48:55]
	s_set_vgpr_msb 0x504
	v_wmma_f32_16x16x32_bf16 v[80:87], v[224:231], v[164:171] /*v[420:427]*/, v[80:87]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[224:231], v[82:89] /*v[338:345]*/, v[0:7]
	s_set_vgpr_msb 0x401
	v_wmma_f32_16x16x32_bf16 v[152:159], v[124:131] /*v[380:387]*/, v[192:199], v[152:159]
	v_wmma_f32_16x16x32_bf16 v[56:63], v[124:131] /*v[380:387]*/, v[208:215], v[56:63]
	v_wmma_f32_16x16x32_bf16 v[144:151], v[132:139] /*v[388:395]*/, v[192:199], v[144:151]
	v_wmma_f32_16x16x32_bf16 v[48:55], v[132:139] /*v[388:395]*/, v[208:215], v[48:55]
	v_wmma_f32_16x16x32_bf16 v[80:87], v[0:7] /*v[256:263]*/, v[192:199], v[80:87]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[0:7] /*v[256:263]*/, v[208:215], v[0:7]
	v_wmma_f32_16x16x32_bf16 v[152:159], v[140:147] /*v[396:403]*/, v[200:207], v[152:159]
	v_wmma_f32_16x16x32_bf16 v[56:63], v[140:147] /*v[396:403]*/, v[216:223], v[56:63]
	v_wmma_f32_16x16x32_bf16 v[144:151], v[148:155] /*v[404:411]*/, v[200:207], v[144:151]
	v_wmma_f32_16x16x32_bf16 v[48:55], v[148:155] /*v[404:411]*/, v[216:223], v[48:55]
	v_wmma_f32_16x16x32_bf16 v[88:95], v[8:15] /*v[264:271]*/, v[200:207], v[88:95]
	v_wmma_f32_16x16x32_bf16 v[8:15], v[8:15] /*v[264:271]*/, v[216:223], v[8:15]
	s_set_vgpr_msb 0x100
	v_wmma_f32_16x16x32_bf16 v[80:87], v[240:247], v[200:207], v[80:87]
	v_wmma_f32_16x16x32_bf16 v[0:7], v[240:247], v[216:223], v[0:7]
	s_cbranch_scc1 .LBB0_24
	v_med3_i32 v192, s50, 0, 0x80
	s_add_co_i32 s2, s52, s48
	s_and_b32 s4, s46, 3
	s_ashr_i32 s3, s2, 31
	s_mul_i32 s6, s4, 0x11800
	v_readfirstlane_b32 s7, v192
	s_mul_u64 s[4:5], s[2:3], s[30:31]
	s_mul_u64 s[2:3], s[2:3], s[56:57]
	s_lshl_b64 s[4:5], s[4:5], 1
	s_lshl_b64 s[2:3], s[2:3], 1
	s_add_nc_u64 s[4:5], s[60:61], s[4:5]
	s_sub_co_i32 s7, s7, s9
	s_add_nc_u64 s[42:43], s[10:11], s[4:5]
	s_max_i32 s4, s7, 0
	s_add_nc_u64 s[2:3], s[62:63], s[2:3]
	s_lshl_b32 s4, s4, 16
	s_add_co_i32 s41, s35, s6
	s_bitset1_b32 s43, 31
	s_or_b32 s14, s4, 0x7fff
	s_mov_b32 s21, s13
	tensor_load_to_lds s[40:43], s[12:19]
	s_add_nc_u64 s[42:43], s[58:59], s[2:3]
	s_add_co_i32 s41, s55, s6
	s_bitset1_b32 s43, 31
	s_mov_b32 s22, s14
	s_mov_b32 s23, s15
	s_mov_b32 s24, s16
	s_mov_b32 s27, s19
	tensor_load_to_lds s[40:43], s[20:27]
	s_branch .LBB0_24
.LBB0_31:
	s_set_vgpr_msb 0x42
	v_mov_b32_e32 v98 /*v354*/, v203 /*v715*/
	s_set_vgpr_msb 0x4200
.LBB0_32:
	s_set_vgpr_msb 10
	v_div_scale_f32 v64, null, v120 /*v632*/, v120 /*v632*/, 1.0
	v_div_scale_f32 v67, vcc_lo, 1.0, v120 /*v632*/, 1.0
	v_cmp_lt_f32_e64 s2, 0, v120 /*v632*/
	v_div_scale_f32 v160, null, v121 /*v633*/, v121 /*v633*/, 1.0
	s_mov_b32 s4, 0
	s_set_vgpr_msb 0xa00
	v_rcp_f32_e32 v65, v64
	s_wait_dscnt 0x0
	v_rcp_f32_e32 v161, v160
	v_fma_f32 v66, -v64, v65, 1.0
	v_fmac_f32_e32 v65, v66, v65
	v_mul_f32_e32 v66, v67, v65
	v_fma_f32 v68, -v64, v66, v67
	v_fmac_f32_e32 v66, v68, v65
	v_fma_f32 v64, -v64, v66, v67
	v_fma_f32 v67, -v160, v161, 1.0
	v_fmac_f32_e32 v161, v67, v161
	v_div_fmas_f32 v64, v64, v65, v66
	s_set_vgpr_msb 8
	v_div_scale_f32 v162, vcc_lo, 1.0, v121 /*v633*/, 1.0
	v_div_fixup_f32 v64, v64, v120 /*v632*/, 1.0
	s_set_vgpr_msb 0x800
	v_dual_mul_f32 v163, v162, v161 :: v_dual_cndmask_b32 v64, 0, v64, s2
	v_fma_f32 v164, -v160, v163, v162
	v_pk_mul_f32 v[106:107], v[136:137], v[64:65] op_sel_hi:[1,0]
	v_fmac_f32_e32 v163, v164, v161
	v_pk_mul_f32 v[102:103], v[64:65], v[102:103] op_sel_hi:[0,1]
	v_pk_mul_f32 v[136:137], v[64:65], v[84:85] op_sel_hi:[0,1]
	v_pk_mul_f32 v[96:97], v[64:65], v[96:97] op_sel_hi:[0,1]
	v_pk_mul_f32 v[130:131], v[64:65], v[88:89] op_sel_hi:[0,1]
	v_fma_f32 v84, -v160, v163, v162
	v_pk_mul_f32 v[88:89], v[64:65], v[90:91] op_sel_hi:[0,1]
	v_pk_mul_f32 v[90:91], v[64:65], v[94:95] op_sel_hi:[0,1]
	v_pk_mul_f32 v[94:95], v[64:65], v[86:87] op_sel_hi:[0,1]
	v_cvt_pk_bf16_f32 v87, v102, v103
	v_div_fmas_f32 v102, v84, v161, v163
	v_cvt_pk_bf16_f32 v84, v96, v97
	s_set_vgpr_msb 8
	v_cmp_lt_f32_e32 vcc_lo, 0, v121 /*v633*/
	s_set_vgpr_msb 0x800
	v_pk_mul_f32 v[68:69], v[152:153], v[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[70:71], v[154:155], v[64:65] op_sel_hi:[1,0]
	s_set_vgpr_msb 8
	v_div_fixup_f32 v96, v102, v121 /*v633*/, 1.0
	s_set_vgpr_msb 0x800
	v_pk_mul_f32 v[72:73], v[156:157], v[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[66:67], v[158:159], v[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[74:75], v[144:145], v[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[104:105], v[150:151], v[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[108:109], v[138:139], v[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[110:111], v[140:141], v[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[112:113], v[64:65], v[112:113] op_sel_hi:[0,1]
	v_pk_mul_f32 v[114:115], v[64:65], v[114:115] op_sel_hi:[0,1]
	v_pk_mul_f32 v[98:99], v[64:65], v[98:99] op_sel_hi:[0,1]
	v_pk_mul_f32 v[100:101], v[64:65], v[100:101] op_sel_hi:[0,1]
	v_cndmask_b32_e32 v96, 0, v96, vcc_lo
	v_pk_mul_f32 v[76:77], v[146:147], v[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[78:79], v[148:149], v[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[128:129], v[142:143], v[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[120:121], v[120:121], v[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[122:123], v[122:123], v[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[124:125], v[124:125], v[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[126:127], v[126:127], v[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[116:117], v[64:65], v[116:117] op_sel_hi:[0,1]
	v_pk_mul_f32 v[118:119], v[64:65], v[118:119] op_sel_hi:[0,1]
	v_pk_mul_f32 v[92:93], v[64:65], v[92:93] op_sel_hi:[0,1]
	v_pk_mul_f32 v[132:133], v[64:65], v[80:81] op_sel_hi:[0,1]
	v_pk_mul_f32 v[134:135], v[64:65], v[82:83] op_sel_hi:[0,1]
	v_cvt_pk_bf16_f32 v67, v66, v67
	v_cvt_pk_bf16_f32 v66, v72, v73
	v_cvt_pk_bf16_f32 v65, v70, v71
	v_cvt_pk_bf16_f32 v64, v68, v69
	v_cvt_pk_bf16_f32 v71, v104, v105
	v_cvt_pk_bf16_f32 v68, v74, v75
	v_cvt_pk_bf16_f32 v74, v110, v111
	v_cvt_pk_bf16_f32 v73, v108, v109
	v_cvt_pk_bf16_f32 v72, v106, v107
	v_cvt_pk_bf16_f32 v81, v114, v115
	v_cvt_pk_bf16_f32 v80, v112, v113
	v_cvt_pk_bf16_f32 v86, v100, v101
	v_cvt_pk_bf16_f32 v85, v98, v99
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
	v_cvt_pk_bf16_f32 v70, v78, v79
	v_cvt_pk_bf16_f32 v69, v76, v77
	v_cvt_pk_bf16_f32 v75, v128, v129
	v_cvt_pk_bf16_f32 v79, v126, v127
	v_cvt_pk_bf16_f32 v78, v124, v125
	v_cvt_pk_bf16_f32 v77, v122, v123
	v_cvt_pk_bf16_f32 v76, v120, v121
	v_cvt_pk_bf16_f32 v83, v118, v119
	v_cvt_pk_bf16_f32 v82, v116, v117
	v_cvt_pk_bf16_f32 v91, v90, v91
	v_cvt_pk_bf16_f32 v90, v92, v93
	v_cvt_pk_bf16_f32 v89, v88, v89
	v_cvt_pk_bf16_f32 v88, v130, v131
	v_cvt_pk_bf16_f32 v95, v94, v95
	v_cvt_pk_bf16_f32 v94, v136, v137
	v_cvt_pk_bf16_f32 v93, v134, v135
	v_cvt_pk_bf16_f32 v92, v132, v133
	s_set_vgpr_msb 2
	v_add3_u32 v116, v192 /*v704*/, s53, 0x47100
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
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 2
	ds_store_b128 v191 /*v703*/, v[64:67]
	ds_store_b128 v191 /*v703*/, v[68:71] offset:32
	ds_store_b128 v191 /*v703*/, v[72:75] offset:64
	ds_store_b128 v191 /*v703*/, v[76:79] offset:96
	ds_store_b128 v191 /*v703*/, v[80:83] offset:128
	ds_store_b128 v191 /*v703*/, v[84:87] offset:160
	ds_store_b128 v191 /*v703*/, v[88:91] offset:192
	ds_store_b128 v191 /*v703*/, v[92:95] offset:224
	s_set_vgpr_msb 0x200
	ds_store_b128 v116, v[0:3]
	ds_store_b128 v116, v[4:7] offset:32
	ds_store_b128 v116, v[8:11] offset:64
	ds_store_b128 v116, v[12:15] offset:96
	ds_store_b128 v116, v[16:19] offset:128
	ds_store_b128 v116, v[20:23] offset:160
	ds_store_b128 v116, v[24:27] offset:192
	ds_store_b128 v116, v[28:31] offset:224
	s_cmp_lt_i32 s51, 1
	s_wait_dscnt 0x0
	s_cbranch_scc1 .LBB0_34
	s_wait_alu depctr_vm_vsrc(6)
	s_set_vgpr_msb 8
	v_dual_lshlrev_b32 v0, 3, v190 /*v702*/ :: v_dual_bitop2_b32 v2, 30, v193 /*v705*/ bitop3:0x54
	v_or_b32_e32 v4, 24, v193 /*v705*/
	s_wait_alu depctr_vm_vsrc(4)
	v_or_b32_e32 v13, 18, v193 /*v705*/
	s_add_co_i32 s51, s51, -1
	v_dual_mov_b32 v1, 0 :: v_dual_bitop2_b32 v3, 28, v193 /*v705*/ bitop3:0x54
	s_min_u32 s5, s51, 31
	v_or_b32_e32 v12, 22, v193 /*v705*/
	s_set_vgpr_msb 0x800
	v_min_u32_e32 v6, s5, v2
	s_set_vgpr_msb 8
	v_or_b32_e32 v2, 26, v193 /*v705*/
	s_wait_alu depctr_vm_vsrc(3)
	s_set_vgpr_msb 0x802
	v_min_u32_e32 v18, s5, v13
	v_min_i32_e32 v10, s5, v4
	s_wait_kmcnt 0x0
	v_mad_nc_i64_i32 v[0:1], s34, s36, v[0:1]
	v_dual_add_nc_u32 v3, s54, v6 :: v_dual_min_i32 v8, s5, v3
	v_min_u32_e32 v9, s5, v2
	v_lshl_add_u32 v32, v190 /*v702*/, 4, s29
	v_min_u32_e32 v12, s5, v12
	v_add_nc_u32_e32 v15, s54, v18
	v_dual_add_nc_u32 v11, s54, v10 :: v_dual_add_nc_u32 v5, s54, v8
	v_add_nc_u32_e32 v7, s54, v9
	v_mad_u32_u24 v34, 0x110, v8, v32
	v_mad_u32_u24 v35, 0x110, v9, v32
	v_mad_u32_u24 v36, 0x110, v10, v32
	v_add_nc_u32_e32 v10, s54, v12
	v_mad_nc_i64_i32 v[8:9], s8, v11, v[0:1]
	s_set_vgpr_msb 0x208
	v_or_b32_e32 v11, 20, v193 /*v705*/
	v_or_b32_e32 v13, 16, v193 /*v705*/
	v_or_b32_e32 v17, 14, v193 /*v705*/
	s_wait_alu depctr_vm_vsrc(2)
	v_or_b32_e32 v21, 12, v193 /*v705*/
	s_wait_alu depctr_vm_vsrc(1)
	v_or_b32_e32 v25, 8, v193 /*v705*/
	v_or_b32_e32 v24, 10, v193 /*v705*/
	v_or_b32_e32 v26, 2, v193 /*v705*/
	s_set_vgpr_msb 0x800
	v_min_u32_e32 v22, s5, v17
	v_min_i32_e32 v20, s5, v13
	v_min_i32_e32 v16, s5, v11
	v_min_i32_e32 v23, s5, v21
	s_wait_alu depctr_vm_vsrc(0)
	v_min_i32_e32 v28, s5, v25
	s_set_vgpr_msb 8
	v_or_b32_e32 v25, 6, v193 /*v705*/
	s_set_vgpr_msb 0x800
	v_add_nc_u32_e32 v19, s54, v20
	v_dual_add_nc_u32 v14, s54, v16 :: v_dual_add_nc_u32 v21, s54, v23
	v_min_u32_e32 v24, s5, v24
	s_set_vgpr_msb 8
	v_or_b32_e32 v27, 4, v193 /*v705*/
	s_load_b64 s[2:3], s[0:1], 0x0 nv
	s_set_vgpr_msb 0x800
	v_min_u32_e32 v44, s5, v25
	s_set_vgpr_msb 8
	v_min_i32_e32 v45, s5, v193 /*v705*/
	s_set_vgpr_msb 0x800
	v_min_u32_e32 v46, s5, v26
	v_mad_u32_u24 v38, 0x110, v16, v32
	v_mad_nc_i64_i32 v[16:17], v19, s8, v[0:1]
	v_add_nc_u32_e32 v19, s54, v22
	v_mad_u32_u24 v41, 0x110, v22, v32
	v_dual_add_nc_u32 v22, s54, v24 :: v_dual_min_i32 v47, s5, v27
	v_mad_u32_u24 v43, 0x110, v24, v32
	v_dual_add_nc_u32 v24, s54, v28 :: v_dual_add_nc_u32 v30, s54, v45
	v_dual_add_nc_u32 v29, s54, v44 :: v_dual_add_nc_u32 v31, s54, v46
	v_add_nc_u32_e32 v48, s54, v47
	v_mad_u32_u24 v49, 0x110, v28, v32
	v_mad_nc_i64_i32 v[2:3], v3, s8, v[0:1]
	v_mad_nc_i64_i32 v[26:27], v29, s8, v[0:1]
	v_mad_nc_i64_i32 v[28:29], v30, s8, v[0:1]
	v_mad_nc_i64_i32 v[30:31], v31, s8, v[0:1]
	v_mad_nc_i64_i32 v[4:5], v5, s8, v[0:1]
	v_mad_u32_u24 v33, 0x110, v6, v32
	v_mad_nc_i64_i32 v[6:7], v7, s8, v[0:1]
	v_mad_nc_i64_i32 v[10:11], v10, s8, v[0:1]
	v_mad_u32_u24 v37, 0x110, v12, v32
	v_mad_nc_i64_i32 v[12:13], v14, s8, v[0:1]
	v_mad_nc_i64_i32 v[14:15], v15, s8, v[0:1]
	v_mad_u32_u24 v39, 0x110, v18, v32
	v_mad_nc_i64_i32 v[18:19], v19, s8, v[0:1]
	v_mad_u32_u24 v40, 0x110, v20, v32
	v_mad_nc_i64_i32 v[20:21], v21, s8, v[0:1]
	v_mad_u32_u24 v42, 0x110, v23, v32
	v_mad_nc_i64_i32 v[22:23], v22, s8, v[0:1]
	v_mad_nc_i64_i32 v[24:25], v24, s8, v[0:1]
	v_mad_nc_i64_i32 v[0:1], v48, s8, v[0:1]
	v_mad_u32_u24 v45, 0x110, v45, v32
	s_wait_kmcnt 0x0
	v_lshl_add_u64 v[28:29], v[28:29], 1, s[2:3]
	v_mad_u32_u24 v44, 0x110, v44, v32
	v_mad_u32_u24 v47, 0x110, v47, v32
	v_mad_u32_u24 v32, 0x110, v46, v32
	v_lshl_add_u64 v[30:31], v[30:31], 1, s[2:3]
	v_lshl_add_u64 v[26:27], v[26:27], 1, s[2:3]
	v_lshl_add_u64 v[0:1], v[0:1], 1, s[2:3]
	v_lshl_add_u64 v[24:25], v[24:25], 1, s[2:3]
	v_lshl_add_u64 v[2:3], v[2:3], 1, s[2:3]
	v_lshl_add_u64 v[4:5], v[4:5], 1, s[2:3]
	v_lshl_add_u64 v[6:7], v[6:7], 1, s[2:3]
	v_lshl_add_u64 v[8:9], v[8:9], 1, s[2:3]
	v_lshl_add_u64 v[10:11], v[10:11], 1, s[2:3]
	v_lshl_add_u64 v[12:13], v[12:13], 1, s[2:3]
	v_lshl_add_u64 v[14:15], v[14:15], 1, s[2:3]
	v_lshl_add_u64 v[16:17], v[16:17], 1, s[2:3]
	v_lshl_add_u64 v[18:19], v[18:19], 1, s[2:3]
	v_lshl_add_u64 v[20:21], v[20:21], 1, s[2:3]
	v_lshl_add_u64 v[22:23], v[22:23], 1, s[2:3]
	s_wait_alu depctr_va_vdst(14)
	global_store_async_from_lds_b128 v[28:29], v45, off
	global_store_async_from_lds_b128 v[30:31], v32, off
	s_wait_alu depctr_va_vdst(12)
	global_store_async_from_lds_b128 v[0:1], v47, off
	global_store_async_from_lds_b128 v[26:27], v44, off
	s_wait_alu depctr_va_vdst(11)
	global_store_async_from_lds_b128 v[24:25], v49, off
	s_wait_alu depctr_va_vdst(0)
	global_store_async_from_lds_b128 v[22:23], v43, off
	global_store_async_from_lds_b128 v[20:21], v42, off
	global_store_async_from_lds_b128 v[18:19], v41, off
	global_store_async_from_lds_b128 v[16:17], v40, off
	global_store_async_from_lds_b128 v[14:15], v39, off
	global_store_async_from_lds_b128 v[12:13], v38, off
	global_store_async_from_lds_b128 v[10:11], v37, off
	global_store_async_from_lds_b128 v[8:9], v36, off
	global_store_async_from_lds_b128 v[6:7], v35, off
	global_store_async_from_lds_b128 v[4:5], v34, off
	global_store_async_from_lds_b128 v[2:3], v33, off
.LBB0_34:
	s_set_vgpr_msb 9
	v_cmp_gt_f32_e32 vcc_lo, 0x800000, v120 /*v632*/
	v_cmp_gt_f32_e64 s2, 0x800000, v121 /*v633*/
	s_load_b64 s[6:7], s[0:1], 0x40 nv
	s_wait_kmcnt 0x0
	s_wait_alu depctr_vm_vsrc(0)
	v_mul_lo_u32 v4, v95 /*v351*/, s37
	v_mul_lo_u32 v5, v94 /*v350*/, s37
	v_cndmask_b32_e64 v2, 0, 32, vcc_lo
	v_cndmask_b32_e64 v3, 0, 32, s2
	v_cndmask_b32_e64 v0, 0, 0x42000000, vcc_lo
	v_cndmask_b32_e64 v1, 0, 0x42000000, s2
	s_mul_i32 s3, s33, s39
	s_set_vgpr_msb 0x902
	v_ldexp_f32 v2, v120 /*v632*/, v2
	v_ldexp_f32 v3, v121 /*v633*/, v3
	s_mul_i32 s5, s34, s38
	s_set_vgpr_msb 0x208
	v_cmp_eq_u32_e32 vcc_lo, 0, v193 /*v705*/
	s_set_vgpr_msb 0x804
	v_cmp_gt_i32_e64 s0, s49, v95 /*v351*/
	v_log_f32_e32 v2, v2
	v_log_f32_e32 v3, v3
	v_cmp_gt_i32_e64 s1, s49, v94 /*v350*/
	s_add_co_i32 s8, s3, s5
	s_add_co_i32 s2, s3, s39
	s_and_b32 s0, vcc_lo, s0
	s_ashr_i32 s3, s2, 31
	s_and_b32 vcc_lo, vcc_lo, s1
	s_set_vgpr_msb 0x400
	v_dual_sub_f32 v0, v2, v0 :: v_dual_sub_f32 v1, v3, v1
	v_add_lshl_u32 v2, s8, v4, 2
	v_add_lshl_u32 v3, s8, v5, 2
	s_lshl_b32 s5, s2, 27
	v_dual_mul_f32 v0, 0x3f317218, v0 :: v_dual_mul_f32 v1, 0x3f317218, v1
	v_cndmask_b32_e64 v2, 0x7fffffff, v2, s0
	v_cndmask_b32_e32 v3, 0x7fffffff, v3, vcc_lo
	s_lshr_b64 s[2:3], s[2:3], 5
	s_set_vgpr_msb 2
	v_add_f32_e32 v0, v198 /*v710*/, v0
	s_set_vgpr_msb 0x201
	v_add_f32_e32 v1, v98 /*v354*/, v1
	s_and_b64 s[2:3], s[2:3], 0x1ffffffffffffff
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
		.amdhsa_next_free_vgpr 800
		.amdhsa_next_free_sgpr 75
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

	.set .Lkn_fmha_fwd_prefill_a16w16_m32x8_bshd_0.num_vgpr, 800
	.set .Lkn_fmha_fwd_prefill_a16w16_m32x8_bshd_0.num_agpr, 0
	.set .Lkn_fmha_fwd_prefill_a16w16_m32x8_bshd_0.numbered_sgpr, 75
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
    .sgpr_count:     77
    .sgpr_spill_count: 0
    .symbol:         kn_fmha_fwd_prefill_a16w16_m32x8_bshd_0.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     800
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa-unknown-gfx1250
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
